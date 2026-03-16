"""Stochastic differential equation tools for mcp-scicomp."""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy import stats

from mcp_scicomp.app import mcp
from mcp_scicomp.plotting import fig_to_base64

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_symbol_dict(
    n_dims: int, params: dict[str, float]
) -> OrderedDict[str, sp.Symbol]:
    """Build an ordered symbol dictionary for sympify.

    Scalar (n_dims=1): {x, t, ...params}
    System  (n_dims>1): {x0, x1, ..., t, ...params}
    """
    d: OrderedDict[str, sp.Symbol] = OrderedDict()
    if n_dims == 1:
        d["x"] = sp.Symbol("x")
    else:
        for i in range(n_dims):
            name = f"x{i}"
            d[name] = sp.Symbol(name)
    d["t"] = sp.Symbol("t")
    for k in params:
        d[k] = sp.Symbol(k)
    return d


def _parse_sde_expr(
    expr_str: str, sym_dict: OrderedDict[str, sp.Symbol]
) -> tuple[sp.Expr, Any]:
    """Parse an SDE expression string into (symbolic expr, numpy callable).

    Returns:
        (sympy_expr, lambdified_function)
    """
    cleaned = expr_str.replace("^", "**")
    sym_expr = sp.sympify(cleaned, locals=dict(sym_dict))
    sym_list = list(sym_dict.values())
    func = sp.lambdify(sym_list, sym_expr, modules=["numpy"])
    return sym_expr, func


def _broadcast(val: Any, n_paths: int) -> np.ndarray:
    """Ensure a lambdify result has shape (n_paths,)."""
    arr = np.asarray(val, dtype=float)
    return np.broadcast_to(arr, (n_paths,)).copy()


def _thin_series(arr: list[float], max_points: int = 500) -> list[float]:
    """Thin a time series to at most max_points evenly-spaced entries."""
    if len(arr) <= max_points:
        return arr
    step = max(1, len(arr) // max_points)
    thinned = arr[::step]
    # Always include the last point
    if thinned[-1] != arr[-1]:
        thinned.append(arr[-1])
    return thinned


# ---------------------------------------------------------------------------
# Tool 1: simulate_sde
# ---------------------------------------------------------------------------


@mcp.tool()
def simulate_sde(
    drift: str | list[str],
    diffusion: str | list[str],
    x0: float | list[float] = 0.0,
    t_end: float = 1.0,
    n_steps: int = 200,
    n_paths: int = 100,
    method: str = "euler_maruyama",
    params: dict[str, float] | None = None,
    analyze: list[str] | None = None,
    seed: Optional[int] = None,
    plot: bool = True,
) -> dict[str, Any]:
    """Simulate a stochastic differential equation dX = f(X,t)dt + g(X,t)dW.

    Use this tool to numerically solve any user-specified SDE, scalar or system,
    with Euler-Maruyama, Milstein, or Heun schemes. Returns path statistics,
    final-value distribution, and optional analysis plots.

    Args:
        drift: Drift function f(X,t) as a string (scalar) or list of strings
            (system). Use 'x' for scalar, 'x0','x1',... for systems. Time is 't'.
            Example (scalar): "mu*x"  Example (2D): ["-x0 + x1", "-x1"]
        diffusion: Diffusion function g(X,t), same format as drift.
            Example: "sigma*x" or ["0.5", "0.3"]
        x0: Initial condition. Float for scalar, list for system. Default 0.0.
        t_end: Simulation time horizon. Default 1.0.
        n_steps: Number of time steps. Default 200.
        n_paths: Number of Monte Carlo paths. Default 100.
        method: Integration scheme — "euler_maruyama", "milstein", or "heun".
            Milstein is scalar-only. Default "euler_maruyama".
        params: Dict of parameter values, e.g. {"mu": 0.05, "sigma": 0.2}.
            Default {}.
        analyze: List of analysis types to produce. Options:
            "paths" — sample path plot,
            "distribution" — histogram of final values,
            "moments" — mean ± std band over time,
            "density" — KDE of final values,
            "fokker_planck" — histogram + theoretical stationary density (scalar only).
            Default ["paths"].
        seed: Random seed for reproducibility. Default None.
        plot: If True, generate plots for requested analyses. Default True.

    Returns:
        dict with keys: method, n_paths, n_steps, n_dims, t_end, dt,
        time_grid (list[float]), path_statistics (dict with mean/std/q05/q95
        per component), final_distribution (dict with mean/std/min/max/median/
        q05/q95/skewness/kurtosis), plots (dict of analysis_name → base64 PNG).

    Example:
        simulate_sde(drift="mu*x", diffusion="sigma*x", x0=1.0, t_end=1.0,
                     params={"mu": 0.05, "sigma": 0.2}, method="milstein")
    """
    try:
        if params is None:
            params = {}
        if analyze is None:
            analyze = ["paths"]

        # --- 1. Input normalization ---
        drift_list = [drift] if isinstance(drift, str) else list(drift)
        diff_list = [diffusion] if isinstance(diffusion, str) else list(diffusion)
        x0_list = [x0] if isinstance(x0, (int, float)) else list(x0)
        n_dims = len(drift_list)

        if len(diff_list) != n_dims:
            return {
                "error": f"drift has {n_dims} components but diffusion has {len(diff_list)}.",
                "suggestion": "Provide the same number of drift and diffusion expressions.",
            }
        if len(x0_list) == 1 and n_dims > 1:
            x0_list = x0_list * n_dims
        if len(x0_list) != n_dims:
            return {
                "error": f"x0 has {len(x0_list)} components but drift has {n_dims}.",
                "suggestion": "Provide x0 with the same dimension as drift.",
            }

        valid_methods = {"euler_maruyama", "milstein", "heun"}
        if method not in valid_methods:
            return {
                "error": f"Unknown method '{method}'.",
                "suggestion": f"Use one of: {', '.join(sorted(valid_methods))}.",
            }
        if method == "milstein" and n_dims > 1:
            return {
                "error": "Milstein method is only supported for scalar SDEs.",
                "suggestion": "Use 'euler_maruyama' or 'heun' for systems.",
            }

        valid_analyze = {"paths", "distribution", "moments", "density", "fokker_planck"}
        for a in analyze:
            if a not in valid_analyze:
                return {
                    "error": f"Unknown analyze option '{a}'.",
                    "suggestion": f"Use: {', '.join(sorted(valid_analyze))}.",
                }
        if "fokker_planck" in analyze and n_dims > 1:
            return {
                "error": "fokker_planck analysis is only supported for scalar SDEs.",
                "suggestion": "Remove 'fokker_planck' or use a scalar SDE.",
            }

        if n_steps <= 0 or n_paths <= 0 or t_end <= 0:
            return {
                "error": "n_steps, n_paths, and t_end must all be positive.",
                "suggestion": "Provide positive values.",
            }

        # --- 2. Expression parsing ---
        sym_dict = _build_symbol_dict(n_dims, params)
        sym_list_vals = list(sym_dict.values())

        drift_syms: list[sp.Expr] = []
        drift_funcs: list[Any] = []
        diff_syms: list[sp.Expr] = []
        diff_funcs: list[Any] = []

        try:
            for s in drift_list:
                se, fn = _parse_sde_expr(s, sym_dict)
                drift_syms.append(se)
                drift_funcs.append(fn)
            for s in diff_list:
                se, fn = _parse_sde_expr(s, sym_dict)
                diff_syms.append(se)
                diff_funcs.append(fn)
        except Exception as parse_err:
            return {
                "error": f"Failed to parse expression: {parse_err}",
                "suggestion": "Use ** for exponentiation, valid variable names (x, x0, x1, t), and check syntax.",
            }

        # Milstein: need g'(x)
        gp_func = None
        if method == "milstein":
            x_sym = sym_dict["x"]
            gp_sym = sp.diff(diff_syms[0], x_sym)
            gp_func = sp.lambdify(sym_list_vals, gp_sym, modules=["numpy"])

        # --- 3. Simulation ---
        rng = np.random.default_rng(seed)
        dt = t_end / n_steps
        sqrt_dt = np.sqrt(dt)

        paths = np.empty((n_steps + 1, n_paths, n_dims))
        for d in range(n_dims):
            paths[0, :, d] = x0_list[d]

        dW = rng.normal(0.0, sqrt_dt, (n_steps, n_paths, n_dims))
        t_grid = np.linspace(0.0, t_end, n_steps + 1)

        def _eval_funcs(funcs: list[Any], X: np.ndarray, t_val: float) -> list[np.ndarray]:
            """Evaluate list of lambdified functions at state X and time t_val."""
            results = []
            for dim_i, fn in enumerate(funcs):
                if n_dims == 1:
                    args = [X[:, 0], t_val] + [params[k] for k in params]
                else:
                    args = [X[:, j] for j in range(n_dims)] + [t_val] + [params[k] for k in params]
                val = fn(*args)
                results.append(_broadcast(val, n_paths))
            return results

        nan_detected = False
        nan_step = -1

        for i in range(n_steps):
            X_curr = paths[i]  # (n_paths, n_dims)
            t_val = t_grid[i]

            if method == "euler_maruyama":
                f_vals = _eval_funcs(drift_funcs, X_curr, t_val)
                g_vals = _eval_funcs(diff_funcs, X_curr, t_val)
                for d in range(n_dims):
                    paths[i + 1, :, d] = X_curr[:, d] + f_vals[d] * dt + g_vals[d] * dW[i, :, d]

            elif method == "milstein":
                f_vals = _eval_funcs(drift_funcs, X_curr, t_val)
                g_vals = _eval_funcs(diff_funcs, X_curr, t_val)
                # g'(x)
                if n_dims == 1:
                    args = [X_curr[:, 0], t_val] + [params[k] for k in params]
                    gp_val = _broadcast(gp_func(*args), n_paths)
                else:
                    gp_val = np.zeros(n_paths)  # unreachable, validated above
                dw = dW[i, :, 0]
                paths[i + 1, :, 0] = (
                    X_curr[:, 0]
                    + f_vals[0] * dt
                    + g_vals[0] * dw
                    + 0.5 * g_vals[0] * gp_val * (dw**2 - dt)
                )

            elif method == "heun":
                f_vals = _eval_funcs(drift_funcs, X_curr, t_val)
                g_vals = _eval_funcs(diff_funcs, X_curr, t_val)
                # Predictor
                X_pred = np.empty_like(X_curr)
                for d in range(n_dims):
                    X_pred[:, d] = X_curr[:, d] + f_vals[d] * dt + g_vals[d] * dW[i, :, d]
                # Evaluate at predicted state
                t_next = t_grid[i + 1]
                f_pred = _eval_funcs(drift_funcs, X_pred, t_next)
                g_pred = _eval_funcs(diff_funcs, X_pred, t_next)
                # Corrector
                for d in range(n_dims):
                    paths[i + 1, :, d] = (
                        X_curr[:, d]
                        + 0.5 * (f_vals[d] + f_pred[d]) * dt
                        + 0.5 * (g_vals[d] + g_pred[d]) * dW[i, :, d]
                    )

            # Check for NaN/Inf
            if np.any(~np.isfinite(paths[i + 1])):
                nan_detected = True
                nan_step = i + 1
                break

        if nan_detected:
            return {
                "error": f"NaN or Inf detected at step {nan_step} (t={t_grid[nan_step]:.4f}).",
                "suggestion": "The SDE may be explosive. Try reducing t_end, increasing n_steps, or using a different drift.",
            }

        # --- 4. Statistics ---
        # Per-timestep stats (per component)
        path_statistics: dict[str, Any] = {}
        for d in range(n_dims):
            comp = paths[:, :, d]  # (n_steps+1, n_paths)
            key = f"x{d}" if n_dims > 1 else "x"
            path_statistics[key] = {
                "mean": _thin_series([float(v) for v in np.mean(comp, axis=1)]),
                "std": _thin_series([float(v) for v in np.std(comp, axis=1)]),
                "q05": _thin_series([float(v) for v in np.quantile(comp, 0.05, axis=1)]),
                "q95": _thin_series([float(v) for v in np.quantile(comp, 0.95, axis=1)]),
            }

        # Final distribution (first component for systems)
        final_vals = paths[-1, :, 0]
        final_distribution = {
            "mean": float(np.mean(final_vals)),
            "std": float(np.std(final_vals, ddof=1)),
            "min": float(np.min(final_vals)),
            "max": float(np.max(final_vals)),
            "median": float(np.median(final_vals)),
            "q05": float(np.quantile(final_vals, 0.05)),
            "q95": float(np.quantile(final_vals, 0.95)),
            "skewness": float(stats.skew(final_vals)),
            "kurtosis": float(stats.kurtosis(final_vals)),
        }

        # --- 5. Plots ---
        plots: dict[str, str] = {}
        if plot:
            time_arr = t_grid
            plot_dim = 0  # plot first component

            if "paths" in analyze:
                fig, ax = plt.subplots(figsize=(8, 5))
                show_paths = min(n_paths, 50)
                for p in range(show_paths):
                    ax.plot(time_arr, paths[:, p, plot_dim], linewidth=0.6, alpha=0.5)
                mean_line = np.mean(paths[:, :, plot_dim], axis=1)
                ax.plot(time_arr, mean_line, color="black", linewidth=2, label="Mean")
                ax.set_title("SDE Sample Paths")
                ax.set_xlabel("Time")
                ax.set_ylabel("X(t)")
                ax.legend()
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                plots["paths"] = fig_to_base64(fig)

            if "distribution" in analyze:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.hist(final_vals, bins=50, density=True, alpha=0.7, color="steelblue")
                ax.axvline(float(np.mean(final_vals)), color="red", linestyle="--", label="Mean")
                ax.axvline(float(np.median(final_vals)), color="green", linestyle="--", label="Median")
                ax.set_title("Final Value Distribution")
                ax.set_xlabel("X(T)")
                ax.set_ylabel("Density")
                ax.legend()
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                plots["distribution"] = fig_to_base64(fig)

            if "moments" in analyze:
                fig, ax = plt.subplots(figsize=(8, 5))
                mean_vals = np.mean(paths[:, :, plot_dim], axis=1)
                std_vals = np.std(paths[:, :, plot_dim], axis=1)
                ax.plot(time_arr, mean_vals, color="blue", linewidth=2, label="Mean")
                ax.fill_between(
                    time_arr,
                    mean_vals - std_vals,
                    mean_vals + std_vals,
                    alpha=0.3,
                    color="blue",
                    label="±1 Std",
                )
                ax.set_title("Mean ± Std Over Time")
                ax.set_xlabel("Time")
                ax.set_ylabel("X(t)")
                ax.legend()
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                plots["moments"] = fig_to_base64(fig)

            if "density" in analyze:
                fig, ax = plt.subplots(figsize=(8, 5))
                kde = stats.gaussian_kde(final_vals)
                x_range = np.linspace(float(np.min(final_vals)), float(np.max(final_vals)), 200)
                ax.plot(x_range, kde(x_range), color="steelblue", linewidth=2)
                ax.fill_between(x_range, kde(x_range), alpha=0.3, color="steelblue")
                ax.set_title("KDE of Final Values")
                ax.set_xlabel("X(T)")
                ax.set_ylabel("Density")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                plots["density"] = fig_to_base64(fig)

            if "fokker_planck" in analyze and n_dims == 1:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.hist(final_vals, bins=50, density=True, alpha=0.6, color="steelblue", label="Empirical")
                # Attempt theoretical stationary density
                fp_note = None
                try:
                    x_sym = sym_dict["x"]
                    f_sym = drift_syms[0]
                    g_sym = diff_syms[0]
                    # Substitute param values
                    subs = {sym_dict[k]: v for k, v in params.items()}
                    f_sub = f_sym.subs(subs)
                    g_sub = g_sym.subs(subs)
                    g2 = g_sub**2
                    if g2.equals(sp.S.Zero):
                        fp_note = "g(x)=0, no stochastic term"
                    else:
                        integrand = 2 * f_sub / g2
                        integral = sp.integrate(integrand, x_sym)
                        if isinstance(integral, sp.Integral):
                            fp_note = "Could not evaluate integral symbolically"
                        else:
                            density_sym = (1 / g2) * sp.exp(integral)
                            density_fn = sp.lambdify([x_sym], density_sym, modules=["numpy"])
                            x_range_fp = np.linspace(
                                float(np.min(final_vals)) - 0.5,
                                float(np.max(final_vals)) + 0.5,
                                300,
                            )
                            y_vals = density_fn(x_range_fp)
                            y_arr = np.asarray(y_vals, dtype=float)
                            if np.all(np.isfinite(y_arr)) and np.any(y_arr > 0):
                                # Normalize via trapezoidal rule
                                area = np.trapz(y_arr, x_range_fp)
                                if area > 0:
                                    y_arr /= area
                                    ax.plot(x_range_fp, y_arr, color="red", linewidth=2, label="Theoretical stationary")
                                else:
                                    fp_note = "Density not normalizable"
                            else:
                                fp_note = "Density has non-finite values"
                except Exception as fp_err:
                    fp_note = f"FP overlay failed: {fp_err}"

                ax.set_title("Fokker-Planck: Empirical vs Theoretical")
                ax.set_xlabel("X(T)")
                ax.set_ylabel("Density")
                ax.legend()
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                plots["fokker_planck"] = fig_to_base64(fig)
                if fp_note:
                    plots["fokker_planck_note"] = fp_note

        # --- 6. Return ---
        time_grid_out = _thin_series([float(v) for v in t_grid])

        return {
            "method": method,
            "n_paths": n_paths,
            "n_steps": n_steps,
            "n_dims": n_dims,
            "t_end": float(t_end),
            "dt": float(dt),
            "time_grid": time_grid_out,
            "path_statistics": path_statistics,
            "final_distribution": final_distribution,
            "plots": plots,
        }

    except Exception as exc:
        logger.exception("simulate_sde failed")
        return {"error": str(exc), "suggestion": "Check expressions and parameters."}


# ---------------------------------------------------------------------------
# Tool 2: analyze_sde
# ---------------------------------------------------------------------------


@mcp.tool()
def analyze_sde(
    drift: str | list[str],
    diffusion: str | list[str],
    params: dict[str, float] | None = None,
    x_range: list[float] | None = None,
    plot: bool = True,
) -> dict[str, Any]:
    """Symbolic analysis of a stochastic differential equation: stationary distribution,
    stability, moment equations, and Kolmogorov operators.

    Use this tool when you need analytical (not numerical) insights about an SDE:
    equilibria, stability classification, whether a stationary distribution exists,
    its closed form, and the forward/backward Kolmogorov equations.

    Args:
        drift: Drift function f(X,t) as string or list of strings.
            Scalar: use 'x'. System: use 'x0','x1',...
            Example: "-theta*(x - mu)" or ["-x0 + x1", "-x1"]
        diffusion: Diffusion function g(X,t), same format as drift.
            Example: "sigma" or ["0.5", "0.3"]
        params: Dict of parameter values, e.g. {"theta": 2.0, "mu": 3.0, "sigma": 1.0}.
            Default {}.
        x_range: Range [xmin, xmax] for plotting drift. Default [-5.0, 5.0].
        plot: If True, produce a drift plot (scalar) or quiver plot (2D system).
            Default True.

    Returns:
        dict with keys: n_dims, stationary_distribution (scalar only, with
        exists/density_latex/identified_distribution/parameters),
        equilibria (list of point/stability/f_prime or eigenvalues),
        moment_equations (dE_dt_latex, dVar_dt_latex),
        kolmogorov (forward_latex, backward_latex), plot (base64 PNG or None).

    Example:
        analyze_sde(drift="-2*(x - 3)", diffusion="1.0",
                    params={"theta": 2.0, "mu": 3.0, "sigma": 1.0})
    """
    try:
        if params is None:
            params = {}
        if x_range is None:
            x_range = [-5.0, 5.0]

        drift_list = [drift] if isinstance(drift, str) else list(drift)
        diff_list = [diffusion] if isinstance(diffusion, str) else list(diffusion)
        n_dims = len(drift_list)

        if len(diff_list) != n_dims:
            return {
                "error": f"drift has {n_dims} components but diffusion has {len(diff_list)}.",
                "suggestion": "Provide the same number of drift and diffusion expressions.",
            }

        # --- Parse expressions ---
        sym_dict = _build_symbol_dict(n_dims, params)

        try:
            drift_syms: list[sp.Expr] = []
            diff_syms: list[sp.Expr] = []
            for s in drift_list:
                cleaned = s.replace("^", "**")
                drift_syms.append(sp.sympify(cleaned, locals=dict(sym_dict)))
            for s in diff_list:
                cleaned = s.replace("^", "**")
                diff_syms.append(sp.sympify(cleaned, locals=dict(sym_dict)))
        except Exception as parse_err:
            return {
                "error": f"Failed to parse expression: {parse_err}",
                "suggestion": "Use ** for exponentiation, valid variable names, and check syntax.",
            }

        # Substitute param values
        subs = {sym_dict[k]: v for k, v in params.items()}
        drift_sub = [expr.subs(subs) for expr in drift_syms]
        diff_sub = [expr.subs(subs) for expr in diff_syms]

        result: dict[str, Any] = {"n_dims": n_dims}

        # --- Stationary distribution (scalar only) ---
        if n_dims == 1:
            x_sym = sym_dict["x"]
            f_sub = drift_sub[0]
            g_sub = diff_sub[0]
            g2 = g_sub**2

            stat_dist: dict[str, Any] = {}
            if g2.equals(sp.S.Zero):
                stat_dist = {
                    "exists": "unknown",
                    "density_latex": "",
                    "identified_distribution": "N/A (deterministic system)",
                    "parameters": {},
                }
            else:
                integrand = 2 * f_sub / g2
                integral = sp.integrate(integrand, x_sym)

                if isinstance(integral, sp.Integral):
                    stat_dist = {
                        "exists": "unknown",
                        "density_latex": str(integral),
                        "identified_distribution": "Could not evaluate integral",
                        "parameters": {},
                    }
                else:
                    density_unnorm = (1 / g2) * sp.exp(integral)
                    density_latex = sp.latex(density_unnorm)

                    # Check if exponent is quadratic with negative leading coeff → Gaussian
                    exponent = integral
                    poly = sp.Poly(exponent, x_sym) if exponent.is_polynomial(x_sym) else None

                    identified = "unknown"
                    dist_params: dict[str, Any] = {}
                    exists: bool | str = "unknown"

                    if poly is not None and poly.degree() == 2:
                        coeffs = poly.all_coeffs()  # [a, b, c] for ax^2 + bx + c
                        a_coeff = float(coeffs[0])
                        b_coeff = float(coeffs[1])

                        if a_coeff < 0:
                            # Gaussian: exponent = a*x^2 + b*x + c
                            # Compare with -(x-mu)^2/(2*sigma^2)
                            # a = -1/(2*sigma^2), b = mu/(sigma^2)
                            sigma2 = -1.0 / (2.0 * a_coeff)
                            mu_val = b_coeff * sigma2
                            identified = f"Normal({mu_val:.4g}, {sigma2:.4g})"
                            dist_params = {"mean": mu_val, "variance": sigma2}
                            exists = True
                        else:
                            # Positive leading coeff → diverges → no stationary
                            exists = False
                    elif poly is not None and poly.degree() == 1:
                        # Linear exponent → exponential tail → may not be normalizable
                        exists = False
                    else:
                        # Can't determine automatically
                        exists = "unknown"

                    stat_dist = {
                        "exists": exists,
                        "density_latex": density_latex,
                        "identified_distribution": identified,
                        "parameters": dist_params,
                    }

            result["stationary_distribution"] = stat_dist

        # --- Equilibria & stability ---
        equilibria: list[dict[str, Any]] = []
        if n_dims == 1:
            x_sym = sym_dict["x"]
            f_sub = drift_sub[0]
            # Remove time dependence for equilibria
            f_eq = f_sub.subs(sym_dict["t"], 0)
            try:
                eq_points = sp.solve(f_eq, x_sym)
            except Exception:
                eq_points = []

            f_prime = sp.diff(f_eq, x_sym)
            for pt in eq_points:
                if not pt.is_real:
                    # Filter complex
                    try:
                        if sp.im(pt) != 0:
                            continue
                    except Exception:
                        continue
                fp_val = float(f_prime.subs(x_sym, pt))
                if fp_val < 0:
                    stability = "stable"
                elif fp_val > 0:
                    stability = "unstable"
                else:
                    stability = "marginal"
                equilibria.append({
                    "point": float(pt),
                    "stability": stability,
                    "f_prime": fp_val,
                })
        else:
            # System: solve f_i = 0 for all i
            state_syms = [sym_dict[f"x{i}"] for i in range(n_dims)]
            f_eq_sys = [f.subs(sym_dict["t"], 0) for f in drift_sub]
            try:
                eq_solutions = sp.solve(f_eq_sys, state_syms, dict=True)
            except Exception:
                eq_solutions = []

            for sol in eq_solutions:
                pt = [float(sol.get(s, 0)) for s in state_syms]
                # Compute Jacobian
                J = sp.Matrix(n_dims, n_dims, lambda i, j: sp.diff(f_eq_sys[i], state_syms[j]))
                J_eval = J.subs(sol)
                try:
                    eigenvals = [complex(ev) for ev in J_eval.eigenvals(multiple=True)]
                    real_parts = [ev.real for ev in eigenvals]
                    if all(r < 0 for r in real_parts):
                        stability = "stable"
                    elif all(r > 0 for r in real_parts):
                        stability = "unstable"
                    elif any(r > 0 for r in real_parts) and any(r < 0 for r in real_parts):
                        stability = "saddle"
                    else:
                        stability = "marginal"
                    eigenvals_out = [
                        float(ev.real) if ev.imag == 0 else [float(ev.real), float(ev.imag)]
                        for ev in eigenvals
                    ]
                except Exception:
                    stability = "unknown"
                    eigenvals_out = []

                equilibria.append({
                    "point": pt,
                    "stability": stability,
                    "eigenvalues": eigenvals_out,
                })

        result["equilibria"] = equilibria

        # --- Moment equations (scalar only) ---
        if n_dims == 1:
            x_sym = sym_dict["x"]
            f_sub = drift_sub[0]
            g_sub = diff_sub[0]

            dE_dt_latex = sp.latex(f_sub)

            # For linear SDEs: f = a*x + b, g = c*x + d
            # dE[X]/dt = a*E[X] + b
            # dVar[X]/dt = 2*a*Var[X] + (c*E[X]+d)^2  (approximate)
            dVar_dt_latex = ""
            try:
                f_poly = sp.Poly(f_sub, x_sym)
                g_poly = sp.Poly(g_sub, x_sym)
                if f_poly.degree() <= 1 and g_poly.degree() <= 1:
                    f_coeffs = f_poly.all_coeffs()
                    a_val = f_coeffs[0] if f_poly.degree() == 1 else sp.S.Zero
                    E_x = sp.Symbol("E_X")
                    Var_x = sp.Symbol("Var_X")
                    dVar = 2 * a_val * Var_x + g_sub.subs(x_sym, E_x) ** 2
                    dVar_dt_latex = sp.latex(dVar)
            except Exception:
                pass

            result["moment_equations"] = {
                "dE_dt_latex": dE_dt_latex,
                "dVar_dt_latex": dVar_dt_latex,
            }

        # --- Kolmogorov operators (scalar only) ---
        if n_dims == 1:
            x_sym = sym_dict["x"]
            f_sub = drift_sub[0]
            g_sub = diff_sub[0]
            g2 = g_sub**2

            p = sp.Function("p")(x_sym)
            u = sp.Function("u")(x_sym)

            # Forward (Fokker-Planck): L*p = -d(f*p)/dx + (1/2)*d^2(g^2*p)/dx^2
            forward = -sp.diff(f_sub * p, x_sym) + sp.Rational(1, 2) * sp.diff(g2 * p, x_sym, 2)

            # Backward: L*u = f*du/dx + (1/2)*g^2*d^2u/dx^2
            backward = f_sub * sp.diff(u, x_sym) + sp.Rational(1, 2) * g2 * sp.diff(u, x_sym, 2)

            result["kolmogorov"] = {
                "forward_latex": sp.latex(forward),
                "backward_latex": sp.latex(backward),
            }

        # --- Plot ---
        plot_b64: str | None = None
        if plot:
            if n_dims == 1:
                x_sym = sym_dict["x"]
                f_sub = drift_sub[0].subs(sym_dict["t"], 0)
                f_fn = sp.lambdify([x_sym], f_sub, modules=["numpy"])
                x_arr = np.linspace(x_range[0], x_range[1], 300)
                y_arr = np.asarray(f_fn(x_arr), dtype=float)
                y_arr = np.broadcast_to(y_arr, x_arr.shape).copy()

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(x_arr, y_arr, color="steelblue", linewidth=2, label="f(x)")
                ax.axhline(0, color="gray", linewidth=0.5)

                # Shade positive/negative drift
                ax.fill_between(x_arr, y_arr, 0, where=(y_arr > 0), alpha=0.15, color="green")
                ax.fill_between(x_arr, y_arr, 0, where=(y_arr < 0), alpha=0.15, color="red")

                # Mark equilibria
                for eq in equilibria:
                    pt_val = eq["point"]
                    if x_range[0] <= pt_val <= x_range[1]:
                        color = "green" if eq["stability"] == "stable" else "red"
                        marker = "o"
                        ax.plot(pt_val, 0, marker, color=color, markersize=10,
                                label=f"{'Stable' if eq['stability'] == 'stable' else 'Unstable'} eq @ {pt_val:.2f}")

                ax.set_title("Drift Function & Equilibria")
                ax.set_xlabel("x")
                ax.set_ylabel("f(x)")
                ax.legend()
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                plot_b64 = fig_to_base64(fig)

            elif n_dims == 2:
                x0_sym = sym_dict["x0"]
                x1_sym = sym_dict["x1"]
                f0_sub = drift_sub[0].subs(sym_dict["t"], 0)
                f1_sub = drift_sub[1].subs(sym_dict["t"], 0)
                f0_fn = sp.lambdify([x0_sym, x1_sym], f0_sub, modules=["numpy"])
                f1_fn = sp.lambdify([x0_sym, x1_sym], f1_sub, modules=["numpy"])

                x0_arr = np.linspace(x_range[0], x_range[1], 20)
                x1_arr = np.linspace(x_range[0], x_range[1], 20)
                X0, X1 = np.meshgrid(x0_arr, x1_arr)
                U = np.asarray(f0_fn(X0, X1), dtype=float)
                V = np.asarray(f1_fn(X0, X1), dtype=float)
                U = np.broadcast_to(U, X0.shape).copy()
                V = np.broadcast_to(V, X0.shape).copy()

                fig, ax = plt.subplots(figsize=(8, 8))
                ax.quiver(X0, X1, U, V, alpha=0.6)

                for eq in equilibria:
                    pt = eq["point"]
                    color = "green" if eq["stability"] == "stable" else "red"
                    ax.plot(pt[0], pt[1], "o", color=color, markersize=10,
                            label=f"{eq['stability']} @ ({pt[0]:.2f}, {pt[1]:.2f})")

                ax.set_title("Drift Vector Field & Equilibria")
                ax.set_xlabel("x0")
                ax.set_ylabel("x1")
                ax.legend()
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                plot_b64 = fig_to_base64(fig)

        result["plot"] = plot_b64
        return result

    except Exception as exc:
        logger.exception("analyze_sde failed")
        return {"error": str(exc), "suggestion": "Check expressions and parameters."}
