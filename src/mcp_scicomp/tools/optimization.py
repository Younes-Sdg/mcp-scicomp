"""Optimization tools: general-purpose optimization and curve fitting."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as sp_opt

from mcp_scicomp.app import mcp
from mcp_scicomp.plotting import fig_to_base64
from mcp_scicomp.utils import parse_expr, resolve_data

logger = logging.getLogger(__name__)

_MINIMIZE_METHODS: dict[str, str] = {
    "nelder_mead": "Nelder-Mead",
    "bfgs": "BFGS",
    "l_bfgs_b": "L-BFGS-B",
    "cobyla": "COBYLA",
    "slsqp": "SLSQP",
}

_BOUNDS_SUPPORTED = {"l_bfgs_b", "slsqp", "nelder_mead"}
_CONSTRAINTS_SUPPORTED = {"slsqp", "cobyla"}


@mcp.tool()
def optimize(
    objective: str,
    variables: list[str],
    method: str = "bfgs",
    mode: str = "minimize",
    bounds: Optional[dict[str, list[float]]] = None,
    constraints: Optional[list[dict[str, str]]] = None,
    x0: Optional[list[float]] = None,
    params: Optional[dict[str, float]] = None,
    plot: bool = True,
) -> dict[str, Any]:
    """Minimize, maximize, or find the root of a mathematical objective function.

    Use this tool for unconstrained/constrained optimization, global optimization,
    or single-variable root finding (finding x where f(x) = 0).

    Args:
        objective: Mathematical expression string in the named variables and optional
                   fixed parameters from 'params'. Use ** for exponentiation.
                   Example: "x**2 + y**2 - 4*x + 3"
        variables: List of variable names that are optimized (the free variables).
                   Example: ["x", "y"]
        method: Solver algorithm. One of:
                  "bfgs"                   — quasi-Newton, efficient for smooth functions (default)
                  "nelder_mead"            — simplex method, no gradient, robust for noisy objectives
                  "l_bfgs_b"              — limited-memory BFGS with box constraints
                  "cobyla"               — linear approximation with inequality constraints
                  "slsqp"               — sequential QP with equality and inequality constraints
                  "differential_evolution" — global stochastic optimizer (requires bounds)
                  "root"                  — find x where f(x) = 0 (single variable recommended)
                Default "bfgs".
        mode: "minimize" (default) to find the minimum, "maximize" to find the maximum.
              Ignored when method="root".
        bounds: Optional dict mapping each variable name to [lower, upper] bound.
                Use null/None for unbounded on one side. Required for method="differential_evolution".
                Example: {"x": [-5.0, 5.0], "y": [0.0, 10.0]}
        constraints: List of constraint dicts, each with:
                       "type": "eq" (f=0) or "ineq" (f≥0)
                       "expr": expression string in the same variables
                     Only supported by method="slsqp" and "cobyla".
                     Example: [{"type": "eq", "expr": "x + y - 1"}]
        x0: Initial guess as a list of floats, one per variable. Defaults to zero
            or the midpoint of each variable's bounds if provided.
        params: Optional dict of fixed parameter values used in the objective expression.
                Example: {"a": 2.0, "b": -3.0} for objective "a*x**2 + b*x"
        plot: If True and len(variables) <= 2, return a line plot (1D) or contour
              plot (2D) with the optimum marked. Default True.

    Returns:
        Dict with keys:
          - optimal_values: dict mapping variable name → optimal float value
          - objective_value: float — f(x*) at the solution
          - success: bool — whether the solver converged
          - message: str — solver status message
          - n_iterations: int — number of iterations (0 if unavailable)
          - n_evaluations: int — number of function evaluations
          - method: str — method used
          - mode: str — minimize / maximize / root
          - plot: base64 PNG string (1- or 2-variable cases, when plot=True)
          - error / suggestion: present only on failure

    Example:
        optimize(
            objective="(x - 2)**2 + (y + 1)**2",
            variables=["x", "y"],
            method="bfgs",
            x0=[0.0, 0.0]
        )
        # Returns optimal_values={"x": 2.0, "y": -1.0}, objective_value≈0.0

        optimize(
            objective="x**3 - x - 2",
            variables=["x"],
            method="root",
            bounds={"x": [1.0, 2.0]}
        )
        # Finds root ≈ 1.5214
    """
    try:
        valid_methods = set(_MINIMIZE_METHODS) | {"differential_evolution", "root"}
        if method not in valid_methods:
            return {
                "error": f"Unknown method '{method}'. Choose from: {sorted(valid_methods)}.",
                "suggestion": "Use one of: " + ", ".join(sorted(valid_methods)),
            }
        if mode not in ("minimize", "maximize"):
            return {
                "error": f"Unknown mode '{mode}'. Use 'minimize' or 'maximize'.",
                "suggestion": "Set mode='minimize' or mode='maximize'.",
            }

        n_vars = len(variables)
        if n_vars == 0:
            return {
                "error": "At least one variable must be specified.",
                "suggestion": "Provide variable names like ['x'] or ['x', 'y'].",
            }

        fixed_params: dict[str, float] = params or {}

        # Validate: no overlap between variable names and param names
        overlap = set(variables) & set(fixed_params)
        if overlap:
            return {
                "error": f"Variable names and param names must not overlap: {sorted(overlap)}.",
                "suggestion": "Rename the fixed parameters to avoid collisions with variables.",
            }

        # Validate x0 length
        if x0 is not None and len(x0) != n_vars:
            return {
                "error": f"x0 has {len(x0)} elements but {n_vars} variables were specified.",
                "suggestion": f"Provide x0 with exactly {n_vars} elements, one per variable.",
            }

        # Build symbol namespace: variables first, then fixed params
        all_symbols: dict[str, str] = {v: v for v in variables}
        all_symbols.update({k: k for k in fixed_params})

        obj_func_raw = parse_expr(objective, all_symbols)
        param_vals = [fixed_params[k] for k in fixed_params]

        def obj_scalar(x_arr: np.ndarray) -> float:
            """Evaluate objective at x_arr (shape (n_vars,)), return scalar."""
            val = obj_func_raw(*list(x_arr), *param_vals)
            return float(np.asarray(val).flat[0])

        sign = -1.0 if mode == "maximize" else 1.0

        def signed_obj(x_arr: np.ndarray) -> float:
            return sign * obj_scalar(x_arr)

        # Build scipy bounds: list of (lo, hi) per variable
        scipy_bounds_list: Optional[list[tuple[Any, Any]]] = None
        if bounds:
            scipy_bounds_list = [
                (bounds[v][0] if v in bounds else None,
                 bounds[v][1] if v in bounds else None)
                for v in variables
            ]

        # Initial guess
        if x0 is not None:
            x_init = np.array(x0, dtype=float)
        elif scipy_bounds_list:
            x_init = np.array([
                0.5 * ((lo if lo is not None else -1.0) + (hi if hi is not None else 1.0))
                for lo, hi in scipy_bounds_list
            ], dtype=float)
        else:
            x_init = np.zeros(n_vars, dtype=float)

        # Build scipy constraint dicts (constraints can reference fixed params too)
        constraint_symbols: dict[str, str] = {v: v for v in variables}
        constraint_symbols.update({k: k for k in fixed_params})
        scipy_constraints: list[dict[str, Any]] = []
        if constraints:
            for c in constraints:
                c_type = c.get("type", "ineq")
                c_expr_str = c.get("expr", "0")
                c_func_raw = parse_expr(c_expr_str, constraint_symbols)

                def _make_cfun(f: Any, pv: list[float] = param_vals) -> Any:
                    def _cfun(x_arr: np.ndarray) -> float:
                        val = f(*list(x_arr), *pv)
                        return float(np.asarray(val).flat[0])
                    return _cfun

                scipy_constraints.append({"type": c_type, "fun": _make_cfun(c_func_raw)})

        # ------------------------------------------------------------------ #
        # Solve
        # ------------------------------------------------------------------ #
        result_dict: dict[str, Any] = {
            "method": method,
            "mode": "root" if method == "root" else mode,
        }

        if method == "root":
            if n_vars == 1:
                if scipy_bounds_list and all(b is not None for b in scipy_bounds_list[0]):
                    lo, hi = scipy_bounds_list[0]
                    brent_res, brent_info = sp_opt.brentq(
                        lambda x: obj_scalar(np.array([x])), lo, hi, full_output=True
                    )
                    x_opt = np.array([brent_res])
                    converged = bool(brent_info.converged)
                    msg = "Brent's method converged" if converged else "Brent's method did not converge"
                    n_iter = int(brent_info.iterations)
                    n_eval = int(brent_info.function_calls)
                else:
                    fsolve_res = sp_opt.fsolve(
                        lambda x: obj_scalar(np.array([x[0]])), x_init, full_output=True
                    )
                    x_opt = np.array([fsolve_res[0][0]])
                    converged = bool(fsolve_res[2] == 1)
                    msg = str(fsolve_res[3])
                    n_iter = 0
                    n_eval = int(fsolve_res[1]["nfev"])
            else:
                fsolve_res = sp_opt.fsolve(
                    lambda x: obj_scalar(np.asarray(x)), x_init, full_output=True
                )
                x_opt = np.asarray(fsolve_res[0])
                converged = bool(fsolve_res[2] == 1)
                msg = str(fsolve_res[3])
                n_iter = 0
                n_eval = int(fsolve_res[1]["nfev"])

            result_dict.update({
                "optimal_values": {v: float(x_opt[i]) for i, v in enumerate(variables)},
                "objective_value": float(obj_scalar(x_opt)),
                "success": converged,
                "message": msg,
                "n_iterations": n_iter,
                "n_evaluations": n_eval,
            })

        elif method == "differential_evolution":
            if not scipy_bounds_list:
                return {
                    "error": "differential_evolution requires bounds for all variables.",
                    "suggestion": "Provide bounds={'x': [-10, 10], 'y': [-10, 10]}.",
                }
            sol = sp_opt.differential_evolution(signed_obj, scipy_bounds_list)
            result_dict.update({
                "optimal_values": {v: float(sol.x[i]) for i, v in enumerate(variables)},
                "objective_value": float(sign * sol.fun),
                "success": bool(sol.success),
                "message": str(sol.message),
                "n_iterations": int(sol.nit),
                "n_evaluations": int(sol.nfev),
            })

        else:
            scipy_method = _MINIMIZE_METHODS[method]
            kwargs: dict[str, Any] = {"method": scipy_method}
            if scipy_bounds_list and method in _BOUNDS_SUPPORTED:
                kwargs["bounds"] = scipy_bounds_list
            if scipy_constraints and method in _CONSTRAINTS_SUPPORTED:
                kwargs["constraints"] = scipy_constraints

            sol = sp_opt.minimize(signed_obj, x_init, **kwargs)
            result_dict.update({
                "optimal_values": {v: float(sol.x[i]) for i, v in enumerate(variables)},
                "objective_value": float(sign * sol.fun),
                "success": bool(sol.success),
                "message": str(sol.message),
                "n_iterations": int(getattr(sol, "nit", 0)),
                "n_evaluations": int(getattr(sol, "nfev", 0)),
            })

        if np.isnan(result_dict["objective_value"]):
            result_dict["warning"] = "Objective value is NaN — the solver may have diverged."

        # ------------------------------------------------------------------ #
        # Plot (1-D or 2-D)
        # ------------------------------------------------------------------ #
        if plot and n_vars <= 2 and method != "root":
            x_opt_arr = np.array(
                [result_dict["optimal_values"][v] for v in variables], dtype=float
            )
            f_opt = result_dict["objective_value"]

            if n_vars == 1:
                v0 = variables[0]
                x_star = x_opt_arr[0]
                half_range = max(abs(x_star) * 1.5, 4.0)
                x_lo = (bounds[v0][0] if (bounds and v0 in bounds) else x_star - half_range)
                x_hi = (bounds[v0][1] if (bounds and v0 in bounds) else x_star + half_range)
                xs = np.linspace(x_lo, x_hi, 400)
                ys = np.array([obj_scalar(np.array([xi])) for xi in xs])

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(xs, ys, color="steelblue", linewidth=2, label=f"f({v0})")
                label = (f"{'min' if mode == 'minimize' else 'max'} "
                         f"= ({x_star:.4g}, {f_opt:.4g})")
                ax.scatter([x_star], [f_opt], color="red", zorder=5, s=80, label=label)
                ax.set_xlabel(v0)
                ax.set_ylabel(f"f({v0})")
                ax.set_title(f"Objective: {objective}")
                ax.legend()
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                result_dict["plot"] = fig_to_base64(fig)

            else:  # 2-D contour
                vx, vy = variables[0], variables[1]
                x_star, y_star = float(x_opt_arr[0]), float(x_opt_arr[1])
                rx = max(abs(x_star) * 1.5, 4.0)
                ry = max(abs(y_star) * 1.5, 4.0)
                x_lo = bounds[vx][0] if (bounds and vx in bounds) else x_star - rx
                x_hi = bounds[vx][1] if (bounds and vx in bounds) else x_star + rx
                y_lo = bounds[vy][0] if (bounds and vy in bounds) else y_star - ry
                y_hi = bounds[vy][1] if (bounds and vy in bounds) else y_star + ry

                xs = np.linspace(x_lo, x_hi, 80)
                ys = np.linspace(y_lo, y_hi, 80)
                Xg, Yg = np.meshgrid(xs, ys)
                # Vectorised evaluation via lambdified expression
                Zg = obj_func_raw(Xg, Yg, *param_vals)

                fig, ax = plt.subplots(figsize=(8, 6))
                cp = ax.contourf(Xg, Yg, Zg, levels=30, cmap="viridis")
                plt.colorbar(cp, ax=ax, label=f"f({vx}, {vy})")
                ax.contour(Xg, Yg, Zg, levels=15, colors="white", alpha=0.3, linewidths=0.5)
                label = (f"{'min' if mode == 'minimize' else 'max'} "
                         f"({x_star:.3g}, {y_star:.3g})")
                ax.scatter([x_star], [y_star], color="red", zorder=5, s=100, label=label)
                ax.set_xlabel(vx)
                ax.set_ylabel(vy)
                ax.set_title(f"Objective: {objective}")
                ax.legend(loc="upper right", fontsize=8)
                fig.tight_layout()
                result_dict["plot"] = fig_to_base64(fig)

        return result_dict

    except Exception as exc:
        logger.exception("optimize failed")
        return {
            "error": str(exc),
            "suggestion": "Check expression syntax, variable names, and method/mode choice.",
        }


@mcp.tool()
def curve_fit_data(
    model_expr: str,
    parameter_names: list[str],
    data_x: Optional[list[float]] = None,
    data_y: Optional[list[float]] = None,
    file_path: Optional[str] = None,
    x_column: Optional[str] = None,
    y_column: Optional[str] = None,
    p0: Optional[list[float]] = None,
    bounds: Optional[dict[str, list[float]]] = None,
    plot: bool = True,
) -> dict[str, Any]:
    """Fit a parametric model expression to (x, y) data using nonlinear least squares.

    Use this tool when you have measurement data and want to estimate the parameters
    of a known model (e.g. exponential decay, Gaussian peak, power law, logistic growth).

    Args:
        model_expr: Expression string in the independent variable 'x' and the named
                    parameters from 'parameter_names'. Use ** for exponentiation.
                    Example: "a * exp(-b * x) + c"
        parameter_names: List of parameter names that will be fitted.
                         Example: ["a", "b", "c"]
        data_x: Independent variable values as a list of floats.
                Mutually exclusive with file_path.
        data_y: Dependent variable (observations) as a list of floats.
                Must be the same length as data_x. Mutually exclusive with file_path.
        file_path: Path to a tabular data file (CSV, TSV, Excel, JSON, Parquet).
                   Mutually exclusive with data_x / data_y.
        x_column: Column name for the independent variable (required when file_path
                  has multiple numeric columns).
        y_column: Column name for the dependent variable (required when file_path
                  has multiple numeric columns).
        p0: Initial guesses for each parameter, one per entry in parameter_names.
            Defaults to all ones if omitted.
        bounds: Optional dict mapping each parameter name to [lower, upper] bound.
                Example: {"a": [0.0, 100.0], "b": [0.0, 10.0], "c": [-5.0, 5.0]}
        plot: If True, return a scatter plot of the data with the fitted model overlaid.
              Default True.

    Returns:
        Dict with keys:
          - parameters: dict mapping parameter name → fitted float value
          - std_errors: dict mapping parameter name → 1-σ uncertainty (sqrt of pcov diagonal)
          - residuals_rms: float — root-mean-square of the residuals
          - r_squared: float — coefficient of determination R²
          - n_points: int — number of data points used
          - covariance: list[list[float]] — full parameter covariance matrix
          - converged: bool — False if covariance is infinite (poor fit or bad initial guess)
          - plot: base64 PNG string (when plot=True)
          - error / suggestion: present only on failure

    Example:
        curve_fit_data(
            model_expr="a * exp(-b * x)",
            parameter_names=["a", "b"],
            data_x=[0, 1, 2, 3, 4],
            data_y=[2.0, 1.2, 0.74, 0.45, 0.27],
            p0=[2.0, 0.5]
        )
        # Returns parameters ≈ {"a": 2.0, "b": 0.5}
    """
    try:
        # ------------------------------------------------------------------
        # Load data
        # ------------------------------------------------------------------
        if data_x is not None or data_y is not None:
            if data_x is None or data_y is None:
                return {
                    "error": "Both data_x and data_y must be provided together.",
                    "suggestion": "Supply both data_x and data_y, or use file_path instead.",
                }
            if file_path is not None:
                return {
                    "error": "Provide either data_x/data_y or file_path, not both.",
                    "suggestion": "Remove file_path or remove data_x/data_y.",
                }
            x_arr = np.asarray(data_x, dtype=float)
            y_arr = np.asarray(data_y, dtype=float)
        elif file_path is not None:
            x_arr, _ = resolve_data(file_path=file_path, column=x_column)
            y_arr, _ = resolve_data(file_path=file_path, column=y_column)
        else:
            return {
                "error": "No data supplied. Provide data_x + data_y or file_path.",
                "suggestion": "Pass inline lists via data_x and data_y.",
            }

        if len(x_arr) != len(y_arr):
            return {
                "error": f"data_x length ({len(x_arr)}) ≠ data_y length ({len(y_arr)}).",
                "suggestion": "Ensure x and y arrays have the same number of elements.",
            }
        if len(x_arr) < len(parameter_names):
            return {
                "error": f"More parameters ({len(parameter_names)}) than data points ({len(x_arr)}).",
                "suggestion": "Provide more data or reduce the number of free parameters.",
            }

        # ------------------------------------------------------------------
        # Parse model
        # ------------------------------------------------------------------
        symbols: dict[str, str] = {"x": "x"}
        symbols.update({p: p for p in parameter_names})
        model_func_raw = parse_expr(model_expr, symbols)

        # scipy.curve_fit expects f(x, p1, p2, ...) → array
        def model_scipy(x_data: np.ndarray, *p_vals: float) -> np.ndarray:
            return np.asarray(model_func_raw(x_data, *p_vals), dtype=float)

        # ------------------------------------------------------------------
        # Build initial guess and bounds
        # ------------------------------------------------------------------
        if p0 is not None and len(p0) != len(parameter_names):
            return {
                "error": f"p0 has {len(p0)} elements but {len(parameter_names)} parameters were specified.",
                "suggestion": f"Provide p0 with exactly {len(parameter_names)} elements.",
            }
        p_init = np.ones(len(parameter_names), dtype=float) if p0 is None else np.array(p0, dtype=float)

        scipy_lower = [-np.inf] * len(parameter_names)
        scipy_upper = [np.inf] * len(parameter_names)
        if bounds:
            for i, p in enumerate(parameter_names):
                if p in bounds:
                    scipy_lower[i] = bounds[p][0]
                    scipy_upper[i] = bounds[p][1]

        # ------------------------------------------------------------------
        # Fit
        # ------------------------------------------------------------------
        popt, pcov = sp_opt.curve_fit(
            model_scipy, x_arr, y_arr,
            p0=p_init,
            bounds=(scipy_lower, scipy_upper),
            maxfev=10_000,
        )

        converged = bool(not np.any(np.isinf(pcov)))
        if converged:
            std_errors = np.sqrt(np.diag(pcov))
        else:
            std_errors = np.full(len(parameter_names), np.inf)
            logger.warning("curve_fit_data: covariance is infinite; fit may not have converged.")

        # Residuals and R²
        y_pred = model_scipy(x_arr, *popt)
        residuals = y_arr - y_pred
        rms = float(np.sqrt(np.mean(residuals ** 2)))
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
        r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")

        result: dict[str, Any] = {
            "parameters": {p: float(popt[i]) for i, p in enumerate(parameter_names)},
            "std_errors": {p: float(std_errors[i]) for i, p in enumerate(parameter_names)},
            "residuals_rms": rms,
            "r_squared": r_squared,
            "n_points": int(len(x_arr)),
            "covariance": [[float(pcov[r, c]) for c in range(len(parameter_names))]
                           for r in range(len(parameter_names))],
            "converged": converged,
        }

        # ------------------------------------------------------------------
        # Plot
        # ------------------------------------------------------------------
        if plot:
            x_dense = np.linspace(float(x_arr.min()), float(x_arr.max()), 500)
            y_dense = model_scipy(x_dense, *popt)

            param_label = ", ".join(
                f"{p}={float(popt[i]):.4g}" for i, p in enumerate(parameter_names)
            )

            fig, axes = plt.subplots(2, 1, figsize=(8, 7),
                                     gridspec_kw={"height_ratios": [3, 1]})
            ax, ax_res = axes

            ax.scatter(x_arr, y_arr, color="steelblue", s=30, zorder=5, label="data")
            ax.plot(x_dense, y_dense, color="red", linewidth=2,
                    label=f"fit: {param_label}")
            ax.set_ylabel("y")
            ax.set_title(f"Curve fit: {model_expr}\n(R²={r_squared:.4f})")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            ax_res.scatter(x_arr, residuals, color="gray", s=20, zorder=5)
            ax_res.axhline(0, color="red", linewidth=1, linestyle="--")
            ax_res.set_xlabel("x")
            ax_res.set_ylabel("residual")
            ax_res.set_title("Residuals")
            ax_res.grid(True, alpha=0.3)

            fig.tight_layout()
            result["plot"] = fig_to_base64(fig)

        return result

    except Exception as exc:
        logger.exception("curve_fit_data failed")
        return {
            "error": str(exc),
            "suggestion": (
                "Check that model_expr uses 'x' as the independent variable, "
                "parameter_names matches the parameter symbols in the expression, "
                "and p0 has the same length as parameter_names."
            ),
        }
