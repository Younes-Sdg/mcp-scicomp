"""PDE tools for mcp-scicomp: 1-D parabolic/hyperbolic PDEs and 2-D Laplace equation."""

from __future__ import annotations

import logging
from typing import Any, Optional, Union

import numpy as np
import sympy as sp
from sympy import Symbol

from mcp_scicomp.app import mcp
from mcp_scicomp.plotting import fig_to_base64

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------

def _parse_bc(
    bc_val: Union[float, int, str],
    grid: np.ndarray,
    var_name: str,
) -> np.ndarray:
    """Convert a boundary-condition spec to a numpy array of length len(grid).

    Accepts:
    - float/int  → constant array
    - str        → SymPy expression in *var_name*, evaluated on *grid*
    """
    if isinstance(bc_val, (int, float)):
        return np.full(len(grid), float(bc_val))
    if isinstance(bc_val, str):
        sym = Symbol(var_name)
        expr = sp.sympify(bc_val.replace("^", "**"), locals={var_name: sym})
        func = sp.lambdify([sym], expr, modules=["numpy"])
        result = np.asarray(func(grid), dtype=float)
        if result.ndim == 0:
            result = np.full(len(grid), float(result))
        return result
    raise TypeError(f"BC value must be float, int, or str; got {type(bc_val)}")


# ---------------------------------------------------------------------------
# Tool 1: solve_pde_1d
# ---------------------------------------------------------------------------

@mcp.tool()
def solve_pde_1d(
    pde_type: str = "heat",
    spatial_rhs: str = "",
    initial_condition: str = "sin(pi*x)",
    initial_velocity: str = "0",
    boundary_left: float = 0.0,
    boundary_right: float = 0.0,
    x_range: list[float] = [0.0, 1.0],
    t_end: float = 1.0,
    n_x: int = 50,
    n_t: int = 100,
    params: dict[str, float] = {},
    plot: bool = True,
) -> dict[str, Any]:
    """Solve a 1-D time-dependent PDE using the Method of Lines (MOL).

    Use this tool when you need to simulate heat diffusion, wave propagation,
    or any custom parabolic/hyperbolic PDE in one spatial dimension.

    The PDE is discretised in space with central finite differences and
    integrated in time with scipy's RK45 solver.

    Convenience presets (set pde_type to trigger them):
      - "heat"  → u_t = alpha * d2u_dx2,  alpha defaults to 0.01
      - "wave"  → u_tt = c**2 * d2u_dx2,  c defaults to 1.0

    For custom equations set pde_type to "parabolic" (u_t = rhs) or
    "hyperbolic" (u_tt = rhs) and supply spatial_rhs.

    Args:
        pde_type: One of "heat", "wave", "parabolic", "hyperbolic".
            "heat" and "wave" are convenience presets. Default "heat".
        spatial_rhs: Right-hand side expression in terms of u, du_dx, d2u_dx2,
            x, t, and any keys in *params*. Leave blank to use the preset.
            Example: "alpha*d2u_dx2 - beta*u".
        initial_condition: SymPy expression in x giving u(x, 0). Default "sin(pi*x)".
        initial_velocity: SymPy expression in x giving u_t(x, 0). Only used for
            hyperbolic PDEs. Default "0".
        boundary_left: Dirichlet BC at x=x_range[0]. Default 0.0.
        boundary_right: Dirichlet BC at x=x_range[1]. Default 0.0.
        x_range: [x_min, x_max] spatial domain. Default [0.0, 1.0].
        t_end: Final time. Must be > 0. Default 1.0.
        n_x: Number of spatial grid points. Must be >= 3. Default 50.
        n_t: Number of output time points for solution storage. Default 100.
        params: Extra named parameters referenced in spatial_rhs, e.g. {"alpha": 0.01}.
        plot: If True, return a heatmap (x vs t) as base64 PNG. Default True.

    Returns:
        dict with keys:
            pde_type (str), x (list[float]), t (list[float]),
            u_final (list[float]), u_min (float), u_max (float), u_mean (float),
            n_x (int), n_t (int),
            solver_success (bool), solver_message (str), solver_nfev (int),
            plot (str | None)

    Example:
        solve_pde_1d(pde_type="heat", t_end=0.5, n_x=60, params={"alpha": 0.05})
        solve_pde_1d(pde_type="wave", initial_condition="sin(pi*x)", t_end=2.0)
        solve_pde_1d(
            pde_type="parabolic",
            spatial_rhs="0.1*d2u_dx2 - u",
            params={},
        )
    """
    # ---- preset expansion (before try, mutate local copies) ----
    params = dict(params)  # defensive copy
    if pde_type == "heat":
        pde_type = "parabolic"
        if not spatial_rhs or spatial_rhs == "d2u_dx2":
            spatial_rhs = "alpha*d2u_dx2"
        params.setdefault("alpha", 0.01)
    elif pde_type == "wave":
        pde_type = "hyperbolic"
        if not spatial_rhs or spatial_rhs == "d2u_dx2":
            spatial_rhs = "c**2*d2u_dx2"
        params.setdefault("c", 1.0)

    try:
        from scipy.integrate import solve_ivp

        # ---- validation ----
        if pde_type not in {"parabolic", "hyperbolic"}:
            return {
                "error": f"Unknown pde_type '{pde_type}'.",
                "suggestion": "Use 'heat', 'wave', 'parabolic', or 'hyperbolic'.",
            }
        if n_x < 3:
            return {"error": "n_x must be >= 3.", "suggestion": "Increase n_x (e.g. 50)."}
        if t_end <= 0:
            return {"error": "t_end must be > 0.", "suggestion": "Set t_end to a positive value."}
        if len(x_range) != 2 or x_range[1] <= x_range[0]:
            return {
                "error": "x_range must be [x_min, x_max] with x_max > x_min.",
                "suggestion": "Example: x_range=[0.0, 1.0].",
            }

        x_grid = np.linspace(x_range[0], x_range[1], n_x)
        dx = x_grid[1] - x_grid[0]
        t_eval = np.linspace(0.0, t_end, max(n_t, 2))

        # ---- expression parsing ----
        sym_dict: dict[str, Any] = {
            "u": Symbol("u"),
            "du_dx": Symbol("du_dx"),
            "d2u_dx2": Symbol("d2u_dx2"),
            "x": Symbol("x"),
            "t": Symbol("t"),
            **{k: Symbol(k) for k in params},
        }
        try:
            rhs_expr = sp.sympify(spatial_rhs.replace("^", "**"), locals=sym_dict)
        except Exception as exc:
            return {
                "error": f"Cannot parse spatial_rhs '{spatial_rhs}': {exc}",
                "suggestion": "Use Python-style expressions with **, not ^.",
            }

        arg_syms = [sym_dict[k] for k in ("u", "du_dx", "d2u_dx2", "x", "t")] + [
            sym_dict[k] for k in params
        ]
        rhs_func = sp.lambdify(arg_syms, rhs_expr, modules=["numpy"])
        param_values = [params[k] for k in params]

        # ---- initial condition ----
        try:
            ic_expr = sp.sympify(
                initial_condition.replace("^", "**"), locals={"x": Symbol("x")}
            )
            ic_func = sp.lambdify([Symbol("x")], ic_expr, modules=["numpy"])
            u0_raw = ic_func(x_grid)
            u0 = np.broadcast_to(np.asarray(u0_raw, dtype=float), x_grid.shape).copy()
        except Exception as exc:
            return {
                "error": f"Cannot parse initial_condition '{initial_condition}': {exc}",
                "suggestion": "Use a SymPy expression in x, e.g. 'sin(pi*x)'.",
            }

        # ---- build ODE system ----
        if pde_type == "parabolic":
            def _rhs_parabolic(t_val: float, y: np.ndarray) -> np.ndarray:
                y = y.copy()
                y[0] = boundary_left
                y[-1] = boundary_right
                du_dx = np.zeros(n_x)
                d2u_dx2 = np.zeros(n_x)
                du_dx[1:-1] = (y[2:] - y[:-2]) / (2.0 * dx)
                d2u_dx2[1:-1] = (y[2:] - 2.0 * y[1:-1] + y[:-2]) / dx**2
                rhs = np.asarray(
                    rhs_func(y, du_dx, d2u_dx2, x_grid, t_val, *param_values),
                    dtype=float,
                )
                if rhs.ndim == 0:
                    rhs = np.full(n_x, float(rhs))
                rhs[0] = rhs[-1] = 0.0
                return rhs

            sol = solve_ivp(
                _rhs_parabolic,
                [0.0, t_end],
                u0,
                t_eval=t_eval,
                method="RK45",
                vectorized=False,
            )
            sol_matrix = sol.y.T  # (n_t, n_x)

        else:  # hyperbolic
            try:
                vel_expr = sp.sympify(
                    initial_velocity.replace("^", "**"), locals={"x": Symbol("x")}
                )
                vel_func = sp.lambdify([Symbol("x")], vel_expr, modules=["numpy"])
                v0_raw = vel_func(x_grid)
                v0 = np.broadcast_to(np.asarray(v0_raw, dtype=float), x_grid.shape).copy()
            except Exception as exc:
                return {
                    "error": f"Cannot parse initial_velocity '{initial_velocity}': {exc}",
                    "suggestion": "Use a SymPy expression in x, e.g. '0' or 'sin(pi*x)'.",
                }

            y0 = np.concatenate([u0, v0])

            def _rhs_hyperbolic(t_val: float, y: np.ndarray) -> np.ndarray:
                u_vec = y[:n_x].copy()
                v_vec = y[n_x:].copy()
                u_vec[0] = boundary_left
                u_vec[-1] = boundary_right
                v_vec[0] = v_vec[-1] = 0.0
                du_dx = np.zeros(n_x)
                d2u_dx2 = np.zeros(n_x)
                du_dx[1:-1] = (u_vec[2:] - u_vec[:-2]) / (2.0 * dx)
                d2u_dx2[1:-1] = (u_vec[2:] - 2.0 * u_vec[1:-1] + u_vec[:-2]) / dx**2
                v_dot = np.asarray(
                    rhs_func(u_vec, du_dx, d2u_dx2, x_grid, t_val, *param_values),
                    dtype=float,
                )
                if v_dot.ndim == 0:
                    v_dot = np.full(n_x, float(v_dot))
                v_dot[0] = v_dot[-1] = 0.0
                return np.concatenate([v_vec, v_dot])

            sol = solve_ivp(
                _rhs_hyperbolic,
                [0.0, t_end],
                y0,
                t_eval=t_eval,
                method="RK45",
                vectorized=False,
            )
            sol_matrix = sol.y[:n_x, :].T  # (n_t, n_x)

        # ---- post-solve checks ----
        if not sol.success:
            return {
                "error": f"solve_ivp failed: {sol.message}",
                "suggestion": "Try increasing n_t, decreasing t_end, or checking the RHS expression.",
            }
        u_final = sol_matrix[-1, :]
        if np.any(np.isnan(u_final)):
            return {
                "error": "NaN values detected in the solution.",
                "suggestion": "Increase n_x, reduce t_end, or check stability of the scheme.",
            }

        # ---- plot ----
        plot_b64: Optional[str] = None
        if plot:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 5))
            pcm = ax.pcolormesh(x_grid, t_eval, sol_matrix, shading="auto", cmap="viridis")
            fig.colorbar(pcm, ax=ax, label="u(x, t)")
            ax.set_xlabel("x")
            ax.set_ylabel("t")
            ax.set_title(f"PDE solution ({pde_type}): {spatial_rhs}")
            fig.tight_layout()
            plot_b64 = fig_to_base64(fig)

        return {
            "pde_type": pde_type,
            "x": [float(v) for v in x_grid],
            "t": [float(v) for v in t_eval],
            "u_final": [float(v) for v in u_final],
            "u_min": float(np.nanmin(sol_matrix)),
            "u_max": float(np.nanmax(sol_matrix)),
            "u_mean": float(np.nanmean(sol_matrix)),
            "n_x": int(n_x),
            "n_t": int(len(t_eval)),
            "solver_success": bool(sol.success),
            "solver_message": str(sol.message),
            "solver_nfev": int(sol.nfev),
            "plot": plot_b64,
        }

    except Exception as exc:
        logger.exception("solve_pde_1d failed")
        return {
            "error": str(exc),
            "suggestion": "Check expression syntax, x_range, t_end, and n_x.",
        }


# ---------------------------------------------------------------------------
# Tool 2: solve_laplace_2d
# ---------------------------------------------------------------------------

@mcp.tool()
def solve_laplace_2d(
    boundary_conditions: Optional[dict[str, Any]] = None,
    nx: int = 30,
    ny: int = 30,
    max_iterations: int = 5000,
    tolerance: float = 1e-5,
    plot: bool = True,
) -> dict[str, Any]:
    """Solve the 2-D Laplace equation ∇²u = 0 via Gauss-Seidel iteration.

    Use this tool to compute steady-state temperature distributions, electrostatic
    potentials, or any harmonic function on a unit square [0,1]×[0,1] with
    specified boundary values.

    Args:
        boundary_conditions: Dict with up to four keys: "top", "bottom", "left", "right".
            Each value can be:
            - float/int  → constant BC (e.g. 1.0)
            - str        → expression in x (for top/bottom) or y (for left/right)
                           e.g. "sin(pi*x)" for the top edge.
            Missing keys default to 0.0. Default: {"top": 1.0, others: 0.0}.
        nx: Number of grid points in x. Must be >= 3. Default 30.
        ny: Number of grid points in y. Must be >= 3. Default 30.
        max_iterations: Maximum Gauss-Seidel iterations. Default 5000.
        tolerance: Convergence threshold (max pointwise change). Default 1e-5.
        plot: If True, return a filled contour plot as base64 PNG. Default True.

    Returns:
        dict with keys:
            u_min (float), u_max (float), u_mean (float),
            nx (int), ny (int),
            convergence: {
                converged (bool),
                iterations (int),
                final_residual (float),
            },
            plot (str | None)

    Example:
        solve_laplace_2d(boundary_conditions={"top": 1.0, "bottom": 0.0,
                                               "left": 0.0, "right": 0.0})
        solve_laplace_2d(boundary_conditions={"top": "sin(pi*x)"}, nx=50, ny=50)
    """
    try:
        # ---- validation ----
        if nx < 3:
            return {"error": "nx must be >= 3.", "suggestion": "Set nx to at least 3."}
        if ny < 3:
            return {"error": "ny must be >= 3.", "suggestion": "Set ny to at least 3."}
        if tolerance <= 0:
            return {
                "error": "tolerance must be > 0.",
                "suggestion": "Set tolerance to a small positive value such as 1e-5.",
            }
        if max_iterations < 1:
            return {
                "error": "max_iterations must be >= 1.",
                "suggestion": "Set max_iterations to a positive integer.",
            }

        # ---- grids ----
        x_grid = np.linspace(0.0, 1.0, nx)
        y_grid = np.linspace(0.0, 1.0, ny)

        # ---- boundary conditions ----
        _DEFAULT_BC: dict[str, Any] = {"top": 1.0, "bottom": 0.0, "left": 0.0, "right": 0.0}
        bc = {**_DEFAULT_BC, **(boundary_conditions or {})}

        try:
            bc_top = _parse_bc(bc["top"], x_grid, "x")       # u[-1, :]
            bc_bottom = _parse_bc(bc["bottom"], x_grid, "x")  # u[0, :]
            bc_left = _parse_bc(bc["left"], y_grid, "y")      # u[:, 0]
            bc_right = _parse_bc(bc["right"], y_grid, "y")    # u[:, -1]
        except Exception as exc:
            return {
                "error": f"Cannot parse boundary condition: {exc}",
                "suggestion": "Use float, int, or a SymPy string expression in x or y.",
            }

        # ---- initialise grid ----
        u = np.zeros((ny, nx))
        u[-1, :] = bc_top
        u[0, :] = bc_bottom
        u[:, 0] = bc_left
        u[:, -1] = bc_right

        # ---- Gauss-Seidel iteration ----
        converged = False
        final_residual = float("inf")
        iterations = 0

        for iterations in range(1, max_iterations + 1):
            u_interior_old = u[1:-1, 1:-1].copy()
            u[1:-1, 1:-1] = 0.25 * (
                u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2]
            )
            final_residual = float(np.max(np.abs(u[1:-1, 1:-1] - u_interior_old)))
            if final_residual < tolerance:
                converged = True
                break

        # ---- plot ----
        plot_b64: Optional[str] = None
        if plot:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7, 6))
            cf = ax.contourf(x_grid, y_grid, u, levels=20, cmap="viridis")
            fig.colorbar(cf, ax=ax, label="u(x, y)")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title("2-D Laplace Equation Solution")
            fig.tight_layout()
            plot_b64 = fig_to_base64(fig)

        return {
            "u_min": float(np.min(u)),
            "u_max": float(np.max(u)),
            "u_mean": float(np.mean(u)),
            "nx": int(nx),
            "ny": int(ny),
            "convergence": {
                "converged": bool(converged),
                "iterations": int(iterations),
                "final_residual": float(final_residual),
            },
            "plot": plot_b64,
        }

    except Exception as exc:
        logger.exception("solve_laplace_2d failed")
        return {
            "error": str(exc),
            "suggestion": "Check boundary_conditions format, nx/ny >= 3, tolerance > 0.",
        }
