"""ODE tools for mcp-scicomp."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from mcp_scicomp.app import mcp
from mcp_scicomp.plotting import fig_to_base64, quick_plot

logger = logging.getLogger(__name__)


@mcp.tool()
def solve_ode(
    equations: list[str],
    y0: list[float],
    t_span: list[float] = [0.0, 10.0],
    method: str = "RK45",
    n_points: int = 300,
    params: Optional[dict[str, float]] = None,
    plot: bool = True,
) -> dict[str, Any]:
    """Solve a system of ODEs as an initial value problem (IVP).

    Use this tool to integrate first-order ODE systems dy/dt = f(t, y).
    Higher-order ODEs must be rewritten as first-order systems. Returns the
    solution time series and a plot.

    Args:
        equations: List of RHS expressions, one per state variable. Use 't'
            for time, 'y[0]', 'y[1]', ... for state components, and any extra
            parameter names defined in `params`. Example for SHO:
            ["y[1]", "-y[0]"]
        y0: Initial conditions, one per equation. Example: [1.0, 0.0]
        t_span: [t_start, t_end]. Default [0.0, 10.0].
        method: Integration method — 'RK45' (default, non-stiff), 'RK23'
            (low-order non-stiff), 'Radau' (stiff), 'BDF' (stiff), 'LSODA'
            (auto stiff/non-stiff).
        n_points: Number of output time points. Default 300.
        params: Optional dict of named scalar parameters accessible in
            expressions. Example: {"omega": 2.0, "k": 0.5}
        plot: If True, return a time-series plot as base64 PNG. Default True.

    Returns:
        dict with keys:
            t (list[float]): time points
            y (list[list[float]]): solution components, y[i] is the i-th variable
            t_span (list), method (str), n_equations (int),
            success (bool), message (str),
            plot (str | None)

    Example:
        # Simple harmonic oscillator: y'' + y = 0  =>  y[0]'=y[1], y[1]'=-y[0]
        solve_ode(equations=["y[1]", "-y[0]"], y0=[1.0, 0.0], t_span=[0, 10])
    """
    try:
        if params is None:
            params = {}

        n = len(equations)
        if len(y0) != n:
            return {"error": f"len(y0)={len(y0)} must equal len(equations)={n}"}

        valid_methods = {"RK45", "RK23", "Radau", "BDF", "LSODA"}
        if method not in valid_methods:
            return {
                "error": f"method must be one of {valid_methods}",
                "suggestion": "Use RK45 for non-stiff, Radau/BDF for stiff systems",
            }

        # Build symbol namespace: t, y as IndexedBase, params
        t_sym = sp.Symbol("t")
        y_sym = sp.IndexedBase("y")
        sym_locals: dict[str, Any] = {"t": t_sym, "y": y_sym}
        sym_locals.update({k: sp.Symbol(k) for k in params})

        # Parse and lambdify each equation
        all_syms = [t_sym] + [y_sym[i] for i in range(n)] + [sp.Symbol(k) for k in params]
        funcs = [
            sp.lambdify(all_syms, sp.sympify(eq, locals=sym_locals), modules=["numpy"])
            for eq in equations
        ]
        param_vals = [params[k] for k in params]

        def rhs(t_val: float, y_val: np.ndarray) -> list[float]:
            args = [t_val] + list(y_val) + param_vals
            return [float(f(*args)) for f in funcs]

        t_eval = np.linspace(float(t_span[0]), float(t_span[1]), n_points)
        sol = solve_ivp(
            rhs,
            [float(t_span[0]), float(t_span[1])],
            list(y0),
            method=method,
            t_eval=t_eval,
        )

        t_list = [float(v) for v in sol.t]
        y_list = [[float(v) for v in row] for row in sol.y]

        result: dict[str, Any] = {
            "t": t_list,
            "y": y_list,
            "t_span": [float(t_span[0]), float(t_span[1])],
            "method": method,
            "n_equations": n,
            "success": bool(sol.success),
            "message": sol.message,
            "plot": None,
        }

        if not sol.success:
            result["suggestion"] = "Try a stiff solver (Radau or BDF) or reduce t_span."

        if plot:
            result["plot"] = quick_plot(
                t_list,
                y_list,
                title="ODE Solution",
                xlabel="t",
                ylabel="y",
                labels=[f"y[{i}]" for i in range(n)],
            )

        return result

    except Exception as exc:
        logger.exception("solve_ode failed")
        return {
            "error": str(exc),
            "suggestion": "Check expression syntax. Use ** for powers, y[0] y[1] for state vars.",
        }


@mcp.tool()
def phase_portrait(
    dx_expr: str,
    dy_expr: str,
    x_range: list[float] = [-3.0, 3.0],
    y_range: list[float] = [-3.0, 3.0],
    grid_points: int = 20,
    params: Optional[dict[str, float]] = None,
    trajectories: Optional[list[list[float]]] = None,
    t_end: float = 10.0,
    n_points: int = 500,
    plot: bool = True,
) -> dict[str, Any]:
    """Plot the 2D phase portrait of an autonomous ODE system dx/dt=f(x,y), dy/dt=g(x,y).

    Use this tool to visualize the vector field and solution trajectories of a
    2D autonomous system. Useful for analyzing fixed points, limit cycles, and stability.

    Args:
        dx_expr: Expression for dx/dt as a string in 'x' and 'y' (and params).
            Example: "y"
        dy_expr: Expression for dy/dt as a string in 'x' and 'y' (and params).
            Example: "mu*(1 - x**2)*y - x"
        x_range: [x_min, x_max] for the plot domain. Default [-3, 3].
        y_range: [y_min, y_max] for the plot domain. Default [-3, 3].
        grid_points: Number of grid points per axis for the vector field. Default 20.
        params: Optional dict of named scalar parameters. Example: {"mu": 1.0}
        trajectories: Optional list of [x0, y0] initial conditions to integrate
            and overlay. Example: [[1.0, 0.0], [-1.0, 0.5]]
        t_end: Integration time for each trajectory. Default 10.0.
        n_points: Time steps per trajectory. Default 500.
        plot: If True, return the phase portrait as base64 PNG. Default True.

    Returns:
        dict with keys:
            dx_expr (str), dy_expr (str), x_range, y_range,
            n_trajectories (int), plot (str | None)

    Example:
        # Van der Pol oscillator
        phase_portrait(dx_expr="y", dy_expr="mu*(1-x**2)*y - x",
                       params={"mu": 1.0}, trajectories=[[0.1, 0.1], [2.0, 0.0]])
    """
    try:
        if params is None:
            params = {}

        x_sym = sp.Symbol("x")
        y_sym = sp.Symbol("y")
        sym_locals: dict[str, Any] = {"x": x_sym, "y": y_sym}
        sym_locals.update({k: sp.Symbol(k) for k in params})

        all_syms = [x_sym, y_sym] + [sp.Symbol(k) for k in params]
        dx_func = sp.lambdify(all_syms, sp.sympify(dx_expr, locals=sym_locals), modules=["numpy"])
        dy_func = sp.lambdify(all_syms, sp.sympify(dy_expr, locals=sym_locals), modules=["numpy"])
        param_vals = [params[k] for k in params]

        def rhs2(t_val: float, state: np.ndarray) -> list[float]:
            xv, yv = state
            return [float(dx_func(xv, yv, *param_vals)), float(dy_func(xv, yv, *param_vals))]

        result: dict[str, Any] = {
            "dx_expr": dx_expr,
            "dy_expr": dy_expr,
            "x_range": [float(v) for v in x_range],
            "y_range": [float(v) for v in y_range],
            "n_trajectories": len(trajectories) if trajectories else 0,
            "plot": None,
        }

        if plot:
            fig, ax = plt.subplots(figsize=(7, 6))

            xs = np.linspace(float(x_range[0]), float(x_range[1]), grid_points)
            ys = np.linspace(float(y_range[0]), float(y_range[1]), grid_points)
            X, Y = np.meshgrid(xs, ys)
            DX = dx_func(X, Y, *param_vals)
            DY = dy_func(X, Y, *param_vals)
            norm = np.sqrt(DX**2 + DY**2)
            norm[norm == 0] = 1.0
            ax.streamplot(
                X, Y, DX, DY,
                color=np.log1p(norm), cmap="Blues",
                linewidth=0.8, arrowsize=1.2, density=1.2,
            )

            if trajectories:
                t_eval = np.linspace(0.0, float(t_end), n_points)
                cmap = plt.cm.tab10  # type: ignore[attr-defined]
                colors = [cmap(i / max(len(trajectories), 1)) for i in range(len(trajectories))]
                for ic, color in zip(trajectories, colors):
                    sol = solve_ivp(rhs2, [0.0, float(t_end)], list(ic), t_eval=t_eval, method="RK45")
                    if sol.success:
                        ax.plot(sol.y[0], sol.y[1], color=color, linewidth=1.5)
                        ax.plot(sol.y[0][0], sol.y[1][0], "o", color=color, markersize=5)

            ax.set_xlim(float(x_range[0]), float(x_range[1]))
            ax.set_ylim(float(y_range[0]), float(y_range[1]))
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title("Phase Portrait")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            result["plot"] = fig_to_base64(fig)

        return result

    except Exception as exc:
        logger.exception("phase_portrait failed")
        return {
            "error": str(exc),
            "suggestion": "Check expression syntax. Use 'x', 'y' as variable names and ** for powers.",
        }
