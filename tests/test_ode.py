"""Tests for ODE tools."""

import math
from mcp_scicomp.tools.ode import solve_ode, phase_portrait


class TestSolveOde:
    def test_happy_path(self):
        result = solve_ode(equations=["y[1]", "-y[0]"], y0=[1.0, 0.0], plot=False)
        assert result.get("success") is True
        assert "t" in result
        assert "y" in result
        assert result["n_equations"] == 2
        assert result["plot"] is None

    def test_sho_analytical(self):
        """y[0] should approximate cos(t)."""
        result = solve_ode(
            equations=["y[1]", "-y[0]"],
            y0=[1.0, 0.0],
            t_span=[0.0, 2 * math.pi],
            n_points=1000,
            plot=False,
        )
        assert result.get("success") is True
        t_vals = result["t"]
        y0_vals = result["y"][0]
        for t, y in zip(t_vals[::100], y0_vals[::100]):
            assert abs(y - math.cos(t)) < 5e-3

    def test_params(self):
        """Damped oscillator: y'' + b*y' + k*y = 0."""
        result = solve_ode(
            equations=["y[1]", "-k*y[0] - b*y[1]"],
            y0=[1.0, 0.0],
            params={"k": 1.0, "b": 0.1},
            t_span=[0.0, 5.0],
            plot=False,
        )
        assert result.get("success") is True
        assert result["n_equations"] == 2

    def test_invalid_method(self):
        result = solve_ode(equations=["y[1]", "-y[0]"], y0=[1.0, 0.0], method="BOGUS", plot=False)
        assert "error" in result

    def test_dimension_mismatch(self):
        result = solve_ode(equations=["y[1]", "-y[0]"], y0=[1.0], plot=False)
        assert "error" in result

    def test_plot_false(self):
        result = solve_ode(equations=["y[1]", "-y[0]"], y0=[1.0, 0.0], plot=False)
        assert result.get("plot") is None


class TestPhasePortrait:
    def test_happy_path(self):
        result = phase_portrait(dx_expr="y", dy_expr="-x", plot=False)
        assert "dx_expr" in result
        assert "dy_expr" in result
        assert "n_trajectories" in result
        assert result["plot"] is None

    def test_with_trajectories(self):
        result = phase_portrait(
            dx_expr="y",
            dy_expr="-x",
            trajectories=[[1.0, 0.0], [-1.0, 0.5]],
            t_end=6.0,
            plot=False,
        )
        assert result["n_trajectories"] == 2

    def test_plot_false(self):
        result = phase_portrait(dx_expr="y", dy_expr="-x", plot=False)
        assert result["plot"] is None

    def test_invalid_expr(self):
        result = phase_portrait(dx_expr="y", dy_expr="@@@invalid@@@", plot=False)
        assert "error" in result
