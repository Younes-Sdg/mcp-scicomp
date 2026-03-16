"""Tests for mcp_scicomp.tools.pde."""

from __future__ import annotations

import math


from mcp_scicomp.tools.pde import solve_laplace_2d, solve_pde_1d


# ===========================================================================
# solve_pde_1d
# ===========================================================================

class TestSolvePde1d:
    def test_heat_preset_happy_path(self):
        result = solve_pde_1d(pde_type="heat", t_end=0.5, n_x=30, n_t=20, plot=False)
        assert "error" not in result
        assert "u_final" in result
        assert "x" in result
        assert len(result["u_final"]) == 30

    def test_heat_analytical(self):
        # u(x, t) = sin(pi*x) * exp(-alpha * pi^2 * t)
        # IC: sin(pi*x), BCs: 0 at x=0 and x=1, alpha=0.01
        # At x~0.5, t=1: u ~ exp(-0.01*pi^2) * sin(pi*0.5) ~ 0.9044
        alpha = 0.01
        result = solve_pde_1d(
            pde_type="heat",
            initial_condition="sin(pi*x)",
            t_end=1.0,
            n_x=50,
            n_t=200,
            params={"alpha": alpha},
            plot=False,
        )
        assert "error" not in result
        x = result["x"]
        u_final = result["u_final"]
        idx = min(range(len(x)), key=lambda i: abs(x[i] - 0.5))
        expected = math.exp(-alpha * math.pi**2) * math.sin(math.pi * x[idx])
        assert abs(u_final[idx] - expected) < 0.02

    def test_wave_preset_happy_path(self):
        result = solve_pde_1d(pde_type="wave", t_end=0.5, n_x=40, n_t=50, plot=False)
        assert "error" not in result
        assert result["pde_type"] == "hyperbolic"
        assert "u_final" in result

    def test_wave_analytical(self):
        # u(x,t) = sin(pi*x)*cos(c*pi*t), c=1
        # At x~0.5, t=0.5: cos(pi*0.5)*sin(pi/2) = 0 * 1 = 0
        result = solve_pde_1d(
            pde_type="wave",
            initial_condition="sin(pi*x)",
            initial_velocity="0",
            t_end=0.5,
            n_x=60,
            n_t=200,
            params={"c": 1.0},
            plot=False,
        )
        assert "error" not in result
        x = result["x"]
        u_final = result["u_final"]
        idx = min(range(len(x)), key=lambda i: abs(x[i] - 0.5))
        assert abs(u_final[idx]) < 0.05

    def test_custom_parabolic(self):
        result = solve_pde_1d(
            pde_type="parabolic",
            spatial_rhs="0.1*d2u_dx2",
            initial_condition="sin(pi*x)",
            t_end=0.2,
            n_x=40,
            n_t=50,
            plot=False,
        )
        assert "error" not in result
        assert result["solver_success"] is True

    def test_invalid_pde_type(self):
        result = solve_pde_1d(pde_type="elliptic", plot=False)
        assert "error" in result

    def test_invalid_n_x(self):
        result = solve_pde_1d(pde_type="heat", n_x=1, plot=False)
        assert "error" in result

    def test_invalid_t_end(self):
        result = solve_pde_1d(pde_type="heat", t_end=-1.0, plot=False)
        assert "error" in result

    def test_invalid_x_range(self):
        result = solve_pde_1d(pde_type="heat", x_range=[1.0, 0.0], plot=False)
        assert "error" in result

    def test_unparseable_rhs(self):
        result = solve_pde_1d(
            pde_type="parabolic",
            spatial_rhs="@@@",
            plot=False,
        )
        assert "error" in result


# ===========================================================================
# solve_laplace_2d
# ===========================================================================

class TestSolveLaplace2d:
    def test_happy_path(self):
        result = solve_laplace_2d(plot=False)
        assert "error" not in result
        assert "u_min" in result
        assert "convergence" in result

    def test_convergence_keys(self):
        result = solve_laplace_2d(plot=False)
        conv = result["convergence"]
        assert "converged" in conv
        assert "iterations" in conv
        assert "final_residual" in conv

    def test_constant_solution(self):
        bcs = {"top": 5.0, "bottom": 5.0, "left": 5.0, "right": 5.0}
        result = solve_laplace_2d(boundary_conditions=bcs, nx=20, ny=20, plot=False)
        assert "error" not in result
        assert abs(result["u_min"] - 5.0) < 0.01
        assert abs(result["u_max"] - 5.0) < 0.01

    def test_zero_solution(self):
        bcs = {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0}
        result = solve_laplace_2d(boundary_conditions=bcs, nx=15, ny=15, plot=False)
        assert "error" not in result
        assert result["u_max"] < 1e-6

    def test_string_boundary(self):
        bcs = {"top": "sin(pi*x)", "bottom": 0.0, "left": 0.0, "right": 0.0}
        result = solve_laplace_2d(boundary_conditions=bcs, nx=25, ny=25, plot=False)
        assert "error" not in result

    def test_maximum_principle(self):
        bcs = {"top": 1.0, "bottom": 0.0, "left": 0.2, "right": 0.8}
        result = solve_laplace_2d(boundary_conditions=bcs, nx=25, ny=25, plot=False)
        assert "error" not in result
        assert result["u_min"] >= -1e-10
        assert result["u_max"] <= 1.0 + 1e-10

    def test_invalid_nx(self):
        result = solve_laplace_2d(nx=1, plot=False)
        assert "error" in result

    def test_invalid_tolerance(self):
        result = solve_laplace_2d(tolerance=-1.0, plot=False)
        assert "error" in result
