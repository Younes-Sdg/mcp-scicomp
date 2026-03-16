"""Tests for optimization tools: optimize and curve_fit_data."""

from __future__ import annotations

import math

import numpy as np

from mcp_scicomp.tools.optimization import curve_fit_data, optimize


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------


def _opt(objective: str, variables: list[str], **kw) -> dict:
    return optimize(objective=objective, variables=variables, plot=False, **kw)


def _fit(model_expr: str, parameter_names: list[str], **kw) -> dict:
    return curve_fit_data(model_expr=model_expr, parameter_names=parameter_names, plot=False, **kw)


# ---------------------------------------------------------------------------
# TestOptimize — minimize
# ---------------------------------------------------------------------------


class TestOptimizeMinimize:
    def test_simple_quadratic_1d(self) -> None:
        """Minimum of (x-3)**2 is at x=3 with value 0."""
        result = _opt("(x - 3)**2", ["x"], method="bfgs", x0=[0.0])
        assert "error" not in result
        assert abs(result["optimal_values"]["x"] - 3.0) < 1e-4
        assert abs(result["objective_value"]) < 1e-8

    def test_simple_quadratic_2d(self) -> None:
        """Minimum of (x-1)**2 + (y+2)**2 is at (1, -2)."""
        result = _opt("(x - 1)**2 + (y + 2)**2", ["x", "y"],
                      method="bfgs", x0=[0.0, 0.0])
        assert "error" not in result
        assert abs(result["optimal_values"]["x"] - 1.0) < 1e-4
        assert abs(result["optimal_values"]["y"] - (-2.0)) < 1e-4
        assert abs(result["objective_value"]) < 1e-8

    def test_nelder_mead_rosenbrock(self) -> None:
        """Rosenbrock minimum at (1, 1)."""
        result = _opt(
            "100*(y - x**2)**2 + (1 - x)**2",
            ["x", "y"],
            method="nelder_mead",
            x0=[-1.0, 1.0],
        )
        assert "error" not in result
        assert result["success"] is True
        assert abs(result["optimal_values"]["x"] - 1.0) < 1e-3
        assert abs(result["optimal_values"]["y"] - 1.0) < 1e-3

    def test_bfgs_with_fixed_params(self) -> None:
        """Minimum of a*x**2 + b*x with a=2, b=-4 is at x=1."""
        result = _opt(
            "a * x**2 + b * x",
            ["x"],
            method="bfgs",
            x0=[0.0],
            params={"a": 2.0, "b": -4.0},
        )
        assert "error" not in result
        assert abs(result["optimal_values"]["x"] - 1.0) < 1e-4

    def test_l_bfgs_b_with_bounds(self) -> None:
        """Minimum of x**2 in [-5, -1] is at x=-1 (boundary)."""
        result = _opt("x**2", ["x"], method="l_bfgs_b",
                      bounds={"x": [-5.0, -1.0]}, x0=[-3.0])
        assert "error" not in result
        assert abs(result["optimal_values"]["x"] - (-1.0)) < 1e-4

    def test_slsqp_with_equality_constraint(self) -> None:
        """Minimize x**2 + y**2 subject to x + y = 1 → min at x=y=0.5."""
        result = _opt(
            "x**2 + y**2",
            ["x", "y"],
            method="slsqp",
            constraints=[{"type": "eq", "expr": "x + y - 1"}],
            x0=[0.5, 0.5],
        )
        assert "error" not in result
        assert result["success"] is True
        assert abs(result["optimal_values"]["x"] - 0.5) < 1e-4
        assert abs(result["optimal_values"]["y"] - 0.5) < 1e-4
        assert abs(result["objective_value"] - 0.5) < 1e-4

    def test_differential_evolution_requires_bounds(self) -> None:
        result = _opt("x**2", ["x"], method="differential_evolution")
        assert "error" in result

    def test_differential_evolution_global_minimum(self) -> None:
        """Global minimum of sin(x)*x in [-10, 10]."""
        result = _opt(
            "sin(x) * x",
            ["x"],
            method="differential_evolution",
            bounds={"x": [-10.0, 10.0]},
        )
        assert "error" not in result
        assert result["success"] is True
        assert result["objective_value"] < -4.5  # global min ≈ -4.81 near x≈±4.71

    def test_returns_expected_keys(self) -> None:
        result = _opt("x**2", ["x"], method="bfgs")
        for key in ("optimal_values", "objective_value", "success",
                    "message", "n_iterations", "n_evaluations", "method", "mode"):
            assert key in result, f"Missing key: {key}"

    def test_invalid_method_returns_error(self) -> None:
        result = _opt("x**2", ["x"], method="gradient_descent_please")
        assert "error" in result

    def test_invalid_mode_returns_error(self) -> None:
        result = optimize(objective="x**2", variables=["x"], mode="solve", plot=False)
        assert "error" in result

    def test_empty_variables_returns_error(self) -> None:
        result = optimize(objective="1.0", variables=[], plot=False)
        assert "error" in result

    def test_nelder_mead_respects_bounds(self) -> None:
        """Nelder-Mead with bounds should constrain the solution."""
        result = _opt("(x - 10)**2", ["x"], method="nelder_mead",
                      bounds={"x": [0.0, 5.0]}, x0=[2.0])
        assert "error" not in result
        # Unconstrained optimum is at x=10, but bounds cap at 5
        assert result["optimal_values"]["x"] <= 5.0 + 1e-6

    def test_variable_param_name_collision_returns_error(self) -> None:
        result = _opt("a * a", ["a"], params={"a": 5.0})
        assert "error" in result
        assert "overlap" in result["error"].lower()

    def test_x0_wrong_length_returns_error(self) -> None:
        result = _opt("x**2 + y**2", ["x", "y"], x0=[0.0])
        assert "error" in result
        assert "x0" in result["error"]

    def test_constraint_with_fixed_param(self) -> None:
        """Constraint expression should be able to reference fixed params."""
        result = _opt(
            "a * x**2",
            ["x"],
            method="slsqp",
            params={"a": 1.0},
            constraints=[{"type": "ineq", "expr": "x - 2"}],  # x >= 2
            x0=[3.0],
        )
        assert "error" not in result
        assert result["optimal_values"]["x"] >= 2.0 - 1e-4


# ---------------------------------------------------------------------------
# TestOptimize — maximize
# ---------------------------------------------------------------------------


class TestOptimizeMaximize:
    def test_maximize_parabola(self) -> None:
        """Maximum of -(x-2)**2 + 5 is at x=2, value=5."""
        result = _opt("-(x - 2)**2 + 5", ["x"], method="bfgs",
                      mode="maximize", x0=[0.0])
        assert "error" not in result
        assert abs(result["optimal_values"]["x"] - 2.0) < 1e-4
        assert abs(result["objective_value"] - 5.0) < 1e-6

    def test_mode_in_result(self) -> None:
        result = _opt("-(x)**2", ["x"], mode="maximize")
        assert result.get("mode") == "maximize"


# ---------------------------------------------------------------------------
# TestOptimize — root finding
# ---------------------------------------------------------------------------


class TestOptimizeRoot:
    def test_root_x_squared_minus_4_brent(self) -> None:
        """x^2 - 4 = 0, root in [0, 3] → x=2."""
        result = _opt("x**2 - 4", ["x"], method="root",
                      bounds={"x": [0.0, 3.0]})
        assert "error" not in result
        assert abs(result["optimal_values"]["x"] - 2.0) < 1e-8
        assert abs(result["objective_value"]) < 1e-8

    def test_root_cubic_fsolve(self) -> None:
        """x^3 - x - 2 = 0 → root ≈ 1.5214."""
        result = _opt("x**3 - x - 2", ["x"], method="root", x0=[1.0])
        assert "error" not in result
        assert abs(result["optimal_values"]["x"] - 1.5213797) < 1e-4

    def test_root_mode_in_result(self) -> None:
        result = _opt("x - 1", ["x"], method="root", x0=[0.0])
        assert result.get("mode") == "root"


# ---------------------------------------------------------------------------
# TestOptimize — plots
# ---------------------------------------------------------------------------


class TestOptimizePlots:
    def test_1d_plot_returned(self) -> None:
        result = optimize(objective="(x-1)**2", variables=["x"],
                          method="bfgs", plot=True, x0=[0.0])
        assert "plot" in result
        assert isinstance(result["plot"], str) and len(result["plot"]) > 200

    def test_2d_plot_returned(self) -> None:
        result = optimize(objective="(x-1)**2 + (y-1)**2",
                          variables=["x", "y"],
                          method="bfgs", plot=True, x0=[0.0, 0.0])
        assert "plot" in result
        assert isinstance(result["plot"], str) and len(result["plot"]) > 200

    def test_no_plot_when_disabled(self) -> None:
        result = optimize(objective="x**2", variables=["x"],
                          method="bfgs", plot=False)
        assert "plot" not in result

    def test_3d_no_plot(self) -> None:
        """3-variable case should not produce a plot even with plot=True."""
        result = optimize(
            objective="x**2 + y**2 + z**2",
            variables=["x", "y", "z"],
            method="bfgs", plot=True, x0=[1.0, 1.0, 1.0],
        )
        assert "error" not in result
        assert "plot" not in result


# ---------------------------------------------------------------------------
# TestCurveFitData — happy path
# ---------------------------------------------------------------------------


class TestCurveFitData:
    # Exponential decay: y = 3 * exp(-0.5 * x)
    _X = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    _Y = [3.0 * math.exp(-0.5 * xi) for xi in _X]

    def test_exponential_decay_recovery(self) -> None:
        result = _fit(
            "a * exp(-b * x)",
            ["a", "b"],
            data_x=self._X,
            data_y=self._Y,
            p0=[1.0, 1.0],
        )
        assert "error" not in result
        assert abs(result["parameters"]["a"] - 3.0) < 1e-4
        assert abs(result["parameters"]["b"] - 0.5) < 1e-4

    def test_linear_model(self) -> None:
        """y = 2*x + 1."""
        xs = list(range(10))
        ys = [2.0 * xi + 1.0 for xi in xs]
        result = _fit("m * x + c", ["m", "c"], data_x=xs, data_y=ys, p0=[1.0, 0.0])
        assert "error" not in result
        assert abs(result["parameters"]["m"] - 2.0) < 1e-6
        assert abs(result["parameters"]["c"] - 1.0) < 1e-6
        assert result["r_squared"] > 0.9999

    def test_gaussian_fit(self) -> None:
        """Fit a Gaussian y = A * exp(-(x-mu)**2 / (2*sigma**2))."""
        xs = np.linspace(-3, 3, 30).tolist()
        A, mu, sigma = 5.0, 0.5, 1.2
        ys = [A * math.exp(-((xi - mu) ** 2) / (2 * sigma ** 2)) for xi in xs]
        result = _fit(
            "A * exp(-(x - mu)**2 / (2 * sigma**2))",
            ["A", "mu", "sigma"],
            data_x=xs,
            data_y=ys,
            p0=[1.0, 0.0, 1.0],
            bounds={"A": [0.0, 100.0], "sigma": [0.01, 10.0]},
        )
        assert "error" not in result
        assert abs(result["parameters"]["A"] - A) < 1e-3
        assert abs(result["parameters"]["mu"] - mu) < 1e-3
        assert abs(result["parameters"]["sigma"] - sigma) < 1e-3

    def test_returns_expected_keys(self) -> None:
        result = _fit(
            "a * x + b", ["a", "b"],
            data_x=[0.0, 1.0, 2.0],
            data_y=[1.0, 3.0, 5.0],
        )
        for key in ("parameters", "std_errors", "residuals_rms",
                    "r_squared", "n_points", "covariance", "converged"):
            assert key in result, f"Missing key: {key}"

    def test_r_squared_perfect_fit(self) -> None:
        xs = list(range(20))
        ys = [3.0 * xi + 1.0 for xi in xs]
        result = _fit("a * x + b", ["a", "b"], data_x=xs, data_y=ys, p0=[1.0, 0.0])
        assert "error" not in result
        assert result["r_squared"] > 0.9999

    def test_std_errors_are_finite_on_good_fit(self) -> None:
        xs = list(range(10))
        ys = [2.0 * xi + 1.0 for xi in xs]
        result = _fit("a * x + b", ["a", "b"], data_x=xs, data_y=ys, p0=[1.0, 0.0])
        assert "error" not in result
        assert result["converged"] is True
        for p in ["a", "b"]:
            assert math.isfinite(result["std_errors"][p])

    def test_rms_residuals_near_zero_perfect_model(self) -> None:
        xs = [0.0, 1.0, 2.0, 3.0]
        ys = [0.0, 1.0, 4.0, 9.0]  # y = x**2
        result = _fit("a * x**2 + b * x + c", ["a", "b", "c"],
                      data_x=xs, data_y=ys, p0=[1.0, 0.0, 0.0])
        assert "error" not in result
        assert result["residuals_rms"] < 1e-6

    def test_with_bounds(self) -> None:
        """Bounded fit should keep parameters inside bounds."""
        xs = list(range(10))
        ys = [3.0 * xi + 2.0 for xi in xs]
        result = _fit(
            "a * x + b", ["a", "b"],
            data_x=xs, data_y=ys, p0=[1.0, 1.0],
            bounds={"a": [0.0, 10.0], "b": [0.0, 10.0]},
        )
        assert "error" not in result
        assert 0.0 <= result["parameters"]["a"] <= 10.0
        assert 0.0 <= result["parameters"]["b"] <= 10.0


# ---------------------------------------------------------------------------
# TestCurveFitData — error handling
# ---------------------------------------------------------------------------


class TestCurveFitDataErrors:
    def test_mismatched_lengths(self) -> None:
        result = _fit("a * x + b", ["a", "b"],
                      data_x=[1.0, 2.0, 3.0], data_y=[1.0, 2.0])
        assert "error" in result

    def test_no_data_provided(self) -> None:
        result = _fit("a * x + b", ["a", "b"])
        assert "error" in result

    def test_both_inline_and_file_provided(self) -> None:
        result = _fit(
            "a * x + b", ["a", "b"],
            data_x=[1.0], data_y=[2.0],
            file_path="dummy.csv",
        )
        assert "error" in result

    def test_only_data_x_provided(self) -> None:
        result = _fit("a * x + b", ["a", "b"], data_x=[1.0, 2.0])
        assert "error" in result

    def test_more_params_than_points(self) -> None:
        result = _fit("a * x**2 + b * x + c", ["a", "b", "c"],
                      data_x=[0.0, 1.0], data_y=[1.0, 2.0])
        assert "error" in result

    def test_bad_expression_returns_error(self) -> None:
        result = _fit("a ** ** x", ["a"],
                      data_x=[1.0, 2.0, 3.0], data_y=[1.0, 2.0, 3.0])
        assert "error" in result

    def test_p0_wrong_length_returns_error(self) -> None:
        result = _fit("a * x + b", ["a", "b"],
                      data_x=[1.0, 2.0, 3.0], data_y=[1.0, 2.0, 3.0],
                      p0=[1.0])
        assert "error" in result
        assert "p0" in result["error"]


# ---------------------------------------------------------------------------
# TestCurveFitData — plots
# ---------------------------------------------------------------------------


class TestCurveFitDataPlots:
    def test_plot_returned(self) -> None:
        result = curve_fit_data(
            model_expr="a * x + b",
            parameter_names=["a", "b"],
            data_x=[0.0, 1.0, 2.0, 3.0],
            data_y=[1.0, 3.0, 5.0, 7.0],
            plot=True,
        )
        assert "plot" in result
        assert isinstance(result["plot"], str) and len(result["plot"]) > 200

    def test_no_plot_when_disabled(self) -> None:
        result = curve_fit_data(
            model_expr="a * x + b",
            parameter_names=["a", "b"],
            data_x=[0.0, 1.0, 2.0, 3.0],
            data_y=[1.0, 3.0, 5.0, 7.0],
            plot=False,
        )
        assert "plot" not in result
