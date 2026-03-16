"""Tests for mcp_scicomp.tools.sde — simulate_sde and analyze_sde."""

from __future__ import annotations

import math

from mcp_scicomp.tools.sde import analyze_sde, simulate_sde


# ===========================================================================
# TestSimulateSde
# ===========================================================================


class TestSimulateSde:
    """Tests for the simulate_sde tool."""

    def test_euler_maruyama_happy_path(self):
        """Basic run with Euler-Maruyama returns all expected keys."""
        result = simulate_sde(
            drift="mu*x",
            diffusion="sigma*x",
            x0=1.0,
            t_end=0.5,
            n_steps=100,
            n_paths=50,
            params={"mu": 0.05, "sigma": 0.2},
            seed=42,
            plot=False,
        )
        assert "error" not in result
        assert result["method"] == "euler_maruyama"
        assert result["n_dims"] == 1
        assert result["n_paths"] == 50
        assert result["n_steps"] == 100
        assert "time_grid" in result
        assert "path_statistics" in result
        assert "final_distribution" in result
        assert "mean" in result["final_distribution"]
        assert "std" in result["final_distribution"]
        assert "skewness" in result["final_distribution"]
        assert "kurtosis" in result["final_distribution"]

    def test_gbm_analytical_mean(self):
        """GBM: E[X(T)] = x0 * exp(mu*T) with many paths."""
        mu, sigma, x0, T = 0.05, 0.2, 1.0, 1.0
        result = simulate_sde(
            drift="mu*x",
            diffusion="sigma*x",
            x0=x0,
            t_end=T,
            n_steps=500,
            n_paths=10000,
            params={"mu": mu, "sigma": sigma},
            seed=123,
            plot=False,
        )
        assert "error" not in result
        expected_mean = x0 * math.exp(mu * T)
        actual_mean = result["final_distribution"]["mean"]
        rel_error = abs(actual_mean - expected_mean) / expected_mean
        assert rel_error < 0.05, f"GBM mean: expected {expected_mean:.4f}, got {actual_mean:.4f}"

    def test_gbm_analytical_variance(self):
        """GBM: Var[X(T)] = x0^2 * exp(2*mu*T) * (exp(sigma^2*T) - 1)."""
        mu, sigma, x0, T = 0.05, 0.2, 1.0, 1.0
        result = simulate_sde(
            drift="mu*x",
            diffusion="sigma*x",
            x0=x0,
            t_end=T,
            n_steps=500,
            n_paths=10000,
            params={"mu": mu, "sigma": sigma},
            seed=456,
            plot=False,
        )
        assert "error" not in result
        expected_var = x0**2 * math.exp(2 * mu * T) * (math.exp(sigma**2 * T) - 1)
        actual_var = result["final_distribution"]["std"] ** 2
        rel_error = abs(actual_var - expected_var) / expected_var
        assert rel_error < 0.10, f"GBM variance: expected {expected_var:.4f}, got {actual_var:.4f}"

    def test_ou_stationary_distribution(self):
        """OU process: final mean ≈ mu, final std ≈ sigma/sqrt(2*theta) after long run."""
        theta, mu, sigma = 2.0, 3.0, 1.0
        result = simulate_sde(
            drift="-theta*(x - mu)",
            diffusion="sigma",
            x0=0.0,
            t_end=10.0,
            n_steps=2000,
            n_paths=5000,
            params={"theta": theta, "mu": mu, "sigma": sigma},
            seed=789,
            plot=False,
        )
        assert "error" not in result
        expected_mean = mu
        expected_std = sigma / math.sqrt(2 * theta)
        actual_mean = result["final_distribution"]["mean"]
        actual_std = result["final_distribution"]["std"]
        assert abs(actual_mean - expected_mean) < 0.15, f"OU mean: expected {expected_mean}, got {actual_mean}"
        assert abs(actual_std - expected_std) < 0.1, f"OU std: expected {expected_std:.3f}, got {actual_std:.3f}"

    def test_milstein_runs(self):
        """Milstein method runs without error."""
        result = simulate_sde(
            drift="mu*x",
            diffusion="sigma*x",
            x0=1.0,
            t_end=0.5,
            n_steps=100,
            n_paths=50,
            method="milstein",
            params={"mu": 0.05, "sigma": 0.2},
            seed=42,
            plot=False,
        )
        assert "error" not in result
        assert result["method"] == "milstein"

    def test_milstein_gbm_accuracy(self):
        """Milstein should be at least as accurate as Euler on GBM mean."""
        mu, sigma, x0, T = 0.05, 0.3, 1.0, 1.0
        expected_mean = x0 * math.exp(mu * T)

        euler_result = simulate_sde(
            drift="mu*x", diffusion="sigma*x", x0=x0, t_end=T,
            n_steps=200, n_paths=10000, method="euler_maruyama",
            params={"mu": mu, "sigma": sigma}, seed=100, plot=False,
        )
        milstein_result = simulate_sde(
            drift="mu*x", diffusion="sigma*x", x0=x0, t_end=T,
            n_steps=200, n_paths=10000, method="milstein",
            params={"mu": mu, "sigma": sigma}, seed=100, plot=False,
        )
        assert "error" not in euler_result
        assert "error" not in milstein_result

        euler_err = abs(euler_result["final_distribution"]["mean"] - expected_mean)
        milstein_err = abs(milstein_result["final_distribution"]["mean"] - expected_mean)
        # Milstein should not be significantly worse
        assert milstein_err < euler_err * 2.0

    def test_heun_runs(self):
        """Heun method runs without error."""
        result = simulate_sde(
            drift="-x",
            diffusion="0.5",
            x0=1.0,
            t_end=1.0,
            n_steps=100,
            n_paths=50,
            method="heun",
            seed=42,
            plot=False,
        )
        assert "error" not in result
        assert result["method"] == "heun"

    def test_system_sde(self):
        """2D system SDE runs and reports correct dimensions."""
        result = simulate_sde(
            drift=["-x0 + x1", "-x1"],
            diffusion=["0.3", "0.3"],
            x0=[1.0, 0.5],
            t_end=1.0,
            n_steps=100,
            n_paths=50,
            seed=42,
            plot=False,
        )
        assert "error" not in result
        assert result["n_dims"] == 2
        assert "x0" in result["path_statistics"]
        assert "x1" in result["path_statistics"]

    def test_milstein_rejects_system(self):
        """Milstein should reject multi-dimensional SDEs."""
        result = simulate_sde(
            drift=["-x0", "-x1"],
            diffusion=["0.5", "0.5"],
            x0=[1.0, 1.0],
            method="milstein",
            plot=False,
        )
        assert "error" in result

    def test_invalid_method(self):
        """Unknown method returns error."""
        result = simulate_sde(
            drift="x", diffusion="1.0", method="rk4", plot=False,
        )
        assert "error" in result

    def test_invalid_expression(self):
        """Unparseable expression returns error."""
        result = simulate_sde(
            drift="x @@ y !!", diffusion="1.0", plot=False,
        )
        assert "error" in result

    def test_mismatched_dimensions(self):
        """Dimension mismatch returns error."""
        result = simulate_sde(
            drift=["-x0", "-x1"],
            diffusion=["0.5"],
            plot=False,
        )
        assert "error" in result

    def test_nan_detection(self):
        """Explosive SDE (drift=x**2) should detect NaN."""
        result = simulate_sde(
            drift="x**2",
            diffusion="0.1",
            x0=1.0,
            t_end=5.0,
            n_steps=100,
            n_paths=10,
            seed=42,
            plot=False,
        )
        assert "error" in result
        assert "NaN" in result["error"] or "Inf" in result["error"]

    def test_analyze_options(self):
        """Each analyze option produces expected plot keys."""
        for option in ["paths", "distribution", "moments", "density"]:
            result = simulate_sde(
                drift="-x",
                diffusion="0.5",
                x0=1.0,
                t_end=1.0,
                n_steps=50,
                n_paths=50,
                analyze=[option],
                seed=42,
                plot=True,
            )
            assert "error" not in result, f"Failed for analyze={option}"
            assert option in result["plots"], f"Missing plot for analyze={option}"
            assert len(result["plots"][option]) > 0


# ===========================================================================
# TestAnalyzeSde
# ===========================================================================


class TestAnalyzeSde:
    """Tests for the analyze_sde tool."""

    def test_ou_stationary_normal(self):
        """OU process should have Normal stationary distribution."""
        result = analyze_sde(
            drift="-theta*(x - mu)",
            diffusion="sigma",
            params={"theta": 2.0, "mu": 3.0, "sigma": 1.0},
            plot=False,
        )
        assert "error" not in result
        sd = result["stationary_distribution"]
        assert sd["exists"] is True
        assert "Normal" in sd["identified_distribution"]
        # Mean should be 3.0, variance should be sigma^2/(2*theta) = 1/(4) = 0.25
        assert abs(sd["parameters"]["mean"] - 3.0) < 0.01
        assert abs(sd["parameters"]["variance"] - 0.25) < 0.01

    def test_ou_equilibrium_stable(self):
        """OU process has single stable equilibrium at mu."""
        result = analyze_sde(
            drift="-2*(x - 3)",
            diffusion="1.0",
            plot=False,
        )
        assert "error" not in result
        eq = result["equilibria"]
        assert len(eq) == 1
        assert abs(eq[0]["point"] - 3.0) < 0.01
        assert eq[0]["stability"] == "stable"

    def test_gbm_no_stationary(self):
        """GBM should not have a normalizable stationary distribution."""
        result = analyze_sde(
            drift="mu*x",
            diffusion="sigma*x",
            params={"mu": 0.05, "sigma": 0.2},
            plot=False,
        )
        assert "error" not in result
        sd = result["stationary_distribution"]
        # GBM: integral of 2*mu*x/(sigma*x)^2 = 2*mu/(sigma^2) * ln(x)
        # density ~ x^(2mu/sigma^2 - 2) which is not normalizable on (0, inf)
        # Our code should report exists=False or "unknown"
        assert sd["exists"] in [False, "unknown"]

    def test_kolmogorov_operators(self):
        """Kolmogorov operators should be non-empty LaTeX strings."""
        result = analyze_sde(
            drift="-x",
            diffusion="1.0",
            plot=False,
        )
        assert "error" not in result
        assert "kolmogorov" in result
        assert len(result["kolmogorov"]["forward_latex"]) > 0
        assert len(result["kolmogorov"]["backward_latex"]) > 0

    def test_moment_equations(self):
        """Moment equations should produce dE_dt_latex."""
        result = analyze_sde(
            drift="-2*x + 1",
            diffusion="0.5",
            plot=False,
        )
        assert "error" not in result
        assert "moment_equations" in result
        assert len(result["moment_equations"]["dE_dt_latex"]) > 0

    def test_invalid_expression(self):
        """Bad expression returns error."""
        result = analyze_sde(
            drift="sin(x +++ )", diffusion="1.0", plot=False,
        )
        assert "error" in result

    def test_system_equilibria(self):
        """2D system: equilibrium with eigenvalue-based stability."""
        result = analyze_sde(
            drift=["-x0", "-2*x1"],
            diffusion=["0.5", "0.5"],
            plot=False,
        )
        assert "error" not in result
        assert result["n_dims"] == 2
        eq = result["equilibria"]
        assert len(eq) >= 1
        # Origin should be stable (both eigenvalues negative)
        origin = eq[0]
        assert origin["stability"] == "stable"
        assert "eigenvalues" in origin

    def test_plot_false(self):
        """plot=False → plot is None."""
        result = analyze_sde(
            drift="-x",
            diffusion="1.0",
            plot=False,
        )
        assert "error" not in result
        assert result["plot"] is None
