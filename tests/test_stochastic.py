"""Tests for mcp_scicomp.tools.stochastic."""

from __future__ import annotations

import math

from mcp_scicomp.tools.stochastic import (
    markov_chain_analysis,
    simulate_brownian_motion,
    simulate_ornstein_uhlenbeck,
    simulate_poisson_process,
)


# ---------------------------------------------------------------------------
# simulate_brownian_motion
# ---------------------------------------------------------------------------

class TestSimulateBrownianMotion:
    def test_abm_happy_path(self):
        result = simulate_brownian_motion(
            t_end=1.0, n_steps=200, n_paths=10, mu=0.0, sigma=1.0, x0=0.0,
            geometric=False, plot=False, seed=42,
        )
        assert "error" not in result
        for key in ("final_mean", "final_std", "final_min", "final_max", "final_median"):
            assert key in result
        assert result["n_paths"] == 10

    def test_abm_mean_variance(self):
        T = 1.0
        mu = 0.5
        sigma = 1.2
        result = simulate_brownian_motion(
            t_end=T, n_steps=500, n_paths=10000, mu=mu, sigma=sigma, x0=2.0,
            geometric=False, plot=False, seed=0,
        )
        assert "error" not in result
        expected_mean = 2.0 + mu * T
        expected_var = sigma**2 * T
        actual_mean = result["final_mean"]
        actual_std = result["final_std"]
        assert abs(actual_mean - expected_mean) < 0.05 * abs(expected_mean) + 0.1
        assert abs(actual_std**2 - expected_var) / expected_var < 0.05

    def test_gbm_mean(self):
        T = 1.0
        mu = 0.3
        sigma = 0.2
        x0 = 10.0
        result = simulate_brownian_motion(
            t_end=T, n_steps=500, n_paths=10000, mu=mu, sigma=sigma, x0=x0,
            geometric=True, plot=False, seed=7,
        )
        assert "error" not in result
        expected_mean = x0 * math.exp(mu * T)
        actual_mean = result["final_mean"]
        assert abs(actual_mean - expected_mean) / expected_mean < 0.05

    def test_invalid_sigma(self):
        result = simulate_brownian_motion(sigma=-1.0, plot=False)
        assert "error" in result

    def test_invalid_n_steps(self):
        result = simulate_brownian_motion(n_steps=0, plot=False)
        assert "error" in result


# ---------------------------------------------------------------------------
# simulate_ornstein_uhlenbeck
# ---------------------------------------------------------------------------

class TestSimulateOrnsteinUhlenbeck:
    def test_happy_path(self):
        result = simulate_ornstein_uhlenbeck(
            theta=1.0, mu=0.0, sigma=1.0, x0=0.0,
            t_end=5.0, n_steps=500, n_paths=10, plot=False, seed=42,
        )
        assert "error" not in result
        for key in (
            "final_mean", "final_std",
            "theoretical_stationary_mean", "theoretical_stationary_std",
        ):
            assert key in result

    def test_stationary_mean(self):
        theta = 2.0
        mu = 5.0
        result = simulate_ornstein_uhlenbeck(
            theta=theta, mu=mu, sigma=0.5, x0=0.0,
            t_end=20.0, n_steps=2000, n_paths=500, plot=False, seed=1,
        )
        assert "error" not in result
        assert abs(result["final_mean"] - mu) < 0.2

    def test_theoretical_stationary_std(self):
        theta = 3.0
        sigma = 1.5
        result = simulate_ornstein_uhlenbeck(
            theta=theta, mu=0.0, sigma=sigma, x0=0.0,
            t_end=5.0, n_steps=500, n_paths=10, plot=False, seed=0,
        )
        assert "error" not in result
        expected = sigma / math.sqrt(2 * theta)
        assert abs(result["theoretical_stationary_std"] - expected) < 1e-10

    def test_invalid_theta(self):
        result = simulate_ornstein_uhlenbeck(theta=-1.0, plot=False)
        assert "error" in result


# ---------------------------------------------------------------------------
# simulate_poisson_process
# ---------------------------------------------------------------------------

class TestSimulatePoissonProcess:
    def test_happy_path(self):
        result = simulate_poisson_process(
            rate=2.0, t_end=5.0, n_paths=5, plot=False, seed=42,
        )
        assert "error" not in result
        assert len(result["event_counts"]) == 5
        for key in (
            "mean_events", "std_events", "expected_events",
            "mean_interarrival", "expected_interarrival",
        ):
            assert key in result

    def test_expected_count(self):
        rate = 3.0
        t_end = 10.0
        result = simulate_poisson_process(
            rate=rate, t_end=t_end, n_paths=1000, plot=False, seed=5,
        )
        assert "error" not in result
        expected = rate * t_end
        assert abs(result["mean_events"] - expected) / expected < 0.10

    def test_expected_interarrival(self):
        rate = 4.0
        result = simulate_poisson_process(
            rate=rate, t_end=20.0, n_paths=500, plot=False, seed=9,
        )
        assert "error" not in result
        expected_ia = 1.0 / rate
        assert abs(result["mean_interarrival"] - expected_ia) / expected_ia < 0.10

    def test_invalid_rate(self):
        result = simulate_poisson_process(rate=0.0, plot=False)
        assert "error" in result


# ---------------------------------------------------------------------------
# markov_chain_analysis
# ---------------------------------------------------------------------------

class TestMarkovChainAnalysis:
    def test_happy_path(self):
        P = [[0.5, 0.5], [0.5, 0.5]]
        result = markov_chain_analysis(
            transition_matrix=P, initial_state=0, n_steps=100, plot=False, seed=42,
        )
        assert "error" not in result
        assert "stationary_distribution" in result
        assert "is_ergodic" in result
        assert "empirical_distribution" in result

    def test_stationary_distribution_2state(self):
        # P = [[0.7, 0.3], [0.4, 0.6]]
        # Stationary: pi_0 = 0.4/(0.3+0.4) = 4/7, pi_1 = 3/7
        P = [[0.7, 0.3], [0.4, 0.6]]
        result = markov_chain_analysis(
            transition_matrix=P, initial_state=0, n_steps=50, plot=False, seed=0,
        )
        assert "error" not in result
        sd = result["stationary_distribution"]
        assert abs(sd[0] - 4 / 7) < 1e-4
        assert abs(sd[1] - 3 / 7) < 1e-4

    def test_absorbing_state(self):
        # Identity matrix: every state is absorbing
        P = [[1.0, 0.0], [0.0, 1.0]]
        result = markov_chain_analysis(
            transition_matrix=P, initial_state=0, n_steps=10, plot=False, seed=0,
        )
        assert "error" not in result
        for cls in result["state_classification"]:
            assert cls == "absorbing"

    def test_invalid_matrix_rows(self):
        P = [[0.5, 0.3], [0.4, 0.6]]  # row 0 sums to 0.8
        result = markov_chain_analysis(transition_matrix=P, plot=False)
        assert "error" in result

    def test_invalid_initial_state(self):
        P = [[0.5, 0.5], [0.5, 0.5]]
        result = markov_chain_analysis(
            transition_matrix=P, initial_state=5, n_steps=10, plot=False,
        )
        assert "error" in result
