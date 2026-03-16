"""Stochastic process tools for mcp-scicomp."""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import numpy as np

from mcp_scicomp.app import mcp
from mcp_scicomp.plotting import fig_to_base64

logger = logging.getLogger(__name__)


@mcp.tool()
def simulate_brownian_motion(
    t_end: float = 1.0,
    n_steps: int = 200,
    n_paths: int = 10,
    mu: float = 0.0,
    sigma: float = 1.0,
    x0: float = 0.0,
    geometric: bool = False,
    plot: bool = True,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """Simulate arithmetic or geometric Brownian motion paths.

    Use this tool to model Wiener processes (random walks), stock prices (GBM),
    or any diffusion process. Returns final-value statistics and a spaghetti plot.

    Args:
        t_end: Total time horizon. Default 1.0.
        n_steps: Number of discrete time steps. Default 200. Must be > 0.
        n_paths: Number of independent simulation paths. Default 10. Capped at 200 internally for the plot.
        mu: Drift coefficient. Default 0.0.
        sigma: Volatility / diffusion coefficient. Must be > 0. Default 1.0.
        x0: Initial value of the process. Default 0.0.
        geometric: If True, simulate Geometric Brownian Motion (GBM) where
            dX = mu*X*dt + sigma*X*dW. Useful for stock prices (always positive).
            If False (default), simulate Arithmetic Brownian Motion (ABM):
            X_t = x0 + mu*t + sigma*W_t.
        plot: If True, return a spaghetti plot of all paths as base64 PNG. Default True.
        seed: Random seed for reproducibility. Default None.

    Returns:
        dict with keys:
            t_end (float), n_steps (int), n_paths (int), mu (float), sigma (float),
            x0 (float), geometric (bool),
            final_mean (float), final_std (float), final_min (float),
            final_max (float), final_median (float),
            plot (str | None)

    Example:
        simulate_brownian_motion(t_end=1.0, n_paths=50, mu=0.05, sigma=0.2, geometric=True)
    """
    try:
        if sigma <= 0:
            return {"error": "sigma must be > 0.", "suggestion": "Set sigma to a positive value."}
        if n_steps <= 0:
            return {"error": "n_steps must be > 0.", "suggestion": "Set n_steps to a positive integer."}
        if n_paths <= 0:
            return {"error": "n_paths must be > 0.", "suggestion": "Set n_paths to a positive integer."}

        rng = np.random.default_rng(seed)
        dt = t_end / n_steps
        t = np.linspace(0.0, t_end, n_steps + 1)

        # dW increments: shape (n_steps, n_paths)
        dW = rng.normal(0.0, math.sqrt(dt), (n_steps, n_paths))

        if not geometric:
            # ABM: X_t = x0 + mu*t + sigma*W_t
            W = np.vstack([np.zeros(n_paths), np.cumsum(dW, axis=0)])  # (n_steps+1, n_paths)
            paths = x0 + mu * t[:, None] + sigma * W
        else:
            # GBM analytical: X_t = x0 * exp((mu - 0.5*sigma^2)*t + sigma*W_t)
            W = np.vstack([np.zeros(n_paths), np.cumsum(dW, axis=0)])
            paths = x0 * np.exp((mu - 0.5 * sigma**2) * t[:, None] + sigma * W)

        final_values = paths[-1, :]

        plot_b64: Optional[str] = None
        if plot:
            import matplotlib.pyplot as plt

            plot_paths = min(n_paths, 200)
            fig, ax = plt.subplots(figsize=(8, 5))
            for i in range(plot_paths):
                ax.plot(t, paths[:, i], linewidth=0.8, alpha=0.6)
            label = "GBM" if geometric else "ABM"
            ax.set_title(f"{'Geometric' if geometric else 'Arithmetic'} Brownian Motion ({label})")
            ax.set_xlabel("Time")
            ax.set_ylabel("X(t)")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            plot_b64 = fig_to_base64(fig)

        return {
            "t_end": float(t_end),
            "n_steps": int(n_steps),
            "n_paths": int(n_paths),
            "mu": float(mu),
            "sigma": float(sigma),
            "x0": float(x0),
            "geometric": bool(geometric),
            "final_mean": float(np.mean(final_values)),
            "final_std": float(np.std(final_values, ddof=1)),
            "final_min": float(np.min(final_values)),
            "final_max": float(np.max(final_values)),
            "final_median": float(np.median(final_values)),
            "plot": plot_b64,
        }

    except Exception as exc:
        logger.exception("simulate_brownian_motion failed")
        return {"error": str(exc), "suggestion": "Check parameters; sigma must be positive, n_steps > 0."}


@mcp.tool()
def simulate_ornstein_uhlenbeck(
    theta: float = 1.0,
    mu: float = 0.0,
    sigma: float = 1.0,
    x0: float = 0.0,
    t_end: float = 5.0,
    n_steps: int = 500,
    n_paths: int = 10,
    plot: bool = True,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """Simulate Ornstein-Uhlenbeck (mean-reverting) processes.

    Use this tool to model interest rates, volatility, or any mean-reverting
    stochastic process. The OU process satisfies dX = theta*(mu - X)*dt + sigma*dW.

    Args:
        theta: Mean-reversion speed. Must be > 0. Larger values pull faster toward mu.
            Default 1.0.
        mu: Long-run mean (equilibrium level). Default 0.0.
        sigma: Volatility / diffusion coefficient. Default 1.0.
        x0: Initial value. Default 0.0.
        t_end: Total simulation time. Default 5.0.
        n_steps: Number of time steps (Euler-Maruyama). Default 500.
        n_paths: Number of independent paths. Default 10.
        plot: If True, return a spaghetti plot with dashed line at mu. Default True.
        seed: Random seed for reproducibility. Default None.

    Returns:
        dict with keys:
            theta (float), mu (float), sigma (float), x0 (float),
            t_end (float), n_steps (int), n_paths (int),
            final_mean (float), final_std (float),
            theoretical_stationary_mean (float),
            theoretical_stationary_std (float),
            plot (str | None)

    Example:
        simulate_ornstein_uhlenbeck(theta=2.0, mu=5.0, sigma=0.5, x0=0.0, t_end=10.0)
    """
    try:
        if theta <= 0:
            return {"error": "theta must be > 0.", "suggestion": "Set theta to a positive value (e.g. 1.0)."}
        if sigma <= 0:
            return {"error": "sigma must be > 0.", "suggestion": "Set sigma to a positive value."}
        if n_steps <= 0:
            return {"error": "n_steps must be > 0.", "suggestion": "Set n_steps to a positive integer."}

        rng = np.random.default_rng(seed)
        dt = t_end / n_steps
        sqrt_dt = math.sqrt(dt)

        # Paths: shape (n_steps+1, n_paths)
        paths = np.empty((n_steps + 1, n_paths))
        paths[0, :] = x0

        Z = rng.standard_normal((n_steps, n_paths))
        for i in range(n_steps):
            paths[i + 1, :] = (
                paths[i, :]
                + theta * (mu - paths[i, :]) * dt
                + sigma * sqrt_dt * Z[i, :]
            )

        final_values = paths[-1, :]
        t = np.linspace(0.0, t_end, n_steps + 1)

        theoretical_stationary_std = sigma / math.sqrt(2 * theta)

        plot_b64: Optional[str] = None
        if plot:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 5))
            for i in range(n_paths):
                ax.plot(t, paths[:, i], linewidth=0.8, alpha=0.6)
            ax.axhline(mu, color="crimson", linestyle="--", linewidth=1.5, label=f"mu = {mu}")
            ax.set_title("Ornstein-Uhlenbeck Process")
            ax.set_xlabel("Time")
            ax.set_ylabel("X(t)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            plot_b64 = fig_to_base64(fig)

        return {
            "theta": float(theta),
            "mu": float(mu),
            "sigma": float(sigma),
            "x0": float(x0),
            "t_end": float(t_end),
            "n_steps": int(n_steps),
            "n_paths": int(n_paths),
            "final_mean": float(np.mean(final_values)),
            "final_std": float(np.std(final_values, ddof=1) if n_paths > 1 else 0.0),
            "theoretical_stationary_mean": float(mu),
            "theoretical_stationary_std": float(theoretical_stationary_std),
            "plot": plot_b64,
        }

    except Exception as exc:
        logger.exception("simulate_ornstein_uhlenbeck failed")
        return {"error": str(exc), "suggestion": "Check parameters; theta and sigma must be positive."}


@mcp.tool()
def simulate_poisson_process(
    rate: float = 1.0,
    t_end: float = 10.0,
    n_paths: int = 5,
    plot: bool = True,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """Simulate homogeneous Poisson processes (counting processes).

    Use this tool to model event arrivals: customer arrivals, photon counts,
    insurance claims, or any process with constant event rate. Returns event
    counts and inter-arrival statistics.

    Args:
        rate: Expected events per unit time (intensity). Must be > 0. Default 1.0.
        t_end: End of observation window. Default 10.0.
        n_paths: Number of independent realizations. Default 5.
        plot: If True, return a step-function plot of N(t) for each path. Default True.
        seed: Random seed for reproducibility. Default None.

    Returns:
        dict with keys:
            rate (float), t_end (float), n_paths (int),
            mean_events (float), std_events (float),
            expected_events (float),
            mean_interarrival (float), expected_interarrival (float),
            event_counts (list[int]),
            plot (str | None)

    Example:
        simulate_poisson_process(rate=3.0, t_end=5.0, n_paths=10)
    """
    try:
        if rate <= 0:
            return {"error": "rate must be > 0.", "suggestion": "Set rate to a positive value (e.g. 1.0)."}
        if t_end <= 0:
            return {"error": "t_end must be > 0.", "suggestion": "Set t_end to a positive value."}
        if n_paths <= 0:
            return {"error": "n_paths must be > 0.", "suggestion": "Set n_paths to a positive integer."}

        rng = np.random.default_rng(seed)

        all_arrival_times: list[Any] = []
        event_counts: list[int] = []
        all_interarrivals: list[float] = []

        for _ in range(n_paths):
            arrivals = []
            t = 0.0
            while True:
                inter = rng.exponential(1.0 / rate)
                t += inter
                if t > t_end:
                    break
                arrivals.append(t)
                all_interarrivals.append(float(inter))
            event_counts.append(len(arrivals))
            all_arrival_times.append(np.array(arrivals))

        counts_arr = np.array(event_counts, dtype=float)
        mean_ia = float(np.mean(all_interarrivals)) if all_interarrivals else float("nan")

        plot_b64: Optional[str] = None
        if plot:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 5))
            plot_paths = min(n_paths, 20)
            for i in range(plot_paths):
                arrivals = all_arrival_times[i]
                # Build step function: times 0, each arrival; counts 0, 1, 2, ...
                t_steps = np.concatenate([[0.0], arrivals, [t_end]])
                n_steps_arr = np.concatenate([[0], np.arange(1, len(arrivals) + 1), [len(arrivals)]])
                ax.step(t_steps, n_steps_arr, where="post", linewidth=1.0, alpha=0.7)
            ax.set_title(f"Poisson Process (rate={rate})")
            ax.set_xlabel("Time")
            ax.set_ylabel("N(t) — cumulative events")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            plot_b64 = fig_to_base64(fig)

        return {
            "rate": float(rate),
            "t_end": float(t_end),
            "n_paths": int(n_paths),
            "mean_events": float(np.mean(counts_arr)),
            "std_events": float(np.std(counts_arr, ddof=1) if n_paths > 1 else 0.0),
            "expected_events": float(rate * t_end),
            "mean_interarrival": mean_ia,
            "expected_interarrival": float(1.0 / rate),
            "event_counts": [int(c) for c in event_counts],
            "plot": plot_b64,
        }

    except Exception as exc:
        logger.exception("simulate_poisson_process failed")
        return {"error": str(exc), "suggestion": "Check that rate and t_end are positive numbers."}


@mcp.tool()
def markov_chain_analysis(
    transition_matrix: list[list[float]] = [[0.7, 0.3], [0.4, 0.6]],
    initial_state: int = 0,
    n_steps: int = 100,
    plot: bool = True,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """Analyze a discrete-time Markov chain: stationary distribution, ergodicity, simulation.

    Use this tool to study Markov chains: find the long-run stationary distribution,
    classify states, check ergodicity, simulate a sample path, and compare empirical
    vs theoretical distributions.

    Args:
        transition_matrix: Row-stochastic matrix as list of lists. Each row must sum to 1.
            Element P[i][j] = probability of transitioning from state i to state j.
            Example: [[0.7, 0.3], [0.4, 0.6]] for a 2-state chain.
        initial_state: Starting state index (0-indexed). Default 0.
        n_steps: Number of simulation steps for empirical distribution. Default 100.
        plot: If True, return a bar chart comparing stationary vs empirical distribution.
            Default True.
        seed: Random seed for reproducibility. Default None.

    Returns:
        dict with keys:
            n_states (int),
            stationary_distribution (list[float]),
            is_ergodic (bool),
            state_classification (list[str]): "absorbing", "recurrent", or "transient" per state,
            mean_first_passage_times (list[list[float]] | None): MFPTs for ergodic chains,
            simulated_path_length (int),
            empirical_distribution (list[float]),
            plot (str | None)

    Example:
        markov_chain_analysis(transition_matrix=[[0.9, 0.1], [0.2, 0.8]], initial_state=0, n_steps=500)
    """
    try:
        P = np.array(transition_matrix, dtype=float)

        # Validate shape
        if P.ndim != 2 or P.shape[0] != P.shape[1]:
            return {
                "error": "transition_matrix must be a square matrix.",
                "suggestion": "Provide an n×n list of lists.",
            }

        n = P.shape[0]

        # Validate rows sum to 1
        row_sums = P.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            bad_rows = [int(i) for i in np.where(np.abs(row_sums - 1.0) > 1e-6)[0]]
            return {
                "error": f"Rows {bad_rows} do not sum to 1. Got sums {[float(row_sums[i]) for i in bad_rows]}.",
                "suggestion": "Each row of transition_matrix must sum to exactly 1.0.",
            }

        # Validate values in [0, 1]
        if np.any(P < -1e-10) or np.any(P > 1 + 1e-10):
            return {
                "error": "All transition probabilities must be in [0, 1].",
                "suggestion": "Check for negative values or values > 1.",
            }

        # Validate initial_state
        if initial_state < 0 or initial_state >= n:
            return {
                "error": f"initial_state={initial_state} is out of range for a {n}-state chain.",
                "suggestion": f"Set initial_state to an integer in [0, {n - 1}].",
            }

        # --- Stationary distribution ---
        # Solve pi @ P = pi with sum(pi) = 1
        # Equivalent: (P^T - I) pi = 0, sum(pi) = 1
        # Build system: replace last equation with sum constraint
        # Build augmented system: (P^T - I) pi = 0, sum(pi) = 1
        A_sys = np.vstack([P.T - np.eye(n), np.ones((1, n))])
        b_sys = np.zeros(n + 1)
        b_sys[-1] = 1.0
        pi, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
        # Clip small numerical negatives
        pi = np.clip(pi, 0.0, None)
        pi /= pi.sum()

        # --- State classification ---
        state_classification: list[str] = []
        for i in range(n):
            if abs(P[i, i] - 1.0) < 1e-10 and np.sum(P[i, :] > 1e-10) == 1:
                state_classification.append("absorbing")
            else:
                # Simplified: check reachability via matrix power
                # Use P^n to approximate communication
                state_classification.append("non-absorbing")

        # Refine: for non-absorbing, check if recurrent or transient
        # via communication classes (simplified: if chain is ergodic, all are recurrent)
        # For now, use absorbing vs recurrent vs transient heuristic
        # A state is recurrent if it's in a closed communicating class
        # We use the stationary distribution as a proxy: pi[i] > 0 → recurrent
        for i in range(n):
            if state_classification[i] != "absorbing":
                if pi[i] > 1e-10:
                    state_classification[i] = "recurrent"
                else:
                    state_classification[i] = "transient"

        # --- Ergodicity check ---
        # Ergodic = irreducible (all states communicate) AND aperiodic
        # Check irreducibility via (I + P)^(n-1) > 0 (all entries positive)
        reach = np.linalg.matrix_power(np.eye(n) + P, n)
        is_irreducible = bool(np.all(reach > 1e-10))
        # Aperiodicity: at least one self-loop (sufficient condition), or check eigenvalues
        has_self_loop = bool(np.any(np.diag(P) > 0))
        # More robust: check if eigenvalue -1 is NOT present (period check)
        eigenvalues = np.linalg.eigvals(P)
        has_neg1 = bool(np.any(np.abs(eigenvalues + 1.0) < 1e-8))
        is_aperiodic = has_self_loop or not has_neg1
        is_ergodic = bool(is_irreducible and is_aperiodic)

        # --- Simulate chain ---
        rng = np.random.default_rng(seed)
        path = [initial_state]
        current = initial_state
        for _ in range(n_steps):
            current = int(rng.choice(n, p=P[current]))
            path.append(current)

        # Empirical distribution
        counts = np.bincount(path, minlength=n)
        empirical_dist = [float(c) / len(path) for c in counts]

        # --- Mean first passage times (ergodic chains only) ---
        mfpt: Optional[list[list[float]]] = None
        if is_ergodic and n <= 50:
            # M[i,j] = E[steps to reach j starting from i]
            # For ergodic chain: M[j,j] = 1/pi[j]
            # For i != j: M[i,j] = 1 + sum_{k != j} P[i,k] * M[k,j]
            # Solve linear system for each target j
            M = np.zeros((n, n))
            for j in range(n):
                # System: M[i,j] = 1 + sum_{k!=j} P[i,k] * M[k,j]
                # Rearrange: M[i,j] - sum_{k!=j} P[i,k] * M[k,j] = 1 for i != j
                # M[j,j] = 1/pi[j]
                idx = [k for k in range(n) if k != j]
                A_mfpt = np.eye(len(idx)) - P[np.ix_(idx, idx)]
                b_mfpt = np.ones(len(idx))
                try:
                    m_col, *_ = np.linalg.lstsq(A_mfpt, b_mfpt, rcond=None)
                    for ii, state in enumerate(idx):
                        M[state, j] = float(m_col[ii])
                    M[j, j] = 1.0 / float(pi[j]) if pi[j] > 1e-15 else float("inf")
                except Exception:
                    pass
            mfpt = [[float(M[i, j]) for j in range(n)] for i in range(n)]

        # --- Plot ---
        plot_b64: Optional[str] = None
        if plot:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 5))
            x = np.arange(n)
            width = 0.35
            ax.bar(x - width / 2, pi, width, label="Stationary", color="steelblue", alpha=0.8)
            ax.bar(x + width / 2, empirical_dist, width, label="Empirical", color="darkorange", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([f"State {i}" for i in range(n)])
            ax.set_ylabel("Probability")
            ax.set_title("Stationary vs Empirical Distribution")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")
            fig.tight_layout()
            plot_b64 = fig_to_base64(fig)

        return {
            "n_states": int(n),
            "stationary_distribution": [float(p) for p in pi],
            "is_ergodic": is_ergodic,
            "state_classification": state_classification,
            "mean_first_passage_times": mfpt,
            "simulated_path_length": int(len(path)),
            "empirical_distribution": empirical_dist,
            "plot": plot_b64,
        }

    except Exception as exc:
        logger.exception("markov_chain_analysis failed")
        return {
            "error": str(exc),
            "suggestion": "Check transition_matrix is a square row-stochastic matrix (rows sum to 1).",
        }
