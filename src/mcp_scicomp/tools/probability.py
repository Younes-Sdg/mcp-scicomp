"""Probability and statistics tools for mcp-scicomp."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import scipy.stats as stats  # type: ignore[import-untyped]

from mcp_scicomp.plotting import fig_to_base64
from mcp_scicomp.app import mcp
from mcp_scicomp.utils import resolve_data

logger = logging.getLogger(__name__)


@mcp.tool()
def describe_data(
    data: Optional[list[float]] = None,
    file_path: Optional[str] = None,
    column: Optional[str] = None,
    bins: int = 30,
    plot: bool = True,
) -> dict[str, Any]:
    """Compute descriptive statistics and normality test for a numeric sample.

    Use this tool for exploratory data analysis: summary stats, distribution shape,
    and a histogram. Accepts inline data or a file (CSV/TSV/Excel/JSON/Parquet).

    Args:
        data: Inline list of floats, e.g. [1.2, 3.4, 5.6]. Mutually exclusive with file_path.
        file_path: Path to a data file (.csv, .tsv, .xlsx, .json, .parquet).
        column: Column name when the file has multiple numeric columns.
        bins: Number of histogram bins. Default 30.
        plot: If True, return a base64 PNG histogram. Default True.

    Returns:
        dict with keys:
            n (int), mean (float), std (float), median (float),
            min (float), max (float), q25 (float), q75 (float),
            skewness (float), kurtosis (float),
            shapiro_stat (float), shapiro_p (float), is_normal (bool),
            plot (str | None), source (str)

    Example:
        describe_data(data=[2.1, 3.5, 2.8, 4.0, 3.1], bins=10)
    """
    try:
        arr, meta = resolve_data(data, file_path, column)

        if len(arr) < 3:
            return {
                "error": "Sample too small",
                "suggestion": "Provide at least 3 data points.",
            }

        q25, q50, q75 = np.quantile(arr, [0.25, 0.50, 0.75])

        # Shapiro-Wilk capped at 5000 for performance
        shapiro_arr = arr[:5000] if len(arr) > 5000 else arr
        shapiro_stat, shapiro_p = stats.shapiro(shapiro_arr)

        alpha = 0.05
        is_normal = bool(shapiro_p > alpha)

        plot_b64: Optional[str] = None
        if plot:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(arr, bins=bins, density=True, alpha=0.7, color="steelblue", edgecolor="white")
            ax.set_title("Data Distribution")
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            plot_b64 = fig_to_base64(fig)

        return {
            "n": int(len(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)),
            "median": float(q50),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "q25": float(q25),
            "q75": float(q75),
            "skewness": float(stats.skew(arr)),
            "kurtosis": float(stats.kurtosis(arr)),
            "shapiro_stat": float(shapiro_stat),
            "shapiro_p": float(shapiro_p),
            "is_normal": is_normal,
            "plot": plot_b64,
            "source": str(meta["source"]),
        }

    except Exception as exc:
        logger.exception("describe_data failed")
        return {"error": str(exc), "suggestion": "Check that data is a list of numbers or file_path is valid."}


@mcp.tool()
def fit_distribution(
    data: Optional[list[float]] = None,
    file_path: Optional[str] = None,
    column: Optional[str] = None,
    distributions: Optional[list[str]] = None,
    plot: bool = True,
) -> dict[str, Any]:
    """Fit parametric distributions to a dataset and rank them by AIC.

    Use this tool to find which probability distribution best describes your data.
    Fits multiple distributions via MLE and ranks them by AIC/BIC and KS test.

    Args:
        data: Inline list of floats. Mutually exclusive with file_path.
        file_path: Path to a data file (.csv, .tsv, .xlsx, .json, .parquet).
        column: Column name when the file has multiple numeric columns.
        distributions: List of scipy.stats distribution names to try.
            Default: ["norm", "lognorm", "expon", "gamma", "beta", "uniform", "weibull_min"]
        plot: If True, return a base64 PNG with histogram + top-3 PDF overlays. Default True.

    Returns:
        dict with keys:
            best_fit (str): name of the best-fitting distribution,
            fits (list[dict]): ranked fits, each with distribution, params, param_names,
                               aic, bic, ks_stat, ks_p, rank,
            plot (str | None),
            n (int)

    Example:
        fit_distribution(data=[0.5, 1.2, 0.8, 2.1, 0.3], distributions=["expon", "gamma"])
    """
    if distributions is None:
        distributions = ["norm", "lognorm", "expon", "gamma", "beta", "uniform", "weibull_min"]

    try:
        arr, meta = resolve_data(data, file_path, column)
        n = len(arr)

        if n < 5:
            return {"error": "Need at least 5 data points to fit distributions.", "suggestion": "Provide more data."}

        fits: list[dict[str, Any]] = []

        for dist_name in distributions:
            try:
                dist = getattr(stats, dist_name)
                params = dist.fit(arr)
                ll = float(dist.logpdf(arr, *params).sum())
                if not np.isfinite(ll):
                    continue
                k = len(params)
                aic = float(2 * k - 2 * ll)
                bic = float(k * np.log(n) - 2 * ll)
                ks_stat, ks_p = stats.kstest(arr, dist_name, args=params)
                fits.append({
                    "distribution": dist_name,
                    "params": [float(p) for p in params],
                    "param_names": list(dist.shapes.split(", ") if dist.shapes else []) + ["loc", "scale"],
                    "aic": aic,
                    "bic": bic,
                    "ks_stat": float(ks_stat),
                    "ks_p": float(ks_p),
                })
            except Exception as exc:
                logger.debug("Skipping distribution %s: %s", dist_name, exc)
                continue

        if not fits:
            return {"error": "No distributions could be fit to this data.", "suggestion": "Check data range; beta requires values in (0,1)."}

        fits.sort(key=lambda x: x["aic"])
        for rank, fit in enumerate(fits, start=1):
            fit["rank"] = rank

        best_fit = fits[0]["distribution"]

        plot_b64: Optional[str] = None
        if plot:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(arr, bins=30, density=True, alpha=0.5, color="steelblue",
                    edgecolor="white", label="Data")
            x_min, x_max = arr.min(), arr.max()
            x_range = np.linspace(x_min - 0.1 * abs(x_min), x_max + 0.1 * abs(x_max), 300)
            colors = ["crimson", "darkorange", "forestgreen"]
            for i, fit in enumerate(fits[:3]):
                dist = getattr(stats, fit["distribution"])
                try:
                    pdf_vals = dist.pdf(x_range, *fit["params"])
                    ax.plot(x_range, pdf_vals, color=colors[i], linewidth=2,
                            label=f"{fit['distribution']} (AIC={fit['aic']:.1f})")
                except Exception:
                    pass
            ax.legend()
            ax.set_title("Distribution Fits")
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            plot_b64 = fig_to_base64(fig)

        return {
            "best_fit": best_fit,
            "fits": fits,
            "plot": plot_b64,
            "n": int(n),
        }

    except Exception as exc:
        logger.exception("fit_distribution failed")
        return {"error": str(exc), "suggestion": "Check data and distribution names are valid scipy.stats names."}


@mcp.tool()
def hypothesis_test(
    test: str,
    data: Optional[list[float]] = None,
    file_path: Optional[str] = None,
    column: Optional[str] = None,
    data2: Optional[list[float]] = None,
    file_path2: Optional[str] = None,
    column2: Optional[str] = None,
    popmean: float = 0.0,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """Perform a frequentist hypothesis test on one or two samples.

    Use this tool to test statistical hypotheses. Supports t-tests, Mann-Whitney U,
    Kolmogorov-Smirnov, and Shapiro-Wilk normality tests.

    Args:
        test: Test to run. One of:
            "t_test_1samp"  — one-sample t-test against a population mean,
            "t_test_2samp"  — two-sample Welch's t-test,
            "mann_whitney"  — Mann-Whitney U (non-parametric two-sample),
            "ks_test"       — two-sample Kolmogorov-Smirnov,
            "shapiro"       — Shapiro-Wilk normality test (one sample).
        data: First sample as inline list of floats.
        file_path: Path to file for first sample.
        column: Column name for first sample file.
        data2: Second sample as inline list of floats (required for two-sample tests).
        file_path2: Path to file for second sample.
        column2: Column name for second sample file.
        popmean: Population mean for t_test_1samp. Default 0.0.
        alpha: Significance level. Default 0.05.
        alternative: "two-sided", "less", or "greater". Default "two-sided".

    Returns:
        dict with keys:
            test (str), statistic (float), p_value (float), alpha (float),
            decision (str: "reject H0" | "fail to reject H0"),
            confidence_interval ([float, float] | None),
            alternative (str), n1 (int), n2 (int | None)

    Example:
        hypothesis_test(test="t_test_1samp", data=[2.1, 3.5, 2.8, 4.0], popmean=3.0)
        hypothesis_test(test="t_test_2samp", data=[1,2,3], data2=[4,5,6])
    """
    VALID_TESTS = {"t_test_1samp", "t_test_2samp", "mann_whitney", "ks_test", "shapiro"}
    TWO_SAMPLE_TESTS = {"t_test_2samp", "mann_whitney", "ks_test"}

    try:
        if test not in VALID_TESTS:
            return {
                "error": f"Unknown test '{test}'.",
                "suggestion": f"Choose one of: {', '.join(sorted(VALID_TESTS))}",
            }

        arr1, meta1 = resolve_data(data, file_path, column)
        n1 = int(len(arr1))

        arr2: Optional[np.ndarray] = None
        n2: Optional[int] = None

        if test in TWO_SAMPLE_TESTS:
            if data2 is None and file_path2 is None:
                return {
                    "error": f"'{test}' requires a second sample.",
                    "suggestion": "Provide data2 or file_path2.",
                }
            arr2, meta2 = resolve_data(data2, file_path2, column2)
            n2 = int(len(arr2))

        statistic: float
        p_value: float
        ci: Optional[list[float]] = None

        if test == "t_test_1samp":
            result = stats.ttest_1samp(arr1, popmean=popmean, alternative=alternative)
            statistic = float(result.statistic)
            p_value = float(result.pvalue)
            # Confidence interval via t-distribution
            df = n1 - 1
            se = float(np.std(arr1, ddof=1)) / np.sqrt(n1)
            if alternative == "two-sided":
                t_crit = float(stats.t.ppf(1 - alpha / 2, df))
                mean = float(np.mean(arr1))
                ci = [mean - t_crit * se, mean + t_crit * se]
            elif alternative == "greater":
                t_crit = float(stats.t.ppf(1 - alpha, df))
                ci = [float(np.mean(arr1)) - t_crit * se, float("inf")]
            else:  # less
                t_crit = float(stats.t.ppf(1 - alpha, df))
                ci = [float("-inf"), float(np.mean(arr1)) + t_crit * se]

        elif test == "t_test_2samp":
            assert arr2 is not None and n2 is not None  # guaranteed by TWO_SAMPLE_TESTS check above
            result = stats.ttest_ind(arr1, arr2, alternative=alternative, equal_var=False)
            statistic = float(result.statistic)
            p_value = float(result.pvalue)
            # Welch's CI (two-sided only for simplicity)
            mean_diff = float(np.mean(arr1) - np.mean(arr2))
            s1, s2 = float(np.std(arr1, ddof=1)), float(np.std(arr2, ddof=1))
            se_diff = np.sqrt(s1**2 / n1 + s2**2 / n2)
            # Welch-Satterthwaite df
            num = (s1**2 / n1 + s2**2 / n2) ** 2
            denom = (s1**2 / n1) ** 2 / (n1 - 1) + (s2**2 / n2) ** 2 / (n2 - 1)
            df_welch = float(num / denom) if denom > 0 else float(n1 + n2 - 2)
            if alternative == "two-sided":
                t_crit = float(stats.t.ppf(1 - alpha / 2, df_welch))
                ci = [mean_diff - t_crit * se_diff, mean_diff + t_crit * se_diff]

        elif test == "mann_whitney":
            assert arr2 is not None  # guaranteed by TWO_SAMPLE_TESTS check above
            result = stats.mannwhitneyu(arr1, arr2, alternative=alternative)
            statistic = float(result.statistic)
            p_value = float(result.pvalue)

        elif test == "ks_test":
            assert arr2 is not None  # guaranteed by TWO_SAMPLE_TESTS check above
            result = stats.ks_2samp(arr1, arr2, alternative=alternative)
            statistic = float(result.statistic)
            p_value = float(result.pvalue)

        elif test == "shapiro":
            stat, pval = stats.shapiro(arr1)
            statistic = float(stat)
            p_value = float(pval)

        decision = "reject H0" if p_value < alpha else "fail to reject H0"

        return {
            "test": test,
            "statistic": statistic,
            "p_value": p_value,
            "alpha": float(alpha),
            "decision": decision,
            "confidence_interval": ci,
            "alternative": alternative,
            "n1": n1,
            "n2": n2,
        }

    except Exception as exc:
        logger.exception("hypothesis_test failed")
        return {"error": str(exc), "suggestion": "Check inputs; two-sample tests require data2 or file_path2."}
