"""Tests for mcp_scicomp.tools.probability."""

from __future__ import annotations

import numpy as np

from mcp_scicomp.tools.probability import describe_data, fit_distribution, hypothesis_test


# ---------------------------------------------------------------------------
# describe_data
# ---------------------------------------------------------------------------

class TestDescribeData:
    def test_happy_path(self):
        result = describe_data(data=[1, 2, 3, 4, 5], plot=False)
        assert result["n"] == 5
        assert abs(result["mean"] - 3.0) < 1e-9
        assert abs(result["std"] - float(np.std([1, 2, 3, 4, 5], ddof=1))) < 1e-9
        assert result["plot"] is None
        assert result["source"] == "inline"

    def test_normal_data_is_normal(self):
        rng = np.random.default_rng(42)
        data = rng.normal(loc=0, scale=1, size=200).tolist()
        result = describe_data(data=data, plot=False)
        assert result["is_normal"] is True

    def test_uniform_not_normal_large_n(self):
        rng = np.random.default_rng(0)
        data = rng.uniform(0, 1, size=500).tolist()
        result = describe_data(data=data, plot=False)
        assert result["is_normal"] is False

    def test_plot_false_returns_none(self):
        result = describe_data(data=[1.0, 2.0, 3.0], plot=False)
        assert result["plot"] is None

    def test_error_no_data(self):
        result = describe_data(plot=False)
        assert "error" in result

    def test_sample_too_small(self):
        result = describe_data(data=[1.0, 2.0], plot=False)
        assert "error" in result

    def test_keys_present(self):
        result = describe_data(data=list(range(1, 11)), plot=False)
        for key in ["n", "mean", "std", "median", "min", "max", "q25", "q75",
                    "skewness", "kurtosis", "shapiro_stat", "shapiro_p", "is_normal", "source"]:
            assert key in result, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# fit_distribution
# ---------------------------------------------------------------------------

class TestFitDistribution:
    def test_normal_data_best_fit(self):
        rng = np.random.default_rng(7)
        data = rng.normal(loc=5, scale=2, size=300).tolist()
        result = fit_distribution(data=data, plot=False)
        assert result["best_fit"] == "norm"

    def test_exponential_data_in_top2(self):
        rng = np.random.default_rng(99)
        data = rng.exponential(scale=2, size=300).tolist()
        result = fit_distribution(data=data, distributions=["norm", "expon", "gamma"], plot=False)
        top2 = [f["distribution"] for f in result["fits"][:2]]
        assert "expon" in top2 or "gamma" in top2

    def test_output_keys(self):
        data = [float(x) for x in range(1, 21)]
        result = fit_distribution(data=data, plot=False)
        assert "fits" in result
        assert "best_fit" in result
        assert "n" in result
        assert "plot" in result

    def test_plot_false_no_plot(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = fit_distribution(data=data, plot=False)
        assert result.get("plot") is None

    def test_fit_entry_keys(self):
        rng = np.random.default_rng(1)
        data = rng.normal(0, 1, 50).tolist()
        result = fit_distribution(data=data, distributions=["norm"], plot=False)
        fit = result["fits"][0]
        for key in ["distribution", "params", "param_names", "aic", "bic", "ks_stat", "ks_p", "rank"]:
            assert key in fit, f"Missing key in fit entry: {key}"

    def test_unknown_distribution_skipped_gracefully(self):
        rng = np.random.default_rng(2)
        data = rng.normal(0, 1, 50).tolist()
        result = fit_distribution(data=data, distributions=["norm", "this_does_not_exist"], plot=False)
        assert "best_fit" in result or "error" in result

    def test_too_few_points(self):
        result = fit_distribution(data=[1.0, 2.0], plot=False)
        assert "error" in result


# ---------------------------------------------------------------------------
# hypothesis_test
# ---------------------------------------------------------------------------

class TestHypothesisTest:
    def test_t_test_1samp_fail_to_reject(self):
        rng2 = np.random.default_rng(42)
        data = rng2.normal(0, 1, 50).tolist()
        result = hypothesis_test(test="t_test_1samp", data=data, popmean=0.0)
        assert result["decision"] == "fail to reject H0"
        assert result["p_value"] > 0.05

    def test_t_test_1samp_reject(self):
        rng = np.random.default_rng(5)
        data = rng.normal(10, 1, 100).tolist()
        result = hypothesis_test(test="t_test_1samp", data=data, popmean=0.0)
        assert result["decision"] == "reject H0"
        assert result["p_value"] < 0.05

    def test_t_test_2samp_identical_fail_to_reject(self):
        rng = np.random.default_rng(3)
        data = rng.normal(0, 1, 50).tolist()
        result = hypothesis_test(test="t_test_2samp", data=data, data2=data)
        assert result["decision"] == "fail to reject H0"

    def test_mann_whitney_returns_keys(self):
        rng = np.random.default_rng(10)
        d1 = rng.normal(0, 1, 30).tolist()
        d2 = rng.normal(5, 1, 30).tolist()
        result = hypothesis_test(test="mann_whitney", data=d1, data2=d2)
        assert "statistic" in result
        assert "p_value" in result

    def test_shapiro_normal_data(self):
        rng = np.random.default_rng(11)
        data = rng.normal(0, 1, 100).tolist()
        result = hypothesis_test(test="shapiro", data=data)
        assert "statistic" in result
        assert "p_value" in result
        assert result["decision"] == "fail to reject H0"

    def test_ks_test_identical_fail_to_reject(self):
        rng = np.random.default_rng(12)
        data = rng.normal(0, 1, 50).tolist()
        result = hypothesis_test(test="ks_test", data=data, data2=data)
        assert result["decision"] == "fail to reject H0"

    def test_invalid_test_name(self):
        result = hypothesis_test(test="not_a_test", data=[1.0, 2.0, 3.0])
        assert "error" in result

    def test_t_test_2samp_missing_second_sample(self):
        result = hypothesis_test(test="t_test_2samp", data=[1.0, 2.0, 3.0])
        assert "error" in result

    def test_output_keys(self):
        result = hypothesis_test(test="t_test_1samp", data=[1.0, 2.0, 3.0, 4.0, 5.0])
        for key in ["test", "statistic", "p_value", "alpha", "decision",
                    "confidence_interval", "alternative", "n1"]:
            assert key in result, f"Missing key: {key}"

    def test_confidence_interval_present_t1(self):
        data = [2.0] * 20
        result = hypothesis_test(test="t_test_1samp", data=data, popmean=0.0)
        assert result["confidence_interval"] is not None
        assert len(result["confidence_interval"]) == 2
