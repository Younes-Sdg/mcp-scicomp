"""Tests for linear algebra tools: eigenanalysis and matrix_decomposition."""

from __future__ import annotations

import math

import numpy as np

from mcp_scicomp.tools.linalg import eigenanalysis, matrix_decomposition

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

# 3×3 symmetric tridiagonal: eigenvalues are 2-√2, 2, 2+√2
SYMMETRIC_3X3 = [[2, 1, 0], [1, 2, 1], [0, 1, 2]]
EXPECTED_EIGS_3X3 = sorted([2 - math.sqrt(2), 2.0, 2 + math.sqrt(2)])

# Simple 3×3 full-rank matrix for decomposition tests
FULL_RANK_3X3 = [[4, 3, 1], [6, 3, 2], [2, 5, 3]]

# Positive-definite 3×3: A = B @ B.T + I guarantees PD
_B = np.array([[1.0, 0.5, 0.2], [0.0, 1.0, 0.3], [0.0, 0.0, 1.0]])
PD_MATRIX = (_B @ _B.T + np.eye(3)).tolist()


# ---------------------------------------------------------------------------
# TestEigenanalysis
# ---------------------------------------------------------------------------


class TestEigenanalysis:
    def test_symmetric_3x3_eigenvalues(self) -> None:
        result = eigenanalysis(matrix=SYMMETRIC_3X3, plot=False)
        assert "error" not in result
        computed = sorted(result["eigenvalues"])
        for got, expected in zip(computed, EXPECTED_EIGS_3X3):
            assert abs(got - expected) < 1e-10

    def test_returns_expected_keys(self) -> None:
        result = eigenanalysis(matrix=SYMMETRIC_3X3, plot=False)
        for key in ("eigenvalues", "eigenvectors", "spectral_radius",
                    "condition_number", "rank", "determinant",
                    "is_symmetric", "is_positive_definite", "shape"):
            assert key in result, f"Missing key: {key}"

    def test_is_symmetric_detected(self) -> None:
        result = eigenanalysis(matrix=SYMMETRIC_3X3, plot=False)
        assert result["is_symmetric"] is True

    def test_non_symmetric_detected(self) -> None:
        result = eigenanalysis(matrix=FULL_RANK_3X3, plot=False)
        assert result["is_symmetric"] is False
        # Non-symmetric -> is_positive_definite should be None
        assert result["is_positive_definite"] is None

    def test_non_square_error(self) -> None:
        result = eigenanalysis(matrix=[[1, 2, 3], [4, 5, 6]], plot=False)
        assert "error" in result

    def test_positive_definite_detection_identity(self) -> None:
        identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        result = eigenanalysis(matrix=identity, plot=False)
        assert result["is_positive_definite"] is True

    def test_non_positive_definite_detection(self) -> None:
        # Indefinite symmetric matrix
        indefinite = [[1.0, 2.0], [2.0, 1.0]]
        result = eigenanalysis(matrix=indefinite, plot=False)
        assert result["is_positive_definite"] is False

    def test_spectral_radius_identity(self) -> None:
        identity = [[1.0, 0.0], [0.0, 1.0]]
        result = eigenanalysis(matrix=identity, plot=False)
        assert abs(result["spectral_radius"] - 1.0) < 1e-12

    def test_rank_singular_matrix(self) -> None:
        singular = [[1.0, 2.0], [2.0, 4.0]]
        result = eigenanalysis(matrix=singular, plot=False)
        assert result["rank"] == 1

    def test_plot_returned_when_requested(self) -> None:
        result = eigenanalysis(matrix=SYMMETRIC_3X3, plot=True)
        assert "plot" in result
        assert isinstance(result["plot"], str) and len(result["plot"]) > 100

    def test_no_plot_when_disabled(self) -> None:
        result = eigenanalysis(matrix=SYMMETRIC_3X3, plot=False)
        assert "plot" not in result


# ---------------------------------------------------------------------------
# TestMatrixDecomposition
# ---------------------------------------------------------------------------


class TestMatrixDecomposition:
    def test_svd_happy_path(self) -> None:
        result = matrix_decomposition(matrix=FULL_RANK_3X3, method="svd", plot=False)
        assert "error" not in result
        assert result["reconstruction_error"] < 1e-10
        assert "singular_values" in result
        assert "rank" in result

    def test_lu_happy_path(self) -> None:
        result = matrix_decomposition(matrix=FULL_RANK_3X3, method="lu", plot=False)
        assert "error" not in result
        assert result["reconstruction_error"] < 1e-10
        assert "P" in result and "L" in result and "U_factor" in result

    def test_qr_happy_path(self) -> None:
        result = matrix_decomposition(matrix=FULL_RANK_3X3, method="qr", plot=False)
        assert "error" not in result
        assert result["reconstruction_error"] < 1e-10
        assert "Q" in result and "R" in result

    def test_cholesky_happy_path(self) -> None:
        result = matrix_decomposition(matrix=PD_MATRIX, method="cholesky", plot=False)
        assert "error" not in result
        assert result["reconstruction_error"] < 1e-10
        assert "L" in result

    def test_cholesky_non_pd_error(self) -> None:
        indefinite = [[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        result = matrix_decomposition(matrix=indefinite, method="cholesky", plot=False)
        assert "error" in result

    def test_invalid_method_error(self) -> None:
        result = matrix_decomposition(matrix=FULL_RANK_3X3, method="foo", plot=False)
        assert "error" in result

    def test_lu_non_square_error(self) -> None:
        result = matrix_decomposition(matrix=[[1, 2, 3], [4, 5, 6]], method="lu", plot=False)
        assert "error" in result

    def test_svd_rank_identity(self) -> None:
        identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        result = matrix_decomposition(matrix=identity, method="svd", plot=False)
        assert result["rank"] == 3

    def test_method_returned_in_result(self) -> None:
        for m in ("svd", "lu", "qr"):
            result = matrix_decomposition(matrix=FULL_RANK_3X3, method=m, plot=False)
            assert result.get("method") == m

    def test_plot_returned_when_requested(self) -> None:
        result = matrix_decomposition(matrix=FULL_RANK_3X3, method="svd", plot=True)
        assert "plot" in result
        assert isinstance(result["plot"], str) and len(result["plot"]) > 100
