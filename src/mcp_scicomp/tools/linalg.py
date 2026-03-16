"""Linear algebra tools: eigenanalysis and matrix decompositions."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu

from mcp_scicomp.app import mcp
from mcp_scicomp.plotting import fig_to_base64
from mcp_scicomp.utils import resolve_matrix

logger = logging.getLogger(__name__)


@mcp.tool()
def eigenanalysis(
    matrix: Optional[list[list[float]]] = None,
    file_path: Optional[str] = None,
    plot: bool = True,
) -> dict[str, Any]:
    """Compute eigenvalues, eigenvectors, and spectral properties of a square matrix.

    Use this tool when you need to analyse the spectral structure of a matrix:
    stability of a dynamical system, PCA preprocessing, condition number
    diagnostics, or positive-definiteness checks.

    Args:
        matrix: Square matrix as a list-of-lists (rows × cols).
                Mutually exclusive with file_path.
        file_path: Path to a tabular file whose numeric columns form the matrix.
                   Mutually exclusive with matrix.
        plot: If True, return a scatter plot of eigenvalues in the complex plane
              with the unit circle drawn for reference. Default True.

    Returns:
        Dict with keys:
          - eigenvalues: list[float] if purely real, else {"real": [...], "imag": [...]}
          - eigenvectors: list[list[float]] — columns are eigenvectors
          - spectral_radius: float — max |eigenvalue|
          - condition_number: float
          - rank: int
          - determinant: float
          - is_symmetric: bool
          - is_positive_definite: bool | None  (None for non-symmetric matrices)
          - shape: [rows, cols]
          - plot: base64 PNG string (only present when plot=True)
          - error / suggestion: present only on failure

    Example:
        eigenanalysis(matrix=[[4, 2], [1, 3]])
        # Returns eigenvalues ≈ [5.0, 2.0] with spectral_radius=5.0
    """
    try:
        A, meta = resolve_matrix(matrix, file_path)

        rows, cols = A.shape
        if rows != cols:
            return {
                "error": f"eigenanalysis requires a square matrix; got shape {A.shape}.",
                "suggestion": "Provide an n×n matrix.",
            }

        is_symmetric = bool(np.allclose(A, A.T))

        if is_symmetric:
            vals_real = np.linalg.eigvalsh(A)  # real, sorted ascending
            # eigvalsh doesn't return vecs in this call; use eigh for both
            vals_real, vecs = np.linalg.eigh(A)
            eigenvalues: Any = [float(v) for v in vals_real]
            eigenvectors = [[float(vecs[r, c]) for r in range(rows)] for c in range(cols)]
        else:
            vals, vecs = np.linalg.eig(A)
            # Decide real vs complex representation
            if np.allclose(vals.imag, 0.0):
                eigenvalues = [float(v.real) for v in vals]
            else:
                eigenvalues = {
                    "real": [float(v.real) for v in vals],
                    "imag": [float(v.imag) for v in vals],
                }
            eigenvectors = [
                [float(vecs[r, c].real) for r in range(rows)] for c in range(cols)
            ]

        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(A))))
        condition_number = float(np.linalg.cond(A))
        rank = int(np.linalg.matrix_rank(A))
        determinant = float(np.linalg.det(A))

        is_positive_definite: Optional[bool] = None
        if is_symmetric:
            try:
                np.linalg.cholesky(A)
                is_positive_definite = True
            except np.linalg.LinAlgError:
                is_positive_definite = False

        result: dict[str, Any] = {
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "spectral_radius": spectral_radius,
            "condition_number": condition_number,
            "rank": rank,
            "determinant": determinant,
            "is_symmetric": is_symmetric,
            "is_positive_definite": is_positive_definite,
            "shape": list(A.shape),
        }

        if plot:
            vals_all = np.linalg.eigvals(A)
            fig, ax = plt.subplots(figsize=(6, 6))
            theta = np.linspace(0, 2 * np.pi, 300)
            ax.plot(np.cos(theta), np.sin(theta), "k--", linewidth=0.8, alpha=0.5, label="unit circle")
            ax.scatter(vals_all.real, vals_all.imag, color="steelblue", zorder=5, s=60, label="eigenvalues")
            ax.axhline(0, color="grey", linewidth=0.5)
            ax.axvline(0, color="grey", linewidth=0.5)
            ax.set_xlabel("Real")
            ax.set_ylabel("Imaginary")
            ax.set_title("Eigenvalue Spectrum")
            ax.set_aspect("equal")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            result["plot"] = fig_to_base64(fig)

        return result

    except Exception as exc:
        logger.exception("eigenanalysis failed")
        return {"error": str(exc), "suggestion": "Check that the matrix is square and numeric."}


@mcp.tool()
def matrix_decomposition(
    matrix: Optional[list[list[float]]] = None,
    file_path: Optional[str] = None,
    method: str = "svd",
    plot: bool = True,
) -> dict[str, Any]:
    """Decompose a matrix using SVD, LU, QR, or Cholesky factorisation.

    Use this tool for dimensionality reduction (SVD), solving linear systems
    (LU/QR), verifying positive-definiteness (Cholesky), or inspecting the
    rank structure of a matrix.

    Args:
        matrix: Matrix as a list-of-lists (rows × cols).
                Mutually exclusive with file_path.
        file_path: Path to a tabular file whose numeric columns form the matrix.
                   Mutually exclusive with matrix.
        method: Factorisation to apply. One of:
                  "svd"       — Singular Value Decomposition (any shape)
                  "lu"        — LU decomposition with partial pivoting (square)
                  "qr"        — QR decomposition (any shape)
                  "cholesky"  — Cholesky L L^T (square, symmetric, positive-definite)
                Default "svd".
        plot: If True, return a visualisation of the decomposition:
              SVD → bar chart of singular values;
              LU/QR/Cholesky → heatmap of the primary factor. Default True.

    Returns:
        Dict with keys (vary by method):
          SVD:       U, singular_values, Vt, rank, reconstruction_error
          LU:        P, L, U_factor, reconstruction_error
          QR:        Q, R, reconstruction_error
          Cholesky:  L, reconstruction_error
          All:       method, shape, plot (optional)
          On error:  error, suggestion

    Example:
        matrix_decomposition(matrix=[[4,3],[6,3]], method="lu")
        # Returns P, L, U factors with reconstruction_error < 1e-14
    """
    try:
        A, meta = resolve_matrix(matrix, file_path)

        valid_methods = {"svd", "lu", "qr", "cholesky"}
        if method not in valid_methods:
            return {
                "error": f"Unknown method '{method}'. Choose from: {sorted(valid_methods)}.",
                "suggestion": "Use one of: svd, lu, qr, cholesky.",
            }

        result: dict[str, Any] = {"method": method, "shape": list(A.shape)}

        if method == "svd":
            U, s, Vt = np.linalg.svd(A, full_matrices=False)
            A_rec = U @ np.diag(s) @ Vt
            threshold = float(np.finfo(float).eps * max(A.shape) * s[0]) if s.size > 0 else 0.0
            rank = int(np.sum(s > threshold))
            result.update(
                {
                    "U": [[float(U[r, c]) for c in range(U.shape[1])] for r in range(U.shape[0])],
                    "singular_values": [float(v) for v in s],
                    "Vt": [[float(Vt[r, c]) for c in range(Vt.shape[1])] for r in range(Vt.shape[0])],
                    "rank": rank,
                    "reconstruction_error": float(np.linalg.norm(A - A_rec)),
                }
            )
            if plot:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(range(1, len(s) + 1), s, color="steelblue")
                ax.set_xlabel("Singular value index")
                ax.set_ylabel("Value")
                ax.set_title("Singular Value Spectrum")
                ax.grid(True, alpha=0.3, axis="y")
                fig.tight_layout()
                result["plot"] = fig_to_base64(fig)

        elif method == "lu":
            rows, cols = A.shape
            if rows != cols:
                return {
                    "error": f"LU decomposition requires a square matrix; got {A.shape}.",
                    "suggestion": "Use method='svd' or 'qr' for non-square matrices.",
                }
            P, L, U_fac = lu(A)
            A_rec = P @ L @ U_fac
            result.update(
                {
                    "P": [[float(P[r, c]) for c in range(P.shape[1])] for r in range(P.shape[0])],
                    "L": [[float(L[r, c]) for c in range(L.shape[1])] for r in range(L.shape[0])],
                    "U_factor": [[float(U_fac[r, c]) for c in range(U_fac.shape[1])] for r in range(U_fac.shape[0])],
                    "reconstruction_error": float(np.linalg.norm(A - A_rec)),
                }
            )
            if plot:
                fig, ax = plt.subplots(figsize=(5, 4))
                im = ax.imshow(L, cmap="RdBu_r", aspect="auto")
                plt.colorbar(im, ax=ax)
                ax.set_title("L factor (LU decomposition)")
                fig.tight_layout()
                result["plot"] = fig_to_base64(fig)

        elif method == "qr":
            Q, R = np.linalg.qr(A)
            A_rec = Q @ R
            result.update(
                {
                    "Q": [[float(Q[r, c]) for c in range(Q.shape[1])] for r in range(Q.shape[0])],
                    "R": [[float(R[r, c]) for c in range(R.shape[1])] for r in range(R.shape[0])],
                    "reconstruction_error": float(np.linalg.norm(A - A_rec)),
                }
            )
            if plot:
                fig, ax = plt.subplots(figsize=(5, 4))
                im = ax.imshow(R, cmap="RdBu_r", aspect="auto")
                plt.colorbar(im, ax=ax)
                ax.set_title("R factor (QR decomposition)")
                fig.tight_layout()
                result["plot"] = fig_to_base64(fig)

        elif method == "cholesky":
            rows, cols = A.shape
            if rows != cols:
                return {
                    "error": f"Cholesky requires a square matrix; got {A.shape}.",
                    "suggestion": "Provide a square symmetric positive-definite matrix.",
                }
            if not np.allclose(A, A.T):
                return {
                    "error": "Cholesky requires a symmetric matrix.",
                    "suggestion": "Symmetrize with (A + A.T) / 2 before passing.",
                }
            try:
                L = np.linalg.cholesky(A)
            except np.linalg.LinAlgError:
                return {
                    "error": "Matrix is not positive-definite; Cholesky failed.",
                    "suggestion": "Use method='lu' or verify the matrix is positive-definite.",
                }
            A_rec = L @ L.T
            result.update(
                {
                    "L": [[float(L[r, c]) for c in range(L.shape[1])] for r in range(L.shape[0])],
                    "reconstruction_error": float(np.linalg.norm(A - A_rec)),
                }
            )
            if plot:
                fig, ax = plt.subplots(figsize=(5, 4))
                im = ax.imshow(L, cmap="RdBu_r", aspect="auto")
                plt.colorbar(im, ax=ax)
                ax.set_title("L factor (Cholesky decomposition)")
                fig.tight_layout()
                result["plot"] = fig_to_base64(fig)

        return result

    except Exception as exc:
        logger.exception("matrix_decomposition failed")
        return {"error": str(exc), "suggestion": "Check matrix dimensions and method choice."}
