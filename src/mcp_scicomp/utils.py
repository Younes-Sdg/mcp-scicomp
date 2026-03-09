"""Shared utility helpers for mcp-scicomp tools."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import sympy as sp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

_LOADERS: dict[str, str] = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".txt": "csv",
    ".xlsx": "excel",
    ".xls": "excel",
    ".json": "json",
    ".parquet": "parquet",
    ".pq": "parquet",
}


def _load_file(path: Path, column: Optional[str]) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a tabular file and extract one column as a 1-D array."""
    import pandas as pd  # local import — optional heavy dep

    ext = path.suffix.lower()
    fmt = _LOADERS.get(ext)
    if fmt is None:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            f"Supported: {', '.join(_LOADERS)}"
        )

    if fmt == "csv":
        df = pd.read_csv(path)
    elif fmt == "tsv":
        df = pd.read_csv(path, sep="\t")
    elif fmt == "excel":
        df = pd.read_excel(path)
    elif fmt == "json":
        df = pd.read_json(path)
    elif fmt == "parquet":
        df = pd.read_parquet(path)
    else:  # pragma: no cover
        raise ValueError(f"Internal error: unknown format '{fmt}'")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    meta: dict[str, Any] = {
        "file": str(path),
        "format": fmt,
        "rows": len(df),
        "columns": df.columns.tolist(),
        "numeric_columns": numeric_cols,
    }

    if column is None:
        if len(numeric_cols) == 1:
            column = numeric_cols[0]
        elif len(numeric_cols) == 0:
            raise ValueError(
                f"No numeric columns found in '{path}'. "
                f"Columns present: {df.columns.tolist()}"
            )
        else:
            raise ValueError(
                f"File '{path}' has multiple numeric columns: {numeric_cols}. "
                "Specify 'column' to select one."
            )

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found in '{path}'. "
            f"Available columns: {df.columns.tolist()}"
        )

    meta["selected_column"] = column
    arr = df[column].dropna().to_numpy(dtype=float)
    return arr, meta


def resolve_data(
    data: Optional[list[float]] = None,
    file_path: Optional[str] = None,
    column: Optional[str] = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a 1-D numpy array from either an inline list or a file.

    Exactly one of *data* or *file_path* must be supplied.

    Args:
        data: Inline list of floats. Mutually exclusive with file_path.
        file_path: Path to a data file. Supports CSV (.csv/.txt), TSV (.tsv),
                   Excel (.xlsx/.xls), JSON (.json), Parquet (.parquet/.pq).
                   ~ is expanded to the home directory.
        column: Column name to extract when loading from a file. Required when
                the file has more than one numeric column; ignored for inline data.

    Returns:
        Tuple of (array, metadata_dict).  metadata_dict always contains 'source'
        ('inline' or the file path) and 'n' (number of elements).

    Raises:
        ValueError: If neither or both of data/file_path are provided, if the
                    file is missing, or if column selection is ambiguous.
    """
    if data is not None and file_path is not None:
        raise ValueError("Provide either 'data' or 'file_path', not both.")
    if data is None and file_path is None:
        raise ValueError("One of 'data' or 'file_path' must be provided.")

    if data is not None:
        arr = np.asarray(data, dtype=float)
        meta: dict[str, Any] = {"source": "inline", "n": len(arr)}
        return arr, meta

    # file path branch
    path = Path(file_path).expanduser()  # type: ignore[arg-type]
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: '{path}'. Check the path and try again."
        )
    arr, meta = _load_file(path, column)
    meta["source"] = str(path)
    meta["n"] = len(arr)
    return arr, meta


def resolve_matrix(
    matrix: Optional[list[list[float]]] = None,
    file_path: Optional[str] = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a 2-D numpy array from either an inline nested list or a file.

    Exactly one of *matrix* or *file_path* must be supplied.

    Args:
        matrix: Inline list-of-lists of floats (rows × cols).
        file_path: Path to a tabular file (same formats as resolve_data).
                   All numeric columns are loaded; non-numeric columns are dropped.
                   ~ is expanded to the home directory.

    Returns:
        Tuple of (2-D array, metadata_dict) where metadata_dict contains
        'source', 'shape', and (for files) 'columns'.

    Raises:
        ValueError: If neither or both inputs are provided, or shape is inconsistent.
    """
    if matrix is not None and file_path is not None:
        raise ValueError("Provide either 'matrix' or 'file_path', not both.")
    if matrix is None and file_path is None:
        raise ValueError("One of 'matrix' or 'file_path' must be provided.")

    if matrix is not None:
        arr = np.asarray(matrix, dtype=float)
        if arr.ndim != 2:
            raise ValueError(
                f"'matrix' must be a 2-D list-of-lists; got shape {arr.shape}."
            )
        meta: dict[str, Any] = {"source": "inline", "shape": list(arr.shape)}
        return arr, meta

    import pandas as pd

    path = Path(file_path).expanduser()  # type: ignore[arg-type]
    if not path.exists():
        raise FileNotFoundError(f"File not found: '{path}'.")

    ext = path.suffix.lower()
    fmt = _LOADERS.get(ext)
    if fmt is None:
        raise ValueError(f"Unsupported file extension '{ext}'.")

    if fmt == "csv":
        df = pd.read_csv(path)
    elif fmt == "tsv":
        df = pd.read_csv(path, sep="\t")
    elif fmt == "excel":
        df = pd.read_excel(path)
    elif fmt == "json":
        df = pd.read_json(path)
    elif fmt == "parquet":
        df = pd.read_parquet(path)
    else:  # pragma: no cover
        raise ValueError(f"Internal error: unknown format '{fmt}'")

    num_df = df.select_dtypes(include="number")
    if num_df.empty:
        raise ValueError(f"No numeric columns found in '{path}'.")

    arr = num_df.to_numpy(dtype=float)
    meta = {
        "source": str(path),
        "shape": list(arr.shape),
        "columns": num_df.columns.tolist(),
    }
    return arr, meta


# ---------------------------------------------------------------------------
# Expression parser
# ---------------------------------------------------------------------------

# Safe symbol set exposed to user expressions
_SAFE_NUMPY_MODULES = ["numpy"]


def parse_expr(
    expr_str: str,
    symbols: dict[str, str],
) -> Callable[..., Any]:
    """Parse a mathematical expression string and return a numpy-callable function.

    Uses sympy.sympify with a restricted namespace — raw eval() is never called.

    Args:
        expr_str: A mathematical expression as a string, e.g. ``"-theta*(x - mu)"``.
                  Use ``**`` for exponentiation (not ``^``).
        symbols: Ordered mapping of parameter name → SymPy symbol name,
                 e.g. ``{"x": "x", "theta": "theta", "mu": "mu"}``.
                 The returned function's positional arguments follow dict insertion order.

    Returns:
        A numpy-vectorized callable ``f(*symbol_values) -> np.ndarray``.

    Raises:
        ValueError: If the expression cannot be parsed or contains unsafe constructs.

    Example:
        >>> f = parse_expr("-theta*(x - mu)", {"x": "x", "theta": "theta", "mu": "mu"})
        >>> f(np.linspace(0, 1, 5), 2.0, 0.5)
    """
    # Replace ^ with ** for user convenience (common mistake)
    expr_str = expr_str.replace("^", "**")

    # Build a restricted local namespace: only the declared symbols
    sym_objects = {name: sp.Symbol(sym_name) for name, sym_name in symbols.items()}

    try:
        expr = sp.sympify(expr_str, locals=sym_objects, evaluate=True)
    except (sp.SympifyError, SyntaxError, TypeError) as exc:
        raise ValueError(
            f"Could not parse expression '{expr_str}': {exc}. "
            "Use ** for exponentiation, not ^."
        ) from exc

    # Lambdify in insertion order so callers can pass positional args predictably
    sym_list = list(sym_objects.values())
    func = sp.lambdify(sym_list, expr, modules=_SAFE_NUMPY_MODULES)
    return func
