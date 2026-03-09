"""Tests for src/mcp_scicomp/utils.py."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mcp_scicomp.utils import parse_expr, resolve_data, resolve_matrix


# ---------------------------------------------------------------------------
# resolve_data — inline
# ---------------------------------------------------------------------------


def test_resolve_data_inline_basic():
    arr, meta = resolve_data(data=[1.0, 2.0, 3.0])
    np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])
    assert meta["source"] == "inline"
    assert meta["n"] == 3


def test_resolve_data_inline_empty():
    arr, meta = resolve_data(data=[])
    assert len(arr) == 0
    assert meta["n"] == 0


# ---------------------------------------------------------------------------
# resolve_data — file loading
# ---------------------------------------------------------------------------


def test_resolve_data_csv_single_column(tmp_path: Path):
    csv = tmp_path / "data.csv"
    csv.write_text("value\n1.0\n2.0\n3.0\n")
    arr, meta = resolve_data(file_path=str(csv))
    np.testing.assert_allclose(arr, [1.0, 2.0, 3.0])
    assert meta["selected_column"] == "value"
    assert meta["n"] == 3


def test_resolve_data_csv_explicit_column(tmp_path: Path):
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,4\n2,5\n3,6\n")
    arr, meta = resolve_data(file_path=str(csv), column="b")
    np.testing.assert_allclose(arr, [4.0, 5.0, 6.0])
    assert meta["selected_column"] == "b"


def test_resolve_data_csv_ambiguous_columns_raises(tmp_path: Path):
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,4\n2,5\n")
    with pytest.raises(ValueError, match="multiple numeric columns"):
        resolve_data(file_path=str(csv))


def test_resolve_data_tsv(tmp_path: Path):
    tsv = tmp_path / "data.tsv"
    tsv.write_text("x\ty\n1.0\t10.0\n2.0\t20.0\n")
    arr, meta = resolve_data(file_path=str(tsv), column="y")
    np.testing.assert_allclose(arr, [10.0, 20.0])


def test_resolve_data_json(tmp_path: Path):
    jfile = tmp_path / "data.json"
    jfile.write_text(json.dumps([{"v": 5.0}, {"v": 6.0}, {"v": 7.0}]))
    arr, meta = resolve_data(file_path=str(jfile), column="v")
    np.testing.assert_allclose(arr, [5.0, 6.0, 7.0])


def test_resolve_data_missing_file_raises():
    with pytest.raises(FileNotFoundError, match="not found"):
        resolve_data(file_path="/nonexistent/path/data.csv")


def test_resolve_data_missing_column_raises(tmp_path: Path):
    csv = tmp_path / "data.csv"
    csv.write_text("value\n1.0\n2.0\n")
    with pytest.raises(ValueError, match="not found"):
        resolve_data(file_path=str(csv), column="nonexistent")


def test_resolve_data_both_raises():
    with pytest.raises(ValueError, match="not both"):
        resolve_data(data=[1.0], file_path="some.csv")


def test_resolve_data_neither_raises():
    with pytest.raises(ValueError, match="must be provided"):
        resolve_data()


def test_resolve_data_unsupported_extension(tmp_path: Path):
    f = tmp_path / "data.xyz"
    f.write_text("1 2 3")
    with pytest.raises(ValueError, match="Unsupported"):
        resolve_data(file_path=str(f))


# ---------------------------------------------------------------------------
# resolve_matrix — inline
# ---------------------------------------------------------------------------


def test_resolve_matrix_inline_basic():
    mat = [[1.0, 2.0], [3.0, 4.0]]
    arr, meta = resolve_matrix(matrix=mat)
    assert arr.shape == (2, 2)
    assert meta["shape"] == [2, 2]
    assert meta["source"] == "inline"


def test_resolve_matrix_inline_not_2d_raises():
    with pytest.raises(ValueError, match="2-D"):
        resolve_matrix(matrix=[1.0, 2.0, 3.0])  # type: ignore[arg-type]


def test_resolve_matrix_both_raises():
    with pytest.raises(ValueError, match="not both"):
        resolve_matrix(matrix=[[1.0]], file_path="x.csv")


def test_resolve_matrix_neither_raises():
    with pytest.raises(ValueError, match="must be provided"):
        resolve_matrix()


def test_resolve_matrix_csv(tmp_path: Path):
    csv = tmp_path / "mat.csv"
    csv.write_text("a,b\n1,2\n3,4\n5,6\n")
    arr, meta = resolve_matrix(file_path=str(csv))
    assert arr.shape == (3, 2)
    assert meta["columns"] == ["a", "b"]
    np.testing.assert_allclose(arr[0], [1.0, 2.0])


# ---------------------------------------------------------------------------
# parse_expr
# ---------------------------------------------------------------------------


def test_parse_expr_linear():
    f = parse_expr("2*x + 1", {"x": "x"})
    result = f(np.array([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(result, [1.0, 3.0, 5.0])


def test_parse_expr_multivar():
    f = parse_expr("-theta * (x - mu)", {"x": "x", "theta": "theta", "mu": "mu"})
    val = f(1.0, 2.0, 0.5)  # x=1, theta=2, mu=0.5
    assert math.isclose(val, -2.0 * (1.0 - 0.5), rel_tol=1e-9)


def test_parse_expr_caret_replaced():
    """^ should be silently treated as ** (common user mistake)."""
    f = parse_expr("x^2", {"x": "x"})
    result = f(np.array([2.0, 3.0]))
    np.testing.assert_allclose(result, [4.0, 9.0])


def test_parse_expr_trig():
    f = parse_expr("sin(x)", {"x": "x"})
    result = f(np.pi / 2)
    assert math.isclose(float(result), 1.0, abs_tol=1e-9)


def test_parse_expr_invalid_raises():
    with pytest.raises(ValueError, match="Could not parse"):
        parse_expr(")(invalid((", {"x": "x"})


def test_parse_expr_constant():
    f = parse_expr("42", {})
    assert float(f()) == 42.0
