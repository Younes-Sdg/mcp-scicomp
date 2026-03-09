"""Shared plotting helpers for mcp-scicomp tools."""

from __future__ import annotations

import base64
import io
import logging
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # non-interactive backend, safe for server use

logger = logging.getLogger(__name__)


def fig_to_base64(fig: plt.Figure, dpi: int = 100) -> str:
    """Convert a matplotlib Figure to a base64-encoded PNG string and close it.

    Args:
        fig: The matplotlib Figure to encode.
        dpi: Resolution for the PNG output. Default 100.

    Returns:
        Base64-encoded PNG string (no data-URI prefix).
    """
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    finally:
        plt.close(fig)
        buf.close()


def quick_plot(
    x: list[float],
    y: list[float] | list[list[float]],
    title: str = "",
    xlabel: str = "x",
    ylabel: str = "y",
    labels: Optional[list[str]] = None,
    dpi: int = 100,
) -> str:
    """Create a line plot and return it as a base64 PNG string.

    Supports a single series (y is a flat list) or multiple series
    (y is a list of lists, each the same length as x).

    Args:
        x: x-axis values.
        y: y-axis values. Either a flat list (one series) or a list of lists
           (multiple series). Each inner list must have the same length as x.
        title: Plot title. Default "".
        xlabel: x-axis label. Default "x".
        ylabel: y-axis label. Default "y".
        labels: Series labels for the legend. Ignored for single series unless
                provided. Default None.
        dpi: Output resolution. Default 100.

    Returns:
        Base64-encoded PNG string.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    x_arr = np.asarray(x, dtype=float)

    # Detect multi-series: y is a list of lists
    is_multi = y and isinstance(y[0], (list, tuple, np.ndarray))

    if is_multi:
        series_list = y  # type: ignore[assignment]
        for i, series in enumerate(series_list):
            label = labels[i] if labels and i < len(labels) else f"series {i + 1}"
            ax.plot(x_arr, np.asarray(series, dtype=float), label=label)
        ax.legend()
    else:
        label = labels[0] if labels else None
        ax.plot(x_arr, np.asarray(y, dtype=float), label=label)
        if label:
            ax.legend()

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig_to_base64(fig, dpi=dpi)
