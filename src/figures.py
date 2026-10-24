"""绘图公共工具：保存、阈值线、删失标注。

只放"多张图都会用到"的东西。具体每张图画什么，放到各自的模块里
（W4–W7 的探索性图表在 explore.py）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import EOL_THRESHOLD_AH, FIG_DPI, FIG_DIR, RATED_CAPACITY_AH, ensure_dirs  # noqa: E402

EOL_LABEL = f"EOL 阈值 {EOL_THRESHOLD_AH:g} Ah（额定 {RATED_CAPACITY_AH:g} Ah 的 70%）"


def save(fig: plt.Figure, name: str) -> Path:
    """保存到 outputs/figures/ 并关闭图形，避免内存堆积。"""
    ensure_dirs()
    path = FIG_DIR / name
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def draw_eol_line(ax: plt.Axes, label: str = "", y: float | None = None) -> None:
    """画 EOL 阈值横线（虚线 + 右端文字）。"""
    y = EOL_THRESHOLD_AH if y is None else y
    ax.axhline(y, ls="--", lw=1.4, color="#555555", zorder=1, label=label or None)


def draw_rated_line(ax: plt.Axes) -> None:
    """画额定容量横线（点线），用于展示 B0006 超过额定的现象。"""
    ax.axhline(
        RATED_CAPACITY_AH,
        ls=":",
        lw=1.2,
        color="#999999",
        zorder=1,
        label=f"额定容量 {RATED_CAPACITY_AH:g} Ah",
    )


def label_right(ax: plt.Axes, text: str, y: float, **kwargs) -> None:
    """在图的右侧边缘、给定 y 处放一个右对齐的小字标签。"""
    ax.annotate(
        text,
        xy=(1.0, y),
        xycoords=("axes fraction", "data"),
        xytext=(4, 0),
        textcoords="offset points",
        va="center",
        ha="left",
        fontsize=9,
        color="#555555",
        annotation_clip=False,
        **kwargs,
    )
