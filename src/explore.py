"""W4：退化曲线与个体差异。

产出
----
fig01_capacity_degradation.png
    四个体的容量退化曲线，含 EOL 阈值线、额定容量线、EOL 交叉点、容量再生事件点，
    以及 B0007 的右删失标注。
fig02_individual_differences.png
    两个子图：(a) 绝对健康度 SOH = C/2.0；(b) 相对首循环归一化。
    前者回答"离额定容量还有多远"，后者回答"去掉初始容量差异后，衰减形状是否不同"。

为什么分成这两张
----------------
绝对曲线（Ah）便于读数和对照阈值；但个体初始容量本身差 9.5%（1.8550–2.0353 Ah），
直接比高低会被初始差异淹没。所以第二张同时给"相对额定"和"相对自身首循环"两个视图。
**注意**：相对首循环那一列是目标变量的仿射变换，只能用于展示，禁止作预测特征（D14）。

运行
----
    python src/explore.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402, F401
    BATTERIES,
    COLORS,
    EOL_THRESHOLD_AH,
    RATED_CAPACITY_AH,
    setup_matplotlib,
)
from figures import draw_eol_line, draw_rated_line, label_right, save  # noqa: E402
from load_data import load_raw, split_by_battery  # noqa: E402
from preprocess import cell_profile, mark_rows  # noqa: E402


def _is_missing(value) -> bool:
    """统一判空：pd.NA / np.nan / None 都要能认出来。

    踩过一次坑：`eol is not None` 对 `pd.NA` 是 True，而
    `isinstance(pd.NA, float)` 是 False，于是 pd.NA 一路溜到 `int()` 才炸。
    """
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):  # 非标量
        return False


def _eol_handle():
    """图例用的「首次跌破 EOL」三角标记（不参与实际绘图）。"""
    from matplotlib.lines import Line2D

    return Line2D(
        [0],
        [0],
        marker="v",
        color="none",
        markerfacecolor="#666666",
        markeredgecolor="white",
        markersize=8,
        label="首次跌破 EOL 阈值",
    )


def fig01_capacity_degradation(df, prof) -> Path:
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(9.2, 5.4))

    by = split_by_battery(df)
    for cell, sub in by.items():
        ax.plot(
            sub["cycle"],
            sub["capacity_Ah"],
            color=COLORS[cell],
            lw=1.6,
            label=cell,
            zorder=3,
        )
        # 容量再生事件：空心圈，避免盖住曲线
        ev = sub[sub["is_regen"]]
        if len(ev):
            ax.scatter(
                ev["cycle"],
                ev["capacity_Ah"],
                s=34,
                facecolors="none",
                edgecolors=COLORS[cell],
                linewidths=1.3,
                zorder=4,
            )
        # EOL 交叉点
        eol = prof.loc[cell, "eol_cycle_at_1p4"]
        if not _is_missing(eol):
            ax.scatter(
                [int(eol)],
                [EOL_THRESHOLD_AH],
                s=52,
                marker="v",
                color=COLORS[cell],
                edgecolors="white",
                linewidths=0.8,
                zorder=5,
            )

    draw_rated_line(ax)
    draw_eol_line(ax)

    # 图例：合并数据线与阈值线
    handles, labels = ax.get_legend_handles_labels()
    # 手工补一个"再生事件"图例项
    from matplotlib.lines import Line2D

    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="none",
            markeredgecolor="#666666",
            markersize=7,
            label="容量再生事件（相邻回升 > 0.05 Ah）",
        )
    )
    handles.append(
        Line2D(
            [0],
            [0],
            marker="v",
            color="none",
            markerfacecolor="#666666",
            markeredgecolor="white",
            markersize=8,
            label="首次跌破 EOL 阈值",
        )
    )
    handles.append(
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor="#2ca02c",
            markeredgecolor="white",
            markersize=13,
            label="B0007 全程未跌破（右删失）",
        )
    )
    labels = labels + [
        "容量再生事件（相邻回升 > 0.05 Ah）",
        "首次跌破 EOL 阈值",
        "B0007 全程未跌破（右删失）",
    ]
    ax.legend(handles, labels, loc="upper right", fontsize=9, framealpha=0.92)

    # B0007 删失标注：箭头指向它的最低点
    b7 = by["B0007"]
    imin = b7["capacity_Ah"].idxmin()
    x7, y7 = int(b7.loc[imin, "cycle"]), float(b7.loc[imin, "capacity_Ah"])
    # 用星标点出 B0007 的最低点，文字放在左下空白区。
    # 不画跨越全图的箭头——那样箭头端点容易被曲线遮住，读者对不准。
    ax.scatter(
        [x7],
        [y7],
        marker="*",
        s=240,
        color="#2ca02c",
        edgecolors="white",
        linewidths=0.9,
        zorder=7,
    )
    ax.text(
        6,
        1.115,
        f"B0007 最低 {y7:.4f} Ah @ 第 {x7} 循环，距阈值仅 {y7 - EOL_THRESHOLD_AH:+.4f} Ah\n"
        "全程未跌破 1.4 Ah → 右删失（观察期结束仍未失效，寿命只能记为 > 168）",
        fontsize=9,
        color="#2ca02c",
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#2ca02c", alpha=0.95),
        zorder=6,
    )

    ax.set_xlabel("循环次数")
    ax.set_ylabel("放电容量 (Ah)")
    ax.set_title("图1　四个电池个体的容量退化曲线（24 °C 恒温）", fontsize=12)
    ax.set_xlim(0, 175)
    ax.set_ylim(1.05, 2.10)
    return save(fig, "fig01_capacity_degradation.png")


def fig02_individual_differences(df, prof) -> Path:
    setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9))

    by = split_by_battery(df)

    # --- (a) 绝对健康度：SOH = C / 额定容量 ---
    ax = axes[0]
    for cell, sub in by.items():
        ax.plot(sub["cycle"], sub["cap_over_rated"], color=COLORS[cell], lw=1.6, label=cell)
        eol = prof.loc[cell, "eol_cycle_at_1p4"]
        if not _is_missing(eol):
            ax.scatter(
                [int(eol)],
                [EOL_THRESHOLD_AH / RATED_CAPACITY_AH],
                s=62,
                marker="v",
                color=COLORS[cell],
                edgecolors="white",
                linewidths=0.9,
                zorder=6,
            )
    ax.axhline(1.0, ls=":", lw=1.2, color="#999999")
    ax.axhline(EOL_THRESHOLD_AH / RATED_CAPACITY_AH, ls="--", lw=1.4, color="#555555")
    label_right(ax, f"额定容量 = 1.0", 1.0)
    label_right(ax, f"EOL = 70% 额定", EOL_THRESHOLD_AH / RATED_CAPACITY_AH)
    ax.annotate(
        "B0006 初始 SOH > 100%\n（早期 7 个循环超过额定 2 Ah）",
        xy=(4, 2.0353 / RATED_CAPACITY_AH),
        xytext=(38, 1.055),
        fontsize=8.5,
        color="#d62728",
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.1),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d62728", alpha=0.9),
    )
    ax.set_xlabel("循环次数")
    ax.set_ylabel("SOH = 容量 / 额定容量")
    ax.set_title("(a) 相对额定容量的健康度", fontsize=11)
    ax.set_xlim(0, 175)
    ax.set_ylim(0.52, 1.10)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(_eol_handle())
    labels.append("首次跌破 EOL 阈值")
    ax.legend(handles, labels, fontsize=9, loc="lower left")

    # --- (b) 相对首循环：去掉初始容量差异 ---
    ax = axes[1]
    for cell, sub in by.items():
        ax.plot(sub["cycle"], sub["cap_over_first"], color=COLORS[cell], lw=1.6, label=cell)
        eol = prof.loc[cell, "eol_cycle_at_1p4"]
        if not _is_missing(eol):
            ratio = EOL_THRESHOLD_AH / float(sub["capacity_Ah"].iloc[0])
            ax.scatter([int(eol)], [ratio], s=48, marker="v", color=COLORS[cell],
                       edgecolors="white", linewidths=0.8, zorder=5)
    ax.axhline(1.0, ls=":", lw=1.2, color="#999999")
    label_right(ax, "起点统一归一到 1.0", 1.0)
    ax.set_xlabel("循环次数")
    ax.set_ylabel("容量 / 该电池首循环容量")
    ax.set_title("(b) 去掉初始容量差异后的衰减形状", fontsize=11)
    ax.set_xlim(0, 175)
    ax.set_ylim(0.55, 1.06)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(_eol_handle())
    labels.append("首次跌破 EOL 阈值")
    ax.legend(handles, labels, fontsize=9, loc="lower left")

    fig.suptitle(
        "图2　个体差异：绝对健康度 vs 相对衰减形状（(b) 仅用于展示，不得作特征）",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    return save(fig, "fig02_individual_differences.png")


def main() -> int:
    raw = load_raw()
    df = mark_rows(raw)
    prof = cell_profile(df).set_index("battery_id")

    p1 = fig01_capacity_degradation(df, prof)
    p2 = fig02_individual_differences(df, prof)

    for p in (p1, p2):
        size_kb = p.stat().st_size / 1024
        print(f"已生成 {p.name}  ({size_kb:.0f} KB)")

    # 顺带把关键数字打印出来，便于对照图
    print("\nEOL 交叉点（阈值 1.4 Ah）：")
    for cell in prof.index:
        eol = prof.loc[cell, "eol_cycle_at_1p4"]
        shown = "未到达（右删失）" if _is_missing(eol) else f"第 {int(eol)} 循环"
        print(f"  {cell}: {shown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
