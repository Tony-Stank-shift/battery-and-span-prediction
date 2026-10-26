"""W4：退化曲线与个体差异。

产出
----
fig01_capacity_degradation.png
    四个体的容量退化曲线，含 EOL 阈值线、额定容量线、EOL 交叉点、容量再生事件点，
    以及 B0007 的右删失标注。
fig02_individual_differences.png
    两个子图：(a) 绝对健康度 SOH = C/2.0；(b) 相对首循环归一化。
    前者回答「离额定容量还有多远」，后者回答「去掉初始容量差异后，衰减形状是否不同」。

为什么分成这两张
----------------
绝对曲线（Ah）便于读数和对照阈值；但个体初始容量本身差 9.5%（1.8550–2.0353 Ah），
直接比高低会被初始差异淹没。所以第二张同时给「相对额定」和「相对自身首循环」两个视图。
**注意**：相对首循环那一列是目标变量的仿射变换，只能用于展示，禁止作预测特征（D14）。

关于坐标轴
----------
所有轴界都由数据范围自动生成，并在设置后断言「数据没有被裁掉」。
早期版本把 ylim 写死成常数，第一轮审查指出：换一份数据就会静默裁剪曲线而不报错。

运行
----
    python src/explore.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    COLORS,
    EOL_THRESHOLD_AH,
    RATED_CAPACITY_AH,
    is_missing as _is_missing,
    setup_matplotlib,
)
from figures import EOL_LABEL, draw_eol_line, draw_rated_line, label_right, save  # noqa: E402
from load_data import load_raw, split_by_battery  # noqa: E402
from preprocess import cell_profile, mark_rows  # noqa: E402


def _set_limits(
    ax: plt.Axes,
    x_arrays: list,
    y_arrays: list,
    pad_frac: float = 0.06,
    top_frac: float = 0.0,
) -> None:
    """按数据范围设轴界，并断言没有把数据裁到可视区外。"""
    xs = np.concatenate([np.asarray(a, dtype=float).ravel() for a in x_arrays])
    ys = np.concatenate([np.asarray(a, dtype=float).ravel() for a in y_arrays])
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    xr = (x1 - x0) or 1.0
    yr = (y1 - y0) or 1.0

    ax.set_xlim(x0 - pad_frac * xr, x1 + pad_frac * xr)
    ax.set_ylim(y0 - pad_frac * yr, y1 + (pad_frac + top_frac) * yr)

    assert ax.get_xlim()[0] <= x0 and ax.get_xlim()[1] >= x1, "x 轴把数据裁掉了"
    assert ax.get_ylim()[0] <= y0 and ax.get_ylim()[1] >= y1, "y 轴把数据裁掉了"


def fig01_capacity_degradation(df, prof) -> Path:
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(9.2, 5.6))

    by = split_by_battery(df)
    xs: list = []
    ys: list = []

    for cell, sub in by.items():
        ax.plot(
            sub["cycle"],
            sub["capacity_Ah"],
            color=COLORS[cell],
            lw=1.6,
            label=cell,
            zorder=3,
        )
        xs.append(sub["cycle"].to_numpy())
        ys.append(sub["capacity_Ah"].to_numpy())

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
        eol = prof.loc[cell, "eol_cycle"]
        if not _is_missing(eol):
            ax.scatter(
                [int(eol)],
                [EOL_THRESHOLD_AH],
                s=62,
                marker="v",
                color=COLORS[cell],
                edgecolors="white",
                linewidths=0.9,
                zorder=5,
            )

    draw_rated_line(ax)
    draw_eol_line(ax, label=EOL_LABEL)

    handles, labels = ax.get_legend_handles_labels()
    extra = [
        (
            Line2D(
                [0], [0], marker="o", color="none", markerfacecolor="none",
                markeredgecolor="#666666", markersize=7,
            ),
            "容量再生事件（相邻回升 > 0.05 Ah）",
        ),
        (
            Line2D(
                [0], [0], marker="v", color="none", markerfacecolor="#666666",
                markeredgecolor="white", markersize=8,
            ),
            "首次跌破 EOL 阈值",
        ),
        (
            Line2D(
                [0], [0], marker="*", color="none", markerfacecolor="#2ca02c",
                markeredgecolor="white", markersize=13,
            ),
            "B0007 全程未跌破（右删失）",
        ),
    ]
    handles += [h for h, _ in extra]
    labels += [t for _, t in extra]
    ax.legend(handles, labels, loc="upper right", fontsize=9, framealpha=0.92)

    # B0007 删失标注：星标定点 + 左下说明（不画跨越全图的箭头，端点容易被曲线遮住）
    b7 = by["B0007"]
    imin = b7["capacity_Ah"].idxmin()
    x7, y7 = int(b7.loc[imin, "cycle"]), float(b7.loc[imin, "capacity_Ah"])
    ax.scatter(
        [x7], [y7], marker="*", s=240, color="#2ca02c",
        edgecolors="white", linewidths=0.9, zorder=7,
    )
    ax.text(
        0.02,
        0.03,
        f"B0007 最低 {y7:.4f} Ah @ 第 {x7} 循环，距阈值仅 {y7 - EOL_THRESHOLD_AH:+.4f} Ah\n"
        f"全程未跌破 {EOL_THRESHOLD_AH:g} Ah → 右删失"
        f"（观察期结束仍未失效，寿命只能记为 > {len(b7)}）",
        transform=ax.transAxes,
        fontsize=9,
        color="#2ca02c",
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#2ca02c", alpha=0.95),
        zorder=6,
    )

    ax.set_xlabel("循环次数")
    ax.set_ylabel("放电容量 (Ah)")
    temp_nunique = df["ambient_temp_C"].nunique()
    if temp_nunique == 1:
        temp_note = f"{df['ambient_temp_C'].iloc[0]:g} °C 恒温"
    else:
        temp_note = f"{temp_nunique} 种温度"
    ax.set_title(f"四个电池个体的容量退化曲线（{temp_note}）", fontsize=12)
    ys.append(np.array([EOL_THRESHOLD_AH, RATED_CAPACITY_AH]))
    _set_limits(ax, xs, ys, pad_frac=0.05)
    return save(fig, "fig01_capacity_degradation.png")


def fig02_individual_differences(df, prof) -> Path:
    setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))

    by = split_by_battery(df)

    # --- (a) 绝对健康度：SOH = C / 额定容量 ---
    ax = axes[0]
    xs: list = []
    ys: list = []
    for cell, sub in by.items():
        ax.plot(sub["cycle"], sub["cap_over_rated"], color=COLORS[cell], lw=1.6, label=cell)
        xs.append(sub["cycle"].to_numpy())
        ys.append(sub["cap_over_rated"].to_numpy())

        eol = prof.loc[cell, "eol_cycle"]
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
    label_right(ax, "额定容量 = 1.0", 1.0)
    label_right(ax, f"EOL = {EOL_THRESHOLD_AH / RATED_CAPACITY_AH:.0%} 额定", EOL_THRESHOLD_AH / RATED_CAPACITY_AH)

    # B0006 初始超过额定：坐标与数值都从数据取
    b6 = by["B0006"]
    i6 = b6["capacity_Ah"].idxmax()
    ax.annotate(
        f"B0006 初始 SOH = {float(b6.loc[i6, 'cap_over_rated']):.1%} > 100%\n（早期循环超过额定 2 Ah）",
        xy=(int(b6.loc[i6, "cycle"]), float(b6.loc[i6, "cap_over_rated"])),
        xytext=(0.20, 0.93),
        textcoords="axes fraction",
        fontsize=8.5,
        color="#d62728",
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.1),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d62728", alpha=0.92),
        zorder=7,
    )

    handles, labels = ax.get_legend_handles_labels()
    handles.append(
        Line2D([0], [0], marker="v", color="none", markerfacecolor="#666666",
               markeredgecolor="white", markersize=8)
    )
    labels.append("首次跌破 EOL 阈值")
    ax.legend(handles, labels, fontsize=9, loc="lower left")

    ax.set_xlabel("循环次数")
    ax.set_ylabel("SOH = 容量 / 额定容量")
    ax.set_title("(a) 相对额定容量的健康度", fontsize=11)
    ys.append(np.array([1.0, EOL_THRESHOLD_AH / RATED_CAPACITY_AH]))
    _set_limits(ax, xs, ys, pad_frac=0.05, top_frac=0.10)

    # --- (b) 相对首循环：去掉初始容量差异 ---
    ax = axes[1]
    xs = []
    ys = []
    for cell, sub in by.items():
        ax.plot(sub["cycle"], sub["cap_over_first"], color=COLORS[cell], lw=1.6, label=cell)
        xs.append(sub["cycle"].to_numpy())
        ys.append(sub["cap_over_first"].to_numpy())

        eol = prof.loc[cell, "eol_cycle"]
        if not _is_missing(eol):
            # 用档案里的 cap_first（首循环，过去信息），不用 iloc[0] 猜行序
            ratio = EOL_THRESHOLD_AH / float(prof.loc[cell, "cap_first"])
            ax.scatter([int(eol)], [ratio], s=52, marker="v", color=COLORS[cell],
                       edgecolors="white", linewidths=0.9, zorder=6)

    ax.axhline(1.0, ls=":", lw=1.2, color="#999999")
    label_right(ax, "起点统一归一到 1.0", 1.0)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(
        Line2D([0], [0], marker="v", color="none", markerfacecolor="#666666",
               markeredgecolor="white", markersize=8)
    )
    labels.append("首次跌破 EOL 阈值")
    ax.legend(handles, labels, fontsize=9, loc="lower left")

    ax.set_xlabel("循环次数")
    ax.set_ylabel("容量 / 该电池首循环容量")
    ax.set_title("(b) 去掉初始容量差异后的衰减形状", fontsize=11)
    ys.append(np.array([1.0]))
    _set_limits(ax, xs, ys, pad_frac=0.05)

    fig.suptitle(
        "个体差异：绝对健康度 vs 相对衰减形状（(b) 仅用于展示，不得作特征）",
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
        print(f"已生成 {p.name}  ({p.stat().st_size / 1024:.0f} KB)")

    print(f"\nEOL 交叉点（阈值 {EOL_THRESHOLD_AH:g} Ah）：")
    for cell in prof.index:
        eol = prof.loc[cell, "eol_cycle"]
        shown = "未到达（右删失）" if _is_missing(eol) else f"第 {int(eol)} 循环"
        print(f"  {cell}: {shown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
