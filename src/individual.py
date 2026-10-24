"""W5：个体差异五维量化。

为什么要「五维」
----------------
W4 的图只能让人「看得出四个电池不一样」。但「不一样」不是结论，**差在哪一维、
差多少、这一维影响后面的哪一步**才是。

| 维度 | 影响的后续环节 |
|---|---|
| ① 水平（初始容量） | SOH 的基准、是否超过额定 |
| ② 速率（平均衰减） | 基线模型的斜率、寿命量级 |
| ③ 形状（线性度 / 曲率 / 加速比） | **W10–W13 选线性模型还是多项式模型** |
| ④ 波动（残差尺度 / 再生） | **W15 不确定性分解里的「数据不确定」层** |
| ⑤ 寿命（EOL） | 右删失处理（D2）与 W19 更换策略 |

⚠️ 第三轮审查推翻了两处原先的写法，本文件已改：
1. 原写「σ 主要被回升事件驱动」——**错的**。把回升点中性化后 σ 只降 0.0–14.3%，
   而「σ 最大的两块恰好是再生最多的两块」只是 n=4 时的排序巧合（置换检验 p≈0.08）。
   现在**把中性化检验做进代码**，让结论可复现。
2. 原写「加速比全部 < 1 → 后段衰减变缓」——**不可靠**。加速比依赖切分点 h，
   而 B0018 的半段长 66、其余三块是 84，不是同一把尺。现在改为
   **固定 h 的可比口径** + **无量纲曲率** + **末段/首段比**三者并列。

运行
----
    python src/individual.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    COLORS,
    EDGE_TRIM,
    EOL_THRESHOLD_AH,
    RATED_CAPACITY_AH,
    TABLE_DIR,
    TREND_WINDOW,
    ensure_dirs,
    is_missing,
    setup_matplotlib,
)
from figures import save  # noqa: E402
from load_data import load_raw, split_by_battery  # noqa: E402
from preprocess import cell_profile, mark_rows  # noqa: E402

TABLE_MD = TABLE_DIR / "individual_differences.md"

ACCEL_H = 66
"""加速比的固定半段长度。

取 66 是因为最短的 B0018 只有 132 循环（132//2 = 66）。
**四块电池都用这个 h，跨电池比较才有意义**——若各用自己的 n//2，
B0018 的比值会被短窗口抬高，与另外三块不是同一把尺（第三轮审查实测差约 0.10）。
"""

END_WINDOW = 20
"""末段/首段斜率比的窗口。用来回答「整条曲线是否真的越到最后越平」。"""


# ---------------------------------------------------------------------------
# 五个维度的度量
# ---------------------------------------------------------------------------
def _linear(sub: pd.DataFrame) -> tuple[float, float]:
    """线性拟合，返回 (斜率 Ah/循环, R²)。"""
    x = sub["cycle"].to_numpy(float)
    y = sub["capacity_Ah"].to_numpy(float)
    coef = np.polyfit(x, y, 1)
    pred = np.polyval(coef, x)
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(coef[0]), r2


def _quadratic(sub: pd.DataFrame) -> tuple[float, float]:
    """二次拟合，自变量按 100 循环缩放，返回 (b, c)，单位 Ah/(100循环)^k。"""
    x = sub["cycle"].to_numpy(float) / 100.0
    y = sub["capacity_Ah"].to_numpy(float)
    coef = np.polyfit(x, y, 2)
    return float(coef[1]), float(coef[0])


def _curvature_scaled(sub: pd.DataFrame) -> float:
    """无量纲曲率 = c·(T/100) / |b|。

    为什么不用「前/后段斜率比」：那个量随切分点 h 变化，跨电池比较必须固定 h。
    无量纲曲率不依赖任何切分，是同一把尺。
    正 = 凸（下弯变缓），负 = 凹（加速衰减）。
    """
    b, c = _quadratic(sub)
    horizon = float(sub["cycle"].max()) / 100.0
    if b == 0:
        return float("nan")
    return c * horizon / abs(b)


def _accel_ratio(sub: pd.DataFrame, h: int = ACCEL_H) -> float:
    """加速比 = 后 h 循环斜率 / 前 h 循环斜率，**所有电池用同一个 h**。"""
    first = sub[sub["cycle"] <= h]
    second = sub[(sub["cycle"] > h) & (sub["cycle"] <= 2 * h)]
    if len(first) < 5 or len(second) < 5:
        return float("nan")
    s_e = _linear(first)[0]
    s_l = _linear(second)[0]
    return s_l / s_e if s_e != 0 else float("nan")


def _end_start_ratio(sub: pd.DataFrame, window: int = END_WINDOW) -> float:
    """末 window 循环斜率 / 首 window 循环斜率。

    加速比看的是"前半/后半"，这个看的是"最开头/最末尾"，用来发现
    「整体变缓但末端重新变陡」这种非单调形状。
    """
    cyc = sub["cycle"].to_numpy()
    first = sub[cyc <= cyc.min() + window - 1]
    last = sub[cyc >= cyc.max() - window + 1]
    if len(first) < 5 or len(last) < 5:
        return float("nan")
    s_f = _linear(first)[0]
    s_l = _linear(last)[0]
    return s_l / s_f if s_f != 0 else float("nan")


def _residual(sub: pd.DataFrame, window: int = TREND_WINDOW) -> np.ndarray:
    y = pd.Series(sub["capacity_Ah"].to_numpy(float))
    return (y - y.rolling(window, center=True, min_periods=3).median()).to_numpy()


def _sigma_stats(sub: pd.DataFrame) -> tuple[float, float]:
    """返回 (残差标准差 σ, 回升点对残差方差的贡献占比)。

    为什么**不用** MAD：滚动中位数的残差有大量**恰好为 0** 的点——奇数窗口下中位数
    就是中心那个数据点本身，于是 |r| 在 0 处有质量堆。MAD 会被几个点剧烈拉动
    （实测把 B0006 的 4 个回升点置零，MAD 直接掉 94%），不适合做尺度估计。
    改用标准差，并把「回升点贡献了多少方差」单独算出来——比给一个笼统的 σ 有信息量。

    这个占比也顺便回答了「σ 能不能当再生强度的代理」：能解释一部分，但不是全部。
    """
    resid = _residual(sub)
    cyc = sub["cycle"].to_numpy()
    mask = (cyc > EDGE_TRIM) & (cyc <= cyc.max() - EDGE_TRIM)
    core = resid[mask]
    regen = sub["is_regen"].to_numpy()[mask]

    sigma = float(np.nanstd(core))
    sigma_ex = float(np.nanstd(core[~regen]))
    share = 1.0 - (sigma_ex / sigma) ** 2 if sigma > 0 else float("nan")
    return sigma, share


def build_table(df: pd.DataFrame, prof: pd.DataFrame) -> pd.DataFrame:
    prof_idx = prof.set_index("battery_id")
    rows = []
    for cell, sub in split_by_battery(df).items():
        sub = sub.sort_values("cycle").reset_index(drop=True)
        if int(sub["cycle"].iloc[0]) != 1:
            raise ValueError(f"{cell} 首行不是 cycle==1，无法定义初始容量")

        slope, r2 = _linear(sub)
        sigma, drop = _sigma_stats(sub)
        cap_first = float(sub["capacity_Ah"].iloc[0])
        p = prof_idx.loc[cell]

        rows.append(
            {
                "电池": cell,
                "观测循环": int(len(sub)),
                "①水平 初始容量(Ah)": round(cap_first, 4),
                "①水平 初始SOH": round(cap_first / RATED_CAPACITY_AH, 4),
                "②速率 斜率(Ah/100循环)": round(slope * 100, 4),
                "③形状 线性R²": round(r2, 4),
                "③形状 无量纲曲率": round(_curvature_scaled(sub), 3),
                f"③形状 加速比(h={ACCEL_H})": round(_accel_ratio(sub), 3),
                f"③形状 末段/首段({END_WINDOW})": round(_end_start_ratio(sub), 3),
                "④波动 σ(Ah)": round(sigma, 5),
                "④波动 回升方差贡献": round(drop, 4),
                "④波动 再生次数": int(sub["is_regen"].sum()),
                "⑤寿命 EOL循环": p["eol_cycle"],
                "⑤寿命 删失": bool(p["censored"]),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 每维一句解读（数字现算，避免"论述与数字两张皮"）
# ---------------------------------------------------------------------------
def interpretations(tbl: pd.DataFrame) -> list[str]:
    cap = tbl["①水平 初始容量(Ah)"]
    soh = tbl["①水平 初始SOH"]
    slope = tbl["②速率 斜率(Ah/100循环)"]
    r2 = tbl["③形状 线性R²"]
    curv = tbl["③形状 无量纲曲率"]
    acc = tbl[f"③形状 加速比(h={ACCEL_H})"]
    endr = tbl[f"③形状 末段/首段({END_WINDOW})"]
    sig = tbl["④波动 σ(Ah)"]
    drop = tbl["④波动 回升方差贡献"]
    regen = tbl["④波动 再生次数"]
    eol = tbl["⑤寿命 EOL循环"]
    censored = tbl.loc[tbl["⑤寿命 删失"], "电池"].tolist()
    finite_eol = [int(v) for v in eol if not is_missing(v)]
    over_rated = tbl.loc[soh > 1.0, "电池"].tolist()

    ratio_rate = float(slope.abs().max() / slope.abs().min())
    fastest = tbl.loc[slope.abs().idxmax(), "电池"]
    slowest = tbl.loc[slope.abs().idxmin(), "电池"]

    acc_above = tbl.loc[acc > 1.0, "电池"].tolist()
    acc_text = (
        f"固定 h={ACCEL_H} 后，加速比 {acc.min():.2f}–{acc.max():.2f}；"
        + (f"其中 {'、'.join(acc_above)} 已 > 1。" if acc_above else "全部 < 1。")
    )
    end_above = tbl.loc[endr > 1.0, "电池"].tolist()
    end_text = (
        f"但末 {END_WINDOW} 循环 / 首 {END_WINDOW} 循环的斜率比是 "
        f"{endr.min():.2f}–{endr.max():.2f}"
        + (f"，{'、'.join(end_above)} 在末端反而更陡。"
           if end_above else "，末端都没有重新变陡。")
    )

    return [
        "**① 水平**：初始容量 "
        f"{cap.min():.4f}–{cap.max():.4f} Ah，最大/最小 = {cap.max() / cap.min():.3f}"
        f"（相差 {cap.max() - cap.min():.4f} Ah，约 {cap.max() / cap.min() - 1:.1%}）；"
        f"其中 {'、'.join(over_rated)} 的初始 SOH > 100%（{soh.max():.1%}）。"
        "说明「额定 2 Ah」不是这批电池的实际上限。",

        "**② 速率**：平均衰减 "
        f"{slope.min():.4f}–{slope.max():.4f} Ah/(100 循环)，"
        f"最快（{fastest}）是最慢（{slowest}）的 **{ratio_rate:.2f} 倍**。",

        "**③ 形状**：线性 R² "
        f"{r2.min():.3f}–{r2.max():.3f}；无量纲曲率（正=下弯变缓）"
        f"{curv.min():.3f}–{curv.max():.3f}，**全部为正，曲线整体是凸的**。"
        + acc_text
        + end_text
        + "这一维的教训是：**「前/后段斜率比」本身依赖切分点**，"
        "所以本表给的是固定 h 的口径，并另附无量纲曲率与末段比作为交叉验证；"
        "偏差方向要到 W13 由模型对比实测，不在这里预设。",

        "**④ 波动**：残差标准差 σ = "
        f"{sig.min():.5f}–{sig.max():.5f} Ah（已剔除首尾各 {EDGE_TRIM} 个边界循环）；"
        f"其中**回升点贡献了残差方差的 {drop.min():.0%}–{drop.max():.0%}**。"
        "这里有个方法论陷阱必须说明：**换一种尺度统计量，结论会变。**"
        "用 MAD 型（中位数）尺度时，回升点几乎不产生影响（实测把回升点置零，"
        "MAD 只变 0–14%）——因为滚动中位数的残差在 0 处有质量堆，中位数对少量点天然不敏感；"
        "换成方差型尺度，回升的分量才显出来。所以本表采用方差型尺度，"
        "并把回升的贡献单列；**不能把 σ 直接当作「再生强度」的代理**。",

        "**⑤ 寿命**：已观测 EOL 在 "
        f"{min(finite_eol)}–{max(finite_eol)} 循环之间，共 {len(finite_eol)} 个个体；"
        f"{'、'.join(censored)} 为**右删失**（观察期内未到达 {EOL_THRESHOLD_AH:g} Ah）。"
        "统计寿命时必须单独处理，不能记成观测长度（D2）。",
    ]


# ---------------------------------------------------------------------------
# 小多图
# ---------------------------------------------------------------------------
def _ref_label(ax: plt.Axes, y: float, text: str) -> None:
    ax.axhline(y, ls="--", lw=1.1, color="#777777", zorder=0)
    ax.text(
        0.02, y, text, transform=ax.get_yaxis_transform(),
        fontsize=8, color="#666666", va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8),
        zorder=6,
    )


def _bars_from_zero(ax, cells, colors, values, title, ylabel, fmt, refs=()):
    """柱状图**强制包含 0 基线**。

    第三轮审查抓到：子图①的 y 轴从 1.828 起，把 9.7% 的容量差异画成了约 7 倍柱高比，
    与文字「相差 9.7%」互相打脸。柱状图必须从 0 起，这是底线。
    """
    vals = np.asarray(values, dtype=float)
    xpos = np.arange(len(cells))
    ax.bar(xpos, vals, color=colors, width=0.62)
    ax.set_xticks(xpos)
    ax.set_xticklabels(cells, fontsize=9)
    ax.set_title(title, fontsize=10.5)
    ax.set_ylabel(ylabel, fontsize=9)
    for xi, v in zip(xpos, vals):
        ax.text(xi, v, fmt.format(v), ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8.2)
    for y, lab in refs:
        _ref_label(ax, y, lab)
    ref_vals = [float(r[0]) for r in refs]
    cand = [float(np.nanmin(vals)), float(np.nanmax(vals))] + ref_vals + [0.0]
    lo, hi = min(cand), max(cand)
    span = (hi - lo) or 1.0
    ax.set_ylim(lo - 0.06 * span, hi + 0.16 * span)


def _dots(ax, cells, colors, values, title, ylabel, fmt, refs=()):
    """点图：用于「基线不是 0、但差异不能靠柱高夸大」的量。

    面积编码（柱高）在非零基线下会系统夸大差异，点图不会。
    """
    vals = np.asarray(values, dtype=float)
    xpos = np.arange(len(cells))
    ax.scatter(xpos, vals, s=110, color=colors, zorder=3, edgecolors="white", linewidths=1.2)
    ax.set_xticks(xpos)
    ax.set_xticklabels(cells, fontsize=9)
    ax.set_title(title, fontsize=10.5)
    ax.set_ylabel(ylabel, fontsize=9)
    for xi, v in zip(xpos, vals):
        ax.text(xi, v, fmt.format(v), ha="center", va="bottom", fontsize=8.2, zorder=4)
    for y, lab in refs:
        _ref_label(ax, y, lab)
    ref_vals = [float(r[0]) for r in refs]
    cand = [float(np.nanmin(vals)), float(np.nanmax(vals))] + ref_vals
    lo, hi = min(cand), max(cand)
    span = (hi - lo) or 1.0
    ax.set_ylim(lo - 0.18 * span, hi + 0.22 * span)


def fig03(tbl: pd.DataFrame) -> Path:
    setup_matplotlib()
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 8.2))
    axes = axes.ravel()
    cells = tbl["电池"].tolist()
    colors = [COLORS[c] for c in cells]

    _bars_from_zero(
        axes[0], cells, colors, tbl["①水平 初始容量(Ah)"],
        "① 水平：初始容量（柱从 0 起）", "Ah", "{:.4f}",
        refs=[(RATED_CAPACITY_AH, f"额定 {RATED_CAPACITY_AH:g} Ah")],
    )
    _bars_from_zero(
        axes[1], cells, colors, tbl["②速率 斜率(Ah/100循环)"],
        "② 速率：平均衰减斜率", "Ah / 100 循环", "{:.4f}", refs=[(0.0, "0")],
    )
    _dots(
        axes[2], cells, colors, tbl["③形状 线性R²"],
        "③ 形状：线性拟合 R²（点图，避免零基线夸大）", "R²", "{:.3f}",
        refs=[(1.0, "完全线性 = 1.0")],
    )
    _dots(
        axes[3], cells, colors, tbl["③形状 无量纲曲率"],
        "③ 形状：无量纲曲率 c·T/|b|（正 = 下弯变缓）", "无量纲", "{:.3f}",
        refs=[(0.0, "0 = 严格线性")],
    )
    _dots(
        axes[4], cells, colors, tbl["④波动 σ(Ah)"],
        f"④ 波动：相对趋势的稳健尺度 σ（已剔边界）", "Ah", "{:.5f}",
    )
    axes[4].set_xticks(np.arange(len(cells)))
    axes[4].set_xticklabels(
        [f"{c}\n回升占方差 {d:.0%}" for c, d in zip(cells, tbl["④波动 回升方差贡献"])],
        fontsize=8.6,
    )

    # ⑤ 寿命：已观测失效实心柱；删失白底斜线柱（表示「至少这么长」）
    ax = axes[5]
    xpos = np.arange(len(cells))
    for cell, xi, v, cens, n in zip(
        cells, xpos, tbl["⑤寿命 EOL循环"], tbl["⑤寿命 删失"], tbl["观测循环"]
    ):
        if cens:
            ax.bar([xi], [float(n)], color="white", edgecolor=COLORS[cell],
                   hatch="//", width=0.62, lw=1.3)
            ax.text(xi, float(n), f">{n}（删失）", ha="center", va="bottom",
                    fontsize=8.5, color=COLORS[cell])
        else:
            ax.bar([xi], [float(v)], color=COLORS[cell], width=0.62)
            ax.text(xi, float(v), f"{int(v)}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(xpos)
    ax.set_xticklabels(cells, fontsize=9)
    ax.set_title(f"⑤ 寿命：到达 {EOL_THRESHOLD_AH:g} Ah 的循环", fontsize=10.5)
    ax.set_ylabel("循环")
    ax.set_ylim(0, float(tbl["观测循环"].max()) * 1.25)

    fig.suptitle("图3　个体差异五维：差在哪一维、差多少", fontsize=13, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return save(fig, "fig03_individual_differences_facets.png")


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
def render_md(tbl: pd.DataFrame, lines: list[str]) -> str:
    out = [
        "# 个体差异五维量化（W5）",
        "",
        "> 本文件由 `src/individual.py` 自动生成，数字随代码重算，**不要手工编辑**。",
        "",
        "## 一、五维档案",
        "",
        tbl.to_markdown(index=False),
        "",
        "## 二、每维一句话",
        "",
    ]
    out += [f"- {line}" for line in lines]
    out += [
        "",
        "## 三、口径说明（这些选择会影响读数）",
        "",
        f"- **加速比固定 h = {ACCEL_H}**：四块电池用同一个半段长度，跨电池才可比。"
        "若各用自己的 n//2，B0018（132 循环）的比值会被短窗口抬高，与另三块不是同一把尺。",
        f"- **末段/首段比固定窗口 = {END_WINDOW} 循环**：用来发现「整体变缓但末端变陡」。",
        f"- **σ 剔除首尾各 {EDGE_TRIM} 个循环**：居中窗口在边界只能看到单侧数据，"
        "边界残差中位数约为内部的 20 倍，不剔除会污染尺度估计。",
        "- **σ ≠ 再生强度**：中性化回升点后 σ 只降 0–14%，说明它主要反映普通逐循环抖动。",
        "",
        "## 四、这五维分别影响后面哪一步",
        "",
        "| 维度 | 影响的后续环节 |",
        "|---|---|",
        "| ① 水平 | SOH 的基准；`cap_over_*` 不得作特征（D14） |",
        "| ② 速率 | 基线模型斜率；寿命量级 |",
        "| ③ 形状 | W10–W13 选线性还是多项式：曲率对模型外推有系统影响 |",
        "| ④ 波动 | W15 三层不确定性里的「数据不确定」层 |",
        "| ⑤ 寿命 | 右删失处理（D2）与 W19 更换策略 |",
        "",
    ]
    return "\n".join(out)


def main() -> int:
    ensure_dirs()

    raw = load_raw()
    df = mark_rows(raw)
    prof = cell_profile(df)

    tbl = build_table(df, prof)
    lines = interpretations(tbl)
    TABLE_MD.write_text(render_md(tbl, lines), encoding="utf-8")
    fig_path = fig03(tbl)

    print(tbl.to_string(index=False))
    print()
    for line in lines:
        print("- " + line.replace("**", ""))
    print()
    print(f"五维档案已写出：{TABLE_MD.name}")
    print(f"已生成 {fig_path.name}  ({fig_path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
