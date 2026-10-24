"""W5：个体差异五维量化。

为什么要「五维」
----------------
W4 的图只能让人「看得出四个电池不一样」。但「不一样」不是结论，**差在哪一维、
差多少、这一维影响后面的哪一步**才是。所以这里把差异拆成五个维度，每个维度给一个数，
并明确它影响后续哪个环节：

| 维度 | 影响的后续环节 |
|---|---|
| ① 水平（初始容量） | SOH 的基准、是否超过额定 |
| ② 速率（平均衰减） | 基线模型的斜率、寿命量级 |
| ③ 形状（线性度 / 加速比） | **W10–W13 选线性模型还是双指数模型** |
| ④ 波动（残差 / 再生） | **W15 不确定性分解里的「数据不确定」层** |
| ⑤ 寿命（EOL） | 右删失处理（D2）与 W19 更换策略 |

产出
----
outputs/tables/individual_differences.md
outputs/figures/fig03_individual_differences_facets.png

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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    COLORS,
    EOL_THRESHOLD_AH,
    RATED_CAPACITY_AH,
    TABLE_DIR,
    ensure_dirs,
    is_missing,
    setup_matplotlib,
)
from figures import save  # noqa: E402
from load_data import load_raw, split_by_battery  # noqa: E402
from preprocess import cell_profile, mark_rows  # noqa: E402

TABLE_MD = TABLE_DIR / "individual_differences.md"
RESID_WINDOW = 11
"""残差尺度的滚动窗口。取 11 是为了让趋势估计不受单点回升影响，又能跟上局部曲率。"""


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


def _accel_ratio(sub: pd.DataFrame) -> float:
    """加速比 = 后半段斜率 / 前半段斜率。

    两条斜率都是负的。比值 > 1 表示后段衰减**更快**（加速老化）；
    ≈ 1 表示近似匀速；< 1 表示后段反而变缓。
    """
    half = len(sub) // 2
    s_early = _linear(sub.iloc[:half])[0]
    s_late = _linear(sub.iloc[half:])[0]
    if s_early == 0:
        return float("nan")
    return s_late / s_early


def _quad_coef(sub: pd.DataFrame) -> float:
    """二次项系数，自变量按 100 循环缩放，单位 Ah/(100 循环)²。负值代表下弯加速。"""
    x = sub["cycle"].to_numpy(float) / 100.0
    y = sub["capacity_Ah"].to_numpy(float)
    return float(np.polyfit(x, y, 2)[0])


def _resid_sigma(sub: pd.DataFrame, window: int = RESID_WINDOW) -> float:
    """相对局部趋势（滚动中位数）的**高频波动**尺度，用 MAD 稳健估计。

    注意口径：它衡量的是逐循环抖动，**不是**趋势拟合误差。
    线性 R² 低不代表这个值大——两者衡量不同的东西。
    """
    series = pd.Series(sub["capacity_Ah"].to_numpy(float))
    trend = series.rolling(window, center=True, min_periods=3).median()
    resid = (series - trend).to_numpy()
    resid = resid[~np.isnan(resid)]
    if resid.size == 0:
        return float("nan")
    mad = float(np.median(np.abs(resid - np.median(resid))))
    return 1.4826 * mad


def build_table(df: pd.DataFrame, prof: pd.DataFrame) -> pd.DataFrame:
    prof_idx = prof.set_index("battery_id")
    rows = []
    for cell, sub in split_by_battery(df).items():
        sub = sub.sort_values("cycle").reset_index(drop=True)
        if int(sub["cycle"].iloc[0]) != 1:
            raise ValueError(f"{cell} 首行不是 cycle==1，无法定义初始容量")

        slope, r2 = _linear(sub)
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
                "③形状 二次项(Ah/(100循环)²)": round(_quad_coef(sub), 3),
                "③形状 加速比": round(_accel_ratio(sub), 3),
                "④波动 相对趋势σ(Ah)": round(_resid_sigma(sub), 4),
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
    acc = tbl["③形状 加速比"]
    sig = tbl["④波动 相对趋势σ(Ah)"]
    regen = tbl["④波动 再生次数"]
    over_rated = tbl.loc[soh > 1.0, "电池"].tolist()

    eol = tbl["⑤寿命 EOL循环"]
    censored = tbl.loc[tbl["⑤寿命 删失"], "电池"].tolist()
    finite_eol = [int(v) for v in eol if not is_missing(v)]

    ratio_rate = float(slope.abs().max() / slope.abs().min())
    fastest = tbl.loc[slope.abs().idxmax(), "电池"]
    slowest = tbl.loc[slope.abs().idxmin(), "电池"]

    if acc.max() < 1.0:
        shape_verdict = "，**全部 < 1 —— 后段衰减反而变缓（凸形），不是加速老化**。"
    elif acc.min() > 1.0:
        shape_verdict = "，全部 > 1 —— 后段衰减更快，是加速老化。"
    else:
        shape_verdict = "，既有 > 1 也有 < 1 —— 个体之间的形状并不一致。"

    sig_top2 = set(tbl.nlargest(2, "④波动 相对趋势σ(Ah)")["电池"])
    regen_top2 = set(tbl.nlargest(2, "④波动 再生次数")["电池"])
    if sig_top2 == regen_top2:
        sig_linked = (
            f"σ 最大的两个个体（{sorted(sig_top2)}）恰好也是再生次数最多的两个，"
            "说明这个尺度**主要被回升事件驱动，不是纯白噪声**——"
            "这正是 D18 要把它拆成「共模 + 个体」的原因。"
        )
    else:
        sig_linked = "σ 的排序与再生次数排序不一致，说明波动还有别的来源。"

    return [
        "**① 水平**：初始容量 "
        f"{cap.min():.4f}–{cap.max():.4f} Ah，极差 {cap.max() / cap.min() - 1:.1%}；"
        f"其中 {over_rated} 的初始 SOH > 100%（{soh.max():.1%}）。"
        "说明「额定 2 Ah」不是这批电池的实际上限，SOH > 100% 是数据性质而非错误。",

        "**② 速率**：平均衰减 "
        f"{slope.min():.4f}–{slope.max():.4f} Ah/(100 循环)，"
        f"最快（{fastest}）是最慢（{slowest}）的 **{ratio_rate:.2f} 倍**。"
        "这一维决定寿命量级，也是基线模型斜率的第一来源。",

        "**③ 形状**：线性 R² 在 "
        f"{r2.min():.3f}–{r2.max():.3f} 之间；加速比（后半段斜率/前半段斜率）"
        f"在 {acc.min():.2f}–{acc.max():.2f} 之间"
        + shape_verdict
        + "二次项系数全部为正，指向同一种形状（下弯变缓）。"
        "这一维决定建模选择，但**偏差方向留到 W13 由模型对比实测**，不在这里预设——"
        "本项目的规则是结论先由数字给出，不由直觉给出。",

        "**④ 波动**：相对局部趋势的稳健波动尺度 σ = "
        f"{sig.min():.4f}–{sig.max():.4f} Ah；再生事件 "
        f"{regen.min()}–{regen.max()} 次/个体。"
        + sig_linked,

        "**⑤ 寿命**：已观测到的 EOL 在 "
        f"{min(finite_eol)}–{max(finite_eol)} 循环之间，共 {len(finite_eol)} 个个体；"
        f"{censored} 为**右删失**（观察期内未到达 {EOL_THRESHOLD_AH:g} Ah）。"
        "统计寿命时必须把删失个体单独处理，不能记成观测长度（D2）。",
    ]


# ---------------------------------------------------------------------------
# 小多图
# ---------------------------------------------------------------------------
def fig03(tbl: pd.DataFrame) -> Path:
    setup_matplotlib()
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.0))
    axes = axes.ravel()

    cells = tbl["电池"].tolist()
    colors = [COLORS[c] for c in cells]
    xpos = np.arange(len(cells))

    def bars(ax, values, title, ylabel, fmt, refs=(), top_pad=0.12):
        vals = np.asarray(values, dtype=float)
        ax.bar(xpos, vals, color=colors, width=0.62)
        ax.set_xticks(xpos)
        ax.set_xticklabels(cells, fontsize=9)
        ax.set_title(title, fontsize=10.5)
        ax.set_ylabel(ylabel, fontsize=9)
        limits = list(vals) + [float(y) for y, _ in refs]
        lo = float(np.nanmin(limits))
        hi = float(np.nanmax(limits))
        span = (hi - lo) or abs(hi) or 1.0
        for xi, v in zip(xpos, vals):
            ax.text(
                xi,
                v,
                fmt.format(v),
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=8.2,
            )
        for y, lab in refs:
            ax.axhline(y, ls="--", lw=1.1, color="#777777", zorder=0)
            # 标签必须放在**轴内**：放到轴外会与相邻子图打架，还会把整图 bbox 撑大
            ax.text(
                0.02,
                y,
                lab,
                transform=ax.get_yaxis_transform(),
                fontsize=8,
                color="#666666",
                va="bottom",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8),
                zorder=6,
            )
        ax.set_ylim(lo - 0.15 * span, hi + top_pad * span)

    bars(
        axes[0],
        tbl["①水平 初始容量(Ah)"],
        "① 水平：初始容量",
        "Ah",
        "{:.4f}",
        refs=[(RATED_CAPACITY_AH, f"额定 {RATED_CAPACITY_AH:g} Ah")],
    )

    bars(
        axes[1],
        tbl["②速率 斜率(Ah/100循环)"],
        "② 速率：平均衰减斜率",
        "Ah / 100 循环",
        "{:.4f}",
        refs=[(0.0, "0")],
    )

    bars(
        axes[2],
        tbl["③形状 线性R²"],
        "③ 形状：线性拟合 R²",
        "R²",
        "{:.3f}",
        refs=[(1.0, "完全线性 = 1.0")],
    )

    bars(
        axes[3],
        tbl["③形状 加速比"],
        "③ 形状：加速比（后半段/前半段斜率）",
        "倍",
        "{:.2f}",
        refs=[(1.0, "1.0 = 匀速衰减")],
    )

    bars(
        axes[4],
        tbl["④波动 相对趋势σ(Ah)"],
        "④ 波动：相对局部趋势的稳健尺度 σ",
        "Ah",
        "{:.4f}",
    )
    axes[4].set_xticks(xpos)
    axes[4].set_xticklabels(
        [f"{c}\n再生 {n} 次" for c, n in zip(cells, tbl["④波动 再生次数"])],
        fontsize=9,
    )

    # ⑤ 寿命：已观测失效画实心柱；删失画白底斜线柱（表示「至少这么长」）
    ax = axes[5]
    for cell, xi, v, cens, n in zip(
        cells, xpos, tbl["⑤寿命 EOL循环"], tbl["⑤寿命 删失"], tbl["观测循环"]
    ):
        if cens:
            # 白底 + 彩色边框斜线：如果先画实心柱再叠同色斜线，斜线会完全看不见
            ax.bar(
                [xi],
                [float(n)],
                color="white",
                edgecolor=COLORS[cell],
                hatch="//",
                width=0.62,
                lw=1.3,
            )
            ax.text(
                xi,
                float(n),
                f">{n}（删失）",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=COLORS[cell],
            )
        else:
            ax.bar([xi], [float(v)], color=COLORS[cell], width=0.62)
            ax.text(xi, float(v), f"{int(v)}", ha="center", va="bottom", fontsize=8.5)

    ax.axhline(0, lw=0.8, color="#333333")
    ax.set_xticks(xpos)
    ax.set_xticklabels(cells, fontsize=9)
    ax.set_title(f"⑤ 寿命：到达 {EOL_THRESHOLD_AH:g} Ah 的循环", fontsize=10.5)
    ax.set_ylabel("循环")
    ax.set_ylim(0, float(tbl["观测循环"].max()) * 1.22)

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
        "## 三、这五维分别影响后面哪一步",
        "",
        "| 维度 | 影响的后续环节 |",
        "|---|---|",
        "| ① 水平 | SOH 的基准；`cap_over_*` 不得作特征（D14） |",
        "| ② 速率 | 基线模型斜率；寿命量级 |",
        "| ③ 形状 | W10–W13 选线性还是双指数：加速比 > 1 说明直线会系统性高估后段剩余寿命 |",
        "| ④ 波动 | W15 三层不确定性里的「数据不确定」层；也是 D18「共模 + 个体」拆分的对象 |",
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
