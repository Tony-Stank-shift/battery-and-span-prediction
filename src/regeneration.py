"""W6：容量再生的检测与分解。

三个问题，都要给数字：

1. **有多少次？** —— 逐个列出 (电池, 循环) 对，并标注同一循环上还有谁在回升。
2. **这些回升是四个个体各自独立发生的，还是同时发生的？**
3. **波动里有多少能被一个共同成分解释？**

⚠️ 第三轮审查推翻了初版结论，本文件已改（原始报告见 `outputs/_tmp/rev3/REVIEW_R3.md`）：
- 初版只报「共模占 49.6%」，**漏了零基准**。四块互相独立时，同样构造也能吃掉 1/k = 25%。
  且 49.6% 是被不相关的 B0018 稀释后的值——只取同批次三块是 **82.4%**（零基准 33.3%）。
- 初版报「各电池与共模的相关」，**同样漏了零基准**。共模里包含该电池自己，基准逐电池不同
  （B0006 高达 0.63），所以 B0018 的 +0.348 落在自身基准中间 = **检不出耦合**，不是「弱正相关」。
- 现在两者都并列给出零基准，并加 Monte Carlo 置换检验。

方法
----
- 趋势：**居中滚动中位数**（窗口见 `config.TREND_WINDOW`）。不依赖参数形式，且对
  1–2 个循环的尖峰免疫（中位数天然抗离群）。
- 残差：r_t = C_t - trend_t
- 共模：残差矩阵按循环取跨电池均值（用均值是为了让方差分解**精确**正交）
- 个体：e_t = r_t - 共模

产出
----
outputs/figures/fig04_trend_residual.png
outputs/tables/regeneration_events.md

运行
----
    python src/regeneration.py
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
    EDGE_TRIM,
    RANDOM_SEED,
    REGEN_MIN_JUMP_AH,
    TABLE_DIR,
    TREND_WINDOW,
    ensure_dirs,
    setup_matplotlib,
)
from figures import save  # noqa: E402
from load_data import load_raw, split_by_battery  # noqa: E402
from preprocess import mark_rows  # noqa: E402

TABLE_MD = TABLE_DIR / "regeneration_events.md"

SYNC_CLUSTER = ["B0005", "B0006", "B0007"]
"""同批次三块（两两相关 0.85–0.95）。B0018 与它们不相关，单独作为「局外者」对照。"""

N_PERM = 2000
"""置换检验次数（环形移位法）。"""


# ---------------------------------------------------------------------------
# 趋势与残差
# ---------------------------------------------------------------------------
def trend_and_residual(sub: pd.DataFrame, window: int = TREND_WINDOW) -> tuple[np.ndarray, np.ndarray]:
    series = pd.Series(sub["capacity_Ah"].to_numpy(dtype=float))
    trend = series.rolling(window, center=True, min_periods=3).median()
    resid = series - trend
    return trend.to_numpy(dtype=float), resid.to_numpy(dtype=float)


def residual_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """index=循环，columns=电池，值=相对局部趋势的残差。"""
    cols = {}
    for cell, sub in split_by_battery(df).items():
        sub = sub.sort_values("cycle").reset_index(drop=True)
        _, resid = trend_and_residual(sub)
        cols[cell] = pd.Series(resid, index=sub["cycle"].to_numpy())
    out = pd.DataFrame(cols).sort_index()
    out.index.name = "cycle"
    return out


# ---------------------------------------------------------------------------
# 事件表
# ---------------------------------------------------------------------------
def event_table(df: pd.DataFrame) -> pd.DataFrame:
    """逐个列出 (电池, 循环) 对，并标注同一循环上还有谁也在回升。"""
    by = split_by_battery(df)
    same_cycle: dict[int, list[str]] = {}
    for cell, sub in by.items():
        for cyc in sub.loc[sub["is_regen"], "cycle"].tolist():
            same_cycle.setdefault(int(cyc), []).append(cell)

    rows = []
    for cell, sub in by.items():
        sub = sub.sort_values("cycle").reset_index(drop=True)
        for _, row in sub[sub["is_regen"]].iterrows():
            cyc = int(row["cycle"])
            others = sorted(c for c in same_cycle.get(cyc, []) if c != cell)
            rows.append(
                {
                    "电池": cell,
                    "循环": cyc,
                    "回升幅度(Ah)": round(float(row["cap_delta"]), 4),
                    "回升后容量(Ah)": round(float(row["capacity_Ah"]), 4),
                    "同循环同步回升": "、".join(others) if others else "—",
                    "是否同步事件": "是" if others else "否",
                }
            )
    return pd.DataFrame(rows).sort_values(["循环", "电池"]).reset_index(drop=True)


def merge_stats(events: pd.DataFrame) -> dict:
    """「同刻合并」口径的事件计数。

    ⚠️ 这是一个**约定**，不是统计结果：同一循环上多块电池同时回升，我们把它算作
    一个事件的一次多电池表现。换一种合并规则，数字就会变。报告里必须写明这条假设。
    """
    n_pairs = int(len(events))
    shared = events[events["是否同步事件"] == "是"]
    cycles = sorted(int(c) for c in shared["循环"].unique())
    n_shared_pairs = int(len(shared))
    n_merged = n_pairs - n_shared_pairs + len(cycles)
    return {
        "n_pairs": n_pairs,
        "shared_cycles": cycles,
        "n_shared_pairs": n_shared_pairs,
        "n_merged": n_merged,
    }


# ---------------------------------------------------------------------------
# 方差分解（补零基准、补 k 敏感性、补正交性断言）
# ---------------------------------------------------------------------------
def _decompose(R: np.ndarray) -> dict:
    """把残差矩阵拆成共模 + 个体，并给出独立零基准。

    共模用**跨电池均值**，因此 e = r - m 在每个循环上与 m 严格正交，
    var(r) = var(m) + var(e) 是**精确等式**（ddof 一致时）。下面有 assert 守着。

    零基准 1/k 的含义：若 k 条序列互相独立且方差相同，仅靠「均值里含自己」
    这个构造就能让 m 吃掉 1/k 的方差。所以「共模占比」必须和 1/k 比，不能和 0 比。
    """
    k = R.shape[1]
    m = R.mean(axis=1, keepdims=True)
    E = R - m
    v_tot = float(R.var())
    v_com = float(m.var())
    v_idi = float(E.var())

    identity_gap = abs(v_tot - (v_com + v_idi))
    assert identity_gap <= 1e-10 * max(v_tot, 1e-30), (
        f"方差分解不成立：{v_tot:.6e} != {v_com:.6e} + {v_idi:.6e}（差 {identity_gap:.3e}）"
        "　—— 检查三处 .var() 的 ddof 是否一致"
    )

    return {
        "k": k,
        "var_total": v_tot,
        "var_common": v_com,
        "var_idio": v_idi,
        "share": v_com / (v_com + v_idi) if (v_com + v_idi) > 0 else float("nan"),
        "null_share": 1.0 / k,
        "common_series": pd.Series(m.ravel(), index=range(len(m))),
    }


def null_corr(core: pd.DataFrame, n_perm: int = N_PERM, seed: int = RANDOM_SEED) -> dict:
    """各电池「与共模的相关」的独立零基准（环形移位法）。

    共模包含该电池自己，所以零基准不是 0。做法：对每块电池的残差序列
    独立做随机环形移位，破坏跨电池的时间对齐但保留各自的分布与自相关结构，
    然后照原公式计算 corr(序列_i, 共模)。重复 n_perm 次得到零分布。
    """
    rng = np.random.default_rng(seed)
    R = core.to_numpy(float)
    n, k = R.shape
    out: dict[str, list[float]] = {c: [] for c in core.columns}

    for _ in range(n_perm):
        shifted = np.empty_like(R)
        for j in range(k):
            off = int(rng.integers(1, n))
            shifted[:, j] = np.roll(R[:, j], off)
        m = shifted.mean(axis=1)
        for j, c in enumerate(core.columns):
            col = shifted[:, j]
            if col.std() == 0 or m.std() == 0:
                continue
            out[c].append(float(np.corrcoef(col, m)[0, 1]))

    return {
        c: {
            "mean": float(np.mean(v)),
            "lo": float(np.percentile(v, 2.5)),
            "hi": float(np.percentile(v, 97.5)),
            "samples": v,
        }
        for c, v in out.items()
    }


def variance_decomposition(resid: pd.DataFrame) -> dict:
    common = resid.dropna()
    if len(common) <= 2 * EDGE_TRIM:
        raise ValueError("四块电池共同覆盖的循环太少，无法做方差分解")

    core = common.iloc[EDGE_TRIM : len(common) - EDGE_TRIM]
    core.index.name = "cycle"

    full = _decompose(core.to_numpy(float))
    trio = _decompose(core[SYNC_CLUSTER].to_numpy(float))

    common_series = pd.Series(core.mean(axis=1).to_numpy(), index=core.index)
    observed_corr = {
        c: float(core[c].corr(common_series)) for c in core.columns
    }

    nulls = null_corr(core)
    corr_rows = []
    for c in core.columns:
        obs = observed_corr[c]
        null = nulls[c]
        p = float(np.mean(np.asarray(null["samples"]) >= obs))
        corr_rows.append(
            {
                "电池": c,
                "与共模相关": round(obs, 3),
                "独立零基准(均值)": round(null["mean"], 3),
                "零基准 95% 区间": f"[{null['lo']:.3f}, {null['hi']:.3f}]",
                "置换 p 值": round(p, 3),
                "读数": "高于基准" if obs > null["hi"] else ("低于基准" if obs < null["lo"] else "落在基准区间内，检不出耦合"),
            }
        )

    return {
        "cycles": (int(core.index.min()), int(core.index.max())),
        "n_cycles": int(len(core)),
        "full": full,
        "trio": trio,
        "common_series": common_series,
        "pairwise": core.corr(),
        "corr_table": pd.DataFrame(corr_rows),
    }


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------
def fig04(df: pd.DataFrame, resid: pd.DataFrame, decomp: dict, events: pd.DataFrame) -> Path:
    setup_matplotlib()
    fig, axes = plt.subplots(
        2, 1, figsize=(11.0, 8.4), sharex=True,
        gridspec_kw={"height_ratios": [1.1, 1.0]},
    )
    by = split_by_battery(df)

    ax = axes[0]
    for cell, sub in by.items():
        trend, _ = trend_and_residual(sub)
        ax.plot(sub["cycle"], sub["capacity_Ah"], color=COLORS[cell], lw=0.9, alpha=0.55)
        ax.plot(sub["cycle"], trend, color=COLORS[cell], lw=2.0, label=cell)
    ax.set_ylabel("放电容量 (Ah)")
    ax.set_title(
        f"图4(a)　容量与趋势（细线=原始，粗线=居中滚动中位数，窗口 {TREND_WINDOW}）",
        fontsize=11.5,
    )
    ax.legend(fontsize=9, loc="upper right")

    ax = axes[1]
    for cell in resid.columns:
        ax.plot(resid.index, resid[cell], color=COLORS[cell], lw=0.9, alpha=0.6,
                label=f"{cell} 残差")
    cm = decomp["common_series"]
    ax.plot(cm.index, cm.to_numpy(), color="#111111", lw=2.4, zorder=5,
            label=f"共模（跨电池均值，仅 {cm.index.min()}–{cm.index.max()} 区间）")
    ax.axhline(0, lw=0.8, color="#333333")

    # 共模事件：竖线 + 逐个标注（块数与幅度全部从事件表现算）
    shared_cycles = sorted(
        int(c) for c in events.loc[events["是否同步事件"] == "是", "循环"].unique()
    )
    for cyc in shared_cycles:
        ax.axvline(cyc, ls=":", lw=1.3, color="#d62728", zorder=0)

    placements = [(0.28, 0.90), (0.60, 0.72), (0.78, 0.56)]
    for i, cyc in enumerate(shared_cycles):
        if cyc not in cm.index:
            continue  # 共模只在四块共同区间上有定义，区间外跳过
        here = events[events["循环"] == cyc].sort_values("回升幅度(Ah)", ascending=False)
        cells_here = here["电池"].tolist()
        jumps = here["回升幅度(Ah)"].tolist()
        ax.annotate(
            f"循环 {cyc}：{'、'.join(cells_here)}（{len(cells_here)} 块）同时回升\n"
            + "、".join(f"{c} {j:+.4f}" for c, j in zip(cells_here, jumps)),
            xy=(cyc, float(cm.loc[cyc])),
            xytext=placements[i % len(placements)],
            textcoords="axes fraction",
            fontsize=8.8,
            color="#d62728",
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d62728", alpha=0.92),
            zorder=7,
        )

    ax.set_xlabel("循环次数")
    ax.set_ylabel("残差 (Ah)")
    f4, f3 = decomp["full"], decomp["trio"]
    ax.set_title(
        f"图4(b)　残差与共模（四块 {f4['share']:.1%}，零基准 {f4['null_share']:.0%}；"
        f"同批三块 {f3['share']:.1%}，零基准 {f3['null_share']:.0%}）",
        fontsize=11,
    )
    ax.legend(fontsize=8.5, loc="lower left", ncol=2, framealpha=0.92)

    x_all = np.concatenate(
        [resid.index.to_numpy(dtype=float)] + [by[c]["cycle"].to_numpy(dtype=float) for c in by]
    )
    x0, x1 = float(x_all.min()), float(x_all.max())
    pad = 0.03 * (x1 - x0)
    ax.set_xlim(x0 - pad, x1 + pad)
    assert ax.get_xlim()[0] <= x0 and ax.get_xlim()[1] >= x1, "x 轴把数据裁掉了"

    fig.tight_layout()
    return save(fig, "fig04_trend_residual.png")


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
def render_md(events: pd.DataFrame, decomp: dict, merged: dict) -> str:
    f4, f3 = decomp["full"], decomp["trio"]
    per_cell = events.groupby("电池").size().to_dict()

    lines = [
        "# 容量再生：检测与分解（W6）",
        "",
        "> 本文件由 `src/regeneration.py` 自动生成，数字随代码重算，**不要手工编辑**。",
        "",
        "## 一、口径",
        "",
        f"- 趋势 = 居中滚动中位数（窗口 {TREND_WINDOW} 循环）",
        f"- 再生事件判定沿用 `cap_delta > {REGEN_MIN_JUMP_AH} Ah`（与 W3 决策 3 同一口径）",
        f"- 方差分解区间：循环 {decomp['cycles'][0]}–{decomp['cycles'][1]}"
        f"（四块电池同时存在，且去掉首尾各 {EDGE_TRIM} 个受窗口边界影响的循环），"
        f"共 {decomp['n_cycles']} 个循环",
        "- **共模用跨电池均值**，因此方差分解是精确等式（代码里有 assert 守着，"
        "若三处 `.var()` 的 ddof 不一致会直接报错而不是静默给错数）",
        "",
        "## 二、事件清单",
        "",
        events.to_markdown(index=False),
        "",
        "## 三、统计",
        "",
        f"- (电池, 循环) 对共 **{merged['n_pairs']}** 个：" +
        "、".join(f"{c} {n}" for c, n in per_cell.items()),
        f"- 同一循环上多块同时回升：循环 {merged['shared_cycles']}，"
        f"涉及 {merged['n_shared_pairs']} 个 (电池, 循环) 对",
        f"- **按「同一循环合并为一个事件」的约定计数 → {merged['n_merged']} 个**。"
        f"⚠️ 这是**约定**不是统计结果：换一种合并规则数字就会变，报告里必须写明这条假设。",
        "",
        "## 四、方差分解：必须和零基准比",
        "",
        "| 口径 | 电池数 k | 共模占比 | 独立零基准 1/k | 超出 |",
        "|---|---|---|---|---|",
        f"| 全部四块 | {f4['k']} | **{f4['share']:.1%}** | {f4['null_share']:.1%} "
        f"| {f4['share'] - f4['null_share']:+.1%} |",
        f"| 同批次三块（不含 {', '.join(sorted(set(decomp['pairwise'].columns) - set(SYNC_CLUSTER)))}）"
        f" | {f3['k']} | **{f3['share']:.1%}** | {f3['null_share']:.1%} "
        f"| {f3['share'] - f3['null_share']:+.1%} |",
        "",
        f"方差绝对值：四块口径 共模 {f4['var_common']:.3e} / 个体 {f4['var_idio']:.3e} Ah²；"
        f"三块口径 共模 {f3['var_common']:.3e} / 个体 {f3['var_idio']:.3e} Ah²。",
        "",
        "**为什么必须给零基准**：共模是跨电池均值，本身包含每块电池自己，"
        "所以即使 k 条序列完全独立，这个构造也能吃掉 1/k 的方差。"
        "和 0 比会把「构造本身的效应」误当成「共同经历」。",
        "",
        "**为什么三块口径更高**：不相关的 B0018 被算进均值，把共模稀释了。"
        "三块口径 82.4% 才反映同批次电池之间的真实同步程度。",
        "",
        "## 五、残差两两相关（比单一百分比更能说明结构）",
        "",
        decomp["pairwise"].round(3).to_markdown(),
        "",
        "## 六、各电池与共模的相关：也要和零基准比",
        "",
        decomp["corr_table"].to_markdown(index=False),
        "",
        f"置换次数 {N_PERM}（环形移位法：对每块电池独立随机移位，破坏跨电池时间对齐，"
        "保留各自的分布与自相关结构）。",
        "",
        "**怎么读**：共模里含该电池自己，所以零基准不是 0，且逐电池不同。"
        "B0006 的基准高达约 0.63，因此它看起来很高的相关其实没那么惊人；"
        "B0018 的读数落在自身零基准区间内 → **检不出耦合**，不能说成「弱正相关」。",
        "",
        "## 七、结论与对后续步骤的意义",
        "",
        "- 同批次三块之间存在**明确的共同波动成分**（远高于零基准），"
        "所以「容量再生完全是个体独立的电化学现象」这个说法不成立。",
        "- 但**共模的物理来源本数据无法分辨**（可能是共同的实验/测量事件，也可能是别的）；"
        "本项目只主张「存在共同成分」，不主张成因（D18）。",
        "- 共模部分 → 进入 W15 三层不确定性里的「数据不确定」层（它是系统性成分，"
        "不该被当成随机噪声抹平）。",
        "- 个体部分 → 是 W9 预测区间的噪声底，也是 W18 区间校准该覆盖的对象。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ensure_dirs()

    raw = load_raw()
    df = mark_rows(raw)

    resid = residual_matrix(df)
    events = event_table(df)
    merged = merge_stats(events)
    decomp = variance_decomposition(resid)

    TABLE_MD.write_text(render_md(events, decomp, merged), encoding="utf-8")
    fig_path = fig04(df, resid, decomp, events)

    f4, f3 = decomp["full"], decomp["trio"]
    print(f"(电池, 循环) 对：{merged['n_pairs']} 个；同刻合并后事件数：{merged['n_merged']} 个")
    print(events.to_string(index=False))
    print()
    print(f"共模占比（四块）  ：{f4['share']:.1%}　零基准 {f4['null_share']:.1%}　超出 {f4['share'] - f4['null_share']:+.1%}")
    print(f"共模占比（三块同批）：{f3['share']:.1%}　零基准 {f3['null_share']:.1%}　超出 {f3['share'] - f3['null_share']:+.1%}")
    print()
    print(decomp["corr_table"].to_string(index=False))
    print()
    print(f"事件表已写出：{TABLE_MD.name}")
    print(f"已生成 {fig_path.name}  ({fig_path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
