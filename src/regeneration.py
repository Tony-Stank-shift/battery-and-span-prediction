"""W6：容量再生的检测与分解。

这一步要回答三个问题，而且都要给数字：

1. **有多少次？** —— 不是"大约十来次"，而是逐个列出 (电池, 循环) 对。
2. **这些回升是四个个体各自独立发生的，还是同时发生的？**
   如果同号率远高于 0.5、且多块电池在同一循环同时达到最大回升，
   那么"容量再生=个体电化学现象"这个说法就站不住（D18 的依据）。
3. **波动里有多少能被一个共同成分解释？** —— 把残差矩阵分解成
   「共模 + 个体」，给一个方差占比。

方法
----
- 趋势：**居中滚动中位数**（窗口 11）。选它是因为不依赖任何参数形式，
  又对 1–2 个循环的尖峰免疫（中位数天然抗离群）。
- 残差：r_t = C_t - trend_t
- 共模：残差矩阵按循环取跨电池均值（用均值是为了让下面的方差分解精确正交）
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
    BATTERIES,
    COLORS,
    REGEN_MIN_JUMP_AH,
    TABLE_DIR,
    ensure_dirs,
    setup_matplotlib,
)
from figures import save  # noqa: E402
from load_data import load_raw, split_by_battery  # noqa: E402
from preprocess import mark_rows  # noqa: E402

TREND_WINDOW = 11
"""居中滚动中位数的窗口。11 个循环能吸收 1–2 个循环的尖峰，又不至于抹平局部曲率。"""

EDGE_TRIM = TREND_WINDOW // 2
"""首尾各剔除多少个循环：居中窗口在边界处只能看到单侧数据，残差不可靠。"""

TABLE_MD = TABLE_DIR / "regeneration_events.md"


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
            others = [c for c in same_cycle.get(cyc, []) if c != cell]
            rows.append(
                {
                    "电池": cell,
                    "循环": cyc,
                    "回升幅度(Ah)": round(float(row["cap_delta"]), 4),
                    "回升后容量(Ah)": round(float(row["capacity_Ah"]), 4),
                    "同循环同步回升": "、".join(sorted(others)) if others else "—",
                    "是否共模事件": "是" if others else "否",
                }
            )
    out = pd.DataFrame(rows).sort_values(["循环", "电池"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# 方差分解
# ---------------------------------------------------------------------------
def variance_decomposition(resid: pd.DataFrame) -> dict:
    """在四块电池同时存在的区间上，把残差方差拆成共模 + 个体。

    共模用**跨电池均值**：这样 e = r - m 与 m 在每个循环上严格正交，
    方差分解 var(r) = var(m) + var(e) 是精确等式，不是近似。
    """
    common = resid.dropna()
    if len(common) <= 2 * EDGE_TRIM:
        raise ValueError("四块电池共同覆盖的循环太少，无法做方差分解")

    # 去掉首尾受窗口边界影响的循环
    core = common.iloc[EDGE_TRIM : len(common) - EDGE_TRIM]
    R = core.to_numpy(dtype=float)
    m = R.mean(axis=1, keepdims=True)
    E = R - m

    var_total = float(R.var())
    var_common = float(m.var())
    var_idio = float(E.var())
    share = var_common / (var_common + var_idio) if (var_common + var_idio) > 0 else float("nan")

    m_series = pd.Series(m.ravel(), index=core.index)
    per_cell_corr = {c: float(core[c].corr(m_series)) for c in core.columns}

    return {
        "cycles": (int(core.index.min()), int(core.index.max())),
        "n_cycles": int(len(core)),
        "var_total": var_total,
        "var_common": var_common,
        "var_idio": var_idio,
        "share_common": share,
        "per_cell_corr": per_cell_corr,
        "common_series": m_series,
        "core_index": core.index,
    }


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------
def fig04(df: pd.DataFrame, resid: pd.DataFrame, decomp: dict, events: pd.DataFrame) -> Path:
    setup_matplotlib()
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.15, 1.0]})
    by = split_by_battery(df)

    # ---- 上：容量 + 趋势 ----
    ax = axes[0]
    for cell, sub in by.items():
        trend, _ = trend_and_residual(sub)
        ax.plot(sub["cycle"], sub["capacity_Ah"], color=COLORS[cell], lw=0.9, alpha=0.55)
        ax.plot(sub["cycle"], trend, color=COLORS[cell], lw=2.0, label=cell)
    ax.set_ylabel("放电容量 (Ah)")
    ax.set_title(
        f"图4(a)　容量曲线与趋势（细线=原始，粗线=居中滚动中位数，窗口 {TREND_WINDOW}）",
        fontsize=11.5,
    )
    ax.legend(fontsize=9, loc="upper right")

    # ---- 下：残差 + 共模 ----
    ax = axes[1]
    for cell in resid.columns:
        ax.plot(resid.index, resid[cell], color=COLORS[cell], lw=0.9, alpha=0.6, label=f"{cell} 残差")
    ax.plot(
        decomp["common_series"].index,
        decomp["common_series"].to_numpy(),
        color="#111111",
        lw=2.4,
        label="共模成分（跨电池均值）",
        zorder=5,
    )
    ax.axhline(0, lw=0.8, color="#333333")

    # 共模事件所在循环：竖线 + 逐个标注。
    # 块数与幅度**全部从事件表现算**——第一版这里写死了「三块电池同时最大回升」，
    # 而循环 48 上其实只有两块、且都不是各自峰值。这类"叙述超前于数据"的错必须由代码杜绝。
    shared_cycles = sorted(events.loc[events["是否共模事件"] == "是", "循环"].unique().tolist())
    for cyc in shared_cycles:
        ax.axvline(cyc, ls=":", lw=1.3, color="#d62728", zorder=0)

    cm = decomp["common_series"]
    placements = [(0.30, 0.88), (0.62, 0.70), (0.80, 0.55)]
    for i, cyc in enumerate(shared_cycles):
        here = events[events["循环"] == cyc].sort_values("回升幅度(Ah)", ascending=False)
        cells_here = here["电池"].tolist()
        jumps = here["回升幅度(Ah)"].tolist()
        text = (
            f"循环 {cyc}：{'、'.join(cells_here)}（{len(cells_here)} 块）同时回升\n"
            + "、".join(f"{c} {j:+.4f}" for c, j in zip(cells_here, jumps))
        )
        ax.annotate(
            text,
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
    ax.set_title(
        f"图4(b)　残差与共模成分（共模解释波动方差的 {decomp['share_common']:.1%}）",
        fontsize=11.5,
    )
    ax.legend(fontsize=8.5, loc="lower left", ncol=2, framealpha=0.92)

    # 统一 x 范围（按数据，不写死）
    x_all = np.concatenate([resid.index.to_numpy(dtype=float)] +
                           [by[c]["cycle"].to_numpy(dtype=float) for c in by])
    x0, x1 = float(x_all.min()), float(x_all.max())
    pad = 0.03 * (x1 - x0)
    ax.set_xlim(x0 - pad, x1 + pad)
    assert ax.get_xlim()[0] <= x0 and ax.get_xlim()[1] >= x1, "x 轴把数据裁掉了"

    fig.tight_layout()
    return save(fig, "fig04_trend_residual.png")


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
def render_md(events: pd.DataFrame, decomp: dict) -> str:
    n_pairs = len(events)
    per_cell = events.groupby("电池").size().to_dict()
    shared = events[events["是否共模事件"] == "是"]
    shared_cycles = sorted(shared["循环"].unique().tolist())
    n_shared_pairs = len(shared)
    n_clusters = len(shared_cycles)
    n_independent = n_pairs - n_shared_pairs + n_clusters

    corr_text = "；".join(f"{c} {v:+.3f}" for c, v in decomp["per_cell_corr"].items())

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
        "",
        "## 二、事件清单",
        "",
        events.to_markdown(index=False),
        "",
        "## 三、统计",
        "",
        f"- (电池, 循环) 对共 **{n_pairs}** 个：" +
        "、".join(f"{c} {n}" for c, n in per_cell.items()),
        f"- 其中同循环多块电池同时回升的「共模事件」**{n_shared_pairs}** 个，"
        f"集中在循环 {shared_cycles}（共 {n_clusters} 个时刻）",
        f"- 因此**独立事件数 ≈ {n_independent}**，不是 {n_pairs}。"
        f"把 {n_pairs} 说成「12 次独立事件」站不住。",
        "",
        "## 四、方差分解（共模 vs 个体）",
        "",
        f"- 残差总方差 {decomp['var_total']:.3e} Ah²",
        f"- 共模部分 {decomp['var_common']:.3e} Ah²，占 **{decomp['share_common']:.1%}**",
        f"- 个体部分 {decomp['var_idio']:.3e} Ah²，占 **{1 - decomp['share_common']:.1%}**",
        f"- 各电池残差与共模的相关系数：{corr_text}",
        "",
        "**怎么读**：共模占比越高，说明「再生」里越多的部分不是个体自己的电化学过程，"
        "而是几块电池共同经历的实验/测量事件。个体部分才是真正需要建模的随机波动。",
        "",
        "## 五、对后续步骤的意义",
        "",
        "- 共模部分 → 进入 W15 三层不确定性里的「数据不确定」层。"
        "它是系统性成分，不该被当成随机噪声抹平（W3 决策 3 已决定不平滑）。",
        "- 个体部分 → 是 W9 预测区间的真实噪声底，也是 W18 区间校准该覆盖的对象。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ensure_dirs()

    raw = load_raw()
    df = mark_rows(raw)

    resid = residual_matrix(df)
    events = event_table(df)
    decomp = variance_decomposition(resid)

    TABLE_MD.write_text(render_md(events, decomp), encoding="utf-8")
    fig_path = fig04(df, resid, decomp, events)

    print(f"(电池, 循环) 对：{len(events)} 个")
    print(events.to_string(index=False))
    print()
    print(f"共模占比：{decomp['share_common']:.1%}（循环 {decomp['cycles'][0]}–{decomp['cycles'][1]}，{decomp['n_cycles']} 个循环）")
    print("各电池残差与共模的相关系数：" +
          "、".join(f"{c} {v:+.3f}" for c, v in decomp["per_cell_corr"].items()))
    print()
    print(f"事件表已写出：{TABLE_MD.name}")
    print(f"已生成 {fig_path.name}  ({fig_path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
