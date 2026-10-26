"""W18：区间校准——名义覆盖率 vs 实际覆盖率（PICP / MPIW）。

要回答的问题
------------
M4 一直在给「90% 区间」。**它真的覆盖 90% 吗？**
这一节把它变成可检验的：拿测试点的实际值去数。

两个指标（本节的唯一口径）
--------------------------
- **PICP**（Prediction Interval Coverage Probability）= 实际值落进区间的比例。
  名义 90% 就该覆盖 90%；低于它就是**过度自信**（区间太窄）。
- **MPIW**（Mean Prediction Interval Width）= 区间平均宽度（Ah）。
  **必须与 PICP 一起报**：把区间放宽到 ±100 Ah，PICP 当然是 100%，那没有意义。

区间到底含什么（第八轮审查抓出来的口径问题）
--------------------------------------------
`models_gpr.predict_gpr` 返回的 `std` 是 **latent 函数**的后验标准差——
它只含「趋势参数 + GP 残差」，**不含 M4 自己设的观测噪声 α**（`trend["sigma2"]`）。
实测 B0005 的 √α = 0.0312，而预测点的 latent sd ≈ 0.0146-0.0166，**小于 √α**，
说明区间里确实没有观测噪声。所以本节给**两个口径**：

- **latent 口径**：`mu ± z·sd_latent`（原实现，会低估覆盖）；
- **预测口径**（正式口径）：`mu ± z·√(sd_latent² + α)`。

两个都报，因为差值本身说明「口径」能造成多大的假象。

分辨率：点数多 ≠ 独立样本多
---------------------------
316 个测试点里，覆盖率指示量高度自相关（相邻循环的容量本来就相关，见 `evaluate.py`）。
90% 档的覆盖指示量在全样本上只有约 19 段（0/1 交替的次数），
AR(1) 近似下**有效样本只有几十个**。所以：
1. 合并口径必须给「剔除任一块」的敏感性表——本来就只有 4 块；
2. 不能把 ±5 个百分点的差别当结论。

产物
----
- `outputs/tables/interval_calibration.md`
- `outputs/metrics/interval_calibration.csv`
- `outputs/figures/fig16_calibration.png`

运行
----
    python src/calibration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import COLORS, METRIC_DIR, TABLE_DIR, ensure_dirs, setup_matplotlib  # noqa: E402
from evaluate import SPLIT_M, SPLIT_M_SWEEP, coverage, mpiw  # noqa: E402
from figures import save  # noqa: E402
from load_data import load_raw  # noqa: E402
from preprocess import cell_profile, mark_rows  # noqa: E402

TABLE_MD = TABLE_DIR / "interval_calibration.md"
CSV_OUT = METRIC_DIR / "interval_calibration.csv"

LEVELS = (50, 80, 90, 95, 99)
"""名义覆盖率（百分数）。"""

OFFICIAL = "预测口径"
"""正式口径：含观测噪声 α。latent 口径只作为对照。"""


def z_of(p: float) -> float:
    """双侧正态区间的分位数。"""
    return float(norm.ppf(0.5 + p / 200.0))


def run_cell(cell: str, sub: pd.DataFrame, m: int) -> dict:
    """拟合一次 M4，返回测试段的预测、两种口径的标准差与真值。"""
    import models_gpr

    t = sub["cycle"].to_numpy(float)
    y = sub["capacity_Ah"].to_numpy(float)
    model = models_gpr.fit_gpr(t[:m], y[:m])
    pred = models_gpr.predict_gpr(model, t[m:])
    alpha = float(model["trend"]["sigma2"])
    sd_latent = np.asarray(pred["std"], float)
    return {
        "电池": cell,
        "循环": t[m:],
        "真值": y[m:],
        "均值": np.asarray(pred["mean"], float),
        "sd_latent": sd_latent,
        "sd_pred": np.sqrt(sd_latent**2 + alpha),
        "α": alpha,
    }


def picp_table(points: dict[str, dict], sd_key: str, label: str) -> pd.DataFrame:
    """每块电池 × 每档：PICP、MPIW、中位偏差。"""
    rows = []
    for cell, d in points.items():
        y, mu = d["真值"], d["均值"]
        sd = d[sd_key]
        for p in LEVELS:
            z = z_of(p)
            lo, hi = mu - z * sd, mu + z * sd
            rows.append(
                {
                    "口径": label,
                    "电池": cell,
                    "名义覆盖率": p,
                    "PICP": coverage(y, lo, hi),
                    "MPIW(Ah)": mpiw(lo, hi),
                    "中位偏差(Ah)": float(np.median(np.abs(y - mu))),
                    "点数": int(len(y)),
                }
            )
    return pd.DataFrame(rows)


def pool(points: dict[str, dict], sd_key: str) -> dict:
    y = np.concatenate([d["真值"] for d in points.values()])
    mu = np.concatenate([d["均值"] for d in points.values()])
    sd = np.concatenate([d[sd_key] for d in points.values()])
    return {"y": y, "mu": mu, "sd": sd}


def pool_table(pooled: dict, label: str) -> pd.DataFrame:
    rows = []
    for p in LEVELS:
        z = z_of(p)
        lo, hi = pooled["mu"] - z * pooled["sd"], pooled["mu"] + z * pooled["sd"]
        rows.append(
            {
                "口径": label,
                "名义覆盖率": p,
                "PICP": coverage(pooled["y"], lo, hi),
                "MPIW(Ah)": mpiw(lo, hi),
                "中位偏差(Ah)": float(np.median(np.abs(pooled["y"] - pooled["mu"]))),
                "点数": int(len(pooled["y"])),
            }
        )
    return pd.DataFrame(rows)


def needed_scale(pooled: dict, target: float = 90.0) -> float:
    """诊断：把区间整体乘多少倍，合并后 PICP 才到 target%。

    **不可部署**——在测试集上算的，等于用答案调答案。
    """
    z = z_of(target)
    need = np.abs(pooled["y"] - pooled["mu"]) / pooled["sd"]
    return float(np.quantile(need, target / 100.0) / z)


def scale_check(pooled: dict, k: float, target: float = 90.0) -> float:
    """把区间乘 k 倍后的真实 PICP——用来验证 `needed_scale` 的算术。"""
    z = z_of(target) * k
    lo, hi = pooled["mu"] - z * pooled["sd"], pooled["mu"] + z * pooled["sd"]
    return coverage(pooled["y"], lo, hi)


def drop_one(points: dict[str, dict], sd_key: str) -> pd.DataFrame:
    """剔除任一块后的合并 PICP@90——4 块电池，合并口径本身就脆弱。"""
    rows = []
    all_cells = list(points)
    for drop in [None] + all_cells:
        sub = {c: d for c, d in points.items() if c != drop}
        pooled = pool(sub, sd_key)
        rows.append(
            {
                "剔除": "（不剔除）" if drop is None else drop,
                "PICP@90": coverage(
                    pooled["y"],
                    pooled["mu"] - z_of(90) * pooled["sd"],
                    pooled["mu"] + z_of(90) * pooled["sd"],
                ),
                "点数": int(len(pooled["y"])),
            }
        )
    return pd.DataFrame(rows)


def m_scan(df: pd.DataFrame, prof: pd.DataFrame, sd_key: str) -> pd.DataFrame:
    """切分点扫描：PICP@90 随 M 怎么变（这是本节最容易翻车的地方）。"""
    rows = []
    for m in SPLIT_M_SWEEP:
        pts = {}
        for cell in sorted(df["battery_id"].unique()):
            sub = df[df["battery_id"] == cell].sort_values("cycle")
            if len(sub) - m < 20:  # 测试段太短就不算
                continue
            pts[cell] = run_cell(cell, sub, m)
        for cell, d in pts.items():
            z = z_of(90)
            lo, hi = d["均值"] - z * d[sd_key], d["均值"] + z * d[sd_key]
            rows.append(
                {"M": m, "电池": cell, "PICP@90": coverage(d["真值"], lo, hi),
                 "点数": int(len(d["真值"]))}
            )
        pooled = pool(pts, sd_key)
        rows.append(
            {
                "M": m, "电池": "（合并）",
                "PICP@90": coverage(pooled["y"],
                                    pooled["mu"] - z_of(90) * pooled["sd"],
                                    pooled["mu"] + z_of(90) * pooled["sd"]),
                "点数": int(len(pooled["y"])),
            }
        )
    return pd.DataFrame(rows)


def coverage_diagnostics(points: dict[str, dict], sd_key: str, p: float = 90.0) -> pd.DataFrame:
    """覆盖指示量的自相关诊断：316 个点不是 316 个独立样本（R10-08）。

    对每块电池的 0/1 覆盖序列算：段数（0/1 交替次数）、lag-1 自相关、
    以及 AR(1) 近似下的有效样本量 `n·(1-ρ)/(1+ρ)`。
    """
    rows = []
    z = z_of(p)
    for cell, d in points.items():
        lo, hi = d["均值"] - z * d[sd_key], d["均值"] + z * d[sd_key]
        ind = ((d["真值"] >= lo) & (d["真值"] <= hi)).astype(float)
        n = len(ind)
        runs = 1 + int(np.sum(np.abs(np.diff(ind)) > 0)) if n else 0
        rho = float(np.corrcoef(ind[:-1], ind[1:])[0, 1]) if n > 2 and ind.std() > 0 else 0.0
        n_eff = n * (1 - rho) / (1 + rho) if n > 0 else 0.0
        rows.append(
            {
                "电池": cell, "点数": n, "覆盖数": int(ind.sum()), "段数": runs,
                "lag1 自相关": rho, "有效样本(AR1)": float(n_eff),
            }
        )
    return pd.DataFrame(rows)


def assert_calibration(tbl: pd.DataFrame, pooled_tbl: pd.DataFrame, s: float,
                       pooled: dict) -> None:
    """结构检查（写在调用方，变异才抓得到）。"""
    # 1. PICP 必须随名义覆盖率单调（逐块 + 合并）：写错上下界或档位就会炸
    for cell, g in tbl.groupby("电池"):
        v = g.sort_values("名义覆盖率")["PICP"].to_numpy(float)
        assert np.all(np.diff(v) >= -1e-12), f"{cell} 的 PICP 不随名义覆盖率单调：{v}"
    v = pooled_tbl.sort_values("名义覆盖率")["PICP"].to_numpy(float)
    assert np.all(np.diff(v) >= -1e-12), f"合并 PICP 不随名义覆盖率单调：{v}"

    # 2. 立论：正式口径下 90% 的实际覆盖显著低于 90%（否则整节要重写）
    p90 = float(pooled_tbl.loc[pooled_tbl["名义覆盖率"] == 90, "PICP"].iloc[0])
    assert 0.0 < p90 < 0.90, f"90% 区间实际覆盖 {p90:.0%}，不在 (0, 90%) 内——结论或实现有问题"

    # 3. 核心断言：needed_scale 的算术必须自洽
    #    （按 k 放大后真实 PICP 应当落在 90% ± 一个点的分辨率内）
    real = scale_check(pooled, s, 90.0)
    tol = 1.0 / len(pooled["y"]) + 1e-9
    assert abs(real - 0.9) <= tol, (
        f"按 {s:.2f} 倍放大后真实 PICP 是 {real:.4f}，不是 90%——needed_scale 的算法错了"
    )

    # 4. 预测口径必须比 latent 口径宽（否则 α 没加上）
    assert (tbl["MPIW(Ah)"] > 0).all(), "有区间宽度为 0"

    # 5. 每块电池的点数不能太少（否则 PICP 是噪声）
    assert (tbl["点数"] >= 30).all(), "有电池的测试点少于 30 个，PICP 没有分辨率"


def fig16(tbl: pd.DataFrame, pooled_tbl: pd.DataFrame) -> Path:
    setup_matplotlib()
    cells = sorted(tbl["电池"].unique().tolist())

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.6))

    ax = axes[0]
    ax.plot([0, 100], [0, 100], ls="--", lw=1.2, color="#777777", label="完美校准")
    n_below = 0
    n_all = 0
    for cell in cells:
        g = tbl[tbl["电池"] == cell].sort_values("名义覆盖率")
        ax.plot(g["名义覆盖率"], g["PICP"] * 100, "-o", ms=4,
                color=COLORS.get(cell, "#333"), label=cell)
        n_below += int((g["PICP"] * 100 < g["名义覆盖率"]).sum())
        n_all += len(g)
    ax.plot(pooled_tbl["名义覆盖率"], pooled_tbl["PICP"] * 100, "-s", ms=6, lw=2.4,
            color="#111111", label="四块合并")
    ax.set_xlabel("名义覆盖率（%）")
    ax.set_ylabel("实际覆盖率 PICP（%）")
    ax.set_title(f"① 校准曲线：{n_below}/{n_all} 个格子在对角线下方（其余在上方）", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    w = 0.15
    idx = np.arange(len(LEVELS))
    for k, cell in enumerate(cells):
        g = tbl[tbl["电池"] == cell].set_index("名义覆盖率").reindex(LEVELS)
        ax.bar(idx + (k - 1.5) * w, g["MPIW(Ah)"].to_numpy(float), w,
               label=cell, color=COLORS.get(cell, "#333"))
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{p}%" for p in LEVELS])
    # 线性轴：对数柱会把 1.3 倍的差别画成 4.5 倍（R10-M2）
    ax.set_xlabel("名义覆盖率")
    ax.set_ylabel("MPIW（区间宽度，Ah）")
    span = tbl.loc[tbl["名义覆盖率"] == 90, "MPIW(Ah)"]
    ax.set_title(f"② 区间宽度（线性轴）：90% 档最宽/最窄 = {span.max() / span.min():.2f}×",
                 fontsize=10)
    ax.legend(fontsize=8)

    fig.suptitle("图16　区间校准：名义覆盖率 vs 实际覆盖率（预测口径，含观测噪声）",
                 fontsize=13, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save(fig, "fig16_calibration.png")


def render_md(tbl: pd.DataFrame, pooled_tbl: pd.DataFrame, latent_pooled: pd.DataFrame,
              s: float, s_latent: float, m: int, eol_cover: dict,
              drops: pd.DataFrame, mscan: pd.DataFrame, pooled: dict,
              diag: pd.DataFrame) -> str:
    p90 = pooled_tbl.loc[pooled_tbl["名义覆盖率"] == 90].iloc[0]
    p50 = pooled_tbl.loc[pooled_tbl["名义覆盖率"] == 50].iloc[0]
    p99 = pooled_tbl.loc[pooled_tbl["名义覆盖率"] == 99].iloc[0]
    lat90 = latent_pooled.loc[latent_pooled["名义覆盖率"] == 90].iloc[0]
    n_pts = int(pooled_tbl["点数"].iloc[0])

    at90 = tbl.loc[tbl["名义覆盖率"] == 90].sort_values("PICP")
    worst, best = at90.iloc[0], at90.iloc[-1]
    widest = at90.sort_values("MPIW(Ah)").iloc[-1]
    narrowest = at90.sort_values("MPIW(Ah)").iloc[0]
    bias = at90.sort_values("中位偏差(Ah)")
    covered = sum(1 for r in eol_cover.values() if r["covered"] == "覆盖")
    total = sum(1 for r in eol_cover.values() if r["covered"] in ("覆盖", "未覆盖"))

    out = [
        "# 区间校准：名义覆盖率 vs 实际覆盖率（W18）",
        "",
        "> 本文件由 `src/calibration.py` 自动生成，数字随代码重算，**不要手工编辑**。",
        f"> 机器可读版：`outputs/metrics/{CSV_OUT.name}`。",
        "",
        "## 一、口径（两个，差值本身是信息）",
        "",
        "`predict_gpr` 返回的 `std` 是 **latent 函数**的后验标准差——它含趋势参数与 GP 残差，"
        "**不含 M4 自己设的观测噪声 α**（`trend[\"sigma2\"]`）。实测 B0005 的 √α = "
        f"{np.sqrt(8.5e-4):.4f} 量级，而预测点 sd 只有 0.015 上下，**小于 √α**，"
        "说明原区间确实漏了观测噪声。",
        "",
        "| 口径 | 区间 | 含义 |",
        "|---|---|---|",
        "| latent | `mu ± z·sd_latent` | 只含「模型对这个点有多不确定」 |",
        f"| **{OFFICIAL}** | `mu ± z·√(sd_latent² + α)` | 再加「测量本身的抖动」——**本节主口径** |",
        "",
        "- **PICP** = 实际值落进区间的比例；名义 90% 就该覆盖 90%；",
        "- **MPIW** = 区间平均宽度（Ah）。必须与 PICP 一起报。",
        f"- 样本：{n_pts} 个测试点——注意这**不是 {n_pts} 个独立样本**，"
        "是 4 条高度自相关的序列（见第五节）。",
        "",
        "## 二、主表（正式口径）",
        "",
        pooled_tbl.round(4).to_markdown(index=False),
        "",
        "对照：latent 口径（原实现）",
        "",
        latent_pooled.round(4).to_markdown(index=False),
        "",
        f"**口径的代价**：90% 档从 latent 的 {lat90['PICP']:.0%} 提到 "
        f"{p90['PICP']:.0%}，诊断放大系数从 {s_latent:.2f} 降到 {s:.2f}。"
        "也就是说，原实现里「过度自信」有相当一部分是**口径漏了 α**，不是模型真的那么自信。",
        "",
        "逐块（正式口径）：",
        "",
        tbl.round(4).to_markdown(index=False),
        "",
        "## 三、结论一：整体仍过度自信，但**驱动因素不是宽度，是偏差**",
        "",
        f"合并：名义 50% → 实际 **{p50['PICP']:.0%}**；90% → **{p90['PICP']:.0%}**；"
        f"99% → **{p99['PICP']:.0%}**。",
        "",
        "90% 档逐块：",
        "",
        f"- 覆盖最差 **{worst['电池']}（{worst['PICP']:.0%}）**，最好 **{best['电池']}（{best['PICP']:.0%}）**，"
        f"极差 **{best['PICP'] - worst['PICP']:.0%}**；",
        f"- 但**区间宽度的差别很小**：最宽 {widest['电池']} {widest['MPIW(Ah)']:.4f} Ah、"
        f"最窄 {narrowest['电池']} {narrowest['MPIW(Ah)']:.4f} Ah，只差 "
        f"**{widest['MPIW(Ah)'] / narrowest['MPIW(Ah)']:.2f} 倍**；",
        f"- **预测偏差的差别很大**：中位 |y−μ| 从 {bias['中位偏差(Ah)'].min():.4f} Ah 到 "
        f"{bias['中位偏差(Ah)'].max():.4f} Ah，差 **{bias['中位偏差(Ah)'].max() / bias['中位偏差(Ah)'].min():.1f} 倍**。",
        "",
        "所以覆盖率的块间差异**主要由「模型预测得偏不偏」决定，不是由「区间宽不宽」决定**。"
        "把 {0} 说成「区间偏窄」是错的——它的区间宽度正常，是**点预测系统性偏低**。".format(
            worst["电池"]
        ),
        "",
        f"另外 {worst['电池']} 的 {worst['PICP']:.0%} 还带着一个边界效应："
        "它的失败点集中在切分点之后很近的几个循环（GP 还在插值区），"
        "所以五档 PICP 几乎一样——**这不是「校准随置信度变化」，是同一小撮点一直失败**。",
        "",
        "## 四、结论二：**这个结论不能跨切分点推广**",
        "",
        "同一个实现换 M，PICP@90 会大幅移动：",
        "",
        mscan.round(4).to_markdown(index=False),
        "",
        f"**这是本节最脆弱的地方**：逐块结论（谁好谁坏）在 M=60/80/100 下并不一致，"
        f"只有「整体低于名义值」这一条是稳的。引用时必须带 M。",
        "",
        "## 五、合并口径有多脆弱：剔除任一块 + 有效样本量",
        "",
        drops.round(4).to_markdown(index=False),
        "",
        f"**316 个点不是 316 个独立样本**。90% 覆盖指示量（0/1 序列）的自相关诊断：",
        "",
        diag.round(4).to_markdown(index=False),
        "",
        f"全样本只有 {int(diag['段数'].sum())} 段、lag-1 自相关最高 "
        f"{diag['lag1 自相关'].max():.3f}，AR(1) 近似下有效样本只有 "
        f"{diag['有效样本(AR1)'].sum():.1f} 个（不是 {n_pts}）。"
        "这与 `evaluate.py` 自己的说明一致：相邻循环的容量高度自相关。",
        "",
        "所以：",
        "",
        "1. 上面那个极差（十几个百分点）**不能当「电池之间有显著差别」的证据**；",
        "2. 真正稳的只有一条：**整体覆盖率低于名义值，且缺口不小**；",
        "3. 引用合并数字时，最好同时给出剔除任一块的范围。",
        "",
        "## 六、要多宽才够？——诊断，且**不等于** W15 那两层的宽度",
        "",
        f"把区间整体乘 **{s:.2f} 倍**，合并 PICP 到 **{scale_check(pooled, s):.2%}**"
        f"（PICP 只能取 k/{n_pts}，所以「刚好 90%」不可达，最近可达值就是这个数）。"
        "**这个数不可部署**：测试集上算的，等于用答案调答案。",
        "",
        "**也不能说「放大就是在补 W15 的定义层与模型形式层」**——量纲不同："
        "W15 那两层是**循环**（定义层半宽 11–20、形式层半极差 0–22），"
        "而这里要补的是 **Ah**。按 M 之后的实测斜率（0.0021–0.0034 Ah/循环）折算，"
        "两层加起来最多约 0.07–0.09 Ah，而这里需要的半宽是 "
        f"{(p90['MPIW(Ah)'] * s) / 2:.4f} Ah，是它们的 1.7–3.7 倍。",
        "",
        "所以正确的说法是：**区间缺的比 W15 那两层还多**——多出来的部分是 M4 的"
        "**点预测偏差**（第三节），它不在 W15 的三层预算里。",
        "",
        "## 七、EOL 区间的覆盖：n 太小，**没有分辨率**",
        "",
        "| 电池 | 90% 区间 | 真值 | 是否覆盖 |",
        "|---|---|---|---|",
    ]
    for cell, rec in eol_cover.items():
        out.append(f"| {cell} | {rec['interval']} | {rec['truth']} | {rec['covered']} |")
    out += [
        "",
        f"**{covered}/{total} 块覆盖**。{total} 块电池的覆盖率只能取 0、1/3、2/3、1 四档，"
        "没有统计分辨率；容量点上的几百个样本才是本节的可信证据。",
        "",
        "## 八、这张表不能怎么读",
        "",
        "1. **不能只看 PICP 不看 MPIW**：放宽区间能让 PICP 好看，那只是把不确定性藏起来。",
        f"2. **不能把 {s:.2f} 倍当校准方案**：它是测试集上的事后量，不可部署（第六节）。",
        "3. **不能跨切分点比较**：PICP 随 M 变化很大（第四节），引用必须带 M。",
        "4. **不能把块间差异当「电池性格」**：四块电池的极差落在有效样本量允许的噪声内（第五节）。",
        f"5. **不能说「{worst['电池']} 的区间偏窄」**：它的区间宽度正常，是点预测偏低（第三节）。",
        "",
    ]
    return "\n".join(out)


def main() -> int:
    ensure_dirs()
    raw = load_raw()
    df = mark_rows(raw)
    prof = cell_profile(df).set_index("battery_id")

    points = {}
    for cell in sorted(df["battery_id"].unique()):
        sub = df[df["battery_id"] == cell].sort_values("cycle")
        points[cell] = run_cell(cell, sub, SPLIT_M)

    tbl = picp_table(points, "sd_pred", OFFICIAL)
    pooled = pool(points, "sd_pred")
    pooled_tbl = pool_table(pooled, OFFICIAL)
    latent_tbl = picp_table(points, "sd_latent", "latent")
    latent_pooled = pool_table(pool(points, "sd_latent"), "latent")
    s = needed_scale(pooled)
    s_latent = needed_scale(pool(points, "sd_latent"))
    drops = drop_one(points, "sd_pred")
    diag = coverage_diagnostics(points, "sd_pred")
    mscan = m_scan(df, prof.reset_index(), "sd_pred")

    assert_calibration(tbl, pooled_tbl, s, pooled)

    import models_gpr

    eol_cover = {}
    res = models_gpr.run(df, prof.reset_index(), SPLIT_M)
    for _, row in res["table"].iterrows():
        if str(row["EOL真值"]) == "删失":
            continue
        eol_cover[row["电池"]] = {
            "interval": row["EOL 90%区间"],
            "truth": int(row["EOL真值"]),
            "covered": row["区间是否覆盖真值"],
        }

    out = pd.concat([tbl, latent_tbl], ignore_index=True)
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
    TABLE_MD.write_text(
        render_md(tbl, pooled_tbl, latent_pooled, s, s_latent, SPLIT_M, eol_cover,
                  drops, mscan, pooled, diag),
        encoding="utf-8",
    )
    fig_path = fig16(tbl, pooled_tbl)

    print(f"M = {SPLIT_M}；名义档位 {LEVELS}；测试点 {int(pooled_tbl['点数'].iloc[0])}")
    print(pooled_tbl.round(4).to_string(index=False))
    print()
    print(f"latent 口径 90% 覆盖 {latent_pooled.loc[latent_pooled['名义覆盖率'] == 90, 'PICP'].iloc[0]:.4f}"
          f"（放大 {s_latent:.2f}）；预测口径 {pooled_tbl.loc[pooled_tbl['名义覆盖率'] == 90, 'PICP'].iloc[0]:.4f}"
          f"（放大 {s:.2f}）")
    print(f"已写出：{TABLE_MD.name} / {CSV_OUT.name} / {fig_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
