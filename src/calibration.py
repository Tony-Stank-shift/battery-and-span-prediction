"""W18：区间校准——名义覆盖率 vs 实际覆盖率（PICP / MPIW）。

要回答的问题
------------
M4 一直在给「90% 区间」。**它真的覆盖 90% 吗？**
这一节把它变成可检验的：拿 316 个测试点的实际值去数。

两个指标（本节的唯一口径）
--------------------------
- **PICP**（Prediction Interval Coverage Probability）= 实际值落进区间的比例。
  名义 90% 就该覆盖 90%；低于它就是**过度自信**（区间太窄）。
- **MPIW**（Mean Prediction Interval Width）= 区间平均宽度（Ah）。
  **必须与 PICP 一起报**：把区间放宽到 ±100 Ah，PICP 当然是 100%，那没有意义。

两种覆盖，分辨率完全不同
------------------------
1. **逐点容量覆盖**：4 块电池合计 316 个测试点 → 有统计分辨率，是本节的主体。
2. **EOL 覆盖**：只有 3 块电池有真值 → 覆盖率只能是 0/3、1/3、2/3、3/3 四档，
   **没有分辨率**。放在最后一节，只作方向性说明。

刻意不做的事
------------
- **不把「事后放大因子」当校准方案**。在测试集上拟合一个让 PICP 恰好等于名义值的
  缩放系数，等于用答案调答案；它会给你一个好看的校准曲线和毫无意义的部署性能。
  这里只把它作为**诊断量**报出来（「要多宽才够」），并明确标注它不可部署。
- **不拿正态假设糊过去**：区间直接用 M4 的后验标准差（`predict_gpr` 的 `std`），
  它已经含残差噪声 + 趋势参数两层（W11 的方差分解）。

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

from config import METRIC_DIR, TABLE_DIR, ensure_dirs, setup_matplotlib  # noqa: E402
from evaluate import SPLIT_M, coverage, mpiw  # noqa: E402
from figures import save  # noqa: E402
from load_data import load_raw  # noqa: E402
from preprocess import cell_profile, mark_rows  # noqa: E402

TABLE_MD = TABLE_DIR / "interval_calibration.md"
CSV_OUT = METRIC_DIR / "interval_calibration.csv"

LEVELS = (50, 80, 90, 95, 99)
"""名义覆盖率（百分数）。"""


def z_of(p: float) -> float:
    """双侧正态区间的分位数。"""
    return float(norm.ppf(0.5 + p / 200.0))


def run_cell(cell: str, sub: pd.DataFrame, m: int) -> dict:
    """拟合一次 M4，返回测试段每点的预测、区间与真值。"""
    import models_gpr

    t = sub["cycle"].to_numpy(float)
    y = sub["capacity_Ah"].to_numpy(float)
    model = models_gpr.fit_gpr(t[:m], y[:m])
    pred = models_gpr.predict_gpr(model, t[m:])
    return {"电池": cell, "循环": t[m:], "真值": y[m:], "均值": pred["mean"], "标准差": pred["std"]}


def picp_table(points: dict[str, dict]) -> pd.DataFrame:
    """每块电池 × 每档：PICP、MPIW、以及「要多宽才够」的诊断放大系数。"""
    rows = []
    for cell, d in points.items():
        y, mu, sd = d["真值"], d["均值"], d["标准差"]
        for p in LEVELS:
            z = z_of(p)
            lo, hi = mu - z * sd, mu + z * sd
            rows.append(
                {
                    "电池": cell,
                    "名义覆盖率": p,
                    "PICP": coverage(y, lo, hi),
                    "MPIW(Ah)": mpiw(lo, hi),
                    "点数": int(len(y)),
                    "平均半宽(Ah)": float(np.mean(z * sd)),
                }
            )
    return pd.DataFrame(rows)


def pool_table(points: dict[str, dict]) -> pd.DataFrame:
    """四块电池合并（按点数加权）——这是本节的主口径。"""
    y = np.concatenate([d["真值"] for d in points.values()])
    mu = np.concatenate([d["均值"] for d in points.values()])
    sd = np.concatenate([d["标准差"] for d in points.values()])
    rows = []
    for p in LEVELS:
        z = z_of(p)
        lo, hi = mu - z * sd, mu + z * sd
        rows.append(
            {
                "名义覆盖率": p,
                "PICP": coverage(y, lo, hi),
                "MPIW(Ah)": mpiw(lo, hi),
                "点数": int(len(y)),
            }
        )
    return pd.DataFrame(rows)


def needed_scale(points: dict[str, dict], target: float = 90.0) -> float:
    """诊断：把区间整体乘多少倍，合并后的 PICP 才刚好到 target%。

    **只是诊断量**——在测试集上拟合它等于用答案调答案，不可部署。
    """
    y = np.concatenate([d["真值"] for d in points.values()])
    mu = np.concatenate([d["均值"] for d in points.values()])
    sd = np.concatenate([d["标准差"] for d in points.values()])
    z = z_of(target)
    need = np.abs(y - mu) / sd  # 每个点需要多少个「标准差」才被盖住
    return float(np.quantile(need, target / 100.0) / z)


def assert_calibration(tbl: pd.DataFrame, pooled: pd.DataFrame, scale: float) -> None:
    """结构检查（写在调用方，变异才抓得到）。"""
    # 1. PICP 必须随名义覆盖率单调不减——写错分位数或上下界就会炸
    v = pooled.sort_values("名义覆盖率")["PICP"].to_numpy(float)
    assert np.all(np.diff(v) >= -1e-12), f"PICP 不随名义覆盖率单调：{v}"
    # 逐块也要单调：md 第三节引用的正是**逐块**的数字，只查合并口径会漏掉这张表
    for cell, g in tbl.groupby("电池"):
        vc = g.sort_values("名义覆盖率")["PICP"].to_numpy(float)
        assert np.all(np.diff(vc) >= -1e-12), (
            f"{cell} 的 PICP 随名义覆盖率不单调：{vc}——逐块表的上下界或档位写反了"
        )

    # 2. 本节的立论是「区间偏窄」，即 90% 的实际覆盖显著低于 90%。
    #    如果哪天它达标了，md 的结论必须重写——所以把它写成断言，而不是叙述。
    p90 = float(pooled.loc[pooled["名义覆盖率"] == 90, "PICP"].iloc[0])
    assert p90 < 0.90, (
        f"90% 区间的实际覆盖是 {p90:.0%}，没有偏窄——"
        "「区间过度自信」这个结论不成立，md 第三节要重写"
    )
    assert p90 > 0.0, "90% 区间的实际覆盖为 0，实现大概率错了（不是校准问题）"

    # 3. 诊断放大系数必须 >1（否则「要多宽才够」这句话说不通）
    assert scale > 1.0, f"诊断放大系数 {scale:.2f} ≤ 1，与偏窄的结论矛盾"

    # 4. 每块电池都要有点，否则合并口径是空跑
    assert (tbl["点数"] >= 40).all(), "有电池的测试点少于 40 个，PICP 没有分辨率"

    # 5. 「校准因电池而异」这条也要钉住：若各电池差不多，第三节必须改写成一句概括
    at90 = tbl.loc[tbl["名义覆盖率"] == 90, "PICP"]
    assert float(at90.max() - at90.min()) >= 0.3, (
        f"90% 档各电池的 PICP 极差只有 {at90.max() - at90.min():.0%}——"
        "「因电池而异」的说法不成立，md 第三节要重写"
    )


def fig16(tbl: pd.DataFrame, pooled: pd.DataFrame) -> Path:
    setup_matplotlib()
    cells = sorted(tbl["电池"].unique().tolist())
    colors = {"B0005": "#1f77b4", "B0006": "#d62728", "B0007": "#9467bd", "B0018": "#2ca02c"}

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.6))

    ax = axes[0]
    ax.plot([0, 100], [0, 100], ls="--", lw=1.2, color="#777777", label="完美校准")
    for cell in cells:
        g = tbl[tbl["电池"] == cell].sort_values("名义覆盖率")
        ax.plot(g["名义覆盖率"], g["PICP"] * 100, "-o", ms=4,
                color=colors.get(cell, "#333"), label=cell)
    ax.plot(pooled["名义覆盖率"], pooled["PICP"] * 100, "-s", ms=6, lw=2.4,
            color="#111111", label="四块合并（主口径）")
    ax.set_xlabel("名义覆盖率（%）")
    ax.set_ylabel("实际覆盖率 PICP（%）")
    ax.set_title("① 校准曲线：全部落在对角线下方 = 过度自信", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    w = 0.15
    idx = np.arange(len(LEVELS))
    for k, cell in enumerate(cells):
        g = tbl[tbl["电池"] == cell].set_index("名义覆盖率").reindex(LEVELS)
        ax.bar(idx + (k - 1.5) * w, g["MPIW(Ah)"].to_numpy(float), w,
               label=cell, color=colors.get(cell, "#333"))
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{p}%" for p in LEVELS])
    ax.set_yscale("log")
    ax.set_xlabel("名义覆盖率")
    ax.set_ylabel("MPIW（区间宽度，Ah，对数轴）")
    ax.set_title(f"② 区间宽度随名义覆盖率增长：99% 档最宽 "
                 f"{tbl.loc[tbl['名义覆盖率'] == 99, 'MPIW(Ah)'].max():.3f} Ah", fontsize=10)
    ax.legend(fontsize=8)

    fig.suptitle("图16　区间校准：名义覆盖率 vs 实际覆盖率", fontsize=13, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save(fig, "fig16_calibration.png")


def render_md(tbl: pd.DataFrame, pooled: pd.DataFrame, scale: float, m: int,
              eol_cover: dict) -> str:
    p90 = pooled.loc[pooled["名义覆盖率"] == 90].iloc[0]
    p95 = pooled.loc[pooled["名义覆盖率"] == 95].iloc[0]
    p50 = pooled.loc[pooled["名义覆盖率"] == 50].iloc[0]
    p99 = pooled.loc[pooled["名义覆盖率"] == 99].iloc[0]
    n_pts = int(pooled["点数"].iloc[0])
    worst = tbl.sort_values("PICP").iloc[0]
    best = tbl.sort_values("PICP").iloc[-1]

    at90 = tbl.loc[tbl["名义覆盖率"] == 90].sort_values("PICP")
    c_worst, c_best = at90.iloc[0], at90.iloc[-1]
    spread = float(c_best["PICP"] - c_worst["PICP"])

    out = [
        "# 区间校准：名义覆盖率 vs 实际覆盖率（W18）",
        "",
        "> 本文件由 `src/calibration.py` 自动生成，数字随代码重算，**不要手工编辑**。",
        f"> 机器可读版：`outputs/metrics/{CSV_OUT.name}`。",
        "",
        "## 一、口径",
        "",
        "M4 给的是**预测区间**（`predict_gpr` 的标准差已含残差噪声 + 趋势参数两层，见 W11 方差分解）。",
        f"取切分点 M = {m}，在测试段逐点检验：",
        "",
        "- **PICP** = 实际值落进区间的比例；名义 90% 就该覆盖 90%；",
        "- **MPIW** = 区间平均宽度（Ah）。**必须与 PICP 一起报**——把区间放到 ±100 Ah，"
        "PICP 当然是 100%，那没有意义。",
        "",
        f"样本：{n_pts} 个测试点（四块合并，按点数加权）。",
        "",
        "## 二、主表（四块合并）",
        "",
        pooled.round(4).to_markdown(index=False),
        "",
        "逐块看（**这才是细节所在，合并会把它抹掉**）：",
        "",
        tbl.round(4).to_markdown(index=False),
        "",
        "## 三、结论一：整体**过度自信**，但**因电池而异**",
        "",
        f"合并口径：名义 50% → 实际 **{p50['PICP']:.0%}**；90% → **{p90['PICP']:.0%}**；"
        f"95% → **{p95['PICP']:.0%}**；99% → **{p99['PICP']:.0%}**。"
        "整条曲线落在对角线下方——**M4 说「90% 概率」，实际只有 "
        f"{p90['PICP']:.0%}**。",
        "",
        "但**这个合并数字掩盖了最大的事实**：90% 档上，"
        f"最差的是 **{c_worst['电池']}（{c_worst['PICP']:.0%}）**，"
        f"最好的是 **{c_best['电池']}（{c_best['PICP']:.0%}）**，"
        f"极差 **{spread:.0%}**。",
        "",
        f"- **{c_worst['电池']}**：区间严重偏窄——名义 90% 只覆盖 {c_worst['PICP']:.0%}，"
        "意味着它的真实不确定性比模型自报的大得多；",
        f"- **{c_best['电池']}**：几乎完美校准（{c_best['PICP']:.0%} vs 90%），"
        "甚至在某些档位偏保守。它的区间为什么这么宽？因为它本身是**删失个体**"
        "（从不穿越 1.4 Ah），GP 对它的远端外推本来就没有把握。",
        "",
        "所以正确的说法是：**「M4 的区间过度自信」这句话只在部分电池上成立。**"
        "把它当全局结论会让报告失真——尤其 B0007 那一块，它的区间是可信的。",
        "",
        "## 四、要多宽才够？——这是**诊断**，不是校准方案",
        "",
        f"如果强行把区间整体乘 **{scale:.2f} 倍**，合并后的 PICP 会刚好到 90%。",
        "",
        "**但这个数不能用**：它是在测试集上算出来的，等于用答案调答案。"
        "把它写进部署代码，会得到一个漂亮的校准曲线和毫无意义的实际性能。"
        "本节只用它说明一件事——**缺的宽度是有限的、可量化的**：",
        "",
        f"名义 90% 的区间平均宽 {p90['MPIW(Ah)']:.4f} Ah，"
        f"要放大到 **{p90['MPIW(Ah)'] * scale:.4f} Ah** 才真的覆盖 90%。",
        "",
        "真正可部署的改进方向是 W15 那条：**把定义层与模型形式层显式加进去**——"
        "区间里现在只装了两层（残差 + 趋势参数），而 W15 指出还有两整层不在里面："
        "定义层（11–20 循环）与模型形式层（0–22 循环，全部来自 M2）。"
        "这个放大系数，本质上就是在补那两层的宽度。",
        "",
        "## 五、EOL 区间的覆盖：n 太小，**没有分辨率**",
        "",
        "| 电池 | 90% 区间 | 真值 | 是否覆盖 |",
        "|---|---|---|---|",
    ]
    for cell, rec in eol_cover.items():
        out.append(
            f"| {cell} | {rec['interval']} | {rec['truth']} | {rec['covered']} |"
        )
    covered = sum(1 for r in eol_cover.values() if r["covered"] == "覆盖")
    total = sum(1 for r in eol_cover.values() if r["covered"] in ("覆盖", "未覆盖"))
    out += [
        "",
        f"**{covered}/{total} 块覆盖**。但 {total} 块电池的覆盖率只能取 0、1/3、2/3、1 四档，"
        "**这个数字没有统计分辨率**——它只能说明方向（偏窄），不能当作校准结论。"
        "容量点上的 316 个样本才是本节的可信证据。",
        "",
        "## 六、这张表不能怎么读",
        "",
        "1. **不能只看 PICP 不看 MPIW**：放宽区间能让 PICP 好看，但那只是把不确定性藏起来。",
        "2. **不能把「诊断放大系数」当校准**：它是测试集上的事后量，不可部署（第四节）。",
        f"3. **不能把 {covered}/{total} 当校准结论**：n 太小，见第五节。",
        "4. **不能说「所有电池都一样差」**：逐块差别很大（见第二节第二张表），"
        "有的电池在低档位就已接近名义值。",
        "5. **不能跨切分点比较 MPIW**：区间宽度随 M 变化（W13 的切分点敏感性表有数）。",
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

    tbl = picp_table(points)
    pooled = pool_table(points)
    scale = needed_scale(points, 90.0)
    assert_calibration(tbl, pooled, scale)

    # EOL 区间的覆盖（n 很小，只作方向性说明）
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

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
    TABLE_MD.write_text(render_md(tbl, pooled, scale, SPLIT_M, eol_cover), encoding="utf-8")
    fig_path = fig16(tbl, pooled)

    print(f"M = {SPLIT_M}；名义档位 {LEVELS}；测试点 {int(pooled['点数'].iloc[0])}")
    print(pooled.round(4).to_string(index=False))
    print()
    print(f"诊断放大系数（到 90% 覆盖）：{scale:.2f}")
    print(f"已写出：{TABLE_MD.name} / {CSV_OUT.name} / {fig_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
