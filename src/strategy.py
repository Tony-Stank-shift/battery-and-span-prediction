"""W19：策略建议——把前面所有结果落成可执行的维护判据。

这一节不产生新的模型，只回答一个问题：
**手上这套东西，一个运维的人明天该怎么用？**

三条策略，每条都必须有【判据 / 代价 / 风险】三栏，缺一栏就是空话：

S1 替换点怎么定
   判据：`替换时刻 = EOL 点估计 − 半宽`，半宽取哪个口径（点估计 / RSS / SUM）由下面的表选。
   代价：提前替换 = 浪费寿命（循环数可算）。
   风险：滞后替换 = 用到失效（循环数可算）。
S2 要采集多少数据才敢下结论
   判据：观测到 W16 的「稳定时刻」t*_stable 之后再报寿命。
   代价：多测 t* 个循环。
   风险：不足 t* 就报，预测还会改主意（W16 有数）。
S3 阈值口径必须先写死
   判据：先定 τ（如 1.40 Ah），并在报告里写明窗口；W7 的 ±0.05 Ah 敏感性给出后果。
   代价：定义层 11–20 循环的带宽必须承认。
   风险：换 τ 结论就变（W7：B0007 只靠 0.0005 Ah 的余量才「删失」）。

三层的量纲在这里统一
--------------------
S1 用**循环**（替换时刻），S2 用**循环**（采集量），S3 用**循环**（阈值带宽）。
所以三条策略可以放在同一张表里比较——这也是整个项目「统一折算成循环数」的落点。

产物
----
- `outputs/tables/strategy.md`
- `outputs/metrics/strategy.csv`
- `outputs/figures/fig17_strategy.png`

运行
----
    python src/strategy.py

（依赖 W15/W16 的产物：`uncertainty.py` 直接调用；`metrics/warning_time.csv` 由
`run_all.py` 的第 15 步生成，本模块读它并断言电池集合一致。）
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import METRIC_DIR, TABLE_DIR, ensure_dirs, setup_matplotlib  # noqa: E402
from evaluate import SPLIT_M  # noqa: E402
from figures import save  # noqa: E402
from load_data import load_raw  # noqa: E402
from preprocess import cell_profile, mark_rows  # noqa: E402

TABLE_MD = TABLE_DIR / "strategy.md"
CSV_OUT = METRIC_DIR / "strategy.csv"
WARNING_CSV = METRIC_DIR / "warning_time.csv"

RULES = ("点估计", "RSS 下界", "SUM 下界")
"""三种替换判据。RSS = 独立假设合并，SUM = 最坏情况合并（W15 的口径）。"""


def replacement_table(budget: pd.DataFrame) -> pd.DataFrame:
    """S1：三种判据下的替换时刻、浪费量（正=提前换，负=漏检）。"""
    rows = []
    for _, r in budget.iterrows():
        truth = r["真值"]
        for rule, half in (("点估计", 0.0), ("RSS 下界", r["rss"]), ("SUM 下界", r["sum"])):
            when = float(r["点估计"]) - float(half)
            lead = (float(truth) - when) if pd.notna(truth) else float("nan")
            rows.append(
                {
                    "电池": r["电池"],
                    "判据": rule,
                    "替换时刻": when,
                    "真值": truth,
                    # 正 = 比真值早换（浪费寿命）；负 = 比真值晚换（用到失效）
                    "提前量(循环)": lead,
                    # 浪费占真实寿命的比例：把循环换算成"这块电池的几成寿命"
                    "浪费比例": (lead / float(truth)) if pd.notna(truth) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def acquisition_table(budget: pd.DataFrame) -> pd.DataFrame:
    """S2：稳定时刻 t*（W16 的产物）与观测长度的比较。"""
    if not WARNING_CSV.exists():
        raise FileNotFoundError(
            f"缺 {WARNING_CSV}——请先跑 `python src/warning.py`（run_all 会按顺序跑）"
        )
    w = pd.read_csv(WARNING_CSV)
    assert set(w["电池"]) == set(budget["电池"]), (
        f"W16 产物里的电池集合与数据不一致：{sorted(w['电池'])}"
    )
    t95 = w.loc[w["置信度"] == 95]
    rows = []
    for _, r in budget.iterrows():
        sub = t95.loc[t95["电池"] == r["电池"]]
        rows.append(
            {
                "电池": r["电池"],
                "观测长度": int(sub["网格最后一点"].iloc[0]) + 10,
                "稳定时刻(95%)": sub["t*_stable 最小"].iloc[0],
                "命中时刻(95%)": sub["t*_hit 最小"].iloc[0],
                "点数不足 40 的风险": bool(sub["在噪声内"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def assert_strategy(rep: pd.DataFrame, acq: pd.DataFrame) -> None:
    """结构检查（写在调用方，变异才抓得到）。"""
    # 1. 判据越保守，替换越早——且必须是**严格**递减：
    #    RSS 半宽 > 0、SUM 半宽 > RSS（W15 保证），所以三种时刻在数学上不可能相等。
    #    写成 >= 会让「SUM 偷懒用 RSS」的变异跑通（第一版就是这么漏的）。
    for cell, g in rep.groupby("电池"):
        order = [g.loc[g["判据"] == r, "替换时刻"].iloc[0] for r in RULES]
        assert order[0] > order[1] > order[2], (
            f"{cell} 的替换时刻没按「点估计 > RSS > SUM」严格递减：{order}"
        )
    # 2. 有真值时，SUM 判据必须把全部电池都换在失效之前（这就是它存在的理由）
    lab = rep[rep["真值"].notna() & (rep["判据"] == "SUM 下界")]
    assert (lab["提前量(循环)"] > 0).all(), (
        f"SUM 判据仍漏检：{lab.loc[lab['提前量(循环)'] <= 0, ['电池']].to_dict('records')}"
    )
    # 3. 不能是空的：至少两块电池有真值可以检验
    assert lab["电池"].nunique() >= 2, "有真值的电池少于 2 块，第 2 条断言形同虚设"
    # 4. 采集策略不能用「观测长度」当稳定时刻（那是默认值，不是结论）
    assert (acq["稳定时刻(95%)"] < acq["观测长度"]).all(), (
        "有电池的稳定时刻等于观测长度——大概率是没算出来而填了默认值"
    )


def fig17(rep: pd.DataFrame, acq: pd.DataFrame) -> Path:
    setup_matplotlib()
    cells = sorted(rep["电池"].unique().tolist())
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.6))

    ax = axes[0]
    w = 0.26
    for k, rule in enumerate(RULES):
        vals = [float(rep.loc[(rep["电池"] == c) & (rep["判据"] == rule), "替换时刻"].iloc[0])
                for c in cells]
        ax.bar(np.arange(len(cells)) + (k - 1) * w, vals, w, label=rule, alpha=0.9)
    for i, c in enumerate(cells):
        t = rep.loc[rep["电池"] == c, "真值"].iloc[0]
        if pd.notna(t):
            ax.scatter([i], [float(t)], marker="*", s=170, color="#c0392b", zorder=6,
                       label="真实失效" if i == 0 else None)
    ax.set_xticks(np.arange(len(cells)))
    ax.set_xticklabels(cells)
    ax.set_ylabel("建议替换时刻（循环）")
    ax.set_title("① S1 替换点：三种判据 vs 真实失效", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1]
    idx = np.arange(len(cells))
    ax.bar(idx - 0.2, acq["观测长度"], 0.4, label="现有观测长度", alpha=0.85)
    ax.bar(idx + 0.2, acq["稳定时刻(95%)"], 0.4, label="S2：稳定时刻（95%）", alpha=0.9)
    for i, row in acq.reset_index(drop=True).iterrows():
        if bool(row["点数不足 40 的风险"]):
            y = max(float(row["观测长度"]), float(row["稳定时刻(95%)"])) * 1.02
            ax.text(i, y, "在噪声内", ha="center", fontsize=7.5, color="#8a6d3b")
    ax.set_xticks(idx)
    ax.set_xticklabels(cells)
    ax.set_ylabel("循环数")
    ax.set_title("② S2 采集量：观测到 t* 就够，不必采满全程", fontsize=10)
    ax.legend(fontsize=8)

    fig.suptitle("图17　三条维护策略：替换点 / 采集量 / 阈值口径", fontsize=13, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save(fig, "fig17_strategy.png")


def render_md(rep: pd.DataFrame, acq: pd.DataFrame, budget: pd.DataFrame) -> str:
    lab = rep[rep["真值"].notna()]
    sum_lab = lab[lab["判据"] == "SUM 下界"]
    rss_lab = lab[lab["判据"] == "RSS 下界"]
    pt_lab = lab[lab["判据"] == "点估计"]
    waste_sum = sum_lab["提前量(循环)"]
    waste_rss = rss_lab["提前量(循环)"]
    missed = pt_lab.loc[pt_lab["提前量(循环)"] < 0, "电池"].tolist()
    zero_margin = pt_lab.loc[pt_lab["提前量(循环)"] == 0, "电池"].tolist()
    t_lo = float(acq["稳定时刻(95%)"].min())
    t_hi = float(acq["稳定时刻(95%)"].max())
    save_lo = float((acq["观测长度"] - acq["稳定时刻(95%)"]).min())
    save_hi = float((acq["观测长度"] - acq["稳定时刻(95%)"]).max())
    tau_def = budget["定义层半宽"]

    out = [
        "# 维护策略：三条可执行判据（W19）",
        "",
        "> 本文件由 `src/strategy.py` 自动生成，数字随代码重算，**不要手工编辑**。",
        f"> 机器可读版：`outputs/metrics/{CSV_OUT.name}`。",
        f"> 切分点 M = {SPLIT_M}；区间与预算口径来自 W15，采集时刻来自 W16 的产物。",
        "",
        "这一节不产生新模型，只回答：**这套东西明天怎么用。**",
        "三条策略都能折算成**循环数**，所以可以放在同一张表里比较。",
        "",
        "## S1 替换点怎么定",
        "",
        "**判据**：`替换时刻 = EOL 点估计 − 半宽`，半宽取三种口径之一：",
        "",
        "| 判据 | 半宽从哪来 | 含义 |",
        "|---|---|---|",
        "| 点估计 | 0 | 相信中位数，不做保险 |",
        "| RSS 下界 | W15 的 `rss`（独立假设） | 乐观的保险 |",
        "| SUM 下界 | W15 的 `sum`（最坏情况） | 保守的保险 |",
        "",
        rep.round(2).to_markdown(index=False),
        "",
        "**代价**（提前换 = 浪费寿命）：",
        "",
        f"- SUM 判决：{waste_sum.min():.0f}–{waste_sum.max():.0f} 循环被提前浪费，"
        f"相当于真实寿命的 **{sum_lab['浪费比例'].min():.0%}–{sum_lab['浪费比例'].max():.0%}**；",
        f"- RSS 判决：{waste_rss.min():.0f}–{waste_rss.max():.0f} 循环，"
        f"**{rss_lab['浪费比例'].min():.0%}–{rss_lab['浪费比例'].max():.0%}**。",
        "",
        "**风险**（滞后换 = 用到失效）：",
        "",
        (
            f"- 点估计判决在 " + "、".join(missed) + " 上**晚于真实失效**——"
            "因为点估计本身偏高（W13/W14 的偏差），不保险；"
            if missed else
            "- 点估计判决没有晚于失效的电池，但那只是这几块的点估计恰好偏高得不多，不是设计使然；"
        ),
        (
            f"- " + "、".join(zero_margin) + " 上点估计**刚好卡在失效点**（提前量 0 循环），"
            "没有任何余量——只要估计偏高 1 个循环就变成「用到坏」。"
            if zero_margin else
            "- 点估计判决没有零余量的电池。"
        ),
        "",
        f"**选哪条**：如果停机代价 ≫ 换电池代价，用 SUM（最多浪费 "
        f"{waste_sum.max():.0f} 个循环）；如果寿命是稀缺资源，用 RSS。"
        f"**两条都要求你先把「浪费多少循环可以接受」写下来**——那是运维决策，不是统计决策。",
        "",
        "## S2 要采集多少数据才敢下结论",
        "",
        "**判据**：观测到 W16 的稳定时刻 t*_stable（95% 档）之后再报寿命。",
        "",
        acq.round(2).to_markdown(index=False),
        "",
        "**代价**：要采到第 "
        f"{t_lo:.0f}–{t_hi:.0f} 个循环才有稳定预测。",
        "",
        "**收益**：相比采满全程，可省 "
        f"{save_lo:.0f}–{save_hi:.0f} 个循环的数据量。",
        "",
        "**风险**：",
        "",
        "1. 上图 `在噪声内` 为真的电池，t* 换一个后验种子就会动 0–10 个循环（W16），"
        "**不能当精确阈值用**；",
        "2. t*_stable 只保证「预测不再改主意」，**不保证预测对**——W16 的例子是稳定了但还差 21 个循环。",
        "   所以 S2 必须与 S1 一起用：先用 S2 决定「什么时候报」，再用 S1 决定「报多少」。",
        "",
        "## S3 阈值口径必须先写死",
        "",
        "**判据**：先定 τ，并在报告里写明窗口（本项目的正式口径是 τ = 1.40 ± 0.05 Ah）。",
        "",
        f"**代价**：定义层半宽 {tau_def.min():.0f}–{tau_def.max():.0f} 循环的带宽必须承认——"
        "它不是模型误差，是「你把坏的标准定在哪」造成的。",
        "",
        "**风险**：换 τ 结论就会变：",
        "",
        "- W7：四块电池的寿命排序在 τ ∈ [1.4025, 1.60] 的 80 个点上**始终不变**"
        "（B0018 < B0006 < B0005 < B0007）——「谁先坏」不依赖阈值；",
        "- 但「坏在第几循环」极依赖阈值，且 **B0007 的删失判定只靠 0.0005 Ah 的余量**"
        "（最低 1.4005 Ah，距 1.4 只有 0.025% 的额定容量）。τ 抬到 1.4005 它就立刻变成「第 166 循环失效」。",
        "",
        "所以 S3 的正确写法是：**报排序可以，报绝对寿命必须带 τ 与窗口**。",
        "",
        "## 三条策略怎么合起来用（一段流程）",
        "",
        "1. 先定 τ，写进维护规范（S3）；",
        "2. 在线监测到第 t* 个循环后开始报寿命（S2）；",
        "3. 报的时候给两个数：点估计与 SUM 下界（S1），由运维按停机代价选一个；",
        "4. 任何时候都不要只说「还能用 X 个循环」——必须带上 τ、区间口径与 M。",
        "",
        "## 这张表不能怎么读",
        "",
        "1. **不能把替换时刻当建议值直接下发**：它依赖 `SPLIT_M = 80`，换切分点会变（W18 第四节）；",
        "2. **不能把 SUM 判据的浪费当「模型不好」**：那是买保险的成本，可以算成钱；",
        "3. **不能省掉「在噪声内」那一列**：9/16 的 t* 格子在蒙特卡洛噪声里（W16）；",
        "4. **不能跨电池套用同一个 t***：本数据只有 4 块，t* 由曲线形状决定。",
        "",
    ]
    return "\n".join(out)


def main() -> int:
    ensure_dirs()
    raw = load_raw()
    df = mark_rows(raw)
    prof = cell_profile(df)

    import uncertainty

    budget = uncertainty.build_budget(df, prof, SPLIT_M)

    rep = replacement_table(budget)
    acq = acquisition_table(budget)
    assert_strategy(rep, acq)

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([rep.assign(表="S1"), acq.assign(表="S2")], ignore_index=True).to_csv(
        CSV_OUT, index=False, encoding="utf-8-sig"
    )
    TABLE_MD.write_text(render_md(rep, acq, budget), encoding="utf-8")
    fig_path = fig17(rep, acq)

    print(f"M = {SPLIT_M}")
    print(rep.round(1).to_string(index=False))
    print()
    print(f"已写出：{TABLE_MD.name} / {CSV_OUT.name} / {fig_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
