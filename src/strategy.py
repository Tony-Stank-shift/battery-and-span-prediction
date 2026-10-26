"""W19：策略建议——把前面所有结果落成可执行的维护判据。

这一节不产生新的模型，只回答一个问题：
**手上这套东西，一个运维的人明天该怎么用？**

三条策略，每条都必须有【判据 / 代价 / 风险】三栏，缺一栏就是空话：

S1 替换点怎么定
   判据：`替换时刻 = EOL(τ_hi) − (模型层 m + 数据层 s)`，
   其中 `τ_hi = 1.40 + 0.05` 是**更严的失效阈值**，`EOL(τ_hi)` 由原始序列直接读穿越点（无模型）。
   代价：提前替换 = 浪费寿命（循环数可算）。
   风险：滞后替换 = 用到失效（循环数可算）。
S2 要采集多少数据才敢下结论
   判据：观测到 W16 的稳定时刻 t* 之后再报寿命；报的是**剩余寿命 = 替换时刻 − t***。
   代价：多测 t* 个循环。
   风险：剩余寿命为负 ⇒ 等数据到齐时已经过了该换的时刻（B0006 就是这个情况）。
S3 阈值口径必须先写死
   判据：先定 τ（本项目 1.40 Ah），并写明 ±0.05 Ah 窗口；定义层带宽单独报，**不进 S1 的裕度**。

为什么 S1 不再直接减「定义层半宽」（第十二轮审查的重写）
------------------------------------------------------
上一版把 W15 的 `sum = d + m + s` 整个从点估计里减掉，把 **d（τ 窗口的对称带宽）**
当成了时间裕度。但 d 不是 EOL 的误差：它是「换个惯例，答案平移多少」。
同一个 d 在 S3 里被称为「必须承认的报告带宽」，在 S1 里又变成「必须牺牲的寿命」，自相矛盾。
而且对 B0006，d=19.5 循环远大于「τ 抬到 1.45 后真实提前量」22（实测），
两者的差被当成保险卖掉，实际并没有保护任何东西。

现在的做法把两件事分开：
- **定义**：用更严的阈值 τ_hi 重新读穿越点（这是「定义」，不是「误差」）；
- **不确定性**：m + s 两层（模型形式 + 统计），这才是「这个数可能偏」的部分。
d 只在 S3 里出现，作为报告时必须写明的带宽。

三条策略的量纲统一
------------------
S1 用**循环**（替换时刻），S2 用**循环**（采集量与剩余寿命），S3 用**循环**（阈值带宽）。
这是整个项目「统一折算成循环数」的落点。

产物
----
- `outputs/tables/strategy.md`
- `outputs/metrics/strategy.csv`
- `outputs/figures/fig17_strategy.png`

运行
----
    python src/strategy.py

依赖：W15 的 `uncertainty.build_budget`；W16 的 `metrics/warning_time.csv`（由 run_all 先跑生成）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sensitivity  # noqa: E402
from config import (  # noqa: E402
    COLORS,
    EOL_THRESHOLD_AH,
    METRIC_DIR,
    TABLE_DIR,
    ensure_dirs,
    setup_matplotlib,
)
from evaluate import SPLIT_M  # noqa: E402
from figures import save  # noqa: E402
from load_data import load_raw, split_by_battery  # noqa: E402
from preprocess import cell_profile, mark_rows  # noqa: E402

TABLE_MD = TABLE_DIR / "strategy.md"
CSV_OUT = METRIC_DIR / "strategy.csv"
WARNING_CSV = METRIC_DIR / "warning_time.csv"

TAU_HI = round(EOL_THRESHOLD_AH + 0.05, 4)
"""S1 用的「更严阈值」：比正式口径高 0.05 Ah（与 W7/W15 的窗口半宽一致）。"""

RULES = ("点估计", "RSS 裕度", "SUM 裕度")
"""三种 S1 判据。点估计 = 不减裕度；RSS/SUM = 减掉 m+s 的两种合并口径。"""

LEVEL_FOR_S2 = 95
"""S2 用的 t* 档位。选它的理由：80/90% 档的 t* 在 B0006 上反而更晚（110/100），
99% 档虽然最早（90）但区间最宽、可用于决策的余量最小。95% 是「够严又不至于过宽」的折中——
这是**判断**，所以四档的 t* 也在 md 里一并列出。"""


def eol_at_tau(series: dict[str, tuple[np.ndarray, np.ndarray]], tau: float) -> dict[str, float]:
    """无模型地读穿越点：`EOL(τ) = 首个容量 ≤ τ 的循环`。没穿越则 NaN。"""
    out = {}
    for cell, (_, cap) in series.items():
        v = sensitivity.eol_at_exact(cap, tau)
        out[cell] = float(v) if v is not None else float("nan")
    return out


def replacement_table(budget: pd.DataFrame, eol_hi: dict[str, float]) -> pd.DataFrame:
    """S1：三种判据下的替换时刻、提前量、浪费比例。

    基准是 `EOL(τ_hi)`（更严阈值下的真实穿越点），裕度只取模型层 + 数据层。
    """
    rows = []
    for _, r in budget.iterrows():
        cell = r["电池"]
        base = eol_hi[cell]  # 更严阈值下的穿越点（无模型）
        margin_m = float(r["模型层半极差"])
        margin_s = float(r["数据层半宽"])
        for rule, half in (
            ("点估计", 0.0),
            # RSS：只对 m 与 s 做平方和（不含定义层 d——d 走 S3，不进裕度）
            ("RSS 裕度", float(np.hypot(margin_m, margin_s))),
            ("SUM 裕度", margin_m + margin_s),
        ):
            when = base - half
            lead = (float(r["真值"]) - when) if pd.notna(r["真值"]) else float("nan")
            rows.append(
                {
                    "电池": cell,
                    "判据": rule,
                    "EOL(τ_hi)": base,
                    "裕度(循环)": half,
                    "替换时刻": when,
                    "真值": r["真值"],
                    # 正 = 比真值早换（浪费寿命）；负 = 比真值晚换（用到失效）
                    "提前量(循环)": lead,
                    "提前量比例": (lead / float(r["真值"])) if pd.notna(r["真值"]) else float("nan"),
                    # 提前量的两个来源，必须分开报（R12-H03）：
                    #   阈值平移 = 真值(τ=1.40) − EOL(τ_hi)，属于「定义」；
                    #   裕度 = m + s，属于「不确定性」。
                    "其中阈值平移": (float(r["真值"]) - base) if pd.notna(r["真值"]) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def acquisition_table(budget: pd.DataFrame, n_obs: dict[str, int],
                      rep: pd.DataFrame) -> pd.DataFrame:
    """S2：稳定时刻 t*（W16 产物）、真实观测长度、以及「剩余寿命」。"""
    if not WARNING_CSV.exists():
        raise FileNotFoundError(
            f"缺 {WARNING_CSV}——请先跑 `python src/warning.py`（run_all 会按顺序跑）"
        )
    w = pd.read_csv(WARNING_CSV)
    assert set(w["电池"]) == set(budget["电池"]), (
        f"W16 产物里的电池集合与数据不一致：{sorted(w['电池'])}"
    )

    rows = []
    for cell in sorted(budget["电池"]):
        g = w[w["电池"] == cell].set_index("置信度")
        t_star = float(g.loc[LEVEL_FOR_S2, "t*_stable 最小"])
        sub = rep[(rep["电池"] == cell) & (rep["判据"] == RULES[2])]
        when = float(sub["替换时刻"].iloc[0]) if len(sub) else float("nan")
        rows.append(
            {
                "电池": cell,
                "观测长度": n_obs[cell],
                f"t*(95%)": t_star,
                "t*(80%)": float(g.loc[80, "t*_stable 最小"]),
                "t*(90%)": float(g.loc[90, "t*_stable 最小"]),
                "t*(99%)": float(g.loc[99, "t*_stable 最小"]),
                "可省循环": n_obs[cell] - t_star,
                "SUM 判据替换时刻": when,
                "剩余寿命(t* 之后还能用)": when - t_star,
                "在噪声内": bool(g.loc[LEVEL_FOR_S2, "在噪声内"]),
            }
        )
    return pd.DataFrame(rows)


def assert_strategy(rep: pd.DataFrame, acq: pd.DataFrame, n_obs: dict[str, int]) -> None:
    """结构检查（写在调用方，变异才抓得到）。**不设无出处的阈值**（D26）。"""
    # 1. 三种判据的替换时刻必须非增，且「点估计 → RSS」必须严格（裕度 > 0）。
    #    「RSS → SUM」只在模型层非零时才严格：m=0 的电池（B0018）两者恰好相等，
    #    所以要求「至少一块电池上 RSS 严格早于 SUM」，否则这条区分是空的。
    n_strict = 0
    for cell, g in rep.groupby("电池"):
        order = [g.loc[g["判据"] == r, "替换时刻"].iloc[0] for r in RULES]
        assert order[0] > order[1] >= order[2] - 1e-9, (
            f"{cell} 的替换时刻没按「点估计 > RSS ≥ SUM」排：{order}"
        )
        n_strict += int(order[1] > order[2] + 1e-9)
    assert n_strict >= 1, (
        "没有任何一块电池的 RSS 严格早于 SUM——两种合并口径没区分开（或模型层全为 0）"
    )
    # W19-M1 会打上面那条；这里再钉一条更直接的：SUM 必须严格早于 RSS 的**裕度**
    assert (rep.loc[rep["判据"] == RULES[2], "裕度(循环)"].to_numpy(float)
            > rep.loc[rep["判据"] == RULES[1], "裕度(循环)"].to_numpy(float)).any(), (
        "SUM 的裕度没有任何一块电池严格大于 RSS 的裕度——最坏情况口径没起作用"
    )
    # 2. 有真值时，SUM 判据必须把全部电池换在失效之前（它存在的理由）
    lab = rep[rep["真值"].notna() & (rep["判据"] == RULES[2])]
    assert lab["电池"].nunique() >= 2, "有真值的电池少于 2 块，这条断言形同虚设"
    assert (lab["提前量(循环)"] > 0).all(), (
        f"SUM 判据仍漏检：{lab.loc[lab['提前量(循环)'] <= 0, ['电池']].to_dict('records')}"
    )

    # 3. 观测长度必须等于数据里的真实长度（H-01：上一版用「网格最后一点+10」，少算了 8 个循环）
    for cell, n in n_obs.items():
        got = int(acq.loc[acq["电池"] == cell, "观测长度"].iloc[0])
        assert got == n, f"{cell} 的观测长度写成 {got}，数据里是 {n}"

    # 4. 至少两块电池的「剩余寿命」为正；为负的必须被 md 点名（H-02）
    assert (acq["剩余寿命(t* 之后还能用)"] > 0).sum() >= 2, (
        "剩余寿命为正的电池少于 2 块——S1 与 S2 完全接不上，md 的流程要重写"
    )
    # 稳定时刻不能等于观测长度（那是默认值，不是结论）
    assert (acq["t*(95%)"] < acq["观测长度"]).all(), (
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
        else:
            ax.text(i, 5, "删失\n无真值", ha="center", va="bottom", fontsize=8, color="#8a6d3b")
    ax.set_xticks(np.arange(len(cells)))
    ax.set_xticklabels(cells)
    ax.set_ylabel("建议替换时刻（循环）")
    ax.set_title("① S1 替换点：τ=1.45 起算 + 模型/数据裕度", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1]
    idx = np.arange(len(cells))
    ax.bar(idx - 0.2, acq["观测长度"], 0.4, label="真实观测长度", alpha=0.85)
    ax.bar(idx + 0.2, acq["t*(95%)"], 0.4, label="S2：稳定时刻 t*(95%)", alpha=0.9)
    for i, row in acq.reset_index(drop=True).iterrows():
        if row["剩余寿命(t* 之后还能用)"] <= 0:
            ax.text(i, max(row["观测长度"], row["t*(95%)"]) * 1.02,
                    "t* 已过替换点", ha="center", fontsize=7.5, color="#c0392b")
    ax.set_xticks(idx)
    ax.set_xticklabels(cells)
    ax.set_ylabel("循环数")
    ax.set_title("② S2 采集量：观测到 t* 就够；B0006 等不到", fontsize=10)
    ax.legend(fontsize=8)

    fig.suptitle("图17　替换点（S1）与采集量（S2）；阈值口径见 S3", fontsize=13, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save(fig, "fig17_strategy.png")


def render_md(rep: pd.DataFrame, acq: pd.DataFrame, budget: pd.DataFrame,
              n_obs: dict[str, int]) -> str:
    lab = rep[rep["真值"].notna()]
    sum_lab = lab[lab["判据"] == RULES[2]]
    rss_lab = lab[lab["判据"] == RULES[1]]
    pt_lab = lab[lab["判据"] == RULES[0]]
    missed = pt_lab.loc[pt_lab["提前量(循环)"] < 0, "电池"].tolist()
    zero_margin = pt_lab.loc[pt_lab["提前量(循环)"] == 0, "电池"].tolist()
    blocked = acq.loc[acq["剩余寿命(t* 之后还能用)"] <= 0, "电池"].tolist()
    t_lo = float(acq["t*(95%)"].min())
    t_hi = float(acq["t*(95%)"].max())
    save_lo = float(acq["可省循环"].min())
    save_hi = float(acq["可省循环"].max())
    # 定义层带宽：B0007 是短窗口，单独标注（R12-M01）
    short = budget.loc[budget["定义层口径"].astype(str).str.contains("下端删失"), "电池"].tolist()
    long_only = budget.loc[budget["定义层口径"].astype(str).str.contains("完整窗口"), "定义层半宽"]
    b7 = budget.loc[budget["电池"].isin(short)]
    b7_note = (
        f"（{'、'.join(short)} 是 0.0495 Ah 短窗，得 "
        f"{b7['定义层半宽'].min():.0f}，同宽替代下为 20）"
        if short else ""
    )

    rep_disp = rep.copy()
    rep_disp["真值"] = rep_disp["真值"].map(lambda v: "删失" if pd.isna(v) else int(v))
    rep_disp["提前量(循环)"] = rep_disp["提前量(循环)"].map(
        lambda v: "—" if pd.isna(v) else round(float(v), 1))
    rep_disp["其中阈值平移"] = rep_disp["其中阈值平移"].map(
        lambda v: "—" if pd.isna(v) else round(float(v), 1))
    rep_disp["提前量比例"] = rep_disp["提前量比例"].map(
        lambda v: "—" if pd.isna(v) else f"{float(v):.1%}")
    acq_disp = acq.copy()
    for c in acq_disp.columns:
        if acq_disp[c].dtype.kind == "f":
            acq_disp[c] = acq_disp[c].map(lambda v: "—" if pd.isna(v) else round(float(v), 1))

    out = [
        "# 维护策略：三条可执行判据（W19）",
        "",
        "> 本文件由 `src/strategy.py` 自动生成，数字随代码重算，**不要手工编辑**。",
        f"> 机器可读版：`outputs/metrics/{CSV_OUT.name}`。",
        f"> 切分点 M = {SPLIT_M}；裕度来自 W15，采集时刻来自 W16 的产物。",
        "",
        "这一节不产生新模型，只回答：**这套东西明天怎么用。**",
        "三条策略都能折算成**循环数**，所以可以放在同一张表里比较。",
        "",
        "## S1 替换点怎么定",
        "",
        "**判据**：`替换时刻 = EOL(τ_hi) − 裕度`，其中",
        "",
        f"- `τ_hi = {TAU_HI:g} Ah`——比正式口径 {EOL_THRESHOLD_AH:g} Ah **更严**的失效阈值，"
        "它的穿越点直接从原始序列读，不经过模型；",
        "- `裕度 = 模型形式层 m + 数据层 s`（W15 的两层），"
        "两种合并口径（RSS / SUM）只在 m 与 s 的相对权重上不同。",
        "",
        "**为什么不用 W15 的 `sum`（它含定义层 d）**：d 是「τ 挪 ±0.05 时答案平移多少」，"
        "是**定义**的不确定性，不是 EOL 的误差。把它当时间裕度减掉，"
        "等于把「你选了哪个惯例」当成「这个预测可能偏」——同一份报告里 S3 又说它是"
        "「必须承认的报告带宽」，两处会打架（第十二轮审查 H-03）。",
        "",
        rep_disp.round(2).to_markdown(index=False),
        "",
        "**代价**（提前换 = 浪费寿命）：",
        "",
        f"- SUM 裕度：{sum_lab['提前量(循环)'].min():.1f}–{sum_lab['提前量(循环)'].max():.1f} 循环，"
        f"相当于真实寿命的 **{sum_lab['提前量比例'].min():.1%}–{sum_lab['提前量比例'].max():.1%}**；",
        f"- RSS 裕度：{rss_lab['提前量(循环)'].min():.1f}–{rss_lab['提前量(循环)'].max():.1f} 循环，"
        f"**{rss_lab['提前量比例'].min():.1%}–{rss_lab['提前量比例'].max():.1%}**。",
        "",
        "**这个代价要拆成两半看**（第十二轮审查 H-03 的要害）：",
        "",
        "| 来源 | 是什么 | 能不能省 |",
        "|---|---|---|",
        "| **阈值平移** | 把失效标准从 1.40 收紧到 1.45，真实失效点提前的循环数 | 不能省——是你选的惯例 |",
        "| **不确定性裕度** | 模型形式层 m + 数据层 s（SUM 口径直接相加） | 可以买小，代价是漏检风险 |",
        "",
        rep_disp[["电池", "判据", "替换时刻", "真值", "提前量(循环)", "其中阈值平移",
                  "裕度(循环)"]].round(2).to_markdown(index=False),
        "",
        "**风险**（滞后换 = 用到失效）：",
        "",
        (
            "- 点估计判决在 " + "、".join(missed) + " 上**晚于真实失效**——"
            "不加减裕度的预测就是不够保守；"
            if missed else
            "- 点估计判决没有晚于失效的电池——但这是这几块电池恰好偏得不多，不是设计使然；"
        ),
        (
            "- " + "、".join(zero_margin) + " 上点估计**刚好卡在失效点**（提前量 0 循环），"
            "没有任何余量。"
            if zero_margin else
            "- 点估计判决没有零余量的电池。"
        ),
        "",
        f"**选哪条**：停机代价 ≫ 换电池代价就用 SUM 裕度（最多浪费 "
        f"{sum_lab['提前量(循环)'].max():.0f} 个循环）；寿命是稀缺资源就用 RSS。"
        "**两条都要求你先把「浪费多少寿命可以接受」写下来**——那是运维决策，不是统计决策。",
        "",
        "## S2 要采集多少数据才敢下结论",
        "",
        f"**判据**：观测到 W16 的稳定时刻 t*（本表用 {LEVEL_FOR_S2}% 档）之后再报寿命；"
        "报的是**剩余寿命 = SUM 替换时刻 − t***。",
        "",
        acq_disp.to_markdown(index=False),
        "",
        f"**代价**：要采到第 {t_lo:.0f}–{t_hi:.0f} 个循环。",
        "",
        f"**收益**：相比采满全程，可省 **{save_lo:.0f}–{save_hi:.0f}** 个循环的数据量。",
        "",
        "**风险**：",
        "",
    ]

    if blocked:
        out += [
            "**这是本节最要紧的一条**：" + "、".join(blocked) + " 的 t\\* **已经晚于 SUM 判据的替换时刻**，"
            "剩余寿命为负——**等数据到齐时已经过了该换的时刻**。",
            "",
            "对这几块电池，可执行的结论只有一句：**不要等预测稳定，按当前（还不稳定的）点估计提前换**。"
            "换句话说，S2 的「等到 t\\* 再报」与 S1 的「保守替换」对它们**互相矛盾**，"
            "你必须先选一个：要么接受提前换，要么接受不稳定的预测。",
            "",
        ]
    out += [
        f"- t* 用的是 {LEVEL_FOR_S2}% 档，这是**判断不是定理**：四档的 t* 并列在表里，"
        "换档位结论会变（B0006 在 80/90/95/99% 是 110/100/100/90）。",
        "- `在噪声内` 为真的电池，t* 的余量 ≤1 循环（W16 的口径），**不能当精确阈值**；"
        "本表 95% 档的四格跨种子移动都是 0，但换步长会动（W16 第六节）。",
        "- S2 只保证「预测不再改主意」，**不保证预测对**——W16 的例子是 B0006 90% 档"
        "稳定在 100、命中要等到 110（差 10 个循环）。",
        "",
        "## S3 阈值口径必须先写死",
        "",
        "**判据**：先定 τ，并在报告里写明窗口（本项目正式口径 "
        f"τ = {EOL_THRESHOLD_AH:g} ± 0.05 Ah）。",
        "",
        f"**代价**：定义层带宽 {long_only.min():.0f}–{long_only.max():.0f} 循环" + b7_note +
        "必须承认——它不是模型误差，是「你把坏的标准定在哪」造成的。",
        "",
        "**风险**：换 τ 结论就会变：",
        "",
        "- W7：四块电池的寿命排序在 τ ∈ [1.4025, 1.60] 的 80 个点上**始终不变**"
        "（B0018 < B0006 < B0005 < B0007）——「谁先坏」不依赖阈值；",
        "- 但「坏在第几循环」极依赖阈值：**B0007 的删失判定只靠 0.0005 Ah 的余量**"
        "（最低 1.4005 Ah，距 1.4 只有额定的 0.025%）。τ 抬到 1.4005，"
        "它就立刻从「删失」变成「第 166 循环失效」。",
        "",
        "所以 S3 的正确写法是：**报排序可以，报绝对寿命必须带 τ 与窗口**。",
        "",
        "## 三条策略怎么合起来用（一段流程）",
        "",
        "1. 先定 τ，写进维护规范（S3）；",
        "2. 在线监测到第 t* 个循环后开始算寿命（S2）；",
        "3. 报两个数：`EOL(τ_hi)` 与 `SUM 裕度替换时刻`；剩余寿命 = 后者 − t*；",
        "4. **剩余寿命为负就不要等了**，按当前点估计直接换（B0006 就是这种）；",
        "5. 任何时候都不要只说「还能用 X 个循环」——必须带 τ、裕度口径与 M。",
        "",
        "## 这张表不能怎么读",
        "",
        f"1. **不能把替换时刻当建议值直接下发**：它依赖 `SPLIT_M = {SPLIT_M}`，"
        "换切分点会变（W18 第四节）；",
        "2. **不能把 SUM 裕度的浪费当「模型不好」**：那是买保险的成本，可以算成钱；",
        "3. **不能省掉「在噪声内」那一列**：t* 的余量常常 ≤1 循环（W16）；",
        "4. **不能跨电池套用同一个 t***：本数据只有 4 块，t* 由曲线形状决定；",
        f"5. **B0007 的 {'、'.join(short)} 行要单独读**：它删失、最低容量 {sensitivity.censored_below.__doc__ or ''}"
        "距 1.4 Ah 只有 0.0005 Ah，且模型预测的 159 循环已被观测否定"
        "（第 168 循环实测仍 1.43 Ah）。",
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
    series = {cell: (sub["cycle"].to_numpy(float), sub["capacity_Ah"].to_numpy(float))
              for cell, sub in split_by_battery(df).items()}
    n_obs = {cell: int(len(sub)) for cell, sub in split_by_battery(df).items()}

    eol_hi = eol_at_tau(series, TAU_HI)
    rep = replacement_table(budget, eol_hi)
    acq = acquisition_table(budget, n_obs, rep)
    assert_strategy(rep, acq, n_obs)

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(
        [rep.assign(表="S1"), acq.assign(表="S2", 判据="—")], ignore_index=True
    ).to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
    TABLE_MD.write_text(render_md(rep, acq, budget, n_obs), encoding="utf-8")
    fig_path = fig17(rep, acq)

    print(f"M = {SPLIT_M}；τ_hi = {TAU_HI:g} Ah")
    print(rep.round(1).to_string(index=False))
    print()
    print(acq.round(1).to_string(index=False))
    print()
    print(f"已写出：{TABLE_MD.name} / {CSV_OUT.name} / {fig_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
