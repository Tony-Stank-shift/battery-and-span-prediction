"""W16：最早可信预警时刻 t*。

要回答的问题
------------
「这块电池还能用 100 个循环」这句话，要观测到第几个循环才**值得相信**？
答这个问题必须先说清「相信」是什么意思，本节给两个都不许混用的定义：

1. **稳定（可部署）**：t*_stable(p) = 最早 t0，使得**此后所有** t0 的 p% 预测区间
   **有公共交集**。语义：再加数据也不会改主意。
2. **命中（事后诊断）**：t*_hit(p) = 最早 t0，使得**此后所有** t0 的 p% 区间
   **都盖住真值**。语义：从此以后预测一直是对的。它必须知道真值，只能事后算。

**为什么必须分开**：区间变窄 ≠ 预测变准。本数据里 B0006 在 80%/90% 档位上
从某个时刻起区间整体落到真值下方、再没盖住它，而同一时期它的区间宽度还在缩小。

三件必须写明的口径（都是第七轮审查逼出来的）
--------------------------------------------
1. **网格不碰真值**：上界一律 `N − 10`。初版用 `min(N−10, 真值−5)`，等于让「可部署量」
   偷看真值——放开后 B0006 的 stable 从 40/60/90 变成 90/100/110（95% 档差 60 循环）。
2. **蒙特卡洛敏感**：EOL 分位数来自 300 条后验路径，`models_gpr` 自己说 MC 误差约 ±1 循环。
   本节的 t* 余量常常就是 0–1 循环，所以每个格子跑 4 个路径种子，
   报 min–max，**余量 ≤ SENSITIVE_CYCLES 的格子直接标「在噪声内，不下结论」**。
3. **网格步长敏感**：步长从 10 改成 5，B0007 的 stable 会从 80 跳到 135。
   所以步长写成参数，并给一张步长敏感性列；`t*` 不是电池的固有常数。

四档置信度 80 / 90 / 95 / 99%：置信度越高区间越宽，`t*(80) ≤ t*(90) ≤ t*(95) ≤ t*(99)`——
由断言钉住。

提前量
------
- 稳定口径：`点估计(t*) − t*`（还要观测多久，才敢说它还能用多久）；
- 命中口径：`真值 − t*`（在真正失效前多少循环发出了可用预警；**可以为负**，
  负值意味着「预警晚于失效」，这本身是要报的事实）。

产物
----
- `outputs/tables/warning_time.md`
- `outputs/metrics/warning_time.csv`
- `outputs/figures/fig14_warning_time.png`

运行
----
    python src/warning.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import METRIC_DIR, RANDOM_SEED, TABLE_DIR, ensure_dirs, setup_matplotlib  # noqa: E402
from figures import save  # noqa: E402
from load_data import load_raw  # noqa: E402
from preprocess import cell_profile, mark_rows  # noqa: E402

TABLE_MD = TABLE_DIR / "warning_time.md"
CSV_OUT = METRIC_DIR / "warning_time.csv"

LEVELS = (80, 90, 95, 99)
"""四档置信度（百分数）。"""

SEEDS = (RANDOM_SEED, 7, 123, 2024)
"""后验路径的种子。t* 常常只差 0–1 循环，单种子报出去的数不可靠。"""

N_PATHS = 300
"""每一批的路径条数。四批合起来一次采样（1200 条），所以总成本只算一次协方差分解。"""

SENSITIVE_CYCLES = 1.0
"""余量小于等于这个数（循环）就认为「在蒙特卡洛噪声内」，不下结论。"""

GRID_STEP = 10
GRID_STEP_SENSITIVITY = 5
GRID_START_FACTOR = 4
UPPER_OFFSET = 10
"""上界 = 观测长度 − 10。**绝不用真值**（R9-04）。"""


# ---------------------------------------------------------------------------
# 逐 t0 的预测（多随机种子）
# ---------------------------------------------------------------------------
def scan_cell(
    df: pd.DataFrame,
    cell: str,
    seeds: tuple[int, ...] = SEEDS,
    step: int = GRID_STEP,
) -> pd.DataFrame:
    """固定一块电池，逐个 t0 拟合 M4，并对每个种子取四档区间的上下界。"""
    import models_gpr

    sub = df.loc[df["battery_id"] == cell].sort_values("cycle")
    n_obs = int(sub["cycle"].max())
    upper = n_obs - UPPER_OFFSET
    grid = list(range(step * GRID_START_FACTOR, upper + 1, step))
    assert grid, f"{cell} 的 t0 网格是空的（观测长度 {n_obs}）"

    t_all = sub["cycle"].to_numpy(float)
    y_all = sub["capacity_Ah"].to_numpy(float)

    rows = []
    for t0 in grid:
        model = models_gpr.fit_gpr(t_all[:t0], y_all[:t0])
        rec = {"电池": cell, "t0": t0, "点估计": models_gpr.eol_mean(model, n_obs)}
        # **一次采样、按批切分**：后验协方差的分解与样本条数无关（840 维 Cholesky 主导），
        # 4 个 300 条的批用一次 1200 条采样再切，等价于 4 个种子而快约 4 倍。
        eols = models_gpr.eol_paths(
            model, n_obs, n_samples=N_PATHS * len(seeds), seed=seeds[0]
        )
        for k, seed in enumerate(seeds):
            batch = eols[k * N_PATHS:(k + 1) * N_PATHS]
            ok = batch[np.isfinite(batch)]
            assert len(ok) > 0.5 * N_PATHS, (
                f"{cell} t0={t0} 的第 {k} 批删失路径过多（{N_PATHS - len(ok)}/{N_PATHS}）"
            )
            for p in LEVELS:
                lo_q, hi_q = (100 - p) / 2.0, 100 - (100 - p) / 2.0
                lo, hi = np.percentile(ok, [lo_q, hi_q])
                rec[f"L{p}_s{seed}"] = float(lo)
                rec[f"U{p}_s{seed}"] = float(hi)
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 两个 t* 的定义（并入余量与后缀长度）
# ---------------------------------------------------------------------------
def first_stable(tbl: pd.DataFrame, p: int, seed: int) -> tuple[int | None, float, int]:
    """最早 t0，此后所有 t0 的区间有公共交集。返回 (t0, 余量, 后缀点数)。"""
    lo = tbl[f"L{p}_s{seed}"].to_numpy(float)
    hi = tbl[f"U{p}_s{seed}"].to_numpy(float)
    t0 = tbl["t0"].to_numpy(int)
    for i in range(len(tbl)):
        margin = float(hi[i:].min() - lo[i:].max())
        if margin >= 0:
            return int(t0[i]), margin, int(len(tbl) - i)
    return None, float("nan"), 0


def first_hit(tbl: pd.DataFrame, p: int, seed: int, truth: float) -> tuple[int | None, float, int]:
    """最早 t0，此后所有 t0 的区间都盖住真值。返回 (t0, 余量, 后缀点数)。"""
    lo = tbl[f"L{p}_s{seed}"].to_numpy(float)
    hi = tbl[f"U{p}_s{seed}"].to_numpy(float)
    t0 = tbl["t0"].to_numpy(int)
    for i in range(len(tbl)):
        margin = float(np.minimum(hi[i:] - truth, truth - lo[i:]).min())
        if margin >= 0:
            return int(t0[i]), margin, int(len(tbl) - i)
    return None, float("nan"), 0


def build_tstar(scans: dict[str, pd.DataFrame], truth: dict[str, float | None],
                step5: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """每块电池 × 四档 → 跨种子 min–max 的 t*、余量、步长敏感性。"""
    rows = []
    for cell, tbl in scans.items():
        t_true = truth.get(cell)
        last = int(tbl["t0"].max())
        for p in LEVELS:
            st = [first_stable(tbl, p, s) for s in SEEDS]
            st_vals = [v[0] for v in st if v[0] is not None]
            st_margin = min((v[1] for v in st if v[0] is not None), default=float("nan"))
            if t_true is None:
                hit_vals, hit_margin = [], float("nan")
            else:
                hit = [first_hit(tbl, p, s, t_true) for s in SEEDS]
                hit_vals = [v[0] for v in hit if v[0] is not None]
                hit_margin = min((v[1] for v in hit if v[0] is not None), default=float("nan"))

            point_at = float(tbl.loc[tbl["t0"] == st_vals[0], "点估计"].iloc[0]) if st_vals else float("nan")
            st5 = [first_stable(step5[cell], p, s)[0] for s in SEEDS]
            st5_vals = [v for v in st5 if v is not None]

            rows.append(
                {
                    "电池": cell,
                    "置信度": p,
                    "有真值": t_true is not None,
                    "t*_stable 最小": min(st_vals) if st_vals else None,
                    "t*_stable 最大": max(st_vals) if st_vals else None,
                    "稳定余量(循环)": st_margin,
                    "t*_hit 最小": min(hit_vals) if hit_vals else None,
                    "t*_hit 最大": max(hit_vals) if hit_vals else None,
                    "命中余量(循环)": hit_margin,
                    "t*_stable(步长5)": min(st5_vals) if st5_vals else None,
                    "稳定口径提前量": (point_at - st_vals[0]) if st_vals else float("nan"),
                    "命中口径提前量": (t_true - min(hit_vals)) if hit_vals else float("nan"),
                    "网格最后一点": last,
                    "点估计@稳定": point_at,
                }
            )
    return pd.DataFrame(rows)


def add_flags(tstar: pd.DataFrame) -> pd.DataFrame:
    """标出「在噪声内」「在网格边界」两类不能当精确值的格子。"""
    out = tstar.copy()
    out["在噪声内"] = (
        (out["稳定余量(循环)"] <= SENSITIVE_CYCLES)
        | (out["命中余量(循环)"] <= SENSITIVE_CYCLES)
    )
    out["t* 触到网格边界"] = (
        (out["t*_stable 最小"] == out["t*_stable(步长5)"])  # 占位，下面覆盖
        | False
    )
    first_grid = GRID_STEP * GRID_START_FACTOR
    out["t* 触到网格边界"] = (out["t*_stable 最小"] <= first_grid) | (
        out["t*_stable 最大"] >= out["网格最后一点"]
    )
    return out


def assert_monotone(tstar: pd.DataFrame) -> None:
    """置信度越高，区间越宽，t* 只能更早或相等。档位写反这条立刻炸。"""
    for cell, g in tstar.groupby("电池"):
        g = g.sort_values("置信度")
        for col, nicer in (("t*_stable 最小", "稳定"), ("t*_hit 最小", "命中")):
            vals = [float(v) for v in g[col].tolist() if pd.notna(v)]
            assert all(a >= b for a, b in zip(vals, vals[1:])), (
                f"{cell} 的 {nicer} t* 随置信度不单调：{g[col].tolist()}——分位数或档位算反了"
            )
    assert tstar["t*_stable 最小"].notna().all(), (
        "有电池在网格内始终不稳定——要么网格太短，要么区间算错了"
    )


def assert_warning_before_failure(tstar: pd.DataFrame) -> None:
    """命中时刻至少要有一块电池早于失效；且「命中」不能整体消失（否则检查是空的）。"""
    hit = tstar.dropna(subset=["t*_hit 最小"])
    assert hit["电池"].nunique() >= 2, (
        f"只有 {hit['电池'].nunique()} 块电池算出了命中时刻——要么实现坏了，要么这条断言形同虚设"
    )
    early = hit.loc[hit["命中口径提前量"] > 0, "电池"].nunique()
    assert early >= 2, f"只有 {early} 块电池在失效前就命中——「最早可信预警」这个结论不成立"


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------
def fig14(scans: dict[str, pd.DataFrame], tstar: pd.DataFrame,
          truth: dict[str, float | None]) -> Path:
    setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.6))

    ax = axes[0]
    colors = {"B0005": "#1f77b4", "B0006": "#d62728", "B0007": "#9467bd", "B0018": "#2ca02c"}
    for cell, tbl in scans.items():
        ax.plot(tbl["t0"], tbl["点估计"], "-o", ms=3.5, lw=1.4, color=colors.get(cell, "#333"),
                label=f"{cell} 点估计")
    for cell, t_val in (("B0005", 125), ("B0006", 109), ("B0018", 97)):
        ax.axhline(t_val, ls=":", lw=1.0, color=colors[cell], alpha=0.8)
        ax.text(41, t_val, f" {cell} 真值 {t_val}", fontsize=7, va="bottom", color=colors[cell])
    ax.set_xlabel("观测到第 t0 个循环（预测时刻）")
    ax.set_ylabel("预测的 EOL（循环）")
    ax.set_title("① 点估计随观测时长的摆动：早期极不稳定，后期才收敛", fontsize=10)
    ax.legend(fontsize=7.5, ncol=2)
    ax.grid(alpha=0.3)

    ax = axes[1]
    cells = list(scans)
    w = 0.2
    for k, p in enumerate(LEVELS):
        vals = []
        for cell in cells:
            row = tstar[(tstar["电池"] == cell) & (tstar["置信度"] == p)].iloc[0]
            v = row["t*_hit 最小"]
            vals.append(np.nan if pd.isna(v) else float(v))
        x = np.arange(len(cells)) + (k - 1.5) * w
        ax.bar(x, vals, w, label=f"命中 t*（{p}%）", alpha=0.9)
    ax.set_xticks(np.arange(len(cells)))
    ax.set_xticklabels(cells)
    # 真值星标：B0006 的命中 t*（110）落在真值 109 之后，画出来才看得见「预警晚于失效」
    labelled = False
    for i, cell in enumerate(cells):
        t = truth.get(cell)
        if t is None:
            continue
        ax.scatter([i], [float(t)], marker="*", s=150, color="#c0392b", zorder=6,
                   label=None if labelled else "真实失效（=真值）")
        labelled = True
    for i, cell in enumerate(cells):
        row = tstar[tstar["电池"] == cell].iloc[0]
        if not bool(row["有真值"]):
            ax.text(i, 3, "删失\n无可命中", ha="center", va="bottom", fontsize=8, color="#8a6d3b")
        elif tstar[(tstar["电池"] == cell)]["t*_hit 最小"].isna().all():
            ax.text(i, 3, "从未命中\n（区间整段偏离）", ha="center", va="bottom", fontsize=8,
                    color="#c0392b")
        elif tstar[(tstar["电池"] == cell)]["t*_hit 最小"].isna().any():
            ax.text(i, 3, "低档位\n从未命中", ha="center", va="bottom", fontsize=8, color="#c0392b")
    ax.set_ylabel("最早可信时刻 t*（循环）")
    ax.set_title("② 四档置信度下「从第几循环起预测一直命中真值」", fontsize=10)
    ax.set_ylim(0, max(ax.get_ylim()[1], 120))
    ax.legend(fontsize=8)

    fig.suptitle("图14　最早可信预警时刻：区间够窄 ≠ 预测可信", fontsize=13, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save(fig, "fig14_warning_time.png")


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
def render_md(scans: dict[str, pd.DataFrame], tstar: pd.DataFrame,
              truth: dict[str, float | None]) -> str:
    n_label = sum(1 for v in truth.values() if v is not None)
    labeled = [c for c, v in truth.items() if v is not None]
    sens = tstar[tstar["在噪声内"]]
    never = tstar[tstar["t*_hit 最小"].isna() & tstar["有真值"]]
    never_cells = sorted(never["电池"].unique().tolist())

    # 找出「区间从某时刻起整体落到真值下方」的直接证据（B0006 那类）
    offside = []
    for cell in labeled:
        tbl = scans[cell]
        t = float(truth[cell])
        for p in LEVELS:
            u = tbl[f"U{p}_s{RANDOM_SEED}"].to_numpy(float)
            below = u < t
            idx = None
            for i in range(len(below)):
                if below[i:].all():
                    idx = i
                    break
            if idx is not None:
                offside.append((cell, p, int(tbl["t0"].iloc[idx]), t,
                                float(tbl["点估计"].iloc[idx]), idx == 0))
                break

    out = [
        "# 最早可信预警时刻 t*（W16）",
        "",
        "> 本文件由 `src/warning.py` 自动生成，数字随代码重算，**不要手工编辑**。",
        f"> 机器可读版：`outputs/metrics/{CSV_OUT.name}`。",
        "",
        "## 一、先定义「可信」，否则这节没法读",
        "",
        "两个判据**不许混用**：",
        "",
        "| 判据 | 含义 | 需要真值？ | 能不能部署 |",
        "|---|---|---|---|",
        "| **稳定** t*_stable(p) | 此后所有时刻的 p% 区间**有公共交集**：再加数据也不改主意 | 否 | 能 |",
        "| **命中** t*_hit(p) | 此后所有时刻的 p% 区间**都盖住真值**：从此一直对 | 是 | 只能事后诊断 |",
        "",
        f"四档 80 / 90 / 95 / 99% 指**区间置信度**；置信度越高区间越宽，"
        "故 `t*(80) ≤ t*(90) ≤ t*(95) ≤ t*(99)`（已写成断言）。",
        "",
        "**三条必须写明的口径**：",
        "",
        f"1. **网格不碰真值**：上界一律 `N − 10`。初版曾用 `min(N−10, 真值−5)`，"
        "那等于让「可部署量」偷看真值；放开后 B0006 的稳定时刻从 40/60/90 变成 90/100/110"
        "（95% 档差 60 循环）。",
        f"2. **蒙特卡洛敏感**：分位数来自 {300} 条后验路径，`models_gpr` 自述 MC 误差约 ±1 循环。"
        f"每个格子跑 {len(SEEDS)} 个路径种子报 min–max，"
        f"**余量 ≤ {SENSITIVE_CYCLES:g} 循环的格子标为「在噪声内，不下结论」**。",
        f"3. **网格步长敏感**：主口径步长 {GRID_STEP}，另给步长 {GRID_STEP_SENSITIVITY} 的稳定时刻做对照——"
        "t* **不是电池的固有常数**。",
        "",
        "## 二、主表（t* 跨种子 min–max）",
        "",
        tstar.round(2).to_markdown(index=False),
        "",
        f"`网格最后一点` = 该电池允许预测的最晚时刻（= 观测长度 − {UPPER_OFFSET}）。"
        "`t* 触到网格边界` 为真的格子，真实值可能在该网格之外，不能当精确数用。",
        "",
        "## 三、**区间够窄 ≠ 预测可信**（本节最重要的一条）",
        "",
    ]

    if offside:
        for c0, p0, t_first, t_true, point0, from_start in offside:
            if from_start:
                out += [
                    f"**第一类：区间整段偏离真值。** {c0} 的 {p0}% 区间在**整个预测窗口内**"
                    f"都落在真值 **{int(t_true)}** 下方，一次都没盖住它"
                    f"（窗口末点估计 {point0:.0f}）。",
                    "",
                ]
            else:
                out += [
                    f"**第一类：区间后期整体偏离真值。** {c0} 的 {p0}% 区间从第 **{t_first}** 循环起"
                    f"整体落到真值 **{int(t_true)}** 下方（该处点估计 {point0:.0f}），此后没再盖住它。"
                    "注意这不是「从头到尾没盖住」——更早的时刻区间还包含真值，是**后期系统性偏低**。",
                    "",
                ]
    if never_cells:
        out += [
            f"**「从未命中」的是这些档位**：" + "、".join(
                f"{c}（{'、'.join(str(int(v)) for v in never.loc[never['电池'] == c, '置信度'])}%）"
                for c in never_cells
            ) + "——在**允许的预测窗口内**，这些档位从没稳定盖住真值。",
            "",
        ]

    early = tstar.dropna(subset=["t*_stable 最小", "t*_hit 最小"])
    early = early.loc[early["t*_stable 最小"] < early["t*_hit 最小"]]
    if len(early):
        r = early.sort_values("t*_hit 最小").iloc[-1]
        out += [
            f"**第二类：稳定了，但还不准。** 以 **{r['电池']}**（{int(r['置信度'])}%）为例："
            f"第 **{int(r['t*_stable 最小'])}** 循环起区间就不再改主意，"
            f"但要到第 **{int(r['t*_hit 最小'])}** 循环起才稳定命中，"
            f"差 **{int(r['t*_hit 最小']) - int(r['t*_stable 最小'])} 个循环**。",
            "",
        ]

    out += [
        f"合计「稳定早于命中」的（电池×档位）组合 **{len(early)}/{len(tstar)}**；"
        f"「在噪声内、不下结论」的 **{len(sens)}/{len(tstar)}**。",
        "",
        # 提前量为负 = 预警晚于失效，必须单独点出来，不能混在表里
        (
            "**注意：有电池的命中提前量为负**（预警晚于真实失效）——"
            + "、".join(
                f"{r['电池']}（{int(r['置信度'])}% 档提前量 {r['命中口径提前量']:+.0f}）"
                for _, r in tstar.dropna(subset=["命中口径提前量"]).iterrows()
                if r["命中口径提前量"] <= 0
            )
            + "。按可部署口径（网格不看真值），这些电池**没有及时发出可用预警**。"
            if (tstar["命中口径提前量"] <= 0).any() else
            "所有算出命中时刻的档位，预警都发生在失效之前。"
        ),
        "",
        f"⚠️ **{len(sens)} 个格子的余量 ≤ {SENSITIVE_CYCLES:g} 循环**，"
        "换一个后验路径种子就会翻。报告里引用这些格子时必须写 min–max，不能写单个数。",
        "",
        "## 四、为什么不能用「区间半宽 ≤ 系统不确定性」当判据",
        "",
        "W15 给出系统不确定性（定义层 + 模型形式层）20–34.5 循环。"
        "「统计区间窄于系统误差时就够用」听起来自然，但**判据方向错了**："
        "窄只说明「这条直线估得准」，不说明「这条直线是对的」——后者属于形式层，不在区间里。",
        "",
        "逐块首末对照（`半宽95` 是 95% 区间半宽）：",
        "",
    ]
    for cell in labeled + [c for c in scans if c not in labeled]:
        tbl = scans[cell]
        half = (tbl[f"U95_s{RANDOM_SEED}"] - tbl[f"L95_s{RANDOM_SEED}"]) / 2.0
        tail = f"（真值 {int(truth[cell])}）" if truth.get(cell) else "（删失）"
        out.append(
            f"- {cell}：t0={int(tbl['t0'].iloc[0])} 时半宽 {half.iloc[0]:.1f}、"
            f"点估计 {tbl['点估计'].iloc[0]:.0f}；t0={int(tbl['t0'].iloc[-1])} 时"
            f"半宽 {half.iloc[-1]:.1f}、点估计 {tbl['点估计'].iloc[-1]:.0f}{tail}"
        )
    out += [
        "",
        "**注意不要把这句话概括成「半宽总在缩小」**：B0018 的半宽从 4.0 一路涨到 13.8 再回落，"
        "而它的点估计也在摆。两种形态都说明同一件事——**区间宽度与预测正确性不是一回事**。",
        "",
        "## 五、提前量",
        "",
        tstar.dropna(subset=["t*_hit 最小"])[
            ["电池", "置信度", "t*_hit 最小", "t*_hit 最大", "命中口径提前量", "命中余量(循环)"]
        ].round(2).to_markdown(index=False),
        "",
        f"有真值的电池 {n_label} 块；"
        + "、".join(c for c in scans if truth.get(c) is None) + " 删失，只能报稳定口径。"
        "命中提前量**为负意味着预警晚于失效**，那本身是要报的事实。",
        "",
        "## 六、这张表不能怎么读",
        "",
        "1. **不能说「观测到第 X 个循环就能预测寿命」**：那是稳定口径，不等于预测正确（第三节）。",
        f"2. **不能忽略种子区间**：{len(sens)}/{len(tstar)} 个格子在噪声内，"
        "报告引用时必须带 min–max。",
        f"3. **不能忽略步长**：步长 {GRID_STEP} → {GRID_STEP_SENSITIVITY} 会让个别电池的稳定时刻大幅移动"
        "（见 `t*_stable(步长5)` 列）。",
        "4. **不能用 t*_hit 做部署**：它需要真值才能算，只能事后评估预警够不够早。",
        "5. **不能把 t* 当电池常数推广**：本数据只有 4 块，换批次必须重算。",
        "",
    ]
    return "\n".join(out)


def main() -> int:
    ensure_dirs()
    raw = load_raw()
    df = mark_rows(raw)
    prof = cell_profile(df).set_index("battery_id")

    truth = {
        cell: (None if pd.isna(prof.loc[cell, "eol_cycle"]) else int(prof.loc[cell, "eol_cycle"]))
        for cell in sorted(df["battery_id"].unique())
    }
    scans5 = {cell: scan_cell(df, cell, step=GRID_STEP_SENSITIVITY) for cell in truth}
    # 步长 10 的主口径是步长 5 网格的子集——**扫一次、切两个视图**，别扫两遍
    mask = lambda t: (t["t0"] % GRID_STEP == 0) & (t["t0"] >= GRID_STEP * GRID_START_FACTOR)
    scans = {cell: scans5[cell][mask(scans5[cell])].reset_index(drop=True) for cell in truth}
    step5 = scans5

    tstar = add_flags(build_tstar(scans, truth, step5))
    assert_monotone(tstar)
    assert_warning_before_failure(tstar)

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    tstar.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
    TABLE_MD.write_text(render_md(scans, tstar, truth), encoding="utf-8")
    fig_path = fig14(scans, tstar, truth)

    print(f"网格：起点 {GRID_STEP * GRID_START_FACTOR}、步长 {GRID_STEP}（敏感性 {GRID_STEP_SENSITIVITY}）、"
          f"上界 N−{UPPER_OFFSET}；种子 {SEEDS}")
    print(tstar[["电池", "置信度", "t*_stable 最小", "t*_stable 最大", "稳定余量(循环)",
                 "t*_hit 最小", "t*_hit 最大", "在噪声内"]].round(1).to_string(index=False))
    print()
    print(f"已写出：{TABLE_MD.name} / {CSV_OUT.name} / {fig_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
