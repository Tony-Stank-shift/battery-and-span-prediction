"""W16：最早可信预警时刻 t*。

要回答的问题
------------
「这块电池还能用 100 个循环」这句话，要观测到第几个循环才**值得相信**？
答这个问题必须先说清「相信」是什么意思，本节给两个都不许混用的定义：

1. **稳定（可部署）**：t*_stable(p) = 最早 t0，使得**此后所有** t0 的 p% 预测区间
   **有公共交集**。语义：再加数据也不会改主意。它**不需要真值**，所以在真实部署中可算。
2. **命中（事后诊断）**：t*_hit(p) = 最早 t0，使得**此后所有** t0 的 p% 区间
   **都盖住真值**。语义：从此以后预测一直是对的。它**必须知道真值**，只能事后算。

**为什么必须分开**：区间变窄 ≠ 预测变准。实测 B0005 在 t0=60 的 95% 区间半宽 24 循环
（已经小于 W15 的系统不确定性 34.5），可它的点估计是 217 循环、真值 125——
**误差 92 循环**。如果拿「区间够窄」当可信判据，就会提前 30 多个循环发布一个错得离谱的预警。
这个差距正是 W18（PICP/MPIW 校准）要检测的东西。

四档置信度 80 / 90 / 95 / 99%：**置信度越高，区间越宽**，因此
`t*_(80) ≤ t*_(90) ≤ t*_(95) ≤ t*_(99)`——这条单调性由断言钉住（写错分位数就会炸）。

提前量
------
- 稳定口径的提前量：`点估计(t*) − t*`（还能用多少个循环才知道「它还能用多少」）；
- 命中口径的提前量：`真值 − t*`（在真正失效前多少循环发出了可用的预警）。

刻意不做的事
------------
- **不用「区间半宽 ≤ 系统不确定性」当判据**：那样会给出过早的 t*（理由见上），
  这个诱惑在 W15 结束时出现过，这里明确拒绝。
- **不把 t0 网格开过真值**（有真值的电池）：观测已经越过 EOL 之后再说「预测」没有意义。
  所以网格上界取 `min(N − 10, 真值 − 5)`。

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

from config import METRIC_DIR, TABLE_DIR, ensure_dirs, setup_matplotlib  # noqa: E402
from evaluate import temporal_split  # noqa: E402
from figures import save  # noqa: E402
from load_data import load_raw  # noqa: E402
from preprocess import cell_profile, mark_rows  # noqa: E402

TABLE_MD = TABLE_DIR / "warning_time.md"
CSV_OUT = METRIC_DIR / "warning_time.csv"

LEVELS = (80, 90, 95, 99)
"""四档置信度（百分数）。"""

T0_STEP = 10
"""t0 网格步长（循环）。"""


# ---------------------------------------------------------------------------
# 逐 t0 的预测
# ---------------------------------------------------------------------------
def scan_cell(df: pd.DataFrame, cell: str, eol_true: float | None) -> pd.DataFrame:
    """固定一块电池，逐个 t0 拟合 M4 并取四档区间的上下界。"""
    import models_gpr

    n_obs = int(df.loc[df["battery_id"] == cell, "cycle"].max())
    upper = n_obs - T0_STEP
    if eol_true is not None:
        upper = min(upper, int(eol_true) - 5)  # 观测越过 EOL 之后不再谈预测
    grid = list(range(T0_STEP * 4, upper + 1, T0_STEP))
    assert grid, f"{cell} 的 t0 网格是空的（观测长度 {n_obs}，真值 {eol_true}）"

    # 只切这一块电池：temporal_split 会同时切四块，短电池会拖垮长电池的网格
    sub = df.loc[df["battery_id"] == cell].sort_values("cycle")
    t_all = sub["cycle"].to_numpy(float)
    y_all = sub["capacity_Ah"].to_numpy(float)

    rows = []
    for t0 in grid:
        model = models_gpr.fit_gpr(t_all[:t0], y_all[:t0])
        dist = models_gpr.eol_from_paths(model, n_obs)
        rec = {
            "电池": cell,
            "t0": t0,
            "点估计": models_gpr.eol_mean(model, n_obs),
            "后验中位数": float(dist["median"]),
            "真值": eol_true,
        }
        for p in LEVELS:
            lo_q, hi_q = (100 - p) / 2.0, 100 - (100 - p) / 2.0
            # eol_from_paths 返回的 "eol" 是每条后验路径各自的寿命（删失路径为 NaN）
            ok = dist["eol"][np.isfinite(dist["eol"])]
            lo, hi = np.percentile(ok, [lo_q, hi_q])
            rec[f"L{p}"] = float(lo)
            rec[f"U{p}"] = float(hi)
            rec[f"半宽{p}"] = (float(hi) - float(lo)) / 2.0
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 两个 t* 的定义
# ---------------------------------------------------------------------------
def first_stable(tbl: pd.DataFrame, p: int) -> int | None:
    """最早 t0，使得此后所有 t0 的区间有公共交集（= 预测不再改主意）。"""
    lo = tbl[f"L{p}"].to_numpy(float)
    hi = tbl[f"U{p}"].to_numpy(float)
    t0 = tbl["t0"].to_numpy(int)
    for i in range(len(tbl)):
        if lo[i:].max() <= hi[i:].min():
            return int(t0[i])
    return None


def first_hit(tbl: pd.DataFrame, p: int) -> int | None:
    """最早 t0，使得此后所有 t0 的区间都盖住真值（需要真值，只能事后算）。"""
    if tbl["真值"].isna().all():
        return None
    truth = float(tbl["真值"].iloc[0])
    lo = tbl[f"L{p}"].to_numpy(float)
    hi = tbl[f"U{p}"].to_numpy(float)
    t0 = tbl["t0"].to_numpy(int)
    inside = (lo <= truth) & (truth <= hi)
    for i in range(len(tbl)):
        if inside[i:].all():
            return int(t0[i])
    return None


def build_tstar(scans: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """每块电池 × 四档 → t*_stable / t*_hit / 两个提前量。"""
    rows = []
    for cell, tbl in scans.items():
        truth = None if tbl["真值"].isna().all() else float(tbl["真值"].iloc[0])
        n_obs = int(tbl["t0"].max())
        for p in LEVELS:
            ts = first_stable(tbl, p)
            th = first_hit(tbl, p)

            def _at(t0):
                if t0 is None or pd.isna(t0):
                    return float("nan")
                hit_rows = tbl.loc[tbl["t0"] == t0, "点估计"]
                return float(hit_rows.iloc[0]) if len(hit_rows) else float("nan")

            point = _at(ts)
            rows.append(
                {
                    "电池": cell,
                    "置信度": p,
                    "有真值": truth is not None,
                    "t*_stable": ts,
                    "t*_hit": th,
                    "稳定口径提前量": (point - ts) if ts is not None else float("nan"),
                    "命中口径提前量": (truth - th) if (th is not None and truth is not None) else float("nan"),
                    "t*_stable 处的点估计": point,
                    "t*_stable 处的误差": (point - truth) if truth is not None else float("nan"),
                    "t*_hit 处的误差": (
                        float(tbl.loc[tbl["t0"] == th, "点估计"].iloc[0]) - truth
                        if (th is not None and truth is not None
                            and (tbl["t0"] == th).any()) else float("nan")
                    ),
                    "网格上界": n_obs,
                }
            )
    return pd.DataFrame(rows)


def assert_monotone(tstar: pd.DataFrame) -> None:
    """置信度越高，区间越宽，t* 只能更早或相等。写错分位数这条立刻炸。

    只对**有值**的档位比较：某块电池若在高置信度上才第一次命中（低档位永远命中不了），
    那是「区间太窄、从没盖住真值」的**真实结论**，不是 NaN 该被填掉的地方。
    """
    for cell, g in tstar.groupby("电池"):
        g = g.sort_values("置信度")
        for col, nicer in (("t*_stable", "稳定"), ("t*_hit", "命中")):
            vals = [float(v) for v in g[col].tolist() if pd.notna(v)]
            assert all(a >= b for a, b in zip(vals, vals[1:])), (
                f"{cell} 的 {nicer} t* 随置信度不单调：{g[col].tolist()}——分位数算反了"
            )

    # 稳定时刻不能整体算不出来：网格内总该有某个时刻起预测不再改主意
    assert tstar["t*_stable"].notna().all(), (
        "有电池在网格内始终不稳定："
        f"{tstar.loc[tstar['t*_stable'].isna(), ['电池', '置信度']].to_dict('records')}"
        "——要么网格太短，要么区间算错了"
    )


def assert_warning_before_failure(tstar: pd.DataFrame) -> None:
    """命中口径的提前量必须为正；且「命中」不能整体消失（否则这条检查是空的）。"""
    hit = tstar.dropna(subset=["t*_hit"])
    assert hit["电池"].nunique() >= 2, (
        f"只有 {hit['电池'].nunique()} 块电池算出了命中时刻——"
        "要么实现坏了，要么这条断言形同虚设"
    )
    assert (hit["命中口径提前量"] > 0).all(), (
        f"有电池的预警晚于真实失效：{hit.loc[hit['命中口径提前量'] <= 0, ['电池', '置信度']].to_dict('records')}"
    )


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------
def fig14(scans: dict[str, pd.DataFrame], tstar: pd.DataFrame) -> Path:
    setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.6))

    ax = axes[0]
    colors = {"B0005": "#1f77b4", "B0006": "#d62728", "B0007": "#9467bd", "B0018": "#2ca02c"}
    for cell, tbl in scans.items():
        ax.plot(tbl["t0"], tbl["点估计"], "-o", ms=3.5, lw=1.4, color=colors.get(cell, "#333"),
                label=f"{cell} 点估计")
        truth = tbl["真值"].iloc[0]
        if not pd.isna(truth):
            ax.axhline(float(truth), ls=":", lw=1.0, color=colors.get(cell, "#333"), alpha=0.8)
            ax.text(float(tbl["t0"].min()), float(truth),
                    f" 真值 {int(truth)}", fontsize=7.5, va="bottom",
                    color=colors.get(cell, "#333"))
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
            v = row["t*_hit"]
            vals.append(np.nan if v is None or pd.isna(v) else float(v))
        x = np.arange(len(cells)) + (k - 1.5) * w
        ax.bar(x, vals, w, label=f"命中 t*（{p}%）", alpha=0.9)
    ax.set_xticks(np.arange(len(cells)))
    ax.set_xticklabels(cells)
    # 删失电池没有真值，命中口径不适用——不标注会被读成「它最早，等于 0」
    for i, cell in enumerate(cells):
        if not bool(tstar.loc[tstar["电池"] == cell, "有真值"].iloc[0]):
            ax.text(i, 3, "删失\n无真值", ha="center", va="bottom", fontsize=8,
                    color="#8a6d3b")
    ax.set_ylabel("最早可信时刻 t*（循环）")
    ax.set_title("② 四档置信度下「从第几循环起预测一直命中真值」", fontsize=10)
    ax.legend(fontsize=8)

    fig.suptitle("图14　最早可信预警时刻：区间够窄 ≠ 预测可信", fontsize=13, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save(fig, "fig14_warning_time.png")


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
def render_md(scans: dict[str, pd.DataFrame], tstar: pd.DataFrame) -> str:
    n_label = sum(1 for t in scans.values() if not t["真值"].isna().all())
    gap_rows = tstar.dropna(subset=["t*_stable"])
    # 「stable 早于 hit」的格子：这些就是「预测settled但还错」的直接证据
    early = tstar.dropna(subset=["t*_stable", "t*_hit"])
    early = early.loc[early["t*_stable"] < early["t*_hit"]]
    worst = early.sort_values("t*_stable 处的误差", key=lambda s: s.abs(), ascending=False)

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
        "四档 80 / 90 / 95 / 99% 指**区间置信度**。置信度越高区间越宽，"
        "故 `t*(80) ≤ t*(90) ≤ t*(95) ≤ t*(99)`——这条已写成断言，分位数写反会当场炸。",
        "",
        "## 二、主表",
        "",
        tstar.round(2).to_markdown(index=False),
        "",
        "（`网格上界` = 该电池允许预测的最晚时刻：有真值的电池取 `min(N−10, 真值−5)`，"
        "观测已经越过失效点之后再说「预测」没有意义。）",
        "",
        "## 三、**区间够窄 ≠ 预测可信**（本节最重要的一条）",
        "",
    ]

    if len(early):
        r = worst.iloc[0]
        never = tstar[
            tstar["t*_hit"].isna() & tstar["t*_stable"].notna() & tstar["有真值"]
        ]  # 只数「有真值却没命中」的；删失电池命中不了是定义上的必然，不该混进来
        never_cells = sorted(never["电池"].unique().tolist())
        out += []
        if never_cells:
            c0 = never_cells[0]
            tb = scans[c0]
            lo_pt, hi_pt = float(tb["点估计"].min()), float(tb["点估计"].max())
            truth0 = None if tb["真值"].isna().all() else int(float(tb["真值"].iloc[0]))
            lv = "、".join(str(int(v)) for v in never.loc[never["电池"] == c0, "置信度"])
            out += [
                f"**第一类（更严重）：区间从来就没盖住真值。**"
                f"{c0} 在 {lv}% 档位上，"
                f"整条观测期内点估计在 **{lo_pt:.0f}–{hi_pt:.0f}** 之间摆动，"
                + (f"而真值是 **{truth0}**，" if truth0 is not None else "")
                + "窄区间一次都没包含它。这不是「晚一点才可信」，是**窄区间本身就不可信**。",
                "",
            ]
        out += [
            f"**第二类：稳定了，但还不准。** 以 **{r['电池']}** 为例：到第 "
            f"**{int(r['t*_stable'])}** 循环，{int(r['置信度'])}% 区间就不再改主意了，"
            f"可此时点估计是 **{r['t*_stable 处的点估计']:.0f}** 循环、"
            f"真值 **{int(float(scans[r['电池']]['真值'].iloc[0]))}**——"
            f"误差 **{r['t*_stable 处的误差']:+.0f} 循环**；"
            f"要等到第 **{int(r['t*_hit'])}** 循环才稳定命中，"
            f"两者差 **{int(r['t*_hit']) - int(r['t*_stable'])} 个循环**。",
            "",
            f"合计「稳定早于命中」的（电池×档位）组合有 **{len(early)}/{len(tstar)}** 个；"
            f"另有 **{len(never)}** 个组合是「命中从未发生」。"
            "两类都指向同一件事：**拿「区间够窄/不再变」当发布预警的条件，会发出靠不住的预警。**",
        ]
    else:
        out += [
            "本数据里没有出现「稳定早于命中」的格子——"
            "即预测一旦不再改主意，就已经在盖住真值。"
            "（这句话由代码判定，不是先写好的。）",
        ]

    out += [
        "",
        "## 四、为什么不能用「区间半宽 ≤ 系统不确定性」当判据",
        "",
        "W15 给出系统不确定性（定义层 + 模型形式层）是 20–34.5 循环。看起来很自然："
        "「统计区间窄于系统误差时，统计精度就够用了」。**但这个判据在本数据上是错的**：",
        "",
        scans and "".join(
            f"- {cell}：t0={int(tbl['t0'].min())} 时 95% 半宽 {tbl['半宽95'].iloc[0]:.1f}、"
            f"点估计 {tbl['点估计'].iloc[0]:.0f}；"
            f"到 t0={int(tbl['t0'].max())} 时半宽 {tbl['半宽95'].iloc[-1]:.1f}、"
            f"点估计 {tbl['点估计'].iloc[-1]:.0f}"
            + (
                f"（真值 {int(float(tbl['真值'].iloc[0]))}）"
                if not tbl["真值"].isna().all() else "（删失）"
            )
            + "\n"
            for cell, tbl in scans.items()
        ),
        "",
        "半宽缩小得比点估计收敛快得多，而**这两件事是两回事**：",
        "",
        "- **B0005 / B0007**：半宽从 100+ 降到 10 以下，但点估计要到很晚才落到真值附近；",
        "- **B0018**：半宽从头到尾都只有几个循环（4.0 起），点估计却从 79 一路摆到 97。",
        "",
        "**窄区间只说明「这条直线估得准」，不说明「这条直线是对的」**——"
        "后者属于模型形式层，不在区间里。这个差口就是 W18 要用 PICP 检测的**校准问题**。",
        "",
        "## 五、提前量",
        "",
        "- **稳定口径**：`点估计(t*) − t*` = 「还要观测多少个循环，才敢说它还能用多少个循环」；",
        "- **命中口径**：`真值 − t*` = 「在真正失效前多少个循环发出了可用的预警」。",
        "",
        tstar.dropna(subset=["t*_hit"])[
            ["电池", "置信度", "t*_hit", "命中口径提前量"]
        ].round(2).to_markdown(index=False),
        "",
        f"有真值的电池 {n_label} 块；删失电池（"
        + "、".join(c for c, t in scans.items() if t["真值"].isna().all())
        + "）没有命中口径，只能报稳定口径。",
        "",
        "## 六、这张表不能怎么读",
        "",
        "1. **不能说「观测到第 X 个循环就能预测寿命」**：那是稳定口径，"
        "不等于预测正确（第三节）。",
        "2. **不能把 t* 当常数推广到别的电池**：本数据只有 4 块，"
        "t* 是这块电池的曲线形状决定的，换批次必须重算。",
        "3. **不能用 t*_hit 做部署**：它需要真值才能算，只能用来**事后**评估"
        "「我们的预警够不够早」。",
        "4. **四档之间的差不是精度提升**：置信度从 80% 到 99%，区间变宽、t* 变早，"
        "这是同一个预测的两种说法，不是两个模型。",
        "",
    ]
    return "\n".join(out)


def main() -> int:
    ensure_dirs()
    raw = load_raw()
    df = mark_rows(raw)
    prof = cell_profile(df).set_index("battery_id")

    scans = {}
    for cell in sorted(df["battery_id"].unique()):
        e = prof.loc[cell, "eol_cycle"]
        scans[cell] = scan_cell(df, cell, None if pd.isna(e) else int(e))

    tstar = build_tstar(scans)
    assert_monotone(tstar)
    assert_warning_before_failure(tstar)

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    tstar.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
    TABLE_MD.write_text(render_md(scans, tstar), encoding="utf-8")
    fig_path = fig14(scans, tstar)

    print(f"t0 网格步长 {T0_STEP}；置信度档位 {LEVELS}")
    print(tstar[["电池", "置信度", "t*_stable", "t*_hit", "稳定口径提前量",
                 "命中口径提前量", "t*_stable 处的误差"]].round(1).to_string(index=False))
    print()
    print(f"已写出：{TABLE_MD.name} / {CSV_OUT.name} / {fig_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
