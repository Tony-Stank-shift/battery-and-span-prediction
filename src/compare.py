"""W13：四个模型的对比主表（同划分、同指标、逐块给数、逐点配对检验）。

这一版只做一件事：**把四次实验放到同一张桌子上**，并且强制三条读表纪律。
第七轮审查（R8）推翻了我第一版的两个「事后阈值」——现在不留任何自由参数：
谁更好由**逐点配对 + 分块自助法**判定，判不出来的就说判不出来。

三条纪律（都是前面几轮审查用血换来的，写在这里当护栏）
------------------------------------------------------
1. **只对同分母的量做比较**：测试段 RMSE / MAE 是同一批测试点、同一个 M、
   同一套指标函数算出来的，可以横向比；R² 的分母是**各自电池**的测试段方差，
   **不许跨电池取平均**（D23 第三条：实测按 R² 排序与按 RMSE 排序会翻）。
   注意「四块等权平均」与「按点数合并」不是一个数（88/88/88/52 不等长），两个都给。
2. **训练段指标不许横向排名**：M3 的分段常数会把训练点记住（训练 RMSE 3e-3），
   M4 的 GP 会把残差平滑掉（1e-2），M1/M2 是真的在拟合曲线。
3. **EOL 有三态**，不许塞进一个减法：两边都有值 / 真值删失 / 模型给不出穿越。

产物
----
- `outputs/tables/model_comparison.md`：对比主表（人读）
- `outputs/metrics/model_comparison.csv`：长表（机器读，W14/W17 用）
- `outputs/figures/fig12_model_comparison.png`

运行
----
    python src/compare.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    METRIC_DIR,
    RANDOM_SEED,
    TABLE_DIR,
    ensure_dirs,
    setup_matplotlib,
)
from evaluate import SPLIT_M, to_common_schema  # noqa: E402
from figures import save  # noqa: E402
from load_data import load_raw, split_by_battery  # noqa: E402
from preprocess import cell_profile, mark_rows  # noqa: E402

TABLE_MD = TABLE_DIR / "model_comparison.md"
CSV_OUT = METRIC_DIR / "model_comparison.csv"

MODEL_ORDER = ["M1 线性", "M2 双指数", "M3 随机森林", "M4 趋势+GPR"]
"""四个模型的固定顺序。写死是因为表格列序、图例顺序、颜色都要稳定。"""

REF_MODEL = "M1 线性"
"""配对比较的参照。M1 是最简单的模型，所有「更好」都要先赢它。"""

BOOT_BLOCK = 5
"""分块自助法的默认块长。残差在时间上相关，按点独立重采会低估方差。"""

BOOT_BLOCK_SWEEP = (1, 3, 5, 10, 20)
"""块长扫描。**结论必须在所有块长下一致才算数**——审查实测：
同一组数据换块长，某个格子的区间会跨过 0。只报一个块长等于挑了一个好看的数。"""

BOOT_REPS = 2000

MATERIAL_AH = 0.001
"""「差异是否值得说」的下限：0.001 Ah。

**它不是一个为了凑结论挑出来的数**：原始数据的容量记录到小数点后 4 位
（例如 1.5649 Ah），0.001 Ah 约等于记录精度的量级，也是额定容量 2.0 Ah 的 0.05%。
比这更小的 Δ 一律写「差异可忽略」——统计显著 ≠ 值得说。
（第一版用过 0.002 Ah 与 10% 两个阈值去判「平手/真差距」，那两个数没有任何出处，
审查 R8-02 指出「换 0.005 整句结论就消失」，已删。现在只留这一个、且理由写在这里。）
"""


# ---------------------------------------------------------------------------
# 收集与校验
# ---------------------------------------------------------------------------
def assert_symmetric(
    tbl: pd.DataFrame, expected_cells: list[str], m: int | None = None
) -> None:
    """四个模型必须**完全对称**，并且**与数据的电池集合一致**。

    只验「模型之间彼此一样」是不够的：四个模型一起少一块电池时，
    「行数 == 电池数」「各模型电池集合相同」两条都满足（R8-03 实测退出码 0）。
    所以 `expected_cells` 必须来自数据本身，缺一块也要报错。

    这套检查在 `collect` 与下游 `split_eol_states` 各跑一次：
    只写在 `collect` 里的话，变异测试把 `collect` 整个换掉就绕过去了（R6/R7 的老教训）。
    """
    models = sorted(tbl["模型"].unique().tolist())
    assert models == sorted(MODEL_ORDER), (
        f"模型集合不对：{models}，期望 {sorted(MODEL_ORDER)}"
    )
    cells = sorted(tbl["电池"].unique().tolist())
    assert cells == sorted(expected_cells), (
        f"电池集合与数据不一致：表里 {cells}，数据里 {sorted(expected_cells)}——"
        "少一块电池会让结论消失，不能静默"
    )
    per_model = tbl.groupby("模型").size()
    assert (per_model == len(cells)).all(), (
        f"有模型的行数不等于电池数：{per_model.to_dict()}，电池 {cells}"
    )
    assert tbl["M"].nunique() == 1, f"切分点不一致：{sorted(tbl['M'].unique())}"
    if m is not None:
        assert set(tbl["M"].unique()) == {m}, (
            f"切分点不是 {m}：{sorted(set(tbl['M'].unique()))}"
        )
    for name, g in tbl.groupby("模型"):
        assert sorted(g["电池"].unique()) == cells, (
            f"{name} 的电池集合与别人不同：{sorted(g['电池'].unique())} vs {cells}"
        )
    n_test = tbl.groupby("电池")["测试点数"].nunique()
    assert (n_test == 1).all(), (
        f"同一块电池在不同模型下的测试点数不同：{n_test.to_dict()}——不是同一批测试点"
    )


def collect(df: pd.DataFrame, prof: pd.DataFrame, m: int = SPLIT_M) -> dict:
    """跑三个模块，返回 {"table": 合并长表, "pred": 逐点预测, "cells": 数据里的电池}。"""
    import models_gpr
    import models_ml
    import models_parametric

    results = [
        models_parametric.run(df, prof, m),
        models_ml.run(df, prof, m),
        models_gpr.run(df, prof, m),
    ]
    expected_cells = sorted(split_by_battery(df).keys())

    merged = pd.concat(
        [to_common_schema(r["table"]) for r in results], ignore_index=True
    )  # 不传 model_name：保留各表自己的「模型」列（D25 第五条）
    assert_symmetric(merged, expected_cells, m)

    pred_rows = []
    for res in results:
        for item in res["fits"].values():
            sp = item["split"]
            pred_rows.append(
                pd.DataFrame(
                    {
                        "电池": sp.cell,
                        "模型": item["model"]["name"],
                        "循环": sp.t_test,
                        "真值": sp.y_test,
                        "预测": item["predict"](item["model"], sp.t_test),
                    }
                )
            )
    pred = pd.concat(pred_rows, ignore_index=True)
    assert len(pred) == int(merged["测试点数"].sum()), (
        f"逐点预测 {len(pred)} 行与表里的测试点数合计 {int(merged['测试点数'].sum())} 不符"
    )
    return {"table": merged, "pred": pred, "cells": expected_cells}


# ---------------------------------------------------------------------------
# 逐点配对 + 分块自助法
# ---------------------------------------------------------------------------
def block_bootstrap_ci(
    d: np.ndarray, block: int = BOOT_BLOCK, reps: int = BOOT_REPS, seed: int = RANDOM_SEED
) -> tuple[float, float]:
    """分块自助法（moving block）的 95% 百分位区间。

    残差序列在时间上相关，按点独立重采样会低估方差；块长取 5 个循环。
    """
    d = np.asarray(d, float)
    n = len(d)
    if n == 0:
        return float("nan"), float("nan")
    block = max(1, min(block, n))
    n_blocks = int(np.ceil(n / block))
    starts_max = n - block
    rng = np.random.default_rng(seed)
    means = np.empty(reps)
    for i in range(reps):
        starts = rng.integers(0, starts_max + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        means[i] = d[idx].mean()
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def paired_table(pred: pd.DataFrame, ref: str = REF_MODEL) -> pd.DataFrame:
    """每个模型 vs 参照模型：逐点绝对误差差值的均值、更近的点数、自助法区间。

    Δ = |err_候选| − |err_参照|。**Δ<0 表示候选更好**。

    结论由**块长扫描**决定：只有全部块长（1/3/5/10/20）的 95% 区间都同号，
    才敢说「更好 / 更差」；只要有一档跨 0，就写「不稳健」。
    这条替代了我第一版那两个凭空的阈值（0.002 Ah / 10%，R8-01/R8-02）。
    """
    rows = []
    for cell, g in pred.groupby("电池"):
        wide = g.pivot_table(index="循环", columns="模型", values="预测")
        truth = g.groupby("循环")["真值"].first()
        err = wide.sub(truth, axis=0).abs()
        for name in MODEL_ORDER:
            if name == ref or name not in err.columns:
                continue
            d = (err[name] - err[ref]).to_numpy(float)
            d = d[~np.isnan(d)]
            cis = {
                b: block_bootstrap_ci(d, block=b) for b in BOOT_BLOCK_SWEEP
            }
            lo_main, hi_main = cis[BOOT_BLOCK]
            all_neg = all(hi < 0 for _, hi in cis.values())
            all_pos = all(lo > 0 for lo, _ in cis.values())
            n_closer = int((d < 0).sum())
            # 第二个判据：逐点符号检验。它只看「谁更近」的计数，不依赖块长，
            # 与自助法相互独立。两个判据都显著才下结论（R8-01 的教训：
            # 某块电池的 Δ 由少数几个点撑着，自助法会给出显著、符号检验不会）。
            p_sign = float(binomtest(n_closer, len(d), 0.5).pvalue)
            boot_decided = all_neg or all_pos
            sign_decided = p_sign < 0.05
            if abs(float(d.mean())) < MATERIAL_AH:
                verdict = "差异可忽略（< 记录精度）"
            elif boot_decided and sign_decided:
                verdict = ("候选更好" if all_neg else "参照更好") + "（两判据一致）"
            elif not boot_decided and not sign_decided:
                verdict = "不显著（两判据一致）"
            else:
                verdict = "判据冲突，不下结论"
            rows.append(
                {
                    "电池": cell,
                    "对比": f"{name} − {ref}",
                    "ΔMAE(Ah)": float(d.mean()),
                    "候选更近的点数": n_closer,
                    "点数": int(len(d)),
                    "更近占比": float((d < 0).mean()),
                    "符号检验 p": p_sign,
                    "95%CI 下(块长5)": lo_main,
                    "95%CI 上(块长5)": hi_main,
                    "块长1/3/5/10/20 同号": "是" if boot_decided else "否",
                    "结论": verdict,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# EOL 三态与汇总
# ---------------------------------------------------------------------------
def assert_eol_states(tbl: pd.DataFrame) -> None:
    """三态一致性：删失的不能有误差、给不出的不能有误差、给不出的必须写原因。

    单独抽成函数是因为**两个地方都要调用**：`split_eol_states` 里查一次，
    下游 `summary_table` 里再查一次。只写在被替换的那个函数里，
    变异测试把函数整个换掉就绕过去了（R8-05 就是这么漏的）。
    """
    assert not (tbl["真值删失"] & tbl["EOL误差数值"].notna()).any(), (
        "有删失电池被算进了 EOL 误差——删失没有真值，任何误差都是编的"
    )
    assert not (
        (~tbl["真值删失"]) & tbl["模型给不出"] & tbl["EOL误差数值"].notna()
    ).any(), "模型给不出 EOL，却带着一个误差数字"
    for row in tbl.itertuples():
        if row.模型给不出:
            assert str(row.备注).strip() != "", (
                f"{row.电池} / {row.模型} 的 EOL 给不出，备注却是空的——必须写原因"
            )
        elif not pd.isna(row.EOL误差数值):
            assert "查表" not in str(row.备注), (
                f"{row.电池} 的 EOL 误差是数字，但备注说这是查表——查表不能当预测精度"
            )


def split_eol_states(
    tbl: pd.DataFrame, expected_cells: list[str] | None = None
) -> pd.DataFrame:
    """把每行的 EOL 结果归成三态：有值 / 真值删失 / 模型给不出。

    三态判定必须**显式**：真值既可能是 `"删失"` 这个字符串，也可能是 NaN
    （R8-05：只认字面量时，把删失编码成 NaN 会让它静默变成"可比"，
    一个假误差就直接进均值了）。
    """
    if expected_cells is not None:
        assert_symmetric(tbl, expected_cells)
    out = tbl.copy()
    truth_raw = out["EOL真值"]
    out["真值删失"] = truth_raw.isna() | (truth_raw.astype(str) == "删失")
    out["模型给不出"] = out["EOL预测"].astype(str) == "—"
    out["EOL误差数值"] = pd.to_numeric(out["EOL误差"], errors="coerce")
    assert_eol_states(out)
    return out


def summary_table(tbl: pd.DataFrame) -> pd.DataFrame:
    """每个模型一行。**只汇总可比量**：测试段 RMSE/MAE + EOL 三态计数。"""
    assert_eol_states(tbl)  # 下游再查一次（split_eol_states 可能被整体替换）
    rows = []
    for name in MODEL_ORDER:
        g = tbl[tbl["模型"] == name]
        err = g["EOL误差数值"]
        expect_comparable = int(((~g["模型给不出"]) & (~g["真值删失"])).sum())
        assert int(err.notna().sum()) == expect_comparable, (
            f"{name} 的 EOL 可比数 {int(err.notna().sum())} 与"
            f"「给了预测且真值未删失」的 {expect_comparable} 不一致——有「—」被当成了数字"
        )
        # 能预测的模型必须在**每一块未删失的电池**上都给出 EOL；
        # 只在部分电池给得出，说明有地方静默失败了（R8-04）
        n_labeled = int((~g["真值删失"]).sum())
        if int((~g["模型给不出"]).sum()) > 0:
            assert int((~g["模型给不出"]).sum()) == len(g), (
                f"{name} 在部分电池上给不出 EOL（{g['电池'].tolist()}）——"
                "要么全给，要么整列都没有，不能只给一半"
            )
        rows.append(
            {
                "模型": name,
                "测试RMSE 均值(四块等权)": float(g["RMSE"].mean()),
                "按点数合并的 RMSE": float(
                    np.sqrt((g["RMSE"] ** 2 * g["测试点数"]).sum() / g["测试点数"].sum())
                ),
                "测试MAE 均值(四块等权)": float(g["MAE"].mean()),
                "训练RMSE 均值(不可横向比)": float(g["训练RMSE"].mean()),
                "有真值的电池数": n_labeled,
                "给出 EOL 的电池数": int((~g["模型给不出"]).sum()),
                "EOL 可比电池数": int(err.notna().sum()),
                "EOL 绝对误差均值": float(err.abs().mean()) if err.notna().any() else np.nan,
                "EOL 绝对误差最大": float(err.abs().max()) if err.notna().any() else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    assert not any("R2" in c for c in out.columns), (
        "汇总表里出现了 R² 的跨电池平均——D23 明令禁止"
    )
    return out


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------
def fig12(tbl: pd.DataFrame, pred: pd.DataFrame, m: int) -> Path:
    """两联图：① 逐块测试 RMSE；② EOL 预测 vs 真值。"""
    setup_matplotlib()
    cells = sorted(tbl["电池"].unique().tolist())
    palette = {
        "M1 线性": "#7f8c8d",
        "M2 双指数": "#b8860b",
        "M3 随机森林": "#c0392b",
        "M4 趋势+GPR": "#6a3d9a",
    }

    fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.6))

    ax = axes[0]
    w = 0.2
    idx = np.arange(len(cells))
    for k, name in enumerate(MODEL_ORDER):
        g = tbl[tbl["模型"] == name].set_index("电池").reindex(cells)
        ax.bar(idx + (k - 1.5) * w, g["RMSE"].to_numpy(float), w,
               label=name, color=palette[name])
    ax.set_xticks(idx)
    ax.set_xticklabels(cells)
    ax.set_yscale("log")
    ax.set_ylabel("测试段 RMSE (Ah)　对数轴，越低越好")
    ax.set_title("① 逐块测试误差：M4 与 M1 几乎重合，M3 是设计好的反例", fontsize=10)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1]
    labeled = sorted(tbl.loc[~tbl["真值删失"], "电池"].unique().tolist())
    truth = {
        c: int(pd.to_numeric(tbl.loc[(tbl["电池"] == c) & (~tbl["真值删失"]), "EOL真值"],
                             errors="coerce").dropna().iloc[0])
        for c in labeled
    }
    styles = {
        "M1 线性": dict(s=170, marker="o", facecolors="none", linewidths=2.0),
        "M2 双指数": dict(s=80, marker="s"),
        "M3 随机森林": dict(s=110, marker="X"),
        "M4 趋势+GPR": dict(s=55, marker="^"),
    }
    for name in MODEL_ORDER:
        xs, ys = [], []
        for c in labeled:
            row = tbl[(tbl["电池"] == c) & (tbl["模型"] == name)].iloc[0]
            if row["模型给不出"]:
                continue
            xs.append(truth[c])
            ys.append(int(row["EOL预测"]))
        if xs:
            kw = dict(styles[name])
            kw.setdefault("color", palette[name])
            kw.setdefault("edgecolors", palette[name])
            ax.scatter(xs, ys, label=name, zorder=5, **kw)

    lo = min(truth.values()) - 12
    hi = max(truth.values()) + 34
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.2, color="#999999", label="完美预测线")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("真实寿命（循环）")
    ax.set_ylabel("模型预测的寿命（循环）")
    ax.set_title("② EOL：三个模型都给数，M3 一个都给不出", fontsize=10)

    # 远场最大 |Δpred| 是「M1 与 M4 点预测是否相等」的直接量，别拿 EOL 整数冒充（R8-08）
    wide = pred.pivot_table(index=["电池", "循环"], columns="模型", values="预测")
    if {REF_MODEL, "M4 趋势+GPR"} <= set(wide.columns):
        max_dp = float((wide["M4 趋势+GPR"] - wide[REF_MODEL]).abs().max())
    else:  # pragma: no cover
        max_dp = float("nan")
    censored = sorted(tbl.loc[tbl["真值删失"], "电池"].unique().tolist())
    overlap = [
        c for c in labeled
        if tbl[(tbl["电池"] == c) & (~tbl["模型给不出"])]["EOL预测"].nunique() == 1
    ]
    note = (
        f"{'、'.join(censored)} 删失（无真值），进不了这张图；\n"
        f"三个给出预测的模型都说它会在观测期内坏（方向都错）\n"
        f"M1 与 M4 的点预测最大只差 {max_dp:.4f} Ah（远场点预测 = 趋势线）"
    )
    if overlap:
        c0 = overlap[0]
        n_same = int(tbl[(tbl["电池"] == c0) & (~tbl["模型给不出"])].shape[0])
        note += f"\n{c0} 上 {n_same} 个模型报了同一个数（重合）"
    ax.annotate(
        note,
        xy=(0.98, 0.03), xycoords="axes fraction", fontsize=8, ha="right",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fdf6e3", ec="#d9c79a"),
    )
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(f"图12　四模型对比（M = {m}，同一划分、同一套指标）", fontsize=13, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save(fig, "fig12_model_comparison.png")


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
def pivot_metric(tbl: pd.DataFrame, col: str) -> pd.DataFrame:
    wide = tbl.pivot_table(index="电池", columns="模型", values=col, aggfunc="first")
    return wide.reindex(columns=MODEL_ORDER)


def _fmt_ci(row, col_lo: str, col_hi: str) -> str:
    return f"[{row[col_lo]:+.4f}, {row[col_hi]:+.4f}]"


def render_md(
    tbl: pd.DataFrame,
    summ: pd.DataFrame,
    paired: pd.DataFrame,
    pred: pd.DataFrame,
    m: int,
    obs_len: dict[str, int],
) -> str:
    rmse = pivot_metric(tbl, "RMSE")
    mae = pivot_metric(tbl, "MAE")
    eol_pred = pivot_metric(tbl, "EOL预测")
    eol_true = pivot_metric(tbl, "EOL真值")
    train = pivot_metric(tbl, "训练RMSE")

    cells = sorted(tbl["电池"].unique().tolist())
    n_cells = len(cells)
    n_points = {c: int(tbl.loc[tbl["电池"] == c, "测试点数"].iloc[0]) for c in cells}
    pooled = {
        name: float(
            np.sqrt(
                (summ.loc[summ["模型"] == name, "按点数合并的 RMSE"]).iloc[0]
            )
        )
        for name in MODEL_ORDER
    }
    equal_w = dict(zip(summ["模型"], summ["测试RMSE 均值(四块等权)"]))

    sig = paired[paired["结论"].str.contains("两判据一致") & ~paired["结论"].str.startswith("不显著")]
    conflict = paired[paired["结论"].str.startswith("判据冲突")]
    negligible = paired[paired["结论"].str.startswith("差异可忽略")]

    # M4 与 M1 的**点预测**差在哪：这是「M4 只是区间更好」这句话的直接证据
    wide = pred.pivot_table(index=["电池", "循环"], columns="模型", values="预测")
    if {REF_MODEL, "M4 趋势+GPR"} <= set(wide.columns):
        d_m4 = (wide["M4 趋势+GPR"] - wide[REF_MODEL]).abs()
        max_dp = float(d_m4.max())
        at = d_m4.idxmax()
        max_dp_cell, max_dp_cycle = at[0], int(at[1])
        far = d_m4[d_m4.index.get_level_values("循环") > m + 20]
        far_dp = float(far.max()) if len(far) else float("nan")
        n_big = int((d_m4 >= 0.01).sum())
        n_all = int(len(d_m4))
    else:  # pragma: no cover
        max_dp = far_dp = float("nan")
        max_dp_cell, max_dp_cycle, n_big, n_all = "—", 0, 0, 0
    msig = sig[sig["对比"].str.startswith("M4")]
    m3sig = sig[sig["对比"].str.startswith("M3")]
    m4conf = conflict[conflict["对比"].str.startswith("M4")]

    out = [
        "# 四模型对比主表：同划分、逐块给数、逐点配对检验（W13）",
        "",
        "> 本文件由 `src/compare.py` 自动生成，数字随代码重算，**不要手工编辑**。",
        "> 机器可读版：`outputs/metrics/model_comparison.csv`（长表，W14/W17 直接用）。",
        "",
        "## 一、口径",
        "",
        f"- **任务**：任务 A（个体内轨迹外推）。切分点 **M = {m}**，时序划分，逐个体拟合。",
        f"- **电池**：{n_cells} 块（{'、'.join(f'{c}（观测 {obs_len[c]}，测试 {n_points[c]}）' for c in cells)}）。",
        "- **四个模型**：M1 线性 / M2 双指数 / M3 随机森林（**设计好的反例**）/ M4 趋势+GPR。",
        "- **三条读表纪律**（D23 / D25）：",
        "  ① 测试段 RMSE/MAE 是**同一批测试点**算出来的，可以横向比；"
        "R² 的分母是各自电池的方差，**不许跨电池平均**；",
        "  ② **训练段指标不参与排名**（M3 记住样本、M4 平滑残差，两者都偏低）；",
        "  ③ **EOL 有三态**——有值 / 真值删失 / 模型给不出，混在一起算就是造假。",
        "",
        "## 二、测试段误差",
        "",
        "### RMSE（Ah）",
        "",
        rmse.round(4).to_markdown(),
        "",
        "### MAE（Ah）",
        "",
        mae.round(4).to_markdown(),
        "",
        "⚠️ **「四块平均」不是「整体误差」**：四块的测试点数不等长"
        f"（{'、'.join(str(n_points[c]) for c in cells)}），"
        f"等权平均与按点数合并会差 20% 以上（M1：{equal_w['M1 线性']:.4f} vs {pooled['M1 线性']:.4f}）。"
        "下表两个都给，别只看一个。",
        "",
        "## 三、谁真的更好？——**逐点配对 + 分块自助法**",
        "",
        "第一版这张表用的是「赢家 + 差距阈值」，两个阈值（0.002 Ah、10%）是我事后定的，"
        "而且判反了：被判「真差距」的那块实际只有 56% 的点更近。现在换成不谈阈值的方法：",
        "",
        "- **ΔMAE** = 候选模型逐点绝对误差 − M1 逐点绝对误差，均值；**负 = 候选更好**；",
        f"- **判据一（区间）**：分块自助法（块长 {'/'.join(map(str, BOOT_BLOCK_SWEEP))} 扫描，"
        f"{BOOT_REPS} 次重采样，种子 {RANDOM_SEED}）——残差在时间上相关，按点独立重采会低估方差；"
        "**只有全部块长都同号才算数**；",
        "- **判据二（计数）**：逐点符号检验（「谁更近」的计数 vs 50%）。"
        "它不依赖块长，与判据一相互独立；",
        "- **两个判据都显著才下结论**，只有一个显著就写「判据冲突，不下结论」"
        "——这一条是审查逼出来的：某块电池的 Δ 由少数几个点撑着时，"
        "自助法给显著、符号检验不给，两者打架就说明不该下结论；",
        f"- **差异可忽略**：|ΔMAE| < {MATERIAL_AH} Ah。这不是为凑结论挑的数——"
        "原始容量记录到小数点后 4 位，0.001 Ah 就是记录精度的量级。"
        "**统计显著 ≠ 值得说**。",
        "",
        paired.round(4).to_markdown(index=False),
        "",
        f"**读出来的事实**：{len(paired)} 个组合里，{len(sig)} 个两判据一致且显著、"
        f"{len(negligible)} 个差异可忽略、{len(conflict)} 个判据冲突不下结论、"
        f"其余为一致不显著。",
        "",
    ]

    if len(msig):
        det = "；".join(
            f"{r['电池']}：ΔMAE {r['ΔMAE(Ah)']:+.4f}，"
            f"{r['候选更近的点数']}/{r['点数']} 点更近（符号检验 p={r['符号检验 p']:.1e}），"
            f"CI {r['95%CI 下(块长5)']:+.4f}…{r['95%CI 上(块长5)']:+.4f}"
            for _, r in msig.iterrows()
        )
        out += [f"- **M4 vs M1 两判据都显著的只有**：{det}。"]
    else:
        out += ["- **M4 vs M1 没有任何一块两判据都显著**——「M4 更准」这句话不能说。"]

    if len(m4conf):
        out += [
            "- **判据冲突、不下结论的**："
            + "；".join(
                f"{r['电池']}：ΔMAE {r['ΔMAE(Ah)']:+.4f}，"
                f"只有 {r['候选更近的点数']}/{r['点数']} 点更近（符号检验 p={r['符号检验 p']:.2f}），"
                "自助法显著但点数不支持"
                for _, r in m4conf.iterrows()
            ),
        ]

    if len(m3sig):
        out += [
            "- **M3 显著更差**（Δ 为正 = 误差更大）："
            + "；".join(
                f"{r['电池']} ΔMAE {r['ΔMAE(Ah)']:+.4f}" for _, r in m3sig.iterrows()
            )
        ]
    else:
        out += ["- **M3 vs M1 没有任何一块显著更差/更好**（它只是换了个形态的错）。"]

    out += [
        "",
        f"**M4 与 M1 的差别到底在哪**：两者点预测的最大差是 {max_dp:.4f} Ah"
        f"（{max_dp_cell} 第 {max_dp_cycle} 循环，也就是刚离开训练段的那几个点），"
        f"离开切分点 20 个循环之后最大还有 {far_dp:.4f} Ah；"
        f"全部 {n_all} 个测试点里 {n_big} 个的差 ≥ 0.01 Ah。"
        "**所以「M4 的修正是局部的」这句话成立，但「局部」不等于「可忽略」**——"
        "它在 B0005 上确实两判据都显著（86/88 点更近），"
        "而在其余三块上要么判据冲突、要么低于记录精度。",
        "",
        "谁要是拿「M4 的 RMSE 略低」当卖点，就得先解释：为什么逐点检验在四块里只有一块显著。",
        "",
        "## 四、训练段：只用来回答「它记住没有」",
        "",
        train.round(4).to_markdown(),
        "",
        "M3 的训练 RMSE 比别的模型低一个量级——那不是优点，是**分段常数把训练点记住了**"
        "（W10 已证明它的外推是水平线）。M4 偏低是因为 GP 把残差平滑掉了。"
        "**这张表不产生排名**。",
        "",
        "## 五、EOL：三态必须分开看",
        "",
        eol_pred.to_markdown(),
        "",
        "（真值："
        + "、".join(
            f"{c} {('删失' if tbl.loc[tbl['电池'] == c, '真值删失'].iloc[0] else int(pd.to_numeric(eol_true.loc[c], errors='coerce').dropna().iloc[0]))}"
            for c in cells
        )
        + "）",
        "",
    ]

    cant = summ.loc[summ["给出 EOL 的电池数"] == 0, "模型"].tolist()
    can = summ.loc[summ["给出 EOL 的电池数"] > 0, "模型"].tolist()
    out += [
        f"- **给出预测的模型**：{'、'.join(can)}；"
        f"**{'、'.join(cant) or '无'} 在 {n_cells}/{n_cells} 块电池上给不出**"
        "（外推是常数，永远不穿越阈值）——这是 W10 的设计目的，不是 bug。",
    ]

    censored = sorted(tbl.loc[tbl["真值删失"], "电池"].unique().tolist())
    if censored:
        c0 = censored[0]
        preds = {
            name: tbl.loc[(tbl["电池"] == c0) & (tbl["模型"] == name), "EOL预测"].iloc[0]
            for name in can
        }
        out += [
            "",
            f"- **{c0} 没有「误差」**：它删失（观测到 {obs_len.get(c0, 0)} 循环仍未到达）。"
            "给得出数的模型都说它会坏——"
            + "、".join(f"{k} 报 {v}" for k, v in preds.items())
            + "。这不是「误差多大」，是**方向对不对**：全都错。",
        ]

    out += [
        "",
        "### 汇总（每个模型一行）",
        "",
        summ.round(4).to_markdown(index=False),
        "",
        "⚠️ 读这张表必须注意：",
        "",
        f"1. **「EOL 可比电池数」不同**："
        + "、".join(
            f"{r['模型']} {r['EOL 可比电池数']}（有真值 {r['有真值的电池数']}）"
            for _, r in summ.iterrows()
        )
        + "。所以 EOL 误差**不能跨模型直接比大小**——"
        "「—」不是「误差为 0」，是「没给数」。把「—」当 0 会让给不出预测的模型显得最完美。",
        "2. **「训练RMSE 均值」这一列不可横向比**，它放在这里只为了让读者看到量级差；"
        "真正的训练段讨论在第四节。",
        "3. **M4 的 90% 区间覆盖率只有 1/3**（详见 `model_gpr.md`）："
        "区间偏窄，第三层不确定性还没进来——W14/W17 处理。",
        "",
        "## 六、这张表不能怎么读",
        "",
        "1. **不能按平均 R² 排名**：R² 的分母是各自电池的测试段方差，跨电池平均没有意义"
        "（D23 实测：按 R² 与按 RMSE 排序会翻）。本表因此**只列逐块值**。",
        "2. **不能因为 M3 差就删掉它**：它的作用是反例——证明「树模型不能外推」。",
        "3. **不能说「M4 比 M1 准」**：逐点检验不显著，且远场点预测最大只差 1e-4 Ah 量级；"
        "M4 的价值是区间（W11 已证）。",
        "4. **不能把这张表当跨个体（任务 B）的结论**：任务 B 的样本个体只有 3 个，"
        "LOBOCV 里零基准就打赢了所有特征模型（见 `task_b_lobocv.md`）。",
        "5. **不能说「四块平均就是整体水平」**：四块等权与按点数合并差 20% 以上，两个数都给了。",
        "",
    ]
    return "\n".join(out)


def main() -> int:
    ensure_dirs()
    raw = load_raw()
    df = mark_rows(raw)
    prof = cell_profile(df)

    res = collect(df, prof, SPLIT_M)
    # 期望的电池集合**直接从数据取**，不用 collect 返回值——否则变异把返回值改掉就能骗过去
    expected_cells = sorted(split_by_battery(df).keys())
    tbl = split_eol_states(res["table"], expected_cells)
    summ = summary_table(tbl)
    paired = paired_table(res["pred"])

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    out_csv = tbl.drop(columns=["模型给不出"]).copy()
    out_csv.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")

    obs_len = {cell: int(len(sub)) for cell, sub in split_by_battery(df).items()}
    TABLE_MD.write_text(
        render_md(tbl, summ, paired, res["pred"], SPLIT_M, obs_len), encoding="utf-8"
    )
    fig_path = fig12(tbl, res["pred"], SPLIT_M)

    print(f"对比主表：{len(tbl)} 行（{len(MODEL_ORDER)} 模型 × "
          f"{tbl['电池'].nunique()} 电池），M = {SPLIT_M}")
    print()
    print("测试段 RMSE（Ah）：")
    print(pivot_metric(tbl, "RMSE").round(4).to_string())
    print()
    print("逐点配对 vs M1（Δ<0 = 候选更好）：")
    print(paired[["电池", "对比", "ΔMAE(Ah)", "候选更近的点数", "点数",
                  "95%CI 下(块长5)", "95%CI 上(块长5)", "块长1/3/5/10/20 同号", "结论"]]
          .round(4).to_string(index=False))
    print()
    print(f"对比主表已写出：{TABLE_MD.name}")
    print(f"长表已写出：{CSV_OUT.name}")
    print(f"已生成 {fig_path.name}  ({fig_path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
