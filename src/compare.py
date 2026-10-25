"""W13：四个模型的对比主表（同划分、同指标、逐块给数）。

这一版只做一件事：**把四次实验放到同一张桌子上**，并且强制三条读表纪律。

三条纪律（都是前面几轮审查用血换来的，写在这里当护栏）
------------------------------------------------------
1. **只对同分母的量做比较**：测试段 RMSE / MAE 是同一批测试点、同一个 M、
   同一套指标函数算出来的，可以横向比；R² 的分母是**各自电池**的测试段方差，
   **不许跨电池取平均**（D23 第三条：实测按 R² 排序与按 RMSE 排序会翻）。
2. **训练段指标不许横向排名**：M3 的分段常数会把训练点记住（训练 RMSE 3e-3），
   M4 的 GP 会把残差平滑掉（1e-2），M1/M2 是真的在拟合曲线。
   这张表把训练段单独放到一张表里，只用来回答「它记住没有」。
3. **EOL 有三态**，不许塞进一个减法：两边都有值 / 真值删失 / 模型给不出穿越。
   M3 在四块电池上都给不出（这是它的设计目的），所以它的「EOL 可比电池数」是 0，
   **不能因此说它「精度最高」**。

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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    METRIC_DIR,
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

TEST_METRICS = ["RMSE", "MAE", "MAPE", "R2"]
"""测试段指标。**同一个 M、同一批测试点**算出来的，可以横向比。"""


# ---------------------------------------------------------------------------
# 收集与校验
# ---------------------------------------------------------------------------
def assert_symmetric(tbl: pd.DataFrame, m: int | None = None) -> None:
    """四个模型必须**完全对称**：同样的四个模型、同样的电池集合、同样的切分点。

    这些检查放在一个独立函数里、并且在**下游**的 `split_eol_states` 里也调用一次：
    只写在 `collect` 结尾的话，一旦有人把 `collect` 整个换掉（变异测试就是这么干的），
    检查就形同虚设——R6/R7 两轮都吃过这个亏。
    """
    models = sorted(tbl["模型"].unique().tolist())
    assert models == sorted(MODEL_ORDER), (
        f"模型集合不对：{models}，期望 {sorted(MODEL_ORDER)}"
    )
    cells = sorted(tbl["电池"].unique().tolist())
    per_model = tbl.groupby("模型").size()
    assert (per_model == len(cells)).all(), (
        f"有模型的行数不等于电池数：{per_model.to_dict()}，电池 {cells}"
    )
    assert tbl["M"].nunique() == 1, (
        f"切分点不一致：{sorted(tbl['M'].unique())}"
    )
    if m is not None:
        assert set(tbl["M"].unique()) == {m}, (
            f"切分点不是 {m}：{sorted(set(tbl['M'].unique()))}"
        )
    for name, g in tbl.groupby("模型"):
        assert sorted(g["电池"].unique()) == cells, (
            f"{name} 的电池集合与别人不同：{sorted(g['电池'].unique())} vs {cells}"
        )


def collect(df: pd.DataFrame, prof: pd.DataFrame, m: int = SPLIT_M) -> pd.DataFrame:
    """跑三个模块、把四张模型表规范成统一 schema 后纵向拼接。

    校验交给 `assert_symmetric`（同一套检查在下游还会再跑一次）。
    """
    import models_gpr
    import models_ml
    import models_parametric

    raw_tables = [
        models_parametric.run(df, prof, m)["table"],
        models_ml.run(df, prof, m)["table"],
        models_gpr.run(df, prof, m)["table"],
    ]
    merged = pd.concat(
        [to_common_schema(t) for t in raw_tables], ignore_index=True
    )  # 不传 model_name：保留各表自己的「模型」列（D25 第五条）
    assert_symmetric(merged, m)
    return merged


def split_eol_states(tbl: pd.DataFrame) -> pd.DataFrame:
    """把每行的 EOL 结果归成三态：数字 / 删失（无真值）/ 模型给不出穿越。

    三态必须分开统计——把它们混成一个减法，就会得出「M3 误差最小」这种荒谬结论。
    """
    out = tbl.copy()
    assert_symmetric(out)  # 下游再查一次：collect 被整体替换时也不放过
    out["真值删失"] = out["EOL真值"] == "删失"
    out["模型给不出"] = out["EOL预测"] == "—"
    out["EOL误差数值"] = pd.to_numeric(out["EOL误差"], errors="coerce")

    # 每一行「没有数字」时，备注必须解释原因；否则就是静默给不出
    for row in out.itertuples():
        if row.模型给不出 or pd.isna(row.EOL误差数值):
            assert str(row.备注).strip() != "", (
                f"{row.电池} 的 EOL 没有数字，备注却是空的——"
                "「给不出」必须写原因，不能静默"
            )
        else:
            assert "查表" not in str(row.备注), (
                f"{row.电池} 的 EOL 误差是数字，但备注说这是查表——"
                "查表出来的数字不能当预测精度进表"
            )
    return out


def summary_table(tbl: pd.DataFrame) -> pd.DataFrame:
    """每个模型一行。**只汇总可比量**：测试段 RMSE/MAE（同分母）+ EOL 三态计数。"""
    rows = []
    for name in MODEL_ORDER:
        g = tbl[tbl["模型"] == name]
        err = pd.to_numeric(g["EOL误差"], errors="coerce")
        # 「可比」必须精确等于「给了预测 且 真值没删失」的块数。
        # 这条守住最危险的一种错：把「—」当成 0 塞进误差统计，
        # 那样 M3（一块都给不出）会显示成「4 块可比、误差 0」，看着最完美。
        expect_comparable = int(
            ((g["EOL预测"] != "—") & (g["EOL真值"] != "删失")).sum()
        )
        assert int(err.notna().sum()) == expect_comparable, (
            f"{name} 的 EOL 可比数 {int(err.notna().sum())} 与"
            f"「给了预测且真值未删失」的 {expect_comparable} 不一致——"
            "有「—」被当成了数字"
        )
        rows.append(
            {
                "模型": name,
                "测试RMSE 均值": float(g["RMSE"].mean()),
                "测试MAE 均值": float(g["MAE"].mean()),
                "测试RMSE 最好的一块": float(g["RMSE"].min()),
                "训练RMSE 均值": float(g["训练RMSE"].mean()),
                "给出 EOL 的电池数": int((g["EOL预测"] != "—").sum()),
                "EOL 可比电池数": int(err.notna().sum()),
                "EOL 绝对误差均值": float(err.abs().mean()) if err.notna().any() else np.nan,
                "EOL 绝对误差最大": float(err.abs().max()) if err.notna().any() else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    # 格式护栏：R² 的分母是各自电池的方差，跨电池平均没有意义（D23 第三条）
    assert not any("R2" in c for c in out.columns), (
        "汇总表里出现了 R² 的跨电池平均——这是 D23 明令禁止的"
    )
    return out


def win_counts(tbl: pd.DataFrame) -> pd.DataFrame:
    """逐块赢家：每块电池上测试段 RMSE 最小的模型，并给出**与第二名的差距**。

    为什么必须给差距：「赢」可能是 0.0001 Ah 的平手。不写差距，
    「M3 赢了两块」会被读成「反例模型其实不错」，而它在真正的差距处是 3 倍差。
    """
    rows = []
    for cell, g in tbl.groupby("电池"):
        s = g.sort_values("RMSE")[["模型", "RMSE"]].reset_index(drop=True)
        best_name, best_val = str(s.loc[0, "模型"]), float(s.loc[0, "RMSE"])
        second_name, second_val = str(s.loc[1, "模型"]), float(s.loc[1, "RMSE"])
        rows.append(
            {
                "电池": cell,
                "最好的模型": best_name,
                "RMSE": best_val,
                "次好": second_name,
                "与次好的差距(Ah)": second_val - best_val,
                "差距占最好的比例": (second_val - best_val) / best_val,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------
def fig12(tbl: pd.DataFrame, m: int) -> Path:
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

    # ① 逐块测试 RMSE（对数轴：M3 的 B0018 与 M2 的 B0006 差一个量级）
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

    # ② EOL：预测 vs 真值（只画有真值的电池；删失与"给不出"单独标注）
    ax = axes[1]
    labeled = sorted(
        tbl.loc[tbl["EOL真值"] != "删失", "电池"].unique().tolist()
    )
    truth = {
        c: int(tbl.loc[(tbl["电池"] == c) & (tbl["EOL真值"] != "删失"), "EOL真值"].iloc[0])
        for c in labeled
    }
    markers = {"M1 线性": "o", "M2 双指数": "s", "M3 随机森林": "X", "M4 趋势+GPR": "^"}
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
            if row["EOL预测"] == "—":
                continue
            xs.append(truth[c])
            ys.append(int(row["EOL预测"]))
        if xs:
            kw = dict(styles[name])
            kw.setdefault("color", palette[name])
            kw.setdefault("edgecolors", palette[name])
            ax.scatter(xs, ys, label=name, zorder=5, **kw)

    # M1 与 M4 的点预测在这份数据上完全重合，必须说明，否则读者以为缺了两组点
    m1_pred = tbl[tbl["模型"] == "M1 线性"].set_index("电池")["EOL预测"]
    m4_pred = tbl[tbl["模型"] == "M4 趋势+GPR"].set_index("电池")["EOL预测"]
    same = bool((m1_pred.astype(str) == m4_pred.astype(str)).all())

    lo = min(truth.values()) - 12
    hi = max(truth.values()) + 32
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.2, color="#999999", label="完美预测线")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("真实寿命（循环）")
    ax.set_ylabel("模型预测的寿命（循环）")
    ax.set_title("② EOL：三个模型都给数，M3 一个都给不出", fontsize=10)
    censored = sorted(tbl.loc[tbl["EOL真值"] == "删失", "电池"].unique().tolist())
    coincide = (
        "\nM1 与 M4 的点预测完全重合（远场点预测 = 趋势线）"
        if same else ""
    )
    ax.annotate(
        f"{'、'.join(censored)} 删失（无真值），无法进这张图；\n"
        f"所有给出预测的模型都说它会在观测期内坏（方向都错）{coincide}"
        if censored else "本数据没有删失电池",
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
    """把某个宽表指标转成「行=电池、列=模型」。"""
    wide = tbl.pivot_table(index="电池", columns="模型", values=col, aggfunc="first")
    return wide.reindex(columns=MODEL_ORDER)


def render_md(
    tbl: pd.DataFrame,
    summ: pd.DataFrame,
    wins: pd.DataFrame,
    m: int,
    n_cover_m4: str,
    obs_len: dict[str, int],
) -> str:
    rmse = pivot_metric(tbl, "RMSE")
    mae = pivot_metric(tbl, "MAE")
    eol_pred = pivot_metric(tbl, "EOL预测")
    train = pivot_metric(tbl, "训练RMSE")
    truth = tbl.drop_duplicates("电池").set_index("电池")["EOL真值"]

    cells = sorted(tbl["电池"].unique().tolist())
    n_cells = len(cells)
    m3_all_none = int((tbl.loc[tbl["模型"] == "M3 随机森林", "EOL预测"] == "—").sum())
    models_with_eol = summ.loc[summ["给出 EOL 的电池数"] > 0, "模型"].tolist()
    best_summary = summ.loc[summ["测试RMSE 均值"].idxmin()]
    tie_cells = [
        row["电池"] for _, row in wins[wins["与次好的差距(Ah)"] < 0.002].iterrows()
    ]
    big_cells = [
        row["电池"] for _, row in wins[wins["差距占最好的比例"] > 0.1].iterrows()
    ]
    m3_win = [
        row["电池"] for _, row in wins[wins["最好的模型"] == "M3 随机森林"].iterrows()
    ]
    # 汇总表里的 NaN 显示成「—」，不许叫人去猜 "nan" 是不是缺数据
    summ_disp = summ.round(4).astype(object).where(summ.notna(), "—")

    out = [
        "# 四模型对比主表：同划分、同指标、逐块给数（W13）",
        "",
        "> 本文件由 `src/compare.py` 自动生成，数字随代码重算，**不要手工编辑**。",
        "> 机器可读版：`outputs/metrics/model_comparison.csv`（长表，W14/W17 直接用）。",
        "",
        "## 一、口径",
        "",
        f"- **任务**：任务 A（个体内轨迹外推）。切分点 **M = {m}**，时序划分，逐个体拟合。",
        f"- **电池**：{n_cells} 块（{'、'.join(cells)}）。",
        "- **四个模型**：M1 线性 / M2 双指数 / M3 随机森林（**设计好的反例**）/ M4 趋势+GPR。",
        "- **三条读表纪律**（见 D23 / D25）：",
        "  ① 只对**同分母**的量做比较——测试段 RMSE/MAE 可以，R² 的跨电池平均不行；",
        "  ② **训练段指标不参与排名**（M3 记住样本、M4 平滑残差，两者都偏低）；",
        "  ③ **EOL 有三态**——有值 / 真值删失 / 模型给不出，混在一起算就是造假。",
        "",
        "## 二、测试段误差（同分母，可横向比）",
        "",
        "### RMSE（Ah）",
        "",
        rmse.round(4).to_markdown(),
        "",
        "### MAE（Ah）",
        "",
        mae.round(4).to_markdown(),
        "",
        f"**逐块看谁最好**（比平均值可信：平均值会被单块的极端值带跑）：",
        "",
        wins.round(4).to_markdown(index=False),
        "",
        "⚠️ 这张「谁最好」必须连着**差距**一起读。",
        "",
        f"- 差距 **< 0.002 Ah 的平手**：{'、'.join(tie_cells) or '无'}；",
        f"- 差距 **超过 10%** 的真差距：{'、'.join(big_cells) or '无'}；",
        f"- M3 在 **{'、'.join(m3_win) or '无'}** 上「最好」，但它一块电池的寿命都给不出"
        f"（EOL 可比数 = 0）——**单看某一块的 RMSE 会把反例读成好模型**。",
        "",
        f"按四块电池的平均测试 RMSE 排，最低的是 **{best_summary['模型']}**"
        f"（{best_summary['测试RMSE 均值']:.4f} Ah）。"
        "平均值只差百分之几时，**逐块赢家 + 差距**才是更可信的说法。",
        "",
        "## 三、训练段：只用来回答「它记住没有」",
        "",
        train.round(4).to_markdown(),
        "",
        "M3 的训练 RMSE 比别的模型低一个量级——那不是优点，是**分段常数把训练点记住了**"
        "（W10 已证明它的外推是水平线）。M4 偏低是因为 GP 把残差平滑掉了。"
        "所以**这一张表不产生排名**，它的作用只是提醒：训练段好看和能不能外推是两回事。",
        "",
        "## 四、EOL：三态必须分开看",
        "",
        eol_pred.to_markdown(),
        "",
        f"- **给出预测的模型**：{'、'.join(models_with_eol)}；"
        f"**M3 在 {m3_all_none}/{n_cells} 块电池上给不出**（它的外推是常数，永远不穿越阈值）——"
        "这是 W10 的设计目的，不是 bug。",
    ]

    # 删失电池单独一段（不许当已知值参与误差）
    censored = tbl.loc[tbl["EOL真值"] == "删失", "电池"].unique().tolist()
    if censored:
        c0 = censored[0]
        preds = {
            name: tbl.loc[(tbl["电池"] == c0) & (tbl["模型"] == name), "EOL预测"].iloc[0]
            for name in models_with_eol
        }
        n_obs = obs_len.get(c0, m)
        out += [
            "",
            f"- **{c0} 没有「误差」**：它删失（观测期内没坏）。给得出数的模型都说它会坏——"
            + "、".join(f"{k} 报 {v}" for k, v in preds.items())
            + f"，而观测只有 {n_obs} 个循环。这不是「误差多大」的问题，"
            "是**方向对不对**的问题：三个模型的方向都错。",
        ]

    out += [
        "",
        summ_disp.to_markdown(index=False),
        "",
        "⚠️ 读这张汇总表必须注意：",
        "",
        f"1. **「EOL 可比电池数」不同**：M3 是 0，其余是 "
        f"{summ.loc[summ['模型'] != 'M3 随机森林', 'EOL 可比电池数'].max()}。"
        "所以 EOL 误差**不能跨模型直接比大小**——M3 的「—」不是「误差为 0」，"
        "是「没给数」。任何把「—」当 0 的写法都会让 M3 显得完美。",
        "2. **M2 的 EOL 是唯一「提前报死」最明显的**（三块有真值的电池两块报早），"
        "这与 W9 的偏差分析一致：它在观测末端系统性偏低 0.29 Ah。",
        f"3. **M4 的 90% 区间覆盖情况：{n_cover_m4}**（详见 `model_gpr.md`）。"
        "它说明区间偏窄，第三层不确定性还没进来——W14/W17 处理。",
        "",
        "## 五、这张表不能怎么读",
        "",
        "1. **不能按平均 R² 排名**：R² 的分母是各自电池的测试段方差，跨电池平均没有意义"
        "（D23 实测：按 R² 与按 RMSE 排序会翻）。本表因此**只列不平均**。",
        "2. **不能因为 M3 差就删掉它**：它的作用是反例——证明「树模型不能外推」；"
        "没有它，这个结论就只是断言。",
        "3. **不能说「M4 比 M1 准」**：两者在远场的点预测完全相同，M4 的价值是区间"
        "（W11 已证）。表里 RMSE 的微小差别来自近场那一点局部修正。",
        "4. **不能把这张表当跨个体（任务 B）的结论**：任务 B 的样本个体只有 3 个，"
        "LOBOCV 里零基准就打赢了所有特征模型（见 `task_b_lobocv.md`）。",
        "",
    ]
    return "\n".join(out)


def main() -> int:
    ensure_dirs()
    raw = load_raw()
    df = mark_rows(raw)
    prof = cell_profile(df)

    merged = collect(df, prof, SPLIT_M)
    tbl = split_eol_states(merged)
    summ = summary_table(tbl)
    wins = win_counts(tbl)

    # M4 的区间覆盖情况（就地从表里数，不重跑 GP）
    m4 = tbl[tbl["模型"] == "M4 趋势+GPR"]
    cover = m4.loc[m4["区间是否覆盖真值"] != "—", "区间是否覆盖真值"]
    n_cover = int((cover == "覆盖").sum())
    n_cover_all = len(cover)
    n_cover_m4 = f"{n_cover}/{n_cover_all} 块有真值的电池" if n_cover_all else "无可比电池"

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")

    obs_len = {
        cell: int(len(sub)) for cell, sub in split_by_battery(df).items()
    }
    TABLE_MD.write_text(
        render_md(tbl, summ, wins, SPLIT_M, n_cover_m4, obs_len), encoding="utf-8"
    )
    fig_path = fig12(tbl, SPLIT_M)

    print(f"对比主表：{len(tbl)} 行（{len(MODEL_ORDER)} 模型 × "
          f"{tbl['电池'].nunique()} 电池），M = {SPLIT_M}")
    print()
    print("测试段 RMSE（Ah）：")
    print(pivot_metric(tbl, "RMSE").round(4).to_string())
    print()
    print("汇总：")
    print(summ.round(4).to_string(index=False))
    print()
    print(f"对比主表已写出：{TABLE_MD.name}")
    print(f"长表已写出：{CSV_OUT.name}")
    print(f"已生成 {fig_path.name}  ({fig_path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
