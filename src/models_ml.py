"""W10：M3 随机森林 —— 「外推失效」的结构性反例。

一句话结论（先给结论，后面全是证据）
------------------------------------
把 `cycle` 当唯一自变量、把容量当标签训练随机森林：它在**训练段**能拟合到
R² 接近 1，但在**测试段**会把整段外推压成一条水平线，R² 深度为负、
寿命一步也预测不出来。原因不是参数没调好，而是模型类的**结构限制**——
树模型的预测是「训练标签的分段常数均值」，它没有"趋势"这个概念；
超出训练区间的所有输入都落进每棵树最右边那个叶子，于是输出与 x 无关。

为什么要把一个「注定失败」的模型写进报告
----------------------------------------
M1（线性）、M2（双指数）、M4（趋势+GPR）都是**函数形式**模型，都能外推；
M3 是**分段常数**模型，只能插值。四个模型放在一起，才能回答
「这类任务该用哪一类模型」，而不是「谁的 R² 高」。如果只挑能在测试段赢的
模型写报告，等于事先把反例筛掉了——那是自欺，也正好是老师最想听的那句
「树模型不适合外推」为什么成立的**实证**。

三个口径必须同时出现（D14/D21，写错就是自相矛盾）
-------------------------------------------------
- **任务 A（本模块）**：`cycle` 是回归的**自变量/时间轴**，不是泄露特征。
- **任务 B（W12 才做）**：跨个体早期预测，`cycle` **禁止**作特征。
  `features.py` 的 `FEATURE_EXCLUDE` 管的是任务 B，不是这里。
- **任务 A 没有 LOBOCV**：模型逐个体拟合，没有"留一块电池不训练"这回事（见 `evaluate.py`）。

本模块自己钉住的四条结构断言（不是注释，是 assert）
--------------------------------------------------
1. 外推段预测**严格是常数**（极差 < 1e-12）——被"手动补一条趋势"就会炸；
2. 该常数**落在训练标签区间内**——分段常数模型的输出只能是训练标签的凸组合；
3. 树长到底（`max_depth=None, min_samples_leaf=1, bootstrap=False`）时，
   外推常数**恰好等于最后一个观测值**——证明它连"斜率"都没有；
4. EOL 若被预测在训练区间内，备注必须标明「查表」——否则会把"查表"当成"预测"。

运行
----
    python src/models_ml.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    COLORS,
    EOL_SEARCH_MULTIPLE,
    EOL_THRESHOLD_AH,
    RANDOM_SEED,
    TABLE_DIR,
    ensure_dirs,
    is_at_eol,
    setup_matplotlib,
)
from evaluate import (  # noqa: E402
    SPLIT_M,
    SPLIT_M_SWEEP,
    eol_error,
    metrics,
    temporal_split,
)
from figures import save  # noqa: E402
from load_data import load_raw, split_by_battery  # noqa: E402
from models_parametric import fit_linear, predict_linear  # noqa: E402
from preprocess import cell_profile, mark_rows  # noqa: E402

TABLE_MD = TABLE_DIR / "model_rf.md"

M3_FEATURES = ["cycle"]
"""任务 A 里 M3 的唯一自变量。**只许有一个**：这是"外推失效"的干净反例。

若把它扩成 [cycle, cycle²] 之类的多项式特征，随机森林在区间外依然落进最右叶子，
结论不变，但"失效原因"会从"没有趋势概念"退化成"连时间轴都还是同一个"——
反例就不干净了。所以这里写死并在 `fit_rf` 里断言。
"""

N_TREES = 300
"""主配置的树数。实验二证明它改变不了外推结论（只改变常数的小数点后第三位）。"""

N_TREES_SWEEP = (10, 50, 300, 1000)
DEPTH_SWEEP = (2, 3, 4, 6, 10, None)
"""深度扫描：`None` = 长到底。用来展示外推常数如何逼近"最后一个观测值"。"""


# ---------------------------------------------------------------------------
# 拟合 / 预测
# ---------------------------------------------------------------------------
def _as_x(t) -> np.ndarray:
    """把一维循环号变成 sklearn 要的 (n, 1) 矩阵。"""
    return np.asarray(t, dtype=float).reshape(-1, 1)


def fit_rf(
    t: np.ndarray,
    y: np.ndarray,
    *,
    n_estimators: int = N_TREES,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    bootstrap: bool = True,
    seed: int = RANDOM_SEED,
    features: list[str] | None = None,
) -> dict:
    """训练一个随机森林回归器。**只用 cycle 一个特征**。

    参数
    ----
    features : 允许调用方显式声明特征列表，但只接受 `M3_FEATURES`。
        这样"有人偷偷加特征"会当场失败，而不是在下游变成一句解释不了的 R²。
    """
    feats = M3_FEATURES if features is None else list(features)
    assert feats == M3_FEATURES, (
        f"任务 A 的 M3 只许用 {M3_FEATURES} 作自变量，收到 {feats}——"
        "加特征会毁掉这个反例的干净程度（见模块文档）"
    )
    # 注：这里**不**再断言 n_features_in_ == 1。第六轮审查（R6-8）指出它是恒真的——
    # `_as_x()` 永远 reshape(-1, 1)，这条断言不可能失败，等于装饰。
    # 真正的守门放在调用方 `run()`（查 estimator.n_features_in_），
    # 因为 `fit_rf` 被整体替换时，函数内部的断言根本不会执行。

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(t) < 10:
        raise ValueError(f"M3 至少需要 10 个训练点，收到 {len(t)}")
    if not np.all(np.isfinite(y)):
        raise ValueError("训练标签里有 inf/nan")

    est = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        bootstrap=bootstrap,
        random_state=seed,
        n_jobs=1,  # 80 行数据，多线程只会引入不可复现的扰动
    )
    est.fit(_as_x(t), y)

    return {
        "name": "M3 随机森林",
        "estimator": est,
        "features": feats,
        "t_min": float(t.min()),
        "t_max": float(t.max()),
        "n_estimators": int(n_estimators),
        "max_depth": max_depth,
        "bootstrap": bool(bootstrap),
        "n_train": int(len(t)),
    }


def predict_rf(model: dict, t) -> np.ndarray:
    """预测。**不做任何截断或修正**——外推段会得到常数，这正是要看见的现象。"""
    pred = model["estimator"].predict(_as_x(t))
    if not np.all(np.isfinite(pred)):
        raise ValueError("随机森林给出了 inf/nan —— 不能当作预测值")
    return pred


def extrapolation_probe(model: dict, t_test) -> dict:
    """量化「外推段预测是不是常数」——**由代码判定，不靠叙述**。

    返回
    ----
    n_extrap : 落在训练区间之外的测试点个数
    ptp      : 这些预测值的极差（随机森林应当严格为 0）
    std      : 标准差（同上）
    constant : 那个常数值
    """
    t_test = np.asarray(t_test, dtype=float)
    pred = predict_rf(model, t_test)
    outside = t_test > model["t_max"]
    pe = pred[outside]
    return {
        "pred": pred,
        "n_extrap": int(outside.sum()),
        "extrap_ptp": float(np.ptp(pe)) if len(pe) else float("nan"),
        "extrap_std": float(np.std(pe)) if len(pe) else float("nan"),
        "extrap_constant": float(pe[0]) if len(pe) else float("nan"),
    }


def _assert_extrapolation_invariants(model: dict, sp, probe: dict) -> None:
    """三条结构断言。任何一条被破坏，说明这个"反例"已经不纯了。"""
    assert probe["n_extrap"] > 0, f"{sp.cell}: 测试段全在训练区间内，测不出外推行为"

    ptp = probe["extrap_ptp"]
    assert ptp < 1e-12, (
        f"{sp.cell}: 外推段预测不是严格常数（极差 {ptp:.3g}）——"
        "要么有人给模型补了趋势项，要么它根本不是树模型"
    )

    lo = float(np.min(sp.y_train))
    hi = float(np.max(sp.y_train))
    c = probe["extrap_constant"]
    assert lo - 1e-12 <= c <= hi + 1e-12, (
        f"{sp.cell}: 外推常数 {c:.4f} 落在训练标签区间 [{lo:.4f}, {hi:.4f}] 之外——"
        "分段常数模型的输出必须是训练标签的凸组合，越界意味着预测不是叶子均值"
    )


def eol_rf(model: dict, n_obs: int, tau: float | None = None) -> tuple[int | None, str]:
    """扫到模型自己认为"容量到达阈值"的第一个循环。

    树模型的预测是**分段常数**，不能像 M1/M2 那样用 brentq 求根（平坦区间没有符号变化），
    所以这里在整数循环上暴力扫描。判定统一走 `config.is_at_eol`（D17）。

    返回
    ----
    (循环号 或 None, 说明)
    """
    hi = int(np.ceil(n_obs * EOL_SEARCH_MULTIPLE))
    grid = np.arange(1, hi + 1)
    hit = is_at_eol(predict_rf(model, grid), tau)
    idx = np.flatnonzero(np.asarray(hit))
    if len(idx) == 0:
        return None, ""
    first = int(grid[idx[0]])
    if first <= model["t_max"]:
        return first, "⚠️ 穿越发生在训练区间内：这是查表，不是预测"
    return first, ""


# ---------------------------------------------------------------------------
# 跑一遍：主表
# ---------------------------------------------------------------------------
def run(df: pd.DataFrame, prof: pd.DataFrame, m: int = SPLIT_M) -> dict:
    prof_idx = prof.set_index("battery_id")
    rows, fits = [], {}
    for sp in temporal_split(df, m):
        eol_true = prof_idx.loc[sp.cell, "eol_cycle"]
        eol_true = None if pd.isna(eol_true) else int(eol_true)
        n_obs = len(sp.train) + len(sp.test)

        model = fit_rf(sp.t_train, sp.y_train)
        # 结构断言：模型实际吃进去的自变量个数（不是它自称的），任务 A 只许 1 个
        assert model["estimator"].n_features_in_ == len(M3_FEATURES), (
            f"{sp.cell}: 任务 A 的 M3 只许吃 {len(M3_FEATURES)} 个自变量，"
            f"实际吃进去 {model['estimator'].n_features_in_} 个"
        )
        pred = predict_rf(model, sp.t_test)
        met = metrics(sp.y_test, pred)
        train_met = metrics(sp.y_train, predict_rf(model, sp.t_train))

        probe = extrapolation_probe(model, sp.t_test)
        _assert_extrapolation_invariants(model, sp, probe)

        eol, why = eol_rf(model, n_obs)
        ee = eol_error(eol, eol_true, n_obs)
        note = "；".join(x for x in (ee["note"], why) if x)

        # 断言 4：训练区间内的"穿越"必须被标明是查表
        if eol is not None and eol <= m:
            assert "查表" in note, (
                f"{sp.cell}: M={m} 时 EOL 预测 {eol} ≤ M，落在训练区间内，"
                "备注必须标明这是查表而不是预测"
            )

        fits[sp.cell] = {"model": model, "split": sp, "probe": probe}
        rows.append(
            {
                "电池": sp.cell,
                "模型": model["name"],
                "M": m,
                "训练RMSE": train_met["RMSE"],
                "训练R2": train_met["R2"],
                "RMSE": met["RMSE"],
                "MAE": met["MAE"],
                "MAPE": met["MAPE"],
                "R2": met["R2"],
                "外推常数": probe["extrap_constant"],
                "外推极差": probe["extrap_ptp"],
                "外推点数": probe["n_extrap"],
                "EOL预测": "—" if eol is None else eol,
                "EOL真值": "删失" if eol_true is None else eol_true,
                "EOL误差": "—" if ee["eol_err"] is None else ee["eol_err"],
                "备注": note,
            }
        )
    return {"table": pd.DataFrame(rows), "fits": fits}


# ---------------------------------------------------------------------------
# 实验一：树数 —— 方差不是病因，加树治不了
# ---------------------------------------------------------------------------
def tree_count_sweep(df: pd.DataFrame, m: int = SPLIT_M) -> pd.DataFrame:
    rows = []
    for n in N_TREES_SWEEP:
        r2, consts = [], {}
        for sp in temporal_split(df, m):
            model = fit_rf(sp.t_train, sp.y_train, n_estimators=n)
            met = metrics(sp.y_test, predict_rf(model, sp.t_test))
            r2.append(met["R2"])
            consts[sp.cell] = extrapolation_probe(model, sp.t_test)["extrap_constant"]
        rows.append({"树数": n, "测试R2均值": float(np.mean(r2)),
                     "测试R2最大": float(np.max(r2)), **consts})
    out = pd.DataFrame(rows)
    base = out.loc[out["树数"] == N_TREES, list(split_by_battery(df).keys())].to_numpy(float)
    out["常数与主配置最大差"] = np.abs(
        out[list(split_by_battery(df).keys())].to_numpy(float) - base
    ).max(axis=1)
    assert (out["测试R2最大"] < 0).all(), "树数扫描里出现了非负的测试 R² —— 主表结论不成立"
    return out


# ---------------------------------------------------------------------------
# 实验二：深度 —— 长到底时，外推常数恰好等于最后一个观测值
# ---------------------------------------------------------------------------
def depth_sweep(df: pd.DataFrame, m: int = SPLIT_M) -> pd.DataFrame:
    """深度扫描（`bootstrap=False`，单棵，完全确定）。

    树越深，最右叶子里的样本越少；长到底时每片叶子只剩一个样本，
    于是外推常数**精确等于最后一个循环的容量**。这就是"没有趋势概念"的定量证明：
    模型能给出的最"远"的信息，就是最后一次观测本身。
    """
    rows = []
    cells = list(split_by_battery(df).keys())
    for depth in DEPTH_SWEEP:
        row = {"max_depth": "长到底" if depth is None else depth}
        for sp in temporal_split(df, m):
            model = fit_rf(sp.t_train, sp.y_train, n_estimators=1,
                           max_depth=depth, bootstrap=False)
            c = extrapolation_probe(model, sp.t_test)["extrap_constant"]
            row[sp.cell] = c
            row[f"{sp.cell}_最后观测"] = float(sp.y_train[-1])
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_diagnostics(model: dict, y_last: float, probe: float = 1000.0) -> dict:
    """自助采样的树里，有多少棵的「最右叶子」恰好落在最后一个训练循环上？

    第六轮审查（R6-5）推翻了原来的解释。旧说法是「自助采样会让最右叶子有时含两个样本」——
    实测不成立：主配置下 300 棵树的最右叶子**全部**是单样本叶子。
    真机制是**约三分之一的树在自助采样里根本没抽到最后一个循环**，于是它们的最右叶子
    落在更早的循环上；容量在下降，取平均就把常数抬高了。

    这条从机制解释变成了可复算的数：命中率应接近 1−(1−1/n)^n ≈ 0.63。
    """
    preds = np.array(
        [float(est.predict([[probe]])[0]) for est in model["estimator"].estimators_]
    )
    n_hit = int(np.sum(np.isclose(preds, y_last, rtol=0.0, atol=1e-12)))
    frac = n_hit / len(preds)
    expect = 1.0 - (1.0 - 1.0 / model["n_train"]) ** model["n_train"]
    assert 0.4 < frac < 0.95, (
        f"{n_hit}/{len(preds)} 棵树的右端点等于最后观测值（{frac:.0%}）——"
        f"自助采样命中率理论上应 ≈ {expect:.2f}，对不上说明 bootstrap 配置或这套解释要重写"
    )
    return {
        "n_trees": len(preds),
        "n_hit": n_hit,
        "frac_hit": frac,
        "expect": expect,
        "min_pred": float(preds.min()),
        "max_pred": float(preds.max()),
    }


def _dep_display(dep: pd.DataFrame, cells: list[str]) -> pd.DataFrame:
    """把深度扫描表压成报告里读得动的形状：列是电池，最后一行是「最后一个训练观测值」。

    逐行对照才能看出「深度加深 → 逼近最后观测值」这条线；原始宽表 8 列，在 Word 里放不下。
    """
    rows = [
        {"max_depth": row["max_depth"], **{c: row[c] for c in cells}}
        for _, row in dep.iterrows()
    ]
    rows.append(
        {"max_depth": "最后观测值", **{c: dep[f"{c}_最后观测"].iloc[-1] for c in cells}}
    )
    return pd.DataFrame(rows).round(4)


def deterministic_probe(df: pd.DataFrame, m: int = SPLIT_M) -> pd.DataFrame:
    """把树长到底、关掉自助采样：外推常数必须**恰好**等于最后观测值。"""
    rows = []
    for sp in temporal_split(df, m):
        model = fit_rf(sp.t_train, sp.y_train, n_estimators=1,
                       max_depth=None, bootstrap=False)
        c = extrapolation_probe(model, sp.t_test)["extrap_constant"]
        last = float(sp.y_train[-1])
        gap = c - last
        assert abs(gap) < 1e-12, (
            f"{sp.cell}: 树长到底后外推常数 {c!r} 与最后观测值 {last!r} 差 {gap:g}——"
            "说明预测不是最右叶子的单点均值"
        )
        rows.append({"电池": sp.cell, "最后观测值": last, "外推常数": c, "差": gap})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 实验三：把趋势剥出去，树就不一样了（M4 的思路）
# ---------------------------------------------------------------------------
def detrend_experiment(df: pd.DataFrame, m: int = SPLIT_M) -> pd.DataFrame:
    """纯 M3 / 「线性趋势 + 残差用树」/ M1 三者对比。

    做法：先用 M1 拟合趋势，把 `y − 趋势` 交给随机森林，预测时再加回趋势。
    这时外推 = **线性项 + 一个常数修正**，所以它能外推——但功劳在趋势项，
    树只负责"残差的局部均值"。这条实验同时解释了 M4 为什么长成"趋势 + GPR"。

    ⚠️ 第一版这里写了个断言：「去趋势后 R² 必须大于纯树」。它失败了（B0006 反而略差），
    因为那个常数修正**可正可负**：线性趋势本身在 B0006 上已经偏低，再叠一个负的残差
    平移只会更低。所以断言改成结构性的那条：**去趋势后的测试段预测必须是一条精确的直线**
    （残差在外推段是常数 ⇒ 预测 = 直线 + 常数 = 还是直线）。R² 大小留给人看，不用断言规定。
    """
    rows = []
    for sp in temporal_split(df, m):
        lin = fit_linear(sp.t_train, sp.y_train)
        resid = sp.y_train - predict_linear(lin, sp.t_train)
        rf_resid = fit_rf(sp.t_train, resid)
        pred_detrend = predict_linear(lin, sp.t_test) + predict_rf(rf_resid, sp.t_test)

        second_diff = np.diff(pred_detrend, n=2)
        assert np.all(np.abs(second_diff) < 1e-9), (
            f"{sp.cell}: 去趋势后的测试段预测不是直线（二阶差分最大 "
            f"{np.abs(second_diff).max():.3g}）——残差模型在外推段应当只输出一个常数"
        )

        rf_pure = fit_rf(sp.t_train, sp.y_train)
        met_pure = metrics(sp.y_test, predict_rf(rf_pure, sp.t_test))
        met_det = metrics(sp.y_test, pred_detrend)
        met_lin = metrics(sp.y_test, predict_linear(lin, sp.t_test))
        r2_pure, r2_det, r2_lin = met_pure["R2"], met_det["R2"], met_lin["R2"]

        shift = float(predict_rf(rf_resid, sp.t_test)[0])
        # 结构断言：残差在外推段是常数，而它必须是训练残差的凸组合
        assert abs(shift) <= float(np.max(np.abs(resid))) + 1e-12, (
            f"{sp.cell}: 残差常数平移 {shift:.5f} 超出了训练残差区间 —— 与「叶子均值」矛盾"
        )

        rows.append(
            {
                "电池": sp.cell,
                "纯 M3": r2_pure,
                "趋势+树": r2_det,
                "M1 线性": r2_lin,
                "RMSE 纯 M3": met_pure["RMSE"],
                "RMSE 趋势+树": met_det["RMSE"],
                "RMSE M1": met_lin["RMSE"],
                "常数平移(Ah)": shift,
                "测试段标准差(Ah)": float(np.std(sp.y_test)),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 切分点敏感性（与 W9 同一教训：误差必须带切分点）
# ---------------------------------------------------------------------------
def split_sensitivity(df: pd.DataFrame, prof: pd.DataFrame) -> pd.DataFrame:
    """换切分点 M。同时记录一件必须被看见的事：M 越过真实寿命时，
    "EOL 预测"就变成了训练区间内的**查表**，误差看着小甚至为 0 —— 纯假象。"""
    prof_idx = prof.set_index("battery_id")
    rows = []
    for m in SPLIT_M_SWEEP:
        tbl = run(df, prof, m)["table"]
        r2 = pd.to_numeric(tbl["R2"], errors="coerce")
        consts = tbl["外推常数"].to_numpy(float)
        eol = tbl["EOL预测"].tolist()
        lookup = [str(c) for c, v in zip(tbl["电池"], eol) if v != "—" and int(v) <= m]
        passed = [
            str(c)
            for c in tbl["电池"]
            if not pd.isna(prof_idx.loc[c, "eol_cycle"]) and int(prof_idx.loc[c, "eol_cycle"]) <= m
        ]
        rows.append(
            {
                "M": m,
                "测试R2均值": float(r2.mean()),
                "测试R2最大": float(r2.max()),
                "外推常数区间": f"{consts.min():.3f}–{consts.max():.3f} Ah",
                "EOL预测": "、".join("—" if v == "—" else str(v) for v in eol),
                "查表电池": "、".join(lookup) or "无",
                "真实寿命已被 M 越过": "、".join(passed) or "无",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------
def fig07(df: pd.DataFrame, res: dict, m: int) -> Path:
    """主图：外推段被压成一条水平线（与 M1 的直线对照）。"""
    setup_matplotlib()
    cells = list(split_by_battery(df).keys())
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.6))
    axes = axes.ravel()

    for ax, cell in zip(axes, cells):
        sp = res["fits"][cell]["split"]
        probe = res["fits"][cell]["probe"]
        model = res["fits"][cell]["model"]

        ax.plot(sp.t_train, sp.y_train, color=COLORS[cell], lw=2.0, zorder=5,
                label="已观测（训练段）")
        ax.plot(sp.t_test, sp.y_test, color=COLORS[cell], lw=1.3, ls=":", alpha=0.9,
                zorder=5, label="真实后段（测试段）")

        t_ext = np.linspace(1, (len(sp.train) + len(sp.test)) * 1.18, 900)
        ax.step(t_ext, predict_rf(model, t_ext), where="post", color="#6a3d9a", lw=1.7,
                zorder=3, label="M3 随机森林（分段常数）")

        lin = fit_linear(sp.t_train, sp.y_train)
        ax.plot(t_ext, predict_linear(lin, t_ext), "--", color="#777777", lw=1.4,
                zorder=2, label="M1 线性（参照）")

        ax.axhline(EOL_THRESHOLD_AH, ls="--", lw=1.1, color="#aaaaaa")
        ax.text(0.02, EOL_THRESHOLD_AH, f"τ={EOL_THRESHOLD_AH:g} Ah", fontsize=8,
                color="#666666", va="bottom", transform=ax.get_yaxis_transform())
        ax.axvline(m, color="#999999", lw=0.9, ls=":")
        ax.text(m, 0.02, f" M={m} ", fontsize=8, color="#777777",
                transform=ax.get_xaxis_transform(), va="bottom")

        # 竖轴范围收紧到"有信息"的区间；M1 跌出框外时明确标注，而不是让它把纵轴压扁
        y_lo = min(float(sp.y_test.min()), EOL_THRESHOLD_AH) - 0.12
        y_hi = float(sp.y_train.max()) + 0.06
        ax.set_ylim(y_lo, y_hi)
        if float(predict_linear(lin, np.array([t_ext[-1]]))[0]) < y_lo:
            ax.annotate("M1 继续跌出图外", xy=(0.985, 0.03), xycoords="axes fraction",
                        ha="right", va="bottom", fontsize=7.4, color="#666666",
                        bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.8))

        ax.text(
            0.035, 0.055,
            f"外推段 {probe['n_extrap']} 个点完全相同：{probe['extrap_constant']:.4f} Ah\n"
            f"（极差 {probe['extrap_ptp']:.1e}）｜测试 R² = "
            f"{float(res['table'].loc[res['table']['电池'] == cell, 'R2'].iloc[0]):.2f}",
            transform=ax.transAxes, fontsize=7.8, color="#3f2a6b",
            bbox=dict(boxstyle="round,pad=0.28", fc="#f3eefc", ec="#b9a6e0", lw=0.7),
        )
        ax.set_title(f"{cell}：训练 {len(sp.train)} → 外推 {len(sp.test)} 循环", fontsize=10.5)
        ax.set_xlabel("循环")
        ax.set_ylabel("容量 (Ah)")
        ax.legend(fontsize=7.4, loc="upper right", framealpha=0.92)

    fig.suptitle(f"图7　M3 随机森林：切分点 M={m}，外推段被压成一条水平线", fontsize=13, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return save(fig, "fig07_rf_extrapolation.png")


def fig08(dep: pd.DataFrame, det: pd.DataFrame, tree: pd.DataFrame, m: int) -> Path:
    """诊断图：① 深度扫描收敛到最后观测值；② 树数无关；③ 去趋势才有救。"""
    setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.9))
    cells = [c for c in dep.columns if not c.endswith("_最后观测") and c != "max_depth"]

    ax = axes[0]
    x = np.arange(len(dep))
    for cell in cells:
        ax.plot(x, dep[cell].to_numpy(float), "o-", lw=1.5, ms=4.5,
                color=COLORS[cell], label=cell)
        ax.axhline(float(dep[f"{cell}_最后观测"].iloc[-1]), ls=":", lw=1.0,
                   color=COLORS[cell], alpha=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in dep["max_depth"]])
    ax.set_xlabel("max_depth（长到底 = 最右）")
    ax.set_ylabel("外推段预测值 (Ah)")
    # 标出"从这里开始精确等于最后观测值"的位置（文字放标题里，不压在线上）
    exact = [
        i
        for i in range(len(dep))
        if max(abs(float(dep.iloc[i][c]) - float(dep.iloc[i][f"{c}_最后观测"])) for c in cells)
        < 1e-12
    ]
    exact_txt = (
        f"\n点线 = 各自最后观测值；depth ≥ {dep['max_depth'].iloc[exact[0]]} 起与之完全重合"
        "\n（左图 bootstrap=False、单棵确定树；中图是主配置 bootstrap=True）"
        if exact else "\n点线 = 各自最后观测值"
    )
    if exact:
        ax.axvline(exact[0], color="#888888", ls=":", lw=1.0, zorder=1)
    ax.set_title(f"① 树越深，外推值越接近「最后一次观测」{exact_txt}", fontsize=9.4)
    ax.legend(fontsize=7.6, ncol=2)

    ax = axes[1]
    for cell in cells:
        ax.plot(tree["树数"], tree[cell].to_numpy(float), "s-", lw=1.5, ms=4.5,
                color=COLORS[cell], label=cell)
    ax.set_xscale("log")
    ax.set_xlabel("树数（对数轴）")
    ax.set_ylabel("外推段预测值 (Ah)")
    ax.set_title("② 加树不改变外推值：机制不是方差（bootstrap=True）", fontsize=9.6)
    ax.legend(fontsize=7.6, ncol=2)

    ax = axes[2]
    w = 0.26
    idx = np.arange(len(det))
    # 用 RMSE（Ah）而不是 R²：跨电池平均 R² 没有意义，而负值区的 R² 会把排序搞反（R6-6）
    for k, (key, color) in enumerate(
        (("RMSE 纯 M3", "#c0392b"), ("RMSE 趋势+树", "#6a3d9a"), ("RMSE M1", "#7f8c8d"))
    ):
        ax.bar(idx + (k - 1) * w, det[key].to_numpy(float), w,
               label=key.replace("RMSE ", ""), color=color)
    ax.set_xticks(idx)
    ax.set_xticklabels(det["电池"].tolist())
    ax.set_ylabel("测试段 RMSE (Ah)　越低越好")
    ax.set_title("③ 去趋势后回到直线：外推能力回来了\n（但常数平移可正可负，不保证更好）", fontsize=9.6)
    ax.legend(fontsize=7.8)

    fig.suptitle(f"图8　M3 的三次诊断（M={m}）", fontsize=13, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return save(fig, "fig08_rf_diagnostics.png")


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
def render_md(
    tbl: pd.DataFrame,
    tbl_m: pd.DataFrame,
    dep: pd.DataFrame,
    det: pd.DataFrame,
    tree: pd.DataFrame,
    det_probe: pd.DataFrame,
    boot: pd.DataFrame,
    m: int,
) -> str:
    """报告。**每一句结论的数字都由上面的表现算**（D19）。"""
    cells = tbl["电池"].tolist()
    n = len(cells)

    train_r2 = tbl["训练R2"].to_numpy(float)
    test_r2 = tbl["R2"].to_numpy(float)
    worst = int(np.argmin(test_r2))

    n_neg = int((test_r2 < 0).sum())
    min_train, max_train = float(train_r2.min()), float(train_r2.max())
    min_test, max_test = float(test_r2.min()), float(test_r2.max())

    # 训练标签区间 vs 外推常数（证明常数只是"区间内"的插值结果）
    range_rows = []
    for cell in cells:
        sub = tbl.loc[tbl["电池"] == cell]
        range_rows.append(
            f"| {cell} | {sub['外推常数'].iloc[0]:.4f} | "
            f"{float(dep[f'{cell}_最后观测'].iloc[-1]):.4f} |"
        )

    none_cells = tbl.loc[tbl["EOL预测"] == "—", "电池"].tolist()
    none_with_truth = tbl.loc[
        (tbl["EOL预测"] == "—") & (tbl["EOL真值"] != "删失"), "电池"
    ].tolist()
    finite_truth = tbl.loc[tbl["EOL真值"] != "删失"]
    truth_pairs = "、".join(f"{r.电池}={r.EOL真值}" for r in finite_truth.itertuples())
    n_extrap_total = int(tbl["外推点数"].sum())
    ptp_max = float(tbl["外推极差"].max())
    ptp_txt = "恰好 0（浮点精确相等）" if ptp_max == 0.0 else f"{ptp_max:.1e}"

    lookup_rows = tbl_m.loc[tbl_m["查表电池"] != "无"]
    if len(lookup_rows):
        parts = []
        for _, r in lookup_rows.iterrows():
            parts.append(
                f"M={int(r['M'])} 时 {r['查表电池']} 的「EOL」落在训练区间内"
                f"（该切分点已越过 {r['真实寿命已被 M 越过']} 的真实寿命）"
            )
        lookup_txt = "；".join(parts)
    else:
        lookup_txt = "本次扫描里没有出现训练区间内的穿越"

    n_trees_spread = float(tree["常数与主配置最大差"].max())
    dep_last = dep.iloc[-1]
    dep_first = dep.iloc[0]
    gap_first = max(
        abs(float(dep_first[c]) - float(dep_first[f"{c}_最后观测"])) for c in cells
    )
    gap_last = max(
        abs(float(dep_last[c]) - float(dep_last[f"{c}_最后观测"])) for c in cells
    )

    # 主配置（bootstrap=True）与确定性探针（bootstrap=False）的常数差多少
    boot_gaps = [
        abs(float(tbl.loc[tbl["电池"] == c, "外推常数"].iloc[0])
            - float(dep[f"{c}_最后观测"].iloc[-1]))
        for c in cells
    ]
    boot_lo, boot_hi = min(boot_gaps), max(boot_gaps)

    boot_n_hit_lo, boot_n_hit_hi = int(boot["n_hit"].min()), int(boot["n_hit"].max())
    boot_frac_lo, boot_frac_hi = float(boot["frac_hit"].min()), float(boot["frac_hit"].max())
    boot_expect = float(boot["expect"].mean())
    boot_pred_lo, boot_pred_hi = float(boot["min_pred"].min()), float(boot["max_pred"].max())

    det_pure = det["纯 M3"].mean()
    det_trend = det["趋势+树"].mean()
    det_lin = det["M1 线性"].mean()
    det_rmse_pure = det["RMSE 纯 M3"].mean()
    det_rmse_trend = det["RMSE 趋势+树"].mean()
    det_rmse_lin = det["RMSE M1"].mean()
    n_det_better_rmse = int((det["RMSE 趋势+树"] < det["RMSE 纯 M3"]).sum())
    n_det_beats_m1 = int((det["RMSE 趋势+树"] < det["RMSE M1"]).sum())
    only_cell = det.loc[det["RMSE 趋势+树"] < det["RMSE M1"], "电池"].tolist()
    shift_lo = float(det["常数平移(Ah)"].abs().min())
    shift_hi = float(det["常数平移(Ah)"].abs().max())
    worst_det = det.loc[(det["趋势+树"] - det["纯 M3"]).idxmin()]
    best_det = det.loc[(det["趋势+树"] - det["纯 M3"]).idxmax()]

    # 深度扫描里第一次「全部与最后观测值精确相等」的深度
    exact_rows = []
    for _, row in dep.iterrows():
        exact_rows.append(
            max(abs(float(row[c]) - float(row[f"{c}_最后观测"])) for c in cells) < 1e-12
        )
    dep_exact = dep.loc[exact_rows, "max_depth"].iloc[0] if any(exact_rows) else "未达到"

    out = [
        "# M3 随机森林：外推失效的结构性反例（W10）",
        "",
        "> 本文件由 `src/models_ml.py` 自动生成，数字随代码重算，**不要手工编辑**。",
        "",
        "## 一、口径",
        "",
        f"- **任务**：任务 A（个体内轨迹外推）。前 **M = {m}** 个循环训练，"
        "预测 M+1..N 的容量，再让模型自己外推到阈值求寿命。",
        f"- **自变量只有一个**：`cycle`（`M3_FEATURES = {M3_FEATURES}`）。"
        "在任务 A 里它是**时间轴**，不是泄露特征；任务 B（跨个体早期预测）里它才被禁止。"
        "两个说法必须同时成立，否则会自相矛盾（D14/D21）。",
        "- **切分**：时序划分，不随机打乱（D6）。任务 A 逐个体拟合，**不存在 LOBOCV**。",
        f"- **超参**：`n_estimators={N_TREES}`、`max_depth=None`、"
        f"`min_samples_leaf=1`、`random_state={RANDOM_SEED}`。",
        f"- **寿命**：扫描 1..{EOL_SEARCH_MULTIPLE}×N 个循环，取第一个使 "
        "`is_at_eol` 为真的循环号（分段常数不能用求根法，见 `eol_rf` 注释）。",
        "",
        "## 二、主表",
        "",
        tbl.to_markdown(index=False),
        "",
        "## 三、现象：训练段几乎完美，测试段彻底失效",
        "",
        f"训练段 R² 在 **{min_train:.3f}–{max_train:.3f}** 之间（树模型拟合样本内数据本来就强）；"
        f"同一批模型在测试段 R² 在 **{min_test:.2f} 到 {max_test:.2f}** 之间，"
        f"**{n_neg}/{n} 块电池为负**。最差的是 {cells[worst]}（{test_r2[worst]:.2f}）。",
        "",
        "R² 的分母是**测试段自身的方差**，也就是参照系是「拿测试段的真实均值当预测」。"
        "负值意味着：连这个偷看了答案的常数都比它准。这不是「稍微差一点」，"
        "是模型在测试段里的信息量低于一个常数。",
        "",
        "寿命上更直接："
        f"EOL 预测为「—」（模型认为不会穿越阈值）的电池是"
        f"**{'、'.join(none_cells)}**，共 {len(none_cells)}/{n} 块；"
        f"其中 {'、'.join(none_with_truth)} 的真实寿命是有限的（{truth_pairs}）。"
        "也就是说，**M3 在已经坏掉的电池上给出的答案是「不会坏」，方向就是反的。**",
        "",
        "## 四、机制：为什么必然是常数（三层证据，全部可复算）",
        "",
        "### 证据 1：外推段预测严格是常数",
        "",
        f"四块电池测试段共 {n_extrap_total} 个点，**全部落在训练区间之外**，"
        f"它们的预测值**完全相同**：外推极差 {ptp_txt}"
        "（代码断言 `极差 < 1e-12`，任何「手动补一条趋势」的改动都会当场让它失败）。",
        "",
        "原因：树的每一次预测都是「沿分裂条件走到某片叶子，取该叶子训练标签的均值」。"
        "`cycle > t_max` 这个条件对所有超出训练区间的输入**永远走同一条路径**，"
        "所以每棵树都输出同一个数，森林输出它们的平均——**与输入无关**。",
        "",
        "### 证据 2：这个常数落在训练标签区间内",
        "",
        "| 电池 | 外推常数 (Ah) | 最后一个训练观测 (Ah) |",
        "|---|---|---|",
        *range_rows,
        "",
        "外推常数是若干叶子均值的平均，而每个叶子均值都是训练标签的凸组合，"
        "所以它必然落在 `[min(y_train), max(y_train)]` 内——代码里也断言了这一点"
        "（越界就说明预测不是叶子均值，而这对树模型是不可能的）。"
        "换句话说：**它只能把已知的容量水平重复一遍，给不出任何「更低」的东西。**",
        "",
        "### 证据 3：树长到底时，外推值恰好等于最后一次观测",
        "",
        f"`bootstrap=False`、`n_estimators=1`、`max_depth=None`："
        f"四块电池的外推常数与最后一个训练观测之差**全部为 0**（浮点精确相等，逐条断言）：",
        "",
        det_probe.to_markdown(index=False),
        "",
        f"深度扫描把这件事连成一条线：`max_depth={dep_first['max_depth']}` 时，"
        f"外推值与最后观测值最多差 **{gap_first:.4f} Ah**；"
        f"从 `max_depth={dep_exact}` 起，四块电池的差**全部为 0**"
        f"（长到底时最大差 {gap_last:.0e}）。",
        "",
        _dep_display(dep, cells).to_markdown(index=False),
        "",
        "这就是「没有趋势概念」的定量表述：**模型能给出的最远信息，就是它见过的最后一个点。**",
        "",
        "⚠️ 两张表的常数不完全相同，别当成矛盾：上面这张是 `bootstrap=False`（关掉自助采样，"
        "每片最右叶子只含一个样本，所以**恰好**等于最后观测）；主表用的是主配置"
        f"（`bootstrap=True`，{N_TREES} 棵树），常数与最后观测差 **{boot_lo:.4f}–{boot_hi:.4f} Ah**。",
        "",
        f"抬高的机制是自助采样：{N_TREES} 棵树里只有 "
        + (f"**{boot_n_hit_lo} 棵**" if boot_n_hit_lo == boot_n_hit_hi
           else f"**{boot_n_hit_lo}–{boot_n_hit_hi} 棵**")
        + "（"
        + (f"{boot_frac_lo:.0%}" if boot_frac_lo == boot_frac_hi
           else f"{boot_frac_lo:.0%}–{boot_frac_hi:.0%}")
        + "）的最右叶子落在最后一个循环上，"
        f"理论命中率 1−(1−1/M)^M = {boot_expect:.2f}（M = {m}）。"
        "剩下的树没抽到最后一个循环，最右叶子落在更早的循环；容量在下降，取平均就把常数抬高了。"
        f"各树右端点的取值范围是 {boot_pred_lo:.4f}–{boot_pred_hi:.4f} Ah。",
        "（这里原本写的是「最右叶子有时含两个样本」——第六轮审查实测 300/300 片都是单样本叶子，"
        "那个解释是错的，已按实测改写。）",
        "",
        "## 五、调参能不能救？不能——这不是方差问题",
        "",
        "树数是方差参数。如果失效来自「树太少、方差大」，加树应该能让预测稳住并靠近真值。"
        "实际是：",
        "",
        tree.round(4).to_markdown(index=False),
        "",
        f"树数从 {N_TREES_SWEEP[0]} 加到 {N_TREES_SWEEP[-1]}，外推常数最多只移动 "
        f"**{n_trees_spread:.2e} Ah**，测试段 R² 依然全为负。"
        "因为失效来自**偏差**（函数类里根本没有「一条下降的直线」这种形状），"
        "而加树只能消方差不消偏差。这条区分（偏差 vs 方差）是本实验最该讲清的一点。",
        "",
        "## 六、把趋势交给外层：能恢复外推，但多出一个没人管的自由度",
        "",
        "先用 M1 拟合线性趋势，把残差交给随机森林，预测时再加回趋势：",
        "",
        det[["电池", "纯 M3", "趋势+树", "M1 线性", "RMSE 纯 M3", "RMSE 趋势+树",
             "RMSE M1", "常数平移(Ah)", "测试段标准差(Ah)"]].round(4).to_markdown(index=False),
        "",
        f"**按测试段 RMSE（Ah，可比的绝对量）看**：纯 M3 **{det_rmse_pure:.4f}** → "
        f"趋势+树 **{det_rmse_trend:.4f}** → 仅 M1 **{det_rmse_lin:.4f}**；"
        f"趋势+树在 **{n_det_better_rmse}/{n}** 块电池上优于纯树"
        f"（最好 {best_det['电池']}：{best_det['RMSE 纯 M3']:.4f} → "
        f"{best_det['RMSE 趋势+树']:.4f}；最差 {worst_det['电池']}："
        f"{worst_det['RMSE 纯 M3']:.4f} → {worst_det['RMSE 趋势+树']:.4f}）。",
        "",
        f"⚠️ 这里**故意不按平均 R² 排序**（那是 {det_pure:.2f} → {det_trend:.2f} → "
        f"{det_lin:.2f}）。R² 的分母是各自电池测试段的方差，跨电池取平均没有意义，"
        "而且它在负值区对绝对误差极度敏感——第六轮审查用 RMSE 重排后发现结论是**反的**。"
        f"但 RMSE 的平均也不能全信：趋势+树只在 {len(only_cell)}/{n} 块电池"
        f"（{'、'.join(only_cell) or '无'}）上真的优于 M1，平均值好看是被那一块拉出来的。"
        "**比误差就比 Ah，并且要看逐块，不能只看均值。**",
        "",
        "它确实**恢复了外推能力**：残差模型在外推段只输出一个常数，"
        "所以「趋势+树」的预测 = **一条直线 + 一个常数平移**，"
        "代码里断言这个预测在测试段上二阶差分 < 1e-9（即精确是直线）。"
        "水平线变成了有斜率的线，这是质变。",
        "",
        "但它同时引入了一个**没人约束的自由度**：那个平移。"
        f"实测平移绝对值 {shift_lo:.4f}–{shift_hi:.4f} Ah（断言保证它落在训练残差区间内，"
        "因为它是叶子均值的平均），可正可负、没有任何机制决定它该是多少。",
        "",
        f"以 {worst_det['电池']} 为例，把两件事拆开算（都在表里）："
        f"把水平线换成直线（趋势项），R² 从 {worst_det['纯 M3']:.2f} 变成 "
        f"{worst_det['M1 线性']:.2f}；再叠加 {worst_det['常数平移(Ah)']:+.4f} Ah 的平移，"
        f"掉到 {worst_det['趋势+树']:.2f}。"
        f"那块电池测试段自身的标准差只有 {float(worst_det['测试段标准差(Ah)']):.4f} Ah，"
        "几十毫安时的平移足以让 R² 翻几倍——**归因必须落到具体哪一步**，"
        "把 −2.41→−6.21 整个算在平移头上是错的（第六轮审查 R6-4）。",
        "",
        "所以这一节的结论**不是**「去趋势能救」，而是：",
        "**去掉趋势之后，剩下的自由度必须被约束。** M1 的斜率是 "
        f"{m} 个循环外推出来的、"
        "完全确定的数；残差平移是纯自由的。两者都没有「我有多不确定」的表达。"
        "M4（趋势 + GPR）正是要把这两样都变成**带方差的量**——那是 W11 的事。",
        "",
        "顺带一条读数纪律：R² 在负值区对绝对误差极度敏感（分母是测试段自身方差），"
        "**不能拿 R² 的差值当「改善幅度」**。要比就比 Ah 或循环数。",
        "",
        "## 七、切分点敏感性",
        "",
        tbl_m.to_markdown(index=False),
        "",
        "⚠️ 最后一列和倒数第二列要一起看。M 一旦越过某块电池的真实寿命，"
        "训练段里就直接出现了容量 ≤ τ 的点，于是 M3 会「预测」一个**落在训练区间内**的循环号。"
        "那不是预测，是**查表**——它的误差看着小甚至恰好为 0，正是最危险的那种假象。"
        "所以 `run()` 里有断言：这种情况备注必须带「查表」字样，否则直接失败。",
        "",
        f"本次 M ∈ {list(SPLIT_M_SWEEP)} 的实际情形：{lookup_txt}。",
        "",
        "## 八、这张表不能怎么读",
        "",
        "1. **不能把它读成「随机森林很差」**：它在这里的任务是外推，而树模型本来就不擅长外推。"
        "换成任务 B（跨个体早期预测，样本单位是个体、不做外推），随机森林是合理的候选——"
        "那部分在 W12 用 LOBOCV 做。",
        "2. **不能拿训练段 R² 证明模型好**：训练段 R² 高恰恰是「它把每个点都记住了」的直接后果。",
        "3. **不能因为预测出常数就认为「电池不会坏」**：常数高于阈值只说明"
        "**这个模型没有能力给出下降趋势**，不说明电池健康。真实寿命是 "
        f"{truth_pairs}，而 M3 的输出见第二节的表。",
        "4. **不能把这里的超参当成「最优」**：没有做过任何调参搜索，因为结论对超参不敏感"
        "（第五节）。唯一影响结果的是深度趋近无穷时收敛到最后一个观测值。",
        "",
    ]
    return "\n".join(out)


def main() -> int:
    ensure_dirs()
    raw = load_raw()
    df = mark_rows(raw)
    prof = cell_profile(df)

    res = run(df, prof, SPLIT_M)
    tbl = res["table"]
    tbl_m = split_sensitivity(df, prof)
    dep = depth_sweep(df, SPLIT_M)
    det = detrend_experiment(df, SPLIT_M)
    tree = tree_count_sweep(df, SPLIT_M)
    det_probe = deterministic_probe(df, SPLIT_M)

    boot = pd.DataFrame(
        [
            {"电池": cell,
             **bootstrap_diagnostics(v["model"], float(v["split"].y_train[-1]))}
            for cell, v in res["fits"].items()
        ]
    )

    # 全场结论的两条硬断言：测试段全部为负、训练段全部优于测试段
    assert (tbl["R2"].to_numpy(float) < 0).all(), (
        "M3 在主切分点上有非负的测试 R² —— 外推失效这个结论不成立，必须重查"
    )
    assert (tbl["R2"].to_numpy(float) < tbl["训练R2"].to_numpy(float)).all(), (
        "有电池的测试 R² 不低于训练 R² —— 这不是预期中的外推失效形态"
    )

    TABLE_MD.write_text(
        render_md(tbl, tbl_m, dep, det, tree, det_probe, boot, SPLIT_M), encoding="utf-8"
    )
    fig_a = fig07(df, res, SPLIT_M)
    fig_b = fig08(dep, det, tree, SPLIT_M)

    print(f"切分点 M = {SPLIT_M}；自变量 = {M3_FEATURES}；树数 = {N_TREES}")
    print()
    print(tbl.to_string(index=False))
    print()
    print("深度扫描（外推段常数）:")
    print(dep.round(4).to_string(index=False))
    print()
    print("去趋势实验（测试段）：")
    print(
        det[["电池", "纯 M3", "趋势+树", "M1 线性", "RMSE 纯 M3", "RMSE 趋势+树",
             "RMSE M1", "常数平移(Ah)"]].round(4).to_string(index=False)
    )
    print()
    print("自助采样诊断（最右叶子落在最后一个循环上的树有多少）:")
    print(boot.round(4).to_string(index=False))
    print()
    print("切分点敏感性:")
    print(tbl_m.to_string(index=False))
    print()
    print(f"M3 报告已写出：{TABLE_MD.name}")
    print(f"已生成 {fig_a.name}  ({fig_a.stat().st_size / 1024:.0f} KB)")
    print(f"已生成 {fig_b.name}  ({fig_b.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
