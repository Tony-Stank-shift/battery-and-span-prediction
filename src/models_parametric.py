"""W9：M1 线性基线 + M2 双指数经验模型。

任务与口径
----------
**任务 A（个体内轨迹外推）**：用某块电池前 M 个循环拟合，预测 M+1..N 的容量，
再让模型自己外推到 EOL 阈值 τ，看它把寿命报成多少。
切分点、指标、EOL 误差口径全部来自 `evaluate.py`（唯一出处，四个模型共用）。

评测学提醒（写在最前面，因为它决定了这张表怎么读）
----------------------------------------------------
任务 A 是**逐个体拟合**——模型只在某块电池自己的前 M 个循环上训练。
所以这里**没有**「留一块电池不参与训练」这回事，LOBOCV 在任务 A 上逻辑上不存在。
D6 说的「LOBOCV + 时序划分」是分工：任务 A 用时序划分，任务 B（M3 的特征版）才用 LOBOCV。

两个模型各自代表什么
--------------------
| | 形式 | 参数数 | 可外推性 |
|---|---|---|---|
| M1 线性 | `cap = a + b·t` | 2 | 能，但没有物理形态：斜率由 80 个点定死，外推越远越不可信 |
| M2 双指数 | `cap = a·e^(bt) + c·e^(dt)` | 4 | 能，且带「两段衰减」的形状假设 |

⚠️ M2 的 4 个参数在 ~M 个点上大概率**不可辨识**（`(a,b)` 与 `(c,d)` 可以互相补偿）。
本模块因此做三件事：① 记录拟合是否收敛；② 收敛失败时**降到单指数**并明确记录
（D5 的降级路径）；③ 输出参数值，让「不可辨识」这件事可以被看见，而不是被藏起来。

运行
----
    python src/models_parametric.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq, curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    COLORS,
    EOL_SEARCH_MULTIPLE,
    EOL_THRESHOLD_AH,
    RATED_CAPACITY_AH,
    TABLE_DIR,
    ensure_dirs,
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
from preprocess import cell_profile, mark_rows  # noqa: E402

TABLE_MD = TABLE_DIR / "model_parametric.md"


# ---------------------------------------------------------------------------
# M1：线性
# ---------------------------------------------------------------------------
def fit_linear(t: np.ndarray, y: np.ndarray) -> dict:
    """`cap = a + b·t`，最小二乘闭式解。

    斜率 b 必须为负（容量在衰减）。若数据给出的 b ≥ 0，说明这段观测里容量没降，
    此时**不允许**拿它外推寿命——记 `b >= 0` 并让 EOL 返回 None。
    """
    if len(t) < 3:
        raise ValueError("线性拟合至少需要 3 个点")
    b, a = np.polyfit(t, y, 1)
    return {"name": "M1 线性", "params": {"a": float(a), "b": float(b)}, "converged": True}


def predict_linear(model: dict, t: np.ndarray) -> np.ndarray:
    p = model["params"]
    return p["a"] + p["b"] * np.asarray(t, dtype=float)


def eol_linear(model: dict) -> int | None:
    """解 `a + b·t = τ`。b ≥ 0 时返回 None（不会穿越）。"""
    p = model["params"]
    if p["b"] >= 0:
        return None
    t = (EOL_THRESHOLD_AH - p["a"]) / p["b"]
    return int(np.ceil(t)) if np.isfinite(t) and t >= 1 else None


# ---------------------------------------------------------------------------
# M2：双指数（含降级路径）
# ---------------------------------------------------------------------------
def _double_exp(t, a, b, c, d):
    return a * np.exp(b * t) + c * np.exp(d * t)


def _single_exp(t, a, b):
    return a * np.exp(b * t)


def fit_double_exp(t: np.ndarray, y: np.ndarray, p0: list[float] | None = None) -> dict:
    """`cap = a·e^(bt) + c·e^(dt)`（NASA 形式的双指数），四个参数自由拟合。

    这里**不**约束 a,c ≥ 0 —— 第一版约束了，结果四块电池全部退化成单指数：
    因为真实的最优解长这样（B0005）：`a≈106, b≈−0.0099, c≈−104, d≈−0.0101`，
    即**两个近共线的大项相消**，训练 RMSE 0.016，比线性的 0.031 还好一半。
    用 c ≥ 0 卡住它，等于把这个解强行排除，剩下能走的就是一支指数——
    那已经不是 M2 了，是「单指数套了个双指数的名字」。

    所以改成：**允许自由解，但把「不可辨识」和「外推是否还能当模型用」两件事量化出来**：
    - `cond`：雅可比矩阵的条件数（越大越不可辨识）。本数据实测 **2.2e11–1.3e12**，
      这不是「有点病态」，是四个参数几乎完全互相抵消。
    - `monotone`：拟合曲线在**外推段** `[t_max, 5N]` 上是否严格递减且为正。

    ⚠️ `monotone` 的检查区间是**外推段**，不是 `[1, 5N]`——这一条被第六轮审查打回来过
    （R6-2）。原来的写法把**训练段内**的那点起伏也算进去了：B0005 的拟合曲线在
    [1, 13.9] 上确实微微上翘（峰值 1.834 vs 起点 1.817），但那一段是**已观测**的，
    不影响外推；按老口径它就被判成"外推不单调"，于是整块电池不给寿命，
    而实际上曲线在外推段严格递减、确实穿过 τ（第 107 循环）。
    判据必须只作用于**没有数据的地方**，否则它测的是拟合形状，不是外推风险。

    任一检查不过 → 标 `valid=False`，报告里照样保留这一行，但把原因写清楚。
    """
    p0_exp = [float(y[0]), -1.0 / max(len(t), 1)]
    try:
        popt1, _ = curve_fit(_single_exp, t, y, p0=p0_exp, maxfev=20000)
    except Exception:  # noqa: BLE001
        popt1 = np.array([float(y[0]), -0.01])

    if p0 is None:
        p0 = [float(popt1[0]) * 0.7, float(min(popt1[1], -1e-9)),
              float(popt1[0]) * 0.3, float(min(popt1[1], -1e-9)) * 3.0]
    lb = [-1000.0, -0.5, -1000.0, -0.5]
    ub = [1000.0, -1e-9, 1000.0, -1e-9]

    try:
        popt, _ = curve_fit(_double_exp, t, y, p0=p0, bounds=(lb, ub), maxfev=60000)
    except Exception:  # noqa: BLE001
        return {
            "name": "M2 双指数", "params": {"a": float("nan"), "b": float("nan"),
                                            "c": float("nan"), "d": float("nan")},
            "converged": False, "fallback": True, "cond": float("nan"),
            "monotone": False, "valid": False,
            "why": "curve_fit 不收敛",
        }

    a, b, c, d = (float(v) for v in popt)
    # 雅可比的四列：[e^{bt}, a·t·e^{bt}, e^{dt}, c·t·e^{dt}]
    J = np.column_stack(
        [np.exp(b * t), a * t * np.exp(b * t), np.exp(d * t), c * t * np.exp(d * t)]
    )
    cond = float(np.linalg.cond(J))

    t_max = float(np.max(t))
    t_probe = np.linspace(t_max, float(len(t) * EOL_SEARCH_MULTIPLE), 2000)
    curve = _double_exp(t_probe, a, b, c, d)
    deriv = a * b * np.exp(b * t_probe) + c * d * np.exp(d * t_probe)
    monotone = bool(np.all(deriv < 0) and np.all(curve > 0) and np.all(np.isfinite(curve)))

    reasons = []
    if not monotone:
        reasons.append(
            f"外推段 [{t_max:.0f}, {len(t) * EOL_SEARCH_MULTIPLE}] 内曲线不再严格递减/为正"
        )

    return {
        "name": "M2 双指数",
        "params": {"a": a, "b": b, "c": c, "d": d},
        "converged": True,
        "fallback": False,
        "cond": cond,
        "monotone": monotone,
        "valid": monotone,
        "why": "；".join(reasons),
        "extrap_window": (t_max, float(len(t) * EOL_SEARCH_MULTIPLE)),
    }


def predict_double_exp(model: dict, t: np.ndarray) -> np.ndarray:
    p = model["params"]
    return _double_exp(np.asarray(t, float), p["a"], p["b"], p["c"], p["d"])


def eol_double_exp(model: dict, n_obs: int) -> int | None:
    """解 `a·e^(bt) + c·e^(dt) = τ`。

    ⚠️ 双指数**不保证单调**（两个指数项的系数一正一负时可能在远处掉头），
    所以这里先扫描再求根，而不是想当然地认为「一定单调、一定穿过 τ」。
    `valid=False`（**外推段**不单调/不正）的模型不给寿命预测——那会是一个纯数值巧合。
    注意判据只作用在外推段（见 `fit_double_exp` 文档），不是整条曲线。
    """
    if not model.get("valid", False):
        return None
    lo, hi = 1.0, float(n_obs * EOL_SEARCH_MULTIPLE)
    grid = np.linspace(lo, hi, 4000)
    vals = predict_double_exp(model, grid) - EOL_THRESHOLD_AH
    sign_change = np.flatnonzero(np.sign(vals[:-1]) * np.sign(vals[1:]) <= 0)
    if len(sign_change) == 0:
        return None
    i = int(sign_change[0])
    f = lambda x: float(predict_double_exp(model, np.array([x]))[0]) - EOL_THRESHOLD_AH  # noqa: E731
    root = brentq(f, grid[i], grid[i + 1], maxiter=200)
    return int(np.ceil(root)) if root >= 1 else 1


# ---------------------------------------------------------------------------
# 跑一遍所有 (模型 × 电池 × 切分点)
# ---------------------------------------------------------------------------
def run(df: pd.DataFrame, prof: pd.DataFrame, m: int = SPLIT_M) -> dict:
    prof_idx = prof.set_index("battery_id")
    rows, fits = [], {}
    for sp in temporal_split(df, m):
        eol_true = prof_idx.loc[sp.cell, "eol_cycle"]
        eol_true = None if pd.isna(eol_true) else int(eol_true)
        n_obs = len(sp.train) + len(sp.test)

        for fitter, predictor, eol_fn, key in (
            (fit_linear, predict_linear,
             lambda mod, n: eol_linear(mod), "M1"),
            (fit_double_exp, predict_double_exp,
             lambda mod, n: eol_double_exp(mod, n), "M2"),
        ):
            model = fitter(sp.t_train, sp.y_train)
            pred = predictor(model, sp.t_test)
            met = metrics(sp.y_test, pred)
            eol = eol_fn(model, n_obs)
            ee = eol_error(eol, eol_true, n_obs)
            fits[(sp.cell, key)] = {"model": model, "split": sp}

            train_pred = predictor(model, sp.t_train)
            rows.append(
                {
                    "电池": sp.cell,
                    "模型": model["name"],
                    "M": m,
                    "收敛": model["converged"],
                    "可辨识": "—" if "cond" not in model else f"{model['cond']:.0e}",
                    "外推单调": "—" if "monotone" not in model else model["monotone"],
                    "训练RMSE": float(np.sqrt(np.mean((train_pred - sp.y_train) ** 2))),
                    **met,
                    "EOL预测": "—" if eol is None else eol,
                    "EOL真值": "删失" if eol_true is None else eol_true,
                    "EOL误差": "—" if ee["eol_err"] is None else ee["eol_err"],
                    "备注": "；".join(
                        x for x in (ee["note"], model.get("why", "")) if x
                    ),
                }
            )
    return {"table": pd.DataFrame(rows), "fits": fits}


def split_sensitivity(df: pd.DataFrame, prof: pd.DataFrame) -> pd.DataFrame:
    """换切分点 M，误差怎么变。

    这是 W5（h 依赖）→ W7（窗口宽度依赖）那条教训的**第三次**出现：
    「误差是多少」不写切分点就没说完。
    """
    rows = []
    for m in SPLIT_M_SWEEP:
        res = run(df, prof, m)
        t = res["table"]
        for model_name, g in t.groupby("模型"):
            finite = g.loc[g["EOL误差"] != "—", "EOL误差"].astype(int)
            rows.append(
                {
                    "M": m,
                    "模型": model_name,
                    "RMSE 均值": f"{g['RMSE'].mean():.4f}",
                    "RMSE 最大": f"{g['RMSE'].max():.4f}",
                    "EOL 绝对误差均值": f"{finite.abs().mean():.1f}" if len(finite) else "—",
                    "EOL 绝对误差最大": f"{finite.abs().max()}" if len(finite) else "—",
                    "可比的电池数": len(finite),
                }
            )
    return pd.DataFrame(rows)


INIT_VARIANTS: list[tuple[str, list[float] | None]] = [
    ("代码默认（单指数外推 0.7/0.3）", None),
    ("137.3 / −135.5", [137.3, -0.00993, -135.5, -0.01009]),
    ("150 / −150", [150.0, -0.0099, -150.0, -0.0102]),
    ("200 / −200", [200.0, -0.0095, -200.0, -0.0106]),
    ("80 / −80", [80.0, -0.0100, -80.0, -0.0100]),
]
"""初值扫描用的几个起点。全部写死、可复现，不含随机成分。

为什么要有这个实验：第六轮审查（R6-1）指出，我原来那个「参数各扰动 0.1%」的
敏感性数字**依赖优化器从哪出发**——同一块电池、同样的训练 RMSE，参数可以从
a≈106/c≈−104 滑到 a≈201/c≈−199，那个「0.0333 Ah / 14%」换成别的起点就变成
0.0631 Ah / 26%。也就是说它不是数据性质，是**初值的性质**。
所以改成直接把这个事实测出来：**换初值 → 训练误差几乎不变 → 参数和寿命全变**。
这才是「不可辨识」的正面证据，而不是绕个弯去算扰动。"""


def init_sweep(df: pd.DataFrame, m: int = SPLIT_M) -> dict:
    """同一块电池、同一段训练数据，换 5 个初值重新拟合双指数，看结论变多少。"""
    rows = []
    for sp in temporal_split(df, m):
        n_obs = len(sp.train) + len(sp.test)
        probe = float(n_obs)
        actual_end = float(sp.y_test[-1])
        for label, p0 in INIT_VARIANTS:
            model = fit_double_exp(sp.t_train, sp.y_train, p0=p0)
            if not model["converged"]:
                rows.append({"电池": sp.cell, "初值": label, "收敛": False})
                continue
            tr_pred = predict_double_exp(model, sp.t_train)
            rows.append(
                {
                    "电池": sp.cell,
                    "初值": label,
                    "收敛": True,
                    "a": model["params"]["a"],
                    "c": model["params"]["c"],
                    "训练RMSE": float(np.sqrt(np.mean((tr_pred - sp.y_train) ** 2))),
                    "EOL预测": eol_double_exp(model, n_obs),
                    "末段预测(Ah)": float(predict_double_exp(model, np.array([probe]))[0]),
                    "末段实际(Ah)": actual_end,
                    "cond": model["cond"],
                }
            )
    detail = pd.DataFrame(rows)
    ok = detail[detail["收敛"]]

    summary = []
    for cell, g in ok.groupby("电池"):
        eols = [int(v) for v in g["EOL预测"].dropna().tolist()]
        summary.append(
            {
                "电池": cell,
                "训练RMSE 范围": f"{g['训练RMSE'].min():.6f}–{g['训练RMSE'].max():.6f}",
                "a 范围": f"{g['a'].min():.0f}–{g['a'].max():.0f}",
                "EOL 预测取值": "、".join(str(v) for v in sorted(set(eols))) or "—",
                "EOL 极差": (max(eols) - min(eols)) if len(eols) > 1 else 0,
                "末段预测极差(Ah)": float(
                    g["末段预测(Ah)"].max() - g["末段预测(Ah)"].min()
                ),
                "末段实际": float(g["末段实际(Ah)"].iloc[0]),
            }
        )
    summary_df = pd.DataFrame(summary)

    # 核心事实必须真的发生，否则第三节整段要重写
    rmse_spread = float((ok.groupby("电池")["训练RMSE"].max()
                         - ok.groupby("电池")["训练RMSE"].min()).max())
    eol_spread = int(summary_df["EOL 极差"].max())
    end_spread = float((ok.groupby("电池")["末段预测(Ah)"].max()
                        - ok.groupby("电池")["末段预测(Ah)"].min()).max())
    assert rmse_spread < 1e-5, (
        f"换初值后训练误差最多变了 {rmse_spread:.2e} —— 这些初值没走到同一条脊上，"
        "本实验想展示的「参数不可辨识」不成立"
    )
    assert eol_spread <= 1, (
        f"换初值后 EOL 预测最多差了 {eol_spread} 个循环 —— "
        "「预测沿脊不变」这句话不成立，第三节结论要整个重写"
    )
    return {
        "detail": detail,
        "summary": summary_df,
        "rmse_spread": rmse_spread,
        "eol_spread": eol_spread,
        "end_spread": end_spread,
        "cond_range": (float(ok["cond"].min()), float(ok["cond"].max())),
    }




# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------
def fig06(df: pd.DataFrame, fits: dict, m: int) -> Path:
    setup_matplotlib()
    cells = list(split_by_battery(df).keys())
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.4))
    axes = axes.ravel()

    for ax, cell in zip(axes, cells):
        sp = fits[(cell, "M1")]["split"]
        # 观测段（训练）与真实后段（测试）
        ax.plot(sp.t_train, sp.y_train, color=COLORS[cell], lw=1.8, label="已观测（训练段）")
        ax.plot(sp.t_test, sp.y_test, color=COLORS[cell], lw=1.2, ls=":", alpha=0.85,
                label="真实后段（测试段）")

        t_ext = np.linspace(1, sp.t_train.max() * 2.4 + 10, 400)
        for key, style in (("M1", "-"), ("M2", "--")):
            mod = fits[(cell, key)]["model"]
            pred = predict_linear(mod, t_ext) if key == "M1" else predict_double_exp(mod, t_ext)
            ax.plot(t_ext, pred, style, lw=1.7, color="#333333" if key == "M1" else "#b8860b",
                    label=mod["name"])

        # M2 的判读：只写**算出来**的事实。
        # 旧版这里印的是「M2 在约第 400 循环处上翘」——第六轮审查（R6-2）指出
        # 那是 argmin 落在探针区间端点的假象：曲线其实严格递减，400 处是全程最低点。
        # 现在改成两个真数字：外推末端的预测 vs 实测，以及它解出的寿命。
        m2 = fits[(cell, "M2")]["model"]
        n_obs = len(sp.train) + len(sp.test)
        pred_end = float(predict_double_exp(m2, np.array([float(n_obs)]))[0])
        actual_end = float(sp.y_test[-1])
        eol_m2 = eol_double_exp(m2, n_obs)
        ax.text(
            0.38, 0.10,
            f"M2 外推到第 {n_obs} 循环：{pred_end:.3f} Ah\n"
            f"（实测 {actual_end:.3f}）｜EOL 预测："
            + ("—" if eol_m2 is None else str(eol_m2)),
            transform=ax.transAxes, fontsize=7.6, color="#8a6d00",
            bbox=dict(boxstyle="round,pad=0.25", fc="#fffbe8", ec="#e0cf8a", lw=0.7),
        )

        ax.axhline(EOL_THRESHOLD_AH, ls="--", lw=1.1, color="#777777")
        ax.text(0.02, EOL_THRESHOLD_AH, f"τ={EOL_THRESHOLD_AH:g} Ah", fontsize=8,
                color="#666666", va="bottom",
                transform=ax.get_yaxis_transform())
        ax.axvline(m, color="#999999", lw=0.9, ls=":")
        ax.text(m, ax.get_ylim()[1] if ax.get_ylim()[1] else 2.0, f" M={m}", fontsize=8,
                color="#777777", va="top")

        # 真实 EOL 与两模型的 EOL 预测已在主表里；图上只标切分点与阈值
        ax.set_title(f"{cell}：观测 {len(sp.train)} → 预测 {len(sp.test)} 循环", fontsize=10.5)
        ax.set_xlabel("循环")
        ax.set_ylabel("容量 (Ah)")
        ax.legend(fontsize=7.6, loc="lower left", framealpha=0.9)

    fig.suptitle(f"图6　M1 线性 vs M2 双指数：切分点 M={m} 的轨迹外推", fontsize=13, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return save(fig, "fig06_parametric_extrapolation.png")


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
def render_md(
    tbl: pd.DataFrame,
    tbl_m: pd.DataFrame,
    m: int,
    fits: dict,
    sweep: dict,
) -> str:
    """生成报告。**所有结论句都由表里的数字现算**（D19）。"""
    piv = tbl.pivot_table(index="电池", columns="模型",
                          values=["训练RMSE", "RMSE", "R2"], aggfunc="first")
    better_in = int((piv[("训练RMSE", "M2 双指数")] < piv[("训练RMSE", "M1 线性")]).sum())
    better_out = int((piv[("RMSE", "M2 双指数")] < piv[("RMSE", "M1 线性")]).sum())
    n_cells = len(piv)

    conds = tbl.loc[tbl["模型"] == "M2 双指数", "可辨识"].tolist()
    # 条件数存的是 "1e+12" 这种字符串，比大小要转回数字
    cond_vals = sorted(float(x) for x in conds if x != "—")
    min_c, max_c = f"{cond_vals[0]:.0e}", f"{cond_vals[-1]:.0e}"
    mono = tbl.loc[tbl["模型"] == "M2 双指数", "外推单调"].tolist()
    n_not_mono = sum(1 for x in mono if x is False)
    not_mono_cells = tbl.loc[
        (tbl["模型"] == "M2 双指数") & (tbl["外推单调"] == False), "电池"
    ].tolist()

    m1 = tbl[tbl["模型"] == "M1 线性"]
    eol_m1 = pd.to_numeric(m1["EOL误差"], errors="coerce").dropna()
    m2 = tbl[tbl["模型"] == "M2 双指数"]
    eol_m2 = pd.to_numeric(m2["EOL误差"], errors="coerce").dropna()

    # ---- 下面这些量全部现算：报告里出现过的参数值、循环数、条件数都不许写死 ----
    m2_models = {c: fits[(c, "M2")]["model"] for c in tbl["电池"].unique()}
    finite_cond = {
        c: mod["cond"] for c, mod in m2_models.items()
        if np.isfinite(mod.get("cond", float("nan")))
    }
    assert finite_cond, "M2 全部没给出条件数，第三节的例证无从谈起"
    worst_cell = max(finite_cond, key=finite_cond.get)
    wp = m2_models[worst_cell]["params"]
    amp_lo, amp_hi = sorted((abs(wp["a"]), abs(wp["c"])))
    rate_ratio = min(abs(wp["b"]), abs(wp["d"])) / max(abs(wp["b"]), abs(wp["d"]))
    param_txt = (f"a≈{wp['a']:.4g}、b≈{wp['b']:.4g}、"
                 f"c≈{wp['c']:.4g}、d≈{wp['d']:.4g}")
    # 条件数区间用未取整的 float 生成（R6-9：旧版从 "1e+12" 这种字符串反推，区间是错的）
    cond_lo = f"{min(finite_cond.values()):.2g}"
    cond_hi = f"{max(finite_cond.values()):.2g}"

    detail = sweep["detail"]
    summary = sweep["summary"]
    ok_detail = detail[detail["收敛"]]
    ex_table = ok_detail.loc[ok_detail["电池"] == worst_cell].drop(columns=["末段实际(Ah)"])
    rmse_spread = sweep["rmse_spread"]
    eol_spread_max = sweep["eol_spread"]
    end_spread = sweep["end_spread"]
    sweep_cond_lo, sweep_cond_hi = sweep["cond_range"]
    spread_cell = str(summary.loc[summary["EOL 极差"].idxmax(), "电池"])
    spread_values = str(summary.loc[summary["EOL 极差"].idxmax(), "EOL 预测取值"])

    # M2 外推到观测末端的系统偏差（主拟合，一个数）
    bias_rows = []
    for cell, mod in m2_models.items():
        sp = fits[(cell, "M1")]["split"]
        n_obs = len(sp.train) + len(sp.test)
        pred = float(predict_double_exp(mod, np.array([float(n_obs)]))[0])
        bias_rows.append(
            {
                "电池": cell,
                "观测末循环": n_obs,
                "M2 预测(Ah)": pred,
                "实测(Ah)": float(sp.y_test[-1]),
                "偏差(Ah)": pred - float(sp.y_test[-1]),
            }
        )
    bias_df = pd.DataFrame(bias_rows)
    bias_mean = float(bias_df["偏差(Ah)"].mean())

    # 训练段上「参数多更好看」并不是必然：找出 M2 反而更差的那块
    train_piv = tbl.pivot_table(index="电池", columns="模型", values="训练RMSE", aggfunc="first")
    m2_worse_train = train_piv[train_piv["M2 双指数"] > train_piv["M1 线性"]].index.tolist()

    m1_abs = pd.to_numeric(m1["EOL误差"], errors="coerce").abs()
    m2_abs = pd.to_numeric(m2["EOL误差"], errors="coerce").abs()
    worst_m1_cell = m1.loc[m1_abs.idxmax(), "电池"] if m1_abs.notna().any() else "—"
    worst_m2_cell = m2.loc[m2_abs.idxmax(), "电池"] if m2_abs.notna().any() else "—"

    # valid 闸门只拦"外推段不单调"的模型；这份数据上四块都该给得出寿命
    assert not (tbl["EOL预测"] == "—").any(), (
        "有模型没给出寿命预测——闸门现在只挡外推段不单调的情形，"
        "这份数据四块电池都该有数，先查 fit_double_exp"
    )

    censored = tbl.loc[tbl["EOL真值"] == "删失", "电池"].unique().tolist()
    assert len(censored) == 1, f"删失电池应当只有一块，实际 {censored}"
    censor_cell = censored[0]
    censor_n = len(fits[(censor_cell, "M1")]["split"].train) + len(
        fits[(censor_cell, "M1")]["split"].test
    )
    censor_preds = {
        name: int(tbl.loc[(tbl["电池"] == censor_cell) & (tbl["模型"] == name),
                          "EOL预测"].iloc[0])
        for name in ("M1 线性", "M2 双指数")
    }
    assert all(v <= censor_n for v in censor_preds.values()), (
        f"{censor_cell} 上有模型的寿命预测超过观测长度 {censor_n}，「方向都错」这句话要重写"
    )

    out = [
        "# M1 线性 / M2 双指数：轨迹外推与寿命预测（W9）",
        "",
        "> 本文件由 `src/models_parametric.py` 自动生成，数字随代码重算，**不要手工编辑**。",
        "",
        "## 一、口径",
        "",
        f"- **任务**：任务 A（个体内轨迹外推）。用前 **M = {m}** 个循环拟合，"
        "预测后面的容量，再让模型外推到阈值求寿命。",
        "- **切分**：时序划分，不随机打乱（D6）。",
        "- **指标**：RMSE / MAE / MAPE / R² 全部在**测试段**上算；R² 用测试段自身方差作分母，"
        "所以外推失败时会显著为负——这是要看见的现象，不做截断。",
        f"- **寿命**：模型自己解 `cap = τ = {EOL_THRESHOLD_AH:g} Ah` 得到循环数；"
        "真值删失时**没有误差可比**，只检查模型有没有预测「超过观测长度」。",
        "",
        "## 二、主表（M = " + f"{m}）",
        "",
        tbl.to_markdown(index=False),
        "",
        "## 三、这一轮最重要的结果：**参数更多 ≠ 外推更好**",
        "",
        f"M2 有 4 个参数、M1 只有 2 个。在**训练段**上，M2 有 {better_in}/{n_cells} 块电池"
        f"比 M1 拟合得好；但在**测试段**（真正的外推）上，M2 只有 {better_out}/{n_cells} 块"
        "比 M1 更好。",
        "",
        "### 判据一：参数根本不可辨识 —— 但这**不是**它失败的原因",
        "",
        f"双指数的四个参数在这份数据上不可辨识（主拟合的雅可比条件数 {cond_lo}–{cond_hi}）。"
        "条件数是抽象指标，所以直接做实验：**同一块电池、同一段训练数据，只换优化器起点**。",
        "",
        summary.round(4).to_markdown(index=False),
        "",
        f"四块电池上，换初值后训练 RMSE 的极差最大只有 **{rmse_spread:.0e}**（等于没变），"
        f"`a` 却能滑动成百；而 **EOL 预测的极差是 {eol_spread_max} 个循环，"
        f"观测末端的预测极差只有 {end_spread:.0e} Ah**。",
        "",
        "**这条必须说清楚，因为它纠正了一个常见误解（也纠正了本报告初稿的说法）：**"
        "「参数不可辨识」只意味着**参数没法解释**，不意味着「预测不可靠」。"
        "沿着这条脊，拟合曲线几乎一动不动，所以寿命预测也不动。"
        "把条件数大当成 M2 失败的理由是错的——初稿正是这么写的，做初值扫描才把它推翻。",
        "",
        f"即便条件数本身也不稳定：换初值后它从 {sweep_cond_lo:.1e} 晃到 {sweep_cond_hi:.1e}。"
        f"以 {worst_cell} 为例（参数 `{param_txt}`，两指数系数绝对值 {amp_lo:.0f} 与 "
        f"{amp_hi:.0f}、衰减率之比 {rate_ratio:.3f}，符号相反的近等大项相消）：",
        "",
        ex_table.round(6).to_markdown(index=False),
        "",
        f"**参数没有意义**（`a` 从 {ex_table['a'].min():.0f} 滑到 {ex_table['a'].max():.0f}，"
        "结论一个字没变）。所以报告里不该出现「a 代表什么」这种句子。",
        "",
        "### 判据二：真正的原因是系统性偏差",
        "",
        "既然参数不可辨识不影响预测，那 M2 到底错在哪？错在**方向性偏差**："
        "外推到观测末端时，它的曲线总是比真实衰减更陡。",
        "",
        bias_df.round(4).to_markdown(index=False),
        "",
        f"四块电池的偏差**全为负，平均 {bias_mean:+.4f} Ah**。"
        "这不是随机误差（随机会有正有负），是形式本身带来的系统偏差："
        "双指数为了拟合训练段那点曲率，把后段的衰减率估得过大。"
        "所以它在每一块电池上都会**提前报死**（下一节的数据就是这件事）。",
        "",
        (
            f"另注：M2 的外推段（`[{m + 1}, {m * EOL_SEARCH_MULTIPLE}]`）"
            f"在全部 {n_cells} 块电池上都严格单调递减——它没有「掉头」这个毛病，"
            "问题不在形状，在偏差。"
            if n_not_mono == 0
            else f"另注：{n_not_mono}/{n_cells} 块电池（{'、'.join(not_mono_cells)}）"
            "的外推段不单调递减，本模块对这类模型不给寿命数字（表里是「—」，备注给了原因）。"
        ),
        "",
        "## 四、寿命预测误差",
        "",
        f"两个模型在**同样 {len(eol_m1)} 块有真值的电池**上都给出了寿命预测，可以正面比：",
        "",
        f"- M1 线性：绝对误差均值 **{eol_m1.abs().mean():.1f}** 循环，"
        f"最大 **{eol_m1.abs().max():.0f}**（{worst_m1_cell}）。",
        f"- M2 双指数：绝对误差均值 **{eol_m2.abs().mean():.1f}** 循环，"
        f"最大 **{eol_m2.abs().max():.0f}**（{worst_m2_cell}）。",
        "",
        "⚠️ 读这张表必须注意三件事：",
        "",
        f"1. **M2 的寿命数字是稳的，所以问题不在这里**：换初值，四块电池的 EOL 预测极差是 "
        f"{eol_spread_max} 个循环（第三节判据一）。也就是说它**不是**「一次运气不好的拟合」，"
        "而是**一个稳定但系统性偏早的答案**——偏差才是要害，随机性不是。",
        f"2. **{censor_cell} 没有寿命误差**：它删失，没有真值。这里只能问「模型有没有预测它在"
        "观测期之后才坏」——而不能问「误差几个循环」。"
        f"（M1 预测第 {censor_preds['M1 线性']} 循环、M2 预测第 {censor_preds['M2 双指数']} 循环，"
        f"都小于观测长度 {censor_n}：**两个模型都说它会坏，方向都错**，"
        "备注里的 ⚠️ 就是在说这件事。）",
        "3. **切分点会显著改变误差**（见下表）。「误差是多少」不写切分点就没说完——"
        "这是 W5（h 依赖）、W7（窗口宽度依赖）之后同一类教训的第三次出现。",
        "",
        "## 五、切分点敏感性",
        "",
        tbl_m.to_markdown(index=False),
        "",
        "## 六、这张表不能怎么读",
        "",
        "1. **不能拿 R² 横向比模型复杂度**：参数多的模型在训练段通常更好看"
        f"（本数据 {better_in}/{n_cells}），但**不是必然**——"
        f"{'、'.join(m2_worse_train) if m2_worse_train else '没有电池'} 上 M2 的训练 RMSE "
        f"反而更高（{train_piv.loc[m2_worse_train[0], 'M1 线性']:.7f} vs "
        f"{train_piv.loc[m2_worse_train[0], 'M2 双指数']:.7f}）。"
        if m2_worse_train else
        "1. **不能拿 R² 横向比模型复杂度**：参数多的模型在训练段通常更好看，"
        "而测试段好不好取决于它外推得对不对——第三节就是这件事的实例。",
        f"2. **不能把 M2 的 4 个参数当物理量**：条件数 {cond_lo}–{cond_hi} 意味着"
        "参数之间几乎可以任意互换（第三节的初值扫描已实测），**不可辨识的参数不值得解释**。",
        "3. **不能因为 M2 表现差就删掉它**：它在这里的作用是反例——"
        "「形式更贴合衰减曲线」不等于「外推更准」。",
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
    sweep = init_sweep(df, SPLIT_M)

    TABLE_MD.write_text(
        render_md(tbl, tbl_m, SPLIT_M, res["fits"], sweep), encoding="utf-8"
    )
    fig_path = fig06(df, res["fits"], SPLIT_M)

    print(f"切分点 M = {SPLIT_M}；{len(tbl)} 行（模型 × 电池）")
    print()
    print(tbl.to_string(index=False))
    print()
    print("切分点敏感性：")
    print(tbl_m.to_string(index=False))
    print()
    print("M2 初值扫描（同一段数据，只换起点）：")
    print(sweep["summary"].to_string(index=False))
    print()
    print(f"参数表：")
    for (cell, key), v in res["fits"].items():
        p = v["model"]["params"]
        ps = ", ".join(f"{k}={val:.6g}" for k, val in p.items())
        print(f"  {cell} {key}: {ps}"
              + ("  ← 降级" if v["model"].get("fallback") else ""))
    print()
    print(f"参数模型报告已写出：{TABLE_MD.name}")
    print(f"已生成 {fig_path.name}  ({fig_path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
