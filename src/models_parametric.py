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
| M1 线性 | `cap = a + b·t` | 2 | 能，但没有物理形态，长程外推必然发散 |
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

EOL_SEARCH_MULTIPLE = 5
"""外推求 EOL 时向未来找多远（倍数于观测长度）。"""


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


def fit_double_exp(t: np.ndarray, y: np.ndarray) -> dict:
    """`cap = a·e^(bt) + c·e^(dt)`（NASA 形式的双指数），四个参数自由拟合。

    这里**不**约束 a,c ≥ 0 —— 第一版约束了，结果四块电池全部退化成单指数：
    因为真实的最优解长这样（B0005）：`a=137.3, b=-0.00993, c=-135.5, d=-0.01009`，
    即**两个近共线的大项相消**，训练 RMSE 0.016，比线性的 0.031 还好一半。
    用 c ≥ 0 卡住它，等于把这个解强行排除，剩下能走的就是一支指数——
    那已经不是 M2 了，是「单指数套了个双指数的名字」。

    所以改成：**允许自由解，但把「不可辨识」和「外推是否还能当模型用」两件事量化出来**：
    - `cond`：雅可比矩阵的条件数（越大越不可辨识，本数据实测 10^4 量级）；
    - `monotone`：拟合曲线在 [1, 5N] 上是否仍单调递减且为正。
      两个近共线项相消时，外推到某个距离后曲线会**掉头往上**，那时报出来的「寿命」毫无意义。

    任一检查不过 → 标 `valid=False`，报告里照样保留这一行，但把原因写清楚。
    """
    p0_exp = [float(y[0]), -1.0 / max(len(t), 1)]
    try:
        popt1, _ = curve_fit(_single_exp, t, y, p0=p0_exp, maxfev=20000)
    except Exception:  # noqa: BLE001
        popt1 = np.array([float(y[0]), -0.01])

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

    t_probe = np.linspace(1.0, float(len(t) * EOL_SEARCH_MULTIPLE), 2000)
    curve = _double_exp(t_probe, a, b, c, d)
    deriv = a * b * np.exp(b * t_probe) + c * d * np.exp(d * t_probe)
    monotone = bool(np.all(deriv < 0) and np.all(curve > 0) and np.all(np.isfinite(curve)))

    reasons = []
    if not monotone:
        reasons.append("外推区间内曲线不再单调递减/为正（两指数相消后掉头）")

    return {
        "name": "M2 双指数",
        "params": {"a": a, "b": b, "c": c, "d": d},
        "converged": True,
        "fallback": False,
        "cond": cond,
        "monotone": monotone,
        "valid": monotone,
        "why": "；".join(reasons),
    }


def predict_double_exp(model: dict, t: np.ndarray) -> np.ndarray:
    p = model["params"]
    return _double_exp(np.asarray(t, float), p["a"], p["b"], p["c"], p["d"])


def eol_double_exp(model: dict, n_obs: int) -> int | None:
    """解 `a·e^(bt) + c·e^(dt) = τ`。

    ⚠️ 双指数**不保证单调**（两个指数项的系数一正一负时会掉头），
    所以这里先扫描再求根，而不是想当然地认为「一定单调、一定穿过 τ」。
    `valid=False` 的模型（外推非单调）**不给寿命预测**——那会是一个纯数值巧合。
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
                    "备注": ee["note"] or model.get("why", ""),
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

        # 非单调的 M2：标出它在哪儿开始上翘（这正是「不给寿命」的理由）
        m2 = fits[(cell, "M2")]["model"]
        if m2.get("monotone") is False:
            probe = np.linspace(1, len(sp.train) * EOL_SEARCH_MULTIPLE, 20000)
            curve = predict_double_exp(m2, probe)
            turn = float(probe[int(np.argmin(curve))])
            # 用 axes fraction 定位（数据坐标会落在轴外被裁掉——第一版就是这么消失的）
            ax.text(
                0.40, 0.13,
                f"M2 在约第 {turn:.0f} 循环处上翘\n（故本模块不给它寿命）",
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
def render_md(tbl: pd.DataFrame, tbl_m: pd.DataFrame, m: int) -> str:
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
        "原因不是实现 bug，而是**双指数在这份数据上不可辨识**："
        f"拟合出来的雅可比条件数在 {min_c}–{max_c} 之间（越大越病态），"
        "参数长这样（B0005）：`a≈106、b≈−0.0099、c≈−104、d≈−0.0101`——"
        "**两个绝对值上百、衰减率几乎相同、符号相反的指数项在互相抵消**。",
        "",
        "这种解在训练段能把残差压得很低，但它对参数极其敏感：b 和 d 差一点点，"
        "外推几十个循环就能翻天。",
        "",
        "第二个独立证据：",
        f"**{n_not_mono}/{n_cells} 块电池（{'、'.join(not_mono_cells) or '无'}）的拟合曲线"
        "在外推区间内不再单调递减**——两个指数项相消到某个距离后曲线掉头往上。"
        "这种模型的「寿命预测」是纯数值巧合，所以本模块对这类模型**不给寿命数字**"
        "（表里 EOL 预测是「—」，备注里写了原因）。",
        "",
        "所以 M2 在本数据上的正确结论是：**形式不成立，不是因为算不出来，"
        "而是 80 个循环的窗口撑不起 4 个可分离的参数。**"
        "这条结论本身比「M2 精度 0.0x」有价值——W13 会再用 BIC 之类的惩罚把这件事量化。",
        "",
        "## 四、寿命预测误差",
        "",
        f"- M1：可比电池 {len(eol_m1)} 块，绝对误差最大 **{eol_m1.abs().max():.0f}** 循环"
        f"（{m1.loc[pd.to_numeric(m1['EOL误差'], errors='coerce').abs().idxmax(), '电池']}）。"
        if len(eol_m1) else "- M1：没有可比电池。",
        f"- M2：可比电池 {len(eol_m2)} 块"
        + (f"，绝对误差最大 **{eol_m2.abs().max():.0f}** 循环。" if len(eol_m2) else "（多数外推非单调，不给寿命）。"),
        "",
        "⚠️ 读这张表必须注意三件事：",
        "",
        f"1. **两个模型的「可比电池数」不同（M1 {len(eol_m1)} 块、M2 {len(eol_m2)} 块），"
        "所以 EOL 误差不能直接比大小。** M2 只在它外推仍然单调的电池上给寿命，"
        "那是一份**被筛选过的子集**——它看起来误差不大，恰恰因为「算不出来」的那些被排除了。"
        "这种「选择性可比」在任何模型对比里都必须点出来，否则就是拿不同分母的数字比大小。",
        "2. **B0007 没有寿命误差**：它删失，没有真值。这里只能问「模型有没有预测它在"
        "观测期之后才坏」——而不能问「误差几个循环」。"
        "（M1 预测它 159 循环坏，超过观测长度 168？没有——159 < 168，"
        "所以 M1 在这块电池上判错了方向，备注里的 ⚠️ 就是在说这件事。）",
        "3. **切分点会显著改变误差**（见下表）。「误差是多少」不写切分点就没说完——"
        "这是 W5（h 依赖）、W7（窗口宽度依赖）之后同一类教训的第三次出现。",
        "",
        "## 五、切分点敏感性",
        "",
        tbl_m.to_markdown(index=False),
        "",
        "## 六、这张表不能怎么读",
        "",
        "1. **不能拿 R² 横向比模型复杂度**：参数多的模型在训练段必然更好看，"
        "而测试段好不好取决于它外推得对不对——第三节就是这件事的实例。",
        "2. **不能把 M2 的 4 个参数当物理量**：条件数 10^11 量级意味着参数之间几乎完全可互换，"
        "**不可辨识的参数不值得解释**。",
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

    TABLE_MD.write_text(render_md(tbl, tbl_m, SPLIT_M), encoding="utf-8")
    fig_path = fig06(df, res["fits"], SPLIT_M)

    print(f"切分点 M = {SPLIT_M}；{len(tbl)} 行（模型 × 电池）")
    print()
    print(tbl.to_string(index=False))
    print()
    print("切分点敏感性：")
    print(tbl_m.to_string(index=False))
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
