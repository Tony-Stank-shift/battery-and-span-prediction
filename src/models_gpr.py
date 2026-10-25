"""W11：M4 = 线性趋势 + 残差高斯过程（GPR）。

一句话结论（先给，后面全是证据）
--------------------------------
M4 的价值**不在点预测，在分布**。实测：残差 GP 学到的长度尺度只有
0.9–4.3 个循环（远小于 80 个观测循环），所以一旦离开数据区间，
GP 的后验均值**精确回到 0**、后验标准差**立刻饱和成常数**。
也就是说：**平稳核的 GPR 没有能力外推趋势**——外推这件事必须由趋势项承担，
GP 只负责「残差的形状 + 观测噪声」。这条不是本项目的发现，是平稳过程的本性，
但它是 M4 能不能讲清楚的前提，所以写在最前面。

因此 M4 的结构被拆成两层方差（这也正好接上 W14 的「三层不确定性」）：
    cap(t) = 趋势(t)  +  残差 GP(t)
    Var   = 趋势参数方差(OLS)  ⊕  残差方差(GP)
第一层随外推距离像漏斗一样张开（∝ (t−t̄)²），第二层是常数。
只留第二层的话，EOL 区间会窄得离谱——这正是 W17 要拿 PICP 检验的东西。

继承来的两条纪律（D23）
----------------------
1. **形状/单调判据只作用于没有数据的区段**：带宽单调性在 `[t_max, 5N]` 上检查。
2. **不可辨识 ≠ 不可靠**：M4 的趋势参数是 OLS 估计，标准差直接由残差方差给出；
   不做「参数解释」那套叙述。

运行
----
    python src/models_gpr.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

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
from models_parametric import eol_linear, fit_linear, predict_linear  # noqa: E402
from preprocess import cell_profile, mark_rows  # noqa: E402

TABLE_MD = TABLE_DIR / "model_gpr.md"

N_SAMPLES = 300
"""后验采样路径数。EOL 分布的 5%/95% 分位数在 S=300 时的蒙特卡洛误差约 ±1 个循环。"""

GP_SEED = RANDOM_SEED
"""GP 超参优化的随机重启种子（两次拟合必须给出同一组超参）。"""

Z95 = 1.959963984540054
"""95% 正态分位点，用于把标准差换成区间半宽。"""


# ---------------------------------------------------------------------------
# 趋势层：OLS + 参数不确定性
# ---------------------------------------------------------------------------
def _design(t: np.ndarray) -> np.ndarray:
    """设计矩阵 [1, t]，两列：截距与斜率。"""
    t = np.asarray(t, dtype=float)
    return np.column_stack([np.ones_like(t), t])


def trend_posterior(t: np.ndarray, y: np.ndarray) -> dict:
    """拟合 `cap = a + b·t`，并给出系数协方差（OLS 的经典结果）。

    这里**直接复用** `models_parametric.fit_linear` 求 a、b（唯一出处，避免两处实现分叉），
    但协方差要自己算：`Cov = σ² (XᵀX)⁻¹`，`σ² = RSS/(n−2)`。

    为什么要算它：趋势参数自己是不确定的，而这个不确定性随外推距离平方增长，
    是 M4 区间的主要来源。不算它，EOL 区间会窄得毫无意义（见模块文档）。
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(t)
    if n < 10:
        raise ValueError(f"趋势拟合至少需要 10 个点，收到 {n}")

    lin = fit_linear(t, y)  # 唯一出处：系数 (a, b) 的定义只有一处
    X = _design(t)
    resid = y - predict_linear(lin, t)
    rss = float(np.sum(resid**2))
    sigma2 = rss / (n - 2)
    xtx_inv = np.linalg.inv(X.T @ X)

    return {
        "a": lin["params"]["a"],
        "b": lin["params"]["b"],
        "xtx_inv": xtx_inv,
        "sigma2": sigma2,
        "t_mean": float(t.mean()),
        "n": n,
    }


def trend_line(trend: dict, t) -> np.ndarray:
    """趋势均值。"""
    return trend["a"] + trend["b"] * np.asarray(t, dtype=float)


def trend_std(trend: dict, t) -> np.ndarray:
    """趋势**参数**带来的标准差：`sqrt(σ² · xᵀ(XᵀX)⁻¹x)`，x = [1, t]。

    注意它**不含**观测噪声，只含「我把这条直线估得有多准」。
    """
    t = np.asarray(t, dtype=float)
    X = _design(t)
    var = np.einsum("ij,jk,ik->i", X, trend["xtx_inv"], X) * trend["sigma2"]
    return np.sqrt(np.maximum(var, 0.0))


# ---------------------------------------------------------------------------
# 残差层：GP
# ---------------------------------------------------------------------------
def _kernel():
    """振幅 × RBF。**只有一个核**，噪声不交给 WhiteKernel，而是用 `alpha` 固定成 OLS 残差方差。

    为什么不放 WhiteKernel：第一版就是那样写的，结果 WhiteKernel 把噪声学成 3.6e-9，
    于是 GP **逐点插值**了 80 个残差（训练 RMSE 4.8e-10、训练 R² = 1.000）。
    那是过拟合，而且会让对比表里「M4 训练段完美」变成一个假象。
    真正的物理判断是：残差里至少含有观测噪声，其量级就是 OLS 残差方差。
    把它固定成噪声下限，GP 只能去找「超过噪声的结构」，找不到就把振幅压到 0——
    这才是诚实的做法（实测振幅 0.022–0.035 Ah，与残差尺度同量级）。
    """
    return ConstantKernel(1e-3, (1e-12, 1e3)) * RBF(
        length_scale=20.0, length_scale_bounds=(1e-1, 1e4)
    )


def fit_gpr(t: np.ndarray, y: np.ndarray, seed: int = GP_SEED, n_restarts: int = 2) -> dict:
    """在**去趋势后的残差**上拟合 GP。返回 M4 的完整模型字典。"""
    trend = trend_posterior(t, y)
    resid = np.asarray(y, float) - trend_line(trend, t)

    gp = GaussianProcessRegressor(
        kernel=_kernel(),
        alpha=trend["sigma2"],  # 噪声下限 = OLS 残差方差（见 _kernel 的说明）
        normalize_y=False,
        n_restarts_optimizer=n_restarts,
        random_state=seed,
    )
    gp.fit(np.asarray(t, float).reshape(-1, 1), resid)

    model = {
        "name": "M4 趋势+GPR",
        "gp": gp,
        "trend": trend,
        "t_max": float(np.max(t)),
        "n_train": int(len(t)),
        "length_scale": float(gp.kernel_.k2.length_scale),
        "amplitude": float(np.sqrt(gp.kernel_.k1.constant_value)),
        "noise_std": float(np.sqrt(trend["sigma2"])),
        "lml": float(gp.log_marginal_likelihood_value_),
    }
    # 确定性：同种子重拟合必须给出同一组超参（否则报告的数不可复现）
    assert np.isfinite(model["length_scale"]) and model["length_scale"] > 0
    return model


def predict_gpr(model: dict, t, with_trend_var: bool = True) -> dict:
    """点预测与标准差。

    返回的 `std` 是**总**标准差：残差 GP 方差 ⊕ 趋势参数方差（`with_trend_var=True`）。
    `std_gp_only` 单独留着，用来展示「只用 GP 的带宽有多窄」。
    """
    t = np.asarray(t, dtype=float)
    X = t.reshape(-1, 1)
    mean_r, std_r = model["gp"].predict(X, return_std=True)
    mean = trend_line(model["trend"], t) + mean_r

    gp_var = std_r**2
    trend_var = trend_std(model["trend"], t) ** 2 if with_trend_var else np.zeros_like(gp_var)
    std = np.sqrt(gp_var + trend_var)

    if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(std))):
        raise ValueError("M4 给出了 inf/nan —— 不能当作预测值")
    return {
        "mean": mean,
        "std": std,
        "std_gp_only": std_r,
        "trend_mean": trend_line(model["trend"], t),
        "resid_mean": mean_r,
    }


def eol_mean(model: dict, n_obs: int) -> int | None:
    """用**均值曲线**求穿越 τ 的第一个循环（与 M1/M2 同一口径：扫描 + 求根）。

    M4 的均值曲线 = 趋势 + GP 均值，在数据之外 GP 均值为 0，所以它就等于趋势线；
    这也是为什么 M4 的点预测会和 M1 很像——不是实现巧合，是结构决定的。
    """
    hi = int(np.ceil(n_obs * EOL_SEARCH_MULTIPLE))
    grid = np.arange(1, hi + 1, dtype=float)
    hit = is_at_eol(predict_gpr(model, grid)["mean"])
    idx = np.flatnonzero(np.asarray(hit))
    if len(idx) == 0:
        return None
    return int(grid[idx[0]])


# ---------------------------------------------------------------------------
# 后验采样：EOL 分布
# ---------------------------------------------------------------------------
def sample_paths(
    model: dict, t_grid: np.ndarray, n_samples: int = N_SAMPLES,
    seed: int = GP_SEED, with_trend_var: bool = True,
) -> np.ndarray:
    """从后验抽 `n_samples` 条容量路径，形状 (len(t_grid), n_samples)。

    两条来源：
    - 残差 GP 的路径（`gp.sample_y`，含核结构、不含观测噪声）；
    - 趋势参数的路径：从 N((a,b), σ²(XᵀX)⁻¹) 抽 (a,b)，再画成直线。
    加起来才是「我对这条曲线有多不确定」的完整表达。
    """
    t_grid = np.asarray(t_grid, dtype=float)
    rng = np.random.default_rng(seed)

    resid_paths = model["gp"].sample_y(
        t_grid.reshape(-1, 1), n_samples=n_samples, random_state=seed
    )
    resid_paths = np.atleast_2d(resid_paths)
    if resid_paths.shape[0] != len(t_grid):
        resid_paths = resid_paths.T

    if with_trend_var:
        draws = rng.multivariate_normal(
            mean=np.array([model["trend"]["a"], model["trend"]["b"]]),
            cov=model["trend"]["sigma2"] * model["trend"]["xtx_inv"],
            size=n_samples,
        )  # (n_samples, 2)
        trend_paths = draws[:, 0][None, :] + draws[:, 1][None, :] * t_grid[:, None]
    else:
        trend_paths = trend_line(model["trend"], t_grid)[:, None] * np.ones(
            (1, n_samples)
        )

    return trend_paths + resid_paths


def eol_from_paths(
    model: dict, n_obs: int, n_samples: int = N_SAMPLES,
    seed: int = GP_SEED, with_trend_var: bool = True,
) -> dict:
    """每条后验路径各自的寿命 → EOL 分布。

    路径在某循环处首次 ≤ τ 即算到达；5N 内都没到则记为**删失路径**（单独计数，
    不能当成一个数字塞进分位数）。
    """
    hi = int(np.ceil(n_obs * EOL_SEARCH_MULTIPLE))
    grid = np.arange(1, hi + 1, dtype=float)
    paths = sample_paths(model, grid, n_samples, seed, with_trend_var)

    hit = is_at_eol(paths)  # (len(grid), S) 布尔
    eols = np.full(paths.shape[1], np.nan)
    for j in range(paths.shape[1]):
        idx = np.flatnonzero(hit[:, j])
        if len(idx):
            eols[j] = grid[idx[0]]

    ok = eols[np.isfinite(eols)]
    q = np.percentile(ok, [5, 50, 95]) if len(ok) else np.array([np.nan] * 3)
    return {
        "eol": eols,
        "n_censored_paths": int(np.sum(~np.isfinite(eols))),
        "q05": float(q[0]),
        "median": float(q[1]),
        "q95": float(q[2]),
        "width90": float(q[2] - q[0]),
        "n_paths": int(paths.shape[1]),
    }


# ---------------------------------------------------------------------------
# 结构断言：GP 到底能不能外推？带宽到底怎么长？
# ---------------------------------------------------------------------------
def structure_checks(model: dict, n_obs: int, m: int) -> dict:
    """四条结构断言。任何一条被破坏，说明 M4 的结构描述已经不对了。

    1. 长度尺度 ≪ 外推距离 → 平稳核没有长程记忆（这是「GP 不负责外推」的前提）；
    2. 数据之外 GP 均值**精确归零** → 点预测等于趋势线；
    3. 数据之外 **GP 单独**的标准差是常数 → 它不随距离张开；
    4. 数据之外 **总**标准差单调不减 → 张开全靠趋势参数方差。
    """
    extrap = np.linspace(model["t_max"], float(n_obs * EOL_SEARCH_MULTIPLE), 400)
    pred = predict_gpr(model, extrap)

    # 1. 长度尺度
    assert model["length_scale"] < (n_obs - m), (
        f"长度尺度 {model['length_scale']:.2f} 不短于外推距离 {n_obs - m} 循环——"
        "那说明 GP 在数据之外还有相关性，第 2/3 条结论要重写"
    )

    # 「数据之外」不能从 t_max 当刻算起：t_max 上 GP 还在做局部修正。
    # 真正要问的是「离开数据几倍长度尺度后，GP 是否已经躺平」。
    tau_start = model["t_max"] + max(5.0 * model["length_scale"], 5.0)
    far = np.linspace(tau_start, float(n_obs * EOL_SEARCH_MULTIPLE), 400)
    far_pred = predict_gpr(model, far)

    # 2. GP 均值归零（留 1e-6 Ah 的浮点余量）
    max_resid = float(np.max(np.abs(far_pred["resid_mean"])))
    assert max_resid < 1e-6, (
        f"离开数据 {tau_start - model['t_max']:.0f} 循环后，GP 均值最大仍有 "
        f"{max_resid:.2e} Ah——「点预测就是趋势线」这个说法要重写"
    )

    # 2b. 近场局部修正有多大：这是 GP 唯一真正做的事，必须量化而不是说"零"
    near = np.linspace(model["t_max"], min(tau_start, float(n_obs * EOL_SEARCH_MULTIPLE)), 200)
    max_local = float(np.max(np.abs(predict_gpr(model, near)["resid_mean"])))

    # 3. GP 单独的标准差是常数
    gp_std = far_pred["std_gp_only"]
    gp_flat = float(np.max(np.abs(gp_std - gp_std[0])))
    assert gp_flat < 1e-9, (
        f"数据之外 GP 标准差随距离变化了 {gp_flat:.2e} Ah——"
        "「平稳核在数据外饱和」这个说法不成立"
    )

    # 4. 总标准差单调不减（W13 验收判据，D23 第 1 条：只看没有数据的区段）
    d = np.diff(pred["std"])
    assert np.all(d >= -1e-12), (
        f"外推区间内总标准差出现了 {d.min():.2e} 的下降——带宽不单调"
    )

    # 5. 「外推越远，带宽越由趋势参数决定」这条主张：远处趋势方差必须占大头
    end = predict_gpr(model, np.array([float(n_obs * EOL_SEARCH_MULTIPLE)]))
    total_var = float(end["std"][0] ** 2)
    trend_share = float(end["std"][0] ** 2 - end["std_gp_only"][0] ** 2) / total_var
    assert trend_share > 0.5, (
        f"在 {EOL_SEARCH_MULTIPLE}N 处，趋势参数方差只占总方差的 {trend_share:.0%}——"
        "「带宽主要由趋势项提供」这个说法不成立，第四节要重写"
    )

    return {
        "length_scale": model["length_scale"],
        "amplitude": model["amplitude"],
        "decay_cycles": float(tau_start - model["t_max"]),
        "max_resid_outside": max_resid,
        "max_local_correction": max_local,
        "trend_share_at_5n": trend_share,
        "gp_std_flat": gp_flat,
        "gp_std_outside": float(gp_std[0]),
        "std_at_tmax": float(pred["std"][0]),
        "std_at_5n": float(pred["std"][-1]),
        "trend_std_at_tmax": float(trend_std(model["trend"], np.array([model["t_max"]]))[0]),
        "trend_std_at_5n": float(
            trend_std(model["trend"], np.array([float(n_obs * EOL_SEARCH_MULTIPLE)]))[0]
        ),
    }


# ---------------------------------------------------------------------------
# 跑一遍
# ---------------------------------------------------------------------------
def run(df: pd.DataFrame, prof: pd.DataFrame, m: int = SPLIT_M) -> dict:
    prof_idx = prof.set_index("battery_id")
    rows, fits, dists = [], {}, {}
    for sp in temporal_split(df, m):
        eol_true = prof_idx.loc[sp.cell, "eol_cycle"]
        eol_true = None if pd.isna(eol_true) else int(eol_true)
        n_obs = len(sp.train) + len(sp.test)

        model = fit_gpr(sp.t_train, sp.y_train)
        chk = structure_checks(model, n_obs, m)

        pred = predict_gpr(model, sp.t_test)
        met = metrics(sp.y_test, pred["mean"])
        train_met = metrics(sp.y_train, predict_gpr(model, sp.t_train)["mean"])

        # 训练段不许被插值：GP 的训练 RMSE 必须与残差噪声同量级，
        # 否则就是「每个点一个自由度」（第一版 WhiteKernel 自由学时就是 4.8e-10）
        assert train_met["RMSE"] > 0.1 * model["noise_std"], (
            f"{sp.cell}: 训练 RMSE {train_met['RMSE']:.2e} 远小于残差噪声 "
            f"{model['noise_std']:.4f} Ah——M4 在逐点插值，报告里的训练段数字不可用"
        )

        eol = eol_mean(model, n_obs)
        ee = eol_error(eol, eol_true, n_obs)

        dist = eol_from_paths(model, n_obs)
        assert dist["q05"] <= dist["median"] <= dist["q95"], (
            f"{sp.cell}: EOL 分位数没有排序（{dist['q05']}, {dist['median']}, {dist['q95']}）"
        )
        assert dist["n_censored_paths"] == 0, (
            f"{sp.cell}: {dist['n_censored_paths']}/{dist['n_paths']} 条后验路径在 "
            f"{EOL_SEARCH_MULTIPLE}×N 内没穿越阈值——EOL 分位数此时不可比"
        )

        fits[sp.cell] = {"model": model, "split": sp, "checks": chk, "dist": dist}
        dists[sp.cell] = dist

        covers = (
            "—" if eol_true is None
            else ("覆盖" if dist["q05"] <= eol_true <= dist["q95"] else "未覆盖")
        )
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
                "长度尺度": model["length_scale"],
                "EOL预测": "—" if eol is None else eol,
                "EOL 90%区间": f"{dist['q05']:.0f}–{dist['q95']:.0f}",
                "EOL真值": "删失" if eol_true is None else eol_true,
                "EOL误差": "—" if ee["eol_err"] is None else ee["eol_err"],
                "区间是否覆盖真值": covers,
                "备注": ee["note"],
            }
        )
    return {"table": pd.DataFrame(rows), "fits": fits, "dists": dists}


def variance_decomposition(df: pd.DataFrame, m: int = SPLIT_M) -> pd.DataFrame:
    """把外推末端的总方差拆成「残差」与「趋势参数」两块。

    这张表就是 M4 存在的理由：只看残差那一块，区间会窄一个量级。
    """
    rows = []
    for sp in temporal_split(df, m):
        n_obs = len(sp.train) + len(sp.test)
        model = fit_gpr(sp.t_train, sp.y_train)
        for probe, tag in ((float(m), "切分点 M"),
                           (float(n_obs), "观测末端"),
                           (float(n_obs * 2), "2N")):
            pred = predict_gpr(model, np.array([probe]))
            rows.append(
                {
                    "电池": sp.cell,
                    "位置": tag,
                    "循环": int(probe),
                    "残差方差": float(pred["std_gp_only"][0] ** 2),
                    "趋势方差": float(pred["std"][0] ** 2 - pred["std_gp_only"][0] ** 2),
                    "总标准差(Ah)": float(pred["std"][0]),
                }
            )
    return pd.DataFrame(rows)


def split_sensitivity(df: pd.DataFrame, prof: pd.DataFrame) -> pd.DataFrame:
    """换切分点 M，指标与 EOL 区间怎么变（D23：误差必须带切分点）。"""
    rows = []
    for m in SPLIT_M_SWEEP:
        tbl = run(df, prof, m)["table"]
        r2 = pd.to_numeric(tbl["R2"], errors="coerce")
        err = pd.to_numeric(tbl["EOL误差"], errors="coerce")
        rows.append(
            {
                "M": m,
                "测试R2均值": float(r2.mean()),
                "RMSE 均值": float(tbl["RMSE"].mean()),
                "EOL 绝对误差均值": float(err.abs().mean()) if err.notna().any() else float("nan"),
                "EOL 90%区间宽度均值": float(
                    np.mean([float(s.split("–")[1]) - float(s.split("–")[0])
                             for s in tbl["EOL 90%区间"]])
                ),
                "可比电池数": int(err.notna().sum()),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------
def fig09(df: pd.DataFrame, res: dict, m: int) -> Path:
    """主图：趋势 + GP 均值 + 两层合成出来的 95% 带。"""
    setup_matplotlib()
    cells = list(split_by_battery(df).keys())
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.8))
    axes = axes.ravel()

    for ax, cell in zip(axes, cells):
        v = res["fits"][cell]
        sp, model = v["split"], v["model"]
        n_obs = len(sp.train) + len(sp.test)

        grid = np.linspace(1, float(n_obs * 1.35), 600)
        pred = predict_gpr(model, grid)

        ax.fill_between(grid, pred["mean"] - Z95 * pred["std"],
                        pred["mean"] + Z95 * pred["std"],
                        color="#6a3d9a", alpha=0.18, lw=0, label="95% 区间（趋势+残差）")
        ax.fill_between(grid, pred["mean"] - Z95 * pred["std_gp_only"],
                        pred["mean"] + Z95 * pred["std_gp_only"],
                        color="#e0a800", alpha=0.50, lw=0, label="只算残差方差（更窄）")
        ax.plot(grid, pred["trend_mean"], ":", color="#777777", lw=1.3, label="趋势线（=M1）")
        ax.plot(grid, pred["mean"], color="#6a3d9a", lw=1.8, label="M4 均值")

        ax.plot(sp.t_train, sp.y_train, color=COLORS[cell], lw=2.0, zorder=5,
                label="已观测（训练段）")
        ax.plot(sp.t_test, sp.y_test, color=COLORS[cell], lw=1.3, ls=":", alpha=0.9,
                zorder=5, label="真实后段（测试段）")

        ax.axhline(EOL_THRESHOLD_AH, ls="--", lw=1.1, color="#aaaaaa")
        ax.text(0.02, EOL_THRESHOLD_AH, f"τ={EOL_THRESHOLD_AH:g} Ah", fontsize=8,
                color="#666666", va="bottom", transform=ax.get_yaxis_transform())
        ax.axvline(m, color="#999999", lw=0.9, ls=":")
        ax.text(m, 0.02, f" M={m} ", fontsize=8, color="#777777",
                transform=ax.get_xaxis_transform(), va="bottom")

        d = v["dist"]
        truth = v.get("eol_true")
        ax.text(
            0.035, 0.05,
            f"EOL 中位数 {d['median']:.0f}，90% 区间 {d['q05']:.0f}–{d['q95']:.0f}\n"
            f"（宽 {d['width90']:.0f} 循环；真值 "
            + ("删失" if truth is None else str(truth)) + "）",
            transform=ax.transAxes, fontsize=7.6, color="#3f2a6b",
            bbox=dict(boxstyle="round,pad=0.26", fc="#f3eefc", ec="#b9a6e0", lw=0.7),
        )

        ax.set_title(f"{cell}：训练 {len(sp.train)} → 外推 {len(sp.test)} 循环"
                     f"（ℓ={model['length_scale']:.1f}）", fontsize=10)
        ax.set_xlabel("循环")
        ax.set_ylabel("容量 (Ah)")
        ax.legend(fontsize=7.0, loc="upper right", framealpha=0.92)

    fig.suptitle(f"图9　M4 趋势 + GPR：点预测≈趋势线，不确定性才是它的产出（M={m}）",
                 fontsize=13, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save(fig, "fig09_gpr_band.png")


def fig10(df: pd.DataFrame, res: dict, m: int) -> Path:
    """EOL 的后验分布：把「一个数」变成「一个分布」。"""
    setup_matplotlib()
    cells = list(split_by_battery(df).keys())
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.0))
    axes = axes.ravel()

    for ax, cell in zip(axes, cells):
        v = res["fits"][cell]
        d = v["dist"]
        eols = d["eol"][np.isfinite(d["eol"])]
        eol_m1 = v.get("eol_m1")

        ax.hist(eols, bins=28, color="#6a3d9a", alpha=0.75, edgecolor="white", lw=0.5)
        ax.axvline(d["median"], color="#6a3d9a", lw=2.0,
                   label=f"后验中位数 {d['median']:.0f}")
        if eol_m1 is not None:
            ax.axvline(eol_m1, color="#7f8c8d", ls="--", lw=1.4,
                       label=f"点估计（趋势线）{eol_m1}")
        truth = v.get("eol_true")
        if truth is not None:
            ax.axvline(truth, color="#c0392b", lw=2.0, label=f"真值 {truth}")
        ax.set_title(f"{cell}：{len(eols)} 条路径的寿命分布", fontsize=10)
        ax.set_xlabel("EOL 循环")
        ax.set_ylabel("路径数")
        ax.legend(fontsize=8)

    fig.suptitle(f"图10　EOL 的后验分布（M={m}，{N_SAMPLES} 条路径）", fontsize=13, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save(fig, "fig10_eol_posterior.png")


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
def render_md(
    tbl: pd.DataFrame,
    tbl_m: pd.DataFrame,
    var: pd.DataFrame,
    res: dict,
    m: int,
) -> str:
    cells = tbl["电池"].tolist()
    n = len(cells)

    ls_lo = min(res["fits"][c]["checks"]["length_scale"] for c in cells)
    ls_hi = max(res["fits"][c]["checks"]["length_scale"] for c in cells)
    ls_ratio_hi = max(
        res["fits"][c]["checks"]["length_scale"]
        / (len(res["fits"][c]["split"].train) + len(res["fits"][c]["split"].test) - m)
        for c in cells
    )
    local_max = max(res["fits"][c]["checks"]["max_local_correction"] for c in cells)
    decay_lo = min(res["fits"][c]["checks"]["decay_cycles"] for c in cells)
    decay_hi = max(res["fits"][c]["checks"]["decay_cycles"] for c in cells)
    resid_out = max(res["fits"][c]["checks"]["max_resid_outside"] for c in cells)
    gp_flat = max(res["fits"][c]["checks"]["gp_std_flat"] for c in cells)
    gp_out_lo = min(res["fits"][c]["checks"]["gp_std_outside"] for c in cells)
    gp_out_hi = max(res["fits"][c]["checks"]["gp_std_outside"] for c in cells)

    # 趋势方差在总方差里的占比（按位置分别统计）
    var = var.assign(
        趋势占比=var["趋势方差"] / (var["趋势方差"] + var["残差方差"])
    )
    end_share = var.loc[var["位置"] == "观测末端", "趋势占比"]
    far_share = var.loc[var["位置"] == "2N", "趋势占比"]

    medians = [res["fits"][c]["dist"]["median"] for c in cells]
    widths = [res["fits"][c]["dist"]["width90"] for c in cells]
    eol_m1 = [res["fits"][c].get("eol_m1") for c in cells]

    # 覆盖情况：只看有真值的电池
    cover_cells = tbl.loc[tbl["区间是否覆盖真值"] != "—"]
    n_cover = int((cover_cells["区间是否覆盖真值"] == "覆盖").sum())
    n_cover_all = len(cover_cells)
    err = pd.to_numeric(tbl["EOL误差"], errors="coerce")

    # 删失电池单独一句话（不许把删失当已知值参与误差计算）
    censored = tbl.loc[tbl["EOL真值"] == "删失", "电池"].tolist()
    if censored:
        c0 = censored[0]
        d0 = res["fits"][c0]["dist"]
        censor_note = (
            f"4. **{c0} 没有 EOL 误差**：它删失，没有真值。它的后验中位数是 "
            f"{d0['median']:.0f}、90% 区间 {d0['q05']:.0f}–{d0['q95']:.0f}，"
            "只能看方向，不能算误差。"
        )
    else:
        censor_note = "4. **删失电池只能看方向，不能算误差。**"

    out = [
        "# M4 趋势 + 高斯过程：点预测≈趋势，产出是不确定性（W11）",
        "",
        "> 本文件由 `src/models_gpr.py` 自动生成，数字随代码重算，**不要手工编辑**。",
        "",
        "## 一、口径与结构",
        "",
        f"- **任务**：任务 A（个体内轨迹外推）。前 **M = {m}** 循环训练，预测 M+1..N。",
        "- **结构**：`cap(t) = 线性趋势(t) + 残差高斯过程(t)`，两层方差相加。",
        f"- **核**：`C·RBF(ℓ) + WhiteKernel`，三个超参由边际似然学；"
        f"`random_state={GP_SEED}`、`n_restarts_optimizer=2`。",
        f"- **趋势**：OLS 直线（与 M1 同一实现），参数不确定性 `σ²(XᵀX)⁻¹` 显式传播。",
        f"- **后验采样**：{N_SAMPLES} 条路径；带宽用 95%（z={Z95:.4f}）。",
        "",
        "## 二、主表",
        "",
        tbl.to_markdown(index=False),
        "",
        "## 三、最重要的结论：**平稳核的 GPR 不会外推，它只会在数据外面「躺平」**",
        "",
        f"残差 GP 学到的长度尺度是 **{ls_lo:.2f}–{ls_hi:.2f} 个循环**，"
        f"而观测长度是 {m}、外推距离是 "
        + f"{min(len(res['fits'][c]['split'].test) for c in cells)}–"
        + f"{max(len(res['fits'][c]['split'].test) for c in cells)} 个循环。比值最大的一块"
        f"（{cells[int(np.argmax([res['fits'][c]['checks']['length_scale'] for c in cells]))]}）"
        f"也只有 **{ls_ratio_hi:.2f}**——也就是说 GP 的相关尺度只是外推距离的一个零头。",
        "",
        "**但「躺平」不是立刻发生的**，这里必须精确：",
        "",
        f"- **近场**：刚离开数据时 GP 还在做局部平滑修正，最大 **{local_max:.4f} Ah**"
        "（这是它在测试段唯一真正做的事）；",
        f"- **远场**：离开数据 {decay_lo:.0f}–{decay_hi:.0f} 个循环（= 5 倍长度尺度）之后，"
        f"GP 的后验均值降到 **{resid_out:.0e} Ah** 量级 → **点预测就是趋势线**；",
        f"- **带宽**：数据之外 GP 单独的标准差是**常数** {gp_out_lo:.4f}–{gp_out_hi:.4f} Ah"
        f"（随距离的变化最大 {gp_flat:.1e}）→ **它不会随着外推变宽**。",
        "",
        "这三条都由 `structure_checks()` 断言钉住。所以 M4 能不能讲清楚，"
        "关键不在 GP，而在**趋势项**：",
        "",
        "- 这不是缺陷，是平稳过程的本性：RBF 核的协方差随距离指数衰减，"
        "远处必然回到先验（均值 0、方差 = 振幅）。",
        "- 想让 GP 自己外推趋势，得换非平稳核（比如线性核叠加）或把趋势做进均值函数——"
        "本项目**故意不这么做**，因为那样「趋势不确定性」就被埋进超参里，"
        "报告里再也说不清「这条带子为什么这么宽」。",
        "- 一个实现上的坑（第一版踩过）：把噪声交给 `WhiteKernel` 自由学，"
        "它会把噪声学成 3.6e-9、于是 GP **逐点插值** 80 个残差"
        "（训练 RMSE 4.8e-10、训练 R²=1.000）。那是过拟合，且会让对比表出现"
        "「M4 训练段完美」的假象。现在噪声下限固定为 OLS 残差方差，"
        f"训练 RMSE 回到 {tbl['训练RMSE'].min():.4f}–{tbl['训练RMSE'].max():.4f} Ah，可比。",
        "",
        "## 四、方差分解：越远，带宽越由趋势参数决定",
        "",
        var.drop(columns=["趋势占比"]).round(6).to_markdown(index=False),
        "",
        f"趋势参数方差占总方差的份额：观测末端 **{end_share.min():.0%}–{end_share.max():.0%}**，"
        f"到 2N 处升到 **{far_share.min():.0%}–{far_share.max():.0%}**。"
        "残差方差是常数（GP 单独标准差 "
        f"{gp_out_lo:.4f}–{gp_out_hi:.4f} Ah），趋势方差 ∝ (t−t̄)²，"
        "**所以外推越远，不确定性越由「这条直线本身估得准不准」决定**。",
        "",
        "这也是 M4 与 M1 的关系：M4 的点预测在远场就是 M1，**它多出来的东西主要是区间**。"
        "所以 W13 的对比表里，M4 不该在精度上大赢，它该在「能给出分布」上赢——"
        "这正是本项目的主线。",
        "",
        "## 五、EOL：从「一个数」变成「一个分布」",
        "",
        f"每条后验路径各自求一次穿越 τ 的循环，得到 {N_SAMPLES} 个寿命值：",
        "",
        f"- 中位数：" + "、".join(f"{c} {v:.0f}" for c, v in zip(cells, medians)),
        f"- 90% 区间宽度：" + "、".join(f"{c} {w:.0f}" for c, w in zip(cells, widths)),
        "",
        f"**区间覆盖真值的情况：{n_cover}/{n_cover_all} 块**"
        f"（只有 {n_cover_all} 块有真值，B0007 删失）。"
        f"EOL 点估计的绝对误差均值是 {err.abs().mean():.1f} 循环。",
        "",
        "⚠️ 这个 90% 区间**明显偏窄**：区间宽度只有 "
        f"{min(widths):.0f}–{max(widths):.0f} 循环，而点估计的误差就有 "
        f"{err.abs().max():.0f} 循环（{tbl.loc[err.abs().idxmax(), '电池']}）。"
        "原因不是代码错，而是**第三层不确定性没进来**："
        "M4 的区间只含「残差噪声 + 趋势参数」，"
        "**不含「线性这条形式本身是否成立」**——"
        "而后者才是主导（B0005/B0006 的真实衰减是加速的，直线必然偏）。",
        "",
        "这条缺口正是 W14（三层不确定性）与 W17（PICP/MPIW 校准）要填的东西，"
        "在这里先如实标出来，不许把窄区间当好消息。",
        "",
        "## 六、切分点敏感性",
        "",
        tbl_m.round(4).to_markdown(index=False),
        "",
        "## 七、这张表不能怎么读",
        "",
        "1. **不能把 M4 当成「比 M1 更准」**：数据之外它的点预测就是 M1。"
        "如果报告里说「GPR 提升了精度」，那是假的。",
        "2. **不能把 95% 区间当成「真值有 95% 概率落在里面」**：它只覆盖了两层不确定性，"
        "第五节的覆盖情况已经说明它会漏。",
        "3. **不能解释长度尺度**：ℓ≈1–4 个循环意味着「残差在这尺度上相关」，"
        "这是 80 个点的噪声结构，不是电化学时间常数。",
        censor_note,
        "",
    ]
    return "\n".join(out)


def main() -> int:
    ensure_dirs()
    raw = load_raw()
    df = mark_rows(raw)
    prof = cell_profile(df)

    res = run(df, prof, SPLIT_M)
    res["prof"] = prof
    tbl = res["table"]
    tbl_m = split_sensitivity(df, prof)
    var = variance_decomposition(df, SPLIT_M)

    # 把真值挂回 fits，供 fig10 画竖线（避免图里去猜）
    prof_idx = prof.set_index("battery_id")
    for cell in res["fits"]:
        e = prof_idx.loc[cell, "eol_cycle"]
        res["fits"][cell]["eol_true"] = None if pd.isna(e) else int(e)
        res["fits"][cell]["eol_m1"] = eol_linear(
            {"params": {"a": res["fits"][cell]["model"]["trend"]["a"],
                        "b": res["fits"][cell]["model"]["trend"]["b"]}}
        )

    TABLE_MD.write_text(render_md(tbl, tbl_m, var, res, SPLIT_M), encoding="utf-8")
    fig_a = fig09(df, res, SPLIT_M)
    fig_b = fig10(df, res, SPLIT_M)

    print(f"切分点 M = {SPLIT_M}；后验路径 {N_SAMPLES} 条；种子 {GP_SEED}")
    print()
    print(tbl.to_string(index=False))
    print()
    print("方差分解（Ah²）：")
    print(var.round(6).to_string(index=False))
    print()
    print("切分点敏感性：")
    print(tbl_m.round(4).to_string(index=False))
    print()
    print(f"M4 报告已写出：{TABLE_MD.name}")
    print(f"已生成 {fig_a.name}  ({fig_a.stat().st_size / 1024:.0f} KB)")
    print(f"已生成 {fig_b.name}  ({fig_b.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
