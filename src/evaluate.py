"""W12 的一部分：时序划分、指标、以及四种模型共用的评测口径。

为什么先写这个（虽然编号排在 W12）
----------------------------------
M1/M2/M3/M4 四个模型都要用同一套「切在哪、怎么算误差」，如果各自写一份，
四份实现迟早会分叉（W5/W6 那次「两个模块各写一份 window=11」就被审查记过）。
所以评测口径先落地成唯一出处，模型模块只负责「拟合 + 预测」。

⚠️ 一个必须现在说清的评测学问题：LOBOCV 在任务 A 上不成立
--------------------------------------------------------
本项目有两种预测任务（见 `features.py` 开头）：

- **任务 A：个体内轨迹外推**（主线，M1/M2/M3/M4 都做）
  用某块电池前 M 个循环拟合，外推它自己后面的容量与 EOL。
  **这里没有「训练集/测试集来自不同个体」这回事**——模型是逐个体拟合的，
  所以「留一电池交叉验证（LOBOCV）」在任务 A 上**在逻辑上就不存在**，
  能用的只有**时序划分**（前 M 循环 → 后 N−M 循环）。

- **任务 B：跨个体早期预测**（M3 的特征版）
  用前 K 循环的特征预测寿命，样本单位是个体 → **这里才用 LOBOCV**（4 折）。

所以 D6 说的「LOBOCV + 时序划分」不是两件事都要对每个模型做一遍，而是
**任务 A 用时序划分、任务 B 用 LOBOCV**。报告里必须这样写，否则会被人问
「你外推一个个体，怎么留一法？」

运行
----
    python src/evaluate.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    BATTERIES,
    EOL_THRESHOLD_AH,
    TABLE_DIR,
    ensure_dirs,
    is_at_eol,
)
from load_data import load_raw, split_by_battery  # noqa: E402
from preprocess import cell_profile, mark_rows  # noqa: E402

SPLIT_M = 80
"""任务 A 的主切分点：用前 M 个循环拟合，预测 M+1..N。

取值理由：最短的 B0018 只有 132 循环，M=80 能留下 52 个测试点；
而 B0005 的真实 EOL 是 125，M=80 意味着要外推 45 个循环才够到寿命终点——
外推距离是观测长度的 56%，不算白送。
**这是折中，不是最优值**，所以下面做了 M ∈ {60, 80, 100} 的敏感性。"""

SPLIT_M_SWEEP = (60, 80, 100)
"""切分点敏感性。这是 W5（h 依赖）与 W7（窗口宽度依赖）那条教训的第三次出现：
**任何「误差是多少」的数字，不说切分点就是没说完。**"""


@dataclass(frozen=True)
class Split:
    """一次时序划分的结果。"""

    cell: str
    train: pd.DataFrame
    test: pd.DataFrame
    m: int

    @property
    def t_train(self) -> np.ndarray:
        return self.train["cycle"].to_numpy(float)

    @property
    def y_train(self) -> np.ndarray:
        return self.train["capacity_Ah"].to_numpy(float)

    @property
    def t_test(self) -> np.ndarray:
        return self.test["cycle"].to_numpy(float)

    @property
    def y_test(self) -> np.ndarray:
        return self.test["capacity_Ah"].to_numpy(float)


def temporal_split(df: pd.DataFrame, m: int = SPLIT_M) -> list[Split]:
    """按循环号切分：cycle ≤ m 训练，cycle > m 测试。

    注意这里**不做随机打乱**。相邻循环的容量高度自相关，随机划分会让
    测试集的信息漏进训练集（D6）。时序划分是外推任务唯一诚实的做法。

    断言：每条序列必须真的被切成两段非空，且循环号连续——否则 M 设得不对。
    """
    out = []
    for cell, sub in split_by_battery(df).items():
        sub = sub.sort_values("cycle").reset_index(drop=True)
        cyc = sub["cycle"].to_numpy(int)
        assert (cyc == np.arange(1, len(sub) + 1)).all(), f"{cell} 循环号不连续"
        train = sub[sub["cycle"] <= m]
        test = sub[sub["cycle"] > m]
        if len(train) < 10 or len(test) < 5:
            raise ValueError(
                f"{cell} 在 M={m} 时切不出可用的训练/测试段："
                f"训练 {len(train)} 行、测试 {len(test)} 行"
            )
        assert len(train) + len(test) == len(sub), "切分丢了行"
        out.append(Split(cell=cell, train=train, test=test, m=m))
    return out


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """容量预测的四个常用指标。

    R² 用的是**测试段自身的方差**作分母。对一条强烈外推失败的模型，
    R² 会显著为负（比如树模型外推成一条水平线）——这正是我们要看清楚的现象，
    所以不做任何截断或重定义。
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"形状不一致：{y_true.shape} vs {y_pred.shape}")
    if not np.all(np.isfinite(y_pred)):
        raise ValueError("预测值里有 inf/nan —— 模型发散了，这不能当作一个误差值")

    resid = y_pred - y_true
    ss_res = float((resid**2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "RMSE": float(np.sqrt((resid**2).mean())),
        "MAE": float(np.abs(resid).mean()),
        "MAPE": float(np.mean(np.abs(resid / y_true)) * 100),
        "R2": r2,
        "n_test": int(len(y_true)),
    }


def eol_error(pred: int | None, true: int | None, n_obs: int) -> dict:
    """寿命预测误差。

    三种情况必须分清，不能都塞进一个减法：
    - 两者都有值 → 差值；
    - 真值删失（观察期内没坏）→ **无真值可比**，只能看模型有没有预测「超过观测长度」；
    - 模型预测不出穿越 → 记为「模型认为不会坏」，同样不可比。
    """
    if true is None:
        return {
            "eol_pred": pred,
            "eol_true": None,
            "eol_err": None,
            "note": f"真值删失（观测到 {n_obs} 循环仍未到达），无误差可比"
            + ("" if pred is None or pred > n_obs else "；⚠️ 模型却预测会坏"),
        }
    if pred is None:
        return {
            "eol_pred": None,
            "eol_true": true,
            "eol_err": None,
            "note": "模型预测不穿越阈值，无误差可比",
        }
    return {
        "eol_pred": pred,
        "eol_true": true,
        "eol_err": int(pred - true),
        "note": "",
    }


# ---------------------------------------------------------------------------
# 自检：切分与指标本身对不对
# ---------------------------------------------------------------------------
def self_test(series: dict[str, pd.DataFrame]) -> None:
    """用构造数据钉住 `metrics` 与 `temporal_split` 的行为。

    这几条是**真的会失败**的检查（不是恒真）：每一条都对应一种具体写错的方式。
    """
    # 完美预测 → 全部零误差
    y = np.array([2.0, 1.8, 1.6, 1.4])
    m = metrics(y, y.copy())
    assert m["RMSE"] == 0.0 and m["MAE"] == 0.0 and m["R2"] == 1.0, m

    # 整体偏移 0.1 → RMSE = MAE = 0.1，MAPE = 0.1/mean(y)
    m = metrics(y, y + 0.1)
    assert abs(m["RMSE"] - 0.1) < 1e-12 and abs(m["MAE"] - 0.1) < 1e-12, m
    assert abs(m["MAPE"] - 0.1 / y.mean() * 100) < 1e-9, m

    # 预测成常数（树模型外推的典型行为）→ R² 必为负
    flat = np.full_like(y, y.mean())
    m = metrics(y, flat)
    assert m["R2"] < 0, f"常数预测的 R² 竟然非负：{m}"

    # 发散必须报错，不能悄悄返回一个「误差」
    try:
        metrics(y, np.array([np.inf, 1, 1, 1]))
    except ValueError:
        pass
    else:
        raise AssertionError("预测含 inf 时 metrics 没有报错")

    # 切分：训练段与测试段必须严格互补且不重叠
    for cell, sub in series.items():
        sp = temporal_split(sub.assign(battery_id=cell), SPLIT_M)[0]
        assert sp.train["cycle"].max() == SPLIT_M, f"{cell} 训练段没有切在 M"
        assert sp.test["cycle"].min() == SPLIT_M + 1, f"{cell} 测试段起点不对"
        assert len(sp.train) + len(sp.test) == len(sub), f"{cell} 切分丢行"


def main() -> int:
    ensure_dirs()
    raw = load_raw()
    df = mark_rows(raw)
    prof = cell_profile(df)
    series = split_by_battery(df)

    self_test(series)

    print(f"任务 A 时序划分：前 M={SPLIT_M} 循环拟合 → 预测 M+1..N")
    print()
    rows = []
    prof_idx = prof.set_index("battery_id")
    for sp in temporal_split(df, SPLIT_M):
        eol_true = prof_idx.loc[sp.cell, "eol_cycle"]
        eol_true = None if pd.isna(eol_true) else int(eol_true)
        rows.append(
            {
                "电池": sp.cell,
                "观测长度": len(sp.train) + len(sp.test),
                "训练": len(sp.train),
                "测试": len(sp.test),
                "外推距离(循环)": len(sp.test),
                "外推/观测": f"{len(sp.test) / (len(sp.train) + len(sp.test)):.0%}",
                "真 EOL": "删失" if eol_true is None else eol_true,
                "需外推至 EOL": "—" if eol_true is None
                else max(0, eol_true - sp.m),
            }
        )
    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False))
    print()
    print(f"EOL 阈值 τ = {EOL_THRESHOLD_AH:g} Ah（{len(series)} 块电池，"
          f"其中 {int(prof['censored'].sum())} 块删失）")

    out = TABLE_DIR / "eval_setup.md"
    out.write_text(
        "\n".join(
            [
                "# 评测口径（W12 前置）",
                "",
                "> 本文件由 `src/evaluate.py` 自动生成，数字随代码重算，**不要手工编辑**。",
                "",
                f"任务 A 时序划分：前 **M = {SPLIT_M}** 循环拟合，预测 M+1..N。",
                "",
                tbl.to_markdown(index=False),
                "",
                "## 为什么任务 A 不能做 LOBOCV",
                "",
                "任务 A 是**逐个体拟合**：模型只在某块电池自己的前 M 个循环上训练。",
                "「留一块电池不参与训练」这件事在任务 A 里没有意义——那块电池本来就是",
                "用自己前半段拟合、预测自己后半段。LOBOCV 只适用于任务 B（跨个体早期预测），",
                "那里的样本单位才是个体。",
                "",
                f"## 切分点敏感性",
                "",
                f"正式口径 M = {SPLIT_M}；报告里同时给 M ∈ "
                f"{list(SPLIT_M_SWEEP)} 的误差，说明「误差是多少」依赖切分点。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print()
    print(f"评测口径已写出：{out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
