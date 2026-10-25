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
# 任务 B：留一电池交叉验证（LOBOCV）—— 以及它在本数据上为什么只能"描述"
# ---------------------------------------------------------------------------
"""任务 B（跨个体早期预测）的样本单位是**个体**，所以这里才用 LOBOCV。

⚠️ 本数据只有 4 块电池，其中 B0007 删失没有寿命标签 → **可用 n = 3**。
于是每一折的训练集只有 **2 个样本**。这不是能靠"好好调参"解决的问题，
而是研究设计的硬约束。W12 的任务不是硬凑一个排名，而是**把这个约束量化出来**。
"""


def lobocv_folds(labels: pd.Series) -> list[tuple[list[str], str]]:
    """留一个体：返回 [(训练个体列表, 被留出的个体), ...]，按标签索引顺序。

    断言：每个个体恰好被留出一次；训练集不含被留出的那个；训练集非空。
    """
    ids = list(labels.index)
    if len(ids) < 3:
        raise ValueError(f"LOBOCV 至少需要 3 个有标签的个体，收到 {len(ids)}")
    folds = [([i for i in ids if i != hold], hold) for hold in ids]
    held = [h for _, h in folds]
    assert sorted(held) == sorted(ids), "有折重复或漏掉了个体"
    for train, hold in folds:
        assert hold not in train, "训练集里混进了被留出的个体"
        assert len(train) == len(ids) - 1, "训练集大小不对"
        assert len(train) > 0, "训练集为空"
    return folds


def linear_2pt(x_train: np.ndarray, y_train: np.ndarray):
    """两点线性拟合。**训练残差恒为 0**（两点确定一条直线），这是恒等式不是优点。"""
    x = np.asarray(x_train, float)
    y = np.asarray(y_train, float)
    if len(x) != 2:
        raise ValueError("linear_2pt 只接受 2 个训练点")
    if abs(x[1] - x[0]) < 1e-15:
        raise ValueError("两个训练点的自变量完全相同，斜率无定义")
    b = (y[1] - y[0]) / (x[1] - x[0])
    a = y[0] - b * x[0]
    resid = (a + b * x) - y
    assert np.max(np.abs(resid)) < 1e-12, (
        f"两点线性拟合的训练残差最大 {np.max(np.abs(resid)):.2e}，不是恒等零——实现错了"
    )
    return float(a), float(b)


def lobocv_one_feature(
    feat: pd.DataFrame, labels: pd.Series, feature: str
) -> list[dict]:
    """单个特征在 LOBOCV 下逐折的预测与误差。"""
    rows = []
    for train, hold in lobocv_folds(labels):
        y_train = labels.loc[train].to_numpy(float)
        y_true = float(labels.loc[hold])
        a, b = linear_2pt(feat.loc[train, feature].to_numpy(float), y_train)
        pred = a + b * float(feat.loc[hold, feature])
        rows.append(
            {
                "特征": feature,
                "留出": hold,
                "斜率": b,
                "训练点数": len(train),
                "训练x跨度": float(np.ptp(feat.loc[train, feature].to_numpy(float))),
                "到最近训练点的距离": float(
                    min(abs(float(feat.loc[hold, feature]) - float(feat.loc[c, feature]))
                        for c in train)
                ),
                "预测": pred,
                "真值": y_true,
                "误差": pred - y_true,
            }
        )
    return rows


def lobocv_null(labels: pd.Series) -> list[dict]:
    """零基准：用训练集的寿命均值去预测被留出的那个个体。

    它没有任何"学习"，只有一个数。**任何特征模型赢不了它，就说明模型没用。**
    """
    rows = []
    for train, hold in lobocv_folds(labels):
        y_train = labels.loc[train].to_numpy(float)
        y_true = float(labels.loc[hold])
        pred = float(y_train.mean())
        rows.append(
            {
                "特征": "零基准（训练均值）",
                "留出": hold,
                "斜率": 0.0,
                "训练点数": len(train),
                "预测": pred,
                "真值": y_true,
                "误差": pred - y_true,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 指标汇总：区间类指标 + 跨模型统一表
# ---------------------------------------------------------------------------
def coverage(y_true, lo, hi) -> float:
    """PICP：真值落在 [lo, hi] 内的比例。"""
    y_true = np.asarray(y_true, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    if not (y_true.shape == lo.shape == hi.shape):
        raise ValueError("coverage 的三个输入形状必须一致")
    if np.any(hi < lo):
        raise ValueError("区间上界小于下界")
    return float(np.mean((y_true >= lo) & (y_true <= hi)))


def mpiw(lo, hi) -> float:
    """MPIW：平均区间宽度。**必须与 coverage 一起报**——只报宽度会被"宽区间"骗。"""
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    if np.any(hi < lo):
        raise ValueError("区间上界小于下界")
    return float(np.mean(hi - lo))


COMMON_COLUMNS: tuple[str, ...] = (
    "电池", "模型", "M", "训练RMSE", "RMSE", "MAE", "MAPE", "R2",
    "EOL预测", "EOL真值", "EOL误差",
)
"""四个模型的表合并成"对比主表"时必须有的列。W13 就靠它来拼。

列名必须一致，否则合并时会出现两列同义不同名（比如 `RMSE` 与 `测试RMSE`），
然后 `mean(axis=1)` 静默算错。所以这里用断言把"缺列"变成硬错误。
"""


def to_common_schema(tbl: pd.DataFrame, model_name: str | None = None) -> pd.DataFrame:
    """把某个模型的输出表规范成统一列序：公共列在前，模型特有列在后。

    缺任何一列都直接报错——**不许静默补 NaN**。

    `model_name` 的语义（第七轮审查 R7-6 抓出来的坑）：
    - **传了**：要求表里只有一个模型、且与它一致，然后写进「模型」列。
      若表里有多个模型（M1/M2 共用一张 8 行的表），直接报错而不是把 M2 的行
      静默改标成 M1。
    - **不传**：保留原「模型」列——多模型表要靠这一列区分身份。
    """
    missing = [c for c in COMMON_COLUMNS if c not in tbl.columns]
    if missing:
        raise ValueError(
            f"表缺少公共列 {missing}；现有列：{list(tbl.columns)}"
        )
    extra = [c for c in tbl.columns if c not in COMMON_COLUMNS]
    out = tbl.loc[:, list(COMMON_COLUMNS) + extra].copy()

    if model_name is not None:
        uniq = out["模型"].unique().tolist()
        if len(uniq) > 1:
            raise ValueError(
                f"表里有多个模型 {uniq}，不能统一标成 {model_name}——先按模型拆表再调用"
            )
        if uniq[0] != model_name:
            raise ValueError(f"表内模型是 {uniq[0]}，与传入的 {model_name} 不一致")
        out["模型"] = model_name
    return out


# ---------------------------------------------------------------------------
# 自检：新加的框架函数是否真的会失败
# ---------------------------------------------------------------------------
def self_test_framework() -> None:
    """钉住 LOBOCV、区间指标与统一表的**行为**（不是恒真检查）。"""
    labels = pd.Series({"A": 100.0, "B": 110.0, "C": 130.0})
    folds = lobocv_folds(labels)
    assert len(folds) == 3, f"3 个个体应得 3 折，实际 {len(folds)}"
    assert sorted(h for _, h in folds) == ["A", "B", "C"], "折的覆盖不对"
    for train, hold in folds:
        assert hold not in train and len(train) == len(labels) - 1, (
            f"折构造不对：训练集 {train}，留出 {hold}"
        )

    # 两点线性拟合：训练残差恒为 0 —— 这是恒等式，不是「拟合得好」
    a, b = linear_2pt(np.array([1.0, 3.0]), np.array([10.0, 20.0]))
    assert abs(a - 5.0) < 1e-12 and abs(b - 5.0) < 1e-12, (a, b)
    assert abs((a + b * 1.0) - 10.0) < 1e-12
    assert abs((a + b * 3.0) - 20.0) < 1e-12

    # 两个训练点 x 完全相同时必须报错（否则斜率是 0/0）
    try:
        linear_2pt(np.array([2.0, 2.0]), np.array([1.0, 5.0]))
    except ValueError:
        pass
    else:
        raise AssertionError("两点 x 相同时 linear_2pt 没有报错")

    # 零基准的预测必须等于训练均值
    for row in lobocv_null(labels):
        others = [v for k, v in labels.items() if k != row["留出"]]
        assert abs(row["预测"] - float(np.mean(others))) < 1e-12, row

    # 区间指标：覆盖率与平均宽度
    assert coverage([1, 2, 3], [0, 1, 2], [2, 3, 4]) == 1.0
    assert coverage([1, 5], [0, 0], [2, 2]) == 0.5
    assert abs(mpiw([0, 0], [2, 4]) - 3.0) < 1e-12
    for fn, args in ((coverage, ([1], [3], [2])), (mpiw, ([3], [2]))):
        try:
            fn(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{fn.__name__} 在上界小于下界时没有报错")

    # 统一表：缺列必须报错；多模型表不许被静默合并成一个标签（R7-6）
    # 顺序有意：先查「缺列」，再查「多模型」，这样两个变异各自命中自己那条断言
    good = pd.DataFrame({c: [1] for c in COMMON_COLUMNS}).assign(模型="M1")
    out = to_common_schema(good, "M1")
    assert out["模型"].iloc[0] == "M1"
    assert to_common_schema(good)["模型"].iloc[0] == "M1", (
        "不传 model_name 时应保留原「模型」列"
    )
    bad = good.drop(columns=["EOL误差"])
    try:
        to_common_schema(bad, "M1")
    except ValueError as exc:
        assert "EOL误差" in str(exc), exc
    else:
        raise AssertionError("缺 EOL误差 列时 to_common_schema 没有报错")
    multi = pd.concat([good, good.assign(模型="M2")], ignore_index=True)
    keep = to_common_schema(multi)
    assert set(keep["模型"]) == {"M1", "M2"}, "多模型表被压成了一个标签"
    try:
        to_common_schema(multi, "M1")
    except ValueError as exc:
        assert "多个模型" in str(exc), exc
    else:
        raise AssertionError("多模型表传了 model_name 却没报错")


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

    # 整体偏移 0.1 → RMSE = MAE = 0.1，MAPE 是 mean(|0.1/y_i|)，**不是** 0.1/mean(y)
    m = metrics(y, y + 0.1)
    assert abs(m["RMSE"] - 0.1) < 1e-12 and abs(m["MAE"] - 0.1) < 1e-12, m
    assert abs(m["MAPE"] - 0.1 * np.mean(1.0 / y) * 100) < 1e-9, m

    # 预测成「测试段均值」的常数 → R² 恰为 0（不是负的；负值要预测成别的常数）
    flat = np.full_like(y, y.mean())
    m = metrics(y, flat)
    assert abs(m["R2"]) < 1e-12, f"均值常数预测的 R² 应为 0，实际 {m}"
    # 偏掉的常数 → R² 必为负。树模型外推时就是把趋势压成一条水平线，
    # 如果那条线落在测试段均值之外，R² 会是负数（本项目 M3 的核心现象）。
    m = metrics(y, np.full_like(y, y.mean() + 0.2))
    assert m["R2"] < 0, f"偏移常数预测的 R² 竟然非负：{m}"

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

    return task_b_main(df, prof)


# ---------------------------------------------------------------------------
# 任务 B 的主流程：LOBOCV + 零基准 + 结论
# ---------------------------------------------------------------------------
def task_b_main(df: pd.DataFrame, prof: pd.DataFrame) -> int:
    """跑任务 B 的 LOBOCV，并把「为什么只能描述、不能排名」写成产物。"""
    import features  # 延迟导入：features 不依赖 evaluate，放这里有向无环

    self_test_framework()

    feat = features.build_features(df, features.K_MAIN).set_index("电池")
    prof_idx = prof.set_index("battery_id")["eol_cycle"]
    labels_all = prof_idx.reindex(feat.index)
    labels = labels_all.dropna()
    censored = labels_all[labels_all.isna()].index.tolist()

    record_rows = lobocv_null(labels)
    for f in features.SCHEMA:
        record_rows += lobocv_one_feature(feat, labels, f)
    long = pd.DataFrame(record_rows)

    summary = (
        long.groupby("特征")
        .agg(
            平均绝对误差=("误差", lambda s: float(np.mean(np.abs(s)))),
            最大绝对误差=("误差", lambda s: float(np.max(np.abs(s)))),
            平均斜率=("斜率", "mean"),
        )
        .reset_index()
        .sort_values("平均绝对误差")
        .reset_index(drop=True)
    )

    null_mae = float(summary.loc[summary["特征"] == "零基准（训练均值）",
                                "平均绝对误差"].iloc[0])
    feat_summary = summary[summary["特征"] != "零基准（训练均值）"]
    best_mae = float(feat_summary["平均绝对误差"].min())
    best_feat = str(feat_summary.loc[feat_summary["平均绝对误差"].idxmin(), "特征"])

    worst = long.loc[long["误差"].abs().idxmax()]
    worst_feat = str(worst["特征"])
    print()
    print(f"任务 B（跨个体早期预测）：n = {len(labels)} 个有标签个体"
          f"（删失：{'、'.join(censored) or '无'}），LOBOCV {len(lobocv_folds(labels))} 折，"
          f"每折训练样本 = {len(labels) - 1}")
    print(summary.round(2).to_string(index=False))

    # --- 硬断言：结论必须靠数据站住 ---
    assert null_mae < best_mae, (
        f"零基准（{null_mae:.1f} 循环）竟然不优于最好的特征模型 "
        f"（{best_feat}，{best_mae:.1f}）——「特征模型不如零基准」这句话不成立，报告要重写"
    )
    assert float(feat_summary["最大绝对误差"].max()) > 10 * null_mae, (
        "没有任何特征出现灾难性外推（>10 倍零基准误差）——第四节的近共线解释要重写"
    )

    fig_path = fig11_lobocv(feat, labels, long, summary, worst_feat)
    md = render_task_b_md(
        labels, censored, long, summary, null_mae, best_feat, best_mae, worst,
        fig_path, feat,
    )
    out = TABLE_DIR / "task_b_lobocv.md"
    out.write_text(md, encoding="utf-8")
    print()
    print(f"任务 B 报告已写出：{out.name}")
    print(f"已生成 {fig_path.name}  ({fig_path.stat().st_size / 1024:.0f} KB)")
    return 0


def fig11_lobocv(feat, labels, long, summary, worst_feat):
    """两联图：① 各方法的平均绝对误差（对数轴，零基准高亮）；② 最坏那一折的失控外推。"""
    from config import setup_matplotlib
    from figures import save

    setup_matplotlib()
    import matplotlib.pyplot as plt

    fx = feat if feat.index.name == "电池" else feat.set_index("电池")
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.4))

    ax = axes[0]
    s = summary.iloc[::-1]
    colors = ["#c0392b" if v == "零基准（训练均值）" else "#5b7db1"
              for v in s["特征"]]
    ax.barh(np.arange(len(s)), s["平均绝对误差"], color=colors, height=0.7)
    ax.set_yticks(np.arange(len(s)))
    ax.set_yticklabels(s["特征"], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("LOBOCV 平均绝对误差（循环，对数轴）")
    ax.set_title(f"① 谁比「零基准」更准？一个都没有（n={len(labels)}，每折训练 2 个样本）",
                 fontsize=10)
    for i, v in enumerate(s["平均绝对误差"]):
        ax.text(v * 1.05, i, f"{v:.0f}", va="center", fontsize=7.6, color="#333333")

    ax = axes[1]
    order = ["B0005", "B0006", "B0018"]
    xs = fx.loc[order, worst_feat].to_numpy(float)
    ys = labels.reindex(order).to_numpy(float)
    ax.scatter(xs, ys, s=90, color="#2c3e50", zorder=5, label="有标签的个体")
    for x, y, c in zip(xs, ys, order):
        ax.annotate(c, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)

    hold = str(long.loc[long["误差"].abs().idxmax(), "留出"])
    others = [c for c in order if c != hold]
    a, b = linear_2pt(fx.loc[others, worst_feat].to_numpy(float),
                      labels.loc[others].to_numpy(float))
    pad = 0.02 * float(np.ptp(xs))
    grid = np.linspace(float(xs.min()) - pad, float(xs.max()) + pad, 100)
    ax.plot(grid, a + b * grid, "--", color="#c0392b", lw=1.6,
            label=f"留出 {hold} 时用另外 2 点拟合的直线")
    pred = a + b * float(fx.loc[hold, worst_feat])
    ax.scatter([fx.loc[hold, worst_feat]], [labels.loc[hold]],
               s=120, facecolor="none", edgecolor="#c0392b", lw=2, zorder=6,
               label=f"{hold} 的真值")
    ax.set_ylim(float(ys.min()) - 15, float(ys.max()) + 15)
    ax.annotate(
        f"同一特征下对 {hold} 的预测 = {pred:.0f} 循环（图外）\n"
        f"训练集这 2 个点的 x 只差 "
        f"{float(np.ptp(fx.loc[others, worst_feat].to_numpy(float))):.4f}，"
        f"却要解释 {abs(float(np.diff(labels.loc[others].to_numpy(float))[0])):.0f} 循环的寿命差",
        xy=(0.02, 0.03), xycoords="axes fraction", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="#fdf2f2", ec="#e6b0b0"),
    )
    ax.set_xlabel(worst_feat)
    ax.set_ylabel("真实寿命（循环）")
    ax.set_title(
        f"② 失控的那一折：2 个点 + 近共线的 x = 斜率 "
        f"{abs(float(long.loc[long['误差'].abs().idxmax(), '斜率'])):.0e}",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(f"图11　任务 B（跨个体早期预测）的 LOBOCV：n={len(labels)} 时只能描述，不能排名",
                 fontsize=12.5, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return save(fig, "fig11_lobocv.png")


def render_task_b_md(
    labels, censored, long, summary, null_mae, best_feat, best_mae, worst,
    fig_path, feat,
) -> str:
    """把结论写成报告。数字现算；K / 特征数 / 删失电池一概不写死。"""
    import features

    feat_summary = summary[summary["特征"] != "零基准（训练均值）"]
    n = len(labels)
    folds = lobocv_folds(labels)
    n_train = len(folds[0][0])
    worst_err = float(worst["误差"])
    worst_span = float(worst["训练x跨度"])
    worst_gap = float(worst["到最近训练点的距离"])
    n_feat = len(feat_summary)
    k_main = features.K_MAIN
    slope_max = float(feat_summary["平均斜率"].abs().max())
    # 斜率放大系数：1/训练x跨度（设计矩阵条件数的主因）
    amp = 1.0 / worst_span if worst_span > 0 else float("nan")

    return "\n".join(
        [
            "# 任务 B：跨个体早期预测的 LOBOCV（W12）",
            "",
            "> 本文件由 `src/evaluate.py` 自动生成，数字随代码重算，**不要手工编辑**。",
            "",
            "## 一、口径：任务 B 与任务 A 是两个问题",
            "",
            "- **任务 A**（W9–W11 的 M1–M4）：给一块电池自己的前 M 个循环，外推它自己的后段。"
            "**逐个体拟合，没有「留一个体」这回事**。",
            "- **任务 B**（本文件）：只看前 K 个循环的**特征**，预测**这块电池的寿命**。"
            "样本单位是**个体**，所以这里才用 LOBOCV（留一电池交叉验证）。",
            "- `cycle` 在任务 B 里**不是**特征（会泄露），"
            f"{n_feat} 个特征全部来自 1..K 的窗口内信息。",
            "",
            f"- 观测窗口 **K = {k_main}**（`features.K_MAIN`）；这张表每个数都依赖它。",
            f"- 有寿命标签的个体：**n = {n}**；删失（无标签）的：{'、'.join(censored) or '无'}。",
            f"- 折数 = {len(folds)}，**每折训练样本 = {n_train}**。",
            "",
            "## 二、结果：**没有一个特征模型赢过零基准**",
            "",
            summary.round(2).to_markdown(index=False),
            "",
            f"零基准（用另外 {n_train} 个个体的寿命均值去猜被留出的那个）平均绝对误差 "
            f"**{null_mae:.1f}** 循环；最好的单特征模型（{best_feat}）是 **{best_mae:.1f}**，"
            f"全部 {len(feat_summary)} 个特征都更差。",
            "",
            "这不是「特征没用」——**这是 n 太小时任何交叉验证都无法回答的问题**"
            f"（本数据 n = {n}）。"
            f"每折只有 {n_train} 个训练点，一个 2 参数的线性模型刚好用 2 个点解出 2 个参数，"
            "训练误差**恒等于 0**（`linear_2pt` 里有断言：这是恒等式，不是拟合得好）。"
            "于是「模型在训练集上完美」在这里没有任何信息量，剩下全是外推运气。",
            "",
            "## 三、最坏的那一折：2 个点 + 跨度极小的 x = 失控",
            "",
            f"最坏一行：用 {worst['特征']} 预测 **{worst['留出']}**，"
            f"预测 {worst['预测']:.0f} 循环、真值 {worst['真值']:.0f}，"
            f"误差 **{worst_err:+.0f}** 循环。",
            "",
            f"- 训练集两个点的 x 只差 **{worst_span:.4f}**"
            f"（这 2 个点的寿命却差 {abs(float(np.diff(labels.loc[[c for c in labels.index if c != worst['留出']]].to_numpy(float))[0])):.0f} 循环）；",
            f"- 斜率的放大系数是 **1/{worst_span:.4f} ≈ {amp:.0f}**，"
            f"于是斜率被放大到 **{worst['斜率']:.1f} 循环/单位特征**"
            f"（该特征三折的平均斜率是 {float(feat_summary.loc[feat_summary['特征'] == worst['特征'], '平均斜率'].iloc[0]):.1f}）；",
            f"- 被留出的那个个体到最近训练点的距离是 **{worst_gap:.4f}**，"
            "乘上被放大的斜率，直线一外推就飞了。",
            "",
            "**这不是「近共线」**——这里只有一个自变量，谈不上共线。准确说法是："
            "**斜率是唯一确定的（可辨识），但它的方差被 1/Δx 放大**，属于「唯一解但极不稳」。"
            "W9 的 M2 是另一种病：那条脊上有无穷多组参数给出同样的拟合（多解）。"
            "两者都让寿命数字不可信，但机制不同，报告里不能混成一句「本质一样」。",
            "",
            f"![LOBOCV](../figures/{fig_path.name})",
            "",
            "## 四、指标框架：这一轮新增了什么",
            "",
            "| 组件 | 作用 | 谁在守门 |",
            "|---|---|---|",
            "| `lobocv_folds` | 留一个体的折生成 | 只挡 n<3；折的正确性由 `self_test_framework` 守 |",
            "| `linear_2pt` | 两点线性拟合，**显式断言训练残差恒为 0** | 函数内断言（写反斜率会炸） |",
            "| `lobocv_null` | 零基准（训练均值）——任何模型必须先赢它 | `self_test_framework` 查预测值 |",
            "| `coverage` / `mpiw` | PICP 覆盖率与平均区间宽度（W17 用） | `self_test_framework` 查全覆盖/半覆盖/边界报错 |",
            "| `to_common_schema` | 模型表合并成统一列序 | **缺列报错；多模型表不许被压成一个标签** |",
            "",
            "## 五、结论：任务 B 在这份数据上只能描述",
            "",
            f"1. **不给排名**：n={n}、每折训练 {n_train} 个样本，任何排序都是噪声。"
            "报告里引用「哪个特征最好」时，必须同时给 n。",
            "2. **不给 p 值当结论**：n 这么小的时候相关系数的区间几乎覆盖 [−1, 1]，"
            f"W8 已经记过这件事（{n_feat} 个特征的 p 全部 > 0.20）。",
            "3. **可以说的是方向**：特征与寿命的**符号**在 W8 里做过描述性记录，"
            "那只能当假设，等更多电池数据才能验证。",
            "4. **任务 A 的结论不受影响**：M1–M4 是逐个体拟合，不依赖跨个体的样本量。",
            "",
            "## 六、这张表不能怎么读",
            "",
            "1. **不能把「零基准最好」读成「机器学习没用」**：它读成的是"
            f"「本数据只有 {n} 个有标签的个体，不够做跨个体学习」。",
            "2. **不能拿训练误差证明模型好**：两点拟合的训练误差恒为 0（第三节）。",
            f"3. **不能把 LOBOCV 的误差当置信区间**：{n} 个数的均值没有方差可言，"
            "任何 `mean ± std` 在这里都不成立。",
            "",
        ]
    )


if __name__ == "__main__":
    sys.exit(main())
