"""数据读入与基础派生列。

原则：本模块只做"读进来 + 加列"，不做任何清洗判断。
清洗决策集中在 preprocess.py（W3 待建）。

⚠️ 关于派生列的重要警告
------------------------
`cap_over_rated` = capacity / 2.0 与 `cap_over_first` = capacity / 该电池首循环容量，
**都是目标变量 capacity_Ah 的精确仿射变换**（同乘一个正数）。

因此它们 **不得作为预测特征**：拿它们当特征去预测容量 / SOH / RUL，等价于把答案
喂给模型，会得到 R²≈1 的假结果。它们只有两个正当用途：
  1. 画图时把纵轴换成"相对健康度"，便于跨个体比较；
  2. 定义预测目标本身（例如把"预测容量"改写成"预测相对健康度"）。
真正的特征只能来自 rolling / expanding 历史窗口，见 features.py（W8 待建）。

另：`cap_over_first` 有一个部署前提——目标电池必须已经有了第 1 个循环的数据。
现场问答时会被问"预测时哪来的首循环容量"，报告里要交代。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import BATTERIES, DATA_RAW, RATED_CAPACITY_AH  # noqa: E402

EXPECTED_COLUMNS = ["battery_id", "cycle", "capacity_Ah", "ambient_temp_C"]


def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    """读入原始 CSV，做类型规整，按 (电池, 循环) 排序。

    编码显式指定 utf-8-sig：这份 CSV 带 UTF-8 BOM（首字节 EF BB BF）。
    pandas 的 C 解析器目前恰好会剥掉它，但任何用 encoding="utf-8" 手工读表头的人
    会拿到 '\ufeffbattery_id'。显式写死，不依赖解析器的宽容。

    Returns
    -------
    DataFrame
        列：battery_id(str), cycle(int), capacity_Ah(float), ambient_temp_C(float)
    """
    path = Path(path) if path is not None else DATA_RAW
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件：{path}")

    df = pd.read_csv(path, encoding="utf-8-sig")

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"数据缺少列：{missing}；实际列为 {list(df.columns)}")

    df = df[EXPECTED_COLUMNS].copy()
    df["battery_id"] = df["battery_id"].astype(str).str.strip()
    df["cycle"] = df["cycle"].astype(int)
    df["capacity_Ah"] = df["capacity_Ah"].astype(float)
    df["ambient_temp_C"] = df["ambient_temp_C"].astype(float)

    return df.sort_values(["battery_id", "cycle"]).reset_index(drop=True)


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """加入派生列（**均为目标变量的单调变换，禁止作预测特征**）。

    新增列：
    - cap_over_rated : capacity_Ah / 额定容量，即相对额定值的健康度
    - cap_over_first : capacity_Ah / **该电池第 1 个循环**的容量

    实现说明（这里踩过一次坑）：
    不能用 `groupby("battery_id")["capacity_Ah"].transform("first")` —— 那是取
    **数据里的第一行**，不是"循环号 = 1 的那一行"。两者只有在输入恰好排好序时
    才相同，属于隐式契约。W8/W9 做时序切分时一定会破坏这个前提，导致基准值
    悄悄变成窗口首值。所以这里改用**显式基准表 + merge**，并在缺少 cycle==1
    记录时直接报错，而不是静默给出错误基准。
    """
    out = df.copy()

    if len(out) == 0:
        raise ValueError("add_derived 收到空 DataFrame")

    # 每个电池必须从 cycle==1 开始，否则"首循环"的定义不存在
    first_cycles = out.groupby("battery_id")["cycle"].min()
    bad = first_cycles[first_cycles != 1]
    if len(bad) > 0:
        raise ValueError(
            "以下电池缺少 cycle==1 的记录，无法定义首循环基准："
            f"{bad.to_dict()}"
        )

    # 显式基准表：只取 cycle==1 的行
    ref = (
        out.loc[out["cycle"].eq(1), ["battery_id", "capacity_Ah"]]
        .rename(columns={"capacity_Ah": "_cap_first"})
        .copy()
    )
    if ref["battery_id"].duplicated().any():
        raise ValueError("存在重复的 cycle==1 记录，首循环基准不唯一")

    out = out.merge(ref, on="battery_id", how="left", validate="many_to_one")
    if out["_cap_first"].isna().any():
        raise ValueError(f"{int(out['_cap_first'].isna().sum())} 行找不到对应电池的首循环基准")

    out["cap_over_rated"] = out["capacity_Ah"] / RATED_CAPACITY_AH
    out["cap_over_first"] = out["capacity_Ah"] / out["_cap_first"]
    out = out.drop(columns=["_cap_first"])

    return out.sort_values(["battery_id", "cycle"]).reset_index(drop=True)


def split_by_battery(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """按电池个体切分，返回 dict（键顺序固定为 BATTERIES）。

    未知的电池 id 会直接报错，不静默丢弃——静默丢弃会让"4 个个体"这类结论
    在数据被换掉时仍然"通过"。
    """
    unknown = sorted(set(df["battery_id"]) - set(BATTERIES))
    if unknown:
        raise ValueError(f"数据中出现未登记的电池 id：{unknown}；已知：{BATTERIES}")

    return {
        b: df[df["battery_id"] == b].sort_values("cycle").reset_index(drop=True)
        for b in BATTERIES
        if (df["battery_id"] == b).any()
    }


if __name__ == "__main__":
    raw = load_raw()
    print(f"读入 {len(raw)} 行 × {raw.shape[1]} 列")
    print(raw.head(3).to_string(index=False))
    print()
    print("按电池分组：")
    for b, sub in split_by_battery(raw).items():
        print(f"  {b}: {len(sub)} 个循环（{sub['cycle'].min()}–{sub['cycle'].max()}）")

    print()
    print("派生列：")
    derived = add_derived(raw)
    print(derived[["battery_id", "cycle", "capacity_Ah", "cap_over_rated", "cap_over_first"]].head(3).to_string(index=False))
