# 电池健康评估与剩余寿命预测

《Python 数据分析编程》课程项目 · 第 20 组 · 题 3

## 一句话

**电池什么时候该换，这个答案有多可靠。**

## 项目在做什么

锂离子电池随充放电循环老化，容量下降。但"下降"这件事在真实数据里并不光滑：容量会回升、阈值往上挪一点寿命就换一个数、有的样本到观察结束还没坏。

本项目把**不确定性**当作主线，做三层拆解：

| 层 | 问的问题 | 怎么量化 |
|---|---|---|
| 定义不确定 | "寿命"的阈值该选多少？ | EOL 阈值全区间扫描 |
| 模型不确定 | 这条退化曲线有多可信？ | 高斯过程后验方差 → 后验采样 |
| 数据不确定 | 曲线为什么不光滑？ | 容量再生事件 + 残差观测噪声 |

三层最后折算到**同一个决策量纲** —— EOL 循环数 ± 多少个循环 —— 这样"哪种不确定性最值得先消除"才能被比较，而不是各说各话。

## 数据

`data/raw/battery_capacity.csv`：636 行 × 4 列，4 个电池个体（B0005 / B0006 / B0007 / B0018）的逐循环放电容量，恒温 24 °C。整理自 NASA PCoE 公开电池老化数据集。

> Saha, B., & Goebel, K. (2007). *Battery Data Set*. NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.

**已知的数据陷阱**（都在 `src/sanity_check.py` 里做成了可复现的断言）：

- `ambient_temp_C` 全部 636 行都是 24.0 —— **零方差**，不能当特征用
- B0006 前 7 个循环容量超过额定 2 Ah（最高 2.0353 Ah）
- 共 12 次 >0.05 Ah 的容量回升，最大 +0.1519 Ah
- B0007 最低容量 1.4005 Ah @ 第 166 循环，**从未跌破 1.4 Ah** → 右删失
- B0018 在第 132 循环处截断，同样未观测到失效

## 运行

```bash
pip install -r requirements.txt
python scripts/run_all.py
```

图表输出到 `outputs/figures/`，指标表输出到 `outputs/tables/` 与 `outputs/metrics/`。

## 目录结构

```
src/       分析模块（单一职责，可单独运行）
scripts/   run_all.py 一键重跑
outputs/   图表与指标（全部由代码生成，不含手工修改）
report/    Word 报告 / PPT / 演讲逐字稿
docs/      项目状态与决策记录
```

## 说明

个人独立完成。每一项分析决策及其理由记录在 `docs/DECISIONS.md`。
