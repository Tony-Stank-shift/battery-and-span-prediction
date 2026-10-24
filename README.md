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

文件带 UTF-8 BOM，代码里统一用 `encoding="utf-8-sig"` 读取。

## 快速开始

```bash
pip install -r requirements.txt
python scripts/run_all.py
```

当前阶段（W1–W2）管线只包含**数据体检**一步，W20 会扩展成完整管线（清洗 → 探索 → 特征 → 建模 → 不确定性 → 图表 → 报告）。

也可以单独跑某一步：

```bash
python src/sanity_check.py
```

### 关于运行环境

本项目**不使用虚拟环境**，依赖直接装在用户级 site-packages（`%APPDATA%\Python\Python312\site-packages`）。
原因不是图省事：在 DSH 文件沙箱下 `pip` 与 `python -m venv` 都会因为无法对临时目录执行 chmod 而失败，详见 `docs/STATE.md` 的「环境坑」。**你在自己的机器上跑完全可以用 venv。**

## 已知数据陷阱

这些都在 `src/sanity_check.py` 里做成了可复现断言（12 条，全绿）：

- `ambient_temp_C` 全部 636 行都是 24.0 —— **零方差**，不能当特征
- B0006 前 7 个循环容量超过额定 2 Ah（最高 2.0353 Ah）
- 共 12 次 >0.05 Ah 的容量回升，最大 +0.1519 Ah
- B0007 最低容量 1.4005 Ah @ 第 166 循环，**从未跌破 1.4 Ah** → 右删失
- B0018 在第 132 循环处截断，同样未观测到失效

## 目录结构

```
src/          分析模块（已建：config / load_data / sanity_check）
              计划中：preprocess / regeneration / eol / features / models / evaluate / uncertainty
scripts/      run_all.py 一键重跑（当前 1 步，W20 扩展为完整管线）
outputs/      图表与指标（全部由代码生成，不含手工修改）
report/       计划中：Word 报告 / PPT / 演讲逐字稿（W21–W23 生成）
docs/         项目状态与决策记录
data/raw/     课程下发的 636 行数据
```

## ⚠️ 两条给未来自己的警告

1. **`cap_over_rated` / `cap_over_first` 不得作为预测特征。**
   它们是 `capacity_Ah` 的精确仿射变换（`cap_over_first × 首循环容量 = capacity`，最大残差 2.22e-16）。拿它们当特征预测容量 / SOH / RUL 等于把答案喂给模型，会得到 R²≈1 的假结果。真正的特征只能来自 rolling / expanding 历史窗口。
   更隐蔽的是：**四块电池的首循环容量恰好都等于各自最大值**（1.8565 / 2.0353 / 1.8911 / 1.855），这层巧合会把「用 first 归一化」的泄漏彻底藏起来。

2. **数据体检里写死了期望常量**（636 行、2.0353、0.1519、1.4005 等）。课程方若更新数据集，断言会全部 FAIL —— 这是特性不是 bug，但要知道这不是代码坏了。

## 说明

个人独立完成。每一项分析决策及其理由记录在 `docs/DECISIONS.md`。
