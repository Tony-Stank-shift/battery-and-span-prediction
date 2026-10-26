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

当前阶段（W1–W16）管线包含 14 步：**数据体检 → 清洗与预处理 → 退化曲线 → 个体差异五维量化 → 容量再生分解 → EOL 阈值敏感性 → 早期特征与泄漏证明 → 评测口径 → M1/M2 参数模型 → M3 随机森林（外推失效反例）→ M4 趋势+GPR → 四模型对比主表 → 三层不确定性 → 最早可信预警时刻**。
W20 会扩展成完整管线（… → 校准 → 报告）。

也可以单独跑某一步：

```bash
python src/sanity_check.py       # 数据体检（11 条断言）
python src/preprocess.py         # 清洗决策表 + 个体档案
python src/explore.py            # 退化曲线与个体差异（fig01/fig02）
python src/individual.py         # 个体差异五维量化（fig03）
python src/regeneration.py       # 容量再生检测与共模分解（fig04）
python src/sensitivity.py        # EOL 阈值敏感性（fig05 + 断点表）
python src/features.py           # 早期特征 + 未来信息泄漏证明
python src/evaluate.py           # 评测口径 + 任务 B 的 LOBOCV（fig11）
python src/models_parametric.py  # M1 线性 / M2 双指数（fig06）
python src/models_ml.py          # M3 随机森林：外推失效（fig07/fig08）
python src/models_gpr.py         # M4 趋势 + GPR：EOL 后验分布（fig09/fig10）
python src/compare.py            # 四模型对比主表（fig12 + 长表 CSV）
python src/uncertainty.py        # 三层不确定性折算成循环数（fig13）
python src/warning.py            # 最早可信预警时刻 t*（fig14）
python src/amplification.py      # 两场景误差放大：容量 vs 寿命（fig15）
python src/calibration.py        # 区间校准 PICP/MPIW（fig16）

python scripts/mutation_test.py  # 只查「断言是不是空的」：38 个注入错误必须全部被抓
python scripts/adversarial_test.py  # 病态输入：43 条（NaN/常数/重复/单点/空表…）不许静默给错答案
```

### 关于运行环境

本项目**不使用虚拟环境**，依赖直接装在用户级 site-packages（`%APPDATA%\Python\Python312\site-packages`）。
原因不是图省事：在 DSH 文件沙箱下 `pip` 与 `python -m venv` 都会因为无法对临时目录执行 chmod 而失败，详见 `docs/STATE.md` 的「环境坑」。**你在自己的机器上跑完全可以用 venv。**

## 已知数据陷阱

这些都在 `src/sanity_check.py` 里做成了可复现断言（11 条，全绿）：

- `ambient_temp_C` 全部 636 行都是 24.0 —— **零方差**，不能当特征
- B0006 前 7 个循环容量超过额定 2 Ah（最高 2.0353 Ah）
- 共 12 个「电池×循环」对出现 >0.05 Ah 的容量回升，最大 +0.1519 Ah
- B0007 最低容量 1.4005 Ah @ 第 166 循环，**从未跌破 1.4 Ah** → 右删失
- B0018 在 1.4 Ah 口径下**第 97 循环就已失效**（观测到此为止共 132 循环，最低 1.3411 Ah）—— 它是**已观测失效**，不是删失，别和 B0007 混为一谈

## 寿命这个数字有多依赖阈值

`src/sensitivity.py` 把 τ 从 1.25 扫到 1.60 Ah（断点精确计算，不用网格）：

| 阈值 τ | B0005 | B0006 | B0007 | B0018 |
|---|---|---|---|---|
| 1.60 Ah（额定的 80%） | 75 | 63 | 86 | 45 |
| **1.40 Ah（额定的 70%，本项目主线）** | **125** | **109** | 删失（>168） | **97** |
| 1.30 Ah | 162 | 140 | 删失（>168） | 删失（>132） |

两个必须知道的事实：

1. **阈值每偏 0.01 Ah（额定容量的 0.5%），寿命差 3.00–4.00 个循环**（三块非删失电池，±0.05 Ah 口径）；
   ±0.05 Ah 的窗口里极差 **30–40 循环**。
   ⚠️ **B0007 不能并进这两个区间**：它在 1.4005 Ah 以下就删失，有效窗口只有 0.0495 Ah，
   所以它的 22 循环 / 每 0.01 Ah 4.44 个循环不是 ±0.05 口径的数；
   换成等宽窗口 [1.4005, 1.5005] 后是 40 循环 / 4.00。
2. **但「谁先坏」的排序一次都没变**：四块都有确定寿命的全部 80 个 τ 上恒为
   B0018 < B0006 < B0005 < B0007。同样稳定的还有一条反面事实——
   **B0007 的「删失」只靠 0.0005 Ah 的余量成立**（= 额定容量的 0.025%），
   阈值抬到 1.4005 Ah 它就变成「第 166 循环失效」。

换一种 EOL 定义（80% 额定 / 70% 或 80% 首循环），寿命能变多少：
**四种口径都算出确定值**的电池里最大是 **2.16 倍**（B0005：75 → 162 循环）；
而 B0007 / B0018 在部分口径下删失，它们的比值只有下界（>1.97 / **>2.96**），
所以真实的最大比值**至少 2.96 倍**——不能写成「最多 2.2 倍」，那是把删失当成了已知值。

## 目录结构

```
src/          已建：config / load_data / sanity_check / preprocess / figures / explore / individual /
              regeneration / sensitivity / features / evaluate / models_parametric / models_ml /
              models_gpr / compare / uncertainty / warning / amplification / calibration
              计划中：策略建议（W19）
scripts/      run_all.py 一键重跑（当前 16 步，W20 扩展为完整管线）；
              mutation_test.py 变异测试（38 条）；adversarial_test.py 病态输入测试（43 条）
outputs/      图表与指标（全部由代码生成，不含手工修改）
              figures/fig01–fig16；tables/… model_comparison.md, uncertainty_budget.md,
              warning_time.md, error_amplification.md, interval_calibration.md；
              metrics/… 同名 CSV
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
