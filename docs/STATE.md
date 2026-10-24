# 项目状态（STATE）

> 每个小项完成后更新此文件。上下文被压缩后，靠这个文件无损接上。

- **项目**：电池健康与寿命预测（Python 数据分析课程项目）
- **组号**：20 ｜ **题号**：3 ｜ **个人独立完成**
- **今天**：**2026-10-24（周六）** ← 已核实（首次记录误写为 10-22，会话跨了两天，已更正）
- **硬截止**：2026-10-31（PPT + 视频，**剩 7 天**）｜ 2026-11-08 24:00（Word 报告，**剩 15 天**）
- **工作区**：`C:\Users\Tony Huang\SHU\大二\python\battery-health-and-span-prediction`
- **Git**：`main` 分支

## 当前进度

```
W1  环境与仓库骨架        [完成] sklearn 1.9.1 可用，git 仓库已建
W2  数据装载与体检脚本    [完成] 10/10 断言通过，退出码 0
W3  清洗与预处理          [下一步]
W4  退化曲线              [待办]
W5  个体差异五维量化      [待办]
W6  容量再生检测          [待办]
W7  EOL 阈值敏感性        [待办]
W8  特征工程              [待办]
W9  评测框架              [待办]
W10 M1 线性基线           [待办]
W11 M2 双指数             [待办]
W12 M3 RF 外推失效        [待办]
W13 M4 趋势+GPR           [待办]
W14 模型对比主表          [待办] ★第三方审查
W15 三层不确定性          [待办]
W16 最早可信预警时刻      [待办]
W17 两场景误差放大        [待办]
W18 区间校准              [待办]
W19 更换策略建议          [待办]
W20 run_all 复现          [待办]
W21 Word 报告             [待办] ★第三方审查
W22 PPT                   [待办]
W23 逐字稿与演示脚本      [待办]
W24 Git 推送与 README     [待办]
W25 收官大检查            [待办]
```

## 环境事实（已核实）

- Python 3.12.10 @ `AppData\Local\Programs\Python\Python312`
- numpy 2.5.1 / pandas 3.0.5 / matplotlib 3.11.1 / scipy 1.18.1 / python-docx 1.2.0
- **scikit-learn 1.9.1 已装**（用户级 site-packages：`AppData\Roaming\Python\Python312\site-packages`），普通权限 shell 下 `import sklearn` 通过
- 附带装的依赖：joblib 1.6.0 / threadpoolctl 3.7.0 / narwhals 2.27.1 / cloudpickle 3.1.2

## 环境坑（重要，别重复踩）

1. **DSH 沙箱禁止 chmod**。`pip install` 与 `python -m venv` 都需要在临时目录做 chmod + rmtree 清理，被拒后会失败：
   - `pip` → `PermissionError: [WinError 5]` 写 `AppData\Local\Temp\dsh-*\...`
   - `venv` → `ensurepip` 返回 exit 1，venv 里没有 pip
   **解法**：装包用一次性 `sandbox_permissions=danger-full-access`；日常运行只需读权限，不受影响。
   本项目因此**不用虚拟环境**，直接装用户级 site-packages。README 需写清这一点。
2. **`curl.exe` 在沙箱里跑不通**，返回 HTTP 000 且极快（0.03–0.15s）。这是**测量假象**，不是网络不通。测网络请用 `python` 的 `urllib.request`。
3. **Windows 控制台 GBK 码页**下 Python 中文输出乱码。已确认源码是 UTF-8，解法：`chcp 65001` 或设 `PYTHONIOENCODING=utf-8`。
4. **matplotlib 字体缓存**默认写到 `%USERPROFILE%\.matplotlib`（工作区外，沙箱只读）。W4 出图前需设 `MPLCONFIGDIR` 到工作区内。

## 已落盘文件

```
README.md                      项目主线与数据陷阱说明
requirements.txt               分析依赖 + 报告生成依赖
.gitignore
data/raw/battery_capacity.csv  636 行（复制自课程下发文件，未改动原文件）
src/config.py                  路径/常量/绘图风格
src/load_data.py               读入 + 派生列（soh, soh_vs_first）
src/sanity_check.py            10 条可复现断言（10/10 PASS）
docs/DECISIONS.md              D1–D12
docs/STATE.md                  本文件
outputs/_tmp/                  临时目录（已 gitignore）
```

## 待决问题

1. **仓库名疑似笔误**：`battery-health-and-span-prediction` 里的 `span` 可能是 `lifespan`。报告附录会写这个链接，老师会点开。**要不要在推送前改名？**（GitHub 设置改名 + 改 remote，约 2 分钟）
2. 第三方审查子代理（7be55d08）结论待回。

## 下一步

W3 清洗与预处理：产出清洗决策表——温度零方差 / B0006 超额定 / 容量再生 / 删失 四件事，各给处理方式与理由。
