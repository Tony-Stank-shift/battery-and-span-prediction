# 项目状态（STATE）

> 每个小项完成后更新此文件。上下文被压缩后，靠这个文件无损接上。

- **项目**：电池健康与寿命预测（Python 数据分析课程项目）
- **组号**：20 ｜ **题号**：3 ｜ **个人独立完成**
- **今天**：2026-10-22
- **硬截止**：2026-10-31（PPT + 视频）｜ 2026-11-08 24:00（Word 报告）
- **工作区**：`C:\Users\Tony Huang\SHU\大二\python\battery-health-and-span-prediction`

## 当前进度

```
W1  环境与仓库骨架        [进行中]
W2  数据装载与体检脚本    [待办]
W3  清洗与预处理          [待办]
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
- numpy 2.5.1 / pandas 3.0.5 / matplotlib 3.11.1 / scipy 1.18.1 / python-docx 1.2.0 已装
- scikit-learn：**安装中**（清单 A 已批准）

## 待决问题

1. **仓库名疑似笔误**：`battery-health-and-span-prediction` 里的 `span` 可能是 `lifespan` 的笔误。报告附录会写这个链接，老师会点。**要不要在推送前改仓库名？**（改 GitHub 仓库设置 + 改 remote 即可，成本 2 分钟）
2. `pip install scikit-learn` 已跑 6 分钟未结束，疑似直连 PyPI 慢。若失败考虑换清华镜像（需另行批准）

## 下一步

W1 收尾：README / requirements / .gitignore 已写，尚缺 `docs/DECISIONS.md` 已写；剩下 `git init` + 首次提交 + 校验 `import sklearn`。
