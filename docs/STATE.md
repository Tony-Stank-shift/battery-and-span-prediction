# 项目状态（STATE）

> 每个小项完成后更新此文件。上下文被压缩后，靠这个文件无损接上。

- **项目**：电池健康与寿命预测（Python 数据分析课程项目）
- **组号**：20 ｜ **题号**：3 ｜ **个人独立完成**
- **今天**：**2026-10-24（周六）** ← 已核实
- **硬截止**：2026-10-31（PPT + 视频，**剩 7 天**）｜ 2026-11-08 24:00（Word 报告，**剩 15 天**）
- **工作区**：`C:\Users\Tony Huang\SHU\大二\python\battery-health-and-lifespan-prediction`
- **GitHub**：`https://github.com/Tony-Stank-shift/battery-health-and-lifespan-prediction`（用户已改名完成）
- **Git**：`main` 分支，已与远端建立追踪（`push -u`）；远端自动生成的 README 已用 `-X ours` 合并（远端提交保留，未破坏性删除）
- **推送状态**：✅ **已于 2026-10-24 成功推送**（`5b2b184..50c82ed main -> main`），远端 8 个顶层条目已核实
- **必须的本地 git 配置**：`git config --local http.sslBackend openssl` ← 没有它推送必失败，见「环境坑 5」

## 当前进度

```
W1  环境与仓库骨架        [完成] sklearn 1.9.1；git 仓库；run_all 入口可用
W2  数据装载与体检脚本    [完成] 12/12 断言通过
W3  清洗与预处理          [完成] 决策表 5 条 + 个体档案表，636 行 10 列，未删任何行
W4  退化曲线              [下一步]
W5–W25                    [待办]
```

## W3 产出（已完成）

- `src/preprocess.py`：行级标记列（`cap_delta` / `is_regen` / `cap_exceeds_rated` / `cell_censored` 等），**不删行、不截断、不插值**
- `outputs/tables/data_cleaning_decisions.md`：**由代码自动生成**的决策表，5 条决策每条含「数据事实 / 决策 / 否掉的方案 / 理由 / 证据」，数字随代码重算
- `FEATURE_EXCLUDE` 字典：6 个明确禁止作特征的列，每列附理由

个体档案（也是 W5 的输入）：

| 电池 | 循环 | 首/末容量 | 最低 | 最低@循环 | 衰减% | 斜率 Ah/100循环 | EOL@1.4 | 删失 | 再生次数 |
|---|---|---|---|---|---|---|---|---|---|
| B0005 | 168 | 1.8565 / 1.3251 | 1.2875 | 166 | 30.65 | −0.3867 | 125 | 否 | 2 |
| B0006 | 168 | 2.0353 / 1.1857 | 1.1538 | 164 | 43.31 | −0.5087 | 109 | 否 | 4 |
| B0007 | 168 | 1.8911 / 1.4325 | 1.4005 | 166 | 25.94 | −0.3269 | **未到达** | **是** | 1 |
| B0018 | 132 | 1.8550 / 1.3411 | 1.3411 | 132 | 27.70 | −0.3926 | 97 | 否 | 5 |

## ⚠️ 本轮我自己造的错误（必须记住）

写 W3 决策表第 5 条时，我先写好了"叙事"再去算数，结果**数字把我的叙事全部否证**：

- 我写「全局 IQR 会把晚期的真实低容量误判成异常」→ 实测 IQR 标记 **0 个**
- 我写「用残差域检测才对」→ 那个检测器在我的窗口/阈值下标记了 **192 个**（占 30%），根本不是异常清单

**教训**：先写结论再找证据 = 编造。正确顺序是先算，再看数字说什么，然后才写结论。这已经是本项目第二次了（第一次是「PyPI 直连不通」的测量假象）。
**写进报告的价值**：改后的结论其实更强——"我们不声称做过统计离群点剔除，因为在这份数据上 3σ 和 IQR 一个点都标不出来，它们没有分辨力"。

## 第三方审查（第一轮，已完成）

独立子代理审查 W1–W2，总裁决 **needs_revision**：3 high / 5 medium / 8 low，无 blocker，**已全部修完**。

| 编号 | 级别 | 问题 | 处理 |
|---|---|---|---|
| F1 | high | README 指向不存在的 `scripts/run_all.py` | ✅ 已补 |
| F2 | high | `transform("first")` 是"位置第一行"而非"循环号最小" | ✅ 显式基准表 merge + 缺 cycle==1 报错；打乱验证偏差 0 |
| F3 | high | `soh`/`soh_vs_first` 是目标精确仿射变换 → 泄漏 | ✅ 改名 `cap_over_*`，明写禁令；残差 2.22e-16 |
| F4–F8 | medium | 恒真断言 / `atol` 被 `rtol` 架空 / 空列表崩溃 / config 耦合 / 依赖未锁上界 | ✅ 全修，断言 10 → 12 条 |
| F9–F16 | low | 死代码 / 冗余列 / BOM / 字体 / 对齐 / 文档引用 | ✅ 全清 |

审查自述的盲区：建模与图表代码（尚不存在）、报告生成、真实 GBK 控制台。**第二轮审查安排在 W14 之后。**

## 环境事实（已核实）

- Python 3.12.10 @ `AppData\Local\Programs\Python\Python312`
- numpy 2.5.1 / pandas 3.0.5 / matplotlib 3.11.1 / scipy 1.18.1 / python-docx 1.2.0
- **scikit-learn 1.9.1 已装**（用户级 site-packages），普通权限 shell 下 `import sklearn` 通过
- 附带依赖：joblib 1.6.0 / threadpoolctl 3.7.0 / narwhals 2.27.1 / cloudpickle 3.1.2

## 环境坑（重要，别重复踩）

1. **DSH 沙箱禁止 chmod**。`pip install` 与 `python -m venv` 都要在临时目录做 chmod + rmtree，被拒即失败。**解法**：装包用一次性 `sandbox_permissions=danger-full-access`。项目因此不用虚拟环境。
2. **`curl.exe` 在沙箱里跑不通**，返回 HTTP 000 且极快——**测量假象**，不是网络不通。测网络用 Python 的 `urllib.request`。
3. **中文输出乱码**：`chcp 65001` + `PYTHONIOENCODING=utf-8` 后正常。审查指出"GBK 码页导致"这个机制描述**存疑**，已降级为"待复现"。
4. **matplotlib 字体缓存**默认写到工作区外（只读）。已在 `config.py` import 期设置 `MPLCONFIGDIR`。
5. **git 推送必须用 openssl 后端**（已 `git config --local` 设好）：
   - 默认 schannel + 代理 → `AcquireCredentialsHandle failed: SEC_E_NO_CREDENTIALS`
   - 绕过代理直连 → `Connection was reset`（被阻断）
   - **`http.sslBackend=openssl` + 保留全局代理 → 读操作成功**（`ls-remote` 退出码 0）
6. **git push 必须提权**：沙箱禁止创建命名管道，凭据管理器起不来
   （`sh.exe: fatal error - couldn't create signal pipe, Win32 error 5`）。
   用一次性 `sandbox_permissions=danger-full-access` 后推送成功。
   **每次 push 都要这样，属于既定代价。**
7. **PowerShell 的 `Get-Content` 默认按 GBK 读文件**，会把 UTF-8 的中文显示成乱码。
   那是显示问题，不是文件问题——核对文件内容请用 Python 按 `encoding="utf-8"` 读，或 `Get-Content -Encoding UTF8`。

## 已落盘文件

```
README.md                      主线、数据陷阱、运行说明、两条给未来的警告
requirements.txt               依赖（全部锁上界，含 tabulate）
.gitignore
data/raw/battery_capacity.csv  636 行（复制自课程下发文件，未改动原文件）
src/config.py                  路径/常量/绘图（含 MPLCONFIGDIR 与字体探测）
src/load_data.py               读入 + 派生列（含泄漏警告与显式基准表）
src/sanity_check.py            12 条可复现断言（12/12 PASS）
src/preprocess.py              清洗决策 + 行级标记 + 个体档案 + 自动生成决策表
scripts/run_all.py             一键入口（当前 2 步）
outputs/tables/data_cleaning_decisions.md   清洗决策表（代码生成，勿手改）
docs/DECISIONS.md              D1–D15
docs/STATE.md                  本文件
outputs/_tmp/                  临时与验收脚本（已 gitignore）
```

## 待决问题

1. ✅ **推送已打通**（原问题：认证未验证）。结论：凭据管理器本身可用，但**在沙箱内无法启动**（禁止创建命名管道 `sh.exe Win32 error 5`）。**以后每次 push 都需要一次 `danger-full-access` 提权**，这是既定代价，不是故障。
2. `outputs/_tmp/pytmp*/` 残留 pip 临时目录无法删除（chmod 受限），已 gitignore。

## 下一步

**W4 退化曲线（图1/图2）**：四体容量退化曲线 + SOH 曲线，含 EOL 阈值线、删失个体标注、再生事件点标记。
验收：图含 EOL 线、4 个体可辨识、删失个体有标注。
