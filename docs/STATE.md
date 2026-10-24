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
W2  数据装载与体检脚本    [完成] 11/11 断言通过
W3  清洗与预处理          [完成] 决策表 5 条 + 个体档案表，636 行 10 列，未删任何行
W4  退化曲线与个体差异    [完成] fig01/fig02 已生成，逐张目视核对并修图
W5  个体差异五维量化      [完成] 五维档案 + fig03（六面板）；⚠️ 加速比全 <1 是重要发现
W6  容量再生检测与分解    [完成] 12 对 / 独立 9 次 / 共模占残差方差 49.6%
W7  EOL 阈值敏感性        [下一步]
W8–W25                    [待办]
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
| F4–F8 | medium | 恒真断言 / `atol` 被 `rtol` 架空 / 空列表崩溃 / config 耦合 / 依赖未锁上界 | ✅ 全修，断言 10 → 12 条（C12 后按 D17 删除，**现为 11 条**） |
| F9–F16 | low | 死代码 / 冗余列 / BOM / 字体 / 对齐 / 文档引用 | ✅ 全清 |

审查自述的盲区：建模与图表代码（尚不存在）、报告生成、真实 GBK 控制台。

## 第三方审查（第二轮：W3–W4，已完成）

独立子代理，总裁决 **needs_revision**（2 high / 3 medium / 6 low，无 blocker）。
它不只看代码：清空 `outputs/` 后重跑管线做**字节级比对**、独立重算决策表 20 个数字、
把 figure 真建出来读 artist 坐标验证 ★/▼ 位置、量化泄漏受影响行数。

被它抓到的、我自己没看到的**硬错**：
**README 写「B0018 同样未观测到失效」是假的** —— B0018 在第 97 循环就跌破 1.4 Ah（最低 1.3411），
而 README 是老师第一眼看的文件，这句还挂在「断言全绿」标题下面。

| 编号 | 级别 | 问题 | 处理 |
|---|---|---|---|
| F1 | high | `FEATURE_EXCLUDE` 漏了 `cell_censored`（真·未来信息，328/636 行受影响）+ `is_regen` + `cap_exceeds_rated`；按「10 列 − 6 列」减法，被允许作特征的正好全是该排除的 | ✅ 补齐为 9 列；新增 `PROFILE_EXCLUDE`（档案表 10 列不可回填）；新增白名单断言 `assert_feature_exclude_complete()`，以后漏登记直接报错 |
| F2 | high | README 的 B0018 陈述与代码/决策表/数据三方矛盾 | ✅ 改正，并写清它与 B0007 的区别（一个是已观测失效，一个是删失） |
| F3 | medium | `EOL_COMPARISON_IS_INCLUSIVE` 是**死代码**；C12 会掐死 W7 的 τ 扫描（改 EOL_RATIO → 断言 FAIL → 管线第 1 步中止） | ✅ 新增 `config.is_at_eol()` 统一入口；**删除 C12**；列名改 `eol_cycle`；图的阈值与观测长度改从 config 与数据取 |
| F4 | medium | 决策 3 把再生归因于「个体电化学现象」，与跨电池共模证据冲突 | ✅ 机制降级为「假设，未经本数据验证」；补同号率 0.840/0.832、循环 90 共现；把波动拆成 **共模 + 个体** |
| F5 | medium | 「0 缺失、0 重复」是**写死字符串** | ✅ 改为在**原始**数据上现算（并发现一个坑：`cap_delta` 的 4 个合法 NaN 不能混进来） |
| F6 | low | `cell_profile` 又用 `iloc[0]` 取首循环 | ✅ 先 `assert cycle==1`；fig02(b) 改用档案的 `cap_first` |
| F7 | low | 斜率差值 +0.0101 与显示值相减 +0.0102 不符 | ✅ 统一按 4 位小数计算差值 |
| F8 | low | `EOL_LABEL` 零调用方，图例里**没有 EOL 线** | ✅ 传 `label=EOL_LABEL` |
| F9 | low | 轴界全部写死，换数据会**静默裁剪**曲线 | ✅ 改为按数据范围自动定界 + `assert` 未裁剪 |
| F10 | low | README/STATE 过期；`tabulate` 无上界 | ✅ 更新；补 `<1` |
| F11 | low | 决策 5 只报 IQR 下界、只做池化口径 | ✅ 补上界 + **分电池**口径（结论仍全为 0） |
| F12 | low | 排除理由口径不统一 | ✅ 改写 `cycle` / `cap_delta` 两条 |

**为什么第二轮提前到 W3–W4**（原定 W14）：D16 规定「文字产出物必须单独审」，
而 W3 决策表正是文字产出物。执行下来证明这条规则立刻见效——F2 就是文字错。

**第三轮审查**：按 D16，覆盖 W5+W6。

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
src/sanity_check.py            11 条可复现断言（11/11 PASS）
src/preprocess.py              清洗决策 + 行级标记 + 个体档案 + 自动生成决策表
src/figures.py                 绘图公共工具（保存 / EOL 线 / 额定线 / 右侧标签）
src/explore.py                 W4 探索性图表（fig01/fig02）
src/individual.py              W5 个体差异五维量化（fig03 + 档案表）
src/regeneration.py            W6 容量再生检测与分解（fig04 + 事件表）
scripts/run_all.py             一键入口（当前 5 步）
outputs/figures/fig01_capacity_degradation.png
outputs/figures/fig02_individual_differences.png
outputs/figures/fig03_individual_differences_facets.png
outputs/figures/fig04_trend_residual.png
outputs/tables/data_cleaning_decisions.md   清洗决策表（代码生成，勿手改）
outputs/tables/individual_differences.md    五维档案（代码生成，勿手改）
outputs/tables/regeneration_events.md       再生事件表（代码生成，勿手改）
docs/DECISIONS.md              D1–D15
docs/STATE.md                  本文件
outputs/_tmp/                  临时与验收脚本（已 gitignore）
```

## 待决问题

1. ✅ **推送已打通**（原问题：认证未验证）。结论：凭据管理器本身可用，但**在沙箱内无法启动**（禁止创建命名管道 `sh.exe Win32 error 5`）。**以后每次 push 都需要一次 `danger-full-access` 提权**，这是既定代价，不是故障。
2. `outputs/_tmp/pytmp*/` 残留 pip 临时目录无法删除（chmod 受限），已 gitignore。

## W4 已完成（附注）

- `fig01`：绝对退化曲线，含 EOL 线（1.4 Ah）、额定线（2.0 Ah）、EOL 交叉点（▼ 97/109/125）、
  容量再生事件（空心圈共 12 个）、B0007 最低点（★ @166, 1.4005）+ 左下删失说明框。
- `fig02`：(a) SOH = C/2.0 含 0.7 线，并标出 B0006 初始 SOH > 100%；
  (b) 相对首循环归一化，用于比较「去掉初始容量差异后」的衰减形状，
  标题已声明「仅用于展示，不得作特征」（D14）。
- 两张图都做了**目视核对**（不是只看退出码）。第一版发现三个问题并修掉：
  ① ▼ 标记太小被曲线盖住；② B0006 尾部被 y 轴下界切掉；③ B0007 箭头端点对不准
  （改为星标定点 + 左侧文字，不画跨图箭头）。

## W5 已完成（第三轮审查后已重写）

| 维度 | 结果 |
|---|---|
| ① 水平 | 初始容量 1.8550–2.0353 Ah（相差 0.1803 Ah，约 9.7%）；B0006 初始 SOH 101.8% |
| ② 速率 | −0.5087–−0.3269 Ah/(100循环)，最快是最慢的 **1.56 倍** |
| ③ 形状 | 线性 R² 0.940–0.976；**无量纲曲率 0.031–0.356 全为正（整体凸）**；固定 h=66 加速比 0.51–1.62（**两块 > 1**） |
| ④ 波动 | 残差标准差 σ 0.0109–0.0223 Ah；回升点贡献方差 28%–45% |
| ⑤ 寿命 | 已观测 EOL 97–125；B0007 右删失 |

⚠️ **被审查推翻的两条（原先写错的）**：

1. 原写「加速比全部 < 1 → 后段变缓，不是加速老化」→ **不可靠**。加速比依赖切分点 h，
   B0018 半段长 66 而其余三块 84，不是同一把尺。改用固定 h=66 后 **B0005=1.62、B0007=1.11 翻到 >1**；
   末段/首段（20 循环）斜率比还显示 B0007 末端重新变陡（1.265）。
   正确表述：曲线**整体是凸的**（无量纲曲率全正），但「变缓」并不贯穿全程。
2. 原写「σ 主要被回升事件驱动」→ **是编的**。方差型尺度下回升贡献 28%–45%；
   而 MAD 型尺度下只变 0–14%（滚动中位数残差在 0 处有质量堆，中位数对少量点不敏感）。
   **同一份数据、不同尺度统计量给出不同结论**——这条本身就该写进报告。

## W6 已完成（第三轮审查后已重写）

| 结果 | 数值 |
|---|---|
| (电池, 循环) 对 | **12** 个（B0005 2 / B0006 4 / B0007 1 / B0018 5） |
| 同步事件 | **2 处**：循环 48（2 块）、循环 90（3 块） |
| 同刻合并后事件数 | **9**（12 − 5 + 2）——**这是约定不是统计结果**，报告必须写明 |
| 共模占残差方差 | **四块 49.6%（零基准 25%）**；**同批三块 82.4%（零基准 33.3%）** |
| 残差两两相关 | 三块之间 0.85–0.95；B0018 与三者 −0.03~−0.13 |

⚠️ **被审查推翻的三条**：

1. 原只报「49.6%」→ **漏零基准**。四块互相独立时，同样的均值构造也能吃掉 1/k = 25%。
   且 49.6% 被不相关的 B0018 稀释了 33pp，**同批三块口径是 82.4%**。
2. 原写「B0018 与共模 +0.348，属弱正相关」→ **读反了**。共模里含该电池自己，
   零基准逐电池不同；B0018 的观测值**低于**自身零基准（蒙特卡洛 p=0.998），
   即它**不属于**这个同步簇。
3. 原写「B0018 与另外三块相关仅 −0.09」→ 环形移位置换检验 p=0.28–0.65，**检不出相关**。
   硬证据应改用同号率置换检验：三块之间 p=0.000，与 B0018 全部 p>0.28。

## 下一步

**W7 EOL 阈值敏感性**：τ ∈ [1.25, 1.60] 扫描，4 条 N_EOL(τ) 曲线；
B0007 画成「无穿越点/删失」；给出常用区间 [1.35, 1.45] 内各电池 EOL 的极差（循环数）。
验收：B0007 标为无穿越、4 条曲线、极差可复现。
实现约束（D17）：**τ 必须作为参数传入，不许改 config 常量**。

**第三轮审查（W5+W6）已完成**：needs_revision，4 high / 8 medium / 8 low。
本轮已按 findings 重写 individual.py 与 regeneration.py，正在收尾。
