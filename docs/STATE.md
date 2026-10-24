# 项目状态（STATE）

> 每个小项完成后更新此文件。上下文被压缩后，靠这个文件无损接上。

- **项目**：电池健康与寿命预测（Python 数据分析课程项目）
- **组号**：20 ｜ **题号**：3 ｜ **个人独立完成**
- **今天**：**2026-10-24（周六）** ← 已核实（首次记录误写为 10-22，会话跨了两天，已更正）
- **硬截止**：2026-10-31（PPT + 视频，**剩 7 天**）｜ 2026-11-08 24:00（Word 报告，**剩 15 天**）
- **工作区**：`C:\Users\Tony Huang\SHU\大二\python\battery-health-and-lifespan-prediction`
  （2026-10-24 由 `...-span-prediction` 改名；本地目录已改，**GitHub 侧改名待用户在网页操作**）
- **Git**：`main` 分支；`origin` = `https://github.com/Tony-Stank-shift/battery-health-and-lifespan-prediction.git`（仓库尚未推送）

## 当前进度

```
W1  环境与仓库骨架        [完成] sklearn 1.9.1；git 仓库；run_all 入口可用
W2  数据装载与体检脚本    [完成] 12/12 断言通过（审查后由 10 条扩到 12 条）
W3  清洗与预处理          [下一步]
W4–W25                    [待办]
```

## 第三方审查（第一轮，已完成）

独立子代理审查 W1–W2，总裁决 **needs_revision**：3 high / 5 medium / 8 low，无 blocker。
处理情况：

| 编号 | 级别 | 问题 | 处理 |
|---|---|---|---|
| F1 | high | README 指向不存在的 `scripts/run_all.py` | ✅ 已补最小可运行的一键入口 |
| F2 | high | `transform("first")` 是"位置第一行"而非"循环号最小" | ✅ 改为显式基准表 merge + 缺 cycle==1 时报错；已用打乱输入反例验证（偏差 0） |
| F3 | high | `soh`/`soh_vs_first` 是目标的精确仿射变换 → 泄漏陷阱 | ✅ 改名 `cap_over_rated`/`cap_over_first`；docstring 与 README 明写"禁止作特征"；实证残差 2.22e-16 |
| F4 | medium | C2 恒真、C1 列维度恒真 | ✅ C2 改为对原始文件头断言（含 BOM），C1 只留行数，新增 C11 行数守恒 |
| F5 | medium | `atol` 被默认 `rtol=1e-5` 架空 | ✅ 统一 `rtol=0, atol=1e-9`；C10 增断言"B0007 未穿越"这个结论本身 |
| F6 | medium | 无再生事件时 `max([])` 崩溃 | ✅ `max(..., default=nan)`；`report()` 加空列表保护 |
| F7 | medium | 断言与 config 互相耦合 | ✅ C8 改用 `RATED_CAPACITY_AH`；C10 绑定 `EOL_REF_AH=1.4` 并注明与 config 解耦是有意的；新增 C12 校验两者一致 |
| F8 | medium | 依赖未锁上界 | ✅ 全部加上界（`pandas>=3,<4` 等） |
| F9–F16 | low | 死代码/冗余列/文档引用/BOM/字体/对齐 | ✅ 全清：删 `add_src_to_path`、删 `is_cell_first_cycle`、`utf-8-sig`、字体探测、取消定宽补齐、未建文件标注"计划中" |

**审查未能覆盖的**（它自己声明的边界）：建模/图表代码（尚不存在）、`preprocess.py`、报告生成、真实 GBK 控制台。第二轮审查安排在 W14 之后。

## 环境事实（已核实）

- Python 3.12.10 @ `AppData\Local\Programs\Python\Python312`
- numpy 2.5.1 / pandas 3.0.5 / matplotlib 3.11.1 / scipy 1.18.1 / python-docx 1.2.0
- **scikit-learn 1.9.1 已装**（用户级 site-packages `%APPDATA%\Python\Python312\site-packages`），普通权限 shell 下 `import sklearn` 通过
- 附带依赖：joblib 1.6.0 / threadpoolctl 3.7.0 / narwhals 2.27.1 / cloudpickle 3.1.2

## 环境坑（重要，别重复踩）

1. **DSH 沙箱禁止 chmod**。`pip install` 与 `python -m venv` 都要在临时目录做 chmod + rmtree，被拒即失败：
   - `pip` → `PermissionError: [WinError 5]` 写 `AppData\Local\Temp\dsh-*\...`
   - `venv` → `ensurepip` 退出码 1，venv 里没有 pip
   **解法**：装包用一次性 `sandbox_permissions=danger-full-access`。日常运行只读，不受影响。
2. **`curl.exe` 在沙箱里跑不通**，返回 HTTP 000 且极快（0.03–0.15s）——这是**测量假象**，不是网络不通。测网络用 Python 的 `urllib.request`。
3. **中文输出乱码**：`chcp 65001` + `PYTHONIOENCODING=utf-8` 后正常。
   ⚠️ 审查指出之前的机制描述（"GBK 码页导致"）**存疑**：脚本用到的字符全部可被 cp936 编码，乱码更可能来自"输出重定向到管道/文件时的编码"。下次遇到要给出可复现命令再断言。
4. **matplotlib 字体缓存**默认写到 `%USERPROFILE%\.matplotlib`（工作区外，沙箱只读）。已在 `config.py` import 期设置 `MPLCONFIGDIR` 到 `outputs/_tmp/mplconfig`。

## 已落盘文件

```
README.md                      主线、数据陷阱、运行说明、两条给未来的警告
requirements.txt               依赖（全部锁上界）
.gitignore
data/raw/battery_capacity.csv  636 行（复制自课程下发文件，未改动原文件）
src/config.py                  路径/常量/绘图（含 MPLCONFIGDIR 与字体探测）
src/load_data.py               读入 + 派生列（含泄漏警告与显式基准表）
src/sanity_check.py            12 条可复现断言（12/12 PASS）
scripts/run_all.py             一键入口（当前 1 步）
docs/DECISIONS.md              D1–D15
docs/STATE.md                  本文件
outputs/_tmp/                  临时与验收脚本（已 gitignore）
```

## 待决问题

1. **GitHub 侧改名未完成**（用户已批准改名，本地已改）。
   目标名：`battery-health-and-lifespan-prediction`。
   待用户在本机网页操作：`https://github.com/Tony-Stank-shift/battery-health-and-span-prediction/settings`
   → 第一栏 Repository name 改为上面那个 → 点 Rename（旧 URL 会自动重定向，不会断）。
   改完我这边 `origin` 已指向新名字，可直接 push。**W24 之前必须完成。**
2. **推送凭据未知**：Windows 的 git credential manager 是否已缓存 GitHub 凭据尚未验证。若 push 时弹出浏览器登录，需要用户配合一次。
3. `outputs/_tmp/pytmp*/` 下残留 pip 临时目录无法删除（chmod 受限），已被 gitignore，不影响。

## 下一步

**W3 清洗与预处理**：产出清洗决策表，四件事各给处理方式与「为什么不选另一种」——
① 温度零方差　② B0006 超额定　③ 12 次容量再生　④ B0007 右删失。
