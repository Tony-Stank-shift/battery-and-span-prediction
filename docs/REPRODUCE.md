# 复现指南

> 目标：拿到这个仓库，**一条命令**重跑全部结果，并证明重跑出来的和仓库里提交的**逐字节相同**。

## 一、环境

| 项 | 要求 | 记录出处 |
|---|---|---|
| Python | 3.12（3.10+ 应可，未验证） | `MANIFEST.sha256` 头行 |
| 依赖 | numpy / pandas / scipy / scikit-learn / matplotlib | 同上 |
| 系统 | 开发环境 Windows 11；路径用 `pathlib`，理论跨平台 | 同上 |

**为什么把版本写进清单头部**：PNG 的字节对 matplotlib 版本敏感。
所以「逐字节一致」这个承诺只在**同一环境**下成立；换版本必须重新生成清单。

安装依赖：

```powershell
pip install numpy pandas scipy scikit-learn matplotlib
```

## 二、跑起来

```powershell
# 1) 重跑全部 17 步（约 3–4 分钟）
python scripts/run_all.py

# 2) 重跑 + 与产物清单对账（推荐，这就是验收命令）
python scripts/run_all.py --check

# 3) 只对账，不重跑
python scripts/run_all.py --check-only
```

退出码约定：

| 退出码 | 含义 |
|---|---|
| 0 | 全部通过（且 `--check` 时产物一致） |
| 非 0 | 某一步失败，等于该步的退出码 |
| 2 | 步骤都过了，但**产物与清单不一致** |

## 三、清空重建（W20 的验收实验）

清单放在**仓库根**（不在 `outputs/` 里），所以删 `outputs/` 不会把它一起删掉：

```powershell
Remove-Item outputs -Recurse -Force
python scripts/run_all.py --check
```

预期：`全部 17 个步骤通过` + `[OK] 产物与提交版本逐字节一致`，退出码 0。
复现记录（含耗时）见 `outputs/tables/reproducibility.md`。

## 四、产物在哪

| 目录 | 内容 | 数量 |
|---|---|---|
| `outputs/figures/` | 论文用图 `fig01`–`fig17`（PNG） | 17 |
| `outputs/tables/` | 每节的方法与结论（Markdown，**全部由代码生成**） | 17 |
| `outputs/metrics/` | 机器可读的长表（CSV） | 6 |
| `outputs/_tmp/` | 审查脚本与缓存，**不进清单、不算交付物** | — |

`MANIFEST.sha256` 记录前三个目录每个文件的 SHA256 与字节数。

## 五、常见坑

1. **中文乱码**：Windows 控制台默认 GBK。设 `$env:PYTHONIOENCODING="utf-8"`，
   或先 `chcp 65001`。`run_all.py` 已对子进程注入该变量。
2. **matplotlib 缓存目录**：设 `$env:MPLCONFIGDIR="outputs/_tmp/mplconfig"`，
   否则会在用户目录下建缓存，个别机器没有写权限会直接崩。
3. **字体**：`config.setup_matplotlib()` 统一设置中文字体与负号显示，不要单独改。
4. **PowerShell 不支持 `&&`**：用 `;` 或分两行。
5. **随机性**：所有模型用固定种子（`config.RANDOM_SEED=42`，GP 用 `GP_SEED`）。
   若哪天清单开始随机失败，说明**新增了未固定的随机源**——那是个真 bug，不要重生成清单糊过去。

## 六、自检工具（不属于管线）

```powershell
python scripts/mutation_test.py     # 42 条注入错误，必须全部被断言抓住
python scripts/adversarial_test.py  # 43 条病态输入，不许静默给错答案
```

这两个不产出数据、不进清单，但它们是「结论可信」的守门员，改代码后应一起跑。

## 七、清单对账的边界

`manifest.check()` 能发现：文件缺失、多出、内容变化。

**发现不了**：
- 「代码错了但每次错得一样」——那要靠 `mutation_test` 和第三方审查；
- 换环境后的字节差异（版本不同）——那要先重新生成清单；
- `outputs/_tmp/` 里的任何东西。
