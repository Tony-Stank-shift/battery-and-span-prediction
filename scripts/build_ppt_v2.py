# -*- coding: utf-8 -*-
"""生成课程汇报 PPT v2：report/20_题3_黄锦添_v2.pptx

16:9，深蓝学术风。13 页正片 + 4 页备用页（问答用）。备注栏 = 逐字稿（含时间预算）。
数字全部来自 outputs/tables/*.md。
"""
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "outputs" / "figures"
OUT = ROOT / "report" / "20_题3_黄锦添_v2.pptx"

DEEP = RGBColor(0x1F, 0x38, 0x64)      # 深蓝主色
DEEP2 = RGBColor(0x2A, 0x4A, 0x7F)     # 深蓝次色
ACCENT = RGBColor(0xE0, 0x7B, 0x27)    # 强调橙
INK = RGBColor(0x26, 0x26, 0x26)
GRAY = RGBColor(0x59, 0x59, 0x59)
LGRAY = RGBColor(0x8A, 0x8A, 0x8A)
LIGHT = RGBColor(0xED, 0xF2, 0xF9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
RED = RGBColor(0xB3, 0x30, 0x30)

FONT = "微软雅黑"
SW, SH = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.55)

prs = Presentation()
prs.slide_width, prs.slide_height = SW, SH
BLANK = prs.slide_layouts[6]

TOTAL_MAIN = 13  # 正片页数（含封面），用于页码


def _set(run, size, bold=False, color=INK, font=FONT, italic=False):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font
    rPr = run._r.get_or_add_rPr()
    ea = rPr.find(qn("a:ea"))
    if ea is None:
        from pptx.oxml.ns import nsmap  # noqa
        ea = rPr.makeelement(qn("a:ea"), {})
        rPr.append(ea)
    ea.set("typeface", font)


def box(slide, l, t, w, h, fill=None, line=None, line_w=0.75, shadow=False, round_=False, radius=0.08):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if round_ else MSO_SHAPE.RECTANGLE
    sp = slide.shapes.add_shape(shape_type, l, t, w, h if (h := None) else h) if False else \
        slide.shapes.add_shape(shape_type, l, t, w, h) if False else None
    # 上面的占位写法不可读，直接写：
    sp = slide.shapes.add_shape(shape_type, l, t, w, h)
    if round_:
        try:
            sp.adjustments[0] = radius
        except Exception:
            pass
    if fill is None:
        sp.fill.background()
    else:
        sp.fill.solid()
        sp.fill.fore_color.rgb = fill
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line
        sp.line.width = Pt(line_w)
    sp.shadow.inherit = False
    return sp


def text(slide, l, t, w, h, runs, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         space_after=4, line_spacing=1.12, wrap=True):
    """runs: list of paragraphs; each paragraph = list of (txt, size, bold, color) 或 dict."""
    tb = slide.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame
    tf.word_wrap = wrap
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, para_runs in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        p.line_spacing = line_spacing
        if isinstance(para_runs, tuple):
            para_runs = [para_runs]
        for rr in para_runs:
            txt, size, bold, color = (rr + (INK,))[:4] if len(rr) == 3 else rr
            r = p.add_run()
            r.text = txt
            _set(r, size, bold, color)
    return tb


def add_fig(slide, fname, l, t, max_w, max_h, border=True):
    """按比例放入图片，限定在 (max_w, max_h) 内，水平居中于 (t, t+max_h) 区域左对齐起点 l。"""
    img = Image.open(FIG / fname)
    iw, ih = img.size
    scale = min(max_w / Emu(iw) if False else max_w, max_h)  # placeholder
    # 用英寸算比例
    ratio = iw / ih
    w_in, h_in = max_w / 914400, max_h / 914400
    if w_in / h_in > ratio:
        h_use, w_use = h_in, h_in * ratio
    else:
        w_use, h_use = w_in, w_in / ratio
    left = l + Inches((max_w / 914400 - w_use) / 2)
    top = t  # 顶对齐，避免内容区上方留白不一致
    pic = slide.shapes.add_picture(str(FIG / fname), left, top, Inches(w_use), Inches(h_use))
    if border:
        pic.line.color.rgb = RGBColor(0xD5, 0xDC, 0xE8)
        pic.line.width = Pt(0.75)
    return pic


def notes(slide, txt):
    slide.notes_slide.notes_text_frame.text = txt


def header(slide, kicker, title_runs, page, total=TOTAL_MAIN, backup=False):
    """页眉：深蓝横带 + kicker + 结论式标题 + 页码。"""
    band = box(slide, 0, 0, SW, Inches(1.02), fill=DEEP)
    box(slide, 0, Inches(1.02), SW, Pt(2.2), fill=ACCENT)
    # kicker：小橙块 + 章节
    box(slide, MARGIN, Inches(0.17), Inches(0.13), Inches(0.13), fill=ACCENT)
    text(slide, MARGIN + Inches(0.22), Inches(0.10), Inches(9), Inches(0.3),
         [[(kicker + ("　·　备用页" if backup else ""), 11, True, RGBColor(0xBF, 0xCD, 0xE6))]])
    text(slide, MARGIN, Inches(0.36), Inches(11.2), Inches(0.6), [title_runs], space_after=0)
    # 页码：备用页用字母标号，不与正片页码混排
    if backup:
        text(slide, SW - Inches(1.3), Inches(0.38), Inches(0.85), Inches(0.3),
             [[("备用 ", 10, False, RGBColor(0x9F, 0xB3, 0xD6)), (chr(64 + page - TOTAL_MAIN), 13, True, WHITE)]],
             align=PP_ALIGN.RIGHT)
    else:
        text(slide, SW - Inches(1.15), Inches(0.38), Inches(0.7), Inches(0.3),
             [[(f"{page:02d}", 13, True, WHITE), (f" / {total:02d}", 10, False, RGBColor(0x9F, 0xB3, 0xD6))]],
             align=PP_ALIGN.RIGHT)


SPLIT_DIR = ROOT / "outputs" / "_tmp" / "ppt_split"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)


def add_fig_split(slide, fname, l, t, w_emu, h_emu, cut, strip, gap_in=0.12):
    """宽幅双子图：剥掉顶部全局标题带（strip 比例），沿子图间隙竖直裁成左右两张（cut 比例），
    上下堆叠放入 (l, t, w_emu, h_emu) 区域，各自保持比例、水平居中。"""
    img = Image.open(FIG / fname)
    iw, ih = img.size
    top_px = int(ih * strip)
    mid_px = int(iw * cut)
    parts = [img.crop((0, top_px, mid_px, ih)), img.crop((mid_px, top_px, iw, ih))]
    stem = fname.rsplit(".", 1)[0]
    w_in = w_emu / 914400
    slot_h = (h_emu / 914400 - gap_in) / 2
    for i, part in enumerate(parts):
        p = SPLIT_DIR / f"{stem}_{'LR'[i]}.png"
        part.save(p)
        ratio = part.size[0] / part.size[1]
        if w_in / slot_h > ratio:
            h_use, w_use = slot_h, slot_h * ratio
        else:
            w_use, h_use = w_in, w_in / ratio
        left = l + Inches((w_in - w_use) / 2)
        top_emu = t + Inches(i * (slot_h + gap_in))
        pic = slide.shapes.add_picture(str(p), left, top_emu, Inches(w_use), Inches(h_use))
        pic.line.color.rgb = RGBColor(0xD5, 0xDC, 0xE8)
        pic.line.width = Pt(0.75)


def chip(slide, l, t, w, h, big, small, big_color=ACCENT, fill=LIGHT):
    """大数字卡片。"""
    box(slide, l, t, w, h, fill=fill, round_=True, radius=0.10)
    text(slide, l, t + Inches(0.10), w, Inches(0.55), [[(big, 25, True, big_color)]],
         align=PP_ALIGN.CENTER, space_after=0)
    text(slide, l + Inches(0.08), t + h - Inches(0.62), w - Inches(0.16), Inches(0.55),
         [[(small, 10.5, False, GRAY)]], align=PP_ALIGN.CENTER, space_after=0, line_spacing=1.0)


def bullets(slide, l, t, w, items, size=13.5, gap=6, lead_color=ACCENT):
    """带小方块的要点列表。items: (加粗前缀, 正文) 或 纯文本。"""
    paras = []
    for it in items:
        if isinstance(it, tuple):
            paras.append([("▪ ", size - 1.5, True, lead_color), (it[0], size, True, DEEP), (it[1], size - 1, False, INK)])
        else:
            paras.append([("▪ ", size - 1.5, True, lead_color), (it, size - 1, False, INK)])
    text(slide, l, t, w, Inches(5.0), paras, space_after=gap, line_spacing=1.18)


# ================= 1 封面 =================
s = prs.slides.add_slide(BLANK)
box(s, 0, 0, SW, SH, fill=DEEP)
box(s, 0, Inches(4.62), SW, Pt(2.4), fill=ACCENT)
text(s, MARGIN, Inches(0.5), Inches(8), Inches(0.35),
     [[("《Python 数据分析编程》课程项目　·　题目三", 13, False, RGBColor(0xBF, 0xCD, 0xE6))]])
text(s, MARGIN, Inches(2.35), Inches(12.2), Inches(1.9),
     [[("无人设备电池健康评估与剩余寿命预测", 40, True, WHITE)],
      [("容量退化建模与不确定性量化", 20, False, RGBColor(0xD8, 0xE2, 0xF2))]],
     space_after=14)
text(s, MARGIN, Inches(4.85), Inches(11), Inches(0.4),
     [[("第 20 组　黄锦添　25121605　·　上海大学 未来技术学院 人工智能", 13, False, RGBColor(0xBF, 0xCD, 0xE6))]])
# 底部三枚数据徽章
cw = Inches(3.5)
for i, (big, small) in enumerate([
        ("4 块 · 636 循环", "NASA PCoE 锂离子电池老化数据"),
        ("4 类模型 · 2 个任务", "个体内外推 / 跨个体早期预测"),
        ("1 条命令重建", "17 步管线 · 40 个产物逐字节一致")]):
    l = MARGIN + i * (cw + Inches(0.35))
    box(s, l, Inches(5.65), cw, Inches(1.05), fill=DEEP2, round_=True, radius=0.12)
    text(s, l, Inches(5.82), cw, Inches(0.45), [[(big, 17, True, ACCENT)]], align=PP_ALIGN.CENTER, space_after=0)
    text(s, l + Inches(0.15), Inches(6.22), cw - Inches(0.3), Inches(0.4),
         [[(small, 10.5, False, RGBColor(0xD8, 0xE2, 0xF2))]], align=PP_ALIGN.CENTER, space_after=0)
notes(s, "各位老师好。我汇报题目三：无人设备电池健康评估与剩余寿命预测，20 组黄锦添。"
         "一句话：我做的不是猜一个寿命数字，而是把这个数字能信到什么程度讲清楚。")

# ================= 2 问题与数据 =================
s = prs.slides.add_slide(BLANK)
header(s, "01 · 问题与数据", [("数据概览：容量退化曲线与实测寿命分布", 20, True, WHITE)], 2)
add_fig(s, "fig01_capacity_degradation.png", MARGIN, Inches(1.35), Inches(7.6), Inches(5.1))
bullets(s, Inches(8.55), Inches(1.5), Inches(4.2), [
    ("NASA PCoE 老化实验：", "4 块锂离子电池、逐循环放电容量、恒温 24 ℃"),
    ("EOL 判据：", "容量首次 ≤ 1.4 Ah（额定的 70%）"),
    ("实测寿命：", "B0005 = 125，B0006 = 109，B0018 = 97 循环"),
    ("B0007：", "观测 168 循环未跌破阈值，右删失，寿命 > 168"),
    ("曲线不光滑：", "存在容量再生式回升，最大单步 +0.1519 Ah"),
], size=13)
notes(s, "先看数据。四块电池、636 个循环，没有缺失、没有重复。横虚线是失效阈值，"
         "四条曲线穿过它的位置从 97 到 125 循环不等；最上面的 B0007 整个观测期都没碰到线——只能给下界，不能给数。")

# ================= 3 数据陷阱 =================
s = prs.slides.add_slide(BLANK)
header(s, "01 · 数据与清洗", [("数据清洗：五类数据陷阱的处理决策", 20, True, WHITE)], 3)
cards = [
    ("温度列零方差", "636 行全为 24.0 ℃——保留但禁止作特征，温度维度不可分析"),
    ("容量超额定", "B0006 前 7 循环 > 2.0 Ah——保留原值并加标记列，截断将改变衰减斜率"),
    ("容量再生", "12 个(电池,循环)对回升 > 0.05 Ah——不做平滑，建模阶段以趋势加残差结构显式处理"),
    ("右删失", "B0007 观测 168 循环未达阈值，寿命记 > 168（右删失），误记为 168 将系统性低估寿命"),
    ("统计离群点", "3σ 与 IQR 两种口径都标不出点——趋势数据没有离群点可言"),
]
cw, ch = Inches(3.95), Inches(1.72)
for i, (t1, t2) in enumerate(cards):
    row, col = divmod(i, 3)
    l = MARGIN + col * (cw + Inches(0.18))
    tt = Inches(1.5) + row * (ch + Inches(0.28))
    box(s, l, tt, cw, ch, fill=LIGHT, round_=True, radius=0.08)
    box(s, l, tt, Inches(0.09), ch, fill=DEEP)
    text(s, l + Inches(0.22), tt + Inches(0.14), cw - Inches(0.36), Inches(0.4),
         [[(f"陷阱 {i+1}｜{t1}", 14, True, DEEP)]], space_after=0)
    text(s, l + Inches(0.22), tt + Inches(0.56), cw - Inches(0.4), Inches(1.1),
         [[(t2, 11.5, False, INK)]], space_after=0, line_spacing=1.15)
# 第六格：防泄漏
l = MARGIN + 2 * (cw + Inches(0.18))
tt = Inches(1.5) + 1 * (ch + Inches(0.28))
box(s, l, tt, cw, ch, fill=RGBColor(0xFB, 0xEE, 0xE0), round_=True, radius=0.08)
box(s, l, tt, Inches(0.09), ch, fill=ACCENT)
text(s, l + Inches(0.22), tt + Inches(0.14), cw - Inches(0.36), Inches(0.4),
     [[("暗线｜特征泄漏防线", 14, True, ACCENT)]], space_after=0)
text(s, l + Inches(0.22), tt + Inches(0.56), cw - Inches(0.4), Inches(1.1),
     [[("cap/额定、cap/首循环为容量列的精确仿射变换（残差 2.2×10⁻¹⁶），用作特征将构成标签泄漏", 11.5, False, INK)]],
     space_after=0, line_spacing=1.15)
text(s, MARGIN, Inches(5.65), Inches(12.2), Inches(0.9),
     [[("清理结果：", 14, True, DEEP),
       ("636 行记录全部保留、未删除任何一条；全部决策由 11 条数据体检断言约束，数据更新将触发报警而非静默通过。", 13, False, INK)]])
notes(s, "这份数据没有缺失、没有重复，真正的工作是五个陷阱。最需要记住的是右删失：把没坏记成 168，"
         "寿命就被系统性低估。右下角是防泄漏：两个派生列是容量的仿射变换，用作特征等于标签泄漏。")

# ================= 4 退化规律 =================
s = prs.slides.add_slide(BLANK)
header(s, "01 · 探索性分析", [("个体差异量化与容量再生的共模成分", 20, True, WHITE)], 4)
add_fig_split(s, "fig02_individual_differences.png", MARGIN, Inches(1.35), Inches(6.1), Inches(5.15), cut=0.494, strip=0.10)
text(s, MARGIN, Inches(6.58), Inches(6.1), Inches(0.3),
     [[("注：（b）归一化形状仅用于展示，不得用作特征。", 9.5, False, GRAY)]],
     align=PP_ALIGN.CENTER, space_after=0)
add_fig(s, "fig04_trend_residual.png", Inches(6.85), Inches(1.35), Inches(5.95), Inches(3.4))
text(s, Inches(6.85), Inches(4.95), Inches(5.95), Inches(1.6), [
    [("五维量化：", 13, True, DEEP), ("水平 / 速率 / 形状 / 波动 / 寿命，逐维给数", 12, False, INK)],
    [("共模分解：", 13, True, DEEP), ("同批三块电池的残差 82.4% 来自共同成分（零基准 33.3%），"
                                        "三块在循环 90 同时出现各自最大回升", 12, False, INK)],
    [("结论：", 13, True, DEEP), ("容量再生不构成个体独立的随机现象，"
                                    "共模成分计入数据层不确定性，不得以平滑方式消除", 12, False, INK)],
], space_after=8)
notes(s, "个体差异按五个维度量化：初始容量差 9.7%，衰减速率最快的是最慢的 1.56 倍。"
         "更有意思的是右边：同批三块的波动 82.4% 来自共同成分，远高于独立基准的三分之一——"
         "回升不是各自的随机抖动，不能被平滑掉。")

# ================= 5 阈值敏感性 =================
s = prs.slides.add_slide(BLANK)
header(s, "01 · 探索性分析", [("EOL 阈值敏感性：读数对阈值敏感，排序对阈值稳健", 20, True, WHITE)], 5)
add_fig(s, "fig05_eol_threshold_sensitivity.png", MARGIN, Inches(1.35), Inches(7.7), Inches(5.15))
bullets(s, Inches(8.6), Inches(1.5), Inches(4.15), [
    ("±0.05 Ah 窗口：", "寿命读数极差 30–40 循环"),
    ("换 EOL 口径：", "同一块电池寿命最大相差 2.16 倍；计入删失下界至少 2.96 倍"),
    ("排序稳定：", "80 个阈值上恒为 B0018 < B0006 < B0005 < B0007"),
    ("B0007 删失裕度极小：", "最低容量距阈值仅 0.0005 Ah（额定的 0.025%）"),
], size=13)
box(s, Inches(8.6), Inches(5.15), Inches(4.15), Inches(1.25), fill=LIGHT, round_=True)
text(s, Inches(8.8), Inches(5.32), Inches(3.8), Inches(1.0),
     [[("方法论结论：", 12.5, True, ACCENT)],
      [("寿命序关系的报告可不带阈值；寿命点估计的报告必须注明阈值与窗口口径。", 12.5, False, INK)]],
     space_after=3, line_spacing=1.15)
notes(s, "这页是全项目的伏笔。阈值每挪 0.01 个 Ah，寿命差 3 到 4 个循环；"
         "但排序在 80 个阈值上一次都没翻。所以后面所有寿命数字，我都带着阈值口径报。")

# ================= 6 方法 =================
s = prs.slides.add_slide(BLANK)
header(s, "02 · 方法", [("任务定义与评测口径：个体内外推与跨个体早期预测", 20, True, WHITE)], 6)
# 两个任务卡片
cw = Inches(5.95)
for i, (title, lines) in enumerate([
    ("任务 A｜个体内轨迹外推（主线）",
     [("输入：", "一块电池自己的前 M = 80 个循环"),
      ("输出：", "后段容量 + EOL（外推 52–88 循环）"),
      ("模型：", "M1 线性 / M2 双指数 / M3 随机森林 / M4 趋势和 GPR"),
      ("循环序号：", "作为时间轴（回归自变量）使用")]),
    ("任务 B｜跨个体早期预测",
     [("输入：", "前 K = 50 循环提取的 10 个统计特征"),
      ("输出：", "一块没见过的电池的寿命"),
      ("验证：", "留一电池交叉验证（LOBOCV）"),
      ("循环序号：", "标签泄漏特征，禁止使用")]),
]):
    l = MARGIN + i * (cw + Inches(0.33))
    box(s, l, Inches(1.45), cw, Inches(2.55), fill=LIGHT, round_=True, radius=0.07)
    box(s, l, Inches(1.45), cw, Inches(0.52), fill=DEEP2, round_=True, radius=0.07)
    text(s, l + Inches(0.2), Inches(1.53), cw - Inches(0.4), Inches(0.4),
         [[(title, 14.5, True, WHITE)]], space_after=0)
    paras = [[("▪ ", 12, True, ACCENT), (k, 12.5, True, DEEP), (v, 12, False, INK)] for k, v in lines]
    text(s, l + Inches(0.25), Inches(2.15), cw - Inches(0.5), Inches(1.8), paras, space_after=5, line_spacing=1.12)
# 评测口径
box(s, MARGIN, Inches(4.35), Inches(12.23), Inches(2.0), fill=RGBColor(0xF6, 0xF8, 0xFC),
    line=RGBColor(0xC9, 0xD4, 0xE6), round_=True, radius=0.06)
text(s, MARGIN + Inches(0.25), Inches(4.55), Inches(11.7), Inches(1.7), [
    [("评测口径三条纪律：", 14, True, ACCENT)],
    [("① 时序划分，不随机打乱；训练段指标不参与排名，记忆样本不代表外推能力；", 12.5, False, INK)],
    [("② 不以平均 R² 排名（分母是各自方差，跨电池平均不具备统计意义），改用两个独立判据："
      "逐点符号检验 + 分块自助法置信区间（块长 1/3/5/10/20 × 2000 次重采样），两判据同时显著方可下结论；", 12.5, False, INK)],
    [("③ EOL 按三态分开报告：有值 / 真值删失 / 模型不越阈——混合统计将引入系统性错误。", 12.5, False, INK)],
], space_after=6)
notes(s, "两个必须分开的问题：任务 A 在一块电池内部沿时间外推；任务 B 拿几块电池预测一块新的——"
         "循环序号在任务 B 里是泄漏特征。评测也先定死：不用平均 R² 排名，"
         "逐点符号检验加分块自助法，两个判据都显著才下结论。")

# ================= 7 四模型对比 =================
s = prs.slides.add_slide(BLANK)
header(s, "03 · 结果", [("四模型对比：逐点符号检验与分块自助法显著性判定", 20, True, WHITE)], 7)
add_fig_split(s, "fig12_model_comparison.png", MARGIN, Inches(1.35), Inches(7.35), Inches(5.15), cut=0.510, strip=0.065)
bullets(s, Inches(8.25), Inches(1.45), Inches(4.5), [
    ("M4 vs M1：", "仅 B0005 两判据一致显著（86/88 点更近，p = 2.5×10⁻²³），其余可忽略或冲突——点预测精度无实质差异"),
    ("M2 双指数：", "三块上显著更差；主因不是参数不可辨识（条件数 10¹¹–10¹²），而是末端系统性低估寿命"),
    ("M3 随机森林：", "两块显著更差；EOL 预测 4/4 不越阈"),
    ("EOL 误差：", "M1 / M4 平均 12.0 循环，M2 平均 12.7 循环"),
], size=12.5)
box(s, Inches(8.25), Inches(5.35), Inches(4.5), Inches(1.05), fill=LIGHT, round_=True)
text(s, Inches(8.45), Inches(5.5), Inches(4.1), Inches(0.85),
     [[("结论：", 12.5, True, ACCENT)],
      [("M4 的增量价值不在点预测精度，而在输出 EOL 后验分布与预测区间。", 12, False, INK)]], space_after=3)
notes(s, "四模型同台，刻意不用均方误差排名——只有四块电池，换个切分点排序就会变。"
         "结论：线性和趋势加高斯过程的点预测无显著差异；双指数在三块上显著更差；随机森林两块显著更差。"
         "注意：不显著不等于一样好，是样本不够分辨。")

# ================= 8 反例：随机森林 =================
s = prs.slides.add_slide(BLANK)
header(s, "03 · 结果｜反例", [("随机森林的外推失效：结构性偏差的可证明机制", 20, True, WHITE)], 8)
add_fig(s, "fig07_rf_extrapolation.png", MARGIN, Inches(1.35), Inches(7.35), Inches(4.15))
# 大数字对比
chip(s, Inches(8.25), Inches(1.45), Inches(2.15), Inches(1.35), "0.997", "训练段 R²（逐点记忆训练样本）", big_color=DEEP)
chip(s, Inches(10.6), Inches(1.35), Inches(2.15), Inches(1.35), "−3.57", "测试段 R² 最差（B0005），4/4 为负", big_color=RED)
bullets(s, Inches(8.25), Inches(3.1), Inches(4.5), [
    ("机制可证明：", "超出训练区间的输入永远走同一条分裂路径，316 个测试点预测值完全相同（极差 = 0）"),
    ("树长到底时：", "外推值等于末次训练观测，即模型外推能力的信息上限"),
    ("加树不能修复：", "10 → 1000 棵，外推常数最多动 0.0037 Ah——失效源于偏差而非方差"),
    ("寿命预测方向错误：", "对三块已失效个体均预测不越阈"),
], size=12, gap=5)
box(s, MARGIN, Inches(5.7), Inches(12.23), Inches(0.85), fill=RGBColor(0xFB, 0xEE, 0xE0), round_=True)
text(s, MARGIN + Inches(0.25), Inches(5.86), Inches(11.7), Inches(0.6),
     [[("由该反例引出主模型设计：", 13, True, ACCENT),
       ("先拟合线性趋势，再以高斯过程建模残差，趋势项与残差项均显式携带方差。", 13, False, INK)]], space_after=0)
notes(s, "这页最想讲。随机森林训练 R² 接近 0.997，测试 R² 却是负的——比直接猜均值还差。"
         "原因不是过拟合：超出训练区间的输入永远落到同一片叶子，输出必然是常数，加树、加深都救不回来。"
         "所以我最后选先拟合趋势线，再让高斯过程学残差。")

# ================= 9 M4 GPR =================
s = prs.slides.add_slide(BLANK)
header(s, "03 · 结果", [("趋势和高斯过程回归：EOL 的后验分布", 20, True, WHITE)], 9)
add_fig(s, "fig09_gpr_band.png", MARGIN, Inches(1.35), Inches(6.15), Inches(4.3))
add_fig(s, "fig10_eol_posterior.png", Inches(6.9), Inches(1.35), Inches(5.9), Inches(3.1))
text(s, Inches(6.9), Inches(4.6), Inches(5.9), Inches(1.9), [
    [("结构：", 12.5, True, DEEP), ("cap(t) = 线性趋势和残差 GP 之和；趋势参数方差显式传播", 11.5, False, INK)],
    [("产出：", 12.5, True, DEEP), ("300 条后验路径各自求穿越，EOL 成为分布（B0005 中位 145，90% 区间 129–164）", 11.5, False, INK)],
    [("适用性说明：", 12.5, True, ACCENT), ("RBF 核相关长度仅 3.6–23 循环，外推 52–88——"
                                              "远场点预测退化为趋势线，GP 的贡献在于如实放大区间而非提升精度", 11.5, False, INK)],
], space_after=7, line_spacing=1.15)
box(s, MARGIN, Inches(5.85), Inches(6.15), Inches(0.75), fill=LIGHT, round_=True)
text(s, MARGIN + Inches(0.2), Inches(5.97), Inches(5.8), Inches(0.55),
     [[("该区间仅含数据层不确定性（残差噪声与趋势参数方差），不含定义层与模型形式层，其校准缺口在第 6 章定量归因。", 11.5, False, INK)]],
     space_after=0)
notes(s, "趋势加高斯过程把寿命从一个数变成一个分布：300 条后验路径各自求穿越点。"
         "诚实声明：平稳核不会自己外推，离开数据后后验均值归零、点预测退化为趋势线。"
         "它的价值不是更准，是能给出区间——而区间偏窄，第六章当面算清。")

# ================= 10 三层不确定性 =================
s = prs.slides.add_slide(BLANK)
header(s, "04 · 不确定性", [("三层不确定性预算：定义、模型形式与数据", 20, True, WHITE)], 10)
add_fig_split(s, "fig13_uncertainty_budget.png", MARGIN, Inches(1.35), Inches(7.1), Inches(5.1), cut=0.499, strip=0.065)
# 三层卡片
layers = [
    ("定义层", "11–20", "循环", "失效判据阈值选取（τ ± 0.05 Ah）引入的 EOL 极差，属口径约定而非模型误差"),
    ("模型形式层", "0–22", "循环", "模型形式选择引入的 EOL 半极差，其宽度全部来自双指数模型，应解释为形式差异的上界"),
    ("数据层", "6–20", "循环", "残差噪声 + 趋势参数（M4 后验 90% 半宽）"),
]
for i, (name, rng, unit, desc) in enumerate(layers):
    tt = Inches(1.5) + i * Inches(1.62)
    box(s, Inches(8.05), tt, Inches(4.7), Inches(1.45), fill=LIGHT, round_=True, radius=0.09)
    box(s, Inches(8.05), tt, Inches(0.09), Inches(1.45), fill=[DEEP, ACCENT, DEEP2][i])
    text(s, Inches(8.32), tt + Inches(0.12), Inches(4.3), Inches(0.5),
         [[(name + "　", 13.5, True, DEEP), (rng, 17, True, ACCENT), (" " + unit, 11, False, GRAY)]], space_after=0)
    text(s, Inches(8.32), tt + Inches(0.62), Inches(4.25), Inches(0.8),
         [[(desc, 10.5, False, INK)]], space_after=0, line_spacing=1.1)
text(s, MARGIN, Inches(6.55), Inches(7.1), Inches(0.5),
     [[("合并口径：RSS（独立假设，乐观）/ SUM（最坏情况，保守，28–53 循环）——运维取 SUM。", 11.5, False, GRAY)]])
notes(s, "把不确定性拆成三层、全部折算成循环，三者同量级。定义层来自阈值惯例，不归模型管。"
         "还要承认一点：模型形式层的宽度全部来自双指数——线性和高斯过程在远场的点预测是同一个数。")

# ================= 11 校准 =================
s = prs.slides.add_slide(BLANK)
header(s, "04 · 不确定性", [("区间校准：名义覆盖率与实际覆盖率的缺口归因", 20, True, WHITE)], 11)
add_fig_split(s, "fig16_calibration.png", MARGIN, Inches(1.35), Inches(6.6), Inches(5.1), cut=0.499, strip=0.065)
chip(s, Inches(7.55), Inches(1.5), Inches(2.5), Inches(1.4), "63%", "名义 90% 区间的实际覆盖率（316 个测试点）", big_color=RED)
chip(s, Inches(10.25), Inches(1.5), Inches(2.5), Inches(1.4), "10% / 99%", "逐块覆盖率最差 / 最好（极差 89 个百分点）", big_color=DEEP)
bullets(s, Inches(7.55), Inches(3.25), Inches(5.2), [
    ("驱动因素是偏差：", "点预测中位偏差逐块差 22.5 倍，区间宽度只差 1.15 倍——区间宽度与预测精度相互独立"),
    ("有效样本量受自相关折减：", "4 条序列高度自相关，有效样本约 109，块间差异在噪声范围内"),
    ("结论随切分点摆动：", "M = 60 / 80 / 100 下合并覆盖率为 34% / 63% / 67%，引用必须带 M"),
    ("补到 90% 需放宽 2.8 倍：", "测试集上的事后诊断量，不可直接部署"),
], size=12, gap=5)
notes(s, "九成区间实际只盖住六成三。更重要的是块间差异：最好的接近九成九，最差的只有一成。"
         "驱动因素是点预测偏差，块间差 22 倍，而不是区间宽度——只差 1.15 倍。区间窄不代表预测准。")

# ================= 12 落地 =================
s = prs.slides.add_slide(BLANK)
header(s, "05 · 落地", [("维护策略：更换时刻、采集时长与阈值口径", 20, True, WHITE)], 12)
add_fig_split(s, "fig17_strategy.png", MARGIN, Inches(1.35), Inches(7.35), Inches(5.15), cut=0.500, strip=0.065)
steps = [
    ("S1 更换时刻", "= 更严阈值（1.45 Ah）实测穿越点 − 模型与数据裕度；构造上保证不晚于失效"),
    ("S2 采集时长", "观测到预测稳定时刻（第 50–100 循环）再报寿命，可省 68–98 循环的数据量"),
    ("S3 阈值口径", "先把 τ 与窗口写进规范：报排序可以不带 τ，报绝对寿命必须带"),
]
for i, (t1, t2) in enumerate(steps):
    tt = Inches(1.5) + i * Inches(1.28)
    box(s, Inches(8.25), tt, Inches(4.5), Inches(1.12), fill=LIGHT, round_=True, radius=0.10)
    text(s, Inches(8.5), tt + Inches(0.10), Inches(4.1), Inches(0.4),
         [[(t1, 13.5, True, DEEP)]], space_after=0)
    text(s, Inches(8.5), tt + Inches(0.48), Inches(4.05), Inches(0.6),
         [[(t2, 10.5, False, INK)]], space_after=0, line_spacing=1.1)
box(s, Inches(8.25), Inches(5.5), Inches(4.5), Inches(0.95), fill=RGBColor(0xFB, 0xEE, 0xE0), round_=True)
text(s, Inches(8.45), Inches(5.62), Inches(4.1), Inches(0.75),
     [[("例外：", 12.5, True, RED),
       ("B0006 到达预测稳定时刻时已晚于建议更换点（在线剩余寿命为负）——对该个体不等待区间收敛，按早期观测直接安排更换。", 11, False, INK)]],
     space_after=0, line_spacing=1.1)
notes(s, "落到能执行的层面：更换时刻等于更严阈值的穿越点减裕度，代价是提前消耗两到四成寿命，属于风险对冲成本。"
         "但有一块电池，等预测稳定时已晚于建议更换点，对它不等待预测稳定，直接更换。")

# ================= 13 结论 =================
s = prs.slides.add_slide(BLANK)
header(s, "06 · 结论", [("结论", 20, True, WHITE)], 13)
concls = [
    ("①", "寿命是依赖阈值口径的统计量", "阈值 ±0.05 Ah → 读数差 30–40 循环；序关系在 80 个阈值上不变"),
    ("②", "高训练精度不构成外推能力的证据", "随机森林训练 R² 0.997、测试全负——树模型函数类不包含趋势成分"),
    ("③", "定义层与模型形式层不确定性不小于数据层", "定义层 / 模型形式层 / 数据层 = 11–20 / 0–22 / 6–20 循环，同量级"),
    ("④", "区间宽度与预测精度相互独立", "名义 90% 实际覆盖 63%，缺口由点预测偏差驱动；区间收敛不构成预测正确的充分条件"),
    ("⑤", "维护判据须将口径约定与不确定性分离报告", "更换时刻 = 更严阈值穿越点 − 裕度；代价 26.3%–41.6% 寿命"),
]
for i, (num, t1, t2) in enumerate(concls):
    tt = Inches(1.4) + i * Inches(0.94)
    box(s, MARGIN, tt, Inches(12.23), Inches(0.82), fill=(LIGHT if i % 2 == 0 else RGBColor(0xF6, 0xF8, 0xFC)), round_=True, radius=0.12)
    text(s, MARGIN + Inches(0.18), tt + Inches(0.16), Inches(0.5), Inches(0.5),
         [[(num, 17, True, ACCENT)]], space_after=0)
    text(s, MARGIN + Inches(0.75), tt + Inches(0.08), Inches(11.3), Inches(0.4),
         [[(t1, 13.5, True, DEEP)]], space_after=0)
    text(s, MARGIN + Inches(0.75), tt + Inches(0.44), Inches(11.3), Inches(0.35),
         [[(t2, 11, False, GRAY)]], space_after=0)
box(s, MARGIN, Inches(6.25), Inches(12.23), Inches(0.62), fill=DEEP, round_=True, radius=0.16)
text(s, MARGIN + Inches(0.25), Inches(6.38), Inches(11.8), Inches(0.4),
     [[("全部结果一条命令重建：", 12.5, True, ACCENT),
       ("python scripts/run_all.py --check（17 步 · 40 个产物逐字节一致 · 42 条变异测试 + 43 条病态输入全过）", 12, False, WHITE)]],
     space_after=0)
notes(s, "五条结论：两条关于模型，两条关于不确定性，一条关于怎么用。最后说句工程上的："
         "全部结果一条命令重建，40 个产物逐字节一致，42 条变异测试加 43 条病态输入全过。谢谢老师。")

# ================= 备用页 =================
# B1 阈值全表
s = prs.slides.add_slide(BLANK)
header(s, "备用 A", [("EOL 阈值敏感性全表与口径换算", 20, True, WHITE)], 14, backup=True)
rows = [["τ = 1.60 Ah（额定 80%）", "75", "63", "86", "45"],
        ["τ = 1.40 Ah（额定 70%，主线）", "125", "109", "删失(>168)", "97"],
        ["τ = 1.30 Ah（额定 65%）", "162", "140", "删失(>168)", "删失(>132)"]]
from pptx.util import Cm as _Cm  # noqa
tbl_shape = s.shapes.add_table(4, 5, MARGIN, Inches(1.5), Inches(7.6), Inches(1.9))
tbl = tbl_shape.table
for j, h in enumerate(["阈值口径", "B0005", "B0006", "B0007", "B0018"]):
    c = tbl.cell(0, j)
    c.text = h
    for p_ in c.text_frame.paragraphs:
        for r_ in p_.runs:
            _set(r_, 12, True, WHITE)
    c.fill.solid(); c.fill.fore_color.rgb = DEEP
for i, row in enumerate(rows):
    for j, v in enumerate(row):
        c = tbl.cell(i + 1, j)
        c.text = v
        for p_ in c.text_frame.paragraphs:
            for r_ in p_.runs:
                _set(r_, 12, False, INK)
        c.fill.solid(); c.fill.fore_color.rgb = LIGHT if i % 2 else WHITE
bullets(s, MARGIN, Inches(3.8), Inches(7.6), [
    ("口径换算：", "四种 EOL 定义下同一块电池寿命最大差 2.16 倍；计入删失下界至少 2.96 倍"),
    ("B0007 的特殊性：", "有效窗口只有 0.0495 Ah，不能与另三块同尺比较"),
    ("容量—寿命换算：", "|b| ≈ 0.003–0.007 Ah/循环 ⇒ 容量差 0.01 Ah ≈ 寿命差 1.4–3.3 循环"),
], size=12.5)
text(s, Inches(8.5), Inches(1.5), Inches(4.3), Inches(4.5), [
    [("备答要点：", 14, True, ACCENT)],
    [("口径选择的依据：70% 为工程惯例（额定 80%/70% 均常见）；本报告不依赖单一选择，τ 全区间扫描加四种口径对照，排序结论在 80 个 τ 上不变。", 12, False, INK)],
    [("B0007 的处理：右删失，全程单独立口径，寿命记 > 168，不与已失效个体混算误差。", 12, False, INK)],
], space_after=8)
notes(s, "备用页：阈值敏感性细节，问答时翻到这里。")

# B2 LOBOCV
s = prs.slides.add_slide(BLANK)
header(s, "备用 B", [("任务 B 的 LOBOCV 与零基准结果", 20, True, WHITE)], 15, backup=True)
add_fig(s, "fig11_lobocv.png", MARGIN, Inches(1.45), Inches(6.9), Inches(4.6))
bullets(s, Inches(7.75), Inches(1.55), Inches(5.0), [
    ("设置：", "前 K = 50 循环提取 10 个特征；留一电池交叉验证，有标签个体 n = 3，每折训练 2 个点"),
    ("结果：", "零基准（训练均值）MAE = 14.7 循环；最好的单特征 26.4——全部特征模型均不及零基准"),
    ("最坏一折：", "用首循环容量预测 B0006 得 3463 循环（真值 109）：两个训练点特征只差 0.0015，斜率被放大 667 倍"),
    ("防泄漏证据：", "扰动臂 + 删行臂两条正实验逐格判等，三条负控制（未来值 / 未来行数 / 未来标签）全部被检出"),
    ("正确读法：", "并非机器学习方法无效，而是 3 个有标签个体不足以支撑跨个体学习"),
], size=12, gap=6)
notes(s, "备用页：任务 B 与 LOBOCV 细节。")

# B3 校准逐块
s = prs.slides.add_slide(BLANK)
header(s, "备用 C", [("区间校准逐块明细与有效样本量", 20, True, WHITE)], 16, backup=True)
tbl_shape = s.shapes.add_table(5, 5, MARGIN, Inches(1.5), Inches(7.2), Inches(2.2))
tbl = tbl_shape.table
for j, h in enumerate(["90% 档逐块", "B0005", "B0006", "B0007", "B0018"]):
    c = tbl.cell(0, j)
    c.text = h
    for p_ in c.text_frame.paragraphs:
        for r_ in p_.runs:
            _set(r_, 12, True, WHITE)
    c.fill.solid(); c.fill.fore_color.rgb = DEEP
for i, row in enumerate([
        ["PICP（实际覆盖率）", "84.1%", "10.2%", "98.9%", "53.8%"],
        ["MPIW（Ah）", "0.1446", "0.1482", "0.1369", "0.1286"],
        ["中位偏差（Ah）", "0.0564", "0.1732", "0.0077", "0.0574"],
        ["有效样本量（AR1 近似）", "12.9", "2.9", "90.1", "3.2"]]):
    for j, v in enumerate(row):
        c = tbl.cell(i + 1, j)
        c.text = v
        for p_ in c.text_frame.paragraphs:
            for r_ in p_.runs:
                _set(r_, 12, False, INK)
        c.fill.solid(); c.fill.fore_color.rgb = LIGHT if i % 2 else WHITE
bullets(s, MARGIN, Inches(4.1), Inches(7.2), [
    ("316 个点 ≠ 316 个独立样本：", "4 条序列 lag-1 自相关最高 0.937，合并有效样本约 109"),
    ("剔除任一块：", "合并 PICP@90 在 48.7%–82.9% 之间摆动——块间差异不应解释为个体固有差异"),
    ("M 依赖性：", "M = 60 / 80 / 100 下合并 PICP@90 = 34% / 63% / 67%"),
], size=12.5)
text(s, Inches(8.15), Inches(1.5), Inches(4.6), Inches(4.5), [
    [("备答要点：", 14, True, ACCENT)],
    [("区间不主动放宽的理由：放宽区间可改善覆盖率读数，但会掩盖不确定性；本报告选择如实暴露缺口并归因于点预测偏差。", 12, False, INK)],
    [("2.8 倍放宽因子的性质：测试集上的事后诊断量而非校准方案，以测试真值调节区间，不可部署。", 12, False, INK)],
], space_after=8)
notes(s, "备用页：区间校准逐块数字。")

# B4 复现与测试
s = prs.slides.add_slide(BLANK)
header(s, "备用 D", [("复现、测试与仓库", 20, True, WHITE)], 17, backup=True)
box(s, MARGIN, Inches(1.5), Inches(6.0), Inches(2.3), fill=DEEP, round_=True, radius=0.07)
text(s, MARGIN + Inches(0.3), Inches(1.75), Inches(5.5), Inches(1.9), [
    [("$ python scripts/run_all.py --check", 13, True, WHITE)],
    [("17 步管线约 90 秒；与 MANIFEST.sha256 对账，40 个产物逐字节一致（干净副本中实测通过）", 11.5, False, RGBColor(0xD8, 0xE2, 0xF2))],
    [("$ python scripts/mutation_test.py", 13, True, WHITE)],
    [("42 条注入错误必须全部被抓——断言不是摆设", 11.5, False, RGBColor(0xD8, 0xE2, 0xF2))],
], space_after=6)
bullets(s, Inches(7.0), Inches(1.6), Inches(5.75), [
    ("仓库：", "github.com/Tony-Stank-shift/battery-health-and-lifespan-prediction（Git 管理，分工与提交记录一一对应）"),
    ("病态输入测试 43 条：", "NaN / 常数序列 / 单点 / 空表……不许静默给错答案"),
    ("数据体检断言 11 条：", "写死关键常量，课程方更新数据会立刻报警（特性，不是 bug）"),
    ("管线 17 步：", "数据体检 → 清洗 → 退化分析 → 再生分解 → 阈值敏感性 → 特征与泄漏证明 → 评测口径 → 四个模型 → 对比 → 三层不确定性 → 预警 → 放大 → 校准 → 策略"),
], size=12, gap=6)
notes(s, "备用页：复现命令与测试清单，老师要求现场跑的时候翻到这里。")

prs.save(OUT)
print("saved:", OUT)

# 自检
p2 = Presentation(OUT)
print("slides:", len(p2.slides))
total_notes = 0
for i, sl in enumerate(p2.slides, 1):
    nt = sl.notes_slide.notes_text_frame.text if sl.has_notes_slide else ""
    total_notes += len(nt)
    pics = sum(1 for sh in sl.shapes if sh.shape_type == 13)
    print(f"  slide {i}: shapes={len(sl.shapes)}, pics={pics}, notes={len(nt)}字")
print("notes total:", total_notes, "字 ≈", round(total_notes / 260 * 60), "秒纯朗读")
