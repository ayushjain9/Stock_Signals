"""
WealthOS McKinsey-style deck builder.
Run: python build_deck.py
Output: WealthOS_Strategy_Deck.pptx
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import copy

# ── Palette ────────────────────────────────────────────────────────────────────
NAVY    = RGBColor(0x0D, 0x1B, 0x3E)   # deep McKinsey navy
BLUE    = RGBColor(0x00, 0x5B, 0xB5)   # accent blue
GOLD    = RGBColor(0xE8, 0xA0, 0x00)   # highlight gold
LGRAY   = RGBColor(0xF4, 0xF5, 0xF7)   # light bg
DGRAY   = RGBColor(0x5A, 0x5A, 0x5A)   # body text
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
GREEN   = RGBColor(0x1A, 0x7A, 0x4A)
RED     = RGBColor(0xC0, 0x39, 0x2B)

W, H = Inches(13.33), Inches(7.5)   # widescreen 16:9

prs = Presentation()
prs.slide_width  = W
prs.slide_height = H

BLANK = prs.slide_layouts[6]   # truly blank layout


# ── Helpers ────────────────────────────────────────────────────────────────────

def add_rect(slide, l, t, w, h, fill=None, line=None, line_w=Pt(0)):
    shape = slide.shapes.add_shape(1, l, t, w, h)   # 1 = MSO_SHAPE_TYPE.RECTANGLE
    shape.line.width = line_w
    if fill:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
    else:
        shape.fill.background()
    if line:
        shape.line.color.rgb = line
    else:
        shape.line.fill.background()
    return shape


def add_text(slide, text, l, t, w, h,
             size=18, bold=False, color=NAVY, align=PP_ALIGN.LEFT,
             wrap=True, italic=False):
    txb = slide.shapes.add_textbox(l, t, w, h)
    tf  = txb.text_frame
    tf.word_wrap = wrap
    p   = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size  = Pt(size)
    run.font.bold  = bold
    run.font.color.rgb = color
    run.font.italic = italic
    return txb


def header_bar(slide, title, subtitle=None):
    """Top navy bar with title."""
    add_rect(slide, 0, 0, W, Inches(1.1), fill=NAVY)
    add_text(slide, title,
             Inches(0.35), Inches(0.08), Inches(11), Inches(0.65),
             size=28, bold=True, color=WHITE)
    if subtitle:
        add_text(slide, subtitle,
                 Inches(0.35), Inches(0.72), Inches(11), Inches(0.35),
                 size=14, color=RGBColor(0xBB, 0xCC, 0xE8))
    # thin gold rule
    add_rect(slide, 0, Inches(1.1), W, Inches(0.05), fill=GOLD)


def footnote(slide, text):
    add_text(slide, text,
             Inches(0.35), Inches(7.1), Inches(12.6), Inches(0.3),
             size=9, color=DGRAY, italic=True)


def kpi_box(slide, label, value, sub, l, t, w=Inches(2.6), h=Inches(1.5),
            val_color=BLUE):
    add_rect(slide, l, t, w, h, fill=LGRAY, line=RGBColor(0xD0,0xD5,0xDD), line_w=Pt(0.5))
    add_text(slide, value, l+Inches(0.12), t+Inches(0.08), w-Inches(0.24), Inches(0.65),
             size=30, bold=True, color=val_color, align=PP_ALIGN.CENTER)
    add_text(slide, label, l+Inches(0.08), t+Inches(0.72), w-Inches(0.16), Inches(0.38),
             size=11, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
    add_text(slide, sub,   l+Inches(0.08), t+Inches(1.08), w-Inches(0.16), Inches(0.35),
             size=9, color=DGRAY, align=PP_ALIGN.CENTER)


def bullet_block(slide, title, bullets, l, t, w, h, title_size=13, body_size=11):
    add_text(slide, title, l, t, w, Inches(0.3),
             size=title_size, bold=True, color=NAVY)
    y = t + Inches(0.32)
    for b in bullets:
        add_text(slide, f"▸  {b}", l, y, w, Inches(0.28),
                 size=body_size, color=DGRAY)
        y += Inches(0.28)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — TITLE / COVER
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
add_rect(sl, 0, 0, W, H, fill=NAVY)
add_rect(sl, 0, 0, Inches(0.5), H, fill=GOLD)

add_text(sl, "WealthOS",
         Inches(1), Inches(1.4), Inches(11), Inches(1.2),
         size=60, bold=True, color=WHITE)
add_text(sl, "Systematic Alpha Framework for Indian Equities",
         Inches(1), Inches(2.65), Inches(10), Inches(0.55),
         size=24, color=RGBColor(0xBB,0xCC,0xE8))
add_rect(sl, Inches(1), Inches(3.35), Inches(9), Inches(0.04), fill=GOLD)
add_text(sl, "Four production strategies · Nifty 50 & Midcap 50 · ₹15L+ AUM",
         Inches(1), Inches(3.5), Inches(10), Inches(0.4),
         size=15, color=RGBColor(0x99,0xAA,0xCC))

add_text(sl, "Confidential  ·  May 2026",
         Inches(1), Inches(6.8), Inches(5), Inches(0.35),
         size=11, color=RGBColor(0x77,0x88,0xAA))


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — EXECUTIVE SUMMARY (the one-page answer)
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
header_bar(sl, "Executive Summary",
           "WealthOS delivers systematic, rules-based alpha across two distinct return drivers")

# Big callout numbers row
kpi_box(sl, "Jensen's Alpha (OOS)", "8.68%", "5-yr walk-forward · Nifty 50 momentum",
        Inches(0.4),  Inches(1.35), val_color=GREEN)
kpi_box(sl, "NiftyShop XIRR", "16–18%", "2-yr backtest · with 200DMA filter",
        Inches(3.2),  Inches(1.35), val_color=BLUE)
kpi_box(sl, "Max Drawdown vs Benchmark", "−15.51% vs −15.99%", "Momentum OOS; smaller drawdown",
        Inches(6.0),  Inches(1.35), val_color=DGRAY)
kpi_box(sl, "Hit Ratio (OOS)", "60.7%", "Months strategy beats NIFTY 50",
        Inches(8.8),  Inches(1.35), val_color=BLUE)
kpi_box(sl, "Total Deployed Sleeve", "₹15L", "₹9L momentum + ₹4L shop + ₹2L×2 midcap",
        Inches(11.6), Inches(1.35), val_color=GOLD)

# Three column summary
cols = [
    ("Strategy Logic",
     ["Momentum: trend-following with 8-signal score",
      "Mean-reversion: weekly 20DMA dip-buying",
      "Value: fundamental quality + cheapness screen",
      "All signals validated out-of-sample (OOS)"]),
    ("Risk Controls",
     ["Sector cap: max 3 stocks per sector",
      "Exit buffer: exit only if score −15+ pts",
      "Hard stop: −15% from entry price",
      "200DMA gate on mean-reversion (quality filter)"]),
    ("Edge Source",
     ["Vol surge signal: +0.95pp alpha OOS",
      "6M Jegadeesh-Titman RS: +2.20pp OOS",
      "Alpha sensitivity: only 0.14pp lost per 1pp Nifty fall",
      "Rejected real ADX & graduated 52W high (noise)"]),
]
for i, (title, bullets) in enumerate(cols):
    x = Inches(0.4 + i * 4.3)
    add_rect(sl, x, Inches(3.15), Inches(4.0), Inches(3.8), fill=LGRAY,
             line=RGBColor(0xD0,0xD5,0xDD), line_w=Pt(0.5))
    bullet_block(sl, title, bullets, x+Inches(0.18), Inches(3.25), Inches(3.7), Inches(3.5))

footnote(sl, "OOS = Out-of-sample walk-forward validation  ·  All backtests use Yahoo Finance / NSE data")


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — PORTFOLIO ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
header_bar(sl, "Portfolio Architecture", "Two return drivers × two universes = four independent sleeves")

# Matrix header
for j, label in enumerate(["Strategy", "Universe", "Frequency", "Sleeve", "Positions", "Target XIRR"]):
    x = Inches(0.35 + j*2.15)
    add_rect(sl, x, Inches(1.3), Inches(2.1), Inches(0.38), fill=NAVY)
    add_text(sl, label, x+Inches(0.05), Inches(1.32), Inches(2.0), Inches(0.36),
             size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

# Data rows
rows = [
    ("Nifty 50\nMomentum",    "Nifty 50",    "Monthly",          "₹9 L", "15 stocks",  "Nifty + 8–9%"),
    ("NiftyShop\nMean-Rev.",  "Nifty 50",    "Weekly (Fri 3pm)", "₹4 L", "≤5 stocks",  "16–18% XIRR"),
    ("Midcap 50\nMomentum",   "Midcap 50",   "Monthly",          "₹2 L", "10 stocks",  "Midcap + ~8%"),
    ("MidcapShop\nMean-Rev.", "Midcap 50",   "Weekly (Fri 3pm)", "₹2 L", "≤5 stocks",  "~18–22% XIRR"),
]
row_colors = [LGRAY, WHITE, LGRAY, WHITE]
for i, (row, bg) in enumerate(zip(rows, row_colors)):
    y = Inches(1.68 + i*0.78)
    add_rect(sl, Inches(0.35), y, Inches(12.6), Inches(0.76), fill=bg)
    for j, cell in enumerate(row):
        x = Inches(0.35 + j*2.15)
        add_text(sl, cell, x+Inches(0.08), y+Inches(0.08), Inches(2.0), Inches(0.62),
                 size=11, color=NAVY if j==0 else DGRAY,
                 bold=(j==0), align=PP_ALIGN.CENTER)

# Independence note
add_rect(sl, Inches(0.35), Inches(5.0), Inches(12.6), Inches(0.06), fill=GOLD)
note = ("The four sleeves are structurally independent: momentum captures trend, "
        "mean-reversion captures short-term dislocation. "
        "Drawdowns are unlikely to coincide — momentum drawdowns occur in trending markets, "
        "mean-reversion drawdowns in choppy/falling markets.")
add_text(sl, note, Inches(0.4), Inches(5.18), Inches(12.5), Inches(0.6),
         size=11, color=DGRAY, italic=True)

# Value screener note
add_rect(sl, Inches(0.35), Inches(5.95), Inches(12.6), Inches(1.2),
         fill=RGBColor(0xE8,0xF4,0xE8), line=GREEN, line_w=Pt(0.5))
add_text(sl, "Bonus Tool — Value Screener (not a trading sleeve)",
         Inches(0.5), Inches(6.0), Inches(12), Inches(0.32),
         size=12, bold=True, color=GREEN)
add_text(sl, ("Scans 400+ NSE mid/small caps weekly on 8 fundamental metrics "
              "(ROE, Revenue Growth, Margins, Debt/Equity, PE vs sector, Promoter holding, OCF, EPS consistency). "
              "Outputs a shortlist for 3–5 year core positions. Not backtested — analyst judgment required."),
         Inches(0.5), Inches(6.35), Inches(12.5), Inches(0.6),
         size=10, color=DGRAY)

footnote(sl, "Midcap strategies are paper-trade phase — validate 2 months before full deployment")


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — SIGNAL STACK DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
header_bar(sl, "Signal Stack — Momentum v6",
           "Eight orthogonal signals, score capped at 100 pts · validated out-of-sample")

signals = [
    ("Trend Direction (DMA Alignment)", "20 pts", "Price > 25D > 50D > 200D = 'Brutal Strength'. Full alignment required for max pts.", GREEN),
    ("RS vs Benchmark — 90 Day",        "25 pts", "90-day excess return vs ^NSEI / ^NSMIDCP. Highest weight: relative strength is the core edge.", BLUE),
    ("RS vs Benchmark — 6 Month",       " 8 pts", "Jegadeesh-Titman 6M lookback, skip last 1M. OOS validated: +2.20pp alpha. Tested and kept.", BLUE),
    ("Volatility-Adj. Momentum (VAM)",  "20 pts", "ROC(66d) / ATR%. Filters stocks with chaotic price spikes. Rewards smooth momentum.", NAVY),
    ("52-Week High Proximity",          " 6 pts", "Within −5% of yearly high. Institutional accumulation signal.", DGRAY),
    ("Multi-Timeframe EMA Alignment",   "12 pts", "E5 > E20 > E50 > E200. Short-term trend confirmation nested inside long-term.", NAVY),
    ("ADX Direction Proxy",             " 7 pts", "Price acceleration over 10-day windows. Note: real ADX tested and REJECTED — underperformed.", RED),
    ("Volume Surge",                    " 6 pts", "5D avg > 1.5–2× 20D avg. OOS validated: +0.95pp alpha. Institutional conviction proxy.", GREEN),
]

y = Inches(1.25)
for name, pts, desc, color in signals:
    add_rect(sl, Inches(0.35), y, Inches(12.6), Inches(0.58), fill=WHITE,
             line=RGBColor(0xD8,0xDC,0xE4), line_w=Pt(0.5))
    # pts badge
    add_rect(sl, Inches(0.35), y, Inches(0.75), Inches(0.58), fill=color)
    add_text(sl, pts, Inches(0.35), y+Inches(0.08), Inches(0.75), Inches(0.42),
             size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(sl, name, Inches(1.18), y+Inches(0.06), Inches(3.4), Inches(0.25),
             size=11, bold=True, color=NAVY)
    add_text(sl, desc, Inches(1.18), y+Inches(0.3), Inches(11.6), Inches(0.25),
             size=9.5, color=DGRAY)
    y += Inches(0.61)

footnote(sl, "Signals tested and REJECTED: real ADX (underperformed every config) · graduated 52W high (+0.21pp OOS — noise level, not kept)")


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 5 — NIFTYSHOP MEAN-REVERSION MECHANICS
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
header_bar(sl, "NiftyShop — Mean-Reversion Mechanics",
           "Weekly signal · Friday 3:15pm · 5-minute execution workflow")

# Decision tree — simplified as flow boxes
flow = [
    ("SCAN", "All 50 stocks\nbelow 20DMA\n(weekly dip)", NAVY),
    ("QUALITY\nGATE", "Price > 200DMA?\nStructural uptrend\nintact?", BLUE),
    ("PRIORITY\nCHECK", "SELL first (≥+8%)\nthen AVERAGE (−3%)\nthen FRESH BUY", GOLD),
    ("EXECUTE", "1 trade/week\nin Zerodha\n3:20–3:30pm", GREEN),
]
for i, (label, desc, color) in enumerate(flow):
    x = Inches(0.5 + i * 3.1)
    add_rect(sl, x, Inches(1.35), Inches(2.5), Inches(1.6), fill=color)
    add_text(sl, label, x, Inches(1.4), Inches(2.5), Inches(0.5),
             size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(sl, desc, x+Inches(0.1), Inches(1.92), Inches(2.3), Inches(0.95),
             size=10, color=WHITE, align=PP_ALIGN.CENTER)
    if i < 3:
        add_text(sl, "→", Inches(3.0 + i*3.1), Inches(1.9), Inches(0.2), Inches(0.5),
                 size=22, bold=True, color=NAVY, align=PP_ALIGN.CENTER)

# Parameters side by side
left_params = [
    ("Entry Sizes (Nifty 50)", [
        "Fresh buy:  ₹10,000 per slot",
        "Averaging:  ₹15,000 (at −3% from cost)",
        "Max invested per stock:  ₹40,000",
        "Max simultaneous positions:  5",
    ]),
    ("Entry Sizes (Midcap 50)", [
        "Fresh buy:  ₹5,000–6,250 per slot",
        "Averaging:  ₹7,500–9,375",
        "Higher risk — midcaps fall harder",
        "Same 200DMA quality gate applies",
    ]),
]
right_params = [
    ("Exit Logic", [
        "Target: price ≥ avg cost × 1.08 (+8%)",
        "No time-based exit — let winners run",
        "Forced exit: 200DMA breach (thesis broken)",
        "Priority: SELL before any new buy",
    ]),
    ("Why 200DMA Filter?", [
        "Without filter: XIRR 13–21%, MDD −4.7 to −7.5%",
        "With filter: XIRR ~16–18% (estimated)",
        "Removes structural declines (Adani-style falls)",
        "Midcap MDD without filter: −11.93%",
    ]),
]
for i, (title, bullets) in enumerate(left_params):
    x = Inches(0.4)
    y = Inches(3.2 + i * 1.85)
    add_rect(sl, x, y, Inches(6.0), Inches(1.75), fill=LGRAY, line=RGBColor(0xD0,0xD5,0xDD), line_w=Pt(0.5))
    bullet_block(sl, title, bullets, x+Inches(0.15), y+Inches(0.1), Inches(5.7), Inches(1.6))

for i, (title, bullets) in enumerate(right_params):
    x = Inches(6.8)
    y = Inches(3.2 + i * 1.85)
    add_rect(sl, x, y, Inches(6.15), Inches(1.75), fill=LGRAY, line=RGBColor(0xD0,0xD5,0xDD), line_w=Pt(0.5))
    bullet_block(sl, title, bullets, x+Inches(0.15), y+Inches(0.1), Inches(5.9), Inches(1.6))

footnote(sl, "Trade log format: date, symbol, action (BUY_FRESH|BUY_AVG|SELL), price, quantity, amount  ·  Manually logged after Zerodha execution")


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 6 — VALIDATED PERFORMANCE KPIs
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
header_bar(sl, "Validated Performance",
           "All numbers from 5-year walk-forward backtests (Nifty 50 momentum) or 2-yr backtests (NiftyShop)")

# Section: Momentum
add_rect(sl, Inches(0.35), Inches(1.3), Inches(12.6), Inches(0.32), fill=BLUE)
add_text(sl, "MOMENTUM STRATEGY — NIFTY 50 (v6, 5-year walk-forward)",
         Inches(0.45), Inches(1.32), Inches(12), Inches(0.28),
         size=12, bold=True, color=WHITE)

mom_kpis = [
    ("Jensen's Alpha\n(OOS)", "8.68%", "vs 8.41% baseline", GREEN),
    ("Sharpe Ratio\n(OOS)", "0.79", "vs 0.87 baseline\n(higher alpha, harder mkt)", BLUE),
    ("Max Drawdown\n(OOS)", "−15.51%", "Benchmark: −15.99%\nSmaller than index", DGRAY),
    ("Hit Ratio\n(OOS)", "60.7%", "Months beating Nifty", BLUE),
    ("Alpha Sensitivity", "−0.14pp", "Per 1pp Nifty decline\n(highly robust)", GREEN),
]
for i, (label, val, sub, color) in enumerate(mom_kpis):
    kpi_box(sl, label, val, sub, Inches(0.35 + i*2.6), Inches(1.72), w=Inches(2.45), h=Inches(1.5), val_color=color)

# Signal improvement table
add_rect(sl, Inches(0.35), Inches(3.45), Inches(12.6), Inches(0.28), fill=NAVY)
add_text(sl, "SIGNAL IMPROVEMENT LOG (what moved the needle)",
         Inches(0.45), Inches(3.47), Inches(12), Inches(0.24),
         size=11, bold=True, color=WHITE)
sig_rows = [
    ("Volume Surge (5D avg > 1.5–2× 20D avg)", "+0.95pp OOS alpha", "ADDED to v6", GREEN),
    ("6-Month RS — Jegadeesh-Titman",          "+2.20pp OOS alpha", "ADDED to v6", GREEN),
    ("Real ADX (14-period)",                    "Underperformed every config tested", "REJECTED", RED),
    ("Graduated 52W High Proximity",            "+0.21pp OOS — within noise band", "REJECTED", RED),
]
for i, (sig, result, status, color) in enumerate(sig_rows):
    y = Inches(3.8 + i*0.38)
    bg = LGRAY if i%2==0 else WHITE
    add_rect(sl, Inches(0.35), y, Inches(12.6), Inches(0.36), fill=bg)
    add_text(sl, sig,    Inches(0.5),  y+Inches(0.06), Inches(6.5), Inches(0.26), size=10, color=DGRAY)
    add_text(sl, result, Inches(7.2),  y+Inches(0.06), Inches(3.5), Inches(0.26), size=10, color=DGRAY)
    add_rect(sl, Inches(10.9), y+Inches(0.05), Inches(1.8), Inches(0.26), fill=color)
    add_text(sl, status, Inches(10.9), y+Inches(0.05), Inches(1.8), Inches(0.26),
             size=9, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

# NiftyShop section
add_rect(sl, Inches(0.35), Inches(5.35), Inches(12.6), Inches(0.3), fill=BLUE)
add_text(sl, "NIFTYSHOP — NIFTY 50 (2-year backtest)",
         Inches(0.45), Inches(5.37), Inches(12), Inches(0.26),
         size=12, bold=True, color=WHITE)

shop_kpis = [
    ("XIRR (no filter)",     "13–21%", "Range across 2yr window", DGRAY),
    ("XIRR (200DMA filter)", "~16–18%","Quality stocks only",      GREEN),
    ("Max Drawdown",         "−4.7 to\n−7.5%", "Nifty 50 universe", GREEN),
    ("Midcap MDD (no filter)","−11.93%","Midcaps fall harder",     RED),
    ("Trade frequency",      "1/week", "Friday 3:15pm · 5 min exec", BLUE),
]
for i, (label, val, sub, color) in enumerate(shop_kpis):
    kpi_box(sl, label, val, sub, Inches(0.35 + i*2.6), Inches(5.75), w=Inches(2.45), h=Inches(1.35), val_color=color)

footnote(sl, "Midcap momentum not yet walk-forward validated — paper trade 2 months before full deployment")


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 7 — RISK FRAMEWORK
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
header_bar(sl, "Risk Framework", "Four layered defences — position, sector, score, and price-based")

risks = [
    ("1", "Position Sizing\n(Equal Weight)", BLUE,
     ["All momentum positions: equal weight (₹60K each on ₹9L)",
      "No single stock can dominate P&L",
      "Mean-reversion: capped total per stock (₹40K max, Nifty 50)",
      "Prevents averaging into a fundamentally broken story"]),
    ("2", "Sector Cap\n(Max 3 per sector)", NAVY,
     ["At most 3 momentum stocks from same GICS sector",
      "Prevents sector concentration (e.g. 8/15 in Financials)",
      "Applied at selection step — not post-hoc",
      "Sector map maintained in config.py (single source of truth)"]),
    ("3", "Exit Buffer\n(Score −15 pts)", GOLD,
     ["Momentum holding kept if score ≥ entry_score − 15",
      "Cuts unnecessary turnover: 474% → 330% annual",
      "Avoids whipsawing around the top-15 boundary",
      "Hard override: always exit if hard stop triggered"]),
    ("4", "Hard Stop\n(Price −15%)", RED,
     ["Force-exit any momentum holding down 15%+ from entry",
      "Protects against fundamental deterioration mid-period",
      "Mean-reversion: 200DMA breach = thesis broken → exit",
      "Midcap extra risk: MDD without filter was −11.93%"]),
]

for i, (num, title, color, bullets) in enumerate(risks):
    x = Inches(0.35 + (i%2)*6.5)
    y = Inches(1.3 + (i//2)*2.9)
    add_rect(sl, x, y, Inches(6.2), Inches(2.7), fill=WHITE,
             line=color, line_w=Pt(1.5))
    add_rect(sl, x, y, Inches(0.55), Inches(2.7), fill=color)
    add_text(sl, num, x, y+Inches(0.9), Inches(0.55), Inches(0.7),
             size=28, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(sl, title, x+Inches(0.65), y+Inches(0.1), Inches(5.4), Inches(0.5),
             size=13, bold=True, color=color)
    for j, b in enumerate(bullets):
        add_text(sl, f"▸  {b}", x+Inches(0.65), y+Inches(0.62+j*0.48), Inches(5.4), Inches(0.45),
                 size=10, color=DGRAY)

footnote(sl, "Exit buffer validated in backtest_v3_final.py  ·  Dynamic sector cap tested in v4 — static cap chosen for simplicity and equivalent performance")


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 8 — FORWARD RETURN EXPECTATIONS
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
header_bar(sl, "Forward Return Expectations",
           "Honest range estimates — base, bull, bear — grounded in validated alpha")

# Disclaimer box
add_rect(sl, Inches(0.35), Inches(1.25), Inches(12.6), Inches(0.4),
         fill=RGBColor(0xFF,0xF3,0xCD), line=GOLD, line_w=Pt(1))
add_text(sl, ("⚠  Forward estimates are probabilistic ranges, not guarantees. "
              "They are derived from OOS backtest alpha + Nifty base-rate return assumptions. "
              "Actual returns depend on Nifty trajectory, volatility, and signal drift."),
         Inches(0.5), Inches(1.28), Inches(12.3), Inches(0.35),
         size=9.5, color=RGBColor(0x7D,0x5A,0x00), italic=True)

# Table header
col_labels = ["Strategy", "Bear\n(Nifty −5%)", "Base\n(Nifty +12%)", "Bull\n(Nifty +20%)", "Key Risk", "Time Horizon"]
col_w = [2.4, 1.7, 1.7, 1.7, 2.8, 1.7]
x_pos = [sum(col_w[:j]) for j in range(len(col_w))]
for j, (label, cw) in enumerate(zip(col_labels, col_w)):
    x = Inches(0.35 + x_pos[j])
    add_rect(sl, x, Inches(1.75), Inches(cw), Inches(0.45), fill=NAVY)
    add_text(sl, label, x, Inches(1.77), Inches(cw), Inches(0.43),
             size=10, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

fwd_rows = [
    ("Nifty 50\nMomentum",   "−3 to 0%\n(alpha holds,\nbenchmark drags)", "+18 to 22%\n(alpha ~8–9pp\nabove Nifty)",  "+26 to 30%",
     "Alpha shrinks in\nstrong mean-reversion\nmarkets", "Monthly · 5 yr+"),
    ("NiftyShop\n(N50)",      "+8 to 12%\n(dips = more\nopportunity)", "+16 to 18%\n(validated\nrange)",           "+18 to 22%\n(fewer dips,\nless opportunity)",
     "Structural bear:\n200DMA filter fires,\nfewer entries", "Weekly · 2 yr+"),
    ("Midcap\nMomentum",     "−8 to −3%\n(midcaps fall\nharder)",  "+18 to 25%\n(higher beta\nto Nifty)",           "+28 to 35%",
     "NOT yet validated OOS\n—paper trade first", "Monthly · paper 2m"),
    ("MidcapShop",           "+5 to 10%\n(higher dip\nfrequency)", "+18 to 22%\n(estimated,\nnot OOS validated)",   "+20 to 25%",
     "MDD −11.93% without\n200DMA filter", "Weekly · paper 2m"),
    ("Combined\nPortfolio",  "+3 to 6%\n(diversification\nbenefit)", "+17 to 21%\n(weighted blend)",                "+24 to 28%",
     "Correlated drawdowns\nin sharp market\nfall", "Ongoing"),
]
row_bg = [LGRAY, WHITE, LGRAY, WHITE, RGBColor(0xE8,0xF0,0xFE)]
for i, (row_data, bg) in enumerate(zip(fwd_rows, row_bg)):
    y = Inches(2.25 + i * 0.88)
    cells = list(row_data)
    for j, (cell, cw) in enumerate(zip(cells, col_w)):
        x = Inches(0.35 + x_pos[j])
        add_rect(sl, x, y, Inches(cw), Inches(0.86), fill=bg,
                 line=RGBColor(0xD0,0xD5,0xDD), line_w=Pt(0.3))
        color = GREEN if j in [2,3] else (RED if j==4 else (NAVY if j==0 else DGRAY))
        bold  = j in [0, 4]
        size  = 10 if j != 0 else 11
        add_text(sl, cell, x+Inches(0.05), y+Inches(0.05),
                 Inches(cw-0.1), Inches(0.76),
                 size=size, color=color, bold=bold, align=PP_ALIGN.CENTER)

# Alpha assumption box
add_rect(sl, Inches(0.35), Inches(6.75), Inches(12.6), Inches(0.42),
         fill=RGBColor(0xE8,0xF4,0xE8), line=GREEN, line_w=Pt(0.5))
add_text(sl, ("Assumptions: Nifty base case = 12% CAGR (10-yr historical average). "
              "Momentum alpha = 8.68pp OOS (may compress 10–20% in efficient periods). "
              "NiftyShop XIRR range from 2yr backtest. "
              "Alpha sensitivity: 0.14pp lost per 1pp Nifty decline — strategy is robust to moderate market weakness."),
         Inches(0.5), Inches(6.79), Inches(12.4), Inches(0.35),
         size=9, color=RGBColor(0x1A,0x5C,0x2A))

footnote(sl, "Forward returns are estimates only · Past OOS performance does not guarantee future results · Review signal health quarterly")


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 9 — OPERATIONAL PLAYBOOK
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
header_bar(sl, "Operational Playbook", "Exact commands and timing — treat as a checklist")

tasks = [
    ("MONTHLY (Sunday evening, ~20 min)", NAVY, [
        ("Nifty 50 Momentum",   "python nifty50_Momentum_deploy.py --sleeve 900000",        "Read rebalancing action list → execute Monday open"),
        ("Midcap 50 Momentum",  "python midcap_momentum_deploy.py --sleeve 200000",          "Paper trade — log expected vs actual"),
        ("Save updated holdings","python nifty50_Momentum_deploy.py --save",                "Writes current_holdings.txt after execution"),
        ("Scores-only check",   "python nifty50_Momentum_deploy.py --scores-only",           "All 50 scores — useful for conviction check"),
    ]),
    ("WEEKLY (Friday 3:15pm, ~5 min)", BLUE, [
        ("Nifty 50 NiftyShop",  "python niftyshop_deploy.py",                               "Read signal → execute 1 trade by 3:30pm → log to trade_log.csv"),
        ("Midcap NiftyShop",    "python midcap_niftyshop_deploy.py",                        "Paper trade — log to midcap_trade_log.csv"),
        ("Status check",        "python niftyshop_deploy.py --status",                      "Portfolio snapshot without generating new signal"),
        ("History / P&L",       "python niftyshop_deploy.py --history",                     "Closed positions and realised P&L from trade_log.csv"),
    ]),
    ("ON-DEMAND", GOLD, [
        ("Momentum screener",   "python momentum_screener_v3.py --min-score 60",             "Institutional-grade Nifty 50 signal report"),
        ("Value screener",      "python value_screener.py --top 10 --export picks.csv",      "400+ mid/small cap fundamental scan"),
        ("Sector filter",       "python value_screener.py --sector IT",                      "Sector-specific value screen"),
    ]),
]

y = Inches(1.28)
for (section, color, items) in tasks:
    add_rect(sl, Inches(0.35), y, Inches(12.6), Inches(0.3), fill=color)
    add_text(sl, section, Inches(0.45), y+Inches(0.03), Inches(12), Inches(0.25),
             size=11, bold=True, color=WHITE)
    y += Inches(0.32)
    for (name, cmd, note) in items:
        add_rect(sl, Inches(0.35), y, Inches(12.6), Inches(0.4), fill=LGRAY if items.index((name,cmd,note))%2==0 else WHITE)
        add_text(sl, name, Inches(0.45), y+Inches(0.06), Inches(2.0), Inches(0.28), size=10, bold=True, color=NAVY)
        add_text(sl, cmd,  Inches(2.5),  y+Inches(0.06), Inches(4.8), Inches(0.28), size=9,  color=BLUE,
                 bold=False, italic=False)
        add_text(sl, note, Inches(7.4),  y+Inches(0.06), Inches(5.4), Inches(0.28), size=9,  color=DGRAY, italic=True)
        y += Inches(0.4)
    y += Inches(0.1)

footnote(sl, "Archive/ folder = superseded scripts, do not run  ·  Backtest/ folder = R&D only  ·  All live scripts in root directory")


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 10 — APPENDIX: BACKTESTER CHAIN
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(BLANK)
header_bar(sl, "Appendix — Backtester R&D Chain", "How each backtest version improved on the previous one")

chain = [
    ("v1 · backtest_momentum.py",          "Foundation",             "Core momentum score, transaction costs, 5-year walk-forward framework"),
    ("v2 · backtest_v2.py",                "Sector Cap",             "Added sector concentration cap + parameter grid search across top-N and sector limits"),
    ("v3 · backtest_v3_final.py",          "Exit Buffer",            "Exit only if score dropped 15+ pts from entry. Turnover cut: 474% → 330%. Baseline alpha 8.41%"),
    ("v4 · backtest_v4_dynamic_sector.py", "Dynamic Sector Cap",     "Cap shrinks when sector trailing returns go negative. Marginal improvement — complexity not worth it"),
    ("v5 · backtest_v5_override.py",       "Score Override",         "Stocks ≥70/100 bypass sector cap (1 slot max). Allows high-conviction names through"),
    ("v6 · backtest_v6_signals.py",        "Signal Sweep (CURRENT)", "Tested 10+ signal variations. Kept: vol surge (+0.95pp), 6M RS (+2.20pp). Final alpha: 8.68% OOS"),
    ("regime · backtest_regime.py",        "ABANDONED",              "Regime filters (bull/bear detection) failed to improve MDD and alpha simultaneously. Dropped."),
    ("shop · niftyshop_universe_backtest.py","NiftyShop Validation", "Mean-reversion on Nifty 50, Nifty200 Momentum 30, Midcap 50. Added 200DMA filter after observing midcap MDD"),
]
y = Inches(1.28)
for i, (file, label, desc) in enumerate(chain):
    bg = LGRAY if i%2==0 else WHITE
    is_current  = "CURRENT" in label
    is_abandoned= "ABANDONED" in label
    border_color = GREEN if is_current else (RED if is_abandoned else RGBColor(0xD0,0xD5,0xDD))
    add_rect(sl, Inches(0.35), y, Inches(12.6), Inches(0.5), fill=bg,
             line=border_color, line_w=Pt(1 if (is_current or is_abandoned) else 0.3))
    badge_color = GREEN if is_current else (RED if is_abandoned else BLUE)
    add_rect(sl, Inches(0.35), y, Inches(2.1), Inches(0.5), fill=badge_color)
    add_text(sl, label, Inches(0.38), y+Inches(0.1), Inches(2.05), Inches(0.3),
             size=9, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(sl, file,  Inches(2.55), y+Inches(0.06), Inches(4.0), Inches(0.2), size=9, color=NAVY, bold=True)
    add_text(sl, desc,  Inches(2.55), y+Inches(0.26), Inches(10.2), Inches(0.2), size=9, color=DGRAY)
    y += Inches(0.53)

footnote(sl, "All backtest files are in the Backtest/ folder and should not be run in production  ·  R&D only")


# ══════════════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════════════
out = r"c:\Users\ayush\Documents\Stock_Signals\WealthOS_Strategy_Deck.pptx"
prs.save(out)
print(f"Saved: {out}")
