"""
WealthOS — Portfolio Dashboard Generator
==========================================
Reads live holdings and trade logs, fetches current prices from Yahoo Finance,
computes KPIs, and writes a self-contained dashboard.html.

Usage:
    python dashboard.py           # generates dashboard.html
    python dashboard.py --open    # generates and auto-opens in browser
    python dashboard.py --out my_report.html   # custom output path
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from datetime import date, datetime
from pathlib import Path

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent

# ── Position sizes (per-position deployed amounts) ────────────────────────────
# Must match sleeve / top_n from the deployers.
NIFTY50_PER_POS  = 60_000   # ₹9L / 15
MIDCAP_PER_POS   = 20_000   # ₹2L / 10

# ── File paths ────────────────────────────────────────────────────────────────
NIFTY_HOLDINGS   = BASE / "current_holdings.txt"
MIDCAP_HOLDINGS  = BASE / "midcap_holdings.txt"
NIFTY_LOG        = BASE / "trade_log.csv"
MIDCAP_LOG       = BASE / "midcap_trade_log.csv"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_holdings(path: Path) -> list[dict]:
    if not path.exists():
        return []
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 3:
            result.append({
                "symbol":      parts[0].upper(),
                "entry_price": float(parts[1]),
                "entry_score": float(parts[2]),
                "entry_date":  parts[3] if len(parts) >= 4 else "",
            })
    return result


def load_trade_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    import csv
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append({
                    "date":     row["date"].strip(),
                    "symbol":   row["symbol"].strip().upper(),
                    "action":   row["action"].strip().upper(),
                    "price":    float(row["price"]),
                    "quantity": float(row["quantity"]),
                    "amount":   float(row["amount"]),
                })
            except (ValueError, KeyError):
                continue
    return rows


def build_open_positions(log: list[dict]) -> list[dict]:
    """Reconstruct open NiftyShop positions from trade log (FIFO)."""
    positions: dict[str, dict] = {}
    for row in log:
        sym, action = row["symbol"], row["action"]
        if action in ("BUY_FRESH", "BUY_AVG"):
            if sym not in positions:
                positions[sym] = {"symbol": sym, "lots": [], "first_date": row["date"]}
            positions[sym]["lots"].append(row)
        elif action == "SELL" and sym in positions:
            # FIFO reduction (mirrors deployer logic)
            to_reduce = row["quantity"]
            new_lots = []
            for lot in positions[sym]["lots"]:
                if to_reduce <= 0:
                    new_lots.append(lot)
                elif lot["quantity"] <= to_reduce:
                    to_reduce -= lot["quantity"]
                else:
                    kept = (lot["quantity"] - to_reduce) / lot["quantity"]
                    new_lots.append({**lot, "quantity": lot["quantity"] - to_reduce,
                                     "amount": lot["amount"] * kept})
                    to_reduce = 0
            positions[sym]["lots"] = new_lots
            if sum(l["quantity"] for l in new_lots) <= 0.5:
                del positions[sym]

    result = []
    for sym, pos in positions.items():
        lots = pos["lots"]
        total_qty = sum(l["quantity"] for l in lots)
        total_invested = sum(l["amount"] for l in lots)
        avg_price = sum(l["price"] * l["quantity"] for l in lots) / total_qty if total_qty > 0 else 0
        result.append({
            "symbol":        sym,
            "entry_date":    pos["first_date"],
            "n_lots":        len(lots),
            "total_qty":     total_qty,
            "total_invested":total_invested,
            "avg_price":     avg_price,
            "last_buy_price":lots[-1]["price"] if lots else 0,
        })
    return result


def get_closed_trades(log: list[dict]) -> list[dict]:
    positions: dict[str, dict] = {}
    closed = []
    for row in log:
        sym, action = row["symbol"], row["action"]
        if action in ("BUY_FRESH", "BUY_AVG"):
            if sym not in positions:
                positions[sym] = {"symbol": sym, "lots": [], "first_date": row["date"]}
            positions[sym]["lots"].append(row)
        elif action == "SELL" and sym in positions:
            pos = positions[sym]
            lots = pos["lots"]
            total_qty = sum(l["quantity"] for l in lots)
            total_invested = sum(l["amount"] for l in lots)
            sold_qty = row["quantity"]
            proceeds = row["amount"]
            sold_cost = (sold_qty / total_qty) * total_invested if total_qty > 0 else total_invested
            pnl = proceeds - sold_cost

            try:
                entry_dt = datetime.strptime(pos["first_date"], "%Y-%m-%d").date()
                exit_dt  = datetime.strptime(row["date"], "%Y-%m-%d").date()
                hold_days = (exit_dt - entry_dt).days
            except ValueError:
                hold_days = 0

            closed.append({
                "symbol":    sym,
                "entry":     pos["first_date"],
                "exit":      row["date"],
                "hold_days": hold_days,
                "n_lots":    len(lots),
                "invested":  round(sold_cost, 2),
                "proceeds":  round(proceeds, 2),
                "pnl":       round(pnl, 2),
                "pnl_pct":   round(pnl / sold_cost * 100, 2) if sold_cost > 0 else 0.0,
            })
            # FIFO reduction
            to_reduce = sold_qty
            new_lots = []
            for lot in lots:
                if to_reduce <= 0:
                    new_lots.append(lot)
                elif lot["quantity"] <= to_reduce:
                    to_reduce -= lot["quantity"]
                else:
                    kept = (lot["quantity"] - to_reduce) / lot["quantity"]
                    new_lots.append({**lot, "quantity": lot["quantity"] - to_reduce,
                                     "amount": lot["amount"] * kept})
                    to_reduce = 0
            if sum(l["quantity"] for l in new_lots) <= 0.5:
                del positions[sym]
            else:
                positions[sym]["lots"] = new_lots
    return closed


# ── Price fetching ────────────────────────────────────────────────────────────

def fetch_prices(symbols_ns: list[str]) -> dict[str, float]:
    """Fetch latest closing prices. symbols_ns are .NS-suffixed tickers."""
    import yfinance as yf
    if not symbols_ns:
        return {}
    raw = yf.download(symbols_ns, period="5d", progress=False, auto_adjust=True, threads=True)
    closes = raw["Close"]
    prices = {}
    for sym_ns in symbols_ns:
        try:
            col = closes[sym_ns] if sym_ns in closes.columns else closes
            prices[sym_ns] = float(col.dropna().iloc[-1])
        except Exception:
            prices[sym_ns] = 0.0
    return prices


# ── KPI computation ───────────────────────────────────────────────────────────

def days_held(entry_date_str: str) -> int:
    if not entry_date_str:
        return 0
    try:
        return (date.today() - datetime.strptime(entry_date_str, "%Y-%m-%d").date()).days
    except ValueError:
        return 0


def annualized_return(cost: float, value: float, days: int) -> float | None:
    """CAGR in percent. Returns None if days < 7 or cost <= 0."""
    if cost <= 0 or days < 7:
        return None
    return ((value / cost) ** (365 / days) - 1) * 100


def build_momentum_rows(holdings: list[dict], prices: dict[str, float],
                        per_pos: float) -> list[dict]:
    rows = []
    for h in holdings:
        sym     = h["symbol"]
        ep      = h["entry_price"]
        qty     = round(per_pos / ep) if ep > 0 else 0
        deployed= round(qty * ep, 2)
        cur_px  = prices.get(sym + ".NS", 0.0)
        cur_val = round(qty * cur_px, 2) if cur_px > 0 else 0.0
        pnl     = round(cur_val - deployed, 2)
        pnl_pct = round((pnl / deployed * 100), 2) if deployed > 0 else 0.0
        d       = days_held(h["entry_date"])
        cagr    = annualized_return(deployed, cur_val, d)
        rows.append({
            "symbol":      sym,
            "entry_date":  h["entry_date"] or "—",
            "days_held":   d,
            "entry_price": ep,
            "cur_price":   cur_px,
            "qty":         qty,
            "deployed":    deployed,
            "cur_value":   cur_val,
            "pnl":         pnl,
            "pnl_pct":     pnl_pct,
            "cagr":        round(cagr, 1) if cagr is not None else None,
        })
    rows.sort(key=lambda r: -r["pnl_pct"])
    return rows


def build_niftyshop_rows(open_pos: list[dict], prices: dict[str, float]) -> list[dict]:
    rows = []
    for pos in open_pos:
        sym     = pos["symbol"]
        cur_px  = prices.get(sym + ".NS", 0.0)
        cur_val = round(cur_px * pos["total_qty"], 2) if cur_px > 0 else 0.0
        pnl     = round(cur_val - pos["total_invested"], 2)
        pnl_pct = round(pnl / pos["total_invested"] * 100, 2) if pos["total_invested"] > 0 else 0.0
        d       = days_held(pos["entry_date"])
        rows.append({
            "symbol":    sym,
            "entry_date":pos["entry_date"],
            "days_held": d,
            "n_lots":    pos["n_lots"],
            "avg_price": round(pos["avg_price"], 2),
            "cur_price": cur_px,
            "qty":       round(pos["total_qty"]),
            "deployed":  round(pos["total_invested"], 2),
            "cur_value": cur_val,
            "pnl":       pnl,
            "pnl_pct":   pnl_pct,
        })
    rows.sort(key=lambda r: -r["pnl_pct"])
    return rows


def strategy_summary(rows: list[dict]) -> dict:
    deployed = sum(r["deployed"] for r in rows)
    cur_val  = sum(r["cur_value"] for r in rows)
    pnl      = cur_val - deployed
    pnl_pct  = pnl / deployed * 100 if deployed > 0 else 0.0
    return {
        "deployed": round(deployed, 2),
        "cur_value": round(cur_val, 2),
        "pnl":       round(pnl, 2),
        "pnl_pct":   round(pnl_pct, 2),
    }


def niftyshop_kpis(closed: list[dict]) -> dict:
    if not closed:
        return {"total_realized": 0, "win_rate": 0, "avg_return": 0,
                "avg_hold_days": 0, "n_trades": 0, "n_wins": 0}
    n        = len(closed)
    wins     = [t for t in closed if t["pnl"] > 0]
    realized = sum(t["pnl"] for t in closed)
    avg_ret  = sum(t["pnl_pct"] for t in closed) / n
    avg_hold = sum(t["hold_days"] for t in closed) / n
    return {
        "total_realized": round(realized, 2),
        "win_rate":       round(len(wins) / n * 100, 1),
        "avg_return":     round(avg_ret, 2),
        "avg_hold_days":  round(avg_hold, 1),
        "n_trades":       n,
        "n_wins":         len(wins),
    }


def get_regime() -> dict:
    try:
        from regime_detector import get_regime as _gr
        r = _gr()
        return {
            "label":    r.label,
            "price":    r.price,
            "vs200":    r.price_vs_200_pct,
            "gap5020":  r.dma50_vs_200_pct,
            "confirmed":r.is_confirmed,
            "days":     r.consecutive_days,
        }
    except Exception:
        return {"label": "UNKNOWN", "price": 0, "vs200": 0, "gap5020": 0,
                "confirmed": False, "days": 0}


# ── HTML generation ───────────────────────────────────────────────────────────

REGIME_COLORS = {
    "BULL":     "#22c55e",
    "RECOVERY": "#f59e0b",
    "SIDEWAYS": "#38bdf8",
    "BEAR":     "#ef4444",
    "UNKNOWN":  "#6b7280",
}


def fmt_inr(v: float) -> str:
    sign = "-" if v < 0 else ""
    abs_v = abs(v)
    if abs_v >= 1e5:
        return f"{sign}₹{abs_v/1e5:.2f}L"
    return f"{sign}₹{abs_v:,.0f}"


def pnl_class(v: float) -> str:
    return "pos" if v >= 0 else "neg"


def render_momentum_table(rows: list[dict], title: str, per_pos: float) -> str:
    if not rows:
        return f"<p class='empty'>No {title} holdings found.</p>"

    tbody = ""
    for r in rows:
        cagr_str = f"{r['cagr']:+.1f}%" if r["cagr"] is not None else "—"
        tbody += f"""
        <tr>
          <td><b>{r['symbol']}</b></td>
          <td>{r['entry_date']}</td>
          <td>{r['days_held'] or '—'}</td>
          <td>₹{r['entry_price']:,.2f}</td>
          <td>₹{r['cur_price']:,.2f}</td>
          <td>{r['qty']}</td>
          <td>{fmt_inr(r['deployed'])}</td>
          <td>{fmt_inr(r['cur_value'])}</td>
          <td class='{pnl_class(r["pnl"])}'>{fmt_inr(r['pnl'])}</td>
          <td class='{pnl_class(r["pnl_pct"])}'>{r['pnl_pct']:+.1f}%</td>
          <td class='{pnl_class(r["cagr"] or 0)}'>{cagr_str}</td>
        </tr>"""

    return f"""
    <table class='data-table'>
      <thead><tr>
        <th>Symbol</th><th>Entry Date</th><th>Days</th>
        <th>Entry ₹</th><th>CMP ₹</th><th>Qty</th>
        <th>Deployed</th><th>Value</th><th>P&amp;L ₹</th><th>P&amp;L %</th><th>Ann. Return</th>
      </tr></thead>
      <tbody>{tbody}</tbody>
    </table>"""


def render_niftyshop_open_table(rows: list[dict]) -> str:
    if not rows:
        return "<p class='empty'>No open positions.</p>"
    tbody = ""
    for r in rows:
        tbody += f"""
        <tr>
          <td><b>{r['symbol']}</b></td>
          <td>{r['entry_date']}</td>
          <td>{r['days_held'] or '—'}</td>
          <td>{r['n_lots']}</td>
          <td>₹{r['avg_price']:,.2f}</td>
          <td>₹{r['cur_price']:,.2f}</td>
          <td>{r['qty']}</td>
          <td>{fmt_inr(r['deployed'])}</td>
          <td>{fmt_inr(r['cur_value'])}</td>
          <td class='{pnl_class(r["pnl"])}'>{fmt_inr(r['pnl'])}</td>
          <td class='{pnl_class(r["pnl_pct"])}'>{r['pnl_pct']:+.1f}%</td>
        </tr>"""
    return f"""
    <table class='data-table'>
      <thead><tr>
        <th>Symbol</th><th>Entry Date</th><th>Days</th><th>Lots</th>
        <th>Avg Price</th><th>CMP</th><th>Qty</th>
        <th>Deployed</th><th>Value</th><th>P&amp;L ₹</th><th>P&amp;L %</th>
      </tr></thead>
      <tbody>{tbody}</tbody>
    </table>"""


def render_closed_table(closed: list[dict]) -> str:
    if not closed:
        return "<p class='empty'>No closed trades yet.</p>"
    tbody = ""
    for t in sorted(closed, key=lambda x: x["exit"], reverse=True):
        tbody += f"""
        <tr>
          <td><b>{t['symbol']}</b></td>
          <td>{t['entry']}</td><td>{t['exit']}</td>
          <td>{t['hold_days']}</td><td>{t['n_lots']}</td>
          <td>{fmt_inr(t['invested'])}</td>
          <td>{fmt_inr(t['proceeds'])}</td>
          <td class='{pnl_class(t["pnl"])}'>{fmt_inr(t['pnl'])}</td>
          <td class='{pnl_class(t["pnl_pct"])}'>{t['pnl_pct']:+.1f}%</td>
        </tr>"""
    return f"""
    <table class='data-table'>
      <thead><tr>
        <th>Symbol</th><th>Entry</th><th>Exit</th><th>Days</th><th>Lots</th>
        <th>Invested</th><th>Proceeds</th><th>P&amp;L ₹</th><th>P&amp;L %</th>
      </tr></thead>
      <tbody>{tbody}</tbody>
    </table>"""


def kpi_card(label: str, value: str, sub: str = "", color: str = "") -> str:
    style = f'style="color:{color}"' if color else ""
    return f"""
    <div class='kpi-card'>
      <div class='kpi-label'>{label}</div>
      <div class='kpi-value' {style}>{value}</div>
      {f"<div class='kpi-sub'>{sub}</div>" if sub else ""}
    </div>"""


def section_kpis(summ: dict, extra: dict | None = None) -> str:
    color = "#22c55e" if summ["pnl"] >= 0 else "#ef4444"
    cards = [
        kpi_card("Deployed", fmt_inr(summ["deployed"])),
        kpi_card("Current Value", fmt_inr(summ["cur_value"])),
        kpi_card("Unrealized P&L", fmt_inr(summ["pnl"]),
                 f"{summ['pnl_pct']:+.2f}%", color),
    ]
    if extra:
        if extra.get("n_trades"):
            cards += [
                kpi_card("Realized P&L", fmt_inr(extra["total_realized"]),
                         f"{extra['n_wins']}/{extra['n_trades']} wins",
                         "#22c55e" if extra["total_realized"] >= 0 else "#ef4444"),
                kpi_card("Win Rate", f"{extra['win_rate']}%"),
                kpi_card("Avg Return", f"{extra['avg_return']:+.2f}%"),
                kpi_card("Avg Hold", f"{extra['avg_hold_days']:.0f} days"),
            ]
    return "<div class='kpi-row'>" + "".join(cards) + "</div>"


def generate_html(data: dict) -> str:
    regime     = data["regime"]
    rcolor     = REGIME_COLORS.get(regime["label"], "#6b7280")
    r_confirm  = "CONFIRMED" if regime["confirmed"] else f"TRANSITIONING ({regime['days']}d)"

    # Portfolio-level aggregates
    total_deployed = (data["n50_summ"]["deployed"] + data["mid_summ"]["deployed"] +
                      data["ns50_summ"]["deployed"] + data["nsmid_summ"]["deployed"])
    total_value    = (data["n50_summ"]["cur_value"] + data["mid_summ"]["cur_value"] +
                      data["ns50_summ"]["cur_value"] + data["nsmid_summ"]["cur_value"])
    total_unreal   = total_value - total_deployed
    total_realized = data["ns50_kpis"]["total_realized"] + data["nsmid_kpis"]["total_realized"]
    total_pnl      = total_unreal + total_realized
    total_pnl_pct  = total_pnl / total_deployed * 100 if total_deployed > 0 else 0

    port_color = "#22c55e" if total_pnl >= 0 else "#ef4444"

    # Allocation chart data
    alloc_labels = ["Nifty50 Momentum", "Midcap Momentum", "NiftyShop N50", "MidcapShop"]
    alloc_values = [data["n50_summ"]["deployed"], data["mid_summ"]["deployed"],
                    data["ns50_summ"]["deployed"], data["nsmid_summ"]["deployed"]]

    # P&L bar chart — all open positions sorted by P&L
    all_open = (
        [(r["symbol"], r["pnl"], "N50-Mom") for r in data["n50_rows"]] +
        [(r["symbol"], r["pnl"], "Mid-Mom") for r in data["mid_rows"]] +
        [(r["symbol"], r["pnl"], "NSop50") for r in data["ns50_rows"]] +
        [(r["symbol"], r["pnl"], "NSmid")  for r in data["nsmid_rows"]]
    )
    all_open.sort(key=lambda x: -x[1])
    bar_labels = [x[0] for x in all_open]
    bar_values = [x[1] for x in all_open]
    bar_colors = ["rgba(34,197,94,0.8)" if v >= 0 else "rgba(239,68,68,0.8)" for v in bar_values]

    ts = datetime.now().strftime("%d %b %Y  %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WealthOS Dashboard</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg:      #0d1117;
    --card-bg: #161b22;
    --border:  #30363d;
    --text:    #e6edf3;
    --muted:   #8b949e;
    --pos:     #3fb950;
    --neg:     #f85149;
    --accent:  #58a6ff;
  }}
  * {{ box-sizing: border-box; }}
  body {{ background: var(--bg); color: var(--text); font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; margin: 0; padding: 0; }}
  a {{ color: var(--accent); }}

  /* Header */
  .site-header {{ background: var(--card-bg); border-bottom: 1px solid var(--border); padding: 18px 24px; display: flex; justify-content: space-between; align-items: center; }}
  .site-header h1 {{ margin: 0; font-size: 1.25rem; font-weight: 600; letter-spacing: .5px; }}
  .site-header .ts {{ color: var(--muted); font-size: .85rem; }}

  /* Regime banner */
  .regime-banner {{ padding: 12px 24px; font-size: .9rem; border-bottom: 1px solid var(--border); display: flex; gap: 24px; align-items: center; }}
  .regime-label {{ font-weight: 700; font-size: 1.05rem; padding: 2px 12px; border-radius: 4px; }}
  .regime-stat {{ color: var(--muted); }}
  .regime-stat b {{ color: var(--text); }}

  /* Main layout */
  .main {{ padding: 20px 24px; max-width: 1600px; margin: 0 auto; }}

  /* KPI cards */
  .kpi-row {{ display: flex; flex-wrap: wrap; gap: 14px; margin-bottom: 20px; }}
  .kpi-card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 16px 20px; min-width: 160px; flex: 1; }}
  .kpi-label {{ font-size: .75rem; color: var(--muted); text-transform: uppercase; letter-spacing: .8px; margin-bottom: 6px; }}
  .kpi-value {{ font-size: 1.4rem; font-weight: 700; }}
  .kpi-sub {{ font-size: .8rem; color: var(--muted); margin-top: 3px; }}

  /* Sections */
  .section {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-bottom: 20px; }}
  .section-title {{ font-size: 1rem; font-weight: 600; margin-bottom: 16px; padding-bottom: 10px; border-bottom: 1px solid var(--border); color: var(--accent); }}

  /* Tables */
  .data-table {{ width: 100%; border-collapse: collapse; font-size: .85rem; overflow-x: auto; display: block; white-space: nowrap; }}
  .data-table th {{ background: #21262d; color: var(--muted); font-weight: 500; padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border); font-size: .78rem; text-transform: uppercase; }}
  .data-table td {{ padding: 8px 12px; border-bottom: 1px solid #21262d; }}
  .data-table tr:last-child td {{ border-bottom: none; }}
  .data-table tr:hover td {{ background: rgba(88,166,255,0.04); }}

  /* Colors */
  .pos {{ color: var(--pos); }}
  .neg {{ color: var(--neg); }}
  .empty {{ color: var(--muted); font-style: italic; padding: 12px 0; }}

  /* Charts */
  .charts-row {{ display: flex; gap: 20px; margin-bottom: 20px; flex-wrap: wrap; }}
  .chart-card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 20px; flex: 1; min-width: 300px; }}
  .chart-title {{ font-size: .85rem; color: var(--muted); text-transform: uppercase; letter-spacing: .8px; margin-bottom: 12px; }}

  /* Divider between open/closed in NiftyShop */
  .sub-heading {{ font-size: .85rem; color: var(--muted); text-transform: uppercase; letter-spacing: .8px; margin: 16px 0 8px; }}

  @media (max-width: 768px) {{
    .charts-row {{ flex-direction: column; }}
    .kpi-card {{ min-width: 140px; }}
  }}
</style>
</head>
<body>

<div class="site-header">
  <h1>WealthOS Portfolio Dashboard</h1>
  <span class="ts">Generated {ts}</span>
</div>

<div class="regime-banner" style="background:rgba({','.join(str(int(rcolor.lstrip('#')[i:i+2],16)) for i in (0,2,4))},0.08)">
  <span class="regime-label" style="background:{rcolor}20;color:{rcolor}">
    {regime['label']}
  </span>
  <span class="regime-stat">NIFTY <b>₹{regime['price']:,.0f}</b></span>
  <span class="regime-stat">vs 200DMA <b style="color:{'var(--pos)' if regime['vs200']>=0 else 'var(--neg)'}">{regime['vs200']:+.1f}%</b></span>
  <span class="regime-stat">50-200DMA gap <b>{regime['gap5020']:+.1f}%</b></span>
  <span class="regime-stat" style="color:{rcolor}">{r_confirm}</span>
</div>

<div class="main">

  <!-- ── Portfolio Summary ── -->
  <div class="section">
    <div class="section-title">Portfolio Summary</div>
    <div class="kpi-row">
      {kpi_card("Total Deployed", fmt_inr(total_deployed))}
      {kpi_card("Current Value", fmt_inr(total_value))}
      {kpi_card("Unrealized P&L", fmt_inr(total_unreal),
                f"{(total_unreal/total_deployed*100 if total_deployed else 0):+.2f}%",
                "#22c55e" if total_unreal >= 0 else "#ef4444")}
      {kpi_card("Realized P&L", fmt_inr(total_realized),
                "NiftyShop closed trades",
                "#22c55e" if total_realized >= 0 else "#ef4444")}
      {kpi_card("Overall P&L", fmt_inr(total_pnl),
                f"{total_pnl_pct:+.2f}% of deployed",
                port_color)}
    </div>
  </div>

  <!-- ── Charts ── -->
  <div class="charts-row">
    <div class="chart-card" style="max-width:320px">
      <div class="chart-title">Allocation by Strategy</div>
      <canvas id="allocChart" height="240"></canvas>
    </div>
    <div class="chart-card">
      <div class="chart-title">Unrealized P&amp;L by Position</div>
      <canvas id="pnlChart" height="240"></canvas>
    </div>
  </div>

  <!-- ── Nifty 50 Momentum ── -->
  <div class="section">
    <div class="section-title">Nifty 50 Momentum  <small style="color:var(--muted);font-weight:400">(monthly · ₹9L sleeve)</small></div>
    {section_kpis(data["n50_summ"])}
    {render_momentum_table(data["n50_rows"], "Nifty 50", NIFTY50_PER_POS)}
  </div>

  <!-- ── Midcap 50 Momentum ── -->
  <div class="section">
    <div class="section-title">Midcap 50 Momentum  <small style="color:var(--muted);font-weight:400">(monthly · ₹2L sleeve)</small></div>
    {section_kpis(data["mid_summ"])}
    {render_momentum_table(data["mid_rows"], "Midcap", MIDCAP_PER_POS)}
  </div>

  <!-- ── NiftyShop ── -->
  <div class="section">
    <div class="section-title">NiftyShop  <small style="color:var(--muted);font-weight:400">(weekly · ₹4L · mean-reversion)</small></div>
    {section_kpis(data["ns50_summ"], data["ns50_kpis"])}
    <div class="sub-heading">Open Positions</div>
    {render_niftyshop_open_table(data["ns50_rows"])}
    <div class="sub-heading" style="margin-top:24px">Closed Trades</div>
    {render_closed_table(data["ns50_closed"])}
  </div>

  <!-- ── MidcapShop ── -->
  <div class="section">
    <div class="section-title">MidcapShop  <small style="color:var(--muted);font-weight:400">(weekly · ₹2L · mean-reversion)</small></div>
    {section_kpis(data["nsmid_summ"], data["nsmid_kpis"])}
    <div class="sub-heading">Open Positions</div>
    {render_niftyshop_open_table(data["nsmid_rows"])}
    <div class="sub-heading" style="margin-top:24px">Closed Trades</div>
    {render_closed_table(data["nsmid_closed"])}
  </div>

</div><!-- /main -->

<script>
// Allocation pie
new Chart(document.getElementById('allocChart'), {{
  type: 'doughnut',
  data: {{
    labels: {json.dumps(alloc_labels)},
    datasets: [{{
      data: {json.dumps(alloc_values)},
      backgroundColor: ['#3b82f6','#8b5cf6','#22c55e','#f59e0b'],
      borderWidth: 0,
    }}]
  }},
  options: {{
    plugins: {{
      legend: {{ position: 'bottom', labels: {{ color: '#8b949e', font: {{ size: 11 }} }} }},
      tooltip: {{
        callbacks: {{
          label: ctx => ' ₹' + (ctx.raw/1e5).toFixed(2) + 'L  (' +
            Math.round(ctx.raw / {max(total_deployed,1)} * 100) + '%)'
        }}
      }}
    }},
    cutout: '60%',
  }}
}});

// P&L bar
new Chart(document.getElementById('pnlChart'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(bar_labels)},
    datasets: [{{
      label: 'Unrealized P&L ₹',
      data: {json.dumps(bar_values)},
      backgroundColor: {json.dumps(bar_colors)},
      borderRadius: 3,
    }}]
  }},
  options: {{
    indexAxis: 'x',
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{
        callbacks: {{
          label: ctx => ' ₹' + ctx.raw.toLocaleString('en-IN', {{maximumFractionDigits:0}})
        }}
      }}
    }},
    scales: {{
      x: {{ ticks: {{ color: '#8b949e', font: {{ size: 10 }} }}, grid: {{ color: '#21262d' }} }},
      y: {{ ticks: {{ color: '#8b949e', font: {{ size: 10 }},
              callback: v => '₹' + (v/1000).toFixed(0) + 'K' }},
           grid: {{ color: '#21262d' }} }},
    }}
  }}
}});
</script>
</body>
</html>"""
    return html


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="WealthOS Portfolio Dashboard Generator")
    p.add_argument("--open", action="store_true", help="Auto-open in browser after generating")
    p.add_argument("--out",  default="dashboard.html", help="Output file path (default: dashboard.html)")
    a = p.parse_args()

    print("WealthOS Dashboard Generator")
    print(f"  {datetime.now().strftime('%d %b %Y  %H:%M')}\n")

    # Load all data
    print("  Loading holdings and trade logs...")
    n50_hold  = load_holdings(NIFTY_HOLDINGS)
    mid_hold  = load_holdings(MIDCAP_HOLDINGS)
    ns50_log  = load_trade_log(NIFTY_LOG)
    nsmid_log = load_trade_log(MIDCAP_LOG)

    ns50_open   = build_open_positions(ns50_log)
    nsmid_open  = build_open_positions(nsmid_log)
    ns50_closed = get_closed_trades(ns50_log)
    nsmid_closed= get_closed_trades(nsmid_log)

    # Collect all symbols needing live prices
    syms_ns = list({h["symbol"] + ".NS" for h in n50_hold + mid_hold} |
                   {p["symbol"] + ".NS" for p in ns50_open + nsmid_open})

    print(f"  Fetching live prices for {len(syms_ns)} symbols...")
    prices = fetch_prices(syms_ns)

    print("  Computing KPIs...")
    regime    = get_regime()

    n50_rows  = build_momentum_rows(n50_hold,  prices, NIFTY50_PER_POS)
    mid_rows  = build_momentum_rows(mid_hold,  prices, MIDCAP_PER_POS)
    ns50_rows = build_niftyshop_rows(ns50_open,  prices)
    nsmid_rows= build_niftyshop_rows(nsmid_open, prices)

    n50_summ   = strategy_summary(n50_rows)
    mid_summ   = strategy_summary(mid_rows)
    ns50_summ  = strategy_summary(ns50_rows)
    nsmid_summ = strategy_summary(nsmid_rows)
    ns50_kpis  = niftyshop_kpis(ns50_closed)
    nsmid_kpis = niftyshop_kpis(nsmid_closed)

    data = {
        "regime":       regime,
        "n50_rows":     n50_rows,   "mid_rows":     mid_rows,
        "ns50_rows":    ns50_rows,  "nsmid_rows":   nsmid_rows,
        "ns50_closed":  ns50_closed,"nsmid_closed": nsmid_closed,
        "n50_summ":     n50_summ,   "mid_summ":     mid_summ,
        "ns50_summ":    ns50_summ,  "nsmid_summ":   nsmid_summ,
        "ns50_kpis":    ns50_kpis,  "nsmid_kpis":   nsmid_kpis,
    }

    print("  Generating HTML...")
    html = generate_html(data)
    out  = Path(a.out)
    out.write_text(html, encoding="utf-8")
    print(f"\n  Dashboard written → {out.resolve()}")

    if a.open:
        import webbrowser
        webbrowser.open(out.resolve().as_uri())
        print("  Opened in browser.")


if __name__ == "__main__":
    main()
