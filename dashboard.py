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
import warnings
from datetime import date, datetime
from pathlib import Path

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent

NIFTY50_PER_POS  = 60_000
MIDCAP_PER_POS   = 20_000
SHOP_PROFIT_TGT  = 0.08   # +8% → exit
SHOP_AVG_TRG     = 0.03   # −3% below last buy → average
SHOP_MAX_PER_STK = 40_000

NIFTY_HOLDINGS   = BASE / "current_holdings.txt"
MIDCAP_HOLDINGS  = BASE / "midcap_holdings.txt"
NIFTY_LOG        = BASE / "trade_log.csv"
MIDCAP_LOG       = BASE / "midcap_trade_log.csv"


# ── Date helpers ──────────────────────────────────────────────────────────────

def parse_date(s: str) -> date | None:
    """Parse DD-Mon-YY, DD Mon YY, DD-Mon-YYYY, DD Mon YYYY, or YYYY-MM-DD."""
    for fmt in ("%d-%b-%y", "%d %b %y", "%d-%b-%Y", "%d %b %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            continue
    return None


def days_held(entry_date_str: str) -> int:
    if not entry_date_str:
        return 0
    d = parse_date(entry_date_str)
    return (date.today() - d).days if d else 0


def fmt_date(s: str) -> str:
    d = parse_date(s)
    return d.strftime("%d %b %y") if d else s


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
    positions: dict[str, dict] = {}
    for row in log:
        sym, action = row["symbol"], row["action"]
        if action in ("BUY_FRESH", "BUY_AVG"):
            if sym not in positions:
                positions[sym] = {"symbol": sym, "lots": [], "first_date": row["date"]}
            positions[sym]["lots"].append(row)
        elif action == "SELL" and sym in positions:
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
        avg_price = (sum(l["price"] * l["quantity"] for l in lots) / total_qty
                     if total_qty > 0 else 0)
        result.append({
            "symbol":         sym,
            "entry_date":     pos["first_date"],
            "n_lots":         len(lots),
            "total_qty":      total_qty,
            "total_invested": total_invested,
            "avg_price":      avg_price,
            "last_buy_price": lots[-1]["price"] if lots else 0,
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

            entry_dt = parse_date(pos["first_date"])
            exit_dt  = parse_date(row["date"])
            hold_days = (exit_dt - entry_dt).days if (entry_dt and exit_dt) else 0

            closed.append({
                "symbol":    sym,
                "entry":     fmt_date(pos["first_date"]),
                "exit":      fmt_date(row["date"]),
                "hold_days": hold_days,
                "n_lots":    len(lots),
                "invested":  round(sold_cost, 2),
                "proceeds":  round(proceeds, 2),
                "pnl":       round(pnl, 2),
                "pnl_pct":   round(pnl / sold_cost * 100, 2) if sold_cost > 0 else 0.0,
            })
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
    import yfinance as yf
    if not symbols_ns:
        return {}
    raw = yf.download(symbols_ns, period="5d", progress=False, auto_adjust=True, threads=True)
    if raw.empty:
        return {}
    closes = raw["Close"] if hasattr(raw["Close"], "columns") else raw
    prices: dict[str, float] = {}
    for sym_ns in symbols_ns:
        try:
            col = closes[sym_ns] if sym_ns in closes.columns else closes
            prices[sym_ns] = round(float(col.dropna().iloc[-1]), 2)
        except Exception:
            prices[sym_ns] = 0.0
    return prices


# ── KPI computation ───────────────────────────────────────────────────────────

def annualized_return(cost: float, value: float, days: int) -> float | None:
    if cost <= 0 or days < 7:
        return None
    return ((value / cost) ** (365 / days) - 1) * 100


def build_momentum_rows(holdings: list[dict], prices: dict[str, float],
                        per_pos: float) -> list[dict]:
    rows = []
    for h in holdings:
        sym      = h["symbol"]
        ep       = h["entry_price"]
        qty      = round(per_pos / ep) if ep > 0 else 0
        deployed = round(qty * ep, 2)
        cur_px   = prices.get(sym + ".NS", 0.0)
        cur_val  = round(qty * cur_px, 2) if cur_px > 0 else 0.0
        pnl      = round(cur_val - deployed, 2)
        pnl_pct  = round(pnl / deployed * 100, 2) if deployed > 0 else 0.0
        d        = days_held(h["entry_date"])
        cagr     = annualized_return(deployed, cur_val, d)
        rows.append({
            "symbol":      sym,
            "entry_date":  fmt_date(h["entry_date"]) if h["entry_date"] else "—",
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
        sym          = pos["symbol"]
        cur_px       = prices.get(sym + ".NS", 0.0)
        avg_px       = pos["avg_price"]
        last_buy     = pos["last_buy_price"]
        cur_val      = round(cur_px * pos["total_qty"], 2) if cur_px > 0 else 0.0
        pnl          = round(cur_val - pos["total_invested"], 2)
        pnl_pct      = round(pnl / pos["total_invested"] * 100, 2) if pos["total_invested"] > 0 else 0.0
        d            = days_held(pos["entry_date"])
        exit_target  = avg_px * (1 + SHOP_PROFIT_TGT)
        to_exit_pct  = ((cur_px / exit_target) - 1) * 100 if (cur_px > 0 and exit_target > 0) else -100.0
        exit_prog    = min(100, max(0, (cur_px / exit_target) * 100)) if exit_target > 0 else 0
        exit_trig    = cur_px >= exit_target if cur_px > 0 else False
        avg_trig     = (cur_px <= last_buy * (1 - SHOP_AVG_TRG) if cur_px > 0 else False)
        can_avg      = pos["total_invested"] + 15_000 <= SHOP_MAX_PER_STK

        if exit_trig:
            signal = "SELL"
        elif avg_trig and can_avg:
            signal = "AVG"
        elif avg_trig and not can_avg:
            signal = "AT CAP"
        else:
            signal = "HOLD"

        rows.append({
            "symbol":       sym,
            "entry_date":   fmt_date(pos["entry_date"]),
            "days_held":    d,
            "n_lots":       pos["n_lots"],
            "avg_price":    round(avg_px, 2),
            "last_buy":     round(last_buy, 2),
            "cur_price":    cur_px,
            "qty":          round(pos["total_qty"]),
            "deployed":     round(pos["total_invested"], 2),
            "cur_value":    cur_val,
            "pnl":          pnl,
            "pnl_pct":      pnl_pct,
            "exit_target":  round(exit_target, 2),
            "to_exit_pct":  round(to_exit_pct, 1),
            "exit_progress":round(exit_prog, 1),
            "signal":       signal,
        })
    rows.sort(key=lambda r: -r["pnl_pct"])
    return rows


def strategy_summary(rows: list[dict]) -> dict:
    deployed = sum(r["deployed"] for r in rows)
    cur_val  = sum(r["cur_value"] for r in rows)
    pnl      = cur_val - deployed
    pnl_pct  = pnl / deployed * 100 if deployed > 0 else 0.0
    return {
        "deployed":  round(deployed, 2),
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


def compute_benchmark(all_rows: list[dict]) -> dict:
    """Compare aggregate open-position P&L to Nifty 50 from the earliest entry date."""
    import yfinance as yf

    entry_dates = []
    for r in all_rows:
        if r.get("entry_date") and r["entry_date"] != "—":
            d = parse_date(r["entry_date"])
            if d:
                entry_dates.append(d)
    if not entry_dates:
        return {"available": False}

    since       = min(entry_dates)
    total_dep   = sum(r["deployed"] for r in all_rows)
    total_val   = sum(r["cur_value"] for r in all_rows)
    port_ret    = (total_val / total_dep - 1) * 100 if total_dep > 0 else 0.0

    try:
        df = yf.download("^NSEI", start=since.isoformat(),
                         progress=False, auto_adjust=True)
        if df.empty:
            return {"available": False}
        raw   = df["Close"]
        close = (raw.iloc[:, 0] if hasattr(raw, "columns") else raw).dropna()
        nifty_ret  = (float(close.iloc[-1]) / float(close.iloc[0]) - 1) * 100
        # Weekly series for the sparkline (max 52 points)
        weekly     = close.resample("W").last().dropna()
        base       = float(weekly.iloc[0])
        nifty_idx  = [round((float(v) / base - 1) * 100, 2) for v in weekly]
        # Portfolio indexed: assume linear P&L accrual (approximation)
        n          = len(nifty_idx)
        port_idx   = [round(port_ret * i / max(n - 1, 1), 2) for i in range(n)]
    except Exception:
        return {"available": False}

    days = (date.today() - since).days
    return {
        "available":  True,
        "since":      since.strftime("%d %b %y"),
        "days":       days,
        "port_ret":   round(port_ret, 2),
        "nifty_ret":  round(nifty_ret, 2),
        "alpha_pp":   round(port_ret - nifty_ret, 2),
        "nifty_idx":  nifty_idx,
        "port_idx":   port_idx,
    }


def _hm_colors(pct: float) -> tuple[str, str]:
    """(background, text) hex for heatmap cells."""
    if pct >= 12:  return "#14532d", "#4ade80"
    if pct >= 6:   return "#166534", "#86efac"
    if pct >= 2:   return "#1a3d22", "#a7f3d0"
    if pct >= 0:   return "#1e2e1e", "#6ee7b7"
    if pct >= -4:  return "#3b1e1e", "#fca5a5"
    if pct >= -8:  return "#7f1d1d", "#fca5a5"
    if pct >= -15: return "#991b1b", "#fef2f2"
    return "#450a0a", "#ff8080"


def render_heatmap(all_rows: list[dict]) -> str:
    if not all_rows:
        return "<p class='empty'>No open positions.</p>"
    icons = {"n50": "🔵", "mid": "🟣", "ns50": "🟢", "nsmid": "🟡"}
    cells = ""
    for r in sorted(all_rows, key=lambda x: -x["pnl_pct"]):
        bg, fg = _hm_colors(r["pnl_pct"])
        icon   = icons.get(r.get("strat", ""), "")
        pnl_rs = r.get("pnl", 0)
        cells += (
            f'<div class="hm-cell" style="background:{bg};color:{fg}" '
            f'title="{r["symbol"]} · {r["pnl_pct"]:+.1f}% · '
            f'₹{abs(pnl_rs):,.0f} {"gain" if pnl_rs>=0 else "loss"}">'
            f'<div class="hm-sym">{r["symbol"]}</div>'
            f'<div class="hm-pct">{r["pnl_pct"]:+.1f}%</div>'
            f'<div class="hm-icon">{icon}</div>'
            f'</div>'
        )
    legend = "".join(
        f'<span class="hm-leg" style="background:{bg};color:{fg}">{lbl}</span>'
        for lbl, bg, fg in [
            (">+6%",  "#166534", "#86efac"),
            ("+2–6%", "#1a3d22", "#a7f3d0"),
            ("0–2%",  "#1e2e1e", "#6ee7b7"),
            ("-4–0%", "#3b1e1e", "#fca5a5"),
            ("<-8%",  "#7f1d1d", "#fca5a5"),
        ]
    )
    return f'<div class="hm-legend">{legend}</div><div class="heatmap">{cells}</div>'


# ── Formatting helpers ────────────────────────────────────────────────────────

REGIME_COLORS = {
    "BULL":     "#22c55e",
    "RECOVERY": "#f59e0b",
    "SIDEWAYS": "#38bdf8",
    "BEAR":     "#ef4444",
    "UNKNOWN":  "#6b7280",
}

SIGNAL_STYLE = {
    "SELL":   ("bg-sell",   "SELL ↑"),
    "AVG":    ("bg-avg",    "AVG ↓"),
    "AT CAP": ("bg-atcap",  "AT CAP"),
    "HOLD":   ("bg-hold",   "HOLD"),
}


def fmt_inr(v: float, lakh: bool = True) -> str:
    sign  = "-" if v < 0 else ""
    abs_v = abs(v)
    if lakh and abs_v >= 1e5:
        return f"{sign}₹{abs_v/1e5:.2f}L"
    return f"{sign}₹{abs_v:,.0f}"


def pnl_cls(v: float) -> str:
    return "pos" if v >= 0 else "neg"


# ── HTML renderers ────────────────────────────────────────────────────────────

def render_momentum_table(rows: list[dict], per_pos: float) -> str:
    if not rows:
        return "<p class='empty'>No holdings found. Create current_holdings.txt to track momentum positions.</p>"
    tbody = ""
    for r in rows:
        cagr_str = f"{r['cagr']:+.1f}%" if r["cagr"] is not None else "—"
        tbody += f"""
        <tr>
          <td><span class="sym">{r['symbol']}</span></td>
          <td class="muted">{r['entry_date']}</td>
          <td class="muted">{r['days_held'] or '—'}</td>
          <td>₹{r['entry_price']:,.2f}</td>
          <td class="{"pos" if r["cur_price"] >= r["entry_price"] else "neg"}">₹{r['cur_price']:,.2f}</td>
          <td>{r['qty']}</td>
          <td>{fmt_inr(r['deployed'])}</td>
          <td>{fmt_inr(r['cur_value'])}</td>
          <td class='{pnl_cls(r["pnl"])}'>{fmt_inr(r['pnl'], False)}</td>
          <td class='{pnl_cls(r["pnl_pct"])} fw'>{r['pnl_pct']:+.1f}%</td>
          <td class='{pnl_cls(r["cagr"] or 0)}'>{cagr_str}</td>
        </tr>"""
    return f"""
    <div class="table-wrap">
    <table class='data-table'>
      <thead><tr>
        <th>Symbol</th><th>Entry</th><th>Days</th>
        <th>Entry ₹</th><th>CMP ₹</th><th>Qty</th>
        <th>Deployed</th><th>Value</th><th>P&amp;L ₹</th><th>P&amp;L%</th><th>CAGR</th>
      </tr></thead>
      <tbody>{tbody}</tbody>
    </table></div>"""


def render_niftyshop_open_table(rows: list[dict]) -> str:
    if not rows:
        return "<p class='empty'>No open positions.</p>"
    tbody = ""
    for r in rows:
        scls, slabel = SIGNAL_STYLE.get(r["signal"], ("bg-hold", r["signal"]))
        prog         = r["exit_progress"]
        tbody += f"""
        <tr>
          <td><span class="sym">{r['symbol']}</span></td>
          <td class="muted">{r['entry_date']}</td>
          <td class="muted">{r['days_held'] or '—'}</td>
          <td class="center">{r['n_lots']}</td>
          <td>₹{r['avg_price']:,.2f}</td>
          <td class="muted">₹{r['last_buy']:,.2f}</td>
          <td class="{"pos" if r["cur_price"] >= r["avg_price"] else "neg"}">₹{r['cur_price']:,.2f}</td>
          <td>{r['qty']}</td>
          <td>{fmt_inr(r['deployed'])}</td>
          <td class='{pnl_cls(r["pnl"])}'>{fmt_inr(r['pnl'], False)}</td>
          <td class='{pnl_cls(r["pnl_pct"])} fw'>{r['pnl_pct']:+.1f}%</td>
          <td>
            <div class="prog-wrap">
              <div class="prog-bar" style="width:{prog}%"></div>
            </div>
            <span class="muted" style="font-size:.75rem">{r['to_exit_pct']:+.1f}% to exit</span>
          </td>
          <td><span class="badge {scls}">{slabel}</span></td>
        </tr>"""
    return f"""
    <div class="table-wrap">
    <table class='data-table'>
      <thead><tr>
        <th>Symbol</th><th>Entry</th><th>Days</th><th>Lots</th>
        <th>Avg ₹</th><th>Last Buy</th><th>CMP ₹</th><th>Qty</th>
        <th>Deployed</th><th>P&amp;L ₹</th><th>P&amp;L%</th><th>To Exit</th><th>Signal</th>
      </tr></thead>
      <tbody>{tbody}</tbody>
    </table></div>"""


def render_closed_table(closed: list[dict]) -> str:
    if not closed:
        return "<p class='empty'>No closed trades yet.</p>"
    tbody = ""
    for t in sorted(closed, key=lambda x: x["exit"], reverse=True):
        tbody += f"""
        <tr>
          <td><span class="sym">{t['symbol']}</span></td>
          <td class="muted">{t['entry']}</td>
          <td class="muted">{t['exit']}</td>
          <td class="muted center">{t['hold_days']}</td>
          <td class="center">{t['n_lots']}</td>
          <td>{fmt_inr(t['invested'])}</td>
          <td>{fmt_inr(t['proceeds'])}</td>
          <td class='{pnl_cls(t["pnl"])}'>{fmt_inr(t['pnl'], False)}</td>
          <td class='{pnl_cls(t["pnl_pct"])} fw'>{t['pnl_pct']:+.1f}%</td>
        </tr>"""
    return f"""
    <div class="table-wrap">
    <table class='data-table'>
      <thead><tr>
        <th>Symbol</th><th>Entry</th><th>Exit</th><th>Days</th><th>Lots</th>
        <th>Invested</th><th>Proceeds</th><th>P&amp;L ₹</th><th>P&amp;L%</th>
      </tr></thead>
      <tbody>{tbody}</tbody>
    </table></div>"""


def kpi_card(label: str, value: str, sub: str = "", color: str = "",
             icon: str = "") -> str:
    style = f'style="color:{color}"' if color else ""
    icon_html = f'<span class="kpi-icon">{icon}</span>' if icon else ""
    return f"""
    <div class='kpi-card'>
      {icon_html}
      <div class='kpi-label'>{label}</div>
      <div class='kpi-value' {style}>{value}</div>
      {f"<div class='kpi-sub'>{sub}</div>" if sub else ""}
    </div>"""


def section_kpis(summ: dict, extra: dict | None = None) -> str:
    color = "#22c55e" if summ["pnl"] >= 0 else "#ef4444"
    cards = [
        kpi_card("Deployed",       fmt_inr(summ["deployed"]),  icon="💼"),
        kpi_card("Current Value",  fmt_inr(summ["cur_value"]), icon="📊"),
        kpi_card("Unrealized P&L", fmt_inr(summ["pnl"]),
                 f'{summ["pnl_pct"]:+.2f}%', color, icon="📈" if summ["pnl"] >= 0 else "📉"),
    ]
    if extra and extra.get("n_trades"):
        cards += [
            kpi_card("Realized P&L",  fmt_inr(extra["total_realized"]),
                     f'{extra["n_wins"]}/{extra["n_trades"]} wins',
                     "#22c55e" if extra["total_realized"] >= 0 else "#ef4444", icon="✅"),
            kpi_card("Win Rate",      f'{extra["win_rate"]}%',  icon="🎯"),
            kpi_card("Avg Return",    f'{extra["avg_return"]:+.2f}%', icon="⚡"),
            kpi_card("Avg Hold",      f'{extra["avg_hold_days"]:.0f}d', icon="⏱"),
        ]
    return "<div class='kpi-row'>" + "".join(cards) + "</div>"


def render_action_card(ns50_rows: list[dict], nsmid_rows: list[dict]) -> str:
    """Top-of-page 'this week's actions' summary."""
    actions = []
    for r in ns50_rows:
        if r["signal"] in ("SELL", "AVG"):
            actions.append((r["signal"], r["symbol"], r["cur_price"],
                            r["pnl_pct"], "Nifty 50 Shop"))
    for r in nsmid_rows:
        if r["signal"] in ("SELL", "AVG"):
            actions.append((r["signal"], r["symbol"], r["cur_price"],
                            r["pnl_pct"], "Midcap Shop"))

    if not actions:
        body = "<span class='action-none'>✓ No action required this week — hold all positions</span>"
    else:
        items = ""
        for sig, sym, px, pp, strat in actions:
            scls, slabel = SIGNAL_STYLE.get(sig, ("bg-hold", sig))
            items += (f'<div class="action-item">'
                      f'<span class="badge {scls}">{slabel}</span> '
                      f'<b>{sym}</b> @ ₹{px:,.2f} '
                      f'<span class="muted">({pp:+.1f}% · {strat})</span>'
                      f'</div>')
        body = items

    return f"""
    <div class="action-card">
      <div class="action-title">📋  This Week's Signals</div>
      <div class="action-body">{body}</div>
    </div>"""


# ── Main HTML ─────────────────────────────────────────────────────────────────

def generate_html(data: dict) -> str:
    regime    = data["regime"]
    rcolor    = REGIME_COLORS.get(regime["label"], "#6b7280")
    r_confirm = "CONFIRMED" if regime["confirmed"] else f"{regime['days']}d"

    total_deployed = sum(data[k]["deployed"] for k in ("n50_summ","mid_summ","ns50_summ","nsmid_summ"))
    total_value    = sum(data[k]["cur_value"] for k in ("n50_summ","mid_summ","ns50_summ","nsmid_summ"))
    total_unreal   = total_value - total_deployed
    total_realized = data["ns50_kpis"]["total_realized"] + data["nsmid_kpis"]["total_realized"]
    total_pnl      = total_unreal + total_realized
    total_pnl_pct  = total_pnl / total_deployed * 100 if total_deployed > 0 else 0
    port_color     = "#22c55e" if total_pnl >= 0 else "#ef4444"

    alloc_labels = ["N50 Momentum", "Midcap Mom.", "NiftyShop N50", "MidcapShop"]
    alloc_values = [data[k]["deployed"] for k in ("n50_summ","mid_summ","ns50_summ","nsmid_summ")]

    # All open rows tagged by strategy for heatmap
    all_open_rows = (
        [{**r, "strat": "n50"}   for r in data["n50_rows"]]  +
        [{**r, "strat": "mid"}   for r in data["mid_rows"]]  +
        [{**r, "strat": "ns50"}  for r in data["ns50_rows"]] +
        [{**r, "strat": "nsmid"} for r in data["nsmid_rows"]]
    )
    heatmap_html = render_heatmap(all_open_rows)

    # Benchmark comparison
    bench = data.get("bench", {"available": False})
    if bench.get("available"):
        alpha_color     = "#22c55e" if bench["alpha_pp"] >= 0 else "#ef4444"
        port_ret_color  = "#22c55e" if bench["port_ret"] >= 0 else "#ef4444"
        alpha_card_html = f"""
        <div class="alpha-stats">
          <div class="alpha-row"><span>Portfolio</span>
            <span style="color:{port_ret_color}">{bench['port_ret']:+.1f}%</span></div>
          <div class="alpha-row"><span>Nifty 50</span>
            <span>{bench['nifty_ret']:+.1f}%</span></div>
          <div class="alpha-row alpha-big"><span>Alpha</span>
            <span style="color:{alpha_color}">{bench['alpha_pp']:+.1f}pp</span></div>
          <div class="alpha-since">Since {bench['since']} &nbsp;·&nbsp; {bench['days']} days</div>
        </div>"""
        bench_canvas_html = '<canvas id="benchChart" height="130" style="margin-top:10px"></canvas>'
        bench_js = f"""
new Chart(document.getElementById('benchChart'), {{
  type: 'line',
  data: {{
    labels: Array.from({{length:{len(bench['nifty_idx'])}}}, (_,i) => i),
    datasets: [
      {{
        label: 'Portfolio',
        data: {json.dumps(bench['port_idx'])},
        borderColor: '#60a5fa', backgroundColor: 'rgba(96,165,250,.1)',
        borderWidth: 2, pointRadius: 0, fill: true, tension: .4,
      }},
      {{
        label: 'Nifty 50',
        data: {json.dumps(bench['nifty_idx'])},
        borderColor: '#f59e0b', backgroundColor: 'transparent',
        borderWidth: 1.5, pointRadius: 0, tension: .4,
        borderDash: [5,3],
      }}
    ]
  }},
  options: {{
    plugins: {{
      legend: {{ position:'bottom', labels:{{ color:'#94a3b8', font:{{size:10}}, padding:10 }} }},
      tooltip: {{ callbacks: {{ label: ctx => ' ' + ctx.raw.toFixed(1) + '%' }} }}
    }},
    scales: {{
      x: {{ display: false }},
      y: {{ ticks: {{ ...darkTick, callback: v => v.toFixed(0)+'%' }}, grid: darkGrid }},
    }}
  }}
}});"""
    else:
        alpha_card_html   = "<p class='empty' style='font-size:.78rem'>Add entry dates to holdings files for benchmark comparison.</p>"
        bench_canvas_html = ""
        bench_js          = ""

    ts = datetime.now().strftime("%d %b %Y  %H:%M")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WealthOS Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
<style>
:root {{
  --bg:       #0a0e1a;
  --card:     #111827;
  --card2:    #1a2235;
  --border:   #1f2d45;
  --text:     #e2e8f0;
  --muted:    #64748b;
  --pos:      #34d399;
  --neg:      #f87171;
  --accent:   #60a5fa;
  --gold:     #fbbf24;
  --radius:   10px;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: var(--bg); color: var(--text); font-family: -apple-system,BlinkMacSystemFont,'Inter','Segoe UI',sans-serif; font-size: 14px; line-height: 1.5; }}

/* ── Top bar ── */
.topbar {{
  position: sticky; top: 0; z-index: 100;
  background: rgba(10,14,26,0.92); backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--border);
  padding: 12px 28px; display: flex; align-items: center; gap: 24px;
}}
.topbar-logo {{ font-weight: 800; font-size: 1rem; letter-spacing: .5px; color: var(--accent); }}
.topbar-pnl {{ font-size: .85rem; }}
.topbar-ts {{ margin-left: auto; color: var(--muted); font-size: .78rem; }}

/* ── Regime banner ── */
.regime {{
  padding: 10px 28px; font-size: .85rem;
  display: flex; gap: 20px; align-items: center;
  border-bottom: 1px solid var(--border);
}}
.regime-pill {{
  font-weight: 700; font-size: .82rem;
  padding: 3px 12px; border-radius: 99px;
  letter-spacing: .5px;
}}
.regime-stat {{ color: var(--muted); }}
.regime-stat b {{ color: var(--text); }}

/* ── Layout ── */
.main {{ padding: 20px 28px 60px; max-width: 1700px; margin: 0 auto; }}

/* ── Action card ── */
.action-card {{
  background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
  border: 1px solid #334155; border-radius: var(--radius);
  padding: 16px 22px; margin-bottom: 18px;
  display: flex; align-items: center; gap: 20px; flex-wrap: wrap;
}}
.action-title {{ font-weight: 700; font-size: .9rem; white-space: nowrap; color: var(--gold); }}
.action-body  {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }}
.action-item  {{ display: flex; gap: 8px; align-items: center; font-size: .85rem; }}
.action-none  {{ color: var(--pos); font-size: .85rem; }}

/* ── KPI cards ── */
.kpi-row {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 18px; }}
.kpi-card {{
  background: var(--card); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 14px 18px; flex: 1; min-width: 140px;
  position: relative; overflow: hidden;
}}
.kpi-card::before {{
  content: ''; position: absolute; inset: 0;
  background: linear-gradient(135deg, rgba(96,165,250,.04) 0%, transparent 60%);
}}
.kpi-icon  {{ font-size: 1rem; margin-bottom: 4px; display: block; }}
.kpi-label {{ font-size: .7rem; color: var(--muted); text-transform: uppercase; letter-spacing: .9px; margin-bottom: 4px; }}
.kpi-value {{ font-size: 1.3rem; font-weight: 700; }}
.kpi-sub   {{ font-size: .75rem; color: var(--muted); margin-top: 3px; }}

/* ── Sections ── */
.section {{
  background: var(--card); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 20px 22px; margin-bottom: 18px;
}}
.section-title {{
  font-size: .9rem; font-weight: 700; color: var(--accent);
  margin-bottom: 14px; padding-bottom: 10px;
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 10px;
}}
.section-sub {{
  font-size: .78rem; color: var(--muted); font-weight: 400;
}}
.sub-heading {{
  font-size: .72rem; color: var(--muted); text-transform: uppercase;
  letter-spacing: .8px; margin: 18px 0 8px;
}}

/* ── Charts ── */
.charts-row {{ display: flex; gap: 16px; margin-bottom: 18px; flex-wrap: wrap; }}
.chart-card {{
  background: var(--card); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 18px 20px; flex: 1; min-width: 280px;
}}
.chart-title {{ font-size: .72rem; color: var(--muted); text-transform: uppercase; letter-spacing: .8px; margin-bottom: 12px; }}

/* ── Tables ── */
.table-wrap {{ overflow-x: auto; }}
.data-table {{
  width: 100%; border-collapse: collapse; font-size: .82rem;
  white-space: nowrap;
}}
.data-table th {{
  background: #0d1828; color: var(--muted); font-weight: 500;
  padding: 8px 12px; text-align: left;
  border-bottom: 1px solid var(--border);
  font-size: .7rem; text-transform: uppercase; letter-spacing: .5px;
}}
.data-table td {{ padding: 9px 12px; border-bottom: 1px solid #151e2e; }}
.data-table tr:last-child td {{ border-bottom: none; }}
.data-table tr:hover td {{ background: rgba(96,165,250,.04); }}

/* ── Utility ── */
.pos    {{ color: var(--pos); }}
.neg    {{ color: var(--neg); }}
.muted  {{ color: var(--muted); }}
.fw     {{ font-weight: 700; }}
.center {{ text-align: center; }}
.empty  {{ color: var(--muted); font-style: italic; padding: 10px 0; font-size: .85rem; }}
.sym    {{ font-weight: 700; letter-spacing: .3px; }}

/* ── Badges ── */
.badge {{
  display: inline-block; padding: 2px 9px; border-radius: 99px;
  font-size: .7rem; font-weight: 700; letter-spacing: .5px;
  white-space: nowrap;
}}
.bg-sell  {{ background: rgba(239,68,68,.18);  color: #f87171; border: 1px solid rgba(239,68,68,.3); }}
.bg-avg   {{ background: rgba(251,191,36,.15); color: #fbbf24; border: 1px solid rgba(251,191,36,.3); }}
.bg-atcap {{ background: rgba(100,116,139,.15);color: #94a3b8; border: 1px solid rgba(100,116,139,.3); }}
.bg-hold  {{ background: rgba(52,211,153,.1);  color: #6ee7b7; border: 1px solid rgba(52,211,153,.2); }}

/* ── Progress bar ── */
.prog-wrap {{
  height: 5px; background: #1e293b; border-radius: 99px;
  overflow: hidden; margin-bottom: 3px; width: 80px;
}}
.prog-bar {{
  height: 100%; border-radius: 99px;
  background: linear-gradient(90deg, #3b82f6, #22c55e);
  transition: width .3s;
}}

/* ── Heatmap ── */
.heatmap {{
  display: flex; flex-wrap: wrap; gap: 7px; padding-top: 8px;
}}
.hm-cell {{
  border-radius: 7px; padding: 8px 10px; min-width: 72px;
  text-align: center; cursor: default; transition: transform .15s;
  border: 1px solid rgba(255,255,255,.06);
}}
.hm-cell:hover {{ transform: scale(1.05); z-index: 2; }}
.hm-sym  {{ font-weight: 700; font-size: .78rem; letter-spacing: .3px; }}
.hm-pct  {{ font-size: .82rem; font-weight: 700; margin-top: 2px; }}
.hm-icon {{ font-size: .65rem; margin-top: 3px; opacity: .8; }}
.hm-legend {{
  display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 10px;
}}
.hm-leg {{
  font-size: .68rem; padding: 2px 8px; border-radius: 99px;
  font-weight: 600; letter-spacing: .3px;
}}

/* ── Alpha card ── */
.alpha-stats {{ display: flex; flex-direction: column; gap: 7px; padding: 4px 0; }}
.alpha-row {{
  display: flex; justify-content: space-between; align-items: center;
  font-size: .85rem; padding: 5px 10px;
  background: rgba(255,255,255,.03); border-radius: 6px;
}}
.alpha-row span:last-child {{ font-weight: 700; }}
.alpha-big {{ background: rgba(255,255,255,.06) !important; font-size: .95rem; }}
.alpha-since {{ font-size: .72rem; color: var(--muted); text-align: right; margin-top: 2px; }}

@media (max-width: 768px) {{
  .charts-row {{ flex-direction: column; }}
  .kpi-card   {{ min-width: 130px; }}
  .topbar     {{ flex-wrap: wrap; gap: 10px; }}
}}
</style>
</head>
<body>

<div class="topbar">
  <span class="topbar-logo">⚡ WealthOS</span>
  <span class="topbar-pnl">
    Total deployed <b>₹{total_deployed/1e5:.1f}L</b>
    &nbsp;·&nbsp;
    Overall P&L <b style="color:{port_color}">{'+' if total_pnl>=0 else ''}{fmt_inr(total_pnl)} ({total_pnl_pct:+.1f}%)</b>
  </span>
  <span class="topbar-ts">⟳ Generated {ts}</span>
</div>

<div class="regime" style="background:rgba({','.join(str(int(rcolor.lstrip('#')[i:i+2],16)) for i in (0,2,4))},0.06)">
  <span class="regime-pill" style="background:{rcolor}22;color:{rcolor};border:1px solid {rcolor}44">
    {regime['label']}
  </span>
  <span class="regime-stat">NIFTY <b>₹{regime['price']:,.0f}</b></span>
  <span class="regime-stat">vs 200DMA <b style="color:{'var(--pos)' if regime['vs200']>=0 else 'var(--neg)'}">{regime['vs200']:+.1f}%</b></span>
  <span class="regime-stat">50-200 gap <b>{regime['gap5020']:+.1f}%</b></span>
  <span class="regime-stat" style="color:{rcolor}">{r_confirm}</span>
</div>

<div class="main">

  {render_action_card(data["ns50_rows"], data["nsmid_rows"])}

  <!-- Portfolio Summary -->
  <div class="section">
    <div class="section-title">Portfolio Summary</div>
    <div class="kpi-row">
      {kpi_card("Total Deployed",  fmt_inr(total_deployed), icon="💼")}
      {kpi_card("Current Value",   fmt_inr(total_value),    icon="📊")}
      {kpi_card("Unrealized P&L",  fmt_inr(total_unreal),
                f'{(total_unreal/total_deployed*100 if total_deployed else 0):+.2f}%',
                "#22c55e" if total_unreal >= 0 else "#ef4444", "📈")}
      {kpi_card("Realized P&L",    fmt_inr(total_realized),
                "NiftyShop closed trades",
                "#22c55e" if total_realized >= 0 else "#ef4444", "✅")}
      {kpi_card("Overall P&L",     fmt_inr(total_pnl),
                f'{total_pnl_pct:+.2f}% on capital deployed',
                port_color, "⭐")}
    </div>
  </div>

  <!-- Insights row -->
  <div class="charts-row">
    <div class="chart-card" style="max-width:260px">
      <div class="chart-title">Allocation by Strategy</div>
      <canvas id="allocChart" height="200"></canvas>
    </div>
    <div class="chart-card" style="max-width:320px">
      <div class="chart-title">Portfolio vs Nifty 50 &nbsp;<span style="color:var(--muted);font-size:.7rem">(since first buy)</span></div>
      {alpha_card_html}
      {bench_canvas_html}
    </div>
    <div class="chart-card" style="flex:2">
      <div class="chart-title">Position Heatmap &nbsp;<span style="color:var(--muted);font-size:.7rem">hover for details · 🔵 N50Mom · 🟣 MidMom · 🟢 NSop50 · 🟡 NSmid</span></div>
      {heatmap_html}
    </div>
  </div>

  <!-- Nifty 50 Momentum -->
  <div class="section">
    <div class="section-title">
      🔵 Nifty 50 Momentum
      <span class="section-sub">monthly rebalance · ₹9L sleeve · 15 positions</span>
    </div>
    {section_kpis(data["n50_summ"])}
    {render_momentum_table(data["n50_rows"], NIFTY50_PER_POS)}
  </div>

  <!-- Midcap 50 Momentum -->
  <div class="section">
    <div class="section-title">
      🟣 Midcap 50 Momentum
      <span class="section-sub">monthly rebalance · ₹2L sleeve · 10 positions · paper trade phase</span>
    </div>
    {section_kpis(data["mid_summ"])}
    {render_momentum_table(data["mid_rows"], MIDCAP_PER_POS)}
  </div>

  <!-- NiftyShop -->
  <div class="section">
    <div class="section-title">
      🟢 NiftyShop — Nifty 50
      <span class="section-sub">weekly · ₹4L · mean-reversion · exit at +8%</span>
    </div>
    {section_kpis(data["ns50_summ"], data["ns50_kpis"])}
    <div class="sub-heading">Open Positions</div>
    {render_niftyshop_open_table(data["ns50_rows"])}
    <div class="sub-heading" style="margin-top:22px">Closed Trades</div>
    {render_closed_table(data["ns50_closed"])}
  </div>

  <!-- MidcapShop -->
  <div class="section">
    <div class="section-title">
      🟡 MidcapShop — Midcap 50
      <span class="section-sub">weekly · ₹2L · mean-reversion · paper trade phase</span>
    </div>
    {section_kpis(data["nsmid_summ"], data["nsmid_kpis"])}
    <div class="sub-heading">Open Positions</div>
    {render_niftyshop_open_table(data["nsmid_rows"])}
    <div class="sub-heading" style="margin-top:22px">Closed Trades</div>
    {render_closed_table(data["nsmid_closed"])}
  </div>

</div>

<script>
const darkTick = {{ color: '#64748b', font: {{ size: 10 }} }};
const darkGrid = {{ color: '#1f2d45' }};

new Chart(document.getElementById('allocChart'), {{
  type: 'doughnut',
  data: {{
    labels: {json.dumps(alloc_labels)},
    datasets: [{{
      data: {json.dumps(alloc_values)},
      backgroundColor: ['#3b82f6','#8b5cf6','#22c55e','#f59e0b'],
      borderWidth: 2, borderColor: '#0a0e1a',
      hoverBorderColor: '#fff',
    }}]
  }},
  options: {{
    plugins: {{
      legend: {{ position:'bottom', labels:{{ color:'#94a3b8', font:{{size:11}}, padding:14 }} }},
      tooltip: {{ callbacks: {{ label: ctx => ' ₹' + (ctx.raw/1e5).toFixed(2) + 'L  (' + Math.round(ctx.raw/{max(total_deployed,1)}*100) + '%)' }} }}
    }},
    cutout: '64%',
  }}
}});

{bench_js}
</script>
</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="WealthOS Portfolio Dashboard Generator")
    p.add_argument("--open", action="store_true", help="Auto-open in browser after generating")
    p.add_argument("--out",  default="dashboard.html", help="Output file (default: dashboard.html)")
    a = p.parse_args()

    print(f"\nWealthOS Dashboard  —  {datetime.now().strftime('%d %b %Y  %H:%M')}\n")

    print("  Loading holdings and trade logs...")
    n50_hold  = load_holdings(NIFTY_HOLDINGS)
    mid_hold  = load_holdings(MIDCAP_HOLDINGS)
    ns50_log  = load_trade_log(NIFTY_LOG)
    nsmid_log = load_trade_log(MIDCAP_LOG)

    ns50_open    = build_open_positions(ns50_log)
    nsmid_open   = build_open_positions(nsmid_log)
    ns50_closed  = get_closed_trades(ns50_log)
    nsmid_closed = get_closed_trades(nsmid_log)

    syms_ns = list({h["symbol"] + ".NS" for h in n50_hold + mid_hold} |
                   {p["symbol"] + ".NS" for p in ns50_open + nsmid_open})

    print(f"  Fetching live prices for {len(syms_ns)} symbols...")
    prices = fetch_prices(syms_ns)

    print("  Computing KPIs...")
    regime    = get_regime()
    n50_rows  = build_momentum_rows(n50_hold, prices, NIFTY50_PER_POS)
    mid_rows  = build_momentum_rows(mid_hold, prices, MIDCAP_PER_POS)
    ns50_rows = build_niftyshop_rows(ns50_open,  prices)
    nsmid_rows= build_niftyshop_rows(nsmid_open, prices)

    all_rows = (
        [{**r, "strat": "n50"}   for r in n50_rows]  +
        [{**r, "strat": "mid"}   for r in mid_rows]  +
        [{**r, "strat": "ns50"}  for r in ns50_rows] +
        [{**r, "strat": "nsmid"} for r in nsmid_rows]
    )
    print("  Fetching Nifty 50 benchmark (for alpha comparison)...")
    bench = compute_benchmark(all_rows)

    data = {
        "regime":       regime,
        "bench":        bench,
        "n50_rows":     n50_rows,    "mid_rows":     mid_rows,
        "ns50_rows":    ns50_rows,   "nsmid_rows":   nsmid_rows,
        "ns50_closed":  ns50_closed, "nsmid_closed": nsmid_closed,
        "n50_summ":     strategy_summary(n50_rows),
        "mid_summ":     strategy_summary(mid_rows),
        "ns50_summ":    strategy_summary(ns50_rows),
        "nsmid_summ":   strategy_summary(nsmid_rows),
        "ns50_kpis":    niftyshop_kpis(ns50_closed),
        "nsmid_kpis":   niftyshop_kpis(nsmid_closed),
    }

    print("  Generating HTML...")
    out = Path(a.out)
    out.write_text(generate_html(data), encoding="utf-8")
    print(f"\n  ✅  Dashboard → {out.resolve()}")

    if a.open:
        import webbrowser
        webbrowser.open(out.resolve().as_uri())
        print("  Opened in browser.")
    else:
        print("  Tip: run with --open to auto-launch in browser.\n")


if __name__ == "__main__":
    main()
