"""
WealthOS — Shared Configuration
================================
Single source of truth for universes, sector maps, and strategy parameters.
All four deployers import from here.

When the Nifty 50 or Midcap 50 index rebalances (stocks added/removed),
update ONLY this file. The deployers read from it automatically.

Update CLAUDE.md too if sleeve sizes change.
"""

# ── Nifty 50 universe ─────────────────────────────────────────────────────────

# List form — used by momentum deployers (they iterate and append .NS)
NIFTY50_LIST: list[str] = [
    "RELIANCE","TCS","HDFCBANK","BHARTIARTL","ICICIBANK","INFY","SBIN","HINDUNILVR",
    "ITC","LT","KOTAKBANK","BAJFINANCE","HCLTECH","AXISBANK","ASIANPAINT","MARUTI",
    "TITAN","SUNPHARMA","ULTRACEMCO","NTPC","POWERGRID","NESTLEIND","WIPRO","ONGC",
    "JSWSTEEL","TMPV","ADANIENT","COALINDIA","ETERNAL","BAJAJFINSV",
    "TATASTEEL","TECHM","HDFCLIFE","INDIGO","DRREDDY","CIPLA","GRASIM","APOLLOHOSP",
    "ADANIPORTS","TRENT","BEL","SHRIRAMFIN","BAJAJ-AUTO","EICHERMOT","M&M",
    "TATACONSUM","MAXHEALTH","JIOFIN","HINDALCO","SBILIFE",
]

# Dict form — used by NiftyShop deployer (symbol → display name)
NIFTY50_NAMES: dict[str, str] = {
    "RELIANCE":"Reliance Industries","HDFCBANK":"HDFC Bank",
    "BHARTIARTL":"Bharti Airtel","TCS":"Tata Consultancy Services",
    "ICICIBANK":"ICICI Bank","SBIN":"State Bank of India",
    "HINDUNILVR":"Hindustan Unilever","INFY":"Infosys",
    "BAJFINANCE":"Bajaj Finance","ITC":"ITC","LT":"Larsen & Toubro",
    "MARUTI":"Maruti Suzuki","M&M":"Mahindra & Mahindra",
    "HCLTECH":"HCL Technologies","KOTAKBANK":"Kotak Mahindra Bank",
    "SUNPHARMA":"Sun Pharmaceutical","ULTRACEMCO":"UltraTech Cement",
    "TITAN":"Titan Company","AXISBANK":"Axis Bank","NTPC":"NTPC",
    "BAJAJFINSV":"Bajaj Finserv","ONGC":"Oil & Natural Gas Corp",
    "ADANIPORTS":"Adani Ports","BEL":"Bharat Electronics",
    "POWERGRID":"Power Grid Corporation","COALINDIA":"Coal India",
    "NESTLEIND":"Nestle India","APOLLOHOSP":"Apollo Hospitals",
    "ETERNAL":"Zomato","CIPLA":"Cipla",
    "TECHM":"Tech Mahindra","TATACONSUM":"Tata Consumer Products",
    "JSWSTEEL":"JSW Steel","INDIGO":"Indigo",
    "ASIANPAINT":"Asian Paints","HINDALCO":"Hindalco Industries",
    "EICHERMOT":"Eicher Motors","SBILIFE":"SBI Life Insurance",
    "JIOFIN":"Jio Financials","DRREDDY":"Dr. Reddy's",
    "BAJAJ-AUTO":"Bajaj Auto","GRASIM":"Grasim Industries",
    "TATASTEEL":"Tata Steel","WIPRO":"Wipro","TRENT":"Trent",
    "ADANIENT":"Adani Enterprises","SHRIRAMFIN":"Shriram Finance",
    "MAXHEALTH":"MAX Health","HDFCLIFE":"HDFCLIFE","TMPV":"Tata Motors",
}

# Sector map — used by Nifty 50 momentum deployer for sector cap enforcement
NIFTY50_SECTOR_MAP: dict[str, str] = {
    "HDFCBANK":"BANKING","ICICIBANK":"BANKING","SBIN":"BANKING","AXISBANK":"BANKING",
    "KOTAKBANK":"BANKING","INDIGO":"Aviation","BAJAJFINSV":"NBFC","BAJFINANCE":"NBFC",
    "SHRIRAMFIN":"NBFC","HDFCLIFE":"INSURANCE","SBILIFE":"INSURANCE",
    "TCS":"IT","INFY":"IT","HCLTECH":"IT","WIPRO":"IT","TECHM":"IT",
    "RELIANCE":"ENERGY","ONGC":"ENERGY",
    "NTPC":"POWER","POWERGRID":"POWER",
    "HINDUNILVR":"FMCG","ITC":"FMCG","NESTLEIND":"FMCG",
    "MARUTI":"AUTO","TMPV":"AUTO","M&M":"AUTO","BAJAJ-AUTO":"AUTO",
    "EICHERMOT":"AUTO","TATACONSUM":"FMCG",
    "LT":"INFRA","JIOFIN":"NBFC",
    "BEL":"DEFENCE","MAXHEALTH":"PHARMA",
    "TITAN":"CONSUMER","TRENT":"RETAIL",
    "ASIANPAINT":"PAINTS",
    "ULTRACEMCO":"CEMENT","GRASIM":"CEMENT",
    "JSWSTEEL":"METALS","TATASTEEL":"METALS","HINDALCO":"METALS","COALINDIA":"METALS",
    "SUNPHARMA":"PHARMA","DRREDDY":"PHARMA","ETERNAL":"E-commerce","CIPLA":"PHARMA",
    "APOLLOHOSP":"HEALTHCARE",
    "ADANIENT":"CONGLOMERATE","ADANIPORTS":"PORTS",
    "BHARTIARTL":"TELECOM",
}

# ── Nifty Midcap 50 universe ──────────────────────────────────────────────────

# List form — used by midcap momentum deployer
MIDCAP50_LIST: list[str] = [
    "BSE","BHEL","POLYCAB","LUPIN","INDUSTOWER","MARICO","GMRAIRPORTS","HINDPETRO",
    "NHPC","DABUR","SRF","PERSISTENT","HAVELLS","POLICYBZR","NMDC","MCX",
    "FEDERALBNK","SUZLON","AUBANK","IDFCFIRSTB","OBEROIRLTY","MFSL","ABCAPITAL",
    "BALKRISIND","TATACHEM","CESC","PIIND","ZYDUSLIFE","KAJARIACER","VOLTAS",
    "SUNDARMFIN","ASTRAL","CONCOR","GLENMARK","GODREJPROP","INDHOTEL","JKCEMENT",
    "LTTS","MRF","PAGEIND","PHOENIXLTD","PRESTIGE","RAMCOCEM","SYNGENE",
    "TATACOMM","TORNTPHARM","ABFRL","CHAMBLFERT","CRISIL","METROPOLIS",
]

# Dict form — used by MidcapShop deployer (symbol → display name)
MIDCAP50_NAMES: dict[str, str] = {
    "BSE":"BSE Ltd","BHEL":"Bharat Heavy Electricals","POLYCAB":"Polycab India",
    "LUPIN":"Lupin","INDUSTOWER":"Indus Towers","MARICO":"Marico",
    "GMRAIRPORTS":"GMR Airports Infrastructure","HINDPETRO":"Hindustan Petroleum",
    "NHPC":"NHPC","DABUR":"Dabur India","SRF":"SRF Ltd",
    "PERSISTENT":"Persistent Systems","HAVELLS":"Havells India",
    "POLICYBZR":"PB Fintech (PolicyBazaar)","NMDC":"NMDC","MCX":"MCX",
    "FEDERALBNK":"Federal Bank","SUZLON":"Suzlon Energy",
    "AUBANK":"AU Small Finance Bank","IDFCFIRSTB":"IDFC First Bank",
    "OBEROIRLTY":"Oberoi Realty","MFSL":"Max Financial Services",
    "ABCAPITAL":"Aditya Birla Capital","BALKRISIND":"Balkrishna Industries",
    "TATACHEM":"Tata Chemicals","CESC":"CESC","PIIND":"PI Industries",
    "ZYDUSLIFE":"Zydus Lifesciences","KAJARIACER":"Kajaria Ceramics",
    "VOLTAS":"Voltas","SUNDARMFIN":"Sundaram Finance","ASTRAL":"Astral Ltd",
    "CONCOR":"Container Corporation","GLENMARK":"Glenmark Pharma",
    "GODREJPROP":"Godrej Properties","INDHOTEL":"Indian Hotels",
    "JKCEMENT":"JK Cement","LTTS":"L&T Technology Services","MRF":"MRF",
    "PAGEIND":"Page Industries","PHOENIXLTD":"Phoenix Mills",
    "PRESTIGE":"Prestige Estates","RAMCOCEM":"Ramco Cements",
    "SYNGENE":"Syngene International","TATACOMM":"Tata Communications",
    "TORNTPHARM":"Torrent Pharmaceuticals","ABFRL":"Aditya Birla Fashion",
    "CHAMBLFERT":"Chambal Fertilisers","CRISIL":"CRISIL",
    "METROPOLIS":"Metropolis Healthcare",
}

# Sector map — used by Midcap 50 momentum deployer for sector cap enforcement
MIDCAP50_SECTOR_MAP: dict[str, str] = {
    "FEDERALBNK":"BANKING","AUBANK":"BANKING","IDFCFIRSTB":"BANKING",
    "BSE":"FINANCE","MCX":"FINANCE","CRISIL":"FINANCE","SUNDARMFIN":"NBFC",
    "ABCAPITAL":"NBFC","MFSL":"INSURANCE","POLICYBZR":"FINTECH",
    "PERSISTENT":"IT","LTTS":"IT",
    "LUPIN":"PHARMA","ZYDUSLIFE":"PHARMA","GLENMARK":"PHARMA",
    "TORNTPHARM":"PHARMA","SYNGENE":"PHARMA","METROPOLIS":"HEALTHCARE",
    "MARICO":"FMCG","DABUR":"FMCG",
    "MRF":"AUTO_ANCIL","BALKRISIND":"AUTO_ANCIL",
    "BHEL":"CAPITAL_GOODS","POLYCAB":"CAPITAL_GOODS",
    "HAVELLS":"CONSUMER_ELEC","VOLTAS":"CONSUMER_ELEC","ASTRAL":"CONSUMER",
    "NHPC":"POWER","SUZLON":"POWER","CESC":"POWER","HINDPETRO":"ENERGY",
    "SRF":"CHEMICALS","TATACHEM":"CHEMICALS","PIIND":"CHEMICALS","CHAMBLFERT":"CHEMICALS",
    "NMDC":"METALS",
    "GMRAIRPORTS":"INFRA","CONCOR":"LOGISTICS",
    "INDUSTOWER":"TELECOM","TATACOMM":"TELECOM",
    "OBEROIRLTY":"REALTY","GODREJPROP":"REALTY","PHOENIXLTD":"REALTY","PRESTIGE":"REALTY",
    "PAGEIND":"CONSUMER","KAJARIACER":"CONSUMER","ABFRL":"RETAIL",
    "JKCEMENT":"CEMENT","RAMCOCEM":"CEMENT",
    "INDHOTEL":"HOSPITALITY",
}

# ── Benchmarks ────────────────────────────────────────────────────────────────
NIFTY50_BENCHMARK  = "^NSEI"
MIDCAP50_BENCHMARK = "^NSMIDCP"

# ── Momentum strategy parameters ──────────────────────────────────────────────
# DO NOT CHANGE without re-running the full backtest chain.
MOMENTUM_TOP_N_NIFTY50    = 15
MOMENTUM_TOP_N_MIDCAP50   = 10
MOMENTUM_SECTOR_CAP       = 3
MOMENTUM_EXIT_BUFFER      = 15.0
MOMENTUM_MIN_SCORE        = 40.0
MOMENTUM_MIN_HISTORY      = 210      # trading days
MOMENTUM_HARD_STOP        = 0.85    # exit if price < entry_price * this
MOMENTUM_LIQUIDITY_NIFTY  = 5.0     # ₹Cr/day minimum
MOMENTUM_LIQUIDITY_MIDCAP = 2.0     # ₹Cr/day minimum (midcaps less liquid)

# ── Sleeve sizes ──────────────────────────────────────────────────────────────
# Change here when you adjust capital allocation. Also update CLAUDE.md.
SLEEVE_NIFTY50_MOMENTUM  = 900_000  # ₹9L
SLEEVE_MIDCAP_MOMENTUM   = 200_000  # ₹2L
SLEEVE_NIFTY50_NIFTYSHOP = 400_000  # ₹4L (NiftyShop default capital)
SLEEVE_MIDCAP_NIFTYSHOP  = 200_000  # ₹2L
