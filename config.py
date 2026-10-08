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
# Synced to NSE's published constituent list (post Sept-2026 semi-annual review):
#   https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv
# NSE reviews in March and September — re-sync this block (LIST, NAMES, SECTOR_MAP) then.
NIFTY50_LIST: list[str] = [
    "RELIANCE","TCS","HDFCBANK","BHARTIARTL","ICICIBANK","INFY","SBIN","HINDUNILVR",
    "ITC","LT","KOTAKBANK","BAJFINANCE","HCLTECH","AXISBANK","ASIANPAINT","MARUTI",
    "TITAN","SUNPHARMA","ULTRACEMCO","NTPC","POWERGRID","NESTLEIND","BSE","ONGC",
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
    "ETERNAL":"Eternal (Zomato)","CIPLA":"Cipla",
    "TECHM":"Tech Mahindra","TATACONSUM":"Tata Consumer Products",
    "JSWSTEEL":"JSW Steel","INDIGO":"InterGlobe Aviation (IndiGo)",
    "ASIANPAINT":"Asian Paints","HINDALCO":"Hindalco Industries",
    "EICHERMOT":"Eicher Motors","SBILIFE":"SBI Life Insurance",
    "JIOFIN":"Jio Financials","DRREDDY":"Dr. Reddy's",
    "BAJAJ-AUTO":"Bajaj Auto","GRASIM":"Grasim Industries",
    "TATASTEEL":"Tata Steel","BSE":"BSE Ltd","TRENT":"Trent",
    "ADANIENT":"Adani Enterprises","SHRIRAMFIN":"Shriram Finance",
    "MAXHEALTH":"MAX Health","HDFCLIFE":"HDFC Life Insurance",
    "TMPV":"Tata Motors Passenger Vehicles",
}

# Sector map — used by Nifty 50 momentum deployer for sector cap enforcement
NIFTY50_SECTOR_MAP: dict[str, str] = {
    "HDFCBANK":"BANKING","ICICIBANK":"BANKING","SBIN":"BANKING","AXISBANK":"BANKING",
    "KOTAKBANK":"BANKING","INDIGO":"Aviation","BAJAJFINSV":"NBFC","BAJFINANCE":"NBFC",
    "SHRIRAMFIN":"NBFC","HDFCLIFE":"INSURANCE","SBILIFE":"INSURANCE",
    "TCS":"IT","INFY":"IT","HCLTECH":"IT","TECHM":"IT","BSE":"FINANCE",
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
# Synced to NSE's published constituent list (post Sept-2026 semi-annual review):
#   https://nsearchives.nseindia.com/content/indices/ind_niftymidcap50list.csv
# NSE reviews in March and September — re-sync this block (LIST, NAMES, SECTOR_MAP) then.
MIDCAP50_LIST: list[str] = [
    "APLAPOLLO","AUBANK","ASHOKLEY","AUROPHARMA","BHARATFORG","BHEL","DABUR","DIXON",
    "NYKAA","FEDERALBNK","FORTIS","GVT&D","GMRAIRPORT","GLENMARK","GODREJPROP","HAVELLS",
    "HEROMOTOCO","HINDPETRO","ICICIGI","IDFCFIRSTB","INDHOTEL","INDUSTOWER","INDUSINDBK",
    "NAUKRI","JSWENERGY","LAURUSLABS","LUPIN","MANKIND","MARICO","MFSL","MCX","NHPC",
    "NMDC","NATIONALUM","OIL","PAYTM","POLICYBZR","PERSISTENT","PHOENIXLTD","PRESTIGE",
    "RECLTD","SRF","SUZLON","SWIGGY","TIINDIA","UPL","UNITDSPR","VMM","WAAREEENER","YESBANK",
]

# Dict form — used by MidcapShop deployer (symbol → display name)
MIDCAP50_NAMES: dict[str, str] = {
    "APLAPOLLO":"APL Apollo Tubes","AUBANK":"AU Small Finance Bank",
    "ASHOKLEY":"Ashok Leyland","AUROPHARMA":"Aurobindo Pharma",
    "BHARATFORG":"Bharat Forge","BHEL":"Bharat Heavy Electricals",
    "DABUR":"Dabur India","DIXON":"Dixon Technologies","NYKAA":"Nykaa (FSN E-Commerce)",
    "FEDERALBNK":"Federal Bank","FORTIS":"Fortis Healthcare",
    "GVT&D":"GE Vernova T&D India","GMRAIRPORT":"GMR Airports",
    "GLENMARK":"Glenmark Pharma","GODREJPROP":"Godrej Properties",
    "HAVELLS":"Havells India","HEROMOTOCO":"Hero MotoCorp",
    "HINDPETRO":"Hindustan Petroleum","ICICIGI":"ICICI Lombard",
    "IDFCFIRSTB":"IDFC First Bank","INDHOTEL":"Indian Hotels",
    "INDUSTOWER":"Indus Towers","INDUSINDBK":"IndusInd Bank",
    "NAUKRI":"Info Edge (Naukri)","JSWENERGY":"JSW Energy",
    "LAURUSLABS":"Laurus Labs","LUPIN":"Lupin","MANKIND":"Mankind Pharma",
    "MARICO":"Marico","MFSL":"Max Financial Services","MCX":"MCX","NHPC":"NHPC",
    "NMDC":"NMDC","NATIONALUM":"National Aluminium (NALCO)","OIL":"Oil India",
    "PAYTM":"Paytm (One 97 Communications)","POLICYBZR":"PB Fintech (PolicyBazaar)",
    "PERSISTENT":"Persistent Systems","PHOENIXLTD":"Phoenix Mills",
    "PRESTIGE":"Prestige Estates","RECLTD":"REC Ltd","SRF":"SRF Ltd",
    "SUZLON":"Suzlon Energy","SWIGGY":"Swiggy","TIINDIA":"Tube Investments of India",
    "UPL":"UPL Ltd","UNITDSPR":"United Spirits","VMM":"Vishal Mega Mart",
    "WAAREEENER":"Waaree Energies","YESBANK":"Yes Bank",
}

# Sector map — used by Midcap 50 momentum deployer for sector cap enforcement
MIDCAP50_SECTOR_MAP: dict[str, str] = {
    "FEDERALBNK":"BANKING","AUBANK":"BANKING","IDFCFIRSTB":"BANKING",
    "INDUSINDBK":"BANKING","YESBANK":"BANKING",
    "MCX":"FINANCE","RECLTD":"NBFC","MFSL":"INSURANCE","ICICIGI":"INSURANCE",
    "POLICYBZR":"FINTECH","PAYTM":"FINTECH",
    "PERSISTENT":"IT",
    "LUPIN":"PHARMA","GLENMARK":"PHARMA","AUROPHARMA":"PHARMA",
    "LAURUSLABS":"PHARMA","MANKIND":"PHARMA","FORTIS":"HEALTHCARE",
    "MARICO":"FMCG","DABUR":"FMCG","UNITDSPR":"FMCG",
    "ASHOKLEY":"AUTO","HEROMOTOCO":"AUTO",
    "BHARATFORG":"AUTO_ANCIL","TIINDIA":"AUTO_ANCIL",
    "BHEL":"CAPITAL_GOODS","GVT&D":"CAPITAL_GOODS","WAAREEENER":"CAPITAL_GOODS",
    "HAVELLS":"CONSUMER_ELEC","DIXON":"CONSUMER_ELEC",
    "NHPC":"POWER","SUZLON":"POWER","JSWENERGY":"POWER",
    "HINDPETRO":"ENERGY","OIL":"ENERGY",
    "SRF":"CHEMICALS","UPL":"CHEMICALS",
    "NMDC":"METALS","NATIONALUM":"METALS","APLAPOLLO":"METALS",
    "GMRAIRPORT":"INFRA","INDUSTOWER":"TELECOM",
    "GODREJPROP":"REALTY","PHOENIXLTD":"REALTY","PRESTIGE":"REALTY",
    "INDHOTEL":"HOSPITALITY",
    "SWIGGY":"INTERNET","NYKAA":"INTERNET","NAUKRI":"INTERNET","VMM":"RETAIL",
}

# ── Benchmarks ────────────────────────────────────────────────────────────────
NIFTY50_BENCHMARK  = "^NSEI"
MIDCAP50_BENCHMARK = "^NSEMDCP50"   # Nifty Midcap 50 (NOT ^NSMIDCP — on Yahoo that is Nifty Next 50)

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
