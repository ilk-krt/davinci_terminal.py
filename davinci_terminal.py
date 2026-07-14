import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import graphviz
import yfinance as yf
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import calendar
import time
import re
import gc

# ==========================================
# 0. AYARLAR & AGRESİF DARK MODE CSS
# ==========================================
st.set_page_config(layout="wide", page_title="AETHER APEX ULTIMATE V134.0", page_icon="🏛️")

st.markdown("""
    <style>
    .stApp { background-color: #050505 !important; color: #e0e0e0 !important; }
    p, h1, h2, h3, h4, h5, h6, span, label, div { color: #e0e0e0 !important; }
    div[data-baseweb="select"] > div { background-color: #111111 !important; color: #ffffff !important; border: 1px solid #00ff88 !important; }
    div[data-baseweb="popover"] > div { background-color: #111111 !important; border: 1px solid #444444 !important; }
    ul[role="listbox"] { background-color: #111111 !important; padding: 0px !important; }
    ul[role="listbox"] li { color: #ffffff !important; background-color: #111111 !important; padding: 10px !important; border-bottom: 1px solid #222222 !important; }
    ul[role="listbox"] li:hover { background-color: #222222 !important; color: #00ff88 !important; font-weight: bold !important; }
    [data-testid="stTable"], [data-testid="stDataFrame"] { background-color: #111111 !important; }
    th { background-color: #222222 !important; color: #00ff88 !important; border-bottom: 1px solid #444 !important; }
    td { border-bottom: 1px solid #333 !important; color: #ffffff !important; }
    [data-testid="stExpander"] { background-color: #111111 !important; border: 1px solid #333 !important; border-radius: 8px !important; border-left: 4px solid #00ff88 !important; }
    [data-testid="stExpander"] summary p { color: #00ff88 !important; font-weight: bold !important; font-size: 1.1rem !important; }
    div.stButton > button { background-color: #1a1a1a !important; color: #ffffff !important; border: 1px solid #00ff88 !important; border-radius: 8px !important; font-weight: bold; }
    div.stButton > button:hover { background-color: #00ff88 !important; color: #000000 !important; }
    .valuation-gap-card { background: linear-gradient(145deg, #111 0%, #0a0a0a 100%); padding: 20px; border-radius: 12px; border: 1px solid #333; margin-bottom: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.5); }
    .valuation-leader { color: #00ff88; font-weight: 900; font-size: 1.2rem; }
    .valuation-laggard { color: #f1c40f; font-weight: bold; }
    .macro-def-box { background-color: #111; border-left: 4px solid #FF1744; padding: 15px; border-radius: 5px; margin-bottom: 15px; }
    .macro-def-title { color: #00ff88; font-weight: bold; font-size: 1.1rem; margin-bottom: 5px; }
    div[role="radiogroup"] > label { background-color: #111; padding: 10px 20px; border-radius: 8px; border: 1px solid #333; margin-right: 10px; cursor: pointer; transition: all 0.3s ease; }
    div[role="radiogroup"] > label:hover { border-color: #3b82f6; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. STATE & GÜNCELLEME KONTROLLERİ
# ==========================================
if 'active_trigger' not in st.session_state: st.session_state.active_trigger = "OPEX PINNING"
if 'macro_nonce' not in st.session_state: st.session_state.macro_nonce = str(time.time())
if 'battery_nonce' not in st.session_state: st.session_state.battery_nonce = str(time.time())
if 'val_nonce' not in st.session_state: st.session_state.val_nonce = str(time.time())
if 'news_nonce' not in st.session_state: st.session_state.news_nonce = str(time.time())

# Varsayılan Bilanço Hisse Listesi (Kullanıcının Verdiği Tekilleştirilmiş Liste)
DEFAULT_EARNINGS_TICKERS = sorted(list(set([
    "AAOI", "ABT", "ADBE", "ADEA", "AEHR", "AI", "ALAB", "AMAT", "AMD", "AMGN", "AMKR", "APLD", 
    "ARM", "ASTI", "ASTS", "ASX", "ATLX", "ATRO", "AVGO", "AXTI", "BA", "BABA", "BE", "BIDU", 
    "BKR", "BKSY", "BMNR", "BTDR", "BULL", "BZAI", "CEG", "CIEN", "CIFR", "CLPT", "CLSK", "COHR", 
    "CORZ", "CPRT", "CPSH", "CRDO", "CRM", "CRML", "CRSP", "CRWV", "DCO", "DELL", "DGXX", "DOCN", 
    "DT", "DVLT", "EMR", "EP", "EQIX", "ETN", "EXTR", "FCX", "FN", "FORM", "GDXJ", "GFS", "GLW", 
    "GRAB", "HEI", "HIMS", "HIMX", "HITI", "HON", "HPE", "IBM", "INTC", "IONQ", "IRDM", "IREN", 
    "ISRG", "IXIC", "KRMN", "KTOS", "LASR", "LEU", "LHX", "LITE", "LMT", "LPTH", "LRCX", "LUNR", 
    "LWLG", "MA", "MARA", "MBLY", "MDA", "META", "MP", "MRVL", "MSFT", "MTSI", "NBIS", "NEE", 
    "NOC", "NOVT", "NOW", "NTAP", "NTLA", "NVDA", "ONDS", "ONTO", "OPEN", "OPTX", "OUST", "P", 
    "PALL", "PCOR", "PFE", "PGY", "PL", "PLD", "PLTR", "POET", "PPLT", "PYPL", "QBTS", "QCLS", 
    "QCOM", "QUBT", "RDNT", "RDW", "RGTI", "RIOT", "RKLB", "RTX", "S", "SANM", "SATL", "SBET", 
    "SIDU", "SILJ", "SLNH", "SMCI", "SMR", "SMTC", "SNOW", "SOUN", "SPCE", "SPIR", "STLA", "STX", 
    "T", "TDOC", "TECK", "TER", "TLN", "TMUS", "TRUG", "TSEM", "TSM", "UFO", "UMC", "UUUU", "VECO", 
    "VIAV", "VOYG", "VSAT", "VST", "WBI", "WDC", "WOLF", "WULF", "YSS"
])))

if 'custom_earnings_list' not in st.session_state:
    st.session_state.custom_earnings_list = DEFAULT_EARNINGS_TICKERS

# ==========================================
# 2. KURUMSAL NİŞ ETF & HİSSE EVRENİ
# ==========================================
MAIN_SECTORS = {
    "XLK": "Ana Sektör: Teknoloji", "XLI": "Ana Sektör: Sanayi", "XLE": "Ana Sektör: Enerji",
    "XLV": "Ana Sektör: Sağlık", "XLF": "Ana Sektör: Finans", "XLY": "Ana Tüketim",
    "XLB": "Ana Sektör: Materyal", "XLC": "Ana Sektör: İletişim", "XLRE": "Ana Sektör: Gayrimenkul",
    "XLU": "Ana Sektör: Kamu Hizmetleri", "IGV": "Tema: Yazılım", "ARKG": "Tema: Genomik", 
    "CIBR": "Tema: Siber Güvenlik", "IBB": "Tema: Biyoteknoloji", "GDX": "Tema: Altın Madenciliği",
    "IBIT": "Tema: Bitcoin", "SMH": "Tema: Yarı İletkenler", "DRIV": "Tema: Akıllı Mobilite"
}

THEME_TRACKER_MAP = {
    "Software": ["IGV", "PSJ", "CLOU"],
    "Genomics": ["ARKG", "IDNA"],
    "HealthCare": ["XLV", "VHT"],
    "Cybersecurity": ["CIBR", "HACK", "BUG", "CYBER"],
    "Social Media": ["SOCL"],
    "Biotechnology": ["IBB", "XBI"],
    "Gold Miners": ["GDX", "GDXJ"],
    "Silver & Miners": ["SIL", "SLV"],
    "Medical": ["XLV"], 
    "Real Estate": ["VNQ", "XLRE", "REZ", "SRVR"],
    "Retail": ["XRT", "XLY"],
    "China Internet": ["KWEB"],
    "Bitcoin Miners": ["WGMI", "MARA", "RIOT", "CLSK", "IREN", "CORZ", "CIFR"],
    "Bitcoin": ["IBIT", "BITO"],
    "Airlines": ["JETS"],
    "Utilities": ["XLU", "URA"],
    "Home Construction": ["ITB", "XHB"],
    "Transports": ["IYT", "HULL"],
    "Telecom": ["XLC"],
    "Banks": ["XLF", "KRE"],
    "Aerospace & Defense": ["XAR", "ARKX", "UFO", "SPACE_RACE", "JEDI"],
    "Materials": ["XLB", "COPX", "LIT", "REMX", "XME"],
    "Oil & Gas": ["XLE", "XOP", "OIH"],
    "Industrials": ["XLI", "PAVE"],
    "Smart Mobility & EV": ["DRIV"],
    "AI": ["AIQ", "BOTZ"],
    "Robotics": ["ROBO", "BOTZ"],
    "Solar": ["TAN", "ICLN"],
    "Steel": ["SLX"],
    "Growth Stocks": ["VUG", "IWF"],
    "Quantum": ["QTUM", "IONQ", "RGTI", "QUBT"],
    "Semiconductors": ["SMH", "SOXX", "EUV", "PHOTON"]
}

ETF_INFO = {
    "XLU": {"area": "Utilities & Şebeke", "stocks": ["NEE", "SO", "DUK", "CEG", "AEP", "SRE", "VST"]},
    "PAVE": {"area": "Altyapı Yenileme", "stocks": ["ETN", "PH", "HUBB", "POWL", "TT", "CARR", "URI", "PWR"]},
    "XLK": {"area": "Teknoloji Devleri", "stocks": ["NVDA", "AAPL", "MSFT", "MU", "AVGO", "AMD", "PLTR"]},
    "IGV": {"area": "Yazılım ve SaaS", "stocks": ["MSFT", "CRM", "ORCL", "ADBE", "NOW", "PLTR", "CRWD", "NET"]},
    "SMH": {"area": "Global Çip Dökümhaneleri", "stocks": ["TSM", "INTC", "ASML", "NVDA", "AMD", "AVGO", "LRCX"]},
    "URA": {"area": "Uranyum ve Nükleer", "stocks": ["CCJ", "KAP", "NXE", "UEC", "UUUU", "SMR", "CEG"]},
    "WGMI": {"area": "Bitcoin Madenciliği", "stocks": ["MARA", "RIOT", "CLSK", "IREN", "WULF", "CORZ", "CIFR"]},
    "PHOTON": {"area": "Fotonik ve Optik", "stocks": ["IQE", "AXTI", "AAOI", "COHR", "LITE", "POET", "LRCX"]},
    "QUANT": {"area": "Kuantum Bilişim", "stocks": ["IONQ", "RGTI", "QUBT", "IBM", "GOOGL", "HON"]},
    "CYBER": {"area": "Global Siber Güvenlik", "stocks": ["CRWD", "PANW", "ZS", "FTNT", "OKTA", "S", "NET"]},
    "SPACE_RACE": {"area": "SpaceX & Uzay", "stocks": ["RKLB", "ASTS", "LUNR", "SATS", "PL", "SPIR", "BKSY", "SIDU"]},
    "TRUMP_PF": {"area": "Trump Portföyü", "stocks": ["MSTR", "MARA", "COIN", "TSLA", "PLTR", "GEO", "CXW"]},
    "COPX": {"area": "Bakır Üreticileri", "stocks": ["FCX", "SCCO", "BHP", "RIO", "TECK", "VALE"]}
}

FUTURE_THEMES_MAP = {
    "Chokepoint Çarpanları": ["NVDA", "AVGO", "CEG", "ETN", "EQIX", "FCX", "PLD"],
    "Agentic AI & Yazılım": ["NOW", "SOUN", "ADBE", "DT", "S", "EXTR"],
    "Uzay Bilişimi & Keşif": ETF_INFO["SPACE_RACE"]["stocks"],
    "Kuantum Bilişim (Quantum)": ETF_INFO["QUANT"]["stocks"],
    "Fotonik & Optik Çipler": ETF_INFO["PHOTON"]["stocks"],
    "Neocloud & Enerji Pivotu": ETF_INFO["WGMI"]["stocks"],
    "Nükleer & Temel Materyal": ["CEG", "TLN", "SMR", "NNE", "UUUU", "MP", "ATLX"]
}

# ==========================================
# 2.1 MAKRO DİNAMİK YARDIMCILARI (OPEX VB)
# ==========================================
def get_next_opex():
    today = datetime.now().date()
    c = calendar.Calendar(firstweekday=calendar.MONDAY)
    month_cal = c.monthdatescalendar(today.year, today.month)
    fridays = [d for week in month_cal for d in week if d.weekday() == calendar.FRIDAY and d.month == today.month]
    opex_date = fridays[2]
    
    if today > opex_date:
        next_month = today.month % 12 + 1
        next_year = today.year + (1 if today.month == 12 else 0)
        month_cal = c.monthdatescalendar(next_year, next_month)
        fridays = [d for week in month_cal for d in week if d.weekday() == calendar.FRIDAY and d.month == next_month]
        opex_date = fridays[2]
        
    days_left = (opex_date - today).days
    return opex_date.strftime("%d-%m-%Y"), days_left

opex_d, opex_dl = get_next_opex()

SYSTEM_TRIGGERS = {
    "GAMMA SQUEEZE": {
        "color": "#00ff88", 
        "battery": {"Stocks": 95, "Bonds": 20, "Crypto": 90, "Commodities": 55, "RealEstate": 65},
        "desc": "Tetikleyici: Beklenmedik güvercin FED açıklamaları, zayıf enflasyon verisi veya meme-hisse çılgınlığıyla piyasa yapıcıların (dealers) call opsiyon satıp delta-hedge için spot hisselere saldırması. Risk/Gerçeklik: Çok hızlı fiyat şişmesi yaratır ancak temel finansallara dayanmadığı için sert düzeltme olasılığı masadadır."
    },
    "OPEX PINNING": {
        "color": "#f1c40f", 
        "battery": {"Stocks": 50, "Bonds": 50, "Crypto": 48, "Commodities": 52, "RealEstate": 50},
        "desc": f"Tetikleyici: Sıradaki opsiyon vade sonuna (OpEx) sadece {opex_dl} gün kaldı ({opex_d}). Market Maker'lar primleri (theta) sıfırlamak için endeksi yüksek açık pozisyon (Open Interest) yoğunluğunun olduğu Max Pain noktasına hapsetmeye çalışıyor. Risk/Gerçeklik: Trend kırılımları bu süreçte çoğunlukla tuzak (whipsaw) çıkar, vade geçmeden pozisyon açmak tehlikelidir."
    },
    "GEOPOLITICAL SHOCK": {
        "color": "#ff3333", 
        "battery": {"Stocks": 25, "Bonds": 85, "Crypto": 35, "Commodities": 95, "RealEstate": 40},
        "desc": "Tetikleyici: Tayvan boğazı krizleri, gümrük tarifesi duyuruları, veya küresel enerji nakil hatlarına saldırılar. Sermaye; teknoloji ve riskten kaçıp altın, savunma, petrol ve hazine tahvillerine sığınır. Risk/Gerçeklik: Tedarik zincirine bağlı şirketleri anında ezer, enflasyonu hortlatma riski taşır."
    },
    "LIQUIDITY CRUNCH (FED)": {
        "color": "#9b59b6", 
        "battery": {"Stocks": 15, "Bonds": 90, "Crypto": 10, "Commodities": 35, "RealEstate": 25},
        "desc": "Tetikleyici: Enflasyonun inatçı çıkması, Hazine'nin devasa tahvil ihracı veya Reverse Repo (RRP) havuzunun kuruması. Piyasadaki dolar miktarı azalır. Risk/Gerçeklik: Yüksek F/K'ya sahip teknoloji, biyoteknoloji ve kripto varlıklarında acımasız likidasyon ve margin call döngüleri yaratır."
    }
}

# ==========================================
# 3. YARDIMCI GÖRSEL VE AĞ FONKSİYONLARI
# ==========================================
def draw_smart_money_flow(trigger_data):
    dot = graphviz.Digraph()
    dot.attr(bgcolor='#050505', rankdir='LR', ranksep='1.5', nodesep='0.8')
    dot.attr('node', fontsize='16', fontname='Arial', margin='0.2,0.1')
    dot.attr('edge', fontsize='14')
    with dot.subgraph(name='cluster_0') as c:
        c.attr(style='dashed', color='#555', label='Kaydi Varlıklar', fontcolor='#e0e0e0', fontsize='18')
        c.node("FIAT", "Fiat\nCurrency", shape='ellipse', style='filled', fillcolor='#4a148c', fontcolor='white')
        c.node("USD", "USD\n(Merkez)", shape='circle', style='filled', fillcolor='#0277bd', fontcolor='white')
        c.node("STOCK", "Borsalar", shape='box', style='filled', fillcolor='#f57f17', fontcolor='white')
        c.node("BOND", "Tahviller", shape='box', style='filled', fillcolor='#2e7d32', fontcolor='white')
        c.node("CRYPTO", "Kripto", shape='box', style='filled', fillcolor='#d81b60', fontcolor='white')
    with dot.subgraph(name='cluster_1') as c:
        c.attr(style='dashed', color='#555', label='Maddi Varlıklar', fontcolor='#e0e0e0', fontsize='18')
        c.node("COMM", "Emtia &\nEnerji", shape='circle', style='filled', fillcolor='#00695c', fontcolor='white')
        c.node("REAL", "Gayrimenkul", shape='box', style='filled', fillcolor='#827717', fontcolor='white')
    bat = trigger_data['battery']
    def get_pen(val): return str(max(2.0, val / 10))
    def get_col(val): return "#00ff88" if val >= 60 else "#ff3333" if val <= 40 else "#888"
    dot.edge("FIAT", "USD", color="#aaa", penwidth="3")
    dot.edge("USD", "STOCK", color=get_col(bat['Stocks']), penwidth=get_pen(bat['Stocks']))
    dot.edge("USD", "BOND", color=get_col(bat['Bonds']), penwidth=get_pen(bat['Bonds']))
    dot.edge("USD", "CRYPTO", color=get_col(bat['Crypto']), penwidth=get_pen(bat['Crypto']))
    dot.edge("USD", "COMM", color=get_col(bat['Commodities']), penwidth=get_pen(bat['Commodities']))
    dot.edge("COMM", "REAL", color=get_col(bat['RealEstate']), penwidth=get_pen(bat['RealEstate']), style="dashed")
    st.graphviz_chart(dot, use_container_width=True)

@st.cache_data(ttl=60)
def fetch_live_trump_news(news_bypass_stamp):
    rss_url = "https://news.google.com/rss/search?q=Trump+OR+Fed+OR+Economy+OR+Tariffs+stock+market&hl=en-US&gl=US&ceid=US:en"
    news_alerts = []
    all_stocks = list(set([t for tkrs in ETF_INFO.values() for t in tkrs['stocks']]))
    
    try:
        response = requests.get(rss_url, timeout=10)
        root = ET.fromstring(response.content)
        for item in root.findall('.//item')[:15]:
            title = item.find('title').text
            link = item.find('link').text
            pub_date = item.find('pubDate').text
            
            detected_tickers = [ticker for ticker in all_stocks if re.search(rf"\b{ticker}\b", title) or f"({ticker})" in title]
            ticker_str = ", ".join(detected_tickers) if detected_tickers else "Genel Makro / Endeks"
            
            title_lower = title.lower()
            impact = "⚖️ Nötr / Sektörel Rotasyon"
            if any(k in title_lower for k in ["tariff", "tax", "china", "trade"]): impact = "🔴 Tech & Çin İthalatı | 🟢 İç Üretim"
            elif any(k in title_lower for k in ["oil", "gas", "energy", "drill", "fossil"]): impact = "🟢 Fosil Yakıt | 🔴 Temiz Enerji"
            elif any(k in title_lower for k in ["crypto", "bitcoin", "sec", "deregulation"]): impact = "🟢 Kripto & Fintek"
            elif any(k in title_lower for k in ["war", "defense", "military", "space"]): impact = "🟢 Savunma & Uzay"
            elif any(k in title_lower for k in ["fed", "rate", "inflation", "powell"]): impact = "📉 Likidite Etkisi"
            elif any(k in title_lower for k in ["ai", "chip", "semiconductor", "tech"]): impact = "🟢 Çip & AI Altyapısı"
            
            news_alerts.append({"Tarih": pub_date[:16], "Haber Başlığı": title, "İlgili Hisse": ticker_str, "Sektör Etkisi": impact, "Link": link})
    except Exception as e:
        return [{"Tarih": "-", "Haber Başlığı": f"Haber motoru başlatılamadı: {e}", "İlgili Hisse": "HATA", "Sektör Etkisi": "HATA", "Link": ""}]
    return news_alerts

# ==========================================
# 4. ŞAHANE V127.0 MATEMATİK & THEME MOTORU
# ==========================================
def get_rma(s, period): return s.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
def get_wma(s, period):
    weights = np.arange(1, period + 1)
    return s.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
def get_rsi(s, period):
    delta = s.diff()
    ma_up = get_rma(delta.clip(lower=0), period)
    ma_down = get_rma(-1 * delta.clip(upper=0), period)
    return 100 - (100 / (1 + (ma_up / ma_down.replace(0, 0.001))))
def get_safe_df(raw_data, ticker):
    if isinstance(raw_data.columns, pd.MultiIndex):
        if ticker in raw_data.columns.levels[0]: return raw_data[ticker].copy()
        elif ticker in raw_data.columns.levels[1]: return raw_data.xs(ticker, level=1, axis=1).copy()
        else: return pd.DataFrame()
    return raw_data.copy()

@st.cache_data(ttl=600)
def fetch_theme_performance(bypass_stamp):
    all_etfs = list(set([t for themes in THEME_TRACKER_MAP.values() for t in themes]))
    start_date = datetime.now() - timedelta(days=400)
    
    try:
        raw_data = yf.download(all_etfs, start=start_date, interval="1d", group_by='ticker', progress=False)
    except:
        return pd.DataFrame()
        
    perf_data = {}
    current_year = datetime.now().year
    
    for t in all_etfs:
        df = get_safe_df(raw_data, t).dropna(subset=['Close'])
        if df.empty or len(df) < 5:
            continue
            
        close = df['Close']
        
        # --- MEVCUT DÖNEM ---
        t_curr  = ((close.iloc[-1] / close.iloc[-2]) - 1) * 100 if len(close) > 1 else 0
        w1_curr = ((close.iloc[-1] / close.iloc[-6]) - 1) * 100 if len(close) > 5 else t_curr
        m1_curr = ((close.iloc[-1] / close.iloc[-22]) - 1) * 100 if len(close) > 21 else w1_curr
        m3_curr = ((close.iloc[-1] / close.iloc[-64]) - 1) * 100 if len(close) > 63 else m1_curr
        
        ytd_df = df[df.index.year == current_year]
        ytd_curr = ((close.iloc[-1] / ytd_df['Close'].iloc[0]) - 1) * 100 if not ytd_df.empty else m3_curr

        # --- ÖNCEKİ EŞDEĞER DÖNEM (DELTA İÇİN) ---
        t_prev  = ((close.iloc[-2] / close.iloc[-3]) - 1) * 100 if len(close) > 2 else 0
        w1_prev = ((close.iloc[-6] / close.iloc[-11]) - 1) * 100 if len(close) > 10 else 0
        m1_prev = ((close.iloc[-22] / close.iloc[-43]) - 1) * 100 if len(close) > 42 else 0
        m3_prev = ((close.iloc[-64] / close.iloc[-127]) - 1) * 100 if len(close) > 126 else 0
        
        prev_year_df = df[(df.index.year == current_year - 1) & (df.index.dayofyear <= datetime.now().timetuple().tm_yday)]
        if len(prev_year_df) > 0 and len(df[df.index.year == current_year - 1]) > 0:
            first_day_prev_year = df[df.index.year == current_year - 1]['Close'].iloc[0]
            ytd_prev = ((prev_year_df['Close'].iloc[-1] / first_day_prev_year) - 1) * 100
        else:
            ytd_prev = 0

        perf_data[t] = {
            "Today": t_curr, "Prev_Today": t_prev,
            "1W": w1_curr, "Prev_1W": w1_prev,
            "1M": m1_curr, "Prev_1M": m1_prev,
            "3M": m3_curr, "Prev_3M": m3_prev,
            "YTD": ytd_curr, "Prev_YTD": ytd_prev
        }
        
    perf_df = pd.DataFrame.from_dict(perf_data, orient='index')
    
    theme_perf = {}
    for theme, tickers in THEME_TRACKER_MAP.items():
        valid_tickers = [t for t in tickers if t in perf_df.index]
        if valid_tickers:
            theme_perf[theme] = perf_df.loc[valid_tickers].mean()
            
    final_df = pd.DataFrame.from_dict(theme_perf, orient='index')
    return final_df

@st.cache_data(ttl=300)
def calculate_signals(ticker_list, interval="1d", bypass_stamp=""):
    if not ticker_list: return pd.DataFrame()
    end_date = datetime.now()
    days_back = 200 if interval == "1wk" else 90 if interval == "1d" else 50
    yf_int = "1h" if interval == "4h" else interval
    
    try: raw_data = yf.download(ticker_list, start=end_date - timedelta(days=days_back), end=end_date, interval=yf_int, group_by='ticker', progress=False)
    except: return pd.DataFrame()

    results = []
    for t in ticker_list:
        try:
            df = get_safe_df(raw_data, t).dropna(subset=['Close'])
            if len(df) < 30: continue
            if interval == "4h":
                df.index = pd.to_datetime(df.index)
                df = df.resample('4h').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()

            close, high, low, open_p, vol = df['Close'], df['High'], df['Low'], df['Open'], df['Volume']
            
            slope_up = (close > close.shift(1)) & (close.shift(1) > close.shift(2))
            slope_dn = (close < close.shift(1)) & (close.shift(1) < close.shift(2))

            pct_1d = (close.iloc[-1] / close.iloc[-2] - 1) * 100 if len(close) > 1 else 0
            pct_1w = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) > 5 else 0

            ema1_s3 = close.ewm(span=9, adjust=False).mean()
            v150_v_avg = vol.rolling(20).mean()
            bear_trap = (low < ema1_s3) & (close > ema1_s3) & (vol > v150_v_avg * 1.5)
            bull_trap = (high > ema1_s3) & (close < ema1_s3) & (vol > v150_v_avg * 1.5)

            wma_cv = get_wma(close * vol, 14)
            wma_v = get_wma(vol, 14).clip(lower=0.001)
            eff_price = get_wma(wma_cv / wma_v, 3)
            price_cross_eff_up = (close > eff_price) & (close.shift(1) <= eff_price.shift(1))
            price_cross_eff_dn = (close < eff_price) & (close.shift(1) >= eff_price.shift(1))

            eff_status = pd.Series("➖ NÖTR", index=close.index)
            eff_status.loc[close > eff_price] = "🟢 POZ"
            eff_status.loc[close < eff_price] = "🔴 NEG"
            eff_status.loc[price_cross_eff_up] = "🚀 UP KIRILIM"
            eff_status.loc[price_cross_eff_dn] = "🩸 DOWN KIRILIM"

            r14 = get_rsi(close, 14)
            c_range_q = (high - low).clip(lower=0.001)
            delta_q = ((close - low) - (high - close)) / c_range_q
            delta_vol_q = (delta_q * vol).rolling(20).mean() / vol.rolling(20).mean().clip(lower=0.001)
            rvol_q = (vol / vol.rolling(20).mean().clip(lower=1)).clip(upper=2.5)

            base_pwr_q = ((r14 - 50) + (delta_vol_q * 50)) * rvol_q * 1.5
            logic_pwr_q = np.log(1 + np.exp(np.clip(base_pwr_q / 5, -50, 50))) * 5
            logic_pwr_q = np.where((low > high.shift(2)) & (close > open_p), logic_pwr_q + 35, logic_pwr_q)

            log_w_q = np.log10(1 + np.clip(logic_pwr_q, 0, None))
            pct_w_q = np.clip((log_w_q * 65)**0.8 * 1.8, 0, 100)
            w_pwr_q = get_wma(pd.Series(pct_w_q, index=close.index), 2)

            pct_pro_q = w_pwr_q.ewm(span=3, adjust=False).mean()
            yellow_rest = (w_pwr_q.shift(1) < pct_pro_q.shift(1)) & (w_pwr_q.shift(2) < pct_pro_q.shift(2))
            whale_re_entry = (w_pwr_q > pct_pro_q) & (w_pwr_q.shift(1) <= pct_pro_q.shift(1)) & yellow_rest

            sma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
            k_mid = sma20
            k_up = k_mid + 1.5 * tr.rolling(20).mean()
            k_low = k_mid - 1.5 * tr.rolling(20).mean()
            is_sqz = (sma20 - 2 * std20 > k_low) & (sma20 + 2 * std20 < k_up)
            vol_hole = is_sqz & (close <= (k_mid - (k_up - k_mid) / 3.0))

            _kin_b = ((vol > v150_v_avg * 1.2) & (close > open_p)).astype(int)
            _tre_b = (close > close.ewm(span=34).mean()).astype(int)
            _kur_b = ((r14 > 50) & (r14.shift(1) <= 50)).astype(int)
            total_score_b = _kin_b + _tre_b + _kur_b

            sig = "⚪ WAIT"
            if interval == "1wk":
                if (w_pwr_q.iloc[-1] > 80) and slope_up.iloc[-1]: sig = "🐋 WHALE ACCUMULATION"
                elif (r14.iloc[-1] < 35): sig = "🕳️ DEEP VALUE (DCA)"
                elif whale_re_entry.iloc[-1]: sig = "🚀 MOMENTUM GAP"
            else:
                if bull_trap.iloc[-1]: sig = "⛔"
                elif bear_trap.iloc[-1]: sig = "✅"
                elif whale_re_entry.iloc[-1]: sig = "🔄 WHALE RE-ENTRY"
                elif price_cross_eff_up.iloc[-1] and slope_up.iloc[-1]: sig = "🚀 UP KIRILIM"
                elif price_cross_eff_dn.iloc[-1] and slope_dn.iloc[-1]: sig = "🩸 DOWN KIRILIM"
                elif vol_hole.iloc[-1]: sig = "🕳️ VOLA HOLE"
                elif w_pwr_q.iloc[-1] >= 85: sig = "🐋 WHALE IN"

            results.append({
                "Ticker": t, "Sinyal": sig, "Efor": eff_status.iloc[-1], "Fiyat": f"${close.iloc[-1]:.2f}",
                "Whale Power": float(f"{w_pwr_q.iloc[-1]:.1f}"), "Fusion": int(total_score_b.iloc[-1]),
                "1 Gün (%)": round(pct_1d, 2), "1 Hafta (%)": round(pct_1w, 2)
            })
        except: continue
    if results: return pd.DataFrame(results).sort_values(by="Fusion", ascending=False)
    return pd.DataFrame()

@st.cache_data(ttl=600)
def fetch_valuation_data(ticker_list, bypass_stamp):
    funds = []
    for t in ticker_list:
        try:
            tk = yf.Ticker(t)
            mc = tk.fast_info.get('marketCap', 0)
            try:
                info = tk.info
                pe = info.get('trailingPE', 0)
                ps = info.get('priceToSalesTrailing12Months', 0)
                peg = info.get('pegRatio', 0)
            except:
                pe, ps, peg = 0, 0, 0
                
            pe = pe if pe is not None else 0
            ps = ps if ps is not None else 0
            peg = peg if peg is not None else 0
            
            funds.append({"Ticker": t, "MarketCap": mc, "PE": pe, "PS": ps, "PEG": peg})
            time.sleep(0.1)
        except: 
            funds.append({"Ticker": t, "MarketCap": 0, "PE": 0, "PS": 0, "PEG": 0})
    return pd.DataFrame(funds)

@st.cache_data(ttl=3600)
def fetch_earnings_fair_value(ticker_list, bypass_stamp):
    data = []
    for t in ticker_list:
        try:
            tk = yf.Ticker(t)
            info = tk.info
            
            # Bilanço Tarihi Bulma
            earn_date_str = "Bilinmiyor"
            days_left = "-"
            days_left_sort = 9999
            
            try:
                cal = tk.calendar
                if cal is not None and not cal.empty and 'Earnings Date' in cal:
                    first_date = cal['Earnings Date'].iloc[0]
                    if isinstance(first_date, list):
                        first_date = first_date[0]
                    if first_date:
                        earn_date = first_date.date()
                        days_left_sort = (earn_date - datetime.now().date()).days
                        days_left = f"{days_left_sort} Gün"
                        earn_date_str = earn_date.strftime("%d-%m-%Y")
            except:
                pass
                
            fv = info.get('targetMeanPrice', info.get('targetMedianPrice', 'Bilinmiyor'))
            current = info.get('currentPrice', info.get('previousClose', 0))
            
            gap = "-"
            if isinstance(fv, (int, float)) and isinstance(current, (int, float)) and current > 0:
                diff = ((fv / current) - 1) * 100
                gap = f"%{diff:.1f} {'Ucuz' if diff > 0 else 'Pahalı'}"
            
            data.append({
                "Hisse": t,
                "Bilanço Tarihi": earn_date_str,
                "Kalan Gün": days_left,
                "Güncel Fiyat": f"${current:.2f}" if current else "-",
                "Adil Değer (Fair Value)": f"${fv:.2f}" if isinstance(fv, (int, float)) else fv,
                "Potansiyel": gap,
                "_sort": days_left_sort
            })
        except:
            data.append({"Hisse": t, "Bilanço Tarihi": "Hata", "Kalan Gün": "-", "Güncel Fiyat": "-", "Adil Değer (Fair Value)": "-", "Potansiyel": "-", "_sort": 9999})
            
    df = pd.DataFrame(data).sort_values(by="_sort")
    df = df.drop(columns=["_sort"])
    return df

def style_signals(val):
    if isinstance(val, str):
        if 'GAP' in val: return 'background-color: #00e676; color: black; font-weight: bold;'
        if 'DEEP' in val: return 'background-color: #00b0ff; color: black; font-weight: bold;'
        if 'WHALE RE-ENTRY' in val: return 'background-color: #006064; color: white; font-weight: bold;'
        if 'WHALE IN' in val: return 'background-color: #01579b; color: white;'
        if 'VOLA HOLE' in val: return 'background-color: #4a148c; color: white;'
        if 'UP KIRILIM' in val: return 'background-color: #00FF88; color: black; font-weight: bold;'
        if 'DOWN KIRILIM' in val: return 'background-color: #FF1744; color: white; font-weight: bold;'
        if val == '⛔': return 'background-color: #b71c1c; color: white; font-size: 1.2rem; text-align: center;'
        if val == '✅': return 'background-color: #004d40; color: white; font-size: 1.2rem; text-align: center;'
    return 'background-color: #111111; color: white;'

def style_efor(val):
    if isinstance(val, str):
        if '🚀' in val: return 'background-color: #00FF88; color: black; font-weight: bold;'
        if '🩸' in val: return 'background-color: #FF1744; color: white; font-weight: bold;'
        if '🟢' in val: return 'color: #00FF88; font-weight: bold;'
        if '🔴' in val: return 'color: #FF1744; font-weight: bold;'
    return 'color: #888;'

def style_percentages(val):
    if isinstance(val, (float, int)): return f"color: {'#00ff88' if val > 0 else '#ff3333'}; font-weight: bold;"
    return ''

# ==========================================
# 5. KOKPİT ARAYÜZÜ
# ==========================================
st.title("🏛️ AETHER APEX ULTIMATE V134.0")
st.markdown("Kararnamelerin rasyonel etki puanlarını, 12 kanallı esneklik yapısını ve **ŞAHANE V127** efor kırılımlarını birleştirir. Tüm sekmeler gerçek zamanlı güncellenebilir.")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🌐 MAKRO & OPEX", 
    "🔥 THEME TRACKER",
    "🦅 KUŞBAKIŞI SEKTÖR",
    "⚖️ VALUATION GAP (Çarpan Uçurumu)",
    "🦈 HAFTALIK MOMENTUM",
    "🚨 4H & OMNI RADAR",
    "🚀 FUTURE THEMES",
    "📅 BİLANÇO & ADİL DEĞER"
])

all_etfs_to_scan = list(MAIN_SECTORS.keys()) + list(ETF_INFO.keys())
portfolio_tickers = sorted(list(set([t for tkrs in ETF_INFO.values() for t in tkrs['stocks']])))

# ---------------------------------------------------------
# TAB 1: MAKRO & OPEX (Güncellendi)
# ---------------------------------------------------------
with tab1:
    st.subheader("⚙️ Institutional Desk: Gelişmiş Makro Tetikleyiciler & RSS Canlı Haber Akışı")
    
    col_update_makro, _ = st.columns([1, 4])
    if col_update_makro.button("🔄 Makro Modeli Zorla Güncelle", use_container_width=True):
        st.session_state.macro_nonce = str(time.time())
        st.rerun()

    t_cols = st.columns(4)
    for i, trig in enumerate(SYSTEM_TRIGGERS.keys()):
        with t_cols[i]:
            if st.button(f"Senaryo: {trig}", use_container_width=True):
                st.session_state.active_trigger = trig

    col_chart, col_docs = st.columns([3, 2])
    with col_chart:
        st.markdown(f"#### 💸 **Sermaye Akış Rotası:** ({st.session_state.active_trigger})")
        draw_smart_money_flow(SYSTEM_TRIGGERS[st.session_state.active_trigger])
        
        # Seçili Senaryonun Açıklaması
        st.info(f"**Neden / Etki Analizi:** {SYSTEM_TRIGGERS[st.session_state.active_trigger]['desc']}")
        
        st.divider()
        st.markdown("#### 🔊 Canlı Medya, Makro ve Sektör Tarayıcı (RSS)")
        
        if st.button("🔄 Canlı Haberleri ve Etkilerini Güncelle", use_container_width=True):
            st.session_state.news_nonce = str(time.time())
            
        with st.spinner("Küresel haber ağları ve yasa tasarıları taranıyor..."):
            df_news = pd.DataFrame(fetch_live_trump_news(st.session_state.news_nonce))
            if not df_news.empty:
                st.dataframe(df_news, use_container_width=True, hide_index=True, column_config={"Link": st.column_config.LinkColumn("Haber Linki")})
                
    with col_docs:
        st.markdown("""
        <div class="macro-def-box"><div class="macro-def-title">📚 Opex Pinning Nedir?</div><p style="font-size: 0.85rem; color: #b0b0b0;">Aylık veya Üç Aylık opsiyon (OpEx) vadelerinin sonuna yaklaşırken, Market Maker'ların fiyatı, yatırımcıların büyük kısmının kaybedeceği Max Pain noktasına doğru çekmesidir. Sahte kırılımlar (Whipsaw) artar.</p></div>
        <div class="macro-def-box"><div class="macro-def-title">📈 Gamma Squeeze Nedir?</div><p style="font-size: 0.85rem; color: #b0b0b0;">Aşırı Call opsiyon alımı sonrası piyasa yapıcıların (Dealers) hedge amaçlı panikle spot hisse alması sonucu oluşan parabolik fiyat erimesi (Melt-Up) döngüsüdür.</p></div>
        <div class="macro-def-box"><div class="macro-def-title">🌍 Geopolitical Shock</div><p style="font-size: 0.85rem; color: #b0b0b0;">Savaş, ambargo veya tedarik zinciri çöküşünde sermayenin büyüme hisselerinden kaçıp sert emtiaya (Bakır, Petrol, Savunma) ve nakde akmasıdır.</p></div>
        <div class="macro-def-box"><div class="macro-def-title">🏦 Liquidity Crunch (Fed)</div><p style="font-size: 0.85rem; color: #b0b0b0;">Merkez bankasının şahin politikalarla (faiz artışı/QT) piyasadan doları çekmesidir. Yüksek çarpanlı varlıklarda devasa margin call satışları getirir.</p></div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 2: THEME TRACKER (PDF/Image İndirme Seçeneği İle)
# ---------------------------------------------------------
with tab2:
    st.subheader("🔥 Theme Tracker: Sektörel İvme ve Performans Matrisi")
    st.markdown("""
        > 💡 **Parantez İçi Değerler Nedir?** Yüzdelerin yanındaki parantez içi değerler (🔺/🔻), **önceki eşdeğer periyoda kıyasla ivmenin (momentumun) yönünü** gösterir. 
        Örneğin *3M (Son 3 Ay)* grafiğine bakıyorsanız, delta, *önceki 3 aya göre* ivmenin ne kadar hızlandığını veya yavaşladığını belirtir. 
        Grafiğin sağ üstündeki menüden (📸 Camera ikonuna tıklayarak) SVG/PNG olarak indirebilirsin.
    """)
    
    col_per, col_btn = st.columns([4, 1])
    with col_per:
        period_selection = st.radio("Zaman Periyodu Seçin:", ["Today", "1W", "1M", "3M", "YTD"], horizontal=True)
    with col_btn:
        if st.button("🔄 Trackeri Yenile", use_container_width=True):
            st.session_state.battery_nonce = str(time.time())
            
    with st.spinner("Piyasa verileri, geçmiş eşdeğer periyot kıyaslamalarıyla birlikte harmanlanıyor..."):
        df_perf = fetch_theme_performance(st.session_state.battery_nonce)
        
        if not df_perf.empty:
            df_sorted = df_perf.sort_values(by=period_selection, ascending=True)
            colors = ['#3b82f6' if val >= 0 else '#ec4899' for val in df_sorted[period_selection]]
            
            y_labels = []
            for theme in df_sorted.index:
                tickers = THEME_TRACKER_MAP.get(theme, [])
                ticker_str = f" ({', '.join(tickers[:3])}+)" if len(tickers) > 3 else f" ({', '.join(tickers)})" if tickers else ""
                y_labels.append(f"{theme} <span style='font-size: 11px; color: #888;'>{ticker_str}</span>")
            
            text_labels = []
            for idx, row in df_sorted.iterrows():
                val = row[period_selection]
                prev_val = row[f"Prev_{period_selection}"]
                delta = val - prev_val
                val_str = f"+{val:.2f}%" if val > 0 else f"{val:.2f}%"
                delta_str = f" (🔺+{delta:.2f}%)" if delta > 0 else f" (🔻{delta:.2f}%)" if delta < 0 else " (➖ 0.00%)"
                text_labels.append(f"{val_str}{delta_str}")
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=y_labels, x=df_sorted[period_selection], orientation='h',
                marker=dict(color=colors, line=dict(width=0)),
                text=text_labels, textposition='outside',
                textfont=dict(color='#e0e0e0', size=13, family="Arial", weight="bold")
            ))
            
            fig.update_layout(
                height=900, margin=dict(l=10, r=50, t=10, b=10),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, zeroline=True, zerolinecolor='#444', zerolinewidth=2, showticklabels=False),
                yaxis=dict(showgrid=False, tickfont=dict(color='#e0e0e0', size=14, weight="bold")),
                showlegend=False
            )
            
            min_x = df_sorted[period_selection].min()
            max_x = df_sorted[period_selection].max()
            padding = (max_x - min_x) * 0.35 if (max_x - min_x) != 0 else 5
            fig.update_xaxes(range=[min_x - padding, max_x + padding])
            
            # PDF/Vektör İndirme Ayarları Aktive Edildi (Streamlit native config)
            st.plotly_chart(fig, use_container_width=True, config={
                'toImageButtonOptions': {
                    'format': 'svg', # Vektörel export (PDF kalitesindedir, bulanıklaşmaz)
                    'filename': f'ThemeTracker_{period_selection}',
                    'height': 900,
                    'width': 1200,
                    'scale': 1
                }
            })

# ---------------------------------------------------------
# TAB 3, 4, 5, 6, 7: DEĞİŞMEDEN KORUNDU
# ---------------------------------------------------------
with tab3:
    st.subheader("🦅 Kuşbakışı ETF Sinyal Radarı")
    if st.button("🔄 Sektör Sinyallerini Güncelle", key="btn_bird"):
        st.session_state.battery_nonce = str(time.time())
    with st.spinner("1D Sinyaller Hesaplanıyor..."):
        df_bird = calculate_signals(all_etfs_to_scan, interval="1d", bypass_stamp=st.session_state.battery_nonce)
        if not df_bird.empty:
            df_bird['Kapsam'] = df_bird['Ticker'].map(MAIN_SECTORS).fillna("Spesifik Tematik ETF")
            st.dataframe(df_bird[['Kapsam', 'Ticker', 'Sinyal', 'Efor', 'Fiyat', '1 Gün (%)', '1 Hafta (%)', 'Whale Power', 'Fusion']].style.map(style_signals, subset=['Sinyal']).map(style_percentages, subset=['1 Gün (%)', '1 Hafta (%)']).map(style_efor, subset=['Efor']), use_container_width=True, hide_index=True)

with tab4:
    st.subheader("⚖️ Tematik Çarpan Uçurumu ve Sektör Kıyaslaması")
    col_v1, col_v2 = st.columns([1, 4])
    group_choice = col_v1.selectbox("Analiz Edilecek Tema:", list(ETF_INFO.keys()))
    if col_v2.button("🔄 Finansal Verileri Güncelle"): st.session_state.val_nonce = str(time.time())
    
    if group_choice:
        with st.spinner("Rasyolar çekiliyor..."):
            tickers = ETF_INFO[group_choice]['stocks']
            df_val = fetch_valuation_data(tickers, st.session_state.val_nonce)
            if not df_val.empty:
                df_val = df_val[df_val['MarketCap'] > 0].sort_values(by='MarketCap', ascending=False)
                avg_pe, avg_ps, avg_peg = df_val[df_val['PE'] > 0]['PE'].mean(), df_val[df_val['PS'] > 0]['PS'].mean(), df_val[df_val['PEG'] > 0]['PEG'].mean()
                avg_pe, avg_ps, avg_peg = avg_pe if pd.notna(avg_pe) else 0, avg_ps if pd.notna(avg_ps) else 0, avg_peg if pd.notna(avg_peg) else 0
                st.markdown(f"""
                <div style="background-color: #111; padding: 15px; border-left: 5px solid #f1c40f; margin-bottom: 20px;">
                    <h4 style="color: #f1c40f; margin:0;">📊 {ETF_INFO[group_choice]['area']} - Sektör Ortalamaları</h4>
                    <p style="margin: 5px 0 0 0; font-size: 1.1rem;">F/K (PE): <strong>{avg_pe:.1f}</strong> | F/S (PS): <strong>{avg_ps:.1f}</strong> | PEG: <strong>{avg_peg:.1f}</strong></p>
                </div>
                """, unsafe_allow_html=True)

                if not df_val.empty:
                    leader = df_val.iloc[0]
                    st.markdown(f"<div class='valuation-leader'>👑 Lider: {leader['Ticker']} (Değer: ${leader['MarketCap']/1e9:.1f}B | F/K: {leader['PE']:.1f} | F/S: {leader['PS']:.1f} | PEG: {leader['PEG']:.1f})</div><hr>", unsafe_allow_html=True)
                    for i in range(1, len(df_val)):
                        row = df_val.iloc[i]
                        pe_diff = row['PE'] - avg_pe
                        pe_color = "#ff3333" if pe_diff > 0 else "#00ff88"
                        pe_txt = f"Ortalamanın {abs(pe_diff):.1f} {'Üzerinde (Pahalı)' if pe_diff > 0 else 'Altında (Ucuz)'}" if avg_pe > 0 and row['PE'] > 0 else "Rasyo verisi anlık çekilemedi"
                        st.markdown(f'''
                            <div class="valuation-gap-card">
                                <h3><span class="valuation-laggard">{row['Ticker']}</span> <span style="font-size:0.9rem; color:#888;">(${row['MarketCap']/1e9:.1f}B | F/K: {row['PE']:.1f} | PEG: {row['PEG']:.1f})</span></h3>
                                <div style="color: {pe_color}; font-weight:bold; margin-bottom: 5px;">Rasyo Analizi: {pe_txt}</div>
                            </div>
                        ''', unsafe_allow_html=True)

with tab5:
    st.subheader("🦈 Haftalık Momentum-Gap Avcısı")
    if st.button("🔄 Haftalık Veriyi Yenile"): st.session_state.battery_nonce = str(time.time())
    with st.spinner("1W Sinyaller Hesaplanıyor..."):
        df_wk = calculate_signals(portfolio_tickers, interval="1wk", bypass_stamp=st.session_state.battery_nonce)
        if not df_wk.empty: st.dataframe(df_wk[['Ticker', 'Sinyal', 'Efor', 'Fiyat', '1 Hafta (%)', 'Whale Power']].style.map(style_signals, subset=['Sinyal']).map(style_efor, subset=['Efor']), use_container_width=True, hide_index=True)

with tab6:
    st.subheader("🚨 4H & Günlük OMNI RADAR")
    if st.button("🔄 Radar Verilerini Yenile"): st.session_state.battery_nonce = str(time.time())
    with st.spinner("Piyasa taranıyor..."):
        df_radar = calculate_signals(portfolio_tickers, interval="1d", bypass_stamp=st.session_state.battery_nonce)
        if not df_radar.empty:
            c1, c2 = st.columns(2)
            c1.markdown("#### 🟢 POZİTİF")
            c1.dataframe(df_radar[df_radar['Sinyal'].isin(['✅', '🚀 UP KIRILIM', '🔄 WHALE RE-ENTRY'])].style.map(style_signals, subset=['Sinyal']), use_container_width=True, hide_index=True)
            c2.markdown("#### 🔴 NEGATİF")
            c2.dataframe(df_radar[df_radar['Sinyal'].isin(['⛔', '🩸 DOWN KIRILIM'])].style.map(style_signals, subset=['Sinyal']), use_container_width=True, hide_index=True)

with tab7:
    st.subheader("🚀 FUTURE THEMES: Geleceğin Teknolojileri")
    if st.button("🔄 Gelecek Temalarını Yenile"): st.session_state.battery_nonce = str(time.time())
    future_tickers = list(set([t for tkrs in FUTURE_THEMES_MAP.values() for t in tkrs]))
    with st.spinner("Future Themes evreni taranıyor..."):
        df_future = calculate_signals(future_tickers, interval="1d", bypass_stamp=st.session_state.battery_nonce)
        if not df_future.empty: st.dataframe(df_future[['Ticker', 'Sinyal', 'Efor', 'Fiyat', '1 Gün (%)', 'Whale Power', 'Fusion']].style.map(style_signals, subset=['Sinyal']).map(style_efor, subset=['Efor']), use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# TAB 8: YENİ - BİLANÇO & ADİL DEĞER (EARNINGS)
# ---------------------------------------------------------
with tab8:
    st.subheader("📅 Bilanço & Adil Değer (Earnings & Fair Value) Radarı")
    
    col_input, col_action = st.columns([4, 1])
    with col_input:
        new_tickers_input = st.text_input("Virgül veya boşluk ile ayırarak yeni hisse sembolleri ekle/çıkar (Örn: AAPL, TSLA MSFT):")
    with col_action:
        st.write("") # Boşluk
        if st.button("➕ Listeyi Güncelle", use_container_width=True):
            if new_tickers_input:
                new_tkrs = [t.strip().upper() for t in re.split(r'[,\s]+', new_tickers_input) if t.strip()]
                # Birleştir ve tekrarları sil
                combined_list = list(set(st.session_state.custom_earnings_list + new_tkrs))
                st.session_state.custom_earnings_list = sorted(combined_list)
                st.success("Liste başarıyla güncellendi!")

    # Aktif Liste Gösterimi (Silme Butonu İçin Expander)
    with st.expander("🛠️ Aktif Hisse Listesini Yönet (Çıkar)"):
        st.write("Aşağıdaki listeden çıkarmak istediğiniz sembolü seçin:")
        tkr_to_remove = st.selectbox("Çıkarılacak Hisse:", ["Seçiniz..."] + st.session_state.custom_earnings_list)
        if st.button("❌ Hissesi Çıkar") and tkr_to_remove != "Seçiniz...":
            st.session_state.custom_earnings_list.remove(tkr_to_remove)
            st.rerun()

    if st.button("🔄 Finansalları & Bilanço Takvimini Tarat"):
        st.session_state.val_nonce = str(time.time())
        
    with st.spinner("Seçili varlıklar için bilanço tarihleri ve Fair Value hedefleri Yahoo Finance'tan çekiliyor..."):
        df_earnings = fetch_earnings_fair_value(st.session_state.custom_earnings_list, st.session_state.val_nonce)
        
        if not df_earnings.empty:
            # Renklendirme Stilleri
            def highlight_days(val):
                if isinstance(val, str) and "Gün" in val:
                    num = int(val.replace(" Gün", ""))
                    if num <= 7: return 'background-color: #ff3333; color: white; font-weight: bold;'
                    elif num <= 15: return 'background-color: #f1c40f; color: black; font-weight: bold;'
                return ''
                
            def highlight_pot(val):
                if isinstance(val, str):
                    if "Ucuz" in val: return 'color: #00ff88; font-weight: bold;'
                    elif "Pahalı" in val: return 'color: #ff3333; font-weight: bold;'
                return ''

            st.dataframe(
                df_earnings.style
                .map(highlight_days, subset=['Kalan Gün'])
                .map(highlight_pot, subset=['Potansiyel']),
                use_container_width=True,
                hide_index=True
            )
