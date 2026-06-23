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
    .battery-container { width: 100%; background-color: #222; border-radius: 10px; margin: 5px 0 15px 0; border: 1px solid #444; position: relative; height: 25px; overflow: hidden; }
    .battery-fill { height: 100%; border-radius: 8px; transition: width 0.5s ease; display: flex; align-items: center; justify-content: flex-end; padding-right: 10px; font-weight: bold; color: #000 !important; font-size: 0.9rem; }
    .valuation-gap-card { background: linear-gradient(145deg, #111 0%, #0a0a0a 100%); padding: 20px; border-radius: 12px; border: 1px solid #333; margin-bottom: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.5); }
    .valuation-leader { color: #00ff88; font-weight: 900; font-size: 1.2rem; }
    .valuation-laggard { color: #f1c40f; font-weight: bold; }
    .macro-def-box { background-color: #111; border-left: 4px solid #FF1744; padding: 15px; border-radius: 5px; margin-bottom: 15px; }
    .macro-def-title { color: #00ff88; font-weight: bold; font-size: 1.1rem; margin-bottom: 5px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. STATE & GÜNCELLEME KONTROLLERİ (CACHE BYPASS)
# ==========================================
if 'active_trigger' not in st.session_state: st.session_state.active_trigger = "OPEX PINNING"
if 'macro_nonce' not in st.session_state: st.session_state.macro_nonce = str(time.time())
if 'battery_nonce' not in st.session_state: st.session_state.battery_nonce = str(time.time())
if 'val_nonce' not in st.session_state: st.session_state.val_nonce = str(time.time())
if 'news_nonce' not in st.session_state: st.session_state.news_nonce = str(time.time())

# ==========================================
# 2. KURUMSAL NİŞ ETF & HİSSE EVRENİ
# ==========================================
MAIN_SECTORS = {
    "XLK": "Ana Sektör: Teknoloji", "XLI": "Ana Sektör: Sanayi", "XLE": "Ana Sektör: Enerji",
    "XLV": "Ana Sektör: Sağlık", "XLF": "Ana Sektör: Finans", "XLY": "Ana Tüketim",
    "XLB": "Ana Sektör: Materyal", "XLC": "Ana Sektör: İletişim", "XLRE": "Ana Sektör: Gayrimenkul",
    "XLU": "Ana Sektör: Kamu Hizmetleri"
}

GLOBAL_MAP = {
    "Teknoloji (Bulut & AI)": ["XLK", "CLOU", "IGV", "AIQ", "CIBR", "BOTZ", "CYBER"],
    "Yarı İletken (Çip Mimarisi)": ["SOXX", "SMH", "EUV", "PHOTON"],
    "Enerji & Altyapı": ["XLE", "XOP", "OIH", "XLU", "URA", "ICLN", "PAVE", "JOUL"],
    "Emtia & Madencilik": ["COPX", "LIT", "REMX", "GDX", "XME"],
    "Lojistik & Havacılık": ["IYT", "JETS", "HULL"],
    "Savunma & Uzay": ["XAR", "ARKX", "UFO", "SPACE_RACE"],
    "Finans & Kripto": ["XLF", "KRE", "ARKF", "IBIT", "WGMI"],
    "Gayrimenkul & Veri Merkezleri": ["XLRE", "REZ", "SRVR", "VNQ"],
    "Tüketim & Perakende": ["XLY", "XRT", "XHB"],
    "Özel Durumlar (IPO/Trump)": ["TRUMP_PF", "RECENT_IPO"]
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

SYSTEM_TRIGGERS = {
    "GAMMA SQUEEZE": {"color": "#00ff88", "battery": {"Stocks": 95, "Bonds": 20, "Crypto": 90, "Commodities": 55, "RealEstate": 65}},
    "OPEX PINNING": {"color": "#f1c40f", "battery": {"Stocks": 50, "Bonds": 50, "Crypto": 48, "Commodities": 52, "RealEstate": 50}},
    "GEOPOLITICAL SHOCK": {"color": "#ff3333", "battery": {"Stocks": 25, "Bonds": 85, "Crypto": 35, "Commodities": 95, "RealEstate": 40}},
    "LIQUIDITY CRUNCH (FED)": {"color": "#9b59b6", "battery": {"Stocks": 15, "Bonds": 90, "Crypto": 10, "Commodities": 35, "RealEstate": 25}}
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

def draw_battery(label, current, color, delta_1d=0.0):
    d1_icon = f"🔺+{delta_1d:.1f}" if delta_1d > 0 else f"🔻{delta_1d:.1f}" if delta_1d < 0 else "➖ 0.0"
    d1_color = "#00ff88" if delta_1d > 0 else "#ff3333" if delta_1d < 0 else "#888888"
    st.markdown(f"""
        <div style="margin-bottom: 2px; font-size: 0.85rem; color: #ccc; display: flex; justify-content: space-between;">
            <span>{label}</span>
            <span style="color: {d1_color}; font-weight: bold; font-size: 0.75rem;">1D Değişim: {d1_icon}</span>
        </div>
        <div class="battery-container" style="height: 20px;">
            <div class="battery-fill" style="width: {min(max(current,0), 100)}%; background-color: {color}; font-size: 0.8rem;">%{int(current)}</div>
        </div>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=60)
def fetch_live_trump_news(news_bypass_stamp):
    # DÜZELTME: Sadece Ticker bazlı değil, ekonomi ve makro kelimelerini de arıyoruz.
    rss_url = "https://news.google.com/rss/search?q=Trump+OR+Fed+OR+Economy+OR+Tariffs+stock+market&hl=en-US&gl=US&ceid=US:en"
    news_alerts = []
    all_stocks = list(set([t for tkrs in ETF_INFO.values() for t in tkrs['stocks']]))
    
    try:
        response = requests.get(rss_url, timeout=10)
        root = ET.fromstring(response.content)
        
        for item in root.findall('.//item')[:15]:  # En güncel 15 haberi al
            title = item.find('title').text
            link = item.find('link').text
            pub_date = item.find('pubDate').text
            
            # Ticker Tespit Et
            detected_tickers = [ticker for ticker in all_stocks if re.search(rf"\b{ticker}\b", title) or f"({ticker})" in title]
            ticker_str = ", ".join(detected_tickers) if detected_tickers else "Genel Makro / Endeks"
            
            # Sektör Etkisi Mantığı (Keyword Algılama)
            title_lower = title.lower()
            impact = "⚖️ Nötr / Sektörel Rotasyon"
            if any(k in title_lower for k in ["tariff", "tax", "china", "trade"]):
                impact = "🔴 Tech & Çin İthalatı | 🟢 İç Üretim (XLI, XME)"
            elif any(k in title_lower for k in ["oil", "gas", "energy", "drill", "fossil"]):
                impact = "🟢 Fosil Yakıt (XLE, XOP) | 🔴 Temiz Enerji (ICLN)"
            elif any(k in title_lower for k in ["crypto", "bitcoin", "sec", "deregulation"]):
                impact = "🟢 Kripto & Fintek (WGMI, ARKF)"
            elif any(k in title_lower for k in ["war", "defense", "military", "space"]):
                impact = "🟢 Savunma & Uzay (XAR, ARKX)"
            elif any(k in title_lower for k in ["fed", "rate", "inflation", "powell", "cpi", "yield"]):
                impact = "📉 Likidite Etkisi (Tüm Piyasayı Etkiler)"
            elif any(k in title_lower for k in ["ai", "chip", "semiconductor", "tech"]):
                impact = "🟢 Çip & AI Altyapısı (SOXX, XLK)"
            
            news_alerts.append({
                "Tarih": pub_date[:16], 
                "Haber Başlığı": title, 
                "İlgili Hisse": ticker_str, 
                "Sektör Etkisi": impact,
                "Link": link
            })
            
    except Exception as e:
        return [{"Tarih": "-", "Haber Başlığı": f"Haber motoru başlatılamadı: {e}", "İlgili Hisse": "HATA", "Sektör Etkisi": "HATA", "Link": ""}]
    
    return news_alerts

# ==========================================
# 4. ŞAHANE V127.0 MATEMATİK MOTORU
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

@st.cache_data(ttl=300)
def fetch_matrix_data(bypass_stamp):
    all_etfs = list(set([etf for etfs in GLOBAL_MAP.values() for etf in etfs]))
    all_etfs.extend(list(MAIN_SECTORS.keys()))
    end_date = datetime.now()
    try:
        raw_data = yf.download(all_etfs, start=end_date - timedelta(days=90), end=end_date, interval="1d", group_by='ticker', progress=False)
    except: return pd.DataFrame()

    matrix_results = []
    for t in all_etfs:
        df = get_safe_df(raw_data, t).dropna(subset=['Close'])
        if len(df) < 25: continue
        close = df['Close']
        rsi_s = get_rsi(close, 14)
        r14_current, r14_1d_ago, r14_1w_ago = rsi_s.iloc[-1], rsi_s.iloc[-2] if len(rsi_s)>1 else rsi_s.iloc[-1], rsi_s.iloc[-6] if len(rsi_s)>5 else rsi_s.iloc[-1]
        
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        current_bbw = (((sma20 + 2*std20) - (sma20 - 2*std20)) / sma20 * 100).iloc[-1]
        cat = next((k for k, v in GLOBAL_MAP.items() if t in v), "Diğer")
        
        state, color = ("Aşırı Alım (Dağıtım)", "#ff3333") if r14_current > 70 else ("Vakum (Contrarian Fırsat)", "#00ff88") if r14_current < 35 else ("Sıkışma (VCP)", "#f1c40f")
        delta_icon = "⬆️" if r14_current > r14_1d_ago else "⬇️" if r14_current < r14_1d_ago else "➖"
        
        matrix_results.append({"Sektör": cat, "ETF": t, "RSI": r14_current, "RSI_1D": r14_1d_ago, "RSI_1W": r14_1w_ago, "BBW": current_bbw, "Durum": state, "Renk": color, "Delta_Icon": delta_icon})
    return pd.DataFrame(matrix_results)

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
            
            # Stratejik Katman: 3-Mum Eğim Kuralı (3-Bar Slope)
            slope_up = (close > close.shift(1)) & (close.shift(1) > close.shift(2))
            slope_dn = (close < close.shift(1)) & (close.shift(1) < close.shift(2))

            pct_1d = (close.iloc[-1] / close.iloc[-2] - 1) * 100 if len(close) > 1 else 0
            pct_1w = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) > 5 else 0

            # Unwritten Traps ✅/⛔ (1-Period Confirmed EMA Breakout)
            ema1_s3 = close.ewm(span=9, adjust=False).mean()
            v150_v_avg = vol.rolling(20).mean()
            bear_trap = (low < ema1_s3) & (close > ema1_s3) & (vol > v150_v_avg * 1.5)
            bull_trap = (high > ema1_s3) & (close < ema1_s3) & (vol > v150_v_avg * 1.5)

            # Sinerji / Efor Çizgisi
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

            # Whale Power (V700 Sinerji Motoru)
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

            # Vola Hole (Squeeze)
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

            # Sinyal Önceliklendirmesi (Asla tekrar etmez)
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

# DÜZELTME: Valuation Gap fonksiyonunda try/except eklendi ve fast_info önceliklendirildi.
@st.cache_data(ttl=600)
def fetch_valuation_data(ticker_list, bypass_stamp):
    funds = []
    for t in ticker_list:
        try:
            tk = yf.Ticker(t)
            # Market Cap fast_info'dan alınarak API bloklaması engellendi
            mc = tk.fast_info.get('marketCap', 0)
            
            # Info verisi Rate Limit yerse sadece onu sıfırlarız, uygulamayı çökertmeyiz
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
            time.sleep(0.1)  # Anti-Ban için ufak gecikme
        except: 
            funds.append({"Ticker": t, "MarketCap": 0, "PE": 0, "PS": 0, "PEG": 0})
    return pd.DataFrame(funds)

# --- STYLER YARDIMCILARI ---
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
st.markdown("Kararnamelerin rasyonel etki puanlarını, 12 kanallı esneklik yapısını ve **ŞAHANE V127** efor kırılımlarını birleştirir. Tüm sekmeler gerçek zamanlı güncellenebilir (Önbellek bypass'ı aktiftir).")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🌐 MAKRO & OPEX", 
    "🔋 OMNI-MATRIX (Piller)",
    "🦅 KUŞBAKIŞI SEKTÖR",
    "⚖️ VALUATION GAP (Çarpan Uçurumu)",
    "🦈 HAFTALIK MOMENTUM",
    "🚨 4H & OMNI RADAR",
    "🚀 FUTURE THEMES"
])

all_etfs_to_scan = list(MAIN_SECTORS.keys()) + list(ETF_INFO.keys())
portfolio_tickers = sorted(list(set([t for tkrs in ETF_INFO.values() for t in tkrs['stocks']])))
etf_name_map = {k: v for k, v in MAIN_SECTORS.items()}
for k, v in ETF_INFO.items(): etf_name_map[k] = f"Alt Sektör: {v['area']}"

# ---------------------------------------------------------
# TAB 1: MAKRO & OPEX & HABERLER
# ---------------------------------------------------------
with tab1:
    st.subheader("⚙️ Institutional Desk: Gelişmiş Makro Tetikleyiciler & RSS Canlı Haber Akışı")
    
    t_cols = st.columns(4)
    for i, trig in enumerate(SYSTEM_TRIGGERS.keys()):
        with t_cols[i]:
            if st.button(f"Senaryo: {trig}", use_container_width=True):
                st.session_state.active_trigger = trig

    col_chart, col_docs = st.columns([3, 2])
    with col_chart:
        st.markdown(f"#### 💸 **Sermaye Akış Rotası:** ({st.session_state.active_trigger})")
        draw_smart_money_flow(SYSTEM_TRIGGERS[st.session_state.active_trigger])
        
        st.divider()
        st.markdown("#### 🔊 Canlı Medya, Makro ve Sektör Tarayıcı (RSS)")
        
        # DÜZELTME: Sadece Haberler İçin Özel Update Butonu
        if st.button("🔄 Canlı Haberleri ve Etkilerini Güncelle", use_container_width=True):
            st.session_state.news_nonce = str(time.time())
            st.success("Küresel haber ağları yeniden tarandı!")
            
        with st.spinner("Küresel haber ağları ve yasa tasarıları taranıyor..."):
            df_news = pd.DataFrame(fetch_live_trump_news(st.session_state.news_nonce))
            if not df_news.empty:
                st.dataframe(
                    df_news, 
                    use_container_width=True, 
                    hide_index=True, 
                    column_config={"Link": st.column_config.LinkColumn("Haber Linki")}
                )
                
    with col_docs:
        st.markdown("""
        <div class="macro-def-box">
            <div class="macro-def-title">📚 Opex Pinning Nedir?</div>
            <p style="font-size: 0.85rem; color: #b0b0b0;">3. Cuma vade sonlarına (OpEx) yaklaşırken, yoğun opsiyon (Open Interest) olan seviyelerde Market Maker'ların fiyatı buraya hapsetmesi durumudur. Fiyat sıkışır, sahte kırılımlar (Whipsaw) üretir.</p>
        </div>
        <div class="macro-def-box">
            <div class="macro-def-title">📈 Gamma Squeeze Nedir?</div>
            <p style="font-size: 0.85rem; color: #b0b0b0;">Aşırı Call opsiyon alımı sonrası piyasa yapıcıların (Dealers) hedge amaçlı panikle spot hisse alması sonucu oluşan parabolik fiyat erimesi (Melt-Up) döngüsüdür.</p>
        </div>
        <div class="macro-def-box">
            <div class="macro-def-title">🌍 Geopolitical Shock</div>
            <p style="font-size: 0.85rem; color: #b0b0b0;">Savaş, ambargo veya tedarik zinciri çöküşünde sermayenin büyüme hisselerinden (Tech/AI) kaçıp sert emtiaya (Bakır, Petrol, Savunma) ve nakde akmasıdır.</p>
        </div>
        <div class="macro-def-box">
            <div class="macro-def-title">🏦 Liquidity Crunch (Fed)</div>
            <p style="font-size: 0.85rem; color: #b0b0b0;">Merkez bankasının şahin politikalarla (faiz artışı/QT) piyasadan doları çekmesidir. Yüksek çarpanlı (Zarar eden Tech/Kripto) varlıklarda devasa margin call satışları getirir.</p>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 2: OMNI-MATRIX (PİLLER)
# ---------------------------------------------------------
with tab2:
    st.subheader("🔋 Tüm Sektörler Pil Enerjisi (Gerçek Zamanlı)")
    if st.button("🔄 Pil Matrisini Zorla Güncelle (Cache Bypass)", key="btn_matrix"):
        st.session_state.battery_nonce = str(time.time())
        
    with st.spinner("Matrix API'den güncel veriler çekiliyor..."):
        df_m = fetch_matrix_data(st.session_state.battery_nonce)
        if not df_m.empty:
            theme_avg = df_m.groupby('Sektör')[['RSI', 'RSI_1D', 'RSI_1W']].mean().reset_index()
            cols = st.columns(4)
            for i, row in theme_avg.iterrows():
                with cols[i % 4]:
                    col = "#00ff88" if row['RSI'] > 60 else "#ff3333" if row['RSI'] < 40 else "#f1c40f"
                    draw_battery(row['Sektör'], row['RSI'], col, delta_1d=(row['RSI'] - row['RSI_1D']))

# ---------------------------------------------------------
# TAB 3: KUŞBAKIŞI SEKTÖR (1D)
# ---------------------------------------------------------
with tab3:
    st.subheader("🦅 Kuşbakışı ETF Sinyal Radarı")
    if st.button("🔄 Sektör Sinyallerini Güncelle", key="btn_bird"):
        st.session_state.battery_nonce = str(time.time())
        
    with st.spinner("1D Sinyaller Hesaplanıyor..."):
        df_bird = calculate_signals(all_etfs_to_scan, interval="1d", bypass_stamp=st.session_state.battery_nonce)
        if not df_bird.empty:
            df_bird['Kapsam'] = df_bird['Ticker'].map(etf_name_map)
            st.dataframe(
                df_bird[['Kapsam', 'Ticker', 'Sinyal', 'Efor', 'Fiyat', '1 Gün (%)', '1 Hafta (%)', 'Whale Power', 'Fusion']]
                .style.map(style_signals, subset=['Sinyal']).map(style_percentages, subset=['1 Gün (%)', '1 Hafta (%)']).map(style_efor, subset=['Efor']),
                use_container_width=True, hide_index=True
            )

# ---------------------------------------------------------
# TAB 4: VALUATION GAP (ÇARPAN UÇURUMU) SEKTÖR ORTALAMALI
# ---------------------------------------------------------
with tab4:
    st.subheader("⚖️ Tematik Çarpan Uçurumu ve Sektör Kıyaslaması")
    st.markdown("Seçilen sektördeki hisseleri tarar, **Sektör Ortalamasını** bulur ve tüm hisseleri bu ortalamaya göre kıyaslar.")
    
    col_v1, col_v2 = st.columns([1, 4])
    group_choice = col_v1.selectbox("Analiz Edilecek Tema:", list(ETF_INFO.keys()))
    
    if col_v2.button("🔄 Finansal Verileri ve Rasyoları Güncelle (API Çağrısı)", use_container_width=True):
        st.session_state.val_nonce = str(time.time())
        
    if group_choice:
        with st.spinner(f"{group_choice} rasyoları Yahoo Finance'tan canlı çekiliyor..."):
            tickers = ETF_INFO[group_choice]['stocks']
            df_val = fetch_valuation_data(tickers, st.session_state.val_nonce)
            
            if not df_val.empty:
                df_val = df_val[df_val['MarketCap'] > 0].sort_values(by='MarketCap', ascending=False)
                
                # Calculate Sector Averages (Ignoring Zeros due to Rate Limits)
                avg_pe = df_val[df_val['PE'] > 0]['PE'].mean()
                avg_ps = df_val[df_val['PS'] > 0]['PS'].mean()
                avg_peg = df_val[df_val['PEG'] > 0]['PEG'].mean()
                
                avg_pe = avg_pe if not pd.isna(avg_pe) else 0
                avg_ps = avg_ps if not pd.isna(avg_ps) else 0
                avg_peg = avg_peg if not pd.isna(avg_peg) else 0

                # DÜZELTME: API Limit uyarısı
                if avg_pe == 0 and avg_ps == 0:
                    st.warning("⚠️ Yahoo Finance anlık rasyo (F/K, PEG) verilerini reddetti (Rate Limit sebebiyle). Sadece Piyasa Değeri (Market Cap) üzerinden kıyaslama yapılıyor. Değerler 0 görünüyorsa API soğuma süresindedir.")

                st.markdown(f"""
                <div style="background-color: #111; padding: 15px; border-left: 5px solid #f1c40f; border-radius: 5px; margin-bottom: 20px;">
                    <h4 style="color: #f1c40f; margin:0;">📊 {ETF_INFO[group_choice]['area']} - Sektör Ortalamaları</h4>
                    <p style="margin: 5px 0 0 0; font-size: 1.1rem;">
                        Ortalama F/K (PE): <strong>{avg_pe:.1f}</strong> | 
                        Ortalama F/S (PS): <strong>{avg_ps:.1f}</strong> | 
                        Ortalama PEG: <strong>{avg_peg:.1f}</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)

                if not df_val.empty:
                    leader = df_val.iloc[0]
                    st.markdown(f"<div class='valuation-leader'>👑 Grup Lideri: {leader['Ticker']} (Değer: ${leader['MarketCap']/1e9:.1f}B | F/K: {leader['PE']:.1f} | F/S: {leader['PS']:.1f} | PEG: {leader['PEG']:.1f})</div><hr>", unsafe_allow_html=True)
                    
                    for i in range(1, len(df_val)):
                        row = df_val.iloc[i]
                        gap_mc = leader['MarketCap'] / row['MarketCap'] if row['MarketCap'] > 0 else 0
                        pe_diff = row['PE'] - avg_pe
                        pe_color = "#ff3333" if pe_diff > 0 else "#00ff88"
                        pe_txt = f"Ortalamanın {abs(pe_diff):.1f} {'Üzerinde (Pahalı)' if pe_diff > 0 else 'Altında (Ucuz)'}" if avg_pe > 0 and row['PE'] > 0 else "Rasyo verisi anlık çekilemedi"
                        
                        st.markdown(f'''
                            <div class="valuation-gap-card">
                                <h3><span class="valuation-laggard">{row['Ticker']}</span> <span style="font-size:0.9rem; color:#888;">(Değer: ${row['MarketCap']/1e9:.1f}B | F/K: {row['PE']:.1f} | PEG: {row['PEG']:.1f})</span></h3>
                                <div style="color: {pe_color}; font-weight:bold; margin-bottom: 5px;">Rasyo Analizi: {pe_txt}</div>
                                <div style="color: #e0e0e0; font-size: 0.9rem;">Lider {leader['Ticker']} ile arasındaki pazar hacmi farkı: <strong>{gap_mc:.1f}x</strong></div>
                            </div>
                        ''', unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 5, 6, 7: OMNI RADAR, HAFTALIK VE GELECEK
# ---------------------------------------------------------
with tab5:
    st.subheader("🦈 Haftalık Momentum-Gap Avcısı")
    if st.button("🔄 Haftalık Veriyi Yenile"): st.session_state.battery_nonce = str(time.time())
    with st.spinner("1W Sinyaller Hesaplanıyor..."):
        df_wk = calculate_signals(portfolio_tickers, interval="1wk", bypass_stamp=st.session_state.battery_nonce)
        if not df_wk.empty: st.dataframe(df_wk[['Ticker', 'Sinyal', 'Efor', 'Fiyat', '1 Hafta (%)', 'Whale Power']].style.map(style_signals, subset=['Sinyal']).map(style_efor, subset=['Efor']), use_container_width=True, hide_index=True)

with tab6:
    st.subheader("🚨 4H & Günlük OMNI RADAR (Tuzaklar ve Kırılımlar)")
    if st.button("🔄 Radar Verilerini Yenile"): st.session_state.battery_nonce = str(time.time())
    with st.spinner("Piyasa taranıyor (Bull/Bear Trap Onaylı)..."):
        df_radar = calculate_signals(portfolio_tickers, interval="1d", bypass_stamp=st.session_state.battery_nonce)
        if not df_radar.empty:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 🟢 POZİTİF (✅, Up, Re-Entry)")
                st.dataframe(df_radar[df_radar['Sinyal'].isin(['✅', '🚀 UP KIRILIM', '🔄 WHALE RE-ENTRY'])].style.map(style_signals, subset=['Sinyal']), use_container_width=True, hide_index=True)
            with c2:
                st.markdown("#### 🔴 NEGATİF (⛔, Down)")
                st.dataframe(df_radar[df_radar['Sinyal'].isin(['⛔', '🩸 DOWN KIRILIM'])].style.map(style_signals, subset=['Sinyal']), use_container_width=True, hide_index=True)

with tab7:
    st.subheader("🚀 FUTURE THEMES: Geleceğin Teknolojileri")
    if st.button("🔄 Gelecek Temalarını Yenile"): st.session_state.battery_nonce = str(time.time())
    future_tickers = list(set([t for tkrs in FUTURE_THEMES_MAP.values() for t in tkrs]))
    with st.spinner("Future Themes evreni taranıyor..."):
        df_future = calculate_signals(future_tickers, interval="1d", bypass_stamp=st.session_state.battery_nonce)
        if not df_future.empty: st.dataframe(df_future[['Ticker', 'Sinyal', 'Efor', 'Fiyat', '1 Gün (%)', 'Whale Power', 'Fusion']].style.map(style_signals, subset=['Sinyal']).map(style_efor, subset=['Efor']), use_container_width=True, hide_index=True)
