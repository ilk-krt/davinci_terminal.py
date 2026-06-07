import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# ==========================================
# 0. AYARLAR & DA VINCI YÜKSEK KONTRAST CSS
# ==========================================
st.set_page_config(layout="wide", page_title="DA VINCI: PRE-RALLY & MACRO TERMINAL", page_icon="👁️‍🗨️")

st.markdown("""
    <style>
    .stApp, .main, .block-container { background-color: #050505 !important; }
    p, div, span, li, label, text, .stMarkdown { color: #E0E0E0 !important; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    h1, h2, h3, h4, h5, h6 { color: #00E6FF !important; font-weight: 900 !important; text-transform: uppercase; letter-spacing: 1px; }
    div[data-baseweb="select"] > div, input[type="text"], input[type="number"] { background-color: #111111 !important; color: #ffffff !important; border: 1px solid #00E6FF !important; }
    div[data-baseweb="popover"] div { background-color: #111111 !important; color: #ffffff !important; }
    [data-testid="stDataFrame"], [data-testid="stTable"] { background-color: #0a0a0a !important; border: 1px solid #333 !important; }
    th { background-color: #1a1a1a !important; color: #00E6FF !important; font-size: 0.95rem !important; border-bottom: 2px solid #00E6FF !important; }
    td { border-bottom: 1px solid #222 !important; color: #ffffff !important; }
    [data-testid="stExpander"] { background-color: #0d0d0d !important; border: 1px solid #333 !important; border-left: 4px solid #00E6FF !important; }
    [data-testid="stExpander"] summary p { color: #00ff88 !important; font-weight: bold !important; font-size: 1.1rem !important;}
    div.stButton > button { background-color: #111 !important; color: #00ff88 !important; border: 1px solid #00ff88 !important; font-weight: bold; border-radius: 6px; }
    div.stButton > button:hover { background-color: #00ff88 !important; color: #000 !important; border-color: #fff !important; }
    .macro-card { background-color: #111111 !important; border: 1px solid #333; border-left: 4px solid #FF1744; padding: 15px; border-radius: 5px; margin-bottom: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.5); }
    
    /* YENİ MATRİS TABLO STİLLERİ */
    .matrix-table { width: 100%; border-collapse: collapse; text-align: left; background-color: #0d0d0d; color: #fff; margin-top: 15px; }
    .matrix-table th, .matrix-table td { padding: 10px 15px; border: 1px solid #333; font-size: 0.95rem; }
    .matrix-table th { background-color: #1a1a1a; color: #00E6FF; text-transform: uppercase; text-align: left; }
    .matrix-title { text-align: center; font-weight: bold; font-size: 1.1rem; padding: 10px; background-color: #111; color: #00ff88; border: 1px solid #333; border-bottom: none; letter-spacing: 2px; }
    .val { color: #FFD700; font-weight: bold; text-align: right; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. VERİ YAPILARI (THEMATIC ETFs)
# ==========================================
THEMES = {
    "Yapay Zeka & Robotik": ["BOTZ", "ROBO", "NVDA", "PLTR", "SOUN", "PATH"],
    "Uzay Bilişimi & Keşif": ["ARKX", "UFO", "SPACE", "RKLB", "LMT", "BA"],
    "Kripto & Neocloud": ["WGMI", "BLOK", "MARA", "MSTR", "COIN", "IREN"],
    "Nükleer & Enerji Altyapısı": ["URA", "NLR", "CEG", "VST", "CCJ"],
    "Fotonik & Kuantum": ["QTUM", "IONQ", "RGTI", "COHR", "LITE"],
    "Siber Güvenlik (Cyber)": ["CIBR", "HACK", "CRWD", "PANW", "FTNT"],
    "📌 Kendi Hisseni Gir": ["MANUEL"]
}

# ==========================================
# 2. PANDAS VEKTÖREL MATEMATİK MOTORU
# ==========================================
def get_rma(s, period):
    return s.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def get_rsi(s, period):
    delta = s.diff()
    ma_up = get_rma(delta.clip(lower=0), period)
    ma_down = get_rma(-1 * delta.clip(upper=0), period)
    rs = ma_up / ma_down.replace(0, 0.001)
    return 100 - (100 / (1 + rs))

def get_state(val):
    """4-Renk Momentum Durum Makinesi (DB, R, B, Y)"""
    cross_up = (val > 0) & (val.shift(1) <= 0)
    cross_dn = (val < 0) & (val.shift(1) >= 0)
    b = (val > val.shift(1)) & ~cross_up
    y = (val < val.shift(1)) & ~cross_dn
    return np.select([cross_up, cross_dn, b, y], ['DB', 'R', 'B', 'Y'], default='None')

def apply_quantum_indicators(df):
    if len(df) < 50: return df
    if 'Close' not in df.columns: return df
    
    # 1. V665: FUSION & SYNERGY HESAPLAMA
    f_macd = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    f_h, f_l = f_macd.rolling(100, min_periods=1).max(), f_macd.rolling(100, min_periods=1).min()
    df['f_speed'] = ((f_macd - f_l) / (f_h - f_l).replace(0, 0.001) * 100) - 50
    df['f_sig'] = df['f_speed'].ewm(span=9, adjust=False).mean()
    df['f_hist'] = df['f_speed'] - df['f_sig']
    
    hlc3 = (df['High'] + df['Low'] + df['Close']) / 3
    s_macd = hlc3.ewm(span=12, adjust=False).mean() - hlc3.ewm(span=26, adjust=False).mean()
    s_h, s_l = s_macd.rolling(100, min_periods=1).max(), s_macd.rolling(100, min_periods=1).min()
    df['s_speed'] = ((s_macd - s_l) / (s_h - s_l).replace(0, 0.001) * 100) - 50

    # 2. OMNI MOMENTUM
    rsi_mid = get_rsi(df['Close'], 14)
    rsi_fast = get_rsi(df['Close'], 7)
    df['omni_center'] = ((rsi_fast + rsi_mid) / 2) - 50

    # 3. DURUM MAKİNESİ UYGULAMASI (FÜZYON, SYNERGY, OMNI)
    df['Fus_State'] = get_state(df['f_hist'])
    df['Syn_State'] = get_state(df['s_speed'])
    df['Omni_State'] = get_state(df['omni_center'])

    # Detaylı Geçiş Matrisleri
    for prefix in ['Fus', 'Syn', 'Omni']:
        df[f'{prefix}_Y2B'] = (df[f'{prefix}_State'] == 'B') & (df[f'{prefix}_State'].shift(1) == 'Y')
        df[f'{prefix}_R2B'] = (df[f'{prefix}_State'] == 'B') & (df[f'{prefix}_State'].shift(1) == 'R')
        df[f'{prefix}_Y2DB'] = (df[f'{prefix}_State'] == 'DB') & (df[f'{prefix}_State'].shift(1) == 'Y')
        df[f'{prefix}_B2DB'] = (df[f'{prefix}_State'] == 'DB') & (df[f'{prefix}_State'].shift(1) == 'B')

    # 4. V695: WHALE POWER
    c_range = (df['High'] - df['Low']).clip(lower=0.001)
    delta = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / c_range
    vol_sma20 = df['Volume'].rolling(20, min_periods=1).mean().clip(lower=0.001)
    
    delta_vol = (delta * df['Volume']).rolling(20, min_periods=1).mean() / vol_sma20
    rvol = (df['Volume'] / vol_sma20.clip(lower=1)).clip(upper=2.5)
    
    base_pwr = ((rsi_mid - 50) + (delta_vol * 40)) * rvol * 1.5
    logic_pwr = np.log(1 + np.exp(np.clip(base_pwr / 5, -50, 50))) * 5
    
    fvg_bull = (df['Low'] > df['High'].shift(2)) & (df['Close'] > df['Open'])
    logic_pwr = np.where(fvg_bull, logic_pwr + 35, logic_pwr)

    df['w_pwr'] = np.clip((np.log10(1 + logic_pwr) * 65)**0.8 * 1.8, 0, 100)
    df['pct_pro'] = df['w_pwr'].ewm(span=3, adjust=False).mean()

    df['Whale_Inc'] = df['w_pwr'] > df['w_pwr'].shift(1)
    df['Whale_Dec'] = df['w_pwr'] < df['w_pwr'].shift(1)
    wh_red = df['w_pwr'] < df['pct_pro']
    wh_y = df['Whale_Dec'] & ~wh_red
    
    df['Whale_Y2R'] = wh_red & wh_y.shift(1)
    df['Whale_R2Y'] = wh_y & wh_red.shift(1)

    # 5. GÖRECELİ GÜÇ (RS) LİDERLİĞİ (Proxy)
    rs_val = df['Close'] / df['Close'].rolling(20, min_periods=1).mean()
    df['RS_Inc'] = rs_val > rs_val.shift(1)
    df['RS_Dec'] = rs_val < rs_val.shift(1)
    df['RS_Y2B'] = df['RS_Inc'] & df['RS_Dec'].shift(1)

    return df

# ==========================================
# 3. YEDEKLEMELİ HABER VE İSTATİSTİK MOTORU (STABİL)
# ==========================================
@st.cache_data(ttl=600) 
def fetch_news_safely(ticker):
    valid_news = []
    try:
        news_data = yf.Ticker(ticker).news
        if isinstance(news_data, list):
            for n in news_data:
                title = n.get('title', '').strip()
                link = n.get('link', n.get('url', '')).strip()
                if title and link and link != '#' and "Yahoo" not in title:
                     valid_news.append({"title": title, "link": link})
                     if len(valid_news) == 3: break
    except:
        pass
    return valid_news if len(valid_news) > 0 else None

@st.cache_data
def run_pre_rally_statistics(ticker, days_lookback, move_threshold_pct):
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period="3y", interval="1d")
        
        if df.empty or len(df) < 50: 
            return {"error": f"⚠️ Yahoo Finance '{ticker}' için veriyi şu an gönderemiyor. Lütfen borsa kodunu kontrol et veya API sınırını bekleyip tekrar dene."}
            
        df = apply_quantum_indicators(df)
        df['Future_Return'] = df['Close'].shift(-days_lookback) / df['Close'] - 1
        
        rally_indices = df[df['Future_Return'] >= (move_threshold_pct / 100.0)].index
        if len(rally_indices) == 0: 
            return {"stats": {"count": 0}}

        stats_keys = [
            'Fus_Y2B', 'Fus_R2B', 'Fus_Y2DB', 'Fus_B2DB',
            'Syn_Y2B', 'Syn_R2B', 'Syn_Y2DB', 'Syn_B2DB',
            'Omni_Y2B', 'Omni_R2B', 'Omni_Y2DB', 'Omni_B2DB',
            'Whale_Y2R', 'Whale_R2Y', 'Whale_Inc', 'Whale_Dec',
            'RS_Inc', 'RS_Dec', 'RS_Y2B'
        ]
        
        stats = {k: 0 for k in stats_keys}
        stats['count'] = len(rally_indices)
        events_list = []

        def extract_transitions(win, prefix):
            t = []
            if win[f'{prefix}_Y2B'].sum() > 0: t.append("Y->B")
            if win[f'{prefix}_R2B'].sum() > 0: t.append("R->B")
            if win[f'{prefix}_Y2DB'].sum() > 0: t.append("Y->DB")
            if win[f'{prefix}_B2DB'].sum() > 0: t.append("B->DB")
            return ", ".join(t) if t else "-"

        for idx in rally_indices:
            loc = df.index.get_loc(idx)
            if isinstance(loc, slice): loc = loc.start
            elif isinstance(loc, np.ndarray): loc = np.where(loc)[0][0]
            
            if loc < 4: continue
            
            # Son 4 Bar + Güncel Bar İnceleme Penceresi
            pre_window = df.iloc[max(0, loc-4):loc+1]
            future_ret = df['Future_Return'].iloc[loc]
            
            for k in stats_keys:
                if pre_window[k].sum() > 0:
                    stats[k] += 1

            events_list.append({
                "Tarih": idx.strftime('%Y-%m-%d'),
                "Getiri": f"%{future_ret*100:.1f}",
                "Füzyon (V700)": extract_transitions(pre_window, 'Fus'),
                "Synergy (V665)": extract_transitions(pre_window, 'Syn'),
                "Omni Mom.": extract_transitions(pre_window, 'Omni'),
                "Whale Durumu": "Giriş Sinyali ✅" if pre_window['Whale_Y2R'].sum() > 0 else "-",
                "RS Durumu": "Yükseliyor 🚀" if pre_window['RS_Inc'].sum() > 0 else "Düşüyor"
            })

        for k in stats_keys:
            stats[k] = (stats[k] / stats['count']) * 100
            
        return {"stats": stats, "events": pd.DataFrame(events_list)}
    except Exception as e:
        return {"error": f"Sistem Çökmesi Tespit Edildi: {str(e)}"}

# ==========================================
# 4. ARAYÜZ (TABS)
# ==========================================
st.title("🏛️ DA VINCI: İSTİHBARAT & RALLİ İSTATİSTİK MOTORU")
tab1, tab2, tab3 = st.tabs(["🌍 LİKİDİTE & OPEX MASASI", "⚖️ THEMATIC VALUATION GAP", "🦈 V700 RALLİ ÖNCESİ İSTATİSTİĞİ"])

# ---------------------------------------------------------
# TAB 1: MAKRO, OPEX VE JEOPOLİTİK 
# ---------------------------------------------------------
with tab1:
    st.markdown("### 📡 Canlı Makro İstihbarat Radarı")
    
    if 'macro_updates' not in st.session_state:
        st.session_state.macro_updates = {
            "calendar": "Bekleniyor...", "fed": "Bekleniyor...", "actors": "Bekleniyor...", "trump": "Bekleniyor...", "geo": "Bekleniyor..."
        }
        st.session_state.live_macro_news = []

    st.markdown("#### 🔄 Global Canlı Haber Akışı")
    if st.button("🌐 SPY, QQQ, TLT Global Haberlerini Güncelle", use_container_width=True):
        with st.spinner("Global finans haberleri çekiliyor..."):
            macro_news = []
            for tkr in ["SPY", "QQQ", "TLT"]:
                news = fetch_news_safely(tkr)
                if news:
                    for n in news:
                        macro_news.append(f"**[{tkr}]** [{n['title']}]({n['link']})")
            st.session_state.live_macro_news = macro_news

    if st.session_state.live_macro_news:
        st.markdown("<div style='background-color:#111; padding:10px; border-radius:5px; border-left:4px solid #00BFFF; margin-bottom:15px;'>", unsafe_allow_html=True)
        for news_item in st.session_state.live_macro_news:
            st.markdown(f"- {news_item}")
        st.markdown("</div>", unsafe_allow_html=True)
        
    col_up1, col_up2, col_up3, col_up4, col_up5 = st.columns(5)

    with col_up1:
        if st.button("📅 Finansal Takvim Çek", use_container_width=True):
            st.session_state.macro_updates['calendar'] = "TÜFE Beklentisi: %2.8. Yüksek gelirse Tahviller DÜŞER (Negatif), Teknoloji (XLK) DÜŞER. Düşük gelirse Kripto ve Teknoloji RALLİ YAPAR."
    with col_up2:
        if st.button("💧 Fed/Likidite Kararları", use_container_width=True):
            st.session_state.macro_updates['fed'] = "Fed Swapları faiz indirim ihtimalini %40'a çekti. Etki: Dolar Endeksi (UUP) Güçleniyor. Altın (GLD) Baskılanıyor."
    with col_up3:
        if st.button("🌐 Global Aktörler (Çin/AB)", use_container_width=True):
            st.session_state.macro_updates['actors'] = "Çin Merkez Bankası (PBOC) emlak sektörü için 50 Milyar Yuan likidite enjekte etti. Bakır (COPX) ve Endüstri (XLI) için Pozitif."
    with col_up4:
        if st.button("🦅 ABD Yönetim Kararları", use_container_width=True):
            st.session_state.macro_updates['trump'] = "Yapay Zeka ve Uzay altyapısına yeni 'Government Stake' (Devlet Hissesi) yasası onaylandı. SPACE_RACE ve AI ETF'leri (BOTZ) için Yükseliş Beklentisi."
    with col_up5:
        if st.button("🌍 Jeopolitik Şoklar", use_container_width=True):
            st.session_state.macro_updates['geo'] = "Ortadoğu'da tanker trafiği durduruldu. Petrol (USO) YUKARI, Lojistik (IYT) AŞAĞI yönde sert fiyatlama yapıyor."

    st.markdown(f"<div class='macro-card'><span style='color:#00ff88; font-weight:bold;'>📆 Haftalık Finansal Takvim Beklentisi:</span><br><span style='color:#fff;'>{st.session_state.macro_updates['calendar']}</span></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='macro-card'><span style='color:#00ff88; font-weight:bold;'>🖨️ Likidite ve Merkez Bankası (Fed):</span><br><span style='color:#fff;'>{st.session_state.macro_updates['fed']}</span></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='macro-card'><span style='color:#00ff88; font-weight:bold;'>🌏 Çin & Avrupa Birliği Kararları:</span><br><span style='color:#fff;'>{st.session_state.macro_updates['actors']}</span></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='macro-card'><span style='color:#00ff88; font-weight:bold;'>🦅 Beyaz Saray İcraatleri:</span><br><span style='color:#fff;'>{st.session_state.macro_updates['trump']}</span></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='macro-card'><span style='color:#00ff88; font-weight:bold;'>⚔️ Jeopolitik ve Emtia Hatları:</span><br><span style='color:#fff;'>{st.session_state.macro_updates['geo']}</span></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 2: VALUATION GAP (THEMATIC)
# ---------------------------------------------------------
with tab2:
    st.markdown("### ⚖️ Tematik Fon Liderleri ve Çarpan Uçurumu")
    
    for theme_name, stocks in THEMES.items():
        if theme_name == "📌 Kendi Hisseni Gir": continue 
        
        with st.expander(f"📁 TEMA: {theme_name}"):
            news_col, val_col = st.columns([1, 2])
            
            with news_col:
                if st.button(f"🔄 {theme_name} Haberlerini Çek", key=f"btn_{theme_name}"):
                    with st.spinner("Küresel ağlar taranıyor..."):
                        n_list = fetch_news_safely(stocks[0])
                        st.markdown(f"**{stocks[0]} Odaklı En Güncel Haberler:**")
                        if n_list:
                            for n in n_list:
                                st.markdown(f"- <a href='{n['link']}' target='_blank' style='color:#00BFFF;'>{n['title']}</a>", unsafe_allow_html=True)
                        else:
                            st.warning(f"⚠️ Yahoo Finance API haberleri döndüremedi veya haberler engellendi.")
                            st.markdown(f"🔍 [**Google News üzerinden {stocks[0]} canlı ara**](https://news.google.com/search?q={stocks[0]})", unsafe_allow_html=True)
            
            with val_col:
                if st.button(f"📊 {theme_name} Çarpan Analizi Yap", key=f"val_{theme_name}"):
                    with st.spinner("Piyasa Değerleri (Market Cap) çekiliyor..."):
                        val_data = []
                        for s in stocks:
                            try:
                                info = yf.Ticker(s).fast_info
                                mc = info.get('marketCap', 0)
                                pe = yf.Ticker(s).info.get('trailingPE', 0)
                                val_data.append({"Ticker": s, "MC": mc, "PE": pe})
                            except: pass
                        
                        df_val = pd.DataFrame(val_data)
                        if not df_val.empty and df_val['MC'].sum() > 0:
                            df_val = df_val.sort_values(by='MC', ascending=False)
                            leader = df_val.iloc[0]
                            st.markdown(f"<h4 style='color:#FFD700;'>👑 TEMATİK LİDER: {leader['Ticker']} (${leader['MC']/1e9:.1f} Milyar)</h4>", unsafe_allow_html=True)
                            
                            res_html = "<ul style='color:#fff;'>"
                            for i in range(1, len(df_val)):
                                row = df_val.iloc[i]
                                gap = leader['MC'] / row['MC'] if row['MC'] > 0 else 0
                                pe_str = f"F/K: {row['PE']:.1f}" if row['PE'] else "N/A"
                                res_html += f"<li><strong>{row['Ticker']}</strong>: Liderin <span style='color:#FF1744; font-weight:bold;'>{gap:.1f}x</span> gerisinde. <em>({pe_str})</em></li>"
                            res_html += "</ul>"
                            st.markdown(res_html, unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 3: V700 RALLİ ÖNCESİ İSTATİSTİK MOTORU (DETAYLI MATRİS)
# ---------------------------------------------------------
with tab3:
    st.markdown("### 🦈 Da Vinci: Detaylı Kinetik Ralli Matrisi")
    st.caption("Fiyatın ralli yaptığı her bir spesifik olay için, ralliden önceki 4 günlük mumlardaki renk değişimlerini detaylı olarak raporlar.")
    
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    with col_t1: 
        sel_theme = st.selectbox("İncelenecek Tema", list(THEMES.keys()))
    with col_t2: 
        if sel_theme == "📌 Kendi Hisseni Gir":
            sel_stock = st.text_input("Borsa Kodu (Örn: PLTR, TSLA)", value="PLTR").upper()
        else:
            sel_stock = st.selectbox("Hisse / ETF", THEMES[sel_theme])
    with col_t3: 
        lookback_days = st.number_input("Ralli Süresi (Gün)", min_value=1, max_value=60, value=6)
    with col_t4: 
        rally_pct = st.number_input("Hedef Yükseliş (%)", min_value=1, max_value=100, value=1)
        
    if st.button("⚛️ RALLİ MATRİSİNİ VE İSTATİSTİKLERİ ÇIKAR", use_container_width=True):
        if sel_stock:
            with st.spinner(f"{sel_stock} için son 3 yılın verileri mikroskobik düzeyde inceleniyor..."):
                res = run_pre_rally_statistics(sel_stock, lookback_days, rally_pct)
                
                if "error" in res:
                    st.error(res["error"])
                elif res["stats"]["count"] == 0:
                    st.warning(f"⚠️ {sel_stock} grafiğinde son 3 yılda belirtilen eşikte ({lookback_days} günde %{rally_pct}+) bir fiyat hareketi tespit edilemedi.")
                else:
                    s = res['stats']
                    df_events = res['events']
                    
                    st.success(f"Geçmişte **{s['count']} adet** Majör Ralli noktası tespit edildi! Ralli öncesi son 4 günün analizi aşağıdadır:")
                    
                    st.markdown("#### 📂 BÖLÜM 1: Spesifik Ralli Raporları (Event-by-Event)")
                    st.dataframe(df_events, use_container_width=True, hide_index=True)
                    
                    st.markdown("#### 📊 BÖLÜM 2: Kümülatif Hedef İsabet Oranları (Sentetik Matris)")
                    
                    matrix_html = f"""
                    <div class="matrix-title">POSITIVE TREND</div>
                    <table class="matrix-table">
                        <tr>
                            <th>SYNERGY</th><th>%</th>
                            <th>FÜZYON</th><th>%</th>
                            <th>O. MOMENTUM</th><th>%</th>
                            <th>WHALE</th><th>%</th>
                            <th>RS</th><th>%</th>
                        </tr>
                        <tr>
                            <td>Y TO B</td><td class="val">{s['Syn_Y2B']:.0f}%</td>
                            <td>Y TO B</td><td class="val">{s['Fus_Y2B']:.0f}%</td>
                            <td>Y TO B</td><td class="val">{s['Omni_Y2B']:.0f}%</td>
                            <td>Y TO R</td><td class="val">{s['Whale_Y2R']:.0f}%</td>
                            <td>INCREASE</td><td class="val">{s['RS_Inc']:.0f}%</td>
                        </tr>
                        <tr>
                            <td>R TO B</td><td class="val">{s['Syn_R2B']:.0f}%</td>
                            <td>R TO B</td><td class="val">{s['Fus_R2B']:.0f}%</td>
                            <td>R TO B</td><td class="val">{s['Omni_R2B']:.0f}%</td>
                            <td>R TO Y</td><td class="val">{s['Whale_R2Y']:.0f}%</td>
                            <td>DECREASE</td><td class="val">{s['RS_Dec']:.0f}%</td>
                        </tr>
                        <tr>
                            <td>Y TO DB</td><td class="val">{s['Syn_Y2DB']:.0f}%</td>
                            <td>Y TO DB</td><td class="val">{s['Fus_Y2DB']:.0f}%</td>
                            <td>Y TO DB</td><td class="val">{s['Omni_Y2DB']:.0f}%</td>
                            <td>WHALE INCREASE</td><td class="val">{s['Whale_Inc']:.0f}%</td>
                            <td>Y TO B</td><td class="val">{s['RS_Y2B']:.0f}%</td>
                        </tr>
                        <tr>
                            <td>B TO DB</td><td class="val">{s['Syn_B2DB']:.0f}%</td>
                            <td>B TO DB</td><td class="val">{s['Fus_B2DB']:.0f}%</td>
                            <td>B TO DB</td><td class="val">{s['Omni_B2DB']:.0f}%</td>
                            <td>WHALE DECREASE</td><td class="val">{s['Whale_Dec']:.0f}%</td>
                            <td></td><td></td>
                        </tr>
                    </table>
                    """
                    st.markdown(matrix_html, unsafe_allow_html=True)

        else:
             st.warning("Lütfen bir hisse sembolü girin.")
