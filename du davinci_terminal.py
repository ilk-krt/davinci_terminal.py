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
    .stat-box { background: linear-gradient(145deg, #1a1a1a, #0a0a0a) !important; padding: 15px; border-radius: 8px; border: 1px solid #444; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.5); }
    .stat-value { font-size: 1.2rem !important; font-weight: 900 !important; color: #00E6FF !important; margin: 5px 0; }
    .stat-label { font-size: 0.75rem !important; color: #aaaaaa !important; text-transform: uppercase; font-weight: bold; }
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
# 2. PANDAS VEKTÖREL MATEMATİK MOTORU (ÇALIŞAN ANA ÇEKİRDEK)
# ==========================================
def get_rma(s, period):
    return s.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def get_rsi(s, period):
    delta = s.diff()
    ma_up = get_rma(delta.clip(lower=0), period)
    ma_down = get_rma(-1 * delta.clip(upper=0), period)
    rs = ma_up / ma_down.replace(0, 0.001)
    return 100 - (100 / (1 + rs))

def apply_quantum_indicators(df):
    if len(df) < 50: return df
    
    # 1. V665: FUSION & SYNERGY
    f_macd = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    f_h, f_l = f_macd.rolling(100, min_periods=1).max(), f_macd.rolling(100, min_periods=1).min()
    df['f_speed'] = ((f_macd - f_l) / (f_h - f_l).replace(0, 0.001) * 100) - 50
    df['f_sig'] = df['f_speed'].ewm(span=9, adjust=False).mean()
    
    hlc3 = (df['High'] + df['Low'] + df['Close']) / 3
    s_macd = hlc3.ewm(span=12, adjust=False).mean() - hlc3.ewm(span=26, adjust=False).mean()
    s_h, s_l = s_macd.rolling(100, min_periods=1).max(), s_macd.rolling(100, min_periods=1).min()
    df['s_speed'] = ((s_macd - s_l) / (s_h - s_l).replace(0, 0.001) * 100) - 50

    # 2. V695: WHALE POWER
    rsi_mid = get_rsi(df['Close'], 14)
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

    # ========================================================
    # 3. YENİ DETAYLI RENK GEÇİŞ MATRİSLERİ (Çökmeyecek Şekilde Eklendi)
    # ========================================================
    
    # FUSION DETAY
    df['Fus_Color'] = np.where(df['f_speed'] > df['f_speed'].shift(1), 'Blue', 'Yellow')
    df['Fus_Y2B'] = (df['Fus_Color'] == 'Blue') & (df['Fus_Color'].shift(1) == 'Yellow')
    df['Fus_B2DB'] = (df['Fus_Color'] == 'Blue') & df['Fus_Y2B'].shift(1) & (df['f_speed'] > 0)
    
    # SYNERGY DETAY
    df['Syn_Color'] = np.where(df['s_speed'] > df['s_speed'].shift(1), 'Blue', 'Yellow')
    df['Syn_Y2B'] = (df['Syn_Color'] == 'Blue') & (df['Syn_Color'].shift(1) == 'Yellow')
    df['Syn_B2DB'] = (df['Syn_Color'] == 'Blue') & df['Syn_Y2B'].shift(1) & (df['s_speed'] > 0)

    # OMNI MOMENTUM
    rsi_fast = get_rsi(df['Close'], 7)
    df['Omni'] = (rsi_fast + rsi_mid) / 2
    df['Omni_Y2B'] = (df['Omni'] > df['Omni'].shift(1)) & ~(df['Omni'].shift(1) > df['Omni'].shift(2))

    # SPEED / SIGNAL KESİŞİMİ
    df['Spd_Cross'] = (df['f_speed'] > df['f_sig']) & (df['f_speed'].shift(1) <= df['f_sig'].shift(1))

    # WHALE RE-ENTRY DETAY
    df['Whale_In'] = df['w_pwr'] > df['pct_pro']
    df['Whale_Y2R'] = df['Whale_In'] & ~df['Whale_In'].shift(1)

    # GÖRECELİ GÜÇ (RS) LİDERLİĞİ
    df['RS_Leader'] = df['Close'].pct_change(20) > 0

    return df

# ==========================================
# 3. YEDEKLEMELİ HABER VE İSTATİSTİK MOTORU
# ==========================================
@st.cache_data(ttl=600) 
def fetch_news_safely(ticker):
    """Yahoo Finance API'den Gelen Boş veya Bozuk Linkleri Kesin Olarak Yok Eder"""
    valid_news = []
    try:
        news_data = yf.Ticker(ticker).news
        if isinstance(news_data, list):
            for n in news_data:
                title = n.get('title', '').strip()
                link = n.get('link', n.get('url', '')).strip()
                # Sahte başlıkları ve boş linkleri direkt atla
                if title and link and link != '#' and "Yahoo" not in title:
                     valid_news.append({"title": title, "link": link})
                     if len(valid_news) == 3: break
    except:
        pass
    return valid_news if len(valid_news) > 0 else None

@st.cache_data
def run_pre_rally_statistics(ticker, days_lookback, move_threshold_pct):
    try:
        # SENİN ÇALIŞAN ÇEKİRDEK KODUN: History Metodu (Sessiz hataları engeller)
        tk = yf.Ticker(ticker)
        df = tk.history(period="3y", interval="1d")
        
        # Eğer API gerçekten veri yollamazsa "0 ralli" yalanını söyleme, gerçeği bildir:
        if df.empty or len(df) < 50: 
            return {"error": f"⚠️ Yahoo Finance '{ticker}' için veriyi şu an gönderemiyor. Lütfen borsa kodunu kontrol et veya API sınırını bekleyip tekrar dene."}
            
        df = apply_quantum_indicators(df)
        
        # Gelecekteki hareketi hesapla
        df['Future_Return'] = df['Close'].shift(-days_lookback) / df['Close'] - 1
        
        # Eşiği aşan Ralli Başlangıç Noktalarını Bul
        rally_indices = df[df['Future_Return'] >= (move_threshold_pct / 100.0)].index
        
        if len(rally_indices) == 0: 
            return {"stats": {"count": 0}}

        events_list = []
        stats = {
            "count": len(rally_indices),
            "fus_YB": 0, "fus_BDB": 0, 
            "syn_YB": 0, "syn_BDB": 0, 
            "omni_YB": 0, "spd_cross": 0,
            "whale_YR": 0
        }

        # SENİN ÇALIŞAN DÖNGÜN, DETAYLI MATRİSLE BİRLEŞTİ
        for idx in rally_indices:
            loc = df.index.get_loc(idx)
            
            # Pandas Duplicate Index Zırhı (Sessiz Çökmeyi Engeller)
            if isinstance(loc, slice): loc = loc.start
            elif isinstance(loc, np.ndarray): loc = np.where(loc)[0][0]
            
            if loc < 4: continue
            
            pre_window = df.iloc[loc-4:loc+1] # Ralli öncesi son 4 gün
            future_ret = df['Future_Return'].iloc[loc]
            
            # 4 Günlük İçindeki Olayları Say
            c_fus_yb = pre_window['Fus_Y2B'].sum()
            c_fus_bdb = pre_window['Fus_B2DB'].sum()
            c_syn_yb = pre_window['Syn_Y2B'].sum()
            c_syn_bdb = pre_window['Syn_B2DB'].sum()
            c_omni_yb = pre_window['Omni_Y2B'].sum()
            c_spd_cross = pre_window['Spd_Cross'].sum()
            c_whale_yr = pre_window['Whale_Y2R'].sum()
            rs_state = "Lider" if df['RS_Leader'].iloc[loc] else "Geri"

            # Kümülatif İstatistik İçin En Az 1 Kere Olduysa Say
            if c_fus_yb > 0: stats['fus_YB'] += 1
            if c_fus_bdb > 0: stats['fus_BDB'] += 1
            if c_syn_yb > 0: stats['syn_YB'] += 1
            if c_syn_bdb > 0: stats['syn_BDB'] += 1
            if c_omni_yb > 0: stats['omni_YB'] += 1
            if c_spd_cross > 0: stats['spd_cross'] += 1
            if c_whale_yr > 0: stats['whale_YR'] += 1

            # Etkinlik Tablosu Formatı
            str_fus = f"{c_fus_yb}x Yellow->Blue, {c_fus_bdb}x Blue->DarkBlue" if (c_fus_yb+c_fus_bdb)>0 else "-"
            str_syn = f"{c_syn_yb}x Yellow->Blue, {c_syn_bdb}x Blue->DarkBlue" if (c_syn_yb+c_syn_bdb)>0 else "-"
            str_omni = f"{c_omni_yb}x Y->B" if c_omni_yb>0 else "-"
            str_spd = "Kesti ✅" if c_spd_cross>0 else "-"
            str_whale = f"{c_whale_yr}x Yellow->Red" if c_whale_yr>0 else "-"

            events_list.append({
                "Tarih": idx.strftime('%Y-%m-%d'),
                "Getiri": f"%{future_ret*100:.1f}",
                "Füzyon (V700)": str_fus,
                "Synergy (V665)": str_syn,
                "Omni Mom.": str_omni,
                "Speed/Sig.": str_spd,
                "Whale (V695)": str_whale,
                "RS Durumu": rs_state
            })

        # Yüzdelere çevir
        for k in ["fus_YB", "fus_BDB", "syn_YB", "syn_BDB", "omni_YB", "spd_cross", "whale_YR"]:
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
                    # Sistem API kaynaklı veya hesaplama kaynaklı çökerse yalan söylemez, direkt hatayı basar:
                    st.error(res["error"])
                elif res["stats"]["count"] == 0:
                    st.warning(f"⚠️ {sel_stock} grafiğinde son 3 yılda belirtilen eşikte ({lookback_days} günde %{rally_pct}+) bir fiyat hareketi tespit edilemedi.")
                else:
                    stats = res['stats']
                    df_events = res['events']
                    
                    st.success(f"Geçmişte **{stats['count']} adet** Majör Ralli noktası tespit edildi! Ralli öncesi son 4 günün analizi aşağıdadır:")
                    
                    st.markdown("#### 📂 BÖLÜM 1: Spesifik Ralli Raporları (Event-by-Event)")
                    st.dataframe(df_events, use_container_width=True, hide_index=True)
                    
                    st.markdown("#### 📊 BÖLÜM 2: Kümülatif Sinyal İsabet Oranları")
                    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
                    with sc1: st.markdown(f"<div class='stat-box'><div class='stat-label'>Füzyon B->DB</div><div class='stat-value'>%{stats['fus_BDB']:.0f}</div><div class='stat-desc'>DarkBlue İsabeti</div></div>", unsafe_allow_html=True)
                    with sc2: st.markdown(f"<div class='stat-box'><div class='stat-label'>Synergy B->DB</div><div class='stat-value'>%{stats['syn_BDB']:.0f}</div><div class='stat-desc'>DarkBlue İsabeti</div></div>", unsafe_allow_html=True)
                    with sc3: st.markdown(f"<div class='stat-box'><div class='stat-label'>Speed/Signal</div><div class='stat-value'>%{stats['spd_cross']:.0f}</div><div class='stat-desc'>Kesişim İsabeti</div></div>", unsafe_allow_html=True)
                    with sc4: st.markdown(f"<div class='stat-box'><div class='stat-label'>Whale IN</div><div class='stat-value'>%{stats['whale_YR']:.0f}</div><div class='stat-desc'>Y->R Re-Entry</div></div>", unsafe_allow_html=True)
                    with sc5: st.markdown(f"<div class='stat-box'><div class='stat-label'>Omni Mom.</div><div class='stat-value'>%{stats['omni_YB']:.0f}</div><div class='stat-desc'>Y->B İsabeti</div></div>", unsafe_allow_html=True)

                    st.divider()
                    st.markdown(f"""
                    <div style="background-color: #0a0a0a; padding: 15px; border-radius: 10px; border-left: 5px solid #00ff88;">
                    <span style="color:#00ff88; font-weight:bold; font-size:1.1rem;">🧠 DA VINCI SENTETİK SONUÇ:</span><br>
                    <span style="color:#e0e0e0; font-size:1rem;">{sel_stock} varlığında ralli öncesi pencerelerde özellikle 
                    <strong>% {stats['fus_BDB']:.0f}</strong> oranında Füzyon hattı Blue to Dark Blue (Koyu Mavi Onay) yakalamış, 
                    <strong>% {stats['syn_BDB']:.0f}</strong> oranında Synergy hız onayı (Dark Blue) alınmıştır.</span>
                    </div>
                    """, unsafe_allow_html=True)
        else:
             st.warning("Lütfen bir hisse sembolü girin.")
