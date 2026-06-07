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
    div[role="radiogroup"] label { color: #00E6FF !important; font-weight: bold; }
    [data-testid="stDataFrame"], [data-testid="stTable"] { background-color: #0a0a0a !important; border: 1px solid #333 !important; }
    th { background-color: #1a1a1a !important; color: #00E6FF !important; font-size: 0.90rem !important; border-bottom: 2px solid #00E6FF !important; }
    td { border-bottom: 1px solid #222 !important; color: #ffffff !important; font-size: 0.85rem !important;}
    [data-testid="stExpander"] { background-color: #0d0d0d !important; border: 1px solid #333 !important; border-left: 4px solid #00E6FF !important; }
    [data-testid="stExpander"] summary p { color: #00ff88 !important; font-weight: bold !important; font-size: 1.1rem !important;}
    div.stButton > button { background-color: #111 !important; color: #00ff88 !important; border: 1px solid #00ff88 !important; font-weight: bold; border-radius: 6px; }
    div.stButton > button:hover { background-color: #00ff88 !important; color: #000 !important; border-color: #fff !important; }
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
# 2. PANDAS VEKTÖREL MATEMATİK MOTORU (4 FAZLI)
# ==========================================
def get_rma(s, period):
    return s.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def get_rsi(s, period):
    delta = s.diff()
    ma_up = get_rma(delta.clip(lower=0), period)
    ma_down = get_rma(-1 * delta.clip(upper=0), period)
    rs = ma_up / ma_down.replace(0, 0.001)
    return 100 - (100 / (1 + rs))

def get_4_state_color(val_series):
    """MACD Hist mantığıyla 4'lü renk döngüsünü tespit eder (DB, B, R, Y)"""
    val_prev = val_series.shift(1)
    cond_DB = (val_series > 0) & (val_series > val_prev)
    cond_B  = (val_series > 0) & (val_series <= val_prev)
    cond_R  = (val_series < 0) & (val_series < val_prev)
    cond_Y  = (val_series < 0) & (val_series >= val_prev)
    return np.select([cond_DB, cond_B, cond_R, cond_Y], ['DB', 'B', 'R', 'Y'], default='N')

def apply_quantum_indicators(df):
    if len(df) < 50 or 'Close' not in df.columns: return df
    
    # 1. FUSION (V700)
    f_macd = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    f_h, f_l = f_macd.rolling(100, min_periods=1).max(), f_macd.rolling(100, min_periods=1).min()
    df['f_speed'] = ((f_macd - f_l) / (f_h - f_l).replace(0, 0.001) * 100) - 50
    df['f_sig'] = df['f_speed'].ewm(span=9, adjust=False).mean()
    df['f_hist'] = df['f_speed'] - df['f_sig']
    
    # 2. SYNERGY (V665)
    hlc3 = (df['High'] + df['Low'] + df['Close']) / 3
    s_macd = hlc3.ewm(span=12, adjust=False).mean() - hlc3.ewm(span=26, adjust=False).mean()
    s_h, s_l = s_macd.rolling(100, min_periods=1).max(), s_macd.rolling(100, min_periods=1).min()
    df['s_speed'] = ((s_macd - s_l) / (s_h - s_l).replace(0, 0.001) * 100) - 50
    df['s_sig'] = df['s_speed'].ewm(span=9, adjust=False).mean()
    df['s_hist'] = df['s_speed'] - df['s_sig']

    # 3. WHALE POWER
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
    # DİNAMİK RENK VE GEÇİŞ MATRİSLERİ
    # ========================================================
    df['Fus_Color'] = get_4_state_color(df['f_hist'])
    df['Fus_Trans'] = df['Fus_Color'].shift(1) + "->" + df['Fus_Color']

    df['Syn_Color'] = get_4_state_color(df['s_hist'])
    df['Syn_Trans'] = df['Syn_Color'].shift(1) + "->" + df['Syn_Color']

    df['Whale_State'] = np.where(df['w_pwr'] > df['pct_pro'], 'IN(Kırmızı)', 'OUT(Sarı)')
    df['Whale_Trans'] = df['Whale_State'].shift(1) + "->" + df['Whale_State']

    df['Spd_Cross'] = np.where(df['f_speed'] > df['f_sig'], 'Yukarı', 'Aşağı')
    df['Cross_Trans'] = df['Spd_Cross'].shift(1) + "->" + df['Spd_Cross']

    df['RS_Leader'] = df['Close'].pct_change(20, fill_method=None) > 0

    return df

# ==========================================
# 3. YEDEKLEMELİ HABER VE İSTATİSTİK MOTORU
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
    except: pass
    return valid_news if len(valid_news) > 0 else None

def extract_unique_transitions(trans_series):
    """4 günlük penceredeki tekrarsız ve geçerli değişimleri bulur (örn: Y->B)"""
    trans_list = trans_series.dropna().tolist()
    valid_trans = [t for t in trans_list if t.split('->')[0] != t.split('->')[-1] and 'N' not in t]
    return ", ".join(dict.fromkeys(valid_trans)) if valid_trans else "-"

@st.cache_data
def run_pre_rally_statistics(ticker, days_lookback, move_threshold_pct, is_bullish):
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period="3y", interval="1d")
        
        if df.empty or len(df) < 50: 
            return {"error": f"⚠️ Yahoo Finance '{ticker}' için veri döndüremedi."}
            
        df.index = pd.to_datetime(df.index).tz_localize(None) # Saat dilimi hatalarını sıfırla
        df = apply_quantum_indicators(df)
        
        # Çift Yönlü Hedef Getiri Hesaplaması
        df['Future_Return'] = df['Close'].shift(-days_lookback) / df['Close'] - 1
        
        if is_bullish:
            rally_mask = df['Future_Return'] >= (move_threshold_pct / 100.0)
        else:
            rally_mask = df['Future_Return'] <= -(move_threshold_pct / 100.0)
            
        rally_indices = df[rally_mask].index
        
        if len(rally_indices) == 0: 
            return {"stats": {"count": 0}}

        events_list = []
        stats = {
            "count": len(rally_indices),
            "fus_bull": 0, "fus_bear": 0, 
            "syn_bull": 0, "syn_bear": 0, 
            "cross_up": 0, "cross_down": 0,
            "whale_in": 0, "whale_out": 0
        }

        for idx in rally_indices:
            loc = df.index.get_loc(idx)
            if isinstance(loc, slice): loc = loc.start
            elif isinstance(loc, np.ndarray): loc = np.where(loc)[0][0]
            
            if loc < 4 or loc + days_lookback >= len(df): continue
            
            pre_window = df.iloc[loc-4:loc+1]
            future_ret = df['Future_Return'].iloc[loc]
            target_date = df.index[loc + days_lookback]
            
            # Dinamik Olayları Çek
            str_fus = extract_unique_transitions(pre_window['Fus_Trans'])
            str_syn = extract_unique_transitions(pre_window['Syn_Trans'])
            str_whale = extract_unique_transitions(pre_window['Whale_Trans'])
            str_cross = extract_unique_transitions(pre_window['Cross_Trans'])
            rs_state = "Lider ✅" if df['RS_Leader'].iloc[loc] else "Zayıf ⛔"

            # Kümülatif İstatistik Puanlaması (Aramaya Yönelik)
            if "Y->B" in str_fus or "B->DB" in str_fus: stats['fus_bull'] += 1
            if "B->Y" in str_fus or "Y->R" in str_fus: stats['fus_bear'] += 1
            
            if "Y->B" in str_syn or "B->DB" in str_syn: stats['syn_bull'] += 1
            if "B->Y" in str_syn or "Y->R" in str_syn: stats['syn_bear'] += 1
            
            if "Aşağı->Yukarı" in str_cross: stats['cross_up'] += 1
            if "Yukarı->Aşağı" in str_cross: stats['cross_down'] += 1
            
            if "OUT(Sarı)->IN(Kırmızı)" in str_whale: stats['whale_in'] += 1
            if "IN(Kırmızı)->OUT(Sarı)" in str_whale: stats['whale_out'] += 1

            events_list.append({
                "Sinyal Tarihi": idx.strftime('%Y-%m-%d'),
                "Hedef Tarih": target_date.strftime('%Y-%m-%d'),
                "Gerçekleşen": f"%{future_ret*100:.1f}",
                "Füzyon Geçişi": str_fus,
                "Synergy Geçişi": str_syn,
                "Speed/Sig": str_cross.replace("Aşağı->Yukarı", "Yukarı Kesti").replace("Yukarı->Aşağı", "Aşağı Kesti"),
                "Whale (V695)": str_whale.replace("OUT(Sarı)->IN(Kırmızı)", "Giriş").replace("IN(Kırmızı)->OUT(Sarı)", "Çıkış"),
                "RS Durumu": rs_state
            })

        for k in stats.keys():
            if k != "count": stats[k] = (stats[k] / stats['count']) * 100
            
        return {"stats": stats, "events": pd.DataFrame(events_list)}
    except Exception as e:
        return {"error": f"Sistem Hatası: {str(e)}"}

# ==========================================
# 4. ARAYÜZ (TABS)
# ==========================================
st.title("🏛️ DA VINCI: İSTİHBARAT & RALLİ İSTATİSTİK MOTORU")
tab1, tab2, tab3 = st.tabs(["🌍 LİKİDİTE & OPEX MASASI", "⚖️ THEMATIC VALUATION GAP", "🦈 V700 HAREKET ÖNCESİ İSTATİSTİĞİ"])

# ---------------------------------------------------------
# TAB 1: MAKRO 
# ---------------------------------------------------------
with tab1:
    st.markdown("### 📡 Canlı Makro İstihbarat Radarı")
    
    if 'macro_updates' not in st.session_state:
        st.session_state.macro_updates = {"calendar": "Bekleniyor...", "fed": "Bekleniyor...", "actors": "Bekleniyor...", "trump": "Bekleniyor...", "geo": "Bekleniyor..."}
        st.session_state.live_macro_news = []

    st.markdown("#### 🔄 Global Canlı Haber Akışı")
    if st.button("🌐 SPY, QQQ, TLT Global Haberlerini Güncelle", use_container_width=True):
        with st.spinner("Haberler çekiliyor..."):
            macro_news = []
            for tkr in ["SPY", "QQQ", "TLT"]:
                news = fetch_news_safely(tkr)
                if news:
                    for n in news: macro_news.append(f"**[{tkr}]** [{n['title']}]({n['link']})")
            st.session_state.live_macro_news = macro_news

    if st.session_state.live_macro_news:
        st.markdown("<div style='background-color:#111; padding:10px; border-radius:5px; border-left:4px solid #00BFFF; margin-bottom:15px;'>", unsafe_allow_html=True)
        for news_item in st.session_state.live_macro_news: st.markdown(f"- {news_item}")
        st.markdown("</div>", unsafe_allow_html=True)
        
    col_up1, col_up2, col_up3, col_up4, col_up5 = st.columns(5)
    with col_up1:
        if st.button("📅 Finansal Takvim Çek"): st.session_state.macro_updates['calendar'] = "TÜFE Beklentisi: %2.8. Yüksek gelirse Tahviller DÜŞER (Negatif)."
    with col_up2:
        if st.button("💧 Fed/Likidite Kararları"): st.session_state.macro_updates['fed'] = "Fed Swapları faiz indirim ihtimalini %40'a çekti. Etki: Dolar Endeksi Güçleniyor."
    with col_up3:
        if st.button("🌐 Global Aktörler"): st.session_state.macro_updates['actors'] = "Çin PBOC emlak sektörü için likidite enjekte etti. Endüstri (XLI) için Pozitif."
    with col_up4:
        if st.button("🦅 ABD Yönetim Kararları"): st.session_state.macro_updates['trump'] = "Yapay Zeka ve Uzay altyapısına yeni 'Government Stake' yasası onaylandı."
    with col_up5:
        if st.button("🌍 Jeopolitik Şoklar"): st.session_state.macro_updates['geo'] = "Ortadoğu'da tanker trafiği durduruldu. Petrol (USO) YUKARI yönde fiyatlama yapıyor."

    st.markdown(f"<div class='macro-card'><span style='color:#00ff88; font-weight:bold;'>📆 Haftalık Finansal Takvim Beklentisi:</span><br><span style='color:#fff;'>{st.session_state.macro_updates['calendar']}</span></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='macro-card'><span style='color:#00ff88; font-weight:bold;'>🖨️ Likidite ve Merkez Bankası:</span><br><span style='color:#fff;'>{st.session_state.macro_updates['fed']}</span></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 2: VALUATION GAP 
# ---------------------------------------------------------
with tab2:
    st.markdown("### ⚖️ Tematik Fon Liderleri ve Çarpan Uçurumu")
    
    for theme_name, stocks in THEMES.items():
        if theme_name == "📌 Kendi Hisseni Gir": continue 
        
        with st.expander(f"📁 TEMA: {theme_name}"):
            news_col, val_col = st.columns([1, 2])
            
            with news_col:
                if st.button(f"🔄 Haberleri Çek", key=f"btn_{theme_name}"):
                    with st.spinner("Taranıyor..."):
                        n_list = fetch_news_safely(stocks[0])
                        if n_list:
                            for n in n_list: st.markdown(f"- <a href='{n['link']}' target='_blank' style='color:#00BFFF;'>{n['title']}</a>", unsafe_allow_html=True)
                        else: st.warning("⚠️ Haber bulunamadı.")
            
            with val_col:
                if st.button(f"📊 Çarpan Analizi Yap", key=f"val_{theme_name}"):
                    with st.spinner("Hesaplanıyor..."):
                        val_data = []
                        for s in stocks:
                            try:
                                info = yf.Ticker(s).fast_info
                                val_data.append({"Ticker": s, "MC": info.get('marketCap', 0), "PE": yf.Ticker(s).info.get('trailingPE', 0)})
                            except: pass
                        
                        df_val = pd.DataFrame(val_data)
                        if not df_val.empty and df_val['MC'].sum() > 0:
                            df_val = df_val.sort_values(by='MC', ascending=False)
                            leader = df_val.iloc[0]
                            st.markdown(f"<h4 style='color:#FFD700;'>👑 LİDER: {leader['Ticker']} (${leader['MC']/1e9:.1f} Milyar)</h4>", unsafe_allow_html=True)
                            
                            res_html = "<ul style='color:#fff;'>"
                            for i in range(1, len(df_val)):
                                row = df_val.iloc[i]
                                ratio = (row['MC'] / leader['MC']) * 100 if leader['MC'] > 0 else 0
                                pe_str = f"F/K: {row['PE']:.1f}" if row['PE'] else "N/A"
                                # YENİ MANTIKLI FORMAT: Liderin %X'i büyüklüğünde
                                res_html += f"<li><strong>{row['Ticker']}</strong>: ${row['MC']/1e9:.1f} Milyar <em>(Liderin %{ratio:.1f}'i büyüklüğünde)</em> — {pe_str}</li>"
                            res_html += "</ul>"
                            st.markdown(res_html, unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 3: DİNAMİK RALLİ/ÇÖKÜŞ MATRİSİ
# ---------------------------------------------------------
with tab3:
    st.markdown("### 🦈 Da Vinci: Kinetik Hareket Matrisi (Long & Short)")
    
    col_t1, col_t2, col_t3, col_t4, col_t5 = st.columns([2, 2, 1.5, 1.5, 2])
    with col_t1: sel_theme = st.selectbox("Tema", list(THEMES.keys()))
    with col_t2: 
        if sel_theme == "📌 Kendi Hisseni Gir": sel_stock = st.text_input("Borsa Kodu", value="PLTR").upper()
        else: sel_stock = st.selectbox("Hisse / ETF", THEMES[sel_theme])
    with col_t3: lookback_days = st.number_input("Süre (Gün)", min_value=1, max_value=60, value=6)
    with col_t4: rally_pct = st.number_input("Hedef Yüzde (%)", min_value=1, max_value=100, value=10)
    with col_t5: 
        # YENİ EKLENTİ: Çift Yönlü Tarama
        scan_direction = st.radio("Tarama Yönü", ["🚀 Yükseliş (Long)", "🩸 Düşüş (Short)"], horizontal=True)
        
    if st.button("⚛️ MATRİSİ ÇALIŞTIR", use_container_width=True):
        if sel_stock:
            is_bull = "Yükseliş" in scan_direction
            with st.spinner(f"{sel_stock} için son 3 yılın verileri taranıyor..."):
                res = run_pre_rally_statistics(sel_stock, lookback_days, rally_pct, is_bull)
                
                if "error" in res:
                    st.error(res["error"])
                elif res["stats"]["count"] == 0:
                    yön_text = "yükseliş" if is_bull else "düşüş"
                    st.warning(f"⚠️ {sel_stock} grafiğinde son 3 yılda {lookback_days} günde %{rally_pct}+ {yön_text} hareketi bulunamadı.")
                else:
                    stats = res['stats']
                    
                    st.success(f"Geçmişte tam **{stats['count']} adet** majör hareket noktası tespit edildi!")
                    st.dataframe(res['events'], use_container_width=True, hide_index=True)
                    
                    st.markdown("#### 📊 Kümülatif Hedef İsabet Oranları")
                    sc1, sc2, sc3, sc4 = st.columns(4)
                    
                    # Dinamik istatistik gösterimi (Seçilen yöne göre en anlamlı veriyi basar)
                    if is_bull:
                        with sc1: st.markdown(f"<div class='stat-box'><div class='stat-label'>Füzyon Pozitif Geçiş</div><div class='stat-value'>%{stats['fus_bull']:.0f}</div><div class='stat-desc'>Y->B veya B->DB</div></div>", unsafe_allow_html=True)
                        with sc2: st.markdown(f"<div class='stat-box'><div class='stat-label'>Synergy Pozitif Geçiş</div><div class='stat-value'>%{stats['syn_bull']:.0f}</div><div class='stat-desc'>Y->B veya B->DB</div></div>", unsafe_allow_html=True)
                        with sc3: st.markdown(f"<div class='stat-box'><div class='stat-label'>Golden Cross (Speed)</div><div class='stat-value'>%{stats['cross_up']:.0f}</div><div class='stat-desc'>Yukarı Kesti</div></div>", unsafe_allow_html=True)
                        with sc4: st.markdown(f"<div class='stat-box'><div class='stat-label'>Whale Accumulation</div><div class='stat-value'>%{stats['whale_in']:.0f}</div><div class='stat-desc'>Giriş Sağlandı</div></div>", unsafe_allow_html=True)
                    else:
                        with sc1: st.markdown(f"<div class='stat-box'><div class='stat-label'>Füzyon Negatif Bozulma</div><div class='stat-value'>%{stats['fus_bear']:.0f}</div><div class='stat-desc'>B->Y veya Y->R</div></div>", unsafe_allow_html=True)
                        with sc2: st.markdown(f"<div class='stat-box'><div class='stat-label'>Synergy Negatif Bozulma</div><div class='stat-value'>%{stats['syn_bear']:.0f}</div><div class='stat-desc'>B->Y veya Y->R</div></div>", unsafe_allow_html=True)
                        with sc3: st.markdown(f"<div class='stat-box'><div class='stat-label'>Death Cross (Speed)</div><div class='stat-value'>%{stats['cross_down']:.0f}</div><div class='stat-desc'>Aşağı Kesti</div></div>", unsafe_allow_html=True)
                        with sc4: st.markdown(f"<div class='stat-box'><div class='stat-label'>Whale Distribution</div><div class='stat-value'>%{stats['whale_out']:.0f}</div><div class='stat-desc'>Çıkış Gözlendi</div></div>", unsafe_allow_html=True)
