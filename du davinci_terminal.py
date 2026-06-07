import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
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
    .stat-box { background: linear-gradient(145deg, #1a1a1a, #0a0a0a) !important; padding: 12px; border-radius: 8px; border: 1px solid #444; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.5); margin-bottom: 8px; }
    .stat-value { font-size: 1.3rem !important; font-weight: 900 !important; color: #00E6FF !important; margin: 2px 0; }
    .stat-label { font-size: 0.80rem !important; color: #aaaaaa !important; text-transform: uppercase; font-weight: bold; }
    .macro-card { background-color: #111111 !important; border: 1px solid #333; border-left: 4px solid #FF1744; padding: 15px; border-radius: 5px; margin-bottom: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.5); }
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
# 2. PANDAS VEKTÖREL MATEMATİK MOTORU (V5 ORİJİNAL)
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
    # ORİJİNAL KİNETİK EĞİM MATEMATİĞİ (DEĞİŞTİRİLMEDİ!)
    # ========================================================
    
    # FUSION MATRİSİ
    is_f_rising = df['f_hist'] > df['f_hist'].shift(1)
    is_f_falling = df['f_hist'] < df['f_hist'].shift(1)
    df['Fus_Y2B'] = is_f_rising & is_f_falling.shift(1)
    df['Fus_B2DB'] = is_f_rising & df['Fus_Y2B'].shift(1) & (df['f_hist'] > 0)
    df['Fus_B2Y'] = is_f_falling & is_f_rising.shift(1)
    df['Fus_Y2R'] = is_f_falling & df['Fus_B2Y'].shift(1) & (df['f_hist'] < 0)

    # SYNERGY MATRİSİ
    is_s_rising = df['s_hist'] > df['s_hist'].shift(1)
    is_s_falling = df['s_hist'] < df['s_hist'].shift(1)
    df['Syn_Y2B'] = is_s_rising & is_s_falling.shift(1)
    df['Syn_B2DB'] = is_s_rising & df['Syn_Y2B'].shift(1) & (df['s_hist'] > 0)
    df['Syn_B2Y'] = is_s_falling & is_s_rising.shift(1)
    df['Syn_Y2R'] = is_s_falling & df['Syn_B2Y'].shift(1) & (df['s_hist'] < 0)

    # OMNI MOMENTUM (V5 GERÇEK MATEMATİK: SADECE Y2B VE B2Y)
    rsi_fast = get_rsi(df['Close'], 7)
    df['Omni'] = (rsi_fast + rsi_mid) / 2
    is_o_rising = df['Omni'] > df['Omni'].shift(1)
    is_o_falling = df['Omni'] < df['Omni'].shift(1)
    df['Omni_Y2B'] = is_o_rising & is_o_falling.shift(1)
    df['Omni_B2Y'] = is_o_falling & is_o_rising.shift(1)

    # SPEED / SIGNAL KESİŞİMİ
    df['Spd_Cross_Up'] = (df['f_speed'] > df['f_sig']) & (df['f_speed'].shift(1) <= df['f_sig'].shift(1))
    df['Spd_Cross_Down'] = (df['f_speed'] < df['f_sig']) & (df['f_speed'].shift(1) >= df['f_sig'].shift(1))

    # WHALE RE-ENTRY MATRİSİ
    df['Whale_In'] = df['w_pwr'] > df['pct_pro']
    df['Whale_Y2R'] = df['Whale_In'] & (df['Whale_In'].shift(1) == False)
    df['Whale_R2Y'] = (df['Whale_In'] == False) & df['Whale_In'].shift(1)

    df['RS_Leader'] = df['Close'].pct_change(20, fill_method=None) > 0

    # ========================================================
    # TABLODAKİ (D-1) YAZILARI İÇİN VİZÜEL OKUYUCU
    # ========================================================
    f_state = np.where((df['f_hist'] > 0) & is_f_rising, 'DB',
              np.where((df['f_hist'] > 0) & is_f_falling, 'B',
              np.where((df['f_hist'] <= 0) & is_f_falling, 'R', 'Y')))
    df['Fus_Trans_Str'] = np.where(f_state != pd.Series(f_state).shift(1), pd.Series(f_state).shift(1) + "->" + f_state, "")

    s_state = np.where((df['s_hist'] > 0) & is_s_rising, 'DB',
              np.where((df['s_hist'] > 0) & is_s_falling, 'B',
              np.where((df['s_hist'] <= 0) & is_s_falling, 'R', 'Y')))
    df['Syn_Trans_Str'] = np.where(s_state != pd.Series(s_state).shift(1), pd.Series(s_state).shift(1) + "->" + s_state, "")

    o_state = np.where(is_o_rising, 'B', 'Y')
    df['Omni_Trans_Str'] = np.where(o_state != pd.Series(o_state).shift(1), pd.Series(o_state).shift(1) + "->" + o_state, "")

    return df

# ==========================================
# 3. YEDEKLEMELİ & KORUMALI VERİ MOTORU
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
    except Exception as e:
        if "429" in str(e): return [{"title": "⚠️ Aşırı İstek (Rate Limit). Lütfen 15 dk bekleyin.", "link": "#"}]
    return valid_news if len(valid_news) > 0 else None

@st.cache_data(ttl=3600)
def fetch_valuation_data_safely(stocks):
    val_data = []
    for s in stocks:
        try:
            tk = yf.Ticker(s)
            info = tk.fast_info
            pe = tk.info.get('trailingPE', 0)
            val_data.append({"Ticker": s, "MC": info.get('marketCap', 0), "PE": pe})
            time.sleep(0.5) 
        except:
            time.sleep(1) 
    return val_data

@st.cache_data(ttl=600)
def run_pre_rally_statistics(ticker, days_lookback, move_threshold_pct, is_bullish):
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period="3y", interval="1d")
        
        if df.empty or len(df) < 50: 
            return {"error": f"⚠️ Yahoo Finance '{ticker}' için veri döndüremedi veya limit aşıldı."}
            
        df.index = pd.to_datetime(df.index).tz_localize(None) 
        df = apply_quantum_indicators(df)
        
        df['Future_Return'] = df['Close'].shift(-days_lookback) / df['Close'] - 1
        
        if is_bullish:
            rally_mask = df['Future_Return'] >= (move_threshold_pct / 100.0)
        else:
            rally_mask = df['Future_Return'] <= -(move_threshold_pct / 100.0)
            
        rally_indices = df[rally_mask].index
        
        if len(rally_indices) == 0: 
            return {"stats": {"count": 0}}

        events_list = []
        
        # SADECE ORİJİNAL MATEMATİKTE VAR OLAN SAYIMLAR
        stats = {
            "count": len(rally_indices),
            "f_y2b": 0, "f_b2db": 0, "f_b2y": 0, "f_y2r": 0,
            "s_y2b": 0, "s_b2db": 0, "s_b2y": 0, "s_y2r": 0,
            "o_y2b": 0, "o_b2y": 0,
            "cross_up": 0, "cross_down": 0, "w_in": 0, "w_out": 0
        }

        for idx in rally_indices:
            loc = df.index.get_loc(idx)
            if isinstance(loc, slice): loc = loc.start
            elif isinstance(loc, np.ndarray): loc = np.where(loc)[0][0]
            
            if loc < 4 or loc + days_lookback >= len(df): continue
            
            pre_window = df.iloc[loc-4:loc+1]
            future_ret = df['Future_Return'].iloc[loc]
            target_date = df.index[loc + days_lookback]

            # %100 ORİJİNAL İSTATİSTİK MATEMATİĞİ (SUM > 0)
            if pre_window['Fus_Y2B'].sum() > 0: stats['f_y2b'] += 1
            if pre_window['Fus_B2DB'].sum() > 0: stats['f_b2db'] += 1
            if pre_window['Fus_B2Y'].sum() > 0: stats['f_b2y'] += 1
            if pre_window['Fus_Y2R'].sum() > 0: stats['f_y2r'] += 1

            if pre_window['Syn_Y2B'].sum() > 0: stats['s_y2b'] += 1
            if pre_window['Syn_B2DB'].sum() > 0: stats['s_b2db'] += 1
            if pre_window['Syn_B2Y'].sum() > 0: stats['s_b2y'] += 1
            if pre_window['Syn_Y2R'].sum() > 0: stats['s_y2r'] += 1

            if pre_window['Omni_Y2B'].sum() > 0: stats['o_y2b'] += 1
            if pre_window['Omni_B2Y'].sum() > 0: stats['o_b2y'] += 1

            if pre_window['Spd_Cross_Up'].sum() > 0: stats['cross_up'] += 1
            if pre_window['Spd_Cross_Down'].sum() > 0: stats['cross_down'] += 1
            if pre_window['Whale_Y2R'].sum() > 0: stats['w_in'] += 1
            if pre_window['Whale_R2Y'].sum() > 0: stats['w_out'] += 1

            # Rapor Tablosu İçin Stringler
            fus_events, syn_events, omni_events, spd_events, whale_events = [], [], [], [], []

            for i in range(4, -1, -1):
                curr_loc = loc - i
                day_label = f"(D-{i})" if i > 0 else "(D-0)"

                if df['Fus_Trans_Str'].iloc[curr_loc]: fus_events.append(f"{df['Fus_Trans_Str'].iloc[curr_loc]} {day_label}")
                if df['Syn_Trans_Str'].iloc[curr_loc]: syn_events.append(f"{df['Syn_Trans_Str'].iloc[curr_loc]} {day_label}")
                if df['Omni_Trans_Str'].iloc[curr_loc]: omni_events.append(f"{df['Omni_Trans_Str'].iloc[curr_loc]} {day_label}")

                if df['Spd_Cross_Up'].iloc[curr_loc]: spd_events.append(f"Yukarı✅ {day_label}")
                elif df['Spd_Cross_Down'].iloc[curr_loc]: spd_events.append(f"Aşağı⛔ {day_label}")

                if df['Whale_Y2R'].iloc[curr_loc]: whale_events.append(f"Y->R {day_label}")
                elif df['Whale_R2Y'].iloc[curr_loc]: whale_events.append(f"R->Y {day_label}")

            events_list.append({
                "Sinyal Tarihi": idx.strftime('%Y-%m-%d'),
                "Hedef Tarih": target_date.strftime('%Y-%m-%d'),
                "Gerçekleşen": f"%{future_ret*100:.1f}",
                "Füzyon (V700)": ", ".join(fus_events) if fus_events else "-",
                "Synergy (V665)": ", ".join(syn_events) if syn_events else "-",
                "Omni Mom.": ", ".join(omni_events) if omni_events else "-",
                "Speed/Sig.": ", ".join(spd_events) if spd_events else "-",
                "Whale (V695)": ", ".join(whale_events) if whale_events else "-",
                "RS Durumu": "Lider ✅" if df['RS_Leader'].iloc[loc] else "Zayıf ⛔"
            })

        for k in stats.keys():
            if k != "count": stats[k] = (stats[k] / stats['count']) * 100
            
        return {"stats": stats, "events": pd.DataFrame(events_list)}
    except Exception as e:
        if "429" in str(e): return {"error": "⚠️ SİSTEM BLOKAJI: Yahoo Finance aşırı istek sebebiyle IP'nizi geçici olarak kısıtladı. Lütfen 15 dakika bekleyin."}
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
        with st.spinner("Haberler çekiliyor... (Ban koruması aktif)"):
            macro_news = []
            for tkr in ["SPY", "QQQ", "TLT"]:
                news = fetch_news_safely(tkr)
                if news:
                    for n in news: macro_news.append(f"**[{tkr}]** [{n['title']}]({n['link']})")
                time.sleep(0.5) 
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
                        else: st.warning("⚠️ Haber bulunamadı veya limite takıldı.")
            
            with val_col:
                if st.button(f"📊 Çarpan Analizi Yap", key=f"val_{theme_name}"):
                    with st.spinner("Veriler güvenli bir şekilde çekiliyor (Anti-Ban Aktif)..."):
                        val_data = fetch_valuation_data_safely(stocks)
                        
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
        if sel_theme == "📌 Kendi Hisseni Gir": sel_stock = st.text_input("Borsa Kodu",
