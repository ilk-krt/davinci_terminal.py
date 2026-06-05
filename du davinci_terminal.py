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
    /* Genel Arkaplan ve Metin Rengi Sabitlemesi */
    .stApp, .main, .block-container { background-color: #050505 !important; }
    
    /* Tüm Metin Elementleri İçin Yüksek Kontrast */
    p, div, span, li, label, text, .stMarkdown { color: #E0E0E0 !important; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    h1, h2, h3, h4, h5, h6 { color: #00E6FF !important; font-weight: 900 !important; text-transform: uppercase; letter-spacing: 1px; }
    
    /* Girdi Kutuları (Selectbox, Number Input) */
    div[data-baseweb="select"] > div, input[type="number"] { background-color: #111111 !important; color: #ffffff !important; border: 1px solid #00E6FF !important; }
    div[data-baseweb="popover"] div { background-color: #111111 !important; color: #ffffff !important; }
    
    /* Tablolar (Dataframes) */
    [data-testid="stDataFrame"], [data-testid="stTable"] { background-color: #0a0a0a !important; border: 1px solid #333 !important; }
    th { background-color: #1a1a1a !important; color: #00E6FF !important; font-size: 0.95rem !important; border-bottom: 2px solid #00E6FF !important; }
    td { border-bottom: 1px solid #222 !important; color: #ffffff !important; }
    
    /* Expander (Açılır Kapanır Menüler) */
    [data-testid="stExpander"] { background-color: #0d0d0d !important; border: 1px solid #333 !important; border-left: 4px solid #00E6FF !important; }
    [data-testid="stExpander"] summary p { color: #00ff88 !important; font-weight: bold !important; font-size: 1.1rem !important;}
    
    /* Butonlar */
    div.stButton > button { background-color: #111 !important; color: #00ff88 !important; border: 1px solid #00ff88 !important; font-weight: bold; border-radius: 6px; }
    div.stButton > button:hover { background-color: #00ff88 !important; color: #000 !important; border-color: #fff !important; }
    
    /* Özel Kartlar (Macro, Stat) */
    .macro-card { background-color: #111111 !important; border: 1px solid #333; border-left: 4px solid #FF1744; padding: 15px; border-radius: 5px; margin-bottom: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.5); }
    .stat-box { background: linear-gradient(145deg, #1a1a1a, #0a0a0a) !important; padding: 15px; border-radius: 8px; border: 1px solid #444; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.5); }
    .stat-value { font-size: 2rem !important; font-weight: 900 !important; color: #00E6FF !important; margin: 10px 0; }
    .stat-label { font-size: 0.85rem !important; color: #aaaaaa !important; text-transform: uppercase; font-weight: bold; }
    .stat-desc { font-size: 0.75rem !important; color: #888888 !important; }
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
    "Siber Güvenlik (Cyber)": ["CIBR", "HACK", "CRWD", "PANW", "FTNT"]
}

# ==========================================
# 2. PANDAS VEKTÖREL MATEMATİK MOTORU (ONARILMIŞ)
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
    
    # V665: FUSION & SYNERGY
    f_macd = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    f_h, f_l = f_macd.rolling(100, min_periods=1).max(), f_macd.rolling(100, min_periods=1).min()
    df['f_speed'] = ((f_macd - f_l) / (f_h - f_l).replace(0, 0.001) * 100) - 50
    df['f_sig'] = df['f_speed'].ewm(span=9, adjust=False).mean()
    
    hlc3 = (df['High'] + df['Low'] + df['Close']) / 3
    s_macd = hlc3.ewm(span=12, adjust=False).mean() - hlc3.ewm(span=26, adjust=False).mean()
    s_h, s_l = s_macd.rolling(100, min_periods=1).max(), s_macd.rolling(100, min_periods=1).min()
    df['s_speed'] = ((s_macd - s_l) / (s_h - s_l).replace(0, 0.001) * 100) - 50

    # V695: WHALE POWER (Düzeltilmiş Pandas Vektörel Mantığı)
    rsi_mid = get_rsi(df['Close'], 14)
    c_range = (df['High'] - df['Low']).clip(lower=0.001)
    delta = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / c_range
    vol_sma20 = df['Volume'].rolling(20, min_periods=1).mean().clip(lower=0.001)
    
    delta_vol = (delta * df['Volume']).rolling(20, min_periods=1).mean() / vol_sma20
    rvol = (df['Volume'] / vol_sma20.clip(lower=1)).clip(upper=2.5)
    
    base_pwr = ((rsi_mid - 50) + (delta_vol * 40)) * rvol * 1.5
    logic_pwr = np.log(1 + np.exp(np.clip(base_pwr / 5, -50, 50))) * 5
    
    # FVG Düzeltmesi (Sessiz çökmeyi engelleyen Numpy Where kullanımı)
    fvg_bull = (df['Low'] > df['High'].shift(2)) & (df['Close'] > df['Open'])
    logic_pwr = np.where(fvg_bull, logic_pwr + 35, logic_pwr)

    df['w_pwr'] = np.clip((np.log10(1 + logic_pwr) * 65)**0.8 * 1.8, 0, 100)
    df['pct_pro'] = df['w_pwr'].ewm(span=3, adjust=False).mean()

    # --- SİNYAL ÜRETİCİLERİ ---
    df['Syn_Color'] = np.where(df['s_speed'] > df['s_speed'].shift(1), 'Mavi/Turkuaz', 'Kırmızı/Pembe')
    df['Fus_Cross'] = np.where(df['f_speed'] > df['f_sig'], 'Pozitif', 'Negatif')
    df['Whale_State'] = np.where(df['w_pwr'] > 70, 'Full Kırmızı (Whale IN)', np.where(df['w_pwr'] < df['pct_pro'], 'Sarı Uç (Whale OUT)', 'Nötr'))
    
    # Kinetik Patlamalar (VSA) ve Trapler
    df['VSA_Ignition'] = (df['Volume'] > vol_sma20 * 2) & (df['Close'] > df['Open'])
    ema9 = df['Close'].ewm(span=9, adjust=False).mean()
    df['Bear_Trap'] = (df['Low'] < ema9) & (df['Close'] > ema9) & (df['Volume'] > vol_sma20 * 1.5)

    return df

# ==========================================
# 3. YEDEKLEMELİ HABER VE İSTATİSTİK MOTORU
# ==========================================
@st.cache_data(ttl=600) # 10 dakika önbellek (API Ban yememek için)
def fetch_news_safely(ticker):
    try:
        news_data = yf.Ticker(ticker).news
        if news_data and len(news_data) > 0:
            return news_data[:3]
    except:
        pass
    return None

@st.cache_data
def run_pre_rally_statistics(ticker, days_lookback, move_threshold_pct):
    try:
        # Hata önleyici History Metodu (yf.download yerine)
        tk = yf.Ticker(ticker)
        df = tk.history(period="3y", interval="1d")
        
        if df.empty or len(df) < 50: 
            return None
            
        df = apply_quantum_indicators(df)
        
        # Gelecekteki hareketi hesapla
        df['Future_Return'] = df['Close'].shift(-days_lookback) / df['Close'] - 1
        
        # Eşiği aşan Ralli Başlangıç Noktalarını Bul
        rally_indices = df[df['Future_Return'] >= (move_threshold_pct / 100.0)].index
        
        if len(rally_indices) == 0: 
            return {"count": 0}

        stats = {"count": len(rally_indices), "syn_blue": 0, "fus_pos": 0, "whale_in": 0, "vsa_ign": 0, "bear_trap": 0}

        for idx in rally_indices:
            loc = df.index.get_loc(idx)
            if loc < 4: continue
            
            pre_window = df.iloc[loc-4:loc+1] # Ralli öncesi son 4 gün
            
            if (pre_window['Syn_Color'] == 'Mavi/Turkuaz').any(): stats['syn_blue'] += 1
            if (pre_window['Fus_Cross'] == 'Pozitif').any(): stats['fus_pos'] += 1
            if (pre_window['Whale_State'] == 'Full Kırmızı (Whale IN)').any(): stats['whale_in'] += 1
            if pre_window['VSA_Ignition'].any(): stats['vsa_ign'] += 1
            if pre_window['Bear_Trap'].any(): stats['bear_trap'] += 1

        # Yüzdelere çevir
        for k in ["syn_blue", "fus_pos", "whale_in", "vsa_ign", "bear_trap"]:
            stats[k] = (stats[k] / stats['count']) * 100
            
        return stats
    except Exception as e:
        return None

# ==========================================
# 4. ARAYÜZ (TABS)
# ==========================================
st.title("🏛️ DA VINCI: İSTİHBARAT & RALLİ İSTATİSTİK MOTORU")
tab1, tab2, tab3 = st.tabs(["🌍 LİKİDİTE & OPEX MASASI", "⚖️ THEMATIC VALUATION GAP", "🦈 V700 RALLİ ÖNCESİ İSTATİSTİĞİ"])

# ---------------------------------------------------------
# TAB 1: MAKRO, OPEX VE JEOPOLİTİK (API BAĞLANTILI MOCK)
# ---------------------------------------------------------
with tab1:
    st.markdown("### 📡 Canlı Makro İstihbarat Radarı")
    
    col_up1, col_up2, col_up3, col_up4, col_up5 = st.columns(5)
    
    if 'macro_updates' not in st.session_state:
        st.session_state.macro_updates = {
            "calendar": "Bekleniyor...", "fed": "Bekleniyor...", "actors": "Bekleniyor...", "trump": "Bekleniyor...", "geo": "Bekleniyor..."
        }

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
                            st.warning(f"⚠️ Yahoo Finance API geçici olarak yanıt vermedi.")
                            st.markdown(f"🔍 [**Google News üzerinden {stocks[0]} haberlerini anında görüntüle**](https://news.google.com/search?q={stocks[0]})", unsafe_allow_html=True)
            
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
# TAB 3: V700 RALLİ ÖNCESİ İSTATİSTİK MOTORU (DİNAMİK)
# ---------------------------------------------------------
with tab3:
    st.markdown("### 🦈 Da Vinci: Esnek Kinetik Ralli Hafıza Taraması")
    st.caption("Fiyatın belirlediğin gün sayısında, belirlediğin yüzde (%) kadar ralli yaptığı tarihsel anları tespit eder. Ralliden önceki 4 günlük mumlardaki sinyallerin isabet oranını hesaplar.")
    
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    with col_t1: 
        sel_theme = st.selectbox("İncelenecek Tema", list(THEMES.keys()))
    with col_t2: 
        sel_stock = st.selectbox("Hisse / ETF", THEMES[sel_theme])
    with col_t3: 
        lookback_days = st.number_input("Ralli Süresi (Gün)", min_value=1, max_value=60, value=10)
    with col_t4: 
        rally_pct = st.number_input("Hedef Yükseliş (%)", min_value=1, max_value=100, value=15)
        
    if st.button("⚛️ RALLİ İSTATİSTİKLERİNİ HESAPLA", use_container_width=True):
        with st.spinner(f"{sel_stock} için son 3 yılın verileri taranıyor..."):
            stats = run_pre_rally_statistics(sel_stock, lookback_days, rally_pct)
            
            if stats is None or stats['count'] == 0:
                st.error(f"⚠️ {sel_stock} için {lookback_days} günde %{rally_pct}+ hareket bulunamadı.")
            else:
                st.success(f"Geçmişte **{stats['count']} adet** Majör Ralli noktası tespit edildi!")
                
                st.markdown("#### 🔍 Ralli Başlamadan Önceki 4 Günün Karakteristiği:")
                sc1, sc2, sc3, sc4, sc5 = st.columns(5)
                
                with sc1: st.markdown(f"<div class='stat-box'><div class='stat-label'>Synergy (V665)</div><div class='stat-value'>%{stats['syn_blue']:.0f}</div><div class='stat-desc'>Maviye Döndü</div></div>", unsafe_allow_html=True)
                with sc2: st.markdown(f"<div class='stat-box'><div class='stat-label'>Fusion (V700)</div><div class='stat-value'>%{stats['fus_pos']:.0f}</div><div class='stat-desc'>Pozitif Kesti</div></div>", unsafe_allow_html=True)
                with sc3: st.markdown(f"<div class='stat-box'><div class='stat-label'>Whale IN (V695)</div><div class='stat-value'>%{stats['whale_in']:.0f}</div><div class='stat-desc'>Giriş Gözlendi</div></div>", unsafe_allow_html=True)
                with sc4: st.markdown(f"<div class='stat-box'><div class='stat-label'>Afterburner</div><div class='stat-value'>%{stats['vsa_ign']:.0f}</div><div class='stat-desc'>Ignition Patlaması</div></div>", unsafe_allow_html=True)
                with sc5: st.markdown(f"<div class='stat-box'><div class='stat-label'>Kurumsal Trap</div><div class='stat-value'>%{stats['bear_trap']:.0f}</div><div class='stat-desc'>Ayı Tuzağı Kuruldu</div></div>", unsafe_allow_html=True)

                st.divider()
                st.markdown(f"""
                <div style="background-color: #0a0a0a; padding: 15px; border-radius: 10px; border-left: 5px solid #00ff88;">
                <span style="color:#00ff88; font-weight:bold; font-size:1.1rem;">🧠 DA VINCI SENTETİK SONUÇ:</span><br>
                <span style="color:#e0e0e0; font-size:1rem;">İstatistikler gösteriyor ki <strong>{sel_stock}</strong> kodlu varlıkta büyük fiyat hareketlerinden hemen önceki 
                {lookback_days} günlük periyotta özellikle <strong>Whale IN (V695)</strong> ve <strong>Synergy Hat (V665)</strong> değişimleri, 
                kurumsal birikimin en güçlü öncü göstergesidir.</span>
                </div>
                """, unsafe_allow_html=True)
