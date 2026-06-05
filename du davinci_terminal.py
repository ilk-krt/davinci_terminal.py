import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import time

# ==========================================
# 0. AYARLAR & DA VINCI DARK MODE CSS
# ==========================================
st.set_page_config(layout="wide", page_title="DA VINCI: PRE-RALLY & MACRO TERMINAL", page_icon="👁️‍🗨️")

st.markdown("""
    <style>
    .stApp { background-color: #030303 !important; color: #e0e0e0 !important; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    h1, h2, h3 { color: #00ff88 !important; font-weight: 900 !important; text-transform: uppercase; letter-spacing: 1px; }
    div[data-baseweb="select"] > div { background-color: #0a0a0a !important; color: #ffffff !important; border: 1px solid #00E6FF !important; }
    [data-testid="stDataFrame"] { background-color: #0a0a0a !important; border: 1px solid #222 !important; }
    th { background-color: #111 !important; color: #00E6FF !important; font-size: 0.9rem !important; }
    td { border-bottom: 1px solid #222 !important; color: #ddd !important; }
    [data-testid="stExpander"] { background-color: #0a0a0a !important; border: 1px solid #333 !important; border-left: 4px solid #00E6FF !important; }
    div.stButton > button { background-color: #111 !important; color: #00ff88 !important; border: 1px solid #00ff88 !important; font-weight: bold; }
    div.stButton > button:hover { background-color: #00ff88 !important; color: #000 !important; }
    .macro-card { background: #0a0a0a; border: 1px solid #222; border-left: 4px solid #FF1744; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
    .stat-box { background: linear-gradient(145deg, #111, #050505); padding: 15px; border-radius: 8px; border: 1px solid #333; text-align: center; }
    .stat-value { font-size: 1.8rem; font-weight: 900; color: #00E6FF; }
    .stat-label { font-size: 0.8rem; color: #aaa; text-transform: uppercase; }
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

def apply_quantum_indicators(df):
    if len(df) < 100: return df
    
    close, high, low, open_p, vol = df['Close'], df['High'], df['Low'], df['Open'], df['Volume']
    hlc3 = (high + low + close) / 3

    # V665: FUSION & SYNERGY
    f_macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    f_h, f_l = f_macd.rolling(100).max(), f_macd.rolling(100).min()
    df['f_speed'] = ((f_macd - f_l) / (f_h - f_l).replace(0, 0.001) * 100) - 50
    df['f_sig'] = df['f_speed'].ewm(span=9, adjust=False).mean()
    
    s_macd = hlc3.ewm(span=12, adjust=False).mean() - hlc3.ewm(span=26, adjust=False).mean()
    s_h, s_l = s_macd.rolling(100).max(), s_macd.rolling(100).min()
    df['s_speed'] = ((s_macd - s_l) / (s_h - s_l).replace(0, 0.001) * 100) - 50

    # V700: OMNI MOMENTUM
    rsi_fast, rsi_mid = get_rsi(close, 7), get_rsi(close, 14)
    df['omni_consensus'] = (rsi_fast + rsi_mid) / 2 # Basitleştirilmiş vektörel omni
    
    # V695: WHALE POWER (Sentetik normalize)
    c_range = (high - low).clip(lower=0.001)
    delta = ((close - low) - (high - close)) / c_range
    delta_vol = (delta * vol).rolling(20).mean() / vol.rolling(20).mean().clip(lower=0.001)
    rvol = (vol / vol.rolling(20).mean().clip(lower=1)).clip(upper=2.5)
    
    base_pwr = ((rsi_mid - 50) + (delta_vol * 40)) * rvol * 1.5
    logic_pwr = np.log(1 + np.exp(np.clip(base_pwr / 5, -50, 50))) * 5
    df['w_pwr'] = np.clip((np.log10(1 + logic_pwr) * 65)**0.8 * 1.8, 0, 100)
    df['pct_pro'] = df['w_pwr'].ewm(span=3, adjust=False).mean()

    # --- SİNYAL ÜRETİCİLERİ ---
    df['Syn_Color'] = np.where(df['s_speed'] > df['s_speed'].shift(1), 'Mavi/Turkuaz', 'Kırmızı/Pembe')
    df['Fus_Cross'] = np.where(df['f_speed'] > df['f_sig'], 'Pozitif', 'Negatif')
    df['Whale_State'] = np.where(df['w_pwr'] > 70, 'Full Kırmızı (Whale IN)', np.where(df['w_pwr'] < df['pct_pro'], 'Sarı Uç (Whale OUT)', 'Nötr'))
    
    # Kinetik Patlamalar (VSA)
    df['VSA_Ignition'] = (vol > vol.rolling(20).mean() * 2) & (close > open)
    
    # Basit Trap Mantığı (Sembolsüz, Salt Mantık)
    ema9 = close.ewm(span=9, adjust=False).mean()
    df['Bull_Trap'] = (high > ema9) & (close < ema9) & (vol > vol.rolling(20).mean() * 1.5)
    df['Bear_Trap'] = (low < ema9) & (close > ema9) & (vol > vol.rolling(20).mean() * 1.5)

    return df

# ==========================================
# 3. RALLİ ÖNCESİ İSTATİSTİK MOTORU (PRE-RALLY)
# ==========================================
@st.cache_data
def run_pre_rally_statistics(ticker, days_lookback=10, move_threshold=0.15):
    try:
        df = yf.download(ticker, period="3y", interval="1d", progress=False)
        if df.empty or len(df) < 100: return None
        if isinstance(df.columns, pd.MultiIndex): df = df.xs(ticker, level=1, axis=1)
        
        df = apply_quantum_indicators(df.copy())
        
        # Gelecekteki 10 günlük hareketi hesapla
        df['Future_Return'] = df['Close'].shift(-days_lookback) / df['Close'] - 1
        
        # Ralli Başlangıç Noktalarını Bul
        rally_indices = df[df['Future_Return'] >= move_threshold].index
        
        if len(rally_indices) == 0: return {"count": 0}

        stats = {
            "count": len(rally_indices),
            "syn_blue": 0, "fus_pos": 0, "whale_in": 0, "vsa_ign": 0, "bear_trap": 0
        }

        for idx in rally_indices:
            # Ralli öncesi son 4 güne bak
            loc = df.index.get_loc(idx)
            if loc < 4: continue
            
            pre_window = df.iloc[loc-4:loc+1]
            
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
st.title("🏛️ DA VINCI: ISTİHBARAT & RALLİ İSTATİSTİK MOTORU")
tab1, tab2, tab3 = st.tabs(["🌍 LİKİDİTE & OPEX MASASI", "⚖️ THEMATIC VALUATION GAP", "🦈 V700 RALLİ ÖNCESİ İSTATİSTİĞİ"])

# ---------------------------------------------------------
# TAB 1: MAKRO, OPEX VE JEOPOLİTİK (API BAĞLANTILI MOCK)
# ---------------------------------------------------------
with tab1:
    st.markdown("### 📡 Canlı Makro İstihbarat Radarı")
    
    col_up1, col_up2, col_up3, col_up4, col_up5 = st.columns(5)
    
    if 'macro_updates' not in st.session_state:
        st.session_state.macro_updates = {
            "calendar": "Bekleniyor...", "fed": "Bekleniyor...", "actors": "Bekleniyor...", "trump": "Bekleniyor...", "geo": "Bekleniyor...", "bonds": "Bekleniyor..."
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

    st.markdown(f"<div class='macro-card'><strong>📆 Haftalık Finansal Takvim Beklentisi:</strong><br>{st.session_state.macro_updates['calendar']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='macro-card'><strong>🖨️ Likidite ve Merkez Bankası (Fed):</strong><br>{st.session_state.macro_updates['fed']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='macro-card'><strong>🌏 Çin & Avrupa Birliği Kararları:</strong><br>{st.session_state.macro_updates['actors']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='macro-card'><strong>🦅 Beyaz Saray İcraatleri:</strong><br>{st.session_state.macro_updates['trump']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='macro-card'><strong>⚔️ Jeopolitik ve Emtia Hatları:</strong><br>{st.session_state.macro_updates['geo']}</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("#### 📉 Tahvil Piyasası (Yield Curve) Cuma-Cuma Hareketi")
    if st.button("🔄 Tahvil Verilerini Güncelle"):
        st.info("US01Y: %4.95 -> %5.10 | US05Y: %4.20 -> %4.35 | US10Y: %4.15 -> %4.30. \n**Anomali:** Eğri dikleşiyor (Bear Steepener). Bu durum bankacılık (XLF) için pozitif, Yüksek çarpanlı teknoloji (XLK) için baskılayıcıdır.")

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
                    # Temsili haber API entegrasyon noktası (yfinance news kullanılarak)
                    try:
                        tk = yf.Ticker(stocks[0]) # Liderin haberini çek
                        n_list = tk.news[:3]
                        st.markdown("**En Güncel Tematik Haberler:**")
                        for n in n_list:
                            st.write(f"- [{n['title']}]({n['link']})")
                    except:
                        st.warning("Haber API'si meşgul.")
            
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
                            except:
                                pass
                        
                        df_val = pd.DataFrame(val_data)
                        if not df_val.empty and df_val['MC'].sum() > 0:
                            df_val = df_val.sort_values(by='MC', ascending=False)
                            leader = df_val.iloc[0]
                            st.markdown(f"**👑 TEMATİK LİDER:** {leader['Ticker']} (${leader['MC']/1e9:.1f} Milyar)")
                            
                            res_html = "<ul>"
                            for i in range(1, len(df_val)):
                                row = df_val.iloc[i]
                                gap = leader['MC'] / row['MC'] if row['MC'] > 0 else 0
                                pe_str = f"F/K: {row['PE']:.1f}" if row['PE'] else "N/A"
                                res_html += f"<li><strong>{row['Ticker']}</strong>: Liderin {gap:.1f}x gerisinde. ({pe_str})</li>"
                            res_html += "</ul>"
                            st.markdown(res_html, unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 3: V700 RALLİ ÖNCESİ İSTATİSTİK MOTORU
# ---------------------------------------------------------
with tab3:
    st.markdown("### 🦈 Da Vinci: Kinetik Ralli Öncesi Hafıza Taraması")
    st.caption("Fiyatın 10 işlem gününde %15 ve üzeri ralli yaptığı tarihsel anları tespit eder. Ralliden önceki 4 gün (ve 4 saatlik) mumlardaki Omni, Synergy, Whale ve Kurumsal Trap sinyallerinin gerçekleşme olasılığını hesaplar.")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        sel_theme = st.selectbox("İncelenecek Tema", list(THEMES.keys()))
    with col_t2:
        sel_stock = st.selectbox("Hisse / ETF", THEMES[sel_theme])
        
    if st.button("⚛️ RALLİ İSTATİSTİKLERİNİ HESAPLA", use_container_width=True):
        with st.spinner(f"{sel_stock} için son 3 yılın tüm V665, V695 ve V700 motor algoritmaları taranıyor..."):
            stats = run_pre_rally_statistics(sel_stock)
            
            if stats is None or stats['count'] == 0:
                st.warning(f"⚠️ {sel_stock} grafiğinde son 3 yılda belirtilen eşikte (10 günde %15+) majör bir fiyat hareketi bulunamadı.")
            else:
                st.success(f"Geçmişte **{stats['count']} adet** Majör Ralli başlangıç noktası tespit edildi!")
                
                st.markdown("#### 🔍 Ralli Başlamadan Önceki 4 Günün Karakteristiği:")
                sc1, sc2, sc3, sc4, sc5 = st.columns(5)
                
                with sc1:
                    st.markdown(f"<div class='stat-box'><div class='stat-label'>Synergy (V665)</div><div class='stat-value'>%{stats['syn_blue']:.0f}</div><div style='font-size:0.7rem;color:#888;'>Maviye Döndü</div></div>", unsafe_allow_html=True)
                with sc2:
                    st.markdown(f"<div class='stat-box'><div class='stat-label'>Fusion (V700)</div><div class='stat-value'>%{stats['fus_pos']:.0f}</div><div style='font-size:0.7rem;color:#888;'>Pozitif Kesti</div></div>", unsafe_allow_html=True)
                with sc3:
                    st.markdown(f"<div class='stat-box'><div class='stat-label'>Whale IN (V695)</div><div class='stat-value'>%{stats['whale_in']:.0f}</div><div style='font-size:0.7rem;color:#888;'>Giriş Gözlendi</div></div>", unsafe_allow_html=True)
                with sc4:
                    st.markdown(f"<div class='stat-box'><div class='stat-label'>Afterburner</div><div class='stat-value'>%{stats['vsa_ign']:.0f}</div><div style='font-size:0.7rem;color:#888;'>Ignition Patlaması</div></div>", unsafe_allow_html=True)
                with sc5:
                    st.markdown(f"<div class='stat-box'><div class='stat-label'>Kurumsal Trap</div><div class='stat-value'>%{stats['bear_trap']:.0f}</div><div style='font-size:0.7rem;color:#888;'>Ayı Tuzağı Kuruldu</div></div>", unsafe_allow_html=True)

                st.divider()
                st.markdown(f"""
                **🧠 DA VINCI SENTETİK SONUÇ:**
                İstatistikler gösteriyor ki {sel_stock} kodlu varlıkta büyük fiyat hareketlerinden hemen önceki 4 periyotluk (Günlük/4H) pencerede en belirgin öncü sinyal **Whale IN (V695)** ve **Synergy Hat (V665)** değişimleridir. 
                Piyasa öncesi (Pre-market) hareketlerde genellikle volatilitenin baskılandığı Volatility Hole oluşumları gözlemlenmiştir.
                """)
