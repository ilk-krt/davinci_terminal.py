"""
AETHER APEX — makro rejim, tema takibi ve TradingView sinyal motorlarının
çok sembollü tarayıcıya dönüştürülmüş hali.

Çalıştırma:  streamlit run app.py
"""

from __future__ import annotations

import datetime as dt
import json
import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from apex import data as dta
from apex import engine as eng
from apex import holdings as hld
from apex import macro as mac
from apex import news as nws
from apex import playbook as pb
from apex import screener as scr
from apex import themes as thm
from apex import universe as uni
from apex.store import Storage, StorageError, storage_from_secrets
from apex.ui import (
    CHART_LAYOUT, SERIES, badge, inject_theme, kpi, section, signal_style,
)

logging.basicConfig(level=logging.INFO)

st.set_page_config(layout="wide", page_title="AETHER APEX", page_icon="🏛️",
                   initial_sidebar_state="collapsed")
inject_theme()

TTL_FAST = 300      # 5 dk  — sinyaller
TTL_SLOW = 900      # 15 dk — makro, tema
TTL_NEWS = 300


# ==========================================================================
# ÖNBELLEKLİ VERİ KATMANI
# ==========================================================================
@st.cache_resource
def get_store() -> Storage:
    return storage_from_secrets(getattr(st, "secrets", None),
                                local_path="apex_watchlist.json")


@st.cache_data(ttl=TTL_SLOW, show_spinner=False)
def load_macro(nonce: str) -> mac.MacroState:
    prices, failed = dta.fetch(mac.MACRO_TICKERS.values(), "1d")
    by_key = {k: prices.get(v, pd.DataFrame()) for k, v in mac.MACRO_TICKERS.items()}
    state = mac.build_macro_state(by_key)
    if failed:
        state.errors.append("Çekilemeyen makro sembol: " + ", ".join(failed))
    return state


@st.cache_data(ttl=TTL_FAST, show_spinner=False)
def scan(tickers: tuple[str, ...], interval: str, nonce: str) -> pd.DataFrame:
    """Verilen sembolleri tarar ve sinyal tablosunu döner."""
    tickers = tuple(sorted(set(tickers)))
    if not tickers:
        return pd.DataFrame()

    need = list(tickers) + [eng.BENCHMARK]
    prices, failed = dta.fetch(need, interval)
    bench = prices.get(eng.BENCHMARK)
    bench_close = bench["Close"] if bench is not None and not bench.empty else None

    weekly_map: dict[str, bool] = {}
    if interval == "1d":
        wk, _ = dta.fetch(tickers, "1wk")
        for t, df in wk.items():
            if len(df) > 12:
                ema12 = df["Close"].ewm(span=12, adjust=False).mean()
                weekly_map[t] = bool(df["Close"].iloc[-1] > ema12.iloc[-1])

    rows: list[dict] = []
    for t in tickers:
        df = prices.get(t)
        if df is None or df.empty:
            rows.append({"Sembol": t, "Sinyal": "⚫ VERİ YOK",
                         "Hata": "Fiyat verisi çekilemedi"})
            continue
        row = eng.analyze(df, t, bench_close=bench_close,
                          weekly_bull=weekly_map.get(t))
        if not row.ok:
            rows.append({"Sembol": t, "Sinyal": "⚫ VERİ YOK", "Hata": row.error})
            continue
        rec = {"Sembol": t, **row.data, "Hata": ""}
        rows.append(rec)

    out = pd.DataFrame(rows)
    if "MAGNITUDE" in out.columns:
        out = out.sort_values("MAGNITUDE", ascending=False)
    return out


@st.cache_data(ttl=TTL_SLOW, show_spinner=False)
def theme_performance(nonce: str) -> pd.DataFrame:
    """Tema bazlı çok periyotlu performans + ivme değişimi."""
    etfs = sorted({t for lst in uni.THEME_TRACKER.values() for t in lst})
    prices, failed = dta.fetch(etfs, "1d")

    perf: dict[str, dict[str, float]] = {}
    year = dt.date.today().year
    for t, df in prices.items():
        c = df["Close"].dropna()
        if len(c) < 8:
            continue

        def chg(a: int, b: int = 0) -> float:
            if len(c) <= a:
                return np.nan
            end = c.iloc[-1 - b]
            start = c.iloc[-1 - a]
            return (end / start - 1) * 100 if start else np.nan

        ytd_df = c[c.index.year == year]
        ytd = ((c.iloc[-1] / ytd_df.iloc[0] - 1) * 100
               if len(ytd_df) > 1 else np.nan)
        prev_year = c[c.index.year == year - 1]
        ytd_prev = np.nan
        if len(prev_year) > 1:
            doy = dt.date.today().timetuple().tm_yday
            upto = prev_year[prev_year.index.dayofyear <= doy]
            if len(upto) > 1:
                ytd_prev = (upto.iloc[-1] / prev_year.iloc[0] - 1) * 100

        perf[t] = {
            "Bugün": chg(1), "Prev_Bugün": chg(2, 1),
            "1H": chg(5), "Prev_1H": chg(10, 5),
            "1A": chg(21), "Prev_1A": chg(42, 21),
            "3A": chg(63), "Prev_3A": chg(126, 63),
            "YBB": ytd, "Prev_YBB": ytd_prev,
        }

    pdf = pd.DataFrame.from_dict(perf, orient="index")
    if pdf.empty:
        return pdf

    rows = {}
    for tema, lst in uni.THEME_TRACKER.items():
        valid = [t for t in lst if t in pdf.index]
        if valid:
            rows[tema] = pdf.loc[valid].mean()
            rows[tema]["Semboller"] = ", ".join(valid)
    out = pd.DataFrame.from_dict(rows, orient="index")
    return out


@st.cache_data(ttl=TTL_NEWS, show_spinner=False)
def load_news(topics: tuple[str, ...], nonce: str):
    return nws.fetch_news(uni.all_stocks() + uni.all_etfs(), topics)


@st.cache_data(ttl=3600, show_spinner=False)
def load_earnings(tickers: tuple[str, ...], nonce: str) -> pd.DataFrame:
    return dta.fetch_earnings_calendar(tickers)


# ==========================================================================
# DURUM
# ==========================================================================
def _init_state() -> None:
    defaults = {
        "nonce_macro": "0", "nonce_scan": "0", "nonce_theme": "0",
        "nonce_news": "0", "nonce_earn": "0",
        "manual_scenario": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

    if "watchlist" not in st.session_state:
        store = get_store()
        try:
            saved = store.load(default={}).data
        except StorageError as exc:
            saved = {}
            st.session_state["store_error"] = str(exc)
        if not isinstance(saved, dict):
            saved = {}
        st.session_state.watchlist = {
            "future_themes": saved.get("future_themes")
            or {k: dict(v) for k, v in uni.DEFAULT_FUTURE_THEMES.items()},
            "earnings": saved.get("earnings") or list(uni.DEFAULT_EARNINGS),
        }


def scan_gate(key: str, tickers: list[str], interval: str, label: str,
              auto: bool = False) -> pd.DataFrame:
    """
    Ağır taramaları butona bağlar.

    Sebep: sekme açılır açılmaz 400 sembol taramak arayüzü dakikalarca
    kilitliyordu. Artık kullanıcı isteyince çalışıyor, sonuç oturumda
    saklanıyor; 5 dakikalık önbellek zaten arka planda devrede.
    """
    state_key = f"scanres_{key}"
    n = len(tickers)
    c1, c2 = st.columns([1, 3])
    clicked = c1.button(f"🔍 {label} ({n} sembol)", key=f"btn_{key}",
                        width="stretch", type="primary")
    have = state_key in st.session_state

    if clicked:
        with st.spinner(f"{n} sembol {interval} taranıyor… "
                        f"(yaklaşık {max(5, int(n * 0.16))} sn)"):
            st.session_state[state_key] = scan(tuple(tickers), interval,
                                               st.session_state.nonce_scan)
        have = True
    elif auto and not have and n <= 60:
        with st.spinner(f"{n} sembol taranıyor…"):
            st.session_state[state_key] = scan(tuple(tickers), interval,
                                               st.session_state.nonce_scan)
        have = True

    if not have:
        c2.caption("Tarama başlatılmadı — yukarıdaki butona basın. Sonuç bu "
                   "oturumda saklanır, sekmeler arasında geçince kaybolmaz.")
        return pd.DataFrame()
    return st.session_state[state_key]


def bump(key: str) -> None:
    st.session_state[key] = str(dt.datetime.now().timestamp())


def save_watchlist() -> bool:
    store = get_store()
    try:
        store.save(st.session_state.watchlist, "APEX izleme listesi güncellendi")
        return True
    except StorageError as exc:
        st.error(f"Kaydedilemedi: {exc}")
        return False


_init_state()
store = get_store()

# ==========================================================================
# BAŞLIK
# ==========================================================================
head_l, head_r = st.columns([4, 1])
with head_l:
    st.markdown(
        "<div class='nx-brand'><h1>AETHER APEX</h1>"
        "<span class='tag'>Live</span></div>", unsafe_allow_html=True)
with head_r:
    if st.button("⚡ Tümünü Yenile", width="stretch", type="primary"):
        for k in ("nonce_macro", "nonce_scan", "nonce_theme", "nonce_news"):
            bump(k)
        st.rerun()

with st.spinner("Makro göstergeler çekiliyor…"):
    M = load_macro(st.session_state.nonce_macro)

meta = [f"Güncelleme <b>{M.asof}</b>",
        f"OPEX <b>{M.opex_date}</b> ({M.opex_days} gün)"]
if M.opex_quad:
    meta.append("<b>ÜÇLÜ CADI</b>")
if M.fomc_days is not None:
    meta.append(f"FOMC <b>{M.fomc_date}</b> ({M.fomc_days} gün)")
meta.append(f"Kayıt <b>{'GitHub' if store.backend == 'github' else 'yerel'}</b>")
st.markdown(f"<div class='nx-meta'>{'  ·  '.join(meta)}</div>",
            unsafe_allow_html=True)

if store.backend == "local":
    st.warning(
        "**Kalıcılık kapalı.** Future Themes ve bilanço listesi sadece geçici "
        "dosyaya yazılıyor; Streamlit Cloud uygulamayı uyuttuğunda silinir. "
        "Secrets içine `[github]` bölümünü ekleyin.", icon="⚠️")
for e in M.errors:
    st.warning(e, icon="⚠️")

MARKET_REGIME_OK = M.scores.get("trend", 50) >= 50

TABS = st.tabs([
    "🌐 Makro & Rejim", "🔥 Tema Takibi", "🦅 ETF Radarı", "⚖️ Çarpan Uçurumu",
    "🦈 Haftalık", "🚨 4H Omni Swing", "🚀 Future Themes", "📅 Bilanço",
])
(tab_macro, tab_theme, tab_etf, tab_val, tab_week, tab_omni,
 tab_future, tab_earn) = TABS


# ==========================================================================
# 1) MAKRO & REJİM
# ==========================================================================
with tab_macro:
    c1, c2, c3, c4 = st.columns(4)
    risk_tone = "pos" if M.risk_score >= 60 else "neg" if M.risk_score <= 40 else ""
    c1.markdown(kpi("Tespit Edilen Rejim", M.regime,
                    f"bileşik risk skoru {M.risk_score:.0f}/100", risk_tone),
                unsafe_allow_html=True)
    vix = M.get("VIX")
    c2.markdown(kpi("VIX", f"{vix:.1f}" if np.isfinite(vix) else "—",
                    M.readings["VIX"].detail if "VIX" in M.readings else "",
                    "neg" if np.isfinite(vix) and vix > 25 else "pos"),
                unsafe_allow_html=True)
    c3.markdown(kpi("OPEX'e Kalan", f"{M.opex_days} gün",
                    ("Üçlü cadı — etki güçlü" if M.opex_quad
                     else f"{M.opex_date}"),
                    "neg" if M.opex_days <= 2 else ""),
                unsafe_allow_html=True)
    c4.markdown(kpi("Piyasa Rejim Kapısı",
                    "AÇIK" if MARKET_REGIME_OK else "KAPALI",
                    "SPY 50 EMA üstünde — long sinyaller geçerli"
                    if MARKET_REGIME_OK else
                    "SPY 50 EMA altında — long sinyal kalitesi düşer",
                    "pos" if MARKET_REGIME_OK else "neg"),
                unsafe_allow_html=True)

    st.info(f"**{M.regime}** — {M.regime_desc}")

    left, right = st.columns([3, 2])

    with left:
        section("Canlı makro göstergeler")
        if M.readings:
            rd = pd.DataFrame([{
                "Gösterge": r.label,
                "Değer": r.value,
                "Değişim %": r.change_pct,
                "Yorum": r.detail,
            } for r in M.readings.values()])
            st.dataframe(
                rd, width="stretch", hide_index=True,
                column_config={
                    "Değer": st.column_config.NumberColumn(format="%.2f"),
                    "Değişim %": st.column_config.NumberColumn(format="%+.2f%%"),
                    "Yorum": st.column_config.TextColumn(width="large"),
                })

        section("Sermaye akış eğilimi")
        st.caption("Elle yazılmış senaryo sabitleri değil — yukarıdaki canlı "
                   "göstergelerden hesaplanır.")
        bat = M.battery
        if st.session_state.manual_scenario:
            bat = mac.MANUAL_SCENARIOS[st.session_state.manual_scenario]["battery"]
        keys = list(bat)
        bar = go.Figure(go.Bar(
            x=[bat[k] for k in keys], y=keys, orientation="h",
            marker=dict(color=[SERIES[i % len(SERIES)] for i in range(len(keys))],
                        cornerradius=4),
            text=[f"{bat[k]}" for k in keys], textposition="outside",
            hovertemplate="%{y}: %{x}/100<extra></extra>"))
        bar.update_layout(height=260, bargap=0.35,
                          xaxis=dict(range=[0, 108], showgrid=False,
                                     showticklabels=False),
                          yaxis=dict(showgrid=False), **CHART_LAYOUT)
        st.plotly_chart(bar, width="stretch")

        section("Senaryo karşılaştırma (elle)")
        cols = st.columns(len(mac.MANUAL_SCENARIOS) + 1)
        if cols[0].button("Ölçülen", width="stretch"):
            st.session_state.manual_scenario = None
            st.rerun()
        for i, name in enumerate(mac.MANUAL_SCENARIOS, start=1):
            if cols[i].button(name, width="stretch"):
                st.session_state.manual_scenario = name
                st.rerun()
        if st.session_state.manual_scenario:
            sc = mac.MANUAL_SCENARIOS[st.session_state.manual_scenario]
            st.caption(f"**{st.session_state.manual_scenario}** — {sc['desc']}")

    with right:
        section("Kavram sözlüğü")
        for title, body in mac.REGIME_GLOSSARY:
            with st.expander(title):
                st.write(body)

    # ---------------- REJİM OYUN KİTABI ----------------
    section("Bu rejimde ne çalışır, ne çalışmaz")
    aktif = pb.drivers_for(M.regime)
    st.caption("Aşağıdaki kartlar, tespit edilen rejimi besleyen sürücüleri ve "
               "her birinin tarihsel olarak hangi tarafı vurduğunu gösterir. "
               "Rejim haberden değil ölçülen göstergelerden belirlenir; haberler "
               "sadece 'neden' sorusunu cevaplar.")

    for d in aktif:
        with st.expander(f"{d.icon} {d.label}", expanded=True):
            st.markdown(d.nedir)
            st.caption(f"**Veriden nasıl anlaşılır:** {d.veri_isareti}")

            lc, rc = st.columns(2)
            if d.lehte_etf or d.lehte_hisse:
                with lc:
                    st.markdown("**🟢 Lehte çalışan**")
                    if d.lehte_etf:
                        st.markdown("ETF: " + " ".join(f"`{x}`" for x in d.lehte_etf))
                    if d.lehte_hisse:
                        st.markdown("Hisse: " + " ".join(f"`{x}`" for x in d.lehte_hisse))
            if d.aleyhte_etf or d.aleyhte_hisse:
                with rc:
                    st.markdown("**🔴 Aleyhte çalışan**")
                    if d.aleyhte_etf:
                        st.markdown("ETF: " + " ".join(f"`{x}`" for x in d.aleyhte_etf))
                    if d.aleyhte_hisse:
                        st.markdown("Hisse: " + " ".join(f"`{x}`" for x in d.aleyhte_hisse))
            if d.islem_notu:
                st.info(f"**İşlem notu:** {d.islem_notu}")

    with st.expander("Diğer rejim sürücüleri (referans)"):
        for key, d in pb.DRIVERS.items():
            if d in aktif:
                continue
            st.markdown(f"**{d.icon} {d.label}** — {d.nedir}")
            if d.lehte_hisse:
                st.caption("Lehte: " + ", ".join(d.lehte_hisse[:8])
                           + (" · Aleyhte: " + ", ".join(d.aleyhte_hisse[:8])
                              if d.aleyhte_hisse else ""))
            st.markdown("---")

    section("Canlı haber akışı")
    nc1, nc2 = st.columns([4, 1])
    topics = nc1.multiselect("Konular", list(nws.FEEDS), default=list(nws.FEEDS),
                             label_visibility="collapsed")
    if nc2.button("🔄 Haberleri yenile", width="stretch"):
        bump("nonce_news")
        st.rerun()
    with st.spinner("Haber akışları taranıyor…"):
        items, nerr = load_news(tuple(topics), st.session_state.nonce_news)
    for e in nerr:
        st.caption(f"⚠️ {e}")
    if items:
        # Her haberi rejim sürücüsüne ve etkilediği sembollere bağla
        enriched = []
        driver_counts: dict[str, int] = {}
        for it in items:
            keys = pb.match_drivers(it["Başlık"])
            for k in keys:
                driver_counts[k] = driver_counts.get(k, 0) + 1
            imp = pb.impacted(keys)
            enriched.append({
                **it,
                "Rejim Sürücüsü": " ".join(
                    f"{pb.DRIVERS[k].icon}{pb.DRIVERS[k].label.split(' /')[0]}"
                    for k in keys) or "—",
                "🟢 Lehte": ", ".join((imp["lehte_etf"] + imp["lehte_hisse"])[:6]),
                "🔴 Aleyhte": ", ".join((imp["aleyhte_etf"] + imp["aleyhte_hisse"])[:6]),
            })

        if driver_counts:
            st.markdown("**Haber akışında şu an baskın olan sürücüler**")
            dc = st.columns(min(4, len(driver_counts)))
            for col, (k, n) in zip(dc, sorted(driver_counts.items(),
                                              key=lambda x: -x[1])):
                d = pb.DRIVERS[k]
                col.markdown(kpi(f"{d.icon} {d.label.split(' /')[0]}",
                                 f"{n} haber",
                                 "bu rejimi besliyor" if d in aktif
                                 else "rejimle eşleşmiyor",
                                 "pos" if d in aktif else ""),
                             unsafe_allow_html=True)
            st.caption("Haber sayısı bir rejimi KANITLAMAZ; ölçülen göstergelerle "
                       "aynı yönü gösteriyorsa teyit, göstermiyorsa erken uyarı "
                       "olarak okuyun.")

        st.dataframe(
            pd.DataFrame(enriched)[["Konu", "Tarih", "Başlık", "Rejim Sürücüsü",
                                    "İlgili", "🟢 Lehte", "🔴 Aleyhte", "Link"]],
            width="stretch", hide_index=True,
            column_config={
                "Link": st.column_config.LinkColumn("Link", width="small"),
                "Başlık": st.column_config.TextColumn(width="large"),
                "Rejim Sürücüsü": st.column_config.TextColumn(width="medium"),
                "🟢 Lehte": st.column_config.TextColumn(width="medium"),
                "🔴 Aleyhte": st.column_config.TextColumn(width="medium"),
            })


# ==========================================================================
# 2) TEMA TAKİBİ
# ==========================================================================
with tab_theme:
    tc1, tc2 = st.columns([4, 1])
    period = tc1.radio("Periyot", ["Bugün", "1H", "1A", "3A", "YBB"],
                       horizontal=True, index=2, label_visibility="collapsed")
    if tc2.button("🔄 Yenile", width="stretch", key="theme_refresh"):
        bump("nonce_theme")
        st.rerun()

    with st.expander("📐 Getiri ve İvme ne demek? (bir kez okuyun)", expanded=False):
        st.markdown(f"""
Her tema için **iki ayrı sayı** var ve bunlar farklı şeyler söyler:

| | Tanım | Örnek (**{period}** seçiliyken) |
|---|---|---|
| **Getiri %** | Seçilen dönemin yüzde değişimi | {thm.PERIOD_LABELS.get(period, period)} içindeki % değişim |
| **İvme** | Bu dönemin getirisi **eksi** bir önceki eşdeğer dönemin getirisi | (bu dönem) − ({thm.PERIOD_PREV.get(period, "önceki dönem")}) |

İvme pozitifse tema **hızlanıyor**, negatifse **yavaşlıyor**.

Neden ikisi de gerekli: bir tema %18 kazandırmış olabilir, ama önceki dönem
%39 kazandırdıysa ivme **−21**'dir — para hâlâ giriyor, fakat giriş hızı
yarıya inmiş; liderlik el değiştirmek üzere olabilir. Tersine −%4 getirili
bir tema önceki dönem −%15 yaptıysa ivmesi **+11**'dir ve dipten dönüş tam
buradan başlar.

Bu iki eksen dört çeyrek üretir; asıl karar çeyrekten çıkar:
""")
        for q in thm.QUADRANTS.values():
            st.markdown(f"- {q.icon} **{q.label}** — {q.aciklama}  \n"
                        f"  _Ne yapmalı:_ {q.aksiyon}")

    with st.spinner("Tema performansı hesaplanıyor…"):
        T = theme_performance(st.session_state.nonce_theme)

    if T.empty:
        st.warning("Tema verisi çekilemedi.", icon="⚠️")
    else:
        TQ = thm.build_table(T, period)
        ornek = thm.worked_example(TQ, period)
        if ornek:
            st.info(ornek)

        section("Momentum × İvme haritası")
        st.caption("Yatay eksen: dönem getirisi. Dikey eksen: ivme (önceki "
                   "eşdeğer döneme göre hızlanma). Sağ üst çeyrek avlanma "
                   "sahası, sol üst çeyrek dipten dönüş adayları. Kalabalık "
                   "olmasın diye sadece merkeze en uzak 16 tema etiketlenir; "
                   "diğerlerinin adı için noktanın üzerine gelin.")
        # Etiket kalabalığını önlemek için sadece merkeze en uzak temalar
        # yazılır; kalanlar noktayla kalır ve üzerine gelince okunur.
        _mesafe = (TQ["Getiri %"].fillna(0) ** 2 + TQ["İvme"].fillna(0) ** 2) ** 0.5
        _etiketli = set(_mesafe.nlargest(16).index)

        qfig = go.Figure()
        for key, q in thm.QUADRANTS.items():
            sel = TQ[TQ["_q"] == key]
            if sel.empty:
                continue
            qfig.add_trace(go.Scatter(
                x=sel["Getiri %"], y=sel["İvme"], mode="markers+text",
                name=f"{q.icon} {q.label}",
                text=[t if t in _etiketli else "" for t in sel.index],
                customdata=list(sel.index),
                textposition="top center", textfont=dict(size=11, color="#d5d5de"),
                marker=dict(size=13, color=q.renk, line=dict(color="#050506",
                                                             width=1.5)),
                hovertemplate=("<b>%{customdata}</b><br>Getiri %{x:.2f}%"
                               "<br>İvme %{y:+.2f}<extra></extra>")))
        qfig.add_hline(y=0, line=dict(color="#3a3a48", width=1))
        qfig.add_vline(x=0, line=dict(color="#3a3a48", width=1))
        qfig.update_layout(
            height=560, legend=dict(orientation="h", y=-0.14),
            xaxis=dict(title="Dönem getirisi %", gridcolor="#1b1b22",
                       zeroline=False),
            yaxis=dict(title="İvme (hızlanma)", gridcolor="#1b1b22",
                       zeroline=False),
            **CHART_LAYOUT)
        st.plotly_chart(qfig, width="stretch")

        özet = thm.summary(TQ)
        if özet:
            cols = st.columns(len(özet))
            for col, (key, info) in zip(cols, özet.items()):
                q = info["quadrant"]
                col.markdown(kpi(f"{q.icon} {q.label}", f"{info['n']} tema",
                                 f"ort. getiri %{info['ort_getiri']:+.1f} · "
                                 f"ort. ivme {info['ort_ivme']:+.1f}",
                                 "pos" if key == "lider_hizlanan"
                                 else "neg" if key == "hizlanan_dusus" else ""),
                            unsafe_allow_html=True)

        section("Sıralı performans")
        srt = T.sort_values(period, ascending=True)
        labels, texts, colors = [], [], []
        for tema, row in srt.iterrows():
            syms = str(row.get("Semboller", ""))
            short = syms if len(syms) < 26 else syms[:24] + "…"
            labels.append(f"{tema}  <span style='font-size:11px;color:#6e6e7a'>"
                          f"{short}</span>")
            val = row[period]
            prev = row.get(f"Prev_{period}", np.nan)
            delta = val - prev if np.isfinite(prev) else np.nan
            dtxt = ("" if not np.isfinite(delta)
                    else f"  (🔺+{delta:.1f})" if delta > 0
                    else f"  (🔻{delta:.1f})")
            texts.append(f"{val:+.2f}%{dtxt}")
            colors.append("#3987e5" if val >= 0 else "#d55181")

        fig = go.Figure(go.Bar(
            y=labels, x=srt[period], orientation="h",
            marker=dict(color=colors, cornerradius=3),
            text=texts, textposition="outside",
            textfont=dict(size=12), hovertemplate="%{x:.2f}%<extra></extra>"))
        rng = srt[period].max() - srt[period].min()
        pad = rng * 0.42 if rng else 5
        fig.update_layout(
            height=max(700, 22 * len(srt)),
            xaxis=dict(range=[srt[period].min() - pad, srt[period].max() + pad],
                       showgrid=False, zeroline=True, zerolinecolor="#2b2b36",
                       showticklabels=False),
            yaxis=dict(showgrid=False, tickfont=dict(size=12)),
            **CHART_LAYOUT)
        st.plotly_chart(fig, width="stretch", config={
            "toImageButtonOptions": {"format": "svg",
                                     "filename": f"TemaTakibi_{period}",
                                     "height": 1000, "width": 1400}})

        with st.expander("Tema tablosu — getiri, ivme ve çeyrek"):
            show = TQ.drop(columns=["_q"])
            st.dataframe(show, width="stretch",
                         column_config={
                             "Getiri %": st.column_config.NumberColumn(format="%+.2f%%"),
                             "Önceki %": st.column_config.NumberColumn(format="%+.2f%%"),
                             "İvme": st.column_config.NumberColumn(format="%+.2f"),
                             "Semboller": st.column_config.TextColumn(width="medium"),
                         })

    # --- Swing yorumu ---
    section("Swing işlem karakteri — canlı tarama")
    st.caption("Aşağıdaki gruplar sabit bir liste değil; ETF bileşenlerinin "
               "güncel ATR, likidite ve sinyal verisinden her yenilemede "
               "yeniden hesaplanır.")

    swing_universe = sorted({t for etf in
                             ["XLK", "SOXX", "SMH", "XLE", "XLI", "XLV", "XLU",
                              "IGV", "CIBR", "ARKX", "WGMI", "PAVE", "URA"]
                             for t in uni.holdings(etf)})
    S = scan_gate("swing_char", swing_universe, "1d",
                  "Swing karakter taramasını çalıştır")

    if not S.empty and "ATR %" in S.columns:
        rows = S[S["Sinyal"] != "⚫ VERİ YOK"].to_dict("records")
        recs = scr.build_recommendations(rows, scr.SwingFilters(), {},
                                         MARKET_REGIME_OK)
        cm = scr.swing_commentary(recs, M.regime, MARKET_REGIME_OK)

        for grup, kayitlar in cm["gruplar"].items():
            st.markdown(f"**{grup}**")
            names = ", ".join(f"`{r['Sembol']}` (ATR %{r['ATR %']:.1f}, "
                              f"skor {r['Skor']})" for r in kayitlar)
            st.markdown(names)
            st.caption(scr.character_note(kayitlar[0]["ATR %"]))

        st.markdown("**Sektör ETF'leri** — `XLE`, `XLV`, `SMH`, `XLI`, `XLU`: "
                    "tek hisse haber riskini seyreltir, çoklu-hisse "
                    "backtest'inde daha temiz istatistik üretir.")

        st.markdown("**Devrede olan pratik filtreler**")
        for ad, aciklama in cm["filtreler"]:
            st.markdown(f"- **{ad}** — {aciklama}")
        for n in cm["notlar"]:
            st.markdown(f"- {n}")


# ==========================================================================
# 3) ETF RADARI
# ==========================================================================
with tab_etf:
    kats = uni.etfs_by_category()
    ec1, ec2 = st.columns([4, 1])
    sec_kat = ec1.multiselect("Kategoriler", list(kats), default=list(kats))
    if ec2.button("🔄 Yenile", width="stretch", key="etf_refresh"):
        bump("nonce_scan")
        st.rerun()

    etf_list = sorted({e for k in sec_kat for e in kats[k]}
                      | set(uni.MAIN_SECTORS))
    E = scan_gate("etf", etf_list, "1d", "ETF sinyallerini tara", auto=True)

    if E.empty:
        st.warning("Veri çekilemedi.", icon="⚠️")
    else:
        E = E.copy()
        E["Kapsam"] = E["Sembol"].map(
            lambda s: uni.ETF.get(s, {}).get("name")
            or uni.MAIN_SECTORS.get(s, "—"))
        cols = ["Sembol", "Kapsam", "Sinyal", "Efor", "Fiyat", "1 Gün %",
                "1 Hafta %", "WHALE", "ΔWHALE", "Whale Yön", "PRO-RET",
                "ΔPRO-RET", "OMNI", "ΔOMNI", "OMNI Yön", "HUD /6",
                "MAGNITUDE", "ΔMAG", "DIRECTION", "ΔDIR", "Hata"]
        cols = [c for c in cols if c in E.columns]
        st.dataframe(
            E[cols].style.map(signal_style, subset=["Sinyal"]),
            width="stretch", hide_index=True,
            column_config={
                "Fiyat": st.column_config.NumberColumn(format="$%.2f"),
                "1 Gün %": st.column_config.NumberColumn(format="%+.2f%%"),
                "1 Hafta %": st.column_config.NumberColumn(format="%+.2f%%"),
                "WHALE": st.column_config.ProgressColumn(
                    format="%.0f", min_value=0, max_value=100),
                "ΔWHALE": st.column_config.NumberColumn(
                    format="%+.1f", help="Bir önceki bara göre WHALE değişimi"),
                "Whale Yön": st.column_config.TextColumn(
                    width="small",
                    help="1 barlık ve 5 barlık değişimin birleşimi"),
                "PRO-RET": st.column_config.NumberColumn(format="%+.1f"),
                "ΔPRO-RET": st.column_config.NumberColumn(format="%+.1f"),
                "OMNI": st.column_config.NumberColumn(format="%.0f"),
                "ΔOMNI": st.column_config.NumberColumn(format="%+.1f"),
                "OMNI Yön": st.column_config.TextColumn(width="small"),
                "ΔMAG": st.column_config.NumberColumn(format="%+.0f"),
                "ΔDIR": st.column_config.NumberColumn(format="%+.0f"),
                "Kapsam": st.column_config.TextColumn(width="medium"),
            })
        st.caption("**Δ sütunları** bir önceki bara göre değişimi gösterir; "
                   "**Yön** sütunu 1 barlık ve 5 barlık değişimi birleştirir: "
                   "⇈ güçleniyor · ↗ dönüyor · → yatay · ↘ soluklanıyor · "
                   "⇊ bozuluyor. WHALE 72 tek başına anlamsızdır — 60'tan mı "
                   "yükseldi yoksa 85'ten mi düştü, karar buna bağlıdır.")

    # --- ETF içeriği ve içerideki hisselerin göreli gücü ---
    section("ETF içi röntgen — kim taşıyor, kim geride kaldı")
    sel = st.selectbox("ETF seçin", uni.all_etfs(),
                       format_func=lambda s: f"{s} — {uni.ETF[s]['name']}")
    if sel:
        d = uni.ETF[sel]
        st.markdown(f"**{d['name']}** · {d['kategori']}")
        if d["aciklama"]:
            st.caption(d["aciklama"])

        hold = uni.holdings(sel)

        # ---------- 1) Bileşen taraması + göreli güç ----------
        if not hold:
            st.info("Bu ETF için tanımlı bileşen listesi yok.")
        else:
            pc1, pc2 = st.columns([1, 2])
            period_col = pc1.radio(
                "Kıyas penceresi", list(hld.PERIOD_COLS),
                format_func=lambda c: hld.PERIOD_COLS[c],
                horizontal=True, index=1, key="hold_period")
            pc2.caption(
                "Getiriler bu pencerede ölçülür. **Akrana Göre** = hissenin "
                "getirisi − aynı listedeki hisselerin medyanı; **Z** bu farkın "
                "medyan mutlak sapmaya bölünmüş hâlidir (tek bir dev bileşenin "
                "listeyi çarpıtmasını engeller). **ETF'e Göre** ise sepetin "
                "kendi getirisiyle kıyastır ve ağırlık etkisini içerir.")

            H = scan_gate(f"hold_{sel}", hold + [sel], "1d",
                          f"{sel} içindeki hisseleri tara",
                          auto=len(hold) <= 30)

            if not H.empty:
                etf_row = H[H["Sembol"] == sel]
                etf_row = etf_row.iloc[0].to_dict() if not etf_row.empty else None
                comp = H[(H["Sembol"] != sel) & (H["Sinyal"] != "⚫ VERİ YOK")]
                HT = hld.build_holdings_table(comp, sel, period_col, etf_row)

                if HT.empty:
                    st.warning("Bileşen verisi çekilemedi.", icon="⚠️")
                else:
                    for line in hld.holdings_narrative(HT, sel, period_col):
                        st.markdown(line)

                    cnt = hld.holdings_counts(HT)
                    kcols = st.columns(5)
                    for col, key in zip(kcols, ["lider", "lider_yorgun",
                                                "uyumlu", "geride_akis_var",
                                                "geride_akis_yok"]):
                        v = hld.VERDICTS[key]
                        tone = ("pos" if key == "lider" else
                                "neg" if key == "geride_akis_yok" else "")
                        col.markdown(
                            kpi(f"{v.icon} {v.label}", str(cnt.get(key, 0)),
                                "hisse", tone), unsafe_allow_html=True)

                    with st.expander("Bu beş durum nasıl belirleniyor?"):
                        for v in hld.VERDICTS.values():
                            st.markdown(
                                f"**{v.icon} {v.label}** — {v.aciklama}  \n"
                                f"↳ *{v.aksiyon}*")
                        st.caption(
                            f"Eşikler: akrana göre Z ≤ −{hld.Z_ESIK} ve fark "
                            f"≥ {hld.MIN_FARK_PP} puan ise 'geride', Z ≥ "
                            f"+{hld.Z_ESIK} ise 'lider'. Geride kalanlar, "
                            "kurumsal akış (WHALE/ΔWHALE, toplama-dağıtım, "
                            "likidite süpürmesi) yönüne göre ikiye ayrılır: "
                            "akış içerideyse yakalama adayı, dışarıdaysa tuzak.")

                    # ---------- 2) Ayrışma haritası ----------
                    fig = go.Figure()
                    for key, v in hld.VERDICTS.items():
                        sub = HT[HT["_durum"] == key]
                        if sub.empty:
                            continue
                        fig.add_trace(go.Scatter(
                            x=sub["Akrana Göre"], y=sub["WHALE"],
                            mode="markers+text", text=sub["Sembol"],
                            textposition="top center",
                            textfont=dict(size=10, color="#c2c2cc"),
                            name=f"{v.icon} {v.label}",
                            marker=dict(size=13, color=v.renk,
                                        line=dict(color="#050506", width=1.5)),
                            customdata=np.stack([
                                sub["Getiri %"].fillna(0),
                                sub["ΔWHALE"].fillna(0) if "ΔWHALE" in sub else
                                pd.Series(0, index=sub.index)], axis=-1),
                            hovertemplate=("<b>%{text}</b><br>Akrana göre "
                                           "%{x:+.2f} puan<br>WHALE %{y:.0f}"
                                           "<br>Getiri %{customdata[0]:+.2f}%"
                                           "<br>ΔWHALE %{customdata[1]:+.1f}"
                                           "<extra></extra>")))
                    fig.add_vline(x=0, line=dict(color="#3a3a48", width=1))
                    fig.add_hline(y=50, line=dict(color="#3a3a48", width=1,
                                                  dash="dot"))
                    fig.update_layout(
                        height=430, **CHART_LAYOUT,
                        xaxis_title="Akran medyanına göre fark (puan)",
                        yaxis_title="WHALE — kurumsal akış",
                        legend=dict(orientation="h", y=1.12, x=0))
                    fig.update_xaxes(gridcolor="#1b1b22", zeroline=False)
                    fig.update_yaxes(gridcolor="#1b1b22", zeroline=False)
                    st.plotly_chart(fig, width="stretch")
                    st.caption(
                        "**Sol üst köşe kritiktir:** fiyat akranlarının "
                        "gerisinde (sol) ama kurumsal akış yüksek (üst) — "
                        "yani hisse geri kaldı, para hâlâ içeride. Sol alt "
                        "köşe ise hem fiyatın hem akışın terk ettiği bacak.")

                    # ---------- 3) Tablo ----------
                    tcols = [c for c in [
                        "Sembol", "Durum", "Ağırlık %", "Getiri %", "ETF'e Göre",
                        "Akrana Göre", "Z", "Katkı pp", "Sinyal", "Efor",
                        "WHALE", "ΔWHALE", "Whale Yön", "OMNI", "ΔOMNI",
                        "MAGNITUDE", "DIRECTION", "ATR %", "RS Sıra", "Rejim",
                        "Neden"] if c in HT.columns]
                    st.dataframe(
                        HT[tcols].style.map(signal_style, subset=["Sinyal"]),
                        width="stretch", hide_index=True, height=460,
                        column_config={
                            "Ağırlık %": st.column_config.NumberColumn(
                                format="%.2f%%"),
                            "Getiri %": st.column_config.NumberColumn(
                                format="%+.2f%%"),
                            "ETF'e Göre": st.column_config.NumberColumn(
                                format="%+.2f", help="Hisse getirisi − ETF getirisi"),
                            "Akrana Göre": st.column_config.NumberColumn(
                                format="%+.2f",
                                help="Hisse getirisi − bileşen medyanı"),
                            "Z": st.column_config.NumberColumn(
                                format="%+.2f",
                                help="Akran farkının MAD ile standartlaştırılmışı"),
                            "Katkı pp": st.column_config.NumberColumn(
                                format="%+.3f",
                                help="Ağırlık × getiri — ETF getirisinin kaç "
                                     "puanını bu isim açıklıyor"),
                            "WHALE": st.column_config.ProgressColumn(
                                format="%.0f", min_value=0, max_value=100),
                            "ΔWHALE": st.column_config.NumberColumn(format="%+.1f"),
                            "OMNI": st.column_config.NumberColumn(format="%.0f"),
                            "ΔOMNI": st.column_config.NumberColumn(format="%+.1f"),
                            "MAGNITUDE": st.column_config.NumberColumn(format="%.0f"),
                            "DIRECTION": st.column_config.NumberColumn(format="%+.0f"),
                            "ATR %": st.column_config.NumberColumn(format="%.1f%%"),
                            "RS Sıra": st.column_config.NumberColumn(format="%.0f"),
                            "Whale Yön": st.column_config.TextColumn(width="small"),
                            "Durum": st.column_config.TextColumn(width="medium"),
                            "Neden": st.column_config.TextColumn(width="large"),
                        })

                    # ---------- 4) Hisse hisse teknik durum ----------
                    section(f"{sel} bileşenlerinin teknik durumu")
                    only_lag = st.checkbox(
                        "Sadece geride kalanları göster", value=False,
                        key=f"lagonly_{sel}")
                    show = HT[HT["_durum"].isin(["geride_akis_var",
                                                 "geride_akis_yok"])] \
                        if only_lag else HT
                    if show.empty:
                        st.info("Bu pencerede geride kalan bileşen yok.")
                    for _, r in show.iterrows():
                        v = hld.VERDICTS[r["_durum"]]
                        w = (f" · ağırlık %{r['Ağırlık %']:.2f}"
                             if pd.notna(r.get("Ağırlık %")) else "")
                        with st.expander(
                                f"{v.icon} {r['Sembol']} — {r['Sinyal']} · "
                                f"{hld.PERIOD_COLS[period_col]} "
                                f"%{r['Getiri %']:+.2f} "
                                f"(akrana göre {r['Akrana Göre']:+.2f}){w}"):
                            if r.get("Rol"):
                                st.caption(f"Roldeki yeri: {r['Rol']}")
                            st.markdown(f"**Durum:** {v.label} — {r['Neden']}")
                            st.markdown(r["Teknik Not"])
                            st.caption(f"Ne yapılır: {v.aksiyon}")
                            diger = [x for x in uni.etfs_containing(r["Sembol"])
                                     if x != sel]
                            if diger:
                                st.caption("Ayrıca şu temalarda: "
                                           + ", ".join(f"`{x}`" for x in diger))

        # ---------- 5) Ağırlık dağılımı (referans) ----------
        with st.expander("Ağırlık dağılımı ve grup listeleri"):
            if d["agirlik"]:
                wdf = pd.DataFrame([{
                    "Sembol": t,
                    "Ağırlık %": d["agirlik"].get(t),
                    "Rol": d["rol"].get(t, ""),
                    "Diğer ETF'ler": ", ".join(
                        x for x in uni.etfs_containing(t) if x != sel)[:60],
                } for t in hold])
                st.dataframe(wdf, width="stretch", hide_index=True,
                             column_config={
                                 "Ağırlık %": st.column_config.NumberColumn(
                                     format="%.2f%%"),
                                 "Rol": st.column_config.TextColumn(
                                     width="large")})
                top = [t for t in hold if d["agirlik"].get(t)][:10]
                if top:
                    pie = go.Figure(go.Pie(
                        labels=top, values=[d["agirlik"][t] for t in top],
                        hole=0.55, sort=False,
                        marker=dict(colors=[SERIES[i % len(SERIES)]
                                            for i in range(len(top))],
                                    line=dict(color="#050506", width=2)),
                        textinfo="label+percent", textposition="inside"))
                    pie.update_layout(height=360, showlegend=False, **CHART_LAYOUT)
                    st.plotly_chart(pie, width="stretch")
            if d["gruplar"]:
                for grup, lst in d["gruplar"].items():
                    st.markdown(f"**{grup}:** "
                                + ", ".join(f"`{t}`" for t in lst))


# ==========================================================================
# 4) ÇARPAN UÇURUMU
# ==========================================================================
with tab_val:
    st.markdown("Bir hissenin kaç farklı temada birden yer aldığını gösterir. "
                "Bu kesişim kümesi, paranın hangi alt temaya gittiğinden bağımsız "
                "olarak pastadan pay alan **altyapı sahiplerini** ortaya çıkarır.")

    section("Chokepoint çarpanları")
    for sym, info in uni.CHOKEPOINTS.items():
        etfs = uni.etfs_containing(sym)
        with st.expander(f"{sym} — {info['rol']}  ({len(etfs)} temada)"):
            st.markdown(f"**Beslendiği CapEx kaynağı:** {info['capex']}")
            st.markdown(info["mantik"])
            if etfs:
                st.caption("Bağladığı temalar: " + ", ".join(f"`{e}`" for e in etfs))

    section("Kesişim yoğunluğu — tüm evren")
    counts = [{"Sembol": t, "Tema Sayısı": len(uni.etfs_containing(t)),
               "Temalar": ", ".join(uni.etfs_containing(t))}
              for t in uni.all_stocks()]
    cdf = pd.DataFrame(counts).sort_values("Tema Sayısı", ascending=False)
    cdf = cdf[cdf["Tema Sayısı"] >= 2]
    st.dataframe(cdf.head(40), width="stretch", hide_index=True,
                 column_config={"Temalar": st.column_config.TextColumn(
                     width="large")})


# ==========================================================================
# 5) HAFTALIK MOMENTUM
# ==========================================================================
with tab_week:
    if st.button("🔄 Haftalık veriyi yenile", key="wk"):
        bump("nonce_scan")
        st.rerun()
    universe = uni.all_stocks()
    W = scan_gate("weekly", universe, "1wk", "Haftalık momentumu tara")
    if not W.empty:
        cols = [c for c in ["Sembol", "Sinyal", "Efor", "Fiyat", "1 Hafta %",
                            "WHALE", "ΔWHALE", "ΔWHALE 5B", "Whale Yön",
                            "PRO-RET", "ΔPRO-RET", "MAGNITUDE", "ΔMAG",
                            "DIRECTION", "ΔDIR", "OMNI", "ΔOMNI", "OMNI Yön",
                            "Hata"] if c in W.columns]
        st.dataframe(W[cols].style.map(signal_style, subset=["Sinyal"]),
                     width="stretch", hide_index=True,
                     column_config={
                         "Fiyat": st.column_config.NumberColumn(format="$%.2f"),
                         "1 Hafta %": st.column_config.NumberColumn(format="%+.2f%%"),
                         "WHALE": st.column_config.ProgressColumn(
                             format="%.0f", min_value=0, max_value=100),
                         "ΔWHALE": st.column_config.NumberColumn(format="%+.1f"),
                         "ΔWHALE 5B": st.column_config.NumberColumn(
                             format="%+.1f",
                             help="5 bar önceki değere göre değişim"),
                         "Whale Yön": st.column_config.TextColumn(width="small"),
                         "ΔPRO-RET": st.column_config.NumberColumn(format="%+.1f"),
                         "ΔMAG": st.column_config.NumberColumn(format="%+.0f"),
                         "ΔDIR": st.column_config.NumberColumn(format="%+.0f"),
                         "ΔOMNI": st.column_config.NumberColumn(format="%+.1f"),
                         "OMNI Yön": st.column_config.TextColumn(width="small")})
        st.caption("Haftalık barlarda Δ, bir hafta önceki değere göre "
                   "değişimdir; Δ5B ise beş hafta öncesine göre. Kurumsal "
                   "toplama (WHALE) yükselirken fiyatın yatay kalması, "
                   "scriptlerdeki 'stealth accumulation' durumudur.")


# ==========================================================================
# 6) 4H OMNI SWING
# ==========================================================================
with tab_omni:
    st.markdown("ETF bileşenleri arasından **swing işlem adayları**. Sıralama "
                "konfluans motorunun MAGNITUDE/DIRECTION skoruna, kurumsal akışa "
                "ve göreli güce göre yapılır; ardından pratik filtreler uygulanır.")

    f1, f2, f3 = st.columns(3)
    src_etfs = f1.multiselect(
        "Kaynak ETF'ler", uni.all_etfs(),
        default=["XLK", "SOXX", "SMH", "IGV", "XLU", "PAVE", "URA", "ARKX"])
    interval = f2.selectbox("Zaman dilimi", ["1d", "4h", "1wk"], index=0,
                            format_func=lambda x: {"1d": "Günlük",
                                                   "4h": "4 Saatlik",
                                                   "1wk": "Haftalık"}[x])
    min_score = f3.slider("Minimum skor", 0, 100, 45, 5)

    g1, g2, g3, g4 = st.columns(4)
    min_liq = g1.number_input("Min likidite ($M)", 0.0, 500.0, 5.0, 1.0)
    earn_buf = g2.number_input("Bilanço tamponu (gün)", 0, 10, 2, 1)
    use_regime = g3.toggle("Rejim kapısı", value=True,
                           help="SPY 50 EMA altındayken long sinyalleri ele")
    use_weekly = g4.toggle("Haftalık teyit", value=False)

    cand = sorted({t for e in src_etfs for t in uni.holdings(e)})
    st.caption(f"{len(cand)} aday sembol · kaynak: "
               + ", ".join(f"`{e}`" for e in src_etfs))

    if cand:
        R = scan_gate(f"omni_{interval}", cand, interval,
                      "Swing taramasını çalıştır")

        earn_map: dict[str, int] = {}
        if earn_buf > 0:
            with st.spinner("Bilanço takvimi kontrol ediliyor…"):
                edf = load_earnings(tuple(cand[:120]), st.session_state.nonce_earn)
            for _, r in edf.iterrows():
                if r.get("Kalan Gün") is not None and np.isfinite(
                        float(r["Kalan Gün"] or np.nan)):
                    earn_map[r["Hisse"]] = int(r["Kalan Gün"])

        if not R.empty:
            rows = R[R["Sinyal"] != "⚫ VERİ YOK"].to_dict("records")
            filt = scr.SwingFilters(min_dollar_vol_m=min_liq,
                                    earnings_buffer_days=int(earn_buf),
                                    require_regime=use_regime,
                                    require_weekly=use_weekly,
                                    min_score=min_score)
            RECS = scr.build_recommendations(rows, filt, earn_map, MARKET_REGIME_OK)

            if RECS.empty:
                st.info("Aday bulunamadı.")
            else:
                # her adayın hangi ETF'ten geldiğini işaretle
                RECS["ETF"] = RECS["Sembol"].map(
                    lambda t: ", ".join(e for e in uni.etfs_containing(t)
                                        if e in src_etfs)[:24])
                uygun = RECS[RECS["Uygun"]]
                k1, k2, k3 = st.columns(3)
                k1.markdown(kpi("Filtreyi Geçen", str(len(uygun)),
                                f"{len(RECS)} aday tarandı",
                                "pos" if len(uygun) else ""), unsafe_allow_html=True)
                en_iyi = uygun.iloc[0] if len(uygun) else None
                k2.markdown(kpi("En Yüksek Skor",
                                en_iyi["Sembol"] if en_iyi is not None else "—",
                                (f"skor {en_iyi['Skor']} · {en_iyi['Sinyal']}"
                                 if en_iyi is not None else "")),
                            unsafe_allow_html=True)
                k3.markdown(kpi("Piyasa Rejimi",
                                "AÇIK" if MARKET_REGIME_OK else "KAPALI",
                                M.regime, "pos" if MARKET_REGIME_OK else "neg"),
                            unsafe_allow_html=True)

                section("Tavsiye edilen adaylar")
                show_cols = [c for c in
                             ["Sembol", "ETF", "Skor", "Sinyal", "Karakter", "Fiyat",
                              "ATR %", "Hacim ($M)", "WHALE", "ΔWHALE",
                              "Whale Yön", "OMNI", "ΔOMNI", "MAGNITUDE", "ΔMAG",
                              "DIRECTION", "RS Sıra", "Stop", "T1", "T2", "R (T2)",
                              "Risk %", "Bilanço Gün"] if c in uygun.columns]
                st.dataframe(
                    uygun[show_cols].style.map(signal_style, subset=["Sinyal"]),
                    width="stretch", hide_index=True,
                    column_config={
                        "Skor": st.column_config.ProgressColumn(
                            format="%d", min_value=0, max_value=100),
                        "Fiyat": st.column_config.NumberColumn(format="$%.2f"),
                        "Stop": st.column_config.NumberColumn(format="$%.2f"),
                        "T1": st.column_config.NumberColumn(format="$%.2f"),
                        "T2": st.column_config.NumberColumn(format="$%.2f"),
                        "ATR %": st.column_config.NumberColumn(format="%.2f%%"),
                        "Risk %": st.column_config.NumberColumn(format="%.2f%%"),
                        "R (T2)": st.column_config.NumberColumn(format="%.2fR"),
                        "Hacim ($M)": st.column_config.NumberColumn(format="%.0f"),
                        "RS Sıra": st.column_config.NumberColumn(format="%.0f"),
                        "ΔWHALE": st.column_config.NumberColumn(format="%+.1f"),
                        "ΔOMNI": st.column_config.NumberColumn(format="%+.1f"),
                        "ΔMAG": st.column_config.NumberColumn(format="%+.0f"),
                        "Whale Yön": st.column_config.TextColumn(width="small"),
                    })
                st.caption("**Stop** = iz süren ATR zırhı (low − 2×ATR, sadece yukarı "
                           "kayar, girişten %20 aşağıda sert taban). **T1/T2** = "
                           "1.8× ve 3.5× ATR, volatilite sıkışmasıyla ölçeklenir. "
                           "**R (T2)** = hedefe giden mesafenin riske oranı.")

                with st.expander(f"Filtreye takılanlar ({len(RECS) - len(uygun)})"):
                    red = RECS[~RECS["Uygun"]]
                    st.dataframe(
                        red[[c for c in ["Sembol", "Skor", "Sinyal", "Engel"]
                             if c in red.columns]],
                        width="stretch", hide_index=True,
                        column_config={"Engel": st.column_config.TextColumn(
                            width="large")})


# ==========================================================================
# 7) FUTURE THEMES
# ==========================================================================
with tab_future:
    wl = st.session_state.watchlist["future_themes"]

    st.markdown("Kendi ayıkladığınız hisse ve ETF'ler. Ekleme/çıkarma "
                + ("**GitHub deposuna kalıcı yazılır**." if store.backend == "github"
                   else "sadece bu oturumda tutulur (Secrets ekleyin)."))

    mc1, mc2 = st.columns([3, 1])
    tema_sec = mc1.selectbox("Tema", list(wl) + ["➕ Yeni tema oluştur…"])
    if tema_sec == "➕ Yeni tema oluştur…":
        yeni = mc2.text_input("Tema adı", key="yeni_tema")
        if st.button("Tema oluştur") and yeni.strip():
            wl[yeni.strip()] = {"hisse": [], "etf": []}
            if save_watchlist():
                st.success(f"'{yeni.strip()}' oluşturuldu.")
                st.rerun()
    else:
        tema = wl[tema_sec]
        a1, a2 = st.columns(2)
        with a1:
            st.markdown("**Hisse ekle**")
            yeni_h = st.text_input("Semboller (virgül/boşluk ile)",
                                   key="fh_add",
                                   placeholder="NVDA, AVGO CEG")
            if st.button("➕ Hisse ekle", width="stretch"):
                import re as _re
                yeni = [x.strip().upper() for x in _re.split(r"[,\s]+", yeni_h)
                        if x.strip()]
                tema["hisse"] = sorted(set(tema["hisse"]) | set(yeni))
                if save_watchlist():
                    st.success(f"{len(yeni)} sembol eklendi.")
                    st.rerun()
            cik_h = st.multiselect("Çıkarılacak hisseler", tema["hisse"],
                                   key="fh_del")
            if cik_h and st.button("🗑️ Seçilen hisseleri çıkar", width="stretch"):
                tema["hisse"] = [t for t in tema["hisse"] if t not in cik_h]
                if save_watchlist():
                    st.rerun()
        with a2:
            st.markdown("**ETF ekle**")
            yeni_e = st.multiselect("ETF seç", uni.all_etfs(), key="fe_add")
            ekstra_e = st.text_input("Listede olmayan ETF", key="fe_txt",
                                     placeholder="ITA, KWEB")
            if st.button("➕ ETF ekle", width="stretch"):
                import re as _re
                extra = [x.strip().upper() for x in _re.split(r"[,\s]+", ekstra_e)
                         if x.strip()]
                tema["etf"] = sorted(set(tema["etf"]) | set(yeni_e) | set(extra))
                if save_watchlist():
                    st.rerun()
            cik_e = st.multiselect("Çıkarılacak ETF'ler", tema["etf"], key="fe_del")
            if cik_e and st.button("🗑️ Seçilen ETF'leri çıkar", width="stretch"):
                tema["etf"] = [t for t in tema["etf"] if t not in cik_e]
                if save_watchlist():
                    st.rerun()

        if st.button(f"❌ '{tema_sec}' temasını tamamen sil"):
            wl.pop(tema_sec, None)
            if save_watchlist():
                st.rerun()

    section("Tüm Future Themes evreni")
    fut_tickers = sorted({t for v in wl.values()
                          for t in (v.get("hisse", []) + v.get("etf", []))})
    st.caption(f"{len(fut_tickers)} sembol · {len(wl)} tema")
    if fut_tickers:
        F = scan_gate("future", fut_tickers, "1d", "Future Themes tara",
                      auto=True)
        if not F.empty:
            F = F.copy()
            F["Tema"] = F["Sembol"].map(
                lambda t: ", ".join(k for k, v in wl.items()
                                    if t in v.get("hisse", []) + v.get("etf", [])))
            cols = [c for c in ["Sembol", "Tema", "Sinyal", "Efor", "Fiyat",
                                "1 Gün %", "WHALE", "OMNI", "MAGNITUDE",
                                "DIRECTION", "Hata"] if c in F.columns]
            st.dataframe(F[cols].style.map(signal_style, subset=["Sinyal"]),
                         width="stretch", hide_index=True,
                         column_config={
                             "Fiyat": st.column_config.NumberColumn(format="$%.2f"),
                             "1 Gün %": st.column_config.NumberColumn(
                                 format="%+.2f%%"),
                             "Tema": st.column_config.TextColumn(width="medium")})

    with st.expander("Yedek al / geri yükle"):
        st.download_button(
            "⬇️ İzleme listesini indir",
            json.dumps(st.session_state.watchlist, indent=2,
                       ensure_ascii=False).encode("utf-8"),
            "apex_watchlist.json", "application/json")
        up = st.file_uploader("Yedekten yükle", type="json")
        if up is not None and st.button("📥 Uygula"):
            try:
                st.session_state.watchlist = json.loads(
                    up.getvalue().decode("utf-8"))
                if save_watchlist():
                    st.success("Yüklendi.")
                    st.rerun()
            except Exception as exc:
                st.error(f"Okunamadı: {exc}")


# ==========================================================================
# 8) BİLANÇO
# ==========================================================================
with tab_earn:
    lst = st.session_state.watchlist["earnings"]
    e1, e2 = st.columns([4, 1])
    add_txt = e1.text_input("Sembol ekle (virgül/boşluk ile)",
                            placeholder="AAPL, TSLA MSFT")
    if e2.button("➕ Ekle", width="stretch"):
        import re as _re
        new = [x.strip().upper() for x in _re.split(r"[,\s]+", add_txt)
               if x.strip()]
        st.session_state.watchlist["earnings"] = sorted(set(lst) | set(new))
        if save_watchlist():
            st.rerun()

    with st.expander(f"Listeyi yönet ({len(lst)} sembol)"):
        rem = st.multiselect("Çıkarılacaklar", lst)
        if rem and st.button("🗑️ Çıkar"):
            st.session_state.watchlist["earnings"] = [t for t in lst if t not in rem]
            if save_watchlist():
                st.rerun()

    if st.button("🔄 Bilanço takvimini tara", type="primary"):
        bump("nonce_earn")
        st.rerun()

    with st.spinner("Bilanço tarihleri ve analist hedefleri çekiliyor…"):
        EARN = load_earnings(tuple(lst), st.session_state.nonce_earn)

    if not EARN.empty:
        def style_days(v):
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                return ""
            if v <= 2:
                return "background-color:#4a0d12;color:#fff;font-weight:700"
            if v <= 7:
                return "background-color:#3a2a05;color:#ffd54f;font-weight:700"
            if v <= 15:
                return "color:#ffd54f"
            return ""

        def style_pot(v):
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                return ""
            return ("color:#2fbe86;font-weight:700" if v > 0
                    else "color:#f0736f;font-weight:700")

        st.dataframe(
            EARN.style.map(style_days, subset=["Kalan Gün"])
                      .map(style_pot, subset=["Potansiyel %"]),
            width="stretch", hide_index=True,
            column_config={
                "Fiyat": st.column_config.NumberColumn(format="$%.2f"),
                "Hedef": st.column_config.NumberColumn(format="$%.2f"),
                "Potansiyel %": st.column_config.NumberColumn(format="%+.1f%%"),
                "Kalan Gün": st.column_config.NumberColumn(format="%d"),
            })
        st.caption("Kırmızı satırlar bilanço tamponu içindedir — swing "
                   "taramasında bu hisseler otomatik elenir. **Hedef**, analist "
                   "ortalama fiyat beklentisidir; bir değerleme modeli değil, "
                   "duyarlılık göstergesidir.")
