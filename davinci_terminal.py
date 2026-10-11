"""
AETHER APEX — Streamlit arayüzü (giriş dosyası)

Hesap kodu apex/ paketindedir; bu dosya yalnızca ekranı kurar.
Ağır hesaplar her akşam tools/build_snapshot.py ile yapılır ve data/snapshot/
klasörüne yazılır; arayüz önce oradan okur, yoksa canlı hesaplar.

Çalıştırma:  streamlit run davinci_terminal.py
"""
from __future__ import annotations

from apex.store import Storage, StorageError, _log, storage_from_secrets
from apex.ui import CHART_LAYOUT, SERIES, badge, inject_theme, kpi, section, signal_style
from apex import data as dta
from apex import decision as dcs
from apex import engine as eng
from apex import fundchart as fch
from apex import funnel as fnl
from apex import holdings as hld
from apex import macro as mac
from apex import news as nws
from apex import playbook as pb
from apex import report as rep
from apex import screener as scr
from apex import shortvol as svm
from apex import themes as thm
from apex import universe as uni
from apex import valuation as fvm
from apex import compute as cmp
from apex import snapshot as snap
from apex import plan as pln
from apex import flows as flw
from apex import gold as gld
from apex import markets as mkt
from apex import reading as rdg
from apex import seasonal as sea
from apex import tfview as tfv
from apex import tvchart as tvc
from apex import tracking as trk
from apex import hype as hyp
from apex import robot as rbt
from apex import robotchart as rbc

# --- Modül kısayolları -------------------------------------------------------
# Modüler sürümde `an` = analytics, `px` = prices modülüydü. Tek dosyada hepsi
# aynı isim alanında olduğundan ikisini de bu dosyanın global alanına bağlıyoruz.
# (sys.modules kullanılmıyor: Streamlit betiği kendi isim alanında çalıştırır.)



import datetime as dt
import json
import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


logging.basicConfig(level=logging.INFO)

# --------------------------------------------------------------------------
# KOYU TEMA — her koşulda
# Uygulama koyu zemine göre tasarlandı. Streamlit temayı yalnızca
# `.streamlit/config.toml` dosyasından okur; dosya yoksa ya da yanlış yerdeyse
# açık temaya düşer ve Streamlit'in kendi yazıları (giriş kutuları, menüler,
# tablolar, grafik açıklamaları) siyah kalır → siyah zeminde görünmez.
# 1) Tema seçenekleri burada da koyuya sabitlenir.
# 2) Grafikler Streamlit temasından bağımsız, koyu Plotly şablonuyla çizilir.
# 3) CSS (inject_theme) widget yazılarını açık renge zorlar.
# --------------------------------------------------------------------------
_theme_changed = False
try:
    from streamlit import config as _st_config
    for _k, _v in {"theme.base": "dark", "theme.primaryColor": "#00e5ff",
                   "theme.backgroundColor": "#050506",
                   "theme.secondaryBackgroundColor": "#0d0d11",
                   "theme.textColor": "#ececf1"}.items():
        if _st_config.get_option(_k) != _v:
            _st_config.set_option(_k, _v)
            _theme_changed = True
except Exception as _exc:                       # pragma: no cover
    logging.info("Tema seçeneği ayarlanamadı: %s", _exc)
# Tema, sayfa çalışmaya başlarken tarayıcıya gönderilir; bu çalıştırmada
# yeni ayarlandıysa bir kez yeniden çalıştırınca tablolar da koyu açılır.
if _theme_changed and not st.session_state.get("_theme_rerun"):
    st.session_state["_theme_rerun"] = True
    st.rerun()

import plotly.io as pio

_apex_tpl = go.layout.Template(pio.templates["plotly_dark"])
_apex_tpl.layout.paper_bgcolor = "rgba(0,0,0,0)"
_apex_tpl.layout.plot_bgcolor = "rgba(0,0,0,0)"
_apex_tpl.layout.font = dict(color="#c2c2cc")
_apex_tpl.layout.legend = dict(font=dict(color="#c2c2cc"))
for _ax in ("xaxis", "yaxis"):
    _apex_tpl.layout[_ax].gridcolor = "#1f1f28"
    _apex_tpl.layout[_ax].zerolinecolor = "#2b2b36"
    _apex_tpl.layout[_ax].linecolor = "#2b2b36"
pio.templates["apex"] = _apex_tpl
pio.templates.default = "apex"

_st_plotly_chart = st.plotly_chart


def _plotly_chart(fig, *args, **kwargs):
    """Streamlit'in açık temasının grafik renklerini ezmesini engeller ve grafiği
    SABİTLER: tıklayınca/sürükleyince yakınlaşma, seri gizleme olmaz (görüntü
    bozulmaz). Kamera düğmesi seçili formatta (PNG/JPEG/SVG) indirir."""
    kwargs.setdefault("theme", None)
    try:
        fig.update_layout(dragmode=False)
        fig.update_xaxes(fixedrange=True)
        fig.update_yaxes(fixedrange=True)
        fig.update_layout(legend=dict(itemclick=False, itemdoubleclick=False))
    except Exception:                       # noqa: BLE001 (alt grafik türleri)
        pass
    fmt = st.session_state.get("img_fmt", "png") if hasattr(st, "session_state") else "png"
    cfg = {"displaylogo": False, "scrollZoom": False, "doubleClick": False,
           "modeBarButtonsToRemove": ["zoom2d", "pan2d", "select2d", "lasso2d",
                                      "zoomIn2d", "zoomOut2d", "autoScale2d",
                                      "resetScale2d", "toggleSpikelines"],
           "toImageButtonOptions": {"format": fmt, "scale": 2,
                                    "filename": "aether_grafik"}}
    cfg.update(kwargs.pop("config", {}) or {})
    return _st_plotly_chart(fig, *args, config=cfg, **kwargs)


st.plotly_chart = _plotly_chart

st.set_page_config(layout="wide", page_title="AETHER APEX", page_icon="🏛️",
                   initial_sidebar_state="collapsed")
inject_theme()

TTL_FAST = 300      # 5 dk  — sinyaller
TTL_SLOW = 900      # 15 dk — makro, tema
TTL_NEWS = 300


# ==========================================================================
# ÖNBELLEKLİ VERİ KATMANI
# ==========================================================================

# --------------------------------------------------------------------------
# Akşam hesabı (data/snapshot) — arayüz önce buradan okur.
# Bir sekmede "🔄 yenile"ye basılınca o sekmenin nonce'u "0"dan çıkar ve
# yalnızca o kısım canlı hesaplanır.
# --------------------------------------------------------------------------
def _use_snap(nonce: str) -> bool:
    return str(nonce) == "0"


def _snap_scan(tickers, interval: str, min_cover: float = 0.0):
    """Akşam taramasında bulunan satırlar ve eksik semboller.
    Döner: (bulunanlar DataFrame, eksik semboller) ya da kapsama yetersizse None."""
    if interval not in ("1d", "1wk"):
        return None
    s = snap.load(f"scan_{interval}")
    if s is None or s.empty or "Sembol" not in s.columns:
        return None
    want = set(tickers)
    have = s[s["Sembol"].isin(want)]
    if not want or len(have) / len(want) < min_cover:
        return None
    return have.reset_index(drop=True), sorted(want - set(have["Sembol"]))


@st.cache_resource
def get_store() -> Storage:
    return storage_from_secrets(getattr(st, "secrets", None),
                                local_path="apex_watchlist.json")


@st.cache_data(ttl=TTL_SLOW, show_spinner=False)
def load_macro_frames(nonce: str) -> dict[str, pd.DataFrame]:
    """Makro seriler — akşam hesabından, yoksa canlı."""
    if _use_snap(nonce):
        s = snap.load("macro_frames")
        if s is not None:
            return s
    return cmp.macro_frames()

@st.cache_data(ttl=TTL_SLOW, show_spinner=False)
def load_macro(nonce: str) -> mac.MacroState:
    return cmp.macro_state(load_macro_frames(nonce))

@st.cache_data(ttl=TTL_SLOW, show_spinner=False)
def macro_history(nonce: str, days: int = 60):
    """Batarya seyri + dönemsel değişim tabloları."""
    return cmp.macro_history(load_macro_frames(nonce), days)

@st.cache_data(ttl=TTL_FAST, show_spinner=False)
def scan(tickers: tuple[str, ...], interval: str, nonce: str) -> pd.DataFrame:
    """Sinyal taraması — akşam taraması kapsıyorsa oradan, yoksa canlı."""
    if _use_snap(nonce):
        r = _snap_scan(tickers, interval)
        if r is not None:
            have, missing = r
            if not missing:
                return have
            if len(have):                       # yalnızca eksikler canlı
                live = cmp.scan(tuple(missing), interval)
                out = pd.concat([have, live], ignore_index=True)
                if "MAGNITUDE" in out.columns:
                    out = out.sort_values("MAGNITUDE", ascending=False)
                return out.reset_index(drop=True)
    return cmp.scan(tickers, interval)

@st.cache_data(ttl=TTL_SLOW, show_spinner=False)
def theme_performance(nonce: str) -> pd.DataFrame:
    """Tema bazlı çok periyotlu performans + ivme değişimi."""
    if _use_snap(nonce):
        s = snap.load("theme_perf")
        if s is not None:
            return s
    return cmp.theme_performance()

@st.cache_data(ttl=TTL_NEWS, show_spinner=False)
def load_news(topics: tuple[str, ...], nonce: str):
    return nws.fetch_news(uni.all_stocks() + uni.all_etfs(), topics)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_earnings_cached(tickers: tuple[str, ...], nonce: str) -> pd.DataFrame:
    if _use_snap(nonce):
        s = snap.load("earnings")
        if s is not None and not s.empty and "Hisse" in s.columns:
            want = set(tickers)
            have = s[s["Hisse"].isin(want)]
            missing = sorted(want - set(have["Hisse"]))
            if len(have):
                out = have if not missing else pd.concat(
                    [have, dta.fetch_earnings_calendar(missing)], ignore_index=True)
                return out.reset_index(drop=True)
    return dta.fetch_earnings_calendar(tickers)

def _earn_ok_ratio(df: pd.DataFrame) -> float:
    if df is None or df.empty or "Fiyat" not in df.columns:
        return 0.0
    return float(pd.to_numeric(df["Fiyat"], errors="coerce").notna().mean())


def load_earnings(tickers: tuple[str, ...], nonce: str) -> pd.DataFrame:
    """Yahoo geçici olarak engellediyse (satırların yarısından fazlası boş)
    sonuç önbellekte tutulmaz — bir sonraki çalıştırmada yeniden denenir."""
    df = _load_earnings_cached(tickers, nonce)
    if tickers and _earn_ok_ratio(df) < 0.5:
        _load_earnings_cached.clear()
    return df


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _load_short_volume_cached(nonce: str):
    return svm.fetch_short_volume(30)


def load_short_volume(nonce: str):
    """FINRA günlük short hacmi. Başarısız sonuç önbellekte tutulmaz —
    bir sonraki çalıştırmada yeniden denenir."""
    df, warns, diag = _load_short_volume_cached(nonce)
    if df.empty:
        _load_short_volume_cached.clear()
    return df, warns, diag


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def _load_short_interest_cached(tickers: tuple[str, ...], nonce: str) -> pd.DataFrame:
    return svm.fetch_short_interest(tickers)


def load_short_interest(tickers: tuple[str, ...], nonce: str) -> pd.DataFrame:
    """yfinance short interest (ayda iki kez güncellenen borsa verisi)."""
    df = _load_short_interest_cached(tickers, nonce)
    if df.empty or df.drop(columns=["Sembol", "SI Tarihi"],
                           errors="ignore").notna().to_numpy().mean() < 0.2:
        _load_short_interest_cached.clear()
    return df


@st.cache_data(ttl=TTL_SLOW, show_spinner=False)
def load_rotation(nonce: str):
    """Karar Hunisi 1–2. adım: rotasyon oranları + repo verileri."""
    if _use_snap(nonce):
        s = snap.load("rotation")
        if s is not None:
            return s
    return cmp.rotation()

@st.cache_data(ttl=TTL_SLOW, show_spinner=False)
def load_theme_rotation(nonce: str):
    """Karar Hunisi tema adımı: tema ve ETF'lerde göreli rotasyon (RRG)."""
    if _use_snap(nonce):
        s = snap.load("theme_rotation")
        if s is not None:
            return s
    return cmp.theme_rotation()

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_stock_raw(sym: str, nonce: str) -> dict:
    """Hisse Analizi: tek hissenin fiyat + mali tablo + tahmin verisi."""
    return fch.fetch_raw(sym)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_peer_pe(peers: tuple[str, ...], basis: str, nonce: str) -> dict:
    """Akranların F/K geçmişi (sırayla — Yahoo hız sınırı için)."""
    out = {}
    for t in peers:
        try:
            out[t] = fch.pe_history(fch.fetch_price_eps_only(t), basis)
        except Exception:
            out[t] = pd.Series(dtype=float)
    return out


@st.cache_data(ttl=TTL_SLOW, show_spinner=False)
def load_decision(nonce: str, rot: pd.DataFrame):
    """Karar Hunisi Adım 3: S&P 500 / Nasdaq / Kripto — G/H/A modül tabloları."""
    if _use_snap(nonce):
        s = snap.load("decision")
        if s is not None:
            return s
    return cmp.decision(rot)

def theme_holdings(tema: str) -> list[str]:
    return cmp.theme_holdings(tema)


@st.cache_data(ttl=TTL_SLOW, show_spinner=False)
def load_plans(tickers: tuple[str, ...], nonce: str) -> pd.DataFrame:
    """Giriş/stop/hedef planları — akşam hesabından, eksikler canlı."""
    if _use_snap(nonce):
        parts = [x for x in (snap.load("plans"), snap.load("plans_hype"))
                 if x is not None and not x.empty and "Sembol" in x.columns]
        s = pd.concat(parts, ignore_index=True).drop_duplicates("Sembol") if parts else None
        if s is not None and not s.empty and "Sembol" in s.columns:
            have = s[s["Sembol"].isin(set(tickers))]
            missing = sorted(set(tickers) - set(have["Sembol"]))
            if not missing:
                return have.reset_index(drop=True)
            if len(have):
                return pd.concat([have, cmp.plans(missing)], ignore_index=True)
    return cmp.plans(tickers)


def theme_stock_table(tema: str, min_liq: float, idx_close) -> tuple[pd.DataFrame, list[str]]:
    """Temanın hisseleri: tarama + bilanço + short → Durum sınıfı (Adım 3b ve 5)."""
    return syms_stock_table(theme_holdings(tema), min_liq, idx_close)


def syms_stock_table(hold: list[str], min_liq: float, idx_close=None
                     ) -> tuple[pd.DataFrame, list[str]]:
    """Herhangi bir sembol listesi: tarama + bilanço + short → Durum sınıfı."""
    hold = [h for h in dict.fromkeys(hold) if h]
    if not hold:
        return pd.DataFrame(), hold
    S = scan(tuple(hold), "1d", st.session_state.nonce_scan)
    if S.empty:
        return pd.DataFrame(), hold
    EARN_T = load_earnings(tuple(hold), st.session_state.nonce_earn)
    if not EARN_T.empty:
        EARN_T = fvm.add_fair_values(EARN_T)
    SVRAW_T, _, _ = load_short_volume(st.session_state.nonce_short)
    pchg = dict(zip(S["Sembol"], pd.to_numeric(S.get("1 Hafta %"), errors="coerce")))
    SVT_T = svm.short_table(SVRAW_T, hold, pchg)
    TB = fnl.build_stock_table(S, idx_close, None, EARN_T, SVT_T, min_liq,
                               swing_score=scr.swing_score)
    return TB, hold


def _px(x) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(x):
        return "—"
    return f"{x:,.0f}" if x >= 1000 else f"{x:.2f}"


def trade_plan_table(TB: pd.DataFrame, P: pd.DataFrame, index_ok: bool) -> pd.DataFrame:
    """Tarama tablosu + plan → sade karar tablosu (sebepleriyle)."""
    if TB is None or TB.empty or P is None or P.empty:
        return pd.DataFrame()
    pm = {r["Sembol"]: r for r in P.to_dict("records")}
    rows = []
    for r in TB.to_dict("records"):
        p = pm.get(r["Sembol"], {"ok": False, "neden": "plan yok"})
        p = dict(p)
        p["ok"] = p.get("ok") is True or p.get("ok") == True  # noqa: E712 (NaN → False)
        karar, why = pln.verdict(p, r.get("Durum", ""), r.get("Sinyal", ""),
                                 r.get("Kalan Gün"), index_ok)
        ok = p["ok"]
        rows.append({
            "Karar": karar, "Hisse": r["Sembol"],
            "Fiyat": _px(p["Fiyat"] if ok else r.get("Fiyat")),
            "Giriş bölgesi": f"{_px(p['Giriş Alt'])} – {_px(p['Giriş Üst'])}" if ok else "—",
            "Stop": (f"{_px(p['Stop'])} ({p['Stop %']:+.1f}%)" if ok else "—"),
            "Kâr al 1": f"{_px(p['Hedef 1'])}" if ok else "—",
            "Kâr al 2": f"{_px(p['Hedef 2'])}" if ok else "—",
            "R:R": round(float(p["R:R"]), 1) if ok else np.nan,
            "Sebep": " · ".join(why),
            "_o": {"✅": 0, "🟡": 1, "⛔": 2}.get(karar[:1], 3),
            "_s": r.get("Huni Skoru", 0),
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["_o", "_s"], ascending=[True, False]).drop(
        columns=["_o", "_s"]).reset_index(drop=True)

# ==========================================================================
# DURUM
# ==========================================================================
def _init_state() -> None:
    defaults = {
        "nonce_macro": "0", "nonce_scan": "0", "nonce_theme": "0",
        "nonce_news": "0", "nonce_earn": "0", "nonce_short": "0",
        "nonce_funnel": "0", "nonce_stock": "0",
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
            # P/S adil değer çarpan tablosu (kaydedilmişse)
            **({"ps_multiples": saved["ps_multiples"]}
               if isinstance(saved.get("ps_multiples"), dict) else {}),
            # takip & portföy (Aşama 2)
            "sector_watch": saved.get("sector_watch") or [],
            "stock_watch": saved.get("stock_watch") or [],
            "portfolio": saved.get("portfolio") or [],
            "biotech_watch": saved.get("biotech_watch") or [],
            "robots": saved.get("robots") or [],
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
    elif (not have and _use_snap(st.session_state.nonce_scan)
          and _snap_scan(tuple(tickers), interval, 0.8) is not None):
        st.session_state[state_key] = scan(tuple(tickers), interval,
                                           st.session_state.nonce_scan)
        have = True
        c2.caption(f"📦 Akşam hesabından · {snap.summary()}. Canlı tarama için "
                   "sekmedeki 🔄 yenile düğmesine basın.")
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
        for k in ("nonce_macro", "nonce_scan", "nonce_theme", "nonce_news",
                  "nonce_funnel"):
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
_snap_sum = snap.summary()
meta.append(f"Akşam hesabı <b>{_snap_sum}</b>" if _snap_sum
            else "Akşam hesabı <b>yok — canlı</b>")
st.markdown(f"<div class='nx-meta'>{'  ·  '.join(meta)}</div>",
            unsafe_allow_html=True)
_snap_st = snap.read_status()
if _snap_st:
    _bad = [k for k, v in _snap_st.items() if not v.get("ok")]
    with st.expander(("⚠️ " if _bad else "📦 ") + "Akşam hesabı durumu"
                     + (f" — {len(_bad)} adım başarısız" if _bad else ""),
                     expanded=False):
        st.dataframe(pd.DataFrame([
            {"Adım": k, "Durum": "✅" if v.get("ok") else "❌",
             "Zaman (UTC)": v.get("ts", ""), "Süre (sn)": v.get("sure_sn"),
             "Not": v.get("not", "")} for k, v in _snap_st.items()]),
            hide_index=True, width="stretch")
        st.caption("Uygulama açılınca her şey bu sonuçlardan okunur; Yahoo'ya "
                   "gidilmez. Bir sekmedeki 🔄 düğmesi yalnızca o kısmı canlı "
                   "hesaplar. Sonuçlar her iş günü GitHub Actions ile yenilenir.")

if store.backend == "local":
    st.warning(
        "**Kalıcılık kapalı.** Future Themes ve bilanço listesi sadece geçici "
        "dosyaya yazılıyor; Streamlit Cloud uygulamayı uyuttuğunda silinir.",
        icon="⚠️")
    with st.expander("🔑 Kalıcılığı açmak için 3 adım (2 dakika)", expanded=False):
        st.markdown("""
**1) GitHub'da jeton üretin**

`github.com` → sağ üst profil → **Settings** → en altta **Developer settings**
→ **Personal access tokens** → **Fine-grained tokens** → **Generate new token**

- *Repository access*: **Only select repositories** → bu uygulamanın deposu
- *Permissions* → *Repository permissions* → **Contents: Read and write**
  (başka hiçbir izne gerek yok)
- *Expiration*: uzun bir süre seçin; süresi dolunca kayıt sessizce durmaz,
  uygulama hata gösterir.

Üretilen `github_pat_...` değerini kopyalayın — sayfadan çıkınca bir daha
gösterilmez.

**2) Streamlit Cloud'a yapıştırın**

Uygulama sayfası → sağ alt **Manage app** → **⋮** → **Settings** → **Secrets**
sekmesine aşağıdakini yapıştırıp **Save** deyin:
""")
        st.code('[github]\n'
                'token  = "github_pat_BURAYA_JETONUNUZ"\n'
                'repo   = "kullanici-adiniz/depo-adiniz"\n'
                'branch = "main"\n'
                'path   = "apex_watchlist.json"\n', language="toml")
        st.markdown("""
`repo` alanı **`kullanıcı/depo`** biçimindedir — tam URL değil.
`branch` deponuzun ana dalı (`main` ya da `master`).
`path` deponun içinde oluşturulacak dosyanın adı; elle oluşturmanıza gerek yok,
ilk kayıtta kendisi commit edilir.

**3) Kaydedin ve uygulamayı yeniden başlatın**

Secrets kaydedilince Streamlit uygulamayı otomatik yeniden başlatır. Bu uyarı
kutusu kaybolur ve üstteki bilgi satırında **Kayıt: GitHub** yazar.

> Yerel bilgisayarda çalıştırıyorsanız aynı içeriği proje klasöründe
> `.streamlit/secrets.toml` dosyasına yazın ve bu dosyayı `.gitignore`'a
> ekleyin — jeton asla depoya girmemeli.
""")
        if st.button("🔌 Bağlantıyı test et", key="store_test"):
            probe = storage_from_secrets(getattr(st, "secrets", None),
                                         local_path="apex_watchlist.json")
            if not probe.enabled:
                st.error("Secrets içinde `[github]` bölümü görünmüyor. "
                         "Kaydettiyseniz uygulamayı **Reboot** edin — Secrets "
                         "değişikliği yeniden başlatmadan okunmaz.")
            else:
                try:
                    res = probe.load(default={})
                    st.success(f"Bağlantı çalışıyor → {probe.describe()}. "
                               + (res.message or "Mevcut kayıt okundu."))
                except StorageError as exc:
                    st.error(
                        f"{exc}\n\nSık görülen sebepler: jetonun bu depoya "
                        "erişimi yok · Contents izni 'Read and write' değil · "
                        "`repo` alanı `kullanıcı/depo` biçiminde değil · "
                        "`branch` adı yanlış.")
for e in M.errors:
    st.warning(e, icon="⚠️")

MARKET_REGIME_OK = M.scores.get("trend", 50) >= 50

def stock_fundamentals(sym: str, basis: str) -> None:
    """F/K + akranlar, medyan çarpan adil fiyatı ve çarpan özeti (Hisse Analizi ve
    Hisse Avcısı ortak)."""
    if sym:
        with st.spinner(f"{sym} mali tabloları çekiliyor…"):
            RAW = load_stock_raw(sym, st.session_state.nonce_stock)
        info = RAW.get("info") or {}
        px_ = fch._price(RAW)
        if px_.empty:
            st.error(f"{sym} için fiyat verisi alınamadı. Sembolü kontrol edin "
                     "ya da birkaç dakika sonra yenileyin (Yahoo sınırlaması).")
        else:
            st.markdown(
                f"**{info.get('longName') or info.get('shortName') or sym}** · "
                f"{info.get('sector', '')} / {info.get('industry', '')} · "
                f"son fiyat **\\${px_.iloc[-1]:.2f}**")

            # ------------------------------------------------ F/K + akranlar
            section("F/K oranı (TTM) ve akran medyanı")
            sug = fch.suggest_peers(sym)
            peers_txt = st.text_input(
                "Akranlar (virgülle; varsayılan: aynı ETF'lerdeki en ağır şirketler)",
                value=", ".join(sug), key=f"sa_peers_{sym}")
            peers = [x.strip().upper() for x in peers_txt.split(",")
                     if x.strip() and x.strip().upper() != sym][:12]
            PE = fch.pe_history(RAW, basis)
            with st.spinner("Akranların F/K geçmişi hesaplanıyor…"):
                PEER = load_peer_pe(tuple(peers), basis,
                                    st.session_state.nonce_stock) if peers else {}
            MED, NPEER = fch.peer_median(PEER)
            if PE.empty:
                st.info("F/K hesaplanamadı (HBK verisi yok ya da şirket zararda).")
            else:
                own_med = float(PE.median())
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=PE.index, y=PE, mode="lines", name=f"{sym} F/K (TTM)",
                    line=dict(color="#3987e5", width=1.8)))
                if not MED.empty:
                    m2 = MED[MED.index >= PE.index.min()]
                    fig.add_trace(go.Scatter(
                        x=m2.index, y=m2, mode="lines+markers",
                        name=f"Akran medyanı ({len(PEER)} şirket)",
                        line=dict(color="#c98500", width=2),
                        marker=dict(size=4)))
                fig.add_hline(y=own_med, line_dash="dot", line_color="#6e6e7a",
                              annotation_text=f"{sym} kendi medyanı {own_med:.1f}",
                              annotation_font_color="#a0a0ab")
                fig.update_layout(height=380, legend=dict(orientation="h", y=1.08),
                                  yaxis_title="F/K", **CHART_LAYOUT)
                st.plotly_chart(fig, width="stretch")
                now = float(PE.iloc[-1])
                parts = [f"Şu an F/K **{now:.1f}**, kendi medyanı {own_med:.1f} "
                         f"({(now / own_med - 1) * 100:+.0f}%)."]
                if not MED.empty:
                    pm = float(MED.iloc[-1])
                    parts.append(f"Akran medyanı **{pm:.1f}** — hisse akranlarına "
                                 f"göre {(now / pm - 1) * 100:+.0f}% "
                                 + ("primli." if now > pm else "iskontolu."))
                st.markdown(" ".join(parts))
                if PEER:
                    eksik = [t for t in peers if t not in PEER or PEER[t].empty]
                    if eksik:
                        st.caption("F/K hesaplanamayan akranlar (zarar ya da veri "
                                   "yok): " + ", ".join(eksik))

            # ------------------------------------------------ adil fiyat çizgisi
            section("Fiyat ve medyan çarpan adil fiyatı")
            b1, b2 = st.columns([2, 1])
            mkey = b1.radio("Metrik", list(fch.METRICS),
                            format_func=lambda k: f"{fch.METRICS[k]['ad']} "
                                                  f"({fch.METRICS[k]['kisa']})",
                            horizontal=True, key="sa_metric")
            win = b2.selectbox("Medyan penceresi", [0, 3, 5],
                               format_func=lambda y: "Tüm geçmiş" if y == 0
                               else f"Son {y} yıl", key="sa_win")
            D = fch.fair_value_chart_data(RAW, mkey, win)
            if not D["ok"]:
                st.info(D["neden"])
            else:
                kp = st.columns(5)
                kp[0].markdown(kpi("Fiyat", f"${D['price'].iloc[-1]:.2f}",
                                   f"toplam {D['total']:+.0f}% · yıllık "
                                   f"{D['cagr']:+.1f}%"), unsafe_allow_html=True)
                kp[1].markdown(kpi(f"Medyan {D['metric']['kisa']}",
                                   f"{D['median']:.2f}",
                                   f"şu an {D['mult_now']:.2f}"),
                               unsafe_allow_html=True)
                kp[2].markdown(kpi("Medyan çarpanla adil fiyat",
                                   f"${D['fair_now']:.2f}", "son bilinen TTM"),
                               unsafe_allow_html=True)
                kp[3].markdown(kpi("Adile göre", f"{D['gap']:+.1f}%",
                                   "fiyat adilin üstünde" if D["gap"] > 0
                                   else "fiyat adilin altında",
                                   "neg" if D["gap"] > 15 else
                                   "pos" if D["gap"] < -10 else ""),
                               unsafe_allow_html=True)
                kp[4].markdown(kpi("Korelasyon", f"%{D['corr'] * 100:.0f}"
                                   if np.isfinite(D["corr"]) else "—",
                                   "fiyat ↔ metrik çizgisi"),
                               unsafe_allow_html=True)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=D["price"].index, y=D["price"], mode="lines", name="Fiyat",
                    line=dict(color="#3987e5", width=1.6)))
                fig.add_trace(go.Scatter(
                    x=D["fair"].index, y=D["fair"], mode="lines",
                    name=f"Medyan {D['metric']['kisa']} = {D['median']:.2f} ile fiyat",
                    line=dict(color="#2fbe86", width=2, shape="hv")))
                pts = D["points"][D["points"].index >= D["start"]]
                fig.add_trace(go.Scatter(
                    x=pts.index, y=pts, mode="markers", showlegend=False,
                    marker=dict(color="#2fbe86", size=7, symbol="diamond"),
                    hovertemplate="%{x|%m.%Y}: $%{y:.2f}<extra>dönem sonu</extra>"))
                if not D["est"].empty:
                    last_d = D["fair"].index[-1]
                    ex = pd.concat([pd.Series({last_d: D["fair_now"]}), D["est"]])
                    fig.add_trace(go.Scatter(
                        x=ex.index, y=ex, mode="lines+markers",
                        name="Tahmin", line=dict(color="#2fbe86", width=2,
                                                 dash="dash"),
                        marker=dict(size=8, symbol="diamond")))
                    fig.add_vrect(x0=last_d, x1=D["est"].index[-1],
                                  fillcolor="#c9b800", opacity=0.10, line_width=0,
                                  annotation_text="Tahminler",
                                  annotation_position="bottom right")
                fig.update_layout(height=460, legend=dict(orientation="h", y=1.08),
                                  yaxis_title="$", **CHART_LAYOUT)
                st.plotly_chart(fig, width="stretch")
                if not D["est"].empty:
                    e_last = D["est"]
                    st.caption(
                        "Tahmin noktaları: " + " · ".join(
                            f"{d:%m.%Y} → \\${v:.2f}" for d, v in e_last.items())
                        + f" — kaynak: {D['est_src']}.")
                st.caption(
                    "Yeşil çizgi = hisse başı TTM " + D["metric"]["ad"].lower()
                    + f" × hissenin {('tüm geçmişteki' if not win else f'son {win} yıldaki')} "
                    f"medyan {D['metric']['kisa']} çarpanı. Korelasyon yüksekse "
                    "(%70+) fiyat bu metriği takip ediyor demektir; o zaman "
                    "çizgiden uzaklaşmalar anlamlıdır.")

            # ------------------------------------------------ özet tablo
            section("Çarpan özeti")
            SUM = fch.multiples_summary(RAW, MED if not MED.empty else None)
            if SUM.empty:
                st.caption("Çarpan hesaplanamadı.")
            else:
                st.dataframe(SUM, width="stretch", hide_index=True,
                             column_config={
                                 c: st.column_config.NumberColumn(format="%.2f")
                                 for c in ("Şu an", "Kendi medyanı", "En düşük",
                                           "En yüksek", "Akran medyanı")} | {
                                 "Medyana göre %": st.column_config.NumberColumn(
                                     format="%+.0f%%")})
            if RAW.get("errors"):
                st.caption("Eksik veri: " + ", ".join(RAW["errors"]))
            with st.expander("Hesap yöntemi ve sınırlar"):
                st.markdown(
                    "- **TTM**: son 4 çeyreğin toplamı. Çeyreklik veri yetmeyen "
                    "eski dönemlerde mali yıl sonu değeri kullanılır.\n"
                    "- **Geçmiş uzunluğu**: Yahoo yıllık tabloları ~4–5 yıl, "
                    "çeyrekliği ~5–6 çeyrek verir; grafikler bu aralıktan "
                    "başlar. Düzeltilmiş HBK ~10 yıl gider.\n"
                    "- **Akran medyanı**: sektörün tamamı değil, seçtiğiniz "
                    "akranların aylık F/K medyanıdır (en az 3 akran gerekir).\n"
                    "- **Tahminler**: satış ve HBK için analist tahminleri "
                    "kullanılır. Yahoo nakit akışı tahmini vermediği için OCF/FCF "
                    "tahmini HBK büyümesiyle ölçeklenir — kaba bir yaklaşımdır.\n"
                    "- Zarardaki dönemlerde F/K ve çarpanlar tanımsızdır, "
                    "grafikte boş kalır.")



# ==========================================================================
# YENİ SAYFA YAPISI — Para Akışı → Endeksler & Risk → Tema Avcısı →
# Hisse Avcısı → Hype & Biotech · eski sekmeler "Araçlar" altında.
# st.tabs yerine sayfa seçici: bir sayfadaki düğme bir sonraki sayfayı açabilir
# ve yalnızca seçili sayfa hesaplanır (uygulama hızlanır).
# ==========================================================================
PAGES = ["🌊 Para Akışı", "🧭 Endeksler & Risk", "🎯 Tema Avcısı",
         "🦅 Hisse Avcısı", "🔥 Hype & Biotech", "🔔 Takip & Portföy",
         "🤖 Robot (simülasyon)", "🗂️ Araçlar"]
for _k, _v in {"nav": PAGES[0], "nonce_chart": "0", "nonce_flow": "0",
               "mk_sel": "S&P 500", "ta_index": "Nasdaq", "ha_tema": None,
               "ha_sym": None, "rb_sym": "", "rb_plan": "kademeli"}.items():
    st.session_state.setdefault(_k, _v)


def _go(page: str, **ctx) -> None:
    """Düğme geri çağrısı: sayfayı değiştirir ve bağlamı taşır."""
    st.session_state["nav"] = page
    for w_ in ("ta_index_radio", "ha_tema_sel", "ha_sym_sel", "ha_free", "rb_sym_in",
               "rb_plan_in"):
        st.session_state.pop(w_, None)          # eski widget seçimi yeni bağlamı ezmesin
    for k_, v_ in ctx.items():
        st.session_state[k_] = v_


st.radio("Sayfa", PAGES, key="nav", horizontal=True, label_visibility="collapsed")
st.sidebar.radio("Etkileşimli grafiklerin indirme formatı (📷 düğmesi)",
                 ["png", "jpeg", "svg"], key="img_fmt", horizontal=True,
                 help="Pine çizimli grafikler ve akış görseli altlarındaki "
                      "PNG / JPEG / PDF düğmeleriyle iner.")
PAGE = st.session_state["nav"]


def _fname(x: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in x).strip("_") or "grafik"


@st.cache_data(ttl=3600, show_spinner=False)
def load_long(syms: tuple[str, ...], nonce: str) -> dict:
    prices, _ = dta.fetch(list(syms), "1d_long")
    return prices


@st.cache_data(ttl=3600, show_spinner=False)
def chart_pack(sym: str, label: str, nonce: str) -> dict:
    """Günlük/haftalık/aylık grafikler (PNG/JPEG/PDF) + teknik okumalar."""
    need = tuple(sorted(set(sym.split("/")) | {"SPY", "QQQ"}))
    prices = load_long(need, nonce)
    vs = mkt.views(sym, prices)
    out: dict = {}
    pngs, texts = [], []
    for tf, v in vs.items():
        f = tvc.draw(v, label)
        out[tf] = {"png": tvc.to_bytes(f, "png"), "jpeg": tvc.to_bytes(f, "jpeg"),
                   "pdf": tvc.to_bytes(f, "pdf"), "text": rdg.read(v),
                   "score": v.get("score", np.nan)}
        pngs.append(out[tf]["png"])
        texts.append(f"{tf}\n\n" + out[tf]["text"])
        tvc.close(f)
    out["_pdf"] = tvc.report_pdf(f"{label} — teknik görünüm "
                                 f"({dt.date.today():%d.%m.%Y})", pngs, texts)
    return out


def img_downloads(data: dict, base: str, key: str, cols=None) -> None:
    """PNG / JPEG / PDF indirme düğmeleri."""
    c = cols or st.columns(3)
    for col, (fmt, mime) in zip(c, (("png", "image/png"), ("jpeg", "image/jpeg"),
                                    ("pdf", "application/pdf"))):
        col.download_button(fmt.upper(), data[fmt], f"{base}.{'jpg' if fmt == 'jpeg' else fmt}",
                            mime, key=f"{key}_{fmt}", width="stretch")


def show_triptych(sym: str, label: str, key: str) -> None:
    """Pine çizimli G/H/A grafikler yan yana + altlarında teknik okuma."""
    with st.spinner(f"{label}: günlük, haftalık ve aylık grafikler çiziliyor…"):
        try:
            R = chart_pack(sym, label, st.session_state.nonce_chart)
        except Exception as exc:                  # noqa: BLE001
            st.error(f"{label} grafikleri çizilemedi: {exc}")
            return
    cols = st.columns(3)
    for col, tf in zip(cols, tfv.TF_RULE):
        with col:
            st.image(R[tf]["png"], width="stretch")
            img_downloads(R[tf], f"{_fname(label)}_{tf}", f"{key}_{tf}")
    st.download_button("📄 Üç grafik + teknik okumalar (tek PDF)", R["_pdf"],
                       f"{_fname(label)}_teknik.pdf", "application/pdf",
                       key=f"{key}_allpdf")
    tcols = st.columns(3)
    for col, tf in zip(tcols, tfv.TF_RULE):
        with col, st.container(border=True):
            st.markdown(f"##### {tf}")
            st.markdown(R[tf]["text"])


@st.cache_data(ttl=TTL_SLOW, show_spinner=False)
def load_flows(nonce: str):
    if _use_snap(nonce):
        s = snap.load("flows")
        if s is not None:
            return s
    return cmp.flows()


@st.cache_data(ttl=TTL_SLOW, show_spinner=False)
def load_markets(nonce: str):
    if _use_snap(nonce):
        s = snap.load("markets")
        if s is not None:
            return s
    return cmp.markets()


@st.cache_data(ttl=TTL_SLOW, show_spinner=False)
def load_gold(nonce: str):
    if _use_snap(nonce):
        s = snap.load("gold")
        if s is not None:
            return s
    return cmp.gold()


@st.cache_data(ttl=TTL_SLOW, show_spinner=False)
def load_hype(nonce: str) -> dict:
    """Akşam hesabından (6 saatten taze ise); yoksa canlı: sosyal + ilk 15 hissenin
    temel verileri + biotech listesi (özel ilgi dahil)."""
    bw = st.session_state.get("watchlist", {}).get("biotech_watch", [])
    if _use_snap(nonce):
        s = snap.load("hype", max_age_h=30)
        if s is not None:
            out = dict(s)
            miss = sorted(set(bw) - set(out["fund"]["Sembol"])) if not out["fund"].empty else bw
            if miss:
                out["fund"] = pd.concat([out["fund"], hyp.fundamentals(miss, dta.yf_info)],
                                        ignore_index=True)
            out["_kaynak"] = f"📦 Akşam hesabından · {snap.summary()}"
            return out
    soc, errs = hyp.social(pages=2)
    top = soc.head(15)["Sembol"].tolist() if not soc.empty else []
    fund = hyp.fundamentals(sorted(set(top) | set(bw)), dta.yf_info)
    return {"social": soc, "errs": errs, "fund": fund, "top": top, "bio": bw,
            "_kaynak": "🔴 Canlı çekildi (biotech evreninin nakit verileri akşam hesabında "
                       "tamamlanır)."}


@st.cache_data(ttl=TTL_NEWS, show_spinner=False)
def load_query_news(query: str, nonce: str):
    return nws.fetch_query(query)


def news_list(items: list, n: int = 8) -> None:
    if not items:
        st.caption("Haber bulunamadı.")
        return
    for it in items[:n]:
        st.markdown(f"- [{it['Başlık']}]({it['Link']}) · {it['Tarih'][:16]}  \n"
                    f"  <span style='color:#7a8394;font-size:0.85em'>{it['Etki']}</span>",
                    unsafe_allow_html=True)


@st.cache_data(ttl=TTL_SLOW, show_spinner=False)
def flow_pack(nonce: str) -> dict:
    FT, _ = load_flows(nonce)
    f = flw.diagram(FT)
    out = {fmt: tvc.to_bytes(f, fmt) for fmt in ("png", "jpeg", "pdf")}
    tvc.close(f)
    return out


def rrg_fig(T: pd.DataFrame, TAILS: dict, names: list[str]):
    fig = go.Figure()
    for i, tema in enumerate(names):
        tl = TAILS.get(tema)
        if tl is None or tl.empty:
            continue
        colr = SERIES[i % len(SERIES)]
        fig.add_trace(go.Scatter(x=tl["RS-Oran"], y=tl["RS-Mom"], mode="lines+markers",
                                 line=dict(color=colr, width=1.5), marker=dict(size=4),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[tl["RS-Oran"].iloc[-1]], y=[tl["RS-Mom"].iloc[-1]],
                                 mode="markers+text", marker=dict(size=11, color=colr),
                                 text=[tema], textposition="top center",
                                 textfont=dict(size=11, color=colr), name=tema,
                                 hovertemplate=f"{tema}<br>RS-Oran %{{x:.1f}}<br>"
                                               f"RS-Mom %{{y:.1f}}<extra></extra>"))
    fig.add_vline(x=100, line_color="#3a3a46")
    fig.add_hline(y=100, line_color="#3a3a46")
    for x_, y_, t_ in ((1, 1, "🚀 Lider"), (0, 1, "🌱 İyileşiyor"),
                       (0, 0, "🩸 Geride"), (1, 0, "🌤️ Yoruluyor")):
        fig.add_annotation(xref="paper", yref="paper", x=0.02 + 0.96 * x_,
                           y=0.02 + 0.96 * y_, text=t_, showarrow=False,
                           xanchor="right" if x_ else "left",
                           font=dict(size=12, color="#6e6e7a"))
    fig.update_layout(height=520, xaxis_title="RS-Oran (güç)",
                      yaxis_title="RS-Momentum (güçteki değişim)", showlegend=False,
                      **CHART_LAYOUT)
    return fig


def market_verdicts(MK: dict, ROT: pd.DataFrame, V: dict, vix_skor: float) -> dict:
    out = {}
    for name, r in MK.items():
        m = mkt.MARKETS.get(name, {})
        dd_ = dcs.asset_decision(r.get("scores", {}), ROT, name, V["etiket"],
                                 vix_skor if m.get("vix") else 0.0)
        out[name] = dd_
    return out


THEME_GROUPS = {**pln.INDEX_THEMES,
                "Russell 2000": ["Küçük Ölçekli (Small Cap)", "Bankalar", "Biyoteknoloji",
                                 "Konut İnşaatı", "Taşımacılık", "Bitcoin Madenciliği",
                                 "Kuantum", "Uzay", "Temiz Enerji", "Metal & Madencilik",
                                 "Petrol & Gaz", "Perakende"],
                "Dow Jones": ["Bankalar", "Sağlık (Geniş)", "Altyapı", "Savunma & Havacılık",
                              "Perakende", "Tüketici Defansif", "Petrol & Gaz", "Temettü",
                              "Değer", "Teknoloji (Geniş)"],
                "Altın": ["Altın Madenciliği", "Gümüş", "Metal & Madencilik"],
                "Gümüş": ["Gümüş", "Altın Madenciliği", "Metal & Madencilik"],
                "Bakır/Altın": ["Bakır", "Metal & Madencilik", "Lityum & Batarya",
                                "Nadir Toprak", "Altyapı"],
                "Petrol (Brent)": ["Petrol & Gaz", "Nükleer & Uranyum", "Temiz Enerji"]}


@st.cache_data(ttl=TTL_SLOW, show_spinner=False)
def load_robot_data(sym: str, nonce: str):
    h, _ = dta.fetch([sym], "1h_long")
    d, _ = dta.fetch([sym], "1d_long")
    return h.get(sym), d.get(sym)


def _rparams(cfg: dict) -> "rbt.Params":
    keys = rbt.Params.__dataclass_fields__
    kw = {k: v for k, v in cfg.items() if k in keys}
    for k_ in ("zone", "tp_r"):
        if k_ in kw:
            kw[k_] = tuple(kw[k_])
    return rbt.Params(**kw)


@st.cache_data(ttl=TTL_SLOW, show_spinner=False)
def robot_run(sym: str, cfg_json: str, start: str, nonce: str) -> dict:
    h1, d = load_robot_data(sym, nonce)
    if h1 is None or d is None or len(h1) < 50:
        return {"ok": False, "neden": f"{sym} için 1 saatlik veri alınamadı"}
    R = rbt.run(h1, d, _rparams(json.loads(cfg_json)), start=start)
    f = rbc.draw(R, sym)
    R["png"] = tvc.to_bytes(f, "png")
    R["jpeg"] = tvc.to_bytes(f, "jpeg")
    R["pdf"] = tvc.to_bytes(f, "pdf")
    tvc.close(f)
    R["ok"] = True
    return R


@st.cache_data(ttl=TTL_SLOW, show_spinner=False)
def robot_optimize(sym: str, start: str, budget: float, nonce: str) -> pd.DataFrame:
    h1, d = load_robot_data(sym, nonce)
    if h1 is None or d is None or len(h1) < 50:
        return pd.DataFrame()
    return rbt.optimize(h1, d, start, budget or 10_000.0)


def show_robot(R: dict, key: str) -> None:
    if not R.get("ok"):
        st.error(R.get("neden", "çalıştırılamadı"))
        return
    s_, stt = R["stats"], R["state"]
    k = st.columns(6)
    k[0].markdown(kpi("Durum", "POZİSYONDA" if stt["pozisyonda"] else "NAKİTTE",
                      f"fiyat {stt['fiyat']:.2f}"), unsafe_allow_html=True)
    k[1].markdown(kpi("Açık K/Z", f"{stt['K/Z % (açık)']:+.1f}%" if stt["pozisyonda"] else "—",
                      f"maliyet {stt['maliyet']:.2f}" if stt["pozisyonda"] else "",
                      "pos" if stt["pozisyonda"] and stt["K/Z % (açık)"] >= 0 else
                      "neg" if stt["pozisyonda"] else ""), unsafe_allow_html=True)
    k[2].markdown(kpi("Hedefe", f"{stt['hedefe %']:+.1f}%" if stt["pozisyonda"]
                      and np.isfinite(stt["hedefe %"]) else "—",
                      f"stop {stt['stop']:.2f}" if stt["pozisyonda"] else ""), unsafe_allow_html=True)
    k[3].markdown(kpi("Getiri", f"{s_['Getiri %']:+.1f}%", f"al-tut {s_['Al-tut %']:+.1f}%",
                      "pos" if s_["Getiri %"] >= 0 else "neg"), unsafe_allow_html=True)
    k[4].markdown(kpi("İşlem", f"{s_['İşlem']}", f"kazanan %{s_['Kazanan %']:.0f}"
                      if np.isfinite(s_["Kazanan %"]) else ""), unsafe_allow_html=True)
    k[5].markdown(kpi("Maks. düşüş", f"{s_['Maks. düşüş %']:.1f}%",
                      f"ort. R {s_['Ort. R']:+.2f}" if np.isfinite(s_["Ort. R"]) else ""),
                  unsafe_allow_html=True)
    st.image(R["png"], width="stretch")
    img_downloads(R, f"robot_{key}", f"rbdl_{key}")
    if stt.get("radar"):
        st.warning("📡 Radar açık — fiyat hedefe giden yolun %85'ini geçti; dönüşte giyotin iner.")
    tabs_ = ["📜 Olay günlüğü", "💱 İşlemler", "🔎 Giriş teşhisi"]
    if R.get("council") is not None:
        tabs_.append("🧠 Konseyin öğrendiği ağırlıklar")
    TT_ = st.tabs(tabs_)
    t1, t2, t3 = TT_[:3]
    if len(TT_) > 3:
        with TT_[3]:
            WT = R["council"]["weights"]
            st.caption("Son eğitimde her sinyalin bu hissede ileri getiriyi ne kadar öngördüğü. "
                       "|t| < 1 olanlar 'gürültü' sayılıp susturulur. Ağırlıklar her ~5 günde bir, "
                       "yalnızca o ana kadarki veriyle yeniden öğrenilir.")
            if WT is None or WT.empty:
                st.caption("Eğitim için yeterli geçmiş yok (en az ~400 saatlik mum).")
            else:
                st.dataframe(WT, width="stretch", hide_index=True, column_config={
                    "Ağırlık": st.column_config.ProgressColumn(format="%+.2f", min_value=-1,
                                                               max_value=1),
                    "t-istatistiği": st.column_config.NumberColumn(format="%+.2f")})
    with t1:
        E = R["events"]
        if E.empty:
            st.caption("Henüz olay yok — robot giriş şartlarını bekliyor.")
        else:
            st.dataframe(E.iloc[::-1], width="stretch", hide_index=True,
                         column_config={"Mesaj": st.column_config.TextColumn(width="large"),
                                        "Fiyat": st.column_config.NumberColumn(format="%.2f")})
    with t2:
        T_ = R["trades"]
        if T_.empty:
            st.caption("Kapanmış işlem yok.")
        else:
            st.dataframe(T_, width="stretch", hide_index=True, column_config={
                "Giriş fiyatı": st.column_config.NumberColumn(format="%.2f"),
                "K/Z $": st.column_config.NumberColumn(format="%+,.0f"),
                "K/Z %": st.column_config.NumberColumn(format="%+.1f%%"),
                "R": st.column_config.NumberColumn(format="%+.2f"),
                "Satışlar": st.column_config.TextColumn(width="large")})
    with t3:
        dg = R.get("diag", {})
        if dg and dg.get("bar"):
            st.caption("Nakitte geçen 1 saatlik mumların kaçında her şart sağlandı — robot neden "
                       "girmediyse en düşük sayı darboğazdır.")
            st.dataframe(pd.DataFrame([{"Şart": k_, "Mum sayısı": v_,
                                        "Oran %": v_ / dg["bar"] * 100}
                                       for k_, v_ in dg.items() if k_ != "bar"]),
                         hide_index=True, column_config={"Oran %": st.column_config.NumberColumn(
                             format="%.1f%%")})
        else:
            st.caption("Bu robot giriş aramıyor (sadece yönet) ya da hep pozisyondaydı.")


# ---------------------------------------------------------------- takip & alarm
def _wl() -> dict:
    return st.session_state.watchlist


def _save_wl(msg: str) -> None:
    if save_watchlist():
        st.toast(msg, icon="✅")
    tracking_digest.clear()


@st.cache_data(ttl=TTL_SLOW, show_spinner=False)
def tracking_digest(wl_json: str, n_funnel: str, n_scan: str, n_earn: str) -> dict:
    """Takip listeleri + portföy için durumlar ve uygulama içi alarmlar."""
    wl = json.loads(wl_json)
    sw = [x["tema"] for x in wl.get("sector_watch", [])]
    kw = wl.get("stock_watch", [])
    pf = wl.get("portfolio", [])
    out = {"sector": {}, "watch": [], "port": [], "alarms": [], "suggest": []}
    syms = sorted({x["sym"] for x in kw} | {x["sym"] for x in pf})
    kg_map = {}
    if syms:
        try:
            E = load_earnings(tuple(syms), n_earn)
            if not E.empty and "Hisse" in E.columns:
                kg_map = dict(zip(E["Hisse"], E.get("Kalan Gün")))
        except Exception:                      # noqa: BLE001
            pass
    if sw:
        T, TAILS, _, _, _ = load_theme_rotation(n_funnel)
        etfs = sorted({e for t in sw for e in uni.THEME_TRACKER.get(t, [])})
        S = scan(tuple(etfs), "1d", n_scan) if etfs else pd.DataFrame()
        rows = S.to_dict("records") if not S.empty else []
        for t in sw:
            er = [r for r in rows if r.get("Sembol") in uni.THEME_TRACKER.get(t, [])]
            st_ = trk.sector_status(t, T, TAILS, er)
            out["sector"][t] = st_
            if st_["alarm"]:
                out["alarms"].append({"Tür": "📌 Sektör", "Konu": t, "Durum": st_["durum"],
                                      "Mesaj": st_["tavsiye"]})
        out["suggest"] = trk.rotation_suggestion(sw, T, out["sector"])
    if kw:
        P = load_plans(tuple(sorted({x["sym"] for x in kw})), n_scan)
        pm = {r["Sembol"]: r for r in P.to_dict("records")} if not P.empty else {}
        for it in kw:
            pl = pm.get(it["sym"])
            if pl is not None and not (pl.get("ok") is True or pl.get("ok") == True):  # noqa: E712
                pl = None
            durum, al = trk.watch_alarms(it, pl, kg_map.get(it["sym"]))
            out["watch"].append({**it, "durum": durum, "fiyat": (pl or {}).get("Fiyat"),
                                 "alarmlar": al})
            for a in al:
                out["alarms"].append({"Tür": "👁️ Takip", "Konu": it["sym"], "Durum": durum,
                                      "Mesaj": a})
    if pf:
        prices, _ = dta.fetch(sorted({x["sym"] for x in pf}), "1d")
        for it in pf:
            lv = trk.position_levels(prices.get(it["sym"]), float(it["maliyet"]),
                                     it.get("tarih"), it.get("ilk_stop"))
            durum, al = trk.position_alarms(it, lv, kg_map.get(it["sym"]))
            out["port"].append({**it, "lv": lv, "durum": durum, "alarmlar": al})
            for a in al:
                out["alarms"].append({"Tür": "💼 Portföy", "Konu": it["sym"], "Durum": durum,
                                      "Mesaj": a})
    for rb_ in wl.get("robots", []):
        try:
            R = robot_run(rb_["sym"], json.dumps(rb_["cfg"], sort_keys=True), rb_["start"], n_scan)
        except Exception:                      # noqa: BLE001
            continue
        if not R.get("ok") or R["events"].empty:
            continue
        E = R["events"]
        last_t = E["Zaman"].max()
        recent = E[E["Zaman"] >= last_t.normalize() - pd.Timedelta(days=1)]
        for _, e in recent.tail(3).iterrows():
            if e["Olay"] in ("✅ Giriş", "⛔ Stop", "⚔️ Giyotin", "📡 Radar", "🎁 Bedava lot") \
                    or str(e["Olay"]).startswith("🎯 TP"):
                out["alarms"].append({"Tür": "🤖 Robot", "Konu": rb_["sym"], "Durum": e["Olay"],
                                      "Mesaj": f"{e['Zaman']:%d.%m %H:%M} · {e['Mesaj']}"})
    return out


def _digest() -> dict:
    wl = _wl()
    if not (wl.get("sector_watch") or wl.get("stock_watch") or wl.get("portfolio")
            or wl.get("robots")):
        return {"sector": {}, "watch": [], "port": [], "alarms": [], "suggest": []}
    return tracking_digest(json.dumps({k: wl.get(k, []) for k in
                                       ("sector_watch", "stock_watch", "portfolio",
                                        "robots")},
                                      sort_keys=True, default=str),
                           st.session_state.nonce_funnel, st.session_state.nonce_scan,
                           st.session_state.nonce_earn)


if PAGE != PAGES[-1] and PAGE != "🔔 Takip & Portföy":
    try:
        _DG = _digest()
    except Exception as _exc:                  # noqa: BLE001
        _DG = {"alarms": []}
        _log.info("alarm özeti: %s", _exc)
    if _DG["alarms"]:
        _b1, _b2 = st.columns([5, 1])
        _b1.warning(f"🔔 **{len(_DG['alarms'])} alarm:** "
                    + " · ".join(f"{a['Konu']} {a['Durum']}" for a in _DG["alarms"][:5])
                    + (" …" if len(_DG["alarms"]) > 5 else ""))
        _b2.button("Alarmları aç", on_click=_go, args=("🔔 Takip & Portföy",),
                   key="alarm_go", width="stretch")


# ---------------------------------------------------------------- ortak veriler
if PAGE != PAGES[-1]:
    ROT_, SER_, ROT_FAILED_, FILES_ = load_rotation(st.session_state.nonce_funnel)
    PIL_ = fnl.risk_pillars(M.scores, ROT_)
    V_ = fnl.risk_verdict(PIL_, M.regime)

# ======================================================================= 1
if PAGE == PAGES[0]:
    h1, h2 = st.columns([4, 1])
    h1.markdown("### 🌊 Akıllı para nereye akıyor?")
    if h2.button("🔄 Yenile", key="fl_ref", width="stretch"):
        bump("nonce_flow")
        st.rerun()
    FT, FT_FAILED = load_flows(st.session_state.nonce_flow)
    FP = flow_pack(st.session_state.nonce_flow)
    g1, g2 = st.columns([3, 2])
    with g1:
        st.image(FP["png"], width="stretch")
        img_downloads(FP, "para_akisi", "fl_dl")
    with g2:
        st.markdown("#### Neden böyle akıyor?")
        for line in flw.reasons(FT):
            st.markdown(f"- {line}")
        if not ROT_.empty:
            with st.expander("Rotasyon oranlarından ek okumalar"):
                for line in fnl.money_flow_summary(ROT_):
                    st.markdown(f"- {line}")
    st.markdown("#### Rakamlarla akış")
    if FT is None or FT.empty:
        st.warning("Akış verisi çekilemedi.")
    else:
        st.dataframe(FT[["Varlık", "Sembol", "Akış", "Akış Skoru", "5G %", "20G %",
                         "60G %", "Hacim trendi"]], width="stretch", hide_index=True,
                     column_config={
                         "Akış Skoru": st.column_config.ProgressColumn(
                             format="%+.2f", min_value=-2.5, max_value=2.5,
                             help="20 günlük göreli performans (ağırlık 0.6) + 5 günlük "
                                  "ivme (0.4), dolar hacmi artışıyla güçlendirilir"),
                         "5G %": st.column_config.NumberColumn(format="%+.1f%%"),
                         "20G %": st.column_config.NumberColumn(format="%+.1f%%"),
                         "60G %": st.column_config.NumberColumn(format="%+.1f%%"),
                         "Hacim trendi": st.column_config.NumberColumn(
                             format="%.2f×", help="Son 10 günün dolar hacmi ÷ son 60 gün")})
    st.markdown("#### İlgili haberler")
    n1, n2 = st.columns(2)
    with n1:
        st.markdown("**Fon akışları**")
        it, _ = load_query_news(nws.FEEDS["Fon akışları"], st.session_state.nonce_news)
        news_list(it, 6)
    with n2:
        st.markdown("**Fed, faiz ve dolar**")
        it, _ = load_query_news(nws.FEEDS["Makro & Fed"], st.session_state.nonce_news)
        news_list(it, 6)
    st.button("➡️ Endeksler & Risk sayfasına geç", on_click=_go, args=(PAGES[1],),
              key="fl_next")

# ======================================================================= 2
elif PAGE == PAGES[1]:
    h1, h2 = st.columns([4, 1])
    h1.markdown("### 🧭 Endeksler, risk iştahı ve değerli metaller")
    if h2.button("🔄 Yenile", key="mk_ref", width="stretch"):
        bump("nonce_funnel")
        st.rerun()
    v1, v2 = st.columns([1, 2])
    v1.markdown(kpi("Piyasa durumu", V_["etiket"],
                    f"skor {V_['skor']:.0f}/100 · {V_['boyut']}"
                    if np.isfinite(V_["skor"]) else V_["boyut"], V_["renk"]),
                unsafe_allow_html=True)
    v2.markdown(f"**Ne demek?** {V_['ozet']}")
    v2.caption(f"{V_['n_ok']} sütun olumlu, {V_['n_bad']} sütun olumsuz · rejim: {M.regime}")
    pc = st.columns(6)
    for col, p in zip(pc, PIL_):
        tone = ("pos" if p["durum"] == "✅" else "neg" if p["durum"] == "❌" else "")
        col.markdown(kpi(p["ad"], f"{p['durum']} {p['skor']:.0f}"
                         if np.isfinite(p["skor"]) else "➖", p["ne"], tone),
                     unsafe_allow_html=True)
    with st.expander("Risk iştahı neden böyle? (sütunların gerekçeleri)"):
        for p in PIL_:
            st.markdown(f"- **{p['ad']}** — {p['ne']} *{p['neden']}*")

    with st.spinner("Endeks kararları yükleniyor…"):
        DEC_ = load_decision(st.session_state.nonce_funnel, ROT_)
        MK, MK_FAILED = load_markets(st.session_state.nonce_funnel)
    vix_sk = DEC_["vix"]["skor"]
    MV = market_verdicts(MK, ROT_, V_, vix_sk)
    if DEC_.get("notes"):
        with st.expander("📌 Piyasa notları (hacim, genişlik, VIX kuralı, QQQ/SPY)",
                         expanded=True):
            for ic, tx in DEC_["notes"]:
                st.markdown(f"{ic} {tx}")

    st.markdown("#### Varlıklar — karar ve zaman dilimi skorları")
    st.caption("🔍 grafiği açar (günlük/haftalık/aylık + teknik okuma). 🎯 yalnızca "
               "girişe uygun varlıklarda çıkar ve o varlığın temalarına götürür.")
    names = list(mkt.MARKETS)
    for i in range(0, len(names), 4):
        cc = st.columns(4)
        for col, name in zip(cc, names[i:i + 4]):
            r, dd_ = MK.get(name, {}), MV.get(name, {})
            sc = r.get("scores", {})
            tfs = " · ".join(f"{tf[0]} {sc[tf]:+.0f}" for tf in tfv.TF_RULE
                             if np.isfinite(sc.get(tf, np.nan)))
            with col, st.container(border=True):
                st.markdown(kpi(name, dd_.get("etiket", "—"),
                                (f"5G {r['chg5']:+.1f}% · " if np.isfinite(r.get("chg5", np.nan))
                                 else "") + tfs, dd_.get("tone", "")),
                            unsafe_allow_html=True)
                b1, b2 = st.columns(2)
                b1.button("🔍 Grafik", key=f"mk_g_{name}", width="stretch",
                          on_click=_go, args=(PAGES[1],), kwargs={"mk_sel": name})
                if (name in THEME_GROUPS and name not in mkt.NO_THEMES
                        and dd_.get("etiket", "").startswith(("✅", "🟡"))):
                    b2.button("🎯 Temalar", key=f"mk_t_{name}", width="stretch",
                              on_click=_go, args=(PAGES[2],), kwargs={"ta_index": name})

    sel = st.session_state.mk_sel if st.session_state.mk_sel in mkt.MARKETS else "S&P 500"
    st.markdown(f"#### 🔍 {sel} — günlük · haftalık · aylık")
    if sel in mkt.ROW_NOTE:
        st.caption(mkt.ROW_NOTE[sel])
    dd_sel = MV.get(sel, {})
    if dd_sel.get("baglam_txt"):
        st.caption("Bağlam: " + " · ".join(dd_sel["baglam_txt"]))
    show_triptych(mkt.MARKETS[sel]["sym"], sel, f"mk_{_fname(sel)}")
    if sel in THEME_GROUPS and sel not in mkt.NO_THEMES:
        st.button(f"🎯 {sel} temalarına git (Tema Avcısı)", on_click=_go,
                  args=(PAGES[2],), kwargs={"ta_index": sel}, key="mk_go_theme",
                  type="primary")
    with st.expander(f"📰 {sel} ile ilgili haberler"):
        q = {"Altın": "gold price", "Gümüş": "silver price", "Kripto (BTC)": "bitcoin",
             "Petrol (Brent)": "oil price Brent", "Dolar (DXY)": "dollar index",
             "ABD 10Y faiz": "Treasury yields", "Bakır/Altın": "copper price",
             "Altın/Gümüş": "gold silver ratio", "Russell 2000": "Russell 2000 small caps",
             "Dow Jones": "Dow Jones"}.get(sel, f"{sel} stock market")
        it, _ = load_query_news(q, st.session_state.nonce_news)
        news_list(it, 8)

    # ---- altın & gümüş
    st.markdown("---")
    st.markdown("### 🥇 Altın & gümüş — fiyatı hareket ettiren zincir")
    st.caption("ABD verisi → Fed beklentisi → tahvil faizi ve dolar → altın. Önem sırası: "
               "reel faiz, dolar, Fed beklentisi. Göstergeler çelişirse en doğru "
               "pozisyon çoğu zaman beklemektir.")
    (GT, GS), _ = load_gold(st.session_state.nonce_funnel)
    k1, k2 = st.columns([1, 2])
    k1.markdown(kpi("Altın için zincir", GS["etiket"],
                    f"ilk 3 halka: {GS['plus']} destek · {GS['minus']} baskı",
                    "pos" if GS["etiket"].startswith("🟢") else
                    "neg" if GS["etiket"].startswith("🔴") else ""),
                unsafe_allow_html=True)
    k2.markdown(GS["ozet"])
    st.dataframe(GT, width="stretch", hide_index=True,
                 column_config={"20G değişim": st.column_config.NumberColumn(format="%+.2f"),
                                "Kural": st.column_config.TextColumn(width="large")})
    gc1, gc2 = st.columns(2)
    gc1.button("🔍 Altın grafiği", on_click=_go, args=(PAGES[1],),
               kwargs={"mk_sel": "Altın"}, key="gd_g", width="stretch")
    gc2.button("🔍 Gümüş grafiği", on_click=_go, args=(PAGES[1],),
               kwargs={"mk_sel": "Gümüş"}, key="sv_g", width="stretch")
    with st.expander("🗓️ Altın takvimi ve rutini"):
        for a_, b_ in gld.CALENDAR:
            st.markdown(f"- **{a_}:** {b_}")
        st.markdown("- **Veri ile beklenti farkı:** beklentiden güçlü veri → artırım "
                    "beklentisi ↑ → faiz ve dolar ↑ → altın ↓. Beklentiden zayıf veri → "
                    "tersi.\n- **Jeopolitik primler** genelde kısa sürer; kalıcı yükseliş "
                    "faiz ve para politikasından gelir.\n- **Panik anında** altın önce "
                    "nakit ihtiyacıyla düşebilir.")
    with st.expander("📰 Altın, faiz ve dolar haberleri"):
        it, _ = load_query_news(nws.FEEDS["Altın & Faiz"], st.session_state.nonce_news)
        news_list(it, 8)

# ======================================================================= 3
elif PAGE == PAGES[2]:
    st.markdown("### 🎯 Tema Avcısı")
    _sw = [x["tema"] for x in _wl().get("sector_watch", [])]
    if _sw:
        with st.expander(f"📌 Takip ettiğim sektörler ({len(_sw)})", expanded=True):
            DG = _digest()
            for t_ in _sw:
                ss = DG["sector"].get(t_, {})
                st.markdown(f"**{t_}** — {ss.get('durum', '—')}: {ss.get('tavsiye', '')}  \n"
                            f"<span style='color:#7a8394;font-size:0.85em'>"
                            f"{' · '.join(ss.get('neden', []))}</span>",
                            unsafe_allow_html=True)
            for sg in DG.get("suggest", []):
                st.info("🔀 " + sg)
    idx_opts = [k for k in THEME_GROUPS if k in mkt.MARKETS]
    cur = st.session_state.ta_index if st.session_state.ta_index in idx_opts else "Nasdaq"
    t1, t2 = st.columns([3, 1])
    ta_index = t1.radio("Endeks / varlık", idx_opts, index=idx_opts.index(cur),
                        horizontal=True, key="ta_index_radio")
    st.session_state.ta_index = ta_index
    if t2.button("🔄 Yenile", key="ta_ref", width="stretch"):
        bump("nonce_funnel")
        st.rerun()
    MK, _ = load_markets(st.session_state.nonce_funnel)
    DEC_ = load_decision(st.session_state.nonce_funnel, ROT_)
    MV = market_verdicts(MK, ROT_, V_, DEC_["vix"]["skor"])
    dd_i = MV.get(ta_index, {})
    index_ok = dd_i.get("etiket", "").startswith(("✅", "🟡"))
    (st.success if index_ok else st.warning)(
        f"**{ta_index}:** {dd_i.get('etiket', '—')} — "
        + ("temalarda alım fırsatı aranabilir." if index_ok else
           "endeks alım bölgesinde değil; temalar yalnızca izleme için."))

    T3, TAILS3, E3, IDX3, _ = load_theme_rotation(st.session_state.nonce_funnel)
    themes3 = [t for t in THEME_GROUPS.get(ta_index, []) if t in uni.THEME_TRACKER]
    TT = T3[T3["Tema"].isin(themes3)].copy() if not T3.empty else pd.DataFrame()
    if TT.empty:
        st.info("Bu endeks için tema verisi yok.")
    else:
        TT["Para"] = TT["_q"].map({"lider": "✅ giriyor (lider)",
                                   "iyilesen": "🌱 girmeye başladı",
                                   "zayif": "⚠️ çıkmaya başladı",
                                   "geride": "⛔ çıkıyor / zayıf"}).fillna("—")
        TT["_o"] = TT["_q"].map({"iyilesen": 0, "lider": 1, "zayif": 2, "geride": 3}).fillna(4)
        TT = TT.sort_values(["_o", "Erken Skor"], ascending=[True, False])
        good3 = TT[TT["_q"].isin(["lider", "iyilesen"])]["Tema"].tolist()
        r1, r2 = st.columns([3, 2])
        with r1:
            st.plotly_chart(rrg_fig(T3, TAILS3, list(TT["Tema"])), width="stretch")
            st.caption("Çizgiler son 6 haftanın izi, büyük nokta bugün. Geride → "
                       "İyileşiyor → Lider geçişi akıllı paranın erken izidir.")
        with r2:
            st.dataframe(TT[["Tema", "Para", "Erken Skor", "1H %", "1A %"]],
                         width="stretch", hide_index=True, height=480,
                         column_config={
                             "Erken Skor": st.column_config.ProgressColumn(
                                 format="%d", min_value=0, max_value=100),
                             "1H %": st.column_config.NumberColumn(format="%+.1f%%"),
                             "1A %": st.column_config.NumberColumn(format="%+.1f%%")})
        opts3 = good3 + [t for t in TT["Tema"] if t not in good3]
        st.session_state.setdefault("ta_tema", opts3[0])
        if st.session_state.ta_tema not in opts3:
            st.session_state.ta_tema = opts3[0]
        tema = st.selectbox("Tema seçin", opts3, key="ta_tema",
                            format_func=lambda t: ("✅ " if t in good3 else "⛔ ") + t)
        etfs = uni.THEME_TRACKER.get(tema, [])
        row = TT[TT["Tema"] == tema].iloc[0]
        st.markdown(f"**{tema}** · {row['Para']} · erken skor {row['Erken Skor']} · "
                    f"1H {row['1H %']:+.1f}% · 1A {row['1A %']:+.1f}% · ETF'ler: "
                    + ", ".join(f"`{e}`" for e in etfs))
        _tracked = tema in [x["tema"] for x in _wl().get("sector_watch", [])]
        if st.button(("📌 Sektör takipten çıkar" if _tracked else "📌 Sektör takibe ekle"),
                     key=f"ta_track_{tema}"):
            if _tracked:
                _wl()["sector_watch"] = [x for x in _wl()["sector_watch"] if x["tema"] != tema]
                _save_wl(f"{tema} takipten çıkarıldı")
            else:
                _wl()["sector_watch"].append({"tema": tema, "eklendi": trk.today(),
                                              "endeks": ta_index})
                _save_wl(f"{tema} sektör takibine eklendi")
            st.rerun()
        if _tracked:
            ss = _digest()["sector"].get(tema, {})
            if ss:
                (st.success if ss["durum"].startswith(("🟢", "🌱")) else st.warning)(
                    f"📌 **Takipte · {ss['durum']}** — {ss['tavsiye']}")
        b1, b2 = st.columns([1, 1])
        etf = b1.selectbox("Grafiği çizilecek ETF", etfs, key=f"ta_etf_{tema}") if etfs else None
        b2.button(f"🦅 {tema} hisselerine git (Hisse Avcısı)", on_click=_go,
                  args=(PAGES[3],), kwargs={"ha_tema": tema}, type="primary",
                  key="ta_go_stock", width="stretch")
        if tema not in good3:
            st.warning(f"{tema} temasından para çıkıyor — yeni alım için uygun değil.")
        if etf:
            show_triptych(etf, f"{tema} ({etf})", f"ta_{_fname(etf)}")
        with st.expander(f"📰 {tema} haberleri"):
            it, _ = load_query_news(f"{tema.split('(')[0]} stocks OR " + " OR ".join(etfs[:2]),
                                    st.session_state.nonce_news)
            news_list(it, 8)

# ======================================================================= 4
elif PAGE == PAGES[3]:
    st.markdown("### 🦅 Hisse Avcısı")
    all_themes = [t for t in uni.THEME_TRACKER if theme_holdings(t)]
    if st.session_state.ha_tema not in all_themes and st.session_state.ha_sym:
        st.session_state.ha_tema = next((t for t in all_themes
                                         if st.session_state.ha_sym in theme_holdings(t)), None)
    cur = st.session_state.ha_tema if st.session_state.ha_tema in all_themes else all_themes[0]
    q1, q2, q3 = st.columns([2, 1, 1])
    tema = q1.selectbox("Tema", all_themes, index=all_themes.index(cur), key="ha_tema_sel")
    st.session_state.ha_tema = tema
    liq = q2.number_input("Min. günlük hacim ($M)", 0.0, 500.0, 10.0, 1.0, key="ha_liq")
    if q3.button("🔄 Yenile", key="ha_ref", width="stretch"):
        bump("nonce_scan")
        bump("nonce_chart")
        st.rerun()
    T3, _, _, IDX3, _ = load_theme_rotation(st.session_state.nonce_funnel)
    trow = T3[T3["Tema"] == tema] if not T3.empty else pd.DataFrame()
    tema_ok = not trow.empty and trow.iloc[0]["_q"] in ("lider", "iyilesen")
    MK, _ = load_markets(st.session_state.nonce_funnel)
    DEC_ = load_decision(st.session_state.nonce_funnel, ROT_)
    MV = market_verdicts(MK, ROT_, V_, DEC_["vix"]["skor"])
    idx_name = st.session_state.ta_index
    index_ok = MV.get(idx_name, {}).get("etiket", "").startswith(("✅", "🟡"))
    st.caption(f"Endeks ({idx_name}): {MV.get(idx_name, {}).get('etiket', '—')} · "
               f"Tema: {trow.iloc[0]['Bölge'] if not trow.empty else '—'}")
    with st.spinner(f"{tema} hisseleri için plan hazırlanıyor…"):
        TBh, holdh = theme_stock_table(tema, liq, IDX3.get(tema))
        Ph = load_plans(tuple(holdh), st.session_state.nonce_scan) if holdh else pd.DataFrame()
    PT = trade_plan_table(TBh, Ph, index_ok and tema_ok)
    if PT.empty:
        st.info("Bu tema için hisse listesi ya da fiyat verisi yok.")
    else:
        n_al = int(PT["Karar"].str.startswith("✅").sum())
        n_bk = int(PT["Karar"].str.startswith("🟡").sum())
        st.markdown(" ".join([badge(f"✅ AL · {n_al}", "pos"),
                              badge(f"🟡 LİMİT EMİR / BEKLE · {n_bk}", ""),
                              badge(f"⛔ UZAK DUR · {len(PT) - n_al - n_bk}", "neg")]),
                    unsafe_allow_html=True)
        pcols = ["Karar", "Hisse", "Fiyat", "Giriş bölgesi", "Stop", "Kâr al 1",
                 "Kâr al 2", "R:R", "Sebep"]
        show_bad = st.toggle("⛔ Uzak durulacakları da göster", False, key="ha_bad")
        view = PT if show_bad else PT[PT["Karar"].str.startswith(("✅", "🟡"))]
        st.dataframe(view[pcols], width="stretch", hide_index=True,
                     column_config={"R:R": st.column_config.NumberColumn(format="%.1f"),
                                    "Sebep": st.column_config.TextColumn(width="large")})
        xl = pln.excel_report(f"AETHER APEX · {tema} hisse planı",
                              [("Tema", tema), ("Endeks", idx_name),
                               ("Hazırlanma", dt.datetime.now().strftime("%d.%m.%Y %H:%M"))],
                              pd.DataFrame(), {tema: PT})
        st.download_button("⬇️ Bu temanın planı (Excel)", xl,
                           f"plan_{_fname(tema)}_{dt.date.today():%Y%m%d}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           key="ha_xl")

        syms = PT["Hisse"].tolist()
        outside = (st.session_state.ha_sym if st.session_state.ha_sym
                   and st.session_state.ha_sym not in syms else "")
        s1_, s2_ = st.columns([2, 1])
        cur_s = st.session_state.ha_sym if st.session_state.ha_sym in syms else syms[0]
        sym = s1_.selectbox("🔍 İncelenecek hisse", syms, index=syms.index(cur_s),
                            key="ha_sym_sel",
                            format_func=lambda s: f"{s} — " + PT.loc[PT['Hisse'] == s, 'Karar'].iloc[0])
        free = s2_.text_input("…veya listede olmayan bir sembol", value=outside,
                              key="ha_free", placeholder="ör. NVDA").strip().upper()
        if free:
            sym = free
            S1_ = scan((sym,), "1d", st.session_state.nonce_scan)
            TBh = fnl.build_stock_table(S1_, None, None, None, None, liq,
                                        swing_score=scr.swing_score)
            Ph = load_plans((sym,), st.session_state.nonce_scan)
            PT1 = trade_plan_table(TBh, Ph, index_ok and tema_ok)
            if PT1.empty:
                st.error(f"{sym} için fiyat verisi alınamadı.")
                st.stop()
            prow = PT1.iloc[0]
        else:
            prow = PT[PT["Hisse"] == sym].iloc[0]
        st.session_state.ha_sym = sym
        trow2 = TBh[TBh["Sembol"] == sym]
        with st.container(border=True):
            st.markdown(f"#### {sym} · {prow['Karar']}")
            k = st.columns(5)
            k[0].markdown(kpi("Fiyat", prow["Fiyat"], ""), unsafe_allow_html=True)
            k[1].markdown(kpi("Giriş bölgesi", prow["Giriş bölgesi"], ""), unsafe_allow_html=True)
            k[2].markdown(kpi("Stop", prow["Stop"], "", "neg"), unsafe_allow_html=True)
            k[3].markdown(kpi("Kâr al 1 / 2", f"{prow['Kâr al 1']} / {prow['Kâr al 2']}", "",
                              "pos"), unsafe_allow_html=True)
            k[4].markdown(kpi("R:R", f"{prow['R:R']:.1f}" if np.isfinite(prow["R:R"]) else "—",
                              "kâr al 1'e göre"), unsafe_allow_html=True)
            st.markdown(f"**Sebep:** {prow['Sebep']}")
            _pl = Ph[Ph["Sembol"] == sym].to_dict("records") if not Ph.empty else []
            _pl = _pl[0] if _pl else {}
            _in_w = sym in [x["sym"] for x in _wl().get("stock_watch", [])]
            _in_p = sym in [x["sym"] for x in _wl().get("portfolio", [])]
            w1, w2, w3 = st.columns(3)
            w3.button("🤖 Robot kur (simülasyon)", key=f"ha_rb_{sym}", width="stretch",
                      on_click=_go, args=("🤖 Robot (simülasyon)",),
                      kwargs={"rb_sym": sym, "rb_plan": "yonet" if _in_p else "kademeli"})
            if w1.button("👁️ Takipten çıkar" if _in_w else "👁️ Hisse takibe al",
                         key=f"ha_w_{sym}", width="stretch"):
                if _in_w:
                    _wl()["stock_watch"] = [x for x in _wl()["stock_watch"] if x["sym"] != sym]
                    _save_wl(f"{sym} takipten çıkarıldı")
                elif _pl.get("ok") is True or _pl.get("ok") == True:  # noqa: E712
                    _wl()["stock_watch"].append({
                        "sym": sym, "tema": tema, "eklendi": trk.today(),
                        "giris_alt": float(_pl["Giriş Alt"]), "giris_ust": float(_pl["Giriş Üst"]),
                        "stop": float(_pl["Stop"]), "hedef1": float(_pl["Hedef 1"]),
                        "hedef2": float(_pl["Hedef 2"])})
                    _save_wl(f"{sym} takibe alındı — giriş bölgesine gelince alarm verir")
                else:
                    st.warning("Bu hisse için plan yok; takibe alınamadı.")
                st.rerun()
            with w2.popover("💼 Portföye ekle" if not _in_p else "💼 Portföyde ✓",
                            width="stretch"):
                def _fin(*xs):
                    for x_ in xs:
                        try:
                            x_ = float(x_)
                        except (TypeError, ValueError):
                            continue
                        if np.isfinite(x_) and x_ > 0:
                            return x_
                    return 0.0
                _px0 = _fin(_pl.get("Fiyat"),
                            trow2.iloc[0].get("Fiyat") if not trow2.empty else None)
                _st0 = _fin(_pl.get("Stop"))
                f_adet = st.number_input("Adet", 0.0, 1e7, 10.0, 1.0, key=f"pf_n_{sym}")
                f_cost = st.number_input("Ortalama maliyet ($)", 0.0, 1e7,
                                         round(_px0, 2), 0.01, key=f"pf_c_{sym}")
                f_date = st.date_input("Alış tarihi", dt.date.today(), key=f"pf_d_{sym}")
                f_stop = st.number_input("İlk stop ($)", 0.0, 1e7,
                                         round(_st0, 2), 0.01,
                                         key=f"pf_s_{sym}")
                if st.button("Kaydet", key=f"pf_save_{sym}", type="primary"):
                    _wl()["portfolio"] = [x for x in _wl()["portfolio"] if x["sym"] != sym]
                    _wl()["portfolio"].append({
                        "sym": sym, "tema": tema, "adet": f_adet, "maliyet": f_cost,
                        "tarih": f_date.isoformat(),
                        "ilk_stop": f_stop if f_stop and f_stop > 0 else None})
                    _wl()["stock_watch"] = [x for x in _wl()["stock_watch"] if x["sym"] != sym]
                    _save_wl(f"{sym} portföye eklendi")
                    st.rerun()
            if not trow2.empty:
                t_ = trow2.iloc[0]
                bits = []
                for c_, lab in (("Temaya Göre 1A", "temaya göre 1 ay"),
                                ("Temaya Göre 1H", "temaya göre 1 hafta")):
                    v_ = pd.to_numeric(t_.get(c_), errors="coerce")
                    if np.isfinite(v_):
                        bits.append(f"{lab} {v_:+.1f}% ({'öne geçmiş' if v_ > 0 else 'geride kalmış'})")
                kg = pd.to_numeric(t_.get("Kalan Gün"), errors="coerce")
                if np.isfinite(kg):
                    bits.append(f"bilançoya {int(kg)} gün")
                if t_.get("P/S Durum"):
                    bits.append(f"P/S değerleme: {t_['P/S Durum']}")
                if bits:
                    st.caption("Tema içi durum: " + " · ".join(bits))

        show_triptych(sym, sym, f"ha_{_fname(sym)}")

        st.markdown("#### 📅 Mevsimsellik")
        LP = load_long(tuple(sorted({sym, "SPY", "QQQ"})), st.session_state.nonce_chart)
        SEAS, stxt = sea.monthly(LP.get(sym))
        if SEAS.empty:
            st.caption(stxt)
        else:
            st.markdown(stxt)
            fig = go.Figure(go.Bar(x=SEAS["Ay"], y=SEAS["Ortalama %"],
                                   marker_color=["#26a69a" if v_ >= 0 else "#ef5350"
                                                 for v_ in SEAS["Ortalama %"]],
                                   text=[f"%{p_:.0f}" for p_ in SEAS["Pozitif %"]],
                                   textposition="outside",
                                   hovertemplate="%{x}: ort. %{y:+.1f}%<extra></extra>"))
            fig.update_layout(height=300, yaxis_title="Ortalama aylık getiri %",
                              **CHART_LAYOUT)
            st.plotly_chart(fig, width="stretch")
            st.caption("Çubuk: o ayın ortalama getirisi · üstteki yüzde: yılların kaçında "
                       "artıda kapandığı.")

        st.markdown("#### 🧮 Temel analiz: çarpanlar — kendi geçmişi ve rakipleri")
        basis = st.radio("HBK tabanı", ["GAAP", "Düzeltilmiş"], horizontal=True,
                         key="ha_basis")
        stock_fundamentals(sym, basis)
        with st.expander(f"📰 {sym} haberleri"):
            it, _ = load_query_news(f"{sym} stock", st.session_state.nonce_news)
            news_list(it, 8)

# ======================================================================= 5
elif PAGE == PAGES[4]:
    h1, h2 = st.columns([4, 1])
    h1.markdown("### 🔥 Hype & Biotech")
    if h2.button("🔄 Yenile", key="hy_ref", width="stretch"):
        bump("nonce_hype")
        st.rerun()
    st.session_state.setdefault("nonce_hype", "0")
    with st.spinner("Reddit, StockTwits ve Yahoo trend listeleri çekiliyor…"):
        HY = load_hype(st.session_state.nonce_hype)
    SOC, FUND = HY.get("social", pd.DataFrame()), HY.get("fund", pd.DataFrame())
    for e in HY.get("errs", []):
        st.caption(f"⚠️ {e}")
    if HY.get("_kaynak"):
        st.caption(HY["_kaynak"])
    t_soc, t_fv, t_bio = st.tabs(["🔥 Sosyal medyada öne çıkanlar", "💎 Gelecek değer adayları",
                                  "🧬 Biotech"])

    # ------------------------------------------------ sosyal
    with t_soc:
        st.caption("Hype skoru: Reddit sırası (40) + StockTwits sırası (30) + Yahoo trend (15) + "
                   "son 24 saatte bahsedilme artışı (±15). Hype bir **dikkat** ölçüsüdür, alım "
                   "sinyali değildir — teknik karar sütununa bakın.")
        if SOC.empty:
            st.warning("Sosyal medya verisi çekilemedi.")
        else:
            topn = st.slider("Kaç hisse", 10, 60, 25, 5, key="hy_n")
            syms = SOC.head(topn)["Sembol"].tolist()
            with st.spinner("Teknik planlar hazırlanıyor…"):
                TBy, _ = syms_stock_table(syms, 0.0)
                Py = load_plans(tuple(syms), st.session_state.nonce_scan)
                PTy = trade_plan_table(TBy, Py, True)
            kar = dict(zip(PTy["Hisse"], PTy["Karar"])) if not PTy.empty else {}
            V_SOC = SOC.head(topn).copy()
            V_SOC["Teknik karar"] = V_SOC["Sembol"].map(kar).fillna("—")
            if not FUND.empty:
                V_SOC = V_SOC.merge(FUND[["Sembol", "Gelir büyümesi %",
                                          "Analist hedef potansiyeli %", "Short % float"]],
                                    on="Sembol", how="left")
            cols = [c for c in ["Sembol", "Ad", "Hype Skoru", "Kaynak", "Bahsedilme",
                                "Bahsedilme Δ", "Teknik karar", "Gelir büyümesi %",
                                "Analist hedef potansiyeli %", "Short % float",
                                "Neden konuşuluyor"] if c in V_SOC.columns]
            st.dataframe(V_SOC[cols], width="stretch", hide_index=True, column_config={
                "Hype Skoru": st.column_config.ProgressColumn(format="%d", min_value=0,
                                                              max_value=100),
                "Bahsedilme Δ": st.column_config.NumberColumn("24s Δ", format="%+.0f%%"),
                "Gelir büyümesi %": st.column_config.NumberColumn(format="%+.0f%%"),
                "Analist hedef potansiyeli %": st.column_config.NumberColumn(
                    "Analist hedefi", format="%+.0f%%"),
                "Short % float": st.column_config.NumberColumn(format="%.1f%%"),
                "Neden konuşuluyor": st.column_config.TextColumn(width="large")})
            hs = st.selectbox("🔍 İncele", syms, key="hy_pick")
            c1, c2 = st.columns(2)
            c1.button("🦅 Hisse Avcısı'nda aç (plan + temel analiz)", on_click=_go,
                      args=(PAGES[3],), kwargs={"ha_sym": hs, "ha_tema": None},
                      key="hy_go", width="stretch")
            if not PTy.empty and hs in set(PTy["Hisse"]):
                r_ = PTy[PTy["Hisse"] == hs].iloc[0]
                c2.markdown(f"**{r_['Karar']}** · giriş {r_['Giriş bölgesi']} · stop {r_['Stop']}"
                            f" · kâr al {r_['Kâr al 1']} / {r_['Kâr al 2']}")
            show_triptych(hs, hs, f"hy_{_fname(hs)}")

    # ------------------------------------------------ gelecek değer
    with t_fv:
        st.caption("Konuşulan hisseler içinde temeli de güçlü olanlar: gelir büyümesi, brüt marj, "
                   "analist hedef potansiyeli, nakit ömrü. Puan 0–100; teknik kararı ⛔ olanlar "
                   "listeye girmez.")
        if FUND.empty or SOC.empty:
            st.info("Temel veriler henüz yok — akşam hesabından sonra dolar (ya da 🔄).")
        else:
            F2 = FUND[FUND["Sembol"].isin(HY.get("top", []))].copy()
            if F2.empty:
                st.info("Hype listesindeki hisseler için temel veri yok.")
            else:
                sc = [hyp.future_value_note(r) for r in F2.to_dict("records")]
                F2["Gelecek Değer"] = [x[0] for x in sc]
                F2["Gerekçe"] = [x[1] for x in sc]
                F2 = F2.merge(SOC[["Sembol", "Hype Skoru"]], on="Sembol", how="left")
                TBf, _ = syms_stock_table(F2["Sembol"].tolist(), 0.0)
                Pf = load_plans(tuple(F2["Sembol"]), st.session_state.nonce_scan)
                PTf = trade_plan_table(TBf, Pf, True)
                F2["Teknik karar"] = F2["Sembol"].map(dict(zip(PTf["Hisse"], PTf["Karar"]))
                                                      if not PTf.empty else {}).fillna("—")
                F2["Giriş bölgesi"] = F2["Sembol"].map(dict(zip(PTf["Hisse"], PTf["Giriş bölgesi"]))
                                                       if not PTf.empty else {})
                F2 = F2[~F2["Teknik karar"].str.startswith("⛔")]
                F2 = F2.sort_values(["Gelecek Değer", "Hype Skoru"], ascending=False)
                st.dataframe(F2[["Sembol", "Ad", "Gelecek Değer", "Hype Skoru", "Teknik karar",
                                 "Giriş bölgesi", "Gelir büyümesi %", "Brüt marj %",
                                 "Analist hedef potansiyeli %", "Piyasa değeri ($B)", "Gerekçe"]],
                             width="stretch", hide_index=True, column_config={
                                 "Gelecek Değer": st.column_config.ProgressColumn(
                                     format="%d", min_value=0, max_value=100),
                                 "Hype Skoru": st.column_config.NumberColumn(format="%d"),
                                 "Gelir büyümesi %": st.column_config.NumberColumn(format="%+.0f%%"),
                                 "Brüt marj %": st.column_config.NumberColumn(format="%.0f%%"),
                                 "Analist hedef potansiyeli %": st.column_config.NumberColumn(
                                     "Analist hedefi", format="%+.0f%%"),
                                 "Piyasa değeri ($B)": st.column_config.NumberColumn(format="%.1f"),
                                 "Gerekçe": st.column_config.TextColumn(width="large")})

    # ------------------------------------------------ biotech
    with t_bio:
        bw = _wl().get("biotech_watch", [])
        b1, b2 = st.columns([3, 1])
        add = b1.text_input("⭐ Özel ilgi listeme sembol ekle (virgülle)", key="bio_add",
                            placeholder="ör. RXRX, CRSP, VKTX")
        if b2.button("Ekle", key="bio_add_b", width="stretch") and add.strip():
            new = [x.strip().upper() for x in add.split(",") if x.strip()]
            _wl()["biotech_watch"] = sorted(set(bw) | set(new))
            _save_wl("Biotech listesi güncellendi")
            load_hype.clear()
            st.rerun()
        if bw:
            r1, r2 = st.columns([3, 1])
            rm = r1.multiselect("⭐ Özel ilgi listem", bw, key="bio_rm",
                                help="Seçip 'Çıkar'a basarak listeden çıkarın")
            if r2.button("Çıkar", key="bio_rm_b", width="stretch") and rm:
                _wl()["biotech_watch"] = [x for x in bw if x not in rm]
                _save_wl("Biotech listesi güncellendi")
                st.rerun()
        only_mine = st.toggle("Yalnızca özel ilgi listem", bool(bw), key="bio_mine")
        bio_syms = (bw if (only_mine and bw) else sorted(set(hyp.BIOTECH) | set(bw)))
        bio_syms = [x for x in bio_syms if x not in ("XBI", "IBB")]
        with st.spinner("Biotech hisseleri: plan, nakit ömrü ve katalizör haberleri…"):
            TBb, _ = syms_stock_table(bio_syms, 0.0)
            Pb = load_plans(tuple(bio_syms), st.session_state.nonce_scan)
            PTb = trade_plan_table(TBb, Pb, True)
            items, _ = load_query_news(nws.FEEDS["Biyoteknoloji"], st.session_state.nonce_news)
            Fb = FUND[FUND["Sembol"].isin(bio_syms)] if not FUND.empty else pd.DataFrame()
            extra = st.session_state.get("bio_fund")
            if extra is not None and not extra.empty:
                Fb = pd.concat([Fb, extra[extra["Sembol"].isin(bio_syms)
                                          & ~extra["Sembol"].isin(Fb["Sembol"] if not Fb.empty
                                                                  else [])]],
                               ignore_index=True)
            names = dict(zip(Fb["Sembol"], Fb["Ad"].fillna(""))) if not Fb.empty else {}
            NM = hyp.match_news(items, bio_syms, names)
        have_f = set(Fb["Sembol"]) if not Fb.empty else set()
        if len(have_f) < 0.5 * len(bio_syms):
            if st.button(f"💵 Nakit ömrü / short / analist verilerini şimdi çek "
                         f"({len(bio_syms) - len(have_f)} hisse, ~1 dk)", key="bio_fund_b"):
                with st.spinner("Yahoo'dan şirket verileri çekiliyor…"):
                    st.session_state["bio_fund"] = hyp.fundamentals(
                        [x for x in bio_syms if x not in have_f], dta.yf_info)
                st.rerun()
        if PTb.empty:
            st.info("Biotech fiyat verisi yok.")
        else:
            B = PTb.rename(columns={"Hisse": "Sembol"})
            if not Fb.empty:
                B = B.merge(Fb[["Sembol", "Nakit ($M)", "Çeyreklik yakım ($M)",
                                "Nakit ömrü (çeyrek)", "Short % float",
                                "Analist hedef potansiyeli %", "Piyasa değeri ($B)"]],
                            on="Sembol", how="left")
            if not TBb.empty and "Kalan Gün" in TBb.columns:
                B = B.merge(TBb[["Sembol", "Kalan Gün"]], on="Sembol", how="left")
            B["Katalizör"] = [(hyp.catalyst_tag(NM[s_][0]["Başlık"]) + " " + NM[s_][0]["Başlık"][:90])
                              if s_ in NM else "—" for s_ in B["Sembol"]]
            B["⭐"] = B["Sembol"].isin(bw).map({True: "⭐", False: ""})
            cols = [c for c in ["⭐", "Karar", "Sembol", "Fiyat", "Giriş bölgesi", "Stop",
                                "Kâr al 1", "R:R", "Nakit ömrü (çeyrek)", "Nakit ($M)",
                                "Çeyreklik yakım ($M)", "Short % float",
                                "Analist hedef potansiyeli %", "Kalan Gün", "Katalizör",
                                "Sebep"] if c in B.columns]
            st.dataframe(B[cols], width="stretch", hide_index=True, column_config={
                "R:R": st.column_config.NumberColumn(format="%.1f"),
                "Nakit ömrü (çeyrek)": st.column_config.NumberColumn(
                    format="%.0f", help="Nakit ÷ çeyreklik nakit yakımı. 4'ün altı: yakında "
                                        "sermaye artırımı (hisse ihracı) riski"),
                "Nakit ($M)": st.column_config.NumberColumn(format="%,.0f"),
                "Çeyreklik yakım ($M)": st.column_config.NumberColumn(format="%,.0f"),
                "Short % float": st.column_config.NumberColumn(format="%.1f%%"),
                "Analist hedef potansiyeli %": st.column_config.NumberColumn(
                    "Analist hedefi", format="%+.0f%%"),
                "Kalan Gün": st.column_config.NumberColumn("Bilançoya gün", format="%d"),
                "Katalizör": st.column_config.TextColumn(width="large"),
                "Sebep": st.column_config.TextColumn(width="large")})
            st.caption("Biotech'e özel: **nakit ömrü** (kaç çeyrek dayanır) ve **katalizör** "
                       "(FDA onayı/ret, PDUFA tarihi, faz 3 sonucu, satın alma, hisse ihracı) "
                       "fiyatı teknikten daha sert hareket ettirir. Katalizör öncesi pozisyonu "
                       "küçük tutun; gap riski stopu anlamsız kılar.")
            bs = st.selectbox("🔍 İncele", B["Sembol"].tolist(), key="bio_pick")
            c1, c2 = st.columns(2)
            if bs not in bw:
                if c1.button(f"⭐ {bs} özel ilgi listeme ekle", key="bio_star", width="stretch"):
                    _wl()["biotech_watch"] = sorted(set(bw) | {bs})
                    _save_wl(f"{bs} özel ilgi listesine eklendi")
                    st.rerun()
            c2.button("🦅 Hisse Avcısı'nda aç", on_click=_go, args=(PAGES[3],),
                      kwargs={"ha_sym": bs, "ha_tema": None}, key="bio_go", width="stretch")
            show_triptych(bs, bs, f"bio_{_fname(bs)}")
            st.markdown(f"#### 📰 {bs} haberleri")
            it2, _ = load_query_news(f"{bs} stock OR {names.get(bs, bs)} FDA OR trial",
                                     st.session_state.nonce_news)
            for it in it2[:8]:
                st.markdown(f"- {hyp.catalyst_tag(it['Başlık'])} [{it['Başlık']}]({it['Link']}) "
                            f"· {it['Tarih'][:16]}")
        with st.expander("🧬 Son biyoteknoloji haberleri (FDA, PDUFA, faz 3)"):
            for it in items[:15]:
                st.markdown(f"- {hyp.catalyst_tag(it['Başlık'])} [{it['Başlık']}]({it['Link']}) "
                            f"· {it['Tarih'][:16]}")

# ======================================================================= 6
elif PAGE == "🔔 Takip & Portföy":
    h1, h2 = st.columns([4, 1])
    h1.markdown("### 🔔 Takip & Portföy")
    if h2.button("🔄 Yenile", key="tp_ref", width="stretch"):
        tracking_digest.clear()
        bump("nonce_scan")
        st.rerun()
    if not store.enabled:
        st.warning("⚠️ Kalıcı kayıt kapalı: listeler yalnızca bu oturumda tutulur ve "
                   "uygulama yeniden başlayınca silinir. **Araçlar → 🚀 Future Themes** "
                   "altındaki '🔑 Kalıcılığı açmak için 3 adım' ile GitHub kaydını açın.")
    DG = _digest()
    st.markdown("#### 🔔 Alarmlar")
    if DG["alarms"]:
        st.dataframe(pd.DataFrame(DG["alarms"]), width="stretch", hide_index=True,
                     column_config={"Mesaj": st.column_config.TextColumn(width="large")})
    else:
        st.success("Şu an alarm yok.")
    st.caption("Alarmlar sayfa her açıldığında son kapanış verisiyle yeniden hesaplanır "
               "(15 dakikalık önbellek). Uygulama kapalıyken bildirim gönderilmez.")

    # ---- sektör takip
    st.markdown("#### 📌 Sektör takip")
    sw = _wl().get("sector_watch", [])
    if not sw:
        st.caption("Tema Avcısı'nda bir temayı açıp **📌 Sektör takibe ekle**'ye basın.")
    else:
        rows = []
        for x in sw:
            ss = DG["sector"].get(x["tema"], {})
            rows.append({"Sektör": x["tema"], "Durum": ss.get("durum", "—"),
                         "Ne yapmalı": ss.get("tavsiye", ""),
                         "Gerekçe": " · ".join(ss.get("neden", [])),
                         "Erken skor": ss.get("skor"), "Eklendi": x.get("eklendi", "")})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True,
                     column_config={"Ne yapmalı": st.column_config.TextColumn(width="large"),
                                    "Gerekçe": st.column_config.TextColumn(width="medium")})
        for sg in DG.get("suggest", []):
            st.info("🔀 " + sg)
        c1, c2 = st.columns([3, 1])
        rm = c1.multiselect("Takipten çıkar", [x["tema"] for x in sw], key="tp_rm_sec")
        if c2.button("Çıkar", key="tp_rm_sec_b", width="stretch") and rm:
            _wl()["sector_watch"] = [x for x in sw if x["tema"] not in rm]
            _save_wl("Sektör takibi güncellendi")
            st.rerun()

    # ---- hisse takip
    st.markdown("#### 👁️ Hisse takip")
    kw = DG.get("watch", [])
    if not kw:
        st.caption("Hisse Avcısı'nda bir hisse seçip **👁️ Hisse takibe al**'a basın. "
                   "Kayıt anındaki giriş/stop/hedef seviyeleri saklanır.")
    else:
        rows = [{"Hisse": x["sym"], "Tema": x.get("tema", ""), "Durum": x["durum"],
                 "Fiyat": _px(x.get("fiyat")),
                 "Giriş bölgesi": f"{_px(x['giris_alt'])} – {_px(x['giris_ust'])}",
                 "Stop": _px(x["stop"]), "Hedef 1": _px(x["hedef1"]),
                 "Hedef 2": _px(x.get("hedef2")), "Alarm": " · ".join(x["alarmlar"]) or "—",
                 "Eklendi": x.get("eklendi", "")} for x in kw]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True,
                     column_config={"Alarm": st.column_config.TextColumn(width="large")})
        c1, c2, c3 = st.columns([2, 1, 1])
        pick = c1.selectbox("Hisse", [x["sym"] for x in kw], key="tp_w_pick")
        c2.button("🔍 Grafik ve analiz", key="tp_w_go", width="stretch", on_click=_go,
                  args=(PAGES[3],), kwargs={"ha_sym": pick, "ha_tema": next(
                      (x.get("tema") for x in kw if x["sym"] == pick), None)})
        if c3.button("Takipten çıkar", key="tp_w_rm", width="stretch"):
            _wl()["stock_watch"] = [x for x in _wl()["stock_watch"] if x["sym"] != pick]
            _save_wl(f"{pick} takipten çıkarıldı")
            st.rerun()

    # ---- portföy
    st.markdown("#### 💼 Portföy")
    pf = DG.get("port", [])
    if not pf:
        st.caption("Hisse Avcısı'nda bir hisse seçip **💼 Portföye ekle** ile adet, maliyet "
                   "ve alış tarihini girin.")
    else:
        rows, tot_v, tot_c = [], 0.0, 0.0
        for x in pf:
            lv = x["lv"]
            ok = lv.get("ok")
            px_ = lv.get("Fiyat", np.nan) if ok else np.nan
            if np.isfinite(px_):
                tot_v += px_ * float(x["adet"])
                tot_c += float(x["maliyet"]) * float(x["adet"])
            rows.append({
                "Hisse": x["sym"], "Durum": x["durum"], "Adet": x["adet"],
                "Maliyet": _px(x["maliyet"]), "Fiyat": _px(px_),
                "K/Z %": lv.get("K/Z %", np.nan) if ok else np.nan,
                "R": lv.get("R", np.nan) if ok else np.nan,
                "İz süren stop": _px(lv.get("Stop")) if ok else "—",
                "Kâr al 1": _px(lv.get("Kâr al 1")) if ok else "—",
                "Kâr al 2": _px(lv.get("Kâr al 2")) if ok else "—",
                "Ekle / maliyet düşür": ", ".join(_px(e) for e in lv.get("Ekle", [])) or
                ("—" if lv.get("Trend sağlam") else "trend zayıf — ekleme yok"),
                "Doygunluk": " · ".join(lv.get("Doygunluk", [])) or "—",
                "Alarm": " · ".join(x["alarmlar"]) or "—"})
        k1, k2, k3 = st.columns(3)
        k1.markdown(kpi("Portföy değeri", f"${tot_v:,.0f}", f"{len(pf)} pozisyon"),
                    unsafe_allow_html=True)
        k2.markdown(kpi("Maliyet", f"${tot_c:,.0f}", ""), unsafe_allow_html=True)
        pl_ = (tot_v / tot_c - 1) * 100 if tot_c else np.nan
        k3.markdown(kpi("Toplam K/Z", f"{pl_:+.1f}%" if np.isfinite(pl_) else "—",
                        f"${tot_v - tot_c:+,.0f}", "pos" if pl_ >= 0 else "neg"),
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True,
                     column_config={
                         "K/Z %": st.column_config.NumberColumn(format="%+.1f%%"),
                         "R": st.column_config.NumberColumn(format="%+.1f",
                                                            help="Kâr ÷ ilk risk (maliyet − ilk stop)"),
                         "Doygunluk": st.column_config.TextColumn(width="medium"),
                         "Alarm": st.column_config.TextColumn(width="large")})
        with st.expander("Seviyeler nasıl hesaplanıyor?"):
            st.markdown(
                "- **İz süren stop:** şu üçünün en yükseği — alıştan beri en yüksek "
                "kapanış − 3 ATR, en yakın S/R desteği − 1 ATR, girdiğiniz ilk stop. "
                "Stop yalnızca yukarı gider.\n"
                "- **Kâr al 1 / 2:** fiyatın üstündeki ilk iki S/R direnci (yoksa 2 ve 3,5 "
                "ATR). Kâr al 1'de pozisyonun bir kısmını satıp stopu maliyete çekmek "
                "klasik swing yönetimidir.\n"
                "- **Doygunluk:** EMA21'den 3 ATR'den fazla uzama, RSI ≥ 75, hacimsiz "
                "yükseliş, alım doruğu, satıcı baskınlığına geçiş, dirence dayanma. "
                "İkisi birlikte gelirse kısmi satış alarmı.\n"
                "- **Ekle / maliyet düşür:** yalnızca trend sağlamsa (fiyat > EMA50 > "
                "EMA200) stopun üstündeki destekler ve EMA21.")
        st.markdown("**Pozisyonları düzenle**")
        ED = pd.DataFrame([{k: x.get(k) for k in ("sym", "tema", "adet", "maliyet", "tarih",
                                                   "ilk_stop")} for x in _wl()["portfolio"]])
        ED = st.data_editor(ED, num_rows="dynamic", width="stretch", key="tp_pf_ed",
                            column_config={"sym": "Hisse", "tema": "Tema", "adet": "Adet",
                                           "maliyet": "Maliyet", "tarih": "Alış tarihi",
                                           "ilk_stop": "İlk stop"})
        e1, e2 = st.columns([1, 3])
        if e1.button("💾 Kaydet", key="tp_pf_save", type="primary", width="stretch"):
            new = []
            for r in ED.to_dict("records"):
                if r.get("sym"):
                    new.append({"sym": str(r["sym"]).upper().strip(), "tema": r.get("tema") or "",
                                "adet": float(r.get("adet") or 0),
                                "maliyet": float(r.get("maliyet") or 0),
                                "tarih": str(r.get("tarih") or trk.today())[:10],
                                "ilk_stop": float(r["ilk_stop"]) if r.get("ilk_stop") not in
                                (None, "") and r.get("ilk_stop") == r.get("ilk_stop") else None})
            _wl()["portfolio"] = new
            _save_wl("Portföy kaydedildi")
            st.rerun()
        pick = e2.selectbox("Grafik ve analiz için hisse", [x["sym"] for x in pf],
                            key="tp_p_pick")
        e2.button("🤖 Bu pozisyonu robota emanet et (sadece yönet)", key="tp_p_rb",
                  on_click=_go, args=("🤖 Robot (simülasyon)",),
                  kwargs={"rb_sym": pick, "rb_plan": "yonet"})
        e2.button("🔍 Hisse Avcısı'nda aç", key="tp_p_go", on_click=_go,
                  args=(PAGES[3],), kwargs={"ha_sym": pick, "ha_tema": next(
                      (x.get("tema") for x in pf if x["sym"] == pick), None)})


# ======================================================================= 7
elif PAGE == "🤖 Robot (simülasyon)":
    h1, h2 = st.columns([4, 1])
    h1.markdown("### 🤖 İşlem robotu — simülasyon")
    if h2.button("🔄 Yenile", key="rb_ref", width="stretch"):
        robot_run.clear()
        load_robot_data.clear()
        st.rerun()
    st.info("Bu robot **gerçek emir göndermez**; 1 saatlik mumlarda bar bar ilerleyip ne "
            "yapacağını kaydeder. Her açılışta başlangıç anından bugüne kadar yeniden oynatılır, "
            "yani sonuçlar her zaman günceldir. Uygulama kapalıyken 'çalışmaz' ama açınca "
            "aradaki her saati geriye dönük işler.")
    with st.expander("Robot nasıl karar veriyor?"):
        st.markdown(
            "**🧠 Akıllı kontrol (giriş):** günlük ana Fibonacci bacağı yükselişte olmalı, fiyat "
            "seçtiğiniz güvenli Fib bölgesine (varsayılan 0.5–0.618) son 6 saatte değmiş olmalı, "
            "ve 1 saatlik **hacimli yeşil mum** gelmeli (hacim ≥ eşik × 20 saatlik ortalama, "
            "kapanış mumun üst yarısında). İsteğe bağlı: günlük kapanış EMA200 üstünde. Bütçe "
            "bölünmez, tek seferde tam giriş.\n\n"
            "**🛡️ Stop:** 0.786 seviyesi / son 10 saatin dibi − 0.25 günlük ATR (en fazla 3 günlük "
            "ATR). +1R'de maliyete, +1.5R'den sonra zirve − 2.5 saatlik ATR ile iz sürer. "
            "**Asla aşağı inmez.**\n\n"
            "**🎯 Hedefler:** bacağın tepesi ve Fib 1.272 / 1.618 / 2.618 uzantıları (ilk hedef "
            "en az 1R). **Dinamik:** her gün başında ADX ≥ 25, +DI > −DI ve RSI > 55 ise aktif "
            "hedef 0.5 günlük ATR yukarı çekilir (bir sonraki Fib uzantısına kadar); ADX < 18 "
            "(yatay) ise hedef daraltılır.\n\n"
            "**📡 %85 Radar & ⚔️ Giyotin:** fiyat hedefe giden yolun %85'ini geçince radar açılır, "
            "zirveyi izler. 1 saatlik kapanış önceki mumun dibini kırar ve zirveden ≥ 1 saatlik "
            "ATR gerilerse kalan pozisyon kapatılır.\n\n"
            "**Planlar:** " + " · ".join(rbt.PLANS.values()))

    # ---------------------------------------------- yeni robot / geçmişte dene
    st.markdown("#### ➕ Yeni robot")
    pf_map = {x["sym"]: x for x in _wl().get("portfolio", [])}
    c1, c2, c3 = st.columns([1, 2, 1])
    sym_r = c1.text_input("Sembol", value=st.session_state.rb_sym or "", key="rb_sym_in",
                          placeholder="ör. NVDA").strip().upper()
    plan_keys = list(rbt.PLANS)
    plan_r = c2.radio("Plan", plan_keys, index=plan_keys.index(st.session_state.rb_plan)
                      if st.session_state.rb_plan in plan_keys else 1,
                      format_func=lambda k: rbt.PLANS[k], key="rb_plan_in")
    with c3:
        if plan_r == "yonet":
            pp = pf_map.get(sym_r, {})
            q_r = st.number_input("Adet", 0.0, 1e7, float(pp.get("adet", 10.0)), 1.0, key="rb_q")
            cost_r = st.number_input("Maliyet", 0.0, 1e7, float(pp.get("maliyet", 0.0)), 0.01,
                                     key="rb_c")
            stop_r = st.number_input("Stop (0 = otomatik)", 0.0, 1e7,
                                     float(pp.get("ilk_stop") or 0.0), 0.01, key="rb_s")
            budget_r = 0.0
        else:
            budget_r = st.number_input("Bütçe ($)", 100.0, 1e8, 10_000.0, 500.0, key="rb_b")
            q_r = cost_r = stop_r = 0.0
    prof_keys = list(rbt.PROFILES)
    prof_r = st.radio("Agresiflik", prof_keys,
                      index=prof_keys.index(st.session_state.get("rb_prof", "akilli"))
                      if st.session_state.get("rb_prof", "akilli") in prof_keys else 0,
                      horizontal=True, key="rb_prof_in",
                      format_func=lambda k: {"temkinli": "🐢 Temkinli", "dengeli": "⚖️ Dengeli",
                                             "agresif": "🔥 Agresif",
                                             "akilli": "🧠 Akıllı (sinyal konseyi)"}[k])
    st.caption(rbt.PROFILE_TXT[prof_r])
    P0 = rbt.PROFILES[prof_r]
    zone_names = list(rbt.ZONES)
    z_default = next((k for k, v in rbt.ZONES.items() if tuple(v) == tuple(P0["zone"])),
                     zone_names[0])
    with st.expander("⚙️ Ayarları elle değiştir (profilin varsayılanları yüklü)"):
        a1, a2, a3, a4 = st.columns(4)
        zone_r = a1.selectbox("Fib giriş bölgesi", zone_names, index=zone_names.index(z_default),
                              key=f"rb_zone_{prof_r}")
        vol_r = a2.slider("Hacim eşiği (× ortalama)", 1.0, 3.0, float(P0["vol_mult"]), 0.1,
                          key=f"rb_vol_{prof_r}")
        radar_r = a3.slider("Radar eşiği", 0.70, 0.95, 0.85, 0.01, key=f"rb_radar_{prof_r}")
        stopm_r = a3.slider("Stop en fazla (× günlük ATR)", 1.0, 4.0, float(P0["stop_atr_max"]),
                            0.25, key=f"rb_stopm_{prof_r}")
        trend_r = a4.toggle(f"Günlük EMA{P0['trend_ma']} filtresi", True, key=f"rb_trend_{prof_r}")
        loop_r = a4.toggle("İşlem bitince yeniden giriş ara", True, key=f"rb_loop_{prof_r}")
        a5, a6, a7 = st.columns(3)
        brk_r = a5.toggle("20 saatlik zirve kırılımıyla gir", bool(P0["breakout"]),
                          key=f"rb_brk_{prof_r}")
        bnc_r = a6.toggle("1s EMA21 sekmesiyle gir", bool(P0["ema_bounce"]),
                          key=f"rb_bnc_{prof_r}")
        tr_r = a7.slider("İz süren stop (× 1s ATR)", 1.5, 4.0, float(P0["trail_atr"]), 0.25,
                         key=f"rb_trail_{prof_r}")
        a8, a9, a10 = st.columns(3)
        risk_r = a8.slider("İşlem başına risk (% sermaye, 0 = bütçenin tamamı)", 0.0, 10.0,
                           float(P0.get("risk_pct", 0.0)) * 100, 0.5, key=f"rb_risk_{prof_r}",
                           help="Stopa gelirse kaybedilecek tutar. Pozisyon büyüklüğü buna göre "
                                "hesaplanır; giriş yine tek seferde yapılır.")
        reg_r = a9.toggle("Yalnızca yükselen trendde gir (fiyat > EMA50 > EMA200)",
                          bool(P0.get("regime_gate", False)), key=f"rb_reg_{prof_r}")
        ride_r = a9.toggle("🏄 Trend modu (güçlü trendde hedef tavanı kalkar)",
                           bool(P0.get("trend_ride", False)), key=f"rb_ride_{prof_r}")
        ml_r = a10.number_input("Devre kesici: art arda kaç zarardan sonra mola (0 = kapalı)",
                                0, 5, int(P0.get("max_losses", 0)), 1, key=f"rb_ml_{prof_r}")
    cfg = {**{k: (list(v) if isinstance(v, tuple) else v) for k, v in P0.items()},
           "profile": prof_r, "plan": plan_r, "budget": budget_r,
           "zone": list(rbt.ZONES[zone_r]), "vol_mult": vol_r, "radar": radar_r,
           "stop_atr_max": stopm_r, "trend_filter": trend_r, "loop": loop_r,
           "breakout": brk_r, "ema_bounce": bnc_r, "trail_atr": tr_r,
           "risk_pct": risk_r / 100, "regime_gate": reg_r, "trend_ride": ride_r,
           "max_losses": int(ml_r),
           "qty": q_r, "cost": cost_r, "user_stop": stop_r}
    b1, b2, b3 = st.columns([1, 1, 2])
    period = b1.selectbox("Geçmişte dene: dönem", ["Son 3 ay", "Son 6 ay", "Son 12 ay",
                                                    "Son 24 ay"], index=2, key="rb_per")
    months = int(period.split()[1])
    if b1.button("🧪 Geçmişte dene", key="rb_bt", width="stretch", disabled=not sym_r):
        st.session_state["rb_bt_res"] = (sym_r, json.dumps(cfg, sort_keys=True),
                                         (pd.Timestamp.today() - pd.DateOffset(months=months))
                                         .normalize().isoformat())
    if b2.button("▶️ Robotu başlat (bugünden itibaren)", key="rb_start", type="primary",
                 width="stretch", disabled=not sym_r):
        rid = f"{sym_r}-{prof_r}-{plan_r}-{dt.datetime.now():%Y%m%d%H%M%S}"
        start_ = (pd.Timestamp.now(tz="America/New_York").tz_localize(None)
                  if plan_r != "yonet" else
                  pd.Timestamp(pf_map.get(sym_r, {}).get("tarih") or pd.Timestamp.today()))
        _wl()["robots"].append({"id": rid, "sym": sym_r, "cfg": cfg,
                                "start": pd.Timestamp(start_).isoformat(),
                                "olusturma": dt.datetime.now().isoformat(timespec="minutes")})
        _save_wl(f"🤖 {sym_r} robotu başlatıldı")
        st.rerun()
    if b3.button("🔬 En iyi ayarı bul (4 profil × 3 plan, ~30 sn)", key="rb_opt",
                 width="stretch", disabled=not sym_r or plan_r == "yonet"):
        st.session_state["rb_opt_res"] = (sym_r, (pd.Timestamp.today() - pd.DateOffset(
            months=months)).normalize().isoformat(), budget_r)
    b3.caption("**Geçmişte dene:** aynı kurallar seçilen dönemde çalıştırılır (sonuç kaydedilmez). "
               "**Robotu başlat:** bugünden itibaren ileriye doğru simüle eder ve listeye eklenir; "
               "'sadece yönet' planında başlangıç portföydeki alış tarihidir.")
    if st.session_state.get("rb_opt_res"):
        sy_, st_, bu_ = st.session_state["rb_opt_res"]
        st.markdown(f"##### 🔬 {sy_} · {pd.Timestamp(st_):%d.%m.%Y}'den bugüne · hangi ayar "
                    "daha iyi çalıştı?")
        with st.spinner("12 kombinasyon geçmiş veride deneniyor…"):
            O = robot_optimize(sy_, st_, bu_, st.session_state.nonce_scan)
        if O.empty:
            st.error("Veri alınamadı.")
        else:
            O2 = O.copy()
            O2["Profil"] = O2["Profil"].map({"temkinli": "🐢 Temkinli", "dengeli": "⚖️ Dengeli",
                                             "agresif": "🔥 Agresif", "akilli": "🧠 Akıllı"})
            O2["Plan"] = O2["Plan"].map(lambda k: rbt.PLANS[k].split(" —")[0])
            st.dataframe(O2, width="stretch", hide_index=True, column_config={
                "Getiri %": st.column_config.NumberColumn(format="%+.1f%%"),
                "Al-tut %": st.column_config.NumberColumn(format="%+.1f%%"),
                "Maks. düşüş %": st.column_config.NumberColumn(format="%.1f%%"),
                "Getiri / düşüş": st.column_config.NumberColumn(format="%.2f"),
                "Kazanan %": st.column_config.NumberColumn(format="%.0f%%"),
                "Ort. R": st.column_config.NumberColumn(format="%+.2f")})
            best = O.iloc[0]
            st.success(f"En yüksek getiri: **{best['Profil']} · {rbt.PLANS[best['Plan']].split(' —')[0]}**"
                       f" ({best['Getiri %']:+.1f}%, maks. düşüş {best['Maks. düşüş %']:.1f}%, "
                       f"al-tut {best['Al-tut %']:+.1f}%). Yukarıdan bu profili ve planı seçip "
                       "başlatabilirsiniz.")
            st.caption("⚠️ Geçmişte en iyi çalışan ayar gelecekte de en iyisi olmayabilir "
                       "(aşırı uydurma riski). Getiri / düşüş oranı yüksek olanlar daha sağlamdır.")
    if st.session_state.get("rb_bt_res"):
        sy_, cj_, st_ = st.session_state["rb_bt_res"]
        st.markdown(f"##### 🧪 {sy_} · {rbt.PLANS[json.loads(cj_)['plan']]} · "
                    f"{pd.Timestamp(st_):%d.%m.%Y}'den bugüne")
        with st.spinner("Geçmiş 1 saatlik mumlar üzerinde oynatılıyor…"):
            show_robot(robot_run(sy_, cj_, st_, st.session_state.nonce_scan), "bt")

    # ---------------------------------------------- robotlarım
    st.markdown("---")
    st.markdown("#### 🤖 Robotlarım")
    RB = _wl().get("robots", [])
    if not store.enabled:
        st.caption("⚠️ Kalıcı kayıt kapalı — robot listesi uygulama uyuyunca silinir.")
    if not RB:
        st.caption("Henüz robot yok. Yukarıdan bir sembol seçip **▶️ Robotu başlat**'a basın.")
    else:
        rows, results = [], {}
        with st.spinner("Robotlar güncelleniyor…"):
            for rb_ in RB:
                R = robot_run(rb_["sym"], json.dumps(rb_["cfg"], sort_keys=True), rb_["start"],
                              st.session_state.nonce_scan)
                results[rb_["id"]] = R
                if not R.get("ok"):
                    rows.append({"Robot": rb_["id"], "Sembol": rb_["sym"], "Durum": "⚠️ veri yok"})
                    continue
                stt, s_ = R["state"], R["stats"]
                E = R["events"]
                rows.append({
                    "Robot": rb_["id"], "Sembol": rb_["sym"],
                    "Plan": rbt.PLANS[rb_["cfg"]["plan"]].split(" —")[0]
                    + f" · {rb_['cfg'].get('profile', 'temkinli')}",
                    "Başlangıç": rb_["start"][:16].replace("T", " "),
                    "Durum": ("📡 radar açık" if stt.get("radar") else "📈 pozisyonda")
                    if stt["pozisyonda"] else "💤 giriş bekliyor",
                    "Açık K/Z %": stt["K/Z % (açık)"] if stt["pozisyonda"] else np.nan,
                    "Hedefe %": stt["hedefe %"] if stt["pozisyonda"] else np.nan,
                    "Getiri %": s_["Getiri %"], "İşlem": s_["İşlem"],
                    "Son olay": (f"{E['Zaman'].iloc[-1]:%d.%m %H:%M} {E['Olay'].iloc[-1]}"
                                 if len(E) else "—")})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, column_config={
            "Açık K/Z %": st.column_config.NumberColumn(format="%+.1f%%"),
            "Hedefe %": st.column_config.NumberColumn(format="%+.1f%%"),
            "Getiri %": st.column_config.NumberColumn(format="%+.1f%%")})
        r1, r2 = st.columns([3, 1])
        pick = r1.selectbox("Robot", [x["id"] for x in RB], key="rb_pick")
        if r2.button("🗑️ Robotu durdur ve sil", key="rb_del", width="stretch"):
            _wl()["robots"] = [x for x in RB if x["id"] != pick]
            _save_wl(f"{pick} silindi")
            st.rerun()
        show_robot(results[pick], _fname(pick))


# ---------------------------------------------------------------- Araçlar
if PAGE != PAGES[-1]:
    st.stop()


TABS = st.tabs([
    "🧭 Karar Hunisi (eski)", "🔬 Hisse Analizi", "🌐 Makro & Rejim", "🔥 Tema Takibi", "🦅 ETF Radarı", "⚖️ Çarpan Uçurumu",
    "🦈 Haftalık", "🚨 4H Omni Swing", "🚀 Future Themes", "📅 Bilanço",
    "📄 Rapor",
])
(tab_funnel, tab_stock, tab_macro, tab_theme, tab_etf, tab_val, tab_week, tab_omni,
 tab_future, tab_earn, tab_report) = TABS


# ==========================================================================
# 0) KARAR HUNİSİ
# ==========================================================================
with tab_funnel:
    st.markdown(
        "Bir yatırım kararını **beş soruya** böler. Her adım bir öncekinin "
        "cevabına dayanır: piyasa risk almayı ödüllendiriyor mu, para hangi "
        "bölgeye akıyor, S&P 500 / Nasdaq / kriptoya girmek için şartlar uygun "
        "mu, hangi tema **erken** ivmeleniyor, o temada hangi hisse öne çıkıyor.")
    h1, h2 = st.columns([1, 4])
    if h1.button("🔄 Huniyi yenile", key="fn_ref", width="stretch"):
        bump("nonce_funnel")
        st.rerun()

    with st.spinner("Rotasyon oranları çekiliyor (15 yıllık günlük veri)…"):
        ROT, SER, ROT_FAILED, FILES = load_rotation(st.session_state.nonce_funnel)
    eksik = [k for k, v in FILES.items() if not v]
    h2.caption(
        "Kripto endeksleri: " + (FILES["kripto"] or "❌ dosya yok") + " · "
        "Fed likiditesi: " + (FILES["likidite"] or "❌ dosya yok")
        + (" · Çekilemeyen: " + ", ".join(ROT_FAILED) if ROT_FAILED else ""))
    if eksik:
        st.info("Kripto endeksleri (TOTAL, TOTAL3, BTC.D, OTHERS/BTC) ve Fed "
                "net likiditesi repodaki `data/` klasöründen okunur. GitHub'da "
                "**Actions → FINRA short hacim güncelle → Run workflow** ile "
                "bir kez çalıştırın; sonra her iş günü kendiliğinden güncellenir.")

    # ------------------------------------------------------------------ 1
    section("Adım 1 · Risk açık mı?")
    PIL = fnl.risk_pillars(M.scores, ROT)
    V = fnl.risk_verdict(PIL, M.regime)
    v1, v2 = st.columns([1, 2])
    v1.markdown(kpi("Piyasa durumu", V["etiket"],
                    f"skor {V['skor']:.0f}/100 · {V['boyut']}"
                    if np.isfinite(V["skor"]) else V["boyut"], V["renk"]),
                unsafe_allow_html=True)
    v2.markdown(f"**Ne demek?** {V['ozet']}")
    v2.caption(f"{V['n_ok']} sütun olumlu, {V['n_bad']} sütun olumsuz. "
               f"Makro sekmesindeki rejim etiketi: {M.regime}")
    pc = st.columns(6)
    for col, p in zip(pc, PIL):
        tone = ("pos" if p["durum"] == "✅" else "neg" if p["durum"] == "❌" else "")
        col.markdown(kpi(p["ad"], f"{p['durum']} {p['skor']:.0f}"
                         if np.isfinite(p["skor"]) else "➖", p["ne"], tone),
                     unsafe_allow_html=True)
    if any(x in M.regime for x in ("OPEX", "FOMC")):
        st.info(f"📅 **Takvim uyarısı — {M.regime}:** {M.regime_desc}")
    with st.expander("Risk açık / risk kapalı ne demek? Sütunlar nasıl okunur?"):
        st.markdown(
            "**Risk açık (risk-on):** Yatırımcılar getiri peşinde; para "
            "tahvil, altın ve nakitten hisseye, küçük şirketlere, büyüme "
            "hisselerine ve kriptoya akar. Alım sinyallerinin isabeti "
            "yüksektir.\n\n"
            "**Risk kapalı (risk-off):** Yatırımcılar korunma peşinde; para "
            "güvenli limana (hazine, altın, dolar, defansif sektörler) kaçar. "
            "İyi görünen alım sinyalleri bile sık başarısız olur.\n\n"
            "Karar altı sütunun ağırlıklı ortalamasıdır (Trend %25, diğerleri "
            "%15). Her sütun 0–100: **60+ ✅**, **40–60 ⚠️**, **40 altı ❌**.")
        for p in PIL:
            st.markdown(f"- **{p['ad']}** — {p['ne']} *{p['neden']}*")

    # ------------------------------------------------------------------ 2
    section("Adım 2 · Para nereye akıyor?")
    if ROT.empty:
        st.warning("Rotasyon verisi çekilemedi. Birkaç dakika sonra yenileyin.")
    else:
        for line in fnl.money_flow_summary(ROT):
            st.markdown(f"- {line}")

        SEG = fnl.segment_verdicts(ROT)
        s1, s2 = st.columns([1, 2])
        seg_pick = s1.selectbox("Alacağınız hisse hangi bölgede?",
                                list(fnl.SEGMENTS), key="fn_seg")
        row = SEG[SEG["Segment"] == seg_pick]
        if not row.empty:
            r = row.iloc[0]
            tone = ("pos" if r["Karar"].startswith("🟢")
                    else "neg" if r["Karar"].startswith("🔴") else "")
            s1.markdown(kpi(seg_pick, r["Karar"], f"skor {r['Skor']:+.0f}", tone),
                        unsafe_allow_html=True)
            s2.markdown(f"**Kapsam:** {r['Kapsam']}")
            s2.markdown("**Dayanak:** " + r["Dayanak"])
            uyum = ("✅ Piyasa risk açık ve bu bölgeye iştah var — ortam uygun."
                    if V["etiket"].startswith("🟢") and r["Skor"] >= 5 else
                    "⚠️ Bölgeye iştah var ama genel piyasa temkinli — küçük "
                    "pozisyon, sadece en güçlü kurulumlar."
                    if r["Skor"] >= 5 else
                    "🌱 Bölgede erken dönüş sinyali var — izleme listesine "
                    "alın, teyit bekleyin."
                    if r["Erken Sinyal"] else
                    "❌ Para bu bölgeye akmıyor — doğru zaman değil. Adım 2'de "
                    "iştahlı bölgelere bakın.")
            s2.markdown(f"**Sonuç:** {uyum}")

        st.dataframe(
            SEG[["Segment", "Karar", "Skor", "Erken Sinyal", "Dayanak"]],
            width="stretch", hide_index=True,
            column_config={
                "Skor": st.column_config.ProgressColumn(
                    format="%+.0f", min_value=-100, max_value=100,
                    help="Bölgeye ait oranların ortalama akış skoru"),
                "Dayanak": st.column_config.TextColumn(width="large")})

        st.markdown("**Bütün akış göstergeleri**")
        gtabs = st.tabs(["Hisse içi rotasyon", "Varlık sınıfları", "Kripto",
                         "Likidite / para basımı"])
        gmap = ["Hisse içi", "Varlık sınıfları", "Kripto", "Likidite"]
        for gt, g in zip(gtabs, gmap):
            with gt:
                sub = ROT[ROT["Grup"] == g].copy()
                if sub.empty:
                    st.caption("Veri yok" + (" — data/ klasöründeki dosya "
                                             "henüz oluşmamış." if g in
                                             ("Kripto", "Likidite") else "."))
                    continue
                sub["Çeyreklik"] = sub.apply(
                    lambda x: ("📐 TAZE KIRILIM" if x["Taze Çeyreklik Kırılım"]
                               else "✅ direnç üstü" if x["Çeyreklik Kırılım"]
                               else f"{x['Dirence Uzaklık %']:+.1f}% uzakta"
                               if np.isfinite(x["Dirence Uzaklık %"]) else "—"),
                    axis=1)
                st.dataframe(
                    sub.rename(columns={"Akış": "Ne oluyor / ne demek",
                                        "Risk Okuması": "Piyasa için"})[
                        ["Gösterge", "Piyasa için", "Ne oluyor / ne demek", "Faz", "20G %",
                         "63G %", "Çeyreklik", "Not"]],
                    width="stretch", hide_index=True,
                    column_config={
                        "20G %": st.column_config.NumberColumn(format="%+.1f%%"),
                        "63G %": st.column_config.NumberColumn(format="%+.1f%%"),
                        "Çeyreklik": st.column_config.TextColumn(
                            help="Çeyrek kapanışı, önceki 8 çeyreğin en yüksek "
                                 "kapanışını geçti mi?"),
                        "Not": st.column_config.TextColumn(width="large")})
                if g == "Kripto":
                    st.caption("TOTAL, TOTAL3, OTHERS ve BTC.D, ilk ~150 coinin "
                               "bugünkü arzı × günlük fiyatla hesaplanan "
                               "yaklaşık değerlerdir; yön ve kırılımlar "
                               "TradingView ile uyumludur, seviye birkaç puan "
                               "sapabilir.")

        # ---- oran grafiği (QQQ/SPY çeyreklik gibi)
        st.markdown("**Oran grafiği ve kırılım**")
        k1, k2 = st.columns([2, 1])
        keys = [k for k in ROT["Anahtar"]]
        ck = k1.selectbox("Gösterge", keys,
                          index=keys.index("QQQ/SPY") if "QQQ/SPY" in keys else 0,
                          format_func=lambda k: fnl.PAIR[k]["ad"], key="fn_ratio")
        per = k2.radio("Zaman dilimi", ["Çeyreklik", "Aylık", "Haftalık"],
                       horizontal=True, key="fn_per")
        sser = SER.get(ck)
        if sser is not None:
            rule, look = {"Çeyreklik": ("QE", 8), "Aylık": ("ME", 12),
                          "Haftalık": ("W-FRI", 26)}[per]
            o = sser.resample(rule).agg(["first", "max", "min", "last"]).dropna()
            o.columns = ["Open", "High", "Low", "Close"]
            o = o.tail({"QE": 48, "ME": 120, "W-FRI": 156}[rule])
            bi = fnl.breakout_info(sser, rule, look)
            fig = go.Figure(go.Candlestick(
                x=o.index, open=o["Open"], high=o["High"], low=o["Low"],
                close=o["Close"], increasing_line_color="#2fbe86",
                decreasing_line_color="#e66767", name=fnl.PAIR[ck]["ad"]))
            if np.isfinite(bi["direnc"]):
                fig.add_hline(y=bi["direnc"], line_dash="dot",
                              line_color="#c98500",
                              annotation_text=f"önceki {look} dönemin zirvesi",
                              annotation_font_color="#c98500")
            fig.update_layout(height=380, xaxis_rangeslider_visible=False,
                              showlegend=False, **CHART_LAYOUT)
            st.plotly_chart(fig, width="stretch")
            if bi["kirilim"]:
                st.success(f"📐 {fnl.PAIR[ck]['ad']} {per.lower()} kapanışta "
                           f"önceki {look} dönemin zirvesinin "
                           f"%{bi['uzaklik']:.1f} üstünde"
                           + (" — **bu dönem kırdı (taze)**." if bi.get("taze")
                              else "."))
            elif np.isfinite(bi["uzaklik"]):
                st.caption(f"Dirence uzaklık: {bi['uzaklik']:+.1f}%")
            st.caption(fnl.PAIR[ck]["not_"])

    # ------------------------------------------------------------------ 3
    section("Adım 3 · Karar: S&P 500 · Nasdaq · Kripto")
    st.markdown(
        "Her varlık **günlük, haftalık ve aylık** mumlarda bütün modüllerle tek "
        "tek incelenir: mum, hacim, trend, EMA'lar, RSI, whale/retail, efor, "
        "konfluans, Fusion, Synergy, Omni, sıkışma ve tükenme; ayrıca Pine "
        "göstergelerinizden çevrilen Mum Gücü, Fibonacci, Elliott, Q-dry, "
        "Volatility Hole, Whale profili, Gap/FVG, S/R matrisi, Wyckoff "
        "bölgeleri, likidite havuzları, GFR ve VSA & Delta. Karar haftalık "
        "ağırlıklıdır (G %30 · H %40 · A %30); Adım 1'de risk kapalıysa "
        "\"uygun\" kararı verilmez.")
    with st.spinner("Endeksler üç zaman diliminde inceleniyor…"):
        DEC = load_decision(st.session_state.nonce_funnel, ROT)
    dcols = st.columns(len(dcs.DEC_ASSETS))
    DECV = {}
    for col, (ad, info) in zip(dcols, DEC["assets"].items()):
        dd = dcs.asset_decision(info["scores"], ROT, ad, V["etiket"],
                                DEC["vix"]["skor"])
        DECV[ad] = dd
        tfs = " · ".join(
            f"{tf[0]} {info['scores'][tf]:+.0f}" for tf in dcs.TFS
            if np.isfinite(info["scores"].get(tf, np.nan)))
        col.markdown(kpi(ad, dd["etiket"],
                         (f"skor {dd['skor']:+.0f} · " if np.isfinite(dd["skor"])
                          else "") + tfs, dd["tone"]), unsafe_allow_html=True)
    if DEC["notes"]:
        st.markdown("**Piyasa notları**")
        for ic, tx in DEC["notes"]:
            st.markdown(f"{ic} {tx}")
    pick = st.radio("Hangi endeksi açalım?", list(dcs.DEC_ASSETS), horizontal=True,
                    key="fn_dec")
    info, dd = DEC["assets"][pick], DECV[pick]
    pos, neg = dcs.reasons(info["mods"])
    r1, r2 = st.columns(2)
    r1.markdown("**👍 Lehte**\n" + "\n".join(f"- {x}" for x in pos)
                if pos else "**👍 Lehte**\n- —")
    r2.markdown("**👎 Aleyhte**\n" + "\n".join(f"- {x}" for x in neg)
                if neg else "**👎 Aleyhte**\n- —")
    if dd["baglam_txt"]:
        st.caption("Bağlam (Adım 2): " + " · ".join(dd["baglam_txt"]))
    st.dataframe(dcs.summary(info["mods"]), width="stretch", hide_index=True,
                 column_config={"Konu": st.column_config.TextColumn(width="medium"),
                                "Öne çıkan (haftalık)": st.column_config.TextColumn(
                                    width="large")})
    with st.expander(f"🔎 Bütün modüller tek tek ({pick}, ayrıntı)"):
        MX = dcs.matrix(info["mods"])
        st.dataframe(MX, width="stretch", hide_index=True,
                     height=min(38 * (len(MX) + 1), 1100),
                     column_config={tf: st.column_config.TextColumn(width="large")
                                    for tf in dcs.TFS})
        st.caption("✅ güçlü olumlu · 🟢 hafif olumlu · ⚪ nötr · 🔴 hafif olumsuz · "
                   "❌ güçlü olumsuz. Haftalık ve aylık mumlar içinde bulunulan "
                   "(henüz kapanmamış) dönemi de gösterir.")

    # ------------------------------------------------------------------ 3b
    section(f"Adım 3b · {pick} için alım planı: tema → hisse → giriş · stop · kâr al")
    index_ok = dd["etiket"].startswith(("✅", "🟡"))
    if index_ok:
        st.success(f"**{pick} alım bölgesinde** ({dd['etiket']}). Aşağıda bu "
                   "endeksin para giren temaları ve o temalarda hangi hissenin hangi "
                   "fiyattan alınabileceği, stopu ve kâr alma seviyeleri var.")
    else:
        st.warning(f"**{pick} şu an alım bölgesinde değil** ({dd['etiket']}). "
                   "Planlar yine gösteriliyor ama hiçbir hisseye ✅ AL verilmez; "
                   "seviyeleri bekleme/limit emir haritası olarak okuyun.")
    T3, _, _, IDX3, _ = load_theme_rotation(st.session_state.nonce_funnel)
    themes3 = [t for t in pln.INDEX_THEMES.get(pick, []) if t in uni.THEME_TRACKER]
    if T3.empty:
        TT = pd.DataFrame({"Tema": themes3})
    else:
        TT = T3[T3["Tema"].isin(themes3)].copy()
    if TT.empty:
        st.info("Bu endeks için tema verisi yok.")
    else:
        if "_q" in TT.columns:
            TT["Para"] = TT["_q"].map({"lider": "✅ giriyor (lider)",
                                       "iyilesen": "🌱 girmeye başladı",
                                       "zayif": "⚠️ çıkmaya başladı",
                                       "geride": "⛔ çıkıyor / zayıf"}).fillna("—")
            TT["_o"] = TT["_q"].map({"iyilesen": 0, "lider": 1, "zayif": 2,
                                     "geride": 3}).fillna(4)
            TT = TT.sort_values(["_o", "Erken Skor"], ascending=[True, False])
        tcols = [c for c in ["Tema", "Para", "Erken Skor", "1H %", "1A %"]
                 if c in TT.columns]
        st.markdown("**1) Temalar** — önce para girmeye başlayan ve lider olanlar:")
        st.dataframe(TT[tcols], width="stretch", hide_index=True,
                     column_config={
                         "Erken Skor": st.column_config.ProgressColumn(
                             format="%d", min_value=0, max_value=100),
                         "1H %": st.column_config.NumberColumn(format="%+.1f%%"),
                         "1A %": st.column_config.NumberColumn(format="%+.1f%%")})
        good = TT[TT["_q"].isin(["lider", "iyilesen"])]["Tema"].tolist() \
            if "_q" in TT.columns else TT["Tema"].tolist()

        # ---- 📥 bütün temaların planı tek Excel dosyasında (internetsiz inceleme)
        xkey = f"xlsx_{pick}"
        e1, e2 = st.columns([1, 1])
        if e1.button(f"📥 {pick} — bütün temaların planını Excel'e hazırla",
                     key=f"xbtn_{pick}", width="stretch"):
            per = {}
            prog = st.progress(0.0, text="Temalar hesaplanıyor…")
            tlist = list(TT["Tema"])
            for i, tm in enumerate(tlist):
                prog.progress((i + 1) / len(tlist), text=f"{tm} ({i + 1}/{len(tlist)})")
                try:
                    tb_, h_ = theme_stock_table(tm, 10.0, IDX3.get(tm))
                    p_ = load_plans(tuple(h_), st.session_state.nonce_scan) if h_ \
                        else pd.DataFrame()
                    per[tm] = trade_plan_table(tb_, p_, index_ok and tm in good)
                except Exception as exc:          # tek tema hatası dosyayı bozmasın
                    _log.info("Excel tema %s: %s", tm, exc)
            prog.empty()
            info = [("Endeks kararı", f"{pick}: {dd['etiket']}"),
                    ("Piyasa (Adım 1)", V["etiket"]),
                    ("Hazırlanma", dt.datetime.now().strftime("%d.%m.%Y %H:%M")),
                    ("Veri", snap.summary() if _use_snap(st.session_state.nonce_scan)
                     else "canlı")]
            st.session_state[xkey] = pln.excel_report(
                f"AETHER APEX · {pick} alım planı", info, TT[tcols], per)
        if xkey in st.session_state:
            safe = "".join(ch for ch in pick if ch.isalnum()) or "endeks"
            e2.download_button(
                "⬇️ Excel dosyasını indir", st.session_state[xkey],
                file_name=f"alim_plani_{safe}_{dt.date.today():%Y%m%d}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"xdl_{pick}", width="stretch", type="primary")
        opts = good + [t for t in TT["Tema"] if t not in good]
        k1, k2 = st.columns([3, 1])
        tema3 = k1.selectbox("2) Hangi temanın hisselerine bakalım?", opts,
                             key=f"plan_tema_{pick}",
                             format_func=lambda t: ("✅ " if t in good else "⛔ ") + t)
        liq3 = k2.number_input("Min. günlük hacim ($M)", 0.0, 500.0, 10.0, 1.0,
                               key="plan_liq")
        if tema3 not in good:
            st.warning(f"**{tema3}** temasından para çıkıyor — bu temada yeni "
                       "alım önerilmez; liste yalnızca bilgi için.")
        with st.spinner(f"{tema3} hisseleri için plan hazırlanıyor…"):
            TB3, hold3 = theme_stock_table(tema3, liq3, IDX3.get(tema3))
            P3 = load_plans(tuple(hold3), st.session_state.nonce_scan) if hold3 \
                else pd.DataFrame()
        PT = trade_plan_table(TB3, P3, index_ok and tema3 in good)
        if PT.empty:
            st.info("Bu tema için hisse listesi ya da fiyat verisi yok.")
        else:
            n_al = int(PT["Karar"].str.startswith("✅").sum())
            n_bk = int(PT["Karar"].str.startswith("🟡").sum())
            n_uz = len(PT) - n_al - n_bk
            st.markdown(" ".join([badge(f"✅ AL · {n_al}", "pos"),
                                  badge(f"🟡 LİMİT EMİR / BEKLE · {n_bk}", ""),
                                  badge(f"⛔ UZAK DUR · {n_uz}", "neg")]),
                        unsafe_allow_html=True)
            pcols = ["Hisse", "Fiyat", "Giriş bölgesi", "Stop", "Kâr al 1",
                     "Kâr al 2", "R:R", "Sebep"]
            pcfg = {"R:R": st.column_config.NumberColumn(
                        format="%.1f", help="Kâr al 1'e kazanç ÷ stopa kayıp"),
                    "Sebep": st.column_config.TextColumn(width="large")}
            st.markdown("**✅ Şimdi alınabilir** — fiyat giriş bölgesinde, "
                        "risk/ödül yeterli")
            al = PT[PT["Karar"].str.startswith("✅")]
            if al.empty:
                st.caption("Şu an giriş bölgesinde olan uygun hisse yok.")
            else:
                st.dataframe(al[pcols], width="stretch", hide_index=True,
                             column_config=pcfg)
            st.markdown("**🟡 Limit emir / bekle** — hisse iyi ama fiyat henüz "
                        "giriş bölgesinde değil (limit emir giriş bölgesinin "
                        "üst sınırına) ya da teyit bekleniyor")
            bk = PT[PT["Karar"].str.startswith("🟡")]
            if bk.empty:
                st.caption("Yok.")
            else:
                st.dataframe(bk[pcols], width="stretch", hide_index=True,
                             column_config=pcfg)
            uz = PT[~PT["Karar"].str.startswith(("✅", "🟡"))]
            with st.expander(f"⛔ Uzak durulacaklar ({len(uz)}) — sebepleriyle"):
                st.dataframe(uz[["Hisse", "Fiyat", "Sebep"]], width="stretch",
                             hide_index=True,
                             column_config={"Sebep": st.column_config.TextColumn(
                                 width="large")})
            with st.expander("Seviyeler nasıl hesaplanıyor?"):
                st.markdown(
                    "- **Giriş bölgesi:** fiyatın altındaki (en fazla 3 ATR) "
                    "desteklerden **en çok çakışanı** — S/R matrisi, açık boşluklar "
                    "(FVG/GAP), Wyckoff alım bölgeleri, EMA21/50/200 ve ana yapı "
                    "Fibonacci 0.382/0.5/0.618. Sebep sütununda hangileri çakıştığı "
                    "yazar.\n"
                    "- **Stop:** giriş bölgesinin dibi − 1 ATR (GFR çalışmasında "
                    "aynı beklentiyi en az gereksiz stopla veren mesafe).\n"
                    "- **Kâr al 1 / 2:** girişten en az 1R yukarıdaki ilk ve ikinci "
                    "direnç (S/R direnci, doldurulmamış boşluk, Wyckoff satış "
                    "bölgesi, 52 hafta zirvesi, Fib 1.272/1.618). Yakında direnç "
                    "yoksa 2R ve 3R.\n"
                    "- **⛔ Uzak dur:** satış/dağıtım sinyali, 7 gün içinde bilanço, "
                    "düşüş trendi, düşük hacim, en iyi hedefte bile R:R < 1.5 ya da "
                    "fiyat yeni dipte (altında destek yok).\n"
                    "- Seviyeler günlük mumdan, son kapanışa göre hesaplanır; "
                    "emir vermeden önce grafikte teyit edin.")

    # ------------------------------------------------------------------ 4
    section("Adım 4 · Hangi tema erken ivmeleniyor?")
    st.markdown(
        "Her tema S&P 500 ile kıyaslanır. **Yatay eksen** temanın endeksten "
        "güçlü mü zayıf mı olduğunu, **dikey eksen** bu gücün artıp "
        "azaldığını gösterir. Akıllı paranın erken izi, **🌱 İyileşiyor** "
        "bölgesinden **🚀 Lider** bölgesine geçiştir: tema henüz endeksi "
        "geçmemiştir ama göreli gücü artmaktadır.")
    with st.spinner("Tema ETF'leri karşılaştırılıyor…"):
        T, TAILS, E, IDX, T_FAILED = load_theme_rotation(
            st.session_state.nonce_funnel)
    if T.empty:
        st.warning("Tema verisi çekilemedi.")
    else:
        top = T.head(3)
        tc = st.columns(3)
        for col, (_, r) in zip(tc, top.iterrows()):
            neden = [r["Bölge"]]
            if r["Taze Lider"]:
                neden.append("liderliğe yeni geçti")
            if r["Hacim Onayı"]:
                neden.append(f"hacim {r['Hacim Oranı']:.1f}×")
            if np.isfinite(r["1H %"]):
                neden.append(f"1H {r['1H %']:+.1f}%")
            col.markdown(kpi(r["Tema"], f"{r['Erken Skor']}/100",
                             " · ".join(neden), "pos"), unsafe_allow_html=True)

        flt = st.radio("Göster", ["🌱 Erken akış", "🚀 Liderler", "Tümü"],
                       horizontal=True, key="fn_tflt")
        view = T
        if flt.startswith("🌱"):
            view = T[(T["_q"] == "iyilesen") | (T["Taze Lider"])]
        elif flt.startswith("🚀"):
            view = T[T["_q"] == "lider"]
        st.dataframe(
            view[["Tema", "Bölge", "Erken Skor", "1H %", "1A %", "3A %",
                  "Hacim Onayı", "RS-Oran", "RS-Mom", "ETF'ler"]],
            width="stretch", hide_index=True,
            column_config={
                "Erken Skor": st.column_config.ProgressColumn(
                    format="%d", min_value=0, max_value=100,
                    help="İyileşen bölge +45, liderliğe taze geçiş +25, göreli "
                         "güç artışı +10, hacim onayı +15, haftalık artı +5"),
                "1H %": st.column_config.NumberColumn(format="%+.1f%%"),
                "1A %": st.column_config.NumberColumn(format="%+.1f%%"),
                "3A %": st.column_config.NumberColumn(format="%+.1f%%"),
                "Hacim Onayı": st.column_config.CheckboxColumn(
                    help="Son 10 günün dolar hacmi önceki 60 günün 1.15 katından "
                         "fazla — fiyat hareketine para eşlik ediyor"),
                "RS-Oran": st.column_config.NumberColumn(format="%.1f"),
                "RS-Mom": st.column_config.NumberColumn(format="%.1f")})

        # ---- RRG grafiği
        pick = st.multiselect("Grafikte göster", list(T["Tema"]),
                              default=list(T["Tema"].head(10)), key="fn_rrg")
        if pick:
            fig = go.Figure()
            for i, tema in enumerate(pick):
                tl = TAILS.get(tema)
                if tl is None or tl.empty:
                    continue
                colr = SERIES[i % len(SERIES)]
                fig.add_trace(go.Scatter(
                    x=tl["RS-Oran"], y=tl["RS-Mom"], mode="lines+markers",
                    line=dict(color=colr, width=1.5), marker=dict(size=4),
                    name=tema, showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(
                    x=[tl["RS-Oran"].iloc[-1]], y=[tl["RS-Mom"].iloc[-1]],
                    mode="markers+text", marker=dict(size=11, color=colr),
                    text=[tema], textposition="top center",
                    textfont=dict(size=11, color=colr), name=tema,
                    hovertemplate=f"{tema}<br>RS-Oran %{{x:.1f}}<br>"
                                  f"RS-Mom %{{y:.1f}}<extra></extra>"))
            fig.add_vline(x=100, line_color="#3a3a46")
            fig.add_hline(y=100, line_color="#3a3a46")
            for x_, y_, t_ in ((1, 1, "🚀 Lider"), (0, 1, "🌱 İyileşiyor"),
                               (0, 0, "🩸 Geride"), (1, 0, "🌤️ Yoruluyor")):
                fig.add_annotation(xref="paper", yref="paper", x=0.02 + 0.96 * x_,
                                   y=0.02 + 0.96 * y_, text=t_, showarrow=False,
                                   xanchor="right" if x_ else "left",
                                   font=dict(size=12, color="#6e6e7a"))
            fig.update_layout(height=520, xaxis_title="RS-Oran (güç)",
                              yaxis_title="RS-Momentum (güçteki değişim)",
                              **CHART_LAYOUT)
            st.plotly_chart(fig, width="stretch")
            st.caption("Çizgiler son 6 haftanın izi, büyük nokta bugün. Saat "
                       "yönünün tersine dönüş (Geride → İyileşiyor → Lider) "
                       "sağlıklı rotasyondur.")
        with st.expander("ETF bazında aynı tablo"):
            if not E.empty:
                st.dataframe(
                    E[["ETF", "Bölge", "Erken Skor", "1H %", "1A %", "3A %",
                       "Hacim Onayı", "RS-Oran", "RS-Mom"]],
                    width="stretch", hide_index=True,
                    column_config={
                        "Erken Skor": st.column_config.ProgressColumn(
                            format="%d", min_value=0, max_value=100),
                        "1H %": st.column_config.NumberColumn(format="%+.1f%%"),
                        "1A %": st.column_config.NumberColumn(format="%+.1f%%"),
                        "3A %": st.column_config.NumberColumn(format="%+.1f%%"),
                        "RS-Oran": st.column_config.NumberColumn(format="%.1f"),
                        "RS-Mom": st.column_config.NumberColumn(format="%.1f")})

    # ------------------------------------------------------------------ 5
    section("Adım 5 · Bu temada hangi hisse?")
    tema_list = list(T["Tema"]) if not T.empty else list(uni.THEME_TRACKER)
    q1, q2 = st.columns([2, 1])
    tema = q1.selectbox("Tema", tema_list, key="fn_tema",
                        help="Liste Adım 4'teki erken skor sırasıyla gelir")
    min_liq = q2.number_input("Min. günlük işlem hacmi ($M)", 0.0, 500.0,
                              10.0, 1.0, key="fn_liq")
    hold = theme_holdings(tema)
    if not hold:
        st.info("Bu temadaki ETF'lerin bileşen listesi tanımlı değil. ETF "
                "Radarı sekmesinde bileşeni olan bir tema seçin.")
    else:
        st.caption(f"{len(hold)} hisse · kaynak: "
                   + ", ".join(f"`{e}`" for e in uni.THEME_TRACKER.get(tema, [])))
        with st.spinner("Tema hisseleri, bilanço tarihleri ve short verisi…"):
            TB, _ = theme_stock_table(tema, min_liq, IDX.get(tema))
        if True:
            if TB.empty:
                st.warning("Tarama sonucu boş.")
            else:
                counts = TB["Durum"].value_counts()
                st.markdown(" ".join(
                    badge(f"{c} · {counts[c]}",
                          "pos" if c.startswith(("🎯", "🌱")) else
                          "neg" if c.startswith(("⛔", "💧")) else "")
                    for c in fnl.CLASS_ORDER if c in counts), unsafe_allow_html=True)
                show = st.multiselect(
                    "Gösterilecek durumlar", [c for c in fnl.CLASS_ORDER
                                              if c in counts],
                    default=[c for c in fnl.CLASS_ORDER[:3] if c in counts]
                    or [c for c in fnl.CLASS_ORDER if c in counts],
                    key=f"fn_show_{tema}")
                vv = TB[TB["Durum"].isin(show)] if show else TB
                full = st.toggle("Bütün sütunları göster", False, key="fn_full")
                base = ["Durum", "Sembol", "Fiyat", "Huni Skoru", "Kalan Gün", "Neden"]
                more = ["Sinyal", "Temaya Göre 1A", "Temaya Göre 1H", "1 Ay %",
                        "Hacim ($M)", "Bilanço", "SV% 5G", "ΔSV pp", "WHALE",
                        "ΔWHALE 5B", "P/S Durum"]
                cols = [c for c in (base[:5] + more + ["Neden"] if full else base)
                        if c in vv.columns]
                st.dataframe(
                    vv[cols].style.map(signal_style, subset=[c for c in ["Sinyal"]
                                                              if c in cols]),
                    width="stretch", hide_index=True,
                    column_config={
                        "Huni Skoru": st.column_config.ProgressColumn(
                            format="%d", min_value=0, max_value=100,
                            help="Swing skoru (Pine sinyalleri) %55 + temaya "
                                 "göre güç + whale yönü − short artışı"),
                        "Fiyat": st.column_config.NumberColumn(format="$%.2f"),
                        "Temaya Göre 1A": st.column_config.NumberColumn(
                            format="%+.1f%%",
                            help="Hissenin 1 aylık getirisi − tema endeksinin"),
                        "Temaya Göre 1H": st.column_config.NumberColumn(
                            format="%+.1f%%"),
                        "1 Ay %": st.column_config.NumberColumn(format="%+.1f%%"),
                        "Hacim ($M)": st.column_config.NumberColumn(
                            "Günlük Ort. İşlem Hacmi ($M)", format="%.1f",
                            help="20 günlük ortalama dolar hacmi"),
                        "Kalan Gün": st.column_config.NumberColumn(
                            "Bilançoya Gün", format="%d"),
                        "SV% 5G": st.column_config.NumberColumn(
                            "Short Hacim %", format="%.1f%%"),
                        "ΔSV pp": st.column_config.NumberColumn(
                            "Short Δ (puan)", format="%+.1f"),
                        "WHALE": st.column_config.ProgressColumn(
                            format="%.0f", min_value=0, max_value=100),
                        "ΔWHALE 5B": st.column_config.NumberColumn(format="%+.1f"),
                        "Neden": st.column_config.TextColumn(width="large")})
                with st.expander("Durumlar ne anlama geliyor?"):
                    for c in fnl.CLASS_ORDER:
                        st.markdown(f"- **{c}** — {fnl.STOCK_CLASSES[c]}")

                # ---- tek paragraf sonuç
                al = TB[TB["Durum"] == "🎯 Alım adayı"]["Sembol"].head(5).tolist()
                ger = TB[TB["Durum"] == "🌱 Geride kaldı, toparlanıyor"][
                    "Sembol"].head(5).tolist()
                tr = T[T["Tema"] == tema].iloc[0] if not T.empty and (
                    T["Tema"] == tema).any() else None
                parca = [f"**Piyasa:** {V['etiket']}"]
                if tr is not None:
                    parca.append(f"**Tema ({tema}):** {tr['Bölge']}, erken skor "
                                 f"{tr['Erken Skor']}/100")
                parca.append("**Alım adayları:** "
                             + (", ".join(al) if al else "yok"))
                parca.append("**Toparlanan geride kalanlar:** "
                             + (", ".join(ger) if ger else "yok"))
                st.success(" · ".join(parca))


# ==========================================================================
# HİSSE ANALİZİ — tek hisse değerleme grafikleri
# ==========================================================================
with tab_stock:
    st.markdown(
        "Bir hisse seçin; **F/K geçmişini akranlarıyla** ve **fiyatı, "
        "hissenin kendi medyan çarpanından türetilen adil fiyat çizgisiyle** "
        "karşılaştırır. Fiyat yeşil çizginin çok üstündeyse hisse kendi "
        "tarihine göre pahalı, altındaysa ucuzdur.")
    a1, a2, a3 = st.columns([1, 1, 1])
    sym = a1.text_input("Sembol", value=st.session_state.get("sa_sym", "AVGO"),
                        key="sa_sym_in").strip().upper()
    basis = a2.radio("HBK tabanı", ["GAAP", "Düzeltilmiş"], horizontal=True,
                     key="sa_basis",
                     help="GAAP: resmi mali tablolardaki seyreltilmiş HBK "
                          "(yaklaşık 5 yıl). Düzeltilmiş: şirketin açıkladığı "
                          "non-GAAP HBK (yaklaşık 10 yıl, genelde daha yüksek, "
                          "F/K daha düşük çıkar).")
    if a3.button("🔄 Veriyi yenile", key="sa_ref", width="stretch"):
        bump("nonce_stock")
        st.rerun()
    st.session_state["sa_sym"] = sym

    stock_fundamentals(sym, basis)


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
                   "göstergelerden hesaplanır. Seviyenin yanındaki fark, aynı "
                   "formülün 1 gün / 1 hafta / 1 ay önceki veriyle yeniden "
                   "çalıştırılmasıyla bulunur.")

        manual_on = bool(st.session_state.manual_scenario)
        bat = (mac.MANUAL_SCENARIOS[st.session_state.manual_scenario]["battery"]
               if manual_on else M.battery)

        with st.spinner("Geçmiş rejim yeniden hesaplanıyor…"):
            BC, SC, BH, SHIFTS = macro_history(st.session_state.nonce_macro)

        delta_period = st.radio(
            "Karşılaştırma dönemi", list(mac.PERIOD_BARS),
            horizontal=True, index=1, key="bat_period",
            help="Bataryanın bu dönem önceki değerine göre farkı gösterilir.")
        dcol = f"Δ {delta_period}"
        deltas = ({} if manual_on else
                  {r["Varlık Sınıfı"]: r.get(dcol) for _, r in BC.iterrows()})

        keys = list(bat)
        prev_vals = [bat[k] - (deltas.get(k) or 0) for k in keys]
        bar = go.Figure()
        # Önceki seviye soluk gölge olarak arkada durur
        if not manual_on:
            bar.add_trace(go.Bar(
                x=prev_vals, y=keys, orientation="h", name=f"{delta_period} önce",
                marker=dict(color="rgba(255,255,255,.10)", cornerradius=4),
                hovertemplate="%{y}: %{x:.0f}/100 (" + delta_period
                              + " önce)<extra></extra>"))
        bar.add_trace(go.Bar(
            x=[bat[k] for k in keys], y=keys, orientation="h", name="şimdi",
            marker=dict(color=[SERIES[i % len(SERIES)] for i in range(len(keys))],
                        cornerradius=4),
            text=[f"{bat[k]}" + ("" if manual_on or deltas.get(k) is None
                                 or not np.isfinite(deltas.get(k, np.nan))
                                 else f"   {deltas[k]:+.0f}")
                  for k in keys],
            textposition="outside",
            hovertemplate="%{y}: %{x}/100<extra></extra>"))
        bar.update_layout(height=300, bargap=0.30, barmode="overlay",
                          showlegend=not manual_on,
                          legend=dict(orientation="h", y=1.18, x=0),
                          xaxis=dict(range=[0, 118], showgrid=False,
                                     showticklabels=False),
                          yaxis=dict(showgrid=False), **CHART_LAYOUT)
        st.plotly_chart(bar, width="stretch")

        if manual_on:
            st.caption("Elle senaryo seçiliyken geçmiş karşılaştırma kapalıdır — "
                       "senaryo sabitlerinin tarihçesi yoktur.")
        else:
            st.dataframe(
                BC, width="stretch", hide_index=True,
                column_config={
                    "Şimdi": st.column_config.ProgressColumn(
                        format="%.0f", min_value=0, max_value=100),
                    "1 gün önce": st.column_config.NumberColumn(format="%.0f"),
                    "1 hafta önce": st.column_config.NumberColumn(format="%.0f"),
                    "1 ay önce": st.column_config.NumberColumn(format="%.0f"),
                    "Δ 1 gün": st.column_config.NumberColumn(format="%+.0f"),
                    "Δ 1 hafta": st.column_config.NumberColumn(format="%+.0f"),
                    "Δ 1 ay": st.column_config.NumberColumn(format="%+.0f"),
                    "Besleyen": st.column_config.TextColumn(width="large"),
                })

            if not BH.empty:
                line = go.Figure()
                for i, k in enumerate(keys):
                    if k in BH.columns:
                        line.add_trace(go.Scatter(
                            x=BH.index, y=BH[k], name=k, mode="lines",
                            line=dict(color=SERIES[i % len(SERIES)], width=2)))
                line.add_trace(go.Scatter(
                    x=BH.index, y=BH["Risk Skoru"], name="Bileşik risk",
                    mode="lines", line=dict(color="#f2f2f6", width=2.5,
                                            dash="dot")))
                line.add_hline(y=50, line=dict(color="#3a3a48", width=1,
                                               dash="dot"))
                line.update_layout(height=330, **CHART_LAYOUT,
                                   yaxis=dict(range=[0, 100],
                                              gridcolor="#1b1b22"),
                                   xaxis=dict(gridcolor="#1b1b22"),
                                   legend=dict(orientation="h", y=1.15, x=0))
                st.plotly_chart(line, width="stretch")
                st.caption(f"Son {len(BH)} işlem günü. Her gün, bugünkü formülle "
                           "yeniden hesaplanır — formül değişse bile geçmiş "
                           "tutarlı kalır. 50 çizgisi nötr seviyedir.")

            if SHIFTS:
                st.markdown("**Bu pencerede rejim kaç kez değişti**")
                sh = pd.DataFrame(SHIFTS[-6:])
                sh["Tarih"] = pd.to_datetime(sh["Tarih"]).dt.strftime("%d.%m.%Y")
                st.dataframe(sh, width="stretch", hide_index=True)
                st.caption("Rejimin sık değişmesi kararsız piyasa demektir; "
                           "trend takip sistemleri bu pencerelerde kötü çalışır.")

        section("Rejimi ne itiyor, ne çekiyor")
        st.caption("Alt skorların dönemsel değişimi. Bileşik risk skoru bunların "
                   "ortalamasıdır; hangi bileşenin rejimi taşıdığı buradan "
                   "okunur.")
        st.dataframe(
            SC, width="stretch", hide_index=True,
            column_config={
                "Şimdi": st.column_config.ProgressColumn(
                    format="%.0f", min_value=0, max_value=100),
                "Δ 1 gün": st.column_config.NumberColumn(format="%+.1f"),
                "Δ 1 hafta": st.column_config.NumberColumn(format="%+.1f"),
                "Δ 1 ay": st.column_config.NumberColumn(format="%+.1f"),
                "Ne ölçüyor": st.column_config.TextColumn(width="large"),
            })

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

        st.session_state["rep_news"] = pd.DataFrame(enriched)
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
        st.session_state["rep_theme"] = (TQ, period)
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
        labels, texts, bar_renk = [], [], []
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
            bar_renk.append("#3987e5" if val >= 0 else "#d55181")

        fig = go.Figure(go.Bar(
            y=labels, x=srt[period], orientation="h",
            marker=dict(color=bar_renk, cornerradius=3),
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
                "ΔPRO-RET", "OMNI", "ΔOMNI", "OMNI Yön", "Boğa /6", "Ayı /6",
                "MAGNITUDE", "ΔMAG", "DIRECTION", "ΔDIR", "Hata"]
        cols = [c for c in cols if c in E.columns]
        st.session_state["rep_etf"] = E[cols]
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
                    anlati = hld.holdings_narrative(HT, sel, period_col)
                    st.session_state["rep_hold"] = {
                        "etf": sel, "period": period_col,
                        "table": HT, "narrative": anlati}
                    for line in anlati:
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
        bump("nonce_short")
        st.rerun()
    universe = uni.all_stocks()
    W = scan_gate("weekly", universe, "1wk", "Haftalık momentumu tara")
    if not W.empty:
        W = W.copy()
        # Haftalık barlarda motorun "1 Gün %" sütunu = 1 bar = 1 HAFTA,
        # "1 Hafta %" sütunu = 5 bar = 5 HAFTA. Etiketler buna göre düzeltildi.
        W = W.rename(columns={"1 Gün %": "Haftalık %", "1 Hafta %": "5 Hafta %"})

        # ---------- FINRA short hacmi ----------
        section("📉 Short verisi")
        with st.spinner("FINRA short hacim dosyaları çekiliyor…"):
            SVRAW, sv_warn, sv_diag = load_short_volume(
                st.session_state.nonce_short)
        # Short interest her sembol için ayrı Yahoo isteği demek; yüzlerce
        # sembolde Yahoo geçici blok uygulayabildiği için isteğe bağlı.
        si_on = st.toggle(
            "Short interest (yfinance) sütunlarını ekle — yavaş, ilk "
            "çekimde 1–3 dk", key="si_on")
        SI = pd.DataFrame(columns=["Sembol"])
        if si_on:
            with st.spinner("Short interest (yfinance) çekiliyor…"):
                SI = load_short_interest(tuple(W["Sembol"].tolist()),
                                         st.session_state.nonce_short)
        n_ok = sum(1 for m in sv_diag.values() if m.startswith("ok"))
        n_repo = sum(1 for m in sv_diag.values() if "repo" in m)
        codes = sorted({m for m in sv_diag.values()
                        if not m.startswith("ok") and not m.startswith("HTTP 404")})
        pchg = dict(zip(W["Sembol"],
                        pd.to_numeric(W.get("Haftalık %"), errors="coerce")))
        SVT = svm.short_table(SVRAW, W["Sembol"].tolist(), pchg)
        n_match = int(SVT["ΔSV pp"].notna().sum()) if not SVT.empty else 0
        n_si = (int(SI["SI Δ% (ay)"].notna().sum())
                if "SI Δ% (ay)" in SI.columns else 0)
        durum = (f"**FINRA günlük short hacmi:** "
                 + (f"✅ {n_ok} gün ({n_repo} gün repodan), {n_match}/{len(W)} sembol eşleşti"
                    if n_ok else f"❌ alınamadı ({', '.join(codes) or 'yanıt yok'})")
                 + "  \n**yfinance short interest:** "
                 + (f"✅ {n_si}/{len(W)} sembol" if si_on and n_si
                    else "⏳ açık ama veri gelmedi — birkaç dk sonra yenileyin"
                    if si_on else "kapalı — yukarıdaki anahtarla açın"))
        (st.success if (n_match or n_si) else st.warning)(durum)
        for w_ in sv_warn:
            st.caption(f"⚠️ {w_}")
        W = W.merge(SVT, on="Sembol", how="left")
        # eşleşmeyen satırlarda NaN kalırsa çizgi grafik sütunu hata verir
        W["SV Seyri"] = W["SV Seyri"].apply(
            lambda x: x if isinstance(x, list) else None)
        W["SV Yön"] = W["SV Yön"].fillna("—")
        if not SI.empty:
            W = W.merge(SI, on="Sembol", how="left")

        # FINRA günlük verisi yoksa yorum, aylık short interest değişiminden
        def _si_yorum(r):
            d, p = r.get("SI Δ% (ay)"), r.get("Haftalık %")
            if pd.isna(d):
                return "Veri yok (OTC/kripto)"
            yon = ("🟠 Short interest artıyor" if d >= 10
                   else "🟢 Short interest azalıyor" if d <= -10
                   else "⚪ Short interest yatay")
            return f"{yon} ({d:+.0f}% / ay)"
        miss = W["Short Yorum"].isna()
        if "SI Δ% (ay)" in W.columns and miss.any():
            W.loc[miss, "Short Yorum"] = [_si_yorum(r) for r in
                                          W[miss].to_dict("records")]
        W["Short Yorum"] = W["Short Yorum"].fillna("Veri yok (OTC/kripto)")

        sv_filter = st.radio(
            "Short filtresi",
            ["Tümü", "⬆️ Short artanlar", "⬇️ Short azalanlar",
             "Aşırı uç (|Z| ≥ 2)"],
            horizontal=True, key="sv_filter")
        # FINRA günlük verisi varsa ΔSV pp, yoksa aylık SI değişimi kullanılır
        have_sv = W["ΔSV pp"].notna().any()
        fkey, fthr = (("ΔSV pp", svm.ESIK_PP) if have_sv
                      else ("SI Δ% (ay)", 10.0))
        if fkey not in W.columns:
            W[fkey] = np.nan
        view = W
        if sv_filter.startswith("⬆️"):
            view = W[W[fkey] >= fthr].sort_values(fkey, ascending=False)
        elif sv_filter.startswith("⬇️"):
            view = W[W[fkey] <= -fthr].sort_values(fkey)
        elif sv_filter.startswith("Aşırı"):
            view = W[W["SV% Z"].abs() >= 2].sort_values("SV% Z",
                                                         ascending=False)

        cols = [c for c in ["Sembol", "Sinyal", "Efor", "Fiyat", "Haftalık %",
                            "5 Hafta %", "WHALE", "ΔWHALE", "ΔWHALE 5B",
                            "Whale Yön", "SV% 5G", "ΔSV pp", "Short Hacim Δ%",
                            "SV% Z", "SV Yön", "SV Seyri", "SI Float %",
                            "SI Δ% (ay)", "Gün Kapama", "Short Yorum",
                            "PRO-RET", "ΔPRO-RET", "MAGNITUDE", "ΔMAG",
                            "DIRECTION", "ΔDIR", "OMNI", "ΔOMNI", "OMNI Yön",
                            "SV Son Gün", "Hata"] if c in view.columns]
        st.dataframe(
            view[cols].style.map(signal_style, subset=["Sinyal"]),
            width="stretch", hide_index=True,
            column_config={
                "Fiyat": st.column_config.NumberColumn(format="$%.2f"),
                "Haftalık %": st.column_config.NumberColumn(
                    format="%+.2f%%", help="Son haftalık barın değişimi"),
                "5 Hafta %": st.column_config.NumberColumn(format="%+.2f%%"),
                "WHALE": st.column_config.ProgressColumn(
                    format="%.0f", min_value=0, max_value=100),
                "ΔWHALE": st.column_config.NumberColumn(format="%+.1f"),
                "ΔWHALE 5B": st.column_config.NumberColumn(
                    format="%+.1f", help="5 bar önceki değere göre değişim"),
                "Whale Yön": st.column_config.TextColumn(width="small"),
                "SV% 5G": st.column_config.NumberColumn(
                    format="%.1f%%",
                    help="Son 5 işlem gününde short hacmin toplam hacme oranı "
                         "(hacim ağırlıklı)"),
                "ΔSV pp": st.column_config.NumberColumn(
                    format="%+.1f",
                    help="Son 5 gün oranı − önceki 5 gün oranı (yüzde puan)"),
                "Short Hacim Δ%": st.column_config.NumberColumn(
                    format="%+.0f%%",
                    help="Short hisse adedinin önceki 5 güne göre değişimi"),
                "SV% Z": st.column_config.NumberColumn(
                    format="%+.1f",
                    help="Son günün oranı, hissenin kendi 20 günlük "
                         "ortalamasından kaç σ uzakta"),
                "SV Yön": st.column_config.TextColumn(width="small"),
                "SV Seyri": st.column_config.LineChartColumn(
                    "SV% Seyri", y_min=0, y_max=100, width="small",
                    help="Son 15 işlem günü günlük short hacim oranı"),
                "SI Float %": st.column_config.NumberColumn(
                    format="%.1f%%",
                    help="Açık short pozisyonun halka açık hisseye oranı "
                         "(borsa verisi, ayda iki kez güncellenir)"),
                "SI Δ% (ay)": st.column_config.NumberColumn(
                    format="%+.0f%%",
                    help="Açık short pozisyonun bir önceki aya göre değişimi"),
                "Gün Kapama": st.column_config.NumberColumn(
                    format="%.1f",
                    help="Short'ların ortalama hacimle kaç günde kapanabileceği"),
                "Short Yorum": st.column_config.TextColumn(width="large"),
                "ΔPRO-RET": st.column_config.NumberColumn(format="%+.1f"),
                "ΔMAG": st.column_config.NumberColumn(format="%+.0f"),
                "ΔDIR": st.column_config.NumberColumn(format="%+.0f"),
                "ΔOMNI": st.column_config.NumberColumn(format="%+.1f"),
                "OMNI Yön": st.column_config.TextColumn(width="small")})
        st.caption("Haftalık barlarda Δ, bir hafta önceki değere göre "
                   "değişimdir; Δ5B ise beş hafta öncesine göre. Kurumsal "
                   "toplama (WHALE) yükselirken fiyatın yatay kalması, "
                   "scriptlerdeki 'stealth accumulation' durumudur.")

        # ---------- en büyük değişimler ----------
        section("Short hacminde en büyük haftalık değişimler")
        sv_ok = W.dropna(subset=[fkey])
        if sv_ok.empty:
            st.info("Short verisi eşleşmedi.")
        else:
            mini = (["Sembol", "SV% 5G", "ΔSV pp", "Haftalık %", "WHALE",
                     "Short Yorum"] if have_sv else
                    ["Sembol", "SI Float %", "SI Δ% (ay)", "Gün Kapama",
                     "Haftalık %", "WHALE", "Short Yorum"])
            mini = [c for c in mini if c in sv_ok.columns]
            mcfg = {"SV% 5G": st.column_config.NumberColumn(format="%.1f%%"),
                    "ΔSV pp": st.column_config.NumberColumn(format="%+.1f"),
                    "Haftalık %": st.column_config.NumberColumn(format="%+.2f%%"),
                    "WHALE": st.column_config.NumberColumn(format="%.0f"),
                    "SI Float %": st.column_config.NumberColumn(format="%.1f%%"),
                    "SI Δ% (ay)": st.column_config.NumberColumn(format="%+.0f%%"),
                    "Gün Kapama": st.column_config.NumberColumn(format="%.1f")}
            a, b = st.columns(2)
            a.markdown("**⬆️ Short en çok artan 10**")
            a.dataframe(sv_ok.nlargest(10, fkey)[mini], width="stretch",
                        hide_index=True, column_config=mcfg)
            b.markdown("**⬇️ Short en çok azalan 10**")
            b.dataframe(sv_ok.nsmallest(10, fkey)[mini], width="stretch",
                        hide_index=True, column_config=mcfg)

        with st.expander("Short hacmi nasıl okunur?"):
            st.markdown(f"""
- **Kaynak:** FINRA Reg SHO günlük dosyası (ücretsiz, her iş günü ABD
  akşamı yayımlanır). Son veri: **{SVT['SV Son Gün'].dropna().max()
  if not SVT.empty else '—'}**.
- **Short HACMİ ≠ short INTEREST.** Piyasa yapıcıların alıcıya hisse
  sağlamak için yaptığı gün içi açığa satışlar da buraya girer; çoğu
  hissede oran zaten %35–55'tir. Bu yüzden **seviyeye değil değişime**
  bakın: ΔSV pp (son 5 gün − önceki 5 gün) ve Z (hissenin kendi 20
  günlük ortalamasından sapma).
- Anlamlı değişim eşiği **±{svm.ESIK_PP:.0f} puan**; |Z| ≥ 2 aşırı uç.
- **Okuma matrisi:** short ↑ + fiyat ↓ = baskı · short ↑ + fiyat ↑ =
  dağıtım ya da squeeze yakıtı (WHALE'e bakın: WHALE yükseliyorsa squeeze
  tarafı) · short ↓ + fiyat ↑ = kapama desteği · short ↓ + fiyat ↓ =
  satış uzun pozisyonlardan geliyor.
- OTC semboller (FANUY, YASKY…) ve kripto FINRA konsolide dosyasında
  olmadığı için boş kalır.
- **SI sütunları** (yfinance): borsanın ayda iki kez yayımladığı *açık*
  short pozisyon. Daha yavaş ama daha anlamlıdır: SI Float % %10'un,
  Gün Kapama 5'in üstündeyse squeeze potansiyeli yüksektir. FINRA
  verisi çekilemezse filtreler ve listeler SI Δ% (ay) ile çalışır.
""")
            st.markdown("**FINRA bağlantı tanısı** (gün → sonuç)")
            st.dataframe(pd.DataFrame({"Gün": list(sv_diag),
                                       "Sonuç": list(sv_diag.values())}),
                         hide_index=True, width="stretch", height=220)


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
            st.session_state["rep_swing"] = RECS

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

    with st.spinner("Bilanço tarihleri, analist hedefleri ve satış verisi çekiliyor…"):
        EARN = load_earnings(tuple(lst), st.session_state.nonce_earn)
    if lst and _earn_ok_ratio(EARN) < 0.5:
        st.warning("Yahoo Finance şu an istekleri sınırlıyor (bulut "
                   "sunucularında sık olur); verilerin çoğu boş geldi. Birkaç "
                   "dakika bekleyip **🔄 Bilanço takvimini tara** düğmesine "
                   "basın — boş sonuç önbelleğe alınmaz.")

    if not EARN.empty:
        # ---------- P/S adil değer ayarları ----------
        wlst = st.session_state.watchlist
        saved_mult = wlst.get("ps_multiples")
        ind_map = ({k: (None if v is None else float(v))
                    for k, v in saved_mult.items()}
                   if isinstance(saved_mult, dict) and saved_mult
                   else dict(fvm.INDUSTRY_PS))

        with st.expander("⚙️ P/S adil değer ayarları (çarpan tablosu, bant, "
                         "pahalı eşiği)"):
            st.markdown(
                "**Yöntem:** Adil orta = *endüstri çarpanı × gelecek yıl "
                "satış/hisse*; adil bant = orta ± bant. **PSG** = (fiyat / "
                "TTM satış-hisse) ÷ büyüme %. **Pahalı eşiği**, PSG'nin "
                "aşağıdaki değere ulaştığı fiyattır; ancak adil ortanın "
                "belirlenen katını geçemez. Bant üstünde ama pahalı "
                "eşiğinin altındaysa hisse *büyüme destekli adil* sayılır.")
            p1, p2, p3 = st.columns(3)
            band = p1.slider("Adil bant ±%", 5, 30,
                             int(fvm.DEFAULT_BAND * 100), 1) / 100
            psg_max = p2.slider("Pahalı eşiği (PSG)", 0.30, 1.50,
                                fvm.DEFAULT_PSG_MAX, 0.05)
            exp_cap = p3.slider(
                "Pahalı eşiği üst sınırı (× adil orta)", 1.2, 3.0,
                fvm.DEFAULT_EXP_CAP, 0.1,
                help="Büyüme ne kadar yüksek olursa olsun pahalı eşiği adil "
                     "ortanın bu katını geçmez.")
            st.caption("Çarpan tablosu endüstri adında anahtar kelime "
                       "arar; ilk eşleşen kazanır. Boş bırakılan çarpan = "
                       "P/S bu iş modelinde anlamsız (banka, sigorta, petrol, "
                       "GYO). Eşleşmeyen endüstri sektör varsayılanına düşer.")
            med = st.data_editor(
                pd.DataFrame({"Endüstri anahtarı": list(ind_map),
                              "İleri P/S çarpanı": list(ind_map.values())}),
                num_rows="dynamic", hide_index=True, width="stretch",
                key="ps_mult_editor",
                column_config={"İleri P/S çarpanı": st.column_config.NumberColumn(
                    min_value=0.1, max_value=60.0, step=0.5, format="%.1fx")})
            ind_map = {str(r["Endüstri anahtarı"]).strip():
                       (None if pd.isna(r["İleri P/S çarpanı"])
                        else float(r["İleri P/S çarpanı"]))
                       for _, r in med.iterrows()
                       if str(r["Endüstri anahtarı"]).strip()
                       and str(r["Endüstri anahtarı"]) != "nan"}
            s1, s2 = st.columns(2)
            if s1.button("💾 Çarpanları kaydet", width="stretch"):
                wlst["ps_multiples"] = ind_map
                if save_watchlist():
                    st.success("Çarpan tablosu kaydedildi.")
            if s2.button("↩️ Varsayılana dön", width="stretch"):
                wlst.pop("ps_multiples", None)
                save_watchlist()
                st.rerun()

        EARN = fvm.add_fair_values(EARN, ind_map=ind_map, band=band,
                                   psg_max=psg_max, exp_cap=exp_cap)

        # İki kaynağın (analist hedefi + P/S orta) aynı yönü gösterip
        # göstermediği — tek başına hiçbiri değerleme modeli değildir.
        def _uyum(r):
            p, h, m = r.get("Fiyat"), r.get("Hedef"), r.get("Adil Orta")
            if not p or pd.isna(h) or pd.isna(m):
                return "—"
            if h > p and m > p:
                return "🟢 İkisi de yukarı"
            if h < p and m < p:
                return "🔴 İkisi de aşağı"
            return "🟡 Ayrışıyor"
        EARN["Kaynak Uyumu"] = [_uyum(r) for r in EARN.to_dict("records")]

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

        def style_ps(v):
            if not isinstance(v, str):
                return ""
            if v.startswith("🟢"):
                return "background-color:#0d3b2a;color:#4fd6a0;font-weight:700"
            if v.startswith("🔴"):
                return "background-color:#4a0d12;color:#ff8f8b;font-weight:700"
            if v.startswith("🟡"):
                return "background-color:#3a3205;color:#ffd54f;font-weight:700"
            return "color:#8a8a95"

        # Uzaklık: fiyat adil ortanın üstündeyse kırmızı, altındaysa yeşil
        def style_dist(v):
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                return ""
            return ("color:#f0736f;font-weight:700" if v > 0
                    else "color:#2fbe86;font-weight:700")

        ecols = [c for c in [
            "Hisse", "Bilanço", "Kalan Gün", "Fiyat", "P/S Durum", "Adil Alt",
            "Adil Orta", "Adil Üst", "Pahalı >", "Adile Uzaklık %", "Hedef",
            "Potansiyel %", "Kaynak Uyumu", "P/S", "İleri P/S", "PSG",
            "Büyüme %", "Çarpan", "Çarpan Kaynağı", "EV/Satış", "İleri F/K",
            "Piyasa Değ. ($B)", "52H Konum %", "Analist"] if c in EARN.columns]
        st.dataframe(
            EARN[ecols].style.map(style_days, subset=["Kalan Gün"])
                             .map(style_pot, subset=["Potansiyel %"])
                             .map(style_ps, subset=["P/S Durum"])
                             .map(style_dist, subset=["Adile Uzaklık %"]),
            width="stretch", hide_index=True,
            column_config={
                "Fiyat": st.column_config.NumberColumn(format="$%.2f"),
                "Hedef": st.column_config.NumberColumn(
                    format="$%.2f", help="Analist ortalama hedef fiyatı"),
                "Potansiyel %": st.column_config.NumberColumn(format="%+.1f%%"),
                "Kalan Gün": st.column_config.NumberColumn(format="%d"),
                "P/S Durum": st.column_config.TextColumn(width="medium"),
                "Adil Alt": st.column_config.NumberColumn(format="$%.2f"),
                "Adil Orta": st.column_config.NumberColumn(format="$%.2f"),
                "Adil Üst": st.column_config.NumberColumn(format="$%.2f"),
                "Pahalı >": st.column_config.NumberColumn(
                    format="$%.2f",
                    help="Bu fiyatın üstü satışlara göre pahalı. Büyüme "
                         "yüksekse esner, en fazla adil ortanın 1.8 katı "
                         "(ayarlardan değişir)."),
                "Adile Uzaklık %": st.column_config.NumberColumn(
                    format="%+.1f%%", help="Fiyatın adil ortaya göre konumu"),
                "P/S": st.column_config.NumberColumn(format="%.1fx"),
                "İleri P/S": st.column_config.NumberColumn(format="%.1fx"),
                "PSG": st.column_config.NumberColumn(format="%.2f"),
                "Büyüme %": st.column_config.NumberColumn(format="%.0f%%"),
                "Çarpan": st.column_config.NumberColumn(format="%.1fx"),
                "EV/Satış": st.column_config.NumberColumn(format="%.1fx"),
                "İleri F/K": st.column_config.NumberColumn(format="%.1fx"),
                "Piyasa Değ. ($B)": st.column_config.NumberColumn(format="%.1f"),
                "52H Konum %": st.column_config.ProgressColumn(
                    format="%.0f%%", min_value=0, max_value=100),
            })
        st.caption("Kırmızı satırlar bilanço tamponu içindedir — swing "
                   "taramasında bu hisseler otomatik elenir. **Hedef** analist "
                   "ortalamasıdır; **P/S adil değer** ise satış çarpanından "
                   "türetilen ikinci, bağımsız bir referanstır. İkisi de "
                   "değerleme modeli değil, bağlam göstergesidir — "
                   "**Kaynak Uyumu** ikisinin aynı yönü gösterip "
                   "göstermediğini özetler.")

        section("Hızlı bakış")
        pick = st.selectbox("Hisse", EARN["Hisse"].tolist(), key="glance_pick")
        if pick:
            r = EARN[EARN["Hisse"] == pick].iloc[0].to_dict()
            st.markdown(fvm.glance_text(r))
            if r.get("Satış Tahmin Kaynağı"):
                st.caption(f"Satış tahmini: {r['Satış Tahmin Kaynağı']} · "
                           f"endüstri: {r.get('Endüstri') or '—'}")


# ==========================================================================
# 9) PDF RAPOR
# ==========================================================================
with tab_report:
    st.markdown(
        "Ekrandaki tüm hesaplamaları tek bir PDF'e toplar: makro rejim ve "
        "sermaye akışı (geçmiş karşılaştırmasıyla), rejim oyun kitabı, tema "
        "haritası, ETF radarı, seçili ETF'in iç röntgeni, swing adayları ve "
        "haber akışı. Grafikler rapora yeniden çizilir — ekran görüntüsü değil, "
        "baskıya uygun vektörel sayfa düzeni üretilir.")

    have_theme = "rep_theme" in st.session_state
    have_etf = "rep_etf" in st.session_state
    have_hold = "rep_hold" in st.session_state
    have_swing = "rep_swing" in st.session_state
    have_news = "rep_news" in st.session_state

    section("Rapora nelerin gireceğini seçin")
    st.caption("Bir bölüm soluksa, o sekmedeki tarama henüz çalıştırılmamıştır. "
               "İlgili sekmeye gidip taramayı başlatın, sonra buraya dönün — "
               "sonuçlar oturumda saklanır.")

    r1, r2, r3 = st.columns(3)
    inc_macro = r1.checkbox("Makro & sermaye akışı", value=True)
    inc_hist = r1.checkbox("Geçmiş seyir grafiği ve rejim değişimleri", value=True)
    inc_play = r1.checkbox("Rejim oyun kitabı", value=True)
    inc_theme = r2.checkbox(
        f"Tema takibi{'' if have_theme else '  (tarama yok)'}",
        value=have_theme, disabled=not have_theme)
    inc_etf = r2.checkbox(
        f"ETF radarı{'' if have_etf else '  (tarama yok)'}",
        value=have_etf, disabled=not have_etf)
    inc_hold = r2.checkbox(
        (f"ETF içi röntgen — {st.session_state['rep_hold']['etf']}"
         if have_hold else "ETF içi röntgen  (tarama yok)"),
        value=have_hold, disabled=not have_hold)
    inc_swing = r3.checkbox(
        f"Swing adayları{'' if have_swing else '  (tarama yok)'}",
        value=have_swing, disabled=not have_swing)
    inc_news = r3.checkbox(
        f"Haber akışı{'' if have_news else '  (yüklenmedi)'}",
        value=have_news, disabled=not have_news)
    delta_p = r3.selectbox("Karşılaştırma dönemi", list(mac.PERIOD_BARS),
                           index=1, key="rep_delta")

    baslik = st.text_input("Rapor başlığı", value="AETHER APEX")

    if st.button("📄 PDF raporu oluştur", type="primary", width="stretch"):
        with st.spinner("Rapor hazırlanıyor — grafikler çiziliyor…"):
            try:
                BC = SC = BH = None
                SHIFTS: list = []
                if inc_macro or inc_hist:
                    BC, SC, BH, SHIFTS = macro_history(
                        st.session_state.nonce_macro)
                theme_tbl, theme_per = (st.session_state["rep_theme"]
                                        if (inc_theme and have_theme)
                                        else (None, ""))
                pdf_bytes = rep.build_report(
                    M,
                    battery_changes=BC if inc_macro else None,
                    score_changes=SC if inc_macro else None,
                    battery_history=BH if inc_hist else None,
                    regime_shifts=SHIFTS if inc_hist else [],
                    drivers=pb.drivers_for(M.regime) if inc_play else (),
                    theme_table=theme_tbl, theme_period=theme_per,
                    etf_table=(st.session_state["rep_etf"]
                               if (inc_etf and have_etf) else None),
                    holdings=(st.session_state["rep_hold"]
                              if (inc_hold and have_hold) else None),
                    swing=(st.session_state["rep_swing"]
                           if (inc_swing and have_swing) else None),
                    news=(st.session_state["rep_news"]
                          if (inc_news and have_news) else None),
                    delta_period=delta_p, baslik=baslik or "AETHER APEX")
                st.session_state["rep_pdf"] = pdf_bytes
                st.session_state["rep_pdf_name"] = rep.report_filename()
            except Exception as exc:                      # pragma: no cover
                st.session_state.pop("rep_pdf", None)
                st.error(f"Rapor üretilemedi: {exc}")
                logging.exception("PDF hatası")

    if st.session_state.get("rep_pdf"):
        boyut = len(st.session_state["rep_pdf"]) / 1024
        st.success(f"Rapor hazır — {boyut:.0f} KB. Aşağıdaki butonla indirin.")
        st.download_button(
            "⬇️ PDF'i indir", data=st.session_state["rep_pdf"],
            file_name=st.session_state.get("rep_pdf_name", "aether_apex.pdf"),
            mime="application/pdf", width="stretch", type="primary")
        if not rep.ensure_fonts():
            st.warning(
                "Sunucuda Türkçe karakter içeren bir TTF font bulunamadı; "
                "raporda harfler ASCII'ye çevrildi (ş→s, ğ→g). "
                "`requirements.txt` içinde `matplotlib` olduğundan emin olun — "
                "DejaVu fontu onunla birlikte gelir.", icon="ℹ️")

    with st.expander("Raporda ne var, nasıl okunur?"):
        st.markdown("""
| Bölüm | İçerik |
|---|---|
| **Kapak** | Tespit edilen rejim, bileşik risk skoru, VIX, rejim kapısı, OPEX/FOMC takvimi |
| **1. Makro göstergeler** | Tüm canlı okumalar ve yorumları; alt skorların 1 gün/1 hafta/1 ay değişimi (grafik + tablo) |
| **2. Sermaye akış eğilimi** | Varlık sınıfı bataryası, seçilen döneme göre farkı, 60 günlük seyir grafiği ve rejim değişim anları |
| **3. Oyun kitabı** | Aktif rejimi besleyen sürücüler; her biri için lehte/aleyhte ETF ve hisse listesi, işlem notu |
| **4. Tema takibi** | Momentum × ivme haritası ve çeyrek sınıflandırmalı tema tablosu |
| **5. ETF radarı** | Taranan ETF'lerin sinyal tablosu |
| **6. ETF içi röntgen** | Seçili ETF'in bileşen bazında göreli gücü, ayrışma haritası ve her hissenin teknik durumu |
| **7. Swing adayları** | Skor, karakter, stop/hedef, R katsayıları ve engel gerekçeleri |
| **8. Haber akışı** | Rejim sürücüsüne bağlanmış başlıklar |

Rapor açık zeminlidir; ekrandaki koyu tema baskıda mürekkep yakar ve telefonda
okunmaz. Grafikler PDF içine gömülü PNG olarak girer, metin ise gerçek metindir —
arama yapılabilir ve kopyalanabilir.
""")
