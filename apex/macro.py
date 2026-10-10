# apex/macro.py — AETHER APEX
from __future__ import annotations

import calendar
import datetime as dt
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Makro göstergelerin Yahoo sembolleri
MACRO_TICKERS: dict[str, str] = {
    "SPY": "SPY",          # geniş piyasa
    "QQQ": "QQQ",          # teknoloji
    "VIX": "^VIX",         # 30 günlük örtük oynaklık
    "VIX3M": "^VIX3M",     # 3 aylık — vade yapısı için
    "TLT": "TLT",          # uzun vadeli hazine
    "HYG": "HYG",          # yüksek getirili şirket tahvili
    "LQD": "LQD",          # yatırım yapılabilir tahvil
    "DXY": "DX-Y.NYB",     # dolar endeksi
    "GOLD": "GC=F",        # altın
    "COPPER": "HG=F",      # bakır
    "OIL": "CL=F",         # ham petrol
    "BTC": "BTC-USD",      # kripto risk iştahı
    "US10Y": "^TNX",       # 10 yıllık tahvil faizi
    "US3M": "^IRX",        # 13 haftalık bono — getiri eğrisi için
    "RSP": "RSP",          # eşit ağırlıklı S&P — piyasa genişliği
    "IWM": "IWM",          # küçük ölçek — riskin ucu
    "XLY": "XLY",          # ihtiyari tüketim
    "XLP": "XLP",          # temel tüketim (defansif)
}

# 2026 FOMC toplantı tarihleri (son gün). Yeni yıl takvimi açıklanınca güncelleyin.
FOMC_DATES_2026 = [
    dt.date(2026, 1, 28), dt.date(2026, 3, 18), dt.date(2026, 4, 29),
    dt.date(2026, 6, 17), dt.date(2026, 7, 29), dt.date(2026, 9, 16),
    dt.date(2026, 11, 4), dt.date(2026, 12, 16),
]
FOMC_DATES_2027 = [
    dt.date(2027, 1, 27), dt.date(2027, 3, 17), dt.date(2027, 4, 28),
    dt.date(2027, 6, 16), dt.date(2027, 7, 28), dt.date(2027, 9, 22),
    dt.date(2027, 11, 3), dt.date(2027, 12, 15),
]


# --------------------------------------------------------------------------
# Takvim
# --------------------------------------------------------------------------
def third_friday(year: int, month: int) -> dt.date:
    """Aylık opsiyon vadesi — ayın üçüncü cuması."""
    c = calendar.Calendar(firstweekday=calendar.MONDAY)
    fridays = [d for week in c.monthdatescalendar(year, month)
               for d in week if d.weekday() == calendar.FRIDAY and d.month == month]
    return fridays[2]


def next_opex(today: dt.date | None = None) -> tuple[dt.date, int, bool]:
    """Sıradaki OPEX tarihi, kalan gün ve üçlü cadı (quad witching) mı."""
    today = today or dt.date.today()
    d = third_friday(today.year, today.month)
    if today > d:
        m = today.month % 12 + 1
        y = today.year + (1 if today.month == 12 else 0)
        d = third_friday(y, m)
    return d, (d - today).days, d.month in (3, 6, 9, 12)


def next_fomc(today: dt.date | None = None) -> tuple[dt.date | None, int | None]:
    today = today or dt.date.today()
    future = [d for d in (FOMC_DATES_2026 + FOMC_DATES_2027) if d >= today]
    if not future:
        return None, None
    return future[0], (future[0] - today).days


# --------------------------------------------------------------------------
# Ölçümler
# --------------------------------------------------------------------------
@dataclass
class MacroReading:
    """Tek bir makro göstergenin okuması."""
    key: str
    label: str
    value: float
    change_pct: float
    detail: str = ""
    tone: str = "neutral"     # good | bad | neutral


@dataclass
class MacroState:
    readings: dict[str, MacroReading] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    regime: str = "BELİRSİZ"
    regime_desc: str = ""
    battery: dict[str, int] = field(default_factory=dict)
    risk_score: float = 50.0
    errors: list[str] = field(default_factory=list)
    opex_date: str = ""
    opex_days: int = 0
    opex_quad: bool = False
    fomc_date: str = ""
    fomc_days: int | None = None
    asof: str = ""

    def get(self, key: str) -> float:
        r = self.readings.get(key)
        return r.value if r else float("nan")


def _pct_change(s: pd.Series, n: int = 1) -> float:
    s = s.dropna()
    if len(s) <= n:
        return float("nan")
    return float((s.iloc[-1] / s.iloc[-1 - n] - 1) * 100)


def _last(s: pd.Series) -> float:
    s = pd.Series(s).dropna()
    return float(s.iloc[-1]) if len(s) else float("nan")


def build_macro_state(prices: dict[str, pd.DataFrame],
                      today: dt.date | None = None) -> MacroState:
    """
    Çekilmiş fiyat serilerinden makro durumu hesaplar.
    `prices`: {anahtar: OHLCV DataFrame} — anahtarlar MACRO_TICKERS ile aynı.
    """
    st = MacroState()
    today = today or dt.date.today()
    st.asof = dt.datetime.now().strftime("%d.%m.%Y %H:%M")

    opex_d, opex_dl, quad = next_opex(today)
    st.opex_date, st.opex_days, st.opex_quad = opex_d.strftime("%d.%m.%Y"), opex_dl, quad
    fomc_d, fomc_dl = next_fomc(today)
    st.fomc_date = fomc_d.strftime("%d.%m.%Y") if fomc_d else "—"
    st.fomc_days = fomc_dl

    def close(key: str) -> pd.Series | None:
        df = prices.get(key)
        if df is None or df.empty or "Close" not in df:
            return None
        s = df["Close"].dropna()
        return s if len(s) > 5 else None

    def add(key: str, label: str, value: float, chg: float,
            detail: str = "", tone: str = "neutral") -> None:
        st.readings[key] = MacroReading(key, label, value, chg, detail, tone)

    # --- VIX ve vade yapısı ---
    vix = close("VIX")
    vix3m = close("VIX3M")
    if vix is not None:
        v = _last(vix)
        if v < 15:
            tone, det = "good", "Rehavet — koruma ucuz, ani şoklara açık"
        elif v < 22:
            tone, det = "neutral", "Normal aralık"
        elif v < 30:
            tone, det = "bad", "Gerginlik — pozisyon boyutunu düşür"
        else:
            tone, det = "bad", "Panik bölgesi — dip arayışı erken"
        add("VIX", "VIX", v, _pct_change(vix), det, tone)
        st.scores["vix"] = float(np.clip((25 - v) / 15 * 100, 0, 100))

        if vix3m is not None:
            v3 = _last(vix3m)
            ratio = v / v3 if v3 else np.nan
            if np.isfinite(ratio):
                inverted = ratio > 1.0
                ts_series = (vix / vix3m.reindex(vix.index).ffill()).dropna()
                ts_chg = _pct_change(ts_series, 5) if len(ts_series) > 6 else 0.0
                add("VIX_TS", "VIX Vade Yapısı", ratio,
                    ts_chg if np.isfinite(ts_chg) else 0.0,
                    "TERSİNE DÖNMÜŞ — yakın vade korkusu uzun vadeyi aştı, "
                    "kısa vadeli stres" if inverted else
                    "Normal contango — piyasa sakin",
                    "bad" if inverted else "good")
                st.scores["vix_ts"] = float(np.clip((1.05 - ratio) / 0.25 * 100, 0, 100))

    # --- SPY trend rejimi ---
    spy = close("SPY")
    if spy is not None and len(spy) > 200:
        e50 = spy.ewm(span=50, adjust=False).mean()
        e200 = spy.ewm(span=200, adjust=False).mean()
        px, m50, m200 = _last(spy), _last(e50), _last(e200)
        above50, above200 = px > m50, px > m200
        golden = m50 > m200
        if above50 and above200 and golden:
            det, tone, sc = "Tam boğa dizilimi (fiyat > 50 EMA > 200 EMA)", "good", 90
        elif above200 and not above50:
            det, tone, sc = "Ana trend yukarı ama kısa vadede zayıf — long sinyal kalitesi düşer", "neutral", 55
        elif not above200 and above50:
            det, tone, sc = "Ayı piyasasında toparlanma — dikkatli", "neutral", 40
        else:
            det, tone, sc = "Fiyat 50 ve 200 EMA altında — long sinyalleri güvenilmez", "bad", 15
        add("SPY_TREND", "SPY Trend Rejimi", (px / m50 - 1) * 100, _pct_change(spy),
            det, tone)
        st.scores["trend"] = float(sc)
        st.readings["SPY_TREND"].detail = det

    # --- Kredi iştahı: HYG / TLT ---
    hyg, tlt = close("HYG"), close("TLT")
    if hyg is not None and tlt is not None:
        rel = (hyg / tlt.reindex(hyg.index).ffill()).dropna()
        if len(rel) > 25:
            mom = _pct_change(rel, 20)
            tone = "good" if mom > 0 else "bad"
            add("CREDIT", "Kredi İştahı (HYG/TLT)", mom, _pct_change(rel, 5),
                "Riskli tahvil güvenli tahvile göre güçleniyor — risk alma isteği var"
                if mom > 0 else
                "Sermaye güvenli tahvile kaçıyor — riskten kaçınma", tone)
            st.scores["credit"] = float(np.clip(50 + mom * 8, 0, 100))

    # --- Dolar likiditesi ---
    dxy = close("DXY")
    if dxy is not None:
        mom = _pct_change(dxy, 20)
        add("DXY", "Dolar Endeksi", _last(dxy), mom,
            "Dolar güçleniyor — küresel likidite sıkışıyor, riskli varlık aleyhine"
            if mom > 0 else
            "Dolar zayıflıyor — likidite gevşiyor, riskli varlık lehine",
            "bad" if mom > 1 else "good" if mom < -1 else "neutral")
        st.scores["dollar"] = float(np.clip(50 - mom * 10, 0, 100))

    # --- Altın / Bakır: korku mu büyüme mi ---
    gold, copper = close("GOLD"), close("COPPER")
    if gold is not None and copper is not None:
        rel = (gold / copper.reindex(gold.index).ffill()).dropna()
        if len(rel) > 25:
            mom = _pct_change(rel, 20)
            add("GOLD_COPPER", "Altın / Bakır", mom, _pct_change(rel, 5),
                "Altın bakırı geçiyor — korku ve durgunluk fiyatlanıyor" if mom > 0
                else "Bakır altını geçiyor — büyüme ve sanayi talebi fiyatlanıyor",
                "bad" if mom > 2 else "good" if mom < -2 else "neutral")
            st.scores["growth"] = float(np.clip(50 - mom * 5, 0, 100))

    # --- Kripto risk iştahı ---
    btc = close("BTC")
    if btc is not None:
        mom = _pct_change(btc, 20)
        add("BTC", "Bitcoin", _last(btc), _pct_change(btc),
            "Risk iştahının ucu güçlü" if mom > 0 else "Riskli uç zayıflıyor",
            "good" if mom > 0 else "bad")
        st.scores["crypto"] = float(np.clip(50 + mom * 2, 0, 100))

    # --- 10 yıllık faiz ---
    tnx = close("US10Y")
    if tnx is not None:
        v = _last(tnx) / 10.0        # ^TNX 10 katı olarak gelir
        mom = _pct_change(tnx, 20)
        add("US10Y", "ABD 10Y Faiz", v, mom,
            "Faizler yükseliyor — yüksek çarpanlı hisseler baskı altında"
            if mom > 0 else "Faizler geriliyor — büyüme hisseleri rahatlar",
            "bad" if mom > 3 else "good" if mom < -3 else "neutral")
        st.scores["rates"] = float(np.clip(50 - mom * 3, 0, 100))

    # --- Piyasa genişliği: eşit ağırlıklı S&P / SPY ---
    rsp = close("RSP")
    if rsp is not None and spy is not None:
        rel = (rsp / spy.reindex(rsp.index).ffill()).dropna()
        if len(rel) > 25:
            mom = _pct_change(rel, 20)
            add("BREADTH", "Piyasa Genişliği (RSP/SPY)", mom, _pct_change(rel, 5),
                "Ortalama hisse endeksten iyi — yükseliş tabana yayılmış, "
                "sağlıklı" if mom > 0 else
                "Endeksi birkaç dev taşıyor — yükseliş dar tabanlı, kırılgan",
                "good" if mom > 0 else "bad")
            st.scores["breadth"] = float(np.clip(50 + mom * 10, 0, 100))

    # --- Riskin ucu: küçük ölçek / SPY ---
    iwm = close("IWM")
    if iwm is not None and spy is not None:
        rel = (iwm / spy.reindex(iwm.index).ffill()).dropna()
        if len(rel) > 25:
            mom = _pct_change(rel, 20)
            add("SMALLCAP", "Küçük Ölçek (IWM/SPY)", mom, _pct_change(rel, 5),
                "Küçük ölçek endeksi geçiyor — risk iştahı gerçek, likidite bol"
                if mom > 0 else
                "Sermaye büyük ve likit isimlere sığınıyor — savunmacı ralli",
                "good" if mom > 0 else "bad")
            st.scores["smallcap"] = float(np.clip(50 + mom * 8, 0, 100))

    # --- Tüketici sinyali: ihtiyari / temel ---
    xly, xlp = close("XLY"), close("XLP")
    if xly is not None and xlp is not None:
        rel = (xly / xlp.reindex(xly.index).ffill()).dropna()
        if len(rel) > 25:
            mom = _pct_change(rel, 20)
            add("CONSUMER", "İhtiyari / Temel Tüketim", mom, _pct_change(rel, 5),
                "İhtiyari tüketim defansifi geçiyor — büyüme bekleniyor"
                if mom > 0 else
                "Sermaye defansif tüketime kaçıyor — büyüme beklentisi zayıflıyor",
                "good" if mom > 0 else "bad")
            st.scores["consumer"] = float(np.clip(50 + mom * 6, 0, 100))

    # --- Getiri eğrisi: 10Y − 3A ---
    irx = close("US3M")
    if tnx is not None and irx is not None:
        y10 = _last(tnx) / 10.0
        y3m = _last(irx) / 10.0
        if np.isfinite(y10) and np.isfinite(y3m):
            spread = y10 - y3m
            # 20 gün önceki eğim — dikleşiyor mu düzleşiyor mu
            prev = np.nan
            t10 = tnx.dropna()
            t3 = irx.dropna()
            if len(t10) > 21 and len(t3) > 21:
                prev = float(t10.iloc[-21] / 10.0 - t3.iloc[-21] / 10.0)
            delta = (spread - prev) if np.isfinite(prev) else 0.0
            if spread < 0:
                det = ("TERS EĞRİ — kısa vade uzun vadeden pahalı. Tarihsel "
                       "olarak durgunluk öncüsü; bankalar ve döngüsel sektörler "
                       "baskı görür.")
                tone = "bad"
            elif spread < 0.5:
                det = ("Eğri neredeyse düz — büyüme beklentisi zayıf, "
                       "faiz makası bankaları sıkıştırıyor.")
                tone = "neutral"
            else:
                det = ("Eğri normal/dik — büyüme fiyatlanıyor, banka ve "
                       "döngüsel sektörler lehine.")
                tone = "good"
            if np.isfinite(prev):
                det += (" Son 1 ayda dikleşiyor." if delta > 0.05
                        else " Son 1 ayda düzleşiyor." if delta < -0.05
                        else "")
            add("CURVE", "Getiri Eğrisi (10Y − 3A)", spread, delta * 100, det, tone)
            st.scores["curve"] = float(np.clip(50 + spread * 25, 0, 100))

    if not st.scores:
        st.errors.append("Hiçbir makro gösterge çekilemedi — rejim hesaplanamadı.")
        st.battery = {"Hisse": 50, "Tahvil": 50, "Kripto": 50,
                      "Emtia": 50, "Gayrimenkul": 50}
        return st

    st.risk_score = float(np.mean(list(st.scores.values())))
    st.regime, st.regime_desc = classify_regime(st)
    st.battery = compute_battery(st)
    return st


def classify_regime(st: MacroState) -> tuple[str, str]:
    """Ölçümlerden rejim etiketi türetir. Sıra önemlidir: en keskin durum önce."""
    vix = st.get("VIX")
    ts = st.get("VIX_TS")
    trend = st.scores.get("trend", 50)
    credit = st.scores.get("credit", 50)
    dollar = st.scores.get("dollar", 50)
    risk = st.risk_score

    if np.isfinite(vix) and vix > 30 and trend < 30:
        return ("🩸 LİKİDİTE KRİZİ",
                f"VIX {vix:.1f} ile panik bölgesinde ve SPY ana trendin altında. "
                f"Yüksek çarpanlı teknoloji, biyoteknoloji ve kriptoda margin call "
                f"döngüsü riski var. Bu rejimde dip alımı erkendir; nakit ve uzun "
                f"vadeli tahvil korunma sağlar.")

    if np.isfinite(ts) and ts > 1.0 and credit < 45:
        return ("🌍 JEOPOLİTİK / OLAY ŞOKU",
                f"VIX vade yapısı tersine dönmüş (oran {ts:.2f}) ve kredi iştahı "
                f"zayıf. Sermaye teknolojiden kaçıp altın, savunma, petrol ve "
                f"hazineye sığınıyor. Tedarik zincirine bağlı şirketler en hızlı "
                f"ezilen taraf.")

    if st.opex_days <= 3:
        quad = " ÜÇLÜ CADI (quad witching) — etki normalden güçlü." if st.opex_quad else ""
        return ("🎯 OPEX PINNING",
                f"Opsiyon vadesine {st.opex_days} gün kaldı ({st.opex_date}).{quad} "
                f"Market maker'lar primi sıfırlamak için endeksi en yüksek açık "
                f"pozisyonun olduğu Max Pain noktasına hapsetmeye çalışıyor. Trend "
                f"kırılımları bu pencerede çoğunlukla tuzak (whipsaw) çıkar; vade "
                f"geçmeden yeni pozisyon açmak risklidir.")

    if np.isfinite(vix) and vix < 15 and trend > 70 and risk > 62:
        return ("🚀 RİSK İŞTAHI / GAMMA",
                f"VIX {vix:.1f} ile rehavet bölgesinde, SPY tam boğa diziliminde ve "
                f"kredi iştahı açık. Dealer'lar call hedge'i için spot almak zorunda "
                f"kaldığında parabolik hızlanma (gamma squeeze) görülebilir. Ancak "
                f"koruma ucuzken piyasa şoka da en açık haldedir.")

    if st.fomc_days is not None and st.fomc_days <= 3:
        return ("🏦 FOMC BEKLEYİŞİ",
                f"FOMC toplantısına {st.fomc_days} gün kaldı ({st.fomc_date}). "
                f"Karar öncesi hacim düşer, karar anında oynaklık patlar. Stop "
                f"mesafeleri normal ATR'ye göre dar kalır; pozisyon boyutunu "
                f"küçültmek stop mantığını korur.")

    if dollar < 35 and credit < 45:
        return ("💵 DOLAR SIKIŞMASI",
                "Dolar güçlenirken kredi iştahı zayıflıyor. Küresel likidite "
                "çekiliyor; yüksek çarpanlı ve kaldıraçlı varlıklar önce satılır.")

    if risk >= 62 and trend >= 50:
        return ("🟢 RİSK AÇIK",
                f"Bileşik risk skoru {risk:.0f}/100 ve SPY ana trendi yukarı. "
                f"Trend, kredi ve oynaklık birlikte long tarafını destekliyor — "
                f"sinyal kalitesi yüksek.")
    if risk >= 62 and trend < 50:
        return ("🟡 KARIŞIK — TREND ZAYIF",
                f"Kredi iştahı ve oynaklık iyi görünüyor (bileşik skor {risk:.0f}/100) "
                f"AMA fiyat SPY 50 EMA'sının altında. Bu ikisi çeliştiğinde rejim "
                f"kapısı kapalı sayılır: long sinyalleri izleme listesi olarak "
                f"kullanılır, pozisyon için trendin dönmesi beklenir.")
    if risk <= 40 or trend < 25:
        return ("🔴 RİSK KAPALI",
                f"Bileşik risk skoru {risk:.0f}/100. Long sinyallerinin isabet "
                f"oranı bu rejimde tarihsel olarak düşer; kısa vadeli işlemlerde "
                f"pozisyon boyutunu küçültmek gerekir.")
    return ("⚖️ GEÇİŞ / KARARSIZ",
            f"Bileşik risk skoru {risk:.0f}/100. Göstergeler birbiriyle çelişiyor; "
            f"tek yönlü agresif pozisyon için teyit bekleyin.")


def compute_battery(st: MacroState) -> dict[str, int]:
    """
    Varlık sınıflarına sermaye akış eğilimi (0–100).
    Elle yazılmış sabitler yerine ölçülen skorlardan türetilir.
    """
    trend = st.scores.get("trend", 50)
    vix = st.scores.get("vix", 50)
    credit = st.scores.get("credit", 50)
    dollar = st.scores.get("dollar", 50)
    growth = st.scores.get("growth", 50)
    crypto = st.scores.get("crypto", 50)
    rates = st.scores.get("rates", 50)
    breadth = st.scores.get("breadth", 50)
    smallcap = st.scores.get("smallcap", 50)
    consumer = st.scores.get("consumer", 50)
    curve = st.scores.get("curve", 50)

    def clip(x: float) -> int:
        return int(np.clip(round(x), 2, 98))

    # Genişlik ve küçük ölçek eklendi: endeksi birkaç dev taşıyorken "Hisse"
    # bataryasının dolu görünmesi yanıltıcıydı.
    hisse = (0.30 * trend + 0.20 * credit + 0.15 * vix + 0.10 * dollar
             + 0.15 * breadth + 0.10 * smallcap)
    tahvil = 100 - (0.45 * credit + 0.35 * vix + 0.20 * breadth)
    kripto = 0.35 * crypto + 0.25 * credit + 0.20 * dollar + 0.20 * smallcap
    emtia = 0.40 * (100 - growth) + 0.25 * dollar + 0.20 * trend + 0.15 * consumer
    gyo = 0.35 * rates + 0.25 * trend + 0.20 * credit + 0.20 * curve

    # OPEX haftasında her şey ortaya çekilir (pinning)
    if st.opex_days <= 2:
        pull = 0.35
        hisse = hisse * (1 - pull) + 50 * pull
        kripto = kripto * (1 - pull) + 50 * pull

    return {"Hisse": clip(hisse), "Tahvil": clip(tahvil), "Kripto": clip(kripto),
            "Emtia": clip(emtia), "Gayrimenkul": clip(gyo)}


# --------------------------------------------------------------------------
# Skor sözlüğü — "Ne değişti?" tablosunda okunur adlar
# --------------------------------------------------------------------------
SCORE_LABELS: dict[str, tuple[str, str]] = {
    "trend":    ("SPY Trend", "Fiyatın 50/200 EMA'ya göre dizilimi"),
    "vix":      ("Oynaklık (VIX)", "Düşük VIX = yüksek skor"),
    "vix_ts":   ("VIX Vade Yapısı", "VIX/VIX3M 1'in altındayken yüksek skor"),
    "credit":   ("Kredi İştahı", "HYG'nin TLT'ye göre 20 günlük momentumu"),
    "dollar":   ("Dolar Likiditesi", "DXY zayıfladıkça skor yükselir"),
    "growth":   ("Büyüme (Bakır/Altın)", "Bakır altını geçtikçe skor yükselir"),
    "crypto":   ("Kripto İştahı", "BTC 20 günlük momentumu"),
    "rates":    ("Faiz Yönü", "10Y gerilerken skor yükselir"),
    "breadth":  ("Piyasa Genişliği", "RSP/SPY — yükseliş tabana yayıldıkça yükselir"),
    "smallcap": ("Küçük Ölçek", "IWM/SPY — riskin ucundaki iştah"),
    "consumer": ("Tüketici Sinyali", "XLY/XLP — büyüme mi savunma mı"),
    "curve":    ("Getiri Eğrisi", "10Y − 3A; ters eğri düşük skor"),
}

# Sermaye akış bataryasının hangi sınıfı hangi skorlardan beslendiği
BATTERY_DRIVERS: dict[str, str] = {
    "Hisse": "SPY trend, kredi iştahı, VIX, dolar, genişlik, küçük ölçek",
    "Tahvil": "Kredi iştahı ve VIX'in tersi — korku arttıkça dolar",
    "Kripto": "BTC momentumu, kredi iştahı, dolar, küçük ölçek",
    "Emtia": "Altın/bakır, dolar, SPY trend, tüketici sinyali",
    "Gayrimenkul": "Faiz yönü, SPY trend, kredi iştahı, getiri eğrisi",
}


def _truncate(prices: dict[str, pd.DataFrame], back: int
              ) -> dict[str, pd.DataFrame]:
    """Tüm serileri `back` bar geriye keser — geçmişteki durumu yeniden kurar."""
    if back <= 0:
        return prices
    out: dict[str, pd.DataFrame] = {}
    for k, df in prices.items():
        if df is None or df.empty:
            out[k] = df
        elif len(df) > back:
            out[k] = df.iloc[:-back]
        else:
            out[k] = df.iloc[:0]
    return out


def state_at(prices: dict[str, pd.DataFrame], back: int,
             today: dt.date | None = None) -> MacroState:
    """
    `back` işlem barı önceki makro durumu.

    Neden gerekli: "Hisse bataryası 68" tek başına anlamsızdır — bir hafta önce
    52 miydi, 81 miydi karar buna bağlıdır. Aynı fonksiyonla geçmişi yeniden
    hesaplamak, ayrı bir kayıt dosyası tutmadan doğru karşılaştırma verir.
    """
    today = today or dt.date.today()
    ref = today - dt.timedelta(days=int(back * 7 / 5))     # işlem günü ≈ takvim
    return build_macro_state(_truncate(prices, back), today=ref)


PERIOD_BARS: dict[str, int] = {"1 gün": 1, "1 hafta": 5, "1 ay": 21}


def battery_changes(prices: dict[str, pd.DataFrame],
                    current: MacroState,
                    today: dt.date | None = None
                    ) -> tuple[pd.DataFrame, dict[str, MacroState]]:
    """
    Batarya ve risk skorunun 1 gün / 1 hafta / 1 ay önceki değerleri ve farkları.
    Döner: (tablo, {dönem: geçmiş durum}).
    """
    past: dict[str, MacroState] = {}
    for label, back in PERIOD_BARS.items():
        try:
            past[label] = state_at(prices, back, today)
        except Exception:                      # pragma: no cover - savunmacı
            continue

    rows: list[dict[str, Any]] = []
    for sinif, simdi in current.battery.items():
        row: dict[str, Any] = {"Varlık Sınıfı": sinif, "Şimdi": simdi}
        for label in PERIOD_BARS:
            p = past.get(label)
            onceki = p.battery.get(sinif) if p and p.battery else None
            row[f"{label} önce"] = onceki
            row[f"Δ {label}"] = (simdi - onceki) if onceki is not None else np.nan
        row["Besleyen"] = BATTERY_DRIVERS.get(sinif, "")
        rows.append(row)

    # Risk skoru da aynı tabloda taşınsın
    risk_row: dict[str, Any] = {"Varlık Sınıfı": "▸ Bileşik Risk Skoru",
                                "Şimdi": round(current.risk_score)}
    for label in PERIOD_BARS:
        p = past.get(label)
        onceki = round(p.risk_score) if p else None
        risk_row[f"{label} önce"] = onceki
        risk_row[f"Δ {label}"] = ((current.risk_score - p.risk_score)
                                  if p else np.nan)
    risk_row["Besleyen"] = "Tüm alt skorların ortalaması"
    rows.append(risk_row)
    return pd.DataFrame(rows), past


def score_changes(current: MacroState,
                  past: dict[str, MacroState]) -> pd.DataFrame:
    """Her alt skorun dönemsel değişimi — rejimi ne itiyor, ne çekiyor."""
    rows: list[dict[str, Any]] = []
    for key, val in current.scores.items():
        label, aciklama = SCORE_LABELS.get(key, (key, ""))
        row: dict[str, Any] = {"Skor": label, "Şimdi": val}
        for plabel in PERIOD_BARS:
            p = past.get(plabel)
            prev = p.scores.get(key) if p else None
            row[f"Δ {plabel}"] = (val - prev) if prev is not None else np.nan
        row["Ne ölçüyor"] = aciklama
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty and "Δ 1 hafta" in df.columns:
        df = df.sort_values("Δ 1 hafta", ascending=False, na_position="last")
    return df


def battery_history(prices: dict[str, pd.DataFrame], days: int = 60,
                    today: dt.date | None = None) -> pd.DataFrame:
    """
    Son `days` işlem günü için batarya ve risk skoru seyri.

    Her gün için tüm makro motoru yeniden çalıştırılır; böylece geçmiş,
    bugünün formülüyle tutarlı olur (kayıt dosyası tutulsaydı formül
    değiştiğinde geçmiş kırılırdı).
    """
    today = today or dt.date.today()
    ref = None
    for df in prices.values():
        if df is not None and not df.empty:
            ref = df.index
            break
    if ref is None:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for back in range(days, -1, -1):
        if back >= len(ref):
            continue
        try:
            s = state_at(prices, back, today)
        except Exception:                      # pragma: no cover
            continue
        if not s.battery:
            continue
        stamp = ref[-1 - back]
        rows.append({"Tarih": stamp, **s.battery,
                     "Risk Skoru": round(s.risk_score, 1),
                     "Rejim": s.regime})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.set_index("Tarih")
    return out


def regime_shifts(history: pd.DataFrame) -> list[dict[str, Any]]:
    """Seyir tablosundaki rejim değişim anları — 'ne zaman döndü' sorusu."""
    if history.empty or "Rejim" not in history:
        return []
    out: list[dict[str, Any]] = []
    prev = None
    for stamp, row in history.iterrows():
        cur = row["Rejim"]
        if prev is not None and cur != prev:
            out.append({"Tarih": stamp, "Önceki": prev, "Yeni": cur,
                        "Risk Skoru": row.get("Risk Skoru")})
        prev = cur
    return out


# --------------------------------------------------------------------------
# Elle seçilebilen senaryolar (referans / eğitim amaçlı)
# --------------------------------------------------------------------------
MANUAL_SCENARIOS: dict[str, dict[str, Any]] = {
    "🚀 GAMMA SQUEEZE": {
        "battery": {"Hisse": 95, "Tahvil": 20, "Kripto": 90, "Emtia": 55,
                    "Gayrimenkul": 65},
        "desc": "Beklenmedik güvercin FED açıklaması, zayıf enflasyon verisi veya "
                "yoğun call alımı sonrası piyasa yapıcıların delta-hedge için spot "
                "hisseye saldırması. Hızlı şişme yaratır ama temele dayanmadığı "
                "için sert düzeltme olasılığı masadadır.",
    },
    "🎯 OPEX PINNING": {
        "battery": {"Hisse": 50, "Tahvil": 50, "Kripto": 48, "Emtia": 52,
                    "Gayrimenkul": 50},
        "desc": "Market maker'lar primleri (theta) sıfırlamak için endeksi en "
                "yüksek açık pozisyon yoğunluğunun olduğu Max Pain noktasına "
                "hapseder. Trend kırılımları çoğunlukla tuzak çıkar.",
    },
    "🌍 JEOPOLİTİK ŞOK": {
        "battery": {"Hisse": 25, "Tahvil": 85, "Kripto": 35, "Emtia": 95,
                    "Gayrimenkul": 40},
        "desc": "Boğaz krizleri, gümrük tarifeleri veya enerji nakil hatlarına "
                "saldırı. Sermaye riskten kaçıp altın, savunma, petrol ve hazineye "
                "sığınır; tedarik zincirine bağlı şirketler anında ezilir.",
    },
    "🏦 LİKİDİTE KRİZİ (FED)": {
        "battery": {"Hisse": 15, "Tahvil": 90, "Kripto": 10, "Emtia": 35,
                    "Gayrimenkul": 25},
        "desc": "İnatçı enflasyon, devasa tahvil ihracı veya Reverse Repo havuzunun "
                "kuruması. Yüksek F/K'lı teknoloji, biyoteknoloji ve kriptoda "
                "acımasız likidasyon ve margin call döngüleri.",
    },
}

REGIME_GLOSSARY: list[tuple[str, str]] = [
    ("OPEX Pinning",
     "Aylık/üç aylık opsiyon vadesine yaklaşırken market maker'lar fiyatı "
     "yatırımcıların büyük kısmının kaybedeceği Max Pain noktasına çeker. "
     "Sahte kırılımlar artar, trend takip sistemleri bu pencerede kötü çalışır."),
    ("Gamma Squeeze",
     "Yoğun call alımı sonrası dealer'ların hedge amaçlı spot hisse almak zorunda "
     "kalmasıyla oluşan parabolik yükseliş döngüsü. Kendi kendini besler, "
     "beslediği kadar da hızlı çöker."),
    ("VIX Vade Yapısı",
     "VIX (30 gün) / VIX3M (3 ay) oranı 1'in üstüne çıkarsa yakın vade korkusu "
     "uzun vadeyi aşmış demektir — kısa vadeli stres göstergesi. Normalde bu oran "
     "1'in altındadır (contango)."),
    ("Kredi İştahı (HYG/TLT)",
     "Yüksek getirili şirket tahvilinin uzun vadeli hazineye göre performansı. "
     "Yükseliyorsa piyasa risk almaya istekli, düşüyorsa sermaye güvenliğe kaçıyor. "
     "Hisse senedi rallilerinin en güvenilir teyit göstergelerinden biridir."),
    ("Altın / Bakır Oranı",
     "Altın korkunun, bakır sanayi büyümesinin göstergesidir. Oran yükseliyorsa "
     "piyasa durgunluk, düşüyorsa büyüme fiyatlıyor."),
    ("Dolar Endeksi (DXY)",
     "Dolar güçlendikçe küresel likidite sıkışır; gelişen piyasalar, emtia ve "
     "yüksek çarpanlı büyüme hisseleri baskı görür."),
]
