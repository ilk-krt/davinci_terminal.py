# apex/funnel.py — AETHER APEX
from __future__ import annotations


# apex/funnel.py — KARAR HUNİSİ
#
# Yatırım kararını dört soruya böler; her adım bir öncekinin cevabına dayanır:
#
#   1) Risk açık mı?            → piyasanın genel risk iştahı (6 sütun)
#   2) Para nereye akıyor?      → endeks / varlık sınıfı / kripto rotasyonu
#   3) Hangi tema ivmeleniyor?  → tema ve ETF'lerde ERKEN akış tespiti (RRG)
#   4) Bu temada hangi hisse?   → lider / geride kalan / alım adayı ayrımı
#
# Fiyat verisi yfinance'ten; kripto endeksleri (TOTAL, TOTAL3, BTC.D,
# OTHERS/BTC) ve Fed net likiditesi repodaki data/*.csv dosyalarından
# (GitHub Actions her iş günü günceller — tools/macro_update.py).


from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Semboller
# --------------------------------------------------------------------------
ROTATION_TICKERS: list[str] = [
    "SPY", "QQQ", "IWM", "RSP", "SMH", "ARKK", "XLY", "XLP", "EEM", "TLT",
    "HYG", "GC=F", "SI=F", "HG=F", "BZ=F", "DX-Y.NYB", "^GSPC", "^IXIC",
    "BTC-USD", "ETH-USD",
]

NAMES: dict[str, str] = {
    "SPY": "S&P 500", "QQQ": "Nasdaq 100", "IWM": "Russell 2000 (küçük ölçek)",
    "RSP": "Eşit ağırlıklı S&P", "SMH": "Yarı iletkenler",
    "ARKK": "Spekülatif büyüme (ARKK)", "XLY": "İhtiyari tüketim",
    "XLP": "Temel tüketim", "EEM": "Gelişen piyasalar", "TLT": "Uzun vadeli tahvil",
    "HYG": "Yüksek getirili tahvil", "GC=F": "Altın", "SI=F": "Gümüş",
    "HG=F": "Bakır", "BZ=F": "Brent petrol", "DX-Y.NYB": "Dolar endeksi",
    "^GSPC": "S&P 500 endeksi", "^IXIC": "Nasdaq Bileşik", "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum", "TOTAL": "Toplam kripto", "TOTAL3": "Altcoinler (TOTAL3)",
    "OTHERS": "Küçük altcoinler (OTHERS)", "BTC.D": "BTC hakimiyeti",
    "NET_LIQ": "Fed net likiditesi",
}


# --------------------------------------------------------------------------
# Rotasyon çiftleri — "pay / payda" yükseliyorsa para paydadan paya akıyor
#   risk: +1 → yükselişi risk iştahı demek, −1 → risk kaçışı demek
# --------------------------------------------------------------------------
PAIRS: list[dict[str, Any]] = [
    # Hisse içi rotasyon
    dict(key="QQQ/SPY", num="QQQ", den="SPY", grup="Hisse içi", risk=+1,
         ad="Nasdaq 100 / S&P 500",
         not_="Yükseliyorsa büyüme ve teknoloji liderleri piyasayı sürüklüyor. "
              "Çeyreklik grafikte kırılım, liderliğin süreceğine ve risk "
              "iştahına işaret eder."),
    dict(key="^IXIC/^GSPC", num="^IXIC", den="^GSPC", grup="Hisse içi", risk=+1,
         ad="Nasdaq Bileşik / S&P 500",
         not_="QQQ/SPY'den farkı, Nasdaq'taki küçük ve orta boy büyüme "
              "şirketlerini de kapsaması. İkisi birlikte yükseliyorsa iştah "
              "devlerle sınırlı değil."),
    dict(key="SMH/SPY", num="SMH", den="SPY", grup="Hisse içi", risk=+1,
         ad="Yarı iletken / S&P 500",
         not_="Yarı iletkenler döngünün öncü sektörü. Liderlik ediyorsa "
              "yapay zekâ ve donanım temaları güçlü."),
    dict(key="ARKK/QQQ", num="ARKK", den="QQQ", grup="Hisse içi", risk=+1,
         ad="Spekülatif büyüme / Nasdaq 100",
         not_="Kârsız, yüksek riskli büyüme şirketlerine iştah. Riskin en "
              "uç noktası — burası ısınıyorsa piyasa cesur."),
    dict(key="IWM/SPY", num="IWM", den="SPY", grup="Hisse içi", risk=+1,
         ad="Küçük ölçek / S&P 500",
         not_="Faiz indirimi ve ekonomik genişleme beklentisinde yükselir. "
              "Yükselişin tabana yayıldığını gösterir."),
    dict(key="RSP/SPY", num="RSP", den="SPY", grup="Hisse içi", risk=+1,
         ad="Eşit ağırlık / S&P 500 (genişlik)",
         not_="Düşüyorsa endeksi birkaç dev hisse taşıyor — sağlıksız "
              "yükseliş. Yükseliyorsa katılım geniş."),
    dict(key="XLY/XLP", num="XLY", den="XLP", grup="Hisse içi", risk=+1,
         ad="İhtiyari / Temel tüketim",
         not_="Tüketicinin keyfi harcaması mı, zorunlu harcaması mı öne "
              "çıkıyor? Klasik ofans/defans göstergesi."),
    dict(key="EEM/SPY", num="EEM", den="SPY", grup="Hisse içi", risk=+1,
         ad="Gelişen piyasalar / ABD",
         not_="Zayıf dolar ve küresel risk iştahında gelişen piyasalara "
              "para akar."),
    # Varlık sınıfları
    dict(key="SPY/TLT", num="SPY", den="TLT", grup="Varlık sınıfları", risk=+1,
         ad="Hisse / Tahvil",
         not_="Para tahvilden hisseye mi geçiyor? En temel risk-on/off "
              "göstergesi."),
    dict(key="HYG/TLT", num="HYG", den="TLT", grup="Varlık sınıfları", risk=+1,
         ad="Riskli tahvil / Hazine",
         not_="Kredi piyasası hisseden önce uyarır. Yükseliyorsa borç "
              "verenler risk almaya istekli."),
    dict(key="GC=F/SPY", num="GC=F", den="SPY", grup="Varlık sınıfları", risk=-1,
         ad="Altın / S&P 500",
         not_="Yükseliyorsa para hisseden güvenli limana kaçıyor ya da "
              "enflasyon/para basımı korkusu var."),
    dict(key="SI=F/GC=F", num="SI=F", den="GC=F", grup="Varlık sınıfları", risk=+1,
         ad="Gümüş / Altın",
         not_="Gümüş hem değerli hem sanayi metali. Altını geçiyorsa metal "
              "rallisi spekülatif/büyüme ayağına geçmiş demektir."),
    dict(key="HG=F/GC=F", num="HG=F", den="GC=F", grup="Varlık sınıfları", risk=+1,
         ad="Bakır / Altın",
         not_="Dr. Bakır: büyüme beklentisi. Yükseliyorsa ekonomi "
              "hızlanıyor; düşüyorsa resesyon korkusu."),
    dict(key="BZ=F", num="BZ=F", den=None, grup="Varlık sınıfları", risk=0,
         ad="Brent petrol",
         not_="Sert yükseliş enflasyon ve jeopolitik risk demek (hisseler "
              "için olumsuz); ılımlı yükseliş talep canlılığı."),
    dict(key="DX-Y.NYB", num="DX-Y.NYB", den=None, grup="Varlık sınıfları", risk=-1,
         ad="Dolar endeksi (DXY)",
         not_="Güçlü dolar küresel likiditeyi emer: riskli varlıklar, emtia, "
              "gelişen piyasalar ve kripto baskılanır."),
    # Kripto
    dict(key="BTC-USD", num="BTC-USD", den=None, grup="Kripto", risk=+1,
         ad="Bitcoin",
         not_="Kripto iştahının ana göstergesi ve küresel likiditeye en "
              "duyarlı varlık."),
    dict(key="TOTAL", num="TOTAL", den=None, grup="Kripto", risk=+1,
         ad="Toplam kripto piyasa değeri (TOTAL)",
         not_="Kripto piyasasına net para girişi."),
    dict(key="BTC.D", num="BTC.D", den=None, grup="Kripto", risk=-1,
         ad="BTC hakimiyeti (BTC.D)",
         not_="Yükseliyorsa kripto içinde para altcoinlerden BTC'ye "
              "kaçıyor (temkin). Düşüyorsa altcoinlere akıyor."),
    dict(key="ETH/BTC", num="ETH/BTC", den=None, grup="Kripto", risk=+1,
         ad="ETH / BTC",
         not_="Altcoin sezonunun ilk basamağı genellikle ETH'nin BTC'yi "
              "geçmesidir."),
    dict(key="TOTAL3", num="TOTAL3", den=None, grup="Kripto", risk=+1,
         ad="Altcoinler (TOTAL3: BTC ve ETH hariç)",
         not_="BTC ve ETH dışındaki tüm piyasaya para girişi."),
    dict(key="OTHERS/BTC", num="OTHERS/BTC", den=None, grup="Kripto", risk=+1,
         ad="Küçük altcoinler / BTC (OTHERS/BTC)",
         not_="Kripto riskinin en uç noktası. Yükselişi tam bir altcoin "
              "sezonu ve aşırı iştah anlamına gelir."),
    # Likidite
    dict(key="NET_LIQ", num="NET_LIQ", den=None, grup="Likidite", risk=+1,
         ad="Fed net likiditesi (bilanço − TGA − ters repo)",
         not_="Piyasadaki 'para musluğu'. Artıyorsa sistem para basıyor "
              "demektir; hisse ve kripto tarihsel olarak bunu takip eder."),
    dict(key="M2SL", num="M2SL", den=None, grup="Likidite", risk=+1,
         ad="M2 para arzı",
         not_="Ekonomideki toplam para miktarı. Yıllık büyümesi hızlanıyorsa "
              "varlık fiyatları desteklenir (aylık veri, gecikmeli)."),
]
PAIR = {p["key"]: p for p in PAIRS}

# Bir yatırımcının "benim hissem hangi segmentte?" sorusuna cevap
SEGMENTS: dict[str, dict[str, Any]] = {
    "Nasdaq / büyüme & teknoloji": dict(
        pairs=["QQQ/SPY", "^IXIC/^GSPC", "SMH/SPY", "ARKK/QQQ"],
        aciklama="Nasdaq'ta işlem gören büyüme, yazılım, yarı iletken ve "
                 "yapay zekâ hisseleri."),
    "S&P 500 / geniş piyasa": dict(
        pairs=["RSP/SPY", "SPY/TLT", "XLY/XLP"],
        aciklama="Büyük, kârlı şirketler; sanayi, finans, sağlık, tüketim."),
    "Küçük ölçek (Russell 2000)": dict(
        pairs=["IWM/SPY", "RSP/SPY", "HYG/TLT"],
        aciklama="Faize ve ekonomik döngüye en duyarlı küçük şirketler."),
    "Gelişen piyasalar": dict(
        pairs=["EEM/SPY", "DX-Y.NYB"],
        aciklama="Çin, Hindistan, Brezilya vb. — dolardan ters etkilenir."),
    "Altın, gümüş ve madenciler": dict(
        pairs=["GC=F/SPY", "SI=F/GC=F", "DX-Y.NYB"],
        aciklama="Değerli metaller ve madenci hisseleri."),
    "Enerji ve emtia": dict(
        pairs=["BZ=F", "HG=F/GC=F", "DX-Y.NYB"],
        aciklama="Petrol, gaz, bakır, sanayi metalleri ve üreticileri."),
    "Kripto ve kripto hisseleri": dict(
        pairs=["BTC-USD", "TOTAL", "BTC.D", "OTHERS/BTC", "NET_LIQ"],
        aciklama="BTC, altcoinler, madenciler (MARA, RIOT…), borsalar."),
    "Tahvil / defansif": dict(
        pairs=["SPY/TLT", "HYG/TLT"],
        aciklama="Hazine tahvilleri, temettü ve defansif sektörler. "
                 "Burada göstergeler TERS okunur: hisse/tahvil düşüyorsa "
                 "tahvil lehine."),
}
# Segment yönü: tahvil/defansif ve altın, risk iştahının tersinden beslenir
SEGMENT_INVERT = {"Tahvil / defansif"}
SEGMENT_ABS = {"Altın, gümüş ve madenciler": {"GC=F/SPY": +1},
               "Enerji ve emtia": {"BZ=F": +1}}


# --------------------------------------------------------------------------
# Repo veri dosyaları (GitHub Actions)
# --------------------------------------------------------------------------
def _data_path(name: str) -> Path | None:
    from apex.snapshot import data_file
    return data_file(name)


def read_crypto_caps() -> pd.DataFrame:
    p = _data_path("crypto_caps.csv")
    if p is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(p, index_col=0, parse_dates=True).sort_index()
    except Exception:
        return pd.DataFrame()


def read_liquidity() -> pd.DataFrame:
    p = _data_path("liquidity.csv")
    if p is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(p, index_col=0, parse_dates=True).sort_index()
    except Exception:
        return pd.DataFrame()


# --------------------------------------------------------------------------
# Seri istatistikleri
# --------------------------------------------------------------------------
def build_series(prices: dict[str, pd.DataFrame], crypto: pd.DataFrame,
                 liq: pd.DataFrame) -> dict[str, pd.Series]:
    """Her rotasyon anahtarı için günlük seri (oran ya da tekil)."""
    close = {t: df["Close"].dropna() for t, df in prices.items()
             if df is not None and not df.empty}
    for c in ("TOTAL", "TOTAL3", "OTHERS", "BTC.D", "ETH/BTC", "OTHERS/BTC"):
        if c in crypto.columns:
            close[c] = crypto[c].dropna()
    for c in ("NET_LIQ", "M2SL"):
        if c in liq.columns:
            close[c] = liq[c].dropna()

    out: dict[str, pd.Series] = {}
    for p in PAIRS:
        a = close.get(p["num"])
        if a is None or len(a) < 30:
            continue
        if p["den"]:
            b = close.get(p["den"])
            if b is None or len(b) < 30:
                continue
            a, b = a.copy(), b.copy()
            a.index = pd.to_datetime(a.index).tz_localize(None).normalize()
            b.index = pd.to_datetime(b.index).tz_localize(None).normalize()
            a = a[~a.index.duplicated()]
            b = b[~b.index.duplicated()]
            j = pd.concat([a, b], axis=1, join="inner").dropna()
            if len(j) < 30:
                continue
            s = j.iloc[:, 0] / j.iloc[:, 1]
        else:
            s = a.copy()
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            s = s[~s.index.duplicated()]
        out[p["key"]] = s.astype(float)
    return out


def _pct(s: pd.Series, n: int) -> float:
    if len(s) <= n or not s.iloc[-1 - n]:
        return np.nan
    return float((s.iloc[-1] / s.iloc[-1 - n] - 1) * 100)


def quarterly_ohlc(s: pd.Series) -> pd.DataFrame:
    q = s.resample("QE").agg(["first", "max", "min", "last"]).dropna()
    q.columns = ["Open", "High", "Low", "Close"]
    return q


def breakout_info(s: pd.Series, freq: str = "QE", look: int = 8) -> dict[str, Any]:
    """
    Kapanış bazlı kırılım: içinde bulunulan dönemin kapanışı, önceki `look`
    dönemin en yüksek KAPANIŞINI (ekran görüntüsündeki yatay direnç) geçti mi?
    """
    r = s.resample(freq).last().dropna()
    if len(r) < look + 2:
        return {"kirilim": False, "direnc": np.nan, "uzaklik": np.nan}
    prev = r.iloc[-look - 1:-1]
    res = float(prev.max())
    last = float(r.iloc[-1])
    br = last > res
    # bir önceki dönem de kırmış mıydı? (taze kırılım ayrımı)
    prev2 = r.iloc[-look - 2:-2]
    br_prev = float(r.iloc[-2]) > float(prev2.max()) if len(prev2) else False
    return {"kirilim": br, "taze": br and not br_prev, "direnc": res,
            "uzaklik": (last / res - 1) * 100}


def series_stats(s: pd.Series, risk: int) -> dict[str, Any]:
    """
    Bir oran/serinin akış durumu.
    Skor −100…+100: pozitif = pay lehine akış (oran yükseliyor).
    Faz: güçlü akış / akış başlıyor (ERKEN) / yavaşlıyor / çıkış.
    """
    s = s.dropna()
    n = len(s)
    c20, c63 = _pct(s, 20), _pct(s, 63)
    sma50 = s.rolling(50).mean()
    sma200 = s.rolling(200).mean()
    last = float(s.iloc[-1])
    above50 = bool(last > sma50.iloc[-1]) if n >= 50 else None
    above200 = bool(last > sma200.iloc[-1]) if n >= 200 else None

    # 20 günlük değişimin tarihsel oynaklığa göre normalize edilmesi
    ch20 = s.pct_change(20) * 100
    sd = float(ch20.dropna().tail(750).std()) if ch20.notna().sum() > 60 else np.nan
    z20 = c20 / sd if sd and np.isfinite(sd) and np.isfinite(c20) else 0.0
    ch63 = s.pct_change(63) * 100
    sd63 = float(ch63.dropna().tail(750).std()) if ch63.notna().sum() > 60 else np.nan
    z63 = c63 / sd63 if sd63 and np.isfinite(sd63) and np.isfinite(c63) else 0.0

    score = (40 * np.tanh(z20) + 30 * np.tanh(z63)
             + (15 if above50 else -15 if above50 is not None else 0)
             + (15 if above200 else -15 if above200 is not None else 0))

    # ivme: 20 günlük değişim 10 gün öncekine göre artıyor mu
    accel = (float(ch20.iloc[-1] - ch20.iloc[-11])
             if n > 31 and np.isfinite(ch20.iloc[-11]) else np.nan)
    # SMA50'yi son 15 günde yukarı kesti mi (erken dönüş)
    cross_up = False
    if n >= 65:
        above = (s > sma50).tail(16)
        cross_up = bool(above.iloc[-1] and not above.iloc[:-1].all()
                        and (~above.iloc[:-1]).any())

    long_neg = (np.isfinite(c63) and c63 <= 0) or above200 is False
    if np.isfinite(c20) and c20 > 0 and long_neg and (
            cross_up or (np.isfinite(accel) and accel > 0)):
        faz = "🌱 Yükseliş başlıyor"
    elif np.isfinite(c20) and c20 > 0 and np.isfinite(c63) and c63 > 0:
        faz = "🚀 Güçlü yükseliş"
    elif np.isfinite(c63) and c63 > 0 and np.isfinite(c20) and c20 <= 0:
        faz = "🌤️ Yükseliş yavaşlıyor"
    elif np.isfinite(c20) and c20 < 0 and np.isfinite(c63) and c63 < 0:
        faz = "🩸 Düşüş"
    else:
        faz = "⚖️ Kararsız"

    q = breakout_info(s, "QE", 8)
    m = breakout_info(s, "ME", 12)
    return {"Son": last, "20G %": c20, "63G %": c63, "Skor": float(score),
            "Risk Etkisi": float(score) * risk if risk else 0.0,
            "Faz": faz, "50G üstü": above50, "200G üstü": above200,
            "İvme": accel, "Çeyreklik Kırılım": q["kirilim"],
            "Taze Çeyreklik Kırılım": q.get("taze", False),
            "Çeyreklik Direnç": q["direnc"], "Dirence Uzaklık %": q["uzaklik"],
            "Aylık Kırılım": m["kirilim"]}


# Her gösterge için "yükseliyorsa / düşüyorsa NE DEMEK" — oran dilinde değil,
# yatırımcı dilinde. Olumlu/olumsuz etiketi PAIRS'taki `risk` yönünden gelir.
MEANING: dict[str, tuple[str, str]] = {
    "QQQ/SPY": ("Nasdaq 100, S&P 500'ü geçiyor — teknoloji/büyüme liderliği sürüyor",
                "Nasdaq 100, S&P 500'ün gerisinde — teknoloji liderliği zayıflıyor"),
    "^IXIC/^GSPC": ("Nasdaq Bileşik, S&P 500'ü geçiyor — büyüme iştahı küçük/orta şirketlere de yayılmış",
                    "Nasdaq Bileşik, S&P 500'ün gerisinde — büyüme hisselerine iştah zayıf"),
    "SMH/SPY": ("Yarı iletkenler piyasayı geçiyor — döngünün öncü sektörü güçlü",
                "Yarı iletkenler piyasanın gerisinde — öncü sektör yoruluyor"),
    "ARKK/QQQ": ("Spekülatif büyüme Nasdaq'ı geçiyor — yatırımcı cesur",
                 "Spekülatif büyüme Nasdaq'ın gerisinde — yatırımcı riskin ucundan kaçıyor"),
    "IWM/SPY": ("Küçük şirketler S&P 500'ü geçiyor — yükseliş tabana yayılıyor",
                "Küçük şirketler S&P 500'ün gerisinde — para büyük ve güvenli şirketlere sığınıyor"),
    "RSP/SPY": ("Ortalama hisse endeksi geçiyor — katılım geniş, yükseliş sağlıklı",
                "Ortalama hisse endeksin gerisinde — yükselişi birkaç dev hisse taşıyor"),
    "XLY/XLP": ("İhtiyari tüketim, temel tüketimi geçiyor — piyasa ofansif modda",
                "Temel tüketim öne geçti — piyasa defansif modda"),
    "EEM/SPY": ("Gelişen piyasalar ABD'yi geçiyor — küresel risk iştahı var",
                "Gelişen piyasalar ABD'nin gerisinde — para ABD'de ve dolarda kalıyor"),
    "SPY/TLT": ("Hisseler tahvili geçiyor — para riskli varlıklara akıyor",
                "Tahvil hisseleri geçiyor — para güvenli limana kaçıyor"),
    "HYG/TLT": ("Riskli şirket tahvili hazineyi geçiyor — kredi piyasası rahat",
                "Hazine riskli şirket tahvilini geçiyor — kredi piyasası tedirgin"),
    "GC=F/SPY": ("Altın hisseleri geçiyor — güvenli liman ya da enflasyon talebi var",
                 "Hisseler altını geçiyor — korku düşük"),
    "SI=F/GC=F": ("Gümüş altını geçiyor — metal rallisi büyüme/spekülatif tarafta",
                  "Altın gümüşü geçiyor — metallerde savunma ağırlıklı talep"),
    "HG=F/GC=F": ("Bakır altını geçiyor — ekonomik büyüme beklentisi artıyor",
                  "Altın bakırı geçiyor — büyüme/resesyon korkusu"),
    "BZ=F": ("Petrol yükseliyor — enflasyon ve jeopolitik baskı artıyor",
             "Petrol düşüyor — enflasyon baskısı azalıyor"),
    "DX-Y.NYB": ("Dolar güçleniyor — küresel likidite daralıyor",
                 "Dolar zayıflıyor — küresel likidite rahatlıyor"),
    "BTC-USD": ("Bitcoin yükseliyor — kriptoya para giriyor",
                "Bitcoin düşüyor — kriptodan para çıkıyor"),
    "TOTAL": ("Toplam kripto piyasa değeri artıyor — kriptoya net para girişi",
              "Toplam kripto piyasa değeri düşüyor — kriptodan net para çıkışı"),
    "BTC.D": ("BTC hakimiyeti artıyor — kripto içinde para altcoinlerden BTC'ye kaçıyor",
              "BTC hakimiyeti düşüyor — kripto içinde para altcoinlere akıyor"),
    "ETH/BTC": ("ETH, BTC'yi geçiyor — altcoin sezonunun ilk basamağı",
                "ETH, BTC'nin gerisinde — altcoin iştahı yok"),
    "TOTAL3": ("Altcoinlerin (BTC ve ETH hariç) değeri artıyor — altcoinlere para giriyor",
               "Altcoinlerin değeri düşüyor — altcoinlerden para çıkıyor"),
    "OTHERS/BTC": ("Küçük altcoinler BTC'yi geçiyor — kriptoda aşırı risk iştahı",
                   "Küçük altcoinler BTC'nin gerisinde — kriptoda temkin"),
    "NET_LIQ": ("Fed net likiditesi artıyor — sisteme para giriyor",
                "Fed net likiditesi azalıyor — sistemden para çekiliyor"),
    "M2SL": ("M2 para arzı büyüyor — ekonomide para miktarı artıyor",
             "M2 para arzı daralıyor — ekonomide para azalıyor"),
}


def flow_verdict(p: dict[str, Any], st: dict[str, Any]) -> str:
    """✅ OLUMLU / ❌ OLUMSUZ / ⚠️ DİKKAT / ➖ NÖTR — risk iştahı açısından."""
    sc = st["Skor"]
    if abs(sc) < 15:
        return "➖ NÖTR"
    if not p["risk"]:                      # petrol: sert yükseliş dikkat
        return "⚠️ DİKKAT" if sc >= 15 else "✅ OLUMLU"
    return "✅ OLUMLU" if sc * p["risk"] > 0 else "❌ OLUMSUZ"


def flow_sentence(p: dict[str, Any], st: dict[str, Any]) -> str:
    """Yatırımcı dilinde tek cümle: ne oluyor ve ne anlama geliyor."""
    sc = st["Skor"]
    up, down = MEANING.get(p["key"], (f"{p['ad']} yükseliyor", f"{p['ad']} düşüyor"))
    if abs(sc) < 15:
        return f"{p['ad']}: belirgin bir hareket yok"
    return up if sc > 0 else down


def rotation_table(series: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for p in PAIRS:
        s = series.get(p["key"])
        if s is None or len(s) < 30:
            continue
        st = series_stats(s, p["risk"])
        okuma = flow_verdict(p, st)
        rows.append({"Anahtar": p["key"], "Grup": p["grup"], "Gösterge": p["ad"],
                     "Akış": flow_sentence(p, st), "Risk Okuması": okuma,
                     **st, "Not": p["not_"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 1) RİSK AÇIK MI?
# --------------------------------------------------------------------------
def _avg(*xs: float) -> float:
    v = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else np.nan


def _durum(x: float) -> str:
    if not np.isfinite(x):
        return "➖"
    return "✅" if x >= 60 else "⚠️" if x >= 40 else "❌"


def risk_pillars(scores: dict[str, float], rot: pd.DataFrame) -> list[dict[str, Any]]:
    """Altı sütun; her biri 0–100 (yüksek = risk iştahı)."""
    def rs(key: str) -> float:
        if rot.empty or key not in set(rot["Anahtar"]):
            return np.nan
        return float(np.clip(50 + rot.loc[rot["Anahtar"] == key, "Risk Etkisi"].iloc[0] / 2,
                             0, 100))

    g = scores.get
    liq = _avg(rs("NET_LIQ"), g("dollar", np.nan))
    pillars = [
        dict(ad="Trend", skor=g("trend", np.nan),
             ne="S&P 500'ün 50 ve 200 günlük ortalamalara göre dizilimi.",
             neden="Fiyat ana trendin üstündeyken alım sinyallerinin isabeti "
                   "belirgin şekilde yüksektir. Trendin altında 'ucuz' görünen "
                   "hisse daha da ucuzlayabilir."),
        dict(ad="Korku (VIX)", skor=_avg(g("vix", np.nan), g("vix_ts", np.nan)),
             ne="VIX seviyesi ve vade yapısı (VIX / VIX3M).",
             neden="VIX 20'nin altında ve vade yapısı normalken piyasa sakin; "
                   "VIX3M'in üstüne çıkan VIX yakın vadeli paniğe işaret eder."),
        dict(ad="Kredi", skor=_avg(g("credit", np.nan), rs("HYG/TLT")),
             ne="Riskli şirket tahvilinin hazineye göre performansı (HYG/TLT).",
             neden="Kredi piyasası hisseden önce uyarır. Borç verenler geri "
                   "çekiliyorsa hisse yükselişi kırılgandır."),
        dict(ad="Likidite", skor=liq,
             ne="Fed net likiditesi (bilanço − TGA − ters repo) ve dolar.",
             neden="Piyasaya giren para artıyorsa varlık fiyatları rüzgârı "
                   "arkasına alır; güçlü dolar ise küresel likiditeyi emer."),
        dict(ad="Katılım", skor=_avg(g("breadth", np.nan), g("smallcap", np.nan)),
             ne="Eşit ağırlıklı S&P ve küçük şirketlerin performansı.",
             neden="Yükselişe geniş katılım varsa sağlıklıdır. Endeksi 5–10 dev "
                   "hisse taşıyorsa, sizin seçtiğiniz hisse büyük ihtimalle "
                   "o rallinin dışında kalır."),
        dict(ad="Spekülatif iştah", skor=_avg(g("crypto", np.nan), rs("ARKK/QQQ"),
                                              rs("OTHERS/BTC")),
             ne="Bitcoin, spekülatif büyüme (ARKK) ve küçük altcoinler.",
             neden="Riskin en uç noktası. Burada iştah varsa yatırımcı cesur; "
                   "yoksa piyasa yalnızca güvenli büyük şirketleri alıyor."),
    ]
    for p in pillars:
        p["durum"] = _durum(p["skor"])
    return pillars


def risk_verdict(pillars: list[dict[str, Any]], regime_label: str = ""
                 ) -> dict[str, Any]:
    w = {"Trend": 0.25, "Korku (VIX)": 0.15, "Kredi": 0.15, "Likidite": 0.15,
         "Katılım": 0.15, "Spekülatif iştah": 0.15}
    tot, ws = 0.0, 0.0
    for p in pillars:
        if np.isfinite(p["skor"]):
            tot += w[p["ad"]] * p["skor"]
            ws += w[p["ad"]]
    score = tot / ws if ws else np.nan
    n_ok = sum(p["durum"] == "✅" for p in pillars)
    n_bad = sum(p["durum"] == "❌" for p in pillars)
    trend_ok = next((p["skor"] for p in pillars if p["ad"] == "Trend"), np.nan)

    if np.isfinite(score) and score >= 60 and (not np.isfinite(trend_ok) or trend_ok >= 50):
        etiket, renk = "🟢 RİSK AÇIK", "pos"
        ozet = ("Piyasa risk almayı ödüllendiriyor. Alım sinyalleri tam "
                "pozisyonla değerlendirilebilir; asıl soru 'nereye' — 2. adıma "
                "geçin.")
        boyut = "Normal pozisyon boyutu"
    elif np.isfinite(score) and (score <= 40 or n_bad >= 4):
        etiket, renk = "🔴 RİSK KAPALI", "neg"
        ozet = ("Piyasa riskten kaçıyor. Yeni alımlar düşük isabetli olur; "
                "nakit, kısa vade ve koruma öncelikli. İstisna: 2. adımda "
                "güçlü akış alan defansif segmentler.")
        boyut = "Yeni pozisyon yok ya da çeyrek boyut"
    else:
        etiket, renk = "🟡 TEMKİNLİ", ""
        ozet = ("Göstergeler karışık. Sadece en güçlü kurulumlar, küçük "
                "pozisyon ve sıkı stop. Zayıf sütunları aşağıda görün — "
                "onlar düzelince risk tam açılır.")
        boyut = "Yarım pozisyon"
    return {"etiket": etiket, "renk": renk, "skor": score, "ozet": ozet,
            "boyut": boyut, "n_ok": n_ok, "n_bad": n_bad, "rejim": regime_label}


# --------------------------------------------------------------------------
# 2) PARA NEREYE AKIYOR? — segment kararları
# --------------------------------------------------------------------------
def segment_verdicts(rot: pd.DataFrame) -> pd.DataFrame:
    if rot.empty:
        return pd.DataFrame()
    by = rot.set_index("Anahtar")
    rows = []
    for seg, cfg in SEGMENTS.items():
        vals, detay = [], []
        for k in cfg["pairs"]:
            if k not in by.index:
                continue
            r = by.loc[k]
            sign = SEGMENT_ABS.get(seg, {}).get(k, PAIR[k]["risk"] or 1)
            v = float(r["Skor"]) * sign
            if seg in SEGMENT_INVERT:
                v = -v
            vals.append(v)
            detay.append(f"{PAIR[k]['ad']}: {r['Faz']}")
        if not vals:
            continue
        sc = float(np.mean(vals))
        erken = any(by.loc[k, "Faz"] == "🌱 Yükseliş başlıyor"
                    for k in cfg["pairs"] if k in by.index)
        if sc >= 25:
            k_ = "🟢 İştahlı"
        elif sc >= 5:
            k_ = "🟡 Isınıyor" if erken else "🟡 Hafif olumlu"
        elif sc > -10:
            k_ = "⚪ Nötr" + (" (erken dönüş sinyali var)" if erken else "")
        else:
            k_ = "🔴 İştahsız"
        rows.append({"Segment": seg, "Karar": k_, "Skor": sc,
                     "Erken Sinyal": "🌱" if erken else "",
                     "Dayanak": " · ".join(detay), "Kapsam": cfg["aciklama"]})
    return pd.DataFrame(rows).sort_values("Skor", ascending=False)


def money_flow_summary(rot: pd.DataFrame, top: int = 4) -> list[str]:
    """En güçlü akışları sade cümlelere çevirir."""
    if rot.empty:
        return []
    r = rot.copy()
    r["abs"] = r["Skor"].abs()
    r = r[r["abs"] >= 25].sort_values("abs", ascending=False)
    out = [f"**{row['Risk Okuması']}** — {row['Akış']}"
           for _, row in r.head(top).iterrows()]
    erken = rot[rot["Faz"] == "🌱 Yükseliş başlıyor"]
    for _, row in erken.head(3).iterrows():
        p = PAIR[row["Anahtar"]]
        etiket = ("✅ olumlu" if p["risk"] > 0 else "❌ olumsuz" if p["risk"] < 0
                  else "⚠️ dikkat")
        anlam = MEANING.get(row["Anahtar"], (f"{p['ad']} yükseliyor", ""))[0]
        out.append(f"🌱 **Yeni yükseliş başlıyor ({etiket})** — {anlam} "
                   f"(son 20 gün yukarı, 3 aylık trend henüz aşağı)")
    kir = rot[rot["Taze Çeyreklik Kırılım"] == True]  # noqa: E712
    for _, row in kir.iterrows():
        out.append(f"📐 {row['Gösterge']} çeyreklik grafikte direnci kırdı "
                   f"(önceki 8 çeyreğin zirvesi)")
    return out


# --------------------------------------------------------------------------
# 3) HANGİ TEMA İVMELENİYOR? — göreli rotasyon grafiği (RRG)
# --------------------------------------------------------------------------
def rrg_lines(close: pd.Series, bench: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    RS-Oranı: (fiyat / SPY) oranının 50 günlük ortalamasına göre konumu ×100.
      100'ün üstü = endeksten güçlü.
    RS-Momentum: RS-Oranının 10 gün önceye göre değişimi ×100.
      100'ün üstü = göreli güç artıyor.
    """
    j = pd.concat([close, bench], axis=1, join="inner").dropna()
    rs = j.iloc[:, 0] / j.iloc[:, 1]
    ratio = (100 * rs / rs.rolling(50).mean()).ewm(span=5, adjust=False).mean()
    mom = (100 * ratio / ratio.shift(10)).ewm(span=3, adjust=False).mean()
    return ratio, mom


RRG_Q = {
    "lider": ("🚀 Lider", "Endeksten güçlü ve güçlenmeye devam ediyor."),
    "zayif": ("🌤️ Yoruluyor", "Hâlâ endeksten güçlü ama göreli güç azalıyor — "
                             "kâr realizasyonu bölgesi."),
    "geride": ("🩸 Geride", "Endeksten zayıf ve zayıflamaya devam ediyor."),
    "iyilesen": ("🌱 İyileşiyor", "Endeksten zayıf AMA göreli güç artıyor — "
                                 "akıllı paranın erken girdiği bölge."),
}


def rrg_quadrant(r: float, m: float) -> str:
    if not (np.isfinite(r) and np.isfinite(m)):
        return "geride"
    if r >= 100 and m >= 100:
        return "lider"
    if r >= 100:
        return "zayif"
    if m >= 100:
        return "iyilesen"
    return "geride"


def theme_index(prices: dict[str, pd.DataFrame], members: list[str]
                ) -> tuple[pd.Series | None, pd.Series | None]:
    """Eşit ağırlıklı tema endeksi (her ETF başlangıçta 1'e normalize) ve
    toplam dolar hacmi."""
    cl, dv = [], []
    for t in members:
        df = prices.get(t)
        if df is None or df.empty or len(df) < 80:
            continue
        c = df["Close"].dropna()
        cl.append(c / c.iloc[0])
        dv.append((df["Close"] * df["Volume"]).rename(t))
    if not cl:
        return None, None
    idx = pd.concat(cl, axis=1).ffill().mean(axis=1)
    vol = pd.concat(dv, axis=1).sum(axis=1, min_count=1)
    return idx, vol


def early_score(ratio: pd.Series, mom: pd.Series, vol: pd.Series | None,
                close: pd.Series) -> dict[str, Any]:
    r, m = float(ratio.iloc[-1]), float(mom.iloc[-1])
    q = rrg_quadrant(r, m)
    m_up = float(mom.iloc[-1] - mom.iloc[-6]) if len(mom) > 6 else np.nan
    r_up = float(ratio.iloc[-1] - ratio.iloc[-6]) if len(ratio) > 6 else np.nan
    # liderliğe yeni mi geçti? (son 10 günde iyileşen → lider)
    qs = [rrg_quadrant(a, b) for a, b in zip(ratio.tail(11), mom.tail(11))]
    taze_lider = q == "lider" and any(x in ("iyilesen", "geride") for x in qs[:-1])

    vol_ok = False
    vr = np.nan
    if vol is not None and len(vol.dropna()) > 70:
        v10 = vol.tail(10).mean()
        v60 = vol.tail(70).head(60).mean()
        vr = float(v10 / v60) if v60 else np.nan
        vol_ok = bool(np.isfinite(vr) and vr >= 1.15)

    s = {"iyilesen": 45, "lider": 30, "zayif": 8, "geride": 0}[q]
    if q == "iyilesen" and np.isfinite(m_up) and m_up > 0:
        s += 15
    if taze_lider:
        s += 25
    if np.isfinite(r_up) and r_up > 0:
        s += 10
    if vol_ok:
        s += 15
    c5 = _pct(close, 5)
    if np.isfinite(c5) and c5 > 0:
        s += 5
    return {"RS-Oran": r, "RS-Mom": m, "Bölge": RRG_Q[q][0], "_q": q,
            "Taze Lider": taze_lider, "Hacim Oranı": vr, "Hacim Onayı": vol_ok,
            "Erken Skor": int(np.clip(s, 0, 100)),
            "1H %": c5, "1A %": _pct(close, 21), "3A %": _pct(close, 63)}


def theme_rotation(prices: dict[str, pd.DataFrame], themes: dict[str, list[str]],
                   bench: str = "SPY") -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Tema bazında RRG + erken akış skoru. Döner: (tablo, {tema: kuyruk})."""
    b = prices.get(bench)
    if b is None or b.empty:
        return pd.DataFrame(), {}
    bc = b["Close"].dropna()
    rows, tails = [], {}
    for tema, members in themes.items():
        idx, vol = theme_index(prices, members)
        if idx is None:
            continue
        ratio, mom = rrg_lines(idx, bc)
        if ratio.dropna().empty or mom.dropna().empty:
            continue
        e = early_score(ratio.dropna(), mom.dropna(), vol, idx)
        rows.append({"Tema": tema, **e, "ETF'ler": ", ".join(members)})
        t = pd.DataFrame({"RS-Oran": ratio, "RS-Mom": mom}).dropna()
        tails[tema] = t.iloc[::-5].head(6).iloc[::-1]   # son 6 hafta
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Erken Skor", ascending=False).reset_index(drop=True)
    return df, tails


def etf_rotation(prices: dict[str, pd.DataFrame], etfs: list[str],
                 bench: str = "SPY") -> pd.DataFrame:
    b = prices.get(bench)
    if b is None or b.empty:
        return pd.DataFrame()
    bc = b["Close"].dropna()
    rows = []
    for t in etfs:
        df = prices.get(t)
        if df is None or len(df) < 80:
            continue
        c = df["Close"].dropna()
        ratio, mom = rrg_lines(c, bc)
        if ratio.dropna().empty or mom.dropna().empty:
            continue
        e = early_score(ratio.dropna(), mom.dropna(), df["Close"] * df["Volume"], c)
        rows.append({"ETF": t, **e})
    out = pd.DataFrame(rows)
    return out.sort_values("Erken Skor", ascending=False) if not out.empty else out


# --------------------------------------------------------------------------
# 4) BU TEMADA HANGİ HİSSE?
# --------------------------------------------------------------------------
BULL = {"💎 DIAMOND AL", "⭐ GOLDEN STAR", "🎣 LİKİDİTE SÜPÜRMESİ", "🐋 TOPLAMA",
        "🚀 AFTERBURNER", "🟢 GİRİŞ BÖLGESİ", "📈 MINERVINI MVP",
        "🎯 SIKIŞMA PATLADI", "⚡ PRO ↗ RETAIL"}
BEAR = {"🩸 GÜÇLÜ RİSK", "⛔ DIAMOND SAT", "🐋 DAĞITIM", "🔻 AFTERBURNER AYI",
        "⚡ PRO ↘ RETAIL"}

STOCK_CLASSES = {
    "🎯 Alım adayı": "Alım sinyali var, temasından güçlü ve kurumsal para "
                    "giriyor. Giriş planı yapılabilir.",
    "🌱 Geride kaldı, toparlanıyor": "Son bir ayda temasının gerisinde ama son "
                                     "hafta göreli gücü dönmüş ve whale "
                                     "artıyor — yetişme (catch-up) adayı.",
    "🚀 Lider": "Temasını sürükleyen hisse. Trend devam ediyor; yeni giriş "
               "için geri çekilme beklemek daha iyi risk/ödül verir.",
    "🔥 Lider ama uzamış": "Lider fakat tükenme işareti var ya da çok hızlı "
                          "yükselmiş. Kovalamayın.",
    "⚪ İzle": "Belirgin bir avantajı yok; izleme listesinde tutun.",
    "⛔ Uzak dur": "Dağıtım, risk ya da satış sinyali var.",
    "📅 Bilanço yakın": "7 gün içinde bilanço var — gap riski stop mantığını "
                       "bozar. Bilanço sonrasına bırakın.",
    "💧 Likidite düşük": "Günlük işlem hacmi eşiğin altında; spread geniş, "
                        "stop kayar.",
}


def classify_stock(d: dict[str, Any], min_liq_m: float = 10.0) -> tuple[str, list[str]]:
    """Döner: (sınıf, [gerekçe parçaları])."""
    why: list[str] = []
    sig = d.get("Sinyal", "")
    rs1a, rs1h = d.get("Temaya Göre 1A", np.nan), d.get("Temaya Göre 1H", np.nan)
    dw5 = d.get("ΔWHALE 5B", np.nan)
    liq = d.get("Hacim ($M)", np.nan)
    kg = pd.to_numeric(d.get("Kalan Gün"), errors="coerce")
    dsv = d.get("ΔSV pp", np.nan)
    p1h = d.get("1 Hafta %", np.nan)

    if np.isfinite(rs1a):
        why.append(f"temaya göre {rs1a:+.1f}% (1A)")
    if np.isfinite(dw5):
        why.append("whale ↑" if dw5 > 1 else "whale ↓" if dw5 < -1 else "whale yatay")
    if np.isfinite(dsv):
        if dsv <= -3:
            why.append(f"short azalıyor ({dsv:+.1f}p)")
        elif dsv >= 3:
            why.append(f"short artıyor ({dsv:+.1f}p)"
                       + (" — fiyat düşerken" if np.isfinite(p1h) and p1h < 0 else ""))
    if np.isfinite(kg) and kg >= 0:
        why.append(f"bilanço {int(kg)} gün")
    if sig and sig != "⚪ BEKLE":
        why.append(sig)

    if np.isfinite(liq) and liq < min_liq_m:
        return "💧 Likidite düşük", [f"günlük ${liq:.1f}M"] + why
    if np.isfinite(kg) and 0 <= kg <= 7:
        return "📅 Bilanço yakın", why
    if sig in BEAR:
        return "⛔ Uzak dur", why
    short_bad = np.isfinite(dsv) and dsv >= 3 and np.isfinite(p1h) and p1h < 0
    if (sig in BULL and d.get("Skor", 0) >= 50
            and (not np.isfinite(rs1h) or rs1h > 0) and not short_bad):
        return "🎯 Alım adayı", why
    if (np.isfinite(rs1a) and rs1a < 0 and np.isfinite(rs1h) and rs1h > 0
            and np.isfinite(dw5) and dw5 > 0):
        return "🌱 Geride kaldı, toparlanıyor", why
    if np.isfinite(rs1a) and rs1a > 0:
        if d.get("_exhausted") or sig == "⚠️ TÜKENME" or (
                np.isfinite(d.get("1 Ay %", np.nan)) and d.get("1 Ay %") > 35):
            return "🔥 Lider ama uzamış", why
        return "🚀 Lider", why
    return "⚪ İzle", why


CLASS_ORDER = ["🎯 Alım adayı", "🌱 Geride kaldı, toparlanıyor", "🚀 Lider",
               "🔥 Lider ama uzamış", "⚪ İzle", "📅 Bilanço yakın",
               "💧 Likidite düşük", "⛔ Uzak dur"]


def funnel_score(d: dict[str, Any]) -> int:
    """0–100: swing skoru + temaya göre güç + whale + short akışı."""
    s = 0.55 * float(d.get("Skor", 0) or 0)
    rs = d.get("Temaya Göre 1A", np.nan)
    if np.isfinite(rs):
        s += 15 * np.tanh(rs / 10) + 10
    rs1h = d.get("Temaya Göre 1H", np.nan)
    if np.isfinite(rs1h):
        s += 8 * np.tanh(rs1h / 4)
    dw = d.get("ΔWHALE 5B", np.nan)
    if np.isfinite(dw):
        s += 8 * np.tanh(dw / 8)
    dsv = d.get("ΔSV pp", np.nan)
    if np.isfinite(dsv):
        s -= 6 * np.tanh(dsv / 5)
    return int(np.clip(round(s), 0, 100))


def build_stock_table(scan_df: pd.DataFrame, theme_close: pd.Series | None,
                      prices: dict[str, pd.DataFrame] | None = None,
                      earn: pd.DataFrame | None = None,
                      short: pd.DataFrame | None = None,
                      min_liq_m: float = 10.0, swing_score=None) -> pd.DataFrame:
    """Tarama tablosunu tema içi göreli güç, bilanço ve short ile birleştirir."""
    if scan_df is None or scan_df.empty:
        return pd.DataFrame()
    df = scan_df.copy()
    if "Sinyal" in df.columns:
        df = df[df["Sinyal"] != "⚫ VERİ YOK"].copy()
    if df.empty:
        return df
    th1a = _pct(theme_close, 21) if theme_close is not None else np.nan
    th1h = _pct(theme_close, 5) if theme_close is not None else np.nan
    df["Temaya Göre 1A"] = pd.to_numeric(df.get("1 Ay %"), errors="coerce") - th1a
    df["Temaya Göre 1H"] = pd.to_numeric(df.get("1 Hafta %"), errors="coerce") - th1h

    if prices:
        hi = {}
        for t in df["Sembol"]:
            p = prices.get(t)
            if p is not None and len(p) > 20:
                c = p["Close"].dropna()
                hi[t] = (c.iloc[-1] / c.tail(252).max() - 1) * 100
        df["Zirveye Uzaklık %"] = df["Sembol"].map(hi)

    if earn is not None and not earn.empty:
        keep = [c for c in ("Hisse", "Kalan Gün", "Bilanço", "P/S Durum",
                            "Hedef", "Potansiyel %") if c in earn.columns]
        e = earn[keep].rename(columns={"Hisse": "Sembol"})
        df = df.merge(e, on="Sembol", how="left")
    if short is not None and not short.empty:
        keep = [c for c in ("Sembol", "SV% 5G", "ΔSV pp") if c in short.columns]
        df = df.merge(short[keep], on="Sembol", how="left")

    if swing_score is not None:
        df["Skor"] = [swing_score(r) for r in df.to_dict("records")]
    cls, why = [], []
    for r in df.to_dict("records"):
        c, w = classify_stock(r, min_liq_m)
        cls.append(c)
        why.append(" · ".join(w))
    df["Durum"] = cls
    df["Neden"] = why
    df["Huni Skoru"] = [funnel_score(r) for r in df.to_dict("records")]
    df["_ord"] = df["Durum"].map({c: i for i, c in enumerate(CLASS_ORDER)})
    return (df.sort_values(["_ord", "Huni Skoru"], ascending=[True, False])
              .drop(columns=["_ord"]).reset_index(drop=True))
