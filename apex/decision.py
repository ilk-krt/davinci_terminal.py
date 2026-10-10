# apex/decision.py — AETHER APEX
from __future__ import annotations

from apex.indicators import atr, ema, rsi
from apex import engine as eng
from apex import funnel as fnl

# apex/decision.py — ENDEKS KARARI: S&P 500 · Nasdaq · Kripto
#
# Her varlık için günlük, haftalık ve aylık mumlarda bütün modüller tek tek
# değerlendirilir (+1 olumlu · 0 nötr · −1 olumsuz). Zaman dilimi skoru
# modüllerin ortalamasıdır; nihai karar haftalık ağırlıklıdır (swing).
# Piyasa notları (VIX kuralı, genişlik, net giriş / rotasyon, QQQ/SPY
# çeyreklik kırılımı) kararın yanında ayrıca gösterilir.
#
# Pine kodu gelince eklenecek modüller PENDING listesindedir.

from typing import Any

import numpy as np
import pandas as pd

DEC_ASSETS: dict[str, str] = {"S&P 500": "SPY", "Nasdaq": "QQQ",
                              "Kripto (BTC)": "BTC-USD"}
SECTORS: dict[str, str] = {
    "XLK": "Teknoloji", "XLF": "Finans", "XLE": "Enerji", "XLV": "Sağlık",
    "XLI": "Sanayi", "XLY": "İhtiyari tüketim", "XLP": "Temel tüketim",
    "XLU": "Kamu hizmetleri", "XLB": "Materyal", "XLRE": "Gayrimenkul",
    "XLC": "İletişim"}
DEC_TICKERS: list[str] = sorted(set(list(DEC_ASSETS.values())
                                    + ["^VIX", "RSP", "IWM"] + list(SECTORS)))
TFS: dict[str, str | None] = {"Günlük": None, "Haftalık": "W-FRI", "Aylık": "ME"}
TF_W: dict[str, float] = {"Günlük": 0.3, "Haftalık": 0.4, "Aylık": 0.3}

PENDING: list[str] = [
    "Fibonacci seviyeleri", "Elliott dalga", "Q-dry", "Volatility Hole",
    "Whale Momentum Profile", "Gap / Fair Value Gap", "Kurumsal alım-satım bölgeleri",
    "Kısa/orta/uzun vade dirençler", "GFR", "VSA & Delta", "Mum Gücü (v666)"]


# --------------------------------------------------------------------------
# Yardımcılar
# --------------------------------------------------------------------------
def resample(df: pd.DataFrame, rule: str | None) -> pd.DataFrame:
    if rule is None or df is None or df.empty:
        return df
    d = df.copy()
    d.index = pd.to_datetime(d.index)
    try:
        d.index = d.index.tz_localize(None)
    except TypeError:
        pass
    return d.resample(rule).agg({"Open": "first", "High": "max", "Low": "min",
                                 "Close": "last", "Volume": "sum"}).dropna(
        subset=["Close"])


def _m(modul: str, durum: float, notu: str, grup: str = "") -> dict[str, Any]:
    return {"Modül": modul, "Durum": float(np.clip(durum, -1, 1)), "Not": notu,
            "Grup": grup}


def _icon(x: float) -> str:
    if not np.isfinite(x):
        return "➖"
    return "✅" if x >= 0.5 else "🟢" if x > 0 else "❌" if x <= -0.5 else \
        "🔴" if x < 0 else "⚪"


# --------------------------------------------------------------------------
# Modüller — tek zaman dilimi
# --------------------------------------------------------------------------
def candle_module(df: pd.DataFrame) -> dict[str, Any]:
    o, h, l, c = (float(df[k].iloc[-1]) for k in ("Open", "High", "Low", "Close"))
    rng = max(h - l, 1e-12)
    body = abs(c - o) / rng
    up_w = (h - max(o, c)) / rng
    lo_w = (min(o, c) - l) / rng
    a = float(atr(df["High"], df["Low"], df["Close"], 14).iloc[-1])
    strength = rng / a if a else np.nan
    green = c >= o
    renk = "yeşil" if green else "kırmızı"
    parts = [f"{renk}, gövde %{body * 100:.0f}"]
    s = 0.0
    if body < 0.15:
        parts.append("doji — kararsızlık")
    elif body >= 0.55 and np.isfinite(strength) and strength >= 1.0:
        s = 1.0 if green else -1.0
        parts.append(f"güçlü ({strength:.1f}× ATR)")
    else:
        s = 0.4 if green else -0.4
    if up_w >= 0.45:
        s -= 0.5
        parts.append(f"uzun üst fitil %{up_w * 100:.0f} — satış baskısı")
    if lo_w >= 0.45:
        s += 0.5
        parts.append(f"uzun alt fitil %{lo_w * 100:.0f} — alıcı savunması")
    return _m("Mum", s, " · ".join(parts), "Fiyat")


def partial_frac(daily: pd.DataFrame, rule: str | None) -> float:
    """Henüz kapanmamış haftalık/aylık mumun ne kadarının geçtiği (0–1)."""
    if rule is None or daily is None or daily.empty:
        return 1.0
    idx = pd.to_datetime(daily.index)
    try:
        idx = idx.tz_localize(None)
    except TypeError:
        pass
    last = idx[-1]
    crypto = (idx.dayofweek >= 5).any()          # hafta sonu işlem → kripto
    if rule.startswith("W"):
        start = last - pd.Timedelta(days=last.dayofweek)
        n = int(((idx >= start.normalize()) & (idx <= last)).sum())
        return min(1.0, n / (7 if crypto else 5))
    start = last.replace(day=1)
    n = int(((idx >= start.normalize()) & (idx <= last)).sum())
    total = (last.days_in_month if crypto else
             len(pd.bdate_range(start, start + pd.offsets.MonthEnd(0))))
    return min(1.0, n / max(total, 1))


def volume_module(df: pd.DataFrame, frac: float = 1.0) -> dict[str, Any]:
    v = df["Volume"]
    if v.tail(25).sum() <= 0:
        return _m("Hacim", 0, "hacim verisi yok", "Fiyat")
    avg = float(v.iloc[-21:-1].mean())
    # kapanmamış dönem: şimdiye kadarki hacim, geçen süreye göre oranlanır
    rv = float(v.iloc[-1] / max(frac, 0.05) / avg) if avg else np.nan
    chg = float(df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1) * 100
    if not np.isfinite(rv):
        return _m("Hacim", 0, "—", "Fiyat")
    if chg > 0 and rv >= 1.2:
        return _m("Hacim", 1, f"hacimli yükseliş ({rv:.1f}× ort.)", "Fiyat")
    if chg > 0 and rv < 0.8:
        return _m("Hacim", -0.5, f"yükseliş ama hacim ortalamanın altında "
                                  f"({rv:.1f}×) — katılım zayıf", "Fiyat")
    if chg < 0 and rv >= 1.2:
        return _m("Hacim", -1, f"hacimli düşüş ({rv:.1f}× ort.) — dağıtım riski",
                  "Fiyat")
    if chg < 0 and rv < 0.8:
        return _m("Hacim", 0.3, f"hacimsiz geri çekilme ({rv:.1f}×) — satıcı "
                                f"istekli değil", "Fiyat")
    return _m("Hacim", 0, f"normal ({rv:.1f}× ort.)", "Fiyat")


def trend_module(c: pd.Series) -> dict[str, Any]:
    e21, e50, e200 = (float(ema(c, n).iloc[-1]) for n in (21, 50, 200))
    p = float(c.iloc[-1])
    if len(c) < 200:
        e200 = np.nan
    if p > e21 > e50 and (not np.isfinite(e200) or e50 > e200):
        return _m("Trend", 1, "boğa dizilimi (fiyat > EMA21 > EMA50 > EMA200)",
                  "Trend")
    if p < e21 < e50 and (not np.isfinite(e200) or e50 < e200):
        return _m("Trend", -1, "ayı dizilimi (fiyat < EMA21 < EMA50 < EMA200)",
                  "Trend")
    if np.isfinite(e200) and p > e200:
        return _m("Trend", 0.3, "karışık — ana trend (EMA200) yukarı, kısa "
                                "vade düzeltmede", "Trend")
    return _m("Trend", -0.3, "karışık — ana trendin altında, kısa vade "
                             "toparlanma denemesi", "Trend")


def ema_module(c: pd.Series) -> dict[str, Any]:
    p = float(c.iloc[-1])
    parts, above = [], 0
    for n in (21, 50, 100, 200):
        if len(c) < n + 5:
            continue
        e = ema(c, n)
        d_now = (p / float(e.iloc[-1]) - 1) * 100
        d_prev = (float(c.iloc[-4]) / float(e.iloc[-4]) - 1) * 100
        yon = "yaklaşıyor" if abs(d_now) < abs(d_prev) else "uzaklaşıyor"
        above += d_now > 0
        parts.append(f"E{n} {d_now:+.1f}% ({yon})")
    n_ = len(parts)
    s = (above / n_ * 2 - 1) if n_ else 0
    return _m("EMA 21/50/100/200", s, " · ".join(parts), "Trend")


def rsi_module(c: pd.Series) -> dict[str, Any]:
    r = rsi(c, 14)
    v, d = float(r.iloc[-1]), float(r.iloc[-1] - r.iloc[-4])
    yon = "yükseliyor" if d > 0 else "düşüyor"
    if v >= 70:
        s, t = -0.5, f"{v:.0f} aşırı alım ({yon})"
    elif v <= 30:
        s, t = 0.5, f"{v:.0f} aşırı satım ({yon}) — dip bölgesi"
    elif v >= 50:
        s = 1 if d > 0 else 0.3
        t = f"{v:.0f} boğa bölgesi, {yon}" + (" — 70'e yaklaşıyor" if v > 62 and d > 0
                                                else "")
    else:
        s = -1 if d < 0 else -0.3
        t = f"{v:.0f} ayı bölgesi, {yon}" + (" — 30'a yaklaşıyor" if v < 38 and d < 0
                                               else "")
    return _m("RSI (14)", s, t, "Momentum")


def engine_modules(d: dict[str, Any]) -> list[dict[str, Any]]:
    """eng.analyze() çıktısından Pine modüllerinin okuması."""
    out = []
    w, dw = d.get("WHALE", np.nan), d.get("ΔWHALE 5B", np.nan)
    pr = d.get("PRO-RET", np.nan)
    if np.isfinite(w):
        s = (1 if w >= 55 and dw > 0 else -1 if w <= 45 and dw < 0
             else 0.4 if w >= 50 else -0.4)
        out.append(_m("Whale / Retail", s,
                      f"WHALE {w:.0f} ({'+' if dw >= 0 else ''}{dw:.1f} / 5 bar) · "
                      f"PRO−RETAIL {pr:+.0f}" if np.isfinite(pr) else f"WHALE {w:.0f}",
                      "Akıllı para"))
    ef = d.get("Efor", "➖")
    out.append(_m("Efor çizgisi", 1 if ("POZ" in ef or "KIRILIM" in ef)
                  else -1 if "NEG" in ef else 0,
                  {"🚀 EFOR KIRILIMI": "fiyat efor çizgisini yukarı kırdı",
                   "🟢 POZ": "fiyat efor çizgisinin üstünde",
                   "🔴 NEG": "fiyat efor çizgisinin altında"}.get(ef, ef),
                  "Akıllı para"))
    mag, dr = d.get("MAGNITUDE", 0), d.get("DIRECTION", 0)
    out.append(_m("Konfluans", 1 if dr >= 3 else -1 if dr <= -3 else dr / 3,
                  f"DIRECTION {dr:+d} · MAGNITUDE {mag}", "Momentum"))
    fu, sy = d.get("Fusion", np.nan), d.get("Synergy", np.nan)
    if np.isfinite(fu):
        out.append(_m("Fusion histogram", np.tanh(fu / (abs(fu) + 1e-9)) if fu else 0,
                      f"{fu:+.2f} ({'pozitif' if fu > 0 else 'negatif'})", "Momentum"))
    if np.isfinite(sy):
        out.append(_m("Synergy histogram", 1 if sy > 0 else -1 if sy < 0 else 0,
                      f"{sy:+.2f} ({'pozitif' if sy > 0 else 'negatif'})", "Momentum"))
    om, dom = d.get("OMNI", np.nan), d.get("ΔOMNI 5B", np.nan)
    if np.isfinite(om):
        s = (1 if om >= 55 and dom > 0 else -1 if om <= 45 and dom < 0
             else 0.4 if om >= 50 else -0.4)
        out.append(_m("Omni momentum", s, f"{om:.0f} ({dom:+.1f} / 5 bar) · "
                                          f"boğa {d.get('Boğa /6', 0)}/6",
                      "Momentum"))
    # sıkışma, tükenme, toplama/dağıtım
    if d.get("_sqz_fire"):
        out.append(_m("Sıkışma", 0.5, "sıkışma bu bar patladı", "Volatilite"))
    elif d.get("Sıkışma"):
        out.append(_m("Sıkışma", 0, f"sıkışma sürüyor ({d.get('Sıkışma Süre', 0)} "
                                    f"bar) — sert hareket yakın, yönü bekleyin",
                      "Volatilite"))
    else:
        out.append(_m("Sıkışma", 0, "sıkışma yok", "Volatilite"))
    if d.get("_exhausted"):
        out.append(_m("Tükenme", -0.7, "tükenme işareti — kovalamayın", "Volatilite"))
    if d.get("_dist") or d.get("_st_out"):
        out.append(_m("Toplama / dağıtım", -1, "kurumsal dağıtım izi", "Akıllı para"))
    elif d.get("_acc") or d.get("_st_in"):
        out.append(_m("Toplama / dağıtım", 1, "kurumsal toplama izi", "Akıllı para"))
    sig = d.get("Sinyal", "⚪ BEKLE")
    bull = sig in ("💎 DIAMOND AL", "⭐ GOLDEN STAR", "🎣 LİKİDİTE SÜPÜRMESİ",
                   "🐋 TOPLAMA", "🚀 AFTERBURNER", "🟢 GİRİŞ BÖLGESİ",
                   "📈 MINERVINI MVP", "🎯 SIKIŞMA PATLADI", "⚡ PRO ↗ RETAIL")
    bear = sig in ("🩸 GÜÇLÜ RİSK", "⛔ DIAMOND SAT", "🐋 DAĞITIM",
                   "🔻 AFTERBURNER AYI", "⚡ PRO ↘ RETAIL")
    out.append(_m("Başlık sinyali", 1 if bull else -1 if bear else 0, sig,
                  "Akıllı para"))
    return out


def evaluate_tf(df: pd.DataFrame, bench: pd.Series | None, t: str,
                frac: float = 1.0) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Tek varlık + tek zaman dilimi → modül tablosu ve motor çıktısı."""
    if df is None or len(df) < 30:
        return pd.DataFrame(), {}
    c = df["Close"]
    mods = [candle_module(df), volume_module(df, frac), trend_module(c),
            ema_module(c), rsi_module(c)]
    d: dict[str, Any] = {}
    if len(df) >= 60:
        row = eng.analyze(df, t, bench_close=bench)
        if row.ok:
            d = row.data
            mods += engine_modules(d)
    return pd.DataFrame(mods), d


def tf_score(mods: pd.DataFrame) -> float:
    if mods is None or mods.empty:
        return np.nan
    return float(mods["Durum"].mean() * 100)


# --------------------------------------------------------------------------
# Piyasa notları
# --------------------------------------------------------------------------
def vix_rule(vix: pd.Series) -> dict[str, Any]:
    v = float(vix.iloc[-1])
    if v >= 45:
        return {"etiket": "🟢🟢 DAHA FAZLA AL", "skor": 1.0,
                "not": f"VIX {v:.1f} ≥ 45 — panik zirvesi. Kural: alımı artır."}
    if v >= 30:
        return {"etiket": "🟢 AL", "skor": 0.7,
                "not": f"VIX {v:.1f} ≥ 30 — korku yüksek. Kural: kademeli al."}
    if v <= 14:
        return {"etiket": "🔴 SAT / KÂR AL", "skor": -0.7,
                "not": f"VIX {v:.1f} ≤ 14 — rehavet. Kural: kâr al, yeni alımda "
                       f"temkin."}
    return {"etiket": "⚪ NÖTR", "skor": 0.0,
            "not": f"VIX {v:.1f} — kural bölgelerinin (≤14 / ≥30) arasında."}


def market_notes(prices: dict[str, pd.DataFrame], rot: pd.DataFrame
                 ) -> list[tuple[str, str]]:
    """(ikon, cümle) listesi — kurallarla üretilen kısa piyasa notları."""
    out: list[tuple[str, str]] = []
    cl = {t: df["Close"].dropna() for t, df in prices.items() if df is not None
          and not df.empty}
    vol = {t: df["Volume"] for t, df in prices.items() if df is not None
           and not df.empty}

    # 1) Endeks hareketi + hacim
    for t, ad in (("QQQ", "Nasdaq 100"), ("SPY", "S&P 500")):
        if t not in cl or len(cl[t]) < 25:
            continue
        ch = (cl[t].iloc[-1] / cl[t].iloc[-2] - 1) * 100
        rv = float(vol[t].iloc[-1] / vol[t].iloc[-21:-1].mean())
        if ch > 0.3 and rv < 0.85:
            out.append(("⚠️", f"{ad} %{ch:+.1f} yükseldi ama hacim ortalamanın "
                              f"altında ({rv:.1f}×) — yükselişin arkasında güçlü "
                              f"katılım yok."))
        elif ch > 0.3 and rv >= 1.2:
            out.append(("✅", f"{ad} %{ch:+.1f} yükseldi, hacim ortalamanın "
                              f"{rv:.1f} katı — güçlü, katılımlı hareket."))
        elif ch < -0.3 and rv >= 1.2:
            out.append(("❌", f"{ad} %{ch:+.1f} düştü, hacim ortalamanın "
                              f"{rv:.1f} katı — kurumsal satış olabilir."))

    # 2) Net giriş mi rotasyon mu? (son gün sektörler)
    secs = [s for s in SECTORS if s in cl and len(cl[s]) > 25]
    if len(secs) >= 8 and "SPY" in cl:
        chg = {s: (cl[s].iloc[-1] / cl[s].iloc[-2] - 1) * 100 for s in secs}
        rvs = {s: float(vol[s].iloc[-1] / vol[s].iloc[-21:-1].mean()) for s in secs}
        up = sum(v > 0 for v in chg.values())
        sharp_down = [s for s, v in chg.items() if v <= -1.0]
        spy_ch = (cl["SPY"].iloc[-1] / cl["SPY"].iloc[-2] - 1) * 100
        avg_rv = float(np.mean(list(rvs.values())))
        lead = max(chg, key=chg.get)
        if spy_ch > 0.2 and len(sharp_down) <= 1 and up >= 7:
            istisna = (f" (yalnızca {SECTORS[sharp_down[0]]} sert düştü)"
                       if sharp_down else ", hiçbir sektörde sert satış yok")
            hacim = (f"ortalama hacim {avg_rv:.1f}×" if avg_rv >= 0.9 else
                     f"ama ortalama hacim düşük ({avg_rv:.1f}×) — giriş zayıf")
            out.append(("✅" if avg_rv >= 0.9 else "⚪",
                        f"{up}/{len(secs)} sektör yükseldi{istisna}; {hacim}. "
                        f"Bu bir rotasyon değil, piyasaya net para girişi. Öncü: "
                        f"{SECTORS[lead]} (%{chg[lead]:+.1f}, hacim {rvs[lead]:.1f}×)."))
        elif spy_ch > 0.2 and (len(sharp_down) >= 2 or up <= 5):
            out.append(("⚠️", f"Endeks yükseldi ama yalnızca {up}/{len(secs)} sektör "
                              f"artıda" + (f"; sert satış: "
                                           + ", ".join(SECTORS[s] for s in sharp_down)
                                           if sharp_down else "")
                              + " — bu net giriş değil, sektörler arası rotasyon."))
        elif spy_ch < -0.2 and up <= 3:
            out.append(("❌", f"{len(secs) - up}/{len(secs)} sektör düştü — geniş "
                              f"tabanlı satış, piyasadan para çıkıyor."))

    # 3) Genişlik
    if "RSP" in cl and "SPY" in cl:
        j = pd.concat([cl["RSP"], cl["SPY"]], axis=1, join="inner").dropna()
        if len(j) > 25:
            r = j.iloc[:, 0] / j.iloc[:, 1]
            c20 = (r.iloc[-1] / r.iloc[-21] - 1) * 100
            above = sum(float(cl[s].iloc[-1]) > float(cl[s].tail(50).mean())
                        for s in secs) if secs else 0
            if c20 < -1:
                out.append(("⚠️", f"Genişlik zayıf: eşit ağırlıklı S&P son 20 günde "
                                  f"endeksin %{c20:.1f} gerisinde; {above}/{len(secs)} "
                                  f"sektör 50 günlük ortalamasının üstünde. "
                                  f"Yükselişin sağlıklı olması için genişlik lazım."))
            elif c20 > 1:
                out.append(("✅", f"Genişlik güçlü: eşit ağırlıklı S&P son 20 günde "
                                  f"endeksi %{c20:+.1f} geçti; {above}/{len(secs)} "
                                  f"sektör 50 günlük ortalamasının üstünde."))
            else:
                out.append(("⚪", f"Genişlik nötr; {above}/{len(secs)} sektör 50 "
                                  f"günlük ortalamasının üstünde."))

    # 4) QQQ/SPY çeyreklik kırılım
    if "QQQ" in cl and "SPY" in cl:
        j = pd.concat([cl["QQQ"], cl["SPY"]], axis=1, join="inner").dropna()
        if len(j) > 600:
            b = fnl.breakout_info(j.iloc[:, 0] / j.iloc[:, 1], "QE", 8)
            if b["kirilim"]:
                out.append(("✅", "QQQ, SPY'ye karşı çeyreklik grafikte direncini "
                                  "kırdı" + (" (bu çeyrek)" if b.get("taze") else "")
                                  + " — öncü hisseler liderliğe devam edecek gibi; "
                                    "aynı zamanda risk iştahı göstergesi."))
            elif np.isfinite(b["uzaklik"]) and b["uzaklik"] > -3:
                out.append(("👀", f"QQQ/SPY çeyreklik direncine %{-b['uzaklik']:.1f} "
                                  f"uzaklıkta — kırılım yakın."))

    # 5) VIX kuralı
    if "^VIX" in cl:
        vr = vix_rule(cl["^VIX"])
        out.append(("🧭", f"VIX kuralı: {vr['etiket']} — {vr['not']}"))

    # 6) Kripto notları (Adım 2 tablosundan)
    if rot is not None and not rot.empty:
        by = rot.set_index("Anahtar")
        if "BTC.D" in by.index and "OTHERS/BTC" in by.index:
            bd, ob = by.loc["BTC.D"], by.loc["OTHERS/BTC"]
            if bd["Skor"] <= -15 and ob["Skor"] >= 15:
                out.append(("✅", "Kripto: BTC hakimiyeti düşüyor, küçük altcoinler "
                                  "BTC'yi geçiyor — altcoin sezonu koşulları."))
            elif bd["Skor"] >= 15:
                out.append(("⚠️", "Kripto: BTC hakimiyeti artıyor — para kripto "
                                  "içinde güvenli tarafa (BTC) kaçıyor; altcoinlerde "
                                  "temkin."))
    return out


# --------------------------------------------------------------------------
# Nihai karar
# --------------------------------------------------------------------------
ASSET_EXTRA: dict[str, list[str]] = {
    "S&P 500": ["RSP/SPY", "SPY/TLT", "HYG/TLT"],
    "Nasdaq": ["QQQ/SPY", "^IXIC/^GSPC", "SMH/SPY"],
    "Kripto (BTC)": ["NET_LIQ", "BTC.D", "OTHERS/BTC", "DX-Y.NYB"],
}


def asset_decision(tf_scores: dict[str, float], rot: pd.DataFrame, asset: str,
                   risk_label: str, vix_skor: float) -> dict[str, Any]:
    ws, tot = 0.0, 0.0
    for tf, w in TF_W.items():
        v = tf_scores.get(tf, np.nan)
        if np.isfinite(v):
            tot += w * v
            ws += w
    base = tot / ws if ws else np.nan
    # bağlam: Adım 2'deki ilgili rotasyon göstergeleri
    ctx, ctx_txt = [], []
    if rot is not None and not rot.empty:
        by = rot.set_index("Anahtar")
        for k in ASSET_EXTRA.get(asset, []):
            if k in by.index:
                r = by.loc[k]
                ctx.append(float(r["Risk Etkisi"]))
                ctx_txt.append(f"{r['Risk Okuması']} {r['Gösterge']}")
    ctx_sc = float(np.mean(ctx)) if ctx else 0.0
    score = (0.75 * base + 0.25 * ctx_sc) if np.isfinite(base) else np.nan
    if asset != "Kripto (BTC)" and np.isfinite(score):
        score += 10 * vix_skor          # VIX kuralı hisse endekslerine
    if np.isfinite(score) and "KAPALI" in risk_label:
        score = min(score, 15)          # risk kapalıyken "uygun" denmez
    wk = tf_scores.get("Haftalık", np.nan)
    if not np.isfinite(score):
        lab, tone = "➖ VERİ YOK", ""
    elif score >= 35 and (not np.isfinite(wk) or wk >= 20):
        lab, tone = "✅ GİRİŞ İÇİN UYGUN", "pos"
    elif score >= 10:
        lab, tone = "🟡 SEÇİCİ / KADEMELİ", ""
    elif score > -15:
        lab, tone = "⏳ BEKLE — teyit yok", ""
    else:
        lab, tone = "⛔ UYGUN DEĞİL", "neg"
    return {"etiket": lab, "tone": tone, "skor": score, "baz": base,
            "baglam": ctx_sc, "baglam_txt": ctx_txt}


def reasons(mods_by_tf: dict[str, pd.DataFrame], n: int = 3
            ) -> tuple[list[str], list[str]]:
    """En güçlü olumlu ve olumsuz gerekçeler (haftalık önce)."""
    rows = []
    for tf in ("Haftalık", "Günlük", "Aylık"):
        m = mods_by_tf.get(tf)
        if m is None or m.empty:
            continue
        for _, r in m.iterrows():
            rows.append((tf, r["Modül"], r["Durum"], r["Not"]))
    pos = [f"{tf}: {mod} — {nt}" for tf, mod, d, nt in
           sorted(rows, key=lambda x: -x[2]) if d >= 0.5][:n]
    neg = [f"{tf}: {mod} — {nt}" for tf, mod, d, nt in
           sorted(rows, key=lambda x: x[2]) if d <= -0.5][:n]
    return pos, neg


def matrix(mods_by_tf: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Satır: modül, sütun: zaman dilimi → 'ikon not' hücreleri."""
    order: list[str] = []
    cells: dict[str, dict[str, str]] = {}
    grp: dict[str, str] = {}
    for tf in TFS:
        m = mods_by_tf.get(tf)
        if m is None or m.empty:
            continue
        for _, r in m.iterrows():
            if r["Modül"] not in order:
                order.append(r["Modül"])
            grp[r["Modül"]] = r["Grup"]
            cells.setdefault(r["Modül"], {})[tf] = (
                r["Not"] if r["Modül"] == "Başlık sinyali"   # sinyalin kendi ikonu var
                else f"{_icon(r['Durum'])} {r['Not']}")
    rows = [{"Grup": grp[mo], "Modül": mo,
             **{tf: cells[mo].get(tf, "—") for tf in TFS}} for mo in order]
    for p in PENDING:
        rows.append({"Grup": "Sırada", "Modül": p,
                     **{tf: "⏳ Pine'dan çevriliyor" for tf in TFS}})
    return pd.DataFrame(rows)
