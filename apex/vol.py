# apex/vol.py — 📏 BEKLENEN HAREKET (VIX/16) · VIX3M − VIX VADE YAPISI · 20 YILLIK OLAY TESTİ
#
# 1) BEKLENEN HAREKET: VIX = S&P 500 için önümüzdeki 30 günün YILLIK örtük oynaklığı (%).
#    Günlük 1σ ≈ VIX / √252 ≈ VIX / 16 · haftalık ≈ VIX / √52 · aylık ≈ VIX / √12.
#    Günlerin ~%68'i ±1σ içinde kalır. Bugünkü hareket kaç σ? (dünkü VIX'e göre)
#    Gerçekleşen oynaklık (20 gün) ile VIX farkı = "korku primi".
# 2) VADE YAPISI: VIX3M − VIX (puan). Normalde pozitif (contango). Negatif = TERSİNE (backwardation,
#    yakın vadede panik). Durumlar: normal · daralıyor · tersine · tersinme bitti.
#    1 yıllık yüzdelik, 5 / 20 günlük değişim.
# 3) OLAY TESTİ (~20 yıl): tersinmenin BAŞLADIĞI ve BİTTİĞİ günlerden sonra S&P 500'ün 5 / 20 / 60
#    günlük getirisi, aynı dönemin bütün günlerine (taban) göre. Olaylar 20 günden sık sayılmaz.
# 4) HİSSE İÇİN: stop mesafesi beklenen hareketin (hissenin kendi ATR'si) içinde mi? VIX'e göre
#    pozisyon büyüklüğü (VIX yükseldikçe risk küçültülür).
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)
SQ = {"Günlük": np.sqrt(252), "Haftalık": np.sqrt(52), "Aylık": np.sqrt(12)}
HZ = (5, 20, 60)


def _close(df) -> pd.Series:
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)
    s = df["Close"] if isinstance(df, pd.DataFrame) else df
    s = s.astype(float).dropna()
    s.index = pd.to_datetime(s.index)
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    s.index = s.index.normalize()
    return s[~s.index.duplicated(keep="last")].sort_index()


# ------------------------------------------------------------------ 1) beklenen hareket
def expected_move(vix: pd.Series, spx: pd.Series) -> dict[str, Any]:
    vix, spx = _close(vix), _close(spx)
    if len(vix) < 25 or len(spx) < 25:
        return {"ok": False, "neden": "VIX ya da S&P 500 verisi yok"}
    j = pd.concat({"vix": vix, "spx": spx}, axis=1).dropna()
    v, px = float(j["vix"].iloc[-1]), float(j["spx"].iloc[-1])
    em = {k: v / q for k, q in SQ.items()}                       # % (1σ)
    r = j["spx"].pct_change() * 100
    sig = r / (j["vix"].shift(1) / SQ["Günlük"])                 # dünkü VIX'e göre kaç σ
    rv20 = float(r.tail(20).std() * np.sqrt(252))
    last = pd.DataFrame({"Tarih": j.index[-5:].strftime("%d.%m"),
                         "S&P 500 %": r.tail(5).round(2).values,
                         "Beklenen ±%": (j["vix"].shift(1).tail(5) / SQ["Günlük"]).round(2).values,
                         "σ": sig.tail(5).round(2).values})
    s_today = float(sig.iloc[-1]) if np.isfinite(sig.iloc[-1]) else np.nan
    # son 1 yılda |σ| ≥ 2 günlerin sıklığı (normal dağılımda ~%4.6)
    big = float((sig.tail(252).abs() >= 2).mean() * 100)
    if not np.isfinite(s_today):
        s_txt = "—"
    elif abs(s_today) < 1:
        s_txt = "sıradan bir gün (beklenen aralığın içinde)"
    elif abs(s_today) < 2:
        s_txt = "belirgin ama olağan bir hareket"
    elif abs(s_today) < 3:
        s_txt = "NADİR hareket (2σ+) — ayda bir görülür; haber/akış etkisi aranmalı"
    else:
        s_txt = "ÇOK NADİR hareket (3σ+) — yılda bir görülür; rejim değişimi olabilir"
    prim = v - rv20
    return {"ok": True, "vix": v, "spx": px, "em": em,
            "em_pt": {k: px * x / 100 for k, x in em.items()},
            "bugun_pct": float(r.iloc[-1]), "bugun_sigma": s_today, "bugun_txt": s_txt,
            "rv20": rv20, "prim": prim, "buyuk_gun %": big, "son5": last,
            "tarih": str(j.index[-1].date()),
            "vix_txt": ("rehavet (≤ 14): piyasa çok sakin — sürprizlere hazırlıksız" if v <= 14 else
                        "sakin (14–20)" if v < 20 else "tedirgin (20–30)" if v < 30 else
                        "korku (30–45): kademeli alım bölgesi" if v < 45 else
                        "panik (≥ 45): tarihsel olarak dipler burada")}


def risk_scale(vix: float) -> tuple[float, str]:
    """VIX'e göre işlem başı risk çarpanı (oynaklık hedefleme mantığı: σ ∝ VIX)."""
    if not np.isfinite(vix):
        return 1.0, "VIX yok — tam risk"
    f = float(np.clip(18.0 / vix, 0.4, 1.0))
    txt = ("VIX düşük — tam risk" if f >= 0.99 else
           f"VIX {vix:.0f} — risk %{f * 100:.0f}'e indirildi (oynaklık yüksekken aynı stop daha "
           "kolay patlar ve gap riski artar)")
    return f, txt


def position(capital: float, risk_pct: float, entry: float, stop: float,
             vix: float = np.nan) -> dict[str, Any]:
    """Pozisyon büyüklüğü: sermaye × risk% × VIX çarpanı ÷ (giriş − stop)."""
    if not (np.isfinite(entry) and np.isfinite(stop) and entry > stop > 0 and capital > 0):
        return {"ok": False}
    f, ftxt = risk_scale(vix)
    risk_usd = capital * risk_pct / 100 * f
    n = int(risk_usd // (entry - stop))
    return {"ok": True, "adet": n, "tutar": n * entry, "risk_usd": n * (entry - stop),
            "pay %": n * entry / capital * 100, "carpan": f, "txt": ftxt}


def stop_check(entry: float, stop: float, atr_pct: float, tf: str = "Günlük") -> tuple[str, str]:
    """Stop mesafesi hissenin beklenen hareketinin (ATR) kaç katı? → (rozet, açıklama)."""
    if not (np.isfinite(entry) and np.isfinite(stop) and entry > stop and np.isfinite(atr_pct)
            and atr_pct > 0):
        return "", ""
    dist = (entry - stop) / entry * 100
    k = dist / atr_pct
    if k < 1:
        return ("🔴", f"Stop girişin %{dist:.1f} altında = {k:.1f} ATR. Bu hisse günde ortalama "
                      f"±%{atr_pct:.1f} oynuyor; stop sıradan bir günlük dalgalanmayla patlar. "
                      "Stopu genişletip pozisyonu küçültün.")
    if k < 1.5:
        return ("🟠", f"Stop %{dist:.1f} = {k:.1f} ATR — dar. Kötü bir gün stopu tetikleyebilir.")
    if k > 4:
        return ("🟡", f"Stop %{dist:.1f} = {k:.1f} ATR — geniş; pozisyon küçük tutulmalı.")
    return ("🟢", f"Stop %{dist:.1f} = {k:.1f} ATR — hissenin normal oynaklığının dışında, makul.")


# ------------------------------------------------------------------ 2) vade yapısı
def term_structure(vix: pd.Series, vix3m: pd.Series) -> dict[str, Any]:
    vix, v3 = _close(vix), _close(vix3m)
    j = pd.concat({"vix": vix, "v3": v3}, axis=1).dropna()
    if len(j) < 30:
        return {"ok": False, "neden": "VIX3M verisi yok"}
    sp = j["v3"] - j["vix"]
    now = float(sp.iloc[-1])
    ch5 = float(sp.iloc[-1] - sp.iloc[-6]) if len(sp) > 6 else np.nan
    ch20 = float(sp.iloc[-1] - sp.iloc[-21]) if len(sp) > 21 else np.nan
    y = sp.tail(252)
    pctl = float((y < now).mean() * 100)
    neg10 = bool((sp.tail(10) < 0).any())
    if now < 0:
        kod, lab = "tersine", "🔴 TERSİNE (yakın vadede panik)"
        txt = (f"1 aylık korku (VIX {j['vix'].iloc[-1]:.1f}) 3 aylık korkunun ({j['v3'].iloc[-1]:.1f}) "
               "üstünde: piyasa ŞU AN için endişeli. Tarihte bu durumlar kısa sürer; tersinme "
               "bitince dip çoğu zaman geride kalır. Panik satışı yerine kademeli alım bölgesi.")
    elif neg10:
        kod, lab = "bitti", "🟢 TERSİNME BİTTİ"
        txt = ("Son 10 günde tersine olan yapı normale döndü. Tarihsel olarak bu, korku dalgasının "
               "bittiğini ve toparlanmanın başladığını gösterir (aşağıdaki 20 yıllık teste bakın).")
    elif (np.isfinite(ch5) and ch5 <= -1.0 and pctl <= 30) or (pctl <= 10 and now < 1.5):
        kod, lab = "daraliyor", "🟠 DARALIYOR"
        txt = ("Fark hızla küçülüyor: yakın vade korkusu artıyor. Tersine dönerse risk azaltma "
               "zamanı; şimdilik dikkat.")
    else:
        kod, lab = "normal", "⚪ NORMAL (contango)"
        txt = ("Uzak vade korkusu yakın vadeden yüksek — sakin piyasanın olağan hâli.")
    return {"ok": True, "spread": now, "ch5": ch5, "ch20": ch20, "pctl": pctl, "kod": kod,
            "etiket": lab, "txt": txt, "vix": float(j["vix"].iloc[-1]),
            "vix3m": float(j["v3"].iloc[-1]), "seri": sp.tail(504), "tarih": str(j.index[-1].date())}


# ------------------------------------------------------------------ 3) olay testi
def _events(mask: pd.Series, gap: int = 20) -> list[int]:
    idx, last = [], -10 ** 9
    a = mask.fillna(False).to_numpy()
    for i in np.flatnonzero(a):
        if i - last >= gap:
            idx.append(int(i))
            last = i
    return idx


def event_test(vix: pd.Series, vix3m: pd.Series, spx: pd.Series,
               min_days: int = 2) -> dict[str, Any]:
    """Tersinme BAŞLADI / BİTTİ / DERİN (< −2 puan) olaylarından sonra S&P 500 ileri getirisi."""
    j = pd.concat({"vix": _close(vix), "v3": _close(vix3m), "spx": _close(spx)}, axis=1).dropna()
    if len(j) < 500:
        return {"ok": False, "neden": f"yetersiz geçmiş ({len(j)} gün)"}
    sp = j["v3"] - j["vix"]
    neg = sp < 0
    run = neg.groupby((neg != neg.shift()).cumsum()).cumcount() + 1
    start = neg & ~neg.shift(1, fill_value=False)
    end = (~neg) & neg.shift(1, fill_value=False) & (run.shift(1) >= min_days)
    deep = (sp < -2) & (sp.shift(1) >= -2)
    c = j["spx"]
    fwd = {h: (c.shift(-h) / c - 1) * 100 for h in HZ}
    base = {h: fwd[h].dropna() for h in HZ}
    rows, ev_list = [], []
    for nm, m in (("🔴 Tersinme başladı", start), ("🔴 Derin tersinme (< −2 puan)", deep),
                  ("🟢 Tersinme bitti", end)):
        ii = _events(m)
        rec = {"Olay": nm, "Olay sayısı": len(ii)}
        for h in HZ:
            x = fwd[h].iloc[ii].dropna()
            rec[f"{h}G ort. %"] = float(x.mean()) if len(x) else np.nan
            rec[f"{h}G pozitif %"] = float((x > 0).mean() * 100) if len(x) else np.nan
            rec[f"{h}G tabana göre"] = float(x.mean() - base[h].mean()) if len(x) else np.nan
        rows.append(rec)
        for i in ii:
            ev_list.append({"Tarih": str(j.index[i].date()), "Olay": nm,
                            "Fark (puan)": round(float(sp.iloc[i]), 2),
                            "VIX": round(float(j["vix"].iloc[i]), 1),
                            **{f"S&P {h}G %": round(float(fwd[h].iloc[i]), 1)
                               if np.isfinite(fwd[h].iloc[i]) else np.nan for h in HZ}})
    rows.append({"Olay": "Taban (bütün günler)", "Olay sayısı": int(len(base[20])),
                 **{f"{h}G ort. %": float(base[h].mean()) for h in HZ},
                 **{f"{h}G pozitif %": float((base[h] > 0).mean() * 100) for h in HZ},
                 **{f"{h}G tabana göre": 0.0 for h in HZ}})
    E = pd.DataFrame(ev_list).sort_values("Tarih", ascending=False).reset_index(drop=True) \
        if ev_list else pd.DataFrame()
    return {"ok": True, "ozet": pd.DataFrame(rows), "olaylar": E,
            "donem": f"{j.index[0]:%Y} – {j.index[-1]:%Y}", "gun": len(j)}


def pack(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """frames: {'^VIX', '^VIX3M', '^GSPC' (ya da 'SPY')} → arayüzün bütün vol bölümü."""
    vix = frames.get("^VIX")
    v3 = frames.get("^VIX3M")
    spx = frames.get("^GSPC")
    if spx is None or len(spx) == 0:
        spx = frames.get("SPY")
    out = {"em": expected_move(vix, spx), "ts": term_structure(vix, v3)}
    try:
        out["test"] = event_test(vix, v3, spx)
    except Exception as exc:                          # noqa: BLE001
        _log.info("vix olay testi: %s", exc)
        out["test"] = {"ok": False, "neden": str(exc)}
    return out
