# apex/plan.py — hisse başına İŞLEM PLANI: giriş bölgesi · stop · kâr al · karar
#
# Seviyeler Pine modüllerinden gelir (hepsi günlük mum):
#   destek adayları : S/R matrisi destekleri (QUANTUM 885), açık boğa boşlukları
#                     (Gap Matrix), Wyckoff alım bölgeleri (VSA PRO), EMA21/EMA50,
#                     ana yapı Fibonacci 0.382 / 0.5 / 0.618 geri çekilmeleri
#   hedef adayları  : S/R dirençleri, doldurulmamış ayı boşlukları, Wyckoff satış
#                     bölgeleri, 52 haftalık zirve, Fib 1.272 / 1.618 uzantıları
# Giriş bölgesi = fiyatın altındaki EN ÇOK ÇAKIŞAN destek kümesi (≤ 3 ATR).
# Stop = kümenin dibi − 1 ATR (GFR'de ölçülen en iyi stop mesafesi).
# Hedef = girişten en az 1 R uzaktaki ilk direnç (T1) ve bir sonraki (T2);
# yakında direnç yoksa 2R / 3R.
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from apex import pine_fib as pfib
from apex import pine_structure as pst
from apex import pine_vsa as pvsa
from apex.indicators import ema

MAX_DEPTH_ATR = 3.0      # giriş desteği fiyatın en fazla bu kadar ATR altında
CLUSTER_ATR = 0.5        # bu yakınlıktaki destekler aynı küme sayılır
STOP_ATR = 1.0
ZONE_ATR = 0.5           # giriş bölgesinin kalınlığı


def _f(x: float) -> str:
    return f"{x:,.0f}" if x >= 1000 else f"{x:.2f}"


def stock_plan(df: pd.DataFrame) -> dict[str, Any]:
    """Günlük OHLCV → plan sözlüğü (ok=False ise 'neden' alanı açıklar)."""
    if df is None or len(df) < 120:
        return {"ok": False, "neden": "yetersiz fiyat geçmişi"}
    d = df[["Open", "High", "Low", "Close", "Volume"]].astype(float).dropna(
        subset=["Close"]).tail(420)
    atr_s = pfib.atr_rma(d, 14)
    atr = float(atr_s.iloc[-1])
    px = float(d["Close"].iloc[-1])
    if not (np.isfinite(atr) and atr > 0 and px > 0):
        return {"ok": False, "neden": "ATR hesaplanamadı"}
    c = d["Close"]
    e21, e50 = float(ema(c, 21).iloc[-1]), float(ema(c, 50).iloc[-1])
    e200 = float(ema(c, 200).iloc[-1]) if len(d) >= 200 else np.nan
    hi52 = float(d["High"].tail(252).max())

    sr = pst.sr_matrix(d, atr_s)
    gm = pst.gap_matrix(d, atr_s, "Günlük")
    vz = pst.vsa_zones(d)
    mf = pfib.main_fib(d)

    sup: list[tuple[float, str, float]] = []      # (fiyat, etiket, ağırlık)
    res: list[tuple[float, str]] = []
    for x in sr["sup"]:
        sup.append((x["px"], f"{pst.HZ_TXT[x['hz']]} vade destek ({x['cnt']} pivot)",
                    1.0 + 0.5 * x["hz"]))
    for x in sr["res"]:
        res.append((x["px"], f"{pst.HZ_TXT[x['hz']]} vade direnç"))
    for b in gm["boxes"]:
        if b["dir"] > 0 and b["top"] < px:
            sup.append((b["top"], f"açık {pst.TIER_TXT[b['tier']]} boşluğu", 0.6 + 0.2 * b["tier"]))
        elif b["dir"] < 0 and b["bot"] > px:
            res.append((b["bot"], f"doldurulmamış {pst.TIER_TXT[b['tier']]} boşluğu"))
    for z in vz["zones"]:
        if z["side"] > 0 and z["top"] <= px:
            sup.append((z["top"], f"Wyckoff alım bölgesi ({z['name']})", 1.0))
        elif z["side"] < 0 and z["bot"] >= px:
            res.append((z["bot"], f"Wyckoff satış bölgesi ({z['name']})"))
    for v, nm in ((e21, "EMA21"), (e50, "EMA50"), (e200, "EMA200")):
        if np.isfinite(v) and v < px:
            sup.append((v, nm, 0.7 if nm != "EMA200" else 1.0))
    if mf.get("ok") and mf["up"]:
        for lv in (0.382, 0.5, 0.618):
            v = mf["levels"][lv]
            if v < px:
                sup.append((v, f"Fib {lv}", 0.8 if lv != 0.618 else 1.0))
        for lv in (1.272, 1.618):
            if mf["levels"][lv] > px:
                res.append((mf["levels"][lv], f"Fib {lv} uzantısı"))
    if hi52 > px * 1.005:
        res.append((hi52, "52 hafta zirvesi"))

    # --- giriş kümesi: fiyatın 0–3 ATR altında en çok çakışan destekler
    cand = [s for s in sup if px - MAX_DEPTH_ATR * atr <= s[0] <= px]
    best = None
    for p0, _, _ in cand:
        grp = [s for s in cand if abs(s[0] - p0) <= CLUSTER_ATR * atr]
        w = sum(s[2] for s in grp)
        dist = (px - p0) / atr
        score = w - 0.15 * dist
        if best is None or score > best[0]:
            best = (score, grp)
    if best is None:
        lo_ = min((s for s in sup if s[0] < px), key=lambda s: px - s[0], default=None)
        if lo_ is None:
            return {"ok": False, "neden": "altta tanımlı destek yok"}
        grp = [lo_]
    else:
        grp = best[1]
    z_lo = min(s[0] for s in grp)
    z_hi = min(max(s[0] for s in grp) + ZONE_ATR * atr * 0.5, px)
    z_hi = max(z_hi, z_lo)
    entry = (z_lo + z_hi) / 2
    stop = z_lo - STOP_ATR * atr
    risk = entry - stop

    # --- hedefler: girişten ≥ 1R yukarıdaki dirençler
    res = sorted(set(res), key=lambda r: r[0])
    ok_t = [r for r in res if r[0] - entry >= risk]
    if ok_t:
        t1 = ok_t[0]
        nxt = [r for r in ok_t[1:] if r[0] - t1[0] >= 0.5 * atr]
        t2 = nxt[0] if nxt else (entry + max(3 * risk, t1[0] - entry + risk), "ölçülen hedef (3R)")
    else:
        t1 = (entry + 2 * risk, "ölçülen hedef (2R) — yakında direnç yok")
        t2 = (entry + 3 * risk, "ölçülen hedef (3R)")
    rr = (t1[0] - entry) / risk if risk > 0 else np.nan
    rr2 = (t2[0] - entry) / risk if risk > 0 else np.nan
    near_res = [r for r in res if px < r[0] <= px + 0.5 * atr]

    # --- teknik durum
    trend = ("yukarı" if (np.isfinite(e200) and px > e50 > e200) else
             "aşağı" if (np.isfinite(e200) and px < e50 < e200) else "karışık")
    try:
        vr = pvsa.compute(d)
        vsa_v = vr["verdict"]
    except Exception:                      # noqa: BLE001
        vsa_v = ""
    in_zone = px <= z_hi + 0.25 * atr
    dist_pct = (px / z_hi - 1) * 100 if z_hi > 0 else np.nan
    return {
        "ok": True, "Fiyat": px, "ATR": atr,
        "Giriş Alt": z_lo, "Giriş Üst": z_hi, "Giriş": entry,
        "Stop": stop, "Stop %": (stop / entry - 1) * 100,
        "Hedef 1": t1[0], "Hedef 1 neden": t1[1],
        "Hedef 2": t2[0], "Hedef 2 neden": t2[1],
        "R:R": rr, "R:R 2": rr2, "Bölgede": bool(in_zone), "Bölgeye %": dist_pct,
        "Destek": " + ".join(sorted({s[1] for s in grp})),
        "Dirence dayandı": bool(near_res),
        "Trend": trend, "VSA": vsa_v,
        "Zirveye %": (px / hi52 - 1) * 100,
    }


def plans(prices: dict[str, pd.DataFrame], tickers) -> pd.DataFrame:
    rows = []
    for t in tickers:
        try:
            p = stock_plan(prices.get(t))
        except Exception as exc:           # noqa: BLE001
            p = {"ok": False, "neden": f"hesap hatası: {exc}"}
        rows.append({"Sembol": t, **p})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Karar: tarama sınıfı + bilanço + endeks kararı + plan
# --------------------------------------------------------------------------
GOOD = {"🎯 Alım adayı", "🌱 Geride kaldı, toparlanıyor", "🚀 Lider"}


def verdict(p: dict[str, Any], cls: str = "", sig: str = "", kalan_gun=None,
            index_ok: bool = True) -> tuple[str, list[str]]:
    """Döner: (karar, [sebepler]). Karar: ✅ AL · 🟡 LİMİT EMİR · ⛔ UZAK DUR."""
    why_bad: list[str] = []
    why: list[str] = []
    if not p.get("ok"):
        if p.get("neden") == "altta tanımlı destek yok":
            return "⛔ UZAK DUR", ["fiyat yeni dipte — altında tutunacağı destek yok"]
        return "⚪ PLAN YOK", [p.get("neden", "veri yok")]
    kg = pd.to_numeric(kalan_gun, errors="coerce")
    if cls.startswith("⛔"):
        why_bad.append(f"satış/dağıtım sinyali ({sig})" if sig else "satış/dağıtım sinyali")
    if np.isfinite(kg) and 0 <= kg <= 7:
        why_bad.append(f"bilanço {int(kg)} gün sonra — gap riski")
    if cls.startswith("💧"):
        why_bad.append("işlem hacmi düşük")
    if p["Trend"] == "aşağı":
        why_bad.append("düşüş trendi (fiyat < EMA50 < EMA200)")
    if p["R:R 2"] < 1.5:
        why_bad.append(f"risk/ödül zayıf (en iyi hedefte bile {p['R:R 2']:.1f})")
    if p["VSA"] == "TEYİTLİ DÜŞÜŞ" and p["Trend"] != "yukarı":
        why_bad.append("hacim satıcıdan yana (VSA: teyitli düşüş)")
    if why_bad:
        return "⛔ UZAK DUR", why_bad

    why.append(f"giriş desteği: {p['Destek']}")
    why.append(f"hedef: {p['Hedef 1 neden']} · R:R {p['R:R']:.1f}")
    if p["Trend"] == "yukarı":
        why.append("trend yukarı")
    if cls:
        why.append(cls.split(" ", 1)[-1].lower())
    if p["VSA"].startswith("ZAYIF"):
        why.append("⚠ hacim zayıf (dağıtım şüphesi)")
    if p["VSA"].startswith("TOPLAMA"):
        why.append("hacim toplama gösteriyor")
    wait = []
    if not p["Bölgede"]:
        wait.append(f"fiyat giriş bölgesinin %{p['Bölgeye %']:.1f} üstünde — "
                    f"limit emir: {_f(p['Giriş Üst'])}")
    if p["Dirence dayandı"]:
        wait.append("fiyat dirence dayandı — kırılım ve üstünde kapanış beklenmeli")
    elif p["R:R"] < 1.5:
        wait.append(f"ilk hedef yakın ({_f(p['Hedef 1'])}, R:R {p['R:R']:.1f}) — "
                    "orası kırılırsa daha iyi risk/ödül")
    if cls and cls not in GOOD:
        wait.append(f"sınıf: {cls}")
    if not index_ok:
        wait.append("endeks kararı alım yönünde değil")
    if p["VSA"] == "TEYİTLİ DÜŞÜŞ":
        wait.append("geri çekilmede satış baskısı sürüyor — dönüş mumu/hacim teyidi bekleyin")
    if wait:
        return "🟡 LİMİT EMİR / BEKLE", wait + why
    return "✅ AL", why


INDEX_THEMES: dict[str, list[str]] = {
    "Nasdaq": ["Yarı İletken", "Yapay Zekâ", "Yazılım & SaaS", "Siber Güvenlik",
               "Teknoloji (Geniş)", "Robotik & Otomasyon", "Kuantum",
               "Veri Merkezi & Dijital GYO", "İletişim & Medya", "Fintek",
               "Biyoteknoloji", "Genomik", "Uzay", "Büyüme"],
    "S&P 500": ["Teknoloji (Geniş)", "Yarı İletken", "Bankalar", "Sağlık (Geniş)",
                "İlaç", "Tıbbi Cihaz", "Petrol & Gaz", "Kamu Hizmetleri",
                "Savunma & Havacılık", "Altyapı", "Taşımacılık", "Konut İnşaatı",
                "Perakende", "Tüketici Defansif", "Materyal", "Gayrimenkul",
                "İletişim & Medya", "Havayolları", "Nükleer & Uranyum",
                "Su Altyapısı", "Metal & Madencilik", "Temettü", "Değer"],
    "Kripto (BTC)": ["Bitcoin", "Bitcoin Madenciliği", "Fintek"],
}
