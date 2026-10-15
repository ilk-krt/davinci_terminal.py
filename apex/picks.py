# apex/picks.py — 🛒 ALIM SEÇKİSİ: temalara ayrılmış, sepetli, vadeli alım listesi
#
# Elle yaptığımız "portföy analizi"nin (sağlık/biotech listesi) otomatiği. Her hisse için:
#
#   SEPET      🏛️ Çekirdek     — büyük (≥ 50 milyar $) ve kârlı (ileri F/K > 0)
#              📈 Büyüme & değer — orta/büyük, kârlı ya da hızlı büyüyen
#              🧪 Spekülatif    — küçük (< 5 milyar $) ya da zararda; küçük pozisyon
#              ⛔ İzle          — 🐈 ölü kedi, kırık bıçak (zirveden ≥ %40 aşağı + ayı fazı)
#                                ya da ana yapı bozulmuş
#   ZAMANLAMA  günlük ANA YAPI FIB bacağı (robotla aynı hesap, pine_fib.main_fib):
#              yükseliş bacağında 0.382–0.786 = 🔑 test edilmiş kapı (2 yıl, 2 dönem +0.28R)
#              < 0.382 → geri çekilme bekle · 0.786–1 → teyit bekle · düşüş bacağı → dip teyidi
#   SEVİYELER  giriş bölgesi = fiyatın içinde/bekleneceği Fib bandı, stop = bandın altı − ATR
#              payı, kâr al 1 = bacağın tepesi (ya da düşüşte 0.382/0.618 geri dönüş),
#              kâr al 2 = Fib 1.272 uzantısı
#   FAIR VALUE analist 12 ay ortalama hedefi (ve en yüksek hedef) — BİLGİ amaçlıdır,
#              geçmişte doğrulanmadı; ileri F/K ve piyasa değeri sepeti belirler
#   VADE       K (kısa): kapıda + ADX > 15 ya da örgü kuralı · O (orta): para giren tema +
#              boğa fazı · U (uzun): çekirdek/büyüme + analist potansiyeli ≥ %15 + trend bozuk değil
#   ÖNCELİK    0–100; vade seçimine göre ağırlıklar değişir (kısa: zamanlama, uzun: değer)
#
# KULLANICI KURALLARI (ilave katman, öncekilerin üstüne):
#   KISA vade için iyi : 🚀 momentum (20G+60G getiri evrenin ilk %20'si, fiyat yükselen EMA21
#                        üstünde) · 🐋 balina girişi (son 5 günde Afterburner boğa / dip süpürmesi /
#                        gerçek toplama ya da WHALE ≥ 60 ve 10 günde +8; son 10 günde DAĞITIM
#                        varsa sayılmaz) · 🧬 tarihsel örgü (R1/R2 ya da örgü puanı ≥ 85)
#   ORTA/UZUN için iyi : 📈 endeks ya da tema bir süredir düşmüş ve pozitife dönüyor · 💰 fiyat
#                        fair value'nun (analist ort. hedefi, ≥ 3 analist) ≥ %35 altında ·
#                        🏛️ temeller sağlam (kârlı, büyüyen, serbest nakit +, borç makul, ROE > %10
#                        — mevcut olanların ≥ %80'i) → üçü birden = 💎 DEĞER DÖNÜŞÜ
#   Kısa vadede momentum + (balina/örgü/20G zirve) olan, EMA200 üstü hisse 🟢 MOMENTUM kararı
#   alır (fiyattan giriş, EMA21 − 0.5 ATR / 1.5 ATR stop, 2R ve 3R hedef).
#
# PİYASA İÇ YAPISI (breadth.py, "boğa-ayı piyasası" mantığı):
#   🧮 ucuzlayan lider — ileri F/K ≤ 0.8 × geçmiş F/K ve fiyat 60 günde artıda (kazanç fiyattan
#      hızlı büyüyor; endeks F/K'sının ≤ 1.25 katıysa ekstra puan)
#   🌊 genişleme adayı — zirveden ≥ %20 aşağıda, temeli sağlam, trendi bozuk değil ve piyasa dar
#      liderlikte (boğa-ayı) ya da genişlemeye başlamış; genişleme başladıysa ve hisse boğa
#      fazındaysa 🌊 GENİŞLEME ADAYI kararı (orta/uzun vade)
#
# Kanıt düzeyi (Sinyal doğrulama + robot toplu testleri): Fib 0.382–0.786 kapısı ✅,
# taze lider tema ✅, ölü kedi uyarısı ✅ (büyük düşüş öncesi parmak izi). Analist hedefi,
# F/K ve EMA fazı geçmişte bu uygulamada test edilmedi → bağlam olarak gösterilir.
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from apex import ema_stack as ems
from apex import pine_fib as pfib

_log = logging.getLogger(__name__)

SEPET = {"cekirdek": "🏛️ Çekirdek", "buyume": "📈 Büyüme & değer",
         "spek": "🧪 Spekülatif", "izle": "⛔ İzle"}
SEPET_TXT = {
    "cekirdek": "Büyük ve kârlı şirketler — portföyün omurgası, uzun vadede geri çekilmelerde "
                "biriktirilir.",
    "buyume": "Orta/büyük, kârlı ya da hızlı büyüyen — orta vadede tema rüzgârıyla taşınır.",
    "spek": "Küçük ya da zararda — yüksek oynaklık; pozisyon küçük tutulur (risk ≤ %0.5).",
    "izle": "Ölü kedi, kırık bıçak ya da bozulmuş yapı — taban oluşmadan alınmaz."}
TIMING = {
    "kapida": "🔑 Fib kapısında",
    "kapida_zayif": "🔑 Fib kapısında (ADX zayıf)",
    "bekle_geri": "⏳ Geri çekilme bekle",
    "derin": "⚠️ Çok derin — teyit bekle",
    "dip": "🔄 Düşüş bacağı — dip teyidi",
    "donus_teyit": "🌱 Dönüş başladı — geri test",
    "bozuk": "⛔ Yapı bozuldu",
    "belirsiz": "➖ Bacak yok",
}
BULL_FAZ = {"tam_boga", "boga_geri", "boga_derin", "erken_donus", "sikisma"}
BEAR_FAZ = {"tam_ayi", "ayi_tepki", "kirilma"}
VADE_W = {  # zamanlama, değer (analist potansiyeli), tema, faz
    "kısa": (1.3, 0.4, 0.8, 1.0),
    "orta": (1.0, 1.0, 1.0, 1.0),
    "uzun": (0.6, 1.6, 0.8, 1.2),
}


def _num(x: Any) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _adx(d: pd.DataFrame, n: int = 14) -> float:
    h, l, c = d["High"], d["Low"], d["Close"]
    up, dn = h.diff(), -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)

    def rma(x):
        return pd.Series(x, index=d.index).ewm(alpha=1 / n, adjust=False).mean()
    a = rma(tr).replace(0, np.nan)
    pdi, mdi = 100 * rma(pdm) / a, 100 * rma(mdm) / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return float(rma(dx.fillna(0)).iloc[-1])


def _band(r: float) -> tuple[float, float]:
    """Fiyatın içinde bulunduğu kapı bandı (sığdan derine)."""
    for a, b in ((0.382, 0.5), (0.5, 0.618), (0.618, 0.786)):
        if r <= b:
            return a, b
    return 0.618, 0.786


def quality(fund: dict | None) -> tuple[int, int, bool, str, str]:
    """Temel sağlamlık: kârlı · büyüyor · serbest nakit + · borç makul · ROE > %10.
    Bilinenlerin (en az 3) ≥ %80'i sağlanırsa 'sağlam'."""
    fund = fund or {}
    marj, roe = _num(fund.get("Kâr Marjı %")), _num(fund.get("ROE %"))
    fcf, de = _num(fund.get("FCF ($B)")), _num(fund.get("Borç/Özkaynak"))
    gr_, fpe_ = _num(fund.get("Büyüme %")), _num(fund.get("İleri F/K"))
    checks = {
        "kârlı": (marj > 0 or fpe_ > 0) if (np.isfinite(marj) or np.isfinite(fpe_)) else None,
        "büyüyor": (gr_ > 0) if np.isfinite(gr_) else None,
        "serbest nakit +": (fcf > 0) if np.isfinite(fcf) else None,
        "borç makul": (de < 150) if np.isfinite(de) else None,
        "ROE > %10": (roe > 10) if np.isfinite(roe) else None,
    }
    have = {k: v for k, v in checks.items() if v is not None}
    t_pass = sum(bool(v) for v in have.values())
    saglam = bool(len(have) >= 3 and t_pass / len(have) >= 0.8)
    txt = (f"{t_pass}/{len(have)}" + (" ✓" if saglam else "")) if have else "—"
    return t_pass, len(have), saglam, txt, ", ".join(k for k, v in have.items() if not v)


# ------------------------------------------------------------------ tek hisse
def stock_metrics(sym: str, d: pd.DataFrame, fund: dict | None = None,
                  orgu: dict | None = None) -> dict[str, Any] | None:
    """Temadan bağımsız ölçümler (akşam hesabında bir kez)."""
    if d is None or len(d) < 120:
        return None
    d = d[["Open", "High", "Low", "Close", "Volume"]].astype(float).dropna(subset=["Close"])
    fund, orgu = fund or {}, orgu or {}
    px = float(d["Close"].iloc[-1])
    atr = float(pfib.atr_rma(d, 14).iloc[-1])
    hi52 = float(d["High"].tail(252).max())
    lo52 = float(d["Low"].tail(252).min())
    from_hi = (px / hi52 - 1) * 100 if hi52 > 0 else np.nan
    liq = float((d["Close"] * d["Volume"]).tail(20).mean() / 1e6)
    F = ems.frame(d)
    faz = str(F["faz"].iloc[-1]) if "faz" in F else "karisik"
    e200 = float(F["e200"].iloc[-1]) if "e200" in F else np.nan
    ust200 = bool(np.isfinite(e200) and px > e200)
    ust50 = bool("e50" in F and px > float(F["e50"].iloc[-1]))
    adx_ = _adx(d)
    mf = pfib.main_fib(d.tail(420))
    lv = mf.get("levels", {}) if mf.get("ok") else {}
    r = float(mf.get("r", np.nan)) if mf.get("ok") else np.nan
    up = bool(mf.get("up")) if mf.get("ok") else False

    z_lo = z_hi = stop = tp1 = tp2 = teyit = np.nan
    if not mf.get("ok"):
        tm = "belirsiz"
    elif up:
        if r > 1:
            tm = "bozuk"
        elif r > 0.786:
            tm = "derin"
            z_lo, z_hi, teyit = lv[1.0], lv[0.786], lv[0.786]
            stop, tp1, tp2 = lv[1.0] - 0.5 * atr, lv[0.0], lv[1.272]
        elif r >= 0.382:
            tm = "kapida" if adx_ > 15 else "kapida_zayif"
            a, b = _band(r)
            z_lo, z_hi = lv[b], lv[a]
            deeper = {0.5: lv[0.618] - 0.25 * atr, 0.618: lv[0.786] - 0.25 * atr}
            stop = min(deeper.get(b, lv[0.786] - 0.75 * atr), px - 1.0 * atr)
            tp1, tp2 = lv[0.0], lv[1.272]
        else:
            tm = "bekle_geri"
            z_lo, z_hi = lv[0.5], lv[0.382]
            stop = lv[0.618] - 0.25 * atr
            tp1 = lv[0.0] if lv[0.0] > z_hi * 1.02 else lv[1.272]
            tp2 = lv[1.272] if tp1 < lv[1.272] * 0.999 else lv[1.618]
    else:                                   # ana bacak DÜŞÜŞ: r = tepki oranı (0 = dip)
        if r > 1:
            tm = "bekle_geri"               # düşüşün tamamı geri alındı → yeni yükseliş
            top_, rng_ = lv[1.0], lv[1.0] - lv[0.0]
            z_lo, z_hi = top_ - 0.236 * rng_, top_          # kırılan tepenin geri testi
            stop = lv[0.618] - 0.25 * atr
            tp1, tp2 = top_ + 0.272 * rng_, top_ + 0.618 * rng_
        elif r >= 0.382:
            tm = "donus_teyit"
            z_lo, z_hi = lv[0.236], lv[0.382]
            stop, tp1, tp2 = lv[0.0] - 0.5 * atr, lv[0.618], lv[1.0]
        else:
            tm = "dip"
            z_lo, z_hi, teyit = lv[0.0], lv[0.236], lv[0.382]
            stop, tp1, tp2 = lv[0.0] - 0.5 * atr, lv[0.382], lv[0.618]
    mid = (z_lo + z_hi) / 2 if np.isfinite(z_lo) else np.nan
    entry = min(px, mid) if np.isfinite(mid) else np.nan
    if tm in ("kapida", "kapida_zayif"):
        entry = px
    rr = ((tp1 - entry) / (entry - stop)
          if np.isfinite(entry) and np.isfinite(stop) and entry > stop and tp1 > entry else np.nan)

    # ---- KISA VADE: momentum ham değerleri (kesitsel sıralama compute() içinde)
    c_ = d["Close"]
    ret20 = (px / float(c_.iloc[-21]) - 1) * 100 if len(c_) > 21 else np.nan
    ret60 = (px / float(c_.iloc[-61]) - 1) * 100 if len(c_) > 61 else np.nan
    e10 = float(F["e10"].iloc[-1]) if "e10" in F else np.nan
    e21 = float(F["e21"].iloc[-1]) if "e21" in F else np.nan
    s21 = float(F["s21"].iloc[-1]) if "s21" in F else np.nan
    hi20 = float(d["High"].tail(20).max())
    mom_ok = bool(np.isfinite(e21) and px > e21 and s21 > 0 and ret20 > 0)
    # momentum planı (yalnız kısa vade 🟢 MOMENTUM kararında kullanılır — test edilmedi)
    m_stop = min(e21 - 0.5 * atr, px - 1.5 * atr) if np.isfinite(e21) else px - 1.5 * atr
    m_lo = max(e10, px - 0.5 * atr) if np.isfinite(e10) else px - 0.5 * atr
    m_risk = px - m_stop
    m_tp1 = px + 2 * m_risk
    m_tp2 = lv[1.272] if up and lv and lv[1.272] > m_tp1 else px + 3 * m_risk
    # ---- KISA VADE: balina girişi (APEX CORE v4 + V670 OMNI)
    wh = {"ev": [], "neg": [], "whale": np.nan, "dwhale": np.nan}
    try:
        from apex import engine as eng
        dd = d.assign(Volume=d["Volume"].fillna(0.0))
        core, omni = eng.apex_core(dd), eng.apex_omni(dd)

        def ago(x: pd.Series, n: int) -> int | None:
            v = x.fillna(False).astype(bool).to_numpy()[-n:]
            idx = np.flatnonzero(v)
            return int(n - 1 - idx[-1]) if len(idx) else None
        for col, src, lab, n in (("ab_bull", omni, "🚀 Afterburner boğa", 5),
                                 ("sweep_low", omni, "🎣 Dip süpürmesi", 5),
                                 ("real_acc", core, "🐋 Gerçek toplama", 5)):
            if col in src:
                g = ago(src[col], n)
                if g is not None:
                    wh["ev"].append(f"{lab} ({'bugün' if g == 0 else f'{g}g önce'})")
        for col, src, lab, n in (("real_dist", core, "🐋 Gerçek dağıtım", 10),
                                 ("exhausted", core, "⚠️ Whale tükenme", 5)):
            if col in src:
                g = ago(src[col], n)
                if g is not None:
                    wh["neg"].append(f"{lab} ({'bugün' if g == 0 else f'{g}g önce'})")
        w_ = core["whale"].astype(float)
        wh["whale"] = float(w_.iloc[-1])
        wh["dwhale"] = float(w_.iloc[-1] - w_.iloc[-11]) if len(w_) > 11 else np.nan
    except Exception as exc:                     # noqa: BLE001
        _log.info("picks balina %s: %s", sym, exc)
    w_rise = bool(np.isfinite(wh["whale"]) and wh["whale"] >= 60 and wh["dwhale"] >= 8)
    whale_ok = bool((wh["ev"] or w_rise) and not any("dağıtım" in x for x in wh["neg"]))
    whale_txt = " · ".join(wh["ev"] + ([f"WHALE {wh['whale']:.0f} ↑{wh['dwhale']:+.0f}/10g"]
                                       if w_rise else []))
    # ---- ORTA / UZUN: fair value iskontosu ve temel sağlamlık
    hedef = _num(fund.get("Hedef"))
    n_an = _num(fund.get("Analist Sayısı"))
    disc = (1 - px / hedef) * 100 if np.isfinite(hedef) and hedef > 0 else np.nan
    t_pass, n_have, saglam, temel_txt, temel_eksik = quality(fund)
    fpe_, tpe_ = _num(fund.get("İleri F/K")), _num(fund.get("Geçmiş F/K"))
    fk_s = fpe_ / tpe_ if np.isfinite(fpe_) and np.isfinite(tpe_) and fpe_ > 0 and tpe_ > 0 \
        else np.nan
    ucuzlayan = bool(np.isfinite(fk_s) and fk_s <= 0.8 and np.isfinite(ret60) and ret60 > 0)
    ucuz = bool(np.isfinite(disc) and disc >= 35 and (not np.isfinite(n_an) or n_an >= 3))

    mcap = _num(fund.get("Piyasa Değ. ($B)"))
    fpe = _num(fund.get("İleri F/K"))
    grow = _num(fund.get("Büyüme %"))
    kedi = bool(orgu.get("🐈 Ölü kedi"))
    kirik = bool(np.isfinite(from_hi) and from_hi <= -40 and faz in BEAR_FAZ)
    if kedi or kirik or tm == "bozuk":
        sep = "izle"
    elif np.isfinite(mcap) and mcap >= 50 and np.isfinite(fpe) and fpe > 0:
        sep = "cekirdek"
    elif (np.isfinite(mcap) and mcap < 5) or (np.isfinite(mcap) and not (fpe > 0)
                                              and not (grow >= 25)):
        sep = "spek"
    else:
        sep = "buyume"

    return {
        "Sembol": sym, "Fiyat": px, "ATR %": atr / px * 100, "Hacim ($M)": liq,
        "52H Yüksek": hi52, "52H Düşük": lo52, "Zirveden %": from_hi,
        "EMA fazı": ems.PHASES.get(faz, ems.PHASES["karisik"])[0], "_faz": faz,
        "_faz_sk": ems.PHASES.get(faz, ems.PHASES["karisik"])[1], "ADX": adx_,
        "_ust200": ust200, "_ust50": ust50,
        "Bacak": ("yükseliş" if up else "düşüş") if mf.get("ok") else "—",
        "Bacak %": _num(mf.get("move_pct")), "Fib r": r, "_tm": tm,
        "Zamanlama": TIMING[tm], "Giriş Alt": z_lo, "Giriş Üst": z_hi, "Teyit": teyit,
        "Stop": stop, "Kâr al 1": tp1, "Kâr al 2": tp2, "R:R": rr,
        "Fair value": hedef, "Analist yüksek": _num(fund.get("Hedef Yüksek")),
        "Potansiyel %": (hedef / px - 1) * 100 if np.isfinite(hedef) and px > 0 else np.nan,
        "Analist": (f"{fund.get('Tavsiye', '')} ({int(_num(fund.get('Analist Sayısı')))})"
                    if np.isfinite(_num(fund.get("Analist Sayısı"))) else
                    str(fund.get("Tavsiye", "") or "")),
        "İleri F/K": fpe, "Piyasa Değ. ($B)": mcap, "Büyüme %": grow,
        "_sep": sep, "Sepet": SEPET[sep], "🐈": kedi, "_kirik": kirik,
        "_R": bool(orgu.get("R1")) or bool(orgu.get("R2")),
        "Örgü puanı": _num(orgu.get("Örgü puanı")),
        "Veri tarihi": str(d.index[-1])[:10],
        # kısa vade sinyalleri
        "Getiri 20G %": ret20, "Getiri 60G %": ret60, "_mom_ok": mom_ok,
        "_hi20": bool(px >= 0.97 * hi20),
        "Balina": whale_txt, "Balina uyarısı": " · ".join(wh["neg"]), "_whale": whale_ok,
        "_whale_ev": bool(wh["ev"]),
        "_orgu": bool(orgu.get("R1")) or bool(orgu.get("R2"))
                 or (_num(orgu.get("Örgü puanı")) >= 85),
        "M Giriş Alt": m_lo, "M Stop": m_stop, "M Kâr al 1": m_tp1, "M Kâr al 2": m_tp2,
        # orta / uzun vade sinyalleri
        "FV iskontosu %": disc, "_ucuz": ucuz, "Temel": temel_txt, "_saglam": saglam,
        "Temel eksik": temel_eksik,
        # piyasa iç yapısı (breadth.py): F/K sıkışması ve genişleme adaylığı
        "Geçmiş F/K": tpe_, "F/K sıkışması": fk_s, "_ucuzlayan": ucuzlayan,
        "_genis_baz": bool(np.isfinite(from_hi) and from_hi <= -20 and saglam
                           and faz not in ("tam_ayi", "kirilma")),
    }


def compute(prices: dict[str, pd.DataFrame], fund: pd.DataFrame | None = None,
            orgu: pd.DataFrame | None = None) -> pd.DataFrame:
    fm = ({r["Hisse"]: r for r in fund.to_dict("records")}
          if fund is not None and not fund.empty and "Hisse" in fund else {})
    om = ({r["Sembol"]: r for r in orgu.to_dict("records")}
          if orgu is not None and not orgu.empty and "Sembol" in orgu else {})
    rows = []
    for s, d in prices.items():
        try:
            m = stock_metrics(s, d, fm.get(s), om.get(s))
        except Exception as exc:                 # noqa: BLE001
            _log.info("picks %s: %s", s, exc)
            m = None
        if m:
            rows.append(m)
    S = pd.DataFrame(rows)
    return add_momentum(S)


def add_momentum(S: pd.DataFrame) -> pd.DataFrame:
    """Momentum puanı: 20 ve 60 günlük getirinin evrendeki yüzdeliği (0–100).
    Evren küçükse (< 40 hisse) mutlak eşik: 20G ≥ %8 ve 60G ≥ %15 → 85."""
    if S is None or S.empty or "Getiri 20G %" not in S:
        return S
    if len(S) >= 40:
        sc = (S["Getiri 20G %"].rank(pct=True) * 0.5 + S["Getiri 60G %"].rank(pct=True) * 0.5) * 100
    else:
        sc = np.where((S["Getiri 20G %"] >= 8) & (S["Getiri 60G %"] >= 15), 85.0,
                      np.where(S["Getiri 20G %"] > 0, 55.0, 25.0))
    S["Momentum"] = np.round(sc, 0)
    S["_mom"] = (S["Momentum"] >= 80) & S["_mom_ok"].astype(bool)
    return S


# ------------------------------------------------------------------ dönüş bağlamı
def theme_turn(row: dict[str, Any] | None) -> str:
    """Tema 'bir süredir düşmüş ve pozitife dönüyor' mu? → kısa açıklama ya da ''."""
    if not row:
        return ""
    a3, a1, h1 = (_num(row.get(k)) for k in ("3A %", "1A %", "1H %"))
    q = row.get("_q", "")
    dusmus = np.isfinite(a3) and a3 < -2
    donuyor = q == "iyilesen" or (np.isfinite(h1) and np.isfinite(a1) and h1 > 0 and a1 > 0)
    if dusmus and donuyor:
        return f"tema 3 ayda %{a3:+.0f} düşmüştü, dönüyor (1A %{a1:+.0f}, 1H %{h1:+.0f})"
    return ""


def index_turn(ema_rows: dict[str, dict] | None, name: str = "endeks") -> str:
    """Endeks EMA fazından dönüş: günlükte erken dönüş, ya da haftalık zayıfken günlük boğa."""
    if not ema_rows:
        return ""
    g = str((ema_rows.get("Günlük") or {}).get("Faz", ""))
    w = str((ema_rows.get("Haftalık") or {}).get("Faz", ""))
    wk_weak = any(x in w for x in ("derin düzeltme", "kırılıyor", "Ayı", "ayı", "Yorulma"))
    if "Erken dönüş" in g or (wk_weak and any(x in g for x in ("Tam boğa", "sağlıklı"))):
        return f"{name} dönüyor (G: {g.split(' ', 1)[-1]}, H: {w.split(' ', 1)[-1]})"
    return ""


# ------------------------------------------------------------------ tema × vade
def _why(m: dict[str, Any], q: str) -> str:
    w = []
    if m["🐈"]:
        w.append("🐈 ölü kedi: düşen trendde ani sıçrama — büyük düşüş öncesi örgü")
    if m["_kirik"]:
        w.append(f"kırık bıçak: zirveden %{m['Zirveden %']:.0f}, {m['EMA fazı']}")
    if m["Bacak"] != "—":
        w.append(f"ana bacak %{m['Bacak %']:.0f} {m['Bacak']}, Fib {m['Fib r']:.2f}"
                 + (" (🔑 test edilmiş bölge)" if m["_tm"] in ("kapida", "kapida_zayif") else ""))
    w.append(f"EMA: {m['EMA fazı']}")
    if q:
        w.append({"lider": "tema lider (para giriyor)", "iyilesen": "tema iyileşiyor (taze para)",
                  "zayif": "temadan para çıkmaya başladı", "geride": "tema zayıf"}.get(q, ""))
    if np.isfinite(m["Potansiyel %"]):
        w.append(f"analist hedefi {m['Potansiyel %']:+.0f}% · {m['Analist']}")
    if np.isfinite(m["İleri F/K"]):
        w.append(f"ileri F/K {m['İleri F/K']:.0f}")
    if m["_R"]:
        w.append("🧬 örgü kuralı (R1/R2)")
    if m.get("_mom"):
        w.append(f"🚀 momentum {m['Momentum']:.0f}/100 (20G %{m['Getiri 20G %']:+.0f}, "
                 f"60G %{m['Getiri 60G %']:+.0f})")
    if m.get("Balina"):
        w.append(f"🐋 {m['Balina']}")
    if m.get("Balina uyarısı"):
        w.append(f"uyarı: {m['Balina uyarısı']}")
    if m.get("_donus"):
        w.append(f"📈 {m['_donus']}")
    if np.isfinite(m.get("FV iskontosu %", np.nan)) and m["FV iskontosu %"] >= 20:
        w.append(f"💰 fair value'nun %{m['FV iskontosu %']:.0f} altında")
    if m.get("_ucuzlayan"):
        w.append(f"🧮 ucuzlayan lider: ileri F/K {m['İleri F/K']:.0f}, geçmiş "
                 f"{m['Geçmiş F/K']:.0f} (kazanç fiyattan hızlı büyüyor)"
                 + (" · endeks F/K'sına yakın" if m.get("_ucuz_idx") else ""))
    if m.get("_genis"):
        w.append(f"🌊 genişleme adayı: zirveden %{m['Zirveden %']:.0f}, temel sağlam — piyasa "
                 + ("genişlemeye başladı" if m.get("_rejim") == "genisleme"
                    else "dar liderlikte; genişlerse yakalama potansiyeli"))
    if m.get("Temel", "—") != "—":
        w.append(f"temel {m['Temel']}" + (f" (eksik: {m['Temel eksik']})" if m.get("Temel eksik")
                                           else ""))
    if not np.isfinite(m["Piyasa Değ. ($B)"]):
        w.append("temel veri yok — sepet varsayılan")
    if np.isfinite(m.get("Bilanço (gün)", np.nan)) and m["Bilanço (gün)"] <= 10:
        w.append(f"📅 bilanço {int(m['Bilanço (gün)'])} gün sonra")
    return " · ".join(x for x in w if x)


def decide(m: dict[str, Any], q: str, vade: str, index_ok: bool = True
           ) -> tuple[str, float, str]:
    """Karar etiketi, öncelik puanı (0–100), vade uyumu."""
    tm, sep = m["_tm"], m["_sep"]
    wt, wv, wq, wf = VADE_W.get(vade, VADE_W["orta"])
    t_sc = {"kapida": 20, "kapida_zayif": 8, "bekle_geri": 6, "donus_teyit": 4, "derin": -4,
            "dip": -6, "belirsiz": -8, "bozuk": -25}[tm]
    if tm == "kapida" and not m.get("_ust200", True):
        t_sc = 8                 # test edilen kural EMA200 üstü trend filtresiyle birlikteydi
    q_sc = {"lider": 8, "iyilesen": 12, "zayif": -5, "geride": -10}.get(q, 0)
    pot = m["Potansiyel %"]
    v_sc = float(np.clip(pot, -20, 40)) / 4 if np.isfinite(pot) else 0.0
    sc = 50 + wt * t_sc + wq * q_sc + wf * 8 * m["_faz_sk"] + wv * v_sc
    sc += 6 if m["_R"] else 0
    sc += 4 if np.isfinite(m["Örgü puanı"]) and m["Örgü puanı"] >= 85 else 0
    sc += 4 if np.isfinite(m["R:R"]) and m["R:R"] >= 2 else 0
    if vade == "uzun" and sep == "cekirdek":
        sc += 5
    if vade == "kısa" and sep == "spek":
        sc -= 3
    if sep == "izle":
        sc -= 40 + (20 if m["🐈"] else 0)
    # ---- kullanıcı kuralları (ilave): kısa = momentum / balina / örgü ·
    #      orta-uzun = dönüş (tema ya da endeks) + fair value'nun ≥ %35 altı + sağlam temel
    ks = {"kısa": 1.0, "orta": 0.5, "uzun": 0.3}.get(vade, 0.5)
    ls_ = {"kısa": 0.3, "orta": 1.0, "uzun": 1.0}.get(vade, 1.0)
    mom, whale, orgu_ = bool(m.get("_mom")), bool(m.get("_whale")), bool(m.get("_orgu"))
    sc += ks * (10 * mom + (10 if m.get("_whale_ev") else 4) * whale + 6 * orgu_)
    if m.get("Balina uyarısı") and "dağıtım" in m["Balina uyarısı"]:
        sc -= 10                                 # DIST: 5 yılda −1.49 puan (doğrulandı)
    donus, ucuz, saglam = bool(m.get("_donus")), bool(m.get("_ucuz")), bool(m.get("_saglam"))
    n_val = donus + ucuz + saglam
    sc += ls_ * (18 if n_val == 3 else 8 if n_val == 2 and ucuz else 3 * n_val)
    deger_donus = n_val == 3 and sep != "izle" and m["_faz"] not in ("tam_ayi", "kirilma")
    # piyasa iç yapısı: ucuzlayan lider (F/K düşerken fiyat yükseliyor) · genişleme adayı
    rj = m.get("_rejim", "")
    sc += ls_ * (6 * bool(m.get("_ucuzlayan")) + 3 * bool(m.get("_ucuz_idx")))
    if m.get("_genis"):
        sc += ls_ * (12 if rj == "genisleme" else 5)
    kisa_sig = (mom or whale or orgu_) and sep != "izle" and m["_faz"] not in ("tam_ayi",)
    bil = m.get("Bilanço (gün)", np.nan)
    near_e = np.isfinite(bil) and bil <= 7

    k = ((tm == "kapida" and m.get("_ust200", True)) or (m["_R"] and m["_faz"] in BULL_FAZ)) \
        and sep != "izle"
    o = q in ("lider", "iyilesen") and m["_faz"] in BULL_FAZ and sep != "izle"
    u = (sep in ("cekirdek", "buyume") and np.isfinite(pot) and pot >= 15
         and m["_faz"] not in ("tam_ayi", "kirilma"))
    k = k or kisa_sig
    o = o or deger_donus
    u = u or deger_donus or (ucuz and saglam and sep != "izle")
    uyum = " ".join(x for x, ok in (("K", k), ("O", o), ("U", u)) if ok) or "—"

    if sep == "izle":
        lab = "⛔ İZLE"
    elif tm == "kapida" and not m.get("_ust200", True):
        lab = "🟡 BÖLGEDE — EMA200 altında"
    elif tm == "kapida" and (not np.isfinite(m["R:R"]) or m["R:R"] >= 1.5):
        lab = "✅ ALIM BÖLGESİNDE"
        if not index_ok:
            lab, sc = "🟡 BÖLGEDE — endeks uygun değil", sc - 8
        elif near_e and vade == "kısa":
            lab, sc = "🟡 BÖLGEDE — bilanço yakın", sc - 6
        elif q in ("zayif", "geride") and vade != "uzun":
            lab = "🟡 BÖLGEDE — tema zayıf"
    elif tm in ("kapida", "kapida_zayif"):
        lab = "🟡 BÖLGEDE — teyit zayıf"
    elif tm == "bekle_geri":
        lab = "🟡 LİMİT EMİR (geri çekilme)"
    elif tm == "donus_teyit":
        lab = "🟡 LİMİT EMİR (dönüş testi)"
    elif tm == "derin":
        lab = "⏳ TEYİT BEKLE (0.786 geri alınsın)"
    elif tm == "dip":
        lab = "🔄 DİP TEYİDİ BEKLE"
    else:
        lab = "⏳ BEKLE"
    # ✅ dışındaki kararlar, kullanıcı kurallarıyla yükseltilir (kanıt: test edilmedi)
    if not lab.startswith(("✅", "⛔")):
        if (vade == "kısa" and mom and (whale or orgu_ or m.get("_hi20"))
                and m.get("_ust200", True) and tm in ("bekle_geri", "belirsiz", "derin")):
            lab = "🟢 MOMENTUM — kısa vade"
        elif vade in ("orta", "uzun") and deger_donus:
            lab = "💎 DEĞER DÖNÜŞÜ — kademeli al"
        elif vade in ("orta", "uzun") and m.get("_genis") and rj == "genisleme" \
                and m["_faz"] in BULL_FAZ:
            lab = "🌊 GENİŞLEME ADAYI — kademeli al"
    return lab, float(np.clip(sc, 0, 100)), uyum


def board(S: pd.DataFrame, theme_map: dict[str, list[str]], theme_q: dict[str, str],
          vade: str = "orta", index_ok: bool = True, min_liq: float = 5.0,
          earnings: pd.DataFrame | None = None, turn: dict[str, str] | None = None,
          idx_turn: str = "", mkt: dict | None = None) -> pd.DataFrame:
    """mkt: piyasa iç yapısı bağlamı (breadth.analyze): {'rejim', 'endeks_fk', 'etiket'}."""
    mkt = mkt or {}
    """Tema × hisse satırları (bir hisse birden çok temada olabilir)."""
    if S is None or S.empty:
        return pd.DataFrame()
    em = {}
    if earnings is not None and not earnings.empty and "Kalan Gün" in earnings:
        em = dict(zip(earnings["Hisse"], pd.to_numeric(earnings["Kalan Gün"], errors="coerce")))
    by = {r["Sembol"]: r for r in S.to_dict("records")}
    rows = []
    for tema, syms in theme_map.items():
        q = theme_q.get(tema, "")
        for s in dict.fromkeys(syms):
            m = by.get(s)
            if m is None or not (m["Hacim ($M)"] >= min_liq):
                continue
            m = dict(m)
            m["Bilanço (gün)"] = _num(em.get(s))
            m["_donus"] = (turn or {}).get(tema, "") or idx_turn
            rj, ifk = mkt.get("rejim", ""), _num(mkt.get("endeks_fk"))
            m["_rejim"] = rj
            m["_genis"] = bool(m.get("_genis_baz") and rj in ("boga_ayi", "genisleme")
                               and m["_sep"] != "izle")
            m["_ucuzlayan"] = bool(m.get("_ucuzlayan")) and m["_sep"] != "izle"
            m["_ucuz_idx"] = bool(m["_ucuzlayan"] and np.isfinite(ifk)
                                  and np.isfinite(m["İleri F/K"]) and m["İleri F/K"] <= 1.25 * ifk)
            lab, sc, uy = decide(m, q, vade, index_ok)
            if lab.startswith("🟢"):                 # momentum planı
                m["Giriş Alt"], m["Giriş Üst"] = m["M Giriş Alt"], m["Fiyat"]
                m["Stop"], m["Kâr al 1"], m["Kâr al 2"] = (m["M Stop"], m["M Kâr al 1"],
                                                           m["M Kâr al 2"])
                m["Zamanlama"] = "🚀 Momentum (fiyattan, sıkı stop)"
                m["R:R"] = 2.0
            ksig = " ".join(x for x, ok in (("🚀", m.get("_mom")), ("🐋", m.get("_whale")),
                                            ("🧬", m.get("_orgu"))) if ok) or "—"
            lsig = " ".join(x for x, ok in (("📈", bool(m["_donus"])), ("💰", m.get("_ucuz")),
                                            ("🏛️", m.get("_saglam")), ("🧮", m["_ucuzlayan"]),
                                            ("🌊", m["_genis"])) if ok) or "—"
            rows.append({"Tema": tema, "Hisse": s, "Karar": lab, "Öncelik": round(sc),
                         "Vade uyumu": uy, "Kısa sinyal": ksig, "Orta-uzun sinyal": lsig, **{k: v for k, v in m.items()},
                         "Gerekçe": _why(m, q), "_q": q})
    if not rows:
        return pd.DataFrame()
    B = pd.DataFrame(rows)
    B["_o"] = B["Karar"].str[:1].map({"✅": 0, "🟡": 1, "🟢": 1, "💎": 1, "🌊": 1, "⏳": 2,
                                      "🔄": 2,
                                      "⛔": 3}).fillna(2)
    return B.sort_values(["Tema", "_o", "Öncelik"], ascending=[True, True, False]
                         ).reset_index(drop=True)


def _px(x: float) -> str:
    if not np.isfinite(x):
        return "—"
    return f"{x:,.0f}" if x >= 1000 else f"{x:.2f}" if x >= 1 else f"{x:.4f}"


def display(B: pd.DataFrame) -> pd.DataFrame:
    """Ekran / Excel için biçimli tablo."""
    if B is None or B.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "Tema": B["Tema"], "Karar": B["Karar"], "Sepet": B["Sepet"], "Hisse": B["Hisse"],
        "Öncelik": B["Öncelik"], "Vade": B["Vade uyumu"],
        "Kısa sinyal": B["Kısa sinyal"], "Orta-uzun sinyal": B["Orta-uzun sinyal"],
        "Fiyat": B["Fiyat"].map(_px),
        "Zamanlama": B["Zamanlama"],
        "Giriş bölgesi": [f"{_px(a)} – {_px(b)}" if np.isfinite(a) else "—"
                          for a, b in zip(B["Giriş Alt"], B["Giriş Üst"])],
        "Stop": [f"{_px(s)} ({(s / min(p, (a + b) / 2) - 1) * 100:+.0f}%)"
                 if np.isfinite(s) and np.isfinite(a) else _px(s)
                 for s, p, a, b in zip(B["Stop"], B["Fiyat"], B["Giriş Alt"], B["Giriş Üst"])],
        "Kâr al 1": B["Kâr al 1"].map(_px), "Kâr al 2": B["Kâr al 2"].map(_px),
        "R:R": B["R:R"].round(1),
        "Fair value": B["Fair value"].map(_px), "Potansiyel %": B["Potansiyel %"].round(0),
        "FV iskontosu %": B["FV iskontosu %"].round(0), "Temel": B["Temel"],
        "F/K sıkışması": B["F/K sıkışması"].round(2),
        "Momentum": B["Momentum"], "Balina": B["Balina"].replace("", "—"),
        "İleri F/K": B["İleri F/K"].round(0), "Piyasa Değ. ($B)": B["Piyasa Değ. ($B)"].round(1),
        "Zirveden %": B["Zirveden %"].round(0), "EMA fazı": B["EMA fazı"],
        "Gerekçe": B["Gerekçe"]})
    return out


def theme_summary(B: pd.DataFrame, top: int = 3) -> pd.DataFrame:
    """Tema Avcısı için: tema başına sayılar ve en iyi adaylar."""
    if B is None or B.empty:
        return pd.DataFrame()
    rows = []
    for tema, g in B.groupby("Tema", sort=False):
        g = g.sort_values(["_o", "Öncelik"], ascending=[True, False])
        act = g[g["_o"] <= 1]
        best = ", ".join(f"{r['Hisse']} ({r['Karar'][:1]} {r['Öncelik']})"
                         for _, r in act.head(top).iterrows()) or "—"
        rows.append({"Tema": tema, "✅ Bölgede": int((g["_o"] == 0).sum()),
                     "🟡 Limit / teyit": int((g["_o"] == 1).sum()),
                     "⏳ Bekle": int((g["_o"] == 2).sum()),
                     "⛔ İzle": int((g["_o"] == 3).sum()),
                     "En iyi adaylar": best,
                     "_s": float(act["Öncelik"].head(top).mean()) if len(act) else 0.0})
    return pd.DataFrame(rows)
