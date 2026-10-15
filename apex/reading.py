# apex/reading.py — grafiğin altına "teknik analizci" okuması
#
# Kurallara dayalı, tekrarlanabilir yorum (yapay zekâ metni değil): her cümle
# tfview.build içindeki bir ölçümden gelir. Sıra:
#   1) Özet hüküm  2) Trend & mum  3) Momentum  4) Yapı (Fib, Elliott, S/R, boşluk)
#   5) Akıllı para & hacim  6) Seviyeler ve senaryolar
from __future__ import annotations

from typing import Any

import numpy as np

from apex.tfview import mod


def _f(x: float) -> str:
    return (f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.2f}" if abs(x) >= 1
            else f"{x:.4g}")


def _pct(a: float, b: float) -> str:
    return f"%{(a / b - 1) * 100:+.1f}"


def _note(v, name, n=140) -> str | None:
    m = mod(v, name)
    if not m:
        return None
    t = str(m["Not"])
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + "…"


def verdict(score: float) -> tuple[str, str]:
    if not np.isfinite(score):
        return "⚪", "değerlendirilemedi"
    if score >= 35:
        return "🟢", "güçlü olumlu — trend ve sinyaller alıcıdan yana"
    if score >= 12:
        return "🟢", "olumlu — alıcılar önde ama her şey aynı yönde değil"
    if score > -12:
        return "⚪", "kararsız — sinyaller çelişiyor, yön teyidi beklenmeli"
    if score > -35:
        return "🔴", "olumsuz — satıcılar önde"
    return "🔴", "güçlü olumsuz — trend ve sinyaller satıcıdan yana"


def read(v: dict[str, Any]) -> str:
    if not v.get("ok"):
        return f"_{v.get('neden', 'veri yok')}_"
    d = v["d"]
    px = float(d["Close"].iloc[-1])
    atr = v["atr"]
    tf = v["tf"]
    unit = tf.lower()          # "günlük kapanış", "haftalık kapanış", "aylık kapanış"
    ic, ver = verdict(v.get("score", np.nan))
    out = [f"**{ic} {tf} hüküm: {ver}** (skor {v.get('score', 0):+.0f})"]

    # 2) trend & mum
    parts = []
    t = _note(v, "Trend")
    if t:
        parts.append(f"Trend: {t}.")
    t = _note(v, "Mum Gücü (v666)", 90)
    if t and "YETERSİZ" not in t:
        parts.append(f"Son mum: {t.split('·')[0].strip()}.")
    t = _note(v, "Hacim", 90)
    if t and v.get("has_vol"):
        parts.append(f"Hacim: {t}.")
    if parts:
        out.append("**Trend ve mum.** " + " ".join(parts))

    # 2b) EMA şeridi (10/21/50/100/200)
    a = v.get("ema_stack") or {}
    if a.get("ok"):
        from apex import ema_stack as ems
        s = [f"{a['etiket']} — {a['sure']} bardır sürüyor; {a['aciklama']}.",
             f"Sıra: {a['sira']}. Eğim: {a['egim']}. Yelpaze {a['yelpaze']:.1f} ATR, "
             f"{a['yelpaze_txt']}. Fiyat EMA21'den {a['uzama']:+.1f} ATR uzakta"
             + (" — aşırı uzama, kovalamak riskli." if a["uzama"] > 3 else
                " — aşırı satım, tepki gelebilir." if a["uzama"] < -3 else ".")]
        if a["kesisim"]:
            s.append("Kesişimler: " + "; ".join(a["kesisim"]) + ".")
        lv = ems.levels_txt(a)
        if lv:
            s.append(lv[0].upper() + lv[1:] + ".")
        ht = ems.history_txt(a, tf)
        if ht:
            s.append(ht)
        out.append("**EMA dizilimi.** " + " ".join(s))

    # 3) momentum
    parts = []
    for nm, lab in (("RSI (14)", "RSI"), ("Konfluans", "Konfluans"),
                    ("Omni momentum", "Omni"), ("Fusion histogram", "Fusion")):
        t = _note(v, nm, 70)
        if t:
            parts.append(f"{lab} {t}")
    t = _note(v, "Tükenme")
    if t and "işaret" in t:
        parts.append("⚠️ tükenme işareti var — yükselişi kovalamak riskli")
    t = _note(v, "Sıkışma")
    if t and "sürüyor" in t:
        parts.append(f"🔒 {t}")
    if parts:
        out.append("**Momentum.** " + " · ".join(parts) + ".")

    # 4) yapı
    parts = []
    fb = v.get("fib") or {}
    if fb.get("ok"):
        yon = "yükseliş" if fb["up"] else "düşüş"
        parts.append(f"Ana yapı bacağı %{fb['move_pct']:.0f}'lik bir {yon}; fiyat bu "
                     f"bacağın {fb['zone']} bölgesinde (oran {fb['r']:.2f}).")
    e = v.get("ell")
    if e and e.get("zone"):
        z = e["zone"]
        if z.get("dead"):
            parts.append(f"Elliott: son sayım bozuldu ({e['phase']}); dalga senaryosu "
                         f"şu an güvenilir değil.")
        elif min(z["zlo"], z["zhi"]) > 0:
            s = (f"Elliott: {e['phase']}. Hedef bölgesi {_f(z['zlo'])}–{_f(z['zhi'])} "
                 f"({_pct(z['near'], px)}).")
            if z.get("stop") is not None:
                s += (f" Senaryo {_f(z['stop'])} {'üzerinde' if z['up'] else 'altında'} "
                      "kaldıkça canlı.")
            if z.get("inv") is not None:
                s += f" {_f(z['inv'])} geçilirse sayım çöker."
            if z.get("hit"):
                s += " Fiyat hedef bölgesine ulaştı — kâr alma bölgesi."
            parts.append(s)
    elif e:
        parts.append(f"Elliott: {e.get('phase', '—')}.")
    t = _note(v, "Gap / Fair Value Gap", 160)
    if t and t != "açık boşluk yok":
        parts.append(f"Boşluklar: {t}.")
    if parts:
        out.append("**Yapı.** " + " ".join(parts))

    # 5) akıllı para
    parts = []
    for nm in ("VSA & Delta", "Whale / Retail", "Kurumsal alım-satım bölgeleri",
               "Likidite havuzları", "Whale Momentum Profile", "Q-dry", "Volatility Hole"):
        t = _note(v, nm, 130)
        if t and not t.startswith(("STANDBY", "normal", "yakın havuz yok",
                                   "aktif Wyckoff bölgesi yok")):
            parts.append(f"{nm}: {t}")
    if not v.get("has_vol"):
        parts.append("bu seride hacim verisi yok (oran/endeks); hacim modülleri atlandı")
    if parts:
        out.append("**Akıllı para ve hacim.** " + " · ".join(parts) + ".")

    # 6) seviyeler ve senaryolar
    sr = v.get("sr") or {}
    res, sup = sr.get("res", []), sr.get("sup", [])
    lv = []
    if res:
        lv.append("Dirençler: " + ", ".join(
            f"{_f(r['px'])} ({_pct(r['px'], px)})" for r in res))
    if sup:
        lv.append("Destekler: " + ", ".join(
            f"{_f(r['px'])} ({_pct(r['px'], px)})" for r in sup))
    if lv:
        out.append("**Seviyeler.** " + " · ".join(lv) + f". Ortalama {unit} oynaklık "
                   f"(ATR) {_f(atr)} ({atr / px * 100:.1f}%).")
    sc = []
    if sup:
        s1 = sup[0]["px"]
        tgt = res[0]["px"] if res else px + 2 * atr
        sc.append(f"🟢 **Boğa senaryosu:** {_f(s1)} desteği üzerinde kaldıkça hedef "
                  f"{_f(tgt)} ({_pct(tgt, px)})"
                  + (f", aşılırsa {_f(res[1]['px'])}" if len(res) > 1 else "") + ".")
        s2 = sup[1]["px"] if len(sup) > 1 else s1 - 2 * atr
        sc.append(f"🔴 **Ayı senaryosu:** {_f(s1)} altında {unit} kapanış gelirse "
                  f"{_f(s2)} ({_pct(s2, px)}) test edilir.")
    elif res:
        sc.append(f"🔴 Fiyatın altında yakın destek yok; {_f(res[0]['px'])} direnci geri "
                  f"alınmadıkça baskı sürer.")
    g = v.get("gfr")
    if g and g.get("top"):
        if g.get("ready"):
            sc.append(f"🎯 GFR kurulumu aktif: doldurulacak boşluk {_f(g['bot'])}–{_f(g['top'])}, "
                      f"stop {_f(g['stop'])}.")
    if sc:
        out.append("\n".join(f"- {x}" for x in sc))
    return "\n\n".join(out)
