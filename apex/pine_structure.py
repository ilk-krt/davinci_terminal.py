# apex/pine_structure.py — QUANTUM 885 yapı modüllerinin Python karşılığı
#
# Pine kaynağı: QUANTUM_885.pine — varsayılan ayarlarla, bar bar (repaint yok):
#   23X  🏛️ Akıllı Destek/Direnç (S/R Matrix) — kısa/orta/uzun pivot kümeleri
#   23Z  🎯 Boşluk Dolum Stratejisi (GFR) — 4 şartlı kontrol listesi
#   24   ⚓ VSA PRO Wyckoff bölgeleri (Spring/Upthrust/Climax/Effort) — mitigasyona kadar
#   27X  🧲 Gap Matrix — FVG / GAP / KURUMSAL (Miller) tek kutu, birleştirme + kısmi dolum
#   29B  💧 Likidite havuzları — eşit tepe/dip kümeleri + süpürme/kırılım olayı
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from apex.indicators import ema, highest, lowest, mfi, percentrank, roc, sma
from apex.pine_fib import atr_rma

# ------------------------------------------------------------------ ayarlar
GAP_MIN = 0.5            # ② GAP minimum yüzde
MILLER_ATR = 1.0         # ③ gövde eşiği (× ATR) — haftalık 0.5×, aylık 0.3×
MILLER_SENS = 3
GAP_MAX = 30
GAP_MERGE_TOL, GAP_MERGE_WIN = 0.3, 20
BOX_AGE = 300            # kutu yaş sınırı (bar)

SRX_WIN, SRX_PV = 250, (5, 20, 60)
SRX_TOL, SRX_N, SRX_MAXD = 0.25, 3, 25.0

GFR_DDLB, GFR_DDMIN, GFR_GAPMIN, GFR_AGEMAX = 60, 15.0, 0.5, 50
GFR_DMIN, GFR_DMAX, GFR_SUPLB = 1.0, 45.0, 20
GFR_CVDLEN, GFR_MFILEN, GFR_MFITHR, GFR_FNEED = 5, 14, 50.0, 2
GFR_BEMA, GFR_STOPATR, GFR_E = 50, 1.0, (40, 35, 25)
RS_LEN = 20

VSA_LB, VSA_HI, VSA_ULTRA, VSA_LO = 20, 80, 93, 25
VSA_CPREV, VSA_CPCLIM, VSA_MAX = 0.50, 0.40, 25

LQ_PIV, LQ_TOL, LQ_MIN, LQ_LOOK, LQ_MAXD = 5, 0.35, 2, 300, 15.0
ZONE_MAXD = 15.0        # fiyattan bu % uzaktaki Wyckoff bölgeleri kararı etkilemez

HZ_TXT = {0: "kısa", 1: "orta", 2: "uzun"}
TIER_TXT = {1: "FVG", 2: "GAP", 3: "KURUMSAL"}


# ------------------------------------------------------------------ yardımcılar
def pivots(x: np.ndarray, left: int, right: int, high: bool) -> np.ndarray:
    """ta.pivothigh/pivotlow — pivot, ONAY barında (i) i-right fiyatıyla döner."""
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(left + right, n):
        c = i - right
        v = x[c]
        if not np.isfinite(v):
            continue
        lw, rw = x[c - left:c], x[c + 1:i + 1]
        if high:
            if (lw < v).all() and (rw <= v).all():
                out[i] = v
        elif (lw > v).all() and (rw >= v).all():
            out[i] = v
    return out


def _prep(df: pd.DataFrame, last_frac: float | None) -> pd.DataFrame:
    d = df[["Open", "High", "Low", "Close", "Volume"]].astype(float).copy()
    d["Volume"] = d["Volume"].fillna(0.0)
    if last_frac and 0 < last_frac < 1:          # açık barın hacmini tam bara ölçekle
        d.iloc[-1, d.columns.get_loc("Volume")] /= last_frac
    return d


# ------------------------------------------------------------------ 23X S/R Matrix
def sr_matrix(d: pd.DataFrame, atr: pd.Series) -> dict[str, Any]:
    h, l, c, v = (d[k].to_numpy() for k in ("High", "Low", "Close", "Volume"))
    vma = sma(d["Volume"], 50).to_numpy()
    a = atr.to_numpy()
    n = len(d)
    piv = []            # (px, w, hz, bar)
    for hz, ln in enumerate(SRX_PV):
        ph, pl = pivots(h, ln, ln, True), pivots(l, ln, ln, False)
        lo = lowest(d["Close"], ln).to_numpy()
        hi = highest(d["Close"], ln).to_numpy()
        for i in np.flatnonzero(np.isfinite(ph) | np.isfinite(pl)):
            b = i - ln
            for px, is_res in ((ph[i], True), (pl[i], False)):
                if not np.isfinite(px):
                    continue
                w = 1.0
                if np.isfinite(vma[b]) and vma[b] > 0 and v[b] > vma[b]:
                    w += min(2.0, v[b] / vma[b] - 1.0)
                if np.isfinite(a[b]) and a[b] > 0:
                    mv = (px - lo[i]) / a[b] if is_res else (hi[i] - px) / a[b]
                    w += min(3.0, max(0.0, mv) * 0.6)
                piv.append((px, w, hz, b))
    last = n - 1
    piv = [p for p in piv if last - p[3] <= SRX_WIN]
    close, tol = c[-1], a[-1] * SRX_TOL
    clusters = []
    if piv:
        piv.sort(key=lambda p: p[0])
        cur: list = []
        for p in piv + [None]:
            if p is None or (cur and p[0] - cur[-1][0] > tol):
                ws = sum(q[1] for q in cur)
                hzs = {q[2] for q in cur}
                clusters.append({"px": sum(q[0] * q[1] for q in cur) / max(ws, 1e-9),
                                 "score": len(cur) + len(hzs) + ws * 0.5,
                                 "hz": max(hzs), "cnt": len(cur),
                                 "b0": min(q[3] for q in cur)})
                cur = []
            if p is not None:
                cur.append(p)
    res, sup = [], []
    for cl in sorted(clusters, key=lambda x: -x["score"]):
        if abs(cl["px"] - close) / close * 100 <= SRX_MAXD and cl["cnt"] >= 2:
            if cl["px"] > close and len(res) < SRX_N:
                res.append(cl)
            if cl["px"] < close and len(sup) < SRX_N:
                sup.append(cl)
    res.sort(key=lambda x: x["px"] - close)
    sup.sort(key=lambda x: close - x["px"])
    return {"res": res, "sup": sup, "atr": a[-1], "close": close}


# ------------------------------------------------------------------ 27X Gap Matrix
def gap_matrix(d: pd.DataFrame, atr: pd.Series, tf: str) -> dict[str, Any]:
    o, h, l, c, v = (d[k].to_numpy() for k in ("Open", "High", "Low", "Close", "Volume"))
    a = atr.to_numpy()
    vavg = sma(d["Volume"], 20).to_numpy()
    mult = MILLER_ATR * (0.3 if tf == "Aylık" else 0.5 if tf == "Haftalık" else 1.0)
    vreq = 1.0 if tf in ("Aylık", "Haftalık") else 1.2
    sens = 1 if tf in ("Aylık", "Haftalık") else MILLER_SENS
    btop, bbot = np.maximum(o, c), np.minimum(o, c)
    boxes: list[dict] = []          # aktif (mitige olmamış) kutular
    born: list[tuple[int, int, int]] = []   # (bar, dir, tier) — yeni kutu olayları
    filled_ev: list[tuple[int, int, int]] = []
    n = len(d)
    for i in range(2, n):
        ai = a[i] if np.isfinite(a[i]) else np.nan
        rv = v[i] / (vavg[i] if np.isfinite(vavg[i]) and vavg[i] > 0 else 1.0)
        volok = rv > vreq
        mup = mdn = False
        if np.isfinite(ai):
            mup = (bbot[i] - btop[i - 1]) > ai * mult and volok
            mdn = (bbot[i - 1] - btop[i]) > ai * mult and volok
            if mup:
                mup = all(bbot[i] > btop[i - k] for k in range(1, sens + 1) if i - k >= 0)
            if mdn:
                mdn = all(btop[i] < bbot[i - k] for k in range(1, sens + 1) if i - k >= 0)
        fvg_up = l[i] > h[i - 2] and c[i] > o[i]
        fvg_dn = h[i] < l[i - 2] and c[i] < o[i]
        pu = (l[i] - h[i - 1]) / h[i - 1] * 100 if h[i - 1] > 0 else 0.0
        pd_ = (l[i - 1] - h[i]) / h[i] * 100 if h[i] > 0 else 0.0
        gup = l[i] > h[i - 1] and pu >= GAP_MIN
        gdn = h[i] < l[i - 1] and pd_ >= GAP_MIN
        t_up = 3 if mup else 2 if gup else 1 if fvg_up else 0
        t_dn = 3 if mdn else 2 if gdn else 1 if fvg_dn else 0
        for tier, dr in ((t_up, 1), (t_dn, -1)):
            if tier == 0:
                continue
            lb = 2 if tier == 1 else 1
            if tier == 3:
                top = bbot[i] if dr > 0 else bbot[i - 1]
                bot = btop[i - 1] if dr > 0 else btop[i]
            else:
                top = l[i] if dr > 0 else l[i - lb]
                bot = h[i - lb] if dr > 0 else h[i]
            if top <= bot:
                continue
            born.append((i, dr, tier))
            mi = -1
            if boxes and np.isfinite(ai):
                tp = ai * GAP_MERGE_TOL
                for j in range(len(boxes) - 1, max(0, len(boxes) - GAP_MERGE_WIN) - 1, -1):
                    bx = boxes[j]
                    if bx["dir"] == dr and top >= bx["bot"] - tp and bot <= bx["top"] + tp:
                        mi = j
                        break
            if mi >= 0:
                bx = boxes[mi]
                bx["tier"] = max(bx["tier"], tier)
                bx["top"], bx["bot"] = max(bx["top"], top), min(bx["bot"], bot)
                bx["last"] = i
            else:
                boxes.append({"top": top, "bot": bot, "dir": dr, "tier": tier,
                              "left": i - lb, "last": i, "top0": top, "bot0": bot})
                if len(boxes) > GAP_MAX:
                    boxes.pop(0)
        # mitigasyon + kısmi dolum + yaş
        for j in range(len(boxes) - 1, -1, -1):
            bx = boxes[j]
            if i <= bx["left"] + 1:
                continue
            dr = bx["dir"]
            if (l[i] <= bx["bot"]) if dr > 0 else (h[i] >= bx["top"]):
                filled_ev.append((i, dr, bx["tier"]))
                boxes.pop(j)
            elif i - bx["left"] > BOX_AGE:
                boxes.pop(j)
            else:
                if dr > 0 and l[i] < bx["top"]:
                    bx["top"] = l[i]
                if dr < 0 and h[i] > bx["bot"]:
                    bx["bot"] = h[i]
    return {"boxes": boxes, "born": born, "filled": filled_ev, "n": n}


# ------------------------------------------------------------------ 24 VSA PRO bölgeleri
def vsa_zones(d: pd.DataFrame) -> dict[str, Any]:
    o, h, l, c, v = (d[k].to_numpy() for k in ("Open", "High", "Low", "Close", "Volume"))
    vol = d["Volume"]
    vavg = sma(vol, 20).to_numpy()
    spread = np.maximum(h - l, 1e-9)
    sp = pd.Series(spread, index=d.index)
    aspr = sma(sp, 20).to_numpy()
    vpct = percentrank(vol, 100).fillna(50.0).to_numpy()
    cpos = (c - l) / spread
    delta = ((c - l) - (h - c)) / spread
    phi = highest(d["High"], VSA_LB).shift(1).to_numpy()
    plo = lowest(d["Low"], VSA_LB).shift(1).to_numpy()
    zones: list[dict] = []
    events: list[tuple[int, str, int, int]] = []    # (bar, ad, yön, güç)
    n = len(d)
    for i in range(1, n):
        ok = np.isfinite(vavg[i]) and vavg[i] > 0
        if not ok or not np.isfinite(aspr[i]):
            continue
        wide, narrow = spread[i] > aspr[i] * 1.4, spread[i] < aspr[i] * 0.7
        hiv, ult, lov = vpct[i] >= VSA_HI, vpct[i] >= VSA_ULTRA, vpct[i] <= VSA_LO
        spring = l[i] < plo[i] and c[i] > plo[i] and hiv and cpos[i] > VSA_CPREV
        upth = h[i] > phi[i] and c[i] < phi[i] and hiv and cpos[i] < 1 - VSA_CPREV
        sclim = (not spring) and c[i] < o[i] and wide and ult and cpos[i] > VSA_CPCLIM
        bclim = (not upth) and c[i] > o[i] and wide and ult and cpos[i] < 1 - VSA_CPCLIM
        effort = ult and spread[i] < aspr[i] * 0.6
        nd = c[i] > c[i - 1] and narrow and lov
        ns = c[i] < c[i - 1] and narrow and lov
        name = ("SPRING" if spring else "UPTHRUST" if upth else "SELL CLIMAX" if sclim
                else "BUY CLIMAX" if bclim else "EFFORT≠RESULT" if effort
                else "NO DEMAND" if nd else "NO SUPPLY" if ns else "")
        if name:
            bias = (1 if (spring or sclim or ns) else -1 if (upth or bclim or nd) else 0)
            if name == "EFFORT≠RESULT":
                bias = 1 if delta[i] >= 0 else -1
            sv = max(0.0, (vpct[i] - 50) / 50)
            ss = min(1.0, spread[i] / max(aspr[i], 1e-9) / 2)
            sc = abs(cpos[i] - 0.5) * 2
            sw = min(1.0, max(h[i] - max(o[i], c[i]), min(o[i], c[i]) - l[i]) / spread[i] * 2)
            stg = int(round(100 * (0.40 * sv + 0.25 * ss + 0.20 * sc + 0.15 * sw)))
            events.append((i, name, bias, stg))
            if name not in ("NO DEMAND", "NO SUPPLY"):       # ND/NS bölge değil, işaret
                zones.append({"top": h[i], "bot": l[i], "side": bias, "name": name,
                              "str": stg, "left": i - 1, "bar": i})
                if len(zones) > VSA_MAX:
                    zones.pop(0)
        for j in range(len(zones) - 1, -1, -1):
            z = zones[j]
            if i <= z["left"] + 2:
                continue
            mit = c[i] < z["bot"] if z["side"] > 0 else c[i] > z["top"]
            if mit or i - z["left"] > BOX_AGE:
                zones.pop(j)
    return {"zones": zones, "events": events, "n": n}


# ------------------------------------------------------------------ 29B Likidite havuzları
def liquidity(d: pd.DataFrame, atr: pd.Series) -> dict[str, Any]:
    h, l, c = (d[k].to_numpy() for k in ("High", "Low", "Close"))
    a = atr.to_numpy()
    ph, pl = pivots(h, LQ_PIV, LQ_PIV, True), pivots(l, LQ_PIV, LQ_PIV, False)
    pools: list[dict] = []
    events: list[tuple[int, int, float, int]] = []   # (bar, ±1/±2, px, adet)
    for i in range(len(d)):
        if not np.isfinite(a[i]):
            continue
        tol, buf = a[i] * LQ_TOL, a[i] * LQ_TOL * 0.5
        ev = None
        for k in range(len(pools) - 1, -1, -1):
            p = pools[k]
            sd, px = p["side"], p["px"]
            pierced = h[i] > px + buf if sd > 0 else l[i] < px - buf
            if pierced or i - p["b1"] > LQ_LOOK:
                if pierced and p["cnt"] >= LQ_MIN:
                    rev = c[i] < px if sd > 0 else c[i] > px
                    if ev is None or p["cnt"] > 2:
                        ev = (i, sd * (2 if rev else 1), px, p["cnt"])
                pools.pop(k)
        if ev:
            events.append(ev)
        for px, sd in ((ph[i], 1), (pl[i], -1)):
            if not np.isfinite(px):
                continue
            pb = i - LQ_PIV
            hit = next((k for k in range(len(pools) - 1, -1, -1)
                        if pools[k]["side"] == sd and abs(pools[k]["px"] - px) <= tol), -1)
            if hit >= 0:
                p = pools[hit]
                p["cnt"] += 1
                p["px"] = max(p["px"], px) if sd > 0 else min(p["px"], px)
                p["b1"] = pb
            else:
                pools.append({"px": px, "side": sd, "cnt": 1, "b0": pb, "b1": pb})
                if len(pools) > 120:
                    pools.pop(0)
    close = c[-1]
    up = [p for p in pools if p["side"] > 0 and p["cnt"] >= LQ_MIN and p["px"] > close]
    dn = [p for p in pools if p["side"] < 0 and p["cnt"] >= LQ_MIN and p["px"] < close]
    return {"up": min(up, key=lambda p: p["px"]) if up else None,
            "dn": max(dn, key=lambda p: p["px"]) if dn else None,
            "events": events, "n": len(d)}


# ------------------------------------------------------------------ 23Z GFR
def gfr(d: pd.DataFrame, atr: pd.Series, bench: pd.Series | None,
        s1: float | None) -> dict[str, Any]:
    h, l, c, v = (d[k].to_numpy() for k in ("High", "Low", "Close", "Volume"))
    n = len(d)
    top = bot = np.nan
    gbar = -1
    for i in range(1, n):
        if h[i] < l[i - 1] * (1 - GFR_GAPMIN / 100):
            top, bot, gbar = l[i - 1], h[i], i
        if np.isfinite(top) and c[i] >= top:
            top = bot = np.nan
            gbar = -1
    close = c[-1]
    age = n - 1 - gbar if gbar >= 0 else None
    hi = float(highest(d["High"], GFR_DDLB).iloc[-1])
    dd = (hi - close) / hi * 100 if hi > 0 else 0.0
    dist = (top / close - 1) * 100 if np.isfinite(top) else None
    c1 = (dist is not None and dd >= GFR_DDMIN and age is not None and age <= GFR_AGEMAX
          and GFR_DMIN <= dist <= GFR_DMAX)
    fb = float(lowest(d["Low"], GFR_SUPLB).iloc[-1])
    sup = s1 if (s1 is not None and s1 < close) else fb
    c2 = close > sup
    rng = np.maximum(h - l, 1e-9)
    delta = v * (2 * (c - l) / rng - 1)
    cvd = float(delta[-GFR_CVDLEN:].sum())
    hlc3 = (d["High"] + d["Low"] + d["Close"]) / 3
    mf = float(mfi(hlc3, d["Volume"], GFR_MFILEN).iloc[-1])
    f_cvd, f_mfi = cvd > 0, (np.isfinite(mf) and mf > GFR_MFITHR)
    fhit = int(f_cvd) + int(f_mfi)
    c3 = fhit >= GFR_FNEED
    regime = rs_up = True
    if bench is not None:
        b = bench.reindex(d.index).ffill()
        be = ema(b, GFR_BEMA)
        regime = bool(np.isfinite(b.iloc[-1]) and b.iloc[-1] > be.iloc[-1])
        rs = roc(d["Close"] / b, RS_LEN).iloc[-1]
        rs_up = bool(np.isfinite(rs) and rs > 0)
    c4 = regime and rs_up
    ready = c1 and c2 and c3 and c4
    a = float(atr.iloc[-1])
    stop = sup - GFR_STOPATR * a
    risk = close - stop if close > stop else None
    rr2 = (top - close) / risk if (risk and np.isfinite(top)) else None
    st2 = np.isfinite(bot) and close > bot
    st3 = st2 and close > bot + (top - bot) * 0.5
    weight = (GFR_E[0] + (GFR_E[1] if st2 else 0) + (GFR_E[2] if st3 else 0)) if ready else 0
    kill_sup = close < stop
    kill_flow = fhit == 0
    kill_risk = not regime
    kill = np.isfinite(top) and (kill_sup or kill_flow or kill_risk
                                 or (age is not None and age > GFR_AGEMAX))
    why = ("destek/stop kırıldı" if kill_sup else "para girişi kesildi" if kill_flow
           else "endeks rejimi bozuldu" if kill_risk else "boşluk fazla eskidi")
    return {"top": top if np.isfinite(top) else None, "bot": bot if np.isfinite(bot) else None,
            "age": age, "dd": dd, "dist": dist, "c": (c1, c2, c3, c4), "ready": ready,
            "kill": bool(kill), "why": why, "sup": sup, "stop": stop, "rr2": rr2,
            "weight": weight, "fhit": fhit, "regime": regime, "st2": st2, "st3": st3}


# ------------------------------------------------------------------ modüller
def _m(modul, durum, notu, grup):
    return {"Modül": modul, "Durum": float(np.clip(durum, -1, 1)), "Not": notu,
            "Grup": grup}


def _fmt(x: float) -> str:
    return f"{x:,.0f}" if x >= 1000 else f"{x:.2f}"


def modules(df: pd.DataFrame, bench: pd.Series | None, tf: str,
            last_frac: float | None = None) -> list[dict[str, Any]]:
    if df is None or len(df) < 80:
        return []
    d = _prep(df, last_frac)
    atr = atr_rma(d, 14)
    close = float(d["Close"].iloc[-1])
    a = float(atr.iloc[-1])
    n = len(d)
    out = []

    # --- Gap / FVG
    g = gap_matrix(d, atr, tf)
    bull = [b for b in g["boxes"] if b["dir"] > 0]
    bear = [b for b in g["boxes"] if b["dir"] < 0]
    s, parts = 0.0, []
    new = [x for x in g["born"] if x[0] >= n - 3]
    if new:
        i, dr, t = max(new, key=lambda x: (x[2], x[0]))
        parts.append(f"{'▲' if dr > 0 else '▼'} yeni {TIER_TXT[t]} ({n - 1 - i} bar önce)")
        s += dr * (0.25 + 0.15 * t)
    near = 8.0                                     # % — "yakın" boşluk
    if bull:
        b = max(bull, key=lambda x: x["top"])
        dp = (close / b["top"] - 1) * 100
        parts.append(f"altta açık destek boşluğu {TIER_TXT[b['tier']]} "
                     f"{_fmt(b['bot'])}–{_fmt(b['top'])} (%{dp:.1f} aşağıda)")
        if dp <= near:
            s += 0.15 + 0.05 * b["tier"]
    if bear:
        b = min(bear, key=lambda x: x["bot"])
        dp = (b["bot"] / close - 1) * 100
        parts.append(f"üstte doldurulmamış boşluk {TIER_TXT[b['tier']]} "
                     f"{_fmt(b['bot'])}–{_fmt(b['top'])} (%{dp:.1f} yukarıda)")
        if dp <= near:
            s -= 0.15 + 0.05 * b["tier"]
    out.append(_m("Gap / Fair Value Gap", s,
                  " · ".join(parts) if parts else "açık boşluk yok", "Yapı"))

    # --- S/R Matrix
    sr = sr_matrix(d, atr)
    rs_, ss_ = sr["res"], sr["sup"]
    txt = []
    if rs_:
        txt.append("Direnç " + ", ".join(f"{_fmt(x['px'])} ({HZ_TXT[x['hz']]})" for x in rs_))
    if ss_:
        txt.append("Destek " + ", ".join(f"{_fmt(x['px'])} ({HZ_TXT[x['hz']]})" for x in ss_))
    dR = (rs_[0]["px"] - close) / a if rs_ else 6.0
    dS = (close - ss_[0]["px"]) / a if ss_ else 6.0
    s = (dR - dS) / max(dR + dS, 1e-9) * 0.5
    if rs_ and dR < 0.5:
        txt.insert(0, "⚠️ dirence dayandı")
        s -= 0.2
    elif ss_ and dS < 0.5:
        txt.insert(0, "🛡️ desteğin üstünde")
        s += 0.2
    if not rs_:
        txt.insert(0, "üstte %25 içinde direnç yok")
    out.append(_m("Kısa/orta/uzun vade dirençler", s, " · ".join(txt) or "seviye yok", "Yapı"))

    # --- Kurumsal bölgeler (VSA Wyckoff) + likidite havuzu
    vz = vsa_zones(d)
    lim = ZONE_MAXD / 100
    dem = [z for z in vz["zones"] if z["side"] > 0 and z["bot"] <= close * 1.02
           and z["top"] >= close * (1 - lim)]
    sup_ = [z for z in vz["zones"] if z["side"] < 0 and z["top"] >= close * 0.98
            and z["bot"] <= close * (1 + lim)]
    s, parts = 0.0, []
    if dem:
        z = max(dem, key=lambda z: z["top"])
        inside = z["bot"] <= close <= z["top"]
        parts.append(f"🟢 alım bölgesi {z['name']} {_fmt(z['bot'])}–{_fmt(z['top'])}"
                     + (" (fiyat içinde)" if inside else ""))
        s += 0.3 + (0.2 if inside else 0) + 0.1 * (len(dem) - 1)
    if sup_:
        z = min(sup_, key=lambda z: z["bot"])
        inside = z["bot"] <= close <= z["top"]
        parts.append(f"🔴 satış bölgesi {z['name']} {_fmt(z['bot'])}–{_fmt(z['top'])}"
                     + (" (fiyat içinde)" if inside else ""))
        s -= 0.3 + (0.2 if inside else 0) + 0.1 * (len(sup_) - 1)
    rec = [e for e in vz["events"] if e[0] >= n - 5 and e[1] not in ("NO DEMAND", "NO SUPPLY")]
    if rec:
        e = rec[-1]
        parts.append(f"son olay {e[1]} güç {e[3]} ({n - 1 - e[0]} bar önce)")
        s += 0.25 * e[2]
    out.append(_m("Kurumsal alım-satım bölgeleri", s,
                  " · ".join(parts) if parts else "aktif Wyckoff bölgesi yok", "Akıllı para"))

    lq = liquidity(d, atr)
    s, parts = 0.0, []
    if lq["up"] and lq["up"]["px"] <= close * (1 + LQ_MAXD / 100):
        p = lq["up"]
        parts.append(f"üst havuz 💧{p['cnt']} {_fmt(p['px'])} (%{(p['px'] / close - 1) * 100:.1f})")
    if lq["dn"] and lq["dn"]["px"] >= close * (1 - LQ_MAXD / 100):
        p = lq["dn"]
        parts.append(f"alt havuz 💧{p['cnt']} {_fmt(p['px'])} (%{(1 - p['px'] / close) * 100:.1f})")
    ev = [e for e in lq["events"] if e[0] >= n - 3]
    if ev:
        i, k, px, cnt = ev[-1]
        nm = {2: "üst havuz SÜPÜRÜLDÜ ↩ (ayı)", 1: "üst havuz KIRILDI → (boğa)",
              -2: "alt havuz SÜPÜRÜLDÜ ↩ (boğa)", -1: "alt havuz KIRILDI → (ayı)"}[k]
        parts.insert(0, f"{nm} {_fmt(px)}")
        s += {2: -0.5, 1: 0.5, -2: 0.5, -1: -0.5}[k]
    out.append(_m("Likidite havuzları", s, " · ".join(parts) or "yakın havuz yok", "Akıllı para"))

    # --- GFR
    s1 = ss_[0]["px"] if ss_ else None
    gf = gfr(d, atr, bench, s1)
    ticks = "".join(("✅" if ok else "❌") + k for ok, k in zip(gf["c"], "①②③④"))
    if gf["top"] is None:
        out.append(_m("GFR", 0.0, "⚪ doldurulacak aşağı boşluk yok", "Yapı"))
    elif gf["kill"]:
        out.append(_m("GFR", -0.4, f"⚠️ YENİDEN DEĞERLENDİR — {gf['why']} · hedef "
                                   f"{_fmt(gf['top'])} (%{gf['dist']:.1f})", "Yapı"))
    elif gf["ready"]:
        rr = f" · {gf['rr2']:.1f}R" if gf["rr2"] else ""
        out.append(_m("GFR", 0.6 + 0.2 * gf["st2"] + 0.2 * gf["st3"],
                      f"🟢 KURULUM AKTİF — kademe %{gf['weight']} · T1 {_fmt(gf['bot'])} / "
                      f"T2 {_fmt(gf['top'])} · stop {_fmt(gf['stop'])}{rr}", "Yapı"))
    else:
        out.append(_m("GFR", 0.1 * sum(gf["c"]) - 0.1,
                      f"⏳ şartlar bekleniyor {ticks} · hedef {_fmt(gf['top'])} "
                      f"(%{gf['dist']:.1f}) · düşüş %{gf['dd']:.0f} · {gf['age']} bar", "Yapı"))
    return out
