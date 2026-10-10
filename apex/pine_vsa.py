# apex/pine_vsa.py — "VSA & Delta Engine v4/v5" Python karşılığı
#
# Pine kaynağı: VSA_DELTA.pine — varsayılan ayarlarla. Alt zaman dilimi verisi
# burada yok; delta, Pine'ın yedek yolu olan HİBRİT TAHMİN ile hesaplanır
# (mum içi kapanış konumu %37.5 + BVC %62.5).
#   • Akış: alış payı (bp), delta, CVD, yönlü akış dengesizliği (−1..+1)
#   • VSA anomalileri (Squat, Buy/Sell Climax, Upthrust, Effort≠Result, ND/NS)
#   • Wyckoff ilişkileri: soğurma / kolay hareket (hacim yoğunluğu, ATR-normalize)
#   • Wyckoff olayları: SC/BC, AR, ST, Yay/Upthrust (sığ+şok), SOS/SOW
#   • ⭐/🔻 Soğurma dönüş sinyali ve Baskınlık dönüşü
#   • Hacim hükmü: 20 barda fiyat × CVD yönü
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from apex.indicators import ema, highest, lowest, sma
from apex.pine_fib import atr_rma

BVC_W, BVC_LEN = 0.625, 20
VOL_LEN, DEV_HIGH, DEV_ULTRA, LOW_MULT = 20, 2.0, 3.0, 0.5
WIDE, NARROW = 1.5, 0.7
REV_THIRD, REV_DROP, REV_DRY = 0.333, 0.15, 0.60
FLIP_RUN, FLIP_CONV, FLIP_WIN, FLIP_CVDW, FLIP_TRB, FLIP_ABSW, FLIP_CD = 3, 0.08, 4, 2, 5, 10, 5
WVD_LEN, WVD_HI, WVD_LO, WVD_MINR = 20, 1.8, 0.45, 1.0
WFLOW_LEN, WFLOW_R = 20, 2.5
WTR_LEN, WTR_MAX, WTR_MIN, WPEN, WST_TOL, WST_BARS = 30, 8.0, 1.0, 0.10, 0.5, 30
VERD_LEN, SCALE_LEN = 20, 100
RECENT = 3              # "son" sayılan bar sayısı

BULL_EV = {"YAY (Spring)", "İKİNCİL TEST ↑", "OTOMATİK RALLİ", "GÜÇ İŞARETİ"}
BEAR_EV = {"UPTHRUST", "İKİNCİL TEST ↓", "OTOMATİK TEPKİ", "ZAYIFLIK İŞARETİ"}


def _ncdf(x: np.ndarray) -> np.ndarray:
    z = np.clip(x, -8, 8)
    t = 1.0 / (1.0 + 0.2316419 * np.abs(z))
    d = 0.3989422804014327 * np.exp(-z * z / 2.0)
    p = d * t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    return np.where(z > 0, 1.0 - p, p)


def compute(df: pd.DataFrame, last_frac: float | None = None) -> dict[str, Any]:
    d = df[["Open", "High", "Low", "Close", "Volume"]].astype(float).copy()
    d["Volume"] = d["Volume"].fillna(0.0)
    if last_frac and 0 < last_frac < 1:
        d.iloc[-1, d.columns.get_loc("Volume")] /= last_frac
    o, h, l, c, v = (d[k].to_numpy() for k in ("Open", "High", "Low", "Close", "Volume"))
    n = len(d)
    rng = h - l
    cprev = np.r_[c[0], c[:-1]]
    cp = np.where(rng > 0, (c - l) / np.where(rng > 0, rng, 1),
                  np.where(c > cprev, 1.0, np.where(c < cprev, 0.0, 0.5)))
    ret = np.log(c / np.maximum(cprev, 1e-12))
    sd = pd.Series(ret).rolling(BVC_LEN).std(ddof=0).to_numpy()
    bvc = _ncdf(np.where(sd > 0, ret / np.where(sd > 0, sd, 1), 0.0))
    bvc = np.where(np.isfinite(sd), bvc, 0.5)
    bp = np.clip((1 - BVC_W) * cp + BVC_W * bvc, 0, 1)
    buy = v * bp
    sell = v - buy
    delta = buy - sell
    cvd = np.cumsum(delta)

    vs = pd.Series(v)
    vma = sma(vs, VOL_LEN).to_numpy()
    vsd = vs.rolling(VOL_LEN).std(ddof=0).to_numpy()
    zv = np.where(vsd > 0, (v - vma) / np.where(vsd > 0, vsd, 1), 0.0)
    rvol = np.where(vma > 0, v / np.where(vma > 0, vma, 1), 0.0)
    sprma = sma(pd.Series(rng), VOL_LEN).to_numpy()
    wide = (sprma > 0) & (rng > sprma * WIDE)
    narrow = (sprma > 0) & (rng < sprma * NARROW)
    vU, vH = zv >= DEV_ULTRA, zv >= DEV_HIGH
    vL = (vma > 0) & (v < vma * LOW_MULT)
    vprev = np.r_[np.nan, v[:-1]]
    vprev2 = np.r_[np.nan, np.nan, v[:-2]]
    less2 = (v < np.nan_to_num(vprev)) & (v < np.nan_to_num(vprev2))
    isUp, isDn = c > cprev, c < cprev
    e20 = ema(d["Close"], 20).to_numpy()
    upC, dnC = c > e20, c < e20
    hprev = np.r_[h[0], h[:-1]]
    lprev = np.r_[l[0], l[:-1]]
    c2 = np.r_[c[0], c[0], c[:-2]] if n > 1 else c
    vHprev = np.r_[False, vH[:-1]]
    anom = {
        "Squat": vU & narrow,
        "Buy Climax": isUp & vH & wide & (cp < 0.45) & upC,
        "Sell Climax": isDn & vH & wide & (cp > 0.55) & dnC,
        "No Demand": isUp & narrow & vL & less2,
        "No Supply": isDn & narrow & vL & less2,
        "Upthrust": (h > hprev) & (c < cprev) & (cp < 0.35) & wide & vH,
        "Effort≠Result": (cprev > c2) & vHprev & (c < lprev),
    }

    # Wyckoff ilişkileri
    atr = atr_rma(d, 14).to_numpy()
    sprN = np.where(atr > 0, rng / np.where(atr > 0, atr, 1), np.nan)
    vdraw = np.where(np.isfinite(sprN) & (sprN > 0), rvol / np.maximum(sprN, 0.05), np.nan)
    vdf = pd.Series(vdraw).ffill().fillna(1.0)
    vdma = sma(vdf, WVD_LEN).to_numpy()
    vdrel = np.where(vdma > 0, vdf.to_numpy() / np.where(vdma > 0, vdma, 1), np.nan)
    absorb = np.isfinite(vdrel) & (vdrel >= WVD_HI) & (rvol >= WVD_MINR)
    absDn, absUp = absorb & (c < o), absorb & (c > o)
    ease = np.isfinite(vdrel) & (vdrel <= WVD_LO) & (np.nan_to_num(sprN) > 0.8)
    bma = ema(pd.Series(buy), WFLOW_LEN).to_numpy()
    sma_ = ema(pd.Series(sell), WFLOW_LEN).to_numpy()
    fsum = bma + sma_
    flow_imb = np.where(fsum > 0, (bma - sma_) / np.where(fsum > 0, fsum, 1), 0.0)

    # Wyckoff olayları
    trHi = highest(d["High"], WTR_LEN).shift(1).to_numpy()
    trLo = lowest(d["Low"], WTR_LEN).shift(1).to_numpy()
    trW = np.where(atr > 0, (trHi - trLo) / np.where(atr > 0, atr, 1), np.nan)
    trOk = np.isfinite(trW) & (trW <= WTR_MAX) & (trW >= WTR_MIN)
    lowV, shockV = rvol <= 0.8, rvol >= 2.0
    events: list[tuple[int, str]] = []
    scB = bcB = -99999
    scL = scV = bcH = bcV = np.nan
    for i in range(n):
        ai = atr[i] if np.isfinite(atr[i]) else 0.0
        tlo = trLo[i] if np.isfinite(trLo[i]) else l[i]
        thi = trHi[i] if np.isfinite(trHi[i]) else h[i]
        sc = vU[i] and wide[i] and c[i] < o[i] and cp[i] > 0.40 and l[i] <= tlo
        bc = vU[i] and wide[i] and c[i] > o[i] and cp[i] < 0.60 and h[i] >= thi
        if sc:
            scB, scL, scV = i, l[i], v[i]
        if bc:
            bcB, bcH, bcV = i, h[i], v[i]
        ar = np.isfinite(scL) and 1 <= i - scB <= 10 and c[i] > o[i] and (c[i] - scL) > ai * 1.5
        ard = np.isfinite(bcH) and 1 <= i - bcB <= 10 and c[i] < o[i] and (bcH - c[i]) > ai * 1.5
        st = (np.isfinite(scL) and abs(l[i] - scL) <= ai * WST_TOL and 2 < i - scB <= WST_BARS
              and v[i] < scV * 0.7 and rng[i] < ai * 1.2)
        stu = (np.isfinite(bcH) and abs(h[i] - bcH) <= ai * WST_TOL and 2 < i - bcB <= WST_BARS
               and v[i] < bcV * 0.7 and rng[i] < ai * 1.2)
        typ = lowV[i] or shockV[i]
        spr = trOk[i] and l[i] < trLo[i] - ai * WPEN and c[i] > trLo[i] and typ
        ut = trOk[i] and h[i] > trHi[i] + ai * WPEN and c[i] < trHi[i] and typ
        sos = trOk[i] and c[i] > trHi[i] and c[i] > o[i] and wide[i] and rvol[i] >= 1.5 and cp[i] > 0.6
        sow = trOk[i] and c[i] < trLo[i] and c[i] < o[i] and wide[i] and rvol[i] >= 1.5 and cp[i] < 0.4
        tag = " sığ" if lowV[i] else " şok" if shockV[i] else ""
        ev = ("YAY (Spring)" if spr else "UPTHRUST" if ut else "SATIŞ DORUĞU" if sc
              else "ALIŞ DORUĞU" if bc else "İKİNCİL TEST ↑" if st else "İKİNCİL TEST ↓" if stu
              else "OTOMATİK RALLİ" if ar else "OTOMATİK TEPKİ" if ard
              else "GÜÇ İŞARETİ" if sos else "ZAYIFLIK İŞARETİ" if sow else "")
        if ev:
            events.append((i, ev, tag if ev in ("YAY (Spring)", "UPTHRUST") else ""))

    # ⭐/🔻 soğurma dönüşü + baskınlık dönüşü (CVD 'çizilen' ölçekte)
    cv = pd.Series(cvd)
    chi, clo = cv.rolling(SCALE_LEN).max().to_numpy(), cv.rolling(SCALE_LEN).min().to_numpy()
    vhi = vs.rolling(SCALE_LEN).max().to_numpy()
    den = chi - clo
    cvdS = np.where(den > 0, (cvd - clo) / np.where(den > 0, den, 1) * vhi, vhi * 0.5)
    rev_up = np.zeros(n, bool)
    rev_dn = np.zeros(n, bool)
    flip_up = np.zeros(n, bool)
    flip_dn = np.zeros(n, bool)
    runS = runB = 0
    runS_prev = runB_prev = 0
    lastU = lastD = -99999
    last_absDn = last_absUp = -99999
    convDn_hist: list[bool] = []
    convUp_hist: list[bool] = []
    tUp_hist: list[bool] = []
    tDn_hist: list[bool] = []
    for i in range(n):
        if absDn[i]:
            last_absDn = i
        if absUp[i]:
            last_absUp = i
        runS_prev, runB_prev = runS, runB
        runS = runS + 1 if bp[i] < 0.5 else 0
        runB = runB + 1 if bp[i] >= 0.5 else 0
        s0 = cvdS[i]
        s1 = cvdS[i - 1] if i >= 1 else np.nan
        s2 = cvdS[i - 2] if i >= 2 else np.nan
        s0, s1, s2 = (0.0 if not np.isfinite(x) else x for x in (s0, s1, s2))
        tU = s0 > s1 and s1 < s2
        tD = s0 < s1 and s1 > s2
        convDn_hist.append(bp[i] < 0.5 and abs(bp[i] - 0.5) <= FLIP_CONV)
        convUp_hist.append(bp[i] >= 0.5 and abs(bp[i] - 0.5) <= FLIP_CONV)
        tUp_hist.append(tU)
        tDn_hist.append(tD)
        if i < 2:
            continue
        bp1 = bp[i - 1]
        # soğurma dönüşü (1 bar soğurma, dönüş okuması 'Herhangi biri')
        absU = REV_THIRD < bp1 < 0.5
        absD = 0.5 <= bp1 < 1 - REV_THIRD
        dry = rvol[i] <= REV_DRY and rvol[i - 1] > REV_DRY
        tUpB = (bp[i] >= 0.5 and bp1 < 0.5) or (bp[i] <= REV_DROP and bp1 > REV_DROP) or dry
        tDnB = (bp[i] < 0.5 and bp1 >= 0.5) or (bp[i] >= 1 - REV_DROP and bp1 < 1 - REV_DROP) or dry
        rev_up[i] = absU and s1 < s2 and tUpB and s0 >= s1
        rev_dn[i] = absD and s1 > s2 and tDnB and s0 <= s1
        # baskınlık dönüşü
        sConvDn = any(convDn_hist[-FLIP_WIN:])
        sConvUp = any(convUp_hist[-FLIP_WIN:])
        sTU, sTD = any(tUp_hist[-FLIP_CVDW:]), any(tDn_hist[-FLIP_CVDW:])
        cb = c[i - FLIP_TRB] if i >= FLIP_TRB else c[i]
        fu = (bp[i] >= 0.5 and bp1 < 0.5 and runS_prev >= FLIP_RUN and sConvDn and sTU
              and c[i] < cb and i - last_absDn <= FLIP_ABSW)
        fd = (bp[i] < 0.5 and bp1 >= 0.5 and runB_prev >= FLIP_RUN and sConvUp and sTD
              and c[i] > cb and i - last_absUp <= FLIP_ABSW)
        if fu and i - lastU > FLIP_CD:
            flip_up[i], lastU = True, i
        if fd and i - lastD > FLIP_CD:
            flip_dn[i], lastD = True, i

    k = min(VERD_LEN, n - 1)
    pu = c[-1] > c[-1 - k]
    cu = cvdS[-1] > cvdS[-1 - k]
    pch = (c[-1] / c[-1 - k] - 1) * 100
    verdict = ("TEYİTLİ YÜKSELİŞ" if pu and cu else "ZAYIF YÜKSELİŞ — dağıtım şüphesi" if pu
               else "TOPLAMA — fiyat düşerken alım" if cu else "TEYİTLİ DÜŞÜŞ")
    return {"bp": bp, "rvol": rvol, "flow_imb": flow_imb, "anom": anom, "events": events,
            "absDn": absDn, "absUp": absUp, "ease": ease, "c": c, "o": o,
            "rev_up": rev_up, "rev_dn": rev_dn, "flip_up": flip_up, "flip_dn": flip_dn,
            "verdict": verdict, "pu": pu, "cu": cu, "pch": pch, "n": n, "k": k}


def _ago(arr: np.ndarray, within: int = RECENT) -> int | None:
    hits = np.flatnonzero(arr[-within:])
    return None if len(hits) == 0 else int(within - 1 - hits[-1])


def modules(df: pd.DataFrame, tf: str, last_frac: float | None = None) -> list[dict[str, Any]]:
    if df is None or len(df) < 60 or df["Volume"].fillna(0).sum() <= 0:
        return []
    r = compute(df, last_frac)
    n = r["n"]
    s = {"TEYİTLİ YÜKSELİŞ": 0.4, "ZAYIF YÜKSELİŞ — dağıtım şüphesi": -0.25,
         "TOPLAMA — fiyat düşerken alım": 0.3, "TEYİTLİ DÜŞÜŞ": -0.4}[r["verdict"]]
    fi = float(r["flow_imb"][-1])
    s += 0.3 * fi
    parts = [f"{r['verdict']} ({r['k']} bar fiyat {r['pch']:+.1f}%, CVD "
             f"{'↑' if r['cu'] else '↓'})",
             f"alış payı %{r['bp'][-1] * 100:.0f} · akış dengesi {fi:+.2f} · hacim {r['rvol'][-1]:.2f}×"]
    sigs = []
    for arr, txt, w in ((r["rev_up"], "⭐ soğurma sonrası yukarı dönüş", 0.35),
                        (r["rev_dn"], "🔻 dağıtım sonrası aşağı dönüş", -0.35),
                        (r["flip_up"], "⭐ baskınlık alıcıya geçti", 0.35),
                        (r["flip_dn"], "🔻 baskınlık satıcıya geçti", -0.35)):
        k = _ago(arr)
        if k is not None:
            sigs.append((k, txt, w))
    if sigs:                                       # yalnızca EN SON sinyal sayılır
        k, txt, w = min(sigs, key=lambda x: x[0])
        parts.append(f"{txt} ({k} bar önce)")
        s += w
    if r["absDn"][-1]:
        parts.append("düşüş barında soğurma — talep giriyor"); s += 0.15
    elif r["absUp"][-1]:
        parts.append("yükseliş barında soğurma — arz giriyor"); s -= 0.15
    ev = [e for e in r["events"] if e[0] >= n - 5]
    if ev:
        i, nm, tag = ev[-1]
        parts.append(f"Wyckoff: {nm}{tag} ({n - 1 - i} bar önce)")
        s += 0.2 if nm in BULL_EV else -0.2 if nm in BEAR_EV else 0.0
    an = [nm for nm, arr in r["anom"].items() if bool(arr[-RECENT:].any())]
    if an:
        parts.append("VSA: " + ", ".join(an))
        bull = {"Sell Climax", "No Supply"}
        bear = {"Buy Climax", "No Demand", "Upthrust", "Effort≠Result"}
        s += 0.1 * sum(1 for a in an if a in bull) - 0.1 * sum(1 for a in an if a in bear)
    return [{"Modül": "VSA & Delta", "Durum": float(np.clip(s, -1, 1)),
             "Not": " · ".join(parts), "Grup": "Akıllı para"}]
