# apex/pine_v710.py — ŞAHANE V710 (HUD HUB) modüllerinin Python karşılığı
#
# Pine kaynağı: V710_2.pine — sabit varsayılan ayarlarla:
#   • RS-düzeltilmiş fiyat f_C ve Füzyon hızı (f_speed_q / f_sig_q, gear_top)
#   • Efor çizgisi ve Efor Kırılımı (EMA200 'Esnek' trend filtresi)
#   • 🕳️ Volatility Hole (ODI Navigator) — volatilite sıçraması × RS dağılımı
#   • ◯ Adaptive Hole Engine — Bollinger ⊂ Keltner sıkışması + bant teması
#   • 🌟 Q-DRY (Qullamaggie kuruması) + Efor Kırılımıyla onay + bölge
#   • 🐋 Whale Momentum Profili — konviksiyon ağırlıklı POC / VAH / VAL
#   • 🌟 Efor Kırılımı güç ölçer (0–8)
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from apex.indicators import atr, ema, highest, lowest, rsi, sma, stdev, wma

LOOKBACK = 200        # RS ölçekleme penceresi
VWM_LEN = 14          # efor çizgisi
ODI_SENS, ODI_CD = 2.0, 15
Q_TREND, Q_VOL, Q_SENS = 15.0, 0.5, 1.0
QZ_K, QZ_TOL = 0.9, 2
QDRY_CONFIRM = 10
ULT_LEN, ULT_MULT = 50, 1.5
MOM_ROWS, MOM_VA = 100, 70.0


def _kc(c: pd.Series, h: pd.Series, l: pd.Series, n: int, mult: float):
    """Pine ta.kc (useTrueRange=true): EMA tabanı ± EMA(TR)·mult."""
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
                   axis=1).max(axis=1)
    mid = ema(c, n)
    rng = ema(tr, n)
    return mid, mid + rng * mult, mid - rng * mult


def _bb(c: pd.Series, n: int, mult: float):
    mid = sma(c, n)
    dev = stdev(c, n) * mult
    return mid, mid + dev, mid - dev


def _w_pwr(df: pd.DataFrame) -> pd.Series:
    """V709 orijinal Whale Power (log tabanlı) — Ultimate kanalı için."""
    h, l, c, v = df["High"], df["Low"], df["Close"], df["Volume"]
    rngc = (h - l).clip(lower=0.001)
    delta = ((c - l) - (h - c)) / rngc
    dv = sma(delta * v, 20) / sma(v, 20).clip(lower=0.001)
    rv_raw = v / sma(v, 20).clip(lower=1)
    rv = np.where(rv_raw > 2.5, 2.5 + np.log(np.maximum(rv_raw - 1.5, 1e-9)), rv_raw)
    x = ((rsi(c, 14) - 50) + dv * 40) * rv * 1.5 / 5
    lp = np.log1p(np.exp(np.clip(x, -50, 50))) * 5
    raw = np.minimum(np.power(np.log10(1 + lp) * 65, 0.8) * 1.8, 100)
    return wma(pd.Series(raw, index=df.index), 2)


def compute(df: pd.DataFrame, bench: pd.Series | None, tf: str) -> dict[str, Any]:
    """Bütün V710 modüllerini hesaplar; son bar durumu + seriler."""
    o, h, l, c, v = (df[k] for k in ("Open", "High", "Low", "Close", "Volume"))
    n = len(df)
    is_macro = tf in ("Haftalık", "Aylık")

    # --- RS düzeltilmiş fiyat (f_C) ve Füzyon hızı
    b = bench.reindex(df.index).ffill() if bench is not None else c
    denom = b.where(b != 0, 1.0).fillna(c)
    scale = sma(c, LOOKBACK) / sma(c / denom, LOOKBACK).clip(lower=0.001)
    scale = scale.bfill().fillna(1.0)
    fC, fH, fL = c / denom * scale, h / denom * scale, l / denom * scale
    L100 = 24 if is_macro else 100
    fmacd = ema(fC, 12) - ema(fC, 26)
    fspeed = (fmacd - lowest(fmacd, L100)) / (highest(fmacd, L100) - lowest(fmacd, L100)).clip(lower=0.001) * 100 - 50
    fsig = ema(fspeed, 9)
    fhist = fspeed - fsig
    gear = (fhist / highest(fhist.abs(), L100).clip(lower=0.001) * 5 + 5).round().clip(0, 10)

    # --- sensörler
    a14 = atr(h, l, c, 14)
    ema5 = ema(c, 5)
    r7, r14, r21 = rsi(c, 7), rsi(c, 14), rsi(c, 21)
    curl_up = (r7 > r7.shift()) & (r14 > r14.shift()) & (r21 > r21.shift())

    # --- Efor çizgisi + kırılım (EMA200 'Esnek' filtre)
    raw_eff = wma(c * v, VWM_LEN) / wma(v, VWM_LEN).clip(lower=0.001)
    eff = wma(raw_eff, 3)
    e200 = ema(c, 200) if n >= 200 else pd.Series(np.nan, index=df.index)
    rising = e200 > e200.shift().fillna(e200)
    ready = e200.notna()
    ok_up = (~ready) | (c > e200) | rising
    ok_dn = (~ready) | (c < e200) | (~rising)
    eff_up = (c > eff) & (c.shift() <= eff.shift()) & ok_up
    eff_dn = (c < eff) & (c.shift() >= eff.shift()) & ok_dn

    # --- Ultimate kanal kesişimi (güç ölçer için)
    wp = _w_pwr(df)
    ub = sma(c, ULT_LEN)
    udev = ULT_MULT * (1 - wp / 250) * stdev(c, ULT_LEN)
    uce, ufl = ub + udev, ub - udev
    ult_up = ((c > uce) & (c.shift() <= uce.shift())) | ((c > ufl) & (c.shift() <= ufl.shift()))

    # --- 🕳️ Volatility Hole (ODI): kaynak = fiyatın kendisi (Pine varsayılanı)
    ret = (fC - fC.shift()) / fC.shift().clip(lower=0.001)
    disp = stdev(ret, 30)
    spike = (c - sma(c, 50)).abs() > stdev(c, 50) * ODI_SENS
    opp = spike & (disp > sma(disp, 50))
    raw_bull = opp & (fC > ema5) & (fspeed > fsig)
    raw_bear = opp & (fC < ema5) & (fspeed < fsig)
    odi_bull = np.zeros(n, bool)
    odi_bear = np.zeros(n, bool)
    last = -10**9
    rb, rr = raw_bull.to_numpy(), raw_bear.to_numpy()
    for i in range(n):
        if (rb[i] or rr[i]) and i - last > ODI_CD:
            odi_bull[i], odi_bear[i] = rb[i], rr[i]
            last = i

    # --- ◯ Adaptive Hole: BB ⊂ KC sıkışması + bant teması
    _, bup, blo = _bb(c, 20, 2.0)
    kmid, kup, klo = _kc(c, h, l, 20, 1.5)
    sqz_vh = (blo > klo) & (bup < kup)
    half = (kup - kmid) / 3
    hole_dn = sqz_vh & (c <= kmid - half)
    hole_up = sqz_vh & (c >= kmid + half)
    in_sqz = (2 * stdev(fC, 20)) < (1.5 * sma(sma((fH - fL).abs(), 14), 20))

    # --- 🌟 Q-DRY
    body_top, body_bot = np.maximum(o, c), np.minimum(o, c)
    safe = c.clip(lower=1e-4)
    adr = sma(h - l, 20).fillna(h - l) / safe * 100
    adj = Q_TREND * np.maximum(0.5, adr / 3.0)
    roc60 = (c / c.shift(60) - 1).fillna(0) * 100
    roc20 = (c / c.shift(20) - 1).fillna(0) * 100
    q_trend = (roc60 > adj) | (roc20 > adj / 2)
    hi60 = highest(h, 60).fillna(h)
    allow = np.clip(adr * 0.05, 0.05, 0.25)
    q_shallow = c > hi60 * (1 - allow)
    av50 = sma(v.shift(), 50).fillna(v.shift()).fillna(1)
    sv3 = sma(v.shift(), 3).fillna(v.shift()).fillna(1)
    q_dry = sv3 < av50 * Q_VOL
    body = (body_top - body_bot) / safe * 100
    q_tight = body <= adr * 1.5 * Q_SENS
    q_ready = (q_trend & q_shallow & q_dry & q_tight).to_numpy()
    eu = eff_up.to_numpy()
    last_q = None
    q_conf = np.zeros(n, bool)
    for i in range(n):
        if q_ready[i]:
            last_q = i
        q_conf[i] = eu[i] and last_q is not None and i - last_q <= QDRY_CONFIRM

    # Q-DRY bölgesi (son tetik için): geriye daralma taraması
    zone = None
    if last_q is not None:
        maxB = 40 if is_macro else 60
        minB = 3 if is_macro else 4
        hh, ll, aa = h.to_numpy(), l.to_numpy(), a14.to_numpy()
        j = miss = 0
        for bb in range(1, maxB + 1):
            k = last_q - bb
            if k < 0:
                break
            if aa[last_q] > 0 and hh[k] - ll[k] <= QZ_K * aa[last_q]:
                j, miss = bb, 0
            else:
                miss += 1
                if miss > QZ_TOL:
                    break
                j = bb
        back = max(j, minB - 1)
        seg = slice(max(0, last_q - back), last_q + 1)
        zone = {"bar": last_q, "bars": back + 1, "top": float(hh[seg].max()),
                "bot": float(ll[seg].min())}

    # --- 🐋 Whale Momentum Profili (konviksiyon ağırlıklı, son ~6 ay)
    prof = whale_profile(df)

    # --- 🌟 Efor kırılımı güç ölçer (0–8)
    omni = _omni5(df)
    score = (q_ready.astype(int) + odi_bull.astype(int) + hole_up.to_numpy().astype(int)
             + ult_up.to_numpy().astype(int) + curl_up.to_numpy().astype(int)
             + (omni >= 55).to_numpy().astype(int) + (gear >= 6).to_numpy().astype(int)
             + eu.astype(int))
    score = np.minimum(score, 8)

    return {"f_speed": fspeed, "f_sig": fsig, "gear": gear, "eff": eff,
            "eff_up": eff_up, "eff_dn": eff_dn, "odi_bull": odi_bull,
            "odi_bear": odi_bear, "hole_up": hole_up, "hole_dn": hole_dn,
            "sqz_vh": sqz_vh, "in_sqz": in_sqz, "q_ready": q_ready,
            "q_conf": q_conf, "q_last": last_q, "q_zone": zone,
            "profile": prof, "eff_score": score, "close": c, "atr": a14}


def _omni5(df: pd.DataFrame) -> pd.Series:
    """V710 mum boyama Omni skoru (5 osilatör, sabit varsayılanlar)."""
    from apex.indicators import cci, mfi
    c = df["Close"]
    hlc3 = (df["High"] + df["Low"] + c) / 3
    pc = c.diff()
    tsi = 100 * ema(ema(pc, 25), 13) / ema(ema(pc.abs(), 25), 13).clip(lower=0.001)
    return (rsi(c, 7) + rsi(c, 14) + mfi(hlc3, df["Volume"], 14)
            + ((cci(hlc3, 20) + 200) / 4).clip(0, 100) + (tsi + 50).clip(0, 100)) / 5


def whale_profile(df: pd.DataFrame, days: int = 180, rows: int = MOM_ROWS,
                  max_bars: int = 1500) -> dict[str, Any] | None:
    """Pine bölüm 5 + 14: konviksiyon ağırlığı, kapanış satırına yatırılır."""
    h, l, c, v = (df[k] for k in ("High", "Low", "Close", "Volume"))
    vol = v.fillna(0)
    vavg = sma(vol, 20)
    volok = vavg > 0
    spread = (h - l).clip(lower=1e-9)
    delta = ((c - l) - (h - c)) / spread
    rvol = np.where(volok, np.minimum(vol / vavg.clip(lower=1), 2.5), 1.0)
    rsv = rsi(c, 14).fillna(50)
    dvol = np.where(volok, sma(delta * vol, 20) / sma(vol, 20).clip(lower=1), sma(delta, 20))
    base = np.clip(((rsv - 50) + dvol * 50) * rvol * 1.5, -500, 500)
    xs = base / 5.0
    lp = np.where(xs > 20, xs, np.log1p(np.exp(np.minimum(xs, 20)))) * 5.0
    pw = pd.Series(np.minimum(np.power(np.log10(1 + np.maximum(lp, 0)) * 65, 0.8) * 1.8, 100),
                   index=df.index)
    med = pw.rolling(252).median()
    med = med.fillna(pw.rolling(60).median()).fillna(pw)
    mad = (pw - med).abs().rolling(252).median()
    mad = mad.fillna((pw - med).abs().rolling(60).median()).fillna(1.0)
    z = np.where(mad > 0, (pw - med) / (1.4826 * mad), 0.0)
    e = np.exp(np.clip(z, -30, 30))
    conv = (e - 1) / (e + 1) * 100
    w = np.abs(conv) * np.sqrt(np.minimum(rvol, 4.0)) * np.maximum(np.abs(delta), 0.05)
    w = w * np.where(delta >= 0, 1.0, -1.0)
    w = pd.Series(w, index=df.index)

    idx = pd.to_datetime(df.index)
    try:
        idx = idx.tz_localize(None)
    except TypeError:
        pass
    start = idx[-1] - pd.Timedelta(days=days)
    sel = (idx >= start)
    sel[:-max_bars] = False
    if sel.sum() <= 5:
        return None
    hs, ls, cs, ws = h[sel], l[sel], c[sel], w[sel].fillna(0)
    mHi, mLo = float(hs.max()), float(ls.min())
    rs = (mHi - mLo) / rows
    if rs <= 0:
        return None
    bull = np.zeros(rows)
    bear = np.zeros(rows)
    rc = np.clip(((cs - mLo) / rs).astype(int), 0, rows - 1).to_numpy()
    for r_, wv in zip(rc, ws.to_numpy()):
        if wv >= 0:
            bull[r_] += abs(wv)
        else:
            bear[r_] += abs(wv)
    tot = bull + bear
    grand = tot.sum()
    if grand <= 0:
        return None
    poc = int(np.argmax(tot))
    acc, up, dn, guard = tot[poc], poc + 1, poc - 1, 0
    target = grand * MOM_VA / 100
    while acc < target and guard < rows * 2:
        guard += 1
        vu = tot[up] if up <= rows - 1 else -1
        vd = tot[dn] if dn >= 0 else -1
        if vu < 0 and vd < 0:
            break
        if vu >= vd:
            acc += vu; up += 1
        else:
            acc += vd; dn -= 1
    return {"poc": mLo + (poc + 0.5) * rs, "vah": mLo + min(up, rows) * rs,
            "val": mLo + max(dn + 1, 0) * rs,
            "bull_share": float(bull.sum() / grand)}


# ------------------------------------------------------------------ karar modülleri
def _m(modul, durum, notu, grup):
    return {"Modül": modul, "Durum": float(np.clip(durum, -1, 1)), "Not": notu,
            "Grup": grup}


def _ago(arr, n_last: int = 10) -> int | None:
    a = np.asarray(arr)[-n_last:]
    hits = np.nonzero(a)[0]
    return None if len(hits) == 0 else int(len(a) - 1 - hits[-1])


def modules(df: pd.DataFrame, bench: pd.Series | None, tf: str) -> list[dict[str, Any]]:
    if df is None or len(df) < 60:
        return []
    r = compute(df, bench, tf)
    out = []
    # Q-DRY
    n = len(df)
    lq = r["q_last"]
    if r["q_conf"][-10:].any():
        k = _ago(r["q_conf"])
        out.append(_m("Q-dry", 1.0, f"🌟✅ CONFIRMED — Q-DRY sonrası efor kırılımı "
                                   f"({k} bar önce)", "Volatilite"))
    elif lq is not None and n - 1 - lq <= QDRY_CONFIRM:
        z = r["q_zone"]
        out.append(_m("Q-dry", 0.5, f"⏳ BEKLEMEDE — {n - 1 - lq} bar önce tetiklendi, "
                                    f"bölge {z['bars']} bar ({z['bot']:.2f}–{z['top']:.2f}); "
                                    f"efor kırılımı bekleniyor", "Volatilite"))
    else:
        out.append(_m("Q-dry", 0.0, "STANDBY — kuruma kurulumu yok", "Volatilite"))
    # Volatility Hole (ODI) + Adaptive Hole
    kb, kr = _ago(r["odi_bull"], 15), _ago(r["odi_bear"], 15)
    hu, hd = bool(r["hole_up"].iloc[-1]), bool(r["hole_dn"].iloc[-1])
    parts, s = [], 0.0
    if kb is not None and (kr is None or kb < kr):
        parts.append(f"🔼 ODI boğa ({kb} bar önce)"); s += 0.7
    elif kr is not None:
        parts.append(f"🔽 ODI ayı ({kr} bar önce)"); s -= 0.7
    if hu:
        parts.append("◯ adaptive hole yukarı patlama"); s += 0.5
    elif hd:
        parts.append("◯ adaptive hole aşağı patlama"); s -= 0.5
    elif bool(r["in_sqz"].iloc[-1]):
        parts.append("🔒 sıkışma (RS fiyatında)")
    out.append(_m("Volatility Hole", s, " · ".join(parts) if parts else "normal",
                  "Volatilite"))
    # Whale Momentum Profili
    p = r["profile"]
    if p:
        px = float(df["Close"].iloc[-1])
        if px > p["vah"]:
            t, s = f"fiyat değer alanının ÜSTÜNDE (VAH {p['vah']:.2f})", 0.6
        elif px < p["val"]:
            t, s = f"fiyat değer alanının ALTINDA (VAL {p['val']:.2f})", -0.6
        else:
            t = f"değer alanı içinde ({p['val']:.2f}–{p['vah']:.2f})"
            s = 0.2 if px >= p["poc"] else -0.2
        out.append(_m("Whale Momentum Profile",
                      s + (0.2 if p["bull_share"] > 0.6 else -0.2 if p["bull_share"] < 0.4 else 0),
                      f"W-POC {p['poc']:.2f} · {t} · alış payı %{p['bull_share'] * 100:.0f}",
                      "Akıllı para"))
    # Efor kırılım gücü
    sc = int(r["eff_score"][-1])
    up_k, dn_k = _ago(r["eff_up"].to_numpy(), 5), _ago(r["eff_dn"].to_numpy(), 5)
    note = f"güç {sc}/8"
    if up_k is not None:
        note = f"▲ efor yukarı kırıldı ({up_k} bar önce) · " + note
    elif dn_k is not None:
        note = f"▼ efor aşağı kırıldı ({dn_k} bar önce) · " + note
    out.append(_m("Efor kırılım gücü", (sc - 3) / 4 if up_k is not None else
                  -0.6 if dn_k is not None else (sc - 4) / 8, note, "Akıllı para"))
    return out
