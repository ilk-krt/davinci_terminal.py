# apex/pine_elliott.py — V722c "🌊 Elliott Dalga Motoru v2" Python karşılığı
#
# Pine kaynağı: V722c.pine bölüm 17 — varsayılan ayarlarla:
#   • ATR(14) × 1.2 eşikli kapanış-onaylı ZigZag
#   • son 6 pivotta 5 dalgalı itki: K1/K2/K3 kuralları + Fibonacci yakınlık skoru
#     + değişim kuralı → güven %; eşik 70; diyagonal (④∩①) güven × 0.45
#   • tamamlanan sayımdan sonra Ⓐ Ⓑ Ⓒ düzeltme fazı
#   • canlı faz: "2 bitti → ③ bekleniyor", "4 bitti → ⑤ bekleniyor", ABC — hedef
#     bölgesi, ⛔ teyit/stop ve ✖ geçersizlik seviyeleriyle
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from apex.pine_fib import atr_rma

WAVE_ATR = 1.2
MIN_CONF = 70
DIAG = True


def _tanh(x: float) -> float:
    e = math.exp(min(max(2.0 * x, -30.0), 30.0))
    return (e - 1.0) / (e + 1.0)


def _fib(x: float, lo: float, hi: float, ideal: float) -> float:
    if x < lo or x > hi:
        return 0.0
    return math.exp(-((x - ideal) / max((hi - lo) * 0.45, 1e-9)) ** 2)


def count(df: pd.DataFrame, last_frac: float | None = None) -> dict[str, Any]:
    h, l, c = (df[k].to_numpy(dtype=float) for k in ("High", "Low", "Close"))
    a = atr_rma(df, 14).to_numpy()
    n = len(df)
    open_last = bool(last_frac and 0 < last_frac < 1)
    B: list[int] = []
    P: list[float] = []
    T: list[int] = []
    dr, ep, ei = 1, c[0], 0
    last_end, abc_left, last_conf, last_dir, counts = -1, 0, 0, 0, 0
    last_count_bar = None
    for i in range(n):
        new, px, ty = False, np.nan, 0
        thr = a[i] * WAVE_ATR if np.isfinite(a[i]) else np.nan
        if dr == 1:
            if h[i] > ep:
                ep, ei = h[i], i
            if np.isfinite(thr) and c[i] < ep - thr:
                new, px, ty, dr, ep = True, ep, 1, -1, l[i]
        else:
            if l[i] < ep:
                ep, ei = l[i], i
            if np.isfinite(thr) and c[i] > ep + thr:
                new, px, ty, dr, ep = True, ep, -1, 1, h[i]
        if not new or (open_last and i == n - 1):     # açık bar onaylı değil
            continue
        B.append(ei); P.append(px); T.append(ty)
        if len(B) > 500:
            B.pop(0); P.pop(0); T.pop(0)
            last_end -= 1
        ei = i
        m = len(B)
        if abc_left > 0:
            abc_left -= 1
        elif m >= 6:
            i0 = m - 6
            q = P[i0:i0 + 6]
            sgn = 1 if T[i0] == -1 else -1
            w1, w3, w5 = (q[1] - q[0]) * sgn, (q[3] - q[2]) * sgn, (q[5] - q[4]) * sgn
            ok = w1 > 0 and w3 > 0 and w5 > 0
            r2 = (q[1] - q[2]) * sgn / w1 if ok else 9.0
            r4 = (q[3] - q[4]) * sgn / w3 if ok else 9.0
            ovl = (q[4] - q[1]) * sgn <= 0
            if r2 >= 1.0 or (ok and w3 < min(w1, w5)) or (ovl and not DIAG):
                ok = False
            sc = 0.0
            if ok:
                sc = _fib(r2, 0.20, 0.95, 0.585)
                sc += _fib(w3 / w1, 0.9, 4.5, 1.618) * 1.4
                sc += _fib(r4, 0.14, 0.62, 0.382)
                sc += _fib(w5 / w1, 0.4, 1.8, 0.618) * 0.8
                d2, d4 = B[i0 + 2] - B[i0 + 1], B[i0 + 4] - B[i0 + 3]
                if d2 > 0 and d4 > 0:
                    sc += 0.4 * _tanh(max(d2, d4) / min(d2, d4) - 1.0)
                if ovl:
                    sc *= 0.45
            conf = int(min(99.0, max(5.0, 100.0 * sc / 4.2))) if ok else 0
            if ok and conf >= MIN_CONF and i0 > last_end:
                last_end, last_conf, last_dir = i0 + 5, conf, sgn
                counts += 1
                abc_left = 3
                last_count_bar = B[i0 + 5]
    return {"B": B, "P": P, "T": T, "abc_left": abc_left, "last_conf": last_conf,
            "last_dir": last_dir, "counts": counts, "last_count_bar": last_count_bar,
            "h": h, "l": l, "c": c, "n": n}


def _zone(st, sp, h1, h2, stop, inv, name, close, hi, lo):
    up = h1 > sp
    zlo, zhi = min(h1, h2), max(h1, h2)
    near = zlo if up else zhi
    dead = inv is not None and (close < inv if up else close > inv)
    hit = hi >= zlo if up else lo <= zhi
    rr = None
    if stop is not None and abs(close - stop) > 0:
        rr = abs(near - close) / abs(close - stop)
    return {"name": name, "up": up, "zlo": zlo, "zhi": zhi, "near": near, "stop": stop,
            "inv": inv, "dead": dead, "hit": hit, "rr": rr}


def live(df: pd.DataFrame, last_frac: float | None = None) -> dict[str, Any]:
    r = count(df, last_frac)
    P, T, n = r["P"], r["T"], len(r["P"])
    close, hi, lo = r["c"][-1], r["h"][-1], r["l"][-1]
    out = {"phase": "yeterli pivot yok", "kind": None, "zone": None, "sgn": 0, **r}
    al = r["abc_left"]
    if al > 0:
        out["phase"] = f"ABC düzeltmesi ({4 - al}/3) · son sayım %{r['last_conf']}"
        out["kind"], out["sgn"] = f"abc{4 - al}", r["last_dir"]
        if al == 3 and n >= 6:
            z0, z5 = P[n - 6], P[n - 1]
            out["zone"] = _zone(None, z5, z5 - (z5 - z0) * 0.382, z5 - (z5 - z0) * 0.618,
                                None, z5, "Ⓐ düzeltme", close, hi, lo)
        elif al == 2 and n >= 2:
            aS, aE = P[n - 2], P[n - 1]
            out["zone"] = _zone(None, aE, aE + (aS - aE) * 0.5, aE + (aS - aE) * 0.618,
                                aE, aS, "Ⓑ tepki", close, hi, lo)
        elif al == 1 and n >= 3:
            aS2, aE2, bE = P[n - 3], P[n - 2], P[n - 1]
            out["zone"] = _zone(None, bE, bE + (aE2 - aS2) * 0.618, bE + (aE2 - aS2) * 1.0,
                                bE, None, "Ⓒ son ayak", close, hi, lo)
        return out
    if n >= 5:
        j0 = n - 5
        a0, a1, a2, a3, a4 = P[j0:j0 + 5]
        sg = 1 if T[j0] == -1 else -1
        v1, v3 = (a1 - a0) * sg, (a3 - a2) * sg
        if (v1 > 0 and v3 > 0 and (a1 - a2) * sg / max(v1, 1e-9) < 1.0
                and ((a4 - a1) * sg > 0 or DIAG)):
            out.update(phase=f"{'BOĞA' if sg > 0 else 'AYI'} · ④ bitti → ⑤. dalga bekleniyor",
                       kind="w5", sgn=sg)
            inv = None if (DIAG and (a4 - a1) * sg <= 0) else a1
            out["zone"] = _zone(None, a4, a4 + v1 * 0.618 * sg, a4 + v1 * 1.0 * sg, a4, inv,
                                "⑤ dalga", close, hi, lo)
            return out
    if n >= 3:
        b0, b1, b2 = P[n - 3:n]
        sg2 = 1 if T[n - 3] == -1 else -1
        u1 = (b1 - b0) * sg2
        if u1 > 0 and (b1 - b2) * sg2 / max(u1, 1e-9) < 1.0:
            out.update(phase=f"{'BOĞA' if sg2 > 0 else 'AYI'} · ② bitti → ③. dalga bekleniyor",
                       kind="w3", sgn=sg2)
            out["zone"] = _zone(None, b2, b2 + u1 * 1.618 * sg2, b2 + u1 * 2.618 * sg2, b2, b0,
                                "③ dalga", close, hi, lo)
            return out
    out["phase"] = "net bir dalga yapısı yok"
    return out


def _f(x: float) -> str:
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.2f}"


def modules(df: pd.DataFrame, tf: str, last_frac: float | None = None) -> list[dict[str, Any]]:
    if df is None or len(df) < 60:
        return []
    r = live(df[["High", "Low", "Close"]].astype(float), last_frac)
    close = float(df["Close"].iloc[-1])
    z, k, sg = r["zone"], r["kind"], r["sgn"]
    base = {"w3": 0.7, "w5": 0.4, "abc1": -0.5, "abc2": 0.1, "abc3": -0.35}.get(k, 0.0) * sg
    note = r["phase"]
    if z and min(z["zlo"], z["zhi"]) <= 0:
        z = None                                   # oran hedefi sıfırın altına düşüyor
        note += " · hedef bölgesi bu ölçekte anlamsız"
    if z:
        pct = (z["near"] / close - 1) * 100
        note += (f" · 🎯 {z['name']} {_f(z['zlo'])}–{_f(z['zhi'])} ({pct:+.1f}%)")
        if z["rr"] is not None:
            note += f" · R:R {z['rr']:.1f}"
        if z["stop"] is not None:
            note += f" · ⛔ {_f(z['stop'])}"
        if z["inv"] is not None:
            note += f" · ✖ {_f(z['inv'])}"
        if z["dead"]:
            note = "✖ SAYIM BOZULDU — " + note
            base = 0.0
        elif z["hit"]:
            note = "✓ hedef bölgesinde — " + note
            base *= 0.4
    if r["last_conf"]:
        note += f" · son 5 dalga güveni %{r['last_conf']}"
    return [{"Modül": "Elliott dalga", "Durum": float(np.clip(base, -1, 1)), "Not": note,
             "Grup": "Yapı"}]
