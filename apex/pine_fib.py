# apex/pine_fib.py — Apex v667.3 "Fib Yapısı × Mum Gücü" Python karşılığı
#
# Pine kaynağı: apex_v667_3_fib_mum_gucu.pine (v666 Mum Gücü motoru dahil).
# Bar bar aynı sırayla hesaplanır (repaint yok): her barda yüzdelikler, o bara
# kadar ONAYLANMIŞ önceki N barın dağılımından okunur; sonra bar dağılıma eklenir.
#
#   mum_gucu()   → her bar için mutlak sınıf (cls) ve sentez (syn) kodu
#   zigzag()     → ATR ZigZag pivotları (fib_study.zigzag ile aynı, causal)
#   fib_state()  → aktif itki / canlı düzeltme / son setup (+ ampirik olasılıklar)
#   main_fib()   → "Ana Yapı" Fib'i (QUANTUM 884 ile birebir) ve "şu an" oranı
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ ayarlar
MIN_SAMP = 15
RVOL_LEN = 12
BETA_LEN = 36
HI_VOL, LO_VOL, BIG_MV, SML_MV = 70.0, 40.0, 70.0, 40.0
REL_HI, REL_LO = 65.0, 35.0
WICK_TH = 0.40

# Ampirik tablolar (fib_study.py, S&P 500, aylık, 1963–2026)
BAND_NAMES = ["0–0.236", "0.236–0.382", "0.382–0.5", "0.5–0.618",
              "0.618–0.65 GP", "0.65–0.786", "0.786–1", ">1 yapı bozuk"]
UP_P1 = [0.94, 0.87, 0.82, 0.79, 0.71, 0.71, 0.67, 0.61]
UP_P127 = [0.67, 0.67, 0.65, 0.66, 0.56, 0.61, 0.57, 0.52]
UP_P161 = [0.49, 0.52, 0.53, 0.53, 0.43, 0.52, 0.46, 0.42]
UP_N = [104, 1159, 1326, 1083, 224, 645, 577, 535]
DN_P1 = [0.18, 0.51, 0.50, 0.39, 0.31, 0.32, 0.20, 0.08]
DN_P127 = [0.00, 0.04, 0.21, 0.18, 0.14, 0.14, 0.09, 0.03]
DN_P161 = [0.00, 0.01, 0.07, 0.09, 0.07, 0.06, 0.03, 0.01]
DN_N = [28, 89, 212, 217, 74, 231, 253, 865]
MG_P127 = [0.56, 0.63, 0.66, 0.74]
MG_STOP = [0.53, 0.51, 0.47, 0.41]

CLS_NAME = {1: "🟢 GÜÇLÜ YÜKSELİŞ", 2: "🟩 ORTA YÜKSELİŞ",
            3: "🟡 HACİMSİZ YÜKSELİŞ (zayıflık uyarısı)",
            -1: "🔴 SERT DÜŞÜŞ (tepki adayı)", -2: "🟥 ORTA DÜŞÜŞ",
            -3: "🟣 HACİMSİZ DÜŞÜŞ", 10: "🔹 HACİMLİ DURAKSAMA (nötr)",
            11: "🟠 DAĞITIM İZİ", 0: "⚪ NÖTR"}
SYN_NAME = {2: "🟢 GÜÇLÜ ALIM MUMU (kısa vade geri çekilme olası)",
            3: "⚠ HACİMLİ DURAKSAMA (sonrası zayıf)",
            4: "🪤 SİLKELEME (en güçlü alım işareti)",
            -2: "🩸 HİSSEYE ÖZEL SERT DÜŞÜŞ (tepki adayı)",
            -3: "🟠 DAĞITIM İZİ (etkisi belirsiz)",
            -4: "❌ FAKE YÜKSELİŞ (zayıflık)"}


def band_of(d: float) -> int:
    return (0 if d < 0.236 else 1 if d < 0.382 else 2 if d < 0.5 else
            3 if d < 0.618 else 4 if d < 0.65 else 5 if d < 0.786 else
            6 if d < 1.0 else 7)


def prob(sgn: int, b: int, which: int) -> float:
    t = {(1, 1): UP_P1, (1, 2): UP_P127, (1, 3): UP_P161,
         (-1, 1): DN_P1, (-1, 2): DN_P127, (-1, 3): DN_P161}[(sgn, which)]
    return t[b]


def mg_index(mg: int) -> int:
    return 0 if mg <= -1 else 1 if mg == 0 else 2 if mg == 1 else 3


def auto_params(tf: str) -> dict[str, float]:
    """Pine 'Otomatik' parametreleri: Aylık / Haftalık / Günlük ve altı."""
    if tf == "Aylık":
        return {"atrLen": 12, "rev": 2.0, "imp": 4.0, "lookN": 120}
    if tf == "Haftalık":
        return {"atrLen": 14, "rev": 2.5, "imp": 5.0, "lookN": 260}
    return {"atrLen": 14, "rev": 3.0, "imp": 6.0, "lookN": 500}


# ------------------------------------------------------------------ yardımcı
def atr_rma(df: pd.DataFrame, n: int) -> pd.Series:
    """Pine ta.atr: gerçek aralığın RMA'sı (ilk değer SMA)."""
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
                   axis=1).max(axis=1)
    tr.iloc[0] = h.iloc[0] - l.iloc[0]
    out = np.full(len(tr), np.nan)
    v = tr.to_numpy()
    if len(v) >= n:
        out[n - 1] = v[:n].mean()
        for i in range(n, len(v)):
            out[i] = (out[i - 1] * (n - 1) + v[i]) / n
    return pd.Series(out, index=df.index)


class _Dist:
    """Pine 'Dist' tipi: sıralı liste + ekleme sırası (pencereden düşürmek için)."""

    def __init__(self):
        self.srt: list[float] = []
        self.q: list[tuple[int, float]] = []

    def pct(self, x: float) -> float:
        n = len(self.srt)
        if n < MIN_SAMP or not np.isfinite(x):
            return np.nan
        lo = bisect.bisect_left(self.srt, x)
        hi = bisect.bisect_right(self.srt, x)
        return (lo + hi) / 2.0 / n * 100.0

    def add(self, x: float, b: int) -> None:
        if np.isfinite(x):
            bisect.insort(self.srt, x)
            self.q.append((b, x))

    def prune(self, cutoff: int) -> None:
        while self.q and self.q[0][0] <= cutoff:
            _, v = self.q.pop(0)
            del self.srt[bisect.bisect_left(self.srt, v)]


# ------------------------------------------------------------------ Mum Gücü
def mum_gucu(df: pd.DataFrame, bench: pd.Series | None, lookN: int,
             last_frac: float = 1.0) -> pd.DataFrame:
    """Her bar için cls (mutlak sınıf) ve syn (mutlak × göreli sentez)."""
    o, h, l, c = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    vol = df["Volume"].to_numpy(float)
    n = len(df)
    prev = np.r_[np.nan, c[:-1]]
    mv = (c - prev) / prev * 100
    dirn = np.where(np.isnan(mv), 0, np.sign(mv)).astype(int)
    absmv = np.abs(mv)
    rng = h - l
    has = rng > 0
    clv = np.where(has, (c - l) / np.where(has, rng, 1), 0.5)
    upw = np.where(has, (h - np.maximum(o, c)) / np.where(has, rng, 1), 0.0)
    low_ = np.where(has, (np.minimum(o, c) - l) / np.where(has, rng, 1), 0.0)
    clstr = np.where(dirn == 1, clv, np.where(dirn == -1, 1 - clv, 0.5))

    veff = vol.copy()
    if n:
        veff[-1] = vol[-1] / max(0.05, min(1.0, last_frac))
    vavg = pd.Series(vol).rolling(RVOL_LEN).mean().shift(1).to_numpy()
    rvol = np.where(vavg > 0, veff / np.where(vavg > 0, vavg, 1), np.nan)

    if bench is not None and len(bench):
        bc = bench.reindex(df.index).ffill().to_numpy(float)
        br = (bc / np.r_[np.nan, bc[:-1]] - 1) * 100
    else:
        br = np.zeros(n)
    s_mv, s_br = pd.Series(mv), pd.Series(br)
    sdS = s_mv.rolling(BETA_LEN).std(ddof=0)
    sdB = s_br.rolling(BETA_LEN).std(ddof=0)
    corr = s_mv.rolling(BETA_LEN).corr(s_br)
    beta_now = np.where(sdB > 0, corr * sdS / sdB.replace(0, np.nan), np.nan)
    beta = pd.Series(beta_now).shift(1).fillna(1.0).to_numpy()
    alpha = mv - beta * br

    dUp, dDn, dVol, dEx = _Dist(), _Dist(), _Dist(), _Dist()
    cls = np.zeros(n, dtype=int)
    syn = np.zeros(n, dtype=int)
    volP = np.full(n, np.nan)
    mvP = np.full(n, np.nan)
    exP = np.full(n, np.nan)
    for i in range(n):
        cut = i - lookN - 1
        for d in (dUp, dDn, dVol, dEx):
            d.prune(cut)
        volP[i] = dVol.pct(rvol[i])
        mvP[i] = (dUp.pct(absmv[i]) if dirn[i] == 1 else
                  dDn.pct(absmv[i]) if dirn[i] == -1 else np.nan)
        exP[i] = dEx.pct(alpha[i])
        confirmed = i < n - 1 or last_frac >= 1.0
        if confirmed:
            dVol.add(rvol[i], i)
            dEx.add(alpha[i], i)
            if dirn[i] == 1:
                dUp.add(absmv[i], i)
            elif dirn[i] == -1:
                dDn.add(absmv[i], i)
        if not (np.isfinite(volP[i]) and np.isfinite(mvP[i])):
            continue
        hv, lv = volP[i] >= HI_VOL, volP[i] <= LO_VOL
        bm, sm = mvP[i] >= BIG_MV, mvP[i] <= SML_MV
        if dirn[i] == 1:
            k = (11 if hv and upw[i] >= WICK_TH else
                 10 if hv and sm and low_[i] >= WICK_TH else
                 1 if hv and bm and clstr[i] >= 0.6 else
                 11 if hv and sm else 3 if bm and lv else
                 2 if hv or bm else 0)
        elif dirn[i] == -1:
            k = (10 if hv and low_[i] >= WICK_TH else
                 11 if hv and sm and upw[i] >= WICK_TH else
                 -1 if hv and bm and clstr[i] >= 0.6 else
                 10 if hv and sm else -3 if bm and lv else
                 -2 if hv or bm else 0)
        else:
            k = 0
        cls[i] = k
        if np.isfinite(exP[i]):
            rS, rW = exP[i] >= REL_HI, exP[i] <= REL_LO
            s = {1: (2 if rS else 0 if rW else 1),
                 2: (1 if rS else 0),
                 3: (-4 if rW else 1 if rS else 0),
                 -1: (-2 if rW else 0 if rS else -1),
                 -2: (-1 if rW else 0),
                 -3: (4 if rS else -1 if rW else 0),
                 10: (3 if rS else 0 if rW else 1),
                 11: (-3 if rW else 0 if rS else -1)}.get(k, 0)
            syn[i] = s
    return pd.DataFrame({"cls": cls, "syn": syn, "volP": volP, "mvP": mvP,
                         "exP": exP, "rvol": rvol, "alpha": alpha},
                        index=df.index)


def candle_name(cls: int, syn: int, ready: bool = True) -> str:
    if syn in SYN_NAME:
        return SYN_NAME[syn]
    if cls in CLS_NAME and (cls != 0 or ready):
        return CLS_NAME[cls]
    return "⚪ VERİ YETERSİZ"


# ------------------------------------------------------------------ ZigZag
@dataclass
class Pivot:
    bar: int
    price: float
    typ: int          # 1 tepe, -1 dip
    conf: int         # onay barı
    atr: float


@dataclass
class ZZState:
    pivots: list[Pivot] = field(default_factory=list)
    zd: int = 0
    zHi: float = np.nan
    zLo: float = np.nan
    zHiB: int = 0
    zLoB: int = 0
    newpiv: np.ndarray | None = None


def zigzag(df: pd.DataFrame, atr_len: int, rev: float) -> ZZState:
    """Pine bölüm 4 — her bar için newPiv bayrağı da tutulur."""
    h, l = df["High"].to_numpy(float), df["Low"].to_numpy(float)
    a = atr_rma(df, atr_len).to_numpy()
    st = ZZState(newpiv=np.zeros(len(df), dtype=bool))
    zHiA = zLoA = np.nan
    for i in range(len(df)):
        if not np.isfinite(a[i]):
            continue
        if not np.isfinite(st.zHi):
            st.zHi, st.zHiB, zHiA = h[i], i, a[i]
            st.zLo, st.zLoB, zLoA = l[i], i, a[i]
            continue
        thr = rev * a[i]
        if st.zd == 0:
            if h[i] > st.zHi:
                st.zHi, st.zHiB, zHiA = h[i], i, a[i]
            if l[i] < st.zLo:
                st.zLo, st.zLoB, zLoA = l[i], i, a[i]
            if st.zHi - st.zLo >= thr:
                if st.zHiB > st.zLoB:
                    st.pivots.append(Pivot(st.zLoB, st.zLo, -1, i, zLoA))
                    st.zd = 1
                else:
                    st.pivots.append(Pivot(st.zHiB, st.zHi, 1, i, zHiA))
                    st.zd = -1
                    seg = l[st.zHiB:i + 1]
                    k = int(np.argmin(seg[::-1]))       # en son en düşük
                    mi = i - k
                    st.zLo, st.zLoB, zLoA = l[mi], mi, a[mi]
                st.newpiv[i] = True
        elif st.zd == 1:
            if h[i] > st.zHi:
                st.zHi, st.zHiB, zHiA = h[i], i, a[i]
            elif st.zHi - l[i] >= thr:
                st.pivots.append(Pivot(st.zHiB, st.zHi, 1, i, zHiA))
                st.newpiv[i] = True
                st.zd = -1
                st.zLo, st.zLoB, zLoA = l[i], i, a[i]
        else:
            if l[i] < st.zLo:
                st.zLo, st.zLoB, zLoA = l[i], i, a[i]
            elif h[i] - st.zLo >= thr:
                st.pivots.append(Pivot(st.zLoB, st.zLo, -1, i, zLoA))
                st.newpiv[i] = True
                st.zd = 1
                st.zHi, st.zHiB, zHiA = h[i], i, a[i]
    return st


# ------------------------------------------------------------------ Fib durumu
def fib_state(df: pd.DataFrame, bench: pd.Series | None, tf: str,
              last_frac: float = 1.0, show_dn: bool = False) -> dict[str, Any]:
    """Pine bölüm 5–6'nın son bardaki durumu: canlı düzeltme + son setup."""
    p = auto_params(tf)
    mg = mum_gucu(df, bench, int(p["lookN"]), last_frac)
    cls, syn = mg["cls"].to_numpy(), mg["syn"].to_numpy()
    h, l, c = (df[k].to_numpy(float) for k in ("High", "Low", "Close"))

    # zigzag'ı bar bar yeniden oynat (pivot listesi her barda o ana kadarki)
    # (ATR pivot kaydında tutuluyor; ayrıca hesaplamaya gerek yok)
    zz = zigzag(df, int(p["atrLen"]), float(p["rev"]))
    piv_all = zz.pivots
    conf_of = [pv.conf for pv in piv_all]

    sS = 0
    sI1 = sP1 = sImp = np.nan
    tr: dict[str, Any] = {"sgn": 0, "done": True}
    for i in np.nonzero(zz.newpiv)[0]:
        pv = [x for x, cf in zip(piv_all, conf_of) if cf <= i]
        if len(pv) >= 2:
            t0, p0, p1 = pv[-2].typ, pv[-2].price, pv[-1].price
            sg = 1 if t0 == -1 else -1
            imp = sg * (p1 - p0)
            a1 = pv[-1].atr
            if imp > 0 and np.isfinite(a1) and imp >= p["imp"] * a1 and (sg == 1 or show_dn):
                sS, sI1, sP1, sImp = sg, pv[-1].bar, p1, imp
        if len(pv) >= 3:
            q0, q1, q2 = pv[-3], pv[-2], pv[-1]
            sg = 1 if q0.typ == -1 else -1
            imp = sg * (q1.price - q0.price)
            if imp > 0 and np.isfinite(q1.atr) and imp >= p["imp"] * q1.atr and (sg == 1 or show_dn):
                dep = sg * (q1.price - q2.price) / imp
                bnd = band_of(dep)
                absP = disP = strL = badL = 0
                for b in range(q1.bar + 1, i + 1):
                    leg = b > q2.bar
                    c_, s_ = cls[b], syn[b]
                    if sg == 1:
                        if not leg:
                            absP += c_ == 10 or s_ in (3, 4)
                            disP += s_ in (-2, -3)
                        else:
                            strL += s_ == 2 or c_ == 1 or (c_ == 2 and s_ == 1)
                            badL += c_ == 11 or s_ in (-3, -4)
                    else:
                        if not leg:
                            absP += c_ == 11 or s_ == -3
                            disP += s_ in (2, 3)
                        else:
                            strL += s_ == -2 or c_ == -1 or (c_ == -2 and s_ == -1)
                            badL += c_ == 10 or s_ in (3, 4)
                mgv = int(absP > 0) + int(strL > 0) - int(badL > 0) - int(disP > 0)
                broken = bnd == 7
                tr = {"sgn": 0 if broken else sg, "bar": int(i), "p0": q0.price,
                      "p1": q1.price, "imp": imp, "stop": q2.price,
                      "entry": c[i], "dep": dep, "band": bnd, "mg": mgv,
                      "pb": q2.bar - q1.bar, "broken": broken, "done": broken,
                      "h1": False, "h127": False, "h161": False,
                      "absP": absP, "disP": disP, "strL": strL, "badL": badL,
                      "p_1": prob(sg, bnd, 1), "p_127": prob(sg, bnd, 2),
                      "p_161": prob(sg, bnd, 3),
                      "tgt127": q0.price + sg * 1.272 * imp,
                      "tgt161": q0.price + sg * 1.618 * imp,
                      "weak": (not broken) and sg == 1 and mgv <= -1,
                      "aplus": (not broken) and sg == 1 and mgv >= 2}
        # takip (aynı mumda stop + hedef → stop)
    if tr.get("sgn") and not tr["done"]:
        sg = tr["sgn"]
        for b in range(tr["bar"] + 1, len(df)):
            if (sg == 1 and l[b] < tr["stop"]) or (sg == -1 and h[b] > tr["stop"]):
                tr["done"], tr["stopped"] = True, True
                break
            ext = sg * ((h[b] if sg == 1 else l[b]) - tr["p0"]) / tr["imp"]
            tr["h1"] |= ext >= 1.0
            tr["h127"] |= ext >= 1.272
            if ext >= 1.618:
                tr["h161"], tr["done"] = True, True
                break

    last_pb = piv_all[-1].bar if piv_all else None
    in_pull = (sS != 0 and last_pb is not None and last_pb == sI1
               and ((sS == 1 and zz.zd == -1) or (sS == -1 and zz.zd == 1)))
    live_ext = zz.zLo if sS == 1 else zz.zHi
    live_dep = sS * (sP1 - live_ext) / sImp if in_pull else np.nan
    live_band = band_of(live_dep) if np.isfinite(live_dep) else None
    return {"in_pull": in_pull, "pull_sgn": sS, "live_dep": live_dep,
            "live_band": live_band,
            "pull_bars": (len(df) - 1 - int(sI1)) if in_pull else None,
            "live_p1": prob(sS, live_band, 1) if live_band is not None else np.nan,
            "live_p127": prob(sS, live_band, 2) if live_band is not None else np.nan,
            "setup": tr, "candle_cls": int(cls[-1]) if len(cls) else 0,
            "candle_syn": int(syn[-1]) if len(syn) else 0,
            "candle_ready": bool(np.isfinite(mg["volP"].iloc[-1])
                                 and np.isfinite(mg["mvP"].iloc[-1])) if len(mg) else False,
            "mg_frame": mg, "pivots": piv_all, "params": p}


# ------------------------------------------------------------------ Ana Yapı Fib
def main_fib(df: pd.DataFrame, mult: float = 3.0, atr_len: int = 14,
             n_sw: int = 8, win: int = 400, min_mv: float = 10.0,
             max_pv: int = 60) -> dict[str, Any]:
    """QUANTUM 884 / v667.3 '📐 ANA YAPI FIB' — son bardaki bacak ve konum."""
    h, l, c = (df[k].to_numpy(float) for k in ("High", "Low", "Close"))
    dev = (atr_rma(df, atr_len) * mult).to_numpy()
    P: list[float] = []
    B: list[int] = []
    d, ext, exb = 1, h[0], 0
    for i in range(len(df)):
        if d > 0:
            if h[i] >= ext:
                ext, exb = h[i], i
            elif np.isfinite(dev[i]) and ext - l[i] > dev[i]:
                P.append(ext); B.append(exb)
                d, ext, exb = -1, l[i], i
        else:
            if l[i] <= ext:
                ext, exb = l[i], i
            elif np.isfinite(dev[i]) and h[i] - ext > dev[i]:
                P.append(ext); B.append(exb)
                d, ext, exb = 1, h[i], i
        if len(P) > max_pv:
            P.pop(0); B.pop(0)
    n = len(P)
    last = len(df) - 1
    if n < 2:
        return {"ok": False}

    def pct(a, b):
        lo, hi = min(a, b), max(a, b)
        return (hi - lo) / max(lo if b > a else hi, 1e-9) * 100

    mx = mn = ext
    mxb = mnb = exb
    for k in range(n - 1, max(0, n - n_sw) - 1, -1):
        if last - B[k] > win:
            break
        if P[k] > mx:
            mx, mxb = P[k], B[k]
        if P[k] < mn:
            mn, mnb = P[k], B[k]
    up = mxb > mnb
    A, Bv = (mn, mx) if up else (mx, mn)
    mv = pct(A, Bv)
    if mv < min_mv:
        fb = 0.0
        for q in range(n - 1):
            if last - B[q] > win:
                continue
            qp = pct(P[q], P[q + 1])
            if qp >= min_mv and qp > fb:
                fb, A, Bv = qp, P[q], P[q + 1]
        lp = pct(P[n - 1], ext)
        if lp >= min_mv and lp > fb and last - B[n - 1] <= win:
            A, Bv = P[n - 1], ext
        mv = pct(A, Bv)
    if mv < min_mv:
        return {"ok": False}
    is_up = Bv > A
    lo, hi = min(A, Bv), max(A, Bv)
    r = (hi - c[-1]) / (hi - lo) if is_up else (c[-1] - lo) / (hi - lo)
    base, diff = (lo, hi - lo) if is_up else (hi, lo - hi)
    levels = {lv: (base + diff * (1 - lv) if lv <= 1 else base + diff * lv)
              for lv in (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0,
                         1.272, 1.618, 2.618)}
    zone = ("bacak sonunun ötesinde (yeni uç)" if r < 0 else
            "0–0.236 (sığ)" if r < 0.236 else "0.236–0.382" if r < 0.382 else
            "0.382–0.5" if r < 0.5 else "0.5–0.618" if r < 0.618 else
            "0.618–0.786 (derin)" if r < 0.786 else
            "0.786–1 (çok derin)" if r <= 1 else "bacak başının ötesinde — yapı bozuldu")
    px = c[-1]
    above = sorted((v, k) for k, v in levels.items() if v > px)
    below = sorted(((v, k) for k, v in levels.items() if v < px), reverse=True)
    return {"ok": True, "up": is_up, "A": A, "B": Bv, "move_pct": mv, "r": r,
            "zone": zone, "levels": levels,
            "next_up": above[0] if above else None,
            "next_dn": below[0] if below else None}


# ------------------------------------------------------------------ karar modülleri
def _m(modul, durum, notu, grup):
    return {"Modül": modul, "Durum": float(np.clip(durum, -1, 1)), "Not": notu,
            "Grup": grup}


# Mum Gücü sınıfının karara etkisi — v667.1 veri bulgularıyla uyumlu:
# hacimsiz yükseliş ve fake yükseliş sonrası zayıf; silkeleme güçlü; güçlü
# absorpsiyon (hacimli duraksama) sonrası zayıf; sert düşüş sonrası tepki.
SYN_SCORE = {4: 1.0, 2: 0.4, 3: -0.5, -2: 0.2, -3: -0.3, -4: -1.0}
CLS_SCORE = {1: 0.6, 2: 0.3, 3: -0.6, -1: -0.2, -2: -0.4, -3: 0.1, 10: 0.0,
             11: -0.5, 0: 0.0}


def modules(df: pd.DataFrame, bench: pd.Series | None, tf: str,
            last_frac: float = 1.0) -> list[dict[str, Any]]:
    out = []
    fs = fib_state(df, bench, tf, last_frac)
    cl, sy = fs["candle_cls"], fs["candle_syn"]
    name = candle_name(cl, sy, fs["candle_ready"])
    sc = SYN_SCORE.get(sy, CLS_SCORE.get(cl, 0.0))
    mgf = fs["mg_frame"].tail(5)
    son5 = [candle_name(int(a), int(b)) .split(" ")[0] for a, b in
            zip(mgf["cls"], mgf["syn"])]
    out.append(_m("Mum Gücü (v666)", sc, f"{name} · son 5: {' '.join(son5)}",
                  "Fiyat"))

    mf = main_fib(df)
    parts, s = [], 0.0
    if mf.get("ok"):
        yon = "yükseliş" if mf["up"] else "düşüş"
        parts.append(f"ana yapı {yon} %{mf['move_pct']:.0f}, şu an "
                     f"{mf['r']:.2f} geri çekilme ({mf['zone']})")
        if mf["next_up"]:
            parts.append(f"üst seviye {mf['next_up'][1]:g} → {mf['next_up'][0]:.2f}")
        if mf["next_dn"]:
            parts.append(f"alt seviye {mf['next_dn'][1]:g} → {mf['next_dn'][0]:.2f}")
        if mf["up"]:
            s += (0.5 if mf["r"] < 0 else 0.3 if mf["r"] < 0.5 else
                  0.0 if mf["r"] < 0.786 else -0.6)
        else:
            s += -0.5 if mf["r"] < 0.5 else -0.2 if mf["r"] < 1 else 0.4
    if fs["in_pull"]:
        yon = "yükseliş" if fs["pull_sgn"] == 1 else "düşüş"
        parts.append(f"{yon} itkisi düzeltiyor: %{fs['live_dep'] * 100:.0f} geri "
                     f"verildi, {fs['pull_bars']} mumdur · burada biterse 100'de "
                     f"{fs['live_p127'] * 100:.0f} 1.272'ye ulaştı")
    tr = fs["setup"]
    if tr.get("sgn") and not tr.get("done"):
        mgt = {2: "güçlü", 1: "pozitif", 0: "nötr"}.get(min(tr["mg"], 2), "zayıf")
        parts.append(f"{'▲' if tr['sgn'] == 1 else '▼'} setup aktif (%{tr['dep'] * 100:.0f} "
                     f"geri çekilme, MG {mgt}) · stop {tr['stop']:.2f} · 1.272 → "
                     f"{tr['tgt127']:.2f} (100'de {tr['p_127'] * 100:.0f})"
                     + (" · ALMA (MG zayıf)" if tr["weak"] else ""))
        s += -0.5 if tr["weak"] else (0.5 if tr["mg"] >= 1 else 0.2)
    elif tr.get("broken"):
        parts.append("son yapı bozuldu (geri çekilme itkinin başını geçti)")
        s -= 0.3
    out.append(_m("Fibonacci seviyeleri", s, " · ".join(parts) if parts else
                  "yeterli itki yok", "Yapı"))
    return out
