# apex/indicators.py — AETHER APEX
from __future__ import annotations


import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Hareketli ortalamalar
# --------------------------------------------------------------------------
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder yumuşatması — ta.rma. RSI ve ATR bunu kullanır."""
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def wma(s: pd.Series, n: int) -> pd.Series:
    """
    ta.wma — en yeni bara n ağırlığı verir.
    rolling().apply() yerine konvolüsyon: yüzlerce sembol taranırken
    tarama süresini kat kat kısaltıyor, sonuç birebir aynı.
    """
    w = np.arange(1, n + 1, dtype=float)
    w /= w.sum()
    x = s.to_numpy(dtype=float)
    if len(x) < n:
        return pd.Series(np.full(len(x), np.nan), index=s.index)
    conv = np.convolve(np.nan_to_num(x), w[::-1], mode="valid")
    out = np.full(len(x), np.nan)
    out[n - 1:] = conv
    # NaN içeren pencereler NaN kalmalı (nan_to_num kirletmesin)
    nan_win = pd.Series(np.isnan(x)).rolling(n).max().to_numpy()
    out[nan_win == 1] = np.nan
    return pd.Series(out, index=s.index)


def vwma(src: pd.Series, vol: pd.Series, n: int) -> pd.Series:
    return sma(src * vol, n) / sma(vol, n).replace(0, np.nan)


def stdev(s: pd.Series, n: int) -> pd.Series:
    """ta.stdev — popülasyon (ddof=0)."""
    return s.rolling(n).std(ddof=0)


# --------------------------------------------------------------------------
# Temel dönüşümler
# --------------------------------------------------------------------------
def highest(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).max()


def lowest(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).min()


def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))


def roc(s: pd.Series, n: int) -> pd.Series:
    prev = s.shift(n)
    return (s - prev) / prev.replace(0, np.nan) * 100.0


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    pc = close.shift(1)
    return pd.concat([high - low, (high - pc).abs(), (low - pc).abs()],
                     axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    return rma(true_range(high, low, close), n)


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    """
    ta.rsi. Kesintisiz yükselişte düşüş ortalaması 0 olur; bölme NaN vermemeli,
    Pine'da olduğu gibi 100 dönmelidir (tersi 0). Bu ayrım önemli: NaN dönerse
    bütün momentum zinciri (OMNI konsensüsü, konfluans) sessizce boşa düşer.
    """
    d = s.diff()
    up = rma(d.clip(lower=0), n)
    dn = rma((-d).clip(lower=0), n)
    out = 100.0 - 100.0 / (1.0 + up / dn.replace(0, np.nan))
    out = out.mask((dn == 0) & (up > 0), 100.0)
    out = out.mask((up == 0) & (dn > 0), 0.0)
    out = out.mask((up == 0) & (dn == 0), 50.0)
    return out


def stoch(src: pd.Series, high: pd.Series, low: pd.Series, n: int) -> pd.Series:
    ll, hh = lowest(low, n), highest(high, n)
    return 100.0 * (src - ll) / (hh - ll).replace(0, np.nan)


def mfi(src: pd.Series, vol: pd.Series, n: int = 14) -> pd.Series:
    d = src.diff()
    up = (vol * src.where(d > 0, 0.0)).rolling(n).sum()
    dn = (vol * src.where(d < 0, 0.0)).rolling(n).sum()
    return 100.0 - 100.0 / (1.0 + up / dn.replace(0, np.nan))


def cci(src: pd.Series, n: int = 20) -> pd.Series:
    m = sma(src, n)
    mad = _rolling_mad(src, n)
    return (src - m) / (0.015 * mad.replace(0, np.nan))


def _rolling_mad(s: pd.Series, n: int) -> pd.Series:
    """Ortalamadan mutlak sapmanın ortalaması — CCI için, vektörleştirilmiş."""
    x = s.to_numpy(dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return pd.Series(out, index=s.index)
    win = np.lib.stride_tricks.sliding_window_view(x, n)
    with np.errstate(invalid="ignore"):
        res = np.abs(win - win.mean(axis=1, keepdims=True)).mean(axis=1)
    out[n - 1:] = res
    return pd.Series(out, index=s.index)


def tsi(close: pd.Series, long_n: int = 25, short_n: int = 13) -> pd.Series:
    pc = close.diff()
    num = ema(ema(pc, long_n), short_n)
    den = ema(ema(pc.abs(), long_n), short_n)
    return 100.0 * num / den.clip(lower=0.001)


def percentrank(s: pd.Series, n: int) -> pd.Series:
    """
    ta.percentrank — önceki n değerin yüzde kaçı mevcut değerden küçük/eşit.
    Kayan pencere görünümüyle vektörleştirildi (rolling.apply yerine).
    """
    x = s.to_numpy(dtype=float)
    m = len(x)
    out = np.full(m, np.nan)
    if m < n + 1:
        return pd.Series(out, index=s.index)
    win = np.lib.stride_tricks.sliding_window_view(x, n + 1)
    prev, cur = win[:, :-1], win[:, -1:]
    with np.errstate(invalid="ignore"):
        cnt = (prev <= cur).sum(axis=1).astype(float)
        bad = np.isnan(win).any(axis=1)
    res = 100.0 * cnt / n
    res[bad] = np.nan
    out[n:] = res
    return pd.Series(out, index=s.index)


def percentile_lin(s: pd.Series, n: int, p: float) -> pd.Series:
    return s.rolling(n).quantile(p / 100.0, interpolation="linear")


def barssince(cond: pd.Series) -> pd.Series:
    idx = np.arange(len(cond), dtype=float)
    last = pd.Series(np.where(cond.to_numpy(), idx, np.nan),
                     index=cond.index).ffill()
    return pd.Series(idx, index=cond.index) - last


def leaky_reservoir(q: pd.Series, alpha: float,
                    negative_leak: float = 1.30) -> pd.Series:
    """
    APEX CORE'un sızdıran haznesi: ch = ch*(1-alpha) + q, negatif akış
    `negative_leak` katıyla hızlı boşalır (satışlar alımlardan hızlı drene eder).
    """
    out = np.empty(len(q), dtype=float)
    acc = 0.0
    vals = q.to_numpy(dtype=float)
    for i, v in enumerate(vals):
        if not np.isfinite(v):
            v = 0.0
        acc = acc * (1.0 - alpha) + (v if v >= 0 else v * negative_leak)
        out[i] = acc
    return pd.Series(out, index=q.index)


def ratcheting_atr_stop(low: pd.Series, atr14: pd.Series, mult: float,
                        entry_price: float | None = None,
                        hard_stop_pct: float = 20.0) -> pd.Series:
    """
    V719'un iz süren zırhı: stop = low - mult*ATR, SADECE yukarı kayar.
    Girişten `hard_stop_pct` kadar aşağıda sert bir taban vardır.
    """
    calc = (low - atr14 * mult).to_numpy(dtype=float)
    out = np.empty(len(calc), dtype=float)
    prev = np.nan
    floor = (entry_price * (1 - hard_stop_pct / 100.0)
             if entry_price else -np.inf)
    for i, c in enumerate(calc):
        if not np.isfinite(c):
            out[i] = prev
            continue
        prev = c if not np.isfinite(prev) else max(prev, c)
        out[i] = max(prev, floor)
    return pd.Series(out, index=low.index)


def normal_cdf(z: np.ndarray) -> np.ndarray:
    """Abramowitz-Stegun yaklaşımı — V719'un hibrit delta motoru için."""
    z = np.clip(z, -8.0, 8.0)
    t = 1.0 / (1.0 + 0.2316419 * np.abs(z))
    d = 0.3989422804014327 * np.exp(-z * z / 2.0)
    p = d * t * (0.319381530 + t * (-0.356563782 + t *
                 (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    return np.where(z >= 0, 1.0 - p, p)


def f_tanh(x):
    return np.tanh(np.clip(2.0 * np.asarray(x, dtype=float), -60.0, 60.0) / 2.0)


def f_contrast(x, gamma: float):
    x = np.asarray(x, dtype=float)
    return 50.0 * (1.0 + np.tanh((x - 50.0) / 50.0 * gamma) / np.tanh(gamma))


def safe_last(s, default=np.nan) -> float:
    """Serinin son geçerli değeri — kısa geçmişte patlamasın."""
    try:
        v = pd.Series(s).dropna()
        return float(v.iloc[-1]) if len(v) else float(default)
    except Exception:
        return float(default)


def safe_bool(s, default: bool = False) -> bool:
    try:
        v = pd.Series(s).dropna()
        return bool(v.iloc[-1]) if len(v) else default
    except Exception:
        return default
