# apex/markets.py — "Endeksler & Risk" sayfasının varlık listesi ve değerlendirmesi
#
# Her varlık için günlük / haftalık / aylık skor (bütün Pine modülleri) ve
# teknik okuma metni üretilir. Oranlar (Bakır/Altın, Altın/Gümüş) iki serinin
# OHLC'sinin oranından kurulur (hacim yok → hacim modülleri atlanır).
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from apex import reading as rdg
from apex import tfview

_log = logging.getLogger(__name__)

MARKETS: dict[str, dict[str, Any]] = {
    "S&P 500": {"sym": "SPY", "grp": "Hisse endeksi", "vix": True},
    "Nasdaq": {"sym": "QQQ", "grp": "Hisse endeksi", "vix": True},
    "Russell 2000": {"sym": "IWM", "grp": "Hisse endeksi", "vix": True},
    "Dow Jones": {"sym": "DIA", "grp": "Hisse endeksi", "vix": True},
    "Kripto (BTC)": {"sym": "BTC-USD", "grp": "Kripto", "vix": False},
    "Altın": {"sym": "GC=F", "grp": "Değerli metal", "vix": False},
    "Gümüş": {"sym": "SI=F", "grp": "Değerli metal", "vix": False},
    "Bakır/Altın": {"sym": "HG=F/GC=F", "grp": "Oran", "vix": False},
    "Altın/Gümüş": {"sym": "GC=F/SI=F", "grp": "Oran", "vix": False},
    "Petrol (Brent)": {"sym": "BZ=F", "grp": "Emtia", "vix": False},
    "Dolar (DXY)": {"sym": "DX-Y.NYB", "grp": "Makro", "vix": False},
    "ABD 10Y faiz": {"sym": "^TNX", "grp": "Makro", "vix": False},
}
# oranlar ve makro göstergeler yatırım aracı değil — tema sayfası açılmaz
NO_THEMES = {"Altın/Gümüş", "Dolar (DXY)", "ABD 10Y faiz"}
ROW_NOTE = {
    "Bakır/Altın": "Yükseliyorsa ekonomi ısınıyor (risk iştahı, faiz yukarı); düşüyorsa "
                   "büyüme endişesi.",
    "Altın/Gümüş": "Yükseliyorsa gümüş zayıf (korku/defansif); düşüyorsa gümüş güçlü "
                   "(sanayi talebi, risk iştahı).",
    "Dolar (DXY)": "Güçlü dolar altın, gelişen piyasalar ve emtia için baskıdır.",
    "ABD 10Y faiz": "Yükselen faiz büyüme hisselerini ve altını baskılar.",
}


def underlying() -> list[str]:
    out = set()
    for m in MARKETS.values():
        out |= set(m["sym"].split("/"))
    return sorted(out | {"SPY", "QQQ"})


def series(sym: str, prices: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    if "/" not in sym:
        df = prices.get(sym)
        return None if df is None or df.empty else df
    a, b = sym.split("/")
    A, B = prices.get(a), prices.get(b)
    if A is None or B is None or A.empty or B.empty:
        return None
    j = A[["Open", "High", "Low", "Close"]].join(
        B[["Open", "High", "Low", "Close"]], how="inner", rsuffix="_b").dropna()
    out = pd.DataFrame(index=j.index)
    out["Open"] = j["Open"] / j["Open_b"]
    out["Close"] = j["Close"] / j["Close_b"]
    out["High"] = np.maximum(j["High"] / j["Low_b"].where(j["Low_b"] > 0), out[["Open", "Close"]].max(axis=1))
    out["Low"] = np.minimum(j["Low"] / j["High_b"].where(j["High_b"] > 0), out[["Open", "Close"]].min(axis=1))
    # gerçekçi fitil: oranın fitili iki serinin uç değerlerinden fazla şişmesin
    body_hi, body_lo = out[["Open", "Close"]].max(axis=1), out[["Open", "Close"]].min(axis=1)
    rng = (out["Close"].pct_change().abs().rolling(20).mean() * out["Close"]).bfill()
    out["High"] = np.minimum(out["High"], body_hi + rng)
    out["Low"] = np.maximum(out["Low"], body_lo - rng)
    out["Volume"] = 0.0
    return out


def views(sym: str, prices: dict[str, pd.DataFrame]) -> dict[str, dict]:
    df = series(sym, prices)
    spy = prices.get("SPY")
    qqq = prices.get("QQQ")
    b = spy["Close"] if spy is not None and sym != "SPY" else None
    q = qqq["Close"] if qqq is not None else None
    out = {}
    for tf in tfview.TF_RULE:
        if df is None or len(df) < 60:
            out[tf] = {"ok": False, "sym": sym, "tf": tf, "neden": "veri çekilemedi"}
        else:
            out[tf] = tfview.build(df, sym, tf, b, q)
    return out


def evaluate(prices: dict[str, pd.DataFrame]) -> dict[str, dict[str, Any]]:
    """Akşam hesabı: her varlık için skorlar + okuma metinleri (grafik çizmeden)."""
    res = {}
    for name, m in MARKETS.items():
        try:
            vs = views(m["sym"], prices)
            df = series(m["sym"], prices)
            px = float(df["Close"].iloc[-1]) if df is not None else np.nan
            ch = float(df["Close"].pct_change(5).iloc[-1] * 100) if df is not None else np.nan
            res[name] = {"sym": m["sym"], "grp": m["grp"], "px": px, "chg5": ch,
                         "scores": {tf: v.get("score", np.nan) for tf, v in vs.items()},
                         "text": {tf: rdg.read(v) for tf, v in vs.items()}}
        except Exception as exc:                   # noqa: BLE001
            _log.info("market %s: %s", name, exc)
            res[name] = {"sym": m["sym"], "grp": m["grp"], "px": np.nan, "chg5": np.nan,
                         "scores": {}, "text": {}}
    return res
