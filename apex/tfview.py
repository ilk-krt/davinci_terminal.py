# apex/tfview.py — tek sembol × tek zaman dilimi için BÜTÜN Pine katmanları
#
# Grafik (tvchart) ve teknik okuma (reading) aynı hesaptan beslenir:
#   mods      → decision.evaluate_tf modül tablosu (puan + not)
#   sr, gaps, vsa_zones, liq, fib, ell, gfr → çizilecek seviyeler/kutular
#   sig       → işaretler (Q-dry, ODI, efor kırılımı, VSA dönüşleri, Wyckoff)
#   mg        → Mum Gücü sınıfı (mum rengi)
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from apex import decision as dcs
from apex import pine_elliott as pell
from apex import pine_fib as pfib
from apex import pine_structure as pst
from apex import pine_v710 as pv710
from apex import pine_vsa as pvsa
from apex.indicators import ema

_log = logging.getLogger(__name__)

TF_RULE = {"Günlük": None, "Haftalık": "W-FRI", "Aylık": "ME"}
VIEW_BARS = {"Günlük": 140, "Haftalık": 130, "Aylık": 120}
CALC_BARS = {"Günlük": 900, "Haftalık": 2000, "Aylık": 2000}


def _safe(name, fn, *a, default=None):
    try:
        return fn(*a)
    except Exception as exc:                     # noqa: BLE001
        _log.info("tfview %s: %s", name, exc)
        return default


def build(daily: pd.DataFrame, sym: str, tf: str, bench: pd.Series | None = None,
          ndx: pd.Series | None = None) -> dict[str, Any]:
    """Günlük OHLCV'den istenen zaman dilimini kurar ve bütün katmanları hesaplar."""
    rule = TF_RULE[tf]
    d = dcs.resample(daily, rule) if rule else daily.copy()
    d = d.dropna(subset=["Close"]).tail(CALC_BARS[tf])
    d["Volume"] = d["Volume"].fillna(0.0) if "Volume" in d else 0.0
    if len(d) < 40:
        return {"ok": False, "sym": sym, "tf": tf, "neden": f"yetersiz veri ({len(d)} mum)"}
    b = dcs.resample(bench.to_frame("Close").assign(Open=bench, High=bench, Low=bench,
                                                    Volume=0), rule)["Close"] \
        if (bench is not None and rule) else bench
    q = dcs.resample(ndx.to_frame("Close").assign(Open=ndx, High=ndx, Low=ndx,
                                                  Volume=0), rule)["Close"] \
        if (ndx is not None and rule) else ndx
    frac = dcs.partial_frac(daily, rule)
    lf = frac if frac and frac < 1 else None
    has_vol = float(d["Volume"].tail(60).sum()) > 0

    mods, _ = _safe("mods", dcs.evaluate_tf, d, b, sym, frac, tf, q,
                    default=(pd.DataFrame(), {}))
    atr_s = pfib.atr_rma(d, 14)
    v: dict[str, Any] = {"ok": True, "sym": sym, "tf": tf, "d": d, "mods": mods,
                         "atr": float(atr_s.iloc[-1]), "frac": frac, "has_vol": has_vol}
    c = d["Close"]
    v["ema"] = {n: ema(c, n) for n in (21, 50, 200)}
    from apex import ema_stack as ems
    v["ema_stack"] = _safe("ema_stack", ems.analyze, d, tf, default={"ok": False})
    v["sr"] = _safe("sr", pst.sr_matrix, d, atr_s, default={"res": [], "sup": []})
    v["gaps"] = _safe("gaps", pst.gap_matrix, d, atr_s, tf, default={"boxes": [], "born": []})
    v["vsa_zones"] = _safe("vsa", pst.vsa_zones, d, default={"zones": [], "events": []}) \
        if has_vol else {"zones": [], "events": []}
    v["liq"] = _safe("liq", pst.liquidity, d, atr_s, default={"up": None, "dn": None, "events": []})
    v["fib"] = _safe("fib", pfib.main_fib, d, default={"ok": False})
    v["ell"] = _safe("ell", pell.live, d[["High", "Low", "Close"]], lf, default=None)
    s1 = v["sr"]["sup"][0]["px"] if v["sr"].get("sup") else None
    v["gfr"] = _safe("gfr", pst.gfr, d, atr_s, b, s1, default=None) if has_vol else None
    v["v710"] = _safe("v710", pv710.compute, d, q if sym != "QQQ" else None, tf) \
        if has_vol else None
    v["vsa"] = _safe("vsaf", pvsa.compute, d, lf) if has_vol else None
    mg = _safe("mg", pfib.mum_gucu, d, b if sym not in ("SPY", "QQQ") else None,
               int(pfib.auto_params(tf)["lookN"]), frac or 1.0)
    v["mg"] = mg
    v["score"] = float(mods["Durum"].mean() * 100) if mods is not None and len(mods) else np.nan
    return v


def mod(v: dict[str, Any], name: str) -> dict[str, Any] | None:
    m = v.get("mods")
    if m is None or m.empty:
        return None
    r = m[m["Modül"] == name]
    return None if r.empty else r.iloc[0].to_dict()
