"""
SİNYAL DOĞRULAMA — uygulamanın sayfalarındaki okumaların GEÇMİŞTE ne kadar isabetli
olduğunu ölçer (walk-forward: her tarihte yalnızca o güne kadarki veri kullanılır, sonra
ileriki 5 / 20 / 60 işlem gününde ne olduğuna bakılır).

  A) 🌊 Para Akışı  : Akış Skoru yüksek varlıklar sonraki günlerde diğerlerini geçti mi?
                      (yeni yöntem: oynaklığa göre düzeltilmiş · eski yöntem: ham %)
  B) 🎯 Tema Avcısı : RRG bölgesi (Lider / İyileşiyor / Yoruluyor / Geride) ve Erken Skor
                      sonraki günlerde temanın SPY'ye göre getirisini öngördü mü?
  C) 🧭 Endeksler   : günlük ve haftalık skorlar ileriki getiriyi öngördü mü?
  E) 🐋 Balina & OMNI: APEX CORE v4 (Golden Star, PRO↗RETAIL, Stealth, ACC/DIST, seesaw…) ve
                      APEX V670 (Diamond, Afterburner, erken uyarı, süpürme, boğa/ayı skoru)
                      olaylarından sonra 5/10/20 günde SPY'ye göre getiri — ~450 hisse, ~5 yıl
  S) 📜 Geçmiş öneriler (bolum=secki): Tema Avcısı ve Alım seçkisi her hafta ne önerirdi,
                      gerçekte ne oldu — ayrıntı tools/secki_gecmis.py
  D) 🦅 Hisse Avcısı: Sinyal, swing skoru, sınıf (🎯/🚀/⛔…), işlem planı kararı (✅ AL /
                      🟡 LİMİT / ⛔ UZAK DUR) ve plan seviyeleri: hedef mi önce geldi, stop mu?

GitHub Actions → "Sinyal doğrulama" → Run workflow. Ya da:
    python tools/validate_signals.py --ay 12 --bolum akis,tema,endeks,hisse

Çıktı: data/validation/ (akis.csv, tema.csv, endeks.csv, hisse.csv, ozet.json,
sinyal_dogrulama.zip). Bilanço takvimi ve short verisi geçmişe dönük olmadığı için hisse
sınıflandırması bunlar olmadan yapılır.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
import warnings
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np                          # noqa: E402
import pandas as pd                         # noqa: E402

from apex import compute as cmp             # noqa: E402
from apex import data as dta                # noqa: E402
from apex import decision as dcs            # noqa: E402
from apex import engine as eng              # noqa: E402
from apex import flows as flw               # noqa: E402
from apex import funnel as fnl              # noqa: E402
from apex import markets as mkt             # noqa: E402
from apex import pine_fib as pfib           # noqa: E402
from apex import plan as pln                # noqa: E402
from apex import screener as scr            # noqa: E402
from apex import universe as uni            # noqa: E402

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
OUT = ROOT / "data" / "validation"
FWD = (5, 20, 60)


# ------------------------------------------------------------------ yardımcılar
def _naive(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    ix = pd.DatetimeIndex(d.index)
    d.index = (ix.tz_localize(None) if ix.tz is not None else ix).normalize()
    return d[~d.index.duplicated(keep="last")].sort_index()


def _fetch(tickers, kind="1d_long") -> dict[str, pd.DataFrame]:
    tickers = sorted(set(tickers))
    out: dict[str, pd.DataFrame] = {}
    for k in range(0, len(tickers), 40):
        part = tickers[k:k + 40]
        p, f = dta.fetch(part, kind)
        out.update({t: _naive(v).tail(1400) for t, v in p.items() if v is not None and len(v)})
        print(f"  veri {k + len(part)}/{len(tickers)} · eksik: {f[:8]}{'…' if len(f) > 8 else ''}",
              flush=True)
        time.sleep(1.0)
    return out


def _upto(prices: dict[str, pd.DataFrame], t: pd.Timestamp) -> dict[str, pd.DataFrame]:
    return {k: v[v.index <= t] for k, v in prices.items()}


def _fwd(c: pd.Series, t: pd.Timestamp, n: int) -> float:
    """t gününden sonraki n işlem günü getirisi (%)."""
    c = c.dropna()
    i = c.index.searchsorted(t, side="right") - 1
    if i < 0 or i + n >= len(c):
        return np.nan
    return float((c.iloc[i + n] / c.iloc[i] - 1) * 100)


def _dates(cal: pd.DatetimeIndex, months: int, step: int, max_fwd: int) -> list[pd.Timestamp]:
    start = cal[-1] - pd.DateOffset(months=months)
    ok = cal[(cal >= start)]
    ok = ok[: max(0, len(ok) - 5)]           # en az 5 günlük ileri veri kalsın
    return list(ok[::step])


def _spearman(a: pd.Series, b: pd.Series) -> float:
    m = a.notna() & b.notna()
    if m.sum() < 4:
        return np.nan
    return float(a[m].rank().corr(b[m].rank()))


def _ic_summary(df: pd.DataFrame, score: str, target: str, by: str = "Tarih") -> dict:
    ics = df.groupby(by).apply(lambda g: _spearman(g[score], g[target])).dropna()
    if len(ics) < 3:
        return {"IC ort": np.nan, "IC t": np.nan, "n tarih": int(len(ics))}
    return {"IC ort": round(float(ics.mean()), 3),
            "IC t": round(float(ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)))), 2),
            "IC>0 %": round(float((ics > 0).mean() * 100), 1), "n tarih": int(len(ics))}


# ------------------------------------------------------------------ A) para akışı
def _old_flow(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Önceki sürümün yöntemi (ham % z-skoru, takvim hizasız) — karşılaştırma için."""
    rows = []
    for ad, (t, _) in flw.ASSETS.items():
        df = prices.get(t)
        if df is None or len(df) < 70:
            continue
        c = df["Close"].dropna()
        dv = (df["Close"] * df["Volume"]).fillna(0)
        v10, v60 = dv.tail(10).mean(), dv.tail(60).mean()
        rows.append({"Sembol": t, "r20": flw._ret(c, 20), "r5": flw._ret(c, 5),
                     "vt": v10 / v60 if v60 > 0 else np.nan})
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    z = (0.6 * (d["r20"] - d["r20"].mean()) / (d["r20"].std(ddof=0) or 1)
         + 0.4 * (d["r5"] - d["r5"].mean()) / (d["r5"].std(ddof=0) or 1))
    d["Akış Skoru"] = z * d["vt"].clip(0.6, 1.6).fillna(1.0)
    return d


def part_flows(months: int) -> pd.DataFrame:
    print("A) Para akışı", flush=True)
    P = _fetch(flw.TICKERS)
    cal = P["SPY"].index
    rows = []
    # ileri getiriler SPY takvimine hizalı (kripto dahil)
    al = {t: flw._align(df, cal)["Close"] for t, df in P.items()}
    for t in _dates(cal, months, 5, 20):
        sub = _upto(P, t)
        for hz in flw.HORIZONS:
            T = flw.table(sub, hz)
            for _, r in T.iterrows():
                rows.append({"Tarih": t.date(), "Yöntem": "yeni", "Dönem": hz, "Sembol": r["Sembol"],
                             "Akış Skoru": r["Akış Skoru"], "Akış": r["Akış"]})
        O = _old_flow(sub)
        for _, r in O.iterrows():
            rows.append({"Tarih": t.date(), "Yöntem": "eski", "Dönem": "1 ay", "Sembol": r["Sembol"],
                         "Akış Skoru": r["Akış Skoru"], "Akış": ""})
    D = pd.DataFrame(rows)
    for n in FWD:
        D[f"İleri {n}G %"] = [_fwd(al[s], pd.Timestamp(t), n) if s in al else np.nan
                             for s, t in zip(D["Sembol"], D["Tarih"])]
        # aynı tarih/yöntem/dönemde varlıkların ortalamasına göre (göreli)
        D[f"Göreli {n}G %"] = D[f"İleri {n}G %"] - D.groupby(["Tarih", "Yöntem", "Dönem"])[
            f"İleri {n}G %"].transform("mean")
    return D


# ------------------------------------------------------------------ B) temalar
def part_themes(months: int) -> pd.DataFrame:
    print("B) Tema rotasyonu", flush=True)
    etfs = sorted({t for lst in uni.THEME_TRACKER.values() for t in lst})
    P = _fetch(etfs + ["SPY"])
    cal = P["SPY"].index
    full_idx = {}
    for tema, mem in uni.THEME_TRACKER.items():
        ix, _ = fnl.theme_index(P, mem)
        if ix is not None:
            full_idx[tema] = ix
    spy = P["SPY"]["Close"]
    rows = []
    for t in _dates(cal, months, 5, 20):
        sub = {k: v[v.index <= t].tail(420) for k, v in P.items()}
        T, _ = fnl.theme_rotation(sub, uni.THEME_TRACKER)
        for _, r in T.iterrows():
            rec = {"Tarih": t.date(), "Tema": r["Tema"], "Bölge": r["Bölge"], "_q": r["_q"],
                   "Erken Skor": r["Erken Skor"], "Taze Lider": bool(r["Taze Lider"]),
                   "Hacim Onayı": bool(r["Hacim Onayı"]), "RS-Oran": r["RS-Oran"],
                   "RS-Mom": r["RS-Mom"]}
            ix = full_idx.get(r["Tema"])
            for n in FWD:
                a, b = (_fwd(ix, t, n) if ix is not None else np.nan), _fwd(spy, t, n)
                rec[f"İleri {n}G %"] = a
                rec[f"SPY'ye göre {n}G %"] = a - b if np.isfinite(a) and np.isfinite(b) else np.nan
            rows.append(rec)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ C) endeksler
def part_markets(months: int) -> pd.DataFrame:
    print("C) Endeksler & Risk", flush=True)
    P = _fetch(mkt.underlying())
    cal = P["SPY"].index
    spy, qqq = P["SPY"], P.get("QQQ")
    rows = []
    for t in _dates(cal, months, 5, 20):
        sub = _upto(P, t)
        for name, m in mkt.MARKETS.items():
            df = mkt.series(m["sym"], sub)
            full = mkt.series(m["sym"], P)
            if df is None or len(df) < 120 or full is None:
                continue
            rec = {"Tarih": t.date(), "Varlık": name, "Grup": m["grp"]}
            for tf, rule in (("Günlük", None), ("Haftalık", "W-FRI")):
                try:
                    d_ = (dcs.resample(df, rule) if rule else df).tail(900)
                    b_ = sub["SPY"] if m["sym"] != "SPY" else None
                    b_ = (dcs.resample(b_, rule) if (rule and b_ is not None) else b_)
                    q_ = sub.get("QQQ")
                    q_ = dcs.resample(q_, rule) if (rule and q_ is not None) else q_
                    frac = dcs.partial_frac(df, rule)
                    mods, _ = dcs.evaluate_tf(d_, b_["Close"] if b_ is not None else None,
                                              m["sym"], frac, tf,
                                              q_["Close"] if q_ is not None else None)
                    rec[f"{tf} skor"] = (float(mods["Durum"].mean() * 100)
                                         if mods is not None and len(mods) else np.nan)
                except Exception as exc:         # noqa: BLE001
                    rec[f"{tf} skor"] = np.nan
                    print(f"   {name} {tf} {t.date()}: {exc}", flush=True)
            for n in FWD:
                rec[f"İleri {n}G %"] = _fwd(full["Close"], t, n)
            rows.append(rec)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ D) hisseler
def _plan_outcome(df: pd.DataFrame, t: pd.Timestamp, p: dict, karar: str, days: int = 20) -> dict:
    """Plan seviyeleriyle ileriye bakış: giriş oldu mu, sonra hedef mi stop mu önce geldi?"""
    out = {"Girişe geldi": np.nan, "Sonuç": "", "Sonuç R": np.nan}
    if not p.get("ok"):
        return out
    fut = df[df.index > t].head(days)
    if len(fut) < 3:
        return out
    entry, stop, t1 = p["Giriş Üst"], p["Stop"], p["Hedef 1"]
    if karar.startswith("✅"):
        entry = p["Fiyat"]
        i0 = 0
    else:
        hit = np.flatnonzero(fut["Low"].to_numpy() <= entry)
        if not len(hit):
            out["Girişe geldi"] = False
            out["Sonuç"] = "giriş olmadı"
            return out
        i0 = int(hit[0])
    out["Girişe geldi"] = True
    risk = entry - stop
    if risk <= 0:
        return out
    for j in range(i0, len(fut)):
        lo, hi = fut["Low"].iloc[j], fut["High"].iloc[j]
        if lo <= stop:                           # aynı gün ikisi de → temkinli: stop
            out["Sonuç"], out["Sonuç R"] = "stop", -1.0
            return out
        if hi >= t1:
            out["Sonuç"], out["Sonuç R"] = "hedef 1", (t1 - entry) / risk
            return out
    last = fut["Close"].iloc[-1]
    out["Sonuç"], out["Sonuç R"] = "süre doldu", (last - entry) / risk
    return out


def part_stocks(months: int, step: int, limit: int = 0) -> pd.DataFrame:
    print("D) Hisse avcısı", flush=True)
    th_map: dict[str, str] = {}
    for tema in uni.THEME_TRACKER:
        for s in cmp.theme_holdings(tema):
            th_map.setdefault(s, tema)
    syms = sorted(th_map)
    if limit:
        syms = syms[:limit]
    etfs = sorted({t for lst in uni.THEME_TRACKER.values() for t in lst})
    P = _fetch(syms + etfs + ["SPY"])
    cal = P["SPY"].index
    spy = P["SPY"]
    th_idx = {}
    for tema, mem in uni.THEME_TRACKER.items():
        ix, _ = fnl.theme_index(P, mem)
        if ix is not None:
            th_idx[tema] = ix
    rows = []
    dates = _dates(cal, months, step, 20)
    t0 = time.time()
    for k, t in enumerate(dates, 1):
        bench = spy[spy.index <= t]["Close"]
        for s in syms:
            df = P.get(s)
            if df is None:
                continue
            d = df[df.index <= t].tail(420)
            if len(d) < 130:
                continue
            try:
                wk = d["Close"].resample("W-FRI").last().dropna()
                wb = bool(wk.iloc[-1] > wk.ewm(span=12, adjust=False).mean().iloc[-1]) \
                    if len(wk) > 12 else None
                r = eng.analyze(d, s, bench_close=bench, weekly_bull=wb)
                if not r.ok:
                    continue
                dd = dict(r.data)
                dd["Sembol"] = s
                dd["Skor"] = scr.swing_score(dd)
                tc = th_idx.get(th_map[s])
                tc = tc[tc.index <= t] if tc is not None else None
                th1a = fnl._pct(tc, 21) if tc is not None else np.nan
                th1h = fnl._pct(tc, 5) if tc is not None else np.nan
                dd["Temaya Göre 1A"] = dd.get("1 Ay %", np.nan) - th1a
                dd["Temaya Göre 1H"] = dd.get("1 Hafta %", np.nan) - th1h
                cls, _ = fnl.classify_stock(dd)
                p = pln.stock_plan(d)
                karar, _ = pln.verdict(p, cls, dd.get("Sinyal", ""), None, True)
                mf = pfib.main_fib(d)
                rec = {"Tarih": t.date(), "Sembol": s, "Tema": th_map[s],
                       "Sinyal": dd.get("Sinyal", ""), "Skor": dd["Skor"],
                       "Huni Skoru": fnl.funnel_score(dd), "Sınıf": cls, "Karar": karar,
                       "Trend": p.get("Trend", ""), "Bölgede": p.get("Bölgede", np.nan),
                       "R:R": p.get("R:R", np.nan), "Fib yukarı": bool(mf.get("up")) if mf.get("ok") else np.nan,
                       "Fib r": float(mf["r"]) if mf.get("ok") else np.nan,
                       "Hacim ($M)": dd.get("Hacim ($M)", np.nan)}
                for n in (5, 20):
                    a, b = _fwd(df["Close"], t, n), _fwd(spy["Close"], t, n)
                    rec[f"İleri {n}G %"] = a
                    rec[f"SPY'ye göre {n}G %"] = a - b if np.isfinite(a) and np.isfinite(b) else np.nan
                rec.update(_plan_outcome(df, t, p, karar))
                rows.append(rec)
            except Exception as exc:             # noqa: BLE001
                print(f"   {s} {t.date()}: {type(exc).__name__}: {exc}", flush=True)
        el = time.time() - t0
        print(f"  tarih {k}/{len(dates)} ({t.date()}) · {len(rows)} satır · "
              f"{el / 60:.1f} dk · kalan ~{el / k * (len(dates) - k) / 60:.0f} dk", flush=True)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ E) balina & OMNI olayları
EVENTS = {   # (kaynak, sütun): etiket — APEX CORE v4 ve APEX V670 OMNI
    ("core", "star"): "⭐ Golden Star", ("core", "cross_up"): "PRO ↗ RETAIL",
    ("core", "cross_dn"): "PRO ↘ RETAIL", ("core", "st_in"): "🕵️ Stealth IN",
    ("core", "st_out"): "🕵️ Stealth OUT", ("core", "real_acc"): "🐋 Gerçek toplama (ACC)",
    ("core", "real_dist"): "🐋 Gerçek dağıtım (DIST)", ("core", "sweep_bar"): "🎣 Whale süpürme",
    ("core", "db_bottom"): "🔵 Dip çıkışı (seesaw)", ("core", "rd_top"): "🔴 Tepe dönüşü (seesaw)",
    ("core", "exhausted"): "⚠️ Whale tükenme", ("core", "sqz_fire"): "💥 Sıkışma patladı",
    ("omni", "dia_buy"): "💎 Diamond AL", ("omni", "dia_sell"): "💎 Diamond SAT",
    ("omni", "ab_bull"): "🚀 Afterburner boğa", ("omni", "ab_bear"): "🚀 Afterburner ayı",
    ("omni", "ew_bull"): "🔵 Erken uyarı AL", ("omni", "ew_bear"): "🔴 Erken uyarı SAT",
    ("omni", "sweep_low"): "🎣 Dip süpürmesi", ("omni", "sweep_high"): "🎣 Tepe süpürmesi",
    ("omni", "bull5"): "🟢 Boğa skoru 5+/6", ("omni", "bear5"): "🔴 Ayı skoru 5+/6",
}


def part_whale(limit: int = 0) -> pd.DataFrame:
    """Her hisse için TÜM geçmiş (≈5 yıl) bir kez hesaplanır — göstergeler nedensel olduğu
    için her bar yalnızca kendisine kadarki veriyi görür. Olay günlerinden sonraki 5/10/20
    günün SPY'ye göre getirisi kaydedilir; ayrıca her 5 günde bir sürekli değerler
    (WHALE, ΔWHALE, PRO−RETAIL, OMNI) örneklenir."""
    print("E) Balina & OMNI olayları", flush=True)
    syms = sorted({s for tema in uni.THEME_TRACKER for s in cmp.theme_holdings(tema)})
    if limit:
        syms = syms[:limit]
    P = _fetch(syms + ["SPY"])
    spy = P["SPY"]["Close"]
    rows = []
    t0 = time.time()
    for k, s in enumerate(syms, 1):
        df = P.get(s)
        if df is None or len(df) < 400:
            continue
        try:
            d = df[["Open", "High", "Low", "Close", "Volume"]].astype(float).dropna(subset=["Close"])
            d["Volume"] = d["Volume"].fillna(0.0)
            core = eng.apex_core(d)
            omni = eng.apex_omni(d)
            omni["bull5"] = (omni["hud"] >= 5) & (omni["hud"].shift(1) < 5)
            omni["bear5"] = (omni["hud_bear"] >= 5) & (omni["hud_bear"].shift(1) < 5)
            c = d["Close"]
            sp = spy.reindex(d.index).ffill()
            fw = {}
            for n in (5, 10, 20):
                fw[n] = ((c.shift(-n) / c - 1) - (sp.shift(-n) / sp - 1)) * 100
            up = (c > c.ewm(span=200, adjust=False).mean())
            valid = np.arange(len(d)) >= 260            # gösterge ısınması
            src = {"core": core, "omni": omni}
            for (sk, col), lab in EVENTS.items():
                ev = src[sk][col].fillna(False).astype(bool).to_numpy() & valid
                for i in np.flatnonzero(ev):
                    rows.append({"Tür": "olay", "Sinyal": lab, "Sembol": s, "Tarih": d.index[i].date(),
                                 "Trend": "EMA200 üstü" if up.iloc[i] else "EMA200 altı",
                                 **{f"SPY'ye göre {n}G %": fw[n].iloc[i] for n in (5, 10, 20)}})
            # sürekli değerler (5 günde bir)
            dw5 = core["whale"] - core["whale"].shift(5)
            for i in range(260, len(d), 10):
                rows.append({"Tür": "seviye", "Sinyal": "", "Sembol": s, "Tarih": d.index[i].date(),
                             "Trend": "EMA200 üstü" if up.iloc[i] else "EMA200 altı",
                             "WHALE": core["whale"].iloc[i], "ΔWHALE 5G": dw5.iloc[i],
                             "PRO−RETAIL": core["pro"].iloc[i] - core["ret_line"].iloc[i],
                             "OMNI": omni["mom"].iloc[i], "HUD net": omni["hud_net"].iloc[i],
                             **{f"SPY'ye göre {n}G %": fw[n].iloc[i] for n in (5, 10, 20)}})
        except Exception as exc:                 # noqa: BLE001
            print(f"   {s}: {type(exc).__name__}: {exc}", flush=True)
        if k % 25 == 0:
            print(f"  {k}/{len(syms)} hisse · {len(rows)} satır · {(time.time() - t0) / 60:.1f} dk",
                  flush=True)
    return pd.DataFrame(rows)


def _whale_summary(E: pd.DataFrame) -> dict:
    out: dict = {}
    lv = E[E["Tür"] == "seviye"]
    base = {n: float(lv[f"SPY'ye göre {n}G %"].mean()) for n in (5, 10, 20)}
    out["taban (tüm günler) SPY'ye göre %"] = {f"{n}G": round(v, 3) for n, v in base.items()}
    ev = E[E["Tür"] == "olay"]
    tab = {}
    for lab, g in ev.groupby("Sinyal"):
        rec = {"n": int(len(g)), "hisse": int(g["Sembol"].nunique())}
        for n in (5, 10, 20):
            x = g[f"SPY'ye göre {n}G %"].dropna()
            if len(x) < 10:
                continue
            m = float(x.mean())
            # örtüşme payı: aynı hissede yakın olaylar bağımsız değil → kabaca n/2 ile t
            t = (m - base[n]) / (x.std(ddof=1) / np.sqrt(max(len(x) / 2, 1)))
            rec[f"{n}G ort %"] = round(m, 2)
            rec[f"{n}G fark (tabana göre)"] = round(m - base[n], 2)
            rec[f"{n}G t"] = round(t, 2)
            rec[f"{n}G pozitif %"] = round(float((x > 0).mean() * 100), 1)
        g20 = g.dropna(subset=["SPY'ye göre 20G %"])
        if len(g20):
            yr = g20.groupby(pd.to_datetime(g20["Tarih"]).dt.year)["SPY'ye göre 20G %"].mean()
            rec["yıllara göre 20G ort %"] = {int(k): round(float(v), 2) for k, v in yr.items()}
            rec["trende göre 20G ort %"] = g20.groupby("Trend")["SPY'ye göre 20G %"].mean() \
                .round(2).to_dict()
        tab[lab] = rec
    out["olaylar"] = tab
    for col in ("WHALE", "ΔWHALE 5G", "PRO−RETAIL", "OMNI", "HUD net"):
        out[f"{col} IC (20G)"] = _ic_summary(lv, col, "SPY'ye göre 20G %")
        try:
            q = pd.qcut(lv[col], 5, labels=["en düşük %20", "2", "3", "4", "en yüksek %20"],
                        duplicates="drop")
            out[f"{col} beşte birlik → 20G %"] = lv.groupby(q, observed=True)[
                "SPY'ye göre 20G %"].mean().round(2).to_dict()
        except Exception:                        # noqa: BLE001
            pass
    return out


# ------------------------------------------------------------------ özet
def summarize(R: dict[str, pd.DataFrame]) -> dict:
    S: dict = {}
    A = R.get("akis")
    if A is not None and len(A):
        s = {}
        for (ym, dn), g in A.groupby(["Yöntem", "Dönem"]):
            key = f"{ym} · {dn}"
            s[key] = {f"{n}G": _ic_summary(g, "Akış Skoru", f"Göreli {n}G %") for n in FWD}
            if ym == "yeni":
                s[key]["etiket ort. göreli 20G %"] = (
                    g.groupby("Akış")["Göreli 20G %"].mean().round(2).to_dict())
        S["akis"] = s
    B = R.get("tema")
    if B is not None and len(B):
        S["tema"] = {
            "Erken Skor IC": {f"{n}G": _ic_summary(B, "Erken Skor", f"SPY'ye göre {n}G %")
                              for n in FWD},
            "bölgeye göre SPY'ye göre ileri getiri %": B.groupby("Bölge")[
                [f"SPY'ye göre {n}G %" for n in FWD]].mean().round(2).to_dict(orient="index"),
            "taze lider": B.groupby("Taze Lider")["SPY'ye göre 20G %"].agg(["mean", "size"])
            .round(2).to_dict(orient="index")}
    C = R.get("endeks")
    if C is not None and len(C):
        out = {}
        for tf in ("Günlük", "Haftalık"):
            col = f"{tf} skor"
            if col not in C:
                continue
            out[tf] = {f"{n}G": _ic_summary(C, col, f"İleri {n}G %") for n in FWD}
            b = pd.cut(C[col], [-101, -20, 20, 101], labels=["≤ −20", "−20…+20", "≥ +20"])
            out[tf]["skor kovası → ileri getiri %"] = C.groupby(b, observed=True)[
                [f"İleri {n}G %" for n in FWD]].mean().round(2).to_dict(orient="index")
            # varlık bazında zaman serisi korelasyonu
            out[tf]["varlık bazında korelasyon (skor ~ 20G)"] = C.groupby("Varlık").apply(
                lambda g: round(float(g[col].corr(g["İleri 20G %"])), 2)
                if g[col].notna().sum() > 8 else np.nan).to_dict()
        S["endeks"] = out
    E = R.get("balina")
    if E is not None and len(E):
        S["balina"] = _whale_summary(E)
    D = R.get("hisse")
    if D is not None and len(D):
        tgt = "SPY'ye göre 20G %"

        def tab(col):
            g = D.groupby(col)
            return pd.DataFrame({"SPY'ye göre 20G ort %": g[tgt].mean(),
                                 "SPY'ye göre 5G ort %": g["SPY'ye göre 5G %"].mean(),
                                 "20G pozitif %": g[tgt].apply(lambda x: (x > 0).mean() * 100),
                                 "n": g.size()}).round(2).to_dict(orient="index")
        S["hisse"] = {"karar": tab("Karar"), "sınıf": tab("Sınıf"), "sinyal": tab("Sinyal"),
                      "Skor IC 20G": _ic_summary(D, "Skor", tgt),
                      "Huni Skoru IC 20G": _ic_summary(D, "Huni Skoru", tgt),
                      "plan sonucu (karara göre)": D.groupby(["Karar", "Sonuç"]).size()
                      .unstack(fill_value=0).to_dict(orient="index"),
                      "plan ort. R (karara göre)": D.groupby("Karar")["Sonuç R"].mean()
                      .round(2).to_dict(),
                      "Fib r kovası (yukarı bacak)": D[D["Fib yukarı"] == True].groupby(  # noqa: E712
                          pd.cut(D["Fib r"], [-1, 0.236, 0.382, 0.5, 0.618, 0.786, 2]),
                          observed=True)[tgt].agg(ort="mean", n="count").round(2)
                      .rename(index=str).to_dict(orient="index")}
    return S


def _js(o):
    if isinstance(o, dict):
        return {str(k): _js(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_js(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def write(R: dict[str, pd.DataFrame], meta: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    names = {}
    for k, df in R.items():
        if k == "balina":                        # büyük: sıkıştırılmış yaz
            names[k] = f"{k}.csv.gz"
            df.to_csv(OUT / names[k], index=False, float_format="%.2f", compression="gzip")
        else:
            names[k] = f"{k}.csv"
            df.to_csv(OUT / names[k], index=False, float_format="%.4f")
    try:
        meta["ozet"] = summarize(R)
    except Exception as exc:                     # noqa: BLE001
        meta["ozet_hata"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    (OUT / "ozet.json").write_text(json.dumps(_js(meta), ensure_ascii=False, indent=1))
    with zipfile.ZipFile(OUT / "sinyal_dogrulama.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in list(R):
            z.write(OUT / names[f], names[f])
        z.write(OUT / "ozet.json", "ozet.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ay", type=int, default=12)
    ap.add_argument("--bolum", default="balina,akis,tema,endeks,hisse")
    ap.add_argument("--hisse-adim", type=int, default=10, help="hisselerde kaç işlem gününde bir")
    ap.add_argument("--hisse-limit", type=int, default=0, help="test için ilk N hisse")
    a = ap.parse_args()
    parts = [x.strip() for x in a.bolum.split(",") if x.strip()]
    if "secki" in parts:                         # 📜 geçmiş öneriler: ayrı klasöre yazar
        parts.remove("secki")
        sys.path.insert(0, str(ROOT / "tools"))
        import secki_gecmis                      # noqa: E402
        secki_gecmis.main(a.ay, 5, 5, a.hisse_limit)
        if not parts:
            return 0
    meta = {"tarih": pd.Timestamp.now().isoformat(timespec="minutes"), "ay": a.ay,
            "bolumler": parts, "surum": "doğrulama v1", "hata": {}}
    R: dict[str, pd.DataFrame] = {}
    t0 = time.time()
    fns = {"akis": lambda: part_flows(a.ay), "tema": lambda: part_themes(a.ay),
           "endeks": lambda: part_markets(a.ay),
           "hisse": lambda: part_stocks(a.ay, a.hisse_adim, a.hisse_limit),
           "balina": lambda: part_whale(a.hisse_limit)}
    for k in parts:
        try:
            ts = time.time()
            R[k] = fns[k]()
            print(f"✅ {k}: {len(R[k])} satır · {(time.time() - ts) / 60:.1f} dk", flush=True)
        except Exception as exc:                 # noqa: BLE001
            meta["hata"][k] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        meta["sure_dk"] = round((time.time() - t0) / 60, 1)
        write(R, meta)
    print(json.dumps(_js(meta.get("ozet", {})), ensure_ascii=False, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
