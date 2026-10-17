"""
GEÇMİŞ ÖNERİLER — Tema Avcısı ve Hisse Avcısı (🛒 Alım seçkisi) geçmişte HER HAFTA ne
önerirdi, ve o öneriler gerçekte ne oldu?

Walk-forward: her tarihte (varsayılan 5 işlem gününde bir) yalnızca O GÜNE KADARKİ fiyat
verisiyle uygulamanın bugünkü kuralları çalıştırılır:
  • Tema Avcısı  : her vade (kısa/orta/uzun) için tema rotasyonu → ✅ para giriyor (lider /
                   iyileşen) ya da ⛔. Sonuç: tema endeksinin sonraki H günde SPY'ye göre getirisi.
  • Hisse Avcısı : seçki kararları (✅ ALIM BÖLGESİNDE · 🟡 LİMİT EMİR · 🟢 MOMENTUM …),
                   öncelik, giriş bölgesi, stop, kâr al 1/2. Her temada en öncelikli N hisse
                   + bütün temalarda genel ilk 10 kaydedilir. Sonuç: limit emir doldu mu, sonra
                   kâr al 1 mi stop mu önce geldi (aynı gün ikisi → temkinli: stop), H günde
                   SPY'ye göre getiri.
  Vade → değerlendirme süresi H: kısa 10, orta 30, uzun 60 işlem günü.

Geçmişe taşınamayanlar (o günkü değeri bilinmiyor → KULLANILMAZ): analist hedefi / fair value,
F/K, temel sağlamlık, piyasa değeri sepeti, örgü şablon benzerliği, piyasa genişliği.
Yani 💰 🏛️ 🧮 💎 🌊 sinyalleri bu testte YOK; 🐈 ölü kedi, R1/R2, momentum, balina, Fib ve
EMA fazı VAR. Tema kadroları BUGÜNKÜ listedir (hayatta kalma yanlılığı: o gün listede
olmayan, sonradan eklenen başarılı hisseler var olabilir).

    python tools/validate_signals.py --ay 12 --bolum secki
Çıktı: data/validation/secki/ (oneriler.csv, tema.csv, taban.csv, ozet.json, gecmis_oneriler.zip)
"""
from __future__ import annotations

import json
import sys
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from apex import compute as cmp             # noqa: E402
from apex import data as dta                # noqa: E402
from apex import decision as dcs            # noqa: E402
from apex import ema_stack as ems           # noqa: E402
from apex import funnel as fnl              # noqa: E402
from apex import orgu as org                # noqa: E402
from apex import picks as pk                # noqa: E402
from apex import backtest as bt             # noqa: E402
from apex import universe as uni            # noqa: E402

OUT = ROOT / "data" / "validation" / "secki"
H = {"kısa": 10, "orta": 30, "uzun": 60}
FILL = {"kısa": 5, "orta": 10, "uzun": 15}
VADES = ("kısa", "orta", "uzun")


def _naive(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    ix = pd.DatetimeIndex(d.index)
    d.index = (ix.tz_localize(None) if ix.tz is not None else ix).normalize()
    return d[~d.index.duplicated(keep="last")].sort_index()


def _fetch(tickers) -> dict[str, pd.DataFrame]:
    tickers = sorted(set(tickers))
    out: dict[str, pd.DataFrame] = {}
    for k in range(0, len(tickers), 40):
        part = tickers[k:k + 40]
        p, f = dta.fetch(part, "1d_long")
        out.update({t: _naive(v).tail(1400) for t, v in p.items() if v is not None and len(v)})
        print(f"  veri {k + len(part)}/{len(tickers)} · eksik: {f[:6]}", flush=True)
        time.sleep(1.0)
    return out


def _fwd(c: pd.Series, t: pd.Timestamp, n: int) -> float:
    i = c.index.searchsorted(t, side="right") - 1
    if i < 0 or i + n >= len(c):
        return np.nan
    return float((c.iloc[i + n] / c.iloc[i] - 1) * 100)


def outcome(df: pd.DataFrame, t: pd.Timestamp, r: dict, vade: str) -> dict:
    """Plan seviyeleriyle ileriye bakış."""
    out = {"Girişe geldi": np.nan, "Giriş": np.nan, "Sonuç": "açık", "Sonuç R": np.nan,
           "Gün": np.nan}
    fut = df[df.index > t].head(H[vade])
    if len(fut) < H[vade]:
        out["Sonuç"] = "açık (süre dolmadı)"
        if len(fut) < 2:
            return out
    stop, t1 = r["Stop"], r["Kâr al 1"]
    if not (np.isfinite(stop) and np.isfinite(t1)):
        out["Sonuç"] = "plan yok"
        return out
    if str(r["Karar"]).startswith(("✅", "🟢")):
        entry, i0 = float(r["Fiyat"]), 0
    else:
        top = r["Giriş Üst"]
        lows = fut["Low"].head(FILL[vade]).to_numpy()
        hit = np.flatnonzero(lows <= top) if np.isfinite(top) else []
        if not len(hit):
            out["Girişe geldi"], out["Sonuç"] = False, "giriş olmadı"
            return out
        i0 = int(hit[0])
        entry = float(min(fut["Open"].iloc[i0], top))
    out["Girişe geldi"], out["Giriş"] = True, entry
    risk = entry - stop
    if risk <= 0 or t1 <= entry:
        out["Sonuç"] = "plan geçersiz"
        return out
    for j in range(i0, len(fut)):
        lo, hi = fut["Low"].iloc[j], fut["High"].iloc[j]
        if lo <= stop:
            out.update({"Sonuç": "❌ stop", "Sonuç R": -1.0, "Gün": j + 1})
            return out
        if hi >= t1:
            out.update({"Sonuç": "✅ hedef 1", "Sonuç R": (t1 - entry) / risk, "Gün": j + 1})
            return out
    last = float(fut["Close"].iloc[-1])
    if len(fut) >= H[vade]:
        out["Sonuç"] = "⏳ süre doldu (+)" if last > entry else "⏳ süre doldu (−)"
    out["Sonuç R"], out["Gün"] = (last - entry) / risk, len(fut)
    return out


def _orgu_at(X: pd.DataFrame | None, t: pd.Timestamp) -> dict:
    if X is None:
        return {}
    sub = X[X.index <= t].tail(5)
    if sub.empty:
        return {}
    r = sub.iloc[-1]
    g = lambda k: float(r.get(k, np.nan))          # noqa: E731
    return {"🐈 Ölü kedi": bool(sub["olu_kedi"].max() > 0),
            "R1": bool(g("atr_pct") > 1.88 and g("x_lambda") > 2.30 and g("atr_ratio") <= 0.69),
            "R2": bool(g("x_lambda_ch") <= -0.34 and g("x_lambda") <= 16.46
                       and g("vol_10_60") <= 0.74 and g("x_amihud") <= -0.23),
            "Örgü puanı": np.nan}


def run(months: int = 12, step: int = 5, top: int = 5, limit: int = 0) -> dict[str, pd.DataFrame]:
    t0 = time.time()
    print("S) Geçmiş öneriler (Tema + Hisse Avcısı)", flush=True)
    tmap = {t: cmp.theme_holdings(t) for t in uni.THEME_TRACKER}
    tmap = {t: v for t, v in tmap.items() if v}
    syms = sorted({x for v in tmap.values() for x in v})
    if limit:
        syms = syms[:limit]
        tmap = {t: [x for x in v if x in set(syms)] for t, v in tmap.items()}
        tmap = {t: v for t, v in tmap.items() if v}
    etfs = sorted({e for lst in uni.THEME_TRACKER.values() for e in lst})
    P = _fetch(syms + etfs + ["SPY", "QQQ"])
    spy = P["SPY"]["Close"]
    cal = P["SPY"].index
    print(f"  ön hesap: {len(syms)} hisse", flush=True)
    PRE, ORG = {}, {}
    for k, s in enumerate(syms, 1):
        d = P.get(s)
        if d is None or len(d) < 260:
            continue
        try:
            PRE[s] = pk.precompute(d)
            ORG[s] = org.series(PRE[s]["d"], spy)
        except Exception as exc:                 # noqa: BLE001
            print(f"  {s}: {exc}", flush=True)
        if k % 100 == 0:
            print(f"  ön hesap {k}/{len(syms)} · {(time.time() - t0) / 60:.1f} dk", flush=True)
    # tema endeksleri (ileri getiri için, ETF'lerden — Tema Avcısı ile aynı)
    full_idx = {}
    for tema, mem in uni.THEME_TRACKER.items():
        ix, _ = fnl.theme_index(P, mem)
        if ix is not None:
            full_idx[tema] = ix
    start = cal[-1] - pd.DateOffset(months=months)
    dates = list(cal[cal >= start][::step])
    rows, trows, brows = [], [], []
    for n_d, t in enumerate(dates, 1):
        # --- hisse ölçümleri (o güne kadarki veri)
        recs = []
        for s, pre in PRE.items():
            i = int(pre["d"].index.searchsorted(t, side="right") - 1)
            if i < 119 or (t - pre["d"].index[i]).days > 7:
                continue
            try:
                m = pk.stock_metrics(s, None, None, _orgu_at(ORG.get(s), t), pre=pre, i=i)
            except Exception:                    # noqa: BLE001
                m = None
            if m:
                recs.append(m)
        S = pk.add_momentum(pd.DataFrame(recs))
        if S is None or S.empty:
            continue
        # taban: o gün bütün evrenin H gün sonra SPY'ye göre ortalaması (eşit ağırlık)
        for v in VADES:
            fw = [(_fwd(P[s]["Close"], t, H[v]), s) for s in S["Sembol"]]
            sp = _fwd(spy, t, H[v])
            ex = [a - sp for a, _ in fw if np.isfinite(a) and np.isfinite(sp)]
            brows.append({"Tarih": t.date(), "Vade": v, "Taban SPY'ye göre %":
                          float(np.mean(ex)) if ex else np.nan,
                          "Taban SPY'yi geçen %": float(np.mean(np.array(ex) > 0) * 100)
                          if ex else np.nan, "Evren": len(ex)})
        # endeks bağlamı (EMA şeridi; MV kararı yerine yaklaşık: günlük faz skoru ≥ 0)
        sd = P["SPY"][P["SPY"].index <= t]
        a_g = ems.analyze(sd, "Günlük")
        a_h = ems.analyze(dcs.resample(sd, "W-FRI"), "Haftalık")
        index_ok = bool(a_g.get("ok") and a_g["skor"] >= 0)
        itn = pk.index_turn({"Günlük": ems.table_row("S&P 500", "Günlük", a_g),
                             "Haftalık": ems.table_row("S&P 500", "Haftalık", a_h)}, "S&P 500")
        sub = {k: v[v.index <= t].tail(420) for k, v in P.items() if k in etfs or k == "SPY"}
        for v in VADES:
            T, _ = fnl.theme_rotation(sub, uni.THEME_TRACKER, horizon=v)
            if T is None or T.empty:
                continue
            tq = dict(zip(T["Tema"], T["_q"]))
            trn = {r_["Tema"]: pk.theme_turn(r_) for r_ in T.to_dict("records")}
            sp = _fwd(spy, t, H[v])
            for r_ in T.to_dict("records"):
                ix = full_idx.get(r_["Tema"])
                a = _fwd(ix, t, H[v]) if ix is not None else np.nan
                ok_ = r_["_q"] in ("lider", "iyilesen")
                exr = a - sp if np.isfinite(a) and np.isfinite(sp) else np.nan
                trows.append({"Tarih": t.date(), "Vade": v, "Tema": r_["Tema"],
                              "Öneri": "✅ para giriyor" if ok_ else "⛔ uzak dur",
                              "Para": r_["_q"], "Bölge": r_.get("Bölge", ""),
                              "Erken Skor": r_.get("Erken Skor", np.nan),
                              "Taze Lider": bool(r_.get("Taze Lider", False)),
                              "Dönüş": trn.get(r_["Tema"], ""),
                              "H (gün)": H[v], "Sonraki H gün tema %": a,
                              "SPY'ye göre %": exr,
                              "Doğru": (exr > 0) == ok_ if np.isfinite(exr) else np.nan})
            tz = set(T.loc[T["Taze Lider"].astype(bool), "Tema"]) if "Taze Lider" in T else set()
            B = pk.board(S, tmap, tq, v, index_ok, 10.0, None, trn, itn, {}, tz)
            if B.empty:
                continue
            act = B[B["_o"] <= 1].sort_values("Öncelik", ascending=False)
            top10 = set(act.drop_duplicates("Hisse").head(10)["Hisse"])
            pick = act.groupby("Tema", sort=False).head(top)
            pick = pd.concat([pick, act[act["Hisse"].isin(top10)].drop_duplicates("Hisse")]
                             ).drop_duplicates(["Tema", "Hisse"])
            for r in pick.to_dict("records"):
                df = P.get(r["Hisse"])
                oc = outcome(df, t, r, v)
                bo = bt.evaluate_one(df, t, r["Karar"], r["Fiyat"], r["Giriş Üst"], r["Stop"],
                                     r["Kâr al 1"], r["Kâr al 2"], None, FILL[v], H[v])
                a = _fwd(df["Close"], t, H[v])
                rows.append({
                    "Tarih": t.date(), "Vade": v, "Tema": r["Tema"], "Para": r["_q"],
                    "Hisse": r["Hisse"], "Karar": r["Karar"], "Öncelik": r["Öncelik"],
                    "Genel ilk 10": r["Hisse"] in top10, "Sepet": r["Sepet"],
                    "Kısa sinyal": r["Kısa sinyal"], "Orta-uzun sinyal": r["Orta-uzun sinyal"],
                    "Zamanlama": r["Zamanlama"], "EMA fazı": r["EMA fazı"],
                    "Fiyat": r["Fiyat"], "Giriş Alt": r["Giriş Alt"], "Giriş Üst": r["Giriş Üst"],
                    "Stop": r["Stop"], "Kâr al 1": r["Kâr al 1"], "Kâr al 2": r["Kâr al 2"],
                    "R:R": r["R:R"], "Momentum": r.get("Momentum", np.nan),
                    "Balina": r.get("Balina", ""), **oc,
                    "BT Durum": bo["Durum"], "BT Seviye": bo["Seviye"], "BT Kâr %": bo["Kâr %"],
                    "BT Kâra geçme (gün)": bo["Kâra geçme (gün)"], "BT Giriş": bo["Giriş"],
                    f"Sonraki H gün %": a,
                    "SPY'ye göre %": a - sp if np.isfinite(a) and np.isfinite(sp) else np.nan})
        print(f"  tarih {n_d}/{len(dates)} {t.date()} · öneri {len(rows)} · "
              f"{(time.time() - t0) / 60:.1f} dk", flush=True)
    return {"oneriler": pd.DataFrame(rows), "tema": pd.DataFrame(trows),
            "taban": pd.DataFrame(brows)}


# ------------------------------------------------------------------ özet
_grp = pk.outcome_summary


def summarize(R: dict[str, pd.DataFrame]) -> dict:
    O, T, Bs = R.get("oneriler", pd.DataFrame()), R.get("tema", pd.DataFrame()), R.get(
        "taban", pd.DataFrame())
    s: dict = {}
    if not O.empty:
        O = O.copy()
        s["hisse_vade"] = _grp(O, "Vade")
        s["hisse_karar"] = _grp(O, ["Vade", "Karar"])
        s["hisse_ilk10"] = _grp(O, ["Vade", "Genel ilk 10"])
        s["hisse_para"] = _grp(O, ["Vade", "Para"])
        for sig in ("🚀", "🐋", "🧬"):
            s[f"hisse_{sig}"] = _grp(O[O["Kısa sinyal"].astype(str).str.contains(sig)], "Vade")
        O["Ay"] = pd.to_datetime(O["Tarih"]).dt.strftime("%Y-%m")
        s["hisse_ay"] = _grp(O, ["Vade", "Ay"])
    if not Bs.empty:
        s["taban"] = (Bs.groupby("Vade")[["Taban SPY'ye göre %", "Taban SPY'yi geçen %"]]
                      .mean().reset_index().to_dict("records"))
    if not T.empty:
        rows = []
        for (v, o), g in T.groupby(["Vade", "Öneri"]):
            ex = g["SPY'ye göre %"].dropna()
            rows.append({"Vade": v, "Öneri": o, "n": len(g),
                         "SPY'ye göre ort %": float(ex.mean()) if len(ex) else np.nan,
                         "SPY'yi geçen %": float((ex > 0).mean() * 100) if len(ex) else np.nan,
                         "Doğru %": float(g["Doğru"].dropna().astype(float).mean() * 100)
                         if g["Doğru"].notna().any() else np.nan})
        s["tema_oneri"] = rows
        rows = []
        for (v, q), g in T.groupby(["Vade", "Para"]):
            ex = g["SPY'ye göre %"].dropna()
            rows.append({"Vade": v, "Para": q, "n": len(g),
                         "SPY'ye göre ort %": float(ex.mean()) if len(ex) else np.nan,
                         "SPY'yi geçen %": float((ex > 0).mean() * 100) if len(ex) else np.nan})
        s["tema_para"] = rows
    return s


def _js(o):
    if isinstance(o, dict):
        return {str(k): _js(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_js(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 3)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def write(R: dict[str, pd.DataFrame], meta: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for k, df in R.items():
        df.to_csv(OUT / f"{k}.csv", index=False, float_format="%.4f")
    try:
        meta["ozet"] = summarize(R)
    except Exception as exc:                     # noqa: BLE001
        meta["ozet_hata"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    (OUT / "ozet.json").write_text(json.dumps(_js(meta), ensure_ascii=False, indent=1))
    with zipfile.ZipFile(OUT / "gecmis_oneriler.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for k in R:
            z.write(OUT / f"{k}.csv", f"{k}.csv")
        z.write(OUT / "ozet.json", "ozet.json")


def main(months: int = 12, step: int = 5, top: int = 5, limit: int = 0) -> None:
    t0 = time.time()
    meta = {"tarih": pd.Timestamp.now().isoformat(timespec="minutes"), "ay": months,
            "adim": step, "tema_basina": top, "surum": "geçmiş öneriler v2 (seçki v2 kuralları)",
            "vade_gun": H, "limit_dolma_gun": FILL}
    try:
        R = run(months, step, top, limit)
    except Exception as exc:                     # noqa: BLE001
        traceback.print_exc()
        meta["hata"] = f"{type(exc).__name__}: {exc}"
        R = {}
    meta["sure_dk"] = round((time.time() - t0) / 60, 1)
    write(R, meta)
    print(json.dumps(_js(meta.get("ozet", {}).get("hisse_karar", [])), ensure_ascii=False,
                     indent=1)[:3000], flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ay", type=int, default=12)
    ap.add_argument("--adim", type=int, default=5)
    ap.add_argument("--tema-basina", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    main(a.ay, a.adim, a.tema_basina, a.limit)
