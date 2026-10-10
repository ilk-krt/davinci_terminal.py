# apex/compute.py — Streamlit'siz hesap katmanı
#
# Arayüzün (davinci_terminal.py) ve akşam hesap motorunun
# (tools/build_snapshot.py) KULLANDIĞI ORTAK fonksiyonlar. Burada önbellek ya
# da ekran kodu yoktur; aynı girdi her yerde aynı sonucu verir.
from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd

from apex import data as dta
from apex import decision as dcs
from apex import engine as eng
from apex import funnel as fnl
from apex import macro as mac
from apex import universe as uni

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------- makro
def macro_frames() -> dict[str, pd.DataFrame]:
    """Makro serileri bir kez çeker; hem güncel durum hem geçmiş bunu kullanır."""
    prices, failed = dta.fetch(mac.MACRO_TICKERS.values(), "1d")
    out = {k: prices.get(v, pd.DataFrame()) for k, v in mac.MACRO_TICKERS.items()}
    out["_failed"] = pd.DataFrame({"sembol": failed}) if failed else pd.DataFrame()
    return out


def macro_state(frames: dict[str, pd.DataFrame]) -> mac.MacroState:
    failed = frames.get("_failed")
    by_key = {k: v for k, v in frames.items() if k != "_failed"}
    state = mac.build_macro_state(by_key)
    if failed is not None and not failed.empty:
        state.errors.append("Çekilemeyen makro sembol: "
                            + ", ".join(failed["sembol"]))
    return state


def macro_history(frames: dict[str, pd.DataFrame], days: int = 60):
    """Batarya seyri + dönemsel değişim tabloları (geçmiş yeniden hesaplanır)."""
    frames = {k: v for k, v in frames.items() if k != "_failed"}
    state = mac.build_macro_state(frames)
    changes, past = mac.battery_changes(frames, state)
    scores = mac.score_changes(state, past)
    hist = mac.battery_history(frames, days)
    return changes, scores, hist, mac.regime_shifts(hist)


# ---------------------------------------------------------------- tarama
def snapshot_universe() -> list[str]:
    """Akşam hesabının taradığı semboller: uygulamanın herhangi bir sekmesinde
    varsayılan olarak görünen her sembol (hisse + ETF)."""
    out: set[str] = set(uni.all_stocks()) | set(uni.all_etfs())
    out |= set(uni.MAIN_SECTORS)
    out |= {t for lst in uni.THEME_TRACKER.values() for t in lst}
    out |= set(uni.DEFAULT_EARNINGS)
    for v in uni.DEFAULT_FUTURE_THEMES.values():
        for lst in v.values():
            out |= set(lst)
    for e in uni.all_etfs():
        out |= set(uni.holdings(e))
    return sorted(t for t in out if t and "." not in t)


def earnings_universe() -> list[str]:
    """Bilanço / değerleme verisi akşam hesabında çekilen hisseler."""
    return sorted(set(uni.all_stocks()) | set(uni.DEFAULT_EARNINGS)
                  | {t for v in uni.DEFAULT_FUTURE_THEMES.values()
                     for t in v.get("hisse", [])})
def scan(tickers, interval: str) -> pd.DataFrame:
    """Verilen sembolleri tarar ve sinyal tablosunu döner."""
    tickers = tuple(sorted(set(tickers)))
    if not tickers:
        return pd.DataFrame()

    need = list(tickers) + [eng.BENCHMARK]
    prices, failed = dta.fetch(need, interval)
    bench = prices.get(eng.BENCHMARK)
    bench_close = bench["Close"] if bench is not None and not bench.empty else None

    weekly_map: dict[str, bool] = {}
    if interval == "1d":
        wk, _ = dta.fetch(tickers, "1wk")
        for t, df in wk.items():
            if len(df) > 12:
                ema12 = df["Close"].ewm(span=12, adjust=False).mean()
                weekly_map[t] = bool(df["Close"].iloc[-1] > ema12.iloc[-1])

    rows: list[dict] = []
    for t in tickers:
        df = prices.get(t)
        if df is None or df.empty:
            rows.append({"Sembol": t, "Sinyal": "⚫ VERİ YOK",
                         "Hata": "Fiyat verisi çekilemedi"})
            continue
        row = eng.analyze(df, t, bench_close=bench_close,
                          weekly_bull=weekly_map.get(t))
        if not row.ok:
            rows.append({"Sembol": t, "Sinyal": "⚫ VERİ YOK", "Hata": row.error})
            continue
        rec = {"Sembol": t, **row.data, "Hata": ""}
        rows.append(rec)

    out = pd.DataFrame(rows)
    if "MAGNITUDE" in out.columns:
        out = out.sort_values("MAGNITUDE", ascending=False)
    return out


def theme_performance() -> pd.DataFrame:
    """Tema bazlı çok periyotlu performans + ivme değişimi."""
    etfs = sorted({t for lst in uni.THEME_TRACKER.values() for t in lst})
    prices, failed = dta.fetch(etfs, "1d")

    perf: dict[str, dict[str, float]] = {}
    year = dt.date.today().year
    for t, df in prices.items():
        c = df["Close"].dropna()
        if len(c) < 8:
            continue

        def chg(a: int, b: int = 0) -> float:
            if len(c) <= a:
                return np.nan
            end = c.iloc[-1 - b]
            start = c.iloc[-1 - a]
            return (end / start - 1) * 100 if start else np.nan

        ytd_df = c[c.index.year == year]
        ytd = ((c.iloc[-1] / ytd_df.iloc[0] - 1) * 100
               if len(ytd_df) > 1 else np.nan)
        prev_year = c[c.index.year == year - 1]
        ytd_prev = np.nan
        if len(prev_year) > 1:
            doy = dt.date.today().timetuple().tm_yday
            upto = prev_year[prev_year.index.dayofyear <= doy]
            if len(upto) > 1:
                ytd_prev = (upto.iloc[-1] / prev_year.iloc[0] - 1) * 100

        perf[t] = {
            "Bugün": chg(1), "Prev_Bugün": chg(2, 1),
            "1H": chg(5), "Prev_1H": chg(10, 5),
            "1A": chg(21), "Prev_1A": chg(42, 21),
            "3A": chg(63), "Prev_3A": chg(126, 63),
            "YBB": ytd, "Prev_YBB": ytd_prev,
        }

    pdf = pd.DataFrame.from_dict(perf, orient="index")
    if pdf.empty:
        return pdf

    rows = {}
    for tema, lst in uni.THEME_TRACKER.items():
        valid = [t for t in lst if t in pdf.index]
        if valid:
            rows[tema] = pdf.loc[valid].mean()
            rows[tema]["Semboller"] = ", ".join(valid)
    out = pd.DataFrame.from_dict(rows, orient="index")
    return out



# ---------------------------------------------------------------- Karar Hunisi
def rotation():
    """Adım 1–2: rotasyon oranları (uzun geçmiş) + repo verileri."""
    prices, failed = dta.fetch(fnl.ROTATION_TICKERS, "1d_long")
    crypto = fnl.read_crypto_caps()
    liq = fnl.read_liquidity()
    series = fnl.build_series(prices, crypto, liq)
    rot = fnl.rotation_table(series)
    files = {
        "kripto": (f"{crypto.index.max():%d.%m.%Y}" if not crypto.empty else None),
        "likidite": (f"{liq.index.max():%d.%m.%Y}" if not liq.empty else None),
    }
    return rot, series, failed, files


def theme_rotation():
    """Adım 4: tema ve ETF'lerde göreli rotasyon (RRG)."""
    etfs = sorted({t for lst in uni.THEME_TRACKER.values() for t in lst})
    prices, failed = dta.fetch(etfs + ["SPY"], "1d")
    T, tails = fnl.theme_rotation(prices, uni.THEME_TRACKER)
    E = fnl.etf_rotation(prices, etfs)
    idx_map = {}
    for tema, members in uni.THEME_TRACKER.items():
        idx, _ = fnl.theme_index(prices, members)
        if idx is not None:
            idx_map[tema] = idx
    return T, tails, E, idx_map, failed


def decision(rot: pd.DataFrame):
    """Adım 3: S&P 500 / Nasdaq / Kripto — G/H/A modül tabloları."""
    prices, failed = dta.fetch(dcs.DEC_TICKERS, "1d_long")
    spy = prices.get("SPY")
    qqq = prices.get("QQQ")
    assets = {}
    for ad, t in dcs.DEC_ASSETS.items():
        df = prices.get(t)
        mods, scores, sig = {}, {}, {}
        if df is not None and not df.empty:
            for tf, rule in dcs.TFS.items():
                d_ = dcs.resample(df, rule)
                b_ = dcs.resample(spy, rule) if spy is not None else None
                q_ = dcs.resample(qqq, rule) if qqq is not None else None
                if rule is None:
                    d_ = d_.tail(900)
                bench = b_["Close"] if (b_ is not None and t != "SPY") else None
                ndx = q_["Close"] if q_ is not None else None
                frac = dcs.partial_frac(df, rule)
                try:
                    m, eng_d = dcs.evaluate_tf(d_, bench, t, frac, tf, ndx)
                except Exception as exc:          # tek zaman dilimi hatası
                    m, eng_d = pd.DataFrame(), {}
                    _log.info("Karar %s %s: %s", t, tf, exc)
                mods[tf] = m
                scores[tf] = dcs.tf_score(m)
                sig[tf] = eng_d.get("Sinyal", "—")
        assets[ad] = {"mods": mods, "scores": scores, "sig": sig, "ticker": t}
    vix = prices.get("^VIX")
    vr = dcs.vix_rule(vix["Close"].dropna()) if vix is not None and not vix.empty         else {"etiket": "—", "skor": 0.0, "not": "VIX verisi yok"}
    notes = dcs.market_notes(prices, rot)
    return {"assets": assets, "vix": vr, "notes": notes, "failed": failed}


def theme_holdings(tema: str) -> list[str]:
    """Temadaki ETF'lerin bileşenleri; bileşeni bilinmeyen sembol hissenin
    kendisidir (ör. IONQ, MARA)."""
    out: list[str] = []
    for t in uni.THEME_TRACKER.get(tema, []):
        h = uni.holdings(t)
        if h:
            out += [x for x in h if x not in out]
        elif t not in uni.ETF and t not in out:
            out.append(t)
    return [x for x in out if "." not in x]      # yabancı borsa sembolleri hariç


def plan_universe() -> list[str]:
    """İşlem planı hesaplanan hisseler: bütün temaların bileşenleri."""
    out: set[str] = set()
    for tema in uni.THEME_TRACKER:
        out |= set(theme_holdings(tema))
    return sorted(out)


def plans(tickers) -> pd.DataFrame:
    """Giriş / stop / hedef planları (günlük mumdan)."""
    from apex import plan as pln
    tickers = sorted(set(tickers))
    if not tickers:
        return pd.DataFrame()
    prices, _ = dta.fetch(tickers, "1d")
    return pln.plans(prices, tickers)
