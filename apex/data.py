# apex/data.py — AETHER APEX
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import numpy as np

import datetime as dt
import logging
import time
from typing import Iterable

import pandas as pd

_log = logging.getLogger(__name__)

# yfinance aralığı -> (geçmiş gün sayısı, yf interval)
INTERVAL_PLAN: dict[str, tuple[int, str]] = {
    "1d": (420, "1d"),
    "1d_long": (5800, "1d"),   # Karar Hunisi: çeyreklik oran grafikleri
    "1wk": (1500, "1wk"),
    "4h": (170, "1h"),      # 1h çekip 4h'e yeniden örnekliyoruz
    "1h": (60, "1h"),
}


def _extract(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """yf.download çıktısından tek sembolün OHLCV tablosunu ayıklar."""
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            lv0 = raw.columns.get_level_values(0)
            lv1 = raw.columns.get_level_values(1)
            if ticker in set(lv0):
                df = raw[ticker].copy()
            elif ticker in set(lv1):
                df = raw.xs(ticker, level=1, axis=1).copy()
            else:
                return pd.DataFrame()
        else:
            df = raw.copy()
    except Exception as exc:
        _log.warning("Sütun ayıklama hatası (%s): %s", ticker, exc)
        return pd.DataFrame()

    need = ["Open", "High", "Low", "Close", "Volume"]
    if not all(col in df.columns for col in need):
        return pd.DataFrame()
    return df[need].dropna(subset=["Close"])


def resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return df.resample("4h").agg({"Open": "first", "High": "max", "Low": "min",
                                  "Close": "last", "Volume": "sum"}).dropna()


def fetch(tickers: Iterable[str], interval: str = "1d",
          retries: int = 1) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """
    Sembolleri toplu çeker. Döner: ({sembol: OHLCV}, [başarısız semboller])
    """
    tickers = sorted({t.strip().upper() for t in tickers if t and t.strip()})
    if not tickers:
        return {}, []

    import yfinance as yf

    days, yf_int = INTERVAL_PLAN.get(interval, INTERVAL_PLAN["1d"])
    start = dt.datetime.now() - dt.timedelta(days=days)
    out: dict[str, pd.DataFrame] = {}

    for attempt in range(retries + 1):
        missing = [t for t in tickers if t not in out]
        if not missing:
            break
        try:
            raw = yf.download(tickers=" ".join(missing), start=start,
                              interval=yf_int, group_by="column",
                              auto_adjust=False, progress=False, threads=True)
            for t in missing:
                df = _extract(raw, t)
                if len(df) >= 30:
                    out[t] = resample_4h(df) if interval == "4h" else df
        except Exception as exc:
            _log.warning("Toplu çekim hatası (deneme %s): %s", attempt + 1, exc)
            time.sleep(1.5 * (attempt + 1))

    # kalanları tek tek dene
    for t in [t for t in tickers if t not in out]:
        try:
            hist = yf.Ticker(t).history(start=start, interval=yf_int)
            if len(hist) >= 30:
                need = ["Open", "High", "Low", "Close", "Volume"]
                if all(c in hist.columns for c in need):
                    df = hist[need].dropna(subset=["Close"])
                    out[t] = resample_4h(df) if interval == "4h" else df
        except Exception as exc:
            _log.info("Tekil çekim hatası (%s): %s", t, exc)

    failed = [t for t in tickers if t not in out]
    return out, failed


# --------------------------------------------------------------------------
# yfinance `info` — ortak, hız sınırına dayanıklı çağrı
# Yahoo, bulut sunucularından gelen yoğun isteklere geçici blok (HTTP 429 /
# "Too Many Requests") uygular. Bu yüzden: aynı anda en fazla 2 istek,
# blokta artan beklemeyle yeniden deneme ve 6 saatlik bellek içi önbellek
# (bilanço sekmesi ile short interest aynı veriyi paylaşır).
# --------------------------------------------------------------------------
import threading

_INFO_CACHE: dict[str, tuple[float, dict]] = {}
_INFO_SEM = threading.Semaphore(2)
_INFO_TTL = 6 * 3600


def _info_ok(info: dict) -> bool:
    return bool(info) and any(info.get(k) for k in (
        "currentPrice", "regularMarketPrice", "previousClose", "quoteType"))


def yf_info(t: str) -> dict:
    import yfinance as yf

    now = time.time()
    hit = _INFO_CACHE.get(t)
    if hit and now - hit[0] < _INFO_TTL:
        return hit[1]
    info: dict = {}
    for attempt in range(4):
        err = None
        with _INFO_SEM:
            try:
                info = yf.Ticker(t).info or {}
            except Exception as exc:
                err, info = exc, {}
        if _info_ok(info):
            _INFO_CACHE[t] = (now, info)
            return info
        txt = f"{type(err).__name__} {err}" if err else ""
        limited = "Rate" in txt or "Too Many" in txt or "429" in txt
        time.sleep((3 * (attempt + 1)) if limited else 0.4)
    return info


def _one_fundamental(t: str, today: dt.date) -> dict:
    import numpy as np
    import yfinance as yf

    rec = {"Hisse": t, "Bilanço": "—", "Kalan Gün": None,
           "Fiyat": None, "Hedef": None, "Potansiyel %": None,
           "Analist": "", "_sort": 99999,
           "Sektör": "", "Endüstri": "", "Açıklama": "",
           "_ttm_sps": np.nan, "_fwd_sps": np.nan, "Büyüme %": np.nan,
           "Satış Tahmin Kaynağı": "", "Piyasa Değ. ($B)": np.nan,
           "İleri F/K": np.nan, "EV/Satış": np.nan, "İleri HBK": np.nan,
           "52H Konum %": np.nan}

    def num(x):
        try:
            v = float(x)
            return v if np.isfinite(v) else np.nan
        except (TypeError, ValueError):
            return np.nan

    try:
        tk = yf.Ticker(t)
        info = yf_info(t)

        # ---------- bilanço tarihi (eski mantık aynen) ----------
        first_date = None
        try:
            cal = tk.calendar
            if isinstance(cal, dict):
                d = cal.get("Earnings Date")
                first_date = d[0] if isinstance(d, (list, tuple)) and d else d
            elif hasattr(cal, "empty") and not cal.empty:
                if "Earnings Date" in getattr(cal, "columns", []):
                    first_date = cal["Earnings Date"].iloc[0]
                elif "Earnings Date" in getattr(cal, "index", []):
                    first_date = cal.loc["Earnings Date"].iloc[0]
        except Exception:
            pass
        if first_date is not None:
            ed = (first_date.date() if hasattr(first_date, "date")
                  else pd.to_datetime(first_date).date())
            delta = (ed - today).days
            rec["Bilanço"] = ed.strftime("%d.%m.%Y")
            if delta >= 0:
                rec["Kalan Gün"] = delta
                rec["_sort"] = delta
            else:
                rec["_sort"] = 90000 - delta

        price = info.get("currentPrice") or info.get("previousClose")
        target = info.get("targetMeanPrice") or info.get("targetMedianPrice")
        rec["Fiyat"] = float(price) if isinstance(price, (int, float)) else None
        rec["Hedef"] = float(target) if isinstance(target, (int, float)) else None
        if rec["Fiyat"] and rec["Hedef"]:
            rec["Potansiyel %"] = (rec["Hedef"] / rec["Fiyat"] - 1) * 100
        n = info.get("numberOfAnalystOpinions")
        key = info.get("recommendationKey", "")
        rec["Analist"] = (f"{key} ({n})" if n else str(key or ""))

        # ---------- P/S adil değer için ham veri ----------
        rec["Sektör"] = info.get("sector", "") or ""
        rec["Endüstri"] = info.get("industry", "") or ""
        summ = (info.get("longBusinessSummary") or "").strip()
        rec["Açıklama"] = (summ.split(". ")[0][:220] + ".") if summ else ""

        shares = num(info.get("sharesOutstanding"))
        ttm_rev = num(info.get("totalRevenue"))
        rps = num(info.get("revenuePerShare"))
        ttm_sps = rps if np.isfinite(rps) and rps > 0 else (
            ttm_rev / shares if shares and np.isfinite(ttm_rev) else np.nan)
        rec["_ttm_sps"] = ttm_sps

        fwd_rev, est_growth, kaynak = np.nan, np.nan, ""
        try:
            re_ = tk.revenue_estimate
            if re_ is not None and not re_.empty and "avg" in re_.columns:
                for idx, lab in (("+1y", "gelecek mali yıl"),
                                 ("0y", "bu mali yıl")):
                    if idx in re_.index and np.isfinite(num(re_.loc[idx, "avg"])):
                        fwd_rev, kaynak = num(re_.loc[idx, "avg"]), lab
                        if "growth" in re_.columns:
                            est_growth = num(re_.loc[idx, "growth"]) * 100
                        break
        except Exception:
            pass

        growth = num(info.get("revenueGrowth")) * 100
        if not np.isfinite(growth):
            growth = est_growth
        if np.isfinite(fwd_rev) and shares:
            rec["_fwd_sps"] = fwd_rev / shares
        elif np.isfinite(ttm_sps) and np.isfinite(growth):
            rec["_fwd_sps"] = ttm_sps * (1 + growth / 100)
            kaynak = "TTM × (1+büyüme) tahmini"
        if not np.isfinite(growth) and np.isfinite(rec["_fwd_sps"]) \
                and np.isfinite(ttm_sps) and ttm_sps > 0:
            growth = (rec["_fwd_sps"] / ttm_sps - 1) * 100
        rec["Büyüme %"] = growth
        rec["Satış Tahmin Kaynağı"] = kaynak

        mc = num(info.get("marketCap"))
        rec["Piyasa Değ. ($B)"] = mc / 1e9 if np.isfinite(mc) else np.nan
        rec["İleri F/K"] = num(info.get("forwardPE"))
        rec["EV/Satış"] = num(info.get("enterpriseToRevenue"))
        rec["İleri HBK"] = num(info.get("forwardEps"))
        hi, lo = num(info.get("fiftyTwoWeekHigh")), num(info.get("fiftyTwoWeekLow"))
        if rec["Fiyat"] and np.isfinite(hi) and np.isfinite(lo) and hi > lo:
            rec["52H Konum %"] = (rec["Fiyat"] - lo) / (hi - lo) * 100
    except Exception as exc:
        _log.info("Bilanço/temel veri alınamadı (%s): %s", t, exc)
    return rec


def fetch_earnings_calendar(tickers: Iterable[str]) -> pd.DataFrame:
    """
    Bilanço tarihleri, analist hedefleri ve P/S adil değer için ham temel veri.
    Adil değer sütunları app tarafında valuation.add_fair_values() ile eklenir.
    """
    from concurrent.futures import ThreadPoolExecutor

    today = dt.date.today()
    syms = sorted({x.strip().upper() for x in tickers if x and x.strip()})
    if not syms:
        return pd.DataFrame()
    with ThreadPoolExecutor(max_workers=2) as ex:
        rows = list(ex.map(lambda s: _one_fundamental(s, today), syms))
    df = pd.DataFrame(rows).sort_values("_sort").drop(columns=["_sort"])
    return df.reset_index(drop=True)
