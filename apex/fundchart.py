# apex/fundchart.py — AETHER APEX
from __future__ import annotations

from apex.data import yf_info
from apex import universe as uni

# apex/fundchart.py — TEK HİSSE DEĞERLEME GRAFİKLERİ
#
# 1) F/K (TTM) geçmişi + akran medyanı
#    Hissenin günlük fiyatı / son 4 çeyreğin HBK toplamı. Yanında seçilen
#    akranların aynı hesapla bulunan F/K medyanı.
#
# 2) Fiyat vs medyan çarpan adil fiyatı ("FAST Graphs" mantığı)
#    Hisse başı TTM metrik (faaliyet nakit akışı, serbest nakit akışı,
#    satış ya da HBK) × hissenin kendi geçmişteki MEDYAN çarpanı = o
#    tarihteki "normal" fiyat. Analist tahminleriyle 2 mali yıl ileri uzatılır.
#
# Veri: yfinance. Yıllık tablolar ~4–5 yıl, çeyreklik ~5–6 çeyrek geri gider;
# düzeltilmiş HBK (earnings_dates) ~10 yıl. Seriler bu ikisinin birleşimidir.

from typing import Any

import numpy as np
import pandas as pd

METRICS: dict[str, dict[str, Any]] = {
    "OCF": dict(ad="Faaliyet nakit akışı", kisa="P/OCF",
                stmt="cf", row=["Operating Cash Flow",
                                "Cash Flow From Continuing Operating Activities"]),
    "FCF": dict(ad="Serbest nakit akışı", kisa="P/FCF",
                stmt="cf", row=["Free Cash Flow"]),
    "REV": dict(ad="Satış", kisa="F/S", stmt="inc",
                row=["Total Revenue", "Operating Revenue"]),
    "EPS": dict(ad="Hisse başı kâr (GAAP)", kisa="F/K", stmt="inc",
                row=["Diluted EPS", "Basic EPS"], per_share=True),
}
SHARE_ROWS = ["Diluted Average Shares", "Basic Average Shares"]


# --------------------------------------------------------------------------
# Ham veri
# --------------------------------------------------------------------------
def fetch_raw(t: str) -> dict[str, Any]:
    """Tek hisse için gereken tüm ham veriyi çeker (hatalar sessizce boş)."""
    import yfinance as yf

    tk = yf.Ticker(t)
    out: dict[str, Any] = {"ticker": t, "errors": []}

    def grab(name, fn):
        try:
            v = fn()
            out[name] = v if v is not None else pd.DataFrame()
        except Exception as exc:
            out[name] = pd.DataFrame()
            out["errors"].append(f"{name}: {type(exc).__name__}")

    grab("price", lambda: tk.history(period="10y", auto_adjust=False)["Close"])
    grab("q_inc", lambda: tk.quarterly_income_stmt)
    grab("a_inc", lambda: tk.income_stmt)
    grab("q_cf", lambda: tk.quarterly_cashflow)
    grab("a_cf", lambda: tk.cashflow)
    grab("edates", lambda: tk.get_earnings_dates(limit=44))
    grab("eps_est", lambda: tk.earnings_estimate)
    grab("rev_est", lambda: tk.revenue_estimate)
    try:
        out["info"] = yf_info(t)
    except Exception:
        out["info"] = {}
    return out


def fetch_price_eps_only(t: str) -> dict[str, Any]:
    """Akranlar için hafif çekim: fiyat + düzeltilmiş HBK geçmişi."""
    import yfinance as yf

    tk = yf.Ticker(t)
    out: dict[str, Any] = {"ticker": t}
    try:
        out["price"] = tk.history(period="10y", auto_adjust=False)["Close"]
    except Exception:
        out["price"] = pd.Series(dtype=float)
    try:
        out["edates"] = tk.get_earnings_dates(limit=44)
    except Exception:
        out["edates"] = pd.DataFrame()
    try:
        out["q_inc"] = tk.quarterly_income_stmt
        out["a_inc"] = tk.income_stmt
    except Exception:
        out["q_inc"], out["a_inc"] = pd.DataFrame(), pd.DataFrame()
    return out


# --------------------------------------------------------------------------
# Yardımcılar
# --------------------------------------------------------------------------
def _naive(idx) -> pd.DatetimeIndex:
    idx = pd.to_datetime(idx)
    try:
        idx = idx.tz_localize(None)
    except TypeError:
        idx = idx.tz_convert(None)
    return idx.normalize()


def _price(raw: dict[str, Any]) -> pd.Series:
    p = raw.get("price")
    if p is None or len(p) == 0:
        return pd.Series(dtype=float)
    p = pd.Series(p).dropna().astype(float)
    p.index = _naive(p.index)
    return p[~p.index.duplicated(keep="last")].sort_index()


def _row(df: pd.DataFrame, names: list[str]) -> pd.Series:
    """Mali tablodan bir satır (sütunlar tarih). Döner: tarih → değer."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.Series(dtype=float)
    for n in names:
        if n in df.index:
            s = pd.to_numeric(df.loc[n], errors="coerce")
            s.index = _naive(s.index)
            return s.dropna().sort_index()
    return pd.Series(dtype=float)


def ttm_events(q: pd.Series, a: pd.Series, flow: bool = True) -> pd.Series:
    """
    Tarih → TTM değer olayları.
    Çeyreklik veri: ardışık 4 çeyrek varsa toplamı (akış kalemleri).
    Yıllık veri: mali yıl sonundaki yıllık değer.
    Çakışan dönemlerde çeyreklik tercih edilir (daha güncel).
    """
    ev: dict[pd.Timestamp, float] = {}
    for d, v in a.items():
        ev[d] = float(v)
    if len(q) >= 4 and flow:
        qs = q.sort_index()
        for i in range(3, len(qs)):
            win = qs.iloc[i - 3:i + 1]
            span = (win.index[-1] - win.index[0]).days
            if 250 <= span <= 300:             # gerçekten ardışık 4 çeyrek
                ev[win.index[-1]] = float(win.sum())
    return pd.Series(ev).sort_index()


def shares_series(raw: dict[str, Any]) -> pd.Series:
    a = _row(raw.get("a_inc"), SHARE_ROWS)
    q = _row(raw.get("q_inc"), SHARE_ROWS)
    s = pd.concat([a, q]).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if s.empty:
        so = (raw.get("info") or {}).get("sharesOutstanding")
        if so:
            s = pd.Series({pd.Timestamp.today().normalize(): float(so)})
    return s


def per_share_ttm(raw: dict[str, Any], key: str) -> pd.Series:
    """Hisse başı TTM metrik olayları (tarih → değer)."""
    m = METRICS[key]
    src_q = raw.get("q_cf" if m["stmt"] == "cf" else "q_inc")
    src_a = raw.get("a_cf" if m["stmt"] == "cf" else "a_inc")
    q, a = _row(src_q, m["row"]), _row(src_a, m["row"])
    ev = ttm_events(q, a, flow=True)
    if ev.empty:
        return ev
    if m.get("per_share"):
        return ev
    sh = shares_series(raw)
    if sh.empty:
        return pd.Series(dtype=float)
    sh_at = sh.reindex(sh.index.union(ev.index)).sort_index().ffill().bfill()
    return (ev / sh_at.reindex(ev.index)).dropna()


def adjusted_eps_ttm(raw: dict[str, Any]) -> pd.Series:
    """earnings_dates'teki 'Reported EPS' (düzeltilmiş) — 4'lü kayan toplam,
    açıklama tarihinde bilinir hale gelir."""
    ed = raw.get("edates")
    if ed is None or not isinstance(ed, pd.DataFrame) or ed.empty:
        return pd.Series(dtype=float)
    col = next((c for c in ed.columns if "Reported" in str(c)), None)
    if col is None:
        return pd.Series(dtype=float)
    s = pd.to_numeric(ed[col], errors="coerce").dropna()
    s.index = _naive(s.index)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.rolling(4).sum().dropna()


def step_daily(events: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """Olay serisini günlük fiyat endeksine 'en son bilinen değer' olarak yayar."""
    if events.empty:
        return pd.Series(np.nan, index=idx)
    u = idx.union(events.index)
    return events.reindex(u).sort_index().ffill().reindex(idx)


# --------------------------------------------------------------------------
# 1) F/K geçmişi + akran medyanı
# --------------------------------------------------------------------------
def pe_history(raw: dict[str, Any], basis: str = "GAAP") -> pd.Series:
    p = _price(raw)
    if p.empty:
        return pd.Series(dtype=float)
    if basis == "GAAP":
        eps = per_share_ttm(raw, "EPS")
        if eps.empty:
            eps = adjusted_eps_ttm(raw)
    else:
        eps = adjusted_eps_ttm(raw)
        if eps.empty:
            eps = per_share_ttm(raw, "EPS")
    if eps.empty:
        return pd.Series(dtype=float)
    e = step_daily(eps, p.index)
    pe = (p / e).where(e > 0)
    return pe[pe.index >= eps.index.min()].dropna()


def peer_median(pe_map: dict[str, pd.Series], freq: str = "ME",
                min_n: int = 3) -> tuple[pd.Series, pd.Series]:
    """Akranların F/K'larının dönem sonu medyanı ve kaç akranın katıldığı."""
    cols = {}
    for t, s in pe_map.items():
        if s is None or s.empty:
            continue
        s = s[(s > 0) & (s < 400)]                 # anlamsız uçları dışarıda
        cols[t] = s.resample(freq).last()
    if not cols:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    df = pd.DataFrame(cols)
    n = df.notna().sum(axis=1)
    med = df.median(axis=1).where(n >= min_n)
    return med.dropna(), n


# --------------------------------------------------------------------------
# 2) Fiyat vs medyan çarpan adil fiyatı
# --------------------------------------------------------------------------
def _growth(est: pd.DataFrame, row: str) -> float:
    if est is None or not isinstance(est, pd.DataFrame) or est.empty:
        return np.nan
    if row in est.index and "growth" in est.columns:
        g = pd.to_numeric(est.loc[row, "growth"], errors="coerce")
        return float(g) if np.isfinite(g) else np.nan
    return np.nan


def estimate_points(raw: dict[str, Any], key: str, last_ev: pd.Series
                    ) -> tuple[pd.Series, str]:
    """
    Gelecek 2 mali yıl için metrik tahmini. Satış → satış tahmini büyümesi;
    HBK → HBK tahmini; nakit akışları → HBK büyümesiyle ölçeklenir
    (yfinance nakit akışı tahmini vermez). Döner: (tarih → değer, açıklama).
    """
    a = _row(raw.get("a_cf" if METRICS[key]["stmt"] == "cf" else "a_inc"),
             METRICS[key]["row"])
    if a.empty or last_ev.empty:
        return pd.Series(dtype=float), ""
    fy_end = a.index.max()
    est = raw.get("rev_est") if key == "REV" else raw.get("eps_est")
    g0, g1 = _growth(est, "0y"), _growth(est, "+1y")
    if not np.isfinite(g0):
        return pd.Series(dtype=float), ""
    base = float(last_ev[last_ev.index <= fy_end].iloc[-1]) if (
        last_ev.index <= fy_end).any() else float(last_ev.iloc[-1])
    if base <= 0:
        return pd.Series(dtype=float), ""
    pts = {fy_end + pd.DateOffset(years=1): base * (1 + g0)}
    if np.isfinite(g1):
        pts[fy_end + pd.DateOffset(years=2)] = base * (1 + g0) * (1 + g1)
    kaynak = {"REV": "analist satış tahmini",
              "EPS": "analist HBK tahmini"}.get(
        key, "analist HBK büyümesiyle ölçeklendi (nakit akışı tahmini yok)")
    return pd.Series(pts).sort_index(), kaynak


def fair_value_chart_data(raw: dict[str, Any], key: str, years: int = 0
                          ) -> dict[str, Any]:
    """
    Döner: price, fair (günlük adil fiyat), points (dönem sonu noktaları),
    est (tahmin noktaları), median multiple, korelasyon, getiri, CAGR.
    `years`: medyan çarpanın hesaplandığı pencere (0 = mevcut tüm geçmiş).
    """
    p = _price(raw)
    ev = per_share_ttm(raw, key)
    if p.empty or ev.empty:
        return {"ok": False, "neden": f"{METRICS[key]['ad']} verisi bulunamadı"}
    ev = ev[ev > 0]
    if len(ev) < 2:
        return {"ok": False, "neden": f"{METRICS[key]['ad']} negatif ya da "
                                      f"yetersiz — çarpan anlamlı değil"}
    p = p[p.index >= ev.index.min()]
    daily = step_daily(ev, p.index)
    mult = (p / daily).dropna()
    if years:
        mult = mult[mult.index >= mult.index.max() - pd.DateOffset(years=years)]
    med = float(mult.median())
    fair = daily * med
    est, kaynak = estimate_points(raw, key, ev)
    corr = float(p.corr(fair)) if len(p) > 30 else np.nan
    total = (p.iloc[-1] / p.iloc[0] - 1) * 100
    yrs = (p.index[-1] - p.index[0]).days / 365.25
    cagr = ((p.iloc[-1] / p.iloc[0]) ** (1 / yrs) - 1) * 100 if yrs > 0.5 else np.nan
    fair_now = float(fair.iloc[-1])
    return {"ok": True, "price": p, "fair": fair, "points": ev * med,
            "est": est * med, "est_src": kaynak, "median": med,
            "mult_now": float(mult.iloc[-1]), "fair_now": fair_now,
            "gap": (p.iloc[-1] / fair_now - 1) * 100 if fair_now else np.nan,
            "corr": corr, "total": total, "cagr": cagr,
            "start": p.index[0], "metric": METRICS[key]}


def multiples_summary(raw: dict[str, Any], peers_pe: pd.Series | None = None
                      ) -> pd.DataFrame:
    """Güncel çarpan, kendi medyanı ve farkı — her metrik için."""
    p = _price(raw)
    rows = []
    for k, m in METRICS.items():
        ev = per_share_ttm(raw, k)
        ev = ev[ev > 0] if not ev.empty else ev
        if p.empty or len(ev) < 2:
            continue
        pp = p[p.index >= ev.index.min()]
        mult = (pp / step_daily(ev, pp.index)).dropna()
        if mult.empty:
            continue
        now, med = float(mult.iloc[-1]), float(mult.median())
        rows.append({"Çarpan": m["kisa"], "Şu an": now, "Kendi medyanı": med,
                     "Medyana göre %": (now / med - 1) * 100,
                     "En düşük": float(mult.min()), "En yüksek": float(mult.max()),
                     "Geçmiş": f"{mult.index[0]:%m.%Y} →"})
    out = pd.DataFrame(rows)
    if peers_pe is not None and not peers_pe.empty and not out.empty:
        out.loc[out["Çarpan"] == "F/K", "Akran medyanı"] = float(peers_pe.iloc[-1])
    return out


def suggest_peers(t: str, n: int = 8) -> list[str]:
    """Aynı ETF'lerdeki en ağır bileşenler — varsayılan akran listesi."""
    out: list[str] = []
    for e in uni.etfs_containing(t):
        for h in uni.holdings(e):
            if h != t and h not in out and "." not in h:
                out.append(h)
        if len(out) >= n * 2:
            break
    return out[:n]
