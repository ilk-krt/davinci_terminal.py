# apex/shortvol.py — AETHER APEX
from __future__ import annotations

from pathlib import Path
from apex.data import yf_info

# apex/shortvol.py — FINRA günlük short hacmi (Reg SHO) ve haftalık değişim
#
# Kaynak: FINRA'nın ücretsiz yayımladığı konsolide günlük dosya
#   https://cdn.finra.org/equity/regsho/daily/CNMSshvol{YYYYMMDD}.txt
#   Biçim: Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market
#   Her iş günü ABD saatiyle ~18:00'den sonra yayımlanır.
#
# ÖNEMLİ — okuma şekli:
#   * Bu "short HACMİ"dir, "short INTEREST" (açık pozisyon) değildir.
#     Piyasa yapıcıların alıcıya hisse sağlamak için yaptığı gün içi açığa
#     satışlar da buraya girer; bu yüzden çoğu hissede oran zaten %35–55
#     arasındadır. Anlamlı olan SEVİYE değil, hissenin KENDİ ortalamasına
#     göre DEĞİŞİMDİR.
#   * Hacim yalnızca FINRA'ya raporlanan (TRF/ADF) işlemleri kapsar,
#     borsa içi (lit) işlemleri tamamen kapsamaz. "TotalVolume" bu yüzden
#     yfinance'teki toplam hacimden düşüktür — oran kendi içinde tutarlıdır.
#   * OTC hisseler (FANUY, YASKY vb.) ve kripto CNMS dosyasında yoktur.


import datetime as dt
import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

import numpy as np
import pandas as pd
import requests

_log = logging.getLogger(__name__)

FINRA_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{d}.txt"
WINDOW = 5          # "bu hafta" = son 5 işlem günü
Z_LEN = 20          # z-skoru için taban pencere
ESIK_PP = 3.0       # anlamlı değişim eşiği (yüzde puan)


def _us_today() -> dt.date:
    try:
        from zoneinfo import ZoneInfo
        return dt.datetime.now(ZoneInfo("America/New_York")).date()
    except Exception:                                   # pragma: no cover
        return dt.date.today()


def _weekdays_back(n_days: int, end: dt.date | None = None) -> list[dt.date]:
    end = end or _us_today()
    out, d = [], end
    while len(out) < n_days:
        if d.weekday() < 5:
            out.append(d)
        d -= dt.timedelta(days=1)
    return out


_FINRA_HEADERS = {
    # Akamai CDN, kısa/robot görünümlü isteklere 403 dönebiliyor —
    # tam bir tarayıcı başlığı gönderiyoruz.
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/126.0 Safari/537.36"),
    "Accept": "text/plain,text/html,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data",
}


def _fetch_day(d: dt.date, timeout: int = 20) -> tuple[pd.DataFrame | None, str]:
    """Döner: (tablo ya da None, tanı metni: 'ok', 'HTTP 404', hata adı…)."""
    url = FINRA_URL.format(d=d.strftime("%Y%m%d"))
    last = ""
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=timeout, headers=_FINRA_HEADERS)
            if r.status_code == 404:
                return None, "HTTP 404 (tatil / henüz yok)"
            if r.status_code != 200 or not r.content:
                last = f"HTTP {r.status_code}"
                time.sleep(1.0 + attempt)
                continue
            df = pd.read_csv(io.BytesIO(r.content), sep="|",
                             dtype={"Symbol": str})
            df["Date"] = pd.to_numeric(df["Date"], errors="coerce")
            df = df[df["Date"].notna()].copy()           # alt bilgi satırı
            df["Date"] = pd.to_datetime(df["Date"].astype("int64").astype(str),
                                        format="%Y%m%d")
            for c in ("ShortVolume", "TotalVolume"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            return df[["Date", "Symbol", "ShortVolume", "TotalVolume"]], "ok"
        except Exception as exc:
            last = type(exc).__name__
            time.sleep(1.0 + attempt)
    return None, last or "bilinmeyen hata"


def _repo_file() -> "Path | None":
    """GitHub Actions'ın repoya yazdığı FINRA dosyası (data/finra_short.csv.gz)."""
    from apex.snapshot import data_file
    return data_file("finra_short.csv.gz")


def _read_repo_file() -> pd.DataFrame | None:
    p = _repo_file()
    if p is None:
        return None
    try:
        df = pd.read_csv(p, dtype={"Symbol": str})
        df["Date"] = pd.to_datetime(df["Date"].astype("int64").astype(str),
                                    format="%Y%m%d")
        for c in ("ShortVolume", "TotalVolume"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df[["Date", "Symbol", "ShortVolume", "TotalVolume"]]
    except Exception as exc:
        _log.info("Repo FINRA dosyası okunamadı: %s", exc)
        return None


def fetch_short_volume(trading_days: int = 30
                       ) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """
    Önce repodaki data/finra_short.csv.gz (GitHub Actions her akşam günceller),
    sonra yalnızca o dosyada olmayan son günler FINRA'dan doğrudan denenir.
    Döner: (uzun tablo, uyarılar, {tarih: tanı}).
    """
    days = _weekdays_back(trading_days + 4)  # tatil payı
    repo = _read_repo_file()
    diag: dict[str, str] = {}
    frames: list[pd.DataFrame] = []
    have: set = set()
    if repo is not None and not repo.empty:
        frames.append(repo)
        have = {d.date() for d in repo["Date"].unique()}
        for d in sorted(have, reverse=True):
            diag[d.strftime("%d.%m.%Y")] = "ok (repo dosyası)"
        # repodaki son günden sonrası doğrudan denenir
        last = max(have)
        days = [d for d in days if d > last]

    if days:
        with ThreadPoolExecutor(max_workers=4) as ex:
            res = list(ex.map(_fetch_day, days))
        for d, (f, msg) in zip(days, res):
            diag[d.strftime("%d.%m.%Y")] = msg
            if f is not None and not f.empty:
                frames.append(f)
    diag = dict(sorted(diag.items(),
                       key=lambda kv: dt.datetime.strptime(kv[0], "%d.%m.%Y"),
                       reverse=True))

    warns: list[str] = []
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        kodlar = sorted({m for m in diag.values() if not m.startswith("HTTP 404")})
        warns.append("FINRA short hacim verisine ulaşılamadı"
                     + (f" ({', '.join(kodlar)})" if kodlar else "")
                     + ". Repoda data/finra_short.csv.gz yok — GitHub Actions "
                       "iş akışını bir kez çalıştırın.")
        return (pd.DataFrame(columns=["Date", "Symbol", "ShortVolume",
                                      "TotalVolume"]), warns, diag)
    df = (pd.concat(frames, ignore_index=True)
            .drop_duplicates(["Date", "Symbol"], keep="last"))
    last = df["Date"].max()
    if (pd.Timestamp(_us_today()) - last).days > 4:
        warns.append(f"En son FINRA verisi {last:%d.%m.%Y} tarihli.")
    n_days = df["Date"].nunique()
    if n_days < 2 * WINDOW:
        warns.append(f"Yalnızca {n_days} günlük FINRA verisi var; "
                     "haftalık karşılaştırma eksik kalabilir.")
    return df, warns, diag


def _one_short_interest(t: str) -> dict:
    """yfinance üzerinden borsa short INTEREST verisi (ayda iki kez güncellenir)."""
    def num(x):
        try:
            v = float(x)
            return v if np.isfinite(v) else np.nan
        except (TypeError, ValueError):
            return np.nan

    rec = {"Sembol": t, "SI Float %": np.nan, "SI Δ% (ay)": np.nan,
           "Gün Kapama": np.nan, "SI Tarihi": ""}
    try:
        info = yf_info(t)
        cur, prev = num(info.get("sharesShort")), num(info.get("sharesShortPriorMonth"))
        pf = num(info.get("shortPercentOfFloat"))
        rec["SI Float %"] = pf * 100 if np.isfinite(pf) else np.nan
        if np.isfinite(cur) and np.isfinite(prev) and prev > 0:
            rec["SI Δ% (ay)"] = (cur / prev - 1) * 100
        rec["Gün Kapama"] = num(info.get("shortRatio"))
        ts = info.get("dateShortInterest")
        if isinstance(ts, (int, float)) and ts > 0:
            rec["SI Tarihi"] = dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%d.%m")
    except Exception as exc:
        _log.info("Short interest alınamadı (%s): %s", t, exc)
    return rec


def fetch_short_interest(tickers: Iterable[str]) -> pd.DataFrame:
    syms = list(dict.fromkeys(t.upper() for t in tickers if t))
    if not syms:
        return pd.DataFrame(columns=["Sembol"])
    with ThreadPoolExecutor(max_workers=2) as ex:
        return pd.DataFrame(list(ex.map(_one_short_interest, syms)))


def to_finra_symbol(t: str) -> str:
    """yfinance → FINRA: BRK-B → BRK/B (FINRA sınıf ayracı olarak '/' kullanır)"""
    return t.replace("-", "/").replace(".", "/").upper()


def _arrow(d_pp: float) -> str:
    if not np.isfinite(d_pp):
        return "—"
    if d_pp >= ESIK_PP:
        return "⬆️ artıyor"
    if d_pp <= -ESIK_PP:
        return "⬇️ azalıyor"
    return "→ yatay"


def interpret(d_pp: float, price_chg: float, z: float) -> str:
    """Short hacim değişimi + fiyat yönü → tek cümlelik okuma."""
    if not np.isfinite(d_pp):
        return "Veri yok"
    up, dn = d_pp >= ESIK_PP, d_pp <= -ESIK_PP
    p_up = np.isfinite(price_chg) and price_chg > 0
    p_dn = np.isfinite(price_chg) and price_chg < 0
    uc = " (aşırı uç)" if np.isfinite(z) and abs(z) >= 2 else ""
    if up and p_dn:
        return "🔴 Short baskısı artıyor, fiyat düşüyor" + uc
    if up and p_up:
        return "🟠 Yükselişe karşı short artıyor — ya dağıtım ya squeeze yakıtı" + uc
    if dn and p_up:
        return "🟢 Short çekiliyor, fiyat yükseliyor — kapama desteği" + uc
    if dn and p_dn:
        return "🟡 Short azalıyor ama fiyat düşüyor — satış uzun taraftan" + uc
    if up:
        return "🟠 Short hacmi artıyor" + uc
    if dn:
        return "🟢 Short hacmi azalıyor" + uc
    return "⚪ Belirgin değişim yok"


def short_table(raw: pd.DataFrame, tickers: Iterable[str],
                price_chg: dict[str, float] | None = None,
                spark_n: int = 15) -> pd.DataFrame:
    """
    Her sembol için: son 5 gün short oranı, önceki 5 gün, fark (pp),
    short hacmindeki % değişim, 20 günlük z-skoru, seyir listesi, yorum.
    """
    price_chg = price_chg or {}
    tickers = list(dict.fromkeys(t.upper() for t in tickers))
    cols = ["Sembol", "SV% 5G", "SV% Önceki 5G", "ΔSV pp", "Short Hacim Δ%",
            "SV% Z", "SV Yön", "SV Seyri", "Short Yorum", "SV Son Gün"]
    if raw is None or raw.empty:
        return pd.DataFrame(columns=cols)

    fmap = {to_finra_symbol(t): t for t in tickers}
    sub = raw[raw["Symbol"].isin(fmap)].copy()
    sub = (sub.groupby(["Symbol", "Date"], as_index=False)
              [["ShortVolume", "TotalVolume"]].sum())
    sub["ratio"] = sub["ShortVolume"] / sub["TotalVolume"].replace(0, np.nan) * 100

    rows = []
    for fsym, g in sub.groupby("Symbol"):
        g = g.sort_values("Date")
        t = fmap[fsym]
        last5, prev5 = g.tail(WINDOW), g.iloc[-2 * WINDOW:-WINDOW]
        # hacim ağırlıklı oran (günlük oranların ortalaması yerine)
        r5 = last5["ShortVolume"].sum() / max(last5["TotalVolume"].sum(), 1) * 100
        rp = (prev5["ShortVolume"].sum() / max(prev5["TotalVolume"].sum(), 1) * 100
              if len(prev5) else np.nan)
        sv5, svp = last5["ShortVolume"].sum(), prev5["ShortVolume"].sum()
        base = g["ratio"].iloc[:-1].tail(Z_LEN)
        z = ((g["ratio"].iloc[-1] - base.mean()) / base.std(ddof=0)
             if len(base) >= 10 and base.std(ddof=0) > 0 else np.nan)
        d_pp = r5 - rp if np.isfinite(rp) else np.nan
        rows.append({
            "Sembol": t,
            "SV% 5G": r5,
            "SV% Önceki 5G": rp,
            "ΔSV pp": d_pp,
            "Short Hacim Δ%": (sv5 / svp - 1) * 100 if svp else np.nan,
            "SV% Z": z,
            "SV Yön": _arrow(d_pp),
            "SV Seyri": [round(float(x), 1) for x in g["ratio"].tail(spark_n)],
            "Short Yorum": interpret(d_pp, price_chg.get(t, np.nan), z),
            "SV Son Gün": g["Date"].iloc[-1].strftime("%d.%m"),
        })
    return pd.DataFrame(rows, columns=cols)
