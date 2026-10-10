"""
Karar Hunisi için Streamlit sunucusunun güvenilir çekemediği iki veri setini
üretir ve repoya yazar (GitHub Actions her iş günü çalıştırır):

1) data/crypto_caps.csv — TradingView kripto endekslerinin yaklaşık karşılığı
   TOTAL, TOTAL2, TOTAL3, OTHERS, BTC.D, ETH/BTC, OTHERS/BTC
   Yöntem: CoinGecko'dan ilk ~150 coinin dolaşımdaki arzı (bugün) ×
   yfinance günlük kapanış fiyatları. Arz geçmişte bugünkünden biraz farklı
   olduğu için seviyeler TradingView'dan birkaç puan sapabilir; YÖN ve
   kırılımlar uyumludur. Son satır CoinGecko'nun gerçek toplamıyla
   ölçeklenir.

2) data/liquidity.csv — Fed net likiditesi ve para arzı (FRED)
   WALCL (Fed bilançosu), WTREGEN (Hazine hesabı, TGA), RRPONTSYD (ters
   repo), M2SL (M2 para arzı).  Net likidite = WALCL − TGA − RRP.

Elle çalıştırma:  python tools/macro_update.py
"""
from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent / "data"
UA = {"User-Agent": "Mozilla/5.0 (davinci_terminal data job)"}
# FRED kısa/robot görünümlü istekleri reddedebiliyor — tarayıcı başlığı
BROWSER = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/126.0 Safari/537.36"),
           "Accept": "text/csv,text/plain,*/*;q=0.8",
           "Accept-Language": "en-US,en;q=0.9",
           "Referer": "https://fred.stlouisfed.org/"}
ERRORS: dict[str, str] = {}


def _status(name: str, ok: bool, note: str) -> None:
    """Sonucu arayüzde görünen durum dosyasına yaz (data/snapshot/_status.json)."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from apex.snapshot import write_status
        write_status(name, ok, 0.0, note)
    except Exception as exc:                      # durum yazılamazsa iş durmasın
        print(f"durum yazılamadı: {exc}", file=sys.stderr)

# Sabit 1$ kabul edilen stablecoin'ler (yfinance fiyatı gürültülü olabiliyor)
STABLES = {"usdt", "usdc", "dai", "usde", "fdusd", "tusd", "usdd", "pyusd",
           "usds", "usd1", "usdtb", "buidl", "usdy", "rlusd", "susde", "gusd",
           "frax", "lusd", "eurc", "usdf", "bfusd", "usdg", "usdx", "ousg"}
# Sarmalanmış / türev tokenlar çift sayım yapar — dışarıda
WRAPPED = {"wbtc", "weth", "steth", "wsteth", "weeth", "cbbtc", "reth",
           "wbeth", "meth", "ezeth", "lbtc", "solvbtc", "jitosol", "bnsol",
           "msol", "rseth", "teth", "clbtc", "wbnb", "sbtc", "tbtc", "lseth",
           "oseth", "ethx", "pufeth", "rsweth", "susds", "sdai", "stkaave"}


def get_json(url: str, tries: int = 4):
    for i in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=30)
            if r.status_code == 200:
                return r.json()
            print(f"{url}: HTTP {r.status_code}", file=sys.stderr)
        except Exception as exc:
            print(f"{url}: {exc}", file=sys.stderr)
        time.sleep(5 * (i + 1))
    return None


# --------------------------------------------------------------------------
# 1) Kripto
# --------------------------------------------------------------------------
def crypto() -> bool:
    import yfinance as yf

    coins = []
    for page in (1, 2):
        js = get_json("https://api.coingecko.com/api/v3/coins/markets"
                      f"?vs_currency=usd&order=market_cap_desc&per_page=100&page={page}")
        if js:
            coins += js
        time.sleep(3)
    glob = get_json("https://api.coingecko.com/api/v3/global")
    if not coins:
        print("CoinGecko verisi alınamadı", file=sys.stderr)
        return False

    meta = []
    for c in coins:
        sym = (c.get("symbol") or "").lower()
        if sym in WRAPPED or not c.get("circulating_supply"):
            continue
        meta.append({"sym": sym, "rank": c.get("market_cap_rank") or 999,
                     "supply": float(c["circulating_supply"]),
                     "price": float(c.get("current_price") or 0),
                     "stable": sym in STABLES})
    meta = (pd.DataFrame(meta).sort_values("rank")
              .drop_duplicates("sym").head(150).reset_index(drop=True))

    vol = meta[~meta["stable"]]
    tickers = [f"{s.upper()}-USD" for s in vol["sym"]]
    raw = yf.download(" ".join(tickers), period="3y", interval="1d",
                      auto_adjust=False, progress=False, threads=True,
                      group_by="column")
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]

    caps = {}
    for _, m in vol.iterrows():
        t = f"{m['sym'].upper()}-USD"
        if t not in close.columns:
            continue
        s = close[t].dropna()
        if len(s) < 60 or m["price"] <= 0:
            continue
        # yfinance yanlış coini eşleştirdiyse (aynı sembol) fiyat tutmaz
        if not (0.6 < s.iloc[-1] / m["price"] < 1.6):
            print(f"  atlandı (fiyat uyuşmuyor): {t}", file=sys.stderr)
            continue
        caps[m["sym"]] = s * m["supply"]
    if "btc" not in caps or "eth" not in caps:
        print("BTC/ETH fiyatı yok", file=sys.stderr)
        return False

    cap = pd.DataFrame(caps).sort_index().ffill()
    cap = cap[cap.index >= cap["btc"].first_valid_index()]
    stable_cap = float((meta[meta["stable"]]["supply"]).sum())  # ≈ 1$ × arz

    # sıralama bugünkü piyasa değerine göre (TradingView OTHERS = ilk 10 hariç)
    top10 = list(meta.sort_values("rank")["sym"].head(10))
    top10_vol = [s for s in top10 if s in cap.columns]
    top10_stable = sum(float(meta.loc[meta["sym"] == s, "supply"].iloc[0])
                       for s in top10 if s in STABLES)

    total = cap.sum(axis=1, min_count=1) + stable_cap
    out = pd.DataFrame(index=cap.index)
    out["TOTAL"] = total
    out["BTC"] = cap["btc"]
    out["ETH"] = cap["eth"]
    out["TOTAL2"] = total - cap["btc"]
    out["TOTAL3"] = total - cap["btc"] - cap["eth"]
    out["OTHERS"] = total - cap[top10_vol].sum(axis=1) - top10_stable
    out["STABLE"] = stable_cap

    # Seviye kalibrasyonu: CoinGecko'nun gerçek toplamı ile vekil toplam
    # arasındaki fark, kapsanmayan küçük coinlerdir → yalnızca OTHERS'a aittir.
    # OTHERS bu oranda büyütülür, fark TOTAL/TOTAL2/TOTAL3'e de eklenir.
    if glob and "data" in glob:
        real_total = float(glob["data"]["total_market_cap"].get("usd") or 0)
        gap = real_total - float(out["TOTAL"].iloc[-1])
        last_oth = float(out["OTHERS"].iloc[-1])
        if real_total > 0 and gap > 0 and last_oth > 0:
            k = (last_oth + gap) / last_oth
            extra = out["OTHERS"] * (k - 1)
            for c in ("TOTAL", "TOTAL2", "TOTAL3", "OTHERS"):
                out[c] = out[c] + extra
            print(f"OTHERS kalibrasyon katsayısı {k:.3f} "
                  f"(kapsanmayan ≈ {gap / 1e9:,.0f} milyar $)")
    out["BTC.D"] = out["BTC"] / out["TOTAL"] * 100
    out["ETH/BTC"] = out["ETH"] / out["BTC"]
    out["OTHERS/BTC"] = out["OTHERS"] / out["BTC"]
    out.index.name = "Date"
    out = out.dropna(subset=["TOTAL"])
    ROOT.mkdir(parents=True, exist_ok=True)
    out.round(6).to_csv(ROOT / "crypto_caps.csv")
    print(f"crypto_caps.csv: {len(out)} gün, {len(caps)} coin, "
          f"BTC.D son {out['BTC.D'].iloc[-1]:.1f}")
    return True


# --------------------------------------------------------------------------
# 2) Likidite (FRED)
# --------------------------------------------------------------------------
FRED = {"WALCL": "Fed Bilançosu", "WTREGEN": "TGA", "RRPONTSYD": "Ters Repo",
        "M2SL": "M2"}


def _fred_api(sid: str) -> pd.Series | None:
    """Resmî FRED API'si — repo Secrets'ta FRED_API_KEY varsa (ücretsiz anahtar)."""
    key = os.environ.get("FRED_API_KEY", "").strip()
    if not key:
        return None
    url = ("https://api.stlouisfed.org/fred/series/observations"
           f"?series_id={sid}&api_key={key}&file_type=json"
           "&observation_start=2015-01-01")
    r = requests.get(url, headers=UA, timeout=30)
    if r.status_code != 200:
        ERRORS[sid] = f"API HTTP {r.status_code}"
        return None
    obs = r.json().get("observations", [])
    s = pd.Series({pd.Timestamp(o["date"]): pd.to_numeric(o["value"], errors="coerce")
                   for o in obs}, name=sid)
    return s.dropna()


def fred_series(sid: str) -> pd.Series | None:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd=2015-01-01"
    for i in range(3):
        try:
            r = requests.get(url, headers=BROWSER, timeout=30)
            if r.status_code == 200 and r.content and b"," in r.content[:200]:
                df = pd.read_csv(io.BytesIO(r.content), na_values=["."])
                dcol = df.columns[0]          # 'DATE' ya da 'observation_date'
                s = pd.Series(pd.to_numeric(df[sid], errors="coerce").values,
                              index=pd.to_datetime(df[dcol]), name=sid)
                return s.dropna()
            ERRORS[sid] = f"HTTP {r.status_code}"
            print(f"FRED {sid}: HTTP {r.status_code}", file=sys.stderr)
        except Exception as exc:
            ERRORS[sid] = type(exc).__name__
            print(f"FRED {sid}: {exc}", file=sys.stderr)
        time.sleep(4 * (i + 1))
    try:
        return _fred_api(sid)
    except Exception as exc:
        ERRORS[sid] = f"API {type(exc).__name__}"
        return None


def liquidity() -> bool:
    ser = {k: fred_series(k) for k in FRED}
    ok = {k: v for k, v in ser.items() if v is not None and len(v)}
    if "WALCL" not in ok:
        msg = ", ".join(f"{k}: {v}" for k, v in ERRORS.items()) or "bilinmiyor"
        print(f"FRED verisi alınamadı ({msg})", file=sys.stderr)
        _status("likidite (FRED)", False, msg
                + (" — repo Secrets'a FRED_API_KEY eklenirse resmî API denenir"
                   if not os.environ.get("FRED_API_KEY") else ""))
        return False
    idx = pd.date_range(min(s.index.min() for s in ok.values()),
                        max(s.index.max() for s in ok.values()), freq="D")
    df = pd.DataFrame({k: v.reindex(idx).ffill() for k, v in ok.items()})
    # milyar $ cinsine çevir: WALCL ve WTREGEN milyon $, RRP ve M2 milyar $
    df["WALCL"] = df["WALCL"] / 1000
    if "WTREGEN" in df:
        df["WTREGEN"] = df["WTREGEN"] / 1000
    df["NET_LIQ"] = (df["WALCL"] - df.get("WTREGEN", 0).fillna(0)
                     - df.get("RRPONTSYD", 0).fillna(0))
    df = df[df.index.dayofweek < 5]
    df.index.name = "Date"
    ROOT.mkdir(parents=True, exist_ok=True)
    df.round(2).to_csv(ROOT / "liquidity.csv")
    print(f"liquidity.csv: {len(df)} gün, son net likidite "
          f"{df['NET_LIQ'].iloc[-1]:,.0f} milyar $")
    _status("likidite (FRED)", True, f"{len(df)} gün, eksik seri: "
            + (", ".join(k for k in FRED if k not in ok) or "yok"))
    return True


def main() -> int:
    a = False
    b = False
    try:
        a = crypto()
        _status("kripto (CoinGecko)", a, "" if a else "CoinGecko/yfinance verisi alınamadı")
    except Exception as exc:
        print(f"kripto hatası: {exc}", file=sys.stderr)
        _status("kripto (CoinGecko)", False, f"{type(exc).__name__}: {exc}")
    try:
        b = liquidity()
    except Exception as exc:
        print(f"likidite hatası: {exc}", file=sys.stderr)
    # biri bile başarılıysa iş başarılı sayılır (diğeri eski dosyayla kalır)
    return 0 if (a or b) else 1


if __name__ == "__main__":
    sys.exit(main())
