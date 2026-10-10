"""
FINRA günlük short hacim dosyalarını indirip repoya tek bir sıkıştırılmış
CSV olarak yazar: data/finra_short.csv.gz

GitHub Actions tarafından her iş günü akşamı çalıştırılır. Streamlit
uygulaması FINRA'ya kendisi bağlanamadığında bu dosyayı okur.
Elle çalıştırma:  python tools/finra_update.py
"""
from __future__ import annotations

import datetime as dt
import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{d}.txt"
OUT = Path(__file__).resolve().parent.parent / "data" / "finra_short.csv.gz"
DAYS = 40            # tutulacak iş günü (tatil payıyla ~30 işlem günü)
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/126.0 Safari/537.36"),
    "Accept": "text/plain,*/*;q=0.8",
}


def weekdays_back(n: int) -> list[dt.date]:
    d, out = dt.date.today(), []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= dt.timedelta(days=1)
    return out


def fetch(d: dt.date) -> pd.DataFrame | None:
    for attempt in range(3):
        try:
            r = requests.get(URL.format(d=d.strftime("%Y%m%d")),
                             headers=HEADERS, timeout=30)
            if r.status_code == 404:
                return None
            if r.status_code == 200 and r.content:
                df = pd.read_csv(io.BytesIO(r.content), sep="|",
                                 dtype={"Symbol": str})
                df["Date"] = pd.to_numeric(df["Date"], errors="coerce")
                df = df[df["Date"].notna()]
                df["Date"] = df["Date"].astype("int64")
                for c in ("ShortVolume", "TotalVolume"):
                    df[c] = pd.to_numeric(df[c], errors="coerce").round().astype("Int64")
                return df[["Date", "Symbol", "ShortVolume", "TotalVolume"]]
            print(f"{d}: HTTP {r.status_code}", file=sys.stderr)
        except Exception as exc:
            print(f"{d}: {exc}", file=sys.stderr)
        time.sleep(2 + 2 * attempt)
    return None


def main() -> int:
    days = weekdays_back(DAYS)
    old = pd.DataFrame()
    if OUT.exists():
        old = pd.read_csv(OUT, dtype={"Symbol": str})
    have = set(old["Date"].unique()) if not old.empty else set()
    frames = [old] if not old.empty else []
    got = 0
    newest = max(have) if have else 0
    for d in days:
        k = int(d.strftime("%Y%m%d"))
        # dosya doluysa yalnızca en son günden sonrası indirilir
        if k in have or (len(have) >= 25 and k < newest):
            continue
        df = fetch(d)
        if df is not None:
            frames.append(df)
            got += 1
            print(f"{d}: {len(df)} satır")
    if not frames:
        print("Hiç veri alınamadı.", file=sys.stderr)
        return 1
    all_ = pd.concat(frames, ignore_index=True).drop_duplicates(["Date", "Symbol"])
    keep = sorted(all_["Date"].unique())[-30:]          # son 30 işlem günü
    all_ = all_[all_["Date"].isin(keep)].sort_values(["Date", "Symbol"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    all_.to_csv(OUT, index=False, compression="gzip")
    print(f"{got} yeni gün eklendi; toplam {len(keep)} gün, {len(all_)} satır → {OUT}")
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from apex.snapshot import write_status
        write_status("short hacim (FINRA)", True, 0.0,
                     f"{len(keep)} gün, son {max(keep)}")
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
