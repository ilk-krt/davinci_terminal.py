"""
AKŞAM HESABI — bütün ağır hesapları bir kez yapar, sonuçları
data/snapshot/ klasörüne yazar. Arayüz (davinci_terminal.py) önce buradan okur.

GitHub Actions her iş günü ABD kapanışından sonra çalıştırır. Kendi
bilgisayarınızda da çalıştırabilirsiniz:

    python tools/build_snapshot.py            # hepsi
    python tools/build_snapshot.py karar tema # yalnızca seçilenler

Her adım bağımsızdır: biri başarısız olursa diğerleri yine yazılır; durum
data/snapshot/_status.json dosyasına kaydedilir (arayüzde görünür).
"""
from __future__ import annotations

import logging
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from apex import compute as cmp            # noqa: E402
from apex import data as dta               # noqa: E402
from apex import snapshot as snap          # noqa: E402
from apex import universe as uni           # noqa: E402

logging.basicConfig(level=logging.WARNING)


def step(name: str, fn, *, need=None):
    """Bir adımı çalıştırır, sonucu kaydeder, süreyi ve hatayı raporlar."""
    t0 = time.time()
    try:
        obj, note = fn()
        size = snap.save(name, obj)
        secs = time.time() - t0
        snap.write_status(name, True, secs, note)
        print(f"✅ {name:<16} {secs:6.1f} sn  {size / 1024:8.0f} KB  {note}")
        return obj
    except Exception as exc:
        secs = time.time() - t0
        snap.write_status(name, False, secs, f"{type(exc).__name__}: {exc}")
        print(f"❌ {name:<16} {secs:6.1f} sn  {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return None


def main(only: list[str]) -> int:
    want = (lambda k: not only or k in only)
    results = {}

    if want("makro"):
        def f():
            fr = cmp.macro_frames()
            failed = fr.get("_failed")
            n_bad = 0 if failed is None or failed.empty else len(failed)
            return fr, f"{len(fr) - 1} seri, {n_bad} çekilemedi"
        results["makro"] = step("macro_frames", f)

    rot = None
    if want("rotasyon") or want("karar"):
        def f():
            r = cmp.rotation()
            return r, f"{len(r[0])} gösterge, çekilemeyen: {len(r[2])}"
        res = step("rotation", f)
        rot = res[0] if res else None

    if want("karar"):
        def f():
            d = cmp.decision(rot)
            return d, " · ".join(f"{a}: {len(v['mods'])} zaman dilimi"
                                 for a, v in d["assets"].items())
        step("decision", f)

    if want("tema"):
        def f():
            r = cmp.theme_rotation()
            return r, f"{len(r[0])} tema, {len(r[2])} ETF"
        step("theme_rotation", f)

        def f():
            p = cmp.theme_performance()
            return p, f"{len(p)} tema"
        step("theme_perf", f)

    universe = cmp.snapshot_universe()
    stocks = cmp.earnings_universe()
    if want("tarama"):
        def f():
            s = cmp.scan(tuple(universe), "1d")
            return s, f"{len(s)} sembol"
        step("scan_1d", f)

        def f():
            s = cmp.scan(tuple(universe), "1wk")
            return s, f"{len(s)} sembol"
        step("scan_1wk", f)

    if want("bilanco"):
        def f():
            e = dta.fetch_earnings_calendar(stocks)
            ok = int(e["Fiyat"].notna().sum()) if "Fiyat" in e else 0
            if ok < len(stocks) * 0.5:
                raise RuntimeError(f"Yahoo sınırladı: {ok}/{len(stocks)} hisse")
            return e, f"{ok}/{len(stocks)} hisse"
        step("earnings", f)

    st = snap.read_status()
    bad = [k for k, v in st.items() if not v.get("ok")]
    print("\nBaşarısız adımlar:", ", ".join(bad) if bad else "yok")
    return 0


if __name__ == "__main__":
    sys.exit(main([a.lower() for a in sys.argv[1:]]))
