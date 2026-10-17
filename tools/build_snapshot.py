"""
AKŞAM HESABI — bütün ağır hesapları bir kez yapar, sonuçları
data/snapshot/ klasörüne yazar. Arayüz (davinci_terminal.py) önce buradan okur.

GitHub Actions her iş günü ABD kapanışından sonra çalıştırır. Kendi
bilgisayarınızda da çalıştırabilirsiniz:

    python tools/build_snapshot.py            # hepsi
    python tools/build_snapshot.py karar plan  # yalnızca seçilenler

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

    if want("akis") or want("karar"):
        def f():
            t, failed = cmp.flows()
            return (t, failed), f"{len(t.get('1 ay', []))} varlık × {len(t)} dönem"
        step("flows", f)

        def f():
            m, failed = cmp.markets()
            return (m, failed), f"{len(m)} varlık · çekilemeyen: {len(failed)}"
        step("markets", f)

        def f():
            g, failed = cmp.gold()
            return (g, failed), f"{len(g[0])} gösterge"
        step("gold", f)

    if want("tema"):
        def f():
            r = cmp.theme_rotation()
            return r, f"{len(r[0])} tema, {len(r[2])} ETF"
        step("theme_rotation", f)

        for hz in ("kısa", "uzun"):
            def f(hz=hz):
                r = cmp.theme_rotation(hz)
                return r, f"{hz} vade · {len(r[0])} tema"
            step("theme_rotation_" + {"kısa": "kisa", "uzun": "uzun"}[hz], f)

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

    if want("plan") or want("tarama"):
        def f():
            pu = cmp.plan_universe()
            p = cmp.plans(pu)
            ok = int(p["ok"].fillna(False).sum()) if "ok" in p else 0
            return p, f"{ok}/{len(pu)} hisse"
        step("plans", f)

    if want("orgu") or want("plan") or want("tarama"):
        def f():
            T = cmp.orgu()
            n_k = int(T["🐈 Ölü kedi"].sum()) if len(T) else 0
            return T, (f"{len(T)} hisse · şablon {'var' if T.attrs.get('sablon') else 'YOK'} · "
                       f"ölü kedi {n_k}")
        step("orgu", f)

    if want("hype"):
        def f():
            h = cmp.hype()
            if h["social"].empty and h["fund"].empty:
                raise RuntimeError("; ".join(h["errs"]) or "veri yok")
            return h, (f"{len(h['social'])} sosyal sembol · {len(h['fund'])} temel veri"
                       + (f" · hata: {'; '.join(h['errs'])}" if h["errs"] else ""))
        step("hype", f)

        def f():
            h = snap.load("hype", max_age_h=6) or {}
            syms = sorted(set(h.get("top", [])) | set(h.get("bio", [])))
            p = cmp.plans(syms)
            return p, f"{len(syms)} hype/biotech hissesi"
        step("plans_hype", f)

    if want("bilanco"):
        def f():
            e = dta.fetch_earnings_calendar(stocks)
            ok = int(e["Fiyat"].notna().sum()) if "Fiyat" in e else 0
            if ok < len(stocks) * 0.5:
                raise RuntimeError(f"Yahoo sınırladı: {ok}/{len(stocks)} hisse")
            return e, f"{ok}/{len(stocks)} hisse"
        step("earnings", f)

    if want("genislik") or want("karar"):
        def f():
            G = cmp.genislik(float(os.environ.get("GENISLIK_BUDGET_S", "600")))
            note = " · ".join(f"{k}: {v['closes'].shape[1]} hisse ({v['kaynak']}), F/K "
                              f"{int(v['fund']['İleri F/K'].notna().sum())}"
                              for k, v in G.items() if not k.startswith("_"))
            gg = G.get("_genis") or {}
            if gg:
                note += (f" · geniş piyasa {gg['closes'].shape[1]} hisse"
                         f" (Russell 2000: {gg.get('kaynak') or 'alınamadı'})")
            return G, note
        step("genislik", f)

    if want("vol") or want("makro") or want("karar"):
        def f():
            V = cmp.vol()
            em, ts, te = V["em"], V["ts"], V["test"]
            return V, (f"VIX {em.get('vix', float('nan')):.1f} · fark {ts.get('spread', float('nan')):+.2f}"
                       f" ({ts.get('etiket', '—')}) · olay testi "
                       f"{te.get('donem', te.get('neden', '—'))}")
        step("vol", f)

    if want("secki") or want("plan") or want("tarama"):
        def f():
            o = snap.load("orgu", max_age_h=30)
            S = cmp.picks(cmp.plan_universe(), budget_s=float(os.environ.get(
                "SECKI_BUDGET_S", "900")), orgu=o)
            n_f = int(S["Fair value"].notna().sum()) if len(S) else 0
            n_k = int((S["_tm"] == "kapida").sum()) if len(S) else 0
            return S, f"{len(S)} hisse · analist hedefi {n_f} · Fib kapısında {n_k}"
        S_secki = step("secki", f)

        def f():
            S = S_secki if S_secki is not None else snap.load("secki", max_age_h=30)
            rot = {}
            for v, nm in (("kısa", "theme_rotation_kisa"), ("orta", "theme_rotation"),
                          ("uzun", "theme_rotation_uzun")):
                r = snap.load(nm, max_age_h=30)
                rot[v] = r[0] if r is not None else None
            new = cmp.alarm_defteri(S, rot)
            E = cmp.defter_degerlendir()
            n_k = int(E["Seviye"].isin(["TP1", "TP2", "TP3", "SL"]).sum()) if len(E) else 0
            return E, f"bugün {len(new)} alarm · defterde {len(E)} · TP/SL gören {n_k}"
        step("alarm_defteri", f)

    st = snap.read_status()
    bad = [k for k, v in st.items() if not v.get("ok")]
    print("\nBaşarısız adımlar:", ", ".join(bad) if bad else "yok")
    return 0


if __name__ == "__main__":
    sys.exit(main([a.lower() for a in sys.argv[1:]]))
