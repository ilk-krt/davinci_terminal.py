"""
ROBOT TOPLU TEST — robotu çok sayıda hissede, bütün profil × plan × yapı katmanı
(açık/kapalı) kombinasyonlarıyla geçmiş veride çalıştırır ve sonuçları CSV yazar.

GitHub Actions → "Robot toplu test" → Run workflow ile çalışır (elle). Kendi
bilgisayarınızda da:

    python tools/robot_batch.py                       # varsayılan 50 hisse, son 12 ay
    python tools/robot_batch.py --semboller NVDA,ORCL --ay 6

Çıktılar (data/robot_batch/):
    ozet.csv      her sembol × profil × plan × yapı için getiri, düşüş, işlem, kazanan %…
    islemler.csv  HER İŞLEM: giriş/çıkış, R, MFE/MAE, giriş nedeni, stop nedeni, hedef
                  etiketleri ve girişteki yapı bağlamı (S1/R1 uzaklığı, gap, Elliott,
                  haftalık trend, Mum Gücü, konsey skoru…)
    durum.json    tarih, dönem, semboller, çekilemeyenler, süre
    robot_toplu_test.zip  üçü bir arada (Claude'a vermek için)

Her sembol bitince dosyalar yeniden yazılır: iş yarıda kesilse bile o ana kadarki
sonuçlar kalır.
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

from apex import data as dta                # noqa: E402
from apex import robot as rbt               # noqa: E402

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

OUT = ROOT / "data" / "robot_batch"

# 50 farklı karakterde hisse/ETF: büyük teknoloji, yarı iletken, yazılım, enerji,
# nükleer/kamu hizmeti, finans, sağlık, biyoteknoloji, tüketim, sanayi, Çin, spekülatif
# büyüme, kripto bağlantılı, emtia/maden, endeks ETF'leri, savunmacı.
UNIVERSE: dict[str, str] = {
    "NVDA": "yarı iletken", "AMD": "yarı iletken", "AVGO": "yarı iletken", "MU": "yarı iletken",
    "INTC": "yarı iletken", "SMCI": "spekülatif donanım",
    "AAPL": "büyük teknoloji", "MSFT": "büyük teknoloji", "AMZN": "büyük teknoloji",
    "GOOGL": "büyük teknoloji", "META": "büyük teknoloji", "TSLA": "büyük teknoloji",
    "PLTR": "yazılım", "ORCL": "yazılım", "CRM": "yazılım", "NOW": "yazılım", "SNOW": "yazılım",
    "XOM": "enerji", "CVX": "enerji", "OXY": "enerji",
    "CEG": "nükleer/enerji", "VST": "nükleer/enerji", "BE": "nükleer/enerji",
    "JPM": "finans", "GS": "finans", "BAC": "finans",
    "LLY": "sağlık", "UNH": "sağlık", "PFE": "sağlık",
    "VRTX": "biyoteknoloji", "MRNA": "biyoteknoloji", "CRSP": "biyoteknoloji",
    "WMT": "tüketim", "COST": "tüketim", "NKE": "tüketim", "MCD": "tüketim",
    "CAT": "sanayi", "BA": "sanayi", "GE": "sanayi",
    "BABA": "Çin", "PDD": "Çin",
    "ASTS": "spekülatif büyüme", "RKLB": "spekülatif büyüme", "IONQ": "spekülatif büyüme",
    "COIN": "kripto bağlantılı", "MSTR": "kripto bağlantılı",
    "FCX": "emtia/maden", "NEM": "emtia/maden",
    "SPY": "endeks ETF", "QQQ": "endeks ETF",
}


def _args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--semboller", default="", help="virgülle ayrılmış; boşsa varsayılan 50")
    ap.add_argument("--ay", type=int, default=12, help="geriye kaç ay test edilsin")
    ap.add_argument("--profiller", default="", help="boşsa hepsi")
    ap.add_argument("--planlar", default="vurkac,kademeli,bedava")
    ap.add_argument("--yapi", default="profil", choices=["profil", "ikisi", "acik", "kapali"],
                    help="profil = her profilin varsayılanı")
    ap.add_argument("--kapi", default="ikisi", choices=["ikisi", "acik", "kapali"],
                    help="🔑 Fib kapısı + ADX filtresi")
    ap.add_argument("--bitis", type=int, default=0,
                    help="test kaç ay önce bitsin (0 = bugün). Örn. --ay 22 --bitis 12 → "
                         "geçen yılın verisiyle örneklem DIŞI doğrulama")
    ap.add_argument("--butce", type=float, default=10_000.0)
    return ap.parse_args()


def _write(ozet: list[dict], islem: list[dict], meta: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    O = pd.DataFrame(ozet)
    T = pd.DataFrame(islem)
    O.to_csv(OUT / "ozet.csv", index=False, float_format="%.4f")
    T.to_csv(OUT / "islemler.csv", index=False, float_format="%.4f")
    (OUT / "durum.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    with zipfile.ZipFile(OUT / "robot_toplu_test.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("ozet.csv", "islemler.csv", "durum.json"):
            z.write(OUT / f, f)


def main() -> int:
    a = _args()
    syms = [s.strip().upper() for s in a.semboller.split(",") if s.strip()] or list(UNIVERSE)
    profs = [p.strip() for p in a.profiller.split(",") if p.strip()] or list(rbt.PROFILES)
    plans = [p.strip() for p in a.planlar.split(",") if p.strip()]
    ystates = {"profil": (None,), "ikisi": (True, False), "acik": (True,), "kapali": (False,)}[a.yapi]
    gstates = {"ikisi": (True, False), "acik": (True,), "kapali": (False,)}[a.kapi]
    start = (pd.Timestamp.today() - pd.DateOffset(months=a.ay)).normalize()
    end = (pd.Timestamp.today() - pd.DateOffset(months=a.bitis)).normalize() if a.bitis else None
    t0 = time.time()
    meta = {"tarih": pd.Timestamp.now().isoformat(timespec="minutes"), "ay": a.ay,
            "baslangic": start.date().isoformat(),
            "bitis": end.date().isoformat() if end is not None else "bugün", "semboller": syms, "profiller": profs,
            "planlar": plans, "yapi": a.yapi, "bitti": [], "cekilemeyen": [], "hata": {},
            "surum": "robot v5 (Fib kapısı)", "kapi": a.kapi}
    ozet: list[dict] = []
    islem: list[dict] = []
    # veriyi 10'arlı paketlerle çek (Yahoo hız sınırı)
    H, D = {}, {}
    for k in range(0, len(syms), 10):
        part = syms[k:k + 10]
        h, fh = dta.fetch(part, "1h_long")
        d, fd = dta.fetch(part, "1d_long")
        H.update(h)
        D.update(d)
        print(f"veri {k + len(part)}/{len(syms)} · 1s eksik {fh} · günlük eksik {fd}", flush=True)
        time.sleep(1.0)
    for n_, sym in enumerate(syms, 1):
        h1, dd = H.get(sym), D.get(sym)
        if h1 is None or dd is None or len(h1) < 500:
            meta["cekilemeyen"].append(sym)
            print(f"⚠️  {sym}: veri yok", flush=True)
            continue
        if end is not None:
            hi_ = pd.DatetimeIndex(h1.index)
            hi_ = hi_.tz_localize(None) if hi_.tz is not None else hi_
            h1 = h1[hi_ < end]
            di_ = pd.DatetimeIndex(dd.index)
            di_ = di_.tz_localize(None) if di_.tz is not None else di_
            dd = dd[di_ < end]
        ts = time.time()
        try:
            for prof in profs:
                for pl in plans:
                    for ys, gt in [(y, g) for y in ystates for g in gstates]:
                        P = rbt.params_for(prof, plan=pl, budget=a.butce, structure=ys)
                        if not gt:
                            P.fib_gate, P.adx_min = (), 0.0
                        R = rbt.run(h1, dd, P, start=start)
                        s_ = R["stats"]
                        dg = R["diag"]
                        mdd = s_["Maks. düşüş %"]
                        key = {"Sembol": sym, "Tür": UNIVERSE.get(sym, "diğer"), "Profil": prof,
                               "Plan": pl, "Yapı": int(P.structure), "Kapı": int(gt)}
                        ozet.append({**key, "Getiri %": s_["Getiri %"], "Al-tut %": s_["Al-tut %"],
                                     "Maks. düşüş %": mdd,
                                     "Getiri/düşüş": s_["Getiri %"] / abs(mdd) if mdd and mdd < 0 else np.nan,
                                     "İşlem": s_["İşlem"], "Açık pozisyon": int(R["state"]["pozisyonda"]),
                                     "Kazanan %": s_["Kazanan %"], "Ort. R": s_["Ort. R"],
                                     "Toplam R": float(R["trades"]["R"].sum()) if len(R["trades"]) else 0.0,
                                     "Yapı vetosu (mum)": dg.get("🏛️ yapı vetosu", 0),
                                     "Dirence yakın (mum)": dg.get("🏛️ dirence çok yakın", 0),
                                     "Aday giriş (mum)": dg.get("hepsi birden", 0),
                                     "Mola": int((R["events"]["Olay"] == "⏸️ Mola").sum())
                                     if len(R["events"]) else 0})
                        T = R["trades"]
                        if len(T):
                            for _, r in T.iterrows():
                                islem.append({**key, **{k: (v.isoformat() if isinstance(v, pd.Timestamp) else v)
                                                        for k, v in r.items()}})
                        if R["state"]["pozisyonda"]:
                            st_ = R["state"]
                            islem.append({**key, "Giriş": None, "Son çıkış": "AÇIK",
                                          "Giriş fiyatı": st_["maliyet"],
                                          "K/Z %": st_["K/Z % (açık)"]})
            meta["bitti"].append(sym)
            print(f"✅ {n_}/{len(syms)} {sym:<6} {time.time() - ts:5.1f} sn", flush=True)
        except Exception as exc:                 # noqa: BLE001
            meta["hata"][sym] = f"{type(exc).__name__}: {exc}"
            print(f"❌ {sym}: {exc}", flush=True)
            traceback.print_exc()
        meta["sure_dk"] = round((time.time() - t0) / 60, 1)
        _write(ozet, islem, meta)
    _write(ozet, islem, meta)
    print(f"bitti · {len(meta['bitti'])} sembol · {len(ozet)} çalıştırma · {len(islem)} işlem · "
          f"{meta['sure_dk']} dk")
    return 0


if __name__ == "__main__":
    sys.exit(main())
