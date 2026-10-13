"""
🧬 PATERN MADENCİSİ — büyük hareketlerden ÖNCE oluşan sinyal / çizgi / histogram ÖRGÜSÜNÜ
günlük grafikte arar.

Tek tek işaretlere bakmak yerine, her hisse için her gün ~120 özellikten oluşan bir
"durum fotoğrafı" çekilir:
  • TradingView göstergelerimiz: APEX CORE v4 (WHALE / DAILY / RETAIL bantları, PRO−RETAIL,
    eğimleri, seesaw, sıkışma süresi), APEX V670 OMNI (momentum, Fusion, Synergy, HUD),
    Şahane / Konfluans, V710, VSA & Delta, Mum Gücü, ana Fib bacağındaki konum, EMA dizilimi,
    göreli güç, oynaklık sıkışması … ve son 10 / 20 günde hangi olayların kaç kez olduğu.
  • TradingView'de KULLANAMADIĞIMIZ balina izleri (burada ilk kez):
      – gece / gün içi getiri ayrışması (kurumlar kapanışta/gece, perakende açılışta işlem yapar)
      – 1 saatlik mumlardan: son saat − ilk saat (Smart Money Index), son saatin hacim payı,
        gün içi yoğunluk (hacim ağırlıklı kapanış konumu), yükselen saatlerin hacim payı
      – FINRA günlük short hacim oranı (5/20 gün, 60 güne göre z)
      – OBV / fiyat ayrışması, Amihud likidite kuruması, Kyle-λ (fiyat etkisi) değişimi
Sonra:
  1) "Büyük yükseliş" (20 günde ≥ +%15 VE SPY'yi ≥ 10 puan geçen) ve "büyük düşüş" günlerinden
     önceki 20 günün ORTALAMA ÖRGÜSÜ (parmak izi) çıkarılır — hangi çizgi/histogram nasıl
     davranıyor, normal günlerden kaç standart sapma farklı.
  2) Walk-forward öğrenme: eski yıllarda eğitilir, sonraki yıllarda (hiç görülmemiş) test
     edilir. Gradient boosting → isabet (AUC) ve "en yüksek skorlu %1 / %5 günlerde büyük
     hareket olasılığı / taban oran" (lift).
  3) Okunabilir ÖRGÜ KURALLARI (sığ karar ağacı): "WHALE > 62 VE sıkışma ≥ 9 gün VE son 10 günde
     Afterburner boğa ≥ 1 …" gibi; her kuralın eğitim ve TEST dönemindeki isabeti ayrı yazılır.
  4) TradingView dışı balina verilerinin katkısı: aynı model bu grupla ve grupsuz eğitilip
     test isabetleri karşılaştırılır.

GitHub Actions → "Patern madenciliği" → Run workflow. Çıktı: data/patterns/
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
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

from apex import compute as cmp             # noqa: E402
from apex import council as cnl             # noqa: E402
from apex import data as dta                # noqa: E402
from apex import engine as eng              # noqa: E402
from apex import pine_fib as pfib           # noqa: E402
from apex import universe as uni            # noqa: E402

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
OUT = ROOT / "data" / "patterns"
H = 20                     # ileri ufuk (işlem günü)
UP_RET, UP_EX = 15.0, 10.0  # büyük yükseliş: ≥ +%15 ve SPY'ye göre ≥ +10 puan

CORE_EV = {"star": "⭐ Golden Star", "st_in": "🕵️ Stealth IN", "st_out": "🕵️ Stealth OUT",
           "cross_up": "PRO↗RETAIL", "cross_dn": "PRO↘RETAIL", "real_acc": "🐋 ACC",
           "real_dist": "🐋 DIST", "sweep_bar": "🎣 Whale süpürme", "db_bottom": "🔵 Dip çıkışı",
           "rd_top": "🔴 Tepe dönüşü", "exhausted": "⚠️ Whale tükenme", "sqz_fire": "💥 Sıkışma patladı"}
OMNI_EV = {"ab_bull": "🚀 Afterburner boğa", "ab_bear": "🚀 Afterburner ayı",
           "dia_buy": "💎 Diamond AL", "dia_sell": "💎 Diamond SAT", "ew_bull": "🔵 Erken uyarı AL",
           "ew_bear": "🔴 Erken uyarı SAT", "sweep_low": "🎣 Dip süpürmesi",
           "sweep_high": "🎣 Tepe süpürmesi"}

LBL: dict[str, str] = {
    "whale": "WHALE bandı", "daily": "DAILY bandı", "retail": "RETAIL bandı",
    "pro_ret": "PRO − RETAIL", "w_pwr": "Whale gücü", "whale_d5": "WHALE 5 günlük değişim",
    "daily_d5": "DAILY 5 günlük değişim", "retail_d5": "RETAIL 5 günlük değişim",
    "pro_ret_d5": "PRO−RETAIL 5 günlük değişim", "red_cov": "WHALE ≥ DAILY (kırmızı örtüyor)",
    "sqz_on": "Sıkışmada", "sqz_dur": "Sıkışma süresi (gün)", "omni": "OMNI momentum",
    "omni_d5": "OMNI 5 günlük değişim", "f_hist": "Fusion histogram", "f_gap": "Speedline − sinyal",
    "s_speed": "Synergy hızı", "hud_net": "HUD net (boğa − ayı)", "mg_cls": "Mum Gücü sınıfı",
    "fib_r": "Ana Fib bacağında konum (0 tepe – 1 dip)", "fib_up": "Ana Fib bacağı yukarı",
    "ret5": "Son 5 gün getiri %", "ret20": "Son 20 gün getiri %", "ret60": "Son 60 gün getiri %",
    "d_e21": "EMA21'e uzaklık (ATR)", "d_e50": "EMA50'ye uzaklık (ATR)",
    "d_e200": "EMA200'e uzaklık (ATR)", "e50_200": "EMA50 > EMA200",
    "pos52": "52 haftalık aralıkta konum (0–1)", "from_hi": "52 hafta zirvesine uzaklık %",
    "atr_pct": "ATR % (oynaklık)", "atr_ratio": "ATR14 / ATR50 (oynaklık sıkışması < 1)",
    "bbw_rank": "Bollinger genişliği yüzdeliği (120 gün)", "vol_10_60": "Hacim 10 gün / 60 gün",
    "updn_vol": "Yükselen/düşen gün hacim oranı (20 gün)", "rs20": "SPY'ye göre 20 gün %",
    "rs60": "SPY'ye göre 60 gün %",
    # TradingView dışı balina izleri
    "x_on_id": "Gece − gün içi getiri farkı (20 gün) — kurumsal birikim",
    "x_on20": "Gece getirisi toplamı (20 gün)", "x_id20": "Gün içi getiri toplamı (20 gün)",
    "x_obv_div": "OBV − fiyat ayrışması (20 gün)", "x_amihud": "Amihud likidite (20/120)",
    "x_lambda": "Kyle-λ fiyat etkisi (60 gün)", "x_lambda_ch": "Kyle-λ değişimi (20 gün)",
    "x_svr5": "FINRA short hacim oranı (5 gün)", "x_svr20": "FINRA short hacim oranı (20 gün)",
    "x_svr_z": "FINRA short oranı z (60 güne göre)",
    "x_smi20": "Smart Money Index: son saat − ilk saat (20 gün)",
    "x_lastvol20": "Son saatin hacim payı (20 gün)",
    "x_ii20": "Gün içi yoğunluk — hacim ağırlıklı kapanış konumu (20 gün)",
    "x_uphr20": "Yükselen saatlerin hacim payı (20 gün)",
}
for k, v in {**CORE_EV, **OMNI_EV}.items():
    LBL[f"n10_{k}"] = f"Son 10 günde {v} sayısı"
    LBL[f"n20_{k}"] = f"Son 20 günde {v} sayısı"
LBL.update({f"tv_{k}": v for k, v in cnl.LABELS.items()})


HOURLY = ("x_smi20", "x_lastvol20", "x_ii20", "x_uphr20")


def is_x(c: str) -> bool:
    """TradingView dışı balina izi mi (pencere/dizi türevleri dahil)?"""
    return c.startswith(("x_", "w10_x_", "w30_x_", "seq_x_"))


def is_hourly(c: str) -> bool:
    return any(h in c for h in HOURLY)


def label(c: str) -> str:
    if c in LBL:
        return LBL[c]
    for pre, txt in (("w10_", "10 gün"), ("w30_", "30 gün")):
        if c.startswith(pre):
            base, stat = c[len(pre):].rsplit("__", 1)
            st_ = {"egim": "eğimi", "konum": "aralığındaki konumu (0 dip–1 tepe)",
                   "artan": "artan gün oranı"}.get(stat, stat)
            return f"{LBL.get(base, base)} — {txt} {st_}"
    if c.startswith("since_"):
        k = c[6:]
        return f"Son {({**CORE_EV, **OMNI_EV}).get(k, k)} olayından bu yana gün"
    if c.startswith("seq_"):
        base, lag = c[4:].rsplit("_", 1)
        return f"{LBL.get(base, base)} ({lag} gün önce)"
    return c


def group_of(c: str) -> str:
    if c.startswith("seq_"):
        return "örgü dizisi (son 30 günün seyri)"
    if c.startswith(("w10_", "w30_")):
        return "pencere eğilimleri (10 / 30 gün)"
    if c.startswith("since_"):
        return "olaydan bu yana geçen gün (sıra)"
    if c.startswith("x_"):
        return "TradingView dışı balina izi"
    if c.startswith(("n10_", "n20_")):
        return "olay sayıları"
    if c.startswith("tv_"):
        return "motor sinyalleri (anlık)"
    if c in ("whale", "daily", "retail", "pro_ret", "w_pwr", "whale_d5", "daily_d5", "retail_d5",
             "pro_ret_d5", "red_cov", "sqz_on", "sqz_dur"):
        return "APEX CORE çizgi/histogram"
    if c in ("omni", "omni_d5", "f_hist", "f_gap", "s_speed", "hud_net"):
        return "APEX 670 OMNI"
    return "fiyat yapısı / Fib / mum"


# ------------------------------------------------------------------ veri
def _naive(df: pd.DataFrame, norm: bool = True) -> pd.DataFrame:
    d = df.copy()
    ix = pd.DatetimeIndex(d.index)
    ix = ix.tz_convert("America/New_York").tz_localize(None) if ix.tz is not None else ix
    d.index = ix.normalize() if norm else ix
    return d[~d.index.duplicated(keep="last")].sort_index()


def fetch_all(syms, kind):
    out = {}
    for k in range(0, len(syms), 40):
        part = syms[k:k + 40]
        try:
            p, f = dta.fetch(part, kind)
        except Exception as exc:                 # noqa: BLE001
            print("  veri hatası", exc, flush=True)
            continue
        out.update({t: v for t, v in p.items() if v is not None and len(v)})
        print(f"  {kind} {k + len(part)}/{len(syms)} · eksik {len(f)}", flush=True)
        time.sleep(1.0)
    return out


def finra_history(dates: list[pd.Timestamp], syms: set[str]) -> pd.DataFrame:
    """FINRA Reg SHO günlük short hacim dosyaları (CNMS) → Date × Symbol tablosu."""
    import requests
    url = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{d}.txt"
    hdr = {"User-Agent": "Mozilla/5.0 (research; AETHER APEX)"}

    def one(d):
        for a in range(3):
            try:
                r = requests.get(url.format(d=d.strftime("%Y%m%d")), headers=hdr, timeout=30)
                if r.status_code == 404:
                    return None
                if r.status_code == 200 and r.content:
                    x = pd.read_csv(io.BytesIO(r.content), sep="|", dtype={"Symbol": str})
                    x = x[x["Symbol"].isin(syms)]
                    x["Date"] = d
                    return x[["Date", "Symbol", "ShortVolume", "TotalVolume"]]
            except Exception:                    # noqa: BLE001
                time.sleep(1 + a)
        return None
    parts = []
    with cf.ThreadPoolExecutor(8) as ex:
        for i, x in enumerate(ex.map(one, dates)):
            if x is not None:
                parts.append(x)
            if (i + 1) % 200 == 0:
                print(f"  FINRA {i + 1}/{len(dates)} gün", flush=True)
    if not parts:
        return pd.DataFrame()
    F = pd.concat(parts)
    F["svr"] = pd.to_numeric(F["ShortVolume"], errors="coerce") / \
        pd.to_numeric(F["TotalVolume"], errors="coerce").replace(0, np.nan)
    return F.pivot_table(index="Date", columns="Symbol", values="svr", aggfunc="mean")


def hourly_daily(h1: pd.DataFrame) -> pd.DataFrame:
    """1 saatlik mumlardan günlük balina izleri."""
    h = _naive(h1, norm=False)
    h = h[(h.index.hour >= 9) & (h.index.hour <= 15)]
    if h.empty:
        return pd.DataFrame()
    day = h.index.normalize()
    rng = (h["High"] - h["Low"]).replace(0, np.nan)
    clv = ((2 * h["Close"] - h["High"] - h["Low"]) / rng).fillna(0.0)
    v = h["Volume"].fillna(0.0)
    g = pd.DataFrame({"day": day, "o": h["Open"], "c": h["Close"], "v": v, "clvv": clv * v,
                      "upv": v.where(h["Close"] > h["Open"], 0.0)})
    agg = g.groupby("day").agg(fo=("o", "first"), fc=("c", "first"), lo=("o", "last"),
                               lc=("c", "last"), v=("v", "sum"), lv=("v", "last"),
                               clvv=("clvv", "sum"), upv=("upv", "sum"), n=("v", "size"))
    agg = agg[agg["n"] >= 5]
    smi = np.log(agg["lc"] / agg["lo"]) - np.log(agg["fc"] / agg["fo"])
    vv = agg["v"].replace(0, np.nan)
    out = pd.DataFrame(index=agg.index)
    out["x_smi20"] = (smi * 100).rolling(20, min_periods=10).sum()
    out["x_lastvol20"] = (agg["lv"] / vv).rolling(20, min_periods=10).mean()
    out["x_ii20"] = (agg["clvv"] / vv).rolling(20, min_periods=10).mean()
    out["x_uphr20"] = (agg["upv"] / vv).rolling(20, min_periods=10).mean()
    return out


# ------------------------------------------------------------------ özellikler
def features(d: pd.DataFrame, spy: pd.Series, svr: pd.Series | None,
             hd: pd.DataFrame | None) -> pd.DataFrame:
    d = d[["Open", "High", "Low", "Close", "Volume"]].astype(float).dropna(subset=["Close"])
    d["Volume"] = d["Volume"].fillna(0.0)
    o, h, l, c, v = (d[k] for k in ("Open", "High", "Low", "Close", "Volume"))
    X = pd.DataFrame(index=d.index)
    core = eng.apex_core(d)
    omni = eng.apex_omni(d)
    X["whale"], X["daily"], X["retail"] = core["whale"], core["daily"], core["retail"]
    X["pro_ret"] = core["pro"] - core["ret_line"]
    X["w_pwr"] = core["w_pwr"]
    for k in ("whale", "daily", "retail", "pro_ret"):
        X[f"{k}_d5"] = X[k] - X[k].shift(5)
    X["red_cov"] = (core["whale"] >= core["daily"]).astype(float)
    X["sqz_on"], X["sqz_dur"] = core["sqz_on"].astype(float), core["sqz_dur"].astype(float)
    X["omni"] = omni["mom"]
    X["omni_d5"] = omni["mom"] - omni["mom"].shift(5)
    X["f_hist"], X["f_gap"] = omni["f_hist"], omni["f_speed"] - omni["f_sig"]
    X["s_speed"], X["hud_net"] = omni["s_speed"], omni["hud_net"]
    for src, evs in ((core, CORE_EV), (omni, OMNI_EV)):
        for k in evs:
            e = src[k].fillna(False).astype(float)
            X[f"n10_{k}"] = e.rolling(10, min_periods=1).sum()
            X[f"n20_{k}"] = e.rolling(20, min_periods=1).sum()
    try:
        F = cnl.engine_features(d, prefix="tv_")
        X = X.join(F[[c for c in F.columns if c not in ("tv_c_whale_up",)]])
    except Exception as exc:                     # noqa: BLE001
        print("   engine_features:", exc)
    try:
        X["mg_cls"] = pfib.mum_gucu(d, None, 300)["cls"].to_numpy().astype(float)
    except Exception:                            # noqa: BLE001
        X["mg_cls"] = np.nan
    # ana Fib bacağı (5 günde bir, arası doldurulur — her nokta yalnız geçmişi görür)
    fr = pd.Series(np.nan, index=d.index)
    fu = pd.Series(np.nan, index=d.index)
    for i in range(260, len(d), 5):
        mf = pfib.main_fib(d.iloc[max(0, i + 1 - 400):i + 1])
        if mf.get("ok"):
            fr.iloc[i], fu.iloc[i] = float(mf["r"]), float(bool(mf["up"]))
    X["fib_r"], X["fib_up"] = fr.ffill(limit=4), fu.ffill(limit=4)
    # fiyat yapısı
    atr = pfib.atr_rma(d, 14)
    for n in (5, 20, 60):
        X[f"ret{n}"] = (c / c.shift(n) - 1) * 100
    for n, nm in ((21, "d_e21"), (50, "d_e50"), (200, "d_e200")):
        X[nm] = (c - c.ewm(span=n, adjust=False).mean()) / atr
    X["e50_200"] = (c.ewm(span=50, adjust=False).mean() > c.ewm(span=200, adjust=False).mean()).astype(float)
    hi, lo = h.rolling(252, min_periods=120).max(), l.rolling(252, min_periods=120).min()
    X["pos52"] = (c - lo) / (hi - lo).replace(0, np.nan)
    X["from_hi"] = (c / hi - 1) * 100
    X["atr_pct"] = atr / c * 100
    X["atr_ratio"] = atr / pfib.atr_rma(d, 50)
    bbw = c.rolling(20).std() / c.rolling(20).mean()
    X["bbw_rank"] = bbw.rolling(120, min_periods=60).rank(pct=True)
    X["vol_10_60"] = v.rolling(10).mean() / v.rolling(60).mean().replace(0, np.nan)
    upv = v.where(c > c.shift(1), 0.0).rolling(20).sum()
    dnv = v.where(c < c.shift(1), 0.0).rolling(20).sum()
    X["updn_vol"] = np.log((upv + 1) / (dnv + 1))
    sp = spy.reindex(d.index).ffill()
    for n in (20, 60):
        X[f"rs{n}"] = X[f"ret{n}"] - (sp / sp.shift(n) - 1) * 100
    # ---- TradingView dışı balina izleri
    on = np.log(o / c.shift(1))
    idr = np.log(c / o)
    X["x_on20"] = on.rolling(20).sum() * 100
    X["x_id20"] = idr.rolling(20).sum() * 100
    X["x_on_id"] = X["x_on20"] - X["x_id20"]
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    obv_z = (obv - obv.shift(20)) / (v.rolling(60).mean() * 20).replace(0, np.nan)
    X["x_obv_div"] = obv_z - X["ret20"] / (X["atr_pct"] * np.sqrt(20)).replace(0, np.nan)
    ret1 = c.pct_change()
    am = (ret1.abs() / (c * v).replace(0, np.nan))
    X["x_amihud"] = np.log(am.rolling(20).mean() / am.rolling(120).mean())
    sq = np.sign(ret1) * np.sqrt((c * v).clip(lower=1))
    cov = ret1.rolling(60).cov(sq)
    var = sq.rolling(60).var().replace(0, np.nan)
    lam = cov / var * 1e6
    X["x_lambda"] = lam
    X["x_lambda_ch"] = lam - lam.shift(20)
    if svr is not None and len(svr.dropna()) > 30:
        s = svr.reindex(d.index)
        X["x_svr5"] = s.rolling(5, min_periods=3).mean()
        X["x_svr20"] = s.rolling(20, min_periods=10).mean()
        m60, s60 = s.rolling(60, min_periods=30).mean(), s.rolling(60, min_periods=30).std()
        X["x_svr_z"] = (X["x_svr5"] - m60) / s60.replace(0, np.nan)
    if hd is not None and len(hd):
        X = X.join(hd.reindex(d.index))
    # ---- ÇOK MUMLU PENCERE: tek günün fotoğrafı değil, son 10 / 30 günün SEYRİ
    W_KEYS = ["whale", "daily", "retail", "pro_ret", "w_pwr", "omni", "f_hist", "f_gap", "s_speed",
              "hud_net", "sqz_dur", "atr_ratio", "bbw_rank", "vol_10_60", "updn_vol", "rs20",
              "x_on_id", "x_obv_div", "x_svr5", "x_ii20", "x_smi20"]
    win = {}
    for k in [k for k in W_KEYS if k in X.columns]:
        y = X[k].astype(float)
        sd = y.rolling(252, min_periods=60).std().replace(0, np.nan)
        yf = y.ffill()
        for W in (10, 30):
            win[f"w{W}_{k}__egim"] = _roll_slope(yf, W) * W / sd            # pencere boyu değişim / std
            mn, mx = yf.rolling(W).min(), yf.rolling(W).max()
            win[f"w{W}_{k}__konum"] = (yf - mn) / (mx - mn).replace(0, np.nan)
            win[f"w{W}_{k}__artan"] = (yf.diff() > 0).astype(float).rolling(W).mean()
    # olayların SIRASI: her olaydan bu yana kaç gün geçti (60'ta kesilir)
    for src, evs in ((core, CORE_EV), (omni, OMNI_EV)):
        for k in evs:
            e = src[k].fillna(False).to_numpy(bool)
            ar = np.arange(len(e), dtype=float)
            last = pd.Series(np.where(e, ar, np.nan)).ffill().to_numpy()
            win[f"since_{k}"] = np.minimum(np.nan_to_num(ar - last, nan=60.0), 60.0)
    # son 30 günün DİZİSİ (3 günde bir nokta) — şablon eşleştirme için
    for k in [k for k in ("whale", "daily", "retail", "pro_ret", "omni", "f_hist", "s_speed",
                          "sqz_dur", "vol_10_60", "rs20", "x_on_id") if k in X.columns]:
        for L in range(0, 31, 3):
            win[f"seq_{k}_{L}"] = X[k].shift(L)
    X = pd.concat([X, pd.DataFrame(win, index=X.index)], axis=1)
    return X.astype("float32")


def _roll_slope(y: pd.Series, W: int) -> pd.Series:
    """Son W mumdaki doğrusal eğim (birim/mum) — evrişimle hızlı."""
    t = np.arange(W, dtype=float)
    k = (t - t.mean()) / ((t - t.mean()) ** 2).sum()
    a = y.to_numpy(float)
    out = np.full(len(a), np.nan)
    if len(a) >= W:
        out[W - 1:] = np.convolve(a, k[::-1], mode="valid")
    return pd.Series(out, index=y.index)


def targets(c: pd.Series, spy: pd.Series, atr_pct: pd.Series) -> pd.DataFrame:
    sp = spy.reindex(c.index).ffill()
    r = (c.shift(-H) / c - 1) * 100
    ex = r - (sp.shift(-H) / sp - 1) * 100
    zz = ex / (atr_pct * np.sqrt(H)).replace(0, np.nan)      # hissenin KENDİ oynaklığına göre
    return pd.DataFrame({"fwd": r, "fwd_ex": ex,
                         "upz": (zz >= 1.5).astype(float).where(zz.notna()),
                         "up": ((r >= UP_RET) & (ex >= UP_EX)).astype(float).where(r.notna()),
                         "dn": ((r <= -UP_RET) & (ex <= -UP_EX)).astype(float).where(r.notna())})


# ------------------------------------------------------------------ analiz
def fingerprint(D: pd.DataFrame, cols: list[str], tgt: str) -> pd.DataFrame:
    """Büyük hareketten önceki örgü: olay günü t için t−20…t'de özelliklerin ortalama z'si."""
    mu, sd = D[cols].mean(), D[cols].std().replace(0, np.nan)
    rows = []
    lags = (-40, -30, -20, -15, -10, -5, -3, -1, 0)
    for s, g in D.groupby("Sembol"):
        g = g.sort_index()
        ev = g.index[g[tgt] == 1]
        keep, last = [], None
        for t in ev:                                     # 20 gün içinde tekrar sayma
            if last is None or (t - last).days > 30:
                keep.append(t)
                last = t
        pos = {t: i for i, t in enumerate(g.index)}
        for t in keep:
            i = pos[t]
            for L in lags:
                j = i + int(np.floor(L / 2))            # satırlar 2 günde bir örneklendi
                if 0 <= j < len(g):
                    rows.append((L, ((g[cols].iloc[j] - mu) / sd).to_numpy()))
    if not rows:
        return pd.DataFrame()
    out = {}
    for L in lags:
        arr = np.array([r[1] for r in rows if r[0] == L], dtype=float)
        out[L] = np.nanmean(arr, axis=0)
    FP = pd.DataFrame(out, index=cols)
    FP.columns = [f"t{L}" if L else "t0" for L in lags]
    FP["etiket"] = [label(c) for c in cols]
    FP["grup"] = [group_of(c) for c in cols]
    FP["n olay"] = len([r for r in rows if r[0] == 0])
    FP["güç"] = FP[["t-5", "t-3", "t-1", "t0"]].abs().mean(axis=1)
    return FP.sort_values("güç", ascending=False)


def _rules(tree, cols, Xtr, ytr, Xte, yte, base_tr, base_te, top=10):
    """Karar ağacının yapraklarını okunabilir 'örgü kuralı'na çevirir."""
    T = tree.tree_
    out = []

    def walk(node, conds):
        if T.children_left[node] == -1:
            out.append((node, conds))
            return
        f, thr = T.feature[node], T.threshold[node]
        walk(T.children_left[node], conds + [(f, "≤", thr)])
        walk(T.children_right[node], conds + [(f, ">", thr)])
    walk(0, [])
    leaf_tr = tree.apply(Xtr)
    leaf_te = tree.apply(Xte)
    rows = []
    for node, conds in out:
        mtr, mte = leaf_tr == node, leaf_te == node
        if mtr.sum() < 50:
            continue
        p_tr = float(ytr[mtr].mean())
        p_te = float(yte[mte].mean()) if mte.sum() else np.nan
        # aynı özelliğin koşullarını aralığa indir
        rng: dict[int, list] = {}
        for f, op, thr in conds:
            lo, hi = rng.get(f, [-np.inf, np.inf])
            if op == "≤":
                hi = min(hi, thr)
            else:
                lo = max(lo, thr)
            rng[f] = [lo, hi]
        parts = []
        for f, (lo, hi) in rng.items():
            nm = label(cols[f])
            if np.isfinite(lo) and np.isfinite(hi):
                parts.append(f"{nm} {lo:.2f} … {hi:.2f}")
            elif np.isfinite(lo):
                parts.append(f"{nm} > {lo:.2f}")
            else:
                parts.append(f"{nm} ≤ {hi:.2f}")
        rows.append({"Örgü kuralı": " VE ".join(parts), "Eğitim n": int(mtr.sum()),
                     "Eğitim olasılık %": round(p_tr * 100, 2),
                     "Eğitim lift": round(p_tr / base_tr, 2) if base_tr else np.nan,
                     "Test n": int(mte.sum()),
                     "Test olasılık %": round(p_te * 100, 2) if np.isfinite(p_te) else np.nan,
                     "Test lift": round(p_te / base_te, 2) if base_te and np.isfinite(p_te) else np.nan})
    R = pd.DataFrame(rows)
    if R.empty:
        return R
    return R.sort_values("Eğitim lift", ascending=False).head(top).reset_index(drop=True)


def experiment(D: pd.DataFrame, cols: list[str], tgt: str, split: str, gap_days: int = 35,
               name: str = "") -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.tree import DecisionTreeClassifier
    sp = pd.Timestamp(split)
    tr = D[(D.index <= sp) & D[tgt].notna()]
    te = D[(D.index > sp + pd.Timedelta(days=gap_days)) & D[tgt].notna()]
    if len(tr) < 5000 or len(te) < 2000 or tr[tgt].sum() < 100:
        return {"not": f"yetersiz veri (eğitim {len(tr)}, test {len(te)})"}
    Xtr, ytr = tr[cols].to_numpy(np.float32), tr[tgt].to_numpy()
    Xte, yte = te[cols].to_numpy(np.float32), te[tgt].to_numpy()
    base_tr, base_te = float(ytr.mean()), float(yte.mean())
    res = {"deney": name, "hedef": tgt, "eğitim": f"{tr.index.min().date()} → {tr.index.max().date()}",
           "test": f"{te.index.min().date()} → {te.index.max().date()}",
           "eğitim satır": int(len(tr)), "test satır": int(len(te)),
           "taban oran eğitim %": round(base_tr * 100, 2), "taban oran test %": round(base_te * 100, 2)}

    def fit_eval(cs):
        m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_leaf_nodes=31,
                                           min_samples_leaf=300, l2_regularization=1.0,
                                           random_state=0)
        ix = [cols.index(c) for c in cs]
        m.fit(Xtr[:, ix], ytr)
        p = m.predict_proba(Xte[:, ix])[:, 1]
        auc = float(roc_auc_score(yte, p))
        lifts = {}
        for q in (0.01, 0.05, 0.10):
            thr = np.quantile(p, 1 - q)
            sel = p >= thr
            lifts[f"en iyi %{int(q * 100)}"] = {
                "olasılık %": round(float(yte[sel].mean()) * 100, 2),
                "lift": round(float(yte[sel].mean()) / base_te, 2) if base_te else None,
                "ileri 20G ort %": round(float(te["fwd"].to_numpy()[sel].mean()), 2)}
        # yıllara göre (test) en iyi %5
        thr5 = np.quantile(p, 0.95)
        yr = pd.DataFrame({"y": yte, "s": p >= thr5}, index=te.index)
        by = yr.groupby(yr.index.year).apply(
            lambda g: round(float(g.loc[g["s"], "y"].mean() / max(g["y"].mean(), 1e-9)), 2)
            if g["s"].sum() > 20 else None).to_dict()
        return m, p, auc, lifts, by
    m, p, auc, lifts, by = fit_eval(cols)
    res["AUC (test)"] = round(auc, 3)
    res["lift (test)"] = lifts
    res["en iyi %5 lift — yıllara göre"] = {int(k): v for k, v in by.items()}
    # karşılaştırma: SADECE oynaklık bilgisiyle ne kadar tahmin edilebiliyor? (oynak hisseler
    # doğası gereği daha sık ±%15 yapar — örgünün katkısı bunun ÜSTÜNDE olmalı)
    vc = [c for c in cols if c in ("atr_pct", "atr_ratio", "bbw_rank") or
          any(c.startswith(f"w{W}_{k}__") for W in (10, 30) for k in ("atr_ratio", "bbw_rank"))]
    if vc:
        _, _, auc_v, lifts_v, _ = fit_eval(vc)
        res["SADECE oynaklık (kıyas)"] = {"AUC (test)": round(auc_v, 3), "lift (test)": lifts_v}
    xc = [c for c in cols if is_x(c)]
    if xc:
        _, _, auc_wo, lifts_wo, _ = fit_eval([c for c in cols if c not in xc])
        res["TradingView dışı balina izleri OLMADAN"] = {"AUC (test)": round(auc_wo, 3),
                                                         "lift (test)": lifts_wo}
    # grup bazında permütasyon önemi (test alt örneği)
    try:
        rs = np.random.default_rng(0)
        sub = rs.choice(len(te), size=min(40000, len(te)), replace=False)
        Xs, ys = Xte[sub], yte[sub]
        base_auc = roc_auc_score(ys, m.predict_proba(Xs)[:, 1])
        imp = {}
        groups: dict[str, list[int]] = {}
        for i, c in enumerate(cols):
            groups.setdefault(group_of(c), []).append(i)
        for g, ix in groups.items():
            Xp = Xs.copy()
            perm = rs.permutation(len(Xp))
            Xp[:, ix] = Xp[perm][:, ix]
            imp[g] = round(float(base_auc - roc_auc_score(ys, m.predict_proba(Xp)[:, 1])), 4)
        res["grup önemi (AUC kaybı)"] = dict(sorted(imp.items(), key=lambda kv: -kv[1]))
        single = []
        for i, c in enumerate(cols):
            Xp = Xs.copy()
            Xp[:, i] = Xp[rs.permutation(len(Xp)), i]
            single.append((c, float(base_auc - roc_auc_score(ys, m.predict_proba(Xp)[:, 1]))))
        single.sort(key=lambda kv: -kv[1])
        res["en önemli 20 özellik"] = [{"özellik": label(c), "grup": group_of(c),
                                        "AUC kaybı": round(v, 4)} for c, v in single[:20]]
    except Exception as exc:                     # noqa: BLE001
        res["önem hatası"] = str(exc)
    # ŞABLON EŞLEŞTİRME: eğitimdeki büyük hareketlerden önceki ORTALAMA 30 günlük örgü;
    # test günlerinde her hissenin son 30 günü bu şablona ne kadar benziyor?
    sq = [c for c in cols if c.startswith("seq_")]
    if sq:
        try:
            ix = [cols.index(c) for c in sq]
            A, B = Xtr[:, ix].astype(float), Xte[:, ix].astype(float)
            mu_, sd_ = np.nanmean(A, 0), np.nanstd(A, 0)
            sd_[sd_ == 0] = 1
            ZA, ZB = np.nan_to_num((A - mu_) / sd_), np.nan_to_num((B - mu_) / sd_)
            tmpl = ZA[ytr == 1].mean(0) - ZA.mean(0)
            nb = np.linalg.norm(ZB, axis=1) * (np.linalg.norm(tmpl) or 1)
            sim = (ZB @ tmpl) / np.where(nb == 0, 1, nb)
            sl = {}
            for q in (0.01, 0.05, 0.10):
                sel = sim >= np.quantile(sim, 1 - q)
                sl[f"en benzer %{int(q * 100)}"] = {
                    "olasılık %": round(float(yte[sel].mean()) * 100, 2),
                    "lift": round(float(yte[sel].mean()) / base_te, 2) if base_te else None}
            res["şablon eşleştirme (test)"] = {"AUC": round(float(roc_auc_score(yte, sim)), 3), **sl}
            res["_sablon"] = {c: float(v) for c, v in zip(sq, tmpl * sd_)}   # özgün birimde fark
        except Exception as exc:                 # noqa: BLE001
            res["şablon hatası"] = str(exc)
    # okunabilir örgü kuralları
    try:
        med = np.nanmedian(Xtr, axis=0)
        Xtr_f = np.where(np.isnan(Xtr), med, Xtr)
        Xte_f = np.where(np.isnan(Xte), med, Xte)
        rules = []
        for depth in (3, 4, 5):
            t = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=max(400, len(tr) // 400),
                                       class_weight={0: 1, 1: 3}, random_state=0)
            t.fit(Xtr_f, ytr)
            R = _rules(t, cols, Xtr_f, ytr, Xte_f, yte, base_tr, base_te)
            if len(R):
                R.insert(0, "Derinlik", depth)
                rules.append(R)
        RR = pd.concat(rules).drop_duplicates("Örgü kuralı") if rules else pd.DataFrame()
        RR = RR.sort_values("Eğitim lift", ascending=False).head(15)
        res["örgü kuralları"] = RR.to_dict(orient="records")
    except Exception as exc:                     # noqa: BLE001
        res["kural hatası"] = str(exc)
        traceback.print_exc()
    return res


def _js(o):
    if isinstance(o, dict):
        return {str(k): _js(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_js(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="test için ilk N hisse")
    ap.add_argument("--finra", default="evet", choices=["evet", "hayir"])
    ap.add_argument("--saat", default="evet", choices=["evet", "hayir"],
                    help="1 saatlik mumlardan balina izleri (son ~2 yıl)")
    a = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    meta: dict = {"tarih": pd.Timestamp.now().isoformat(timespec="minutes"), "surum": "patern v1",
                  "ufuk gün": H, "büyük yükseliş": f"{H} günde ≥ +%{UP_RET:g} ve SPY'ye göre ≥ +{UP_EX:g} puan",
                  "büyük düşüş": f"{H} günde ≤ −%{UP_RET:g} ve SPY'ye göre ≤ −{UP_EX:g} puan",
                  "hata": {}}
    syms = sorted({s for tema in uni.THEME_TRACKER for s in cmp.theme_holdings(tema)})
    if a.limit:
        syms = syms[:a.limit]
    print(f"{len(syms)} hisse", flush=True)
    P = fetch_all(syms + ["SPY"], "1d_long")
    P = {k: _naive(v).tail(1500) for k, v in P.items()}
    spy = P["SPY"]["Close"]
    cal = P["SPY"].index
    HR = {}
    if a.saat == "evet":
        try:
            HR = {k: v for k, v in fetch_all(syms, "1h_long").items()}
        except Exception as exc:                 # noqa: BLE001
            meta["hata"]["1s"] = str(exc)
    SV = pd.DataFrame()
    if a.finra == "evet":
        try:
            dates = list(cal[cal >= cal[-1] - pd.DateOffset(years=5)])
            SV = finra_history(dates, set(syms))
            print(f"  FINRA: {SV.shape}", flush=True)
        except Exception as exc:                 # noqa: BLE001
            meta["hata"]["finra"] = str(exc)
    parts = []
    for k, s in enumerate(syms, 1):
        d = P.get(s)
        if d is None or len(d) < 400:
            continue
        try:
            hd = hourly_daily(HR[s]) if s in HR else None
            svr = SV[s] if s in SV.columns else None
            X = features(d, spy, svr, hd)
            Y = targets(d["Close"].astype(float), spy, X["atr_pct"].astype(float))
            Z = X.join(Y).iloc[260::2]                  # ısınma sonrası, 2 günde bir
            Z["Sembol"] = s
            parts.append(Z)
        except Exception as exc:                 # noqa: BLE001
            meta["hata"][s] = f"{type(exc).__name__}: {exc}"
        if k % 50 == 0:
            print(f"  özellik {k}/{len(syms)} · {(time.time() - t0) / 60:.1f} dk", flush=True)
    D = pd.concat(parts)
    D.index.name = "Tarih"
    cols = [c for c in D.columns if c not in ("fwd", "fwd_ex", "up", "dn", "upz", "Sembol")]
    cols = [c for c in cols if D[c].notna().mean() > 0.05 and D[c].std() > 0]
    meta["satır"], meta["hisse"], meta["özellik"] = int(len(D)), int(D["Sembol"].nunique()), len(cols)
    meta["özellik grupları"] = pd.Series([group_of(c) for c in cols]).value_counts().to_dict()
    meta["taban oran (tümü) %"] = {"büyük yükseliş": round(float(D["up"].mean()) * 100, 2),
                                   "büyük düşüş": round(float(D["dn"].mean()) * 100, 2),
                                   "oynaklığına göre büyük yükseliş": round(float(D["upz"].mean()) * 100, 2)}
    meta["hedefler"] = {"up": "büyük yükseliş (mutlak)", "dn": "büyük düşüş (mutlak)",
                        "upz": "oynaklığına göre büyük yükseliş: SPY'ye göre 20G getiri ≥ 1.5 × "
                               "(ATR% × √20) — oynak hisse olmanın etkisini ayıklar"}
    print("örnek tablo:", D.shape, flush=True)
    # 1) parmak izleri
    fp_cols = [c for c in cols if not c.startswith("seq_")]
    FPu = fingerprint(D, fp_cols, "up")
    FPd = fingerprint(D, fp_cols, "dn")
    FPz = fingerprint(D, fp_cols, "upz")
    FPz.to_csv(OUT / "parmakizi_yukselis_oynakliga_gore.csv", float_format="%.3f")
    meta["parmak izi — oynaklığına göre büyük yükselişten önce (en güçlü 25)"] = FPz.head(25)[
        ["etiket", "grup", "t-40", "t-20", "t-10", "t-5", "t-1", "t0", "n olay"]].round(2).to_dict(orient="records")
    FPu.to_csv(OUT / "parmakizi_yukselis.csv", float_format="%.3f")
    FPd.to_csv(OUT / "parmakizi_dusus.csv", float_format="%.3f")
    meta["parmak izi — büyük yükselişten önce (en güçlü 25)"] = FPu.head(25)[
        ["etiket", "grup", "t-40", "t-20", "t-10", "t-5", "t-1", "t0", "n olay"]].round(2).to_dict(orient="records")
    meta["parmak izi — büyük düşüşten önce (en güçlü 25)"] = FPd.head(25)[
        ["etiket", "grup", "t-40", "t-20", "t-10", "t-5", "t-1", "t0", "n olay"]].round(2).to_dict(orient="records")
    # 2–4) walk-forward deneyler
    last = D.index.max()
    exps = {}
    for tgt in ("up", "upz", "dn"):
        try:
            exps[f"A · {tgt} · 5 yıl (1s verisi hariç)"] = experiment(
                D, [c for c in cols if not is_hourly(c)],
                tgt, str((last - pd.DateOffset(months=27)).date()), name="A")
        except Exception as exc:                 # noqa: BLE001
            meta["hata"][f"A_{tgt}"] = str(exc)
            traceback.print_exc()
        try:
            D2 = D[D.index >= last - pd.DateOffset(months=24)]
            exps[f"B · {tgt} · son 2 yıl (1s balina izleri dahil)"] = experiment(
                D2, cols, tgt, str((last - pd.DateOffset(months=11)).date()), name="B")
        except Exception as exc:                 # noqa: BLE001
            meta["hata"][f"B_{tgt}"] = str(exc)
            traceback.print_exc()
        print(f"  deney {tgt} bitti · {(time.time() - t0) / 60:.1f} dk", flush=True)
    # şablonları ayrı tabloya (seri × gün önce) — ozet.json'u şişirmesin
    trows = []
    for k_, e_ in exps.items():
        for c_, v_ in (e_.pop("_sablon", {}) or {}).items():
            base_, lag_ = c_[4:].rsplit("_", 1)
            trows.append({"Deney": k_, "Seri": label(base_), "Gün önce": int(lag_), "Fark": v_})
    if trows:
        pd.DataFrame(trows).pivot_table(index=["Deney", "Seri"], columns="Gün önce",
                                        values="Fark").round(3).to_csv(OUT / "sablon.csv")
    meta["deneyler"] = exps
    meta["sure_dk"] = round((time.time() - t0) / 60, 1)
    (OUT / "ozet.json").write_text(json.dumps(_js(meta), ensure_ascii=False, indent=1))
    rows = []
    for k, e in exps.items():
        for r in e.get("örgü kuralları", []):
            rows.append({"Deney": k, **r})
    pd.DataFrame(rows).to_csv(OUT / "kurallar.csv", index=False)
    with zipfile.ZipFile(OUT / "patern_madenciligi.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("ozet.json", "kurallar.csv", "parmakizi_yukselis.csv", "parmakizi_dusus.csv",
                  "parmakizi_yukselis_oynakliga_gore.csv", "sablon.csv"):
            if (OUT / f).exists():
                z.write(OUT / f, f)
    print(json.dumps(_js({k: {kk: vv for kk, vv in v.items() if kk in ("AUC (test)", "lift (test)")}
                          for k, v in exps.items()}), ensure_ascii=False, indent=1))
    print(f"bitti · {meta['sure_dk']} dk")
    return 0


if __name__ == "__main__":
    sys.exit(main())
