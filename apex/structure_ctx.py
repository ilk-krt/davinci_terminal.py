# apex/structure_ctx.py — robotun "YAPI KATMANI"
#
# TradingView'de çizdirdiğimiz bütün YAPI çıktılarını gün gün ve NEDENSEL olarak
# (her gün yalnızca o güne kadarki mumlarla) hesaplar; robot bunları
#   • giriş vetosu  (dirence çok yakın, Elliott ayı itkisi, haftalık trend aşağı,
#                    dağıtım/fake yükseliş mumu)
#   • yeni giriş tetiği (destek / boğa FVG-GAP kutusu / Wyckoff talep bölgesi /
#                    likidite süpürmesi üstünde hacimli dönüş)
#   • yapısal STOP  (en yakın anlamlı desteğin, kutunun ya da Elliott teyit
#                    seviyesinin altı)
#   • yapısal HEDEF (sıradaki direnç, açık gap dolumu, arz bölgesi, likidite havuzu,
#                    Elliott hedef kutusu)
#   • konseye ek özellik (fiyatın bu seviyelere ATR cinsinden uzaklığı vb.)
# olarak kullanır.
#
# Kaynak modüller: pine_structure (S/R Matrix, Gap Matrix, Wyckoff bölgeleri, likidite,
# GFR), pine_elliott (canlı faz + hedef kutusu), pine_fib (ana bacak, Mum Gücü).
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from apex import pine_elliott as pell
from apex import pine_fib as pfib
from apex import pine_structure as pst
from apex.indicators import ema

_log = logging.getLogger(__name__)

CALC_TAIL = 320            # her gün için kullanılan geçmiş (günlük mum)
_CACHE: dict = {}

KIND_CODE = {"w3": 3, "w5": 5}

COLS = ["atr", "close", "s1", "s1_sc", "s2", "r1", "r1_sc", "r2",
        "gb_top", "gb_bot", "gb_tier", "gr_bot", "gr_top", "gr_tier",
        "dz_top", "dz_bot", "sz_bot", "sz_top", "lq_up", "lq_dn", "sweep",
        "vsa_bias", "ell_sgn", "ell_kind", "ell_zlo", "ell_zhi", "ell_stop", "ell_inv",
        "ell_dead", "gfr_ready", "gfr_top", "mg_cls", "mg_syn", "fib_up", "fib_r",
        "wk_trend", "wk_fib_up", "wk_ell_sgn"]

LABELS = {
    "y_near_s1": "Fiyat güçlü DESTEĞE yakın", "y_room_r1": "Dirence kalan mesafe",
    "y_in_bgap": "Boğa FVG/GAP kutusunun içinde", "y_gap_above": "Yukarıda açık gap (dolum hedefi)",
    "y_in_demand": "Wyckoff TALEP bölgesinde", "y_supply_near": "Yakında Wyckoff ARZ bölgesi",
    "y_sweep": "Likidite süpürmesi (dip + / tepe −)", "y_vsa": "VSA bölge işareti (son 3 gün)",
    "y_ell_bull": "Elliott boğa itkisi (③/⑤ bekleniyor)", "y_ell_bear": "Elliott ayı itkisi",
    "y_gfr": "GFR boşluk dolum kurulumu hazır", "y_mg_cls": "Günlük Mum Gücü sınıfı",
    "y_fib_r": "Ana Fib bacağında konum (0 tepe – 1 dip)", "y_wk_trend": "HAFTALIK trend",
    "y_wk_fib": "HAFTALIK Fib bacağı yukarı", "y_wk_ell": "HAFTALIK Elliott yönü",
}
MG_VAL = {1: 1.0, 2: 0.6, 3: -0.2, -1: -0.3, -2: -0.6, -3: -0.3, 10: 0.0, 11: -0.8, 0: 0.0}


def _row(sub: pd.DataFrame) -> dict[str, float]:
    """Verilen günlük geçmişin SON barı için yapı özeti."""
    out = dict.fromkeys(COLS, np.nan)
    atr_s = pfib.atr_rma(sub, 14)
    a = float(atr_s.iloc[-1])
    c = float(sub["Close"].iloc[-1])
    out["atr"], out["close"] = a, c
    try:
        sr = pst.sr_matrix(sub, atr_s)
        if sr["sup"]:
            out["s1"], out["s1_sc"] = sr["sup"][0]["px"], sr["sup"][0]["score"]
            if len(sr["sup"]) > 1:
                out["s2"] = sr["sup"][1]["px"]
        if sr["res"]:
            out["r1"], out["r1_sc"] = sr["res"][0]["px"], sr["res"][0]["score"]
            if len(sr["res"]) > 1:
                out["r2"] = sr["res"][1]["px"]
    except Exception as exc:                     # noqa: BLE001
        _log.info("yapı sr: %s", exc)
    try:
        gm = pst.gap_matrix(sub, atr_s, "Günlük")
        bull = [b for b in gm["boxes"] if b["dir"] > 0 and b["bot"] < c]
        bear = [b for b in gm["boxes"] if b["dir"] < 0 and b["top"] > c]
        if bull:
            b = max(bull, key=lambda x: x["top"])
            out["gb_top"], out["gb_bot"], out["gb_tier"] = b["top"], b["bot"], b["tier"]
        if bear:
            b = min(bear, key=lambda x: x["bot"])
            out["gr_bot"], out["gr_top"], out["gr_tier"] = b["bot"], b["top"], b["tier"]
    except Exception as exc:                     # noqa: BLE001
        _log.info("yapı gap: %s", exc)
    if float(sub["Volume"].tail(60).sum()) > 0:
        try:
            vz = pst.vsa_zones(sub)
            dem = [z for z in vz["zones"] if z["side"] > 0 and z["bot"] < c
                   and (c - z["top"]) / c * 100 <= pst.ZONE_MAXD]
            sup = [z for z in vz["zones"] if z["side"] < 0 and z["top"] > c
                   and (z["bot"] - c) / c * 100 <= pst.ZONE_MAXD]
            if dem:
                z = max(dem, key=lambda x: x["top"])
                out["dz_top"], out["dz_bot"] = z["top"], z["bot"]
            if sup:
                z = min(sup, key=lambda x: x["bot"])
                out["sz_bot"], out["sz_top"] = z["bot"], z["top"]
            n = vz["n"]
            rec = [e for e in vz["events"] if e[0] >= n - 3]
            out["vsa_bias"] = float(np.clip(sum(e[2] * e[3] / 100 for e in rec), -1, 1)) if rec else 0.0
        except Exception as exc:                 # noqa: BLE001
            _log.info("yapı vsa: %s", exc)
    try:
        lq = pst.liquidity(sub, atr_s)
        if lq["up"]:
            out["lq_up"] = lq["up"]["px"]
        if lq["dn"]:
            out["lq_dn"] = lq["dn"]["px"]
        n = lq["n"]
        ev = [e for e in lq["events"] if e[0] >= n - 2 and abs(e[1]) == 2]
        # −2: dip havuzu süpürüldü + geri döndü (boğa) · +2: tepe süpürüldü + döndü (ayı)
        out["sweep"] = float(-np.sign(ev[-1][1])) if ev else 0.0
    except Exception as exc:                     # noqa: BLE001
        _log.info("yapı lq: %s", exc)
    try:
        el = pell.live(sub[["High", "Low", "Close"]])
        out["ell_sgn"] = float(el.get("sgn") or 0)
        k = el.get("kind") or ""
        out["ell_kind"] = float(KIND_CODE.get(k, -1 if k.startswith("abc") else 0))
        z = el.get("zone")
        if z:
            out["ell_zlo"], out["ell_zhi"] = z["zlo"], z["zhi"]
            out["ell_stop"] = z["stop"] if z["stop"] is not None else np.nan
            out["ell_inv"] = z["inv"] if z["inv"] is not None else np.nan
            out["ell_dead"] = float(bool(z["dead"]))
    except Exception as exc:                     # noqa: BLE001
        _log.info("yapı ell: %s", exc)
    try:
        s1 = out["s1"] if np.isfinite(out["s1"]) else None
        g = pst.gfr(sub, atr_s, None, s1)
        out["gfr_ready"] = float(bool(g["ready"]))
        out["gfr_top"] = g["top"] if g["top"] is not None else np.nan
    except Exception as exc:                     # noqa: BLE001
        _log.info("yapı gfr: %s", exc)
    try:
        mg = pfib.mum_gucu(sub, None, 300)
        out["mg_cls"], out["mg_syn"] = float(mg["cls"].iloc[-1]), float(mg["syn"].iloc[-1])
    except Exception as exc:                     # noqa: BLE001
        _log.info("yapı mg: %s", exc)
    try:
        mf = pfib.main_fib(sub)
        if mf.get("ok"):
            out["fib_up"], out["fib_r"] = float(bool(mf["up"])), float(mf["r"])
    except Exception as exc:                     # noqa: BLE001
        _log.info("yapı fib: %s", exc)
    return out


def _weekly(d: pd.DataFrame) -> pd.DataFrame:
    """Tamamlanmış haftalar için haftalık trend / Fib / Elliott (hafta sonu tarihli)."""
    w = d.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min",
                                 "Close": "last", "Volume": "sum"}).dropna(subset=["Close"])
    rows = {}
    e10, e30 = ema(w["Close"], 10), ema(w["Close"], 30)
    for j in range(40, len(w)):
        sub = w.iloc[:j + 1]
        c, a, b = w["Close"].iloc[j], e10.iloc[j], e30.iloc[j]
        tr = 1.0 if c > a > b else -1.0 if c < a < b else 0.0
        fu = el = 0.0
        try:
            mf = pfib.main_fib(sub.tail(300))
            fu = (1.0 if mf["up"] else -1.0) if mf.get("ok") else 0.0
        except Exception:                        # noqa: BLE001
            pass
        try:
            ll = pell.live(sub.tail(300)[["High", "Low", "Close"]])
            el = float(ll.get("sgn") or 0)
        except Exception:                        # noqa: BLE001
            pass
        rows[w.index[j]] = {"wk_trend": tr, "wk_fib_up": fu, "wk_ell_sgn": el}
    return pd.DataFrame.from_dict(rows, orient="index")


def timeline(d: pd.DataFrame, since: pd.Timestamp | None = None) -> pd.DataFrame:
    """Her günlük T satırı = T günü KAPANIŞINDAN sonra bilinen yapı (T dahil).
    Kullanıcı D gününde index < D olan son satırı okur (geleceği görmez)."""
    d = d[["Open", "High", "Low", "Close", "Volume"]].astype(float).dropna(subset=["Close"])
    d = d.assign(Volume=d["Volume"].fillna(0.0))
    key = (len(d), str(d.index[0]), str(d.index[-1]), round(float(d["Close"].iloc[-1]), 6),
           round(float(d["Close"].sum()), 4), str(since))
    if key in _CACHE:
        return _CACHE[key]
    start = 60
    if since is not None:
        start = max(start, int(np.searchsorted(d.index.values, np.datetime64(since))) - 2)
    rows = {}
    for j in range(start, len(d)):
        rows[d.index[j]] = _row(d.iloc[max(0, j + 1 - CALC_TAIL):j + 1])
    T = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=COLS)
    try:
        W = _weekly(d)
        if len(W):
            pos = np.searchsorted(W.index.values, T.index.values, side="right") - 1
            for col in ("wk_trend", "wk_fib_up", "wk_ell_sgn"):
                vals = np.full(len(T), np.nan)
                ok = pos >= 0
                vals[ok] = W[col].to_numpy()[pos[ok]]
                T[col] = vals
    except Exception as exc:                     # noqa: BLE001
        _log.info("yapı haftalık: %s", exc)
    if len(_CACHE) > 24:
        _CACHE.clear()
    _CACHE[key] = T
    return T


def align(h_index: pd.DatetimeIndex, T: pd.DataFrame) -> pd.DataFrame:
    """Her 1 saatlik bara ÖNCEKİ günün yapısını eşler."""
    if T.empty:
        return pd.DataFrame(np.nan, index=h_index, columns=COLS)
    Ti = T.copy()
    Ti.index = pd.to_datetime(Ti.index).normalize()
    Ti = Ti[~Ti.index.duplicated(keep="last")].sort_index()
    days = pd.DatetimeIndex(h_index).normalize()
    pos = np.searchsorted(Ti.index.values, days.values, side="left") - 1
    out = pd.DataFrame(np.nan, index=h_index, columns=Ti.columns)
    ok = pos >= 0
    out.loc[ok, :] = Ti.to_numpy()[pos[ok]]
    return out


def hourly_features(close: np.ndarray, A: pd.DataFrame) -> pd.DataFrame:
    """Konsey için yapı özellikleri: 1 saatlik kapanışın dünkü seviyelere göre konumu."""
    c = np.asarray(close, float)
    a = A["atr"].to_numpy(float)
    a = np.where(np.isfinite(a) & (a > 0), a, np.nan)

    def dist(x):                                   # (fiyat − seviye) / ATR
        return (c - A[x].to_numpy(float)) / a

    F = pd.DataFrame(index=A.index)
    ds = dist("s1")
    F["y_near_s1"] = np.where(np.isfinite(ds) & (ds >= -0.3) & (ds <= 1.0), 1.0 - np.clip(ds, 0, 1), 0.0)
    rr = -dist("r1")
    F["y_room_r1"] = np.where(np.isfinite(rr), np.clip(rr, 0, 6) / 6, 1.0)
    gt, gb = A["gb_top"].to_numpy(float), A["gb_bot"].to_numpy(float)
    F["y_in_bgap"] = np.where(np.isfinite(gt) & (c >= gb) & (c <= gt + 0.3 * a),
                              A["gb_tier"].fillna(1).to_numpy(float) / 3, 0.0)
    ga = (A["gr_bot"].to_numpy(float) - c) / a
    F["y_gap_above"] = np.where(np.isfinite(ga) & (ga > 0) & (ga <= 4), 1 - ga / 4, 0.0)
    dt, db = A["dz_top"].to_numpy(float), A["dz_bot"].to_numpy(float)
    F["y_in_demand"] = np.where(np.isfinite(dt) & (c >= db) & (c <= dt + 0.5 * a), 1.0, 0.0)
    sz = (A["sz_bot"].to_numpy(float) - c) / a
    F["y_supply_near"] = np.where(np.isfinite(sz) & (sz >= -0.3) & (sz <= 1.5), 1.0, 0.0)
    F["y_sweep"] = A["sweep"].fillna(0).to_numpy(float)
    F["y_vsa"] = A["vsa_bias"].fillna(0).to_numpy(float)
    sg, kd = A["ell_sgn"].fillna(0).to_numpy(float), A["ell_kind"].fillna(0).to_numpy(float)
    dead = A["ell_dead"].fillna(0).to_numpy(float) > 0
    imp = (kd >= 3) & ~dead
    F["y_ell_bull"] = np.where(imp & (sg > 0), 1.0, 0.0)
    F["y_ell_bear"] = np.where(imp & (sg < 0), 1.0, 0.0)
    F["y_gfr"] = A["gfr_ready"].fillna(0).to_numpy(float)
    F["y_mg_cls"] = A["mg_cls"].map(lambda x: MG_VAL.get(int(x), 0.0) if np.isfinite(x) else 0.0)
    fr = A["fib_r"].to_numpy(float)
    fu = A["fib_up"].to_numpy(float)
    F["y_fib_r"] = np.where((fu > 0) & np.isfinite(fr), np.clip(fr, -0.5, 1.2), 0.0)
    F["y_wk_trend"] = A["wk_trend"].fillna(0).to_numpy(float)
    F["y_wk_fib"] = A["wk_fib_up"].fillna(0).to_numpy(float)
    F["y_wk_ell"] = A["wk_ell_sgn"].fillna(0).to_numpy(float)
    return F.fillna(0.0)


# ------------------------------------------------------------------ robot yardımcıları
def supports(s: dict[str, float], px: float) -> list[tuple[float, str]]:
    """Fiyatın altındaki yapısal seviyeler (yüksekten alçağa)."""
    cand = []
    for key, nm in (("s1", "S1 destek"), ("s2", "S2 destek"), ("gb_bot", "boğa gap kutusu dibi"),
                    ("dz_bot", "Wyckoff talep bölgesi dibi"), ("lq_dn", "dip likidite havuzu")):
        v = s.get(key, np.nan)
        if v is not None and np.isfinite(v) and v < px:
            cand.append((float(v), nm))
    if s.get("ell_sgn", 0) > 0 and s.get("ell_kind", 0) >= 3 and not s.get("ell_dead", 0):
        for key, nm in (("ell_stop", "Elliott teyit seviyesi"), ("ell_inv", "Elliott geçersizlik")):
            v = s.get(key, np.nan)
            if v is not None and np.isfinite(v) and v < px:
                cand.append((float(v), nm))
    return sorted(cand, key=lambda t: -t[0])


def resistances(s: dict[str, float], px: float) -> list[tuple[float, str]]:
    """Fiyatın üstündeki yapısal engeller/hedefler (yakından uzağa)."""
    cand = []
    for key, nm in (("r1", "R1 direnç"), ("r2", "R2 direnç"), ("gr_bot", "açık gap (dolum başı)"),
                    ("gr_top", "açık gap tamamen dolar"), ("sz_bot", "Wyckoff arz bölgesi"),
                    ("lq_up", "tepe likidite havuzu"), ("gfr_top", "GFR boşluk üstü")):
        v = s.get(key, np.nan)
        if v is not None and np.isfinite(v) and v > px:
            cand.append((float(v), nm))
    if s.get("ell_sgn", 0) > 0 and s.get("ell_kind", 0) >= 3 and not s.get("ell_dead", 0):
        for key, nm in (("ell_zlo", "Elliott hedef kutusu"), ("ell_zhi", "Elliott hedef kutusu üstü")):
            v = s.get(key, np.nan)
            if v is not None and np.isfinite(v) and v > px:
                cand.append((float(v), nm))
    return sorted(cand, key=lambda t: t[0])


def blockers(s: dict[str, float], px: float) -> list[tuple[float, str]]:
    """Girişi engelleyebilecek gerçek dirençler (likidite havuzu/hedef kutusu hariç)."""
    out = [t for t in resistances(s, px) if t[1] in ("R1 direnç", "R2 direnç",
                                                      "Wyckoff arz bölgesi")]
    if (s.get("gr_tier") or 0) >= 2:                   # FVG küçük; GAP/KURUMSAL gerçek engel
        out += [t for t in resistances(s, px) if t[1] == "açık gap (dolum başı)"]
    return sorted(out)


def touched_support(s: dict[str, float], lo_win: float, px: float, atr: float) -> str:
    """Son mumların dibi bir yapısal desteğe değip fiyat üstünde kapattıysa adını döndürür."""
    tol = 0.3 * atr
    for key, nm in (("s1", "S1 destek"), ("s2", "S2 destek")):
        v = s.get(key, np.nan)
        if np.isfinite(v) and lo_win <= v + tol and px > v:
            return nm
    gt, gb = s.get("gb_top", np.nan), s.get("gb_bot", np.nan)
    if np.isfinite(gt) and lo_win <= gt + tol and px > gb and px >= lo_win:
        return {3: "KURUMSAL kutu", 2: "boğa GAP kutusu"}.get(int(s.get("gb_tier") or 1), "boğa FVG kutusu")
    dt, db = s.get("dz_top", np.nan), s.get("dz_bot", np.nan)
    if np.isfinite(dt) and lo_win <= dt + tol and px > db:
        return "Wyckoff talep bölgesi"
    if s.get("sweep", 0) > 0:
        return "dip likidite süpürmesi"
    return ""


def vetoes(s: dict[str, float]) -> list[str]:
    """Girişi durduran yapı uyarıları. 50 hisse testinde Elliott yönü ve Mum Gücü dağıtım
    mumu girişlerde OLUMSUZ sonuç öngörmedi (aksine) → veto olarak kaldırıldı; yalnızca
    haftalık düşüş trendi kaldı (n=25, ort. −0.26R)."""
    out = []
    if s.get("wk_trend", 0) < 0:
        out.append("haftalık trend aşağı")
    return out
