# apex/breadth.py — 🫧 PİYASANIN İÇ YAPISI: genişlik · yoğunlaşma · değerleme ("Boğa-Ayı" analizi)
#
# Mantık (Kobeissi "Bull-Bear Market" analizinin sistematik hâli):
#   1) GETİRİ KATKISI  — bir başlangıçtan (varsayılan: son ~15 ayın en düşük kapanışı = dip)
#      bu yana endeks kazancının ne kadarını hangi hisseler taşıdı? Başlangıç ağırlığı
#      w0 ∝ bugünkü ağırlık / (1 + getiri); katkı = w0 × getiri (endeks puanı).
#      "10 hisse kazancın %70'i" → dar liderlik.
#   2) GENİŞLİK        — üyelerin yüzde kaçı zirvesinin %10 / %20 / %30 / %50 altında (2 yıllık
#      zirve), yüzde kaçı EMA50 / EMA200 üstünde; sektör sektör.
#   3) DEĞERLEME       — endeksin ve sektörlerin kazanç ağırlıklı ileri F/K'sı (Σ piyasa değeri /
#      Σ piyasa değeri ÷ F/K); fiyatı yükselirken F/K'sı düşen ("ucuzlayan lider": ileri F/K,
#      geçmiş F/K'nın ≤ 0.8'i) hisseler.
#   4) REJİM           — endeks zirvesine yakınken üyelerin çoğu ayıdaysa: 🟠 BOĞA-AYI PİYASASI
#      (dar liderlik). Genişlik 10 günde hızla artıyorsa: 🌱 GENİŞLEME BAŞLIYOR — geride kalan
#      sağlam hisselerde yakalama potansiyeli (makaledeki "genişlik biraz artarsa alım baskısı
#      hızla büyür" tezi).
# Not: bu bir bağlam ölçüsüdür; tezin yönü (dar liderlik boğa mı ayı mı) geçmişte test edilmedi.
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "breadth_members.csv"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
INDEXES = {"S&P 500": "SPY", "Nasdaq 100": "QQQ"}
SSGA_SPY = ("https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/"
            "etfs/us/holdings-daily-us-en-spy.xlsx")
WIKI = {"S&P 500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "Nasdaq 100": "https://en.wikipedia.org/wiki/Nasdaq-100"}
SECTOR_TR = {
    "Information Technology": "Teknoloji", "Technology": "Teknoloji",
    "Health Care": "Sağlık", "Healthcare": "Sağlık", "Financials": "Finans",
    "Financial Services": "Finans", "Consumer Discretionary": "İhtiyari tüketim",
    "Consumer Cyclical": "İhtiyari tüketim", "Communication Services": "İletişim",
    "Industrials": "Sanayi", "Consumer Staples": "Temel tüketim",
    "Consumer Defensive": "Temel tüketim", "Energy": "Enerji", "Utilities": "Kamu hizmetleri",
    "Real Estate": "Gayrimenkul", "Materials": "Materyal", "Basic Materials": "Materyal"}


def _yf_sym(t: str) -> str:
    return str(t).strip().upper().replace(".", "-").replace(" ", "")


# ------------------------------------------------------------------ üyeler
def _ssga_spy() -> pd.DataFrame:
    import requests
    r = requests.get(SSGA_SPY, headers=UA, timeout=30)
    r.raise_for_status()
    raw = pd.read_excel(io.BytesIO(r.content), header=None)
    hdr = next(i for i in range(min(len(raw), 15))
               if "Ticker" in [str(x).strip() for x in raw.iloc[i].tolist()])
    df = raw.iloc[hdr + 1:].copy()
    df.columns = [str(x).strip() for x in raw.iloc[hdr].tolist()]
    df = df[pd.to_numeric(df.get("Weight"), errors="coerce").notna()]
    df = df[df["Ticker"].astype(str).str.match(r"^[A-Za-z][A-Za-z.\- ]{0,6}$")]
    out = pd.DataFrame({"Sembol": df["Ticker"].map(_yf_sym),
                        "Şirket": df.get("Name", ""),
                        "Sektör": df.get("Sector", "").map(lambda x: SECTOR_TR.get(str(x), str(x))),
                        "Ağırlık": pd.to_numeric(df["Weight"], errors="coerce")})
    out = out[~out["Sembol"].isin(["CASH", "USD", "-"])]
    return out.drop_duplicates("Sembol").reset_index(drop=True)


def _wiki(name: str) -> pd.DataFrame:
    import requests
    r = requests.get(WIKI[name], headers=UA, timeout=30)
    r.raise_for_status()
    tabs = pd.read_html(io.StringIO(r.text))
    for t in tabs:
        cols = [str(c) for c in t.columns]
        sym = next((c for c in cols if c in ("Symbol", "Ticker")), None)
        sec = next((c for c in cols if "Sector" in c), None)
        if sym and len(t) >= (400 if name == "S&P 500" else 90):
            comp = next((c for c in cols if c in ("Security", "Company")), None)
            return pd.DataFrame({
                "Sembol": t[sym].map(_yf_sym),
                "Şirket": t[comp] if comp else "",
                "Sektör": t[sec].map(lambda x: SECTOR_TR.get(str(x), str(x))) if sec else "",
                "Ağırlık": np.nan}).drop_duplicates("Sembol").reset_index(drop=True)
    raise RuntimeError(f"{name}: Wikipedia tablosu bulunamadı")


def members(name: str) -> tuple[pd.DataFrame, str]:
    """Endeks üyeleri → (tablo, kaynak). Sıra: SSGA (yalnız SPY, ağırlıklı) → Wikipedia →
    son başarılı liste (data/breadth_members.csv) → ETF'in bilinen bileşenleri."""
    tries = ([("SSGA SPY dosyası", _ssga_spy)] if name == "S&P 500" else []) + \
        [("Wikipedia", lambda: _wiki(name))]
    for src, fn in tries:
        try:
            df = fn()
            if len(df) >= (400 if name == "S&P 500" else 90):
                _save_cache(name, df)
                return df, src
        except Exception as exc:                 # noqa: BLE001
            _log.info("üyeler %s %s: %s", name, src, exc)
    if CACHE.exists():
        try:
            c = pd.read_csv(CACHE)
            c = c[c["Endeks"] == name].drop(columns=["Endeks"])
            if len(c):
                return c.reset_index(drop=True), "son kayıtlı liste"
        except Exception as exc:                 # noqa: BLE001
            _log.info("üye önbelleği: %s", exc)
    from apex import universe as uni
    syms = uni.holdings(INDEXES[name])
    return pd.DataFrame({"Sembol": syms, "Şirket": "", "Sektör": "", "Ağırlık": np.nan}), \
        "ETF'in bilinen bileşenleri (eksik liste)"


def _save_cache(name: str, df: pd.DataFrame) -> None:
    try:
        old = pd.read_csv(CACHE) if CACHE.exists() else pd.DataFrame()
        if len(old):
            old = old[old["Endeks"] != name]
        new = pd.concat([old, df.assign(Endeks=name)], ignore_index=True)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        new.to_csv(CACHE, index=False)
    except Exception as exc:                     # noqa: BLE001
        _log.info("üye önbelleği yazılamadı: %s", exc)


# ------------------------------------------------------------------ hesap
def wide_close(prices: dict[str, pd.DataFrame], syms) -> pd.DataFrame:
    cols = {}
    for s in syms:
        d = prices.get(s)
        if d is not None and len(d) > 60:
            c = d["Close"].astype("float32")
            c.index = pd.to_datetime(c.index).tz_localize(None) if getattr(
                c.index, "tz", None) else pd.to_datetime(c.index)
            cols[s] = c[~c.index.duplicated(keep="last")]
    return pd.DataFrame(cols).sort_index()


def auto_start(idx_close: pd.Series, look: int = 320) -> pd.Timestamp:
    """Varsayılan başlangıç: son ~15 ayın en düşük kapanışı (dip). Dip son 20 günse
    (yeni düşüş), 1 yıl öncesi."""
    s = idx_close.dropna().tail(look)
    t = s.idxmin()
    if len(s) - s.index.get_loc(t) < 20:
        t = s.index[max(0, len(s) - 252)]
    return pd.Timestamp(t)


def weights(M: pd.DataFrame, fund: pd.DataFrame | None) -> pd.Series:
    """Bugünkü ağırlık: SSGA ağırlığı, yoksa piyasa değeri (temel veriden)."""
    w = M.set_index("Sembol")["Ağırlık"].astype(float)
    if w.notna().sum() < len(w) * 0.8 and fund is not None and len(fund):
        mc = fund.set_index("Hisse")["Piyasa Değ. ($B)"].astype(float)
        w = mc.reindex(w.index)
    return (w / w.sum() * 100).fillna(0.0)


def contribution(C: pd.DataFrame, w_now: pd.Series, start) -> tuple[pd.DataFrame, dict]:
    sub = C[C.index >= pd.Timestamp(start)]
    if len(sub) < 2:
        return pd.DataFrame(), {}
    first = sub.bfill().iloc[0]
    last = sub.ffill().iloc[-1]
    r = (last / first - 1).astype(float)
    w = w_now.reindex(r.index).fillna(0.0)
    w0 = w / (1 + r.fillna(0))
    w0 = w0 / w0.sum()
    k = (w0 * r.fillna(0) * 100)
    T = pd.DataFrame({"Sembol": k.index, "Getiri %": r.values * 100, "Ağırlık %": w.values,
                      "Katkı (puan)": k.values}).dropna(subset=["Getiri %"])
    tot = float(T["Katkı (puan)"].sum())
    T["Kazancın payı %"] = T["Katkı (puan)"] / tot * 100 if tot > 0 else np.nan
    T = T.sort_values("Katkı (puan)", ascending=False).reset_index(drop=True)
    top10 = float(T["Katkı (puan)"].head(10).sum())
    info = {"endeks_getiri": tot, "top10_puan": top10,
            "top10_pay": top10 / tot * 100 if tot > 0 else np.nan,
            "n": len(T), "pozitif %": float((T["Getiri %"] > 0).mean() * 100),
            "endeksi_gecen %": float((T["Getiri %"] > tot).mean() * 100),
            "baslangic": str(sub.index[0].date())}
    return T, info


def breadth_now(C: pd.DataFrame, M: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    last = C.ffill().iloc[-1]
    dd = (last / C.max() - 1) * 100
    e50 = C.ewm(span=50, adjust=False).mean().iloc[-1]
    e200 = C.ewm(span=200, adjust=False).mean().iloc[-1]
    X = pd.DataFrame({"Zirveden %": dd, "EMA50 üstü": last > e50, "EMA200 üstü": last > e200})
    X["Sektör"] = M.set_index("Sembol")["Sektör"].reindex(X.index).fillna("—").replace("", "—")
    X = X.dropna(subset=["Zirveden %"])
    info = {f"≥%{k} düşük": float((X["Zirveden %"] <= -k).mean() * 100) for k in (10, 20, 30, 50)}
    info["EMA50 üstü %"] = float(X["EMA50 üstü"].mean() * 100)
    info["EMA200 üstü %"] = float(X["EMA200 üstü"].mean() * 100)
    info["medyan düşüş %"] = float(X["Zirveden %"].median())
    S = X.groupby("Sektör").agg(
        Hisse=("Zirveden %", "size"),
        **{"≥%10 düşük": ("Zirveden %", lambda x: (x <= -10).mean() * 100),
           "≥%20 düşük": ("Zirveden %", lambda x: (x <= -20).mean() * 100),
           "Medyan düşüş %": ("Zirveden %", "median"),
           "EMA200 üstü %": ("EMA200 üstü", lambda x: x.mean() * 100)}).reset_index()
    S = S.sort_values("≥%10 düşük", ascending=False).reset_index(drop=True)
    return S, info


def breadth_series(C: pd.DataFrame, idx_close: pd.Series | None) -> pd.DataFrame:
    e50 = C.ewm(span=50, adjust=False).mean()
    e200 = C.ewm(span=200, adjust=False).mean()
    valid = C.notna()
    out = pd.DataFrame({
        "EMA50 üstü %": ((C > e50) & valid).sum(axis=1) / valid.sum(axis=1) * 100,
        "EMA200 üstü %": ((C > e200) & valid).sum(axis=1) / valid.sum(axis=1) * 100})
    roll_hi = C.cummax()
    out["≥%20 düşük %"] = ((C / roll_hi - 1) <= -0.20).sum(axis=1) / valid.sum(axis=1) * 100
    if idx_close is not None:
        ic = idx_close.reindex(out.index).ffill()
        out["Endeks"] = ic
        out["Endeks zirveden %"] = (ic / ic.cummax() - 1) * 100
    return out.iloc[200:] if len(out) > 260 else out


def valuation(M: pd.DataFrame, fund: pd.DataFrame | None, r: pd.Series | None = None
              ) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Kazanç ağırlıklı ileri F/K: endeks ve sektör + ucuzlayan liderler."""
    if fund is None or fund.empty:
        return pd.DataFrame(), {}, pd.DataFrame()
    F = M.merge(fund, left_on="Sembol", right_on="Hisse", how="left")
    mc = pd.to_numeric(F["Piyasa Değ. ($B)"], errors="coerce")
    fpe = pd.to_numeric(F["İleri F/K"], errors="coerce")
    tpe = pd.to_numeric(F.get("Geçmiş F/K"), errors="coerce")
    ok = mc.notna() & fpe.notna() & (fpe > 0) & (fpe < 400)

    def wpe(m):
        m = m & ok
        return float(mc[m].sum() / (mc[m] / fpe[m]).sum()) if m.sum() >= 3 else np.nan
    idx_pe = wpe(pd.Series(True, index=F.index))
    rows = []
    for sec, g in F.groupby("Sektör"):
        m = F["Sektör"] == sec
        rows.append({"Sektör": sec or "—", "İleri F/K": wpe(m),
                     "Endekse göre": wpe(m) / idx_pe if np.isfinite(idx_pe) else np.nan,
                     "Veri (hisse)": int((m & ok).sum())})
    V = pd.DataFrame(rows).sort_values("İleri F/K", ascending=False).reset_index(drop=True)
    tech = V.loc[V["Sektör"] == "Teknoloji", "İleri F/K"]
    info = {"endeks_fk": idx_pe, "tek_fk": float(tech.iloc[0]) if len(tech) else np.nan,
            "kapsama %": float(ok.mean() * 100)}
    # ucuzlayan liderler: ileri F/K ≤ 0.8 × geçmiş F/K (kazanç fiyattan hızlı büyüyor)
    F["Getiri %"] = (r.reindex(F["Sembol"]).values * 100) if r is not None else np.nan
    F["F/K sıkışması"] = fpe / tpe
    U = F[(F["F/K sıkışması"] <= 0.8) & (fpe > 0) & (F["Getiri %"] > 0)]
    U = U.assign(**{"Piyasa Değ. ($B)": mc[U.index]}).sort_values("Piyasa Değ. ($B)",
                                                                   ascending=False)
    U = U[["Sembol", "Sektör", "Getiri %", "İleri F/K", "Geçmiş F/K", "F/K sıkışması",
           "Piyasa Değ. ($B)"]].head(25).reset_index(drop=True)
    return V, info, U


def regime(br: dict, idx_dd: float, contrib: dict, series: pd.DataFrame | None
           ) -> tuple[str, str, str]:
    """→ (kod, etiket, açıklama)."""
    up50 = np.nan
    if series is not None and len(series) > 11:
        up50 = float(series["EMA50 üstü %"].iloc[-1] - series["EMA50 üstü %"].iloc[-11])
    near_hi = np.isfinite(idx_dd) and idx_dd >= -5
    narrow = (br.get("≥%20 düşük", 0) >= 40 or br.get("EMA200 üstü %", 100) < 50
              or (contrib.get("top10_pay", 0) or 0) >= 50)
    if near_hi and narrow and np.isfinite(up50) and up50 >= 10:
        return ("genisleme", "🌱 Genişleme başlıyor",
                f"Endeks zirveye yakın, liderlik dardı; ama son 10 günde EMA50 üstündeki hisse "
                f"oranı {up50:+.0f} puan arttı. Geride kalan sağlam hisselere alım baskısı "
                "yayılıyor olabilir — genişleme adayları öne çıkar.")
    if near_hi and narrow:
        return ("boga_ayi", "🟠 Boğa-ayı piyasası (dar liderlik)",
                "Endeks zirvesine yakın ama üyelerin çoğu ayı bölgesinde; kazancı birkaç dev "
                "taşıyor. Endeksi taşıyan liderler (özellikle F/K'sı düşenler) güçlü kalır; "
                "genişlik artarsa geride kalan sağlam hisselerde hızlı yakalama olur. "
                "Genişlik sinyalini (EMA50 üstü oranında hızlı artış) bekleyin.")
    if near_hi:
        return ("genis_boga", "🟢 Geniş katılımlı boğa",
                "Endeks zirveye yakın ve üyelerin çoğu da yükseliyor — sağlıklı trend.")
    if np.isfinite(idx_dd) and idx_dd <= -10 and br.get("EMA200 üstü %", 100) < 40:
        return ("genis_ayi", "🔴 Geniş ayı",
                "Endeks ve üyelerin çoğu düşüşte — risk iştahı kapalı; seçici olun.")
    return ("karisik", "⚪ Karışık", "Endeks zirveden uzak, genişlik orta seviyede.")


def analyze(C: pd.DataFrame, M: pd.DataFrame, idx_close: pd.Series,
            fund: pd.DataFrame | None, start=None) -> dict[str, Any]:
    start = pd.Timestamp(start) if start is not None else auto_start(idx_close)
    w = weights(M, fund)
    K, ki = contribution(C, w, start)
    S, bi = breadth_now(C, M)
    ser = breadth_series(C, idx_close)
    ic = idx_close.dropna()
    idx_dd = float((ic.iloc[-1] / ic.max() - 1) * 100) if len(ic) else np.nan
    sub = ic[ic.index >= start]
    ki["endeks_gercek"] = float((sub.iloc[-1] / sub.iloc[0] - 1) * 100) if len(sub) > 1 else np.nan
    r = (K.set_index("Sembol")["Getiri %"] / 100) if len(K) else None
    V, vi, U = valuation(M, fund, r)
    kod, lab, txt = regime(bi, idx_dd, ki, ser)
    return {"start": start, "katki": K, "katki_info": ki, "sektor": S, "genislik": bi,
            "seri": ser, "deger": V, "deger_info": vi, "ucuzlayan": U, "endeks_dd": idx_dd,
            "rejim": kod, "rejim_etiket": lab, "rejim_txt": txt}


# ------------------------------------------------------------------ temalar (seçki verisinden)
def theme_breadth(B: pd.DataFrame, idx_pe: float = np.nan) -> pd.DataFrame:
    """Tema iç yapısı: genişlik, yoğunlaşma (60G katkıda ilk 3'ün payı), değerleme."""
    if B is None or B.empty:
        return pd.DataFrame()
    rows = []
    for tema, g in B.drop_duplicates(["Tema", "Hisse"]).groupby("Tema", sort=False):
        mc = pd.to_numeric(g["Piyasa Değ. ($B)"], errors="coerce")
        r60 = pd.to_numeric(g["Getiri 60G %"], errors="coerce") / 100
        w = mc.fillna(mc.median() if mc.notna().any() else 1.0).fillna(1.0)
        w0 = w / (1 + r60.fillna(0))
        w0 = w0 / w0.sum()
        k = (w0 * r60.fillna(0) * 100)
        tot = float(k.sum())
        top3 = float(k.sort_values(ascending=False).head(3).sum())
        fpe = pd.to_numeric(g["İleri F/K"], errors="coerce")
        okp = fpe.notna() & (fpe > 0) & (fpe < 400) & mc.notna()
        t_pe = float(mc[okp].sum() / (mc[okp] / fpe[okp]).sum()) if okp.sum() >= 3 else np.nan
        dd = pd.to_numeric(g["Zirveden %"], errors="coerce")
        p200 = float(g["_ust200"].astype(bool).mean() * 100)
        p50 = float(g["_ust50"].astype(bool).mean() * 100) if "_ust50" in g else np.nan
        p20 = float((dd <= -20).mean() * 100)
        if tot > 0 and top3 / tot >= 0.7 and p20 >= 40:
            dur = "🟠 Dar liderlik — birkaç hisse taşıyor"
        elif np.isfinite(p50) and p50 - p200 >= 15 and p200 < 60:
            dur = "🌱 Genişleme başlıyor (EMA50 üstü oranı EMA200'ü geçti)"
        elif p200 >= 60 and p20 < 30:
            dur = "🟢 Geniş katılım"
        elif p200 < 35:
            dur = "🔴 Çoğu ayıda"
        else:
            dur = "⚪ Karışık"
        rows.append({"Tema": tema, "Hisse": len(g), "İç yapı": dur,
                     "60G tema %": tot, "İlk 3'ün payı %": top3 / tot * 100 if tot > 0 else np.nan,
                     "≥%20 düşük %": p20, "EMA50 üstü %": p50, "EMA200 üstü %": p200,
                     "İleri F/K": t_pe,
                     "Endekse göre F/K": t_pe / idx_pe if np.isfinite(idx_pe) and idx_pe else np.nan,
                     "Genişleme adayı": int(g["_genis"].astype(bool).sum()) if "_genis" in g else 0,
                     "Ucuzlayan lider": int(g["_ucuzlayan"].astype(bool).sum())
                     if "_ucuzlayan" in g else 0})
    return pd.DataFrame(rows)
