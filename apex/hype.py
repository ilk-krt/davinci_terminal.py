# apex/hype.py — HYPE & BIOTECH: sosyal medyada konuşulanlar, gelecek değer adayları,
# biyoteknolojiye özel nakit ömrü / katalizör takibi.
#
# Ücretsiz kaynaklar (anahtar gerekmez):
#   • ApeWisdom  — Reddit (wallstreetbets, stocks, investing …) bahsedilme sayıları
#   • StockTwits — trend semboller + "neden trendde" özeti
#   • Yahoo      — ABD trend semboller
# Her kaynak bağımsızdır: biri çekilemezse diğerleriyle devam edilir.
from __future__ import annotations

import logging
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests

_log = logging.getLogger(__name__)
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
      "Accept": "application/json"}

# biyoteknoloji evreni (XBI/IBB büyük bileşenleri + popüler orta/küçük ölçekli)
BIOTECH = ["VRTX", "REGN", "AMGN", "GILD", "BIIB", "MRNA", "BNTX", "ALNY", "SRPT", "INCY",
           "EXEL", "NBIX", "BMRN", "IONS", "CRSP", "BEAM", "NTLA", "VKTX", "MDGL", "AXSM",
           "KRYS", "INSM", "ARGX", "RXRX", "SMMT", "CYTK", "PCVX", "RVMD", "ACAD", "HALO",
           "UTHR", "ROIV", "LEGN", "IMVT", "TGTX", "VERA", "NUVL", "APLS", "ARWR", "CORT",
           "MRUS", "RYTM", "ALT", "TWST", "DNLI", "SANA", "IOVA", "NVAX", "OCGN", "XBI", "IBB"]


def _get(url: str, timeout: int = 15) -> Any:
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    return r.json()


def apewisdom(filt: str = "all-stocks", pages: int = 2) -> pd.DataFrame:
    rows = []
    for p in range(1, pages + 1):
        js = _get(f"https://apewisdom.io/api/v1.0/filter/{filt}/page/{p}")
        rows += js.get("results", [])
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for c in ("rank", "mentions", "upvotes", "rank_24h_ago", "mentions_24h_ago"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    return df.rename(columns={"ticker": "Sembol", "name": "Ad"})


def stocktwits() -> pd.DataFrame:
    js = _get("https://api.stocktwits.com/api/2/trending/symbols.json")
    rows = []
    for s in js.get("symbols", []):
        if s.get("instrument_class") == "CRYPTO" or str(s.get("symbol", "")).endswith(".X"):
            continue
        rows.append({"Sembol": s.get("symbol"), "Ad": s.get("title"),
                     "st_rank": s.get("rank"), "st_score": s.get("trending_score"),
                     "st_ozet": ((s.get("trends") or {}).get("summary") or "")[:400],
                     "Sektör": s.get("sector"), "Endüstri": s.get("industry"),
                     "st_watch": s.get("watchlist_count")})
    return pd.DataFrame(rows)


def yahoo_trending() -> list[str]:
    js = _get("https://query1.finance.yahoo.com/v1/finance/trending/US?count=30")
    res = (js.get("finance") or {}).get("result") or []
    return [q.get("symbol") for q in (res[0].get("quotes", []) if res else []) if q.get("symbol")]


def social(pages: int = 2) -> tuple[pd.DataFrame, list[str]]:
    """Üç kaynağı birleştirir. Döner: (tablo, hatalar)."""
    errs = []
    try:
        ape = apewisdom(pages=pages)
    except Exception as exc:                      # noqa: BLE001
        ape, _ = pd.DataFrame(), errs.append(f"Reddit (ApeWisdom): {exc}")
    try:
        stw = stocktwits()
    except Exception as exc:                      # noqa: BLE001
        stw, _ = pd.DataFrame(), errs.append(f"StockTwits: {exc}")
    try:
        yt = yahoo_trending()
    except Exception as exc:                      # noqa: BLE001
        yt, _ = [], errs.append(f"Yahoo trend: {exc}")
    syms = set(ape["Sembol"]) if not ape.empty else set()
    syms |= set(stw["Sembol"]) if not stw.empty else set()
    syms |= {s for s in yt if "-" not in s and "=" not in s and "^" not in s}
    syms = {s for s in syms if isinstance(s, str) and s.isascii() and 1 <= len(s) <= 5}
    rows = []
    a = ape.set_index("Sembol") if not ape.empty else pd.DataFrame()
    b = stw.set_index("Sembol") if not stw.empty else pd.DataFrame()
    for s in syms:
        r: dict[str, Any] = {"Sembol": s}
        if s in a.index:
            x = a.loc[s]
            r.update({"Ad": x.get("Ad"), "Reddit sıra": x["rank"], "Bahsedilme": x["mentions"],
                      "Bahsedilme 24s önce": x["mentions_24h_ago"], "Upvote": x["upvotes"],
                      "Reddit sıra 24s önce": x["rank_24h_ago"]})
        if s in b.index:
            x = b.loc[s]
            r.update({"Ad": r.get("Ad") or x.get("Ad"), "StockTwits sıra": x["st_rank"],
                      "Neden konuşuluyor": x["st_ozet"], "Sektör": x.get("Sektör")})
        r["Yahoo trend"] = s in yt
        rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        return df, errs
    for c in ("Reddit sıra", "Bahsedilme", "Bahsedilme 24s önce", "StockTwits sıra",
              "Reddit sıra 24s önce", "Upvote"):
        if c not in df:
            df[c] = np.nan
    if "Neden konuşuluyor" not in df:
        df["Neden konuşuluyor"] = ""
    m, m0 = df["Bahsedilme"], df["Bahsedilme 24s önce"]
    df["Bahsedilme Δ"] = np.where((m0 > 0) & m.notna(), (m / m0 - 1) * 100, np.nan)
    # hype skoru 0–100: Reddit sırası, bahsedilme artışı, StockTwits sırası, Yahoo
    s = pd.Series(0.0, index=df.index)
    s += np.where(df["Reddit sıra"].notna(), 40 * (1 - (df["Reddit sıra"].clip(1, 100) - 1) / 100), 0)
    s += np.where(df["StockTwits sıra"].notna(), 30 * (1 - (df["StockTwits sıra"].clip(1, 30) - 1) / 30), 0)
    s += np.where(df["Yahoo trend"], 15, 0)
    s += np.clip(np.nan_to_num(df["Bahsedilme Δ"].to_numpy(float)) / 10, -10, 15)
    df["Hype Skoru"] = s.clip(0, 100).round(0)
    df["Kaynak"] = [" · ".join(x for x in (
        f"Reddit #{int(a_)}" if pd.notna(a_) else "", f"StockTwits #{int(b_)}" if pd.notna(b_) else "",
        "Yahoo trend" if c_ else "") if x) for a_, b_, c_ in
        zip(df["Reddit sıra"], df["StockTwits sıra"], df["Yahoo trend"])]
    return df.sort_values("Hype Skoru", ascending=False).reset_index(drop=True), errs


# --------------------------------------------------------------------------
# temel veriler (Yahoo info) — gelecek değer ve biyoteknoloji nakit ömrü
# --------------------------------------------------------------------------
def fundamentals(syms: Iterable[str], info_fn) -> pd.DataFrame:
    rows = []
    for s in syms:
        try:
            i = info_fn(s) or {}
        except Exception:                          # noqa: BLE001
            i = {}
        px = i.get("currentPrice") or i.get("regularMarketPrice")
        tgt = i.get("targetMeanPrice")
        cash = i.get("totalCash")
        ocf = i.get("operatingCashflow")
        burn_q = (-ocf / 4) if isinstance(ocf, (int, float)) and ocf < 0 else np.nan
        runway = (cash / burn_q) if cash and np.isfinite(burn_q) and burn_q > 0 else np.nan
        rows.append({
            "Sembol": s, "Ad": i.get("shortName"), "Sektör": i.get("sector"),
            "Endüstri": i.get("industry"), "Piyasa değeri ($B)": (i.get("marketCap") or np.nan) / 1e9,
            "Gelir büyümesi %": (i.get("revenueGrowth") or np.nan) * 100,
            "Kâr büyümesi %": (i.get("earningsGrowth") or np.nan) * 100,
            "Brüt marj %": (i.get("grossMargins") or np.nan) * 100,
            "Analist hedef potansiyeli %": ((tgt / px - 1) * 100) if tgt and px else np.nan,
            "Analist sayısı": i.get("numberOfAnalystOpinions"),
            "Tavsiye": i.get("recommendationKey"),
            "Short % float": (i.get("shortPercentOfFloat") or np.nan) * 100,
            "Nakit ($M)": (cash or np.nan) / 1e6,
            "Çeyreklik yakım ($M)": burn_q / 1e6 if np.isfinite(burn_q) else np.nan,
            "Nakit ömrü (çeyrek)": runway,
            "Nakit akışı pozitif": isinstance(ocf, (int, float)) and ocf > 0,
        })
    return pd.DataFrame(rows)


def future_value_note(r: dict) -> tuple[int, str]:
    """0–100 gelecek değer puanı + kısa gerekçe."""
    sc, why = 0, []
    g = r.get("Gelir büyümesi %", np.nan)
    if np.isfinite(g):
        if g >= 40:
            sc += 35; why.append(f"gelir +%{g:.0f}")
        elif g >= 20:
            sc += 25; why.append(f"gelir +%{g:.0f}")
        elif g >= 10:
            sc += 12
        elif g < 0:
            sc -= 10; why.append(f"gelir {g:.0f}%")
    gm = r.get("Brüt marj %", np.nan)
    if np.isfinite(gm) and gm >= 60:
        sc += 10; why.append(f"brüt marj %{gm:.0f}")
    up = r.get("Analist hedef potansiyeli %", np.nan)
    if np.isfinite(up):
        if up >= 30:
            sc += 25; why.append(f"analist hedefi +%{up:.0f}")
        elif up >= 15:
            sc += 15; why.append(f"analist hedefi +%{up:.0f}")
        elif up < 0:
            sc -= 10; why.append("fiyat analist hedefinin üstünde")
    rw = r.get("Nakit ömrü (çeyrek)", np.nan)
    if np.isfinite(rw):
        if rw < 4:
            sc -= 25; why.append(f"⚠️ nakit ömrü {rw:.0f} çeyrek — sermaye artırımı riski")
        elif rw >= 12:
            sc += 10; why.append(f"nakit ömrü {rw:.0f}+ çeyrek")
    elif r.get("Nakit akışı pozitif"):
        sc += 10; why.append("nakit akışı pozitif")
    sf = r.get("Short % float", np.nan)
    if np.isfinite(sf) and sf >= 15:
        why.append(f"short %{sf:.0f} — sıkışma (squeeze) potansiyeli + risk")
    return int(np.clip(sc + 20, 0, 100)), " · ".join(why)


def match_news(items: list[dict], syms: Iterable[str], names: dict[str, str] | None = None
               ) -> dict[str, list[dict]]:
    """Haber başlıklarını sembol / şirket adıyla eşler (katalizör takibi)."""
    import re
    out: dict[str, list[dict]] = {}
    names = names or {}
    for s in syms:
        pats = [rf"\b{re.escape(s)}\b"]
        nm = (names.get(s) or "").split(",")[0].replace(" Inc.", "").replace(" Inc", "").strip()
        if len(nm) >= 4:
            pats.append(re.escape(nm.split(" ")[0]) if len(nm.split(" ")[0]) >= 5 else re.escape(nm))
        for it in items:
            if any(re.search(p, it["Başlık"], flags=re.I if i_ else 0) for i_, p in enumerate(pats)):
                out.setdefault(s, []).append(it)
    return out


CATALYST_WORDS = ("FDA", "approval", "approves", "PDUFA", "Phase 3", "phase 3", "Phase III",
                  "topline", "trial", "data", "CRL", "complete response", "breakthrough",
                  "fast track", "acquire", "acquisition", "buyout", "offering")


def catalyst_tag(title: str) -> str:
    t = title.lower()
    if "complete response" in t or "crl" in t.split():
        return "🔴 CRL (ret)"
    if "approv" in t:
        return "🟢 onay"
    if "pdufa" in t:
        return "📅 PDUFA"
    if "phase 3" in t or "phase iii" in t or "topline" in t:
        return "🧪 faz 3 / sonuç"
    if "acqui" in t or "buyout" in t:
        return "🤝 satın alma"
    if "offering" in t:
        return "⚠️ hisse ihracı"
    if "trial" in t or "data" in t:
        return "🧪 klinik veri"
    return "📰"
