# apex/news.py — AETHER APEX
from __future__ import annotations


import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Iterable

import requests

_log = logging.getLogger(__name__)

FEEDS: dict[str, str] = {
    "Makro & Fed": "Fed OR FOMC OR inflation OR CPI OR rate cut stock market",
    "Tarife & Ticaret": "tariff OR trade war OR export controls chips",
    "Yapay Zekâ & Çip": "AI chips OR semiconductor OR data center capex",
    "Enerji & Nükleer": "nuclear power OR uranium OR grid OR utilities data center",
    "Savunma & Uzay": "defense contract OR space launch OR satellite",
    "Kripto": "bitcoin OR crypto ETF OR SEC crypto",
}

BASE = ("https://news.google.com/rss/search?q={q}"
        "&hl=en-US&gl=US&ceid=US:en")

IMPACT_RULES: list[tuple[tuple[str, ...], str]] = [
    (("tariff", "trade war", "export control", "sanction"),
     "🔴 Tedarik zinciri & Çin ithalatı | 🟢 İç üretim"),
    (("nuclear", "uranium", "smr", "reactor"),
     "🟢 Nükleer & Uranyum (URA, CEG, VST)"),
    (("data center", "hyperscaler", "capex"),
     "🟢 Veri merkezi zinciri (SRVR, XLU, PAVE)"),
    (("oil", "opec", "crude", "gas", "drill"),
     "🟢 Fosil yakıt (XLE, XOP) | 🔴 Temiz enerji"),
    (("crypto", "bitcoin", "sec ", "stablecoin"),
     "🟢 Kripto & Fintek (IBIT, WGMI, ARKF)"),
    (("defense", "military", "missile", "space"),
     "🟢 Savunma & Uzay (XAR, ARKX, UFO)"),
    (("fed", "fomc", "powell", "rate", "inflation", "cpi"),
     "📉 Likidite etkisi — tüm risk varlıkları"),
    (("ai ", "artificial intelligence", "chip", "semiconductor", "gpu"),
     "🟢 Çip & YZ altyapısı (SMH, SOXX, EUV)"),
    (("copper", "lithium", "rare earth", "mining"),
     "🟢 Emtia & Madencilik (COPX, LIT, REMX)"),
    (("layoff", "recession", "slowdown", "downgrade"),
     "🔴 Büyüme endişesi — döngüsel hisseler"),
]


def classify_impact(title: str) -> str:
    t = title.lower()
    for keys, impact in IMPACT_RULES:
        if any(k in t for k in keys):
            return impact
    return "⚖️ Nötr / sektörel rotasyon"


def fetch_news(known_tickers: Iterable[str], topics: Iterable[str] | None = None,
               per_feed: int = 8, timeout: int = 12) -> tuple[list[dict], list[str]]:
    """Seçili konu akışlarını çeker. Döner: (haberler, hatalar)."""
    tickers = sorted({t.upper() for t in known_tickers if t and len(t) >= 2})
    topics = list(topics or FEEDS)
    items: list[dict] = []
    errors: list[str] = []
    seen: set[str] = set()

    for topic in topics:
        q = FEEDS.get(topic)
        if not q:
            continue
        url = BASE.format(q=requests.utils.quote(q))
        try:
            resp = requests.get(url, timeout=timeout,
                                headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
        except Exception as exc:
            errors.append(f"{topic}: {exc}")
            continue

        for node in root.findall(".//item")[:per_feed]:
            title = (node.findtext("title") or "").strip()
            if not title or title in seen:
                continue
            seen.add(title)
            link = node.findtext("link") or ""
            pub = (node.findtext("pubDate") or "")[:22]

            hits = [t for t in tickers
                    if re.search(rf"\b{re.escape(t)}\b", title)]
            items.append({
                "Konu": topic,
                "Tarih": pub,
                "Başlık": title,
                "İlgili": ", ".join(hits[:6]) if hits else "Genel makro",
                "Etki": classify_impact(title),
                "Link": link,
            })

    items.sort(key=lambda r: _parse_date(r["Tarih"]), reverse=True)
    return items, errors


def _parse_date(s: str) -> datetime:
    for fmt in ("%a, %d %b %Y %H:%M:%S", "%a, %d %b %Y %H:%M"):
        try:
            return datetime.strptime(s.strip()[:len(fmt) + 2].strip(), fmt)
        except Exception:
            continue
    return datetime.min
