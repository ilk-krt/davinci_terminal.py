# apex/gold.py — ALTIN & GÜMÜŞ: fiyatı hareket ettiren zincir
#
# Zincir: ABD verisi → Fed beklentisi → tahvil faizi ve dolar → altın.
# Önem sırası (kullanıcının notlarından): 1) reel faiz / 10Y  2) dolar
# 3) Fed beklentisi (kısa vadeli faiz)  4) tahvil oynaklığı (MOVE)  5) petrol
# (çift taraflı)  6) fon (ETF) talebi  7) gümüş/bakır sinyalleri  8) gram altın.
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

TICKERS = {"GOLD": "GC=F", "SILVER": "SI=F", "US10Y": "^TNX", "TIP": "TIP",
           "DXY": "DX-Y.NYB", "SHORT": "^IRX", "MOVE": "^MOVE", "OIL": "BZ=F",
           "GLD": "GLD", "COPPER": "HG=F", "USDTRY": "TRY=X"}


def _ch(s: pd.Series | None, n: int) -> float:
    if s is None:
        return np.nan
    s = s.dropna()
    return float((s.iloc[-1] / s.iloc[-1 - n] - 1) * 100) if len(s) > n else np.nan


def _last(s):
    return float(s.dropna().iloc[-1]) if s is not None and len(s.dropna()) else np.nan


def drivers(prices: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Döner: (gösterge tablosu, özet). 'Altın için' sütunu: 🟢 destek · 🔴 baskı."""
    c = {k: (prices.get(t)["Close"] if prices.get(t) is not None and not prices.get(t).empty
             else None) for k, t in TICKERS.items()}
    rows: list[dict[str, Any]] = []

    def add(rank, name, val, ch20, effect, rule, fmt="{:.2f}"):
        rows.append({"Sıra": rank, "Gösterge": name,
                     "Son": fmt.format(val) if np.isfinite(val) else "—",
                     "20G değişim": ch20, "Altın için": effect, "Kural": rule})

    # 1) 10Y ve reel faiz (TIP fiyatı ters vekil: TIP düşüyorsa reel faiz yükseliyor)
    tnx = c["US10Y"]
    d10 = (_last(tnx) - float(tnx.dropna().iloc[-21])) if tnx is not None and len(tnx.dropna()) > 21 else np.nan
    add(1, "ABD 10Y faiz (%)", _last(tnx), d10,
        "🔴 baskı" if d10 > 0.1 else "🟢 destek" if d10 < -0.1 else "⚪ nötr",
        "Faiz yükselirse altın baskılanır, düşerse rahatlar (20G değişim: puan).")
    tip20 = _ch(c["TIP"], 20)
    add(1, "Reel faiz vekili (TIP fonu)", _last(c["TIP"]), tip20,
        "🔴 baskı" if tip20 < -0.5 else "🟢 destek" if tip20 > 0.5 else "⚪ nötr",
        "TIP fiyatı düşüyorsa reel faiz yükseliyor → altına ters. Bir numaralı pusula.")
    # 2) dolar
    dxy20 = _ch(c["DXY"], 20)
    add(2, "Dolar endeksi (DXY)", _last(c["DXY"]), dxy20,
        "🔴 baskı" if dxy20 > 0.7 else "🟢 destek" if dxy20 < -0.7 else "⚪ nötr",
        "DXY yukarı → altın aşağı. Faiz ile dolar ters sinyal verirse faiz kazanır.")
    # 3) Fed beklentisi (3 aylık hazine faizi)
    irx = c["SHORT"]
    dirx = (_last(irx) - float(irx.dropna().iloc[-21])) if irx is not None and len(irx.dropna()) > 21 else np.nan
    add(3, "Kısa vade faiz (3A, Fed beklentisi)", _last(irx), dirx,
        "🔴 baskı" if dirx > 0.1 else "🟢 destek" if dirx < -0.1 else "⚪ nötr",
        "Artırım beklentisi yükselirse altına ters, indirim beklentisi dosttur.")
    # 4) MOVE
    mv20 = _ch(c["MOVE"], 20)
    add(4, "Tahvil oynaklığı (MOVE)", _last(c["MOVE"]), mv20,
        "⚠️ sert dalgalanma" if mv20 > 10 else "⚪ sakin",
        "MOVE yükseliyorsa tahvil piyasası geriliyor → altında sert hareket habercisi.")
    # 5) petrol (çift taraflı)
    oil20 = _ch(c["OIL"], 20)
    add(5, "Petrol (Brent)", _last(c["OIL"]), oil20,
        "⚖️ çift taraflı" if abs(oil20) > 5 else "⚪ nötr",
        "Enflasyon korkusu yaratırsa destekler; faiz artırımına dönüşürse baskılar.")
    # 6) fon (ETF) talebi: GLD dolar hacmi trendi + fiyat yönü
    g = prices.get("GLD")
    if g is not None and not g.empty and len(g) > 60:
        dv = (g["Close"] * g["Volume"]).fillna(0)
        vr = dv.tail(10).mean() / max(dv.tail(60).mean(), 1)
        gp = _ch(g["Close"], 10)
        eff = ("🟢 fon girişi" if vr > 1.15 and gp > 0 else
               "🔴 fon çıkışı" if vr > 1.15 and gp < 0 else "⚪ normal")
        add(6, "Altın fonu talebi (GLD hacim × yön)", vr, gp, eff,
            "Fonlar hızlı ve duygusaldır: dalgayı belirler. (Son: hacim oranı, "
            "değişim: 10G fiyat %)")
    # 7) altın/gümüş ve bakır/altın
    if c["GOLD"] is not None and c["SILVER"] is not None:
        gs = (c["GOLD"] / c["SILVER"]).dropna()
        gs20 = _ch(gs, 20)
        add(7, "Altın/Gümüş oranı", _last(gs), gs20,
            "⚪ korku artıyor" if gs20 > 3 else "🟢 metal iştahı" if gs20 < -3 else "⚪ nötr",
            "Oran düşüyorsa gümüş önde → değerli metallerde iştah geniş.")
    if c["COPPER"] is not None and c["GOLD"] is not None:
        cg = (c["COPPER"] / c["GOLD"]).dropna()
        cg20 = _ch(cg, 20)
        add(7, "Bakır/Altın oranı", _last(cg) * 1000, cg20,
            "🔴 faiz yukarı baskısı" if cg20 > 4 else "🟢 büyüme endişesi → altın" if cg20 < -4 else "⚪ nötr",
            "Bakır/altın yükseliyorsa ekonomi ısınıyor ve faizler yukarı → altına ters.")
    # 8) gram altın (TL)
    if c["GOLD"] is not None and c["USDTRY"] is not None:
        gram = (c["GOLD"] / 31.1035 * c["USDTRY"]).dropna()
        add(8, "Gram altın (TL)", _last(gram), _ch(gram, 20), "🇹🇷 ons × kur",
            "Gram = ons ÷ 31,1 × USD/TRY. Ons yatayken kur yükselirse gram yükselir.",
            fmt="{:,.0f}")
        add(8, "USD/TRY", _last(c["USDTRY"]), _ch(c["USDTRY"], 20), "🇹🇷 kur",
            "Gram altının ikinci bileşeni.", fmt="{:.2f}")

    df = pd.DataFrame(rows)
    core = df[df["Sıra"] <= 3]
    plus = int(core["Altın için"].str.startswith("🟢").sum())
    minus = int(core["Altın için"].str.startswith("🔴").sum())
    if plus >= 2 and minus == 0:
        hk = ("🟢 DESTEKLİ", "Zincirin ilk üç halkası (faiz, dolar, Fed beklentisi) altını "
              "destekliyor — yön güçlü.")
    elif minus >= 2 and plus == 0:
        hk = ("🔴 BASKI ALTINDA", "Faiz/dolar/Fed beklentisi altına ters çalışıyor — "
              "yükselişler satış baskısıyla karşılaşabilir.")
    else:
        hk = ("⚖️ KARARSIZ", "Göstergeler çelişiyor; kararsız piyasada en doğru pozisyon "
              "çoğu zaman beklemektir.")
    return df, {"etiket": hk[0], "ozet": hk[1], "plus": plus, "minus": minus}


CALENDAR = [
    ("Her gün", "ABD 10Y faiz, dolar endeksi, ons ve USD/TRY"),
    ("Her Perşembe", "ABD haftalık işsizlik başvuruları"),
    ("Her Cuma", "CFTC COT raporu (vadeli pozisyonlar) · ayın ilk Cuması tarım dışı "
                 "istihdam (TSİ 15:30, Kasım'dan itibaren 16:30)"),
    ("Ayın başı", "Çin Merkez Bankası altın rezervleri · PMI / ISM"),
    ("Ayın ortası", "ABD TÜFE (CPI) — özellikle çekirdek · ÜFE (PPI)"),
    ("Ay sonu", "PCE — Fed'in en çok önemsediği enflasyon ölçüsü"),
    ("Fed toplantıları", "Karar, metin ve basın toplantısı (notlarınıza göre 28 Ekim "
                         "21:00 ve 9 Aralık 22:00 TSİ; Aralık'ta nokta grafiği)"),
    ("Her çeyrek", "Dünya Altın Konseyi talep raporu"),
]
