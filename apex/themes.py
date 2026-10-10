# apex/themes.py — AETHER APEX
from __future__ import annotations


from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Quadrant:
    key: str
    label: str
    icon: str
    renk: str
    aciklama: str
    aksiyon: str


QUADRANTS: dict[str, Quadrant] = {
    "lider_hizlanan": Quadrant(
        "lider_hizlanan", "Hızlanan lider", "🚀", "#2fbe86",
        "Getiri pozitif VE ivme pozitif. Tema kazandırıyor ve kazandırma hızı "
        "artıyor — para bu temaya yeni giriyor.",
        "Ana avlanma sahası. Swing adaylarını önce burada arayın; trend takip "
        "sistemleri en yüksek isabeti bu çeyrekte verir."),
    "lider_yavaslayan": Quadrant(
        "lider_yavaslayan", "Yavaşlayan lider", "🌤️", "#c98500",
        "Getiri pozitif AMA ivme negatif. Tema hâlâ kazandırıyor, ancak önceki "
        "döneme göre daha yavaş — giriş azalıyor.",
        "Yeni pozisyon için geç kalınmış olabilir. Mevcut pozisyonlarda kısmi "
        "kâr alma ve stop yukarı çekme zamanı."),
    "dipten_donen": Quadrant(
        "dipten_donen", "Dipten dönen", "🌱", "#3987e5",
        "Getiri negatif AMA ivme pozitif. Tema hâlâ ekside, fakat düşüş hızı "
        "kesiliyor — taban oluşumu buradan başlar.",
        "En yüksek getiri potansiyeli burada, en yüksek yanılma payı da. "
        "Rejim kapısı açıkken ve hisse bazında likidite süpürmesi/toplama "
        "sinyali varken anlamlı."),
    "hizlanan_dusus": Quadrant(
        "hizlanan_dusus", "Hızlanan düşüş", "🩸", "#e66767",
        "Getiri negatif VE ivme negatif. Tema kaybettiriyor ve kaybettirme "
        "hızı artıyor — çıkış devam ediyor.",
        "Dip arayışı erken. Bu temadaki long sinyalleri rejim kapısı kapalıyken "
        "gelen sinyaller gibi ele alınmalı: izle, alma."),
}

PERIOD_LABELS: dict[str, str] = {
    "Bugün": "son 1 işlem günü",
    "1H": "son 5 işlem günü",
    "1A": "son 21 işlem günü",
    "3A": "son 63 işlem günü",
    "YBB": "yılbaşından bugüne",
}

PERIOD_PREV: dict[str, str] = {
    "Bugün": "ondan önceki gün",
    "1H": "ondan önceki 5 gün",
    "1A": "ondan önceki 21 gün",
    "3A": "ondan önceki 63 gün",
    "YBB": "geçen yılın aynı dönemi",
}


def classify_quadrant(getiri: float, ivme: float, esik: float = 0.0) -> str:
    """Getiri/ivme ikilisini çeyreğe yerleştirir."""
    if not np.isfinite(getiri) or not np.isfinite(ivme):
        return "hizlanan_dusus" if getiri < 0 else "lider_yavaslayan"
    if getiri >= esik and ivme >= 0:
        return "lider_hizlanan"
    if getiri >= esik and ivme < 0:
        return "lider_yavaslayan"
    if getiri < esik and ivme >= 0:
        return "dipten_donen"
    return "hizlanan_dusus"


def build_table(perf: pd.DataFrame, period: str) -> pd.DataFrame:
    """
    Tema performans tablosuna GETİRİ, İVME ve ÇEYREK sütunlarını ekler.
    `perf`: theme_performance() çıktısı (Bugün/1H/… ve Prev_* sütunları).
    """
    if perf.empty or period not in perf.columns:
        return pd.DataFrame()

    prev_col = f"Prev_{period}"
    out = pd.DataFrame(index=perf.index)
    out["Getiri %"] = perf[period]
    out["Önceki %"] = perf[prev_col] if prev_col in perf.columns else np.nan
    out["İvme"] = out["Getiri %"] - out["Önceki %"]
    out["Çeyrek"] = [
        QUADRANTS[classify_quadrant(g, i)].icon + " " + QUADRANTS[classify_quadrant(g, i)].label
        for g, i in zip(out["Getiri %"], out["İvme"])
    ]
    out["_q"] = [classify_quadrant(g, i) for g, i in zip(out["Getiri %"], out["İvme"])]
    out["Semboller"] = perf["Semboller"] if "Semboller" in perf.columns else ""
    return out.sort_values("Getiri %", ascending=False)


def summary(table: pd.DataFrame) -> dict[str, Any]:
    """Çeyrek bazında özet: hangi temalar nerede."""
    if table.empty:
        return {}
    out: dict[str, Any] = {}
    for key, q in QUADRANTS.items():
        sel = table[table["_q"] == key]
        if sel.empty:
            continue
        out[key] = {
            "quadrant": q,
            "temalar": list(sel.index),
            "n": len(sel),
            "ort_getiri": float(sel["Getiri %"].mean()),
            "ort_ivme": float(sel["İvme"].mean()),
        }
    return out


def worked_example(table: pd.DataFrame, period: str) -> str:
    """
    Gerçek veriden somut bir örnek cümle üretir — açıklama soyut kalmasın.
    En büyük ivme farkına sahip temayı seçer.
    """
    if table.empty or table["İvme"].isna().all():
        return ""
    row = table.loc[table["İvme"].abs().idxmax()]
    tema = row.name
    g, o, i = row["Getiri %"], row["Önceki %"], row["İvme"]
    if not np.isfinite(o):
        return ""
    yon = "hızlanıyor" if i > 0 else "yavaşlıyor"
    q = QUADRANTS[row["_q"]]
    return (
        f"**Örnek — {tema}:** {PERIOD_LABELS.get(period, period)} getirisi "
        f"**%{g:+.2f}**, {PERIOD_PREV.get(period, 'önceki dönem')} getirisi "
        f"**%{o:+.2f}** idi. İvme = {g:+.2f} − ({o:+.2f}) = **{i:+.2f}** → tema "
        f"{yon}. Çeyrek: {q.icon} **{q.label}**. {q.aksiyon}"
    )
