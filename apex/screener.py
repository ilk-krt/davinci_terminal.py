# apex/screener.py — AETHER APEX
from __future__ import annotations


from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# İşlem karakteri sınıfları — ATR%'ye göre
KARAKTER_ESIK = {
    "yuksek_beta": 3.5,     # ATR% >= 3.5 -> geniş ATR, yüksek beta
    "orta": 1.8,            # 1.8–3.5 -> temiz trend, orta volatilite
}                            # < 1.8 -> sakin / defansif


@dataclass
class SwingFilters:
    """Kullanıcının pratik filtreleri — arayüzden açılıp kapatılır."""
    min_dollar_vol_m: float = 5.0      # günlük ortalama işlem hacmi ($M)
    earnings_buffer_days: int = 2      # bilanço ±N gün pozisyon açma
    require_regime: bool = True        # SPY 50 EMA rejim kapısı
    require_weekly: bool = False       # haftalık trend teyidi
    min_price: float = 5.0             # penny hisse eleme
    max_atr_pct: float = 12.0          # aşırı oynak olanları ele
    min_score: int = 0


def trade_character(atr_pct: float) -> str:
    if not np.isfinite(atr_pct):
        return "—"
    if atr_pct >= KARAKTER_ESIK["yuksek_beta"]:
        return "⚡ Yüksek beta / geniş ATR"
    if atr_pct >= KARAKTER_ESIK["orta"]:
        return "📈 Temiz trend / orta volatilite"
    return "🛡️ Sakin / defansif"


def character_note(atr_pct: float) -> str:
    if not np.isfinite(atr_pct):
        return ""
    if atr_pct >= KARAKTER_ESIK["yuksek_beta"]:
        return ("Kademeli ATR stop sistemi bu grupta anlamlı çalışır; hareket "
                "geniş olduğu için stop mesafesi de geniş tutulmalı.")
    if atr_pct >= KARAKTER_ESIK["orta"]:
        return ("Sinyal-gürültü oranı iyi, whipsaw az. Trend takip ve kırılım "
                "sistemleri bu bantta en verimli çalışır.")
    return ("Hareket dar; swing için getiri/risk zayıf kalabilir. Pozisyon "
            "boyutunu büyütmek yerine daha oynak bir aday aramak daha mantıklı.")


def swing_score(d: dict[str, Any]) -> int:
    """
    0–100 swing uygunluk skoru.
    Ağırlıklar V719 KONFLUANS'ın ölçülmüş lift değerleriyle uyumlu tutuldu:
    en yüksek katkı Afterburner ve MVP tarafında, teyit katmanları daha düşük.
    """
    s = 0.0
    # Konfluans motoru (en ağır bileşen)
    s += np.clip(d.get("MAGNITUDE", 0), 0, 18) / 18 * 22
    s += np.clip((d.get("DIRECTION", 0) + 5) / 10, 0, 1) * 18
    # Kurumsal akış
    s += np.clip(d.get("WHALE", 50), 0, 100) / 100 * 14
    s += np.clip((d.get("PRO-RET", 0) + 50) / 100, 0, 1) * 8
    # Momentum konsensüsü
    s += np.clip(d.get("OMNI", 50), 0, 100) / 100 * 10
    s += np.clip(d.get("Boğa /6", 0), 0, 6) / 6 * 8
    s += np.clip(d.get("Efor /8", 0), 0, 8) / 8 * 6
    # Göreli güç
    rs = d.get("RS Sıra", np.nan)
    if np.isfinite(rs):
        s += rs / 100 * 8
    # Olay primleri
    if d.get("_dia_buy"):
        s += 6
    if d.get("_sweep"):
        s += 5
    if d.get("_star"):
        s += 4
    if d.get("_ab_bull"):
        s += 4
    if d.get("_mvp"):
        s += 4
    if d.get("Sıkışma"):
        s += 3
    # Cezalar
    if d.get("_risk_hard"):
        s -= 25
    elif d.get("_risk"):
        s -= 12
    if d.get("_exhausted"):
        s -= 8
    if d.get("_dist") or d.get("_st_out"):
        s -= 10
    if not d.get("Rejim", True):
        s -= 6
    return int(np.clip(round(s), 0, 100))


def apply_filters(d: dict[str, Any], f: SwingFilters,
                  earnings_days: int | None,
                  market_regime_ok: bool) -> tuple[bool, list[str]]:
    """Filtreleri uygular. Döner: (geçti mi, [engel gerekçeleri])."""
    blocks: list[str] = []

    price = d.get("Fiyat", np.nan)
    if np.isfinite(price) and price < f.min_price:
        blocks.append(f"Fiyat ${price:.2f} < ${f.min_price:.0f}")

    dv = d.get("Hacim ($M)", np.nan)
    if np.isfinite(dv) and dv < f.min_dollar_vol_m:
        blocks.append(f"Likidite ${dv:.1f}M < ${f.min_dollar_vol_m:.0f}M "
                      f"(spread genişler, stop kayar)")

    atrp = d.get("ATR %", np.nan)
    if np.isfinite(atrp) and atrp > f.max_atr_pct:
        blocks.append(f"ATR %{atrp:.1f} aşırı oynak")

    if earnings_days is not None and abs(earnings_days) <= f.earnings_buffer_days:
        blocks.append(f"Bilanço {earnings_days} gün içinde — gap riski stop "
                      f"mantığını bozar")

    if f.require_regime and not market_regime_ok:
        blocks.append("Piyasa rejimi kapalı (SPY 50 EMA altında)")

    if f.require_regime and not d.get("Rejim", True):
        blocks.append("Hisse 200 EMA altında")

    if f.require_weekly and d.get("Haftalık") is False:
        blocks.append("Haftalık trend teyidi yok")

    if d.get("Skor", 0) < f.min_score:
        blocks.append(f"Skor {d.get('Skor', 0)} < {f.min_score}")

    return (not blocks), blocks


def build_recommendations(rows: list[dict[str, Any]], f: SwingFilters,
                          earnings_map: dict[str, int] | None,
                          market_regime_ok: bool) -> pd.DataFrame:
    """Sinyal satırlarını skorlayıp filtreleyerek tavsiye tablosu üretir."""
    earnings_map = earnings_map or {}
    out: list[dict[str, Any]] = []

    for d in rows:
        d = dict(d)
        d["Skor"] = swing_score(d)
        ed = earnings_map.get(d.get("Sembol", ""))
        ok, blocks = apply_filters(d, f, ed, market_regime_ok)
        price = d.get("Fiyat", np.nan)
        stop = d.get("Stop", np.nan)
        t1, t2 = d.get("T1", np.nan), d.get("T2", np.nan)
        risk = price - stop if np.isfinite(price) and np.isfinite(stop) else np.nan

        d["Uygun"] = ok
        d["Engel"] = " · ".join(blocks)
        d["Karakter"] = trade_character(d.get("ATR %", np.nan))
        d["Risk %"] = (risk / price * 100) if np.isfinite(risk) and price else np.nan
        d["R (T1)"] = ((t1 - price) / risk) if np.isfinite(risk) and risk > 0 else np.nan
        d["R (T2)"] = ((t2 - price) / risk) if np.isfinite(risk) and risk > 0 else np.nan
        d["Bilanço Gün"] = ed
        out.append(d)

    df = pd.DataFrame(out)
    if df.empty:
        return df
    return df.sort_values(["Uygun", "Skor"], ascending=[False, False])


# --------------------------------------------------------------------------
# Swing yorumu — canlı veriden üretilir
# --------------------------------------------------------------------------
def swing_commentary(df: pd.DataFrame, macro_regime: str,
                     regime_ok: bool, top_n: int = 6) -> dict[str, Any]:
    """
    Kullanıcının elle yazdığı swing notunun canlı veriyle yeniden üretilmiş hali:
    hangi hisseler hangi karakterde, hangi filtreler devrede, rejim ne diyor.
    """
    if df.empty:
        return {"gruplar": {}, "filtreler": [], "rejim": macro_regime, "notlar": []}

    uygun = df[df["Uygun"]] if "Uygun" in df else df
    gruplar: dict[str, list[dict[str, Any]]] = {}

    for karakter in ["⚡ Yüksek beta / geniş ATR",
                     "📈 Temiz trend / orta volatilite",
                     "🛡️ Sakin / defansif"]:
        sel = uygun[uygun["Karakter"] == karakter].head(top_n)
        if not sel.empty:
            gruplar[karakter] = sel[["Sembol", "Skor", "ATR %", "Sinyal",
                                     "RS Sıra"]].to_dict("records")

    filtreler = [
        ("Bilanço ±2 gün", "Kazanç tarihine 2 günden az kalan hisselerde pozisyon "
                           "açılmaz — gap riski ATR stop mantığını bozar."),
        ("Likidite > $5M", "Günlük ortalama işlem hacmi eşiğin altındaysa spread "
                           "genişler, stop gerçekleşen fiyattan uzağa kayar."),
        ("Rejim kapısı", "SPY 50 EMA altındayken long sinyallerin isabet oranı "
                         "düşer. Bu, likidite grabı korumasıyla aynı mantıkta ayrı "
                         "bir gating katmanıdır: sinyal doğru olsa da ortam yanlışsa "
                         "işlem açılmaz."),
        ("200 EMA teyidi", "Hissenin kendi ana trendi aşağıysa swing long, trende "
                           "karşı işlem olur."),
    ]

    notlar = []
    if not regime_ok:
        notlar.append(
            "⚠️ Piyasa rejimi şu an KAPALI (SPY 50 EMA altında). Bu pencerede long "
            "sinyallerin kalitesi tarihsel olarak düşer; tarama sonuçlarını izleme "
            "listesi olarak kullanın, pozisyon açmak için rejimin dönmesini bekleyin."
        )
    if "ETF" in df.columns:
        etf_sayisi = uygun["ETF"].nunique() if not uygun.empty else 0
        notlar.append(
            f"Sektör ETF'leri ({etf_sayisi} tema taranıyor) tek hisse haber riskini "
            f"seyreltir; çoklu-hisse istatistiği tek isimden daha temiz çıkar."
        )
    notlar.append(
        "Uzun vade ve swing listelerinin çakışması sorun değil, ancak pozisyonları "
        "ayrı hesapta tutmak gerekir — aksi halde swing stopu uzun vadeli tezi keser."
    )
    return {"gruplar": gruplar, "filtreler": filtreler,
            "rejim": macro_regime, "notlar": notlar}
