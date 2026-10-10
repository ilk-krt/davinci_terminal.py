# apex/valuation.py — AETHER APEX
from __future__ import annotations


# apex/valuation.py — Satış çarpanına dayalı adil değer (P/S Fair Value)
#
# Mantık ("Trader's glance" kartındaki yöntem):
#
#   1. Sektör/endüstri için "normal" ileri P/S çarpanı seçilir
#      (ör. yazılım ≈ 10x gelecek yıl satışı).
#   2. Adil orta = çarpan × gelecek yıl satış / hisse
#      Adil bant  = orta × (1 ± bant)   (varsayılan ±%12)
#   3. PSG = (fiyat / TTM satış-hisse) / büyüme%   — büyümeye göre P/S
#      Pahalı eşiği = PSG'nin `psg_max` olduğu fiyat
#      = psg_max × büyüme% × TTM satış/hisse       (varsayılan 0.60)
#   4. Durum:
#        fiyat < bant altı                → 🟢 UCUZ
#        bant altı ≤ fiyat ≤ pahalı eşiği → 🟡 ADİL (bant üstündeyse
#                                            "büyüme destekli" notu)
#        fiyat > pahalı eşiği             → 🔴 PAHALI
#
# Not: Pahalı eşiği kartta açıkça yazmıyor; RBRK kartındaki rakamlardan
# (fiyat 113.28, P/S 15x, PSG 0.39, pahalı 175.17) geri hesaplanınca
# PSG ≈ 0.60'a denk geliyor. Arayüzden değiştirilebilir.
#
# Veri katmanı (data.py) yalnızca ham alanları çeker; hesap burada yapılır.
# Böylece çarpan tablosu arayüzde düzenlendiğinde yeniden veri çekilmez.


from typing import Any

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Varsayılan ileri P/S çarpanları — BAŞLANGIÇ DEĞERLERİDİR, arayüzden düzenlenir.
# None = satış çarpanı bu iş modelinde anlamsız (banka, sigorta, petrol, GYO):
# bu şirketlerde P/S adil değeri hesaplanmaz.
# Eşleşme endüstri adında anahtar kelime aramasıyla yapılır (yfinance
# `industry` alanı); ilk eşleşen kazanır, bu yüzden özelden genele sıralı.
# --------------------------------------------------------------------------
INDUSTRY_PS: dict[str, float | None] = {
    "Software": 10.0,
    "Semiconductor Equipment": 7.0,
    "Semiconductors": 8.0,
    "Internet Content": 6.0,
    "Information Technology Services": 3.0,
    "Communication Equipment": 3.5,
    "Computer Hardware": 2.5,
    "Electronic Components": 3.5,
    "Scientific & Technical Instruments": 4.0,
    "Biotechnology": 6.0,
    "Drug Manufacturers": 4.0,
    "Medical Devices": 5.0,
    "Medical Instruments": 5.0,
    "Diagnostics": 4.0,
    "Health Information": 5.0,
    "Aerospace & Defense": 2.5,
    "Utilities": 3.0,
    "Uranium": 8.0,
    "Solar": 3.0,
    "Electrical Equipment": 3.0,
    "Specialty Industrial Machinery": 3.0,
    "Engineering & Construction": 1.5,
    "Auto Manufacturers": 1.5,
    "Internet Retail": 2.5,
    "Copper": 2.5,
    "Gold": 4.0,
    "Other Industrial Metals": 2.5,
    "Banks": None,
    "Insurance": None,
    "Capital Markets": None,
    "Asset Management": None,
    "Oil & Gas": None,
    "REIT": None,
    "Real Estate": None,
}

# Endüstri eşleşmezse sektöre düşülür
SECTOR_PS: dict[str, float | None] = {
    "Technology": 6.0,
    "Communication Services": 3.0,
    "Healthcare": 4.0,
    "Industrials": 2.0,
    "Consumer Cyclical": 1.5,
    "Consumer Defensive": 1.5,
    "Basic Materials": 2.0,
    "Utilities": 3.0,
    "Energy": None,
    "Financial Services": None,
    "Real Estate": None,
}

DEFAULT_BAND = 0.12      # adil bant ±%12
DEFAULT_PSG_MAX = 0.60   # bu PSG'nin üstü "pahalı"
DEFAULT_EXP_CAP = 1.80   # pahalı eşiği en fazla adil ortanın bu katı
                         # (RBRK kartında 175.17 / 98.33 ≈ 1.78)


def pick_multiple(industry: str, sector: str,
                  ind_map: dict[str, float | None] | None = None,
                  sec_map: dict[str, float | None] | None = None
                  ) -> tuple[float | None, str]:
    """Endüstri → sektör sırasıyla çarpanı bulur. Döner: (çarpan, kaynak)."""
    ind_map = INDUSTRY_PS if ind_map is None else ind_map
    sec_map = SECTOR_PS if sec_map is None else sec_map
    ind = (industry or "").lower()
    for key, mult in ind_map.items():
        if key.lower() in ind:
            return mult, key
    if sector in sec_map:
        return sec_map[sector], f"{sector} (sektör)"
    return None, "eşleşme yok"


def _f(x: Any) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def compute_fair_value(rec: dict[str, Any],
                       ind_map: dict[str, float | None] | None = None,
                       sec_map: dict[str, float | None] | None = None,
                       band: float = DEFAULT_BAND,
                       psg_max: float = DEFAULT_PSG_MAX,
                       exp_cap: float = DEFAULT_EXP_CAP) -> dict[str, Any]:
    """
    `rec`: data.fetch_fundamentals() satırı (Fiyat, TTM SPS, İleri SPS,
    Büyüme %, Endüstri, Sektör). Döner: adil değer alanları.
    """
    price = _f(rec.get("Fiyat"))
    ttm_sps = _f(rec.get("_ttm_sps"))
    fwd_sps = _f(rec.get("_fwd_sps"))
    g_pct = _f(rec.get("Büyüme %"))
    mult, kaynak = pick_multiple(rec.get("Endüstri", ""), rec.get("Sektör", ""),
                                 ind_map, sec_map)

    out: dict[str, Any] = {
        "Çarpan": mult, "Çarpan Kaynağı": kaynak,
        "P/S": price / ttm_sps if ttm_sps and np.isfinite(price) else np.nan,
        "İleri P/S": price / fwd_sps if fwd_sps and np.isfinite(price) else np.nan,
        "PSG": np.nan, "Adil Alt": np.nan, "Adil Orta": np.nan,
        "Adil Üst": np.nan, "Pahalı >": np.nan, "Adile Uzaklık %": np.nan,
        "P/S Durum": "➖ Hesaplanamadı",
    }

    if mult is None:
        out["P/S Durum"] = "➖ P/S uygun değil"
        return out
    if not (np.isfinite(price) and np.isfinite(fwd_sps) and fwd_sps > 0):
        out["P/S Durum"] = "➖ Satış tahmini yok"
        return out

    mid = mult * fwd_sps
    lo, hi = mid * (1 - band), mid * (1 + band)
    out.update({"Adil Alt": lo, "Adil Orta": mid, "Adil Üst": hi,
                "Adile Uzaklık %": (price / mid - 1) * 100})

    expensive = np.nan
    if np.isfinite(g_pct) and g_pct > 0 and np.isfinite(ttm_sps) and ttm_sps > 0:
        out["PSG"] = (price / ttm_sps) / g_pct
        expensive = psg_max * g_pct * ttm_sps
    # Büyüme yoksa/negatifse PSG tanımsız: bandın üstüne 2 bant daha
    if not np.isfinite(expensive):
        expensive = mid * (1 + 3 * band)
    # PSG eşiği büyümeyle orantılıdır ama sektör çarpanını bilmez: düşük
    # çarpanlı sektörde ya da küçük tabandan %100+ büyümede uçar. Bu yüzden
    # alttan adil banda, üstten adil ortanın `exp_cap` katına sıkıştırılır.
    expensive = min(max(expensive, hi), mid * max(exp_cap, 1 + band))
    out["Pahalı >"] = expensive

    if price < lo:
        out["P/S Durum"] = "🟢 UCUZ"
    elif price > expensive:
        out["P/S Durum"] = "🔴 PAHALI"
    elif price > hi:
        out["P/S Durum"] = "🟡 ADİL (bant üstü, büyüme destekli)"
    else:
        out["P/S Durum"] = "🟡 ADİL"
    return out


def add_fair_values(df: pd.DataFrame, **kw) -> pd.DataFrame:
    """Temel veri tablosunun her satırına adil değer sütunlarını ekler."""
    if df is None or df.empty:
        return df
    fv = pd.DataFrame([compute_fair_value(r, **kw)
                       for r in df.to_dict("records")], index=df.index)
    return pd.concat([df, fv], axis=1)


def glance_text(r: dict[str, Any]) -> str:
    """Tek hisse için 'Trader's glance' kartının Türkçe karşılığı (markdown)."""
    def m(x, spec=".2f", pre="$"):
        return f"{pre}{x:{spec}}" if np.isfinite(_f(x)) else "—"

    durum = r.get("P/S Durum", "➖")
    lines = [f"**Hızlı bakış — `{r.get('Hisse', '')}` {m(r.get('Fiyat'))}**", "",
             f"{durum} — satışlarına göre."]
    if r.get("Açıklama"):
        lines += ["", r["Açıklama"]]
    ps, fps = _f(r.get("P/S")), _f(r.get("İleri P/S"))
    if np.isfinite(ps):
        lines += ["", f"Şu an {ps:.1f}x satış"
                  + (f", gelecek yılın {fps:.1f}x katı ile işlem görüyor."
                     if np.isfinite(fps) else " ile işlem görüyor.")]
    mult = r.get("Çarpan")
    if mult:
        lines += [f"**{r.get('Çarpan Kaynağı', '')}** için normal çarpan ≈ {mult:g}x."]
        lines += [f"Adil aralık **{m(r.get('Adil Alt'))} – {m(r.get('Adil Üst'))}** "
                  f"({mult:g}x gelecek yıl satışı, ±bant).",
                  f"Pahalı eşiği: **{m(r.get('Pahalı >'))}** üstü."]
    ek = []
    if np.isfinite(_f(r.get("Piyasa Değ. ($B)"))):
        ek.append(f"PD ${_f(r['Piyasa Değ. ($B)']):.1f}B")
    if np.isfinite(_f(r.get("İleri F/K"))):
        ek.append(f"İleri F/K {_f(r['İleri F/K']):.1f}x")
    if np.isfinite(_f(r.get("EV/Satış"))):
        ek.append(f"EV/Satış {_f(r['EV/Satış']):.1f}x")
    if np.isfinite(_f(r.get("PSG"))):
        ek.append(f"PSG {_f(r['PSG']):.2f}")
    if np.isfinite(_f(r.get("Büyüme %"))):
        ek.append(f"Büyüme %{_f(r['Büyüme %']):.0f}")
    if np.isfinite(_f(r.get("52H Konum %"))):
        ek.append(f"52H %{_f(r['52H Konum %']):.0f}")
    if np.isfinite(_f(r.get("Hedef"))):
        ek.append(f"Analist hedefi {m(r.get('Hedef'))}")
    if ek:
        lines += ["", " · ".join(ek)]
    return "  \n".join(lines)
