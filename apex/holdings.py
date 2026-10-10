# apex/holdings.py — AETHER APEX
from __future__ import annotations

from apex import universe as uni

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd


# Hangi getiri sütunu hangi pencereyi temsil ediyor
PERIOD_COLS: dict[str, str] = {
    "1 Gün %": "1 gün",
    "1 Hafta %": "1 hafta",
    "1 Ay %": "1 ay",
}

# Akrana göre z eşiği: bu değerin altı 'geride', üstü 'lider'
Z_ESIK = 0.75
# Gürültü tabanı: dağılım çok darsa (herkes aynı) etiketleme yapma
MIN_FARK_PP = 1.0


@dataclass
class Verdict:
    key: str
    label: str
    icon: str
    renk: str
    aciklama: str
    aksiyon: str


VERDICTS: dict[str, Verdict] = {
    "lider": Verdict(
        "lider", "Sepeti taşıyan", "🟢", "#2fbe86",
        "Akranlarının belirgin üstünde getiri. ETF'in yükselişi büyük ölçüde "
        "bu isimlerden geliyor.",
        "Trend takip mantığı burada çalışır; ekleme yapılacaksa geri çekilmede "
        "yapılır, yeni tepede değil."),
    "lider_yorgun": Verdict(
        "lider_yorgun", "Lider ama yorgun", "🟠", "#c98500",
        "Akranların üstünde ama tükenme/risk bayrağı açık — hareket "
        "istatistiksel olarak uzamış.",
        "Yeni giriş için kötü nokta. Mevcut pozisyonda kısmi kâr al, iz süren "
        "stopu yukarı çek."),
    "uyumlu": Verdict(
        "uyumlu", "Sepetle uyumlu", "⚪", "#9a9aa8",
        "Getirisi akran medyanına yakın. Ayrışma yok, ETF ile birlikte "
        "hareket ediyor.",
        "Tek hisse tercih etmenin ek getirisi yok; sepetle aynı işi yapar."),
    "geride_akis_var": Verdict(
        "geride_akis_var", "Geride ama para giriyor", "🟡", "#3987e5",
        "Fiyat akranlarının gerisinde, fakat kurumsal akış hâlâ pozitif "
        "(WHALE yüksek/yükseliyor ya da toplama-süpürme sinyali var). "
        "Klasik gecikmeli katılım profili.",
        "Yakalama adayı. Rejim kapısı açıkken ve akış yönü ⇈/↗ iken izlenir; "
        "tetik, kendi direncinin hacimle kırılmasıdır."),
    "geride_akis_yok": Verdict(
        "geride_akis_yok", "Geride ve akış negatif", "🔴", "#e66767",
        "Hem fiyat akranlarının gerisinde hem kurumsal akış çıkışta "
        "(dağıtım/stealth çıkış ya da düşen WHALE). Geri kalması haklı.",
        "Ucuz görünmesi tuzak. ETF yükselirken bunu almak, sepetin en zayıf "
        "bacağını satın almaktır — akış dönene kadar uzak dur."),
}


# --------------------------------------------------------------------------
# Göreli güç hesapları
# --------------------------------------------------------------------------
def robust_z(values: pd.Series) -> pd.Series:
    """
    Medyan ve MAD tabanlı z skoru.

    Neden ortalama/standart sapma değil: bir ETF listesinde tek bir isim
    %40 kazanmışsa ortalama yukarı kayar ve sağlıklı isimler yapay olarak
    'geride' görünür. Medyan bu tek aykırı değerden etkilenmez.
    """
    v = pd.to_numeric(values, errors="coerce")
    ok = v.dropna()
    if len(ok) < 3:
        return pd.Series(np.nan, index=v.index)
    med = float(ok.median())
    mad = float((ok - med).abs().median()) * 1.4826
    if not np.isfinite(mad) or mad <= 1e-9:
        std = float(ok.std(ddof=0))
        if not np.isfinite(std) or std <= 1e-9:
            return pd.Series(0.0, index=v.index).where(v.notna())
        mad = std
    return (v - med) / mad


def classify_holding(z: float, fark_akran: float, d: dict[str, Any]) -> str:
    """Bir bileşeni beş durumdan birine yerleştirir."""
    akis_pozitif = (
        bool(d.get("_acc")) or bool(d.get("_st_in")) or bool(d.get("_sweep"))
        or bool(d.get("_dia_buy")) or bool(d.get("_star"))
        or (np.isfinite(d.get("WHALE", np.nan)) and d.get("WHALE", 0) >= 55
            and d.get("ΔWHALE", 0) >= 0)
        or (np.isfinite(d.get("ΔWHALE 5B", np.nan)) and d.get("ΔWHALE 5B", 0) > 2)
    )
    akis_negatif = (
        bool(d.get("_dist")) or bool(d.get("_st_out")) or bool(d.get("_risk_hard"))
        or bool(d.get("_dia_sell")) or bool(d.get("_smc_sell"))
        or (np.isfinite(d.get("WHALE", np.nan)) and d.get("WHALE", 100) < 45
            and d.get("ΔWHALE", 0) <= 0)
    )

    geride = (np.isfinite(z) and z <= -Z_ESIK
              and np.isfinite(fark_akran) and fark_akran <= -MIN_FARK_PP)
    onde = (np.isfinite(z) and z >= Z_ESIK
            and np.isfinite(fark_akran) and fark_akran >= MIN_FARK_PP)

    if geride:
        if akis_negatif and not akis_pozitif:
            return "geride_akis_yok"
        if akis_pozitif:
            return "geride_akis_var"
        return "geride_akis_yok" if not d.get("Rejim", True) else "geride_akis_var"
    if onde:
        if d.get("_exhausted") or d.get("_risk") or d.get("_risk_hard"):
            return "lider_yorgun"
        return "lider"
    return "uyumlu"


def build_holdings_table(
        scan_df: pd.DataFrame, etf_sym: str, period_col: str,
        etf_row: dict[str, Any] | None = None,
        meta_fn: Callable[[str, str], dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """
    Tarama çıktısından ETF içi göreli güç tablosu üretir.

    `scan_df` sadece bileşenleri içermelidir (ETF'in kendi satırı ayrı gelir).
    `etf_row` ETF'in kendi tarama satırıdır; yoksa ETF'e göre kıyas boş kalır.
    """
    if scan_df is None or scan_df.empty or period_col not in scan_df.columns:
        return pd.DataFrame()

    meta_fn = meta_fn or uni.holding_meta
    df = scan_df.copy()
    df = df[df["Sembol"].notna()]
    ret = pd.to_numeric(df[period_col], errors="coerce")
    etf_ret = np.nan
    if etf_row:
        etf_ret = pd.to_numeric(pd.Series([etf_row.get(period_col)]),
                                errors="coerce").iloc[0]

    med = float(ret.dropna().median()) if ret.notna().any() else np.nan
    z = robust_z(ret)

    out = pd.DataFrame(index=df.index)
    out["Sembol"] = df["Sembol"]
    metas = [meta_fn(etf_sym, s) for s in df["Sembol"]]
    out["Ağırlık %"] = [m.get("agirlik") for m in metas]
    out["Rol"] = [m.get("rol") or m.get("grup") or "" for m in metas]
    out["Getiri %"] = ret
    out["ETF %"] = etf_ret
    out["ETF'e Göre"] = ret - etf_ret
    out["Akran Medyanı %"] = med
    out["Akrana Göre"] = ret - med
    out["Z"] = z
    # Ağırlıklı katkı: bu isim ETF getirisinin kaç puanını açıkladı (yaklaşık)
    out["Katkı pp"] = [
        (w / 100.0 * r) if (w is not None and np.isfinite(r)) else np.nan
        for w, r in zip(out["Ağırlık %"], ret)
    ]

    recs = df.to_dict("records")
    durum_keys = [classify_holding(zz, ff, d)
                  for zz, ff, d in zip(out["Z"], out["Akrana Göre"], recs)]
    out["_durum"] = durum_keys
    out["Durum"] = [f"{VERDICTS[k].icon} {VERDICTS[k].label}" for k in durum_keys]

    for col in ["Sinyal", "Efor", "Fiyat", "WHALE", "ΔWHALE", "Whale Yön",
                "PRO-RET", "OMNI", "ΔOMNI", "OMNI Yön", "Boğa /6", "Ayı /6",
                "MAGNITUDE",
                "DIRECTION", "ATR %", "RS Sıra", "Stop", "T1", "T2",
                "Hacim ($M)", "Rejim", "Haftalık", "Hata"]:
        if col in df.columns:
            out[col] = df[col]

    out["Teknik Not"] = [technical_note(d) for d in recs]
    out["Neden"] = [
        lag_reason(k, d, zz) for k, d, zz in zip(durum_keys, recs, out["Z"])
    ]
    return out.sort_values("Z", ascending=False, na_position="last")


# --------------------------------------------------------------------------
# Açıklama üretimi
# --------------------------------------------------------------------------
def _fmt(x: float, spec: str = "+.1f") -> str:
    return format(x, spec) if np.isfinite(x) else "—"


def technical_note(d: dict[str, Any]) -> str:
    """
    Bir hissenin son teknik durumunu düz cümlelerle özetler.
    Motorun ürettiği ham sayılar burada okunur hâle gelir.
    """
    if d.get("Hata"):
        return f"Veri yok — {d['Hata']}"

    parts: list[str] = []

    # 1) Ana trend / rejim
    rejim = d.get("Rejim")
    hafta = d.get("Haftalık")
    t = "200 EMA **üstünde**" if rejim else "200 EMA **altında**"
    if hafta is True:
        t += ", haftalık trend teyitli"
    elif hafta is False:
        t += ", haftalık teyit yok"
    parts.append(f"Ana trend: {t}.")

    # 2) Kurumsal akış
    wh = d.get("WHALE", np.nan)
    dw = d.get("ΔWHALE", np.nan)
    yon = d.get("Whale Yön", "")
    pr = d.get("PRO-RET", np.nan)
    if np.isfinite(wh):
        akis = f"Kurumsal akış: WHALE {wh:.0f} ({_fmt(dw)} son barda, {yon})"
        if np.isfinite(pr):
            akis += f", PRO−RETAIL {_fmt(pr)}"
        parts.append(akis + ".")

    # 3) Momentum
    om = d.get("OMNI", np.nan)
    do = d.get("ΔOMNI", np.nan)
    oy = d.get("OMNI Yön", "")
    boga, ayi = d.get("Boğa /6"), d.get("Ayı /6")
    if np.isfinite(om):
        mom = f"Momentum: OMNI {om:.0f} ({_fmt(do)}, {oy})"
        if boga is not None and ayi is not None:
            baskin = ("boğa" if boga > ayi else "ayı" if ayi > boga else "dengede")
            mom += f", konsensüs boğa {boga}/6 · ayı {ayi}/6 ({baskin})"
        parts.append(mom + ".")

    # 4) Konfluans
    mag, dr = d.get("MAGNITUDE"), d.get("DIRECTION")
    if mag is not None and dr is not None:
        parts.append(f"Konfluans: MAGNITUDE {mag}/18, DIRECTION {dr:+d} "
                     f"(0'ın üstü alıcı baskısı).")

    # 5) Volatilite karakteri ve seviyeler
    atrp = d.get("ATR %", np.nan)
    if np.isfinite(atrp):
        kar = ("geniş ATR / yüksek beta" if atrp >= 3.5
               else "temiz trend bandı" if atrp >= 1.8 else "dar, defansif")
        parts.append(f"Oynaklık: ATR %{atrp:.1f} — {kar}.")

    price = d.get("Fiyat", np.nan)
    stop, t1, t2 = d.get("Stop", np.nan), d.get("T1", np.nan), d.get("T2", np.nan)
    if np.isfinite(price) and np.isfinite(stop) and price:
        risk = (price - stop) / price * 100
        if risk >= 0:
            seviye = f"Seviyeler: iz süren stop {stop:.2f} (%{risk:.1f} aşağıda)"
        else:
            # Ratchet stop yukarı çekildikten sonra fiyat altına düştüyse
            seviye = (f"Seviyeler: iz süren stop {stop:.2f} fiyatın "
                      f"%{abs(risk):.1f} ÜSTÜNDE — stop çoktan tetiklenmiş, "
                      f"long kurgu bu seviyenin geri alınmasına bağlı")
        hedefler = [f"{ad} {x:.2f}" + (" (fiyatın altında, geçilmiş hedef)"
                                       if np.isfinite(price) and x < price else "")
                    for ad, x in (("T1", t1), ("T2", t2)) if np.isfinite(x)]
        if hedefler:
            seviye += ", " + ", ".join(hedefler)
        parts.append(seviye + ".")

    # 6) Başlık sinyali ve olaylar
    olaylar = []
    for flag, isim in [("_sweep", "likidite süpürmesi"), ("_acc", "toplama"),
                       ("_dist", "dağıtım"), ("_st_in", "stealth giriş"),
                       ("_st_out", "stealth çıkış"), ("_ab_bull", "afterburner"),
                       ("_exhausted", "tükenme"), ("Sıkışma", "sıkışma"),
                       ("_mvp", "Minervini MVP"), ("_star", "golden star")]:
        if d.get(flag):
            olaylar.append(isim)
    if olaylar:
        parts.append("Açık bayraklar: " + ", ".join(olaylar) + ".")

    return " ".join(parts)


def lag_reason(durum: str, d: dict[str, Any], z: float) -> str:
    """Durum etiketinin tek cümlelik gerekçesi — tabloda hızlı okunsun diye."""
    wh, dw = d.get("WHALE", np.nan), d.get("ΔWHALE", np.nan)
    if durum == "geride_akis_var":
        if d.get("_sweep"):
            return "Geride ama likidite süpürmesi var — stop avı sonrası dönüş kalıbı"
        if d.get("_acc") or d.get("_st_in"):
            return "Geride ama toplama sinyali açık — sessiz birikim"
        return (f"Geride ama WHALE {wh:.0f} ({_fmt(dw)}) — akış hâlâ içeride"
                if np.isfinite(wh) else "Geride, akış bozulmamış")
    if durum == "geride_akis_yok":
        if d.get("_dist") or d.get("_st_out"):
            return "Geride ve dağıtım açık — düşüşün sebebi satış"
        if not d.get("Rejim", True):
            return "Geride ve 200 EMA altında — trend zaten aşağı"
        return (f"Geride, WHALE {wh:.0f} ({_fmt(dw)}) — akış da destek vermiyor"
                if np.isfinite(wh) else "Geride, akış desteği yok")
    if durum == "lider_yorgun":
        return "Akranların üstünde ama tükenme/risk bayrağı açık"
    if durum == "lider":
        return f"Akran medyanının {abs(z):.1f} MAD üstünde — sepeti taşıyor" \
            if np.isfinite(z) else "Akranların üstünde"
    return "Akran medyanına yakın — ayrışma yok"


def holdings_narrative(table: pd.DataFrame, etf_sym: str, period_col: str) -> list[str]:
    """Tablonun tepesine konacak 2–4 cümlelik canlı yorum."""
    if table.empty:
        return []
    lab = PERIOD_COLS.get(period_col, period_col)
    etf_ret = table["ETF %"].iloc[0] if "ETF %" in table else np.nan
    med = table["Akran Medyanı %"].iloc[0] if "Akran Medyanı %" in table else np.nan
    n = len(table)
    poz = int((pd.to_numeric(table["Getiri %"], errors="coerce") > 0).sum())

    out: list[str] = []
    if np.isfinite(etf_ret):
        katilim = f"{poz}/{n} bileşen artıda"
        out.append(
            f"**{etf_sym}** {lab} penceresinde **%{etf_ret:+.2f}**; içindeki "
            f"{n} hissenin medyanı **%{med:+.2f}** ve {katilim}. "
            + ("Yükselişi az sayıda isim taşıyor — katılım dar."
               if etf_ret > 0 and poz < n * 0.5 else
               "Katılım geniş, hareket sepetin geneline yayılmış."
               if etf_ret > 0 else
               "Sepet ekside; aşağıdaki ayrışma dip arayışı için kullanılır.")
        )
    else:
        out.append(f"**{etf_sym}** içindeki {n} hissenin {lab} medyanı "
                   f"**%{med:+.2f}**.")

    for key in ["geride_akis_var", "geride_akis_yok", "lider_yorgun"]:
        sel = table[table["_durum"] == key]
        if sel.empty:
            continue
        v = VERDICTS[key]
        isim = ", ".join(f"`{s}`" for s in sel["Sembol"].head(6))
        out.append(f"{v.icon} **{v.label}** ({len(sel)}): {isim} — {v.aksiyon}")

    return out


def holdings_counts(table: pd.DataFrame) -> dict[str, int]:
    if table.empty or "_durum" not in table:
        return {}
    return {k: int((table["_durum"] == k).sum()) for k in VERDICTS}
