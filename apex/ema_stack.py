# apex/ema_stack.py — 📐 EMA DİZİLİM ANALİZİ (EMA şeridi: 10 · 21 · 50 · 100 · 200)
#
# Eski okuma yalnızca "fiyat > EMA21 > EMA50 > EMA200 mü?" diye bakıyordu. Bu modül
# şeridi bir bütün olarak okur:
#   • Dizilim skoru  : fiyat + 5 EMA'nın bütün ikilileri boğa sırasında mı (−1…+1)
#   • Eğimler        : her EMA son 5 barda yükseliyor mu (ATR cinsinden)
#   • Yelpaze        : en yüksek − en düşük EMA (ATR cinsinden) ve 5 bardaki değişimi
#                      (açılıyor = trend güçleniyor · kapanıyor = trend yaşlanıyor)
#   • Uzama          : fiyatın EMA21'e uzaklığı (ATR) — aşırı uzamada kovalama riski
#   • Kesişimler     : 50/200 (altın / ölüm kesişimi) ve 21/50 — kaç bar önce
#   • Dinamik seviye : fiyatın altındaki en yakın EMA = destek, üstündeki = direnç
#   • FAZ            : yukarıdakilerden tek bir durum (tam boğa, sağlıklı geri çekilme,
#                      derin düzeltme, erken dönüş, yorulma, ayı tepkisi, tam ayı, sıkışma)
#   • Geçmiş         : AYNI FAZIN bu varlığın kendi geçmişinde h bar sonraki getirisi
#                      (medyan, pozitif oranı) ve bütün barların tabanı — örneklem içi,
#                      tek varlık; kesinlik değil, bağlam.
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from apex.indicators import ema

LENS = (10, 21, 50, 100, 200)
FWD = {"Günlük": 20, "Haftalık": 8, "Aylık": 3}

PHASES: dict[str, tuple[str, float, str]] = {
    # kod: (etiket, skor, açıklama)
    "tam_boga": ("🚀 Tam boğa dizilimi", 1.0,
                 "fiyat ve bütün EMA'lar sıralı, hepsi yükseliyor"),
    "boga_geri": ("🟢 Boğa — sağlıklı geri çekilme", 0.6,
                  "uzun EMA'lar yükseliyor, fiyat kısa EMA'lara geri çekildi (EMA50 üstünde)"),
    "boga_derin": ("🟠 Boğa — derin düzeltme", 0.15,
                   "ana trend (50 > 200) sağlam ama fiyat EMA50 altına indi"),
    "kirilma": ("⚠️ Trend kırılıyor", -0.4,
                "50 > 200 hâlâ ama fiyat ana trend EMA'sının (200) altına indi"),
    "yorulma": ("🟡 Yorulma / dağıtım", -0.35,
                "kısa EMA'lar aşağı döndü, uzunlar hâlâ yukarı — tepe oluşumu riski"),
    "erken_donus": ("🌱 Erken dönüş", 0.3,
                    "ana trend aşağı ama kısa EMA'lar yukarı döndü, fiyat EMA21 üstünde"),
    "ayi_tepki": ("🔴 Ayı — tepki yükselişi", -0.5,
                  "uzun EMA'lar düşüyor; fiyat kısa EMA'ların üstünde ama uzunların altında"),
    "tam_ayi": ("🔻 Tam ayı dizilimi", -1.0,
                "fiyat ve bütün EMA'lar ters sırada, hepsi düşüyor"),
    "sikisma": ("⚪ Sıkışma — EMA'lar iç içe", 0.0,
                "şerit 1.5 ATR'den dar: yön kararı yakın, kırılım yönüne bakın"),
    "karisik": ("⚪ Karışık", 0.0, "net bir dizilim yok"),
}


def _atr(d: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = d["High"], d["Low"], d["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def frame(d: pd.DataFrame) -> pd.DataFrame:
    """Bütün geçmiş için şerit ölçümleri ve faz (nedensel — her bar yalnız geçmişi görür)."""
    c = d["Close"].astype(float)
    a = _atr(d).replace(0, np.nan)
    lens = [n for n in LENS if len(c) >= n + 10]
    F = pd.DataFrame(index=d.index)
    F["px"], F["atr"] = c, a
    for n in lens:
        F[f"e{n}"] = ema(c, n)
        F[f"s{n}"] = (F[f"e{n}"] - F[f"e{n}"].shift(5)) / a          # 5 bar eğim (ATR)
    if not {10, 21, 50} <= set(lens):
        F["faz"] = "karisik"
        return F
    # dizilim skoru: fiyat + EMA'ların bütün ikilileri (kısa > uzun = +1)
    seq = ["px"] + [f"e{n}" for n in lens]
    pairs = [(i, j) for i in range(len(seq)) for j in range(i + 1, len(seq))]
    F["dizilim"] = sum(np.sign(F[seq[i]] - F[seq[j]]) for i, j in pairs) / len(pairs)
    E = F[[f"e{n}" for n in lens]]
    F["yelpaze"] = (E.max(axis=1) - E.min(axis=1)) / a
    F["yelpaze_d"] = F["yelpaze"] - F["yelpaze"].shift(5)
    F["uzama"] = (c - F["e21"]) / a
    S = F[[f"s{n}" for n in lens]]
    F["yukselen"] = (S > 0.05).sum(axis=1)
    F["dusen"] = (S < -0.05).sum(axis=1)
    nE = len(lens)
    long_ = "e200" if "e200" in F else ("e100" if "e100" in F else "e50")
    s_long = "s" + long_[1:]
    bull_lt = F["e50"] > F[long_] if long_ != "e50" else F["e21"] > F["e50"]
    lt_up = F[s_long] > -0.02
    lt_dn = F[s_long] < 0.02

    faz = pd.Series("karisik", index=F.index, dtype=object)
    conds = [
        ("tam_boga", (F["dizilim"] >= 0.9) & (F["yukselen"] == nE)),
        ("tam_ayi", (F["dizilim"] <= -0.9) & (F["dusen"] == nE)),
        ("sikisma", F["yelpaze"] < 1.5),
        ("yorulma", bull_lt & (F["e10"] < F["e21"]) & (F["s21"] < 0) & (c < F["e21"])
         & (c > F["e50"])),
        ("boga_geri", bull_lt & lt_up & (c > F["e50"]) & (F["s50"] > 0)),
        ("boga_derin", bull_lt & (c <= F["e50"]) & (c > F[long_])),
        ("kirilma", bull_lt & (c <= F[long_])),
        ("erken_donus", ~bull_lt & (c > F["e21"]) & (F["e10"] > F["e21"]) & (F["s21"] > 0)),
        ("ayi_tepki", ~bull_lt & lt_dn & (c > F["e21"])),
    ]
    done = pd.Series(False, index=F.index)
    for k, m in conds:
        m = m.fillna(False) & ~done
        faz[m] = k
        done |= m
    # yeterli geçmiş olmayan barlar
    faz[F[long_].isna() | F["atr"].isna()] = "karisik"
    F["faz"] = faz
    return F


def _since(cross: pd.Series) -> int | None:
    idx = np.flatnonzero(cross.fillna(False).to_numpy())
    return int(len(cross) - 1 - idx[-1]) if len(idx) else None


def analyze(d: pd.DataFrame, tf: str = "Günlük") -> dict[str, Any]:
    """Son bar için EMA dizilim okuması + geçmiş istatistiği."""
    if d is None or len(d) < 40:
        return {"ok": False, "neden": "yetersiz veri"}
    F = frame(d)
    if "dizilim" not in F:
        return {"ok": False, "neden": "EMA için yetersiz geçmiş"}
    r = F.iloc[-1]
    kod = str(r["faz"])
    lab, sc, ack = PHASES[kod]
    lens = [n for n in LENS if f"e{n}" in F]
    px = float(r["px"])
    # boğa fazında aşırı uzama → skor kırpılır
    uz = float(r["uzama"])
    if sc > 0 and uz > 3:
        sc -= 0.3
    elif sc < 0 and uz < -3:
        sc += 0.2                                   # aşırı satım: ayıda kovalama riski
    yel, yel_d = float(r["yelpaze"]), float(r["yelpaze_d"])
    dz = float(r["dizilim"])
    yon = "boğa" if dz >= 0.3 else "ayı" if dz <= -0.3 else ""
    yel_txt = ((f"{yon} yönünde açılıyor ({yon} trendi güçleniyor)" if yon else
                "açılıyor (yön belirsiz)") if yel_d > 0.25 else
               (f"kapanıyor ({yon} trendi yaşlanıyor)" if yon else "kapanıyor")
               if yel_d < -0.25 else "sabit")
    sl = {n: float(r[f"s{n}"]) for n in lens}
    egim = " ".join(f"E{n}{'↗' if v > 0.05 else '↘' if v < -0.05 else '→'}" for n, v in sl.items())
    sira = sorted([("Fiyat", px)] + [(f"E{n}", float(r[f"e{n}"])) for n in lens],
                  key=lambda x: -x[1])
    sira_txt = " > ".join(k for k, _ in sira)
    gc = gx = None
    if "e200" in F:
        up_ = (F["e50"] > F["e200"])
        gc = _since(up_ & ~up_.shift(1, fill_value=False))
        gx = _since(~up_ & up_.shift(1, fill_value=True))
    up2 = F["e21"] > F["e50"]
    c21u = _since(up2 & ~up2.shift(1, fill_value=False))
    c21d = _since(~up2 & up2.shift(1, fill_value=True))
    kes = []
    if gc is not None and (gx is None or gc < gx):
        kes.append(f"altın kesişim (50↑200) {gc} bar önce")
    elif gx is not None:
        kes.append(f"ölüm kesişimi (50↓200) {gx} bar önce")
    if c21u is not None and (c21d is None or c21u < c21d):
        kes.append(f"21↑50 {c21u} bar önce")
    elif c21d is not None:
        kes.append(f"21↓50 {c21d} bar önce")
    below = sorted([(float(r[f"e{n}"]), n) for n in lens if float(r[f"e{n}"]) < px], reverse=True)
    above = sorted([(float(r[f"e{n}"]), n) for n in lens if float(r[f"e{n}"]) > px])
    # geçmiş: aynı fazda h bar sonraki getiri
    h = FWD.get(tf, 20)
    fwd = (F["px"].shift(-h) / F["px"] - 1) * 100
    hist = F.assign(fwd=fwd).iloc[:-h].dropna(subset=["fwd"])
    hist = hist[hist["faz"] != "karisik"] if kod != "karisik" else hist
    same = hist[hist["faz"] == kod]["fwd"]
    base = hist["fwd"]
    gecmis = {"h": h, "n": int(len(same)),
              "medyan": float(same.median()) if len(same) else np.nan,
              "pozitif": float((same > 0).mean() * 100) if len(same) else np.nan,
              "taban_medyan": float(base.median()) if len(base) else np.nan,
              "taban_pozitif": float((base > 0).mean() * 100) if len(base) else np.nan}
    # fazın süresi
    fz = F["faz"].to_numpy()
    k = len(fz) - 1
    while k > 0 and fz[k - 1] == kod:
        k -= 1
    return {"ok": True, "faz": kod, "etiket": lab, "aciklama": ack, "skor": float(np.clip(sc, -1, 1)),
            "dizilim": float(r["dizilim"]), "sira": sira_txt, "egim": egim,
            "yelpaze": yel, "yelpaze_d": yel_d, "yelpaze_txt": yel_txt, "uzama": uz,
            "kesisim": kes, "sure": int(len(fz) - k),
            "destek": [(v, f"E{n}") for v, n in below[:2]],
            "direnc": [(v, f"E{n}") for v, n in above[:2]],
            "emas": {n: float(r[f"e{n}"]) for n in lens}, "px": px, "gecmis": gecmis}


def _f(x: float) -> str:
    return (f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.2f}" if abs(x) >= 1 else f"{x:.4g}")


def note(a: dict[str, Any], short: bool = False) -> str:
    """Modül tablosu ve okuma için metin."""
    if not a.get("ok"):
        return a.get("neden", "—")
    s = f"{a['etiket']} ({a['sure']} bardır) — {a['aciklama']}"
    if short:
        return s
    parts = [s, f"sıra: {a['sira']}", f"eğim: {a['egim']}",
             f"yelpaze {a['yelpaze']:.1f} ATR, {a['yelpaze_txt']}",
             f"EMA21'e uzaklık {a['uzama']:+.1f} ATR"
             + (" ⚠️ aşırı uzama — kovalamayın" if a["uzama"] > 3 else
                " ⚠️ aşırı satım" if a["uzama"] < -3 else "")]
    if a["kesisim"]:
        parts.append("; ".join(a["kesisim"]))
    return " · ".join(parts)


def levels_txt(a: dict[str, Any]) -> str:
    if not a.get("ok"):
        return ""
    px = a["px"]
    bits = []
    if a["destek"]:
        bits.append("dinamik destek " + ", ".join(
            f"{k} {_f(v)} ({(v / px - 1) * 100:+.1f}%)" for v, k in a["destek"]))
    if a["direnc"]:
        bits.append("dinamik direnç " + ", ".join(
            f"{k} {_f(v)} ({(v / px - 1) * 100:+.1f}%)" for v, k in a["direnc"]))
    return " · ".join(bits)


def history_txt(a: dict[str, Any], tf: str) -> str:
    g = a.get("gecmis") or {}
    if not g or not g.get("n"):
        return ""
    if g["n"] < 15:
        return f"Bu faz geçmişte çok az görülmüş ({g['n']} bar) — istatistik yok."
    unit = {"Günlük": "gün", "Haftalık": "hafta", "Aylık": "ay"}.get(tf, "bar")
    edge = g["medyan"] - g["taban_medyan"]
    return (f"Bu varlıkta aynı faz geçmişte {g['n']} barda görülmüş: {g['h']} {unit} sonra "
            f"medyan getiri %{g['medyan']:+.1f}, pozitif oran %{g['pozitif']:.0f} "
            f"(bütün barlar: %{g['taban_medyan']:+.1f} / %{g['taban_pozitif']:.0f} → fark "
            f"{edge:+.1f} puan). Örneklem içi, tek varlık — bağlam, kesinlik değil.")


def table_row(name: str, tf: str, a: dict[str, Any]) -> dict[str, Any]:
    g = a.get("gecmis") or {}
    ok = a.get("ok")
    return {"Varlık": name, "Zaman": tf,
            "Faz": a["etiket"] if ok else "—",
            "Süre (bar)": a.get("sure") if ok else None,
            "Dizilim": round(a["dizilim"], 2) if ok else np.nan,
            "Eğim": a.get("egim", "") if ok else "",
            "Yelpaze (ATR)": round(a["yelpaze"], 1) if ok else np.nan,
            "Yelpaze": a.get("yelpaze_txt", "") if ok else "",
            "Uzama (ATR)": round(a["uzama"], 1) if ok else np.nan,
            "Kesişim": "; ".join(a.get("kesisim", [])) if ok else "",
            "Geçmiş medyan %": g.get("medyan", np.nan) if ok and g.get("n", 0) >= 15 else np.nan,
            "Geçmiş pozitif %": g.get("pozitif", np.nan) if ok and g.get("n", 0) >= 15 else np.nan,
            "Taban medyan %": g.get("taban_medyan", np.nan) if ok else np.nan,
            "Örnek (bar)": g.get("n", 0) if ok else 0}
