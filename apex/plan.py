# apex/plan.py — hisse başına İŞLEM PLANI: giriş bölgesi · stop · kâr al · karar
#
# Seviyeler Pine modüllerinden gelir (hepsi günlük mum):
#   destek adayları : S/R matrisi destekleri (QUANTUM 885), açık boğa boşlukları
#                     (Gap Matrix), Wyckoff alım bölgeleri (VSA PRO), EMA21/EMA50,
#                     ana yapı Fibonacci 0.382 / 0.5 / 0.618 geri çekilmeleri
#   hedef adayları  : S/R dirençleri, doldurulmamış ayı boşlukları, Wyckoff satış
#                     bölgeleri, 52 haftalık zirve, Fib 1.272 / 1.618 uzantıları
# Giriş bölgesi = fiyatın altındaki EN ÇOK ÇAKIŞAN destek kümesi (≤ 3 ATR).
# Stop = kümenin dibi − 1 ATR (GFR'de ölçülen en iyi stop mesafesi).
# Hedef = girişten en az 1 R uzaktaki ilk direnç (T1) ve bir sonraki (T2);
# yakında direnç yoksa 2R / 3R.
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from apex import pine_fib as pfib
from apex import pine_structure as pst
from apex import pine_vsa as pvsa
from apex.indicators import ema

MAX_DEPTH_ATR = 3.0      # giriş desteği fiyatın en fazla bu kadar ATR altında
CLUSTER_ATR = 0.5        # bu yakınlıktaki destekler aynı küme sayılır
STOP_ATR = 1.0
ZONE_ATR = 0.5           # giriş bölgesinin kalınlığı


def _f(x: float) -> str:
    return f"{x:,.0f}" if x >= 1000 else f"{x:.2f}"


def stock_plan(df: pd.DataFrame) -> dict[str, Any]:
    """Günlük OHLCV → plan sözlüğü (ok=False ise 'neden' alanı açıklar)."""
    if df is None or len(df) < 120:
        return {"ok": False, "neden": "yetersiz fiyat geçmişi"}
    d = df[["Open", "High", "Low", "Close", "Volume"]].astype(float).dropna(
        subset=["Close"]).tail(420)
    atr_s = pfib.atr_rma(d, 14)
    atr = float(atr_s.iloc[-1])
    px = float(d["Close"].iloc[-1])
    if not (np.isfinite(atr) and atr > 0 and px > 0):
        return {"ok": False, "neden": "ATR hesaplanamadı"}
    c = d["Close"]
    e21, e50 = float(ema(c, 21).iloc[-1]), float(ema(c, 50).iloc[-1])
    e200 = float(ema(c, 200).iloc[-1]) if len(d) >= 200 else np.nan
    hi52 = float(d["High"].tail(252).max())

    sr = pst.sr_matrix(d, atr_s)
    gm = pst.gap_matrix(d, atr_s, "Günlük")
    vz = pst.vsa_zones(d)
    mf = pfib.main_fib(d)

    sup: list[tuple[float, str, float]] = []      # (fiyat, etiket, ağırlık)
    res: list[tuple[float, str]] = []
    for x in sr["sup"]:
        sup.append((x["px"], f"{pst.HZ_TXT[x['hz']]} vade destek ({x['cnt']} pivot)",
                    1.0 + 0.5 * x["hz"]))
    for x in sr["res"]:
        res.append((x["px"], f"{pst.HZ_TXT[x['hz']]} vade direnç"))
    for b in gm["boxes"]:
        if b["dir"] > 0 and b["top"] < px:
            sup.append((b["top"], f"açık {pst.TIER_TXT[b['tier']]} boşluğu", 0.6 + 0.2 * b["tier"]))
        elif b["dir"] < 0 and b["bot"] > px:
            res.append((b["bot"], f"doldurulmamış {pst.TIER_TXT[b['tier']]} boşluğu"))
    for z in vz["zones"]:
        if z["side"] > 0 and z["top"] <= px:
            sup.append((z["top"], f"Wyckoff alım bölgesi ({z['name']})", 1.0))
        elif z["side"] < 0 and z["bot"] >= px:
            res.append((z["bot"], f"Wyckoff satış bölgesi ({z['name']})"))
    for v, nm in ((e21, "EMA21"), (e50, "EMA50"), (e200, "EMA200")):
        if np.isfinite(v) and v < px:
            sup.append((v, nm, 0.7 if nm != "EMA200" else 1.0))
    if mf.get("ok") and mf["up"]:
        for lv in (0.382, 0.5, 0.618):
            v = mf["levels"][lv]
            if v < px:
                sup.append((v, f"Fib {lv}", 0.8 if lv != 0.618 else 1.0))
        for lv in (1.272, 1.618):
            if mf["levels"][lv] > px:
                res.append((mf["levels"][lv], f"Fib {lv} uzantısı"))
    if hi52 > px * 1.005:
        res.append((hi52, "52 hafta zirvesi"))

    # --- giriş kümesi: fiyatın 0–3 ATR altında en çok çakışan destekler
    cand = [s for s in sup if px - MAX_DEPTH_ATR * atr <= s[0] <= px]
    best = None
    for p0, _, _ in cand:
        grp = [s for s in cand if abs(s[0] - p0) <= CLUSTER_ATR * atr]
        w = sum(s[2] for s in grp)
        dist = (px - p0) / atr
        score = w - 0.15 * dist
        if best is None or score > best[0]:
            best = (score, grp)
    if best is None:
        lo_ = min((s for s in sup if s[0] < px), key=lambda s: px - s[0], default=None)
        if lo_ is None:
            return {"ok": False, "neden": "altta tanımlı destek yok"}
        grp = [lo_]
    else:
        grp = best[1]
    z_lo = min(s[0] for s in grp)
    z_hi = min(max(s[0] for s in grp) + ZONE_ATR * atr * 0.5, px)
    z_hi = max(z_hi, z_lo)
    entry = (z_lo + z_hi) / 2
    stop = z_lo - STOP_ATR * atr
    risk = entry - stop

    # --- hedefler: girişten ≥ 1R yukarıdaki dirençler
    res = sorted(set(res), key=lambda r: r[0])
    ok_t = [r for r in res if r[0] - entry >= risk]
    if ok_t:
        t1 = ok_t[0]
        nxt = [r for r in ok_t[1:] if r[0] - t1[0] >= 0.5 * atr]
        t2 = nxt[0] if nxt else (entry + max(3 * risk, t1[0] - entry + risk), "ölçülen hedef (3R)")
    else:
        t1 = (entry + 2 * risk, "ölçülen hedef (2R) — yakında direnç yok")
        t2 = (entry + 3 * risk, "ölçülen hedef (3R)")
    rr = (t1[0] - entry) / risk if risk > 0 else np.nan
    rr2 = (t2[0] - entry) / risk if risk > 0 else np.nan
    near_res = [r for r in res if px < r[0] <= px + 0.5 * atr]

    # --- teknik durum
    trend = ("yukarı" if (np.isfinite(e200) and px > e50 > e200) else
             "aşağı" if (np.isfinite(e200) and px < e50 < e200) else "karışık")
    try:
        vr = pvsa.compute(d)
        vsa_v = vr["verdict"]
    except Exception:                      # noqa: BLE001
        vsa_v = ""
    in_zone = px <= z_hi + 0.25 * atr
    dist_pct = (px / z_hi - 1) * 100 if z_hi > 0 else np.nan
    return {
        "ok": True, "Fiyat": px, "ATR": atr,
        "Giriş Alt": z_lo, "Giriş Üst": z_hi, "Giriş": entry,
        "Stop": stop, "Stop %": (stop / entry - 1) * 100,
        "Hedef 1": t1[0], "Hedef 1 neden": t1[1],
        "Hedef 2": t2[0], "Hedef 2 neden": t2[1],
        "R:R": rr, "R:R 2": rr2, "Bölgede": bool(in_zone), "Bölgeye %": dist_pct,
        "Destek": " + ".join(sorted({s[1] for s in grp})),
        "Dirence dayandı": bool(near_res),
        "Trend": trend, "VSA": vsa_v,
        "Zirveye %": (px / hi52 - 1) * 100,
    }


def plans(prices: dict[str, pd.DataFrame], tickers) -> pd.DataFrame:
    rows = []
    for t in tickers:
        try:
            p = stock_plan(prices.get(t))
        except Exception as exc:           # noqa: BLE001
            p = {"ok": False, "neden": f"hesap hatası: {exc}"}
        rows.append({"Sembol": t, **p})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Karar: tarama sınıfı + bilanço + endeks kararı + plan
# --------------------------------------------------------------------------
GOOD = {"🎯 Alım adayı", "🌱 Geride kaldı, toparlanıyor", "🚀 Lider"}


def verdict(p: dict[str, Any], cls: str = "", sig: str = "", kalan_gun=None,
            index_ok: bool = True) -> tuple[str, list[str]]:
    """Döner: (karar, [sebepler]). Karar: ✅ AL · 🟡 LİMİT EMİR · ⛔ UZAK DUR."""
    why_bad: list[str] = []
    why: list[str] = []
    if not p.get("ok"):
        if p.get("neden") == "altta tanımlı destek yok":
            return "⛔ UZAK DUR", ["fiyat yeni dipte — altında tutunacağı destek yok"]
        return "⚪ PLAN YOK", [p.get("neden", "veri yok")]
    kg = pd.to_numeric(kalan_gun, errors="coerce")
    if cls.startswith("⛔"):
        why_bad.append(f"satış/dağıtım sinyali ({sig})" if sig else "satış/dağıtım sinyali")
    if np.isfinite(kg) and 0 <= kg <= 7:
        why_bad.append(f"bilanço {int(kg)} gün sonra — gap riski")
    if cls.startswith("💧"):
        why_bad.append("işlem hacmi düşük")
    if p["Trend"] == "aşağı":
        why_bad.append("düşüş trendi (fiyat < EMA50 < EMA200)")
    if p["R:R 2"] < 1.5:
        why_bad.append(f"risk/ödül zayıf (en iyi hedefte bile {p['R:R 2']:.1f})")
    if p["VSA"] == "TEYİTLİ DÜŞÜŞ" and p["Trend"] != "yukarı":
        why_bad.append("hacim satıcıdan yana (VSA: teyitli düşüş)")
    if why_bad:
        return "⛔ UZAK DUR", why_bad

    why.append(f"giriş desteği: {p['Destek']}")
    why.append(f"hedef: {p['Hedef 1 neden']} · R:R {p['R:R']:.1f}")
    if p["Trend"] == "yukarı":
        why.append("trend yukarı")
    if cls:
        why.append(cls.split(" ", 1)[-1].lower())
    if p["VSA"].startswith("ZAYIF"):
        why.append("⚠ hacim zayıf (dağıtım şüphesi)")
    if p["VSA"].startswith("TOPLAMA"):
        why.append("hacim toplama gösteriyor")
    wait = []
    if not p["Bölgede"]:
        wait.append(f"fiyat giriş bölgesinin %{p['Bölgeye %']:.1f} üstünde — "
                    f"limit emir: {_f(p['Giriş Üst'])}")
    if p["Dirence dayandı"]:
        wait.append("fiyat dirence dayandı — kırılım ve üstünde kapanış beklenmeli")
    elif p["R:R"] < 1.5:
        wait.append(f"ilk hedef yakın ({_f(p['Hedef 1'])}, R:R {p['R:R']:.1f}) — "
                    "orası kırılırsa daha iyi risk/ödül")
    if cls and cls not in GOOD:
        wait.append(f"sınıf: {cls}")
    if not index_ok:
        wait.append("endeks kararı alım yönünde değil")
    if p["VSA"] == "TEYİTLİ DÜŞÜŞ":
        wait.append("geri çekilmede satış baskısı sürüyor — dönüş mumu/hacim teyidi bekleyin")
    if wait:
        return "🟡 LİMİT EMİR / BEKLE", wait + why
    return "✅ AL", why


INDEX_THEMES: dict[str, list[str]] = {
    "Nasdaq": ["Yarı İletken", "Yapay Zekâ", "Yazılım & SaaS", "Siber Güvenlik",
               "Teknoloji (Geniş)", "Robotik & Otomasyon", "Kuantum",
               "Veri Merkezi & Dijital GYO", "İletişim & Medya", "Fintek",
               "Biyoteknoloji", "Genomik", "Uzay", "Büyüme"],
    "S&P 500": ["Teknoloji (Geniş)", "Yarı İletken", "Bankalar", "Sağlık (Geniş)",
                "İlaç", "Tıbbi Cihaz", "Petrol & Gaz", "Kamu Hizmetleri",
                "Savunma & Havacılık", "Altyapı", "Taşımacılık", "Konut İnşaatı",
                "Perakende", "Tüketici Defansif", "Materyal", "Gayrimenkul",
                "İletişim & Medya", "Havayolları", "Nükleer & Uranyum",
                "Su Altyapısı", "Metal & Madencilik", "Temettü", "Değer"],
    "Kripto (BTC)": ["Bitcoin", "Bitcoin Madenciliği", "Fintek"],
}


# --------------------------------------------------------------------------
# Excel raporu (internetsiz inceleme için)
# --------------------------------------------------------------------------
EXCEL_HELP = [
    ("Giriş bölgesi", "Fiyatın altındaki (en fazla 3 ATR) desteklerden en çok çakışanı: "
                      "S/R matrisi, açık boşluklar (FVG/GAP), Wyckoff alım bölgeleri, "
                      "EMA21/50/200, ana yapı Fibonacci 0.382/0.5/0.618."),
    ("Stop", "Giriş bölgesinin dibi − 1 ATR. Parantezde girişe göre yüzde."),
    ("Kâr al 1 / 2", "Girişten en az 1R yukarıdaki ilk ve ikinci direnç; yakında "
                     "direnç yoksa 2R ve 3R."),
    ("R:R", "Kâr al 1'e kazanç ÷ stopa kayıp. 2 üstü iyi, 1.5 altı zayıf."),
    ("✅ AL", "Fiyat giriş bölgesinde, risk/ödül yeterli, endeks ve tema uygun."),
    ("🟡 LİMİT EMİR / BEKLE", "Hisse iyi ama fiyat bölgenin üstünde (limit emir "
                             "giriş bölgesinin üst sınırına) ya da teyit bekleniyor."),
    ("⛔ UZAK DUR", "Satış/dağıtım sinyali, 7 gün içinde bilanço, düşüş trendi, "
                   "düşük hacim, zayıf risk/ödül ya da altında destek yok."),
    ("Not", "Seviyeler son günlük kapanışa göre hesaplanır; emir vermeden önce "
            "grafikte teyit edin."),
]


def _sheet_name(name: str, used: set[str]) -> str:
    bad = '[]:*?/\\'
    s = "".join("-" if ch in bad else ch for ch in name)[:31] or "Sayfa"
    base, k = s, 2
    while s in used:
        s = f"{base[:28]}~{k}"
        k += 1
    used.add(s)
    return s


def excel_report(title: str, info_rows: list[tuple[str, str]], themes: pd.DataFrame,
                 per_theme: dict[str, pd.DataFrame]) -> bytes:
    """Özet + Temalar + her tema için bir sayfa + Açıklama → .xlsx baytları."""
    import io

    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    fills = {"✅": PatternFill("solid", fgColor="D9F2E3"),
             "🟡": PatternFill("solid", fgColor="FFF4CC"),
             "⛔": PatternFill("solid", fgColor="F8D7DA")}
    head_fill = PatternFill("solid", fgColor="1F2937")
    head_font = Font(bold=True, color="FFFFFF")
    widths = {"Tema": 22, "Karar": 22, "Hisse": 8, "Fiyat": 10,
              "Giriş bölgesi": 18, "Stop": 18, "Kâr al 1": 10, "Kâr al 2": 10,
              "R:R": 6, "Sebep": 110, "Para": 22, "Erken Skor": 11,
              "1H %": 8, "1A %": 8, "Başlık": 26, "Açıklama": 110}

    allrows = []
    for tema, df in per_theme.items():
        if df is not None and not df.empty:
            allrows.append(df.assign(Tema=tema))
    top = pd.concat(allrows, ignore_index=True) if allrows else pd.DataFrame()
    if not top.empty:
        top = top[top["Karar"].str.startswith(("✅", "🟡"))]
        top["_o"] = top["Karar"].str[:1].map({"✅": 0, "🟡": 1})
        top = top.sort_values(["_o", "R:R"], ascending=[True, False]).drop(columns="_o")
        top = top[["Tema"] + [c for c in top.columns if c != "Tema"]]

    buf = io.BytesIO()
    used: set[str] = set()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        start = len(info_rows) + 3
        sheet = _sheet_name("Özet", used)
        (top if not top.empty else pd.DataFrame({"Karar": ["Alınabilir ya da beklenecek hisse yok"]})
         ).to_excel(xw, sheet_name=sheet, index=False, startrow=start)
        ws = xw.sheets[sheet]
        ws.cell(row=1, column=1, value=title).font = Font(bold=True, size=14)
        for i, (k, v) in enumerate(info_rows, start=2):
            ws.cell(row=i, column=1, value=k).font = Font(bold=True)
            ws.cell(row=i, column=2, value=v)
        if themes is not None and not themes.empty:
            themes.round(1).to_excel(xw, sheet_name=_sheet_name("Temalar", used), index=False)
        for tema, df in per_theme.items():
            if df is not None and not df.empty:
                df.to_excel(xw, sheet_name=_sheet_name(tema, used), index=False)
        pd.DataFrame(EXCEL_HELP, columns=["Başlık", "Açıklama"]).to_excel(
            xw, sheet_name=_sheet_name("Açıklama", used), index=False)

        for ws in xw.book.worksheets:
            hdr = start + 1 if ws.title == "Özet" else 1
            cols = {}
            for cell in ws[hdr]:
                if cell.value is None:
                    continue
                cols[cell.value] = cell.column
                cell.fill, cell.font = head_fill, head_font
                ws.column_dimensions[get_column_letter(cell.column)].width = \
                    widths.get(str(cell.value), 14)
            ws.freeze_panes = ws.cell(row=hdr + 1, column=1)
            kcol = cols.get("Karar")
            for row in ws.iter_rows(min_row=hdr + 1):
                for cell in row:
                    cell.alignment = Alignment(vertical="top", wrap_text=(
                        cell.column in (cols.get("Sebep"), cols.get("Açıklama"))))
                if kcol:
                    v = str(row[kcol - 1].value or "")[:1]
                    if v in fills:
                        for cell in row:
                            cell.fill = fills[v]
    return buf.getvalue()
