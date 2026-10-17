# apex/backtest.py — 📊 HAFTALIK BACKTEST: alarmlar gerçekte TP1 / TP2 / TP3 mü, SL mi gördü?
#
# İki kaynak aynı değerlendiriciden geçer:
#   • Canlı alarm defteri — akşam hesabı her gün seçkinin ✅ / 🟢 (ve isteğe bağlı 🟡 limit)
#     alarmlarını data/alarms/alarm_defteri.csv'ye yazar; her akşam bütün açık alarmlar yeni
#     fiyatlarla yeniden değerlendirilir (data/alarms/defter_sonuc.csv).
#   • Geçmiş simülasyon — Sinyal doğrulama → secki testi (data/validation/secki/oneriler.csv).
#
# Kurallar (bir alarm için):
#   giriş   : ✅ / 🟢 → alarm günü kapanışı (değerlendirme ertesi günden başlar)
#             🟡 limit → ilk `fill_days` işlem gününde fiyat giriş bölgesinin üstüne değerse,
#             min(açılış, giriş üst) — değmezse "giriş olmadı" (sayılmaz)
#   seviyeler: SL = plan stopu · TP1 · TP2 · TP3 = TP2 + (TP2 − TP1) (yoksa giriş + 3R)
#   iz süren: TP1 görülünce stop girişe (başa baş), TP2 görülünce TP1'e çekilir
#   sınıf   : TP1 / TP2 / TP3 = ulaşılan en yüksek hedef · SL = TP1'den önce stop ·
#             devam ediyor = henüz hiçbiri · süre doldu = max_days içinde hiçbiri (kapanışla)
#   kâr %   : TP gören → ulaşılan en yüksek TP fiyatı (TP sınıfı, ekran görüntüsündeki gibi) ·
#             SL → stop fiyatı · süre doldu → son kapanış
#   aynı gün hem stop hem hedef → temkinli: önce stop
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

CLOSED = ("TP1", "TP2", "TP3", "SL")
AYLAR = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül",
         "Ekim", "Kasım", "Aralık"]


def tr_date(d, year: bool = True) -> str:
    d = pd.Timestamp(d)
    return f"{d.day} {AYLAR[d.month - 1]}" + (f" {d.year}" if year else "")


def _f(x) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def tp3_of(entry: float, stop: float, tp1: float, tp2: float) -> float:
    if np.isfinite(tp2) and np.isfinite(tp1) and tp2 > tp1:
        return tp2 + (tp2 - tp1)
    return entry + 3 * (entry - stop)


def evaluate_one(df: pd.DataFrame | None, t, karar: str, fiyat: float, giris_ust: float,
                 stop: float, tp1: float, tp2: float, tp3: float | None = None,
                 fill_days: int = 10, max_days: int = 60) -> dict[str, Any]:
    out = {"Durum": "veri yok", "Seviye": "", "Giriş": np.nan, "Çıkış": np.nan, "Kâr %": np.nan,
           "Kâra geçme (gün)": np.nan, "Kapanış": "", "Gün": np.nan, "Anlık %": np.nan}
    if df is None or df.empty:
        return out
    t = pd.Timestamp(t)
    fut = df[df.index > t].head(max_days)
    stop, tp1, tp2 = _f(stop), _f(tp1), _f(tp2)
    if not (np.isfinite(stop) and np.isfinite(tp1)):
        out["Durum"] = "plan yok"
        return out
    now = str(karar).startswith(("✅", "🟢"))
    if now:
        entry, i0 = _f(fiyat), 0
    else:
        top = _f(giris_ust)
        lows = fut["Low"].head(fill_days).to_numpy()
        hit = np.flatnonzero(lows <= top) if np.isfinite(top) else []
        if not len(hit):
            out["Durum"] = "giriş olmadı" if len(fut) >= fill_days else "giriş bekleniyor"
            return out
        i0 = int(hit[0])
        entry = float(min(fut["Open"].iloc[i0], top))
    if not np.isfinite(entry) or entry <= stop or tp1 <= entry:
        out["Durum"] = "plan geçersiz"
        return out
    if not np.isfinite(tp2) or tp2 <= tp1:
        tp2 = tp1 + (tp1 - entry)
    tp3 = _f(tp3) if tp3 is not None else tp3_of(entry, stop, tp1, tp2)
    if not np.isfinite(tp3) or tp3 <= tp2:
        tp3 = tp2 + (tp2 - tp1)
    out["Giriş"] = entry
    sl, best, first_tp = stop, 0, None
    lvl = {1: tp1, 2: tp2, 3: tp3}
    for j in range(i0, len(fut)):
        lo, hi = float(fut["Low"].iloc[j]), float(fut["High"].iloc[j])
        if lo <= sl:                                     # temkinli: önce stop
            if best == 0:
                out.update({"Durum": "kapandı", "Seviye": "SL", "Çıkış": sl,
                            "Kâr %": (sl / entry - 1) * 100, "Gün": j - i0 + 1,
                            "Kapanış": str(fut.index[j].date())})
            else:
                out.update({"Durum": "kapandı", "Seviye": f"TP{best}", "Çıkış": lvl[best],
                            "Kâr %": (lvl[best] / entry - 1) * 100, "Gün": j - i0 + 1,
                            "Kapanış": str(fut.index[j].date())})
            out["Kâra geçme (gün)"] = first_tp
            return out
        for k in (3, 2, 1):
            if hi >= lvl[k] and k > best:
                best = k
                if first_tp is None:
                    first_tp = j - i0 + 1
                break
        if best == 3:
            out.update({"Durum": "kapandı", "Seviye": "TP3", "Çıkış": tp3,
                        "Kâr %": (tp3 / entry - 1) * 100, "Gün": j - i0 + 1,
                        "Kapanış": str(fut.index[j].date()), "Kâra geçme (gün)": first_tp})
            return out
        if best >= 1:
            sl = max(sl, entry if best == 1 else tp1)    # iz süren stop
    last = float(fut["Close"].iloc[-1]) if len(fut) else entry
    out["Anlık %"] = (last / entry - 1) * 100
    out["Kâra geçme (gün)"] = first_tp
    out["Gün"] = len(fut) - i0
    if best >= 1:
        out.update({"Durum": "açık (TP gördü)", "Seviye": f"TP{best}",
                    "Kâr %": (lvl[best] / entry - 1) * 100})
    elif len(fut) >= max_days:
        out.update({"Durum": "süre doldu", "Seviye": "süre", "Çıkış": last,
                    "Kâr %": (last / entry - 1) * 100})
    else:
        out.update({"Durum": "devam ediyor", "Seviye": ""})
    return out


def evaluate(alarms: pd.DataFrame, prices: dict[str, pd.DataFrame], fill_days: int = 10,
             max_days: int = 60) -> pd.DataFrame:
    rows = []
    for r in alarms.to_dict("records"):
        df = prices.get(r["Hisse"])
        o = evaluate_one(df, r["Tarih"], r["Karar"], r.get("Fiyat"), r.get("Giriş Üst"),
                         r.get("Stop"), r.get("Kâr al 1"), r.get("Kâr al 2"), r.get("Kâr al 3"),
                         fill_days, max_days)
        rows.append({**r, **o})
    return pd.DataFrame(rows)


def week_bounds(d) -> tuple[pd.Timestamp, pd.Timestamp]:
    d = pd.Timestamp(d).normalize()
    mon = d - pd.Timedelta(days=d.weekday())
    return mon, mon + pd.Timedelta(days=4)


def summary(E: pd.DataFrame, d0, d1, idx_close: pd.Series | None = None) -> dict[str, Any]:
    """Seçilen tarih aralığında AÇILAN alarmların durumu (ekran görüntüsündeki tablo)."""
    d0, d1 = pd.Timestamp(d0), pd.Timestamp(d1)
    W = E[(pd.to_datetime(E["Tarih"]) >= d0) & (pd.to_datetime(E["Tarih"]) <= d1)]
    W = W[~W["Durum"].isin(["giriş olmadı", "plan geçersiz", "plan yok", "veri yok"])]
    tp = W[W["Seviye"].isin(["TP1", "TP2", "TP3"])]
    sl = W[W["Seviye"] == "SL"]
    gor = len(tp) + len(sl)
    cnt = {k: int((W["Seviye"] == k).sum()) for k in CLOSED}
    pct = {k: (cnt[k] / gor * 100 if gor else np.nan) for k in CLOSED}
    closed = W[W["Seviye"].isin(list(CLOSED) + ["süre"])]
    avg = float(closed["Kâr %"].mean()) if len(closed) else np.nan
    days = float(tp["Kâra geçme (gün)"].dropna().mean()) if len(tp) else np.nan
    idx_ret = np.nan
    if idx_close is not None and len(idx_close):
        ic = idx_close.dropna()
        ic.index = pd.to_datetime(ic.index).tz_localize(None) if getattr(
            ic.index, "tz", None) else pd.to_datetime(ic.index)
        before = ic[ic.index < d0]
        upto = ic[ic.index <= d1 + pd.Timedelta(hours=23)]
        if len(before) and len(upto):
            idx_ret = float((upto.iloc[-1] / before.iloc[-1] - 1) * 100)
    return {"d0": d0, "d1": d1, "toplam": len(W), "gor": gor,
            "devam": int(len(W) - gor - (W["Seviye"] == "süre").sum()),
            "sure": int((W["Seviye"] == "süre").sum()), "cnt": cnt, "pct": pct,
            "ort_kar": avg, "kar_gun": days, "idx": idx_ret,
            "fark": avg - idx_ret if np.isfinite(avg) and np.isfinite(idx_ret) else np.nan,
            "rows": W}


def weekly_table(E: pd.DataFrame, idx_close: pd.Series | None = None) -> pd.DataFrame:
    if E is None or E.empty:
        return pd.DataFrame()
    wk = sorted({week_bounds(x)[0] for x in pd.to_datetime(E["Tarih"])})
    rows = []
    for w in wk:
        s = summary(E, w, w + pd.Timedelta(days=4), idx_close)
        rows.append({"Hafta": f"{w:%d.%m} – {w + pd.Timedelta(days=4):%d.%m.%Y}", "_w": w,
                     "Alarm": s["toplam"], "TP/SL gören": s["gor"], "Devam": s["devam"],
                     "Süre doldu": s["sure"],
                     "TP1": s["cnt"]["TP1"], "TP2": s["cnt"]["TP2"], "TP3": s["cnt"]["TP3"],
                     "SL": s["cnt"]["SL"],
                     "TP oranı %": (100 - s["pct"]["SL"]) if s["gor"] else np.nan,
                     "Ort. kâr %": s["ort_kar"], "Kâra geçme (gün)": s["kar_gun"],
                     "Endeks %": s["idx"], "Fark (puan)": s["fark"]})
    return pd.DataFrame(rows).sort_values("_w", ascending=False).reset_index(drop=True)


def card_html(s: dict[str, Any], borsa: str, idx_name: str, strat: str) -> str:
    """Ekran görüntüsündeki koyu kart."""
    def p(x, d=1):
        return "—" if not np.isfinite(x) else f"{x:.{d}f}"
    c, pc = s["cnt"], s["pct"]
    kar = s["ort_kar"]
    idx = s["idx"]
    fark = s["fark"]
    kar_txt = (f"<span style='color:{'#2fbe86' if kar >= 0 else '#ef5350'}'>"
               f"{'+' if kar >= 0 else '−'} % {abs(kar):.2f} {'Kar' if kar >= 0 else 'Zarar'}</span>"
               if np.isfinite(kar) else "—")
    idx_txt = (f"{'+' if idx >= 0 else '−'} % {abs(idx):.2f} {'Kazanç' if idx >= 0 else 'Kayıp'}"
               if np.isfinite(idx) else "—")
    if np.isfinite(fark):
        son = (f"🏆 {strat}, bu dönemi {idx_name} endeksine kıyasla % {abs(fark):.1f} daha "
               f"{'yüksek' if fark >= 0 else 'düşük'} getiri ile tamamladı.")
    else:
        son = "Henüz sonuçlanan işlem yok."
    row = "<div style='margin:2px 0'>{}</div>"
    return f"""
<div style="background:#0f1420;border:1px solid #24304a;border-radius:14px;padding:18px 20px;
            color:#e6e9ef;font-size:15.5px;line-height:1.55;max-width:560px">
 <div style="color:#aab3c5">📅 {tr_date(s['d0'], False)} – {tr_date(s['d1'])} tarihleri arası</div>
 <div style="font-size:20px;font-weight:700;margin:6px 0 10px">📊 HAFTALIK BACKTEST TABLOSU</div>
 <div style="font-weight:700;margin-bottom:6px">Borsa: {borsa}</div>
 {row.format(f"Toplam alarm sayısı: <b>{s['toplam']}</b> adet")}
 {row.format(f"TP / SL gören işlem sayısı: <b>{s['gor']}</b> adet")}
 {row.format(f"Devam eden işlem sayısı: <b>{s['devam']}</b> adet")}
 {row.format(f"Süresi dolan (TP/SL görmeden kapanan): <b>{s['sure']}</b> adet") if s['sure'] else ""}
 <div style="height:8px"></div>
 <div style="color:#2fbe86">✅ TP1 sayısı: {c['TP1']} adet (%{p(pc['TP1'])})</div>
 <div style="color:#2fbe86">✅ TP2 sayısı: {c['TP2']} adet (%{p(pc['TP2'])})</div>
 <div style="color:#2fbe86">✅ TP3 sayısı: {c['TP3']} adet (%{p(pc['TP3'])})</div>
 <div style="color:#ef5350">🔴 SL sayısı: {c['SL']} adet (%{p(pc['SL'])})</div>
 <div style="height:8px"></div>
 <div style="font-weight:700">Haftalık ortalama yüzdesel kâr = {kar_txt}</div>
 <div style="color:#aab3c5">Ortalama kâra geçme vadesi = {p(s['kar_gun'])} iş günü</div>
 <div style="font-weight:700;margin-top:6px">📈 {idx_name} haftalık getirisi = {idx_txt}</div>
 <div style="font-weight:700;margin-top:8px">{son}</div>
</div>"""
