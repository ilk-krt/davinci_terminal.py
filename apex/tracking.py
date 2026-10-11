# apex/tracking.py — SEKTÖR TAKİP · HİSSE TAKİP · PORTFÖY ve uygulama içi ALARMLAR
#
# Kayıt biçimi (watchlist sözlüğünün içinde, GitHub'a yazılır):
#   sector_watch : [{"tema", "eklendi", "not"}]
#   stock_watch  : [{"sym", "tema", "eklendi", "giris_alt", "giris_ust", "stop",
#                    "hedef1", "hedef2", "not"}]
#   portfolio    : [{"sym", "tema", "adet", "maliyet", "tarih", "ilk_stop", "not"}]
from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pandas as pd

from apex import pine_fib as pfib
from apex import pine_structure as pst
from apex import pine_vsa as pvsa
from apex.indicators import ema, rsi

QN = {"lider": "🚀 Lider", "iyilesen": "🌱 İyileşiyor", "zayif": "🌤️ Yoruluyor",
      "geride": "🩸 Geride"}


def _q(r: float, m: float) -> str:
    if not (np.isfinite(r) and np.isfinite(m)):
        return "geride"
    if r >= 100 and m >= 100:
        return "lider"
    if r >= 100:
        return "zayif"
    if m >= 100:
        return "iyilesen"
    return "geride"


def _f(x: float) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(x):
        return "—"
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.2f}"


def today() -> str:
    return dt.date.today().isoformat()


# --------------------------------------------------------------------------
# SEKTÖR TAKİP
# --------------------------------------------------------------------------
def sector_status(tema: str, T: pd.DataFrame, tails: dict, etf_rows: list[dict]
                  ) -> dict[str, Any]:
    """Swing gözüyle sektör durumu: gir / tut / çıkış yaklaşıyor / çık."""
    row = T[T["Tema"] == tema]
    if row.empty:
        return {"durum": "⚪ VERİ YOK", "tavsiye": "Tema verisi yok.", "alarm": False,
                "neden": []}
    r = row.iloc[0]
    tl = tails.get(tema)
    qs = [_q(a, b) for a, b in zip(tl["RS-Oran"], tl["RS-Mom"])] if tl is not None else []
    q_now = r["_q"]
    q_prev = qs[-3] if len(qs) >= 3 else q_now          # ~2 hafta önce
    mom_d = float(tl["RS-Mom"].iloc[-1] - tl["RS-Mom"].iloc[-3]) if tl is not None and len(tl) >= 3 else np.nan
    neden = [f"RRG: {QN.get(q_prev, q_prev)} → {QN.get(q_now, q_now)}",
             f"göreli momentum 2 haftada {mom_d:+.1f}" if np.isfinite(mom_d) else "",
             f"1H {r['1H %']:+.1f}% · 1A {r['1A %']:+.1f}%"]
    ex = sum(1 for e in etf_rows if e.get("_exhausted") or "TÜKENME" in str(e.get("Sinyal", "")))
    bear = sum(1 for e in etf_rows if any(k in str(e.get("Sinyal", ""))
                                          for k in ("SAT", "DAĞITIM", "RİSK")))
    if ex:
        neden.append(f"{ex} ETF'te tükenme işareti")
    if bear:
        neden.append(f"{bear} ETF'te satış/dağıtım sinyali")

    if q_now == "geride":
        d, adv, al = ("⛔ ÇIK / UZAK DUR",
                      "Tema endeksin gerisinde ve gücü azalıyor — pozisyon varsa çıkış, "
                      "yeni alım yok.", True)
    elif q_now == "zayif":
        d, adv, al = ("🟠 ÇIKIŞ VAKTİ YAKLAŞIYOR",
                      "Tema hâlâ endeksten güçlü ama ivmesi düşüyor (Lider → Yoruluyor). "
                      "Kâr almaya başlayın, stopları sıkılaştırın, yeni alım yapmayın.", True)
    elif q_now == "lider" and ((np.isfinite(mom_d) and mom_d < -3) or ex or bear):
        d, adv, al = ("🟡 LİDER AMA YORULUYOR",
                      "Liderlik sürüyor ama momentum zirveden dönüyor ya da tükenme var. "
                      "Kovalamayın; kısmi kâr ve iz süren stop.", True)
    elif q_now == "lider":
        d, adv, al = ("🟢 TUT", "Lider ve ivme sağlam — trendi takip edin, geri "
                                "çekilmelerde ekleme yapılabilir.", False)
    else:  # iyilesen
        fresh = q_prev in ("geride", "iyilesen")
        d, adv, al = ("🌱 ERKEN GİRİŞ", "Endeksin gerisinde ama göreli gücü artıyor — "
                      "akıllı paranın erken izi. Küçük pozisyonla girip Lider'e geçişte "
                      "artırın." if fresh else "İyileşme sürüyor — giriş için uygun.",
                      q_prev == "geride")
    return {"durum": d, "tavsiye": adv, "alarm": al, "neden": [x for x in neden if x],
            "q": q_now, "skor": int(r["Erken Skor"])}


def rotation_suggestion(tracked: list[str], T: pd.DataFrame, statuses: dict) -> list[str]:
    """Çıkış sinyali veren takipli temalar için geçilecek tema önerisi."""
    if T is None or T.empty:
        return []
    cand = T[T["_q"].isin(["iyilesen", "lider"]) & ~T["Tema"].isin(tracked)]
    cand = cand.sort_values("Erken Skor", ascending=False).head(3)["Tema"].tolist()
    out = []
    for t in tracked:
        s = statuses.get(t, {})
        if s.get("durum", "").startswith(("⛔", "🟠", "🟡")) and cand:
            out.append(f"**{t}** {s['durum'].split(' ', 1)[1].lower()} → geçiş adayları: "
                       + ", ".join(f"**{c}**" for c in cand))
    return out


# --------------------------------------------------------------------------
# HİSSE TAKİP
# --------------------------------------------------------------------------
def watch_alarms(item: dict, plan: dict | None, kalan_gun=None) -> tuple[str, list[str]]:
    """Takipteki hisse: kaydedilen giriş/stop seviyelerine göre durum + alarmlar."""
    if not plan or not plan.get("ok"):
        return "⚪ veri yok", []
    px = float(plan["Fiyat"])
    atr = float(plan.get("ATR", np.nan))
    lo, hi = float(item.get("giris_alt", np.nan)), float(item.get("giris_ust", np.nan))
    stop = float(item.get("stop", np.nan))
    t1 = float(item.get("hedef1", np.nan))
    al = []
    durum = "👀 izleniyor"
    if np.isfinite(stop) and px < stop:
        durum = "⛔ stop altında"
        al.append(f"⛔ Fiyat {_f(px)} kayıtlı stopun ({_f(stop)}) altına indi — plan "
                  "geçersiz, yeni plan bekleyin.")
    elif np.isfinite(lo) and np.isfinite(hi) and lo <= px <= hi * 1.002:
        durum = "🎯 giriş bölgesinde"
        al.append(f"🎯 Fiyat {_f(px)} giriş bölgesinde ({_f(lo)}–{_f(hi)}). Stop {_f(stop)}, "
                  f"hedef {_f(t1)}.")
    elif np.isfinite(hi) and hi < px <= hi * 1.02:
        durum = "⏳ girişe yaklaştı"
        al.append(f"⏳ Fiyat giriş bölgesine %{(px / hi - 1) * 100:.1f} uzaklıkta "
                  f"(üst sınır {_f(hi)}).")
    elif np.isfinite(t1) and px >= t1:
        durum = "🚀 hedefe ulaştı (girmeden)"
        al.append(f"🚀 Fiyat girmeden hedefe ({_f(t1)}) ulaştı — kovalamayın, yeni plan "
                  "bekleyin.")
    kg = pd.to_numeric(kalan_gun, errors="coerce")
    if np.isfinite(kg) and 0 <= kg <= 7:
        al.append(f"📅 Bilanço {int(kg)} gün sonra — girişi bilanço sonrasına bırakın.")
    # güncel plan kayıtlı plandan çok farklıysa haber ver
    if np.isfinite(hi) and plan.get("Giriş Üst") and np.isfinite(atr) and atr > 0:
        if abs(float(plan["Giriş Üst"]) - hi) > 1.5 * atr:
            al.append(f"🔄 Güncel giriş bölgesi değişti: {_f(plan['Giriş Alt'])}–"
                      f"{_f(plan['Giriş Üst'])} (kayıtlı {_f(lo)}–{_f(hi)}). "
                      "Planı güncellemek için listeden yeniden takibe alın.")
    return durum, al


# --------------------------------------------------------------------------
# PORTFÖY
# --------------------------------------------------------------------------
def position_levels(df: pd.DataFrame, cost: float, since: str | None = None,
                    first_stop: float | None = None) -> dict[str, Any]:
    """Portföydeki pozisyon için: iz süren stop, kâr al, doygunluk ve ekleme seviyeleri."""
    if df is None or len(df) < 60:
        return {"ok": False, "neden": "fiyat verisi yok"}
    d = df[["Open", "High", "Low", "Close", "Volume"]].astype(float).dropna(subset=["Close"]).tail(420)
    d["Volume"] = d["Volume"].fillna(0)
    atr_s = pfib.atr_rma(d, 14)
    atr = float(atr_s.iloc[-1])
    c = d["Close"]
    px = float(c.iloc[-1])
    e21, e50 = float(ema(c, 21).iloc[-1]), float(ema(c, 50).iloc[-1])
    e200 = float(ema(c, 200).iloc[-1]) if len(d) >= 200 else np.nan
    r14 = float(rsi(c, 14).iloc[-1])
    # giriş tarihinden beri en yüksek kapanış → chandelier (3 ATR) iz süren stop
    try:
        since_ts = pd.Timestamp(since) if since else d.index[-60]
        idx = pd.to_datetime(d.index)
        try:
            idx = idx.tz_localize(None)
        except TypeError:
            pass
        hh = float(c[idx >= since_ts].max()) if (idx >= since_ts).any() else px
    except Exception:                          # noqa: BLE001
        hh = float(c.tail(60).max())
    chand = hh - 3.0 * atr
    sr = pst.sr_matrix(d, atr_s)
    sup = [x["px"] for x in sr["sup"]]
    res = sorted(x["px"] for x in sr["res"])
    struct_stop = (sup[0] - 1.0 * atr) if sup else np.nan
    cands = [x for x in (chand, struct_stop, first_stop) if x is not None and np.isfinite(x)]
    stop = max(cands) if cands else px - 2 * atr     # en sıkı (en yüksek) stop geçerli
    # kâr al: fiyatın üstündeki ilk iki direnç; yoksa ATR katları
    tps = [x for x in res if x > px + 0.3 * atr][:2]
    while len(tps) < 2:
        tps.append(px + (2 if not tps else 3.5) * atr)
    # doygunluk (satış) işaretleri
    sat = []
    ext = (px - e21) / atr if atr > 0 else 0
    if ext > 3:
        sat.append(f"fiyat EMA21'in {ext:.1f} ATR üstünde — aşırı uzama")
    if r14 >= 75:
        sat.append(f"RSI {r14:.0f} — aşırı alım")
    try:
        vr = pvsa.compute(d)
        if vr["verdict"].startswith("ZAYIF"):
            sat.append("yükseliş hacimsiz (VSA: dağıtım şüphesi)")
        if bool(vr["anom"]["Buy Climax"][-3:].any()):
            sat.append("VSA: alım doruğu (Buy Climax)")
        if bool(vr["rev_dn"][-3:].any()) or bool(vr["flip_dn"][-3:].any()):
            sat.append("VSA: satıcı baskınlığa geçti")
    except Exception:                          # noqa: BLE001
        pass
    if res and px >= res[0] - 0.3 * atr:
        sat.append(f"dirence dayandı ({_f(res[0])})")
    # ekleme / maliyet düşürme: trend sağlamsa altındaki destekler
    trend_ok = np.isfinite(e200) and px > e200 and e50 > e200
    ekle = []
    if trend_ok:
        for s_ in sup[:2]:
            if s_ > stop:
                ekle.append(s_)
        if e21 < px and e21 > stop and all(abs(e21 - x) > 0.5 * atr for x in ekle):
            ekle.append(e21)
    ekle = sorted(ekle, reverse=True)[:2]
    pl = (px / cost - 1) * 100 if cost else np.nan
    risk0 = cost - (first_stop if first_stop else stop)
    r_mult = (px - cost) / risk0 if risk0 and risk0 > 0 else np.nan
    return {"ok": True, "Fiyat": px, "ATR": atr, "K/Z %": pl, "R": r_mult,
            "Stop": stop, "Kâr al 1": tps[0], "Kâr al 2": tps[1],
            "Ekle": ekle, "Doygunluk": sat, "Trend sağlam": trend_ok,
            "Zirve": hh}


def position_alarms(p: dict, lv: dict, kalan_gun=None) -> tuple[str, list[str]]:
    if not lv.get("ok"):
        return "⚪ veri yok", [lv.get("neden", "")]
    px, atr = lv["Fiyat"], lv["ATR"]
    al = []
    durum = "🟢 tut"
    if px <= lv["Stop"]:
        durum = "⛔ SAT — stop"
        al.append(f"⛔ Fiyat {_f(px)} iz süren stopun ({_f(lv['Stop'])}) altında — "
                  f"zirveden ({_f(lv['Zirve'])}) geri çekilme stop mesafesini aştı; "
                  "pozisyonu kapatın.")
    elif px <= lv["Stop"] + 0.5 * atr:
        durum = "⚠️ stopa yakın"
        al.append(f"⚠️ Fiyat stopa yaklaştı ({_f(lv['Stop'])}, %{(px / lv['Stop'] - 1) * 100:.1f} "
                  "uzakta).")
    if px >= lv["Kâr al 1"] - 0.3 * atr:
        durum = "💰 kâr al"
        al.append(f"💰 Kâr al 1 bölgesi ({_f(lv['Kâr al 1'])}) — pozisyonun bir kısmını "
                  f"(ör. üçte bir) satın, stopu maliyete çekin. Sonraki hedef "
                  f"{_f(lv['Kâr al 2'])}.")
    if len(lv["Doygunluk"]) >= 2:
        durum = "🟠 doygunluk — kısmi sat"
        al.append("🟠 Doygunluk: " + " · ".join(lv["Doygunluk"]) + " — kârın bir kısmını "
                  "realize edin.")
    for e in lv["Ekle"]:
        if abs(px - e) <= 0.4 * atr:
            al.append(f"➕ Ekleme bölgesinde ({_f(e)}) — trend sağlam; "
                      + ("maliyet düşürme fırsatı." if px < p.get("maliyet", px) else
                         "pozisyon büyütülebilir."))
            if durum == "🟢 tut":
                durum = "➕ ekleme bölgesi"
    kg = pd.to_numeric(kalan_gun, errors="coerce")
    if np.isfinite(kg) and 0 <= kg <= 7:
        al.append(f"📅 Bilanço {int(kg)} gün sonra — pozisyon büyüklüğünü gözden geçirin.")
    return durum, al
