# apex/robot.py — SİMÜLASYON (kâğıt üstü) işlem robotu
#
# Gerçek emir GÖNDERMEZ. 1 saatlik mumlarda bar bar ilerler; günlük bağlamı
# (Fibonacci bacağı, ATR, ADX, EMA200) yalnızca ÖNCEKİ günlerin kapanışından okur
# (geleceği görmez). Aynı kurallar hem geçmişte denemede (backtest) hem de
# "robotu başlat" ile ileriye doğru simülasyonda kullanılır: robot = ayarlar +
# başlangıç zamanı; her açılışta o andan bugüne kadar yeniden oynatılır.
#
# PLANLAR
#   vurkac   — tek hedef: hedefte hepsini sat
#   kademeli — 3 hedef: her birinde 1/3 sat; TP1'de stop maliyete, TP2'de TP1'e
#   bedava   — "bedava lot": TP1'de ANA PARAYI çıkaracak kadar sat, kalan lotlar
#              bedava; onlar iz süren stop + dinamik son hedefle taşınır
#   yonet    — portföydeki mevcut pozisyonu (adet, maliyet) yalnızca yönet
#
# AKILLI KONTROL (giriş): fiyat günlük ana bacağın güvenli Fibonacci bölgesine
#   (varsayılan 0.5–0.618) gelmiş olmalı + 1 saatlik HACİMLİ YEŞİL mum (hacim ≥
#   1.5× 20 bar ortalaması, kapanış mumun üst yarısında) + günlük trend filtresi
#   (kapanış > EMA200). Bütçe bölünmez: tek seferde tam giriş.
# DİNAMİK HEDEF: her gün başında ADX/DI/RSI'ya bakar — trend güçlüyse aktif hedefi
#   yukarı çeker, yataysa daraltır. STOP asla aşağı inmez (+1R'de maliyete,
#   +1.5R'den sonra zirve − 2.5 ATR(1s) iz süren stop).
# %85 RADAR & GİYOTİN: fiyat hedefe giden yolun %85'ini geçince radar açılır ve
#   zirveyi izler; 1 saatlik kapanış önceki mumun dibini kırar VE zirveden ≥ 1 ATR
#   gerilerse kalan pozisyon kapatılır (kâr kilitlenir).
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from apex import pine_fib as pfib
from apex.indicators import ema, rsi

PLANS = {
    "vurkac": "🎯 Vur-Kaç — tek hedef, hedefte tamamı satılır",
    "kademeli": "🪜 Kademeli (swing) — 3 hedefte 1/3'er satış, stop kademeli yukarı",
    "bedava": "🎁 Bedava lot — hedefte ana para çekilir, kâr lotları içeride koşar",
    "yonet": "🛡️ Sadece yönet — portföydeki mevcut pozisyonu yönetir (giriş yapmaz)",
}
MAX_STOP_ATR = 3.0
ZONES = {"0.5–0.618 (altın bölge)": (0.5, 0.618), "0.382–0.618": (0.382, 0.618),
         "0.618–0.786 (derin)": (0.618, 0.786)}


@dataclass
class Params:
    plan: str = "kademeli"
    budget: float = 10_000.0
    zone: tuple[float, float] = (0.5, 0.618)
    vol_mult: float = 1.5
    trend_filter: bool = True
    radar: float = 0.85
    adx_strong: float = 25.0
    adx_flat: float = 18.0
    loop: bool = True              # işlem bitince yeni giriş ara
    qty: float = 0.0               # yonet
    cost: float = 0.0              # yonet
    user_stop: float = 0.0         # yonet (0 = otomatik)


# ------------------------------------------------------------------ göstergeler
def adx(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    h, l, c = df["High"], df["Low"], df["Close"]
    up, dn = h.diff(), -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)

    def rma(x):
        return pd.Series(x, index=df.index).ewm(alpha=1 / n, adjust=False).mean()
    atr_ = rma(tr)
    pdi = 100 * rma(pdm) / atr_.replace(0, np.nan)
    mdi = 100 * rma(mdm) / atr_.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return pd.DataFrame({"adx": rma(dx.fillna(0)), "pdi": pdi, "mdi": mdi}, index=df.index)


def _naive(idx) -> pd.DatetimeIndex:
    idx = pd.to_datetime(idx)
    try:
        return idx.tz_convert("America/New_York").tz_localize(None)
    except (TypeError, AttributeError):
        try:
            return idx.tz_localize(None)
        except TypeError:
            return idx


def daily_context(dfd: pd.DataFrame) -> pd.DataFrame:
    """Günlük göstergeler (her satır: o günün KAPANIŞINDAN sonra bilinen)."""
    d = dfd[["Open", "High", "Low", "Close", "Volume"]].astype(float).dropna(subset=["Close"]).copy()
    d.index = _naive(d.index).normalize()
    d = d[~d.index.duplicated(keep="last")]
    ctx = pd.DataFrame(index=d.index)
    ctx["atr"] = pfib.atr_rma(d, 14)
    ctx["ema200"] = ema(d["Close"], 200)
    ctx["rsi"] = rsi(d["Close"], 14)
    ctx["close"] = d["Close"]
    ctx = ctx.join(adx(d))
    return ctx, d


@dataclass
class Leg:
    ok: bool
    up: bool = False
    levels: dict = field(default_factory=dict)


def _leg_cache(d: pd.DataFrame):
    cache: dict = {}

    def get(day: pd.Timestamp) -> Leg:
        if day in cache:
            return cache[day]
        sub = d[d.index < day]
        if len(sub) < 60:
            cache[day] = Leg(False)
            return cache[day]
        mf = pfib.main_fib(sub.tail(420))
        lg = Leg(bool(mf.get("ok")), bool(mf.get("up")), mf.get("levels", {}))
        cache[day] = lg
        return lg
    return get


# ------------------------------------------------------------------ motor
def run(df1h: pd.DataFrame, dfd: pd.DataFrame, p: Params,
        start: pd.Timestamp | str | None = None) -> dict[str, Any]:
    h1 = df1h[["Open", "High", "Low", "Close", "Volume"]].astype(float).dropna(subset=["Close"]).copy()
    h1.index = _naive(h1.index)
    h1 = h1[~h1.index.duplicated(keep="last")]
    ctx, d = daily_context(dfd)
    leg_at = _leg_cache(d)
    st_ts = pd.Timestamp(start) if start is not None else h1.index[0]
    try:
        st_ts = st_ts.tz_localize(None)
    except TypeError:
        pass
    # 1 saatlik göstergeler (tüm geçmişle ısıtılır)
    atr1 = pfib.atr_rma(h1, 14)
    vma = h1["Volume"].rolling(20).mean()
    o, hi, lo, c, v = (h1[k].to_numpy() for k in ("Open", "High", "Low", "Close", "Volume"))
    a1, vm = atr1.to_numpy(), vma.to_numpy()
    idx = h1.index
    first = int(np.searchsorted(idx.values, np.datetime64(st_ts)))

    n = len(h1)
    sl_line = np.full(n, np.nan)
    tp_line = np.full(n, np.nan)
    rd_line = np.full(n, np.nan)
    cost_line = np.full(n, np.nan)
    zt = np.full(n, np.nan)
    zb = np.full(n, np.nan)
    equity = np.full(n, np.nan)
    events: list[dict] = []
    fills: list[dict] = []
    trades: list[dict] = []

    cash = p.budget if p.plan != "yonet" else 0.0
    qty = 0.0
    entry = stop = r0 = 0.0
    tps: list[float] = []
    tp_i = 0
    peak = hh = 0.0
    radar_on = False
    cur_day = None
    cool = 0
    tr_open: dict | None = None
    free_done = False
    exts: list[float] = []
    tps0: list[float] = []
    diag = {"bar": 0, "yükselen bacak": 0, "trend (EMA200)": 0, "Fib bölgesinde": 0,
            "yeşil mum": 0, "hacim ≥ eşik": 0, "hepsi birden": 0}

    def ev(i, kind, msg, px=None):
        events.append({"Zaman": idx[i], "Olay": kind, "Mesaj": msg,
                       "Fiyat": float(px) if px is not None else float(c[i])})

    def fill(i, side, q, px, why):
        nonlocal cash, qty
        q = float(min(q, qty)) if side == "SAT" else float(q)
        if q <= 0:
            return 0.0
        if side == "AL":
            cash -= q * px
            qty += q
        else:
            cash += q * px
            qty -= q
            if tr_open is not None:
                tr_open["sold"] += q * px
                tr_open["parts"].append(f"{why} {q:g}@{px:.2f}")
        fills.append({"Zaman": idx[i], "İşlem": side, "Adet": q, "Fiyat": px, "Neden": why})
        return q

    def close_trade(i, why):
        nonlocal tr_open, radar_on, tps, tp_i, cool, free_done
        if tr_open is None:
            return
        pnl = tr_open["sold"] - tr_open["cost"]
        trades.append({"Giriş": tr_open["t"], "Giriş fiyatı": tr_open["px"],
                       "Çıkış": idx[i], "Adet": tr_open["q"],
                       "K/Z $": pnl, "K/Z %": pnl / tr_open["cost"] * 100 if tr_open["cost"] else np.nan,
                       "R": pnl / (tr_open["risk"] * tr_open["q"]) if tr_open["risk"] > 0 else np.nan,
                       "Son çıkış": why, "Satışlar": " · ".join(tr_open["parts"])})
        tr_open, radar_on, tps, tp_i, cool, free_done = None, False, [], 0, 5, False

    def targets(lg: Leg, ent: float, risk: float) -> list[float]:
        L = lg.levels
        cands = sorted(x for x in (L.get(0.0), L.get(1.272), L.get(1.618), L.get(2.618))
                       if x is not None and x > ent)
        # ilk hedef en az 1.5R uzakta olmalı
        cands = [x for x in cands if x - ent >= 1.0 * risk] or [ent + 2 * risk, ent + 3 * risk]
        while len(cands) < 3:
            cands.append(cands[-1] + risk)
        if p.plan == "vurkac":
            t = next((x for x in cands if x - ent >= 1.5 * risk), cands[0])
            return [t]
        if p.plan == "bedava":
            return [cands[0], cands[min(2, len(cands) - 1)]]
        return cands[:3]

    # "sadece yönet": başlangıçta pozisyonda
    if p.plan == "yonet" and first < n:
        i0 = first
        day = idx[i0].normalize()
        cx = ctx[ctx.index < day]
        atr_d = float(cx["atr"].iloc[-1]) if len(cx) else a1[i0] * 3
        qty = float(p.qty)
        entry = float(p.cost) if p.cost else float(c[i0])
        stop = float(p.user_stop) if p.user_stop else float(c[i0]) - 2 * atr_d
        r0 = max(entry - stop, 0.5 * atr_d)
        lg = leg_at(day)
        tps = targets(lg if lg.ok else Leg(True, True, {}), max(entry, float(c[i0])), r0)
        tr_open = {"t": idx[i0], "px": entry, "q": qty, "cost": qty * entry, "sold": 0.0,
                   "risk": r0, "parts": []}
        tps0 = list(tps)
        exts = sorted(x for x in (lg.levels.get(k) for k in (0.0, 1.272, 1.618, 2.618))
                      if x is not None) if lg.ok else []
        hh = float(c[i0])
        ev(i0, "🛡️ Yönetim başladı", f"{qty:g} adet · maliyet {entry:.2f} · stop {stop:.2f} · "
           f"hedefler {', '.join(f'{x:.2f}' for x in tps)}")

    for i in range(max(first, 1), n):
        day = idx[i].normalize()
        new_day = day != cur_day
        cur_day = day
        cx = ctx[ctx.index < day]
        if len(cx) < 30:
            equity[i] = cash + qty * c[i]
            continue
        cr = cx.iloc[-1]
        atr_d = float(cr["atr"])
        lg = leg_at(day)
        if lg.ok and lg.up:
            z_hi, z_lo = lg.levels.get(p.zone[0]), lg.levels.get(p.zone[1])
            if z_hi is not None and z_lo is not None:
                zt[i], zb[i] = z_hi + 0.15 * atr_d, z_lo - 0.25 * atr_d

        # ------------------------------------------------ pozisyondayken
        if qty > 0:
            # 1) gap/stop
            if lo[i] <= stop:
                px = min(o[i], stop) if o[i] < stop else stop
                fill(i, "SAT", qty, px, "stop")
                ev(i, "⛔ Stop", f"{px:.2f} seviyesinden çıkıldı", px)
                close_trade(i, "stop")
            else:
                # 2) hedefler
                while qty > 0 and tp_i < len(tps) and hi[i] >= tps[tp_i]:
                    t = tps[tp_i]
                    px = max(o[i], t) if o[i] > t else t
                    if p.plan == "vurkac" or tp_i == len(tps) - 1:
                        fill(i, "SAT", qty, px, f"TP{tp_i + 1}")
                        ev(i, f"🎯 TP{tp_i + 1}", f"son hedef {px:.2f} — pozisyon kapandı", px)
                        close_trade(i, f"TP{tp_i + 1}")
                        break
                    if p.plan == "bedava" and not free_done:
                        need = math.ceil(tr_open["cost"] / px) if tr_open else qty
                        q = fill(i, "SAT", min(need, qty), px, "ana para")
                        free_done = True
                        stop = max(stop, entry)
                        ev(i, "🎁 Bedava lot", f"{q:g} adet satılıp ana para çekildi; kalan "
                           f"{qty:g} adet bedava. Stop maliyete ({stop:.2f}).", px)
                    else:
                        part = qty / (len(tps) - tp_i)
                        part = max(1.0, math.floor(part)) if qty >= 2 else qty
                        q = fill(i, "SAT", part, px, f"TP{tp_i + 1}")
                        prev = stop
                        stop = max(stop, entry if tp_i == 0 else tps[tp_i - 1])
                        ev(i, f"🎯 TP{tp_i + 1}", f"{q:g} adet satıldı; stop {prev:.2f} → {stop:.2f}", px)
                    tp_i += 1
                    radar_on = False
                if qty > 0:
                    hh = max(hh, c[i])
                    # 3) stop asla aşağı inmez
                    prev = stop
                    if c[i] >= entry + r0:
                        stop = max(stop, entry)
                    if c[i] >= entry + 1.5 * r0 and np.isfinite(a1[i]):
                        stop = max(stop, hh - 2.5 * a1[i])
                    if stop > prev + 0.25 * atr_d:
                        ev(i, "🛡️ Stop yukarı", f"{prev:.2f} → {stop:.2f}")
                    # 4) dinamik hedef (gün başında)
                    if new_day and tp_i < len(tps):
                        t = tps[tp_i]
                        ax, pdi, mdi, rs = cr["adx"], cr["pdi"], cr["mdi"], cr["rsi"]
                        base = tps0[tp_i] if tp_i < len(tps0) else t
                        cap = next((x for x in exts if x > base + 1e-9), base + 1.5 * atr_d)
                        if ax >= p.adx_strong and pdi > mdi and rs > 55:
                            nt = min(t + 0.5 * atr_d, cap)
                            if nt > t + 1e-9:
                                tps[tp_i] = nt
                                ev(i, "📈 Hedef yukarı", f"trend güçlü (ADX {ax:.0f}, RSI {rs:.0f}): "
                                   f"TP{tp_i + 1} {t:.2f} → {nt:.2f}")
                        elif ax < p.adx_flat:
                            nt = max(c[i] + 0.3 * atr_d, t - 0.5 * atr_d, entry + 0.8 * r0)
                            if nt < t - 1e-9:
                                tps[tp_i] = nt
                                ev(i, "📉 Hedef daraltıldı", f"yatay piyasa (ADX {ax:.0f}): "
                                   f"TP{tp_i + 1} {t:.2f} → {nt:.2f}")
                    # 5) %85 radar & giyotin
                    if tp_i < len(tps):
                        t = tps[tp_i]
                        prog = (c[i] - entry) / (t - entry) if t > entry else 0
                        if not radar_on and prog >= p.radar:
                            radar_on, peak = True, hi[i]
                            ev(i, "📡 Radar", f"hedefe giden yolun %{prog * 100:.0f}'i geçildi — "
                               "tepe izleniyor")
                        elif radar_on:
                            peak = max(peak, hi[i])
                            if (c[i] < lo[i - 1] and np.isfinite(a1[i])
                                    and c[i] <= peak - 1.0 * a1[i]):
                                q = qty
                                fill(i, "SAT", q, c[i], "giyotin")
                                ev(i, "⚔️ Giyotin", f"tepeden ({peak:.2f}) dönüş — {q:g} adet "
                                   f"{c[i]:.2f}'den satıldı, kâr kilitlendi", c[i])
                                close_trade(i, "giyotin")
        # ------------------------------------------------ düzken: akıllı kontrol
        elif p.plan != "yonet" and (p.loop or not trades) and cool <= 0:
            ok_trend = (not p.trend_filter) or (np.isfinite(cr["ema200"]) and cr["close"] > cr["ema200"])
            diag["bar"] += 1
            up_ok = lg.ok and lg.up and np.isfinite(zt[i])
            touched = up_ok and np.nanmin(lo[max(0, i - 6):i + 1]) <= zt[i]
            in_zone = bool(up_ok and c[i] >= zb[i] and touched and c[i] <= zt[i] + 0.5 * atr_d)
            rng = hi[i] - lo[i]
            green = c[i] > o[i] and rng > 0 and (c[i] - lo[i]) / rng >= 0.5
            vol_ok = bool(np.isfinite(vm[i]) and vm[i] > 0 and v[i] >= p.vol_mult * vm[i])
            diag["yükselen bacak"] += int(up_ok)
            diag["trend (EMA200)"] += int(ok_trend)
            diag["Fib bölgesinde"] += int(in_zone)
            diag["yeşil mum"] += int(green)
            diag["hacim ≥ eşik"] += int(vol_ok)
            if up_ok and ok_trend:
                if in_zone and green and vol_ok:
                    diag["hepsi birden"] += 1
                    sl0 = min(lg.levels.get(0.786, zb[i]), float(np.min(lo[max(0, i - 10):i + 1]))) - 0.25 * atr_d
                    px = c[i]
                    sl0 = max(sl0, px - MAX_STOP_ATR * atr_d)     # stop en fazla 3 günlük ATR uzakta
                    risk = px - sl0
                    q = math.floor(cash / px) if px > 0 else 0
                    if risk > 0 and q > 0:
                        fill(i, "AL", q, px, "akıllı kontrol")
                        entry, stop, r0, hh = px, sl0, risk, px
                        exts = sorted(x for x in (lg.levels.get(k) for k in (0.0, 1.272, 1.618, 2.618))
                                      if x is not None)
                        tps, tp_i, radar_on, free_done = targets(lg, px, risk), 0, False, False
                        tps0 = list(tps)
                        tr_open = {"t": idx[i], "px": px, "q": q, "cost": q * px, "sold": 0.0,
                                   "risk": risk, "parts": []}
                        ev(i, "✅ Giriş", f"Fib bölgesi + hacimli yeşil 1s mum (hacim "
                           f"{v[i] / vm[i]:.1f}×): {q:g} adet @ {px:.2f} · stop {sl0:.2f} · "
                           f"hedefler {', '.join(f'{x:.2f}' for x in tps)}", px)
        cool -= 1
        if qty > 0:
            sl_line[i], cost_line[i] = stop, entry
            if tp_i < len(tps):
                tp_line[i] = tps[tp_i]
                rd_line[i] = entry + p.radar * (tps[tp_i] - entry)
        equity[i] = cash + qty * c[i]

    eq = pd.Series(equity, index=idx).ffill()
    eq = eq[eq.index >= st_ts]
    start_val = (p.budget if p.plan != "yonet" else float(p.qty) * float(p.cost or c[first]))
    T = pd.DataFrame(trades)
    stats = {"İşlem": len(T),
             "Kazanan %": float((T["K/Z $"] > 0).mean() * 100) if len(T) else np.nan,
             "Toplam K/Z $": float(eq.iloc[-1] - start_val) if len(eq) else 0.0,
             "Getiri %": float((eq.iloc[-1] / start_val - 1) * 100) if len(eq) and start_val else np.nan,
             "Ort. R": float(T["R"].mean()) if len(T) else np.nan,
             "Maks. düşüş %": float(((eq / eq.cummax()) - 1).min() * 100) if len(eq) else np.nan,
             "Al-tut %": float((c[-1] / c[first] - 1) * 100) if first < n else np.nan}
    state = {"pozisyonda": qty > 0, "adet": qty, "maliyet": entry, "stop": stop,
             "hedef": tps[tp_i] if qty > 0 and tp_i < len(tps) else np.nan,
             "hedefler": tps, "radar": radar_on, "fiyat": float(c[-1]), "nakit": cash,
             "K/Z % (açık)": (c[-1] / entry - 1) * 100 if qty > 0 and entry else np.nan,
             "hedefe %": (tps[tp_i] / c[-1] - 1) * 100 if qty > 0 and tp_i < len(tps) else np.nan}
    lines = pd.DataFrame({"sl": sl_line, "tp": tp_line, "radar": rd_line, "cost": cost_line,
                          "zt": zt, "zb": zb}, index=idx)
    return {"events": pd.DataFrame(events), "fills": pd.DataFrame(fills), "trades": T,
            "equity": eq, "stats": stats, "state": state, "lines": lines, "h1": h1,
            "start": st_ts, "diag": diag}
