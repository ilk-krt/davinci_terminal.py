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
from apex import structure_ctx as sctx
from apex.indicators import ema, rsi

PLANS = {
    "vurkac": "🎯 Vur-Kaç — tek hedef, hedefte tamamı satılır",
    "kademeli": "🪜 Kademeli (swing) — 3 hedefte 1/3'er satış, stop kademeli yukarı",
    "bedava": "🎁 Bedava lot — hedefte ana para çekilir, kâr lotları içeride koşar",
    "yonet": "🛡️ Sadece yönet — portföydeki mevcut pozisyonu yönetir (giriş yapmaz)",
}
MAX_STOP_ATR = 3.0
ZONES = {"0.5–0.618 (altın bölge)": (0.5, 0.618), "0.382–0.618": (0.382, 0.618),
         "0.236–0.618 (geniş)": (0.236, 0.618),
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
    profile: str = "temkinli"
    breakout: bool = False         # 20 saatlik zirve kırılımı ile giriş
    ema_bounce: bool = False       # 1s EMA21'den sekme ile giriş
    trend_ma: int = 200            # günlük trend filtresi ortalaması (200 / 50)
    stop_atr_max: float = 3.0      # stop en fazla kaç günlük ATR uzakta
    be_r: float = 1.0              # kaç R kârda stop maliyete
    trail_r: float = 1.5           # kaç R'den sonra iz süren stop
    trail_atr: float = 2.5         # iz süren stop: zirve − k × 1s ATR
    tp_r: tuple = ()               # boş değilse hedefler R katı (ör. 1.5, 3, 5)
    cooldown: int = 5              # çıkıştan sonra kaç mum bekle
    smart: bool = False            # 🧠 sinyal konseyi (öğrenen skor) ile giriş/çıkış
    horizon: int = 14              # konseyin öğrendiği ileri getiri ufku (1s mum)
    q_in: float = 0.85             # giriş eşiği: eğitim skorlarının yüzdeliği
    q_out: float = 0.35            # çıkış eşiği
    regime_gate: bool = False      # yalnızca günlük yükselen trendde (fiyat>EMA50>EMA200) gir
    max_losses: int = 0            # art arda bu kadar zararlı işlemden sonra mola (0 = kapalı)
    pause_days: int = 10           # mola süresi (işlem günü)
    risk_pct: float = 0.0          # işlem başına riske edilecek sermaye (0 = bütçenin tamamı)
    edge_gate: bool = False        # konsey: öğrenilen sinyallerin geçmişte kâr ettirdiği kanıtlanmalı
    trend_ride: bool = False       # güçlü trendde hedef tavanı kalkar, radar susar → kâr koşar
    structure: bool = True         # 🏛️ yapı katmanı: S/R, gap/FVG, Wyckoff, likidite, Elliott,
                                   #    Mum Gücü, haftalık trend → veto + stop + hedef + konsey
    struct_entry: bool = False     # 🏛️ yapısal destekten (S/R, boğa gap, talep, süpürme) giriş
    room_r: float = 1.2            # en yakın dirence en az kaç R mesafe olmalı


PROFILES: dict[str, dict] = {
    "temkinli": {"zone": (0.5, 0.618), "vol_mult": 1.5, "trend_ma": 200, "breakout": False,
                 "ema_bounce": False, "stop_atr_max": 3.0, "be_r": 1.0, "trail_r": 1.5,
                 "trail_atr": 2.5, "tp_r": (), "cooldown": 5, "trend_ride": True},
    "dengeli": {"zone": (0.382, 0.618), "vol_mult": 1.3, "trend_ma": 50, "breakout": False,
                "ema_bounce": True, "stop_atr_max": 2.5, "be_r": 1.0, "trail_r": 1.2,
                "trail_atr": 2.5, "tp_r": (), "cooldown": 3, "regime_gate": True,
                "max_losses": 2, "pause_days": 10, "risk_pct": 0.04, "trend_ride": True},
    "agresif": {"zone": (0.236, 0.618), "vol_mult": 1.2, "trend_ma": 50, "breakout": True,
                "ema_bounce": True, "stop_atr_max": 2.0, "be_r": 0.8, "trail_r": 1.0,
                "trail_atr": 3.0, "tp_r": (1.5, 3.0, 5.0), "cooldown": 2, "regime_gate": True,
                "max_losses": 2, "pause_days": 10, "risk_pct": 0.05, "trend_ride": True},
    "akilli": {"zone": (0.236, 0.618), "vol_mult": 1.2, "trend_ma": 50, "breakout": False,
               "ema_bounce": False, "stop_atr_max": 2.0, "be_r": 1.0, "trail_r": 1.5,
               "trail_atr": 3.0, "tp_r": (2.0, 4.0, 6.0), "cooldown": 3, "smart": True,
               "regime_gate": True, "max_losses": 2, "pause_days": 10, "risk_pct": 0.05,
               "edge_gate": True, "trend_ride": True},
    "yapisal": {"zone": (0.382, 0.618), "vol_mult": 1.3, "trend_ma": 50, "breakout": False,
                "ema_bounce": False, "stop_atr_max": 2.5, "be_r": 1.0, "trail_r": 1.5,
                "trail_atr": 2.5, "tp_r": (), "cooldown": 3, "struct_entry": True,
                "max_losses": 2, "pause_days": 10, "risk_pct": 0.05, "trend_ride": True,
                "room_r": 1.5},
}
PROFILE_TXT = {
    "temkinli": "🐢 Temkinli — yalnızca 0.5–0.618 Fib + güçlü hacim (1.5×), EMA200 üstü. Az ama "
                "seçici işlem.",
    "dengeli": "⚖️ Dengeli — 0.382–0.618 Fib + 1s EMA21 sekmesi, hacim 1.3×, EMA50 üstü.",
    "akilli": "🧠 Akıllı (sinyal konseyi) — APEX core, OMNI, Şahane, Konfluans, V710 (Q-dry, ODI, "
              "efor), VSA & Delta ve Mum Gücü'nden ~50 sinyali BİRLİKTE oylatır. Her sinyalin "
              "ağırlığını bu hissenin GEÇMİŞİNDEN öğrenir (yalnızca o ana kadarki veriyle, her "
              "~5 günde yeniden); işe yaramayan sinyaller kendiliğinden susar. Skor eğitim "
              "döneminin en iyi %15'ine çıkınca girer, en zayıf %35'ine düşünce 'sinyal bozuldu' "
              "diye çıkar. Yalnızca yükselen trend rejiminde ve öğrenilen sinyaller geçmişte "
              "gerçekten kâr ettirdiyse işlem açar. İşlem başına risk %5, art arda 2 zararda "
              "10 gün mola, güçlü trendde 🏄 trend modu.",
    "agresif": "🔥 Agresif — Fib 0.236–0.618 + EMA21 sekmesi + 20 saatlik zirve KIRILIMI, hacim "
               "1.2×, EMA50 üstü; daha sıkı stop (≤2 ATR), hedefler 1.5R / 3R / 5R, stop +0.8R'de "
               "maliyete, trend sürerken geniş iz süren stop (3 ATR) ile kâr koşturur. Yalnızca yükselen "
               "trendde girer, işlem başına risk %5, art arda 2 zararda 10 gün mola.",
    "yapisal": "🏛️ Yapısal — TradingView'de çizdiğimiz seviyelerle işlem yapar: fiyat S1/S2 "
               "desteğine, boğa FVG/GAP/KURUMSAL kutusuna, Wyckoff talep bölgesine ya da dip "
               "likidite süpürmesine değip hacimli yeşil 1s mumla dönünce (veya Fib 0.382–0.618 "
               "bölgesinde) girer. Stop en yakın anlamlı yapının altına, hedefler sıradaki direnç / "
               "açık gap dolumu / arz bölgesi / Elliott hedef kutusuna konur. Elliott ayı itkisi, "
               "haftalık düşüş trendi, dağıtım mumu ya da dirence çok yakınken girmez.",
}
STRUCT_TXT = ("🏛️ Yapı katmanı (bütün profillerde açık): girişten önce Elliott ayı itkisi, haftalık "
              "trend, Mum Gücü (dağıtım/fake) ve en yakın dirence mesafe kontrol edilir; stop en "
              "yakın S/R, gap kutusu, Wyckoff bölgesi ya da Elliott teyit seviyesinin altına; hedefler "
              "direnç, açık gap, arz bölgesi, likidite havuzu ve Elliott hedef kutusuna konur. "
              "Akıllı profilde bu seviyeler konseye ek sinyal olarak da girer.")


def params_for(profile: str, **over) -> "Params":
    kw = dict(PROFILES.get(profile, PROFILES["temkinli"]))
    kw.update({k: v for k, v in over.items() if v is not None})
    kw["profile"] = profile
    return Params(**kw)


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
    ctx["ema50"] = ema(d["Close"], 50)
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
    e21h = ema(h1["Close"], 21).to_numpy()
    e50h = ema(h1["Close"], 50).to_numpy()
    o, hi, lo, c, v = (h1[k].to_numpy() for k in ("Open", "High", "Low", "Close", "Volume"))
    a1, vm = atr1.to_numpy(), vma.to_numpy()
    idx = h1.index
    first = int(np.searchsorted(idx.values, np.datetime64(st_ts)))
    A = None
    if p.structure:
        try:
            TL = sctx.timeline(d, since=idx[0])
            A = sctx.align(idx, TL)
        except Exception:                        # noqa: BLE001
            A = None
    Acol = {k: A[k].to_numpy(float) for k in A.columns} if A is not None else {}

    def srow(i):
        return {k: v[i] for k, v in Acol.items()}
    CL = None
    if p.smart:
        from apex import council as cnl
        FH = cnl.engine_features(h1)
        FD = cnl.align_daily(idx, cnl.daily_features(d))
        parts = [FH, FD]
        if A is not None:
            parts.append(sctx.hourly_features(c, A))
            cnl.LABELS.update(sctx.LABELS)
        FALL = pd.concat(parts, axis=1).fillna(0.0)
        CL = cnl.learn_scores(FALL, c, horizon=p.horizon, q_in=p.q_in, q_out=p.q_out)
        CL["F"] = FALL
        sc_, ti_, to_ = CL["score"], CL["thr_in"], CL["thr_out"]
        w_ptr = [0]

        def _w_at(i):
            Wh = CL["W"]
            while w_ptr[0] + 1 < len(Wh) and Wh[w_ptr[0] + 1][0] <= i:
                w_ptr[0] += 1
            return Wh[w_ptr[0]][1] if Wh else None

        def _why(i):
            w_ = _w_at(i)
            if w_ is None:
                return ""
            return cnl.top_contrib(FALL.iloc[i], w_, CL["mu"], CL["sd"], CL["cols"])
    t_in_pos = 0
    loss_streak = 0
    pause_until = -1
    trend_on = False

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
    tp_lbl: list[str] = []
    diag = {"bar": 0, "yükselen bacak": 0, "Fib bölgesinde": 0,
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
        nonlocal tr_open, radar_on, tps, tp_i, cool, free_done, loss_streak, pause_until, trend_on
        trend_on = False
        if tr_open is None:
            return
        pnl = tr_open["sold"] - tr_open["cost"]
        trades.append({"Giriş": tr_open["t"], "Giriş fiyatı": tr_open["px"],
                       "Çıkış": idx[i], "Adet": tr_open["q"],
                       "K/Z $": pnl, "K/Z %": pnl / tr_open["cost"] * 100 if tr_open["cost"] else np.nan,
                       "R": pnl / (tr_open["risk"] * tr_open["q"]) if tr_open["risk"] > 0 else np.nan,
                       "Son çıkış": why, "Satışlar": " · ".join(tr_open["parts"])})
        tr_open, radar_on, tps, tp_i, cool, free_done = None, False, [], 0, p.cooldown, False
        loss_streak = loss_streak + 1 if pnl < 0 else 0
        if p.max_losses and loss_streak >= p.max_losses:
            pause_until = i + p.pause_days * 7
            ev(i, "⏸️ Mola", f"art arda {loss_streak} zararlı işlem — {p.pause_days} işlem günü "
               "yeni giriş yok (devre kesici)")
            loss_streak = 0

    def targets(lg: Leg, ent: float, risk: float, s: dict | None = None,
                atr_d: float = 0.0) -> list[float]:
        tp_lbl.clear()
        if s is not None and p.structure:
            return _s_targets(lg, ent, risk, s, atr_d)
        if p.tp_r:
            rr = [ent + k * risk for k in p.tp_r]
            if p.plan == "vurkac":
                return [rr[1] if len(rr) > 1 else rr[0]]
            if p.plan == "bedava":
                return [rr[0], rr[-1]]
            return rr[:3]
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

    def _s_targets(lg, ent, risk, s, atr_d):
        """Yapısal hedefler: direnç/gap/arz/likidite/Elliott (+ Fib uzantıları, R katları)."""
        buf = 0.1 * atr_d
        min_first = (p.tp_r[0] if p.tp_r else 1.2) * risk
        cand = [(v - buf, nm) for v, nm in sctx.resistances(s, ent)]
        if lg.ok:
            for k in (0.0, 1.272, 1.618, 2.618):
                v = lg.levels.get(k)
                if v is not None and v > ent:
                    cand.append((v - buf, "Fib tepe" if k == 0.0 else f"Fib {k}"))
        cand = sorted(t for t in cand if t[0] - ent >= min_first)
        merged: list[tuple[float, str]] = []
        for v, nm in cand:                       # birbirine çok yakın seviyeleri birleştir
            if merged and v - merged[-1][0] < max(0.3 * atr_d, 0.5 * risk):
                continue
            merged.append((v, nm))
        rmult = [ent + k * risk for k in (p.tp_r or (2.0, 3.0, 4.5))]
        for rv in rmult:
            if not any(abs(v - rv) < 0.75 * risk for v, _ in merged):
                merged.append((rv, f"{(rv - ent) / risk:.1f}R"))
        merged.sort()
        if p.plan == "vurkac":
            t = next((x for x in merged if x[0] - ent >= 1.5 * risk), merged[0])
            out = [t]
        elif p.plan == "bedava":
            out = [merged[0], merged[min(2, len(merged) - 1)]]
        else:
            out = merged[:3]
        tp_lbl.extend(nm for _, nm in out)
        return [v for v, _ in out]

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
        tps = targets(lg if lg.ok else Leg(True, True, {}), max(entry, float(c[i0])), r0,
                      srow(i0) if A is not None else None, atr_d)
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
            t_in_pos += 1
            # 0) 🧠 konsey: sinyal bozulduysa çık
            if (p.smart and t_in_pos >= 4 and np.isfinite(sc_[i]) and np.isfinite(to_[i])
                    and sc_[i] <= to_[i] and lo[i] > stop
                    and np.isfinite(e21h[i]) and c[i] < e21h[i]):     # fiyat da 1s EMA21 altında
                q_ = qty
                fill(i, "SAT", q_, c[i], "sinyal bozuldu")
                ev(i, "🧠 Sinyal bozuldu", f"konsey skoru {sc_[i]:+.2f} ≤ çıkış eşiği {to_[i]:+.2f} "
                   f"— {q_:g} adet {c[i]:.2f}'den satıldı. Ağırlıklı etkenler: {_why(i)}", c[i])
                close_trade(i, "sinyal bozuldu")
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
                    if c[i] >= entry + p.be_r * r0:
                        stop = max(stop, entry)
                    if c[i] >= entry + p.trail_r * r0 and np.isfinite(a1[i]):
                        stop = max(stop, hh - p.trail_atr * a1[i])
                    if stop > prev + 0.25 * atr_d:
                        ev(i, "🛡️ Stop yukarı", f"{prev:.2f} → {stop:.2f}")
                    # 4) dinamik hedef (gün başında)
                    ax, pdi, mdi, rs = cr["adx"], cr["pdi"], cr["mdi"], cr["rsi"]
                    strong = bool(p.trend_ride and ax >= p.adx_strong and pdi > mdi
                                  and cr["close"] > cr["ema50"] > cr["ema200"])
                    if strong and new_day and tp_i < len(tps):
                        t = tps[tp_i]
                        nt = max(t, c[i] + 1.0 * atr_d)        # tavan yok: hedef fiyatın 1 ATR önünde
                        if nt > t + 0.25 * atr_d:
                            tps[tp_i] = nt
                            if not trend_on:
                                ev(i, "🏄 Trend modu", f"güçlü trend (ADX {ax:.0f}, fiyat>EMA50>EMA200): "
                                   "hedef tavanı kalktı, radar sustu — kâr iz süren stopla koşturuluyor")
                    if strong != trend_on:
                        if trend_on and not strong:
                            ev(i, "🏁 Trend modu bitti", f"ADX {ax:.0f} — normal hedef/radar kuralları")
                        trend_on = strong
                    if (not strong) and new_day and tp_i < len(tps):
                        t = tps[tp_i]
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
                    # 5) %85 radar & giyotin (trend modunda susar)
                    if tp_i < len(tps) and not trend_on:
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
        elif p.plan != "yonet" and (p.loop or not trades) and cool <= 0 and i >= pause_until:
            ma = cr["ema200"] if p.trend_ma >= 200 else cr["ema50"]
            ok_trend = (not p.trend_filter) or (np.isfinite(ma) and cr["close"] > ma)
            if p.regime_gate:
                reg = bool(np.isfinite(cr["ema50"]) and np.isfinite(cr["ema200"])
                           and cr["close"] > cr["ema50"] > cr["ema200"])
                diag["yükselen trend rejimi"] = diag.get("yükselen trend rejimi", 0) + int(reg)
                ok_trend = ok_trend and reg
            diag["bar"] += 1
            up_ok = lg.ok and lg.up and np.isfinite(zt[i])
            touched = up_ok and np.nanmin(lo[max(0, i - 6):i + 1]) <= zt[i]
            in_zone = bool(up_ok and c[i] >= zb[i] and touched and c[i] <= zt[i] + 0.5 * atr_d)
            rng = hi[i] - lo[i]
            green = c[i] > o[i] and rng > 0 and (c[i] - lo[i]) / rng >= 0.5
            vol_ok = bool(np.isfinite(vm[i]) and vm[i] > 0 and v[i] >= p.vol_mult * vm[i])
            vol_lite = bool(np.isfinite(vm[i]) and vm[i] > 0 and v[i] >= 1.0 * vm[i])
            brk = bool(p.breakout and i > 20 and c[i] > np.max(hi[i - 20:i]) and green and vol_ok)
            bounce = bool(p.ema_bounce and np.isfinite(e21h[i]) and lo[i] <= e21h[i] < c[i]
                          and e21h[i] > e50h[i] and green and vol_lite)
            fib_sig = bool(up_ok and in_zone and green and vol_ok)
            diag["yükselen bacak"] += int(up_ok)
            diag["trend (EMA200)" if p.trend_ma >= 200 else "trend (EMA50)"] = \
                diag.get("trend (EMA200)" if p.trend_ma >= 200 else "trend (EMA50)", 0) + int(ok_trend)
            diag["Fib bölgesinde"] += int(in_zone)
            diag["yeşil mum"] += int(green)
            diag["hacim ≥ eşik"] += int(vol_ok)
            if p.breakout:
                diag["zirve kırılımı"] = diag.get("zirve kırılımı", 0) + int(brk)
            if p.ema_bounce:
                diag["EMA21 sekmesi"] = diag.get("EMA21 sekmesi", 0) + int(bounce)
            why = ("Fib bölgesi + hacimli yeşil 1s mum" if fib_sig else
                   "20 saatlik zirve kırılımı (hacimli)" if brk else
                   "1s EMA21 sekmesi" if bounce else "")
            sname = ""
            if A is not None and p.struct_entry:
                sname = sctx.touched_support(srow(i), float(np.min(lo[max(0, i - 6):i + 1])),
                                             c[i], atr_d)
                s_sig = bool(sname and green and vol_lite)
                diag["🏛️ yapısal destekten dönüş"] = diag.get("🏛️ yapısal destekten dönüş", 0) + int(s_sig)
                if s_sig and not why:
                    why = f"🏛️ {sname} üstünde hacimli yeşil 1s mum"
            if p.smart:
                smart_ok = bool(np.isfinite(sc_[i]) and np.isfinite(ti_[i]) and sc_[i] >= ti_[i])
                if p.edge_gate:
                    e_, b_ = CL["edge"][i], CL["base"][i]
                    edge_ok = bool(np.isfinite(e_) and e_ > 0 and e_ > b_)
                    diag["öğrenilen avantaj > 0"] = diag.get("öğrenilen avantaj > 0", 0) + int(edge_ok)
                    smart_ok = smart_ok and edge_ok
                diag["konsey skoru ≥ eşik"] = diag.get("konsey skoru ≥ eşik", 0) + int(smart_ok)
                why = (f"🧠 konsey skoru {sc_[i]:+.2f} ≥ eşik {ti_[i]:+.2f} · {_why(i)}"
                       if smart_ok else "")
                fib_sig = False
            S = srow(i) if A is not None else None
            if S is not None and why and ok_trend:
                vts = sctx.vetoes(S)
                if vts:
                    diag["🏛️ yapı vetosu"] = diag.get("🏛️ yapı vetosu", 0) + 1
                    why = ""
            if ok_trend and why:
                px = c[i]
                swing_lo = float(np.min(lo[max(0, i - 10):i + 1]))
                if fib_sig:
                    sl0 = min(lg.levels.get(0.786, zb[i]), swing_lo) - 0.25 * atr_d
                else:
                    sl0 = min(swing_lo, e50h[i] if np.isfinite(e50h[i]) else swing_lo) - 0.25 * atr_d
                sl0 = max(sl0, px - p.stop_atr_max * atr_d)
                stop_why = "salınım dibi" if not fib_sig else "Fib 0.786 / salınım dibi"
                if S is not None:
                    for lv, nm in sctx.supports(S, px):
                        if 0.5 * atr_d <= px - lv <= p.stop_atr_max * atr_d:
                            # Fib girişinde Fib stopu korunur; yapı yalnızca DAHA AŞAĞIDAYSA
                            # (stop avı yapılan kümenin altına) stopu taşır
                            if (not fib_sig) or lv - 0.25 * atr_d < sl0:
                                sl0, stop_why = lv - 0.25 * atr_d, nm
                            break
                risk = px - sl0
                # Fib geri çekilme girişi tasarım gereği ara direncin altından alır (hedef bacak
                # tepesi) → mesafe kontrolü yalnızca diğer giriş türlerine uygulanır
                if S is not None and risk > 0 and not fib_sig:
                    blk = sctx.blockers(S, px * 1.001)
                    if blk and blk[0][0] - px < min(p.room_r * risk, 2.5 * atr_d):
                        diag["🏛️ dirence çok yakın"] = diag.get("🏛️ dirence çok yakın", 0) + 1
                        risk = -1.0
                if risk > 0:
                    diag["hepsi birden"] += 1
                q = math.floor(cash / px) if px > 0 else 0
                if p.risk_pct and risk > 0:
                    eq_now = cash + qty * px
                    q = min(q, math.floor(eq_now * p.risk_pct / risk))
                if risk > 0 and q > 0:
                    fill(i, "AL", q, px, "akıllı kontrol")
                    entry, stop, r0, hh = px, sl0, risk, px
                    t_in_pos = 0
                    exts = sorted(x for x in (lg.levels.get(k) for k in (0.0, 1.272, 1.618, 2.618))
                                  if x is not None) if lg.ok else []
                    tps, tp_i, radar_on, free_done = targets(lg if lg.ok else Leg(True, True, {}),
                                                             px, risk, S, atr_d), 0, False, False
                    tps0 = list(tps)
                    tr_open = {"t": idx[i], "px": px, "q": q, "cost": q * px, "sold": 0.0,
                               "risk": risk, "parts": []}
                    if tp_lbl:
                        ht = ", ".join(f"{x:.2f} ({nm})" for x, nm in zip(tps, tp_lbl))
                    else:
                        ht = ", ".join(f"{x:.2f}" for x in tps)
                    ev(i, "✅ Giriş", f"{why} (hacim {v[i] / vm[i]:.1f}×): {q:g} adet @ {px:.2f} · "
                       f"stop {sl0:.2f} ({stop_why} altı) · hedefler {ht}", px)
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
    if A is not None:
        for k in ("s1", "r1", "gb_top", "gb_bot", "gr_bot", "gr_top", "dz_top", "dz_bot",
                  "ell_zlo", "ell_zhi", "ell_sgn", "ell_kind", "ell_dead"):
            lines[k] = A[k].to_numpy(float)
    return {"events": pd.DataFrame(events), "fills": pd.DataFrame(fills), "trades": T,
            "equity": eq, "stats": stats, "state": state, "lines": lines, "h1": h1,
            "start": st_ts, "diag": diag,
            "council": ({"score": pd.Series(CL["score"], index=idx),
                         "thr_in": pd.Series(CL["thr_in"], index=idx),
                         "thr_out": pd.Series(CL["thr_out"], index=idx),
                         "weights": __import__("apex.council", fromlist=["x"]).weights_table(CL)}
                        if CL is not None else None)}


def optimize(df1h: pd.DataFrame, dfd: pd.DataFrame, start, budget: float = 10_000.0,
             plans=("vurkac", "kademeli", "bedava"), compare_structure: bool = True) -> pd.DataFrame:
    """Aynı sembol ve dönemde profiller × planları dener; getiriye göre sıralar.
    compare_structure: her profil bir kez de yapı katmanı KAPALI (kademeli) denenir."""
    rows = []
    combos = [(prof, pl, True) for prof in PROFILES for pl in plans]
    if compare_structure:
        combos += [(prof, "kademeli", False) for prof in PROFILES if prof != "yapisal"]
    for prof, pl, stc in combos:
        try:
            R = run(df1h, dfd, params_for(prof, plan=pl, budget=budget, structure=stc), start=start)
        except Exception as exc:                 # noqa: BLE001
            rows.append({"Profil": prof, "Plan": pl, "Yapı": "✅" if stc else "❌", "Hata": str(exc)})
            continue
        s_ = R["stats"]
        dd = s_["Maks. düşüş %"]
        rows.append({"Profil": prof, "Plan": pl, "Yapı": "✅" if stc else "❌",
                     "Getiri %": s_["Getiri %"],
                     "Al-tut %": s_["Al-tut %"], "Maks. düşüş %": dd,
                     "Getiri / düşüş": (s_["Getiri %"] / abs(dd)) if dd and dd < 0 else np.nan,
                     "İşlem": s_["İşlem"] + (1 if R["state"]["pozisyonda"] else 0),
                     "Kazanan %": s_["Kazanan %"], "Ort. R": s_["Ort. R"],
                     "Şu an": "pozisyonda" if R["state"]["pozisyonda"] else "nakitte"})
    out = pd.DataFrame(rows)
    if "Getiri %" in out:
        out = out.sort_values("Getiri %", ascending=False).reset_index(drop=True)
    return out
