# apex/orgu.py — 🧬 ÖRGÜ TARAYICI: büyük hareket öncesi örgüyü bugün arar
#
# Patern madenciliği (tools/pattern_miner.py, ~730 hisse × 5 yıl, walk-forward) üç
# kullanılabilir bulgu verdi; burada her gün bütün hisselere uygulanır:
#
#  1) ŞABLON BENZERLİĞİ — oynaklığına göre büyük yükselişten (SPY'ye göre 20 günde
#     ≥ 1.5 × ATR%·√20) önceki ORTALAMA 30 günlük örgü (WHALE, DAILY, RETAIL, PRO−RETAIL,
#     OMNI, Fusion, Synergy, sıkışma süresi, hacim, RS, gece−gündüz farkı; 3 günde bir nokta).
#     Hissenin son 30 günü bu şablona ne kadar benziyor? Test döneminde en benzer %1:
#     büyük hareket olasılığı tabanın 1.8–2.2 katı (iki ayrı dönemde).
#  2) SESSİZ SIKIŞMA — parmak izinin özeti: oynaklık daralması (ATR14/ATR50, Bollinger),
#     hacim kuruması, yüksek ama azalan fiyat etkisi (Kyle-λ), iyileşen likidite (Amihud),
#     yavaşça artan Whale gücü, uzayan sıkışma. Testte tutan iki kural ayrıca işaretlenir.
#  3) 🐈 ÖLÜ KEDİ — büyük DÜŞÜŞTEN önceki parmak izi: EMA200 altında, zirveden uzak bir
#     hissede ani OMNI/HUD/fiyat sıçraması. (Parmak izinden türetildi; kural olarak ayrıca
#     test edilmesi patern madenciliğinin bir sonraki çalıştırmasında yapılır.)
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from apex import engine as eng
from apex import pine_fib as pfib

_log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
PAT = ROOT / "data" / "patterns"

SEQ_KEYS = ["whale", "daily", "retail", "pro_ret", "omni", "f_hist", "s_speed", "sqz_dur",
            "vol_10_60", "rs20", "x_on_id"]
LAGS = list(range(0, 31, 3))
# sablon.csv'deki etiket → seri
SEQ_LABEL = {"WHALE bandı": "whale", "DAILY bandı": "daily", "RETAIL bandı": "retail",
             "PRO − RETAIL": "pro_ret", "OMNI momentum": "omni", "Fusion histogram": "f_hist",
             "Synergy hızı": "s_speed", "Sıkışma süresi (gün)": "sqz_dur",
             "Hacim 10 gün / 60 gün": "vol_10_60", "SPY'ye göre 20 gün %": "rs20",
             "Gece − gün içi getiri farkı (20 gün) — kurumsal birikim": "x_on_id"}
SHOW = {"whale": "WHALE", "pro_ret": "PRO − RETAIL", "omni": "OMNI momentum",
        "s_speed": "Synergy hızı", "rs20": "SPY'ye göre 20G %", "sqz_dur": "Sıkışma süresi"}
EXP = "A · upz · 5 yıl (1s verisi hariç)"


# ------------------------------------------------------------------ seriler
def series(d: pd.DataFrame, spy: pd.Series | None) -> pd.DataFrame:
    """Örgü serileri — tools/pattern_miner.py ile BİREBİR aynı tanımlar (nedensel)."""
    d = d[["Open", "High", "Low", "Close", "Volume"]].astype(float).dropna(subset=["Close"])
    d = d.assign(Volume=d["Volume"].fillna(0.0))
    o, h, l, c, v = (d[k] for k in ("Open", "High", "Low", "Close", "Volume"))
    X = pd.DataFrame(index=d.index)
    core, omni = eng.apex_core(d), eng.apex_omni(d)
    X["whale"], X["daily"], X["retail"] = core["whale"], core["daily"], core["retail"]
    X["pro_ret"] = core["pro"] - core["ret_line"]
    X["w_pwr"] = core["w_pwr"]
    X["sqz_dur"] = core["sqz_dur"].astype(float)
    X["omni"] = omni["mom"]
    X["omni_d5"] = omni["mom"] - omni["mom"].shift(5)
    X["f_hist"], X["s_speed"], X["hud_net"] = omni["f_hist"], omni["s_speed"], omni["hud_net"]
    X["vol_10_60"] = v.rolling(10).mean() / v.rolling(60).mean().replace(0, np.nan)
    X["ret5"] = (c / c.shift(5) - 1) * 100
    X["ret20"] = (c / c.shift(20) - 1) * 100
    if spy is not None:
        sp = spy.reindex(d.index).ffill()
        X["rs20"] = X["ret20"] - (sp / sp.shift(20) - 1) * 100
    else:
        X["rs20"] = np.nan
    X["x_on_id"] = (np.log(o / c.shift(1)).rolling(20).sum()
                    - np.log(c / o).rolling(20).sum()) * 100
    atr = pfib.atr_rma(d, 14)
    X["atr_pct"] = atr / c * 100
    X["atr_ratio"] = atr / pfib.atr_rma(d, 50)
    bbw = c.rolling(20).std() / c.rolling(20).mean()
    X["bbw_rank"] = bbw.rolling(120, min_periods=60).rank(pct=True)
    ret1 = c.pct_change()
    am = ret1.abs() / (c * v).replace(0, np.nan)
    X["x_amihud"] = np.log(am.rolling(20).mean() / am.rolling(120).mean())
    sq = np.sign(ret1) * np.sqrt((c * v).clip(lower=1))
    lam = ret1.rolling(60).cov(sq) / sq.rolling(60).var().replace(0, np.nan) * 1e6
    X["x_lambda"], X["x_lambda_ch"] = lam, lam - lam.shift(20)
    e200 = c.ewm(span=200, adjust=False).mean()
    X["below200"] = (c < e200).astype(float)
    hi = h.rolling(252, min_periods=120).max()
    X["from_hi"] = (c / hi - 1) * 100
    X["wpwr_30"] = X["w_pwr"] - X["w_pwr"].shift(30)
    X["olu_kedi"] = dead_cat(X).astype(float)
    return X


def dead_cat(X: pd.DataFrame) -> pd.Series:
    """🐈 Düşen trendde (EMA200 altı, zirveden ≥ %30 uzak) ani boğa sıçraması."""
    jump = X["ret5"] >= np.maximum(5.0, 1.5 * X["atr_pct"])
    mom = (X["omni_d5"] >= 10) | (X["hud_net"] >= 3)
    return (X["below200"] > 0) & (X["from_hi"] <= -30) & jump & mom


# ------------------------------------------------------------------ şablon
def load_template(exp: str = EXP) -> dict[str, Any] | None:
    """Önce sablon_model.json (ortalama/std ile), yoksa sablon.csv (özgün birimde fark)."""
    mj = PAT / "sablon_model.json"
    if mj.exists():
        try:
            m = json.loads(mj.read_text())
            if exp in m:
                return {"tip": "model", **m[exp]}
        except Exception as exc:                 # noqa: BLE001
            _log.info("sablon_model: %s", exc)
    sc = PAT / "sablon.csv"
    if not sc.exists():
        return None
    try:
        S = pd.read_csv(sc)
        S = S[S["Deney"] == exp]
        if S.empty:
            return None
        diff = {}
        for _, r in S.iterrows():
            k = SEQ_LABEL.get(r["Seri"])
            if not k:
                continue
            for L in LAGS:
                if str(L) in S.columns and np.isfinite(r[str(L)]):
                    diff[f"{k}_{L}"] = float(r[str(L)])
        return {"tip": "fark", "fark": diff}
    except Exception as exc:                     # noqa: BLE001
        _log.info("sablon.csv: %s", exc)
        return None


def seq_vector(X: pd.DataFrame) -> dict[str, float]:
    out = {}
    for k in SEQ_KEYS:
        s = X[k].to_numpy(float)
        for L in LAGS:
            out[f"{k}_{L}"] = float(s[-1 - L]) if len(s) > L else np.nan
    return out


# ------------------------------------------------------------------ tarama
RULES = {
    "R1": "🧬 Sıkışmış oynaklık + yüksek fiyat etkisi (ATR > %1.9, Kyle-λ > 2.3, "
          "ATR14/ATR50 ≤ 0.69) — testte 2.0 kat",
    "R2": "🧬 Hacim kuruması + likidite iyileşmesi + azalan fiyat etkisi (λ değişimi ≤ −0.34, "
          "λ ≤ 16.5, hacim 10/60 ≤ 0.74, Amihud ≤ −0.23) — testte 1.6 kat",
}


def scan(prices: dict[str, pd.DataFrame], spy: pd.Series | None,
         theme_of: dict[str, str] | None = None) -> pd.DataFrame:
    """Bütün hisseler için bugünkü örgü durumu (en güncel bar)."""
    rows, seqs = [], {}
    for s, d in prices.items():
        if d is None or len(d) < 160:
            continue
        try:
            X = series(d, spy)
        except Exception as exc:                 # noqa: BLE001
            _log.info("orgu %s: %s", s, exc)
            continue
        r = X.iloc[-1]
        k5 = X["olu_kedi"].tail(5)
        rows.append({
            "Sembol": s, "Tema": (theme_of or {}).get(s, ""),
            "Fiyat": float(d["Close"].iloc[-1]), "ATR %": float(r["atr_pct"]),
            "ATR14/50": float(r["atr_ratio"]), "BB yüzdelik": float(r["bbw_rank"]),
            "Hacim 10/60": float(r["vol_10_60"]), "Kyle-λ": float(r["x_lambda"]),
            "λ değişimi": float(r["x_lambda_ch"]), "Amihud": float(r["x_amihud"]),
            "Whale gücü Δ30": float(r["wpwr_30"]), "Sıkışma (gün)": float(r["sqz_dur"]),
            "SPY'ye göre 20G %": float(r["rs20"]),
            "🐈 Ölü kedi": bool(k5.max() > 0),
            "Ölü kedi günü": (str(k5[k5 > 0].index[-1].date()) if k5.max() > 0 else ""),
            "Veri tarihi": str(d.index[-1])[:10]})
        seqs[s] = seq_vector(X)
    if not rows:
        return pd.DataFrame()
    T = pd.DataFrame(rows).set_index("Sembol")
    # sessiz sıkışma puanı (0–100)
    lam_rank = T["Kyle-λ"].rank(pct=True)
    comp = pd.DataFrame({
        "oynaklık daralması": ((1.05 - T["ATR14/50"]) / 0.35).clip(0, 1),
        "dar Bollinger": (1 - T["BB yüzdelik"]).clip(0, 1),
        "hacim kuruması": ((1.0 - T["Hacim 10/60"]) / 0.35).clip(0, 1),
        "yüksek fiyat etkisi": lam_rank,
        "azalan fiyat etkisi": (T["λ değişimi"] < 0).astype(float),
        "iyileşen likidite": (-T["Amihud"] / 0.5).clip(0, 1),
        "artan whale gücü": (T["Whale gücü Δ30"] / 20).clip(0, 1),
        "uzayan sıkışma": (T["Sıkışma (gün)"] / 10).clip(0, 1),
    }).fillna(0.0)
    W = {"oynaklık daralması": .20, "dar Bollinger": .15, "hacim kuruması": .15,
         "yüksek fiyat etkisi": .15, "azalan fiyat etkisi": .10, "iyileşen likidite": .10,
         "artan whale gücü": .10, "uzayan sıkışma": .05}
    T["Sessiz sıkışma"] = (sum(comp[k] * w for k, w in W.items()) * 100).round(0)
    T["Öne çıkan"] = comp.apply(lambda r: " · ".join(k for k in r.sort_values(ascending=False)
                                                       .index[:3] if r[k] >= 0.6), axis=1)
    T["R1"] = (T["ATR %"] > 1.88) & (T["Kyle-λ"] > 2.30) & (T["ATR14/50"] <= 0.69)
    T["R2"] = ((T["λ değişimi"] <= -0.34) & (T["Kyle-λ"] <= 16.46) & (T["Hacim 10/60"] <= 0.74)
               & (T["Amihud"] <= -0.23))
    # şablon benzerliği
    tp = load_template()
    T["Şablon benzerliği"] = np.nan
    if tp is not None and seqs:
        Q = pd.DataFrame.from_dict(seqs, orient="index")
        if tp["tip"] == "model":
            cols = [c for c in tp["cols"] if c in Q.columns]
            mu = pd.Series(tp["mu"], index=tp["cols"])[cols]
            sd = pd.Series(tp["sd"], index=tp["cols"])[cols].replace(0, 1)
            tz = pd.Series(tp["tmpl_z"], index=tp["cols"])[cols]
        else:
            cols = [c for c in tp["fark"] if c in Q.columns]
            mu, sd = Q[cols].mean(), Q[cols].std().replace(0, 1)
            tz = pd.Series(tp["fark"])[cols] / sd
        Z = ((Q[cols] - mu) / sd).fillna(0.0)
        num = Z.to_numpy() @ tz.to_numpy()
        den = np.linalg.norm(Z.to_numpy(), axis=1) * (np.linalg.norm(tz.to_numpy()) or 1)
        sim = pd.Series(num / np.where(den == 0, 1, den), index=Q.index)
        T["Şablon benzerliği"] = sim.reindex(T.index)
    T["Benzerlik yüzdeliği"] = (T["Şablon benzerliği"].rank(pct=True) * 100).round(1)
    T["Örgü puanı"] = (0.6 * T["Benzerlik yüzdeliği"].fillna(50)
                       + 0.4 * T["Sessiz sıkışma"]).round(0)
    T.attrs["sablon"] = tp is not None
    return T.reset_index().sort_values("Örgü puanı", ascending=False).reset_index(drop=True)


def verdict_note(row: dict[str, Any]) -> tuple[str, str]:
    """Hisse Avcısı için kısa örgü etiketi ve açıklama."""
    if row.get("🐈 Ölü kedi"):
        return "🐈 ÖLÜ KEDİ", (f"düşen trendde ani sıçrama ({row.get('Ölü kedi günü', '')}) — "
                               "büyük düşüşlerden önceki örgüye benziyor")
    tags = []
    by = row.get("Benzerlik yüzdeliği")
    if by is not None and np.isfinite(by) and by >= 95:
        tags.append(f"şablona benzerlik ilk %{100 - by:.0f}")
    if row.get("R1"):
        tags.append("R1 sıkışma kuralı")
    if row.get("R2"):
        tags.append("R2 kuruma kuralı")
    ss = row.get("Sessiz sıkışma")
    if ss is not None and np.isfinite(ss) and ss >= 70:
        tags.append(f"sessiz sıkışma {ss:.0f}")
    if tags:
        return "🧬 ÖRGÜ", " · ".join(tags)
    return "", ""


# ------------------------------------------------------------------ grafik
def chart(d: pd.DataFrame, spy: pd.Series | None, sym: str):
    """Hissenin son 30 günü ile şablonun ŞEKİL karşılaştırması.

    Benzerlik ölçüsü (kosinüs) ölçekten bağımsız olduğu için iki eğri de standart birime
    çevrilir ve şablon, hissenin eğrisiyle aynı yüksekliğe ölçeklenir: önemli olan
    seviyeler değil, şeklin (ne zaman düşüp ne zaman döndüğünün) benzerliğidir."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from apex.tvchart import BG, GRID, MUTED, PANEL, TXT
    tp = load_template()
    X = series(d, spy)
    fig, axs = plt.subplots(2, 3, figsize=(10.5, 4.6), dpi=110, facecolor=BG)
    for ax, k in zip(axs.ravel(), list(SHOW)):
        ax.set_facecolor(PANEL)
        ax.grid(color=GRID, lw=0.5)
        ax.tick_params(colors=MUTED, labelsize=6.5)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        full = X[k].astype(float)
        if tp is not None and tp["tip"] == "model" and f"{k}_0" in tp["cols"]:
            j = tp["cols"].index(f"{k}_0")
            m_, s_ = tp["mu"][j], tp["sd"][j] or 1.0
        else:
            m_, s_ = float(full.tail(252).mean()), float(full.tail(252).std() or 1.0)
        z = ((full.tail(31) - m_) / s_).to_numpy(float)
        xs = np.arange(-len(z) + 1, 1)
        ax.axhline(0, color=GRID, lw=0.8)
        ax.plot(xs, z, color="#29b6f6", lw=1.6, label=sym)
        if tp is not None:
            if tp["tip"] == "model":
                tz = dict(zip(tp["cols"], tp["tmpl_z"]))
                pts = [(-L, tz[f"{k}_{L}"]) for L in LAGS if f"{k}_{L}" in tz]
            else:
                pts = [(-L, tp["fark"][f"{k}_{L}"] / s_) for L in LAGS if f"{k}_{L}" in tp["fark"]]
            if pts:
                px, py = zip(*sorted(pts))
                py = np.array(py, float)
                sc = (np.nanmax(np.abs(z)) / np.nanmax(np.abs(py))) if np.nanmax(np.abs(py)) > 0 else 1
                ax.plot(px, py * sc, color="#ffd54f", lw=1.3, ls="--", marker="o", ms=2.5,
                        label="şablon şekli")
        ax.set_title(SHOW[k], color=TXT, fontsize=8)
    axs[0, 0].legend(fontsize=6, facecolor=PANEL, edgecolor=GRID, labelcolor=TXT)
    fig.suptitle(f"{sym} · son 30 gün (mavi) ve büyük yükseliş öncesi ortalama örgü (sarı) — "
                 "şekil karşılaştırması, standart birim", color=TXT, fontsize=9)
    fig.tight_layout()
    return fig
