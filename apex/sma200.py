# apex/sma200.py — 🔻 200 GÜNLÜK ORTALAMA (SMA200) OLAY ÇALIŞMASI
#
# "Son kez 200 günlük ortalamanın altına indiğinde %300 koştu" türü iddiaları TEK hisseye bakarak
# değil, sistematik ölçer:
#   OLAYLAR (her hisse, nedensel — yalnız o güne kadarki veri):
#     🔻 kırılım : ≥ 50 gün SMA200 üstünde kaldıktan sonra İLK kapanış altında
#     🔻 dokunuş : son 60 günde SMA200'ün ≥ %10 üstüne çıkmış; bugün düşük ≤ SMA200 × 1.01 ama
#                  kapanış üstünde (ortalamadan sekme) — "dip alımı" klasiği
#     🔺 geri alış: ≥ 20 gün altta kaldıktan sonra İLK kapanış üstünde
#   Her olayda: SMA200 eğimi (20 günde yükselen mi), sonraki 20/60/120 gün getirisi, SPY'ye göre
#   fark, 120 gün içindeki en yüksek yükseliş ve en derin düşüş.
#   HİSSE BAŞINA özet (arayüz) + EVREN testi (tools/validate_signals.py --bolum sma200).
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)
HZ = (20, 60, 120)
EV = {"kirilim": "🔻 Kırılım (ilk kapanış altında)",
      "dokunus": "🔻 Dokunuş (ortalamadan sekme)",
      "geri_alis": "🔺 Geri alış (ilk kapanış üstünde)"}


def frame(d: pd.DataFrame) -> pd.DataFrame:
    c, lo = d["Close"].astype(float), d["Low"].astype(float)
    s = c.rolling(200, min_periods=200).mean()
    above = c > s
    run_up = above.groupby((above != above.shift()).cumsum()).cumcount() + 1
    up_prev = run_up.shift(1).where(above.shift(1, fill_value=False), 0)
    dn_prev = run_up.shift(1).where(~above.shift(1, fill_value=True), 0)
    ext60 = (c / s - 1).rolling(60, min_periods=20).max()
    F = pd.DataFrame({"c": c, "s": s, "egim": s / s.shift(20) - 1, "above": above})
    F["kirilim"] = (~above) & above.shift(1, fill_value=False) & (up_prev >= 50) & s.notna()
    F["dokunus"] = (above & (lo <= s * 1.01) & (ext60.shift(1) >= 0.10) & s.notna()
                    & ~(above.shift(1, fill_value=False) & (lo.shift(1) <= s.shift(1) * 1.01)))
    F["geri_alis"] = above & ~above.shift(1, fill_value=True) & (dn_prev >= 20) & s.notna()
    return F


def _decluster(mask: pd.Series, gap: int = 20) -> np.ndarray:
    out, last = [], -10 ** 9
    for i in np.flatnonzero(mask.fillna(False).to_numpy()):
        if i - last >= gap:
            out.append(i)
            last = i
    return np.asarray(out, int)


def events(d: pd.DataFrame, bench: pd.Series | None = None) -> pd.DataFrame:
    """Tek hissenin bütün SMA200 olayları ve sonrası."""
    if d is None or len(d) < 260:
        return pd.DataFrame()
    F = frame(d)
    c = F["c"]
    hi, lo = d["High"].astype(float), d["Low"].astype(float)
    b = bench.reindex(c.index).ffill() if bench is not None else None
    rows = []
    n = len(c)
    for kod, lab in EV.items():
        for i in _decluster(F[kod]):
            rec = {"Tarih": c.index[i], "Olay": lab, "_kod": kod, "Fiyat": float(c.iloc[i]),
                   "SMA200": float(F["s"].iloc[i]),
                   "SMA200 eğimi": "↗ yükselen" if F["egim"].iloc[i] > 0 else "↘ düşen",
                   "_egim_up": bool(F["egim"].iloc[i] > 0)}
            for h in HZ:
                j = i + h
                if j < n:
                    r = (c.iloc[j] / c.iloc[i] - 1) * 100
                    rec[f"{h}G %"] = float(r)
                    if b is not None and np.isfinite(b.iloc[i]) and np.isfinite(b.iloc[j]):
                        rec[f"{h}G SPY'ye göre"] = float(r - (b.iloc[j] / b.iloc[i] - 1) * 100)
                    else:
                        rec[f"{h}G SPY'ye göre"] = np.nan
                else:
                    rec[f"{h}G %"] = rec[f"{h}G SPY'ye göre"] = np.nan
            w = slice(i + 1, min(i + 121, n))
            if i + 1 < n:
                rec["120G en yüksek %"] = float((hi.iloc[w].max() / c.iloc[i] - 1) * 100)
                rec["120G en derin %"] = float((lo.iloc[w].min() / c.iloc[i] - 1) * 100)
            else:
                rec["120G en yüksek %"] = rec["120G en derin %"] = np.nan
            rec["Gün önce"] = n - 1 - i
            rows.append(rec)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Tarih", ascending=False).reset_index(drop=True)


def now(d: pd.DataFrame) -> dict[str, Any]:
    """Bugünkü durum: SMA200'e uzaklık, eğim, son 5 günde olay var mı?"""
    if d is None or len(d) < 205:
        return {"ok": False}
    F = frame(d)
    i = len(F) - 1
    s, c = float(F["s"].iloc[i]), float(F["c"].iloc[i])
    rec = {"ok": True, "sma200": s, "uzaklik %": (c / s - 1) * 100,
           "egim_up": bool(F["egim"].iloc[i] > 0), "ustte": bool(c > s), "olay": "", "olay_gun": None}
    for kod in ("dokunus", "kirilim", "geri_alis"):
        a = F[kod].to_numpy()[-5:]
        if a.any():
            g = 4 - int(np.flatnonzero(a)[-1])
            rec["olay"], rec["olay_gun"], rec["_kod"] = EV[kod], g, kod
            break
    return rec


def summary(E: pd.DataFrame, by=("Olay",)) -> pd.DataFrame:
    if E is None or E.empty:
        return pd.DataFrame()
    rows = []
    for k, g in E.groupby(list(by), sort=False):
        k = k if isinstance(k, tuple) else (k,)
        rec = dict(zip(by, k))
        rec["Olay sayısı"] = len(g)
        for h in HZ:
            x = g[f"{h}G %"].dropna()
            xs = g[f"{h}G SPY'ye göre"].dropna()
            rec[f"{h}G medyan %"] = float(x.median()) if len(x) else np.nan
            rec[f"{h}G pozitif %"] = float((x > 0).mean() * 100) if len(x) else np.nan
            rec[f"{h}G SPY'ye göre ort."] = float(xs.mean()) if len(xs) else np.nan
        rec["120G en yüksek medyan %"] = float(g["120G en yüksek %"].median())
        rec["120G en derin medyan %"] = float(g["120G en derin %"].median())
        rows.append(rec)
    return pd.DataFrame(rows)


def stock_text(sym: str, E: pd.DataFrame, nw: dict[str, Any]) -> list[str]:
    """Hisse kartı için sade özet."""
    out = []
    if nw.get("ok"):
        out.append(f"{sym} şu an 200 günlük ortalamanın **%{abs(nw['uzaklik %']):.1f} "
                   f"{'üstünde' if nw['ustte'] else 'altında'}**; ortalama "
                   f"{'yükseliyor' if nw['egim_up'] else 'düşüyor'}.")
        if nw.get("olay"):
            out.append(f"Son 5 günde olay: **{nw['olay']}** "
                       f"({'bugün' if nw['olay_gun'] == 0 else str(nw['olay_gun']) + ' gün önce'}).")
    if E is not None and not E.empty:
        for kod, lab in EV.items():
            g = E[(E["_kod"] == kod)]
            g60 = g["60G %"].dropna()
            if len(g60) >= 2:
                out.append(f"{lab}: geçmişte {len(g)} kez; 60 gün sonra medyan %{g60.median():+.0f}, "
                           f"{(g60 > 0).mean() * 100:.0f}% artıda; 120 gün içinde medyan en yüksek "
                           f"%{g['120G en yüksek %'].median():+.0f}.")
        last = E[E["_kod"] == "kirilim"].head(1)
        if len(last):
            r = last.iloc[0]
            out.append(f"Son kırılım {r['Tarih']:%d.%m.%Y}: 120 gün içinde en yüksek "
                       f"%{r['120G en yüksek %']:+.0f}, en derin %{r['120G en derin %']:+.0f}. "
                       "Tek hissenin geçmişi tesadüf olabilir — evren testine bakın.")
    return out
