# apex/flows.py — VARLIKLAR ARASI PARA AKIŞI (temsilî görsel + rakamlar + gerekçeler)
#
# Her varlık sınıfı likit bir ETF ile temsil edilir. "Akış" = son 20 günün göreli
# performansı (bütün varlıkların ortalamasına göre) + dolar hacminin artışı.
# Para en çok geride kalanlardan en çok öne çıkanlara "akıyor" gibi çizilir —
# bu gerçek fon akışı verisi değil, fiyat × hacimden çıkarılan bir TEMSİLDİR.
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

ASSETS: dict[str, tuple[str, str]] = {      # ad: (sembol, kısa etiket)
    "ABD Hisse (S&P 500)": ("SPY", "HİSSE"),
    "Teknoloji / Büyüme": ("QQQ", "TEKNO"),
    "Küçük Şirketler": ("IWM", "KÜÇÜK"),
    "Defansif Hisse": ("XLP", "DEFANSİF"),
    "Gelişen Piyasalar": ("EEM", "GELİŞEN"),
    "Uzun Tahvil": ("TLT", "TAHVİL"),
    "Nakit (kısa vade)": ("BIL", "NAKİT"),
    "Dolar": ("UUP", "DOLAR $"),
    "Altın": ("GLD", "ALTIN Au"),
    "Gümüş": ("SLV", "GÜMÜŞ Ag"),
    "Petrol / Emtia": ("DBC", "EMTİA"),
    "Kripto": ("BTC-USD", "KRİPTO"),
}
TICKERS = [v[0] for v in ASSETS.values()]


def _ret(c: pd.Series, n: int) -> float:
    c = c.dropna()
    return float((c.iloc[-1] / c.iloc[-1 - n] - 1) * 100) if len(c) > n else np.nan


def table(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for ad, (t, kisa) in ASSETS.items():
        df = prices.get(t)
        if df is None or df.empty or len(df) < 70:
            continue
        c = df["Close"].dropna()
        dv = (df["Close"] * df["Volume"]).fillna(0)
        v10, v60 = dv.tail(10).mean(), dv.tail(60).mean()
        rows.append({"Varlık": ad, "Kısa": kisa, "Sembol": t,
                     "5G %": _ret(c, 5), "20G %": _ret(c, 20), "60G %": _ret(c, 60),
                     "Hacim trendi": (v10 / v60) if v60 > 0 else np.nan})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    m20, m5 = df["20G %"].mean(), df["5G %"].mean()
    s20 = df["20G %"].std(ddof=0) or 1.0
    s5 = df["5G %"].std(ddof=0) or 1.0
    df["Göreli 20G"] = df["20G %"] - m20
    # akış skoru: 20 gün (ağırlık 0.6) + 5 gün ivme (0.4), hacim artışıyla güçlenir
    z = 0.6 * (df["20G %"] - m20) / s20 + 0.4 * (df["5G %"] - m5) / s5
    vol_boost = df["Hacim trendi"].clip(0.6, 1.6).fillna(1.0)
    df["Akış Skoru"] = (z * vol_boost).round(2)
    df["Akış"] = np.select([df["Akış Skoru"] >= 0.6, df["Akış Skoru"] <= -0.6],
                           ["🟢 para giriyor", "🔴 para çıkıyor"], "⚪ nötr")
    return df.sort_values("Akış Skoru", ascending=False).reset_index(drop=True)


def reasons(df: pd.DataFrame) -> list[str]:
    """Akışın gerekçeleri — klasik varlıklar arası ilişkiler."""
    if df is None or df.empty:
        return []
    r = {row["Sembol"]: row for row in df.to_dict("records")}

    def g(t, k="20G %"):
        return r[t][k] if t in r and np.isfinite(r[t][k]) else np.nan

    out = []
    spy, tlt, gld, uup, btc = g("SPY"), g("TLT"), g("GLD"), g("UUP"), g("BTC-USD")
    qqq, iwm, xlp, eem, slv, dbc = g("QQQ"), g("IWM"), g("XLP"), g("EEM"), g("SLV"), g("DBC")
    if np.isfinite(spy) and np.isfinite(tlt):
        if spy > tlt + 2:
            out.append(f"📈 **Risk iştahı:** hisseler ({spy:+.1f}%) tahvili ({tlt:+.1f}%) "
                       "geçiyor — para güvenli limandan riskli varlıklara geçiyor.")
        elif tlt > spy + 2:
            out.append(f"🛡️ **Riskten kaçış:** tahvil ({tlt:+.1f}%) hisseyi ({spy:+.1f}%) "
                       "geçiyor — para güvenli limana sığınıyor.")
    if np.isfinite(gld) and np.isfinite(uup):
        if gld > 0 and uup < 0:
            out.append(f"🥇 **Dolar zayıf, altın güçlü:** dolar {uup:+.1f}%, altın {gld:+.1f}% "
                       "— klasik ters ilişki çalışıyor (dolar düşünce altın pahalılaşmaz).")
        elif gld < 0 and uup > 0:
            out.append(f"💵 **Dolar güçlü, altın baskı altında:** dolar {uup:+.1f}%, "
                       f"altın {gld:+.1f}%.")
        elif gld > 0 and uup > 0:
            out.append(f"⚠️ **Altın ve dolar birlikte yükseliyor** (altın {gld:+.1f}%, dolar "
                       f"{uup:+.1f}%) — genelde kriz/güvenli liman talebi işaretidir.")
    if np.isfinite(slv) and np.isfinite(gld) and slv - gld > 3:
        out.append(f"🥈 **Gümüş altını geçiyor** ({slv:+.1f}% / {gld:+.1f}%) — sanayi "
                   "talebi ve risk iştahıyla uyumlu; değerli metallerde iştah yüksek.")
    if np.isfinite(iwm) and np.isfinite(spy):
        if iwm > spy + 1.5:
            out.append(f"🌱 **Yükseliş genişliyor:** küçük şirketler ({iwm:+.1f}%) büyükleri "
                       f"({spy:+.1f}%) geçiyor — sağlıklı katılım.")
        elif spy > iwm + 1.5:
            out.append(f"🏰 **Para büyük şirketlere sığınıyor:** küçükler ({iwm:+.1f}%) "
                       f"S&P 500'ün ({spy:+.1f}%) gerisinde — dar katılım.")
    if np.isfinite(xlp) and np.isfinite(spy) and xlp > spy + 1.5:
        out.append(f"🧺 **Defansife kaçış:** temel tüketim ({xlp:+.1f}%) endeksi ({spy:+.1f}%) "
                   "geçiyor — kurumsal para temkinli.")
    if np.isfinite(qqq) and np.isfinite(btc) and qqq > spy and btc > 5:
        out.append(f"🚀 **Spekülatif iştah:** teknoloji ({qqq:+.1f}%) ve kripto ({btc:+.1f}%) "
                   "birlikte güçlü.")
    if np.isfinite(eem) and np.isfinite(spy) and np.isfinite(uup) and eem > spy + 1.5 and uup < 0:
        out.append(f"🌍 **Gelişen piyasalara akış:** zayıf dolar ({uup:+.1f}%) gelişen "
                   f"piyasaları ({eem:+.1f}%) destekliyor.")
    if np.isfinite(dbc) and dbc > 4:
        out.append(f"🛢️ **Emtia yükselişi** ({dbc:+.1f}%) — enflasyon beklentisi ve faiz "
                   "baskısı riski; altın için çift taraflı.")
    top = df.head(3)["Varlık"].tolist()
    bot = df.tail(3)["Varlık"].tolist()[::-1]
    out.insert(0, f"🔀 **Özet:** para en çok **{', '.join(bot)}** tarafından çıkıp "
                  f"**{', '.join(top)}** tarafına yöneliyor (son 20 gün, göreli).")
    return out


def diagram(df: pd.DataFrame, title: str = "Varlıklar arası para akışı (son 20 gün)"):
    """Varlıklar daire üzerinde; kalın oklar çıkıştan girişe."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch

    BG, TXT, MUT = "#0b0e11", "#d1d4dc", "#7a8394"
    fig, ax = plt.subplots(figsize=(9.5, 7.2), dpi=110, facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.30, 1.38)
    ax.set_aspect("equal")
    ax.axis("off")
    if df is None or df.empty:
        ax.text(0, 0, "akış verisi yok", color=TXT, ha="center")
        return fig
    n = len(df)
    order = df.sort_values("Varlık").reset_index(drop=True)
    pos = {}
    for i, row in order.iterrows():
        a = math.pi / 2 - 2 * math.pi * i / n
        pos[row["Sembol"]] = (math.cos(a), math.sin(a))
    smax = max(df["Akış Skoru"].abs().max(), 0.5)

    # oklar: en zayıf 3 → en güçlü 3
    src = df.tail(3).to_dict("records")
    dst = df.head(3).to_dict("records")
    for s in src:
        for t in dst:
            w = (t["Akış Skoru"] - s["Akış Skoru"])
            if w <= 0.4:
                continue
            (x0, y0), (x1, y1) = pos[s["Sembol"]], pos[t["Sembol"]]
            ar = FancyArrowPatch((x0 * 0.86, y0 * 0.86), (x1 * 0.86, y1 * 0.86),
                                 connectionstyle="arc3,rad=0.18", arrowstyle="-|>",
                                 mutation_scale=10 + 6 * w / smax,
                                 lw=0.8 + 4.5 * w / (2 * smax), color="#ffd54f",
                                 alpha=0.30 + 0.45 * min(1, w / (2 * smax)), zorder=2)
            ax.add_patch(ar)
    for row in df.to_dict("records"):
        x, y = pos[row["Sembol"]]
        s = row["Akış Skoru"]
        col = "#00c853" if s >= 0.6 else "#ff1744" if s <= -0.6 else "#607d8b"
        r = 0.11 + 0.07 * min(1, abs(s) / smax)
        ax.add_patch(Circle((x, y), r, color=col, alpha=0.9, zorder=3))
        ax.add_patch(Circle((x, y), r, fill=False, ec="#ffffff", lw=0.6, alpha=0.4, zorder=4))
        ax.text(x, y + 0.012, row["Kısa"], color="#ffffff", fontsize=7.6,
                fontweight="bold", ha="center", va="center", zorder=5)
        ax.text(x, y - r - 0.055, f"{row['20G %']:+.1f}%  (5G {row['5G %']:+.1f}%)",
                color=TXT, fontsize=7, ha="center", va="center", zorder=5)
    ax.text(0, 0.06, "PARA", color=MUT, fontsize=16, ha="center", va="center",
            fontweight="bold", alpha=0.35)
    ax.text(0, -0.08, "çıkış → giriş", color=MUT, fontsize=8, ha="center", alpha=0.6)
    ax.text(-1.52, 1.33, title, color=TXT, fontsize=11.5, fontweight="bold", va="top")
    ax.text(-1.52, 1.24, "yeşil: para giriyor · kırmızı: çıkıyor · daire büyüklüğü: akış gücü · "
            "ok kalınlığı: göreli fark", color=MUT, fontsize=7, va="top")
    ax.text(-1.52, -1.27, "Temsilîdir: fiyat performansı × dolar hacmi trendinden çıkarılır; "
            "gerçek fon akış verisi değildir.", color=MUT, fontsize=6.5)
    return fig


def summary_numbers(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or df.empty:
        return {}
    return {"giren": df[df["Akış"].str.startswith("🟢")]["Varlık"].tolist(),
            "cikan": df[df["Akış"].str.startswith("🔴")]["Varlık"].tolist()}
