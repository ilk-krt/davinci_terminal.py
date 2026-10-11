# apex/seasonal.py — mevsimsellik: hangi aylarda tarihsel olarak güçlü/zayıf
from __future__ import annotations

import pandas as pd

AYLAR = ["Oca", "Şub", "Mar", "Nis", "May", "Haz", "Tem", "Ağu", "Eyl", "Eki", "Kas", "Ara"]


def monthly(df: pd.DataFrame, min_years: int = 4) -> tuple[pd.DataFrame, str]:
    """Döner: (ay tablosu, yorum). Tabloda ortalama/medyan getiri ve pozitif ay oranı."""
    if df is None or df.empty:
        return pd.DataFrame(), "veri yok"
    c = df["Close"].dropna()
    c.index = pd.to_datetime(c.index)
    try:
        c.index = c.index.tz_localize(None)
    except TypeError:
        pass
    m = c.resample("ME").last().pct_change().dropna() * 100
    if m.empty or len(m) < 12 * min_years:
        return pd.DataFrame(), f"mevsimsellik için en az {min_years} yıllık veri gerekir"
    g = m.groupby(m.index.month)
    t = pd.DataFrame({"Ay": [AYLAR[i - 1] for i in g.mean().index],
                      "Ortalama %": g.mean().values, "Medyan %": g.median().values,
                      "Pozitif %": (g.apply(lambda s: (s > 0).mean()) * 100).values,
                      "Yıl": g.size().values})
    now = pd.Timestamp.today().month
    nxt = now % 12 + 1
    r_now, r_nxt = t.iloc[now - 1], t.iloc[nxt - 1]
    best = t.sort_values("Ortalama %", ascending=False).head(3)["Ay"].tolist()
    worst = t.sort_values("Ortalama %").head(3)["Ay"].tolist()

    def word(r):
        return ("tarihsel olarak GÜÇLÜ" if r["Ortalama %"] > 1 and r["Pozitif %"] >= 60 else
                "tarihsel olarak ZAYIF" if r["Ortalama %"] < -0.5 and r["Pozitif %"] <= 45 else
                "belirgin bir eğilim yok")
    txt = (f"Bu ay ({r_now['Ay']}) {word(r_now)}: ortalama {r_now['Ortalama %']:+.1f}%, "
           f"yılların %{r_now['Pozitif %']:.0f}'inde artıda ({int(r_now['Yıl'])} yıl). "
           f"Gelecek ay ({r_nxt['Ay']}) {word(r_nxt)}: ortalama {r_nxt['Ortalama %']:+.1f}%, "
           f"%{r_nxt['Pozitif %']:.0f} pozitif. En iyi aylar: {', '.join(best)} · "
           f"en zayıf: {', '.join(worst)}.")
    if t["Yıl"].min() < 8:
        txt += " ⚠️ Yıl sayısı az — mevsimselliği güçlü bir sinyal olarak okumayın."
    return t.round(1), txt
