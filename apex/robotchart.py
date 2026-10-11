# apex/robotchart.py — robot simülasyonunun grafiği: seviyeler grafiğe "çivi gibi çakılı"
#
# 1 saatlik mumlar + Fibonacci giriş bölgesi (gri bant) + MALİYET (beyaz) + STOP
# (kırmızı, yalnızca yukarı basamaklanır) + HEDEF (yeşil, dinamik) + %85 RADAR
# (turuncu kesikli) + alım/satım işaretleri; sağ kenarda canlı K/Z ve hedefe uzaklık;
# altta simülasyon sermaye eğrisi.
from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

from apex.tvchart import BG, DN, GRID, MUTED, PANEL, TXT, UP  # noqa: E402


def draw(R: dict[str, Any], label: str, bars: int = 220, width: float = 12, height: float = 6.6):
    h1, L = R["h1"], R["lines"]
    fig = plt.figure(figsize=(width, height), dpi=110, facecolor=BG)
    ev = R["events"]
    # görünüm: son işlemin girişinden biraz önce ya da son 'bars' mum
    end = len(h1)
    s0 = max(0, end - bars)
    if len(ev):
        last_in = ev[ev["Olay"].isin(["✅ Giriş", "🛡️ Yönetim başladı"])]
        if len(last_in):
            pos = int(np.searchsorted(h1.index.values, np.datetime64(last_in["Zaman"].iloc[-1])))
            s0 = max(0, min(s0, pos - 40))
    d = h1.iloc[s0:end]
    Lv = L.iloc[s0:end]
    n = len(d)
    x = np.arange(n)
    o, hi, lo, c = (d[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
    CN = R.get("council")
    if CN is not None:
        gs = fig.add_gridspec(3, 1, height_ratios=[4, 1.0, 1.0], hspace=0.06, left=0.015,
                              right=0.86, top=0.90, bottom=0.08)
        ax, axs, axe = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
        axes = (ax, axs, axe)
    else:
        gs = fig.add_gridspec(2, 1, height_ratios=[4, 1.1], hspace=0.05, left=0.015, right=0.86,
                              top=0.90, bottom=0.08)
        ax, axe = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
        axs, axes = None, (ax, axe)
    for a in axes:
        a.set_facecolor(PANEL)
        a.grid(color=GRID, lw=0.6)
        a.tick_params(colors=MUTED, labelsize=7)
        for sp in a.spines.values():
            sp.set_color(GRID)
        a.yaxis.tick_right()
    col = np.where(c >= o, UP, DN)
    ax.vlines(x, lo, hi, colors=col, lw=0.7)
    ax.bar(x, np.maximum(np.abs(c - o), (hi.max() - lo.min()) * 0.001),
           bottom=np.minimum(o, c), width=0.7, color=col)
    # fib bölgesi
    ax.fill_between(x, Lv["zb"].to_numpy(), Lv["zt"].to_numpy(), color="#8d99ae", alpha=0.12,
                    step="mid", label="Fib giriş bölgesi")
    for key, colr, ls, lw, nm in (("cost", "#ffffff", "-", 1.1, "Maliyet"),
                                  ("sl", "#ff1744", "-", 1.5, "Stop"),
                                  ("tp", "#00e676", "-", 1.5, "Hedef"),
                                  ("radar", "#ffa726", "--", 1.0, "Radar %85")):
        y = Lv[key].to_numpy()
        if np.isfinite(y).any():
            ax.step(x, y, where="mid", color=colr, ls=ls, lw=lw, label=nm, zorder=5)
    # alım/satım işaretleri
    fl = R["fills"]
    if len(fl):
        for _, r in fl.iterrows():
            pos = int(np.searchsorted(d.index.values, np.datetime64(r["Zaman"])))
            if 0 <= pos < n and d.index[pos] == r["Zaman"]:
                buy = r["İşlem"] == "AL"
                ax.scatter(pos, r["Fiyat"], marker="^" if buy else "v", s=70,
                           color="#00e5ff" if buy else "#ffd54f", edgecolors=BG, zorder=8)
                ax.annotate(("AL " if buy else f"{r['Neden']} ") + f"{r['Fiyat']:.2f}",
                            (pos, r["Fiyat"]), xytext=(0, -14 if buy else 9),
                            textcoords="offset points", ha="center", fontsize=6.3,
                            color="#00e5ff" if buy else "#ffd54f", zorder=9)
    # radar / giyotin olayları
    if len(ev):
        for _, r in ev[ev["Olay"].isin(["📡 Radar", "⚔️ Giyotin"])].iterrows():
            pos = int(np.searchsorted(d.index.values, np.datetime64(r["Zaman"])))
            if 0 <= pos < n:
                ax.axvline(pos, color="#ffa726" if "Radar" in r["Olay"] else "#ff1744",
                           lw=0.8, ls=":", alpha=0.8)
                ax.text(pos, hi.max(),
                        "RADAR" if "Radar" in r["Olay"] else "GİYOTİN",
                        color="#ffa726" if "Radar" in r["Olay"] else "#ff1744",
                        fontsize=6.3, ha="center", va="bottom", rotation=90)
    # sağ kenar: canlı değerler
    stt = R["state"]
    px = float(c[-1])
    labels = [(px, f"Fiyat {px:.2f}", "#ffffff")]
    if stt["pozisyonda"]:
        cost, sl, tp = stt["maliyet"], stt["stop"], stt["hedef"]
        labels.append((cost, f"Maliyet {cost:.2f}  K/Z {(px / cost - 1) * 100:+.1f}%", "#d1d4dc"))
        labels.append((sl, f"Stop {sl:.2f}  ({(sl / px - 1) * 100:+.1f}%)", "#ff1744"))
        if np.isfinite(tp):
            labels.append((tp, f"Hedef {tp:.2f}  ({(tp / px - 1) * 100:+.1f}%)", "#00e676"))
            rd = cost + 0.85 * (tp - cost)
            labels.append((rd, f"Radar {rd:.2f}", "#ffa726"))
    lv_vals = np.concatenate([Lv[k].to_numpy() for k in ("cost", "sl", "tp", "radar")])
    lv_vals = lv_vals[np.isfinite(lv_vals)]
    y0 = float(min(lo.min(), lv_vals.min() if len(lv_vals) else lo.min()))
    y1 = float(max(hi.max(), lv_vals.max() if len(lv_vals) else hi.max()))
    pad = (y1 - y0) * 0.05
    y0, y1 = y0 - pad, y1 + pad
    ys = [v for v, _, _ in labels if np.isfinite(v)]
    y0, y1 = min([y0] + ys) * 0.995, max([y1] + ys) * 1.005
    ax.set_ylim(y0, y1)
    labels.sort(key=lambda t: t[0])
    gap = (y1 - y0) * 0.045
    last = None
    for yv, t, colr in labels:
        if not np.isfinite(yv):
            continue
        yy = yv if last is None or yv - last >= gap else last + gap
        last = yy
        ax.text(1.005, (yy - y0) / (y1 - y0), t, transform=ax.transAxes, fontsize=7,
                color=BG if colr == "#ffffff" else colr, va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc=colr if colr == "#ffffff" else PANEL,
                          ec=colr, lw=0.6), clip_on=False)
    ax.set_xlim(-1, n + 3)
    ax.set_yticklabels([])
    plt.setp(ax.get_xticklabels(), visible=False)
    leg = ax.legend(loc="upper left", fontsize=6.5, facecolor=PANEL, edgecolor=GRID)
    for t in leg.get_texts():
        t.set_color(TXT)
    # konsey skoru
    if axs is not None:
        scv = CN["score"].iloc[s0:end].to_numpy()
        axs.plot(x, scv, color="#b39dff", lw=1.0, label="konsey skoru")
        axs.plot(x, CN["thr_in"].iloc[s0:end].to_numpy(), color="#00e676", lw=0.8, ls="--",
                 label="giriş eşiği")
        axs.plot(x, CN["thr_out"].iloc[s0:end].to_numpy(), color="#ff5252", lw=0.8, ls="--",
                 label="çıkış eşiği")
        axs.axhline(0, color=GRID, lw=0.8)
        axs.set_xlim(-1, n + 3)
        axs.set_yticklabels([])
        plt.setp(axs.get_xticklabels(), visible=False)
        lg2 = axs.legend(loc="upper left", fontsize=6, facecolor=PANEL, edgecolor=GRID, ncol=3)
        for t in lg2.get_texts():
            t.set_color(TXT)
    # sermaye eğrisi
    eq = R["equity"].reindex(d.index).ffill()
    axe.plot(x, eq.to_numpy(), color="#29b6f6", lw=1.2)
    axe.fill_between(x, eq.to_numpy(), np.nanmin(eq.to_numpy()) if eq.notna().any() else 0,
                     color="#29b6f6", alpha=0.08)
    axe.set_ylabel("sermaye", color=MUTED, fontsize=7)
    step = max(1, n // 7)
    axe.set_xticks(list(range(0, n, step)))
    axe.set_xticklabels([d.index[i].strftime("%d %b %H:%M") for i in range(0, n, step)],
                        fontsize=6.5, color=MUTED)
    axe.set_xlim(-1, n + 3)
    s = R["stats"]
    durum = ("POZİSYONDA" if stt["pozisyonda"] else "NAKİTTE — giriş bekleniyor")
    fig.text(0.015, 0.955, f"ROBOT · {label} · simülasyon · 1 saatlik", color=TXT, fontsize=11,
             fontweight="bold", va="center")
    fig.text(0.015, 0.92, f"{durum} · {s['İşlem']} işlem · getiri {s['Getiri %']:+.1f}% · "
             f"al-tut {s['Al-tut %']:+.1f}% · maks. düşüş {s['Maks. düşüş %']:.1f}%",
             color=UP if s["Getiri %"] >= 0 else DN, fontsize=8, va="center")
    return fig
