# apex/tvchart.py — TradingView görünümlü statik grafik (matplotlib)
#
# Statik resim: tıklayınca/sürükleyince bozulmaz; PNG / JPEG / PDF indirilir.
# Çizilenler (tfview.build çıktısından):
#   mumlar Mum Gücü (v666) sınıfına göre renkli · EMA21/50/200
#   S/R matrisi (kısa ····  orta ‒ ‒  uzun ───) · Gap Matrix kutuları (FVG/GAP/KURUMSAL)
#   Wyckoff bölgeleri · likidite havuzları · ana yapı Fibonacci seviyeleri
#   Elliott sayımı + hedef kutusu + stop/geçersizlik · GFR hedef boşluğu
#   işaretler: Q-dry ★, ODI ▲▼, efor kırılımı ↑↓, VSA dönüşü ◆, Wyckoff olayları
#   alt panel: hacim (alış/satış baskısına göre) + 20 bar ortalaması
from __future__ import annotations

import io
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                    # noqa: E402
from matplotlib.patches import Rectangle              # noqa: E402

BG, PANEL, GRID, TXT, MUTED = "#0b0e11", "#0f1318", "#1c222b", "#d1d4dc", "#7a8394"
UP, DN = "#26a69a", "#ef5350"
CLS_COL = {1: "#00e676", 2: "#26a69a", 3: "#9ccc65", -1: "#ff1744", -2: "#ef5350",
           -3: "#ba68c8", 10: "#42a5f5", 11: "#ffa726", 0: "#787b86"}
CLS_TXT = {1: "güçlü yükseliş", 2: "orta yükseliş", 3: "hacimsiz yükseliş",
           -1: "sert düşüş", -2: "orta düşüş", -3: "hacimsiz düşüş",
           10: "hacimli duraksama", 11: "dağıtım izi", 0: "nötr"}
EMA_COL = {21: "#ffd54f", 50: "#29b6f6", 200: "#e040fb"}
GAP_COL = "#f7e08a"
RES_COL, SUP_COL = "#ff5252", "#00e676"
FIB_COL = "#8d99ae"
ELL_COL = "#b39dff"
HZ_STYLE = {0: (":", 0.9), 1: ("--", 1.2), 2: ("-", 1.6)}
TIER = {1: "FVG", 2: "GAP", 3: "KURUMSAL"}


def _fmt(x: float) -> str:
    return (f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.2f}" if abs(x) >= 1
            else f"{x:.4g}")


def _arr(x, n):
    if x is None:
        return np.zeros(n, bool)
    a = np.asarray(x)
    return a.astype(bool) if len(a) == n else np.zeros(n, bool)


def draw(v: dict[str, Any], label: str = "", width: float = 7.2, height: float = 5.6):
    """tfview sözlüğünden figür üretir."""
    fig = plt.figure(figsize=(width, height), dpi=110, facecolor=BG)
    if not v.get("ok"):
        fig.text(0.5, 0.5, f"{label} · {v.get('tf', '')}\n{v.get('neden', 'veri yok')}",
                 color=TXT, ha="center", va="center")
        return fig
    d = v["d"]
    n_all = len(d)
    nv = min(n_all, {"Günlük": 140, "Haftalık": 130, "Aylık": 120}.get(v["tf"], 130))
    s0 = n_all - nv
    dv = d.iloc[s0:]
    o, h, l, c = (dv[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    x = np.arange(nv)
    pad_r = max(16, nv // 6)
    gs = fig.add_gridspec(2, 1, height_ratios=[4.2, 1], hspace=0.04,
                          left=0.015, right=0.88, top=0.90, bottom=0.10)
    ax = fig.add_subplot(gs[0])
    axv = fig.add_subplot(gs[1], sharex=ax)
    for a in (ax, axv):
        a.set_facecolor(PANEL)
        a.grid(color=GRID, lw=0.6)
        a.tick_params(colors=MUTED, labelsize=7)
        for sp in a.spines.values():
            sp.set_color(GRID)
        a.yaxis.tick_right()
    plt.setp(ax.get_xticklabels(), visible=False)

    # ---- y aralığı: görünen fiyat + yakın seviyeler
    lo, hi = float(np.nanmin(l)), float(np.nanmax(h))
    px = float(c[-1])
    rng = hi - lo
    y0, y1 = lo - rng * 0.06, hi + rng * 0.10

    def inview(y):
        return y0 - rng * 0.25 <= y <= y1 + rng * 0.25

    # ---- mumlar (Mum Gücü renkleri)
    mg = v.get("mg")
    cls = mg["cls"].to_numpy()[s0:] if mg is not None and len(mg) == n_all else None
    cols = []
    for i in range(nv):
        if cls is not None and int(cls[i]) in CLS_COL and int(cls[i]) != 0:
            cols.append(CLS_COL[int(cls[i])])
        else:
            cols.append(UP if c[i] >= o[i] else DN)
    ax.vlines(x, l, h, colors=cols, lw=0.8, zorder=3)
    body_lo, body_hi = np.minimum(o, c), np.maximum(o, c)
    body_h = np.maximum(body_hi - body_lo, rng * 0.002)
    ax.bar(x, body_h, bottom=body_lo, width=0.7, color=cols, edgecolor=cols,
           lw=0.4, zorder=4)

    # ---- EMA'lar
    for n_, s in v.get("ema", {}).items():
        yy = s.to_numpy(float)[s0:]
        if np.isfinite(yy).sum() > 5:
            ax.plot(x, yy, color=EMA_COL[n_], lw=1.0 if n_ != 200 else 1.4,
                    alpha=0.9, zorder=5, label=f"EMA{n_}")

    xr = nv - 1 + pad_r
    labels: list[tuple[float, str, str]] = []

    # ---- Fibonacci (ana yapı)
    fb = v.get("fib") or {}
    if fb.get("ok"):
        for lv in (0.236, 0.382, 0.5, 0.618, 0.786, 1.272, 1.618):
            y = fb["levels"][lv]
            if inview(y):
                ax.hlines(y, 0, nv - 1 + pad_r * 0.55, colors=FIB_COL, lw=0.6,
                          linestyles=(0, (2, 3)), alpha=0.55, zorder=2)
                ax.text(nv - 1 + pad_r * 0.56, y, f"F{lv}", color=FIB_COL,
                        fontsize=5.8, va="center", alpha=0.85, zorder=2, clip_on=True)

    # ---- Gap Matrix kutuları
    gm = v.get("gaps") or {}
    boxes = [b for b in gm.get("boxes", []) if inview(b["top"]) or inview(b["bot"])]
    boxes = sorted(boxes, key=lambda b: abs((b["top"] + b["bot"]) / 2 - px))[:6]
    for b in boxes:
        left = max(b["left"] - s0, 0)
        ax.add_patch(Rectangle((left, b["bot"]), xr - left, b["top"] - b["bot"],
                               facecolor=GAP_COL, alpha=0.10 + 0.05 * b["tier"],
                               edgecolor=GAP_COL, lw=0.6 + 0.3 * b["tier"],
                               ls="--" if b["dir"] > 0 else ":", zorder=1))
        ax.text(left + 0.5, b["top"], f"{'▲' if b['dir'] > 0 else '▼'}{TIER[b['tier']]}",
                color=GAP_COL, fontsize=5.8, va="bottom", zorder=6, clip_on=True)

    # ---- Wyckoff (VSA PRO) bölgeleri
    for z in (v.get("vsa_zones") or {}).get("zones", []):
        if not (inview(z["top"]) or inview(z["bot"])) or abs(z["bot"] / px - 1) > 0.25:
            continue
        left = max(z["left"] - s0, 0)
        col = SUP_COL if z["side"] > 0 else RES_COL
        ax.add_patch(Rectangle((left, z["bot"]), xr - left, z["top"] - z["bot"],
                               facecolor=col, alpha=0.07, edgecolor=col, lw=0.7, zorder=1))
        ax.text(left + 0.5, z["bot"], z["name"].replace("EFFORT≠RESULT", "EFFORT"),
                color=col, fontsize=5.6, va="top", zorder=6, alpha=0.9, clip_on=True)

    # ---- S/R matrisi
    sr = v.get("sr") or {}
    for side, lst, col in (("D", sr.get("res", []), RES_COL), ("S", sr.get("sup", []), SUP_COL)):
        for k, cl in enumerate(lst):
            y = cl["px"]
            if not inview(y):
                continue
            ls, lw = HZ_STYLE[cl["hz"]]
            left = max(cl["b0"] - s0, 0)
            ax.hlines(y, left, xr, colors=col, lw=lw, linestyles=ls, alpha=0.85, zorder=6)
            labels.append((y, f"{side}{k + 1} {_fmt(y)}", col))

    # ---- likidite havuzları
    lq = v.get("liq") or {}
    for key, col in (("up", "#4fc3f7"), ("dn", "#4fc3f7")):
        p = lq.get(key)
        if p and inview(p["px"]) and abs(p["px"] / px - 1) < 0.15:
            ax.hlines(p["px"], max(p["b0"] - s0, 0), xr, colors=col, lw=0.8,
                      linestyles=(0, (1, 2)), zorder=6)
            labels.append((p["px"], f"LQ×{p['cnt']} {_fmt(p['px'])}", col))

    # ---- GFR hedef boşluğu
    g = v.get("gfr")
    if g and g.get("top") and inview(g["top"]):
        ax.add_patch(Rectangle((nv - 30, g["bot"]), xr - (nv - 30), g["top"] - g["bot"],
                               facecolor="#b39dff", alpha=0.12, edgecolor="#b39dff",
                               lw=0.8, zorder=1))
        ax.text(nv - 29, g["top"], "GFR hedef", color="#b39dff", fontsize=5.8,
                va="bottom", zorder=6, clip_on=True)

    # ---- Elliott
    e = v.get("ell")
    if e and e.get("B"):
        B, P = e["B"], e["P"]
        names = None
        k = e.get("kind")
        n_p = len(P)
        if k == "w3" and n_p >= 3:
            idx, names = list(range(n_p - 3, n_p)), ["0", "1", "2"]
        elif k == "w5" and n_p >= 5:
            idx, names = list(range(n_p - 5, n_p)), ["0", "1", "2", "3", "4"]
        elif k and k.startswith("abc") and e.get("last_end") is not None:
            le = e["last_end"]
            idx = [i for i in range(le - 5, n_p) if 0 <= i < n_p]
            base = ["0", "1", "2", "3", "4", "5", "A", "B", "C"]
            names = base[:len(idx)]
        if names:
            xs = [B[i] - s0 for i in idx]
            ys = [P[i] for i in idx]
            ok = [i for i, xx in enumerate(xs) if xx >= 0]
            if len(ok) >= 2:
                ax.plot([xs[i] for i in ok], [ys[i] for i in ok], color=ELL_COL,
                        lw=1.1, alpha=0.9, zorder=7)
                for i in ok:
                    top = i + 1 < len(ys) and ys[i] > ys[i + 1] or (i > 0 and ys[i] > ys[i - 1])
                    ax.annotate(names[i], (xs[i], ys[i]),
                                xytext=(0, 7 if top else -9), textcoords="offset points",
                                ha="center", fontsize=7, color="#0b0e11", zorder=8,
                                bbox=dict(boxstyle="circle,pad=0.18", fc=ELL_COL, ec="none"))
        z = e.get("zone")
        if z and not z.get("dead") and min(z["zlo"], z["zhi"]) > 0 \
                and abs(z["near"] / px - 1) < 0.6:
            zx = nv + 1
            ax.add_patch(Rectangle((zx, z["zlo"]), xr - zx, z["zhi"] - z["zlo"],
                                   facecolor=ELL_COL, alpha=0.16, edgecolor=ELL_COL,
                                   lw=0.9, zorder=2))
            ax.text(zx + 0.5, z["zhi"], f"Elliott {z['name']}", color=ELL_COL,
                    fontsize=6, va="bottom", zorder=8, clip_on=True)
            for yv, col, nm in ((z.get("stop"), "#ffa726", "stop"), (z.get("inv"), "#ff5252", "✖")):
                if yv is not None and inview(yv):
                    ax.hlines(yv, nv - 12, xr, colors=col, lw=0.8, linestyles="--", zorder=6)
            y0 = min(y0, z["zlo"] - rng * 0.03)
            y1 = max(y1, z["zhi"] + rng * 0.03)

    # ---- işaretler
    n_all_ = len(d)
    marks = []
    vv = v.get("v710")
    if vv:
        for key, sym, col, below, sz in (("q_conf", "*", "#ffd54f", True, 60),
                                         ("odi_bull", "h", "#00e5ff", True, 26),
                                         ("odi_bear", "h", "#ff4081", False, 26),
                                         ("eff_up", "^", "#00e676", True, 16),
                                         ("eff_dn", "v", "#ff1744", False, 16)):
            a = _arr(vv.get(key), n_all_)[s0:]
            marks.append((a, sym, col, below, sz))
    vs = v.get("vsa")
    recent = np.zeros(nv, bool)
    recent[-40:] = True                      # VSA dönüşleri sık — yalnızca son 40 mum
    if vs:
        for key, sym, col, below in (("rev_up", "D", "#00e676", True),
                                     ("rev_dn", "D", "#ff1744", False),
                                     ("flip_up", "P", "#69f0ae", True),
                                     ("flip_dn", "X", "#ff8a80", False)):
            marks.append((_arr(vs.get(key), n_all_)[s0:] & recent, sym, col, below, 18))
    off = rng * 0.025
    for a, sym, col, below, sz in marks:
        ii = np.flatnonzero(a)
        if len(ii):
            ax.scatter(ii, (l[ii] - off) if below else (h[ii] + off), marker=sym, s=sz,
                       color=col, zorder=9, linewidths=0.6, edgecolors=BG)
    for (bi, nm, bias, stg) in (v.get("vsa_zones") or {}).get("events", []):
        xi = bi - s0
        if xi < 0 or nm in ("NO DEMAND", "NO SUPPLY") or stg < 55:
            continue
        short = {"SPRING": "SPR", "UPTHRUST": "UT", "SELL CLIMAX": "SC",
                 "BUY CLIMAX": "BC", "EFFORT≠RESULT": "E≠R"}.get(nm, nm[:3])
        ax.text(xi, (l[xi] - off * 2.2) if bias > 0 else (h[xi] + off * 2.2), short,
                color=SUP_COL if bias > 0 else RES_COL, fontsize=5.4, ha="center",
                va="top" if bias > 0 else "bottom", zorder=9)

    # ---- sağ kenar etiketleri (çakışmasın diye aralıklı)
    labels.append((px, f"{_fmt(px)}", "#ffffff"))
    labels.sort(key=lambda t: t[0])
    min_gap = (y1 - y0) * 0.035
    placed: list[float] = []
    for y, t, col in labels:
        yy = y
        if placed and yy - placed[-1] < min_gap:
            yy = placed[-1] + min_gap
        placed.append(yy)
        ax.text(1.005, (yy - y0) / (y1 - y0), t, transform=ax.transAxes, fontsize=6.2,
                color=BG if col == "#ffffff" else col, va="center", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc=col if col == "#ffffff" else PANEL,
                          ec=col, lw=0.6), clip_on=False)
    ax.axhline(px, color="#ffffff", lw=0.5, ls=":", alpha=0.5, zorder=2)
    ax.set_ylim(y0, y1)
    ax.set_xlim(-1, xr)
    ax.set_yticklabels([])

    # ---- hacim
    vol = dv["Volume"].to_numpy(float)
    if np.nansum(vol) > 0:
        bp = np.asarray(vs["bp"])[s0:] if vs is not None and len(vs["bp"]) == n_all_ \
            else np.where(c >= o, 0.6, 0.4)
        vcol = np.where(bp >= 0.5, UP, DN)
        axv.bar(x, vol, width=0.7, color=vcol, alpha=0.75)
        axv.bar(x, vol * np.minimum(bp, 1 - bp) * 2 * 0.5, width=0.7,
                color=np.where(bp >= 0.5, DN, UP), alpha=0.55)
        ma = np.convolve(np.nan_to_num(d["Volume"].to_numpy(float)), np.ones(20) / 20,
                         mode="full")[:n_all_][s0:]
        axv.plot(x, ma, color="#ffd54f", lw=0.8)
        axv.set_ylim(0, np.nanmax(vol) * 1.15)
    else:
        axv.text(0.5, 0.5, "hacim verisi yok (oran / endeks)", color=MUTED, fontsize=7,
                 ha="center", va="center", transform=axv.transAxes)
    axv.set_yticklabels([])

    # ---- tarih ekseni
    idx = dv.index
    step = max(1, nv // 6)
    ticks = list(range(0, nv, step))
    fmt = "%d %b" if v["tf"] == "Günlük" else "%b %y" if v["tf"] == "Haftalık" else "%Y"
    axv.set_xticks(ticks)
    axv.set_xticklabels([idx[i].strftime(fmt) for i in ticks], fontsize=6.5, color=MUTED)

    # ---- başlık
    chg = (c[-1] / c[-2] - 1) * 100 if nv > 1 else 0
    ccls = int(cls[-1]) if cls is not None else 0
    sc = v.get("score", np.nan)
    fig.text(0.015, 0.955, f"{label}  ·  {v['tf']}", color=TXT, fontsize=10.5,
             fontweight="bold", va="center")
    fig.text(0.015, 0.918, f"{_fmt(px)}  ({chg:+.2f}%)   ·   mum: {CLS_TXT.get(ccls, 'nötr')}",
             color=UP if chg >= 0 else DN, fontsize=7.5, va="center")
    if np.isfinite(sc):
        col = UP if sc >= 15 else DN if sc <= -15 else MUTED
        fig.text(0.88, 0.955, f"Skor {sc:+.0f}", color=col, fontsize=10, ha="right",
                 fontweight="bold", va="center")
    leg = ("EMA21 sarı · EMA50 mavi · EMA200 mor  |  S/R: ··· kısa  ‒ ‒ orta  ─ uzun  |  "
           "sarı kutu: boşluk  |  mor: Elliott  |  ★ Q-dry  ⬢ ODI  ▲▼ efor kırılımı  "
           "◆ VSA dönüş  ✚/✖ baskınlık  LQ likidite")
    fig.text(0.015, 0.035, leg, color=MUTED, fontsize=5.4, va="center")
    return fig


def to_bytes(fig, fmt: str = "png") -> bytes:
    buf = io.BytesIO()
    kw = {"facecolor": fig.get_facecolor()}
    if fmt in ("png", "jpeg", "jpg"):
        kw["dpi"] = 160
    fig.savefig(buf, format="jpeg" if fmt == "jpg" else fmt, **kw)
    return buf.getvalue()


def multi_pdf(figs: list) -> bytes:
    from matplotlib.backends.backend_pdf import PdfPages
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        for f in figs:
            pdf.savefig(f, facecolor=f.get_facecolor())
    return buf.getvalue()


def close(fig) -> None:
    plt.close(fig)


def _plain(t: str) -> str:
    """PDF metni: markdown işaretleri ve fontta olmayan emojiler temizlenir."""
    import re
    t = t.replace("**", "").replace("_", "")
    rep = {"🟢": "[+]", "🔴": "[-]", "⚪": "[=]", "⚠️": "(!)", "🔒": "", "🎯": "",
           "✅": "[+]", "❌": "[-]", "💧": "", "🌱": "", "🚀": "",
           "Ⓐ": "(A)", "Ⓑ": "(B)", "Ⓒ": "(C)"}
    for k, v in rep.items():
        t = t.replace(k, v)
    return re.sub(r"[\U00010000-\U0010ffff]", "", t)


def report_pdf(title: str, pngs: list[bytes], texts: list[str]) -> bytes:
    """Her zaman dilimi için bir A4 sayfa: üstte grafik, altta teknik okuma."""
    import textwrap
    from matplotlib.backends.backend_pdf import PdfPages
    from PIL import Image
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        for png, txt in zip(pngs, texts):
            fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
            img = Image.open(io.BytesIO(png))
            ax = fig.add_axes([0.04, 0.50, 0.92, 0.46])
            ax.imshow(img)
            ax.axis("off")
            fig.text(0.05, 0.975, title, fontsize=12, fontweight="bold", va="top")
            lines = []
            for para in _plain(txt).split("\n"):
                lines += textwrap.wrap(para, 118) or [""]
            fig.text(0.05, 0.48, "\n".join(lines[:58]), fontsize=7.2, va="top",
                     family="DejaVu Sans", linespacing=1.35)
            pdf.savefig(fig)
            plt.close(fig)
    return buf.getvalue()
