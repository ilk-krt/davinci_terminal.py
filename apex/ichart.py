# apex/ichart.py — TradingView benzeri ETKİLEŞİMLİ grafik (Plotly) + "Durum ne?" sade anlatım
#
# tvchart.py'nin (statik, matplotlib) etkileşimli kardeşi. Aynı tfview.build sözlüğünü kullanır;
# her katman aç/kapa yapılabilir, üzerine gelince her çizginin ne anlama geldiği SADE dille yazar:
#   mumlar (Mum Gücü renkleri) · EMA şeridi 10/21/50/100/200 · ana yapı Fibonacci · destek/direnç
#   · boşluklar (gap/FVG) · Elliott sayımı ve hedef kutusu · sinyal işaretleri (balina olayları,
#   Q-dry, ODI, efor kırılımı, VSA dönüşü) · işlem planı (giriş bandı, stop, kâr al 1/2/3)
# Ayrıca TradingView kısayolu (borsa önekiyle) burada üretilir.
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_log = logging.getLogger(__name__)

BG, PANEL, GRID, TXT, MUTED = "#0b0e11", "#0f1318", "#1c222b", "#d1d4dc", "#7a8394"
UP, DN = "#26a69a", "#ef5350"
CLS_COL = {1: "#00e676", 2: "#26a69a", 3: "#9ccc65", -1: "#ff1744", -2: "#ef5350",
           -3: "#ba68c8", 10: "#42a5f5", 11: "#ffa726"}
CLS_TXT = {1: "güçlü yükseliş mumu (geniş gövde, yüksek hacim — alıcılar istekli)",
           2: "orta güçte yükseliş mumu", 3: "hacimsiz yükseliş (alıcı az — güvenilmez)",
           -1: "sert düşüş mumu (satıcılar baskın, hacim yüksek)", -2: "orta düşüş mumu",
           -3: "hacimsiz düşüş (satıcı az — düşüş yorgun olabilir)",
           10: "hacimli duraksama (çok işlem ama fiyat ilerlemedi — biri karşı tarafı emiyor)",
           11: "dağıtım izi (yüksek hacimde zayıf kapanış — büyükler satıyor olabilir)",
           0: "sıradan mum"}
EMA_COL = {10: "#ffffff", 21: "#ffd54f", 50: "#29b6f6", 100: "#7e57c2", 200: "#e040fb"}
EMA_TXT = {10: "EMA10 — çok kısa vadeli ortalama; hızlı tepki verir",
           21: "EMA21 — yaklaşık 1 aylık ortalama; kısa vadeli trendin omurgası",
           50: "EMA50 — yaklaşık 2,5 aylık ortalama; orta vadeli trend",
           100: "EMA100 — yaklaşık 5 aylık ortalama",
           200: "EMA200 — yaklaşık 1 yıllık ortalama; üstü 'boğa', altı 'ayı' bölgesi sayılır"}
FIB_TXT = {
    0.0: "bacağın ucu (son tepe). Burası geçilirse yükseliş devam ediyor demektir",
    0.236: "sığ geri çekilme — güçlü trendlerde fiyat ancak buraya kadar iner",
    0.382: "ilk ciddi destek — sağlıklı geri çekilmenin başladığı yer",
    0.5: "yükselişin yarısı geri verildi — alıcıların sık devreye girdiği orta nokta",
    0.618: "'altın oran' — en çok izlenen dönüş seviyesi",
    0.786: "derin geri çekilme — bunun altı trendin bozulma riskini artırır",
    1.0: "bacağın başı (yükselişin başladığı dip). Altında yükseliş senaryosu biter",
    1.272: "uzantı hedefi — tepe geçilirse ilk kâr alma bölgesi",
    1.618: "ikinci uzantı hedefi (altın oran uzantısı)",
    2.618: "üçüncü uzantı (çok güçlü trendlerde)"}
TIER = {1: "FVG (küçük boşluk)", 2: "GAP (fiyat boşluğu)", 3: "KURUMSAL boşluk (büyük, hacimli)"}
LAYERS = ["Mum gücü renkleri", "EMA şeridi", "Fibonacci", "Destek / direnç", "Boşluklar (gap)",
          "Elliott", "Sinyal işaretleri", "İşlem planı", "Hacim"]
DEFAULT_LAYERS = ["Mum gücü renkleri", "EMA şeridi", "Fibonacci", "Destek / direnç",
                  "Sinyal işaretleri", "İşlem planı", "Hacim"]

# ------------------------------------------------------------------ TradingView
EXCH_TV = {"NMS": "NASDAQ", "NGM": "NASDAQ", "NCM": "NASDAQ", "NAS": "NASDAQ", "NASDAQ": "NASDAQ",
           "NYQ": "NYSE", "NYS": "NYSE", "NYSE": "NYSE", "ASE": "AMEX", "AMEX": "AMEX",
           "PCX": "AMEX", "NYSEARCA": "AMEX", "BTS": "CBOE", "CBOE": "CBOE", "PNK": "OTC",
           "OQB": "OTC", "OQX": "OTC"}
SPECIAL_TV = {"^VIX": "TVC:VIX", "^VIX3M": "CBOE:VIX3M", "^TNX": "TVC:US10Y", "^GSPC": "SP:SPX",
              "^NDX": "NASDAQ:NDX", "^RUT": "TVC:RUT", "DX-Y.NYB": "TVC:DXY",
              "GC=F": "COMEX:GC1!", "SI=F": "COMEX:SI1!", "HG=F": "COMEX:HG1!",
              "BZ=F": "NYMEX:BB1!", "CL=F": "NYMEX:CL1!", "BTC-USD": "COINBASE:BTCUSD",
              "ETH-USD": "COINBASE:ETHUSD", "HG=F/GC=F": "COMEX:HG1!/COMEX:GC1!",
              "GC=F/SI=F": "COMEX:GC1!/COMEX:SI1!"}
# Borsa bilgisi gelmediğinde: yaygın ETF'ler NYSE Arca'da (TradingView 'AMEX' öneki)
ARCA_ETF = {"SPY", "IWM", "DIA", "GLD", "SLV", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP",
            "XLU", "XLB", "XLRE", "XLC", "SMH", "XBI", "IBB", "ARKK", "ARKG", "TAN", "ICLN", "URA",
            "LIT", "KWEB", "FXI", "EEM", "EFA", "VTI", "VOO", "RSP", "HYG", "LQD", "JNK", "XME",
            "GDX", "GDXJ", "SIL", "COPX", "XHB", "ITB", "KRE", "XRT", "XOP", "OIH", "JETS", "UFO",
            "ITA", "XAR", "BOTZ", "ROBO", "HACK", "CIBR", "SKYY", "IGV", "SOXX", "VNQ", "TLT",
            "IEF", "SHY", "USO", "UNG", "DBA", "PAVE", "IGF", "MOO", "VIXY", "BITO", "IBIT"}


def tv_symbol(sym: str, exch: str | None = None) -> str:
    s = str(sym).strip().upper()
    if s in SPECIAL_TV:
        return SPECIAL_TV[s]
    tvs = s.replace("-", ".")                       # BRK-B → BRK.B
    e = str(exch or "").strip().upper()
    if e and e != "NAN" and e in EXCH_TV:
        return f"{EXCH_TV[e]}:{tvs}"
    if s in ARCA_ETF:
        return f"AMEX:{tvs}"
    return tvs                                      # TradingView çıplak sembolü kendisi çözer


def tv_url(sym: str, exch: str | None = None) -> str:
    from urllib.parse import quote
    return "https://www.tradingview.com/chart/?symbol=" + quote(tv_symbol(sym, exch), safe="")


# ------------------------------------------------------------------ yardımcılar
def _f(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.2f}" if abs(x) >= 1 else f"{x:.4g}"


def _pct(a: float, b: float) -> str:
    return f"%{(a / b - 1) * 100:+.1f}" if b else "—"


def _arr(x, n):
    if x is None:
        return np.zeros(n, bool)
    a = np.asarray(x)
    return a.astype(bool) if len(a) == n else np.zeros(n, bool)


def _hline(fig, x0, x1, y, color, name, hover, dash="solid", width=1.0, row=1, group=None,
           show=False, opacity=0.9):
    fig.add_trace(go.Scatter(x=[x0, x1], y=[y, y], mode="lines", name=name,
                             line=dict(color=color, dash=dash, width=width), opacity=opacity,
                             hovertemplate=f"<b>{name}</b> {_f(y)}<br>{hover}<extra></extra>",
                             legendgroup=group or name, showlegend=show), row=row, col=1)


def whale_events(d: pd.DataFrame) -> dict[str, np.ndarray]:
    """Balina olayları (APEX CORE v4 + V670 OMNI) — yalnız günlükte anlamlı."""
    try:
        from apex import engine as eng
        dd = d[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        core, omni = eng.apex_core(dd), eng.apex_omni(dd)
    except Exception as exc:                      # noqa: BLE001
        _log.info("ichart balina: %s", exc)
        return {}
    out = {}
    for col, src in (("ab_bull", omni), ("sweep_low", omni), ("st_in", core),
                     ("real_dist", core), ("exhausted", core), ("rd_top", core)):
        if src is not None and col in src:
            out[col] = src[col].fillna(False).astype(bool).to_numpy()
    return out


WHALE_MARK = {  # sütun: (etiket, sembol, renk, altta mı, sade açıklama)
    "ab_bull": ("🚀 Afterburner boğa", "triangle-up", "#00e676", True,
                "Büyük alıcılar sert ve hacimli girdi. 5 yılda 730 hissede 20 gün içinde piyasayı "
                "ortalama +1.65 puan geçti — kanıtlı en güçlü olay"),
    "sweep_low": ("🎣 Dip süpürmesi", "triangle-up", "#4fc3f7", True,
                  "Fiyat önceki dibin altına sarkıp stopları tetikledi, sonra geri toparlandı — "
                  "büyükler ucuz mal topladı (5 yılda +0.47 puan)"),
    "st_in": ("🕵️ Stealth IN", "diamond", "#80deea", True,
              "Fiyat pek oynamadan sessiz, hacimli alım (5 yılda +0.45 puan)"),
    "real_dist": ("🐋 Gerçek dağıtım", "triangle-down", "#ff1744", False,
                  "Büyükler yükselişe mal satıyor — 5 yılda sonraki 20 günde −1.07 puan"),
    "exhausted": ("⚠️ Whale tükenme", "x", "#ffa726", False,
                  "Alıcı gücü bitiyor — 5 yılda −0.54 puan (en tutarlı olumsuz işaret)"),
    "rd_top": ("🔴 Tepe dönüşü", "triangle-down", "#ff8a80", False,
               "Tepede dönüş örgüsü — 5 yılda −0.78 puan"),
}
V710_MARK = {
    "q_conf": ("★ Q-dry", "star", "#ffd54f", True,
               "Satıcılar kurudu: düşüşte hacim çok azaldı, ardından alım geldi — dip işareti"),
    "odi_bull": ("⬢ ODI alış", "hexagon", "#00e5ff", True,
                 "Emir akışı dengesizliği alıcı yönünde"),
    "odi_bear": ("⬢ ODI satış", "hexagon", "#ff4081", False,
                 "Emir akışı dengesizliği satıcı yönünde"),
    "eff_up": ("▲ Efor kırılımı", "triangle-up", "#00e676", True,
               "Hacimli (çaba harcanan) yukarı kırılım"),
    "eff_dn": ("▼ Efor kırılımı", "triangle-down", "#ff1744", False, "Hacimli aşağı kırılım"),
}
VSA_MARK = {
    "rev_up": ("◆ VSA dönüş ↑", "diamond", "#00e676", True,
               "Hacim-fiyat analizine göre aşağıdan yukarı dönüş mumu"),
    "rev_dn": ("◆ VSA dönüş ↓", "diamond", "#ff1744", False,
               "Hacim-fiyat analizine göre yukarıdan aşağı dönüş mumu"),
}


# ------------------------------------------------------------------ grafik
def figure(v: dict[str, Any], label: str = "", plan: dict | None = None,
           layers: list[str] | None = None, bars: int | None = None,
           height: int = 640) -> go.Figure:
    """tfview sözlüğünden etkileşimli Plotly grafiği. plan: {'giris_alt', 'giris_ust', 'stop',
    'tp1', 'tp2', 'tp3', 'karar'} (opsiyonel)."""
    L = set(layers if layers is not None else DEFAULT_LAYERS)
    show_vol = "Hacim" in L
    fig = make_subplots(rows=2 if show_vol else 1, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02, row_heights=[0.8, 0.2] if show_vol else [1.0])
    if not v.get("ok"):
        fig.add_annotation(text=f"{label}: {v.get('neden', 'veri yok')}", x=0.5, y=0.5,
                           xref="paper", yref="paper", showarrow=False, font=dict(color=TXT))
        fig.update_layout(height=300, paper_bgcolor=BG, plot_bgcolor=PANEL)
        return fig
    d = v["d"]
    n = len(d)
    x = d.index
    o, h, l, c = (d[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    px = float(c[-1])
    tf = v.get("tf", "Günlük")
    nv = min(n, bars or {"Günlük": 160, "Haftalık": 130, "Aylık": 120}.get(tf, 150))
    s0 = n - nv
    lo_v, hi_v = float(np.nanmin(l[s0:])), float(np.nanmax(h[s0:]))
    rng = hi_v - lo_v
    # sağda boşluk (gelecek): seviyelerin etiketi ve Elliott kutusu için
    step = (x[-1] - x[-2]) if n > 1 else pd.Timedelta(days=1)
    x_end = x[-1] + step * max(8, nv // 8)
    x_left = x[s0]

    def inview(y):
        return y is not None and np.isfinite(y) and lo_v - rng * 0.35 <= y <= hi_v + rng * 0.35

    # ---- mumlar
    mg = v.get("mg")
    cls = (mg["cls"].to_numpy() if mg is not None and "cls" in mg and len(mg) == n else None)
    chg = np.r_[np.nan, c[1:] / c[:-1] - 1] * 100
    vol = d["Volume"].to_numpy(float) if "Volume" in d else np.zeros(n)
    vma = pd.Series(vol).rolling(20, min_periods=5).mean().to_numpy()
    groups: dict[int, np.ndarray] = {}
    if "Mum gücü renkleri" in L and cls is not None:
        for k_ in set(int(z) for z in np.nan_to_num(cls)):
            groups[k_] = np.flatnonzero(np.nan_to_num(cls).astype(int) == k_)
    else:
        groups[0] = np.arange(n)
    for k_, ii in groups.items():
        if not len(ii):
            continue
        col = CLS_COL.get(k_)
        txt = [f"{x[i]:%d.%m.%Y}<br>Açılış {_f(o[i])} · Yüksek {_f(h[i])}<br>Düşük {_f(l[i])} · "
               f"Kapanış {_f(c[i])} ({chg[i]:+.2f}%)"
               + (f"<br>Hacim ortalamanın {vol[i] / vma[i]:.1f} katı"
                  if np.isfinite(vma[i]) and vma[i] > 0 and vol[i] > 0 else "")
               + (f"<br><i>{CLS_TXT.get(k_, '')}</i>" if k_ in CLS_TXT and cls is not None
                  and "Mum gücü renkleri" in L else "") for i in ii]
        fig.add_trace(go.Candlestick(
            x=x[ii], open=o[ii], high=h[ii], low=l[ii], close=c[ii], name=CLS_TXT.get(k_, "mum")
            .split(" (")[0], text=txt, hoverinfo="text",
            increasing=dict(line=dict(color=col or UP, width=1), fillcolor=col or UP),
            decreasing=dict(line=dict(color=col or DN, width=1), fillcolor=col or DN),
            showlegend=False), row=1, col=1)

    # ---- EMA şeridi
    if "EMA şeridi" in L:
        from apex.indicators import ema
        for p_ in (10, 21, 50, 100, 200):
            s = ema(d["Close"], p_).to_numpy(float)
            if np.isfinite(s).sum() < 5:
                continue
            fig.add_trace(go.Scatter(
                x=x, y=s, mode="lines", name=f"EMA{p_}",
                line=dict(color=EMA_COL[p_], width=1.6 if p_ == 200 else 1.0),
                opacity=0.85 if p_ != 10 else 0.5,
                hovertemplate=f"<b>EMA{p_}</b> %{{y:.2f}}<br>{EMA_TXT[p_]}<extra></extra>"),
                row=1, col=1)

    # ---- Fibonacci (ana yapı)
    fb = v.get("fib") or {}
    if "Fibonacci" in L and fb.get("ok"):
        yon = "yükseliş" if fb["up"] else "düşüş"
        for lv in (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618):
            y = fb["levels"].get(lv)
            if not inview(y):
                continue
            t = FIB_TXT[lv] if fb["up"] else (
                "düşüş bacağının dibi" if lv == 0.0 else "düşüş bacağının başı (tepe)" if lv == 1.0
                else f"düşüşün %{lv * 100:.1f}'i geri alındı — tepki direnci")
            key = lv in (0.382, 0.5, 0.618)
            _hline(fig, x_left, x_end, y, "#8d99ae" if not key else "#b0bec5", f"Fib {lv}",
                   f"Ana {yon} bacağı (%{fb['move_pct']:.0f}): {t}. Fiyata uzaklık {_pct(y, px)}",
                   dash="dot", width=1.2 if key else 0.8, group="Fibonacci",
                   show=lv == 0.5, opacity=0.8)
            fig.add_annotation(x=x_end, y=y, text=f"F{lv}", showarrow=False, xanchor="right",
                               yanchor="top", font=dict(size=9, color="#8d99ae"), row=1, col=1)
        if fb["up"]:
            a_, b_ = fb["levels"].get(0.786), fb["levels"].get(0.382)
            if inview(a_) and inview(b_):
                fig.add_shape(type="rect", x0=x_left, x1=x_end, y0=a_, y1=b_, line_width=0,
                              fillcolor="rgba(141,153,174,0.06)", row=1, col=1)

    # ---- destek / direnç
    sr = v.get("sr") or {}
    if "Destek / direnç" in L:
        hz_txt = {0: "kısa vadeli", 1: "orta vadeli", 2: "uzun vadeli (güçlü)"}
        dash = {0: "dot", 1: "dash", 2: "solid"}
        for side, lst, col in (("Direnç", sr.get("res", []), "#ff5252"),
                               ("Destek", sr.get("sup", []), "#00e676")):
            for k, cl in enumerate(lst):
                y = cl["px"]
                if not inview(y):
                    continue
                b0 = int(cl.get("b0", 0))
                x0 = x[min(max(b0, 0), n - 1)]
                hz = int(cl.get("hz", 0))
                t = (f"{hz_txt.get(hz, '')} {side.lower()}: fiyat bu bölgede {cl.get('cnt', 2)} kez "
                     f"döndü. " + ("Yukarı çıkarken satıcıların beklediği yer — aşılırsa yol açılır."
                                   if side == "Direnç" else
                                   "Düşerken alıcıların beklediği yer — kırılırsa düşüş hızlanabilir.")
                     + f" Fiyata uzaklık {_pct(y, px)}")
                _hline(fig, max(x0, x_left), x_end, y, col, f"{side[0]}{k + 1}", t,
                       dash=dash.get(hz, "dash"), width=1.0 + 0.4 * hz,
                       group=side, show=k == 0)
                fig.add_annotation(x=x_end, y=y, text=f"{side[0]}{k + 1} {_f(y)}", showarrow=False,
                                   xanchor="right", yanchor="bottom",
                                   font=dict(size=9, color=col), row=1, col=1)

    # ---- boşluklar
    gm = v.get("gaps") or {}
    if "Boşluklar (gap)" in L:
        boxes = [b for b in gm.get("boxes", []) if inview(b["top"]) or inview(b["bot"])]
        boxes = sorted(boxes, key=lambda b: abs((b["top"] + b["bot"]) / 2 - px))[:6]
        for b in boxes:
            x0 = x[min(max(int(b["left"]), 0), n - 1)]
            up_ = b["dir"] > 0
            fig.add_shape(type="rect", x0=max(x0, x_left), x1=x_end, y0=b["bot"], y1=b["top"],
                          line=dict(color="#f7e08a", width=0.8, dash="dash" if up_ else "dot"),
                          fillcolor=f"rgba(247,224,138,{0.08 + 0.04 * b['tier']})", row=1, col=1)
            t = (f"{TIER[b['tier']]} {'yukarı' if up_ else 'aşağı'}: fiyat buradan hızla geçti, "
                 "arada işlem az oldu. Fiyat çoğu zaman bu boşluğu doldurmaya geri döner; "
                 + ("doldururken destek olur." if up_ else "doldururken direnç olur."))
            fig.add_trace(go.Scatter(x=[max(x0, x_left)], y=[(b["top"] + b["bot"]) / 2],
                                     mode="markers", marker=dict(size=8, color="#f7e08a",
                                                                 symbol="square-open"),
                                     name="Boşluk", legendgroup="gap", showlegend=False,
                                     hovertemplate=f"<b>Boşluk</b> {_f(b['bot'])}–{_f(b['top'])}"
                                                   f"<br>{t}<extra></extra>"), row=1, col=1)

    # ---- Elliott
    e = v.get("ell")
    if "Elliott" in L and e and e.get("B"):
        Bx, P = e["B"], e["P"]
        k = e.get("kind")
        n_p = len(P)
        idx, names = [], None
        if k == "w3" and n_p >= 3:
            idx, names = list(range(n_p - 3, n_p)), ["0", "1", "2"]
        elif k == "w5" and n_p >= 5:
            idx, names = list(range(n_p - 5, n_p)), ["0", "1", "2", "3", "4"]
        elif k and str(k).startswith("abc") and e.get("last_end") is not None:
            le = e["last_end"]
            idx = [i for i in range(le - 5, n_p) if 0 <= i < n_p]
            names = ["0", "1", "2", "3", "4", "5", "A", "B", "C"][:len(idx)]
        if names:
            xs = [x[min(max(int(Bx[i]), 0), n - 1)] for i in idx]
            ys = [P[i] for i in idx]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines+markers+text", text=names, textposition="top center",
                textfont=dict(color="#b39dff", size=12), name="Elliott",
                line=dict(color="#b39dff", width=1.4), marker=dict(size=7, color="#b39dff"),
                hovertemplate="<b>Elliott dalga %{text}</b> %{y:.2f}<br>Elliott: piyasa 5 dalga "
                              "ilerler, 3 dalga düzeltir. Şu an: " + str(e.get("phase", ""))
                              + "<extra></extra>"), row=1, col=1)
        z = e.get("zone")
        if z and not z.get("dead") and min(z["zlo"], z["zhi"]) > 0 and abs(z["near"] / px - 1) < 0.6:
            fig.add_shape(type="rect", x0=x[-1] + step, x1=x_end, y0=z["zlo"], y1=z["zhi"],
                          line=dict(color="#b39dff", width=1), fillcolor="rgba(179,157,255,0.15)",
                          row=1, col=1)
            fig.add_trace(go.Scatter(
                x=[x[-1] + step * 2], y=[(z["zlo"] + z["zhi"]) / 2], mode="markers",
                marker=dict(size=9, color="#b39dff", symbol="square"), name="Elliott hedef",
                showlegend=False,
                hovertemplate=f"<b>Elliott hedef bölgesi</b> {_f(z['zlo'])}–{_f(z['zhi'])}<br>"
                              f"Dalga sayımına göre fiyatın gitmesi beklenen yer ({z.get('name', '')})."
                              f"<br>Fiyata uzaklık {_pct(z['near'], px)}<extra></extra>"),
                row=1, col=1)
            for yv, col, nm, t in ((z.get("stop"), "#ffa726", "Elliott stop",
                                    "bu seviye kırılırsa dalga senaryosu zayıflar"),
                                   (z.get("inv"), "#ff5252", "Elliott geçersizlik",
                                    "bu seviye geçilirse sayım tamamen çöker")):
                if yv is not None and inview(yv):
                    _hline(fig, x[max(n - 15, 0)], x_end, yv, col, nm, t, dash="dash",
                           group="Elliott")

    # ---- sinyal işaretleri
    if "Sinyal işaretleri" in L:
        off = rng * 0.03
        recent = np.zeros(n, bool)
        recent[s0:] = True
        sets = []
        if tf == "Günlük" and v.get("has_vol"):
            we = whale_events(d)
            for k_, a in we.items():
                sets.append((a, WHALE_MARK[k_]))
        vv = v.get("v710") or {}
        for k_, spec in V710_MARK.items():
            sets.append((_arr(vv.get(k_), n), spec))
        vs = v.get("vsa") or {}
        rec40 = np.zeros(n, bool)
        rec40[-40:] = True
        for k_, spec in VSA_MARK.items():
            sets.append((_arr(vs.get(k_), n) & rec40, spec))
        for a, (nm, sym, col, below, txt) in sets:
            ii = np.flatnonzero(np.asarray(a, bool) & recent)
            if not len(ii):
                continue
            yy = (l[ii] - off) if below else (h[ii] + off)
            fig.add_trace(go.Scatter(
                x=x[ii], y=yy, mode="markers", name=nm,
                marker=dict(symbol=sym, size=11 if "Afterburner" in nm else 9, color=col,
                            line=dict(color=BG, width=0.6)),
                hovertemplate=f"<b>{nm}</b> %{{x|%d.%m.%Y}}<br>{txt}<extra></extra>"),
                row=1, col=1)

    # ---- işlem planı
    if "İşlem planı" in L and plan:
        pl = {k: _num(plan.get(k)) for k in ("giris_alt", "giris_ust", "stop", "tp1", "tp2", "tp3")}
        x0p = x[max(n - 25, 0)]
        if np.isfinite(pl["giris_alt"]) and np.isfinite(pl["giris_ust"]):
            lo_, hi_ = sorted((pl["giris_alt"], pl["giris_ust"]))
            if hi_ - lo_ < rng * 0.004:
                hi_ = lo_ + rng * 0.004
            fig.add_shape(type="rect", x0=x0p, x1=x_end, y0=lo_, y1=hi_, line_width=0,
                          fillcolor="rgba(0,229,255,0.16)", row=1, col=1)
            fig.add_trace(go.Scatter(
                x=[x0p], y=[(lo_ + hi_) / 2], mode="markers", name="Giriş bölgesi",
                marker=dict(symbol="square", size=9, color="#00e5ff"),
                hovertemplate=f"<b>Giriş bölgesi</b> {_f(lo_)}–{_f(hi_)}<br>Alım için beklenen "
                              f"fiyat aralığı ({plan.get('karar', '')}). Fiyata uzaklık "
                              f"{_pct((lo_ + hi_) / 2, px)}<extra></extra>"), row=1, col=1)
        ent = np.nanmean([pl["giris_alt"], pl["giris_ust"]]) if np.isfinite(pl["giris_alt"]) else px
        for key, col, nm, t in (
                ("stop", "#ff1744", "Stop", "zarar kes: fiyat buranın altında kapanırsa pozisyon "
                                            "kapatılır, plan geçersiz"),
                ("tp1", "#00e676", "Kâr al 1", "ilk hedef — pozisyonun bir kısmı satılır, stop "
                                               "giriş fiyatına çekilir"),
                ("tp2", "#69f0ae", "Kâr al 2", "ikinci hedef — kalan pozisyonun bir kısmı"),
                ("tp3", "#b9f6ca", "Kâr al 3", "üçüncü hedef — iz süren stopla kalan kısım")):
            y = pl[key]
            if not np.isfinite(y):
                continue
            r_ = (y - ent) / (ent - pl["stop"]) if np.isfinite(pl["stop"]) and ent > pl["stop"] \
                else np.nan
            extra = f" · {r_:+.1f}R" if np.isfinite(r_) and key != "stop" else ""
            _hline(fig, x0p, x_end, y, col, nm, f"{t}. Fiyata uzaklık {_pct(y, px)}{extra}",
                   dash="dash" if key != "stop" else "solid", width=1.4, group="plan", show=False)
            fig.add_annotation(x=x0p, y=y, text=f"{nm} {_f(y)}", showarrow=False, xanchor="left",
                               yanchor="bottom", font=dict(size=10, color=col), row=1, col=1)

    # ---- hacim
    if show_vol:
        if np.nansum(vol) > 0:
            vcol = np.where(c >= o, "rgba(38,166,154,0.6)", "rgba(239,83,80,0.6)")
            fig.add_trace(go.Bar(x=x, y=vol, marker_color=vcol, name="Hacim", showlegend=False,
                                 hovertemplate="Hacim %{y:,.0f}<extra></extra>"), row=2, col=1)
            fig.add_trace(go.Scatter(x=x, y=vma, mode="lines", line=dict(color="#ffd54f", width=1),
                                     name="20 bar hacim ort.", showlegend=False,
                                     hovertemplate="20 bar ortalama hacim %{y:,.0f}<extra></extra>"),
                          row=2, col=1)
        else:
            fig.add_annotation(text="hacim verisi yok (oran / endeks)", x=0.5, y=0.1,
                               xref="paper", yref="paper", showarrow=False,
                               font=dict(color=MUTED, size=10))

    # ---- son fiyat çizgisi
    fig.add_hline(y=px, line=dict(color="#ffffff", width=0.6, dash="dot"), opacity=0.5,
                  row=1, col=1)
    fig.add_annotation(x=x_end, y=px, text=f"<b>{_f(px)}</b>", showarrow=False,
                       xanchor="left", bgcolor="#ffffff", font=dict(color=BG, size=10),
                       row=1, col=1)

    # ---- düzen
    y0, y1 = lo_v - rng * 0.08, hi_v + rng * 0.12
    if plan:
        for k_ in ("stop", "tp1"):
            yv = _num(plan.get(k_))
            if np.isfinite(yv) and lo_v - rng * 0.6 < yv < hi_v + rng * 0.6:
                y0, y1 = min(y0, yv - rng * 0.03), max(y1, yv + rng * 0.03)
    has_wk = tf == "Günlük" and len(x) > 10 and not any(t.weekday() >= 5 for t in x[-30:])
    fig.update_layout(
        height=height, paper_bgcolor=BG, plot_bgcolor=PANEL, font=dict(color=TXT, size=11),
        margin=dict(l=6, r=6, t=34, b=6), dragmode="pan", hovermode="closest",
        hoverlabel=dict(bgcolor="#131319", bordercolor="#2b2b36", font=dict(color="#ececf1",
                                                                             size=12),
                        align="left"),
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        title=dict(text=f"{label} · {tf} · {_f(px)} ({chg[-1]:+.2f}%)", x=0.01, y=0.99,
                   font=dict(size=14)),
        xaxis_rangeslider_visible=False)
    fig.update_xaxes(gridcolor=GRID, showspikes=True, spikecolor=MUTED, spikethickness=1,
                     spikemode="across", range=[x_left, x_end])
    fig.update_yaxes(gridcolor=GRID, side="right", showspikes=True, spikecolor=MUTED,
                     spikethickness=1)
    fig.update_yaxes(range=[y0, y1], row=1, col=1)
    if has_wk:
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    if show_vol and np.nansum(vol[s0:]) > 0:
        fig.update_yaxes(range=[0, float(np.nanmax(vol[s0:])) * 1.15], row=2, col=1,
                         showticklabels=False)
    return fig


def _num(x) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


# ------------------------------------------------------------------ "Durum ne?"
def durum(v: dict[str, Any], label: str = "", plan: dict | None = None,
          extra: list[str] | None = None) -> list[str]:
    """Grafiğin SADE dille özeti — terim bilmeyen birinin anlayacağı cümleler."""
    if not v.get("ok"):
        return [f"{label}: {v.get('neden', 'veri yok')}."]
    d = v["d"]
    c = d["Close"]
    px = float(c.iloc[-1])
    out: list[str] = []
    tf = v.get("tf", "Günlük").lower()
    sc = v.get("score", np.nan)
    if np.isfinite(sc):
        out.append("🧭 **Genel tablo:** " + (
            "alıcılar açıkça önde; göstergelerin çoğu yükselişi destekliyor." if sc >= 35 else
            "alıcılar biraz önde ama her gösterge aynı fikirde değil." if sc >= 12 else
            "kararsız — göstergeler birbirini tutmuyor, acele etmeye gerek yok." if sc > -12 else
            "satıcılar önde; düşüş baskısı var." if sc > -35 else
            "satıcılar açıkça önde; göstergelerin çoğu düşüşü gösteriyor.")
            + f" ({tf} skor {sc:+.0f}, −100 ile +100 arası)")
    a = v.get("ema_stack") or {}
    if a.get("ok"):
        faz = a.get("faz", "")
        txt = {"tam_boga": "Fiyat bütün ortalamaların üstünde ve hepsi yukarı bakıyor — güçlü "
                           "yükseliş trendi.",
               "boga_geri": "Uzun vadeli trend yukarı; fiyat kısa bir mola/geri çekilme yaşıyor.",
               "boga_derin": "Uzun vadeli trend hâlâ yukarı ama fiyat orta vadeli ortalamanın "
                             "altına sarktı — derin düzeltme.",
               "kirilma": "Fiyat 1 yıllık ortalamanın (EMA200) altına indi — trend bozuluyor.",
               "yorulma": "Yükseliş yoruldu; kısa vadeli ortalamalar aşağı dönmeye başladı.",
               "erken_donus": "Düşüşten sonra ilk toparlanma işaretleri — dönüş henüz erken.",
               "ayi_tepki": "Ana trend aşağı; şu anki yükseliş bir tepki olabilir.",
               "tam_ayi": "Fiyat bütün ortalamaların altında ve hepsi aşağı bakıyor — güçlü "
                          "düşüş trendi.",
               "sikisma": "Ortalamalar birbirine çok yaklaştı (sıkışma) — yakında sert bir hareket "
                          "gelebilir, yönü henüz belli değil.",
               "karisik": "Ortalamalar karışık — net bir trend yok."}.get(faz, "")
        if txt:
            out.append("📈 **Trend:** " + txt)
        uz = a.get("uzama", np.nan)
        if np.isfinite(uz) and uz > 3:
            out.append("⚠️ **Fiyat çok koştu:** kısa vadeli ortalamadan çok uzaklaştı; bu "
                       "noktadan almak 'kovalamak' olur, geri çekilme beklemek daha güvenli.")
        elif np.isfinite(uz) and uz < -3:
            out.append("🔻 **Aşırı satılmış:** fiyat ortalamanın çok altında; kısa bir tepki "
                       "yükselişi gelebilir ama trend dönmüş sayılmaz.")
    fb = v.get("fib") or {}
    if fb.get("ok"):
        r = fb["r"]
        if fb["up"]:
            if r < 0:
                s = "fiyat son yükselişin tepesini aştı — yeni zirve bölgesinde."
            elif r < 0.236:
                s = "fiyat son yükselişin tepesine çok yakın; geri çekilme gelmedi."
            elif r < 0.382:
                s = "fiyat yükselişin küçük bir kısmını geri verdi (sığ geri çekilme)."
            elif r <= 0.786:
                s = (f"fiyat son yükselişin yaklaşık %{r * 100:.0f}'ini geri verdi. Bu bölge "
                     "(0.382–0.786) alıcıların sık devreye girdiği 'indirim bölgesi'dir; "
                     "dönüş mumu gelirse alım fırsatı olabilir.")
            elif r <= 1:
                s = "fiyat yükselişin neredeyse tamamını geri verdi — çok derin, dikkat."
            else:
                s = "fiyat yükselişin başladığı yerin de altına indi — yükseliş yapısı bozuldu."
            out.append(f"📐 **Son hareket (%{fb['move_pct']:.0f} yükseliş):** " + s)
        else:
            s = ("düşüşün dibine yakın, henüz toparlanma yok." if r < 0.236 else
                 f"düşüşün %{r * 100:.0f}'i geri alındı — tepki sürüyor ama ana yön hâlâ aşağı."
                 if r <= 1 else "düşüşün tamamı geri alındı — yeni yükseliş başlıyor olabilir.")
            out.append(f"📐 **Son hareket (%{fb['move_pct']:.0f} düşüş):** " + s)
    sr = v.get("sr") or {}
    res, sup = sr.get("res", []), sr.get("sup", [])
    if sup or res:
        bits = []
        if sup:
            bits.append(f"en yakın destek (alıcıların beklediği yer) **{_f(sup[0]['px'])}** "
                        f"({_pct(sup[0]['px'], px)})")
        if res:
            bits.append(f"en yakın direnç (satıcıların beklediği yer) **{_f(res[0]['px'])}** "
                        f"({_pct(res[0]['px'], px)})")
        out.append("🧱 **Seviyeler:** " + "; ".join(bits) + ".")
        if sup:
            out.append(f"👉 **Basitçe:** {_f(sup[0]['px'])} üzerinde kaldıkça olumlu senaryo "
                       "geçerli" + (f", {_f(res[0]['px'])} aşılırsa yol açılır" if res else "")
                       + f"; {_f(sup[0]['px'])} altında {tf} kapanış gelirse senaryo bozulur.")
    if tf == "günlük" and v.get("has_vol"):
        we = whale_events(d)
        last = []
        for k_, a_ in we.items():
            idx = np.flatnonzero(a_[-10:])
            if len(idx):
                g = 9 - int(idx[-1])
                last.append((g, k_))
        for g, k_ in sorted(last)[:3]:
            nm, _, _, _, txt = WHALE_MARK[k_]
            out.append(f"🐋 **{nm}** ({'bugün' if g == 0 else f'{g} gün önce'}): {txt}.")
    e = v.get("ell")
    if e and e.get("zone") and not e["zone"].get("dead") and min(e["zone"]["zlo"],
                                                                   e["zone"]["zhi"]) > 0:
        z = e["zone"]
        out.append(f"🌊 **Elliott dalga sayımı:** beklenen hedef {_f(z['zlo'])}–{_f(z['zhi'])} "
                   f"({_pct(z['near'], px)}). Dalga sayımı yorumdur, kesin değildir.")
    if plan:
        pl = {k: _num(plan.get(k)) for k in ("giris_alt", "giris_ust", "stop", "tp1", "tp2")}
        if np.isfinite(pl["stop"]) and np.isfinite(pl["tp1"]):
            ent = (np.nanmean([pl["giris_alt"], pl["giris_ust"]])
                   if np.isfinite(pl["giris_alt"]) else px)
            risk = (ent - pl["stop"]) / ent * 100
            gain = (pl["tp1"] - ent) / ent * 100
            out.append(f"🎯 **Plan ({plan.get('karar', '')}):** giriş ~{_f(ent)}, zarar kes "
                       f"{_f(pl['stop'])} (−%{risk:.1f}), ilk hedef {_f(pl['tp1'])} (+%{gain:.1f}). "
                       f"Yani 1 birim riske karşı ~{gain / risk:.1f} birim kazanç hedefleniyor."
                       if risk > 0 else f"🎯 **Plan:** {plan.get('karar', '')}")
    for t in extra or []:
        out.append(t)
    return out
