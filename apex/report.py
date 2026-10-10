# apex/report.py — AETHER APEX
from __future__ import annotations

from apex.macro import battery_changes, battery_history, regime_shifts, score_changes
from apex.universe import holdings

import datetime as dt
import io
import os
import re
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402
from matplotlib.figure import Figure                       # noqa: E402

from reportlab.lib import colors                           # noqa: E402
from reportlab.lib.enums import TA_LEFT                    # noqa: E402
from reportlab.lib.pagesizes import A4                     # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm                         # noqa: E402
from reportlab.pdfbase import pdfmetrics                   # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont               # noqa: E402
from reportlab.platypus import (                           # noqa: E402
    HRFlowable, Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)

# --------------------------------------------------------------------------
# Renk paleti (baskı için açık zemin)
# --------------------------------------------------------------------------
INK = colors.HexColor("#14141a")
INK2 = colors.HexColor("#4a4a58")
LINE = colors.HexColor("#d5d5df")
ACCENT = colors.HexColor("#007a8c")
POS = colors.HexColor("#137a4d")
NEG = colors.HexColor("#b3271f")
BAND = colors.HexColor("#f2f4f7")
CHART_SERIES = ["#2b6cb0", "#c05621", "#2f855a", "#b7791f", "#b83280",
                "#4c51bf", "#c53030", "#2c7a7b"]

_FONT = "Helvetica"
_FONT_B = "Helvetica-Bold"
_UNICODE_OK = False
_FONT_CHECKED = False


# --------------------------------------------------------------------------
# Font
# --------------------------------------------------------------------------
def _font_candidates() -> list[tuple[str, str]]:
    """(normal, bold) TTF yolları — en güvenilir kaynak önce."""
    out: list[tuple[str, str]] = []
    try:
        base = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
        out.append((os.path.join(base, "DejaVuSans.ttf"),
                    os.path.join(base, "DejaVuSans-Bold.ttf")))
    except Exception:                                      # pragma: no cover
        pass
    out += [
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
         "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
        ("/Library/Fonts/Arial Unicode.ttf", "/Library/Fonts/Arial Unicode.ttf"),
        ("C:\\Windows\\Fonts\\arial.ttf", "C:\\Windows\\Fonts\\arialbd.ttf"),
    ]
    return out


def ensure_fonts() -> bool:
    """Türkçe destekli fontu kaydeder. Döner: unicode font bulundu mu."""
    global _FONT, _FONT_B, _UNICODE_OK, _FONT_CHECKED
    if _UNICODE_OK or _FONT_CHECKED:
        return _UNICODE_OK
    _FONT_CHECKED = True
    for regular, bold in _font_candidates():
        try:
            if not os.path.exists(regular):
                continue
            pdfmetrics.registerFont(TTFont("APEXSans", regular))
            pdfmetrics.registerFont(
                TTFont("APEXSans-Bold", bold if os.path.exists(bold) else regular))
            _FONT, _FONT_B, _UNICODE_OK = "APEXSans", "APEXSans-Bold", True
            return True
        except Exception:                                  # pragma: no cover
            continue
    return False


_TR_ASCII = str.maketrans({
    "ğ": "g", "Ğ": "G", "ş": "s", "Ş": "S", "ı": "i", "İ": "I",
    "ç": "c", "Ç": "C", "ö": "o", "Ö": "O", "ü": "u", "Ü": "U",
    "─": "-", "•": "-", "’": "'", "“": '"', "”": '"', "…": "...",
})

# Emoji ve piktogram blokları (DejaVu bunları taşımaz)
_EMOJI = re.compile(
    "[" "\U0001F000-\U0001FAFF" "\u2190-\u21FF" "\u2300-\u27BF"
    "\u2B00-\u2BFF" "\uFE0F" "\u200D" "]+")

# Ok karakterleri anlam taşıyor — kelimeye çevrilir, silinmez
_ARROWS = {"⇈": "artiyor", "↗": "donuyor", "⇊": "bozuluyor",
           "↘": "soluluyor", "→": "yatay", "▸": ">", "−": "-"}


def clean(text: Any) -> str:
    """Emoji'yi atar, markdown kalıntısını sadeleştirir, fontu yoksa ASCII'ye düşer."""
    if text is None or (isinstance(text, float) and not np.isfinite(text)):
        return ""
    ensure_fonts()          # font durumu ASCII'ye düşüp düşmeyeceğimizi belirler
    s = str(text)
    for k, v in _ARROWS.items():
        s = s.replace(k, v)
    s = _EMOJI.sub("", s)
    s = s.replace("**", "").replace("`", "")
    s = re.sub(r"\s+", " ", s).strip()
    if not _UNICODE_OK:
        s = s.translate(_TR_ASCII)
        s = s.encode("ascii", "ignore").decode("ascii")
    return s


def _num(x: Any, spec: str = ".2f", suffix: str = "") -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return clean(x)
    if not np.isfinite(v):
        return "-"
    return format(v, spec) + suffix


# --------------------------------------------------------------------------
# Stiller
# --------------------------------------------------------------------------
def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    s = {
        "title": ParagraphStyle("apex_title", parent=base["Title"],
                                fontName=_FONT_B, fontSize=22, leading=26,
                                textColor=INK, alignment=TA_LEFT,
                                spaceAfter=2),
        "sub": ParagraphStyle("apex_sub", parent=base["Normal"],
                              fontName=_FONT, fontSize=9.5, leading=13,
                              textColor=INK2, spaceAfter=10),
        "h1": ParagraphStyle("apex_h1", parent=base["Heading1"],
                             fontName=_FONT_B, fontSize=14, leading=18,
                             textColor=ACCENT, spaceBefore=14, spaceAfter=6),
        "h2": ParagraphStyle("apex_h2", parent=base["Heading2"],
                             fontName=_FONT_B, fontSize=11, leading=14,
                             textColor=INK, spaceBefore=9, spaceAfter=4),
        "body": ParagraphStyle("apex_body", parent=base["Normal"],
                               fontName=_FONT, fontSize=9, leading=13,
                               textColor=INK, spaceAfter=5),
        "small": ParagraphStyle("apex_small", parent=base["Normal"],
                                fontName=_FONT, fontSize=7.8, leading=10.5,
                                textColor=INK2, spaceAfter=4),
        "cell": ParagraphStyle("apex_cell", parent=base["Normal"],
                               fontName=_FONT, fontSize=7.2, leading=9),
    }
    return s


# --------------------------------------------------------------------------
# Grafikler (matplotlib -> PNG akışı)
# --------------------------------------------------------------------------
def _fig_to_image(fig: Figure, width_mm: float, height_mm: float) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=width_mm * mm, height=height_mm * mm)


def _style_axes(ax) -> None:
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c8c8d2")
    ax.tick_params(colors="#4a4a58", labelsize=7.5)
    ax.grid(True, color="#e6e6ee", linewidth=0.7)
    ax.set_axisbelow(True)


def battery_chart(battery: dict[str, int], deltas: dict[str, float] | None,
                  period: str = "") -> Image:
    """Varlık sınıfı bataryası; önceki dönem soluk çubuk olarak arkada."""
    keys = list(battery)
    vals = [battery[k] for k in keys]
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    y = np.arange(len(keys))
    if deltas:
        prev = [vals[i] - (deltas.get(k) or 0) for i, k in enumerate(keys)]
        ax.barh(y, prev, color="#dfe3ea", height=0.62,
                label=f"{clean(period)} once")
    ax.barh(y, vals, color=[CHART_SERIES[i % len(CHART_SERIES)]
                            for i in range(len(keys))], height=0.42,
            label="simdi")
    for i, v in enumerate(vals):
        d = (deltas or {}).get(keys[i])
        etiket = f"{v}"
        if d is not None and np.isfinite(d):
            etiket += f"  ({d:+.0f})"
        ax.text(v + 2, i, etiket, va="center", fontsize=7.5, color="#14141a")
    ax.set_yticks(y, [clean(k) for k in keys], fontsize=8)
    ax.invert_yaxis()          # tablo sırasıyla aynı olsun
    ax.set_xlim(0, 118)
    ax.axvline(50, color="#b9b9c6", linewidth=0.9, linestyle=":")
    ax.set_xticks([0, 25, 50, 75, 100])
    _style_axes(ax)
    if deltas:
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    return _fig_to_image(fig, 168, 66)


def battery_history_chart(hist: pd.DataFrame) -> Image | None:
    """Batarya ve bileşik risk skorunun son N günlük seyri."""
    if hist is None or hist.empty:
        return None
    cols = [c for c in hist.columns if c not in ("Risk Skoru", "Rejim")]
    fig, ax = plt.subplots(figsize=(6.6, 2.5))
    for i, c in enumerate(cols):
        ax.plot(hist.index, hist[c], label=clean(c),
                color=CHART_SERIES[i % len(CHART_SERIES)], linewidth=1.5)
    if "Risk Skoru" in hist.columns:
        ax.plot(hist.index, hist["Risk Skoru"], label="Bilesik risk",
                color="#14141a", linewidth=1.8, linestyle="--")
    ax.axhline(50, color="#b9b9c6", linewidth=0.9, linestyle=":")
    ax.set_ylim(0, 100)
    _style_axes(ax)
    ax.legend(fontsize=6.5, frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.24))
    fig.autofmt_xdate(rotation=0, ha="center")
    return _fig_to_image(fig, 168, 64)


def score_chart(scores: pd.DataFrame, delta_col: str = "Δ 1 hafta") -> Image | None:
    """Alt skorların dönemsel değişimi — rejimi ne itiyor, ne çekiyor."""
    if scores is None or scores.empty or delta_col not in scores.columns:
        return None
    df = scores.dropna(subset=[delta_col]).copy()
    if df.empty:
        return None
    df = df.sort_values(delta_col)
    fig, ax = plt.subplots(figsize=(6.6, max(2.0, 0.28 * len(df))))
    renk = [POS.hexval()[2:] if v >= 0 else NEG.hexval()[2:]
            for v in df[delta_col]]
    ax.barh(np.arange(len(df)), df[delta_col],
            color=["#" + c for c in renk], height=0.6)
    ax.set_yticks(np.arange(len(df)), [clean(x) for x in df["Skor"]], fontsize=7.5)
    ax.axvline(0, color="#8a8a98", linewidth=1)
    _style_axes(ax)
    ax.set_xlabel(clean(delta_col) + " (puan)", fontsize=7.5, color="#4a4a58")
    return _fig_to_image(fig, 168, max(46, 7.5 * len(df)))


def quadrant_chart(table: pd.DataFrame, label_n: int = 14) -> Image | None:
    """Tema momentum x ivme dağılımı."""
    if table is None or table.empty:
        return None
    if "Getiri %" not in table.columns or "İvme" not in table.columns:
        return None
    df = table.dropna(subset=["Getiri %", "İvme"])
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    def _q_renk(g: float, i: float) -> str:
        if g >= 0:
            return "#2f855a" if i >= 0 else "#b7791f"   # hızlanan / yavaşlayan lider
        return "#2b6cb0" if i >= 0 else "#c53030"       # dipten dönen / hızlanan düşüş

    renk = [_q_renk(g, i) for g, i in zip(df["Getiri %"], df["İvme"])]
    ax.scatter(df["Getiri %"], df["İvme"], s=42, c=renk,
               edgecolors="white", linewidths=0.8, zorder=3)
    ax.axhline(0, color="#8a8a98", linewidth=1)
    ax.axvline(0, color="#8a8a98", linewidth=1)
    uzak = (df["Getiri %"] ** 2 + df["İvme"] ** 2).sort_values(ascending=False)
    for tema in uzak.head(label_n).index:
        r = df.loc[tema]
        ax.annotate(clean(tema)[:22], (r["Getiri %"], r["İvme"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=6.4,
                    color="#14141a")
    ax.set_xlabel("Getiri % (donem)", fontsize=8, color="#4a4a58")
    ax.set_ylabel("Ivme (bu donem - onceki donem)", fontsize=8, color="#4a4a58")
    _style_axes(ax)
    return _fig_to_image(fig, 168, 92)


def holdings_chart(table: pd.DataFrame) -> Image | None:
    """ETF içi ayrışma haritası: akran farkı x kurumsal akış."""
    if table is None or table.empty:
        return None
    if "Akrana Göre" not in table.columns or "WHALE" not in table.columns:
        return None
    df = table.dropna(subset=["Akrana Göre"])
    if df.empty:
        return None
    renk_map = {"lider": "#2f855a", "lider_yorgun": "#b7791f",
                "uyumlu": "#8a8a98", "geride_akis_var": "#2b6cb0",
                "geride_akis_yok": "#c53030"}
    renk = [renk_map.get(k, "#8a8a98") for k in df.get("_durum", [])] \
        if "_durum" in df.columns else "#2b6cb0"
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    ax.scatter(df["Akrana Göre"], df["WHALE"], s=48, c=renk,
               edgecolors="white", linewidths=0.8, zorder=3)
    for _, r in df.iterrows():
        ax.annotate(clean(r["Sembol"]), (r["Akrana Göre"], r["WHALE"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=6.6,
                    color="#14141a")
    ax.axvline(0, color="#8a8a98", linewidth=1)
    ax.axhline(50, color="#b9b9c6", linewidth=0.9, linestyle=":")
    ax.set_xlabel("Akran medyanina gore fark (puan)", fontsize=8, color="#4a4a58")
    ax.set_ylabel("WHALE - kurumsal akis", fontsize=8, color="#4a4a58")
    _style_axes(ax)
    return _fig_to_image(fig, 168, 80)


# --------------------------------------------------------------------------
# Tablo yardımcısı
# --------------------------------------------------------------------------
def df_table(df: pd.DataFrame, columns: Sequence[str] | None = None,
             max_rows: int = 25, widths: Sequence[float] | None = None,
             formats: dict[str, str] | None = None,
             wrap: Iterable[str] = (), total_width: float = 168.0) -> Table | None:
    """DataFrame'i baskıya uygun tabloya çevirir."""
    if df is None or df.empty:
        return None
    cols = [c for c in (columns or df.columns) if c in df.columns]
    if not cols:
        return None
    st_ = _styles()
    formats = formats or {}
    wrap = set(wrap)

    head = [Paragraph(f"<b>{clean(c)}</b>", st_["cell"]) for c in cols]
    body: list[list[Any]] = [head]
    for _, row in df.head(max_rows).iterrows():
        line: list[Any] = []
        for c in cols:
            v = row[c]
            if c in formats and isinstance(v, (int, float, np.floating)):
                txt = _num(v, formats[c])
            elif isinstance(v, (float, np.floating)):
                txt = _num(v, ".2f")
            elif isinstance(v, (bool, np.bool_)):
                txt = "Evet" if v else "Hayir"
            else:
                txt = clean(v)
            line.append(Paragraph(txt, st_["cell"]) if c in wrap else txt)
        body.append(line)

    if widths:
        w = [x * mm for x in widths]
    else:
        w = [total_width / len(cols) * mm] * len(cols)

    t = Table(body, colWidths=w, repeatRows=1, hAlign="LEFT")
    style = [
        ("FONTNAME", (0, 0), (-1, -1), _FONT),
        ("FONTNAME", (0, 0), (-1, 0), _FONT_B),
        ("FONTSIZE", (0, 0), (-1, -1), 7.2),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("TEXTCOLOR", (0, 1), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BAND]),
    ]
    t.setStyle(TableStyle(style))
    return t


def kpi_row(items: Sequence[tuple[str, str, str]]) -> Table:
    """Üstteki özet kutuları: (etiket, değer, alt not)."""
    st_ = _styles()
    cells = []
    for label, value, sub in items:
        cells.append([
            Paragraph(f"<font size=7 color='#4a4a58'>{clean(label).upper()}</font>",
                      st_["cell"]),
            Paragraph(f"<font size=12><b>{clean(value)}</b></font>", st_["cell"]),
            Paragraph(f"<font size=6.6 color='#4a4a58'>{clean(sub)}</font>",
                      st_["cell"]),
        ])
    grid = [[c[0] for c in cells], [c[1] for c in cells], [c[2] for c in cells]]
    w = 168.0 / max(1, len(items))
    t = Table(grid, colWidths=[w * mm] * len(items), hAlign="LEFT")
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("BACKGROUND", (0, 0), (-1, -1), BAND),
        ("LINEBEFORE", (0, 0), (-1, -1), 2, ACCENT),
        ("BOX", (0, 0), (-1, -1), 0.4, LINE),
    ]))
    return t


# --------------------------------------------------------------------------
# Sayfa altlığı
# --------------------------------------------------------------------------
def _footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont(_FONT, 7)
    canvas.setFillColor(INK2)
    canvas.drawString(21 * mm, 12 * mm, clean(
        "AETHER APEX — otomatik uretilmis rapor. Yatirim tavsiyesi degildir."))
    canvas.drawRightString(A4[0] - 21 * mm, 12 * mm, f"{doc.page}")
    canvas.setStrokeColor(LINE)
    canvas.line(21 * mm, 15 * mm, A4[0] - 21 * mm, 15 * mm)
    canvas.restoreState()


# --------------------------------------------------------------------------
# Rapor
# --------------------------------------------------------------------------
def build_report(macro_state: Any, *,
                 battery_changes: pd.DataFrame | None = None,
                 score_changes: pd.DataFrame | None = None,
                 battery_history: pd.DataFrame | None = None,
                 regime_shifts: list[dict[str, Any]] | None = None,
                 drivers: Sequence[Any] = (),
                 theme_table: pd.DataFrame | None = None,
                 theme_period: str = "",
                 theme_summary: dict[str, Any] | None = None,
                 etf_table: pd.DataFrame | None = None,
                 holdings: dict[str, Any] | None = None,
                 swing: pd.DataFrame | None = None,
                 news: pd.DataFrame | None = None,
                 delta_period: str = "1 hafta",
                 baslik: str = "AETHER APEX") -> bytes:
    """Tüm bölümleri tek PDF'e derler ve baytları döner."""
    ensure_fonts()
    s = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=21 * mm, rightMargin=21 * mm,
        topMargin=18 * mm, bottomMargin=20 * mm,
        title=clean(baslik), author="AETHER APEX")

    M = macro_state
    story: list[Any] = []

    # ---------------- Kapak ----------------
    story.append(Paragraph(clean(baslik), s["title"]))
    story.append(Paragraph(
        clean(f"Piyasa rejimi ve sinyal raporu · {getattr(M, 'asof', '')} · "
              f"OPEX {getattr(M, 'opex_date', '-')} "
              f"({getattr(M, 'opex_days', 0)} gun)"
              + (" · UCLU CADI" if getattr(M, "opex_quad", False) else "")
              + (f" · FOMC {M.fomc_date} ({M.fomc_days} gun)"
                 if getattr(M, "fomc_days", None) is not None else "")),
        s["sub"]))
    story.append(HRFlowable(width="100%", color=ACCENT, thickness=1.6,
                            spaceAfter=10))

    trend = getattr(M, "scores", {}).get("trend", 50)
    story.append(kpi_row([
        ("Tespit edilen rejim", getattr(M, "regime", "-"),
         f"bilesik risk {getattr(M, 'risk_score', 0):.0f}/100"),
        ("VIX", _num(M.get("VIX") if hasattr(M, "get") else np.nan, ".1f"),
         (M.readings["VIX"].detail if getattr(M, "readings", {}).get("VIX")
          else "")),
        ("Rejim kapisi", "ACIK" if trend >= 50 else "KAPALI",
         "SPY 50 EMA ustunde" if trend >= 50 else "SPY 50 EMA altinda"),
    ]))
    story.append(Spacer(1, 8))
    if getattr(M, "regime_desc", ""):
        story.append(Paragraph(f"<b>{clean(M.regime)}</b> — "
                               f"{clean(M.regime_desc)}", s["body"]))

    # ---------------- Makro ----------------
    story.append(Paragraph("1. Makro gostergeler", s["h1"]))
    if getattr(M, "readings", None):
        rd = pd.DataFrame([{
            "Gosterge": r.label, "Deger": r.value,
            "Degisim %": r.change_pct, "Yorum": r.detail,
        } for r in M.readings.values()])
        t = df_table(rd, ["Gosterge", "Deger", "Degisim %", "Yorum"],
                     max_rows=30, widths=[34, 18, 20, 96],
                     formats={"Deger": ".2f", "Degisim %": "+.2f"},
                     wrap=["Yorum", "Gosterge"])
        if t:
            story.append(t)

    if score_changes is not None and not score_changes.empty:
        story.append(Paragraph("Rejimi ne itiyor, ne cekiyor", s["h2"]))
        story.append(Paragraph(clean(
            "Alt skorlarin donemsel degisimi. Bilesik risk skoru bunlarin "
            "ortalamasidir; hangi bilesenin rejimi tasidigi buradan okunur."),
            s["small"]))
        img = score_chart(score_changes, f"Δ {delta_period}")
        if img:
            story.append(img)
        t = df_table(score_changes,
                     ["Skor", "Şimdi", "Δ 1 gün", "Δ 1 hafta", "Δ 1 ay",
                      "Ne ölçüyor"],
                     max_rows=15, widths=[32, 15, 17, 19, 15, 70],
                     formats={"Şimdi": ".0f", "Δ 1 gün": "+.1f",
                              "Δ 1 hafta": "+.1f", "Δ 1 ay": "+.1f"},
                     wrap=["Ne ölçüyor", "Skor"])
        if t:
            story.append(Spacer(1, 4))
            story.append(t)

    # ---------------- Sermaye akışı ----------------
    story.append(PageBreak())
    story.append(Paragraph("2. Sermaye akis egilimi", s["h1"]))
    story.append(Paragraph(clean(
        "Elle yazilmis senaryo sabitleri degil; yukaridaki canli gostergelerden "
        "hesaplanir. Yanindaki fark, ayni formulun gecmis veriyle yeniden "
        "calistirilmasiyla bulunur — yani gecmis bugunun formuluyle tutarlidir."),
        s["small"]))

    deltas = None
    if battery_changes is not None and not battery_changes.empty:
        dcol = f"Δ {delta_period}"
        if dcol in battery_changes.columns:
            deltas = {r["Varlık Sınıfı"]: r[dcol]
                      for _, r in battery_changes.iterrows()}
    if getattr(M, "battery", None):
        story.append(battery_chart(M.battery, deltas, delta_period))

    if battery_changes is not None and not battery_changes.empty:
        t = df_table(battery_changes,
                     ["Varlık Sınıfı", "Şimdi", "1 gün önce", "1 hafta önce",
                      "1 ay önce", "Δ 1 gün", "Δ 1 hafta", "Δ 1 ay", "Besleyen"],
                     max_rows=12, widths=[24, 12, 14, 16, 13, 13, 15, 12, 49],
                     formats={"Şimdi": ".0f", "1 gün önce": ".0f",
                              "1 hafta önce": ".0f", "1 ay önce": ".0f",
                              "Δ 1 gün": "+.0f", "Δ 1 hafta": "+.0f",
                              "Δ 1 ay": "+.0f"},
                     wrap=["Besleyen", "Varlık Sınıfı"])
        if t:
            story.append(Spacer(1, 6))
            story.append(t)

    img = battery_history_chart(battery_history)
    if img:
        story.append(Spacer(1, 8))
        story.append(Paragraph("Son donem seyri", s["h2"]))
        story.append(img)

    if regime_shifts:
        sh = pd.DataFrame(regime_shifts[-8:])
        if "Tarih" in sh:
            sh["Tarih"] = pd.to_datetime(sh["Tarih"]).dt.strftime("%d.%m.%Y")
        story.append(Paragraph("Rejim degisim anlari", s["h2"]))
        story.append(Paragraph(clean(
            "Rejimin sik degismesi kararsiz piyasa demektir; trend takip "
            "sistemleri bu pencerelerde kotu calisir."), s["small"]))
        t = df_table(sh, ["Tarih", "Önceki", "Yeni", "Risk Skoru"],
                     max_rows=8, widths=[24, 56, 56, 32],
                     formats={"Risk Skoru": ".1f"}, wrap=["Önceki", "Yeni"])
        if t:
            story.append(t)

    # ---------------- Oyun kitabı ----------------
    if drivers:
        story.append(Paragraph("3. Bu rejimde ne calisir, ne calismaz", s["h1"]))
        for d in drivers:
            blok = [Paragraph(clean(f"{d.label}"), s["h2"]),
                    Paragraph(clean(d.nedir), s["body"]),
                    Paragraph("<b>Veriden nasil anlasilir:</b> "
                              + clean(d.veri_isareti), s["small"])]
            lehte = list(getattr(d, "lehte_etf", [])) + list(getattr(d, "lehte_hisse", []))
            aleyhte = list(getattr(d, "aleyhte_etf", [])) + list(getattr(d, "aleyhte_hisse", []))
            if lehte:
                blok.append(Paragraph("<b>Lehte:</b> " + clean(", ".join(lehte)),
                                      s["small"]))
            if aleyhte:
                blok.append(Paragraph("<b>Aleyhte:</b> " + clean(", ".join(aleyhte)),
                                      s["small"]))
            if getattr(d, "islem_notu", ""):
                blok.append(Paragraph("<b>Islem notu:</b> " + clean(d.islem_notu),
                                      s["small"]))
            story.append(KeepTogether(blok))

    # ---------------- Tema ----------------
    if theme_table is not None and not theme_table.empty:
        story.append(PageBreak())
        story.append(Paragraph(f"4. Tema takibi ({clean(theme_period)})", s["h1"]))
        story.append(Paragraph(clean(
            "Yatay eksen donemin getirisi, dikey eksen ivme (bu donem eksi "
            "onceki esdeger donem). Sag ust: hizlanan lider. Sag alt: yavaslayan "
            "lider. Sol ust: dipten donen. Sol alt: hizlanan dusus."), s["small"]))
        img = quadrant_chart(theme_table)
        if img:
            story.append(img)
        cols = [c for c in ["Getiri %", "Önceki %", "İvme", "Çeyrek", "Semboller"]
                if c in theme_table.columns]
        tt = theme_table.copy()
        tt.insert(0, "Tema", tt.index)
        t = df_table(tt, ["Tema"] + cols, max_rows=22,
                     widths=[36, 16, 17, 13, 34, 52],
                     formats={"Getiri %": "+.2f", "Önceki %": "+.2f",
                              "İvme": "+.2f"},
                     wrap=["Tema", "Çeyrek", "Semboller"])
        if t:
            story.append(Spacer(1, 6))
            story.append(t)

    # ---------------- ETF radarı ----------------
    if etf_table is not None and not etf_table.empty:
        story.append(PageBreak())
        story.append(Paragraph("5. ETF radari", s["h1"]))
        cols = [c for c in ["Sembol", "Sinyal", "Fiyat", "1 Gün %", "1 Hafta %",
                            "WHALE", "ΔWHALE", "OMNI", "MAGNITUDE", "DIRECTION"]
                if c in etf_table.columns]
        genis = {"Sembol": 16, "Sinyal": 27, "Fiyat": 15, "1 Gün %": 15,
                 "1 Hafta %": 16, "WHALE": 14, "ΔWHALE": 14, "OMNI": 13,
                 "MAGNITUDE": 21, "DIRECTION": 20}
        t = df_table(etf_table, cols, max_rows=30,
                     widths=[genis.get(c, 16) for c in cols],
                     formats={"Fiyat": ".2f", "1 Gün %": "+.2f",
                              "1 Hafta %": "+.2f", "WHALE": ".0f",
                              "ΔWHALE": "+.1f", "OMNI": ".0f",
                              "MAGNITUDE": ".0f", "DIRECTION": "+.0f"},
                     wrap=["Sinyal"])
        if t:
            story.append(t)

    # ---------------- ETF içi röntgen ----------------
    if holdings and holdings.get("table") is not None \
            and not holdings["table"].empty:
        HT = holdings["table"]
        sym = clean(holdings.get("etf", ""))
        story.append(PageBreak())
        story.append(Paragraph(f"6. {sym} ici rontgen", s["h1"]))
        for line in holdings.get("narrative", []):
            story.append(Paragraph(clean(line), s["body"]))
        img = holdings_chart(HT)
        if img:
            story.append(img)
        cols = [c for c in ["Sembol", "Durum", "Ağırlık %", "Getiri %",
                            "ETF'e Göre", "Akrana Göre", "Z", "Sinyal", "WHALE",
                            "ΔWHALE", "Neden"] if c in HT.columns]
        genis_h = {"Sembol": 13, "Durum": 20, "Ağırlık %": 11, "Getiri %": 12,
                   "ETF'e Göre": 12, "Akrana Göre": 13, "Z": 9, "Sinyal": 18,
                   "WHALE": 11, "ΔWHALE": 11, "Neden": 38}
        t = df_table(HT, cols, max_rows=40,
                     widths=[genis_h.get(c, 14) for c in cols],
                     formats={"Ağırlık %": ".2f", "Getiri %": "+.2f",
                              "ETF'e Göre": "+.2f", "Akrana Göre": "+.2f",
                              "Z": "+.2f", "WHALE": ".0f", "ΔWHALE": "+.1f"},
                     wrap=["Durum", "Sinyal", "Neden"])
        if t:
            story.append(Spacer(1, 6))
            story.append(t)

        story.append(Paragraph("Bilesenlerin teknik durumu", s["h2"]))
        for _, r in HT.iterrows():
            if not r.get("Teknik Not"):
                continue
            basl = clean(f"{r['Sembol']} — {r.get('Durum', '')} · "
                         f"{r.get('Sinyal', '')}")
            story.append(KeepTogether([
                Paragraph(f"<b>{basl}</b>", s["body"]),
                Paragraph(clean(r["Teknik Not"]), s["small"]),
                Paragraph(clean(r.get("Neden", "")), s["small"]),
            ]))

    # ---------------- Swing ----------------
    if swing is not None and not swing.empty:
        story.append(PageBreak())
        story.append(Paragraph("7. Swing adaylari", s["h1"]))
        cols = [c for c in ["Sembol", "Skor", "Sinyal", "Karakter", "Fiyat",
                            "Stop", "T1", "T2", "Risk %", "R (T1)", "R (T2)",
                            "Engel"] if c in swing.columns]
        t = df_table(swing, cols, max_rows=30,
                     formats={"Skor": ".0f", "Fiyat": ".2f", "Stop": ".2f",
                              "T1": ".2f", "T2": ".2f", "Risk %": ".1f",
                              "R (T1)": ".2f", "R (T2)": ".2f"},
                     wrap=["Sinyal", "Karakter", "Engel"])
        if t:
            story.append(t)

    # ---------------- Haberler ----------------
    if news is not None and not news.empty:
        story.append(Paragraph("8. Haber akisi", s["h1"]))
        cols = [c for c in ["Tarih", "Konu", "Başlık", "Rejim Sürücüsü",
                            "🟢 Lehte", "🔴 Aleyhte"] if c in news.columns]
        t = df_table(news, cols, max_rows=20,
                     widths=[18, 20, 66, 26, 19, 19], wrap=cols)
        if t:
            story.append(t)

    # ---------------- Yöntem ----------------
    story.append(Paragraph("Yontem notu", s["h1"]))
    story.append(Paragraph(clean(
        "Sinyaller TradingView betiklerinden (APEX CORE, APEX V665 OMNI, "
        "SAHANE V710/V719, QUANTUM V883) pandas'a birebir port edilmistir. "
        "Rejim etiketi ve batarya degerleri elle girilmis sabitler degil, "
        "canli piyasa verisinden hesaplanir. Gecmise donuk degerler ayni "
        "formulun kesilmis seriyle yeniden calistirilmasiyla uretilir; boylece "
        "formul degistiginde gecmis de tutarli kalir."), s["small"]))
    story.append(Paragraph(clean(
        "Bu rapor otomatik uretilmistir ve yatirim tavsiyesi degildir. "
        "Fiyat verisi Yahoo Finance kaynaklidir ve gecikmeli olabilir."),
        s["small"]))

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()


def report_filename(prefix: str = "aether_apex") -> str:
    return f"{prefix}_{dt.datetime.now():%Y%m%d_%H%M}.pdf"
