# apex/ui.py — AETHER APEX
from __future__ import annotations

from apex.engine import SIGNAL_COLORS

import streamlit as st


# Doğrulanmış kategorik palet (koyu zemin adımları)
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181",
          "#2f9e44", "#9085e9", "#e66767"]

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#a0a0ab", size=12,
              family='-apple-system, "Segoe UI", Roboto, Inter, sans-serif'),
    margin=dict(t=8, b=8, l=8, r=8),
    hoverlabel=dict(bgcolor="#131319", bordercolor="#2b2b36",
                    font=dict(color="#ececf1", size=12)),
)

_CSS = """
<style>
:root {
  --bg:#050506; --surface:#0d0d11; --surface-2:#131319;
  --line:#24242e; --line-soft:#1b1b22; --edge:#3a3a48;
  /* Kontrast: --ink-3 önceden #6e6e7a idi ve #050506 zemin üzerinde
     yaklaşık 3.4:1 kalıyordu — bölüm başlıkları ve açıklamalar okunmuyordu.
     Yeni değerler zemine karşı en az 7:1 (ink-2) ve 5.5:1 (ink-3). */
  --ink:#f2f2f6; --ink-2:#c2c2cc; --ink-3:#9a9aa8;
  --accent:#00e5ff; --pos:#2fbe86; --neg:#f0736f;
}
.stApp { background: var(--bg); color: var(--ink); }
.block-container { padding-top: 2.1rem; padding-bottom: 3rem; max-width: 1600px; }
.stApp, .stApp p, .stApp div, .stApp span,
[data-testid="stDataFrame"], [data-testid="stDataEditor"] {
  font-variant-numeric: tabular-nums; -webkit-font-smoothing: antialiased;
}

/* Başlık */
.nx-brand { display:flex; align-items:baseline; gap:.6rem; }
.nx-brand h1 { font-size:1.55rem; font-weight:700; letter-spacing:-.02em;
  margin:0; color:var(--ink); }
.nx-brand .tag { font-size:.62rem; font-weight:700; letter-spacing:.18em;
  text-transform:uppercase; color:var(--bg); background:var(--accent);
  padding:.18rem .45rem; border-radius:4px; }
.nx-meta { color:var(--ink-3); font-size:.8rem; margin-top:.35rem; }
.nx-meta b { color:var(--ink-2); font-weight:600; }

/* KPI kartları */
.kpi { background:linear-gradient(160deg,var(--surface-2) 0%,var(--surface) 100%);
  border:1px solid var(--line); border-radius:14px; padding:1rem 1.15rem 1.05rem;
  position:relative; overflow:hidden; height:100%; }
.kpi::before { content:""; position:absolute; inset:0 auto 0 0; width:3px;
  background:var(--accent); opacity:.85; }
.kpi.pos::before { background:var(--pos); } .kpi.neg::before { background:var(--neg); }
.kpi-label { font-size:.68rem; letter-spacing:.12em; text-transform:uppercase;
  color:var(--ink-2); font-weight:700; margin-bottom:.5rem; }
.kpi-value { font-size:1.45rem; font-weight:700; letter-spacing:-.02em;
  line-height:1.2; color:var(--ink); }
.kpi-sub { font-size:.78rem; color:var(--ink-2); margin-top:.4rem; }
.kpi-value.pos,.kpi-sub.pos { color:var(--pos); }
.kpi-value.neg,.kpi-sub.neg { color:var(--neg); }
.badge { display:inline-block; font-size:.72rem; font-weight:600;
  padding:.12rem .42rem; border-radius:5px; background:rgba(255,255,255,.05); }
.badge.pos { background:rgba(47,190,134,.13); color:var(--pos); }
.badge.neg { background:rgba(240,115,111,.13); color:var(--neg); }

/* Bölüm başlığı */
.nx-section { font-size:.74rem; letter-spacing:.13em; text-transform:uppercase;
  color:var(--ink-2); font-weight:700; margin:1.7rem 0 .75rem;
  padding-bottom:.45rem; border-bottom:1px solid var(--line);
  display:flex; align-items:center; gap:.5rem; }
.nx-section::before { content:""; width:3px; height:13px; border-radius:2px;
  background:var(--accent); display:inline-block; }

/* Sekmeler */
.stTabs [data-baseweb="tab-list"] { gap:.15rem;
  border-bottom:1px solid var(--line); flex-wrap:wrap; }
.stTabs [data-baseweb="tab"] { height:44px; padding:0 .95rem;
  background:transparent; color:var(--ink-2); font-size:.88rem;
  font-weight:600; border-radius:8px 8px 0 0; }
.stTabs [data-baseweb="tab"]:hover { color:var(--ink)!important; }
.stTabs [aria-selected="true"] { color:var(--ink)!important;
  background:var(--surface)!important; border-bottom:2px solid var(--accent)!important; }

/* Tablolar */
[data-testid="stDataFrame"], [data-testid="stDataEditor"] {
  border:1px solid var(--line); border-radius:12px; overflow:hidden; }

/* Butonlar — koyu zeminde görünür olsun (hover'a gerek kalmadan) */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button,
[data-testid="stBaseButton-secondary"],
[data-testid="stFileUploaderDropzone"] button {
  background:#1c1c25!important; color:var(--ink)!important;
  border:1px solid var(--edge)!important; border-radius:9px;
  font-weight:600; font-size:.86rem; transition:all .12s ease;
  box-shadow:0 1px 0 rgba(255,255,255,.04) inset; }
.stButton > button:hover, .stDownloadButton > button:hover,
.stFormSubmitButton > button:hover {
  background:#1b1b23!important; border-color:var(--accent)!important;
  color:var(--accent)!important; }
.stButton > button *, .stDownloadButton > button *,
.stFormSubmitButton > button * { color:inherit!important; }
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"],
[data-testid="stBaseButton-primary"] {
  background:var(--accent)!important; color:#04141a!important;
  border-color:var(--accent)!important; }
.stButton > button[kind="primary"] * { color:#04141a!important; }

/* Giriş alanları ve etiket çipleri */
.stTextInput input, .stNumberInput input, .stSelectbox > div > div,
.stMultiSelect > div > div {
  background:var(--surface)!important; border-color:var(--line)!important;
  border-radius:9px!important; }
[data-baseweb="tag"] { background-color:rgba(0,229,255,.13)!important;
  color:var(--accent)!important; border:1px solid rgba(0,229,255,.28)!important;
  border-radius:7px!important; }
[data-baseweb="tag"] span, [data-baseweb="tag"] svg { color:var(--accent)!important; }
div[role="radiogroup"] > label { background:var(--surface-2);
  border:1px solid var(--edge); border-radius:8px; padding:.3rem .7rem;
  margin-right:.35rem; }
div[role="radiogroup"] > label:hover { border-color:var(--accent); }
div[data-testid="stExpander"] { border:1px solid var(--line);
  border-radius:12px; background:var(--surface); }
div[data-testid="stExpander"] summary { color:var(--ink)!important; }
div[data-testid="stAlert"] { border-radius:11px; border:1px solid var(--line); }
a, a:visited { color:var(--accent); }

/* Streamlit'in soluk metinleri: caption, widget etiketi, yardım ikonu */
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
.stMarkdown small, small { color:var(--ink-2)!important; }
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label,
.stSlider label, .stRadio label p, .stCheckbox label p, .stToggle label p {
  color:var(--ink)!important; font-weight:600; }
[data-testid="stMarkdownContainer"] p { color:var(--ink); }
svg[data-testid="stTooltipHoverTarget"] { fill:var(--ink-2)!important; }
[data-testid="stMetricLabel"] { color:var(--ink-2)!important; }
[data-testid="stExpander"] summary p, [data-testid="stExpander"] summary span {
  color:var(--ink)!important; font-weight:600; }
[data-testid="stElementToolbar"] button { color:var(--ink)!important; }

/* Tablo başlıkları */
[data-testid="stDataFrame"] th, [data-testid="stDataEditor"] th {
  color:var(--ink)!important; font-weight:700!important; }

/* Delta rozetleri */
.delta { font-size:.72rem; font-weight:700; padding:.1rem .35rem;
  border-radius:5px; margin-left:.3rem; white-space:nowrap; }
.delta.up { background:rgba(47,190,134,.16); color:#4fd6a0; }
.delta.dn { background:rgba(240,115,111,.16); color:#ff8f8b; }
.delta.flat { background:rgba(255,255,255,.06); color:var(--ink-2); }
hr { border-color:var(--line-soft); }

/* ---- Açık tema emniyeti: Streamlit açık temaya düşse bile okunur kalsın ---- */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  background:var(--bg)!important; color:var(--ink)!important; }
[data-testid="stHeader"], header[data-testid="stHeader"] {
  background:var(--bg)!important; }
[data-testid="stHeader"] *, [data-testid="stToolbar"] * { color:var(--ink-2)!important; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp li,
.stApp label, .stApp strong, .stApp b, .stApp td, .stApp th,
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] * {
  color:inherit; }
[data-testid="stMarkdownContainer"] { color:var(--ink); }
/* metin girişleri */
.stTextInput input, .stNumberInput input, .stTextArea textarea,
.stDateInput input, input[type="text"], input[type="number"], textarea {
  color:var(--ink)!important; -webkit-text-fill-color:var(--ink)!important;
  caret-color:var(--accent)!important; background:var(--surface)!important; }
input::placeholder, textarea::placeholder { color:var(--ink-3)!important;
  -webkit-text-fill-color:var(--ink-3)!important; }
.stNumberInput button { background:var(--surface-2)!important;
  color:var(--ink)!important; border-color:var(--line)!important; }
/* seçim kutuları ve açılır menüler */
[data-baseweb="select"] > div { background:var(--surface)!important;
  border-color:var(--line)!important; }
[data-baseweb="select"] *, [data-baseweb="select"] input {
  color:var(--ink)!important; -webkit-text-fill-color:var(--ink)!important; }
[data-baseweb="select"] svg { fill:var(--ink-2)!important; }
[data-baseweb="popover"], [data-baseweb="popover"] > div,
[data-baseweb="menu"], [data-baseweb="popover"] ul,
div[role="listbox"], ul[role="listbox"] {
  background:var(--surface-2)!important; color:var(--ink)!important; }
li[role="option"], [data-baseweb="menu"] li { background:var(--surface-2)!important;
  color:var(--ink)!important; }
li[role="option"]:hover, li[role="option"][aria-selected="true"] {
  background:#1f2a33!important; color:var(--accent)!important; }
li[role="option"] * { color:inherit!important; }
/* radyo, onay kutusu, anahtar yazıları */
.stRadio label, .stRadio label *, .stCheckbox label *, .stToggle label *,
div[role="radiogroup"] label * { color:var(--ink)!important; }
/* uyarı kutuları */
[data-testid="stAlert"] p, [data-testid="stAlert"] li,
[data-testid="stAlert"] span { color:var(--ink)!important; }
/* açıklama baloncukları */
[data-baseweb="tooltip"], [data-baseweb="tooltip"] * {
  background:var(--surface-2)!important; color:var(--ink)!important; }
/* kod ve satır içi kod */
code { color:var(--accent)!important; background:rgba(0,229,255,.08)!important; }
/* kaydırıcı */
.stSlider [data-baseweb="slider"] div { color:var(--ink)!important; }
[data-testid="stTickBar"] *, [data-testid="stSliderThumbValue"] {
  color:var(--ink-2)!important; }
/* dosya yükleme alanı */
[data-testid="stFileUploaderDropzone"] { background:var(--surface)!important;
  color:var(--ink)!important; }
</style>
"""


def inject_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f"<div class='nx-section'>{title}</div>", unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = "", tone: str = "") -> str:
    cls = f" {tone}" if tone else ""
    return (f"<div class='kpi{cls}'><div class='kpi-label'>{label}</div>"
            f"<div class='kpi-value{cls}'>{value}</div>"
            f"<div class='kpi-sub{cls}'>{sub}</div></div>")


def badge(text: str, tone: str = "") -> str:
    return f"<span class='badge {tone}'>{text}</span>"


def signal_style(val) -> str:
    """Sinyal hücresini scriptlerdeki renk diliyle boyar."""
    if not isinstance(val, str):
        return ""
    for key, (bg, fg) in SIGNAL_COLORS.items():
        if val == key:
            return f"background-color:{bg};color:{fg};font-weight:700"
    if val == "⚫ VERİ YOK":
        return "background-color:#101014;color:#5a5a63"
    return "background-color:#15151c;color:#8a8a95"
