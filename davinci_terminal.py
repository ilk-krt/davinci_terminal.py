"""
AETHER APEX — makro, tema ve sinyal tarayıcı (TEK DOSYA SÜRÜMÜ)

Bu dosya build_single_file.py tarafından üretilmiştir; elle düzenlemeyin.
Kaynak: apex/*.py + app.py

Çalıştırma:  streamlit run apex_app.py
"""

from __future__ import annotations


# ==========================================================================
# KAYNAK: apex/indicators.py
# ==========================================================================


import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Hareketli ortalamalar
# --------------------------------------------------------------------------
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder yumuşatması — ta.rma. RSI ve ATR bunu kullanır."""
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def wma(s: pd.Series, n: int) -> pd.Series:
    """
    ta.wma — en yeni bara n ağırlığı verir.
    rolling().apply() yerine konvolüsyon: yüzlerce sembol taranırken
    tarama süresini kat kat kısaltıyor, sonuç birebir aynı.
    """
    w = np.arange(1, n + 1, dtype=float)
    w /= w.sum()
    x = s.to_numpy(dtype=float)
    if len(x) < n:
        return pd.Series(np.full(len(x), np.nan), index=s.index)
    conv = np.convolve(np.nan_to_num(x), w[::-1], mode="valid")
    out = np.full(len(x), np.nan)
    out[n - 1:] = conv
    # NaN içeren pencereler NaN kalmalı (nan_to_num kirletmesin)
    nan_win = pd.Series(np.isnan(x)).rolling(n).max().to_numpy()
    out[nan_win == 1] = np.nan
    return pd.Series(out, index=s.index)


def vwma(src: pd.Series, vol: pd.Series, n: int) -> pd.Series:
    return sma(src * vol, n) / sma(vol, n).replace(0, np.nan)


def stdev(s: pd.Series, n: int) -> pd.Series:
    """ta.stdev — popülasyon (ddof=0)."""
    return s.rolling(n).std(ddof=0)


# --------------------------------------------------------------------------
# Temel dönüşümler
# --------------------------------------------------------------------------
def highest(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).max()


def lowest(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).min()


def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))


def roc(s: pd.Series, n: int) -> pd.Series:
    prev = s.shift(n)
    return (s - prev) / prev.replace(0, np.nan) * 100.0


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    pc = close.shift(1)
    return pd.concat([high - low, (high - pc).abs(), (low - pc).abs()],
                     axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    return rma(true_range(high, low, close), n)


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    """
    ta.rsi. Kesintisiz yükselişte düşüş ortalaması 0 olur; bölme NaN vermemeli,
    Pine'da olduğu gibi 100 dönmelidir (tersi 0). Bu ayrım önemli: NaN dönerse
    bütün momentum zinciri (OMNI konsensüsü, konfluans) sessizce boşa düşer.
    """
    d = s.diff()
    up = rma(d.clip(lower=0), n)
    dn = rma((-d).clip(lower=0), n)
    out = 100.0 - 100.0 / (1.0 + up / dn.replace(0, np.nan))
    out = out.mask((dn == 0) & (up > 0), 100.0)
    out = out.mask((up == 0) & (dn > 0), 0.0)
    out = out.mask((up == 0) & (dn == 0), 50.0)
    return out


def stoch(src: pd.Series, high: pd.Series, low: pd.Series, n: int) -> pd.Series:
    ll, hh = lowest(low, n), highest(high, n)
    return 100.0 * (src - ll) / (hh - ll).replace(0, np.nan)


def mfi(src: pd.Series, vol: pd.Series, n: int = 14) -> pd.Series:
    d = src.diff()
    up = (vol * src.where(d > 0, 0.0)).rolling(n).sum()
    dn = (vol * src.where(d < 0, 0.0)).rolling(n).sum()
    return 100.0 - 100.0 / (1.0 + up / dn.replace(0, np.nan))


def cci(src: pd.Series, n: int = 20) -> pd.Series:
    m = sma(src, n)
    mad = _rolling_mad(src, n)
    return (src - m) / (0.015 * mad.replace(0, np.nan))


def _rolling_mad(s: pd.Series, n: int) -> pd.Series:
    """Ortalamadan mutlak sapmanın ortalaması — CCI için, vektörleştirilmiş."""
    x = s.to_numpy(dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return pd.Series(out, index=s.index)
    win = np.lib.stride_tricks.sliding_window_view(x, n)
    with np.errstate(invalid="ignore"):
        res = np.abs(win - win.mean(axis=1, keepdims=True)).mean(axis=1)
    out[n - 1:] = res
    return pd.Series(out, index=s.index)


def tsi(close: pd.Series, long_n: int = 25, short_n: int = 13) -> pd.Series:
    pc = close.diff()
    num = ema(ema(pc, long_n), short_n)
    den = ema(ema(pc.abs(), long_n), short_n)
    return 100.0 * num / den.clip(lower=0.001)


def percentrank(s: pd.Series, n: int) -> pd.Series:
    """
    ta.percentrank — önceki n değerin yüzde kaçı mevcut değerden küçük/eşit.
    Kayan pencere görünümüyle vektörleştirildi (rolling.apply yerine).
    """
    x = s.to_numpy(dtype=float)
    m = len(x)
    out = np.full(m, np.nan)
    if m < n + 1:
        return pd.Series(out, index=s.index)
    win = np.lib.stride_tricks.sliding_window_view(x, n + 1)
    prev, cur = win[:, :-1], win[:, -1:]
    with np.errstate(invalid="ignore"):
        cnt = (prev <= cur).sum(axis=1).astype(float)
        bad = np.isnan(win).any(axis=1)
    res = 100.0 * cnt / n
    res[bad] = np.nan
    out[n:] = res
    return pd.Series(out, index=s.index)


def percentile_lin(s: pd.Series, n: int, p: float) -> pd.Series:
    return s.rolling(n).quantile(p / 100.0, interpolation="linear")


def barssince(cond: pd.Series) -> pd.Series:
    idx = np.arange(len(cond), dtype=float)
    last = pd.Series(np.where(cond.to_numpy(), idx, np.nan),
                     index=cond.index).ffill()
    return pd.Series(idx, index=cond.index) - last


def leaky_reservoir(q: pd.Series, alpha: float,
                    negative_leak: float = 1.30) -> pd.Series:
    """
    APEX CORE'un sızdıran haznesi: ch = ch*(1-alpha) + q, negatif akış
    `negative_leak` katıyla hızlı boşalır (satışlar alımlardan hızlı drene eder).
    """
    out = np.empty(len(q), dtype=float)
    acc = 0.0
    vals = q.to_numpy(dtype=float)
    for i, v in enumerate(vals):
        if not np.isfinite(v):
            v = 0.0
        acc = acc * (1.0 - alpha) + (v if v >= 0 else v * negative_leak)
        out[i] = acc
    return pd.Series(out, index=q.index)


def ratcheting_atr_stop(low: pd.Series, atr14: pd.Series, mult: float,
                        entry_price: float | None = None,
                        hard_stop_pct: float = 20.0) -> pd.Series:
    """
    V719'un iz süren zırhı: stop = low - mult*ATR, SADECE yukarı kayar.
    Girişten `hard_stop_pct` kadar aşağıda sert bir taban vardır.
    """
    calc = (low - atr14 * mult).to_numpy(dtype=float)
    out = np.empty(len(calc), dtype=float)
    prev = np.nan
    floor = (entry_price * (1 - hard_stop_pct / 100.0)
             if entry_price else -np.inf)
    for i, c in enumerate(calc):
        if not np.isfinite(c):
            out[i] = prev
            continue
        prev = c if not np.isfinite(prev) else max(prev, c)
        out[i] = max(prev, floor)
    return pd.Series(out, index=low.index)


def normal_cdf(z: np.ndarray) -> np.ndarray:
    """Abramowitz-Stegun yaklaşımı — V719'un hibrit delta motoru için."""
    z = np.clip(z, -8.0, 8.0)
    t = 1.0 / (1.0 + 0.2316419 * np.abs(z))
    d = 0.3989422804014327 * np.exp(-z * z / 2.0)
    p = d * t * (0.319381530 + t * (-0.356563782 + t *
                 (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    return np.where(z >= 0, 1.0 - p, p)


def f_tanh(x):
    return np.tanh(np.clip(2.0 * np.asarray(x, dtype=float), -60.0, 60.0) / 2.0)


def f_contrast(x, gamma: float):
    x = np.asarray(x, dtype=float)
    return 50.0 * (1.0 + np.tanh((x - 50.0) / 50.0 * gamma) / np.tanh(gamma))


def safe_last(s, default=np.nan) -> float:
    """Serinin son geçerli değeri — kısa geçmişte patlamasın."""
    try:
        v = pd.Series(s).dropna()
        return float(v.iloc[-1]) if len(v) else float(default)
    except Exception:
        return float(default)


def safe_bool(s, default: bool = False) -> bool:
    try:
        v = pd.Series(s).dropna()
        return bool(v.iloc[-1]) if len(v) else default
    except Exception:
        return default

# ==========================================================================
# KAYNAK: apex/universe.py
# ==========================================================================


from typing import Any

# --------------------------------------------------------------------------
# ANA SEKTÖR ETF'LERİ
# --------------------------------------------------------------------------
MAIN_SECTORS: dict[str, str] = {
    "XLK": "Teknoloji", "XLI": "Sanayi", "XLE": "Enerji", "XLV": "Sağlık",
    "XLF": "Finans", "XLY": "Tüketim (Döngüsel)", "XLP": "Tüketim (Defansif)",
    "XLB": "Materyal", "XLC": "İletişim", "XLRE": "Gayrimenkul",
    "XLU": "Kamu Hizmetleri", "SPY": "S&P 500", "QQQ": "Nasdaq 100",
    "IWM": "Russell 2000",
}

# --------------------------------------------------------------------------
# ETF İÇERİKLERİ
#   agirlik : portföy ağırlığı (%) — bilinenler
#   rol     : şirketin temadaki rolü
#   gruplar : alt kırılım (ağırlık bilinmeyen ETF'ler için)
# --------------------------------------------------------------------------
ETF: dict[str, dict[str, Any]] = {}


def _add(sym: str, name: str, kategori: str, aciklama: str = "", *,
         agirlik: dict[str, float] | None = None,
         rol: dict[str, str] | None = None,
         gruplar: dict[str, list[str]] | None = None) -> None:
    ETF[sym] = {"name": name, "kategori": kategori, "aciklama": aciklama,
                "agirlik": agirlik or {}, "rol": rol or {},
                "gruplar": gruplar or {}}


# ========================= 1. TEKNOLOJİ & YAPAY ZEKÂ ======================
_add("XLK", "Technology Select Sector", "Teknoloji & YZ",
     "Donanım devleri, yazılım ekosistemleri ve yarı iletken liderleri.",
     agirlik={"NVDA": 15.2, "AAPL": 11.9, "MSFT": 8.3, "MU": 5.75, "AVGO": 5.45,
              "AMD": 3.9, "INTC": 3.6, "CSCO": 2.85, "PLTR": 2.5, "AMAT": 2.35},
     rol={"NVDA": "Yapay zekâ ve GPU", "AAPL": "Tüketici elektroniği",
          "MSFT": "Bulut ve kurumsal yazılım", "MU": "Bellek (DRAM/NAND)",
          "AVGO": "Altyapı ve kablosuz haberleşme çipleri",
          "AMD": "İşlemci ve GPU tasarımı", "INTC": "Yarı iletken üretimi",
          "CSCO": "Ağ donanımı ve telekom", "PLTR": "YZ veri analitiği",
          "AMAT": "Çip üretim ekipmanları"},
     gruplar={"Yazılım & SaaS": ["MSFT", "CRM", "ADBE", "ORCL", "NOW", "INTU"],
              "Donanım & Altyapı": ["AAPL", "CSCO", "IBM", "HPE"],
              "Yarı İletken Tasarım": ["NVDA", "AVGO", "AMD", "QCOM", "INTC"]})

_add("IGV", "iShares Expanded Tech-Software", "Teknoloji & YZ",
     "Donanım içermez; bulut altyapısı, kurumsal yazılım ve güvenlik yazılımı.",
     gruplar={"Kurumsal Yazılım": ["MSFT", "CRM", "ORCL", "ADBE"],
              "İş Süreçleri & İK": ["NOW", "INTU", "WDAY", "PLTR", "PAYC"],
              "Veri ve Bulut": ["SNOW", "DDOG", "DT", "TEAM"],
              "Yazılım Tabanlı Güvenlik": ["PANW", "CRWD", "NET"]})

_add("CLOU", "Global X Cloud Computing", "Teknoloji & YZ",
     "Saf bulut altyapısı, gözlemlenebilirlik ve SaaS.",
     agirlik={"DOCN": 6.05, "DDOG": 5.75, "AKAM": 5.55, "TWLO": 4.95, "ZS": 4.4,
              "SNOW": 4.2, "PAYC": 4.0, "ZM": 3.9, "NOW": 3.8, "NET": 3.6},
     rol={"DOCN": "Geliştirici bulut altyapısı (IaaS)",
          "DDOG": "Bulut uygulama izleme (observability)",
          "AKAM": "İçerik dağıtım ağı ve güvenlik",
          "TWLO": "Bulut iletişim platformu (CPaaS)",
          "ZS": "Zero Trust kurumsal güvenlik",
          "SNOW": "Bulut veri deposu ve analitik",
          "PAYC": "Bulut İK ve bordro", "ZM": "Video konferans",
          "NOW": "Dijital iş akışı yönetimi",
          "NET": "Web altyapısı, güvenlik ve CDN"})

_add("CIBR", "First Trust Nasdaq Cybersecurity", "Teknoloji & YZ",
     "Ağ güvenliği, uç nokta koruması, bulut güvenliği ve siber danışmanlık.",
     gruplar={"Yeni Nesil Bulut & Uç Nokta": ["CRWD", "PANW", "ZS"],
              "Ağ Güvenliği (Firewall)": ["FTNT", "CHKP", "CSCO", "JNPR"],
              "Kimlik & Tehdit Analizi": ["OKTA", "CYBR", "TENB", "QLYS"],
              "Tüketici & Dijital Altyapı": ["GEN", "NET", "AKAM"]})

_add("BOTZ", "Global X Robotics & AI", "Teknoloji & YZ",
     "YZ algoritmalarından endüstriyel robot kollarına ve cerrahi robota.",
     gruplar={"YZ İşlemcileri": ["NVDA"],
              "Tıbbi Robotik": ["ISRG"],
              "Fabrika Otomasyonu (Japonya)": ["KEYS", "FANUY", "YASKY", "OMRNY"],
              "YZ Yazılımı ve Görü": ["PATH", "AI", "CGNX"],
              "Ağır Otomasyon": ["ROK"]})

_add("AIQ", "Global X Artificial Intelligence & Technology", "Teknoloji & YZ",
     "Bellek ve çip tarafı ağırlıklı, küresel YZ ekosistemi.",
     agirlik={"MU": 4.8, "INTC": 4.45, "AMD": 4.4, "CSCO": 4.0, "AVGO": 3.4,
              "NVDA": 3.2, "TSM": 3.2, "GOOGL": 3.05, "AAPL": 3.0},
     rol={"MU": "YZ sunucuları için DRAM/NAND",
          "INTC": "Veri merkezi ve AI PC işlemcileri",
          "AMD": "MI300 serisi YZ hızlandırıcıları",
          "CSCO": "YZ veri merkezi ağ altyapısı",
          "AVGO": "Özel YZ ASIC çipleri ve hızlı bağlantı",
          "NVDA": "GPU ve CUDA platformunun lideri",
          "TSM": "En gelişmiş YZ çiplerini üreten dökümhane",
          "GOOGL": "LLM ve bulut ekosistemi",
          "AAPL": "Cihaz üstü YZ entegrasyonu"})
ETF["AIQ"]["aciklama"] += (" SK Hynix (000660.KS) ve Samsung (005930.KS) fonun "
                           "ilk sıralarında ama ABD hattında işlem görmediği için "
                           "taramaya dahil edilmedi.")

# ========================= 2. YARI İLETKENLER =============================
_add("SOXX", "iShares Semiconductor", "Yarı İletken",
     "ABD menşeili çip tasarımcıları ve üretim ekipmanı sağlayıcıları.",
     agirlik={"MU": 10.2, "AMD": 8.95, "INTC": 7.1, "AVGO": 7.0, "NVDA": 6.75,
              "MRVL": 5.65, "AMAT": 4.65, "QCOM": 3.95, "MPWR": 3.8, "TXN": 3.75},
     rol={"MU": "Bellek ve veri depolama çipleri",
          "AMD": "CPU ve GPU tasarımı", "INTC": "Entegre cihaz üreticisi (IDM)",
          "AVGO": "Ağ ve altyapı çipleri", "NVDA": "YZ ve grafik işlemcileri",
          "MRVL": "Veri altyapısı ve bulut çipleri",
          "AMAT": "Çip üretim ekipmanları", "QCOM": "Mobil işlemci ve 5G",
          "MPWR": "Güç yönetimi çözümleri", "TXN": "Analog ve gömülü işlemciler"})

_add("SMH", "VanEck Semiconductor", "Yarı İletken",
     "SOXX'tan farkı: TSMC ve ASML çok yüksek ağırlıkta.",
     gruplar={"Dökümhaneler": ["TSM", "INTC"],
              "Litografi": ["ASML"],
              "GPU / YZ": ["NVDA", "AMD"],
              "Ağ ve Veri Merkezi": ["AVGO", "MRVL", "QCOM"],
              "Üretim Ekipmanı": ["AMAT", "LRCX", "KLAC"]})

_add("EUV", "Lithography & Semiconductor Photonics", "Yarı İletken",
     "Litografi, optik ve çip üretim metrolojisi — YZ altyapısının darboğazı.",
     agirlik={"TSM": 9.6, "ASML": 8.0, "GLW": 5.2, "LRCX": 5.0, "AMAT": 4.8,
              "LITE": 4.3, "CIEN": 4.3, "KLAC": 4.1, "COHR": 4.1, "MTSI": 3.3},
     rol={"TSM": "3nm ve altı çipleri üreten dev dökümhane",
          "ASML": "EUV litografi makinelerinin tek üreticisi",
          "GLW": "Veri merkezi camı ve fiber optik altyapı",
          "LRCX": "Gofret işleme ve çip üretim ekipmanı",
          "AMAT": "Yarı iletken malzeme mühendisliği",
          "LITE": "Optik veri iletimi ve YZ ağları için lazer",
          "CIEN": "Yüksek hızlı veri merkezi optik ağ mimarisi",
          "KLAC": "Optik denetim ve metroloji sistemleri",
          "COHR": "Endüstriyel lazer ve optik bileşenler",
          "MTSI": "Yüksek hızlı veri iletimi bileşenleri"})

_add("PHOTON", "Fotonik & Optik (özel sepet)", "Yarı İletken",
     "Kullanıcı tanımlı fotonik sepeti.",
     gruplar={"Fotonik": ["AAOI", "COHR", "LITE", "POET", "AXTI", "IQE", "LRCX"]})

_add("QTUM", "Defiance Quantum", "Yarı İletken",
     "Kuantum bilişim ve yüksek performanslı hesaplama.",
     gruplar={"Saf Kuantum": ["IONQ", "RGTI", "QUBT", "QBTS"],
              "Kurumsal Ar-Ge": ["IBM", "GOOGL", "HON", "NVDA"]})

# ========================= 3. ENERJİ, EMTİA, MADENCİLİK ===================
_add("XLE", "Energy Select Sector", "Enerji & Emtia",
     "Dev entegre petrol, gaz ve rafine ürün şirketleri.",
     gruplar={"Entegre Devler": ["XOM", "CVX"],
              "Arama ve Üretim (E&P)": ["COP", "EOG", "OXY", "DVN"],
              "Petrol Sahası Hizmetleri": ["SLB", "BKR", "HAL"],
              "Rafineri ve Dağıtım": ["MPC", "VLO", "PSX"],
              "Boru Hattı (Midstream)": ["WMB", "OKE", "KMI"]})

_add("XOP", "SPDR Oil & Gas Exploration", "Enerji & Emtia",
     "Eşit ağırlığa yakın; petrol fiyatına en duyarlı upstream şirketleri.",
     gruplar={"Bağımsız Upstream": ["FANG", "CTRA", "EQT", "APA", "AR", "CHK",
                                    "RRC", "MTDR"],
              "Büyük Entegre": ["COP", "XOM", "CVX", "OXY"]})

_add("OIH", "VanEck Oil Services", "Enerji & Emtia",
     "Kuyu açan, sismik analiz yapan, platform kuran teknoloji sağlayıcıları.",
     gruplar={"Büyük Üçlü": ["SLB", "HAL", "BKR"],
              "Açık Deniz Sondaj": ["RIG", "NE", "VAL", "SDRL"],
              "Karada Sondaj": ["HP", "PTEN", "NBR"],
              "Ekipman ve Kuyu Teknolojisi": ["NOV", "CHX", "WHD", "TDW"]})

_add("COPX", "Global X Copper Miners", "Enerji & Emtia",
     "Elektrifikasyon, YZ veri merkezleri ve yeşil dönüşümün ana hammaddesi.",
     gruplar={"Saf Bakır Madenleri": ["FCX", "SCCO", "IVPAF", "ANFGY", "LUNMF",
                                      "FQVLF"],
              "Çeşitlendirilmiş Devler": ["BHP", "RIO", "TECK", "GLNCY", "VALE"]})

_add("LIT", "Global X Lithium & Battery Tech", "Enerji & Emtia",
     "Maden çıkarmadan batarya hücresine ve elektrikli araca uzanan zincir.",
     gruplar={"Lityum Madenciliği": ["ALB", "SQM", "ALTM"],
              "Batarya Hücresi": ["PCRFY", "TTDKY"],
              "Elektrikli Araç": ["TSLA", "RIVN", "LCID"]})

_add("URA", "Global X Uranium", "Enerji & Emtia",
     "YZ veri merkezlerinin baz yük ihtiyacıyla canlanan nükleer döngü.",
     gruplar={"Uranyum Üreticileri": ["CCJ", "NXE", "UEC", "UUUU", "DNN"],
              "Nükleer Teknoloji ve SMR": ["BWXT", "LEU", "SMR", "CEG"]})

_add("REMX", "VanEck Rare Earth & Strategic Metals", "Enerji & Emtia",
     "Mıknatıs, savunma jetleri, rüzgâr türbini için kritik elementler.",
     gruplar={"Batı Üreticileri": ["MP", "LYSDY"],
              "Stratejik Metaller": ["ALB", "ALTM"]})

_add("GDX", "VanEck Gold Miners", "Enerji & Emtia",
     "Altın fiyatını kaldıraçlı yansıtan üreticiler.",
     gruplar={"Küresel Devler": ["NEM", "GOLD", "AEM", "GFI", "AU", "KGC"],
              "Royalty / Streaming": ["WPM", "FNV", "RGLD"],
              "Bölgesel Üreticiler": ["EGO", "BTG", "HMY", "SBSW"]})

_add("XME", "SPDR Metals & Mining", "Enerji & Emtia",
     "ABD yerleşik demir, çelik, alüminyum ve kömür üreticileri.",
     gruplar={"Demir & Çelik": ["NUE", "STLD", "CLF", "X", "RS"],
              "Alüminyum & Bakır": ["AA", "KALU", "FCX"],
              "Kömür": ["AMR", "HCC", "ARCH"],
              "Değerli Metaller": ["HL", "RGLD"]})

_add("ICLN", "iShares Global Clean Energy", "Enerji & Emtia",
     "Güneş, rüzgâr, hidroelektrik ve hidrojen odaklı küresel ekosistem.",
     gruplar={"Güneş": ["ENPH", "FSLR", "SEDG"],
              "Rüzgâr": ["VWDRY", "ORSTY"],
              "Yenilenebilir Üretim": ["IBDRY"],
              "Hidrojen ve Yakıt Hücresi": ["PLUG", "BE"]})

_add("TAN", "Invesco Solar", "Enerji & Emtia", "Saf güneş enerjisi zinciri.",
     gruplar={"Panel ve İnverter": ["FSLR", "ENPH", "SEDG", "RUN", "NXT"]})

# ========================= 4. ALTYAPI, TAŞIMA, SAVUNMA ====================
_add("XLU", "Utilities Select Sector", "Altyapı & Savunma",
     "Elektrik şebekeleri, regüle gaz hatları ve baz yük üreten holdingler.",
     agirlik={"NEE": 14.15, "SO": 7.35, "DUK": 6.9, "CEG": 6.45, "AEP": 5.05,
              "SRE": 4.25, "D": 3.8, "ETR": 3.65, "VST": 3.45, "XEL": 3.35},
     rol={"NEE": "Dünyanın en büyük rüzgâr ve güneş üreticisi",
          "SO": "Güneydoğu ABD'nin dev regüle elektrik ve gaz şebekesi",
          "DUK": "Geniş ölçekli elektrik altyapısı",
          "CEG": "ABD'nin en büyük nükleer üreticisi (YZ/veri merkezi odaklı)",
          "AEP": "Dev elektrik iletim şebekesi",
          "SRE": "Enerji altyapısı, doğalgaz dağıtımı ve LNG",
          "D": "Veri merkezlerinin kalbi Virginia'nın ana sağlayıcısı",
          "ETR": "Körfez bölgesi nükleer ve temiz elektrik",
          "VST": "Nükleer ve bağımsız üretim, veri merkezi partneri",
          "XEL": "Yenilenebilir entegrasyonlu regüle şebeke"})

_add("XLI", "Industrial Select Sector", "Altyapı & Savunma",
     "Havacılık, lojistik, ağır makine ve savunma devleri.",
     gruplar={"Ağır İş Makinası": ["CAT", "DE"],
              "Konglomera": ["GE", "HON", "MMM", "EMR"],
              "Savunma ve Havacılık": ["LMT", "RTX", "BA"],
              "Demiryolu": ["UNP", "NSC", "CSX"],
              "Kargo ve Lojistik": ["UPS", "FDX"]})

_add("PAVE", "Global X US Infrastructure Development", "Altyapı & Savunma",
     "Şebeke yenileme, fabrika kurulumu ve elektrifikasyondan beslenenler.",
     gruplar={"Veri Merkezi ve Şebeke": ["ETN", "PH", "HUBB", "POWL"],
              "İklimlendirme": ["TT", "CARR", "JCI"],
              "Kiralama ve Tedarik": ["URI", "FAST", "GWW"],
              "Çimento ve Agrega": ["VMC", "MLM", "EXP"],
              "Mühendislik ve İnşaat": ["J", "ACM", "PWR", "EME"]})

_add("IYT", "iShares Transportation Average", "Altyapı & Savunma",
     "Demiryolları, kargo, havayolları ve yolculuk paylaşım platformları.",
     gruplar={"Demiryolu": ["UNP", "CSX", "NSC"],
              "Kargo ve Dağıtım": ["UPS", "FDX", "EXPD", "JBHT", "ODFL"],
              "Yeni Nesil Mobilite": ["UBER", "LYFT"],
              "Havayolları": ["DAL", "UAL", "LUV"]})

_add("JETS", "US Global Jets", "Altyapı & Savunma",
     "Havayolları, uçak üreticileri ve havalimanı işletmecileri.",
     gruplar={"ABD Bayrak Taşıyıcı": ["DAL", "UAL", "AAL"],
              "Düşük Maliyetli": ["LUV", "JBLU", "ALK", "ALGT", "ULCC", "SKYW"],
              "Uçak Üreticileri": ["BA", "ERJ", "EADSY"]})

_add("XAR", "SPDR Aerospace & Defense", "Altyapı & Savunma",
     "Eşit ağırlıklı; büyük yükleniciler + jet motoru ve alt sistem üreticileri.",
     gruplar={"Ana Yükleniciler": ["LMT", "RTX", "NOC", "GD", "LHX"],
              "Jet Motoru ve Yapısal": ["TDG", "HWM", "HEI", "SPR", "CW", "TXT"],
              "Alt Sistem": ["BWXT", "HII", "PSN"]})

_add("ARKX", "ARK Space Exploration & Innovation", "Altyapı & Savunma",
     "Uzay, savunma ve otonom sistem inovasyonu.",
     agirlik={"RKLB": 10.05, "AMD": 7.3, "LHX": 7.1, "TER": 6.3, "DE": 5.4,
              "KTOS": 5.45, "AVAV": 4.0, "AMZN": 4.0, "ACHR": 3.9, "GOOG": 3.8},
     rol={"RKLB": "Küçük fırlatma aracı ve uydu üretimi",
          "AMD": "Uzay/savunma sistemleri için işlemci",
          "LHX": "Savunma haberleşme ve uzay sistemleri",
          "TER": "Test ve otomasyon ekipmanı", "DE": "Otonom tarım makineleri",
          "KTOS": "İnsansız savunma sistemleri", "AVAV": "Taktik İHA",
          "AMZN": "Kuiper uydu takımyıldızı ve bulut",
          "ACHR": "eVTOL hava taksisi", "GOOG": "Uzay verisi ve YZ altyapısı"})

_add("UFO", "Procure Space", "Altyapı & Savunma",
     "Uydu haberleşmesi, roket fırlatma ve küresel konumlandırma.",
     gruplar={"Uydu Haberleşme": ["SIRI", "IRDM", "SATS", "VSAT"],
              "Konumlandırma": ["GRMN"],
              "Uzay Savunma": ["LMT", "BA", "NOC", "LHX"],
              "Yeni Nesil Uzay": ["RKLB", "SPCE"]})

_add("SPACE_RACE", "SpaceX & Yeni Uzay (özel sepet)", "Altyapı & Savunma",
     "Kullanıcı tanımlı yeni nesil uzay sepeti.",
     gruplar={"Fırlatma ve Uydu": ["RKLB", "ASTS", "LUNR", "SATS", "PL", "SPIR",
                                   "BKSY", "SIDU"]})

# ========================= 5. FİNANS & FİNTEK =============================
_add("XLF", "Financial Select Sector", "Finans",
     "Yatırım bankaları, sigorta, kredi kartı ağları ve ticari bankalar.",
     gruplar={"Ticari Bankalar": ["JPM", "BAC", "WFC", "C"],
              "Yatırım Bankacılığı": ["GS", "MS", "BLK", "SCHW"],
              "Ödeme Ağları": ["V", "MA", "AXP"],
              "Sigorta ve Holding": ["BRK-B", "MMC", "CB", "PGR"]})

_add("KRE", "SPDR Regional Banking", "Finans",
     "Ticari gayrimenkul ve KOBİ kredilerini fonlayan orta ölçekli bankalar.",
     gruplar={"Büyük Bölgesel": ["MTB", "RF", "HBAN", "FITB", "KEY", "CFG",
                                 "TFC", "FHN"],
              "Niş / Ticari": ["WAL", "ZION", "CMA"]})

_add("ARKF", "ARK Fintech Innovation", "Finans",
     "Dijital cüzdanlar, kripto borsaları ve yeni nesil ödeme sistemleri.",
     gruplar={"Kripto Altyapı": ["COIN", "HOOD"],
              "Ödeme ve Cüzdan": ["XYZ", "PYPL", "TOST", "AFRM", "SOFI"],
              "E-Ticaret Tabanı": ["SHOP", "MELI", "SE"],
              "Dijital Bankacılık": ["NU", "INTR"]})

# ========================= 6. İLETİŞİM & TÜKETİCİ =========================
_add("XLC", "Communication Services Select Sector", "Tüketici & Medya",
     "Sosyal medya, arama, streaming ve telekom liderleri.",
     gruplar={"Sosyal ve Reklam": ["META", "GOOGL", "PINS"],
              "Streaming": ["NFLX", "DIS", "WBD", "PARA"],
              "Telekom": ["TMUS", "T", "VZ"],
              "Kablo / Genişbant": ["CMCSA", "CHTR"]})

_add("XLY", "Consumer Discretionary Select Sector", "Tüketici & Medya",
     "E-ticaret, otomotiv, restoran ve giyim devleri.",
     gruplar={"E-Ticaret": ["AMZN", "EBAY"],
              "Otomotiv": ["TSLA", "F", "GM"],
              "Yapı Marketleri": ["HD", "LOW"],
              "Restoran ve Otel": ["MCD", "SBUX", "MAR", "HLT", "CMG"],
              "Giyim ve Lüks": ["NKE", "TJX", "LULU"],
              "Seyahat": ["BKNG", "ABNB"]})

_add("XRT", "SPDR Retail", "Tüketici & Medya",
     "Eşit ağırlıklı; süpermarketten online otomobil perakendesine.",
     gruplar={"Dijital Perakende": ["CVNA", "AMZN", "CHWY"],
              "Giyim Mağazaları": ["ANF", "GAP", "BOOT", "ROST", "TJX", "M", "JWN"],
              "Süpermarket ve Toptan": ["COST", "WMT", "TGT", "DLTR", "DG"],
              "Otomotiv ve Elektronik": ["AZO", "BBY"]})

_add("XHB", "SPDR Homebuilders", "Tüketici & Medya",
     "Konut üretim döngüsü ve ev içi donanım markaları.",
     gruplar={"Konut Geliştiriciler": ["DHI", "LEN", "PHM", "NVR", "TOL", "KBH"],
              "Yapı Marketleri": ["HD", "LOW", "BLDR"],
              "Boya ve Kimyasal": ["SHW"],
              "Ev Aletleri ve Mobilya": ["WHR", "TT", "MHK", "OC"]})

# ========================= 7. SAĞLIK & GENOMİK ============================
_add("XLV", "Health Care Select Sector", "Sağlık",
     "Dev ilaç üreticileri ile sağlık sigortası şirketlerinin birleşimi.",
     gruplar={"Mega İlaç": ["LLY", "MRK", "ABBV", "PFE", "BMY", "JNJ"],
              "Sigorta ve Bakım": ["UNH", "ELV", "HUM", "CVS"],
              "Biyoteknoloji Devleri": ["AMGN", "GILD", "REGN", "VRTX"],
              "Yaşam Bilimleri": ["TMO", "DHR"]})

_add("IHI", "iShares Medical Devices", "Sağlık",
     "Cerrahi cihazlar, protezler ve laboratuvar tanı ekipmanları.",
     gruplar={"Cerrahi Robotik": ["ISRG"],
              "Kardiyovasküler ve Ortopedi": ["SYK", "BSX", "MDT", "EW", "ZBH"],
              "Diyabet Takibi": ["DXCM", "ABT"],
              "Laboratuvar Tanı": ["TMO", "BDX"]})

_add("XBI", "SPDR Biotech", "Sağlık",
     "Eşit ağırlıklı; klinik aşamadaki yenilikçi moleküller ve FDA adayları.",
     gruplar={"Büyük Klinik": ["VRTX", "AMGN", "GILD", "MRNA", "BIIB", "REGN"],
              "Nadir Hastalık ve Gen Tedavisi": ["ALNY", "BMRN", "INCY", "UTHR",
                                                 "EXAS"]})

_add("ARKG", "ARK Genomic Revolution", "Sağlık",
     "CRISPR, DNA dizileme, hücresel tedavi ve YZ destekli ilaç keşfi.",
     gruplar={"Gen Düzenleme": ["CRSP", "NTLA", "BEAM", "EDIT"],
              "Erken Teşhis": ["EXAS", "GH"],
              "DNA Dizileme": ["TWST", "PACB", "ILMN"],
              "RNA Tedavileri": ["IONS", "ALNY"],
              "YZ Tabanlı İlaç Keşfi": ["SDGR", "RXRX"]})

# ========================= 8. GAYRİMENKUL & VERİ MERKEZİ ==================
_add("XLRE", "Real Estate Select Sector", "Gayrimenkul",
     "S&P 500'deki en büyük kurumsal gayrimenkul sahipleri.",
     gruplar={"Lojistik": ["PLD"],
              "Verici Kuleleri": ["AMT", "CCI", "SBAC"],
              "Veri Merkezi GYO": ["EQIX", "DLR"],
              "Perakende Alanları": ["SPG", "O"],
              "Sağlık Tesisleri": ["WELL", "VTR"],
              "Depolama": ["PSA", "EXR"]})

_add("SRVR", "Pacer Data & Infrastructure REIT", "Gayrimenkul",
     "Bulut, YZ sunucuları ve 5G'nin fiziksel binaları ile fiber ağları.",
     gruplar={"Veri Merkezleri": ["EQIX", "DLR"],
              "Telekom Kuleleri": ["AMT", "CCI", "SBAC"],
              "Fiber Altyapı": ["IRM", "UNIT"]})

_add("REZ", "iShares Residential & Multisector REIT", "Gayrimenkul",
     "Barınma, yaşlanma ve kişisel depolama odaklı GYO'lar.",
     gruplar={"Sağlık ve Kıdemli Yaşam": ["WELL", "VTR", "OHI"],
              "Kiralık Apartman": ["EQR", "AVB", "UDR", "CPT"],
              "Müstakil Kiralık": ["INVH", "AMH"],
              "Prefabrik Siteler": ["SUI", "ELS"],
              "Bireysel Depolama": ["PSA", "CUBE"]})

_add("VNQ", "Vanguard Real Estate", "Gayrimenkul",
     "XLRE'ye göre çok daha geniş kapsamlı gayrimenkul havuzu.",
     gruplar={"Lojistik": ["PLD"],
              "Dijital Altyapı": ["AMT", "EQIX", "CCI", "DLR"],
              "Perakende ve Net-Lease": ["SPG", "O", "KIM"],
              "Depolama ve Konut": ["PSA", "AVB", "EQR"],
              "Ormancılık GYO": ["WY", "RYN"]})

# ========================= 9. KRİPTO ======================================
_add("IBIT", "iShares Bitcoin Trust", "Kripto",
     "Hisse havuzu yok; %100 spot Bitcoin tutar.",
     gruplar={"Doğrudan Varlık": ["BTC-USD"]})

_add("WGMI", "Valkyrie Bitcoin Miners", "Kripto",
     "Madencilik şirketleri, veri merkezleri ve ASIC donanım tedarikçileri.",
     gruplar={"Endüstriyel Madenciler": ["MARA", "RIOT", "CLSK", "HUT", "CIFR",
                                          "IREN", "WULF", "CORZ", "HIVE", "BTDR"],
              "YZ/HPC Dönüşümü Yapanlar": ["CORZ", "HUT", "IREN"],
              "Donanım Tedarikçileri": ["NVDA", "AMD"]})

def holdings(sym: str) -> list[str]:
    """Bir ETF'in bilinen tüm bileşenleri (ağırlıklılar önce, sonra gruplar)."""
    d = ETF.get(sym)
    if not d:
        return []
    out = sorted(d["agirlik"], key=lambda t: -d["agirlik"][t])
    for grup in d["gruplar"].values():
        for t in grup:
            if t not in out:
                out.append(t)
    return out


def holding_meta(sym: str, ticker: str) -> dict[str, Any]:
    d = ETF.get(sym, {})
    grup = next((g for g, lst in d.get("gruplar", {}).items() if ticker in lst), "")
    return {"agirlik": d.get("agirlik", {}).get(ticker),
            "rol": d.get("rol", {}).get(ticker, ""),
            "grup": grup}


def all_etfs() -> list[str]:
    return sorted(ETF)


def etfs_by_category() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for sym, d in ETF.items():
        out.setdefault(d["kategori"], []).append(sym)
    return {k: sorted(v) for k, v in sorted(out.items())}


def all_stocks() -> list[str]:
    seen: list[str] = []
    for sym in ETF:
        for t in holdings(sym):
            if t not in seen:
                seen.append(t)
    return sorted(seen)


def etfs_containing(ticker: str) -> list[str]:
    """Bir hissenin geçtiği tüm ETF'ler — çarpan etkisini görmek için."""
    return sorted(s for s in ETF if ticker in holdings(s))


# --------------------------------------------------------------------------
# THEME TRACKER — sektörel ivme matrisi
# --------------------------------------------------------------------------
THEME_TRACKER: dict[str, list[str]] = {
    "Yarı İletken": ["SMH", "SOXX", "EUV"],
    "Yapay Zekâ": ["AIQ", "BOTZ"],
    "Kuantum": ["QTUM", "IONQ", "RGTI", "QUBT"],
    "Yazılım & SaaS": ["IGV", "CLOU", "PSJ"],
    "Siber Güvenlik": ["CIBR", "HACK", "BUG"],
    "Robotik & Otomasyon": ["BOTZ", "ROBO"],
    "Teknoloji (Geniş)": ["XLK", "QQQ"],
    "Veri Merkezi & Dijital GYO": ["SRVR"],
    "Gayrimenkul": ["VNQ", "XLRE", "REZ"],
    "Nükleer & Uranyum": ["URA", "NLR"],
    "Kamu Hizmetleri": ["XLU"],
    "Temiz Enerji": ["ICLN", "TAN"],
    "Petrol & Gaz": ["XLE", "XOP", "OIH"],
    "Bakır": ["COPX"],
    "Lityum & Batarya": ["LIT"],
    "Nadir Toprak": ["REMX"],
    "Altın Madenciliği": ["GDX", "GDXJ"],
    "Gümüş": ["SIL", "SLV"],
    "Metal & Madencilik": ["XME", "SLX"],
    "Savunma & Havacılık": ["XAR", "ITA"],
    "Uzay": ["ARKX", "UFO"],
    "Havayolları": ["JETS"],
    "Taşımacılık": ["IYT"],
    "Altyapı": ["PAVE", "XLI"],
    "Konut İnşaatı": ["ITB", "XHB"],
    "Bankalar": ["XLF", "KRE"],
    "Fintek": ["ARKF"],
    "Bitcoin": ["IBIT", "BITO"],
    "Bitcoin Madenciliği": ["WGMI", "MARA", "RIOT", "CLSK", "IREN"],
    "Biyoteknoloji": ["IBB", "XBI"],
    "Genomik": ["ARKG", "IDNA"],
    "Tıbbi Cihaz": ["IHI"],
    "Sağlık (Geniş)": ["XLV", "VHT"],
    "İlaç": ["PPH", "XPH"],
    "Perakende": ["XRT", "XLY"],
    "İletişim & Medya": ["XLC", "SOCL"],
    "Tüketici Defansif": ["XLP"],
    "Materyal": ["XLB"],
    "Tarım & Gıda": ["MOO", "DBA"],
    "Su Altyapısı": ["PHO", "FIW"],
    "Çin İnterneti": ["KWEB"],
    "Hindistan": ["INDA"],
    "Japonya": ["EWJ"],
    "Avrupa": ["VGK"],
    "Gelişen Piyasalar": ["EEM"],
    "Küçük Ölçekli (Small Cap)": ["IWM"],
    "Büyüme": ["VUG", "IWF"],
    "Değer": ["VTV", "IWD"],
    "Temettü": ["SCHD", "VIG"],
    "Volatilite": ["VIXY"],
    "Uzun Vadeli Tahvil": ["TLT"],
    "Yüksek Getirili Tahvil": ["HYG"],
}

# --------------------------------------------------------------------------
# ÇARPAN (CHOKEPOINT) HİSSELERİ
# Paranın hangi alt temaya gittiğinden bağımsız pay alan altyapı sahipleri.
# --------------------------------------------------------------------------
CHOKEPOINTS: dict[str, dict[str, Any]] = {
    "NVDA": {
        "rol": "İşlem gücü ve algoritma standardı",
        "capex": "YZ Ar-Ge ve bulut yatırımları",
        "mantik": "Sadece GPU üreticisi değil; CUDA ekosistemi YZ yazılımının "
                  "çalıştığı tek efektif platform. Robotikten ilaç keşfine, "
                  "otonom araçtan Bitcoin madenciliğine işlem gücü gerektiren "
                  "her trend dönüp dolaşıp NVDA'ya bütçe ayırıyor.",
    },
    "AVGO": {
        "rol": "Veri trafiği ve kurumsal entegrasyon",
        "capex": "Veri merkezi ve siber güvenlik bütçeleri",
        "mantik": "YZ veri merkezindeki binlerce çipin ultra hızlı konuşmasını "
                  "sağlayan ağ çiplerini üretir. VMware ve Symantec ile kurumsal "
                  "bulut yazılımı ve güvenliği de kontrol eder — sunucu yatırımı "
                  "arttıkça hem çip hem yazılım bacağından çift yönlü nakit akışı.",
    },
    "CEG": {
        "rol": "Temiz ve kesintisiz baz yük",
        "capex": "Teknoloji devlerinin karbon-nötr enerji arayışı",
        "mantik": "Hyperscaler'lar fosil kullanmayan, 7/24 kesintisiz baz yük "
                  "talep ediyor. CEG ABD'nin en büyük nükleer filosuna sahip; "
                  "Microsoft ve Amazon doğrudan PPA imzalıyor.",
    },
    "VST": {
        "rol": "Bağımsız nükleer üretim",
        "capex": "Veri merkezi elektrik anlaşmaları",
        "mantik": "CEG ile aynı tez: nükleer baz yük + serbest piyasa fiyatlaması. "
                  "Veri merkezi talebi arttıkça marjı doğrudan genişler.",
    },
    "ETN": {
        "rol": "Güç dağıtımı ve elektrifikasyon",
        "capex": "Şebeke yenileme, veri merkezi ve fabrika kurulumları",
        "mantik": "Enerji üretilebilir, çip tasarlanabilir; ama transformatör ve "
                  "güç yönetimi olmadan veri merkezine dağıtılamaz. Yeşil dönüşüm, "
                  "veri merkezi inşası ve üretimin ABD'ye dönmesi — üçü de ETN'e "
                  "doğrudan yeni sipariş demek.",
    },
    "EQIX": {
        "rol": "Küresel veri otobanlarının kesişimi",
        "capex": "Bulut ve YZ sunucu barındırma",
        "mantik": "Bulut soyut görünür ama betonarme bina, sıvı soğutma ve fiber "
                  "girişi ister. EQIX en kritik kesişim noktalarındaki binaları "
                  "işletir; YZ patlaması metrekare kirasını teknoloji çarpanıyla "
                  "fiyatlatır.",
    },
    "DLR": {
        "rol": "Hiperölçek veri merkezi mülkiyeti",
        "capex": "Bulut sağlayıcı kiralamaları",
        "mantik": "EQIX ile aynı tez, daha büyük ölçekli tekil kiracılara odaklı.",
    },
    "FCX": {
        "rol": "Elektrifikasyonun temel hammaddesi",
        "capex": "EV şebekeleri, veri merkezi kablolaması, ağır sanayi",
        "mantik": "Veri merkezi güç kabloları, rüzgâr türbinleri, EV bataryaları "
                  "ve altyapı projeleri muazzam bakır tüketir. Hangi teknolojinin "
                  "kazandığından bağımsız olarak elektrifikasyon içeren her "
                  "senaryoda FCX pay alır.",
    },
    "PLD": {
        "rol": "Lojistik mülkiyeti + mikro şebeke",
        "capex": "E-ticaret dağıtım ağı ve çatı üstü güneş",
        "mantik": "Küresel e-ticaretin en büyük depo sahibi. Çatılarını güneş "
                  "paneliyle donatıp kendi mikro şebekesini kuruyor — hem lojistik "
                  "büyümesinden kira topluyor hem enerji satıyor.",
    },
    "TSM": {
        "rol": "Gelişmiş çip üretiminin tek kapısı",
        "capex": "Tüm fabless çip tasarımcılarının üretim bütçesi",
        "mantik": "NVDA, AMD, AAPL dahil en gelişmiş çipleri fiziksel olarak "
                  "üreten tek dökümhane. Çip savaşını kim kazanırsa kazansın "
                  "üretim TSM'de yapılır.",
    },
    "ASML": {
        "rol": "EUV litografi tekeli",
        "capex": "Dökümhane kapasite yatırımları",
        "mantik": "3nm ve altı üretim için gereken EUV makinesini dünyada başka "
                  "kimse yapamıyor. Çip kapasitesi artacaksa yolu ASML'den geçer.",
    },
}

# --------------------------------------------------------------------------
# FUTURE THEMES — varsayılan liste (kullanıcı arayüzden ekler/çıkarır)
# --------------------------------------------------------------------------
DEFAULT_FUTURE_THEMES: dict[str, dict[str, list[str]]] = {
    "Chokepoint Çarpanları": {
        "hisse": list(CHOKEPOINTS), "etf": []},
    "Agentic AI & Yazılım": {
        "hisse": ["NOW", "SOUN", "ADBE", "DT", "S", "EXTR", "PLTR", "AI"],
        "etf": ["IGV", "CLOU"]},
    "Uzay Bilişimi & Keşif": {
        "hisse": holdings("SPACE_RACE"), "etf": ["ARKX", "UFO"]},
    "Kuantum Bilişim": {
        "hisse": ["IONQ", "RGTI", "QUBT", "QBTS"], "etf": ["QTUM"]},
    "Fotonik & Optik Çipler": {
        "hisse": holdings("PHOTON"), "etf": ["EUV"]},
    "Neocloud & Enerji Pivotu": {
        "hisse": ["CORZ", "IREN", "WULF", "APLD", "NBIS", "CIFR"],
        "etf": ["WGMI"]},
    "Nükleer & Temel Materyal": {
        "hisse": ["CEG", "VST", "TLN", "SMR", "NNE", "UUUU", "MP", "LEU"],
        "etf": ["URA", "REMX"]},
}

# --------------------------------------------------------------------------
# BİLANÇO TAKİP LİSTESİ (varsayılan)
# --------------------------------------------------------------------------
DEFAULT_EARNINGS = sorted(set("""
AAOI ABT ADBE AEHR AI ALAB AMAT AMD AMGN AMKR APLD ARM ASTS ASX ATRO AVGO
BA BE BKR BTDR CEG CIEN CIFR CLSK COHR CORZ CRDO CRM CRSP CRWV DELL DOCN
DT EMR EQIX ETN FCX FN FORM GFS GLW HEI HIMS HON HPE IBM INTC IONQ IRDM
IREN ISRG KTOS LEU LHX LITE LMT LRCX LUNR MA MARA MBLY META MP MRVL MSFT
NBIS NEE NOC NOW NTAP NTLA NVDA ONTO OUST PL PLTR POET PYPL QBTS QCOM QUBT
RDW RGTI RIOT RKLB RTX SANM SMCI SMR SNOW SOUN SPIR STX TER TLN TMUS TSM
UUUU VECO VSAT VST WDC WOLF WULF
""".split()))

# ==========================================================================
# KAYNAK: apex/engine.py
# ==========================================================================


from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Script varsayılanları (Pine input'larının default değerleri)
# --------------------------------------------------------------------------
PARAMS: dict[str, Any] = {
    # APEX CORE v3
    "sens": 1.5,          # i_sens        Hassasiyet
    "lenW": 21,           # i_lenW        Whale ataleti
    "lenD": 8,            # i_lenD        Daily ataleti
    "lenR": 13,           # i_lenR        Retail ataleti
    "effort": 0.8,        # i_effort      Çaba ≠ Sonuç ağırlığı
    "persist": 0.65,      # i_persist     Kalıcılık ρ
    "anchor": 0.70,       # i_anchor      Çapa bileşimi
    "trendL": 27,         # i_trendL      Trend penceresi
    "dFastL": 5,          # i_dFastL      Daily hızlı çapa
    "gamma": 1.2,         # i_gamma       Kontrast
    "proMix": 0.65,       # i_proMix      PRO içinde whale ağırlığı
    "smooth": 5,          # i_smooth      Final EMA
    "lamLen": 60,         # i_lamLen      Kyle λ penceresi
    "normLen": 252,       # i_normLen     Normalizasyon penceresi
    "stSens": 2.0,        # i_stSens      Stealth hassasiyeti
    "botCd": 10,          # i_botCd       Bot soğuma
    "dotN": 3,            # i_dotN        Seesaw bar sayısı
    "sqzLen": 20,         # i_sqzLen      Sıkışma penceresi
    # APEX V670 OMNI  (V665'ten düzeltilerek güncellendi — README'ye bakın)
    "volMult": 2.0,       # i_vol_mult    Afterburner hacim çarpanı
    "emaBreak": 9,        # i_ema_break   Diamond geri alım EMA'sı
    "diaLen": 60,         # i_dia_lookbk  Süpürülecek dip/tepe penceresi
    "diaWindow": 5,       # i_dia_window  Süpürme sonrası geri alım penceresi
    "diaCool": 10,        # i_dia_cool    Aynı yönde sinyaller arası bekleme
    "diaRegime": False,   # i_dia_regime  EMA200 rejim filtresi (varsayılan kapalı)
    "abReset": 20,        # i_ab_reset    Afterburner kilit açılma süresi
    "rsiFast": 7,         # i_rsi_fast
    "rsiMid": 14,         # i_rsi_mid     (MFI de bunu kullanır)
    "rsiSlow": 21,        # i_rsi_slow    V665'te ölü koddu, artık konsensüste
    "rocLen": 9,          # i_roc_len
    "rocScale": 10.0,     # i_roc_scale   roc*ölçek+50 ile 0-100'e taşınır
    "cciLen": 20,         # i_cci_len
    "cciScale": 200.0,    # i_cci_scale
    "tsiLong": 25,        # i_tsi_long
    "tsiShort": 13,       # i_tsi_short
    "omniSmooth": 3,      # i_omni_smooth
    "omniW": {            # konsensüs ağırlıkları (0 = bileşeni kapat)
        "rsiFast": 1.0, "rsiMid": 1.5, "rsiSlow": 1.0, "mfi": 1.5,
        "cci": 1.0, "tsi": 1.5, "roc": 0.5,
    },
    "exhSigma": 2.0,      # i_exh_sigma   Tükenme eşiği (σ)
    "bbMult": 2.0,        # i_bb_mult     Sıkışma: Bollinger çarpanı
    "kcMult": 1.5,        # i_kc_mult     Sıkışma: Keltner çarpanı
    "flatTol": 0.05,      # i_flat_tol    "Yatay" sayılma toleransı
    # ŞAHANE / V719
    "vwmLen": 14,         # i_vwm_len     Efor çizgisi penceresi
    "ultLen": 50,         # i_ult_len     Ultimate kanal
    "ultMult": 1.5,       # i_ult_mult
    "smcFast": 5,         # MSS tetik penceresi
    "smcMid": 10,         # likidite havuzu penceresi
    "kfLb": 3,            # i_kf_lb       Konfluans bileşen hafızası (bar)
    "kfExpThr": 6,        # i_kf_expThr   MAGNITUDE eşiği
    "kfEntryDir": 5,      # i_kf_entryDir DIRECTION giriş eşiği
    "stopMult": 2.0,      # i_ouStopMult  İz süren stop ATR çarpanı
    "hardStopPct": 20.0,  # i_hardStopPct Sert stop yüzdesi
    # QUANTUM V883
    "minLiqM": 5.0,       # i_min_liq     Min günlük hacim ($M)
}

BENCHMARK = "SPY"     # rejim kapısı ve göreli güç için
TROY = 31.1034768     # (portföy tarafıyla ortak sabit; burada kullanılmıyor)


@dataclass
class SignalRow:
    """Bir sembol için hesaplanmış tüm sinyal alanları."""
    ticker: str
    ok: bool = False
    error: str = ""
    data: dict[str, Any] = field(default_factory=dict)


# ==========================================================================
# APEX CORE v3 — Kyle-λ ayrıştırması
# ==========================================================================
def _f_prank(src: pd.Series, norm_len: int) -> pd.Series:
    """3 kademeli yedekli yüzdelik sıra — kısa geçmişte de değer üretir."""
    mid = max(int(round(norm_len / 3.0)), 20)
    a = percentrank(src, norm_len)
    b = percentrank(src, mid)
    c = percentrank(src, 20)
    return a.fillna(b).fillna(c).fillna(50.0)


def _f_medabs(src: pd.Series, norm_len: int) -> pd.Series:
    mid = max(int(round(norm_len / 3.0)), 20)
    x = src.abs()
    m = (x.rolling(norm_len).median()
         .fillna(x.rolling(mid).median())
         .fillna(x.rolling(20).median())
         .fillna(x))
    mn = sma(x, 20)
    return pd.Series(
        np.where(m > 0, m, np.where(mn.fillna(0) > 0, mn * 0.6745, 1.0)),
        index=src.index)


def _f_norm(src: pd.Series, norm_len: int) -> pd.Series:
    sc = _f_medabs(src, norm_len)
    return pd.Series(
        f_tanh(np.where(sc > 0, src / (1.4826 * sc), 0.0) / 2.0),
        index=src.index)


def apex_core(df: pd.DataFrame, p: dict[str, Any] | None = None) -> pd.DataFrame:
    """APEX CORE v3'ün bant ve sinyal serilerini üretir."""
    p = {**PARAMS, **(p or {})}
    o, h, l, c, v = (df["Open"], df["High"], df["Low"], df["Close"], df["Volume"])
    n_len = p["normLen"]
    out = pd.DataFrame(index=df.index)

    # --- temel güç zinciri ---
    rsi14 = rsi(c, 14)
    c_range = (h - l).clip(lower=0.001)
    delta = ((c - l) - (h - c)) / c_range
    up_w = h - np.maximum(o, c)
    dn_w = np.minimum(o, c) - l
    wick_d = (dn_w - up_w) / c_range

    d_vol = sma(delta * v, 20) / sma(v, 20).clip(lower=0.001)
    v_avg = sma(v, 20)
    rv_raw = v / v_avg.clip(lower=1)
    rvol = pd.Series(np.where(rv_raw > 2.5, 2.5 + np.log(rv_raw.clip(lower=1.6) - 1.5),
                              rv_raw), index=df.index)

    base_pwr = ((rsi14 - 50) + d_vol * 40 + wick_d * 20) * rvol * p["sens"]

    logic_p = np.log1p(np.exp(np.minimum(base_pwr / 5, 600.0))) * 5
    w_pwr = wma(pd.Series(np.minimum((np.log10(1 + logic_p) * 65) ** 0.8 * 1.8, 100),
                          index=df.index), 2)

    # --- Kyle λ: hacimle açıklanan hareket (WHALE) vs artık (RETAIL) ---
    ret1 = (c / c.shift(1) - 1.0).fillna(0.0)
    s_raw = np.sign(ret1) * np.sqrt(np.maximum(v * c, 1.0))
    s_sd = stdev(s_raw, p["lamLen"])
    s_n = pd.Series(np.where(s_sd > 0, s_raw / s_sd.replace(0, np.nan), 0.0),
                    index=df.index).fillna(0.0)
    corr = ret1.rolling(p["lamLen"]).corr(s_n)
    sd_r, sd_s = stdev(ret1, p["lamLen"]), stdev(s_n, p["lamLen"])
    lam = pd.Series(np.where((sd_s > 0) & corr.notna(),
                             corr * sd_r / sd_s.replace(0, np.nan), 0.0),
                    index=df.index).fillna(0.0)
    expl = lam * s_n
    resd = ret1 - expl

    # --- üç aktörün akışı ---
    trend_pos = stoch(c, h, l, p["trendL"])
    trend_norm = pd.Series(f_tanh((trend_pos - 50.0) / 25.0), index=df.index)
    fast_pos = stoch(c, h, l, p["dFastL"])
    fast_norm = pd.Series(f_tanh((fast_pos - 50.0) / 25.0), index=df.index)
    vol_w = np.sqrt(np.minimum(rvol, 4.0))

    ret_sd = stdev(ret1, 20)
    eff_res = (np.where(delta >= 0, 1.0, -1.0)
               * np.clip(rvol - 1.0, 0.0, 3.0)
               / np.maximum(1.0 + ret1.abs() / ret_sd.replace(0, np.nan).fillna(1e-9),
                            0.5))
    eff_res = pd.Series(eff_res, index=df.index).fillna(0.0)

    k_per = p["persist"] * 0.38
    w_sum = 1.5 + p["effort"]

    q_w = (vol_w * (_f_norm(base_pwr, n_len)
                    + 0.50 * _f_norm(expl, n_len)
                    + p["effort"] * _f_norm(eff_res, n_len)) / w_sum
           * (1.0 - k_per) + k_per * trend_norm)
    fast_raw = ((rsi(c, 5) - 50) + delta * 30) * np.minimum(rvol, 3.0)
    q_d = _f_norm(fast_raw, n_len) * (1.0 - k_per) + k_per * fast_norm
    q_r = _f_norm(resd, n_len) * np.minimum(1.6, 1.0 / np.maximum(rvol, 0.6))

    a_w, a_d, a_r = 2 / (p["lenW"] + 1), 2 / (p["lenD"] + 1), 2 / (p["lenR"] + 1)
    ch_w = leaky_reservoir(q_w.fillna(0.0), a_w)
    ch_d = leaky_reservoir(q_d.fillna(0.0), a_d)
    ch_r = leaky_reservoir(q_r.fillna(0.0), a_r)

    w_hd = 50.0 * (1.0 + _f_norm(base_pwr, n_len))
    anchor = p["anchor"] * trend_pos + (1.0 - p["anchor"]) * w_hd
    anchor_d = p["anchor"] * fast_pos + (1.0 - p["anchor"]) * w_hd

    lv_w = (1 - p["persist"]) * _f_prank(ch_w, n_len) + p["persist"] * anchor
    lv_d = (1 - p["persist"]) * _f_prank(ch_d, n_len) + p["persist"] * anchor_d
    lv_r = (1 - p["persist"]) * _f_prank(ch_r, n_len) + p["persist"] * (100.0 - anchor)

    whale = ema(pd.Series(f_contrast(lv_w, p["gamma"]), index=df.index), p["smooth"])
    daily = ema(pd.Series(f_contrast(lv_d, p["gamma"]), index=df.index), p["smooth"])
    retail = ema(pd.Series(f_contrast(lv_r, p["gamma"]), index=df.index), p["smooth"])

    pro = ema(p["proMix"] * whale + (1 - p["proMix"]) * daily, 2)
    ret_line = ema(retail, 2)

    # --- sinyaller ---
    w_inc = whale > whale.shift(1)
    w_dec = whale < whale.shift(1)
    red_cov = whale >= daily
    y_bars = barssince(red_cov).fillna(0)
    r_bars = barssince(~red_cov).fillna(0)

    db_bottom = red_cov & (y_bars.shift(1).fillna(0) >= p["dotN"]) & w_inc
    rd_top = (~red_cov) & (r_bars.shift(1).fillna(0) >= p["dotN"]) & w_dec
    cross_up = crossover(pro, ret_line)
    cross_dn = crossunder(pro, ret_line)

    st_in = (c < c.shift(1)) & (whale > whale.shift(1) + p["stSens"]) & (rvol > 0.8)
    st_out = (c > c.shift(1)) & (whale < whale.shift(1) - p["stSens"]) & (rvol > 0.8)
    star = cross_up & w_inc
    exhausted = (whale > 85.0) & (rvol < 0.8)

    # adaptif toplama/dağıtım + LİKİDİTE SÜPÜRMESİ (stop avı)
    w_p25 = percentile_lin(whale, n_len, 25)
    w_p75 = percentile_lin(whale, n_len, 75)
    lp5, hp5 = lowest(c, 5), highest(c, 5)
    lw5, hw5 = lowest(whale, 5), highest(whale, 5)
    lo20 = lowest(l, 20)

    real_acc = ((c <= lp5.shift(1).fillna(c) * 1.005) & (whale > lw5 * 1.10)
                & (lw5 <= w_p25.fillna(25.0)))
    real_dist = ((c >= hp5.shift(1).fillna(c) * 0.995) & (whale < hw5 * 0.90)
                 & (hw5 >= w_p75.fillna(75.0)))
    sweep_bar = ((l <= lo20.shift(1).fillna(l)) & (c > lo20.shift(1).fillna(l))
                 & w_inc)

    # sıkışma (BB ⊂ KC)
    bb_dev = 2.0 * stdev(c, p["sqzLen"])
    kc_dev = 1.5 * sma(true_range(h, l, c), p["sqzLen"])
    sqz_on = bb_dev < kc_dev
    sqz_fire = (~sqz_on) & sqz_on.shift(1).fillna(False)
    sqz_dur = sqz_on.groupby((~sqz_on).cumsum()).cumsum()

    # ATR hedefleri (T1 = 1.8×, T2 = 3.5×; VCP ve sıkışma süresiyle ölçeklenir)
    atr14 = atr(h, l, c, 14)
    vcp = sma(atr14, 50) / atr14.clip(lower=0.001)
    emlt = np.minimum(1.0 + sqz_dur / 25.0, 2.5)
    t1 = c + atr14 * 1.8 * vcp * emlt
    t2 = c + atr14 * 3.5 * vcp * emlt

    out["whale"], out["daily"], out["retail"] = whale, daily, retail
    out["pro"], out["ret_line"] = pro, ret_line
    out["w_pwr"], out["rvol"], out["atr14"] = w_pwr, rvol, atr14
    out["eff_res"] = eff_res
    out["db_bottom"], out["rd_top"] = db_bottom, rd_top
    out["cross_up"], out["cross_dn"] = cross_up, cross_dn
    out["st_in"], out["st_out"], out["star"] = st_in, st_out, star
    out["exhausted"] = exhausted
    out["real_acc"], out["real_dist"], out["sweep_bar"] = real_acc, real_dist, sweep_bar
    out["sqz_on"], out["sqz_fire"], out["sqz_dur"] = sqz_on, sqz_fire, sqz_dur
    out["t1"], out["t2"] = t1, t2
    out["w_inc"] = w_inc
    return out


# ==========================================================================
# APEX V665 OMNI — momentum konsensüsü ve füzyon
# ==========================================================================
def apex_omni(df: pd.DataFrame, p: dict[str, Any] | None = None) -> pd.DataFrame:
    p = {**PARAMS, **(p or {})}
    o, h, l, c, v = (df["Open"], df["High"], df["Low"], df["Close"], df["Volume"])
    out = pd.DataFrame(index=df.index)
    hlc3 = (h + l + c) / 3.0
    len100, len20 = 100, 20

    # ---- Ağırlıklı konsensüs (7 bileşen) --------------------------------
    # V665 beş bileşen kullanıyor, yavaş RSI ile ROC'u hesaplayıp çöpe atıyordu.
    w = p["omniW"]
    rsi_f = rsi(c, p["rsiFast"])
    rsi_m = rsi(c, p["rsiMid"])
    rsi_s = rsi(c, p["rsiSlow"])
    mfi_v = mfi(hlc3, v, p["rsiMid"])
    roc_r = roc(c, p["rocLen"])
    roc_n = ((roc_r * p["rocScale"]) + 50.0).clip(0, 100)
    cci_n = ((cci(hlc3, p["cciLen"]) + p["cciScale"])
             / (2.0 * p["cciScale"]) * 100.0).clip(0, 100)
    # DÜZELTME: tsi ±100 salınır; V665'in `tsi + 50` formülü üst yarıyı 100'e,
    # alt yarıyı 0'a kırpıyordu (ölçülen barların ~%3'ü). Doğrusu (tsi+100)/2.
    tsi_n = ((tsi(c, p["tsiLong"], p["tsiShort"]) + 100.0) / 2.0).clip(0, 100)

    w_sum = sum(w.values())
    if w_sum <= 0:
        raw_omni = pd.Series(50.0, index=df.index)
    else:
        raw_omni = (rsi_f * w["rsiFast"] + rsi_m * w["rsiMid"] + rsi_s * w["rsiSlow"]
                    + mfi_v * w["mfi"] + cci_n * w["cci"] + tsi_n * w["tsi"]
                    + roc_n * w["roc"]) / w_sum
    mom = wma(raw_omni, p["omniSmooth"])
    omni_center = mom - 50.0

    # KALİBRASYON KOPYASI — konfluans motoru için 5 bileşenli eski konsensüs.
    # Konfluansın HAREKET/YÖN eşikleri (kfExpThr, kfEntryDir, "mom > 60 / < 45",
    # "mom >= 55") 7.800 bar üzerinde BU formülle ölçülerek seçildi. Yukarıdaki
    # 7 bileşenli ağırlıklı sürüm daha doğrudur ama dağılımı kaydırır; ölçülmüş
    # eşikleri onunla kullanmak kalibrasyonu sessizce geçersiz kılar.
    # Bu yüzden gösterim/skor `mom`, konfluans `mom5` kullanır.
    # (TSI düzeltmesi ikisinde de var: kırpılan uçları geri kazandırır,
    #  formülü değiştirmez.)
    mom5 = wma((rsi_f + rsi_m + mfi_v + cci_n + tsi_n) / 5.0, p["omniSmooth"])

    # ---- Fusion / Synergy hız motorları ---------------------------------
    f_macd = ema(c, 12) - ema(c, 26)
    f_speed = _center_norm(f_macd, len100)
    f_sig = ema(f_speed, 9)
    f_hist = (f_speed - f_sig) * 1.5

    s_macd = ema(hlc3, 12) - ema(hlc3, 26)
    s_speed = _center_norm(s_macd, len100)

    # Tükenme YÖNLÜ olmalı. V665 mutlak sapma kullanıyordu; bu tanım, uzun bir
    # düşüşün ardından gelen sert toparlanmayı da "tükenmiş" sayıyor ve dip
    # dönüşünü — yani Diamond'ın yakalamak için var olduğu kalıbı — bloke
    # ediyordu. Doğru anlam: "zaten uzandığı yönde fazla uzamış".
    #   yukarı tükenme = ortalamasının çok üstünde VE hız zaten pozitif bölgede
    #   aşağı tükenme  = ortalamasının çok altında VE hız zaten negatif bölgede
    # Sapma sıfırsa (tam yatay seri) her fark sonsuz σ sayılırdı; koruma var.
    s_dev = stdev(s_speed, len20)
    s_off = s_speed - sma(s_speed, len20)
    exh_up = (s_dev > 0) & (s_off > p["exhSigma"] * s_dev) & (s_speed > 0)
    exh_dn = (s_dev > 0) & (s_off < -p["exhSigma"] * s_dev) & (s_speed < 0)
    is_exhausted = exh_up | exh_dn

    # ---- Sıkışma: Bollinger, Keltner'ın içinde --------------------------
    # V665 `stdev*2 < sma(tr)*1.5` yaklaşık ölçüsünü kullanıyordu; burada
    # literatürdeki tanım var. (Bantların ortak tabanı sadeleştiği için
    # matematiksel olarak dev < kc_range*kcMult'a indirgenir.)
    bb_dev = p["bbMult"] * stdev(c, len20)
    kc_rng = sma(true_range(h, l, c), len20)
    sqz_on = bb_dev < kc_rng * p["kcMult"]
    sqz_fire = (~sqz_on) & sqz_on.shift(1).fillna(False)
    grp = (~sqz_on).cumsum()
    sqz_dur = sqz_on.groupby(grp).cumsum().where(sqz_on, 0).astype(int)

    # ---- Afterburner (kilit açmalı) -------------------------------------
    # V665: mandal yalnızca TERS yönde tetikle sıfırlanıyordu; uzun bir
    # trendde sinyal ömür boyu bir kez yanıyordu.
    vsa_anom = v > sma(v, len20) * p["volMult"]
    rvol = v / sma(v, len20).replace(0, np.nan)
    roc_acc = roc_r - roc_r.shift(1)
    trig_up = vsa_anom & (roc_acc > 0) & (c > o) & (mom >= 50)
    trig_dn = vsa_anom & (roc_acc < 0) & (c < o) & (mom <= 50)
    mom_x = crossover(mom, pd.Series(50.0, index=df.index)) | \
        crossunder(mom, pd.Series(50.0, index=df.index))
    ab_bull, ab_bear = _latch_direction(trig_up, trig_dn, reset=mom_x,
                                        reset_bars=p["abReset"])

    # ---- Diamond: likidite süpürmesi + geri alım ------------------------
    # V665'in koşulu ("60 barın dibi OL" ve "iki bardır EMA9 ÜSTÜNDE kapat")
    # birbirini dışlıyordu; 16.000 barlık sınamada sinyal sıfır kez yandı.
    ema_focus = ema(c, p["emaBreak"])
    ema200 = ema(c, 200)
    prior_low = lowest(l, p["diaLen"]).shift(1)
    prior_high = highest(h, p["diaLen"]).shift(1)
    sweep_low = (l < prior_low) & (c > prior_low)
    sweep_high = (h > prior_high) & (c < prior_high)
    since_low = barssince(sweep_low)
    since_high = barssince(sweep_high)

    mom_turn_up = (mom > mom.shift(1)) & (s_speed > s_speed.shift(1))
    mom_turn_dn = (mom < mom.shift(1)) & (s_speed < s_speed.shift(1))
    gate_bull = (c > ema200) if p["diaRegime"] else pd.Series(True, index=df.index)
    gate_bear = (c < ema200) if p["diaRegime"] else pd.Series(True, index=df.index)

    # Veto yönlüdür: yukarı uzamışken ALMA, aşağı uzamışken SATMA.
    # Ters yöndeki tükenme zaten dönüş kurgusunun kendisidir.
    raw_buy = (crossover(c, ema_focus) & (since_low <= p["diaWindow"])
               & mom_turn_up & ~exh_up & gate_bull)
    raw_sell = (crossunder(c, ema_focus) & (since_high <= p["diaWindow"])
                & mom_turn_dn & ~exh_dn & gate_bear)
    dia_buy = _cooldown(raw_buy, p["diaCool"])
    dia_sell = _cooldown(raw_sell, p["diaCool"])

    # ---- Yönlü skorlar ---------------------------------------------------
    # V665'te tek yönsüz skor vardı: sıkışma ve hacim anomalisi düşüşte bile
    # artı sayılıyordu, üstelik ikisi aynı anda barların ~%0.5'inde oluştuğu
    # için 6/6 fiilen erişilemezdi.
    tol = p["flatTol"]
    d_omni = omni_center - omni_center.shift(1)
    rv1 = rvol.fillna(1.0) >= 1.0
    bull = ((mom >= 50).astype(int) + (d_omni > tol).astype(int)
            + (f_hist > 0).astype(int) + (f_speed > f_sig).astype(int)
            + (s_speed > s_speed.shift(1)).astype(int)
            + (rv1 & (c > o)).astype(int))
    bear = ((mom < 50).astype(int) + (d_omni < -tol).astype(int)
            + (f_hist < 0).astype(int) + (f_speed < f_sig).astype(int)
            + (s_speed < s_speed.shift(1)).astype(int)
            + (rv1 & (c < o)).astype(int))

    out["mom"], out["mom5"] = mom, mom5
    out["f_speed"], out["f_sig"], out["f_hist"] = f_speed, f_sig, f_hist
    out["s_speed"], out["is_exhausted"] = s_speed, is_exhausted
    out["exh_up"], out["exh_dn"] = exh_up, exh_dn
    out["vsa_anom"], out["rvol"] = vsa_anom, rvol
    out["ab_bull"], out["ab_bear"] = ab_bull, ab_bear
    out["dia_buy"], out["dia_sell"] = dia_buy, dia_sell
    out["sweep_low"], out["sweep_high"] = sweep_low, sweep_high
    out["hud"], out["hud_bear"] = bull, bear          # hud = BOĞA skoru
    out["hud_net"] = bull - bear
    out["sqz_on"], out["sqz_fire"], out["sqz_dur"] = sqz_on, sqz_fire, sqz_dur
    out["ew_bull"] = crossover(f_speed, f_sig)
    out["ew_bear"] = crossunder(f_speed, f_sig)
    return out


def _center_norm(src: pd.Series, n: int) -> pd.Series:
    """
    Seriyi n barlık aralığına göre 0-100'e taşır, merkezini sıfıra çeker.

    V665 burada `.clip(lower=0.001)` kullanıyordu. Bu sabit taban 300 dolarlık
    bir hissede zararsız, 0.0004 dolarlık bir coinde MACD aralığının tamamından
    büyük olduğu için tüm seriyi eziyordu. Aralık gerçekten sıfırsa değer
    tanımsızdır; nötr (0) dönmek doğrusudur. Isınma barları NaN kalır.
    """
    hi, lo = highest(src, n), lowest(src, n)
    rng = hi - lo
    out = pd.Series(np.nan, index=src.index)
    ok = rng > 0
    out[ok] = (src[ok] - lo[ok]) / rng[ok] * 100.0 - 50.0
    out[rng.notna() & ~ok] = 0.0
    return out


def _cooldown(sig: pd.Series, bars: int) -> pd.Series:
    """Aynı yönde art arda sinyalleri bastırır (ilkini geçirir)."""
    if bars <= 0:
        return sig.fillna(False)
    vals = sig.fillna(False).to_numpy()
    out = np.zeros(len(vals), dtype=bool)
    last = -(10 ** 9)
    for i in range(len(vals)):
        if vals[i] and (i - last) > bars:
            out[i] = True
            last = i
    return pd.Series(out, index=sig.index)


def _latch_direction(trig_up: pd.Series, trig_dn: pd.Series,
                     reset: pd.Series | None = None,
                     reset_bars: int | None = None):
    """
    Yön değişiminde bir kez ateşleyen mandal.

    `reset` / `reset_bars` verilirse kilit ters yön beklemeden de açılır:
    V665'te uzun bir yükselişte Afterburner ömür boyu tek kez yanıyordu.
    """
    up = np.zeros(len(trig_up), dtype=bool)
    dn = np.zeros(len(trig_up), dtype=bool)
    state, last = 0, -(10 ** 9)
    tu, td = trig_up.fillna(False).to_numpy(), trig_dn.fillna(False).to_numpy()
    rs = (reset.fillna(False).to_numpy() if reset is not None
          else np.zeros(len(tu), dtype=bool))
    for i in range(len(tu)):
        if state != 0 and (rs[i] or (reset_bars is not None
                                     and (i - last) >= reset_bars)):
            state = 0
        if tu[i] and state != 1:
            up[i], state, last = True, 1, i
        if td[i] and state != -1:
            dn[i], state, last = True, -1, i
    return pd.Series(up, index=trig_up.index), pd.Series(dn, index=trig_up.index)


# ==========================================================================
# ŞAHANE — SMC likidite süpürmesi, rejim kapısı, efor çizgisi
# ==========================================================================
def sahane_layer(df: pd.DataFrame, w_pwr: pd.Series,
                 p: dict[str, Any] | None = None) -> pd.DataFrame:
    p = {**PARAMS, **(p or {})}
    o, h, l, c, v = (df["Open"], df["High"], df["Low"], df["Close"], df["Volume"])
    out = pd.DataFrame(index=df.index)

    # Efor çizgisi: hacim ağırlıklı fiyat
    raw_effort = (wma(c * v, p["vwmLen"]) / wma(v, p["vwmLen"]).clip(lower=0.001))
    eff_price = wma(raw_effort, 3)
    out["eff_price"] = eff_price
    out["eff_up"] = crossover(c, eff_price)
    out["eff_dn"] = crossunder(c, eff_price)

    # Adaptive Ultimate: whale gücü arttıkça bant daralır
    ult_basis = sma(c, p["ultLen"])
    adaptive_mult = p["ultMult"] * (1.0 - w_pwr.fillna(0) / 250.0)
    ult_dev = adaptive_mult * stdev(c, p["ultLen"])
    out["ult_ceil"] = ult_basis + ult_dev
    out["ult_floor"] = ult_basis - ult_dev
    out["ult_up_cross"] = crossover(c, out["ult_ceil"])

    # EMA200 rejim kapısı
    ema200 = ema(c, 200)
    out["ema200"] = ema200
    out["regime_bull"] = c > ema200

    # SMC — likidite havuzu süpürmesi + market structure shift teyidi
    s_low = lowest(l, p["smcMid"]).shift(1)
    s_high = highest(h, p["smcMid"]).shift(1)
    mss_buy = highest(h, p["smcFast"]).shift(1)
    mss_sell = lowest(l, p["smcFast"]).shift(1)

    sweep_low = (l < s_low) & (c > s_low)
    sweep_high = (h > s_high) & (c < s_high)
    out["sweep_low"], out["sweep_high"] = sweep_low, sweep_high
    out["smc_buy"] = sweep_low & (c > mss_buy) & (c > o)
    out["smc_sell"] = sweep_high & (c < mss_sell) & (c < o)
    return out


# ==========================================================================
# V719 KONFLUANS — MAGNITUDE 0–18 / DIRECTION −5…+5
# ==========================================================================
def confluence(df: pd.DataFrame, core: pd.DataFrame, omni: pd.DataFrame,
               sah: pd.DataFrame, p: dict[str, Any] | None = None) -> pd.DataFrame:
    p = {**PARAMS, **(p or {})}
    c, v, l = df["Close"], df["Volume"], df["Low"]
    out = pd.DataFrame(index=df.index)
    lb = p["kfLb"]

    kf_rvol = v / sma(v, 20).replace(0, np.nan)
    kf_sma50 = sma(c, 50)

    # Minervini MVP: hacim + 15 barda %10 + alıcı baskısı + 50MA üstü
    cp = ((c - l) / (df["High"] - l).replace(0, np.nan)).fillna(0.5)
    ret_log = np.log(c / c.shift(1)).replace([np.inf, -np.inf], np.nan)
    rsd = stdev(ret_log, 20)
    bvc = pd.Series(normal_cdf((ret_log / rsd.replace(0, np.nan)).fillna(0).to_numpy()),
                    index=df.index)
    bp = (0.375 * cp + 0.625 * bvc).clip(0, 1)
    mvp = ((kf_rvol >= 1.5) & (c >= c.shift(15) * 1.10) & (bp > 0.5) & (c > kf_sma50))
    out["mvp"], out["bp"] = mvp, bp

    def recent(s: pd.Series) -> pd.Series:
        return s.fillna(False).rolling(lb).max().fillna(0).astype(bool)

    # MAGNITUDE — ölçülmüş ağırlıklarla (scriptte kayıtlı lift değerlerine göre)
    mag = (3 * recent(omni["ab_bull"]).astype(int)
           + 3 * recent(mvp).astype(int)
           + 2 * recent(kf_rvol >= 2.0).astype(int)
           + 2 * recent(crossover(omni["s_speed"], pd.Series(0.0, index=df.index))).astype(int)
           + 2 * recent(sah["ult_up_cross"]).astype(int)
           + 2 * recent(_effort_score(core, omni, sah) >= 4).astype(int)
           + 1 * recent(core["star"]).astype(int)
           + 1 * recent(omni["dia_buy"]).astype(int)
           + 1 * recent(omni["ew_bull"]).astype(int)
           + 1 * recent(kf_rvol >= 1.5).astype(int))

    # DIRECTION — beş bağımsız üçlü
    z = pd.Series(0.0, index=df.index)
    d1 = np.sign(omni["s_speed"].fillna(0))
    d2 = np.where(omni["s_speed"] > 25, 1, np.where(omni["s_speed"] < -25, -1, 0))
    d3 = np.sign(omni["f_hist"].fillna(0))
    # mom5: eşiklerin ölçüldüğü 5 bileşenli konsensüs (bkz. apex_omni)
    d4 = np.where(omni["mom5"] > 60, 1, np.where(omni["mom5"] < 45, -1, 0))
    d5 = np.where(c > kf_sma50, 1, np.where(c < kf_sma50, -1, 0))
    direction = (d1 + d2 + d3 + d4 + d5).astype(int)

    out["magnitude"] = mag
    out["direction"] = pd.Series(direction, index=df.index)
    out["risk_state"] = (mag >= p["kfExpThr"]) & (out["direction"] <= 0)
    out["risk_hard"] = out["risk_state"] & (out["direction"] <= -2)
    out["expand_state"] = (mag >= p["kfExpThr"]) & (out["direction"] > 0)
    out["entry_state"] = (mag < p["kfExpThr"]) & (out["direction"] >= p["kfEntryDir"])

    # İz süren ATR zırhı
    out["trail_stop"] = ratcheting_atr_stop(
        l, core["atr14"], p["stopMult"],
        entry_price=safe_last(c), hard_stop_pct=p["hardStopPct"])
    return out


def _effort_score(core: pd.DataFrame, omni: pd.DataFrame,
                  sah: pd.DataFrame) -> pd.Series:
    """Efor kırılım gücü 0–8 (ŞAHANE V710 §1.15)."""
    return (core["sqz_on"].astype(int)
            + sah["smc_buy"].astype(int)
            + core["sweep_bar"].astype(int)
            + sah["ult_up_cross"].astype(int)
            + (omni["mom5"] >= 55).astype(int)
            + (core["whale"] >= 60).astype(int)
            + omni["vsa_anom"].astype(int)
            + sah["eff_up"].astype(int)).clip(upper=8)


# ==========================================================================
# TEK SEMBOL ÖZETİ
# ==========================================================================
def _delta(series: pd.Series, back: int = 1) -> float:
    """
    Serinin `back` bar önceki değerine göre değişimi.
    Tarayıcıda "WHALE 72" tek başına anlamsız — 72 ve YÜKSELİYOR mu, yoksa
    85'ten düşerek mi 72'ye geldi, karar bunu bilmeye bağlı.
    """
    v = pd.Series(series).dropna()
    if len(v) <= back:
        return float("nan")
    return float(v.iloc[-1] - v.iloc[-1 - back])


def analyze(df: pd.DataFrame, ticker: str,
            bench_close: pd.Series | None = None,
            weekly_bull: bool | None = None,
            p: dict[str, Any] | None = None) -> SignalRow:
    """Bir sembolün tüm katmanlarını hesaplayıp son bardaki durumu döner."""
    p = {**PARAMS, **(p or {})}
    row = SignalRow(ticker=ticker)

    df = df.dropna(subset=["Close"]).copy()
    if len(df) < 60:
        row.error = f"Yetersiz veri ({len(df)} bar, en az 60 gerekli)"
        return row

    try:
        core = apex_core(df, p)
        omni = apex_omni(df, p)
        sah = sahane_layer(df, core["w_pwr"], p)
        conf = confluence(df, core, omni, sah, p)
    except Exception as exc:          # pragma: no cover - savunmacı
        row.error = f"Hesaplama hatası: {exc}"
        return row

    c = df["Close"]
    v = df["Volume"]
    price = safe_last(c)
    atr14 = safe_last(core["atr14"])
    dollar_vol_m = safe_last(sma(c * v, 20)) / 1e6

    # göreli güç (benchmark'a karşı 20 barlık momentum yüzdeliği)
    rs_mom = rs_rank = np.nan
    if bench_close is not None and len(bench_close) > 25:
        rel = (c / bench_close.reindex(c.index).ffill()).dropna()
        if len(rel) > 25:
            rs_mom = safe_last(roc(rel, 20))
            rs_rank = safe_last(percentrank(roc(rel, 20), min(100, len(rel) - 2)))

    eff_score = safe_last(_effort_score(core, omni, sah))
    whale = safe_last(core["whale"])
    pro = safe_last(core["pro"])
    ret_l = safe_last(core["ret_line"])
    mag = safe_last(conf["magnitude"])
    direction = safe_last(conf["direction"])

    # Önceki bara ve önceki haftaya göre değişimler
    d_whale = _delta(core["whale"])
    d_whale5 = _delta(core["whale"], 5)
    d_pro_ret = _delta(core["pro"] - core["ret_line"])
    d_omni = _delta(omni["mom"])
    d_omni5 = _delta(omni["mom"], 5)
    d_mag = _delta(conf["magnitude"])
    d_dir = _delta(conf["direction"])
    d_wpwr = _delta(core["w_pwr"])

    row.ok = True
    row.data = {
        "Fiyat": price,
        "ATR": atr14,
        "ATR %": (atr14 / price * 100.0) if price else np.nan,
        "Hacim ($M)": dollar_vol_m,
        "WHALE": whale,
        "ΔWHALE": d_whale,
        "ΔWHALE 5B": d_whale5,
        "Whale Yön": _trend_arrow(d_whale, d_whale5),
        "DAILY": safe_last(core["daily"]),
        "RETAIL": safe_last(core["retail"]),
        "PRO": pro,
        "PRO-RET": pro - ret_l,
        "ΔPRO-RET": d_pro_ret,
        "Whale Power": safe_last(core["w_pwr"]),
        "ΔWhale Power": d_wpwr,
        "RVOL": safe_last(core["rvol"]),
        "OMNI": safe_last(omni["mom"]),
        "ΔOMNI": d_omni,
        "ΔOMNI 5B": d_omni5,
        "OMNI Yön": _trend_arrow(d_omni, d_omni5),
        "Fusion": safe_last(omni["f_speed"]),
        "Synergy": safe_last(omni["s_speed"]),
        "Boğa /6": int(safe_last(omni["hud"], 0)),
        "Ayı /6": int(safe_last(omni["hud_bear"], 0)),
        "Skor Net": int(safe_last(omni["hud_net"], 0)),
        "Efor /8": int(eff_score) if np.isfinite(eff_score) else 0,
        "MAGNITUDE": int(mag) if np.isfinite(mag) else 0,
        "ΔMAG": d_mag,
        "DIRECTION": int(direction) if np.isfinite(direction) else 0,
        "ΔDIR": d_dir,
        "RS %": rs_mom,
        "RS Sıra": rs_rank,
        "T1": safe_last(core["t1"]),
        "T2": safe_last(core["t2"]),
        "Stop": safe_last(conf["trail_stop"]),
        "Rejim": bool(safe_bool(sah["regime_bull"])),
        "Haftalık": weekly_bull,
        "Sıkışma": bool(safe_bool(core["sqz_on"])),
        "Sıkışma Süre": int(safe_last(core["sqz_dur"], 0)),
        "1 Gün %": safe_last(roc(c, 1)),
        "1 Hafta %": safe_last(roc(c, 5)),
        "1 Ay %": safe_last(roc(c, 21)),
        # olay bayrakları
        "_star": safe_bool(core["star"]),
        "_dia_buy": safe_bool(omni["dia_buy"]),
        "_dia_sell": safe_bool(omni["dia_sell"]),
        "_sweep": safe_bool(core["sweep_bar"]) or safe_bool(sah["smc_buy"]),
        "_smc_sell": safe_bool(sah["smc_sell"]),
        "_st_in": safe_bool(core["st_in"]),
        "_st_out": safe_bool(core["st_out"]),
        "_acc": safe_bool(core["real_acc"]),
        "_dist": safe_bool(core["real_dist"]),
        "_ab_bull": safe_bool(omni["ab_bull"]),
        "_ab_bear": safe_bool(omni["ab_bear"]),
        "_exhausted": safe_bool(core["exhausted"]) or safe_bool(omni["is_exhausted"]),
        "_sqz_fire": safe_bool(core["sqz_fire"]),
        "_cross_up": safe_bool(core["cross_up"]),
        "_cross_dn": safe_bool(core["cross_dn"]),
        "_entry": safe_bool(conf["entry_state"]),
        "_risk": safe_bool(conf["risk_state"]),
        "_risk_hard": safe_bool(conf["risk_hard"]),
        "_expand": safe_bool(conf["expand_state"]),
        "_mvp": safe_bool(conf["mvp"]),
        "_eff_up": safe_bool(sah["eff_up"]),
    }
    row.data["Sinyal"] = headline_signal(row.data)
    row.data["Efor"] = effort_state(row.data, safe_last(c), safe_last(sah["eff_price"]))
    return row


def _trend_arrow(d1: float, d5: float) -> str:
    """
    Bir barlık ve beş barlık değişimi tek okla özetler.

      ⇈ hem dün hem hafta boyunca artıyor  (hızlanan güçlenme)
      ↗ kısa vadede artıyor ama haftalık zayıf (yeni dönüş)
      ↘ kısa vadede düşüyor ama haftalık güçlü (soluklanma)
      ⇊ ikisi de düşüyor (hızlanan bozulma)
    """
    if not np.isfinite(d1) and not np.isfinite(d5):
        return "—"
    a = d1 if np.isfinite(d1) else 0.0
    b = d5 if np.isfinite(d5) else 0.0
    if a > 0.5 and b > 0.5:
        return "⇈ güçleniyor"
    if a > 0.5 >= b:
        return "↗ dönüyor"
    if a < -0.5 and b < -0.5:
        return "⇊ bozuluyor"
    if a < -0.5 <= b:
        return "↘ soluklanıyor"
    return "→ yatay"


def headline_signal(d: dict[str, Any]) -> str:
    """
    Tek satırlık başlık sinyali — scriptlerin işaret önceliğini korur:
    risk uyarıları her şeyin önünde, sonra en güçlü giriş tetikleyicileri.
    """
    if d.get("_risk_hard"):
        return "🩸 GÜÇLÜ RİSK"
    if d.get("_dia_sell") or d.get("_smc_sell"):
        return "⛔ DIAMOND SAT"
    if d.get("_dist") or d.get("_st_out"):
        return "🐋 DAĞITIM"
    if d.get("_ab_bear"):
        return "🔻 AFTERBURNER AYI"
    if d.get("_exhausted"):
        return "⚠️ TÜKENME"
    if d.get("_dia_buy"):
        return "💎 DIAMOND AL"
    if d.get("_star"):
        return "⭐ GOLDEN STAR"
    if d.get("_sweep"):
        return "🎣 LİKİDİTE SÜPÜRMESİ"
    if d.get("_st_in") or d.get("_acc"):
        return "🐋 TOPLAMA"
    if d.get("_ab_bull"):
        return "🚀 AFTERBURNER"
    if d.get("_entry"):
        return "🟢 GİRİŞ BÖLGESİ"
    if d.get("_mvp"):
        return "📈 MINERVINI MVP"
    if d.get("_sqz_fire"):
        return "🎯 SIKIŞMA PATLADI"
    if d.get("_cross_up"):
        return "⚡ PRO ↗ RETAIL"
    if d.get("_cross_dn"):
        return "⚡ PRO ↘ RETAIL"
    if d.get("_risk"):
        return "⚠️ RİSK"
    if d.get("Sıkışma"):
        return "🕳️ SIKIŞMA"
    if d.get("_expand"):
        return "🟡 GENİŞLEME"
    return "⚪ BEKLE"


def effort_state(d: dict[str, Any], price: float, eff_price: float) -> str:
    if not np.isfinite(price) or not np.isfinite(eff_price):
        return "➖"
    if d.get("_eff_up"):
        return "🚀 EFOR KIRILIMI"
    return "🟢 POZ" if price > eff_price else "🔴 NEG"


SIGNAL_COLORS = {
    "🩸 GÜÇLÜ RİSK": ("#4a0d12", "#ffffff"),
    "⛔ DIAMOND SAT": ("#b71c1c", "#ffffff"),
    "🐋 DAĞITIM": ("#7b1f14", "#ffffff"),
    "🔻 AFTERBURNER AYI": ("#5c1a3a", "#ffffff"),
    "⚠️ TÜKENME": ("#4a3a05", "#ffea00"),
    "💎 DIAMOND AL": ("#00e6ff", "#04141a"),
    "⭐ GOLDEN STAR": ("#ffd700", "#1a1400"),
    "🎣 LİKİDİTE SÜPÜRMESİ": ("#00bfff", "#04141a"),
    "🐋 TOPLAMA": ("#006064", "#ffffff"),
    "🚀 AFTERBURNER": ("#ff9800", "#1a0d00"),
    "🟢 GİRİŞ BÖLGESİ": ("#00e676", "#04140a"),
    "📈 MINERVINI MVP": ("#1b5e20", "#ffffff"),
    "🎯 SIKIŞMA PATLADI": ("#4a148c", "#ffffff"),
    "⚡ PRO ↗ RETAIL": ("#0d3b52", "#00e5ff"),
    "⚡ PRO ↘ RETAIL": ("#3b1020", "#ff80ab"),
    "⚠️ RİSK": ("#3a2a05", "#ffd54f"),
    "🕳️ SIKIŞMA": ("#26134a", "#b39dff"),
    "🟡 GENİŞLEME": ("#3a3205", "#ffd600"),
    "⚪ BEKLE": ("#15151c", "#8a8a95"),
}

# ==========================================================================
# KAYNAK: apex/ui.py
# ==========================================================================


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

# ==========================================================================
# KAYNAK: apex/macro.py
# ==========================================================================


import calendar
import datetime as dt
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Makro göstergelerin Yahoo sembolleri
MACRO_TICKERS: dict[str, str] = {
    "SPY": "SPY",          # geniş piyasa
    "QQQ": "QQQ",          # teknoloji
    "VIX": "^VIX",         # 30 günlük örtük oynaklık
    "VIX3M": "^VIX3M",     # 3 aylık — vade yapısı için
    "TLT": "TLT",          # uzun vadeli hazine
    "HYG": "HYG",          # yüksek getirili şirket tahvili
    "LQD": "LQD",          # yatırım yapılabilir tahvil
    "DXY": "DX-Y.NYB",     # dolar endeksi
    "GOLD": "GC=F",        # altın
    "COPPER": "HG=F",      # bakır
    "OIL": "CL=F",         # ham petrol
    "BTC": "BTC-USD",      # kripto risk iştahı
    "US10Y": "^TNX",       # 10 yıllık tahvil faizi
    "US3M": "^IRX",        # 13 haftalık bono — getiri eğrisi için
    "RSP": "RSP",          # eşit ağırlıklı S&P — piyasa genişliği
    "IWM": "IWM",          # küçük ölçek — riskin ucu
    "XLY": "XLY",          # ihtiyari tüketim
    "XLP": "XLP",          # temel tüketim (defansif)
}

# 2026 FOMC toplantı tarihleri (son gün). Yeni yıl takvimi açıklanınca güncelleyin.
FOMC_DATES_2026 = [
    dt.date(2026, 1, 28), dt.date(2026, 3, 18), dt.date(2026, 4, 29),
    dt.date(2026, 6, 17), dt.date(2026, 7, 29), dt.date(2026, 9, 16),
    dt.date(2026, 11, 4), dt.date(2026, 12, 16),
]
FOMC_DATES_2027 = [
    dt.date(2027, 1, 27), dt.date(2027, 3, 17), dt.date(2027, 4, 28),
    dt.date(2027, 6, 16), dt.date(2027, 7, 28), dt.date(2027, 9, 22),
    dt.date(2027, 11, 3), dt.date(2027, 12, 15),
]


# --------------------------------------------------------------------------
# Takvim
# --------------------------------------------------------------------------
def third_friday(year: int, month: int) -> dt.date:
    """Aylık opsiyon vadesi — ayın üçüncü cuması."""
    c = calendar.Calendar(firstweekday=calendar.MONDAY)
    fridays = [d for week in c.monthdatescalendar(year, month)
               for d in week if d.weekday() == calendar.FRIDAY and d.month == month]
    return fridays[2]


def next_opex(today: dt.date | None = None) -> tuple[dt.date, int, bool]:
    """Sıradaki OPEX tarihi, kalan gün ve üçlü cadı (quad witching) mı."""
    today = today or dt.date.today()
    d = third_friday(today.year, today.month)
    if today > d:
        m = today.month % 12 + 1
        y = today.year + (1 if today.month == 12 else 0)
        d = third_friday(y, m)
    return d, (d - today).days, d.month in (3, 6, 9, 12)


def next_fomc(today: dt.date | None = None) -> tuple[dt.date | None, int | None]:
    today = today or dt.date.today()
    future = [d for d in (FOMC_DATES_2026 + FOMC_DATES_2027) if d >= today]
    if not future:
        return None, None
    return future[0], (future[0] - today).days


# --------------------------------------------------------------------------
# Ölçümler
# --------------------------------------------------------------------------
@dataclass
class MacroReading:
    """Tek bir makro göstergenin okuması."""
    key: str
    label: str
    value: float
    change_pct: float
    detail: str = ""
    tone: str = "neutral"     # good | bad | neutral


@dataclass
class MacroState:
    readings: dict[str, MacroReading] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    regime: str = "BELİRSİZ"
    regime_desc: str = ""
    battery: dict[str, int] = field(default_factory=dict)
    risk_score: float = 50.0
    errors: list[str] = field(default_factory=list)
    opex_date: str = ""
    opex_days: int = 0
    opex_quad: bool = False
    fomc_date: str = ""
    fomc_days: int | None = None
    asof: str = ""

    def get(self, key: str) -> float:
        r = self.readings.get(key)
        return r.value if r else float("nan")


def _pct_change(s: pd.Series, n: int = 1) -> float:
    s = s.dropna()
    if len(s) <= n:
        return float("nan")
    return float((s.iloc[-1] / s.iloc[-1 - n] - 1) * 100)


def _last(s: pd.Series) -> float:
    s = pd.Series(s).dropna()
    return float(s.iloc[-1]) if len(s) else float("nan")


def build_macro_state(prices: dict[str, pd.DataFrame],
                      today: dt.date | None = None) -> MacroState:
    """
    Çekilmiş fiyat serilerinden makro durumu hesaplar.
    `prices`: {anahtar: OHLCV DataFrame} — anahtarlar MACRO_TICKERS ile aynı.
    """
    st = MacroState()
    today = today or dt.date.today()
    st.asof = dt.datetime.now().strftime("%d.%m.%Y %H:%M")

    opex_d, opex_dl, quad = next_opex(today)
    st.opex_date, st.opex_days, st.opex_quad = opex_d.strftime("%d.%m.%Y"), opex_dl, quad
    fomc_d, fomc_dl = next_fomc(today)
    st.fomc_date = fomc_d.strftime("%d.%m.%Y") if fomc_d else "—"
    st.fomc_days = fomc_dl

    def close(key: str) -> pd.Series | None:
        df = prices.get(key)
        if df is None or df.empty or "Close" not in df:
            return None
        s = df["Close"].dropna()
        return s if len(s) > 5 else None

    def add(key: str, label: str, value: float, chg: float,
            detail: str = "", tone: str = "neutral") -> None:
        st.readings[key] = MacroReading(key, label, value, chg, detail, tone)

    # --- VIX ve vade yapısı ---
    vix = close("VIX")
    vix3m = close("VIX3M")
    if vix is not None:
        v = _last(vix)
        if v < 15:
            tone, det = "good", "Rehavet — koruma ucuz, ani şoklara açık"
        elif v < 22:
            tone, det = "neutral", "Normal aralık"
        elif v < 30:
            tone, det = "bad", "Gerginlik — pozisyon boyutunu düşür"
        else:
            tone, det = "bad", "Panik bölgesi — dip arayışı erken"
        add("VIX", "VIX", v, _pct_change(vix), det, tone)
        st.scores["vix"] = float(np.clip((25 - v) / 15 * 100, 0, 100))

        if vix3m is not None:
            v3 = _last(vix3m)
            ratio = v / v3 if v3 else np.nan
            if np.isfinite(ratio):
                inverted = ratio > 1.0
                ts_series = (vix / vix3m.reindex(vix.index).ffill()).dropna()
                ts_chg = _pct_change(ts_series, 5) if len(ts_series) > 6 else 0.0
                add("VIX_TS", "VIX Vade Yapısı", ratio,
                    ts_chg if np.isfinite(ts_chg) else 0.0,
                    "TERSİNE DÖNMÜŞ — yakın vade korkusu uzun vadeyi aştı, "
                    "kısa vadeli stres" if inverted else
                    "Normal contango — piyasa sakin",
                    "bad" if inverted else "good")
                st.scores["vix_ts"] = float(np.clip((1.05 - ratio) / 0.25 * 100, 0, 100))

    # --- SPY trend rejimi ---
    spy = close("SPY")
    if spy is not None and len(spy) > 200:
        e50 = spy.ewm(span=50, adjust=False).mean()
        e200 = spy.ewm(span=200, adjust=False).mean()
        px, m50, m200 = _last(spy), _last(e50), _last(e200)
        above50, above200 = px > m50, px > m200
        golden = m50 > m200
        if above50 and above200 and golden:
            det, tone, sc = "Tam boğa dizilimi (fiyat > 50 EMA > 200 EMA)", "good", 90
        elif above200 and not above50:
            det, tone, sc = "Ana trend yukarı ama kısa vadede zayıf — long sinyal kalitesi düşer", "neutral", 55
        elif not above200 and above50:
            det, tone, sc = "Ayı piyasasında toparlanma — dikkatli", "neutral", 40
        else:
            det, tone, sc = "Fiyat 50 ve 200 EMA altında — long sinyalleri güvenilmez", "bad", 15
        add("SPY_TREND", "SPY Trend Rejimi", (px / m50 - 1) * 100, _pct_change(spy),
            det, tone)
        st.scores["trend"] = float(sc)
        st.readings["SPY_TREND"].detail = det

    # --- Kredi iştahı: HYG / TLT ---
    hyg, tlt = close("HYG"), close("TLT")
    if hyg is not None and tlt is not None:
        rel = (hyg / tlt.reindex(hyg.index).ffill()).dropna()
        if len(rel) > 25:
            mom = _pct_change(rel, 20)
            tone = "good" if mom > 0 else "bad"
            add("CREDIT", "Kredi İştahı (HYG/TLT)", mom, _pct_change(rel, 5),
                "Riskli tahvil güvenli tahvile göre güçleniyor — risk alma isteği var"
                if mom > 0 else
                "Sermaye güvenli tahvile kaçıyor — riskten kaçınma", tone)
            st.scores["credit"] = float(np.clip(50 + mom * 8, 0, 100))

    # --- Dolar likiditesi ---
    dxy = close("DXY")
    if dxy is not None:
        mom = _pct_change(dxy, 20)
        add("DXY", "Dolar Endeksi", _last(dxy), mom,
            "Dolar güçleniyor — küresel likidite sıkışıyor, riskli varlık aleyhine"
            if mom > 0 else
            "Dolar zayıflıyor — likidite gevşiyor, riskli varlık lehine",
            "bad" if mom > 1 else "good" if mom < -1 else "neutral")
        st.scores["dollar"] = float(np.clip(50 - mom * 10, 0, 100))

    # --- Altın / Bakır: korku mu büyüme mi ---
    gold, copper = close("GOLD"), close("COPPER")
    if gold is not None and copper is not None:
        rel = (gold / copper.reindex(gold.index).ffill()).dropna()
        if len(rel) > 25:
            mom = _pct_change(rel, 20)
            add("GOLD_COPPER", "Altın / Bakır", mom, _pct_change(rel, 5),
                "Altın bakırı geçiyor — korku ve durgunluk fiyatlanıyor" if mom > 0
                else "Bakır altını geçiyor — büyüme ve sanayi talebi fiyatlanıyor",
                "bad" if mom > 2 else "good" if mom < -2 else "neutral")
            st.scores["growth"] = float(np.clip(50 - mom * 5, 0, 100))

    # --- Kripto risk iştahı ---
    btc = close("BTC")
    if btc is not None:
        mom = _pct_change(btc, 20)
        add("BTC", "Bitcoin", _last(btc), _pct_change(btc),
            "Risk iştahının ucu güçlü" if mom > 0 else "Riskli uç zayıflıyor",
            "good" if mom > 0 else "bad")
        st.scores["crypto"] = float(np.clip(50 + mom * 2, 0, 100))

    # --- 10 yıllık faiz ---
    tnx = close("US10Y")
    if tnx is not None:
        v = _last(tnx) / 10.0        # ^TNX 10 katı olarak gelir
        mom = _pct_change(tnx, 20)
        add("US10Y", "ABD 10Y Faiz", v, mom,
            "Faizler yükseliyor — yüksek çarpanlı hisseler baskı altında"
            if mom > 0 else "Faizler geriliyor — büyüme hisseleri rahatlar",
            "bad" if mom > 3 else "good" if mom < -3 else "neutral")
        st.scores["rates"] = float(np.clip(50 - mom * 3, 0, 100))

    # --- Piyasa genişliği: eşit ağırlıklı S&P / SPY ---
    rsp = close("RSP")
    if rsp is not None and spy is not None:
        rel = (rsp / spy.reindex(rsp.index).ffill()).dropna()
        if len(rel) > 25:
            mom = _pct_change(rel, 20)
            add("BREADTH", "Piyasa Genişliği (RSP/SPY)", mom, _pct_change(rel, 5),
                "Ortalama hisse endeksten iyi — yükseliş tabana yayılmış, "
                "sağlıklı" if mom > 0 else
                "Endeksi birkaç dev taşıyor — yükseliş dar tabanlı, kırılgan",
                "good" if mom > 0 else "bad")
            st.scores["breadth"] = float(np.clip(50 + mom * 10, 0, 100))

    # --- Riskin ucu: küçük ölçek / SPY ---
    iwm = close("IWM")
    if iwm is not None and spy is not None:
        rel = (iwm / spy.reindex(iwm.index).ffill()).dropna()
        if len(rel) > 25:
            mom = _pct_change(rel, 20)
            add("SMALLCAP", "Küçük Ölçek (IWM/SPY)", mom, _pct_change(rel, 5),
                "Küçük ölçek endeksi geçiyor — risk iştahı gerçek, likidite bol"
                if mom > 0 else
                "Sermaye büyük ve likit isimlere sığınıyor — savunmacı ralli",
                "good" if mom > 0 else "bad")
            st.scores["smallcap"] = float(np.clip(50 + mom * 8, 0, 100))

    # --- Tüketici sinyali: ihtiyari / temel ---
    xly, xlp = close("XLY"), close("XLP")
    if xly is not None and xlp is not None:
        rel = (xly / xlp.reindex(xly.index).ffill()).dropna()
        if len(rel) > 25:
            mom = _pct_change(rel, 20)
            add("CONSUMER", "İhtiyari / Temel Tüketim", mom, _pct_change(rel, 5),
                "İhtiyari tüketim defansifi geçiyor — büyüme bekleniyor"
                if mom > 0 else
                "Sermaye defansif tüketime kaçıyor — büyüme beklentisi zayıflıyor",
                "good" if mom > 0 else "bad")
            st.scores["consumer"] = float(np.clip(50 + mom * 6, 0, 100))

    # --- Getiri eğrisi: 10Y − 3A ---
    irx = close("US3M")
    if tnx is not None and irx is not None:
        y10 = _last(tnx) / 10.0
        y3m = _last(irx) / 10.0
        if np.isfinite(y10) and np.isfinite(y3m):
            spread = y10 - y3m
            # 20 gün önceki eğim — dikleşiyor mu düzleşiyor mu
            prev = np.nan
            t10 = tnx.dropna()
            t3 = irx.dropna()
            if len(t10) > 21 and len(t3) > 21:
                prev = float(t10.iloc[-21] / 10.0 - t3.iloc[-21] / 10.0)
            delta = (spread - prev) if np.isfinite(prev) else 0.0
            if spread < 0:
                det = ("TERS EĞRİ — kısa vade uzun vadeden pahalı. Tarihsel "
                       "olarak durgunluk öncüsü; bankalar ve döngüsel sektörler "
                       "baskı görür.")
                tone = "bad"
            elif spread < 0.5:
                det = ("Eğri neredeyse düz — büyüme beklentisi zayıf, "
                       "faiz makası bankaları sıkıştırıyor.")
                tone = "neutral"
            else:
                det = ("Eğri normal/dik — büyüme fiyatlanıyor, banka ve "
                       "döngüsel sektörler lehine.")
                tone = "good"
            if np.isfinite(prev):
                det += (" Son 1 ayda dikleşiyor." if delta > 0.05
                        else " Son 1 ayda düzleşiyor." if delta < -0.05
                        else "")
            add("CURVE", "Getiri Eğrisi (10Y − 3A)", spread, delta * 100, det, tone)
            st.scores["curve"] = float(np.clip(50 + spread * 25, 0, 100))

    if not st.scores:
        st.errors.append("Hiçbir makro gösterge çekilemedi — rejim hesaplanamadı.")
        st.battery = {"Hisse": 50, "Tahvil": 50, "Kripto": 50,
                      "Emtia": 50, "Gayrimenkul": 50}
        return st

    st.risk_score = float(np.mean(list(st.scores.values())))
    st.regime, st.regime_desc = classify_regime(st)
    st.battery = compute_battery(st)
    return st


def classify_regime(st: MacroState) -> tuple[str, str]:
    """Ölçümlerden rejim etiketi türetir. Sıra önemlidir: en keskin durum önce."""
    vix = st.get("VIX")
    ts = st.get("VIX_TS")
    trend = st.scores.get("trend", 50)
    credit = st.scores.get("credit", 50)
    dollar = st.scores.get("dollar", 50)
    risk = st.risk_score

    if np.isfinite(vix) and vix > 30 and trend < 30:
        return ("🩸 LİKİDİTE KRİZİ",
                f"VIX {vix:.1f} ile panik bölgesinde ve SPY ana trendin altında. "
                f"Yüksek çarpanlı teknoloji, biyoteknoloji ve kriptoda margin call "
                f"döngüsü riski var. Bu rejimde dip alımı erkendir; nakit ve uzun "
                f"vadeli tahvil korunma sağlar.")

    if np.isfinite(ts) and ts > 1.0 and credit < 45:
        return ("🌍 JEOPOLİTİK / OLAY ŞOKU",
                f"VIX vade yapısı tersine dönmüş (oran {ts:.2f}) ve kredi iştahı "
                f"zayıf. Sermaye teknolojiden kaçıp altın, savunma, petrol ve "
                f"hazineye sığınıyor. Tedarik zincirine bağlı şirketler en hızlı "
                f"ezilen taraf.")

    if st.opex_days <= 3:
        quad = " ÜÇLÜ CADI (quad witching) — etki normalden güçlü." if st.opex_quad else ""
        return ("🎯 OPEX PINNING",
                f"Opsiyon vadesine {st.opex_days} gün kaldı ({st.opex_date}).{quad} "
                f"Market maker'lar primi sıfırlamak için endeksi en yüksek açık "
                f"pozisyonun olduğu Max Pain noktasına hapsetmeye çalışıyor. Trend "
                f"kırılımları bu pencerede çoğunlukla tuzak (whipsaw) çıkar; vade "
                f"geçmeden yeni pozisyon açmak risklidir.")

    if np.isfinite(vix) and vix < 15 and trend > 70 and risk > 62:
        return ("🚀 RİSK İŞTAHI / GAMMA",
                f"VIX {vix:.1f} ile rehavet bölgesinde, SPY tam boğa diziliminde ve "
                f"kredi iştahı açık. Dealer'lar call hedge'i için spot almak zorunda "
                f"kaldığında parabolik hızlanma (gamma squeeze) görülebilir. Ancak "
                f"koruma ucuzken piyasa şoka da en açık haldedir.")

    if st.fomc_days is not None and st.fomc_days <= 3:
        return ("🏦 FOMC BEKLEYİŞİ",
                f"FOMC toplantısına {st.fomc_days} gün kaldı ({st.fomc_date}). "
                f"Karar öncesi hacim düşer, karar anında oynaklık patlar. Stop "
                f"mesafeleri normal ATR'ye göre dar kalır; pozisyon boyutunu "
                f"küçültmek stop mantığını korur.")

    if dollar < 35 and credit < 45:
        return ("💵 DOLAR SIKIŞMASI",
                "Dolar güçlenirken kredi iştahı zayıflıyor. Küresel likidite "
                "çekiliyor; yüksek çarpanlı ve kaldıraçlı varlıklar önce satılır.")

    if risk >= 62 and trend >= 50:
        return ("🟢 RİSK AÇIK",
                f"Bileşik risk skoru {risk:.0f}/100 ve SPY ana trendi yukarı. "
                f"Trend, kredi ve oynaklık birlikte long tarafını destekliyor — "
                f"sinyal kalitesi yüksek.")
    if risk >= 62 and trend < 50:
        return ("🟡 KARIŞIK — TREND ZAYIF",
                f"Kredi iştahı ve oynaklık iyi görünüyor (bileşik skor {risk:.0f}/100) "
                f"AMA fiyat SPY 50 EMA'sının altında. Bu ikisi çeliştiğinde rejim "
                f"kapısı kapalı sayılır: long sinyalleri izleme listesi olarak "
                f"kullanılır, pozisyon için trendin dönmesi beklenir.")
    if risk <= 40 or trend < 25:
        return ("🔴 RİSK KAPALI",
                f"Bileşik risk skoru {risk:.0f}/100. Long sinyallerinin isabet "
                f"oranı bu rejimde tarihsel olarak düşer; kısa vadeli işlemlerde "
                f"pozisyon boyutunu küçültmek gerekir.")
    return ("⚖️ GEÇİŞ / KARARSIZ",
            f"Bileşik risk skoru {risk:.0f}/100. Göstergeler birbiriyle çelişiyor; "
            f"tek yönlü agresif pozisyon için teyit bekleyin.")


def compute_battery(st: MacroState) -> dict[str, int]:
    """
    Varlık sınıflarına sermaye akış eğilimi (0–100).
    Elle yazılmış sabitler yerine ölçülen skorlardan türetilir.
    """
    trend = st.scores.get("trend", 50)
    vix = st.scores.get("vix", 50)
    credit = st.scores.get("credit", 50)
    dollar = st.scores.get("dollar", 50)
    growth = st.scores.get("growth", 50)
    crypto = st.scores.get("crypto", 50)
    rates = st.scores.get("rates", 50)
    breadth = st.scores.get("breadth", 50)
    smallcap = st.scores.get("smallcap", 50)
    consumer = st.scores.get("consumer", 50)
    curve = st.scores.get("curve", 50)

    def clip(x: float) -> int:
        return int(np.clip(round(x), 2, 98))

    # Genişlik ve küçük ölçek eklendi: endeksi birkaç dev taşıyorken "Hisse"
    # bataryasının dolu görünmesi yanıltıcıydı.
    hisse = (0.30 * trend + 0.20 * credit + 0.15 * vix + 0.10 * dollar
             + 0.15 * breadth + 0.10 * smallcap)
    tahvil = 100 - (0.45 * credit + 0.35 * vix + 0.20 * breadth)
    kripto = 0.35 * crypto + 0.25 * credit + 0.20 * dollar + 0.20 * smallcap
    emtia = 0.40 * (100 - growth) + 0.25 * dollar + 0.20 * trend + 0.15 * consumer
    gyo = 0.35 * rates + 0.25 * trend + 0.20 * credit + 0.20 * curve

    # OPEX haftasında her şey ortaya çekilir (pinning)
    if st.opex_days <= 2:
        pull = 0.35
        hisse = hisse * (1 - pull) + 50 * pull
        kripto = kripto * (1 - pull) + 50 * pull

    return {"Hisse": clip(hisse), "Tahvil": clip(tahvil), "Kripto": clip(kripto),
            "Emtia": clip(emtia), "Gayrimenkul": clip(gyo)}


# --------------------------------------------------------------------------
# Skor sözlüğü — "Ne değişti?" tablosunda okunur adlar
# --------------------------------------------------------------------------
SCORE_LABELS: dict[str, tuple[str, str]] = {
    "trend":    ("SPY Trend", "Fiyatın 50/200 EMA'ya göre dizilimi"),
    "vix":      ("Oynaklık (VIX)", "Düşük VIX = yüksek skor"),
    "vix_ts":   ("VIX Vade Yapısı", "VIX/VIX3M 1'in altındayken yüksek skor"),
    "credit":   ("Kredi İştahı", "HYG'nin TLT'ye göre 20 günlük momentumu"),
    "dollar":   ("Dolar Likiditesi", "DXY zayıfladıkça skor yükselir"),
    "growth":   ("Büyüme (Bakır/Altın)", "Bakır altını geçtikçe skor yükselir"),
    "crypto":   ("Kripto İştahı", "BTC 20 günlük momentumu"),
    "rates":    ("Faiz Yönü", "10Y gerilerken skor yükselir"),
    "breadth":  ("Piyasa Genişliği", "RSP/SPY — yükseliş tabana yayıldıkça yükselir"),
    "smallcap": ("Küçük Ölçek", "IWM/SPY — riskin ucundaki iştah"),
    "consumer": ("Tüketici Sinyali", "XLY/XLP — büyüme mi savunma mı"),
    "curve":    ("Getiri Eğrisi", "10Y − 3A; ters eğri düşük skor"),
}

# Sermaye akış bataryasının hangi sınıfı hangi skorlardan beslendiği
BATTERY_DRIVERS: dict[str, str] = {
    "Hisse": "SPY trend, kredi iştahı, VIX, dolar, genişlik, küçük ölçek",
    "Tahvil": "Kredi iştahı ve VIX'in tersi — korku arttıkça dolar",
    "Kripto": "BTC momentumu, kredi iştahı, dolar, küçük ölçek",
    "Emtia": "Altın/bakır, dolar, SPY trend, tüketici sinyali",
    "Gayrimenkul": "Faiz yönü, SPY trend, kredi iştahı, getiri eğrisi",
}


def _truncate(prices: dict[str, pd.DataFrame], back: int
              ) -> dict[str, pd.DataFrame]:
    """Tüm serileri `back` bar geriye keser — geçmişteki durumu yeniden kurar."""
    if back <= 0:
        return prices
    out: dict[str, pd.DataFrame] = {}
    for k, df in prices.items():
        if df is None or df.empty:
            out[k] = df
        elif len(df) > back:
            out[k] = df.iloc[:-back]
        else:
            out[k] = df.iloc[:0]
    return out


def state_at(prices: dict[str, pd.DataFrame], back: int,
             today: dt.date | None = None) -> MacroState:
    """
    `back` işlem barı önceki makro durumu.

    Neden gerekli: "Hisse bataryası 68" tek başına anlamsızdır — bir hafta önce
    52 miydi, 81 miydi karar buna bağlıdır. Aynı fonksiyonla geçmişi yeniden
    hesaplamak, ayrı bir kayıt dosyası tutmadan doğru karşılaştırma verir.
    """
    today = today or dt.date.today()
    ref = today - dt.timedelta(days=int(back * 7 / 5))     # işlem günü ≈ takvim
    return build_macro_state(_truncate(prices, back), today=ref)


PERIOD_BARS: dict[str, int] = {"1 gün": 1, "1 hafta": 5, "1 ay": 21}


def battery_changes(prices: dict[str, pd.DataFrame],
                    current: MacroState,
                    today: dt.date | None = None
                    ) -> tuple[pd.DataFrame, dict[str, MacroState]]:
    """
    Batarya ve risk skorunun 1 gün / 1 hafta / 1 ay önceki değerleri ve farkları.
    Döner: (tablo, {dönem: geçmiş durum}).
    """
    past: dict[str, MacroState] = {}
    for label, back in PERIOD_BARS.items():
        try:
            past[label] = state_at(prices, back, today)
        except Exception:                      # pragma: no cover - savunmacı
            continue

    rows: list[dict[str, Any]] = []
    for sinif, simdi in current.battery.items():
        row: dict[str, Any] = {"Varlık Sınıfı": sinif, "Şimdi": simdi}
        for label in PERIOD_BARS:
            p = past.get(label)
            onceki = p.battery.get(sinif) if p and p.battery else None
            row[f"{label} önce"] = onceki
            row[f"Δ {label}"] = (simdi - onceki) if onceki is not None else np.nan
        row["Besleyen"] = BATTERY_DRIVERS.get(sinif, "")
        rows.append(row)

    # Risk skoru da aynı tabloda taşınsın
    risk_row: dict[str, Any] = {"Varlık Sınıfı": "▸ Bileşik Risk Skoru",
                                "Şimdi": round(current.risk_score)}
    for label in PERIOD_BARS:
        p = past.get(label)
        onceki = round(p.risk_score) if p else None
        risk_row[f"{label} önce"] = onceki
        risk_row[f"Δ {label}"] = ((current.risk_score - p.risk_score)
                                  if p else np.nan)
    risk_row["Besleyen"] = "Tüm alt skorların ortalaması"
    rows.append(risk_row)
    return pd.DataFrame(rows), past


def score_changes(current: MacroState,
                  past: dict[str, MacroState]) -> pd.DataFrame:
    """Her alt skorun dönemsel değişimi — rejimi ne itiyor, ne çekiyor."""
    rows: list[dict[str, Any]] = []
    for key, val in current.scores.items():
        label, aciklama = SCORE_LABELS.get(key, (key, ""))
        row: dict[str, Any] = {"Skor": label, "Şimdi": val}
        for plabel in PERIOD_BARS:
            p = past.get(plabel)
            prev = p.scores.get(key) if p else None
            row[f"Δ {plabel}"] = (val - prev) if prev is not None else np.nan
        row["Ne ölçüyor"] = aciklama
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty and "Δ 1 hafta" in df.columns:
        df = df.sort_values("Δ 1 hafta", ascending=False, na_position="last")
    return df


def battery_history(prices: dict[str, pd.DataFrame], days: int = 60,
                    today: dt.date | None = None) -> pd.DataFrame:
    """
    Son `days` işlem günü için batarya ve risk skoru seyri.

    Her gün için tüm makro motoru yeniden çalıştırılır; böylece geçmiş,
    bugünün formülüyle tutarlı olur (kayıt dosyası tutulsaydı formül
    değiştiğinde geçmiş kırılırdı).
    """
    today = today or dt.date.today()
    ref = None
    for df in prices.values():
        if df is not None and not df.empty:
            ref = df.index
            break
    if ref is None:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for back in range(days, -1, -1):
        if back >= len(ref):
            continue
        try:
            s = state_at(prices, back, today)
        except Exception:                      # pragma: no cover
            continue
        if not s.battery:
            continue
        stamp = ref[-1 - back]
        rows.append({"Tarih": stamp, **s.battery,
                     "Risk Skoru": round(s.risk_score, 1),
                     "Rejim": s.regime})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.set_index("Tarih")
    return out


def regime_shifts(history: pd.DataFrame) -> list[dict[str, Any]]:
    """Seyir tablosundaki rejim değişim anları — 'ne zaman döndü' sorusu."""
    if history.empty or "Rejim" not in history:
        return []
    out: list[dict[str, Any]] = []
    prev = None
    for stamp, row in history.iterrows():
        cur = row["Rejim"]
        if prev is not None and cur != prev:
            out.append({"Tarih": stamp, "Önceki": prev, "Yeni": cur,
                        "Risk Skoru": row.get("Risk Skoru")})
        prev = cur
    return out


# --------------------------------------------------------------------------
# Elle seçilebilen senaryolar (referans / eğitim amaçlı)
# --------------------------------------------------------------------------
MANUAL_SCENARIOS: dict[str, dict[str, Any]] = {
    "🚀 GAMMA SQUEEZE": {
        "battery": {"Hisse": 95, "Tahvil": 20, "Kripto": 90, "Emtia": 55,
                    "Gayrimenkul": 65},
        "desc": "Beklenmedik güvercin FED açıklaması, zayıf enflasyon verisi veya "
                "yoğun call alımı sonrası piyasa yapıcıların delta-hedge için spot "
                "hisseye saldırması. Hızlı şişme yaratır ama temele dayanmadığı "
                "için sert düzeltme olasılığı masadadır.",
    },
    "🎯 OPEX PINNING": {
        "battery": {"Hisse": 50, "Tahvil": 50, "Kripto": 48, "Emtia": 52,
                    "Gayrimenkul": 50},
        "desc": "Market maker'lar primleri (theta) sıfırlamak için endeksi en "
                "yüksek açık pozisyon yoğunluğunun olduğu Max Pain noktasına "
                "hapseder. Trend kırılımları çoğunlukla tuzak çıkar.",
    },
    "🌍 JEOPOLİTİK ŞOK": {
        "battery": {"Hisse": 25, "Tahvil": 85, "Kripto": 35, "Emtia": 95,
                    "Gayrimenkul": 40},
        "desc": "Boğaz krizleri, gümrük tarifeleri veya enerji nakil hatlarına "
                "saldırı. Sermaye riskten kaçıp altın, savunma, petrol ve hazineye "
                "sığınır; tedarik zincirine bağlı şirketler anında ezilir.",
    },
    "🏦 LİKİDİTE KRİZİ (FED)": {
        "battery": {"Hisse": 15, "Tahvil": 90, "Kripto": 10, "Emtia": 35,
                    "Gayrimenkul": 25},
        "desc": "İnatçı enflasyon, devasa tahvil ihracı veya Reverse Repo havuzunun "
                "kuruması. Yüksek F/K'lı teknoloji, biyoteknoloji ve kriptoda "
                "acımasız likidasyon ve margin call döngüleri.",
    },
}

REGIME_GLOSSARY: list[tuple[str, str]] = [
    ("OPEX Pinning",
     "Aylık/üç aylık opsiyon vadesine yaklaşırken market maker'lar fiyatı "
     "yatırımcıların büyük kısmının kaybedeceği Max Pain noktasına çeker. "
     "Sahte kırılımlar artar, trend takip sistemleri bu pencerede kötü çalışır."),
    ("Gamma Squeeze",
     "Yoğun call alımı sonrası dealer'ların hedge amaçlı spot hisse almak zorunda "
     "kalmasıyla oluşan parabolik yükseliş döngüsü. Kendi kendini besler, "
     "beslediği kadar da hızlı çöker."),
    ("VIX Vade Yapısı",
     "VIX (30 gün) / VIX3M (3 ay) oranı 1'in üstüne çıkarsa yakın vade korkusu "
     "uzun vadeyi aşmış demektir — kısa vadeli stres göstergesi. Normalde bu oran "
     "1'in altındadır (contango)."),
    ("Kredi İştahı (HYG/TLT)",
     "Yüksek getirili şirket tahvilinin uzun vadeli hazineye göre performansı. "
     "Yükseliyorsa piyasa risk almaya istekli, düşüyorsa sermaye güvenliğe kaçıyor. "
     "Hisse senedi rallilerinin en güvenilir teyit göstergelerinden biridir."),
    ("Altın / Bakır Oranı",
     "Altın korkunun, bakır sanayi büyümesinin göstergesidir. Oran yükseliyorsa "
     "piyasa durgunluk, düşüyorsa büyüme fiyatlıyor."),
    ("Dolar Endeksi (DXY)",
     "Dolar güçlendikçe küresel likidite sıkışır; gelişen piyasalar, emtia ve "
     "yüksek çarpanlı büyüme hisseleri baskı görür."),
]

# ==========================================================================
# KAYNAK: apex/store.py
# ==========================================================================


import base64
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import requests

_log = logging.getLogger(__name__)

API = "https://api.github.com"
DEFAULT_PATH = "apex_watchlist.json"


class StorageError(RuntimeError):
    pass


@dataclass
class LoadResult:
    data: Any
    sha: str | None
    backend: str          # "github" | "local"
    message: str = ""


class Storage:
    """GitHub'a yazar; yapılandırma yoksa yerel dosyaya düşer."""

    def __init__(self, config: dict[str, Any] | None = None,
                 local_path: str = DEFAULT_PATH):
        cfg = dict(config or {})
        self.token = (cfg.get("token") or "").strip()
        self.repo = (cfg.get("repo") or "").strip()
        self.branch = (cfg.get("branch") or "main").strip()
        self.path = (cfg.get("path") or local_path).strip()
        self.local_path = local_path
        self.committer_name = cfg.get("committer_name") or "aether-nexus-bot"
        self.committer_email = cfg.get("committer_email") or "bot@users.noreply.github.com"
        self._sha: str | None = None

    # -- durum -------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return bool(self.token and self.repo)

    @property
    def backend(self) -> str:
        return "github" if self.enabled else "local"

    def describe(self) -> str:
        if self.enabled:
            return f"GitHub → {self.repo}@{self.branch}/{self.path}"
        return f"Yerel dosya → {self.local_path} (kalıcı değil!)"

    # -- iç yardımcılar ----------------------------------------------------
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def _url(self) -> str:
        return f"{API}/repos/{self.repo}/contents/{self.path}"

    # -- okuma -------------------------------------------------------------
    def load(self, default: Any = None) -> LoadResult:
        if default is None:
            default = []

        if self.enabled:
            try:
                r = requests.get(self._url(), headers=self._headers(),
                                 params={"ref": self.branch}, timeout=20)
                if r.status_code == 404:
                    self._sha = None
                    return LoadResult(default, None, "github",
                                      "Depoda dosya yok, ilk kayıtta oluşturulacak.")
                r.raise_for_status()
                payload = r.json()
                self._sha = payload.get("sha")
                raw = base64.b64decode(payload.get("content", "")).decode("utf-8")
                return LoadResult(json.loads(raw or "[]"), self._sha, "github")
            except Exception as exc:
                _log.error("GitHub okuma hatası: %s", exc)
                raise StorageError(f"GitHub'dan okunamadı: {exc}") from exc

        if os.path.exists(self.local_path):
            with open(self.local_path, "r", encoding="utf-8") as f:
                return LoadResult(json.load(f), None, "local")
        return LoadResult(default, None, "local", "Yerel dosya bulunamadı.")

    # -- yazma -------------------------------------------------------------
    def save(self, data: Any, message: str = "portföy güncellendi") -> str:
        body = json.dumps(data, indent=2, ensure_ascii=False) + "\n"

        # Yerel kopya her zaman yazılır (aynı oturumda hızlı okuma için)
        try:
            with open(self.local_path, "w", encoding="utf-8") as f:
                f.write(body)
        except OSError as exc:
            _log.warning("Yerel kopya yazılamadı: %s", exc)

        if not self.enabled:
            return "local"

        payload = {
            "message": message,
            "content": base64.b64encode(body.encode("utf-8")).decode("ascii"),
            "branch": self.branch,
            "committer": {"name": self.committer_name, "email": self.committer_email},
        }

        for attempt in range(2):
            if self._sha:
                payload["sha"] = self._sha
            else:
                payload.pop("sha", None)
            r = requests.put(self._url(), headers=self._headers(),
                             json=payload, timeout=25)
            if r.status_code in (200, 201):
                self._sha = (r.json().get("content") or {}).get("sha")
                return self._sha or "ok"
            if r.status_code == 409 and attempt == 0:
                # Başka bir yerden commit gelmiş; sha'yı tazeleyip bir kez daha dene
                _log.info("GitHub 409 çakışması, sha tazeleniyor.")
                try:
                    self.load()
                except StorageError:
                    pass
                continue
            raise StorageError(
                f"GitHub'a yazılamadı (HTTP {r.status_code}): {r.text[:300]}"
            )
        raise StorageError("GitHub'a yazılamadı: çakışma çözülemedi.")


def storage_from_secrets(secrets: Any, local_path: str = DEFAULT_PATH) -> Storage:
    """st.secrets nesnesinden Storage üretir; bölüm yoksa yerel moda düşer."""
    cfg: dict[str, Any] = {}
    try:
        if secrets is not None and "github" in secrets:
            cfg = dict(secrets["github"])
    except Exception as exc:
        _log.info("secrets okunamadı: %s", exc)
    return Storage(cfg, local_path=local_path)

# ==========================================================================
# KAYNAK: apex/data.py
# ==========================================================================


import datetime as dt
import logging
import time
from typing import Iterable

import pandas as pd

_log = logging.getLogger(__name__)

# yfinance aralığı -> (geçmiş gün sayısı, yf interval)
INTERVAL_PLAN: dict[str, tuple[int, str]] = {
    "1d": (420, "1d"),
    "1d_long": (5800, "1d"),   # Karar Hunisi: çeyreklik oran grafikleri
    "1wk": (1500, "1wk"),
    "4h": (170, "1h"),      # 1h çekip 4h'e yeniden örnekliyoruz
    "1h": (60, "1h"),
}


def _extract(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """yf.download çıktısından tek sembolün OHLCV tablosunu ayıklar."""
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            lv0 = raw.columns.get_level_values(0)
            lv1 = raw.columns.get_level_values(1)
            if ticker in set(lv0):
                df = raw[ticker].copy()
            elif ticker in set(lv1):
                df = raw.xs(ticker, level=1, axis=1).copy()
            else:
                return pd.DataFrame()
        else:
            df = raw.copy()
    except Exception as exc:
        _log.warning("Sütun ayıklama hatası (%s): %s", ticker, exc)
        return pd.DataFrame()

    need = ["Open", "High", "Low", "Close", "Volume"]
    if not all(col in df.columns for col in need):
        return pd.DataFrame()
    return df[need].dropna(subset=["Close"])


def resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return df.resample("4h").agg({"Open": "first", "High": "max", "Low": "min",
                                  "Close": "last", "Volume": "sum"}).dropna()


def fetch(tickers: Iterable[str], interval: str = "1d",
          retries: int = 1) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """
    Sembolleri toplu çeker. Döner: ({sembol: OHLCV}, [başarısız semboller])
    """
    tickers = sorted({t.strip().upper() for t in tickers if t and t.strip()})
    if not tickers:
        return {}, []

    import yfinance as yf

    days, yf_int = INTERVAL_PLAN.get(interval, INTERVAL_PLAN["1d"])
    start = dt.datetime.now() - dt.timedelta(days=days)
    out: dict[str, pd.DataFrame] = {}

    for attempt in range(retries + 1):
        missing = [t for t in tickers if t not in out]
        if not missing:
            break
        try:
            raw = yf.download(tickers=" ".join(missing), start=start,
                              interval=yf_int, group_by="column",
                              auto_adjust=False, progress=False, threads=True)
            for t in missing:
                df = _extract(raw, t)
                if len(df) >= 30:
                    out[t] = resample_4h(df) if interval == "4h" else df
        except Exception as exc:
            _log.warning("Toplu çekim hatası (deneme %s): %s", attempt + 1, exc)
            time.sleep(1.5 * (attempt + 1))

    # kalanları tek tek dene
    for t in [t for t in tickers if t not in out]:
        try:
            hist = yf.Ticker(t).history(start=start, interval=yf_int)
            if len(hist) >= 30:
                need = ["Open", "High", "Low", "Close", "Volume"]
                if all(c in hist.columns for c in need):
                    df = hist[need].dropna(subset=["Close"])
                    out[t] = resample_4h(df) if interval == "4h" else df
        except Exception as exc:
            _log.info("Tekil çekim hatası (%s): %s", t, exc)

    failed = [t for t in tickers if t not in out]
    return out, failed


# --------------------------------------------------------------------------
# yfinance `info` — ortak, hız sınırına dayanıklı çağrı
# Yahoo, bulut sunucularından gelen yoğun isteklere geçici blok (HTTP 429 /
# "Too Many Requests") uygular. Bu yüzden: aynı anda en fazla 2 istek,
# blokta artan beklemeyle yeniden deneme ve 6 saatlik bellek içi önbellek
# (bilanço sekmesi ile short interest aynı veriyi paylaşır).
# --------------------------------------------------------------------------
import threading

_INFO_CACHE: dict[str, tuple[float, dict]] = {}
_INFO_SEM = threading.Semaphore(2)
_INFO_TTL = 6 * 3600


def _info_ok(info: dict) -> bool:
    return bool(info) and any(info.get(k) for k in (
        "currentPrice", "regularMarketPrice", "previousClose", "quoteType"))


def yf_info(t: str) -> dict:
    import yfinance as yf

    now = time.time()
    hit = _INFO_CACHE.get(t)
    if hit and now - hit[0] < _INFO_TTL:
        return hit[1]
    info: dict = {}
    for attempt in range(4):
        err = None
        with _INFO_SEM:
            try:
                info = yf.Ticker(t).info or {}
            except Exception as exc:
                err, info = exc, {}
        if _info_ok(info):
            _INFO_CACHE[t] = (now, info)
            return info
        txt = f"{type(err).__name__} {err}" if err else ""
        limited = "Rate" in txt or "Too Many" in txt or "429" in txt
        time.sleep((3 * (attempt + 1)) if limited else 0.4)
    return info


def _one_fundamental(t: str, today: dt.date) -> dict:
    import numpy as np
    import yfinance as yf

    rec = {"Hisse": t, "Bilanço": "—", "Kalan Gün": None,
           "Fiyat": None, "Hedef": None, "Potansiyel %": None,
           "Analist": "", "_sort": 99999,
           "Sektör": "", "Endüstri": "", "Açıklama": "",
           "_ttm_sps": np.nan, "_fwd_sps": np.nan, "Büyüme %": np.nan,
           "Satış Tahmin Kaynağı": "", "Piyasa Değ. ($B)": np.nan,
           "İleri F/K": np.nan, "EV/Satış": np.nan, "İleri HBK": np.nan,
           "52H Konum %": np.nan}

    def num(x):
        try:
            v = float(x)
            return v if np.isfinite(v) else np.nan
        except (TypeError, ValueError):
            return np.nan

    try:
        tk = yf.Ticker(t)
        info = yf_info(t)

        # ---------- bilanço tarihi (eski mantık aynen) ----------
        first_date = None
        try:
            cal = tk.calendar
            if isinstance(cal, dict):
                d = cal.get("Earnings Date")
                first_date = d[0] if isinstance(d, (list, tuple)) and d else d
            elif hasattr(cal, "empty") and not cal.empty:
                if "Earnings Date" in getattr(cal, "columns", []):
                    first_date = cal["Earnings Date"].iloc[0]
                elif "Earnings Date" in getattr(cal, "index", []):
                    first_date = cal.loc["Earnings Date"].iloc[0]
        except Exception:
            pass
        if first_date is not None:
            ed = (first_date.date() if hasattr(first_date, "date")
                  else pd.to_datetime(first_date).date())
            delta = (ed - today).days
            rec["Bilanço"] = ed.strftime("%d.%m.%Y")
            if delta >= 0:
                rec["Kalan Gün"] = delta
                rec["_sort"] = delta
            else:
                rec["_sort"] = 90000 - delta

        price = info.get("currentPrice") or info.get("previousClose")
        target = info.get("targetMeanPrice") or info.get("targetMedianPrice")
        rec["Fiyat"] = float(price) if isinstance(price, (int, float)) else None
        rec["Hedef"] = float(target) if isinstance(target, (int, float)) else None
        if rec["Fiyat"] and rec["Hedef"]:
            rec["Potansiyel %"] = (rec["Hedef"] / rec["Fiyat"] - 1) * 100
        n = info.get("numberOfAnalystOpinions")
        key = info.get("recommendationKey", "")
        rec["Analist"] = (f"{key} ({n})" if n else str(key or ""))

        # ---------- P/S adil değer için ham veri ----------
        rec["Sektör"] = info.get("sector", "") or ""
        rec["Endüstri"] = info.get("industry", "") or ""
        summ = (info.get("longBusinessSummary") or "").strip()
        rec["Açıklama"] = (summ.split(". ")[0][:220] + ".") if summ else ""

        shares = num(info.get("sharesOutstanding"))
        ttm_rev = num(info.get("totalRevenue"))
        rps = num(info.get("revenuePerShare"))
        ttm_sps = rps if np.isfinite(rps) and rps > 0 else (
            ttm_rev / shares if shares and np.isfinite(ttm_rev) else np.nan)
        rec["_ttm_sps"] = ttm_sps

        fwd_rev, est_growth, kaynak = np.nan, np.nan, ""
        try:
            re_ = tk.revenue_estimate
            if re_ is not None and not re_.empty and "avg" in re_.columns:
                for idx, lab in (("+1y", "gelecek mali yıl"),
                                 ("0y", "bu mali yıl")):
                    if idx in re_.index and np.isfinite(num(re_.loc[idx, "avg"])):
                        fwd_rev, kaynak = num(re_.loc[idx, "avg"]), lab
                        if "growth" in re_.columns:
                            est_growth = num(re_.loc[idx, "growth"]) * 100
                        break
        except Exception:
            pass

        growth = num(info.get("revenueGrowth")) * 100
        if not np.isfinite(growth):
            growth = est_growth
        if np.isfinite(fwd_rev) and shares:
            rec["_fwd_sps"] = fwd_rev / shares
        elif np.isfinite(ttm_sps) and np.isfinite(growth):
            rec["_fwd_sps"] = ttm_sps * (1 + growth / 100)
            kaynak = "TTM × (1+büyüme) tahmini"
        if not np.isfinite(growth) and np.isfinite(rec["_fwd_sps"]) \
                and np.isfinite(ttm_sps) and ttm_sps > 0:
            growth = (rec["_fwd_sps"] / ttm_sps - 1) * 100
        rec["Büyüme %"] = growth
        rec["Satış Tahmin Kaynağı"] = kaynak

        mc = num(info.get("marketCap"))
        rec["Piyasa Değ. ($B)"] = mc / 1e9 if np.isfinite(mc) else np.nan
        rec["İleri F/K"] = num(info.get("forwardPE"))
        rec["EV/Satış"] = num(info.get("enterpriseToRevenue"))
        rec["İleri HBK"] = num(info.get("forwardEps"))
        hi, lo = num(info.get("fiftyTwoWeekHigh")), num(info.get("fiftyTwoWeekLow"))
        if rec["Fiyat"] and np.isfinite(hi) and np.isfinite(lo) and hi > lo:
            rec["52H Konum %"] = (rec["Fiyat"] - lo) / (hi - lo) * 100
    except Exception as exc:
        _log.info("Bilanço/temel veri alınamadı (%s): %s", t, exc)
    return rec


def fetch_earnings_calendar(tickers: Iterable[str]) -> pd.DataFrame:
    """
    Bilanço tarihleri, analist hedefleri ve P/S adil değer için ham temel veri.
    Adil değer sütunları app tarafında valuation.add_fair_values() ile eklenir.
    """
    from concurrent.futures import ThreadPoolExecutor

    today = dt.date.today()
    syms = sorted({x.strip().upper() for x in tickers if x and x.strip()})
    if not syms:
        return pd.DataFrame()
    with ThreadPoolExecutor(max_workers=2) as ex:
        rows = list(ex.map(lambda s: _one_fundamental(s, today), syms))
    df = pd.DataFrame(rows).sort_values("_sort").drop(columns=["_sort"])
    return df.reset_index(drop=True)

# ==========================================================================
# KAYNAK: apex/holdings.py
# ==========================================================================


from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd


# Hangi getiri sütunu hangi pencereyi temsil ediyor
PERIOD_COLS: dict[str, str] = {
    "1 Gün %": "1 gün",
    "1 Hafta %": "1 hafta",
    "1 Ay %": "1 ay",
}

# Akrana göre z eşiği: bu değerin altı 'geride', üstü 'lider'
Z_ESIK = 0.75
# Gürültü tabanı: dağılım çok darsa (herkes aynı) etiketleme yapma
MIN_FARK_PP = 1.0


@dataclass
class Verdict:
    key: str
    label: str
    icon: str
    renk: str
    aciklama: str
    aksiyon: str


VERDICTS: dict[str, Verdict] = {
    "lider": Verdict(
        "lider", "Sepeti taşıyan", "🟢", "#2fbe86",
        "Akranlarının belirgin üstünde getiri. ETF'in yükselişi büyük ölçüde "
        "bu isimlerden geliyor.",
        "Trend takip mantığı burada çalışır; ekleme yapılacaksa geri çekilmede "
        "yapılır, yeni tepede değil."),
    "lider_yorgun": Verdict(
        "lider_yorgun", "Lider ama yorgun", "🟠", "#c98500",
        "Akranların üstünde ama tükenme/risk bayrağı açık — hareket "
        "istatistiksel olarak uzamış.",
        "Yeni giriş için kötü nokta. Mevcut pozisyonda kısmi kâr al, iz süren "
        "stopu yukarı çek."),
    "uyumlu": Verdict(
        "uyumlu", "Sepetle uyumlu", "⚪", "#9a9aa8",
        "Getirisi akran medyanına yakın. Ayrışma yok, ETF ile birlikte "
        "hareket ediyor.",
        "Tek hisse tercih etmenin ek getirisi yok; sepetle aynı işi yapar."),
    "geride_akis_var": Verdict(
        "geride_akis_var", "Geride ama para giriyor", "🟡", "#3987e5",
        "Fiyat akranlarının gerisinde, fakat kurumsal akış hâlâ pozitif "
        "(WHALE yüksek/yükseliyor ya da toplama-süpürme sinyali var). "
        "Klasik gecikmeli katılım profili.",
        "Yakalama adayı. Rejim kapısı açıkken ve akış yönü ⇈/↗ iken izlenir; "
        "tetik, kendi direncinin hacimle kırılmasıdır."),
    "geride_akis_yok": Verdict(
        "geride_akis_yok", "Geride ve akış negatif", "🔴", "#e66767",
        "Hem fiyat akranlarının gerisinde hem kurumsal akış çıkışta "
        "(dağıtım/stealth çıkış ya da düşen WHALE). Geri kalması haklı.",
        "Ucuz görünmesi tuzak. ETF yükselirken bunu almak, sepetin en zayıf "
        "bacağını satın almaktır — akış dönene kadar uzak dur."),
}


# --------------------------------------------------------------------------
# Göreli güç hesapları
# --------------------------------------------------------------------------
def robust_z(values: pd.Series) -> pd.Series:
    """
    Medyan ve MAD tabanlı z skoru.

    Neden ortalama/standart sapma değil: bir ETF listesinde tek bir isim
    %40 kazanmışsa ortalama yukarı kayar ve sağlıklı isimler yapay olarak
    'geride' görünür. Medyan bu tek aykırı değerden etkilenmez.
    """
    v = pd.to_numeric(values, errors="coerce")
    ok = v.dropna()
    if len(ok) < 3:
        return pd.Series(np.nan, index=v.index)
    med = float(ok.median())
    mad = float((ok - med).abs().median()) * 1.4826
    if not np.isfinite(mad) or mad <= 1e-9:
        std = float(ok.std(ddof=0))
        if not np.isfinite(std) or std <= 1e-9:
            return pd.Series(0.0, index=v.index).where(v.notna())
        mad = std
    return (v - med) / mad


def classify_holding(z: float, fark_akran: float, d: dict[str, Any]) -> str:
    """Bir bileşeni beş durumdan birine yerleştirir."""
    akis_pozitif = (
        bool(d.get("_acc")) or bool(d.get("_st_in")) or bool(d.get("_sweep"))
        or bool(d.get("_dia_buy")) or bool(d.get("_star"))
        or (np.isfinite(d.get("WHALE", np.nan)) and d.get("WHALE", 0) >= 55
            and d.get("ΔWHALE", 0) >= 0)
        or (np.isfinite(d.get("ΔWHALE 5B", np.nan)) and d.get("ΔWHALE 5B", 0) > 2)
    )
    akis_negatif = (
        bool(d.get("_dist")) or bool(d.get("_st_out")) or bool(d.get("_risk_hard"))
        or bool(d.get("_dia_sell")) or bool(d.get("_smc_sell"))
        or (np.isfinite(d.get("WHALE", np.nan)) and d.get("WHALE", 100) < 45
            and d.get("ΔWHALE", 0) <= 0)
    )

    geride = (np.isfinite(z) and z <= -Z_ESIK
              and np.isfinite(fark_akran) and fark_akran <= -MIN_FARK_PP)
    onde = (np.isfinite(z) and z >= Z_ESIK
            and np.isfinite(fark_akran) and fark_akran >= MIN_FARK_PP)

    if geride:
        if akis_negatif and not akis_pozitif:
            return "geride_akis_yok"
        if akis_pozitif:
            return "geride_akis_var"
        return "geride_akis_yok" if not d.get("Rejim", True) else "geride_akis_var"
    if onde:
        if d.get("_exhausted") or d.get("_risk") or d.get("_risk_hard"):
            return "lider_yorgun"
        return "lider"
    return "uyumlu"


def build_holdings_table(
        scan_df: pd.DataFrame, etf_sym: str, period_col: str,
        etf_row: dict[str, Any] | None = None,
        meta_fn: Callable[[str, str], dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """
    Tarama çıktısından ETF içi göreli güç tablosu üretir.

    `scan_df` sadece bileşenleri içermelidir (ETF'in kendi satırı ayrı gelir).
    `etf_row` ETF'in kendi tarama satırıdır; yoksa ETF'e göre kıyas boş kalır.
    """
    if scan_df is None or scan_df.empty or period_col not in scan_df.columns:
        return pd.DataFrame()

    meta_fn = meta_fn or uni.holding_meta
    df = scan_df.copy()
    df = df[df["Sembol"].notna()]
    ret = pd.to_numeric(df[period_col], errors="coerce")
    etf_ret = np.nan
    if etf_row:
        etf_ret = pd.to_numeric(pd.Series([etf_row.get(period_col)]),
                                errors="coerce").iloc[0]

    med = float(ret.dropna().median()) if ret.notna().any() else np.nan
    z = robust_z(ret)

    out = pd.DataFrame(index=df.index)
    out["Sembol"] = df["Sembol"]
    metas = [meta_fn(etf_sym, s) for s in df["Sembol"]]
    out["Ağırlık %"] = [m.get("agirlik") for m in metas]
    out["Rol"] = [m.get("rol") or m.get("grup") or "" for m in metas]
    out["Getiri %"] = ret
    out["ETF %"] = etf_ret
    out["ETF'e Göre"] = ret - etf_ret
    out["Akran Medyanı %"] = med
    out["Akrana Göre"] = ret - med
    out["Z"] = z
    # Ağırlıklı katkı: bu isim ETF getirisinin kaç puanını açıkladı (yaklaşık)
    out["Katkı pp"] = [
        (w / 100.0 * r) if (w is not None and np.isfinite(r)) else np.nan
        for w, r in zip(out["Ağırlık %"], ret)
    ]

    recs = df.to_dict("records")
    durum_keys = [classify_holding(zz, ff, d)
                  for zz, ff, d in zip(out["Z"], out["Akrana Göre"], recs)]
    out["_durum"] = durum_keys
    out["Durum"] = [f"{VERDICTS[k].icon} {VERDICTS[k].label}" for k in durum_keys]

    for col in ["Sinyal", "Efor", "Fiyat", "WHALE", "ΔWHALE", "Whale Yön",
                "PRO-RET", "OMNI", "ΔOMNI", "OMNI Yön", "Boğa /6", "Ayı /6",
                "MAGNITUDE",
                "DIRECTION", "ATR %", "RS Sıra", "Stop", "T1", "T2",
                "Hacim ($M)", "Rejim", "Haftalık", "Hata"]:
        if col in df.columns:
            out[col] = df[col]

    out["Teknik Not"] = [technical_note(d) for d in recs]
    out["Neden"] = [
        lag_reason(k, d, zz) for k, d, zz in zip(durum_keys, recs, out["Z"])
    ]
    return out.sort_values("Z", ascending=False, na_position="last")


# --------------------------------------------------------------------------
# Açıklama üretimi
# --------------------------------------------------------------------------
def _fmt(x: float, spec: str = "+.1f") -> str:
    return format(x, spec) if np.isfinite(x) else "—"


def technical_note(d: dict[str, Any]) -> str:
    """
    Bir hissenin son teknik durumunu düz cümlelerle özetler.
    Motorun ürettiği ham sayılar burada okunur hâle gelir.
    """
    if d.get("Hata"):
        return f"Veri yok — {d['Hata']}"

    parts: list[str] = []

    # 1) Ana trend / rejim
    rejim = d.get("Rejim")
    hafta = d.get("Haftalık")
    t = "200 EMA **üstünde**" if rejim else "200 EMA **altında**"
    if hafta is True:
        t += ", haftalık trend teyitli"
    elif hafta is False:
        t += ", haftalık teyit yok"
    parts.append(f"Ana trend: {t}.")

    # 2) Kurumsal akış
    wh = d.get("WHALE", np.nan)
    dw = d.get("ΔWHALE", np.nan)
    yon = d.get("Whale Yön", "")
    pr = d.get("PRO-RET", np.nan)
    if np.isfinite(wh):
        akis = f"Kurumsal akış: WHALE {wh:.0f} ({_fmt(dw)} son barda, {yon})"
        if np.isfinite(pr):
            akis += f", PRO−RETAIL {_fmt(pr)}"
        parts.append(akis + ".")

    # 3) Momentum
    om = d.get("OMNI", np.nan)
    do = d.get("ΔOMNI", np.nan)
    oy = d.get("OMNI Yön", "")
    boga, ayi = d.get("Boğa /6"), d.get("Ayı /6")
    if np.isfinite(om):
        mom = f"Momentum: OMNI {om:.0f} ({_fmt(do)}, {oy})"
        if boga is not None and ayi is not None:
            baskin = ("boğa" if boga > ayi else "ayı" if ayi > boga else "dengede")
            mom += f", konsensüs boğa {boga}/6 · ayı {ayi}/6 ({baskin})"
        parts.append(mom + ".")

    # 4) Konfluans
    mag, dr = d.get("MAGNITUDE"), d.get("DIRECTION")
    if mag is not None and dr is not None:
        parts.append(f"Konfluans: MAGNITUDE {mag}/18, DIRECTION {dr:+d} "
                     f"(0'ın üstü alıcı baskısı).")

    # 5) Volatilite karakteri ve seviyeler
    atrp = d.get("ATR %", np.nan)
    if np.isfinite(atrp):
        kar = ("geniş ATR / yüksek beta" if atrp >= 3.5
               else "temiz trend bandı" if atrp >= 1.8 else "dar, defansif")
        parts.append(f"Oynaklık: ATR %{atrp:.1f} — {kar}.")

    price = d.get("Fiyat", np.nan)
    stop, t1, t2 = d.get("Stop", np.nan), d.get("T1", np.nan), d.get("T2", np.nan)
    if np.isfinite(price) and np.isfinite(stop) and price:
        risk = (price - stop) / price * 100
        if risk >= 0:
            seviye = f"Seviyeler: iz süren stop {stop:.2f} (%{risk:.1f} aşağıda)"
        else:
            # Ratchet stop yukarı çekildikten sonra fiyat altına düştüyse
            seviye = (f"Seviyeler: iz süren stop {stop:.2f} fiyatın "
                      f"%{abs(risk):.1f} ÜSTÜNDE — stop çoktan tetiklenmiş, "
                      f"long kurgu bu seviyenin geri alınmasına bağlı")
        hedefler = [f"{ad} {x:.2f}" + (" (fiyatın altında, geçilmiş hedef)"
                                       if np.isfinite(price) and x < price else "")
                    for ad, x in (("T1", t1), ("T2", t2)) if np.isfinite(x)]
        if hedefler:
            seviye += ", " + ", ".join(hedefler)
        parts.append(seviye + ".")

    # 6) Başlık sinyali ve olaylar
    olaylar = []
    for flag, isim in [("_sweep", "likidite süpürmesi"), ("_acc", "toplama"),
                       ("_dist", "dağıtım"), ("_st_in", "stealth giriş"),
                       ("_st_out", "stealth çıkış"), ("_ab_bull", "afterburner"),
                       ("_exhausted", "tükenme"), ("Sıkışma", "sıkışma"),
                       ("_mvp", "Minervini MVP"), ("_star", "golden star")]:
        if d.get(flag):
            olaylar.append(isim)
    if olaylar:
        parts.append("Açık bayraklar: " + ", ".join(olaylar) + ".")

    return " ".join(parts)


def lag_reason(durum: str, d: dict[str, Any], z: float) -> str:
    """Durum etiketinin tek cümlelik gerekçesi — tabloda hızlı okunsun diye."""
    wh, dw = d.get("WHALE", np.nan), d.get("ΔWHALE", np.nan)
    if durum == "geride_akis_var":
        if d.get("_sweep"):
            return "Geride ama likidite süpürmesi var — stop avı sonrası dönüş kalıbı"
        if d.get("_acc") or d.get("_st_in"):
            return "Geride ama toplama sinyali açık — sessiz birikim"
        return (f"Geride ama WHALE {wh:.0f} ({_fmt(dw)}) — akış hâlâ içeride"
                if np.isfinite(wh) else "Geride, akış bozulmamış")
    if durum == "geride_akis_yok":
        if d.get("_dist") or d.get("_st_out"):
            return "Geride ve dağıtım açık — düşüşün sebebi satış"
        if not d.get("Rejim", True):
            return "Geride ve 200 EMA altında — trend zaten aşağı"
        return (f"Geride, WHALE {wh:.0f} ({_fmt(dw)}) — akış da destek vermiyor"
                if np.isfinite(wh) else "Geride, akış desteği yok")
    if durum == "lider_yorgun":
        return "Akranların üstünde ama tükenme/risk bayrağı açık"
    if durum == "lider":
        return f"Akran medyanının {abs(z):.1f} MAD üstünde — sepeti taşıyor" \
            if np.isfinite(z) else "Akranların üstünde"
    return "Akran medyanına yakın — ayrışma yok"


def holdings_narrative(table: pd.DataFrame, etf_sym: str, period_col: str) -> list[str]:
    """Tablonun tepesine konacak 2–4 cümlelik canlı yorum."""
    if table.empty:
        return []
    lab = PERIOD_COLS.get(period_col, period_col)
    etf_ret = table["ETF %"].iloc[0] if "ETF %" in table else np.nan
    med = table["Akran Medyanı %"].iloc[0] if "Akran Medyanı %" in table else np.nan
    n = len(table)
    poz = int((pd.to_numeric(table["Getiri %"], errors="coerce") > 0).sum())

    out: list[str] = []
    if np.isfinite(etf_ret):
        katilim = f"{poz}/{n} bileşen artıda"
        out.append(
            f"**{etf_sym}** {lab} penceresinde **%{etf_ret:+.2f}**; içindeki "
            f"{n} hissenin medyanı **%{med:+.2f}** ve {katilim}. "
            + ("Yükselişi az sayıda isim taşıyor — katılım dar."
               if etf_ret > 0 and poz < n * 0.5 else
               "Katılım geniş, hareket sepetin geneline yayılmış."
               if etf_ret > 0 else
               "Sepet ekside; aşağıdaki ayrışma dip arayışı için kullanılır.")
        )
    else:
        out.append(f"**{etf_sym}** içindeki {n} hissenin {lab} medyanı "
                   f"**%{med:+.2f}**.")

    for key in ["geride_akis_var", "geride_akis_yok", "lider_yorgun"]:
        sel = table[table["_durum"] == key]
        if sel.empty:
            continue
        v = VERDICTS[key]
        isim = ", ".join(f"`{s}`" for s in sel["Sembol"].head(6))
        out.append(f"{v.icon} **{v.label}** ({len(sel)}): {isim} — {v.aksiyon}")

    return out


def holdings_counts(table: pd.DataFrame) -> dict[str, int]:
    if table.empty or "_durum" not in table:
        return {}
    return {k: int((table["_durum"] == k).sum()) for k in VERDICTS}

# ==========================================================================
# KAYNAK: apex/screener.py
# ==========================================================================


from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# İşlem karakteri sınıfları — ATR%'ye göre
KARAKTER_ESIK = {
    "yuksek_beta": 3.5,     # ATR% >= 3.5 -> geniş ATR, yüksek beta
    "orta": 1.8,            # 1.8–3.5 -> temiz trend, orta volatilite
}                            # < 1.8 -> sakin / defansif


@dataclass
class SwingFilters:
    """Kullanıcının pratik filtreleri — arayüzden açılıp kapatılır."""
    min_dollar_vol_m: float = 5.0      # günlük ortalama işlem hacmi ($M)
    earnings_buffer_days: int = 2      # bilanço ±N gün pozisyon açma
    require_regime: bool = True        # SPY 50 EMA rejim kapısı
    require_weekly: bool = False       # haftalık trend teyidi
    min_price: float = 5.0             # penny hisse eleme
    max_atr_pct: float = 12.0          # aşırı oynak olanları ele
    min_score: int = 0


def trade_character(atr_pct: float) -> str:
    if not np.isfinite(atr_pct):
        return "—"
    if atr_pct >= KARAKTER_ESIK["yuksek_beta"]:
        return "⚡ Yüksek beta / geniş ATR"
    if atr_pct >= KARAKTER_ESIK["orta"]:
        return "📈 Temiz trend / orta volatilite"
    return "🛡️ Sakin / defansif"


def character_note(atr_pct: float) -> str:
    if not np.isfinite(atr_pct):
        return ""
    if atr_pct >= KARAKTER_ESIK["yuksek_beta"]:
        return ("Kademeli ATR stop sistemi bu grupta anlamlı çalışır; hareket "
                "geniş olduğu için stop mesafesi de geniş tutulmalı.")
    if atr_pct >= KARAKTER_ESIK["orta"]:
        return ("Sinyal-gürültü oranı iyi, whipsaw az. Trend takip ve kırılım "
                "sistemleri bu bantta en verimli çalışır.")
    return ("Hareket dar; swing için getiri/risk zayıf kalabilir. Pozisyon "
            "boyutunu büyütmek yerine daha oynak bir aday aramak daha mantıklı.")


def swing_score(d: dict[str, Any]) -> int:
    """
    0–100 swing uygunluk skoru.
    Ağırlıklar V719 KONFLUANS'ın ölçülmüş lift değerleriyle uyumlu tutuldu:
    en yüksek katkı Afterburner ve MVP tarafında, teyit katmanları daha düşük.
    """
    s = 0.0
    # Konfluans motoru (en ağır bileşen)
    s += np.clip(d.get("MAGNITUDE", 0), 0, 18) / 18 * 22
    s += np.clip((d.get("DIRECTION", 0) + 5) / 10, 0, 1) * 18
    # Kurumsal akış
    s += np.clip(d.get("WHALE", 50), 0, 100) / 100 * 14
    s += np.clip((d.get("PRO-RET", 0) + 50) / 100, 0, 1) * 8
    # Momentum konsensüsü
    s += np.clip(d.get("OMNI", 50), 0, 100) / 100 * 10
    s += np.clip(d.get("Boğa /6", 0), 0, 6) / 6 * 8
    s += np.clip(d.get("Efor /8", 0), 0, 8) / 8 * 6
    # Göreli güç
    rs = d.get("RS Sıra", np.nan)
    if np.isfinite(rs):
        s += rs / 100 * 8
    # Olay primleri
    if d.get("_dia_buy"):
        s += 6
    if d.get("_sweep"):
        s += 5
    if d.get("_star"):
        s += 4
    if d.get("_ab_bull"):
        s += 4
    if d.get("_mvp"):
        s += 4
    if d.get("Sıkışma"):
        s += 3
    # Cezalar
    if d.get("_risk_hard"):
        s -= 25
    elif d.get("_risk"):
        s -= 12
    if d.get("_exhausted"):
        s -= 8
    if d.get("_dist") or d.get("_st_out"):
        s -= 10
    if not d.get("Rejim", True):
        s -= 6
    return int(np.clip(round(s), 0, 100))


def apply_filters(d: dict[str, Any], f: SwingFilters,
                  earnings_days: int | None,
                  market_regime_ok: bool) -> tuple[bool, list[str]]:
    """Filtreleri uygular. Döner: (geçti mi, [engel gerekçeleri])."""
    blocks: list[str] = []

    price = d.get("Fiyat", np.nan)
    if np.isfinite(price) and price < f.min_price:
        blocks.append(f"Fiyat ${price:.2f} < ${f.min_price:.0f}")

    dv = d.get("Hacim ($M)", np.nan)
    if np.isfinite(dv) and dv < f.min_dollar_vol_m:
        blocks.append(f"Likidite ${dv:.1f}M < ${f.min_dollar_vol_m:.0f}M "
                      f"(spread genişler, stop kayar)")

    atrp = d.get("ATR %", np.nan)
    if np.isfinite(atrp) and atrp > f.max_atr_pct:
        blocks.append(f"ATR %{atrp:.1f} aşırı oynak")

    if earnings_days is not None and abs(earnings_days) <= f.earnings_buffer_days:
        blocks.append(f"Bilanço {earnings_days} gün içinde — gap riski stop "
                      f"mantığını bozar")

    if f.require_regime and not market_regime_ok:
        blocks.append("Piyasa rejimi kapalı (SPY 50 EMA altında)")

    if f.require_regime and not d.get("Rejim", True):
        blocks.append("Hisse 200 EMA altında")

    if f.require_weekly and d.get("Haftalık") is False:
        blocks.append("Haftalık trend teyidi yok")

    if d.get("Skor", 0) < f.min_score:
        blocks.append(f"Skor {d.get('Skor', 0)} < {f.min_score}")

    return (not blocks), blocks


def build_recommendations(rows: list[dict[str, Any]], f: SwingFilters,
                          earnings_map: dict[str, int] | None,
                          market_regime_ok: bool) -> pd.DataFrame:
    """Sinyal satırlarını skorlayıp filtreleyerek tavsiye tablosu üretir."""
    earnings_map = earnings_map or {}
    out: list[dict[str, Any]] = []

    for d in rows:
        d = dict(d)
        d["Skor"] = swing_score(d)
        ed = earnings_map.get(d.get("Sembol", ""))
        ok, blocks = apply_filters(d, f, ed, market_regime_ok)
        price = d.get("Fiyat", np.nan)
        stop = d.get("Stop", np.nan)
        t1, t2 = d.get("T1", np.nan), d.get("T2", np.nan)
        risk = price - stop if np.isfinite(price) and np.isfinite(stop) else np.nan

        d["Uygun"] = ok
        d["Engel"] = " · ".join(blocks)
        d["Karakter"] = trade_character(d.get("ATR %", np.nan))
        d["Risk %"] = (risk / price * 100) if np.isfinite(risk) and price else np.nan
        d["R (T1)"] = ((t1 - price) / risk) if np.isfinite(risk) and risk > 0 else np.nan
        d["R (T2)"] = ((t2 - price) / risk) if np.isfinite(risk) and risk > 0 else np.nan
        d["Bilanço Gün"] = ed
        out.append(d)

    df = pd.DataFrame(out)
    if df.empty:
        return df
    return df.sort_values(["Uygun", "Skor"], ascending=[False, False])


# --------------------------------------------------------------------------
# Swing yorumu — canlı veriden üretilir
# --------------------------------------------------------------------------
def swing_commentary(df: pd.DataFrame, macro_regime: str,
                     regime_ok: bool, top_n: int = 6) -> dict[str, Any]:
    """
    Kullanıcının elle yazdığı swing notunun canlı veriyle yeniden üretilmiş hali:
    hangi hisseler hangi karakterde, hangi filtreler devrede, rejim ne diyor.
    """
    if df.empty:
        return {"gruplar": {}, "filtreler": [], "rejim": macro_regime, "notlar": []}

    uygun = df[df["Uygun"]] if "Uygun" in df else df
    gruplar: dict[str, list[dict[str, Any]]] = {}

    for karakter in ["⚡ Yüksek beta / geniş ATR",
                     "📈 Temiz trend / orta volatilite",
                     "🛡️ Sakin / defansif"]:
        sel = uygun[uygun["Karakter"] == karakter].head(top_n)
        if not sel.empty:
            gruplar[karakter] = sel[["Sembol", "Skor", "ATR %", "Sinyal",
                                     "RS Sıra"]].to_dict("records")

    filtreler = [
        ("Bilanço ±2 gün", "Kazanç tarihine 2 günden az kalan hisselerde pozisyon "
                           "açılmaz — gap riski ATR stop mantığını bozar."),
        ("Likidite > $5M", "Günlük ortalama işlem hacmi eşiğin altındaysa spread "
                           "genişler, stop gerçekleşen fiyattan uzağa kayar."),
        ("Rejim kapısı", "SPY 50 EMA altındayken long sinyallerin isabet oranı "
                         "düşer. Bu, likidite grabı korumasıyla aynı mantıkta ayrı "
                         "bir gating katmanıdır: sinyal doğru olsa da ortam yanlışsa "
                         "işlem açılmaz."),
        ("200 EMA teyidi", "Hissenin kendi ana trendi aşağıysa swing long, trende "
                           "karşı işlem olur."),
    ]

    notlar = []
    if not regime_ok:
        notlar.append(
            "⚠️ Piyasa rejimi şu an KAPALI (SPY 50 EMA altında). Bu pencerede long "
            "sinyallerin kalitesi tarihsel olarak düşer; tarama sonuçlarını izleme "
            "listesi olarak kullanın, pozisyon açmak için rejimin dönmesini bekleyin."
        )
    if "ETF" in df.columns:
        etf_sayisi = uygun["ETF"].nunique() if not uygun.empty else 0
        notlar.append(
            f"Sektör ETF'leri ({etf_sayisi} tema taranıyor) tek hisse haber riskini "
            f"seyreltir; çoklu-hisse istatistiği tek isimden daha temiz çıkar."
        )
    notlar.append(
        "Uzun vade ve swing listelerinin çakışması sorun değil, ancak pozisyonları "
        "ayrı hesapta tutmak gerekir — aksi halde swing stopu uzun vadeli tezi keser."
    )
    return {"gruplar": gruplar, "filtreler": filtreler,
            "rejim": macro_regime, "notlar": notlar}

# ==========================================================================
# KAYNAK: apex/news.py
# ==========================================================================


import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Iterable

import requests

_log = logging.getLogger(__name__)

FEEDS: dict[str, str] = {
    "Makro & Fed": "Fed OR FOMC OR inflation OR CPI OR rate cut stock market",
    "Tarife & Ticaret": "tariff OR trade war OR export controls chips",
    "Yapay Zekâ & Çip": "AI chips OR semiconductor OR data center capex",
    "Enerji & Nükleer": "nuclear power OR uranium OR grid OR utilities data center",
    "Savunma & Uzay": "defense contract OR space launch OR satellite",
    "Kripto": "bitcoin OR crypto ETF OR SEC crypto",
}

BASE = ("https://news.google.com/rss/search?q={q}"
        "&hl=en-US&gl=US&ceid=US:en")

IMPACT_RULES: list[tuple[tuple[str, ...], str]] = [
    (("tariff", "trade war", "export control", "sanction"),
     "🔴 Tedarik zinciri & Çin ithalatı | 🟢 İç üretim"),
    (("nuclear", "uranium", "smr", "reactor"),
     "🟢 Nükleer & Uranyum (URA, CEG, VST)"),
    (("data center", "hyperscaler", "capex"),
     "🟢 Veri merkezi zinciri (SRVR, XLU, PAVE)"),
    (("oil", "opec", "crude", "gas", "drill"),
     "🟢 Fosil yakıt (XLE, XOP) | 🔴 Temiz enerji"),
    (("crypto", "bitcoin", "sec ", "stablecoin"),
     "🟢 Kripto & Fintek (IBIT, WGMI, ARKF)"),
    (("defense", "military", "missile", "space"),
     "🟢 Savunma & Uzay (XAR, ARKX, UFO)"),
    (("fed", "fomc", "powell", "rate", "inflation", "cpi"),
     "📉 Likidite etkisi — tüm risk varlıkları"),
    (("ai ", "artificial intelligence", "chip", "semiconductor", "gpu"),
     "🟢 Çip & YZ altyapısı (SMH, SOXX, EUV)"),
    (("copper", "lithium", "rare earth", "mining"),
     "🟢 Emtia & Madencilik (COPX, LIT, REMX)"),
    (("layoff", "recession", "slowdown", "downgrade"),
     "🔴 Büyüme endişesi — döngüsel hisseler"),
]


def classify_impact(title: str) -> str:
    t = title.lower()
    for keys, impact in IMPACT_RULES:
        if any(k in t for k in keys):
            return impact
    return "⚖️ Nötr / sektörel rotasyon"


def fetch_news(known_tickers: Iterable[str], topics: Iterable[str] | None = None,
               per_feed: int = 8, timeout: int = 12) -> tuple[list[dict], list[str]]:
    """Seçili konu akışlarını çeker. Döner: (haberler, hatalar)."""
    tickers = sorted({t.upper() for t in known_tickers if t and len(t) >= 2})
    topics = list(topics or FEEDS)
    items: list[dict] = []
    errors: list[str] = []
    seen: set[str] = set()

    for topic in topics:
        q = FEEDS.get(topic)
        if not q:
            continue
        url = BASE.format(q=requests.utils.quote(q))
        try:
            resp = requests.get(url, timeout=timeout,
                                headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
        except Exception as exc:
            errors.append(f"{topic}: {exc}")
            continue

        for node in root.findall(".//item")[:per_feed]:
            title = (node.findtext("title") or "").strip()
            if not title or title in seen:
                continue
            seen.add(title)
            link = node.findtext("link") or ""
            pub = (node.findtext("pubDate") or "")[:22]

            hits = [t for t in tickers
                    if re.search(rf"\b{re.escape(t)}\b", title)]
            items.append({
                "Konu": topic,
                "Tarih": pub,
                "Başlık": title,
                "İlgili": ", ".join(hits[:6]) if hits else "Genel makro",
                "Etki": classify_impact(title),
                "Link": link,
            })

    items.sort(key=lambda r: _parse_date(r["Tarih"]), reverse=True)
    return items, errors


def _parse_date(s: str) -> datetime:
    for fmt in ("%a, %d %b %Y %H:%M:%S", "%a, %d %b %Y %H:%M"):
        try:
            return datetime.strptime(s.strip()[:len(fmt) + 2].strip(), fmt)
        except Exception:
            continue
    return datetime.min

# ==========================================================================
# KAYNAK: apex/playbook.py
# ==========================================================================


import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Driver:
    """Bir rejim sürücüsü (gamma, opex, vix, jeopolitik…)."""
    key: str
    label: str
    icon: str
    nedir: str
    tetikleyiciler: list[str]          # haber başlığında aranan ifadeler
    veri_isareti: str                  # hangi göstergeden anlaşılır
    lehte_etf: list[str] = field(default_factory=list)
    lehte_hisse: list[str] = field(default_factory=list)
    aleyhte_etf: list[str] = field(default_factory=list)
    aleyhte_hisse: list[str] = field(default_factory=list)
    islem_notu: str = ""


DRIVERS: dict[str, Driver] = {}


def _d(**kw) -> None:
    DRIVERS[kw["key"]] = Driver(**kw)


# ---------------------------------------------------------------- GAMMA
_d(
    key="gamma",
    label="Gamma Squeeze / Melt-Up",
    icon="🚀",
    nedir=(
        "Yoğun call opsiyon alımı sonrası piyasa yapıcılar (dealer) açığa "
        "sattıkları call'ları hedge etmek için spot hisse almak ZORUNDA kalır. "
        "Fiyat yükseldikçe hedge ihtiyacı büyür — kendi kendini besleyen yukarı "
        "sarmal oluşur. Temele değil opsiyon mekaniğine dayandığı için vade "
        "geçince aynı hızla sönebilir."
    ),
    tetikleyiciler=[
        "rate cut", "dovish", "soft cpi", "cooler inflation", "short squeeze",
        "record high", "melt up", "call volume", "retail buying", "meme stock",
        "beats estimates", "raises guidance", "blowout quarter", "squeeze",
    ],
    veri_isareti="VIX 15 altında + SPY tam boğa dizilimi + kredi iştahı açık",
    lehte_etf=["QQQ", "XLK", "SOXX", "SMH", "ARKF", "ARKX", "WGMI", "IBIT"],
    lehte_hisse=["NVDA", "AMD", "PLTR", "TSLA", "COIN", "MARA", "RIOT", "SMCI",
                 "IONQ", "RKLB"],
    aleyhte_etf=["TLT", "XLP", "XLU"],
    aleyhte_hisse=["KO", "PG", "JNJ"],
    islem_notu=(
        "Yüksek beta ve geniş ATR'li isimler en çok hareket edeni olur; ATR "
        "kademeli stop bu grupta anlamlı çalışır. Ancak koruma ucuzken piyasa "
        "şoka en açık haldedir — pozisyonu vade haftasına taşımayın."
    ),
)

# ---------------------------------------------------------------- OPEX
_d(
    key="opex",
    label="OPEX Pinning / Max Pain",
    icon="🎯",
    nedir=(
        "Aylık (üçüncü cuma) ve üç aylık opsiyon vadelerine yaklaşırken market "
        "maker'lar taşıdıkları pozisyonun primini (theta) sıfırlamak için "
        "endeksi en yüksek açık pozisyonun bulunduğu Max Pain seviyesine "
        "çekmeye çalışır. Fiyat dar bir bantta hapsolur, kırılımlar tuzağa "
        "dönüşür. Üçlü cadı (mart/haziran/eylül/aralık) aylarında etki en güçlü."
    ),
    tetikleyiciler=[
        "options expiration", "opex", "quad witching", "triple witching",
        "max pain", "open interest", "gamma exposure", "0dte",
    ],
    veri_isareti="OPEX'e 3 gün veya daha az kalması (üçlü cadı ayrıca işaretlenir)",
    lehte_etf=[],
    lehte_hisse=[],
    aleyhte_etf=["Tüm trend takip stratejileri"],
    aleyhte_hisse=[],
    islem_notu=(
        "Bu pencerede yeni kırılım pozisyonu AÇMAYIN. Sinyal doğru olsa bile "
        "fiyat vade sonuna kadar geri çekilip stopu tetikler. Mevcut pozisyonda "
        "stopu biraz genişletmek ya da kısmi kâr almak, yeni giriş yapmaktan "
        "daha mantıklıdır. Vade cumasından sonraki pazartesi bant çözülür."
    ),
)

# ---------------------------------------------------------------- VIX
_d(
    key="vix",
    label="Oynaklık Şoku / VIX Sıçraması",
    icon="⚡",
    nedir=(
        "VIX'in hızla yükselmesi ve özellikle VIX/VIX3M oranının 1'in üstüne "
        "çıkması (vade yapısının tersine dönmesi) yakın vadeli korkunun uzun "
        "vadeyi aştığını gösterir. Bu, kaldıraçlı fonların pozisyon küçültmeye "
        "zorlandığı andır — satış satışı besler."
    ),
    tetikleyiciler=[
        "vix", "volatility", "selloff", "plunge", "correction", "crash",
        "margin call", "risk off", "flight to safety", "circuit breaker",
    ],
    veri_isareti="VIX > 25 veya VIX/VIX3M oranı > 1.00",
    lehte_etf=["TLT", "GLD", "XLP", "XLU", "VIXY"],
    lehte_hisse=["NEM", "GOLD", "KO", "PG", "WMT"],
    aleyhte_etf=["ARKG", "ARKF", "XBI", "IWM", "WGMI", "SOXX"],
    aleyhte_hisse=["PLTR", "COIN", "MARA", "IONQ", "RKLB", "SMCI"],
    islem_notu=(
        "Oynaklık yükselirken ATR de genişler; sabit yüzdelik stop kullanan "
        "sistemler erken kesilir. Pozisyon boyutunu ATR ile ters orantılı "
        "küçültmek stop mantığını korur. VIX tepe yaptıktan sonra düşerken "
        "alım yapmak, tepe anında almaktan tarihsel olarak çok daha isabetli."
    ),
)

# ---------------------------------------------------------------- JEOPOLİTİK
_d(
    key="geo",
    label="Jeopolitik / Tedarik Zinciri Şoku",
    icon="🌍",
    nedir=(
        "Boğaz krizleri, gümrük tarifeleri, ihracat kontrolleri, enerji nakil "
        "hatlarına saldırı. Sermaye büyümeden kaçıp sert varlığa (altın, "
        "petrol, savunma) ve hazineye sığınır. Etki simetrik değildir: aynı "
        "olay savunma ve enerjiyi yukarı, tedarik zincirine bağlı üretimi "
        "aşağı çeker."
    ),
    tetikleyiciler=[
        "tariff", "sanction", "export control", "taiwan", "strait", "war",
        "missile", "invasion", "opec", "pipeline", "embargo", "trade war",
        "chip ban", "rare earth restriction", "port strike",
    ],
    veri_isareti="VIX vade yapısı tersine dönmüş + altın/bakır oranı yükseliyor",
    lehte_etf=["XAR", "ITA", "UFO", "ARKX", "GDX", "XLE", "XOP", "OIH", "REMX",
               "URA", "TLT"],
    lehte_hisse=["LMT", "RTX", "NOC", "GD", "LHX", "KTOS", "AVAV", "MP", "XOM",
                 "CVX", "NEM"],
    aleyhte_etf=["SOXX", "SMH", "EUV", "XRT", "JETS", "KWEB", "IYT"],
    aleyhte_hisse=["TSM", "ASML", "AAPL", "NVDA", "NKE", "TSLA", "BA", "DAL"],
    islem_notu=(
        "Çip tarafında ikili etki vardır: ihracat kısıtı TSM ve ASML'yi vurur "
        "ama ABD içi üretim teşviki INTC ve GFS lehine çalışır. Nadir toprak "
        "kısıtı REMX ve MP için doğrudan yukarı katalizördür. Haber anında "
        "değil, ilk paniğin geri çekilmesinde konumlanmak daha iyi fiyat verir."
    ),
)

# ---------------------------------------------------------------- LİKİDİTE
_d(
    key="liquidity",
    label="Likidite Sıkışması (FED / Hazine)",
    icon="🏦",
    nedir=(
        "İnatçı enflasyon, şahin FED, devasa tahvil ihracı veya Reverse Repo "
        "havuzunun kuruması piyasadaki dolar miktarını azaltır. En yüksek F/K'lı "
        "ve nakit akışı en uzak vadeli varlıklar önce satılır — çünkü iskonto "
        "oranı yükseldikçe uzak nakit akışı en çok değer kaybeder."
    ),
    tetikleyiciler=[
        "hawkish", "rate hike", "quantitative tightening", "qt", "hot cpi",
        "sticky inflation", "reverse repo", "treasury issuance", "debt ceiling",
        "yields surge", "dollar surges", "liquidity",
    ],
    veri_isareti="DXY yükseliyor + 10Y faiz yükseliyor + kredi iştahı zayıflıyor",
    lehte_etf=["TLT", "XLP", "XLV", "XLU"],
    lehte_hisse=["BRK-B", "JNJ", "PG", "KO", "MRK"],
    aleyhte_etf=["ARKG", "ARKF", "XBI", "IGV", "CLOU", "WGMI", "IBIT", "IWM"],
    aleyhte_hisse=["SNOW", "DDOG", "NET", "CRWD", "COIN", "MARA", "RIVN",
                   "IONQ", "RGTI"],
    islem_notu=(
        "Bu rejimde 'ucuzladı' diye alım en pahalı hatadır; likidite çekilirken "
        "çarpanlar aylarca sıkışabilir. Kâr eden, nakit üreten ve borcu düşük "
        "şirketler görece korunur. Kripto ve kâr etmeyen teknoloji en uçtaki "
        "kaldıraç olduğu için ilk ve en sert satılan taraftır."
    ),
)

# ---------------------------------------------------------------- FOMC
_d(
    key="fomc",
    label="FOMC / Veri Bekleyişi",
    icon="🏛️",
    nedir=(
        "Toplantı öncesi hacim çekilir, oynaklık bastırılır; karar anında ise "
        "tek barda haftalık ATR kadar hareket olur. Bu, stop mesafelerinin "
        "normal ATR'ye göre YETERSİZ kaldığı nadir durumlardan biridir."
    ),
    tetikleyiciler=[
        "fomc", "powell", "fed meeting", "dot plot", "jackson hole",
        "cpi report", "pce", "jobs report", "nonfarm payrolls", "fed minutes",
    ],
    veri_isareti="FOMC'a 3 gün veya daha az kalması",
    lehte_etf=[],
    lehte_hisse=[],
    aleyhte_etf=["Kaldıraçlı ve yüksek beta her şey"],
    aleyhte_hisse=[],
    islem_notu=(
        "Karar öncesi yeni pozisyon açmayın ya da normal boyutun yarısıyla "
        "açın. Karar sonrası ilk 30 dakikadaki hareket sık sık ters döner; "
        "kapanışı beklemek yanlış yönde girmekten korur."
    ),
)

# ---------------------------------------------------------------- YZ CAPEX
_d(
    key="ai_capex",
    label="YZ Sermaye Harcaması Döngüsü",
    icon="🤖",
    nedir=(
        "Hyperscaler'ların veri merkezi yatırım bütçesi, bu döngünün ana "
        "yakıtıdır. Bütçe artışı zinciri yukarıdan aşağı besler: çip → ağ → "
        "güç dağıtımı → elektrik üretimi → soğutma ve gayrimenkul. Bütçe "
        "kesintisi aynı zinciri ters yönde vurur."
    ),
    tetikleyiciler=[
        "capex", "data center", "hyperscaler", "ai spending", "gpu order",
        "cloud growth", "training cluster", "inference demand", "nuclear ppa",
        "power purchase agreement",
    ],
    veri_isareti="Yarı iletken ve kamu hizmetleri temalarının birlikte güçlenmesi",
    lehte_etf=["SOXX", "SMH", "EUV", "XLU", "URA", "PAVE", "SRVR", "AIQ"],
    lehte_hisse=["NVDA", "AVGO", "TSM", "ASML", "MU", "VRT", "ETN", "CEG",
                 "VST", "EQIX", "DLR", "MRVL", "CRDO"],
    aleyhte_etf=[],
    aleyhte_hisse=[],
    islem_notu=(
        "Zincirin ucundaki çarpan hisseleri (NVDA, AVGO, ETN, CEG, EQIX) "
        "paranın hangi alt temaya gittiğinden bağımsız pay alır. Alt temalar "
        "dönüşümlü modaya girer; çarpanlar döngü boyunca kalır."
    ),
)

# ---------------------------------------------------------------- ROTASYON
_d(
    key="rotation",
    label="Sektör Rotasyonu",
    icon="🔄",
    nedir=(
        "Paranın piyasadan çıkmadan sektör değiştirmesi. Endeks yatay görünse "
        "de altında büyük bir yer değiştirme olur; tema takibi bunu endeksten "
        "önce gösterir."
    ),
    tetikleyiciler=[
        "rotation", "value over growth", "small cap", "breadth", "laggard",
        "sector leadership", "defensive",
    ],
    veri_isareti="Tema takibinde liderlerin yavaşlarken diplerin hızlanması",
    lehte_etf=["XLV", "XLP", "XLE", "KRE", "IWM"],
    lehte_hisse=[],
    aleyhte_etf=["QQQ", "XLK"],
    aleyhte_hisse=[],
    islem_notu=(
        "Rotasyon rejiminde 'dipten dönen' çeyrek (negatif getiri, pozitif "
        "ivme) en yüksek getiriyi verir; 'yavaşlayan lider' çeyreğindeki "
        "pozisyonlar ise kâr almanın zamanının geldiğini söyler."
    ),
)


# --------------------------------------------------------------------------
# Rejim -> sürücü eşlemesi
# --------------------------------------------------------------------------
REGIME_TO_DRIVERS: dict[str, list[str]] = {
    "🩸 LİKİDİTE KRİZİ": ["liquidity", "vix"],
    "🌍 JEOPOLİTİK / OLAY ŞOKU": ["geo", "vix"],
    "🎯 OPEX PINNING": ["opex"],
    "🚀 RİSK İŞTAHI / GAMMA": ["gamma", "ai_capex"],
    "🏦 FOMC BEKLEYİŞİ": ["fomc", "liquidity"],
    "💵 DOLAR SIKIŞMASI": ["liquidity"],
    "🟢 RİSK AÇIK": ["gamma", "ai_capex", "rotation"],
    "🟡 KARIŞIK — TREND ZAYIF": ["rotation", "vix"],
    "🔴 RİSK KAPALI": ["liquidity", "vix", "rotation"],
    "⚖️ GEÇİŞ / KARARSIZ": ["rotation", "opex"],
}


def drivers_for(regime: str) -> list[Driver]:
    keys = REGIME_TO_DRIVERS.get(regime, ["rotation"])
    return [DRIVERS[k] for k in keys if k in DRIVERS]


_KW_CACHE: dict[str, re.Pattern] = {}


def _kw_pattern(kw: str) -> re.Pattern:
    """
    Kelime sınırlı desen.

    Düz alt dize araması "award" içindeki "war"ı jeopolitik sürücüsü sanıyordu.
    \b sınırı bu tür yanlış eşleşmeleri engeller; çok kelimeli ifadelerde
    aradaki boşluk esnek bırakılır.
    """
    if kw not in _KW_CACHE:
        parts = [re.escape(w) for w in kw.split()]
        _KW_CACHE[kw] = re.compile(r"\b" + r"\s+".join(parts) + r"\b")
    return _KW_CACHE[kw]


def match_drivers(title: str) -> list[str]:
    """Bir haber başlığının hangi rejim sürücülerini beslediğini bulur."""
    t = (title or "").lower()
    return [key for key, d in DRIVERS.items()
            if any(_kw_pattern(kw).search(t) for kw in d.tetikleyiciler)]


def impacted(driver_keys: list[str]) -> dict[str, list[str]]:
    """Bir veya birden çok sürücünün etkilediği ETF ve hisseler."""
    lehte_e, lehte_h, aleyhte_e, aleyhte_h = [], [], [], []
    for k in driver_keys:
        d = DRIVERS.get(k)
        if not d:
            continue
        for src, dst in ((d.lehte_etf, lehte_e), (d.lehte_hisse, lehte_h),
                         (d.aleyhte_etf, aleyhte_e), (d.aleyhte_hisse, aleyhte_h)):
            for x in src:
                if x not in dst:
                    dst.append(x)
    return {"lehte_etf": lehte_e, "lehte_hisse": lehte_h,
            "aleyhte_etf": aleyhte_e, "aleyhte_hisse": aleyhte_h}

# ==========================================================================
# KAYNAK: apex/themes.py
# ==========================================================================


from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Quadrant:
    key: str
    label: str
    icon: str
    renk: str
    aciklama: str
    aksiyon: str


QUADRANTS: dict[str, Quadrant] = {
    "lider_hizlanan": Quadrant(
        "lider_hizlanan", "Hızlanan lider", "🚀", "#2fbe86",
        "Getiri pozitif VE ivme pozitif. Tema kazandırıyor ve kazandırma hızı "
        "artıyor — para bu temaya yeni giriyor.",
        "Ana avlanma sahası. Swing adaylarını önce burada arayın; trend takip "
        "sistemleri en yüksek isabeti bu çeyrekte verir."),
    "lider_yavaslayan": Quadrant(
        "lider_yavaslayan", "Yavaşlayan lider", "🌤️", "#c98500",
        "Getiri pozitif AMA ivme negatif. Tema hâlâ kazandırıyor, ancak önceki "
        "döneme göre daha yavaş — giriş azalıyor.",
        "Yeni pozisyon için geç kalınmış olabilir. Mevcut pozisyonlarda kısmi "
        "kâr alma ve stop yukarı çekme zamanı."),
    "dipten_donen": Quadrant(
        "dipten_donen", "Dipten dönen", "🌱", "#3987e5",
        "Getiri negatif AMA ivme pozitif. Tema hâlâ ekside, fakat düşüş hızı "
        "kesiliyor — taban oluşumu buradan başlar.",
        "En yüksek getiri potansiyeli burada, en yüksek yanılma payı da. "
        "Rejim kapısı açıkken ve hisse bazında likidite süpürmesi/toplama "
        "sinyali varken anlamlı."),
    "hizlanan_dusus": Quadrant(
        "hizlanan_dusus", "Hızlanan düşüş", "🩸", "#e66767",
        "Getiri negatif VE ivme negatif. Tema kaybettiriyor ve kaybettirme "
        "hızı artıyor — çıkış devam ediyor.",
        "Dip arayışı erken. Bu temadaki long sinyalleri rejim kapısı kapalıyken "
        "gelen sinyaller gibi ele alınmalı: izle, alma."),
}

PERIOD_LABELS: dict[str, str] = {
    "Bugün": "son 1 işlem günü",
    "1H": "son 5 işlem günü",
    "1A": "son 21 işlem günü",
    "3A": "son 63 işlem günü",
    "YBB": "yılbaşından bugüne",
}

PERIOD_PREV: dict[str, str] = {
    "Bugün": "ondan önceki gün",
    "1H": "ondan önceki 5 gün",
    "1A": "ondan önceki 21 gün",
    "3A": "ondan önceki 63 gün",
    "YBB": "geçen yılın aynı dönemi",
}


def classify_quadrant(getiri: float, ivme: float, esik: float = 0.0) -> str:
    """Getiri/ivme ikilisini çeyreğe yerleştirir."""
    if not np.isfinite(getiri) or not np.isfinite(ivme):
        return "hizlanan_dusus" if getiri < 0 else "lider_yavaslayan"
    if getiri >= esik and ivme >= 0:
        return "lider_hizlanan"
    if getiri >= esik and ivme < 0:
        return "lider_yavaslayan"
    if getiri < esik and ivme >= 0:
        return "dipten_donen"
    return "hizlanan_dusus"


def build_table(perf: pd.DataFrame, period: str) -> pd.DataFrame:
    """
    Tema performans tablosuna GETİRİ, İVME ve ÇEYREK sütunlarını ekler.
    `perf`: theme_performance() çıktısı (Bugün/1H/… ve Prev_* sütunları).
    """
    if perf.empty or period not in perf.columns:
        return pd.DataFrame()

    prev_col = f"Prev_{period}"
    out = pd.DataFrame(index=perf.index)
    out["Getiri %"] = perf[period]
    out["Önceki %"] = perf[prev_col] if prev_col in perf.columns else np.nan
    out["İvme"] = out["Getiri %"] - out["Önceki %"]
    out["Çeyrek"] = [
        QUADRANTS[classify_quadrant(g, i)].icon + " " + QUADRANTS[classify_quadrant(g, i)].label
        for g, i in zip(out["Getiri %"], out["İvme"])
    ]
    out["_q"] = [classify_quadrant(g, i) for g, i in zip(out["Getiri %"], out["İvme"])]
    out["Semboller"] = perf["Semboller"] if "Semboller" in perf.columns else ""
    return out.sort_values("Getiri %", ascending=False)


def summary(table: pd.DataFrame) -> dict[str, Any]:
    """Çeyrek bazında özet: hangi temalar nerede."""
    if table.empty:
        return {}
    out: dict[str, Any] = {}
    for key, q in QUADRANTS.items():
        sel = table[table["_q"] == key]
        if sel.empty:
            continue
        out[key] = {
            "quadrant": q,
            "temalar": list(sel.index),
            "n": len(sel),
            "ort_getiri": float(sel["Getiri %"].mean()),
            "ort_ivme": float(sel["İvme"].mean()),
        }
    return out


def worked_example(table: pd.DataFrame, period: str) -> str:
    """
    Gerçek veriden somut bir örnek cümle üretir — açıklama soyut kalmasın.
    En büyük ivme farkına sahip temayı seçer.
    """
    if table.empty or table["İvme"].isna().all():
        return ""
    row = table.loc[table["İvme"].abs().idxmax()]
    tema = row.name
    g, o, i = row["Getiri %"], row["Önceki %"], row["İvme"]
    if not np.isfinite(o):
        return ""
    yon = "hızlanıyor" if i > 0 else "yavaşlıyor"
    q = QUADRANTS[row["_q"]]
    return (
        f"**Örnek — {tema}:** {PERIOD_LABELS.get(period, period)} getirisi "
        f"**%{g:+.2f}**, {PERIOD_PREV.get(period, 'önceki dönem')} getirisi "
        f"**%{o:+.2f}** idi. İvme = {g:+.2f} − ({o:+.2f}) = **{i:+.2f}** → tema "
        f"{yon}. Çeyrek: {q.icon} **{q.label}**. {q.aksiyon}"
    )

# ==========================================================================
# KAYNAK: apex/report.py
# ==========================================================================


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

# ==========================================================================
# KAYNAK: apex/valuation.py
# ==========================================================================
# apex/valuation.py — Satış çarpanına dayalı adil değer (P/S Fair Value)
#
# Mantık ("Trader's glance" kartındaki yöntem):
#
#   1. Sektör/endüstri için "normal" ileri P/S çarpanı seçilir
#      (ör. yazılım ≈ 10x gelecek yıl satışı).
#   2. Adil orta = çarpan × gelecek yıl satış / hisse
#      Adil bant  = orta × (1 ± bant)   (varsayılan ±%12)
#   3. PSG = (fiyat / TTM satış-hisse) / büyüme%   — büyümeye göre P/S
#      Pahalı eşiği = PSG'nin `psg_max` olduğu fiyat
#      = psg_max × büyüme% × TTM satış/hisse       (varsayılan 0.60)
#   4. Durum:
#        fiyat < bant altı                → 🟢 UCUZ
#        bant altı ≤ fiyat ≤ pahalı eşiği → 🟡 ADİL (bant üstündeyse
#                                            "büyüme destekli" notu)
#        fiyat > pahalı eşiği             → 🔴 PAHALI
#
# Not: Pahalı eşiği kartta açıkça yazmıyor; RBRK kartındaki rakamlardan
# (fiyat 113.28, P/S 15x, PSG 0.39, pahalı 175.17) geri hesaplanınca
# PSG ≈ 0.60'a denk geliyor. Arayüzden değiştirilebilir.
#
# Veri katmanı (data.py) yalnızca ham alanları çeker; hesap burada yapılır.
# Böylece çarpan tablosu arayüzde düzenlendiğinde yeniden veri çekilmez.


from typing import Any

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Varsayılan ileri P/S çarpanları — BAŞLANGIÇ DEĞERLERİDİR, arayüzden düzenlenir.
# None = satış çarpanı bu iş modelinde anlamsız (banka, sigorta, petrol, GYO):
# bu şirketlerde P/S adil değeri hesaplanmaz.
# Eşleşme endüstri adında anahtar kelime aramasıyla yapılır (yfinance
# `industry` alanı); ilk eşleşen kazanır, bu yüzden özelden genele sıralı.
# --------------------------------------------------------------------------
INDUSTRY_PS: dict[str, float | None] = {
    "Software": 10.0,
    "Semiconductor Equipment": 7.0,
    "Semiconductors": 8.0,
    "Internet Content": 6.0,
    "Information Technology Services": 3.0,
    "Communication Equipment": 3.5,
    "Computer Hardware": 2.5,
    "Electronic Components": 3.5,
    "Scientific & Technical Instruments": 4.0,
    "Biotechnology": 6.0,
    "Drug Manufacturers": 4.0,
    "Medical Devices": 5.0,
    "Medical Instruments": 5.0,
    "Diagnostics": 4.0,
    "Health Information": 5.0,
    "Aerospace & Defense": 2.5,
    "Utilities": 3.0,
    "Uranium": 8.0,
    "Solar": 3.0,
    "Electrical Equipment": 3.0,
    "Specialty Industrial Machinery": 3.0,
    "Engineering & Construction": 1.5,
    "Auto Manufacturers": 1.5,
    "Internet Retail": 2.5,
    "Copper": 2.5,
    "Gold": 4.0,
    "Other Industrial Metals": 2.5,
    "Banks": None,
    "Insurance": None,
    "Capital Markets": None,
    "Asset Management": None,
    "Oil & Gas": None,
    "REIT": None,
    "Real Estate": None,
}

# Endüstri eşleşmezse sektöre düşülür
SECTOR_PS: dict[str, float | None] = {
    "Technology": 6.0,
    "Communication Services": 3.0,
    "Healthcare": 4.0,
    "Industrials": 2.0,
    "Consumer Cyclical": 1.5,
    "Consumer Defensive": 1.5,
    "Basic Materials": 2.0,
    "Utilities": 3.0,
    "Energy": None,
    "Financial Services": None,
    "Real Estate": None,
}

DEFAULT_BAND = 0.12      # adil bant ±%12
DEFAULT_PSG_MAX = 0.60   # bu PSG'nin üstü "pahalı"
DEFAULT_EXP_CAP = 1.80   # pahalı eşiği en fazla adil ortanın bu katı
                         # (RBRK kartında 175.17 / 98.33 ≈ 1.78)


def pick_multiple(industry: str, sector: str,
                  ind_map: dict[str, float | None] | None = None,
                  sec_map: dict[str, float | None] | None = None
                  ) -> tuple[float | None, str]:
    """Endüstri → sektör sırasıyla çarpanı bulur. Döner: (çarpan, kaynak)."""
    ind_map = INDUSTRY_PS if ind_map is None else ind_map
    sec_map = SECTOR_PS if sec_map is None else sec_map
    ind = (industry or "").lower()
    for key, mult in ind_map.items():
        if key.lower() in ind:
            return mult, key
    if sector in sec_map:
        return sec_map[sector], f"{sector} (sektör)"
    return None, "eşleşme yok"


def _f(x: Any) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def compute_fair_value(rec: dict[str, Any],
                       ind_map: dict[str, float | None] | None = None,
                       sec_map: dict[str, float | None] | None = None,
                       band: float = DEFAULT_BAND,
                       psg_max: float = DEFAULT_PSG_MAX,
                       exp_cap: float = DEFAULT_EXP_CAP) -> dict[str, Any]:
    """
    `rec`: data.fetch_fundamentals() satırı (Fiyat, TTM SPS, İleri SPS,
    Büyüme %, Endüstri, Sektör). Döner: adil değer alanları.
    """
    price = _f(rec.get("Fiyat"))
    ttm_sps = _f(rec.get("_ttm_sps"))
    fwd_sps = _f(rec.get("_fwd_sps"))
    g_pct = _f(rec.get("Büyüme %"))
    mult, kaynak = pick_multiple(rec.get("Endüstri", ""), rec.get("Sektör", ""),
                                 ind_map, sec_map)

    out: dict[str, Any] = {
        "Çarpan": mult, "Çarpan Kaynağı": kaynak,
        "P/S": price / ttm_sps if ttm_sps and np.isfinite(price) else np.nan,
        "İleri P/S": price / fwd_sps if fwd_sps and np.isfinite(price) else np.nan,
        "PSG": np.nan, "Adil Alt": np.nan, "Adil Orta": np.nan,
        "Adil Üst": np.nan, "Pahalı >": np.nan, "Adile Uzaklık %": np.nan,
        "P/S Durum": "➖ Hesaplanamadı",
    }

    if mult is None:
        out["P/S Durum"] = "➖ P/S uygun değil"
        return out
    if not (np.isfinite(price) and np.isfinite(fwd_sps) and fwd_sps > 0):
        out["P/S Durum"] = "➖ Satış tahmini yok"
        return out

    mid = mult * fwd_sps
    lo, hi = mid * (1 - band), mid * (1 + band)
    out.update({"Adil Alt": lo, "Adil Orta": mid, "Adil Üst": hi,
                "Adile Uzaklık %": (price / mid - 1) * 100})

    expensive = np.nan
    if np.isfinite(g_pct) and g_pct > 0 and np.isfinite(ttm_sps) and ttm_sps > 0:
        out["PSG"] = (price / ttm_sps) / g_pct
        expensive = psg_max * g_pct * ttm_sps
    # Büyüme yoksa/negatifse PSG tanımsız: bandın üstüne 2 bant daha
    if not np.isfinite(expensive):
        expensive = mid * (1 + 3 * band)
    # PSG eşiği büyümeyle orantılıdır ama sektör çarpanını bilmez: düşük
    # çarpanlı sektörde ya da küçük tabandan %100+ büyümede uçar. Bu yüzden
    # alttan adil banda, üstten adil ortanın `exp_cap` katına sıkıştırılır.
    expensive = min(max(expensive, hi), mid * max(exp_cap, 1 + band))
    out["Pahalı >"] = expensive

    if price < lo:
        out["P/S Durum"] = "🟢 UCUZ"
    elif price > expensive:
        out["P/S Durum"] = "🔴 PAHALI"
    elif price > hi:
        out["P/S Durum"] = "🟡 ADİL (bant üstü, büyüme destekli)"
    else:
        out["P/S Durum"] = "🟡 ADİL"
    return out


def add_fair_values(df: pd.DataFrame, **kw) -> pd.DataFrame:
    """Temel veri tablosunun her satırına adil değer sütunlarını ekler."""
    if df is None or df.empty:
        return df
    fv = pd.DataFrame([compute_fair_value(r, **kw)
                       for r in df.to_dict("records")], index=df.index)
    return pd.concat([df, fv], axis=1)


def glance_text(r: dict[str, Any]) -> str:
    """Tek hisse için 'Trader's glance' kartının Türkçe karşılığı (markdown)."""
    def m(x, spec=".2f", pre="$"):
        return f"{pre}{x:{spec}}" if np.isfinite(_f(x)) else "—"

    durum = r.get("P/S Durum", "➖")
    lines = [f"**Hızlı bakış — `{r.get('Hisse', '')}` {m(r.get('Fiyat'))}**", "",
             f"{durum} — satışlarına göre."]
    if r.get("Açıklama"):
        lines += ["", r["Açıklama"]]
    ps, fps = _f(r.get("P/S")), _f(r.get("İleri P/S"))
    if np.isfinite(ps):
        lines += ["", f"Şu an {ps:.1f}x satış"
                  + (f", gelecek yılın {fps:.1f}x katı ile işlem görüyor."
                     if np.isfinite(fps) else " ile işlem görüyor.")]
    mult = r.get("Çarpan")
    if mult:
        lines += [f"**{r.get('Çarpan Kaynağı', '')}** için normal çarpan ≈ {mult:g}x."]
        lines += [f"Adil aralık **{m(r.get('Adil Alt'))} – {m(r.get('Adil Üst'))}** "
                  f"({mult:g}x gelecek yıl satışı, ±bant).",
                  f"Pahalı eşiği: **{m(r.get('Pahalı >'))}** üstü."]
    ek = []
    if np.isfinite(_f(r.get("Piyasa Değ. ($B)"))):
        ek.append(f"PD ${_f(r['Piyasa Değ. ($B)']):.1f}B")
    if np.isfinite(_f(r.get("İleri F/K"))):
        ek.append(f"İleri F/K {_f(r['İleri F/K']):.1f}x")
    if np.isfinite(_f(r.get("EV/Satış"))):
        ek.append(f"EV/Satış {_f(r['EV/Satış']):.1f}x")
    if np.isfinite(_f(r.get("PSG"))):
        ek.append(f"PSG {_f(r['PSG']):.2f}")
    if np.isfinite(_f(r.get("Büyüme %"))):
        ek.append(f"Büyüme %{_f(r['Büyüme %']):.0f}")
    if np.isfinite(_f(r.get("52H Konum %"))):
        ek.append(f"52H %{_f(r['52H Konum %']):.0f}")
    if np.isfinite(_f(r.get("Hedef"))):
        ek.append(f"Analist hedefi {m(r.get('Hedef'))}")
    if ek:
        lines += ["", " · ".join(ek)]
    return "  \n".join(lines)

# ==========================================================================
# KAYNAK: apex/shortvol.py
# ==========================================================================
# apex/shortvol.py — FINRA günlük short hacmi (Reg SHO) ve haftalık değişim
#
# Kaynak: FINRA'nın ücretsiz yayımladığı konsolide günlük dosya
#   https://cdn.finra.org/equity/regsho/daily/CNMSshvol{YYYYMMDD}.txt
#   Biçim: Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market
#   Her iş günü ABD saatiyle ~18:00'den sonra yayımlanır.
#
# ÖNEMLİ — okuma şekli:
#   * Bu "short HACMİ"dir, "short INTEREST" (açık pozisyon) değildir.
#     Piyasa yapıcıların alıcıya hisse sağlamak için yaptığı gün içi açığa
#     satışlar da buraya girer; bu yüzden çoğu hissede oran zaten %35–55
#     arasındadır. Anlamlı olan SEVİYE değil, hissenin KENDİ ortalamasına
#     göre DEĞİŞİMDİR.
#   * Hacim yalnızca FINRA'ya raporlanan (TRF/ADF) işlemleri kapsar,
#     borsa içi (lit) işlemleri tamamen kapsamaz. "TotalVolume" bu yüzden
#     yfinance'teki toplam hacimden düşüktür — oran kendi içinde tutarlıdır.
#   * OTC hisseler (FANUY, YASKY vb.) ve kripto CNMS dosyasında yoktur.


import datetime as dt
import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

import numpy as np
import pandas as pd
import requests

_log = logging.getLogger(__name__)

FINRA_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{d}.txt"
WINDOW = 5          # "bu hafta" = son 5 işlem günü
Z_LEN = 20          # z-skoru için taban pencere
ESIK_PP = 3.0       # anlamlı değişim eşiği (yüzde puan)


def _us_today() -> dt.date:
    try:
        from zoneinfo import ZoneInfo
        return dt.datetime.now(ZoneInfo("America/New_York")).date()
    except Exception:                                   # pragma: no cover
        return dt.date.today()


def _weekdays_back(n_days: int, end: dt.date | None = None) -> list[dt.date]:
    end = end or _us_today()
    out, d = [], end
    while len(out) < n_days:
        if d.weekday() < 5:
            out.append(d)
        d -= dt.timedelta(days=1)
    return out


_FINRA_HEADERS = {
    # Akamai CDN, kısa/robot görünümlü isteklere 403 dönebiliyor —
    # tam bir tarayıcı başlığı gönderiyoruz.
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/126.0 Safari/537.36"),
    "Accept": "text/plain,text/html,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data",
}


def _fetch_day(d: dt.date, timeout: int = 20) -> tuple[pd.DataFrame | None, str]:
    """Döner: (tablo ya da None, tanı metni: 'ok', 'HTTP 404', hata adı…)."""
    url = FINRA_URL.format(d=d.strftime("%Y%m%d"))
    last = ""
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=timeout, headers=_FINRA_HEADERS)
            if r.status_code == 404:
                return None, "HTTP 404 (tatil / henüz yok)"
            if r.status_code != 200 or not r.content:
                last = f"HTTP {r.status_code}"
                time.sleep(1.0 + attempt)
                continue
            df = pd.read_csv(io.BytesIO(r.content), sep="|",
                             dtype={"Symbol": str})
            df["Date"] = pd.to_numeric(df["Date"], errors="coerce")
            df = df[df["Date"].notna()].copy()           # alt bilgi satırı
            df["Date"] = pd.to_datetime(df["Date"].astype("int64").astype(str),
                                        format="%Y%m%d")
            for c in ("ShortVolume", "TotalVolume"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            return df[["Date", "Symbol", "ShortVolume", "TotalVolume"]], "ok"
        except Exception as exc:
            last = type(exc).__name__
            time.sleep(1.0 + attempt)
    return None, last or "bilinmeyen hata"


def _repo_file() -> "Path | None":
    """GitHub Actions'ın repoya yazdığı FINRA dosyası (data/finra_short.csv.gz)."""
    from pathlib import Path
    for base in (Path(__file__).resolve().parent, Path.cwd()):
        p = base / "data" / "finra_short.csv.gz"
        if p.exists():
            return p
    return None


def _read_repo_file() -> pd.DataFrame | None:
    p = _repo_file()
    if p is None:
        return None
    try:
        df = pd.read_csv(p, dtype={"Symbol": str})
        df["Date"] = pd.to_datetime(df["Date"].astype("int64").astype(str),
                                    format="%Y%m%d")
        for c in ("ShortVolume", "TotalVolume"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df[["Date", "Symbol", "ShortVolume", "TotalVolume"]]
    except Exception as exc:
        _log.info("Repo FINRA dosyası okunamadı: %s", exc)
        return None


def fetch_short_volume(trading_days: int = 30
                       ) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """
    Önce repodaki data/finra_short.csv.gz (GitHub Actions her akşam günceller),
    sonra yalnızca o dosyada olmayan son günler FINRA'dan doğrudan denenir.
    Döner: (uzun tablo, uyarılar, {tarih: tanı}).
    """
    days = _weekdays_back(trading_days + 4)  # tatil payı
    repo = _read_repo_file()
    diag: dict[str, str] = {}
    frames: list[pd.DataFrame] = []
    have: set = set()
    if repo is not None and not repo.empty:
        frames.append(repo)
        have = {d.date() for d in repo["Date"].unique()}
        for d in sorted(have, reverse=True):
            diag[d.strftime("%d.%m.%Y")] = "ok (repo dosyası)"
        # repodaki son günden sonrası doğrudan denenir
        last = max(have)
        days = [d for d in days if d > last]

    if days:
        with ThreadPoolExecutor(max_workers=4) as ex:
            res = list(ex.map(_fetch_day, days))
        for d, (f, msg) in zip(days, res):
            diag[d.strftime("%d.%m.%Y")] = msg
            if f is not None and not f.empty:
                frames.append(f)
    diag = dict(sorted(diag.items(),
                       key=lambda kv: dt.datetime.strptime(kv[0], "%d.%m.%Y"),
                       reverse=True))

    warns: list[str] = []
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        kodlar = sorted({m for m in diag.values() if not m.startswith("HTTP 404")})
        warns.append("FINRA short hacim verisine ulaşılamadı"
                     + (f" ({', '.join(kodlar)})" if kodlar else "")
                     + ". Repoda data/finra_short.csv.gz yok — GitHub Actions "
                       "iş akışını bir kez çalıştırın.")
        return (pd.DataFrame(columns=["Date", "Symbol", "ShortVolume",
                                      "TotalVolume"]), warns, diag)
    df = (pd.concat(frames, ignore_index=True)
            .drop_duplicates(["Date", "Symbol"], keep="last"))
    last = df["Date"].max()
    if (pd.Timestamp(_us_today()) - last).days > 4:
        warns.append(f"En son FINRA verisi {last:%d.%m.%Y} tarihli.")
    n_days = df["Date"].nunique()
    if n_days < 2 * WINDOW:
        warns.append(f"Yalnızca {n_days} günlük FINRA verisi var; "
                     "haftalık karşılaştırma eksik kalabilir.")
    return df, warns, diag


def _one_short_interest(t: str) -> dict:
    """yfinance üzerinden borsa short INTEREST verisi (ayda iki kez güncellenir)."""
    def num(x):
        try:
            v = float(x)
            return v if np.isfinite(v) else np.nan
        except (TypeError, ValueError):
            return np.nan

    rec = {"Sembol": t, "SI Float %": np.nan, "SI Δ% (ay)": np.nan,
           "Gün Kapama": np.nan, "SI Tarihi": ""}
    try:
        info = yf_info(t)
        cur, prev = num(info.get("sharesShort")), num(info.get("sharesShortPriorMonth"))
        pf = num(info.get("shortPercentOfFloat"))
        rec["SI Float %"] = pf * 100 if np.isfinite(pf) else np.nan
        if np.isfinite(cur) and np.isfinite(prev) and prev > 0:
            rec["SI Δ% (ay)"] = (cur / prev - 1) * 100
        rec["Gün Kapama"] = num(info.get("shortRatio"))
        ts = info.get("dateShortInterest")
        if isinstance(ts, (int, float)) and ts > 0:
            rec["SI Tarihi"] = dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%d.%m")
    except Exception as exc:
        _log.info("Short interest alınamadı (%s): %s", t, exc)
    return rec


def fetch_short_interest(tickers: Iterable[str]) -> pd.DataFrame:
    syms = list(dict.fromkeys(t.upper() for t in tickers if t))
    if not syms:
        return pd.DataFrame(columns=["Sembol"])
    with ThreadPoolExecutor(max_workers=2) as ex:
        return pd.DataFrame(list(ex.map(_one_short_interest, syms)))


def to_finra_symbol(t: str) -> str:
    """yfinance → FINRA: BRK-B → BRK/B (FINRA sınıf ayracı olarak '/' kullanır)"""
    return t.replace("-", "/").replace(".", "/").upper()


def _arrow(d_pp: float) -> str:
    if not np.isfinite(d_pp):
        return "—"
    if d_pp >= ESIK_PP:
        return "⬆️ artıyor"
    if d_pp <= -ESIK_PP:
        return "⬇️ azalıyor"
    return "→ yatay"


def interpret(d_pp: float, price_chg: float, z: float) -> str:
    """Short hacim değişimi + fiyat yönü → tek cümlelik okuma."""
    if not np.isfinite(d_pp):
        return "Veri yok"
    up, dn = d_pp >= ESIK_PP, d_pp <= -ESIK_PP
    p_up = np.isfinite(price_chg) and price_chg > 0
    p_dn = np.isfinite(price_chg) and price_chg < 0
    uc = " (aşırı uç)" if np.isfinite(z) and abs(z) >= 2 else ""
    if up and p_dn:
        return "🔴 Short baskısı artıyor, fiyat düşüyor" + uc
    if up and p_up:
        return "🟠 Yükselişe karşı short artıyor — ya dağıtım ya squeeze yakıtı" + uc
    if dn and p_up:
        return "🟢 Short çekiliyor, fiyat yükseliyor — kapama desteği" + uc
    if dn and p_dn:
        return "🟡 Short azalıyor ama fiyat düşüyor — satış uzun taraftan" + uc
    if up:
        return "🟠 Short hacmi artıyor" + uc
    if dn:
        return "🟢 Short hacmi azalıyor" + uc
    return "⚪ Belirgin değişim yok"


def short_table(raw: pd.DataFrame, tickers: Iterable[str],
                price_chg: dict[str, float] | None = None,
                spark_n: int = 15) -> pd.DataFrame:
    """
    Her sembol için: son 5 gün short oranı, önceki 5 gün, fark (pp),
    short hacmindeki % değişim, 20 günlük z-skoru, seyir listesi, yorum.
    """
    price_chg = price_chg or {}
    tickers = list(dict.fromkeys(t.upper() for t in tickers))
    cols = ["Sembol", "SV% 5G", "SV% Önceki 5G", "ΔSV pp", "Short Hacim Δ%",
            "SV% Z", "SV Yön", "SV Seyri", "Short Yorum", "SV Son Gün"]
    if raw is None or raw.empty:
        return pd.DataFrame(columns=cols)

    fmap = {to_finra_symbol(t): t for t in tickers}
    sub = raw[raw["Symbol"].isin(fmap)].copy()
    sub = (sub.groupby(["Symbol", "Date"], as_index=False)
              [["ShortVolume", "TotalVolume"]].sum())
    sub["ratio"] = sub["ShortVolume"] / sub["TotalVolume"].replace(0, np.nan) * 100

    rows = []
    for fsym, g in sub.groupby("Symbol"):
        g = g.sort_values("Date")
        t = fmap[fsym]
        last5, prev5 = g.tail(WINDOW), g.iloc[-2 * WINDOW:-WINDOW]
        # hacim ağırlıklı oran (günlük oranların ortalaması yerine)
        r5 = last5["ShortVolume"].sum() / max(last5["TotalVolume"].sum(), 1) * 100
        rp = (prev5["ShortVolume"].sum() / max(prev5["TotalVolume"].sum(), 1) * 100
              if len(prev5) else np.nan)
        sv5, svp = last5["ShortVolume"].sum(), prev5["ShortVolume"].sum()
        base = g["ratio"].iloc[:-1].tail(Z_LEN)
        z = ((g["ratio"].iloc[-1] - base.mean()) / base.std(ddof=0)
             if len(base) >= 10 and base.std(ddof=0) > 0 else np.nan)
        d_pp = r5 - rp if np.isfinite(rp) else np.nan
        rows.append({
            "Sembol": t,
            "SV% 5G": r5,
            "SV% Önceki 5G": rp,
            "ΔSV pp": d_pp,
            "Short Hacim Δ%": (sv5 / svp - 1) * 100 if svp else np.nan,
            "SV% Z": z,
            "SV Yön": _arrow(d_pp),
            "SV Seyri": [round(float(x), 1) for x in g["ratio"].tail(spark_n)],
            "Short Yorum": interpret(d_pp, price_chg.get(t, np.nan), z),
            "SV Son Gün": g["Date"].iloc[-1].strftime("%d.%m"),
        })
    return pd.DataFrame(rows, columns=cols)

# ==========================================================================
# KAYNAK: apex/funnel.py
# ==========================================================================
# apex/funnel.py — KARAR HUNİSİ
#
# Yatırım kararını dört soruya böler; her adım bir öncekinin cevabına dayanır:
#
#   1) Risk açık mı?            → piyasanın genel risk iştahı (6 sütun)
#   2) Para nereye akıyor?      → endeks / varlık sınıfı / kripto rotasyonu
#   3) Hangi tema ivmeleniyor?  → tema ve ETF'lerde ERKEN akış tespiti (RRG)
#   4) Bu temada hangi hisse?   → lider / geride kalan / alım adayı ayrımı
#
# Fiyat verisi yfinance'ten; kripto endeksleri (TOTAL, TOTAL3, BTC.D,
# OTHERS/BTC) ve Fed net likiditesi repodaki data/*.csv dosyalarından
# (GitHub Actions her iş günü günceller — tools/macro_update.py).


from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Semboller
# --------------------------------------------------------------------------
ROTATION_TICKERS: list[str] = [
    "SPY", "QQQ", "IWM", "RSP", "SMH", "ARKK", "XLY", "XLP", "EEM", "TLT",
    "HYG", "GC=F", "SI=F", "HG=F", "BZ=F", "DX-Y.NYB", "^GSPC", "^IXIC",
    "BTC-USD", "ETH-USD",
]

NAMES: dict[str, str] = {
    "SPY": "S&P 500", "QQQ": "Nasdaq 100", "IWM": "Russell 2000 (küçük ölçek)",
    "RSP": "Eşit ağırlıklı S&P", "SMH": "Yarı iletkenler",
    "ARKK": "Spekülatif büyüme (ARKK)", "XLY": "İhtiyari tüketim",
    "XLP": "Temel tüketim", "EEM": "Gelişen piyasalar", "TLT": "Uzun vadeli tahvil",
    "HYG": "Yüksek getirili tahvil", "GC=F": "Altın", "SI=F": "Gümüş",
    "HG=F": "Bakır", "BZ=F": "Brent petrol", "DX-Y.NYB": "Dolar endeksi",
    "^GSPC": "S&P 500 endeksi", "^IXIC": "Nasdaq Bileşik", "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum", "TOTAL": "Toplam kripto", "TOTAL3": "Altcoinler (TOTAL3)",
    "OTHERS": "Küçük altcoinler (OTHERS)", "BTC.D": "BTC hakimiyeti",
    "NET_LIQ": "Fed net likiditesi",
}


# --------------------------------------------------------------------------
# Rotasyon çiftleri — "pay / payda" yükseliyorsa para paydadan paya akıyor
#   risk: +1 → yükselişi risk iştahı demek, −1 → risk kaçışı demek
# --------------------------------------------------------------------------
PAIRS: list[dict[str, Any]] = [
    # Hisse içi rotasyon
    dict(key="QQQ/SPY", num="QQQ", den="SPY", grup="Hisse içi", risk=+1,
         ad="Nasdaq 100 / S&P 500",
         not_="Yükseliyorsa büyüme ve teknoloji liderleri piyasayı sürüklüyor. "
              "Çeyreklik grafikte kırılım, liderliğin süreceğine ve risk "
              "iştahına işaret eder."),
    dict(key="^IXIC/^GSPC", num="^IXIC", den="^GSPC", grup="Hisse içi", risk=+1,
         ad="Nasdaq Bileşik / S&P 500",
         not_="QQQ/SPY'den farkı, Nasdaq'taki küçük ve orta boy büyüme "
              "şirketlerini de kapsaması. İkisi birlikte yükseliyorsa iştah "
              "devlerle sınırlı değil."),
    dict(key="SMH/SPY", num="SMH", den="SPY", grup="Hisse içi", risk=+1,
         ad="Yarı iletken / S&P 500",
         not_="Yarı iletkenler döngünün öncü sektörü. Liderlik ediyorsa "
              "yapay zekâ ve donanım temaları güçlü."),
    dict(key="ARKK/QQQ", num="ARKK", den="QQQ", grup="Hisse içi", risk=+1,
         ad="Spekülatif büyüme / Nasdaq 100",
         not_="Kârsız, yüksek riskli büyüme şirketlerine iştah. Riskin en "
              "uç noktası — burası ısınıyorsa piyasa cesur."),
    dict(key="IWM/SPY", num="IWM", den="SPY", grup="Hisse içi", risk=+1,
         ad="Küçük ölçek / S&P 500",
         not_="Faiz indirimi ve ekonomik genişleme beklentisinde yükselir. "
              "Yükselişin tabana yayıldığını gösterir."),
    dict(key="RSP/SPY", num="RSP", den="SPY", grup="Hisse içi", risk=+1,
         ad="Eşit ağırlık / S&P 500 (genişlik)",
         not_="Düşüyorsa endeksi birkaç dev hisse taşıyor — sağlıksız "
              "yükseliş. Yükseliyorsa katılım geniş."),
    dict(key="XLY/XLP", num="XLY", den="XLP", grup="Hisse içi", risk=+1,
         ad="İhtiyari / Temel tüketim",
         not_="Tüketicinin keyfi harcaması mı, zorunlu harcaması mı öne "
              "çıkıyor? Klasik ofans/defans göstergesi."),
    dict(key="EEM/SPY", num="EEM", den="SPY", grup="Hisse içi", risk=+1,
         ad="Gelişen piyasalar / ABD",
         not_="Zayıf dolar ve küresel risk iştahında gelişen piyasalara "
              "para akar."),
    # Varlık sınıfları
    dict(key="SPY/TLT", num="SPY", den="TLT", grup="Varlık sınıfları", risk=+1,
         ad="Hisse / Tahvil",
         not_="Para tahvilden hisseye mi geçiyor? En temel risk-on/off "
              "göstergesi."),
    dict(key="HYG/TLT", num="HYG", den="TLT", grup="Varlık sınıfları", risk=+1,
         ad="Riskli tahvil / Hazine",
         not_="Kredi piyasası hisseden önce uyarır. Yükseliyorsa borç "
              "verenler risk almaya istekli."),
    dict(key="GC=F/SPY", num="GC=F", den="SPY", grup="Varlık sınıfları", risk=-1,
         ad="Altın / S&P 500",
         not_="Yükseliyorsa para hisseden güvenli limana kaçıyor ya da "
              "enflasyon/para basımı korkusu var."),
    dict(key="SI=F/GC=F", num="SI=F", den="GC=F", grup="Varlık sınıfları", risk=+1,
         ad="Gümüş / Altın",
         not_="Gümüş hem değerli hem sanayi metali. Altını geçiyorsa metal "
              "rallisi spekülatif/büyüme ayağına geçmiş demektir."),
    dict(key="HG=F/GC=F", num="HG=F", den="GC=F", grup="Varlık sınıfları", risk=+1,
         ad="Bakır / Altın",
         not_="Dr. Bakır: büyüme beklentisi. Yükseliyorsa ekonomi "
              "hızlanıyor; düşüyorsa resesyon korkusu."),
    dict(key="BZ=F", num="BZ=F", den=None, grup="Varlık sınıfları", risk=0,
         ad="Brent petrol",
         not_="Sert yükseliş enflasyon ve jeopolitik risk demek (hisseler "
              "için olumsuz); ılımlı yükseliş talep canlılığı."),
    dict(key="DX-Y.NYB", num="DX-Y.NYB", den=None, grup="Varlık sınıfları", risk=-1,
         ad="Dolar endeksi (DXY)",
         not_="Güçlü dolar küresel likiditeyi emer: riskli varlıklar, emtia, "
              "gelişen piyasalar ve kripto baskılanır."),
    # Kripto
    dict(key="BTC-USD", num="BTC-USD", den=None, grup="Kripto", risk=+1,
         ad="Bitcoin",
         not_="Kripto iştahının ana göstergesi ve küresel likiditeye en "
              "duyarlı varlık."),
    dict(key="TOTAL", num="TOTAL", den=None, grup="Kripto", risk=+1,
         ad="Toplam kripto piyasa değeri (TOTAL)",
         not_="Kripto piyasasına net para girişi."),
    dict(key="BTC.D", num="BTC.D", den=None, grup="Kripto", risk=-1,
         ad="BTC hakimiyeti (BTC.D)",
         not_="Yükseliyorsa kripto içinde para altcoinlerden BTC'ye "
              "kaçıyor (temkin). Düşüyorsa altcoinlere akıyor."),
    dict(key="ETH/BTC", num="ETH/BTC", den=None, grup="Kripto", risk=+1,
         ad="ETH / BTC",
         not_="Altcoin sezonunun ilk basamağı genellikle ETH'nin BTC'yi "
              "geçmesidir."),
    dict(key="TOTAL3", num="TOTAL3", den=None, grup="Kripto", risk=+1,
         ad="Altcoinler (TOTAL3: BTC ve ETH hariç)",
         not_="BTC ve ETH dışındaki tüm piyasaya para girişi."),
    dict(key="OTHERS/BTC", num="OTHERS/BTC", den=None, grup="Kripto", risk=+1,
         ad="Küçük altcoinler / BTC (OTHERS/BTC)",
         not_="Kripto riskinin en uç noktası. Yükselişi tam bir altcoin "
              "sezonu ve aşırı iştah anlamına gelir."),
    # Likidite
    dict(key="NET_LIQ", num="NET_LIQ", den=None, grup="Likidite", risk=+1,
         ad="Fed net likiditesi (bilanço − TGA − ters repo)",
         not_="Piyasadaki 'para musluğu'. Artıyorsa sistem para basıyor "
              "demektir; hisse ve kripto tarihsel olarak bunu takip eder."),
    dict(key="M2SL", num="M2SL", den=None, grup="Likidite", risk=+1,
         ad="M2 para arzı",
         not_="Ekonomideki toplam para miktarı. Yıllık büyümesi hızlanıyorsa "
              "varlık fiyatları desteklenir (aylık veri, gecikmeli)."),
]
PAIR = {p["key"]: p for p in PAIRS}

# Bir yatırımcının "benim hissem hangi segmentte?" sorusuna cevap
SEGMENTS: dict[str, dict[str, Any]] = {
    "Nasdaq / büyüme & teknoloji": dict(
        pairs=["QQQ/SPY", "^IXIC/^GSPC", "SMH/SPY", "ARKK/QQQ"],
        aciklama="Nasdaq'ta işlem gören büyüme, yazılım, yarı iletken ve "
                 "yapay zekâ hisseleri."),
    "S&P 500 / geniş piyasa": dict(
        pairs=["RSP/SPY", "SPY/TLT", "XLY/XLP"],
        aciklama="Büyük, kârlı şirketler; sanayi, finans, sağlık, tüketim."),
    "Küçük ölçek (Russell 2000)": dict(
        pairs=["IWM/SPY", "RSP/SPY", "HYG/TLT"],
        aciklama="Faize ve ekonomik döngüye en duyarlı küçük şirketler."),
    "Gelişen piyasalar": dict(
        pairs=["EEM/SPY", "DX-Y.NYB"],
        aciklama="Çin, Hindistan, Brezilya vb. — dolardan ters etkilenir."),
    "Altın, gümüş ve madenciler": dict(
        pairs=["GC=F/SPY", "SI=F/GC=F", "DX-Y.NYB"],
        aciklama="Değerli metaller ve madenci hisseleri."),
    "Enerji ve emtia": dict(
        pairs=["BZ=F", "HG=F/GC=F", "DX-Y.NYB"],
        aciklama="Petrol, gaz, bakır, sanayi metalleri ve üreticileri."),
    "Kripto ve kripto hisseleri": dict(
        pairs=["BTC-USD", "TOTAL", "BTC.D", "OTHERS/BTC", "NET_LIQ"],
        aciklama="BTC, altcoinler, madenciler (MARA, RIOT…), borsalar."),
    "Tahvil / defansif": dict(
        pairs=["SPY/TLT", "HYG/TLT"],
        aciklama="Hazine tahvilleri, temettü ve defansif sektörler. "
                 "Burada göstergeler TERS okunur: hisse/tahvil düşüyorsa "
                 "tahvil lehine."),
}
# Segment yönü: tahvil/defansif ve altın, risk iştahının tersinden beslenir
SEGMENT_INVERT = {"Tahvil / defansif"}
SEGMENT_ABS = {"Altın, gümüş ve madenciler": {"GC=F/SPY": +1},
               "Enerji ve emtia": {"BZ=F": +1}}


# --------------------------------------------------------------------------
# Repo veri dosyaları (GitHub Actions)
# --------------------------------------------------------------------------
def _data_path(name: str) -> Path | None:
    for base in (Path(__file__).resolve().parent, Path.cwd()):
        p = base / "data" / name
        if p.exists():
            return p
    return None


def read_crypto_caps() -> pd.DataFrame:
    p = _data_path("crypto_caps.csv")
    if p is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(p, index_col=0, parse_dates=True).sort_index()
    except Exception:
        return pd.DataFrame()


def read_liquidity() -> pd.DataFrame:
    p = _data_path("liquidity.csv")
    if p is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(p, index_col=0, parse_dates=True).sort_index()
    except Exception:
        return pd.DataFrame()


# --------------------------------------------------------------------------
# Seri istatistikleri
# --------------------------------------------------------------------------
def build_series(prices: dict[str, pd.DataFrame], crypto: pd.DataFrame,
                 liq: pd.DataFrame) -> dict[str, pd.Series]:
    """Her rotasyon anahtarı için günlük seri (oran ya da tekil)."""
    close = {t: df["Close"].dropna() for t, df in prices.items()
             if df is not None and not df.empty}
    for c in ("TOTAL", "TOTAL3", "OTHERS", "BTC.D", "ETH/BTC", "OTHERS/BTC"):
        if c in crypto.columns:
            close[c] = crypto[c].dropna()
    for c in ("NET_LIQ", "M2SL"):
        if c in liq.columns:
            close[c] = liq[c].dropna()

    out: dict[str, pd.Series] = {}
    for p in PAIRS:
        a = close.get(p["num"])
        if a is None or len(a) < 30:
            continue
        if p["den"]:
            b = close.get(p["den"])
            if b is None or len(b) < 30:
                continue
            a, b = a.copy(), b.copy()
            a.index = pd.to_datetime(a.index).tz_localize(None).normalize()
            b.index = pd.to_datetime(b.index).tz_localize(None).normalize()
            a = a[~a.index.duplicated()]
            b = b[~b.index.duplicated()]
            j = pd.concat([a, b], axis=1, join="inner").dropna()
            if len(j) < 30:
                continue
            s = j.iloc[:, 0] / j.iloc[:, 1]
        else:
            s = a.copy()
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            s = s[~s.index.duplicated()]
        out[p["key"]] = s.astype(float)
    return out


def _pct(s: pd.Series, n: int) -> float:
    if len(s) <= n or not s.iloc[-1 - n]:
        return np.nan
    return float((s.iloc[-1] / s.iloc[-1 - n] - 1) * 100)


def quarterly_ohlc(s: pd.Series) -> pd.DataFrame:
    q = s.resample("QE").agg(["first", "max", "min", "last"]).dropna()
    q.columns = ["Open", "High", "Low", "Close"]
    return q


def breakout_info(s: pd.Series, freq: str = "QE", look: int = 8) -> dict[str, Any]:
    """
    Kapanış bazlı kırılım: içinde bulunulan dönemin kapanışı, önceki `look`
    dönemin en yüksek KAPANIŞINI (ekran görüntüsündeki yatay direnç) geçti mi?
    """
    r = s.resample(freq).last().dropna()
    if len(r) < look + 2:
        return {"kirilim": False, "direnc": np.nan, "uzaklik": np.nan}
    prev = r.iloc[-look - 1:-1]
    res = float(prev.max())
    last = float(r.iloc[-1])
    br = last > res
    # bir önceki dönem de kırmış mıydı? (taze kırılım ayrımı)
    prev2 = r.iloc[-look - 2:-2]
    br_prev = float(r.iloc[-2]) > float(prev2.max()) if len(prev2) else False
    return {"kirilim": br, "taze": br and not br_prev, "direnc": res,
            "uzaklik": (last / res - 1) * 100}


def series_stats(s: pd.Series, risk: int) -> dict[str, Any]:
    """
    Bir oran/serinin akış durumu.
    Skor −100…+100: pozitif = pay lehine akış (oran yükseliyor).
    Faz: güçlü akış / akış başlıyor (ERKEN) / yavaşlıyor / çıkış.
    """
    s = s.dropna()
    n = len(s)
    c20, c63 = _pct(s, 20), _pct(s, 63)
    sma50 = s.rolling(50).mean()
    sma200 = s.rolling(200).mean()
    last = float(s.iloc[-1])
    above50 = bool(last > sma50.iloc[-1]) if n >= 50 else None
    above200 = bool(last > sma200.iloc[-1]) if n >= 200 else None

    # 20 günlük değişimin tarihsel oynaklığa göre normalize edilmesi
    ch20 = s.pct_change(20) * 100
    sd = float(ch20.dropna().tail(750).std()) if ch20.notna().sum() > 60 else np.nan
    z20 = c20 / sd if sd and np.isfinite(sd) and np.isfinite(c20) else 0.0
    ch63 = s.pct_change(63) * 100
    sd63 = float(ch63.dropna().tail(750).std()) if ch63.notna().sum() > 60 else np.nan
    z63 = c63 / sd63 if sd63 and np.isfinite(sd63) and np.isfinite(c63) else 0.0

    score = (40 * np.tanh(z20) + 30 * np.tanh(z63)
             + (15 if above50 else -15 if above50 is not None else 0)
             + (15 if above200 else -15 if above200 is not None else 0))

    # ivme: 20 günlük değişim 10 gün öncekine göre artıyor mu
    accel = (float(ch20.iloc[-1] - ch20.iloc[-11])
             if n > 31 and np.isfinite(ch20.iloc[-11]) else np.nan)
    # SMA50'yi son 15 günde yukarı kesti mi (erken dönüş)
    cross_up = False
    if n >= 65:
        above = (s > sma50).tail(16)
        cross_up = bool(above.iloc[-1] and not above.iloc[:-1].all()
                        and (~above.iloc[:-1]).any())

    long_neg = (np.isfinite(c63) and c63 <= 0) or above200 is False
    if np.isfinite(c20) and c20 > 0 and long_neg and (
            cross_up or (np.isfinite(accel) and accel > 0)):
        faz = "🌱 Yükseliş başlıyor"
    elif np.isfinite(c20) and c20 > 0 and np.isfinite(c63) and c63 > 0:
        faz = "🚀 Güçlü yükseliş"
    elif np.isfinite(c63) and c63 > 0 and np.isfinite(c20) and c20 <= 0:
        faz = "🌤️ Yükseliş yavaşlıyor"
    elif np.isfinite(c20) and c20 < 0 and np.isfinite(c63) and c63 < 0:
        faz = "🩸 Düşüş"
    else:
        faz = "⚖️ Kararsız"

    q = breakout_info(s, "QE", 8)
    m = breakout_info(s, "ME", 12)
    return {"Son": last, "20G %": c20, "63G %": c63, "Skor": float(score),
            "Risk Etkisi": float(score) * risk if risk else 0.0,
            "Faz": faz, "50G üstü": above50, "200G üstü": above200,
            "İvme": accel, "Çeyreklik Kırılım": q["kirilim"],
            "Taze Çeyreklik Kırılım": q.get("taze", False),
            "Çeyreklik Direnç": q["direnc"], "Dirence Uzaklık %": q["uzaklik"],
            "Aylık Kırılım": m["kirilim"]}


def flow_sentence(p: dict[str, Any], st: dict[str, Any]) -> str:
    """İnsan dilinde tek cümle: para nereden nereye."""
    sc = st["Skor"]
    if p["den"]:
        a, b = NAMES.get(p["num"], p["num"]), NAMES.get(p["den"], p["den"])
        if sc >= 15:
            return f"Para {b} → {a} yönünde akıyor"
        if sc <= -15:
            return f"Para {a} → {b} yönünde akıyor"
        return f"{a} ile {b} arasında belirgin akış yok"
    ad = p["ad"]
    if p["key"] == "BTC.D":
        if sc >= 15:
            return "Kripto içinde para altcoinlerden BTC'ye kaçıyor"
        if sc <= -15:
            return "Kripto içinde para BTC'den altcoinlere akıyor"
        return "BTC hakimiyeti yatay"
    if sc >= 15:
        return f"{ad} yükseliyor — para giriyor"
    if sc <= -15:
        return f"{ad} düşüyor — para çıkıyor"
    return f"{ad} yatay"


def rotation_table(series: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for p in PAIRS:
        s = series.get(p["key"])
        if s is None or len(s) < 30:
            continue
        st = series_stats(s, p["risk"])
        re_ = st["Risk Etkisi"]
        okuma = ("➖ nötr" if not p["risk"] or abs(re_) < 15
                 else "✅ risk iştahı" if re_ > 0 else "❌ risk kaçışı")
        rows.append({"Anahtar": p["key"], "Grup": p["grup"], "Gösterge": p["ad"],
                     "Akış": flow_sentence(p, st), "Risk Okuması": okuma,
                     **st, "Not": p["not_"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 1) RİSK AÇIK MI?
# --------------------------------------------------------------------------
def _avg(*xs: float) -> float:
    v = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else np.nan


def _durum(x: float) -> str:
    if not np.isfinite(x):
        return "➖"
    return "✅" if x >= 60 else "⚠️" if x >= 40 else "❌"


def risk_pillars(scores: dict[str, float], rot: pd.DataFrame) -> list[dict[str, Any]]:
    """Altı sütun; her biri 0–100 (yüksek = risk iştahı)."""
    def rs(key: str) -> float:
        if rot.empty or key not in set(rot["Anahtar"]):
            return np.nan
        return float(np.clip(50 + rot.loc[rot["Anahtar"] == key, "Risk Etkisi"].iloc[0] / 2,
                             0, 100))

    g = scores.get
    liq = _avg(rs("NET_LIQ"), g("dollar", np.nan))
    pillars = [
        dict(ad="Trend", skor=g("trend", np.nan),
             ne="S&P 500'ün 50 ve 200 günlük ortalamalara göre dizilimi.",
             neden="Fiyat ana trendin üstündeyken alım sinyallerinin isabeti "
                   "belirgin şekilde yüksektir. Trendin altında 'ucuz' görünen "
                   "hisse daha da ucuzlayabilir."),
        dict(ad="Korku (VIX)", skor=_avg(g("vix", np.nan), g("vix_ts", np.nan)),
             ne="VIX seviyesi ve vade yapısı (VIX / VIX3M).",
             neden="VIX 20'nin altında ve vade yapısı normalken piyasa sakin; "
                   "VIX3M'in üstüne çıkan VIX yakın vadeli paniğe işaret eder."),
        dict(ad="Kredi", skor=_avg(g("credit", np.nan), rs("HYG/TLT")),
             ne="Riskli şirket tahvilinin hazineye göre performansı (HYG/TLT).",
             neden="Kredi piyasası hisseden önce uyarır. Borç verenler geri "
                   "çekiliyorsa hisse yükselişi kırılgandır."),
        dict(ad="Likidite", skor=liq,
             ne="Fed net likiditesi (bilanço − TGA − ters repo) ve dolar.",
             neden="Piyasaya giren para artıyorsa varlık fiyatları rüzgârı "
                   "arkasına alır; güçlü dolar ise küresel likiditeyi emer."),
        dict(ad="Katılım", skor=_avg(g("breadth", np.nan), g("smallcap", np.nan)),
             ne="Eşit ağırlıklı S&P ve küçük şirketlerin performansı.",
             neden="Yükselişe geniş katılım varsa sağlıklıdır. Endeksi 5–10 dev "
                   "hisse taşıyorsa, sizin seçtiğiniz hisse büyük ihtimalle "
                   "o rallinin dışında kalır."),
        dict(ad="Spekülatif iştah", skor=_avg(g("crypto", np.nan), rs("ARKK/QQQ"),
                                              rs("OTHERS/BTC")),
             ne="Bitcoin, spekülatif büyüme (ARKK) ve küçük altcoinler.",
             neden="Riskin en uç noktası. Burada iştah varsa yatırımcı cesur; "
                   "yoksa piyasa yalnızca güvenli büyük şirketleri alıyor."),
    ]
    for p in pillars:
        p["durum"] = _durum(p["skor"])
    return pillars


def risk_verdict(pillars: list[dict[str, Any]], regime_label: str = ""
                 ) -> dict[str, Any]:
    w = {"Trend": 0.25, "Korku (VIX)": 0.15, "Kredi": 0.15, "Likidite": 0.15,
         "Katılım": 0.15, "Spekülatif iştah": 0.15}
    tot, ws = 0.0, 0.0
    for p in pillars:
        if np.isfinite(p["skor"]):
            tot += w[p["ad"]] * p["skor"]
            ws += w[p["ad"]]
    score = tot / ws if ws else np.nan
    n_ok = sum(p["durum"] == "✅" for p in pillars)
    n_bad = sum(p["durum"] == "❌" for p in pillars)
    trend_ok = next((p["skor"] for p in pillars if p["ad"] == "Trend"), np.nan)

    if np.isfinite(score) and score >= 60 and (not np.isfinite(trend_ok) or trend_ok >= 50):
        etiket, renk = "🟢 RİSK AÇIK", "pos"
        ozet = ("Piyasa risk almayı ödüllendiriyor. Alım sinyalleri tam "
                "pozisyonla değerlendirilebilir; asıl soru 'nereye' — 2. adıma "
                "geçin.")
        boyut = "Normal pozisyon boyutu"
    elif np.isfinite(score) and (score <= 40 or n_bad >= 4):
        etiket, renk = "🔴 RİSK KAPALI", "neg"
        ozet = ("Piyasa riskten kaçıyor. Yeni alımlar düşük isabetli olur; "
                "nakit, kısa vade ve koruma öncelikli. İstisna: 2. adımda "
                "güçlü akış alan defansif segmentler.")
        boyut = "Yeni pozisyon yok ya da çeyrek boyut"
    else:
        etiket, renk = "🟡 TEMKİNLİ", ""
        ozet = ("Göstergeler karışık. Sadece en güçlü kurulumlar, küçük "
                "pozisyon ve sıkı stop. Zayıf sütunları aşağıda görün — "
                "onlar düzelince risk tam açılır.")
        boyut = "Yarım pozisyon"
    return {"etiket": etiket, "renk": renk, "skor": score, "ozet": ozet,
            "boyut": boyut, "n_ok": n_ok, "n_bad": n_bad, "rejim": regime_label}


# --------------------------------------------------------------------------
# 2) PARA NEREYE AKIYOR? — segment kararları
# --------------------------------------------------------------------------
def segment_verdicts(rot: pd.DataFrame) -> pd.DataFrame:
    if rot.empty:
        return pd.DataFrame()
    by = rot.set_index("Anahtar")
    rows = []
    for seg, cfg in SEGMENTS.items():
        vals, detay = [], []
        for k in cfg["pairs"]:
            if k not in by.index:
                continue
            r = by.loc[k]
            sign = SEGMENT_ABS.get(seg, {}).get(k, PAIR[k]["risk"] or 1)
            v = float(r["Skor"]) * sign
            if seg in SEGMENT_INVERT:
                v = -v
            vals.append(v)
            detay.append(f"{PAIR[k]['ad']}: {r['Faz']}")
        if not vals:
            continue
        sc = float(np.mean(vals))
        erken = any(by.loc[k, "Faz"] == "🌱 Yükseliş başlıyor"
                    for k in cfg["pairs"] if k in by.index)
        if sc >= 25:
            k_ = "🟢 İştahlı"
        elif sc >= 5:
            k_ = "🟡 Isınıyor" if erken else "🟡 Hafif olumlu"
        elif sc > -10:
            k_ = "⚪ Nötr" + (" (erken dönüş sinyali var)" if erken else "")
        else:
            k_ = "🔴 İştahsız"
        rows.append({"Segment": seg, "Karar": k_, "Skor": sc,
                     "Erken Sinyal": "🌱" if erken else "",
                     "Dayanak": " · ".join(detay), "Kapsam": cfg["aciklama"]})
    return pd.DataFrame(rows).sort_values("Skor", ascending=False)


def money_flow_summary(rot: pd.DataFrame, top: int = 4) -> list[str]:
    """En güçlü akışları sade cümlelere çevirir."""
    if rot.empty:
        return []
    r = rot.copy()
    r["abs"] = r["Skor"].abs()
    r = r[r["abs"] >= 25].sort_values("abs", ascending=False)
    out = [f"{row['Akış']} — {row['Risk Okuması']}" for _, row in r.head(top).iterrows()]
    erken = rot[rot["Faz"] == "🌱 Yükseliş başlıyor"]
    for _, row in erken.head(3).iterrows():
        out.append(f"🌱 Erken: {row['Gösterge']} — {row['Akış']}")
    kir = rot[rot["Taze Çeyreklik Kırılım"] == True]  # noqa: E712
    for _, row in kir.iterrows():
        out.append(f"📐 {row['Gösterge']} çeyreklik grafikte direnci kırdı "
                   f"(önceki 8 çeyreğin zirvesi)")
    return out


# --------------------------------------------------------------------------
# 3) HANGİ TEMA İVMELENİYOR? — göreli rotasyon grafiği (RRG)
# --------------------------------------------------------------------------
def rrg_lines(close: pd.Series, bench: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    RS-Oranı: (fiyat / SPY) oranının 50 günlük ortalamasına göre konumu ×100.
      100'ün üstü = endeksten güçlü.
    RS-Momentum: RS-Oranının 10 gün önceye göre değişimi ×100.
      100'ün üstü = göreli güç artıyor.
    """
    j = pd.concat([close, bench], axis=1, join="inner").dropna()
    rs = j.iloc[:, 0] / j.iloc[:, 1]
    ratio = (100 * rs / rs.rolling(50).mean()).ewm(span=5, adjust=False).mean()
    mom = (100 * ratio / ratio.shift(10)).ewm(span=3, adjust=False).mean()
    return ratio, mom


RRG_Q = {
    "lider": ("🚀 Lider", "Endeksten güçlü ve güçlenmeye devam ediyor."),
    "zayif": ("🌤️ Yoruluyor", "Hâlâ endeksten güçlü ama göreli güç azalıyor — "
                             "kâr realizasyonu bölgesi."),
    "geride": ("🩸 Geride", "Endeksten zayıf ve zayıflamaya devam ediyor."),
    "iyilesen": ("🌱 İyileşiyor", "Endeksten zayıf AMA göreli güç artıyor — "
                                 "akıllı paranın erken girdiği bölge."),
}


def rrg_quadrant(r: float, m: float) -> str:
    if not (np.isfinite(r) and np.isfinite(m)):
        return "geride"
    if r >= 100 and m >= 100:
        return "lider"
    if r >= 100:
        return "zayif"
    if m >= 100:
        return "iyilesen"
    return "geride"


def theme_index(prices: dict[str, pd.DataFrame], members: list[str]
                ) -> tuple[pd.Series | None, pd.Series | None]:
    """Eşit ağırlıklı tema endeksi (her ETF başlangıçta 1'e normalize) ve
    toplam dolar hacmi."""
    cl, dv = [], []
    for t in members:
        df = prices.get(t)
        if df is None or df.empty or len(df) < 80:
            continue
        c = df["Close"].dropna()
        cl.append(c / c.iloc[0])
        dv.append((df["Close"] * df["Volume"]).rename(t))
    if not cl:
        return None, None
    idx = pd.concat(cl, axis=1).ffill().mean(axis=1)
    vol = pd.concat(dv, axis=1).sum(axis=1, min_count=1)
    return idx, vol


def early_score(ratio: pd.Series, mom: pd.Series, vol: pd.Series | None,
                close: pd.Series) -> dict[str, Any]:
    r, m = float(ratio.iloc[-1]), float(mom.iloc[-1])
    q = rrg_quadrant(r, m)
    m_up = float(mom.iloc[-1] - mom.iloc[-6]) if len(mom) > 6 else np.nan
    r_up = float(ratio.iloc[-1] - ratio.iloc[-6]) if len(ratio) > 6 else np.nan
    # liderliğe yeni mi geçti? (son 10 günde iyileşen → lider)
    qs = [rrg_quadrant(a, b) for a, b in zip(ratio.tail(11), mom.tail(11))]
    taze_lider = q == "lider" and any(x in ("iyilesen", "geride") for x in qs[:-1])

    vol_ok = False
    vr = np.nan
    if vol is not None and len(vol.dropna()) > 70:
        v10 = vol.tail(10).mean()
        v60 = vol.tail(70).head(60).mean()
        vr = float(v10 / v60) if v60 else np.nan
        vol_ok = bool(np.isfinite(vr) and vr >= 1.15)

    s = {"iyilesen": 45, "lider": 30, "zayif": 8, "geride": 0}[q]
    if q == "iyilesen" and np.isfinite(m_up) and m_up > 0:
        s += 15
    if taze_lider:
        s += 25
    if np.isfinite(r_up) and r_up > 0:
        s += 10
    if vol_ok:
        s += 15
    c5 = _pct(close, 5)
    if np.isfinite(c5) and c5 > 0:
        s += 5
    return {"RS-Oran": r, "RS-Mom": m, "Bölge": RRG_Q[q][0], "_q": q,
            "Taze Lider": taze_lider, "Hacim Oranı": vr, "Hacim Onayı": vol_ok,
            "Erken Skor": int(np.clip(s, 0, 100)),
            "1H %": c5, "1A %": _pct(close, 21), "3A %": _pct(close, 63)}


def theme_rotation(prices: dict[str, pd.DataFrame], themes: dict[str, list[str]],
                   bench: str = "SPY") -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Tema bazında RRG + erken akış skoru. Döner: (tablo, {tema: kuyruk})."""
    b = prices.get(bench)
    if b is None or b.empty:
        return pd.DataFrame(), {}
    bc = b["Close"].dropna()
    rows, tails = [], {}
    for tema, members in themes.items():
        idx, vol = theme_index(prices, members)
        if idx is None:
            continue
        ratio, mom = rrg_lines(idx, bc)
        if ratio.dropna().empty or mom.dropna().empty:
            continue
        e = early_score(ratio.dropna(), mom.dropna(), vol, idx)
        rows.append({"Tema": tema, **e, "ETF'ler": ", ".join(members)})
        t = pd.DataFrame({"RS-Oran": ratio, "RS-Mom": mom}).dropna()
        tails[tema] = t.iloc[::-5].head(6).iloc[::-1]   # son 6 hafta
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Erken Skor", ascending=False).reset_index(drop=True)
    return df, tails


def etf_rotation(prices: dict[str, pd.DataFrame], etfs: list[str],
                 bench: str = "SPY") -> pd.DataFrame:
    b = prices.get(bench)
    if b is None or b.empty:
        return pd.DataFrame()
    bc = b["Close"].dropna()
    rows = []
    for t in etfs:
        df = prices.get(t)
        if df is None or len(df) < 80:
            continue
        c = df["Close"].dropna()
        ratio, mom = rrg_lines(c, bc)
        if ratio.dropna().empty or mom.dropna().empty:
            continue
        e = early_score(ratio.dropna(), mom.dropna(), df["Close"] * df["Volume"], c)
        rows.append({"ETF": t, **e})
    out = pd.DataFrame(rows)
    return out.sort_values("Erken Skor", ascending=False) if not out.empty else out


# --------------------------------------------------------------------------
# 4) BU TEMADA HANGİ HİSSE?
# --------------------------------------------------------------------------
BULL = {"💎 DIAMOND AL", "⭐ GOLDEN STAR", "🎣 LİKİDİTE SÜPÜRMESİ", "🐋 TOPLAMA",
        "🚀 AFTERBURNER", "🟢 GİRİŞ BÖLGESİ", "📈 MINERVINI MVP",
        "🎯 SIKIŞMA PATLADI", "⚡ PRO ↗ RETAIL"}
BEAR = {"🩸 GÜÇLÜ RİSK", "⛔ DIAMOND SAT", "🐋 DAĞITIM", "🔻 AFTERBURNER AYI",
        "⚡ PRO ↘ RETAIL"}

STOCK_CLASSES = {
    "🎯 Alım adayı": "Alım sinyali var, temasından güçlü ve kurumsal para "
                    "giriyor. Giriş planı yapılabilir.",
    "🌱 Geride kaldı, toparlanıyor": "Son bir ayda temasının gerisinde ama son "
                                     "hafta göreli gücü dönmüş ve whale "
                                     "artıyor — yetişme (catch-up) adayı.",
    "🚀 Lider": "Temasını sürükleyen hisse. Trend devam ediyor; yeni giriş "
               "için geri çekilme beklemek daha iyi risk/ödül verir.",
    "🔥 Lider ama uzamış": "Lider fakat tükenme işareti var ya da çok hızlı "
                          "yükselmiş. Kovalamayın.",
    "⚪ İzle": "Belirgin bir avantajı yok; izleme listesinde tutun.",
    "⛔ Uzak dur": "Dağıtım, risk ya da satış sinyali var.",
    "📅 Bilanço yakın": "7 gün içinde bilanço var — gap riski stop mantığını "
                       "bozar. Bilanço sonrasına bırakın.",
    "💧 Likidite düşük": "Günlük işlem hacmi eşiğin altında; spread geniş, "
                        "stop kayar.",
}


def classify_stock(d: dict[str, Any], min_liq_m: float = 10.0) -> tuple[str, list[str]]:
    """Döner: (sınıf, [gerekçe parçaları])."""
    why: list[str] = []
    sig = d.get("Sinyal", "")
    rs1a, rs1h = d.get("Temaya Göre 1A", np.nan), d.get("Temaya Göre 1H", np.nan)
    dw5 = d.get("ΔWHALE 5B", np.nan)
    liq = d.get("Hacim ($M)", np.nan)
    kg = pd.to_numeric(d.get("Kalan Gün"), errors="coerce")
    dsv = d.get("ΔSV pp", np.nan)
    p1h = d.get("1 Hafta %", np.nan)

    if np.isfinite(rs1a):
        why.append(f"temaya göre {rs1a:+.1f}% (1A)")
    if np.isfinite(dw5):
        why.append("whale ↑" if dw5 > 1 else "whale ↓" if dw5 < -1 else "whale yatay")
    if np.isfinite(dsv):
        if dsv <= -3:
            why.append(f"short azalıyor ({dsv:+.1f}p)")
        elif dsv >= 3:
            why.append(f"short artıyor ({dsv:+.1f}p)"
                       + (" — fiyat düşerken" if np.isfinite(p1h) and p1h < 0 else ""))
    if np.isfinite(kg) and kg >= 0:
        why.append(f"bilanço {int(kg)} gün")
    if sig and sig != "⚪ BEKLE":
        why.append(sig)

    if np.isfinite(liq) and liq < min_liq_m:
        return "💧 Likidite düşük", [f"günlük ${liq:.1f}M"] + why
    if np.isfinite(kg) and 0 <= kg <= 7:
        return "📅 Bilanço yakın", why
    if sig in BEAR:
        return "⛔ Uzak dur", why
    short_bad = np.isfinite(dsv) and dsv >= 3 and np.isfinite(p1h) and p1h < 0
    if (sig in BULL and d.get("Skor", 0) >= 50
            and (not np.isfinite(rs1h) or rs1h > 0) and not short_bad):
        return "🎯 Alım adayı", why
    if (np.isfinite(rs1a) and rs1a < 0 and np.isfinite(rs1h) and rs1h > 0
            and np.isfinite(dw5) and dw5 > 0):
        return "🌱 Geride kaldı, toparlanıyor", why
    if np.isfinite(rs1a) and rs1a > 0:
        if d.get("_exhausted") or sig == "⚠️ TÜKENME" or (
                np.isfinite(d.get("1 Ay %", np.nan)) and d.get("1 Ay %") > 35):
            return "🔥 Lider ama uzamış", why
        return "🚀 Lider", why
    return "⚪ İzle", why


CLASS_ORDER = ["🎯 Alım adayı", "🌱 Geride kaldı, toparlanıyor", "🚀 Lider",
               "🔥 Lider ama uzamış", "⚪ İzle", "📅 Bilanço yakın",
               "💧 Likidite düşük", "⛔ Uzak dur"]


def funnel_score(d: dict[str, Any]) -> int:
    """0–100: swing skoru + temaya göre güç + whale + short akışı."""
    s = 0.55 * float(d.get("Skor", 0) or 0)
    rs = d.get("Temaya Göre 1A", np.nan)
    if np.isfinite(rs):
        s += 15 * np.tanh(rs / 10) + 10
    rs1h = d.get("Temaya Göre 1H", np.nan)
    if np.isfinite(rs1h):
        s += 8 * np.tanh(rs1h / 4)
    dw = d.get("ΔWHALE 5B", np.nan)
    if np.isfinite(dw):
        s += 8 * np.tanh(dw / 8)
    dsv = d.get("ΔSV pp", np.nan)
    if np.isfinite(dsv):
        s -= 6 * np.tanh(dsv / 5)
    return int(np.clip(round(s), 0, 100))


def build_stock_table(scan_df: pd.DataFrame, theme_close: pd.Series | None,
                      prices: dict[str, pd.DataFrame] | None = None,
                      earn: pd.DataFrame | None = None,
                      short: pd.DataFrame | None = None,
                      min_liq_m: float = 10.0, swing_score=None) -> pd.DataFrame:
    """Tarama tablosunu tema içi göreli güç, bilanço ve short ile birleştirir."""
    if scan_df is None or scan_df.empty:
        return pd.DataFrame()
    df = scan_df.copy()
    if "Sinyal" in df.columns:
        df = df[df["Sinyal"] != "⚫ VERİ YOK"].copy()
    if df.empty:
        return df
    th1a = _pct(theme_close, 21) if theme_close is not None else np.nan
    th1h = _pct(theme_close, 5) if theme_close is not None else np.nan
    df["Temaya Göre 1A"] = pd.to_numeric(df.get("1 Ay %"), errors="coerce") - th1a
    df["Temaya Göre 1H"] = pd.to_numeric(df.get("1 Hafta %"), errors="coerce") - th1h

    if prices:
        hi = {}
        for t in df["Sembol"]:
            p = prices.get(t)
            if p is not None and len(p) > 20:
                c = p["Close"].dropna()
                hi[t] = (c.iloc[-1] / c.tail(252).max() - 1) * 100
        df["Zirveye Uzaklık %"] = df["Sembol"].map(hi)

    if earn is not None and not earn.empty:
        keep = [c for c in ("Hisse", "Kalan Gün", "Bilanço", "P/S Durum",
                            "Hedef", "Potansiyel %") if c in earn.columns]
        e = earn[keep].rename(columns={"Hisse": "Sembol"})
        df = df.merge(e, on="Sembol", how="left")
    if short is not None and not short.empty:
        keep = [c for c in ("Sembol", "SV% 5G", "ΔSV pp") if c in short.columns]
        df = df.merge(short[keep], on="Sembol", how="left")

    if swing_score is not None:
        df["Skor"] = [swing_score(r) for r in df.to_dict("records")]
    cls, why = [], []
    for r in df.to_dict("records"):
        c, w = classify_stock(r, min_liq_m)
        cls.append(c)
        why.append(" · ".join(w))
    df["Durum"] = cls
    df["Neden"] = why
    df["Huni Skoru"] = [funnel_score(r) for r in df.to_dict("records")]
    df["_ord"] = df["Durum"].map({c: i for i, c in enumerate(CLASS_ORDER)})
    return (df.sort_values(["_ord", "Huni Skoru"], ascending=[True, False])
              .drop(columns=["_ord"]).reset_index(drop=True))

# ==========================================================================
# KAYNAK: apex/fundchart.py
# ==========================================================================
# apex/fundchart.py — TEK HİSSE DEĞERLEME GRAFİKLERİ
#
# 1) F/K (TTM) geçmişi + akran medyanı
#    Hissenin günlük fiyatı / son 4 çeyreğin HBK toplamı. Yanında seçilen
#    akranların aynı hesapla bulunan F/K medyanı.
#
# 2) Fiyat vs medyan çarpan adil fiyatı ("FAST Graphs" mantığı)
#    Hisse başı TTM metrik (faaliyet nakit akışı, serbest nakit akışı,
#    satış ya da HBK) × hissenin kendi geçmişteki MEDYAN çarpanı = o
#    tarihteki "normal" fiyat. Analist tahminleriyle 2 mali yıl ileri uzatılır.
#
# Veri: yfinance. Yıllık tablolar ~4–5 yıl, çeyreklik ~5–6 çeyrek geri gider;
# düzeltilmiş HBK (earnings_dates) ~10 yıl. Seriler bu ikisinin birleşimidir.

from typing import Any

import numpy as np
import pandas as pd

METRICS: dict[str, dict[str, Any]] = {
    "OCF": dict(ad="Faaliyet nakit akışı", kisa="P/OCF",
                stmt="cf", row=["Operating Cash Flow",
                                "Cash Flow From Continuing Operating Activities"]),
    "FCF": dict(ad="Serbest nakit akışı", kisa="P/FCF",
                stmt="cf", row=["Free Cash Flow"]),
    "REV": dict(ad="Satış", kisa="F/S", stmt="inc",
                row=["Total Revenue", "Operating Revenue"]),
    "EPS": dict(ad="Hisse başı kâr (GAAP)", kisa="F/K", stmt="inc",
                row=["Diluted EPS", "Basic EPS"], per_share=True),
}
SHARE_ROWS = ["Diluted Average Shares", "Basic Average Shares"]


# --------------------------------------------------------------------------
# Ham veri
# --------------------------------------------------------------------------
def fetch_raw(t: str) -> dict[str, Any]:
    """Tek hisse için gereken tüm ham veriyi çeker (hatalar sessizce boş)."""
    import yfinance as yf

    tk = yf.Ticker(t)
    out: dict[str, Any] = {"ticker": t, "errors": []}

    def grab(name, fn):
        try:
            v = fn()
            out[name] = v if v is not None else pd.DataFrame()
        except Exception as exc:
            out[name] = pd.DataFrame()
            out["errors"].append(f"{name}: {type(exc).__name__}")

    grab("price", lambda: tk.history(period="10y", auto_adjust=False)["Close"])
    grab("q_inc", lambda: tk.quarterly_income_stmt)
    grab("a_inc", lambda: tk.income_stmt)
    grab("q_cf", lambda: tk.quarterly_cashflow)
    grab("a_cf", lambda: tk.cashflow)
    grab("edates", lambda: tk.get_earnings_dates(limit=44))
    grab("eps_est", lambda: tk.earnings_estimate)
    grab("rev_est", lambda: tk.revenue_estimate)
    try:
        out["info"] = yf_info(t)
    except Exception:
        out["info"] = {}
    return out


def fetch_price_eps_only(t: str) -> dict[str, Any]:
    """Akranlar için hafif çekim: fiyat + düzeltilmiş HBK geçmişi."""
    import yfinance as yf

    tk = yf.Ticker(t)
    out: dict[str, Any] = {"ticker": t}
    try:
        out["price"] = tk.history(period="10y", auto_adjust=False)["Close"]
    except Exception:
        out["price"] = pd.Series(dtype=float)
    try:
        out["edates"] = tk.get_earnings_dates(limit=44)
    except Exception:
        out["edates"] = pd.DataFrame()
    try:
        out["q_inc"] = tk.quarterly_income_stmt
        out["a_inc"] = tk.income_stmt
    except Exception:
        out["q_inc"], out["a_inc"] = pd.DataFrame(), pd.DataFrame()
    return out


# --------------------------------------------------------------------------
# Yardımcılar
# --------------------------------------------------------------------------
def _naive(idx) -> pd.DatetimeIndex:
    idx = pd.to_datetime(idx)
    try:
        idx = idx.tz_localize(None)
    except TypeError:
        idx = idx.tz_convert(None)
    return idx.normalize()


def _price(raw: dict[str, Any]) -> pd.Series:
    p = raw.get("price")
    if p is None or len(p) == 0:
        return pd.Series(dtype=float)
    p = pd.Series(p).dropna().astype(float)
    p.index = _naive(p.index)
    return p[~p.index.duplicated(keep="last")].sort_index()


def _row(df: pd.DataFrame, names: list[str]) -> pd.Series:
    """Mali tablodan bir satır (sütunlar tarih). Döner: tarih → değer."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.Series(dtype=float)
    for n in names:
        if n in df.index:
            s = pd.to_numeric(df.loc[n], errors="coerce")
            s.index = _naive(s.index)
            return s.dropna().sort_index()
    return pd.Series(dtype=float)


def ttm_events(q: pd.Series, a: pd.Series, flow: bool = True) -> pd.Series:
    """
    Tarih → TTM değer olayları.
    Çeyreklik veri: ardışık 4 çeyrek varsa toplamı (akış kalemleri).
    Yıllık veri: mali yıl sonundaki yıllık değer.
    Çakışan dönemlerde çeyreklik tercih edilir (daha güncel).
    """
    ev: dict[pd.Timestamp, float] = {}
    for d, v in a.items():
        ev[d] = float(v)
    if len(q) >= 4 and flow:
        qs = q.sort_index()
        for i in range(3, len(qs)):
            win = qs.iloc[i - 3:i + 1]
            span = (win.index[-1] - win.index[0]).days
            if 250 <= span <= 300:             # gerçekten ardışık 4 çeyrek
                ev[win.index[-1]] = float(win.sum())
    return pd.Series(ev).sort_index()


def shares_series(raw: dict[str, Any]) -> pd.Series:
    a = _row(raw.get("a_inc"), SHARE_ROWS)
    q = _row(raw.get("q_inc"), SHARE_ROWS)
    s = pd.concat([a, q]).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if s.empty:
        so = (raw.get("info") or {}).get("sharesOutstanding")
        if so:
            s = pd.Series({pd.Timestamp.today().normalize(): float(so)})
    return s


def per_share_ttm(raw: dict[str, Any], key: str) -> pd.Series:
    """Hisse başı TTM metrik olayları (tarih → değer)."""
    m = METRICS[key]
    src_q = raw.get("q_cf" if m["stmt"] == "cf" else "q_inc")
    src_a = raw.get("a_cf" if m["stmt"] == "cf" else "a_inc")
    q, a = _row(src_q, m["row"]), _row(src_a, m["row"])
    ev = ttm_events(q, a, flow=True)
    if ev.empty:
        return ev
    if m.get("per_share"):
        return ev
    sh = shares_series(raw)
    if sh.empty:
        return pd.Series(dtype=float)
    sh_at = sh.reindex(sh.index.union(ev.index)).sort_index().ffill().bfill()
    return (ev / sh_at.reindex(ev.index)).dropna()


def adjusted_eps_ttm(raw: dict[str, Any]) -> pd.Series:
    """earnings_dates'teki 'Reported EPS' (düzeltilmiş) — 4'lü kayan toplam,
    açıklama tarihinde bilinir hale gelir."""
    ed = raw.get("edates")
    if ed is None or not isinstance(ed, pd.DataFrame) or ed.empty:
        return pd.Series(dtype=float)
    col = next((c for c in ed.columns if "Reported" in str(c)), None)
    if col is None:
        return pd.Series(dtype=float)
    s = pd.to_numeric(ed[col], errors="coerce").dropna()
    s.index = _naive(s.index)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.rolling(4).sum().dropna()


def step_daily(events: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """Olay serisini günlük fiyat endeksine 'en son bilinen değer' olarak yayar."""
    if events.empty:
        return pd.Series(np.nan, index=idx)
    u = idx.union(events.index)
    return events.reindex(u).sort_index().ffill().reindex(idx)


# --------------------------------------------------------------------------
# 1) F/K geçmişi + akran medyanı
# --------------------------------------------------------------------------
def pe_history(raw: dict[str, Any], basis: str = "GAAP") -> pd.Series:
    p = _price(raw)
    if p.empty:
        return pd.Series(dtype=float)
    if basis == "GAAP":
        eps = per_share_ttm(raw, "EPS")
        if eps.empty:
            eps = adjusted_eps_ttm(raw)
    else:
        eps = adjusted_eps_ttm(raw)
        if eps.empty:
            eps = per_share_ttm(raw, "EPS")
    if eps.empty:
        return pd.Series(dtype=float)
    e = step_daily(eps, p.index)
    pe = (p / e).where(e > 0)
    return pe[pe.index >= eps.index.min()].dropna()


def peer_median(pe_map: dict[str, pd.Series], freq: str = "ME",
                min_n: int = 3) -> tuple[pd.Series, pd.Series]:
    """Akranların F/K'larının dönem sonu medyanı ve kaç akranın katıldığı."""
    cols = {}
    for t, s in pe_map.items():
        if s is None or s.empty:
            continue
        s = s[(s > 0) & (s < 400)]                 # anlamsız uçları dışarıda
        cols[t] = s.resample(freq).last()
    if not cols:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    df = pd.DataFrame(cols)
    n = df.notna().sum(axis=1)
    med = df.median(axis=1).where(n >= min_n)
    return med.dropna(), n


# --------------------------------------------------------------------------
# 2) Fiyat vs medyan çarpan adil fiyatı
# --------------------------------------------------------------------------
def _growth(est: pd.DataFrame, row: str) -> float:
    if est is None or not isinstance(est, pd.DataFrame) or est.empty:
        return np.nan
    if row in est.index and "growth" in est.columns:
        g = pd.to_numeric(est.loc[row, "growth"], errors="coerce")
        return float(g) if np.isfinite(g) else np.nan
    return np.nan


def estimate_points(raw: dict[str, Any], key: str, last_ev: pd.Series
                    ) -> tuple[pd.Series, str]:
    """
    Gelecek 2 mali yıl için metrik tahmini. Satış → satış tahmini büyümesi;
    HBK → HBK tahmini; nakit akışları → HBK büyümesiyle ölçeklenir
    (yfinance nakit akışı tahmini vermez). Döner: (tarih → değer, açıklama).
    """
    a = _row(raw.get("a_cf" if METRICS[key]["stmt"] == "cf" else "a_inc"),
             METRICS[key]["row"])
    if a.empty or last_ev.empty:
        return pd.Series(dtype=float), ""
    fy_end = a.index.max()
    est = raw.get("rev_est") if key == "REV" else raw.get("eps_est")
    g0, g1 = _growth(est, "0y"), _growth(est, "+1y")
    if not np.isfinite(g0):
        return pd.Series(dtype=float), ""
    base = float(last_ev[last_ev.index <= fy_end].iloc[-1]) if (
        last_ev.index <= fy_end).any() else float(last_ev.iloc[-1])
    if base <= 0:
        return pd.Series(dtype=float), ""
    pts = {fy_end + pd.DateOffset(years=1): base * (1 + g0)}
    if np.isfinite(g1):
        pts[fy_end + pd.DateOffset(years=2)] = base * (1 + g0) * (1 + g1)
    kaynak = {"REV": "analist satış tahmini",
              "EPS": "analist HBK tahmini"}.get(
        key, "analist HBK büyümesiyle ölçeklendi (nakit akışı tahmini yok)")
    return pd.Series(pts).sort_index(), kaynak


def fair_value_chart_data(raw: dict[str, Any], key: str, years: int = 0
                          ) -> dict[str, Any]:
    """
    Döner: price, fair (günlük adil fiyat), points (dönem sonu noktaları),
    est (tahmin noktaları), median multiple, korelasyon, getiri, CAGR.
    `years`: medyan çarpanın hesaplandığı pencere (0 = mevcut tüm geçmiş).
    """
    p = _price(raw)
    ev = per_share_ttm(raw, key)
    if p.empty or ev.empty:
        return {"ok": False, "neden": f"{METRICS[key]['ad']} verisi bulunamadı"}
    ev = ev[ev > 0]
    if len(ev) < 2:
        return {"ok": False, "neden": f"{METRICS[key]['ad']} negatif ya da "
                                      f"yetersiz — çarpan anlamlı değil"}
    p = p[p.index >= ev.index.min()]
    daily = step_daily(ev, p.index)
    mult = (p / daily).dropna()
    if years:
        mult = mult[mult.index >= mult.index.max() - pd.DateOffset(years=years)]
    med = float(mult.median())
    fair = daily * med
    est, kaynak = estimate_points(raw, key, ev)
    corr = float(p.corr(fair)) if len(p) > 30 else np.nan
    total = (p.iloc[-1] / p.iloc[0] - 1) * 100
    yrs = (p.index[-1] - p.index[0]).days / 365.25
    cagr = ((p.iloc[-1] / p.iloc[0]) ** (1 / yrs) - 1) * 100 if yrs > 0.5 else np.nan
    fair_now = float(fair.iloc[-1])
    return {"ok": True, "price": p, "fair": fair, "points": ev * med,
            "est": est * med, "est_src": kaynak, "median": med,
            "mult_now": float(mult.iloc[-1]), "fair_now": fair_now,
            "gap": (p.iloc[-1] / fair_now - 1) * 100 if fair_now else np.nan,
            "corr": corr, "total": total, "cagr": cagr,
            "start": p.index[0], "metric": METRICS[key]}


def multiples_summary(raw: dict[str, Any], peers_pe: pd.Series | None = None
                      ) -> pd.DataFrame:
    """Güncel çarpan, kendi medyanı ve farkı — her metrik için."""
    p = _price(raw)
    rows = []
    for k, m in METRICS.items():
        ev = per_share_ttm(raw, k)
        ev = ev[ev > 0] if not ev.empty else ev
        if p.empty or len(ev) < 2:
            continue
        pp = p[p.index >= ev.index.min()]
        mult = (pp / step_daily(ev, pp.index)).dropna()
        if mult.empty:
            continue
        now, med = float(mult.iloc[-1]), float(mult.median())
        rows.append({"Çarpan": m["kisa"], "Şu an": now, "Kendi medyanı": med,
                     "Medyana göre %": (now / med - 1) * 100,
                     "En düşük": float(mult.min()), "En yüksek": float(mult.max()),
                     "Geçmiş": f"{mult.index[0]:%m.%Y} →"})
    out = pd.DataFrame(rows)
    if peers_pe is not None and not peers_pe.empty and not out.empty:
        out.loc[out["Çarpan"] == "F/K", "Akran medyanı"] = float(peers_pe.iloc[-1])
    return out


def suggest_peers(t: str, n: int = 8) -> list[str]:
    """Aynı ETF'lerdeki en ağır bileşenler — varsayılan akran listesi."""
    out: list[str] = []
    for e in uni.etfs_containing(t):
        for h in uni.holdings(e):
            if h != t and h not in out and "." not in h:
                out.append(h)
        if len(out) >= n * 2:
            break
    return out[:n]

# ==========================================================================
# KAYNAK: app.py
# ==========================================================================


# --- Modül kısayolları -------------------------------------------------------
# Modüler sürümde `an` = analytics, `px` = prices modülüydü. Tek dosyada hepsi
# aynı isim alanında olduğundan ikisini de bu dosyanın global alanına bağlıyoruz.
# (sys.modules kullanılmıyor: Streamlit betiği kendi isim alanında çalıştırır.)
class _Namespace:
    def __getattr__(self, name):
        try:
            return globals()[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


dta = eng = hld = mac = nws = pb = rep = scr = thm = uni = fvm = svm = fnl = fch = _Namespace()



import datetime as dt
import json
import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


logging.basicConfig(level=logging.INFO)

# --------------------------------------------------------------------------
# KOYU TEMA — her koşulda
# Uygulama koyu zemine göre tasarlandı. Streamlit temayı yalnızca
# `.streamlit/config.toml` dosyasından okur; dosya yoksa ya da yanlış yerdeyse
# açık temaya düşer ve Streamlit'in kendi yazıları (giriş kutuları, menüler,
# tablolar, grafik açıklamaları) siyah kalır → siyah zeminde görünmez.
# 1) Tema seçenekleri burada da koyuya sabitlenir.
# 2) Grafikler Streamlit temasından bağımsız, koyu Plotly şablonuyla çizilir.
# 3) CSS (inject_theme) widget yazılarını açık renge zorlar.
# --------------------------------------------------------------------------
_theme_changed = False
try:
    from streamlit import config as _st_config
    for _k, _v in {"theme.base": "dark", "theme.primaryColor": "#00e5ff",
                   "theme.backgroundColor": "#050506",
                   "theme.secondaryBackgroundColor": "#0d0d11",
                   "theme.textColor": "#ececf1"}.items():
        if _st_config.get_option(_k) != _v:
            _st_config.set_option(_k, _v)
            _theme_changed = True
except Exception as _exc:                       # pragma: no cover
    logging.info("Tema seçeneği ayarlanamadı: %s", _exc)
# Tema, sayfa çalışmaya başlarken tarayıcıya gönderilir; bu çalıştırmada
# yeni ayarlandıysa bir kez yeniden çalıştırınca tablolar da koyu açılır.
if _theme_changed and not st.session_state.get("_theme_rerun"):
    st.session_state["_theme_rerun"] = True
    st.rerun()

import plotly.io as pio

_apex_tpl = go.layout.Template(pio.templates["plotly_dark"])
_apex_tpl.layout.paper_bgcolor = "rgba(0,0,0,0)"
_apex_tpl.layout.plot_bgcolor = "rgba(0,0,0,0)"
_apex_tpl.layout.font = dict(color="#c2c2cc")
_apex_tpl.layout.legend = dict(font=dict(color="#c2c2cc"))
for _ax in ("xaxis", "yaxis"):
    _apex_tpl.layout[_ax].gridcolor = "#1f1f28"
    _apex_tpl.layout[_ax].zerolinecolor = "#2b2b36"
    _apex_tpl.layout[_ax].linecolor = "#2b2b36"
pio.templates["apex"] = _apex_tpl
pio.templates.default = "apex"

_st_plotly_chart = st.plotly_chart


def _plotly_chart(fig, *args, **kwargs):
    """Streamlit'in açık temasının grafik renklerini ezmesini engeller."""
    kwargs.setdefault("theme", None)
    return _st_plotly_chart(fig, *args, **kwargs)


st.plotly_chart = _plotly_chart

st.set_page_config(layout="wide", page_title="AETHER APEX", page_icon="🏛️",
                   initial_sidebar_state="collapsed")
inject_theme()

TTL_FAST = 300      # 5 dk  — sinyaller
TTL_SLOW = 900      # 15 dk — makro, tema
TTL_NEWS = 300


# ==========================================================================
# ÖNBELLEKLİ VERİ KATMANI
# ==========================================================================
@st.cache_resource
def get_store() -> Storage:
    return storage_from_secrets(getattr(st, "secrets", None),
                                local_path="apex_watchlist.json")


@st.cache_data(ttl=TTL_SLOW, show_spinner=False)
def load_macro_frames(nonce: str) -> dict[str, pd.DataFrame]:
    """Makro serileri bir kez çeker; hem güncel durum hem geçmiş bunu kullanır."""
    prices, failed = dta.fetch(mac.MACRO_TICKERS.values(), "1d")
    out = {k: prices.get(v, pd.DataFrame()) for k, v in mac.MACRO_TICKERS.items()}
    out["_failed"] = pd.DataFrame({"sembol": failed}) if failed else pd.DataFrame()
    return out


@st.cache_data(ttl=TTL_SLOW, show_spinner=False)
def load_macro(nonce: str) -> mac.MacroState:
    frames = load_macro_frames(nonce)
    failed = frames.get("_failed")
    by_key = {k: v for k, v in frames.items() if k != "_failed"}
    state = mac.build_macro_state(by_key)
    if failed is not None and not failed.empty:
        state.errors.append("Çekilemeyen makro sembol: "
                            + ", ".join(failed["sembol"]))
    return state


@st.cache_data(ttl=TTL_SLOW, show_spinner=False)
def macro_history(nonce: str, days: int = 60):
    """Batarya seyri + dönemsel değişim tabloları (geçmiş yeniden hesaplanır)."""
    frames = {k: v for k, v in load_macro_frames(nonce).items() if k != "_failed"}
    state = mac.build_macro_state(frames)
    changes, past = mac.battery_changes(frames, state)
    scores = mac.score_changes(state, past)
    hist = mac.battery_history(frames, days)
    return changes, scores, hist, mac.regime_shifts(hist)


@st.cache_data(ttl=TTL_FAST, show_spinner=False)
def scan(tickers: tuple[str, ...], interval: str, nonce: str) -> pd.DataFrame:
    """Verilen sembolleri tarar ve sinyal tablosunu döner."""
    tickers = tuple(sorted(set(tickers)))
    if not tickers:
        return pd.DataFrame()

    need = list(tickers) + [eng.BENCHMARK]
    prices, failed = dta.fetch(need, interval)
    bench = prices.get(eng.BENCHMARK)
    bench_close = bench["Close"] if bench is not None and not bench.empty else None

    weekly_map: dict[str, bool] = {}
    if interval == "1d":
        wk, _ = dta.fetch(tickers, "1wk")
        for t, df in wk.items():
            if len(df) > 12:
                ema12 = df["Close"].ewm(span=12, adjust=False).mean()
                weekly_map[t] = bool(df["Close"].iloc[-1] > ema12.iloc[-1])

    rows: list[dict] = []
    for t in tickers:
        df = prices.get(t)
        if df is None or df.empty:
            rows.append({"Sembol": t, "Sinyal": "⚫ VERİ YOK",
                         "Hata": "Fiyat verisi çekilemedi"})
            continue
        row = eng.analyze(df, t, bench_close=bench_close,
                          weekly_bull=weekly_map.get(t))
        if not row.ok:
            rows.append({"Sembol": t, "Sinyal": "⚫ VERİ YOK", "Hata": row.error})
            continue
        rec = {"Sembol": t, **row.data, "Hata": ""}
        rows.append(rec)

    out = pd.DataFrame(rows)
    if "MAGNITUDE" in out.columns:
        out = out.sort_values("MAGNITUDE", ascending=False)
    return out


@st.cache_data(ttl=TTL_SLOW, show_spinner=False)
def theme_performance(nonce: str) -> pd.DataFrame:
    """Tema bazlı çok periyotlu performans + ivme değişimi."""
    etfs = sorted({t for lst in uni.THEME_TRACKER.values() for t in lst})
    prices, failed = dta.fetch(etfs, "1d")

    perf: dict[str, dict[str, float]] = {}
    year = dt.date.today().year
    for t, df in prices.items():
        c = df["Close"].dropna()
        if len(c) < 8:
            continue

        def chg(a: int, b: int = 0) -> float:
            if len(c) <= a:
                return np.nan
            end = c.iloc[-1 - b]
            start = c.iloc[-1 - a]
            return (end / start - 1) * 100 if start else np.nan

        ytd_df = c[c.index.year == year]
        ytd = ((c.iloc[-1] / ytd_df.iloc[0] - 1) * 100
               if len(ytd_df) > 1 else np.nan)
        prev_year = c[c.index.year == year - 1]
        ytd_prev = np.nan
        if len(prev_year) > 1:
            doy = dt.date.today().timetuple().tm_yday
            upto = prev_year[prev_year.index.dayofyear <= doy]
            if len(upto) > 1:
                ytd_prev = (upto.iloc[-1] / prev_year.iloc[0] - 1) * 100

        perf[t] = {
            "Bugün": chg(1), "Prev_Bugün": chg(2, 1),
            "1H": chg(5), "Prev_1H": chg(10, 5),
            "1A": chg(21), "Prev_1A": chg(42, 21),
            "3A": chg(63), "Prev_3A": chg(126, 63),
            "YBB": ytd, "Prev_YBB": ytd_prev,
        }

    pdf = pd.DataFrame.from_dict(perf, orient="index")
    if pdf.empty:
        return pdf

    rows = {}
    for tema, lst in uni.THEME_TRACKER.items():
        valid = [t for t in lst if t in pdf.index]
        if valid:
            rows[tema] = pdf.loc[valid].mean()
            rows[tema]["Semboller"] = ", ".join(valid)
    out = pd.DataFrame.from_dict(rows, orient="index")
    return out


@st.cache_data(ttl=TTL_NEWS, show_spinner=False)
def load_news(topics: tuple[str, ...], nonce: str):
    return nws.fetch_news(uni.all_stocks() + uni.all_etfs(), topics)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_earnings_cached(tickers: tuple[str, ...], nonce: str) -> pd.DataFrame:
    return dta.fetch_earnings_calendar(tickers)


def _earn_ok_ratio(df: pd.DataFrame) -> float:
    if df is None or df.empty or "Fiyat" not in df.columns:
        return 0.0
    return float(pd.to_numeric(df["Fiyat"], errors="coerce").notna().mean())


def load_earnings(tickers: tuple[str, ...], nonce: str) -> pd.DataFrame:
    """Yahoo geçici olarak engellediyse (satırların yarısından fazlası boş)
    sonuç önbellekte tutulmaz — bir sonraki çalıştırmada yeniden denenir."""
    df = _load_earnings_cached(tickers, nonce)
    if tickers and _earn_ok_ratio(df) < 0.5:
        _load_earnings_cached.clear()
    return df


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _load_short_volume_cached(nonce: str):
    return svm.fetch_short_volume(30)


def load_short_volume(nonce: str):
    """FINRA günlük short hacmi. Başarısız sonuç önbellekte tutulmaz —
    bir sonraki çalıştırmada yeniden denenir."""
    df, warns, diag = _load_short_volume_cached(nonce)
    if df.empty:
        _load_short_volume_cached.clear()
    return df, warns, diag


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def _load_short_interest_cached(tickers: tuple[str, ...], nonce: str) -> pd.DataFrame:
    return svm.fetch_short_interest(tickers)


def load_short_interest(tickers: tuple[str, ...], nonce: str) -> pd.DataFrame:
    """yfinance short interest (ayda iki kez güncellenen borsa verisi)."""
    df = _load_short_interest_cached(tickers, nonce)
    if df.empty or df.drop(columns=["Sembol", "SI Tarihi"],
                           errors="ignore").notna().to_numpy().mean() < 0.2:
        _load_short_interest_cached.clear()
    return df


@st.cache_data(ttl=TTL_SLOW, show_spinner=False)
def load_rotation(nonce: str):
    """Karar Hunisi 1–2. adım: rotasyon oranları (uzun geçmiş) + repo verileri."""
    prices, failed = dta.fetch(fnl.ROTATION_TICKERS, "1d_long")
    crypto = fnl.read_crypto_caps()
    liq = fnl.read_liquidity()
    series = fnl.build_series(prices, crypto, liq)
    rot = fnl.rotation_table(series)
    files = {
        "kripto": (f"{crypto.index.max():%d.%m.%Y}" if not crypto.empty else None),
        "likidite": (f"{liq.index.max():%d.%m.%Y}" if not liq.empty else None),
    }
    return rot, series, failed, files


@st.cache_data(ttl=TTL_SLOW, show_spinner=False)
def load_theme_rotation(nonce: str):
    """Karar Hunisi 3. adım: tema ve ETF'lerde göreli rotasyon (RRG)."""
    etfs = sorted({t for lst in uni.THEME_TRACKER.values() for t in lst})
    prices, failed = dta.fetch(etfs + ["SPY"], "1d")
    T, tails = fnl.theme_rotation(prices, uni.THEME_TRACKER)
    E = fnl.etf_rotation(prices, etfs)
    idx_map = {}
    for tema, members in uni.THEME_TRACKER.items():
        idx, _ = fnl.theme_index(prices, members)
        if idx is not None:
            idx_map[tema] = idx
    return T, tails, E, idx_map, failed


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_stock_raw(sym: str, nonce: str) -> dict:
    """Hisse Analizi: tek hissenin fiyat + mali tablo + tahmin verisi."""
    return fch.fetch_raw(sym)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_peer_pe(peers: tuple[str, ...], basis: str, nonce: str) -> dict:
    """Akranların F/K geçmişi (sırayla — Yahoo hız sınırı için)."""
    out = {}
    for t in peers:
        try:
            out[t] = fch.pe_history(fch.fetch_price_eps_only(t), basis)
        except Exception:
            out[t] = pd.Series(dtype=float)
    return out


def theme_holdings(tema: str) -> list[str]:
    """Temadaki ETF'lerin bileşenleri; bileşeni bilinmeyen sembol hissenin
    kendisidir (ör. IONQ, MARA)."""
    out: list[str] = []
    for t in uni.THEME_TRACKER.get(tema, []):
        h = uni.holdings(t)
        if h:
            out += [x for x in h if x not in out]
        elif t not in uni.ETF and t not in out:
            out.append(t)
    return [x for x in out if "." not in x]      # yabancı borsa sembolleri hariç


# ==========================================================================
# DURUM
# ==========================================================================
def _init_state() -> None:
    defaults = {
        "nonce_macro": "0", "nonce_scan": "0", "nonce_theme": "0",
        "nonce_news": "0", "nonce_earn": "0", "nonce_short": "0",
        "nonce_funnel": "0", "nonce_stock": "0",
        "manual_scenario": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

    if "watchlist" not in st.session_state:
        store = get_store()
        try:
            saved = store.load(default={}).data
        except StorageError as exc:
            saved = {}
            st.session_state["store_error"] = str(exc)
        if not isinstance(saved, dict):
            saved = {}
        st.session_state.watchlist = {
            "future_themes": saved.get("future_themes")
            or {k: dict(v) for k, v in uni.DEFAULT_FUTURE_THEMES.items()},
            "earnings": saved.get("earnings") or list(uni.DEFAULT_EARNINGS),
            # P/S adil değer çarpan tablosu (kaydedilmişse)
            **({"ps_multiples": saved["ps_multiples"]}
               if isinstance(saved.get("ps_multiples"), dict) else {}),
        }


def scan_gate(key: str, tickers: list[str], interval: str, label: str,
              auto: bool = False) -> pd.DataFrame:
    """
    Ağır taramaları butona bağlar.

    Sebep: sekme açılır açılmaz 400 sembol taramak arayüzü dakikalarca
    kilitliyordu. Artık kullanıcı isteyince çalışıyor, sonuç oturumda
    saklanıyor; 5 dakikalık önbellek zaten arka planda devrede.
    """
    state_key = f"scanres_{key}"
    n = len(tickers)
    c1, c2 = st.columns([1, 3])
    clicked = c1.button(f"🔍 {label} ({n} sembol)", key=f"btn_{key}",
                        width="stretch", type="primary")
    have = state_key in st.session_state

    if clicked:
        with st.spinner(f"{n} sembol {interval} taranıyor… "
                        f"(yaklaşık {max(5, int(n * 0.16))} sn)"):
            st.session_state[state_key] = scan(tuple(tickers), interval,
                                               st.session_state.nonce_scan)
        have = True
    elif auto and not have and n <= 60:
        with st.spinner(f"{n} sembol taranıyor…"):
            st.session_state[state_key] = scan(tuple(tickers), interval,
                                               st.session_state.nonce_scan)
        have = True

    if not have:
        c2.caption("Tarama başlatılmadı — yukarıdaki butona basın. Sonuç bu "
                   "oturumda saklanır, sekmeler arasında geçince kaybolmaz.")
        return pd.DataFrame()
    return st.session_state[state_key]


def bump(key: str) -> None:
    st.session_state[key] = str(dt.datetime.now().timestamp())


def save_watchlist() -> bool:
    store = get_store()
    try:
        store.save(st.session_state.watchlist, "APEX izleme listesi güncellendi")
        return True
    except StorageError as exc:
        st.error(f"Kaydedilemedi: {exc}")
        return False


_init_state()
store = get_store()

# ==========================================================================
# BAŞLIK
# ==========================================================================
head_l, head_r = st.columns([4, 1])
with head_l:
    st.markdown(
        "<div class='nx-brand'><h1>AETHER APEX</h1>"
        "<span class='tag'>Live</span></div>", unsafe_allow_html=True)
with head_r:
    if st.button("⚡ Tümünü Yenile", width="stretch", type="primary"):
        for k in ("nonce_macro", "nonce_scan", "nonce_theme", "nonce_news",
                  "nonce_funnel"):
            bump(k)
        st.rerun()

with st.spinner("Makro göstergeler çekiliyor…"):
    M = load_macro(st.session_state.nonce_macro)

meta = [f"Güncelleme <b>{M.asof}</b>",
        f"OPEX <b>{M.opex_date}</b> ({M.opex_days} gün)"]
if M.opex_quad:
    meta.append("<b>ÜÇLÜ CADI</b>")
if M.fomc_days is not None:
    meta.append(f"FOMC <b>{M.fomc_date}</b> ({M.fomc_days} gün)")
meta.append(f"Kayıt <b>{'GitHub' if store.backend == 'github' else 'yerel'}</b>")
st.markdown(f"<div class='nx-meta'>{'  ·  '.join(meta)}</div>",
            unsafe_allow_html=True)

if store.backend == "local":
    st.warning(
        "**Kalıcılık kapalı.** Future Themes ve bilanço listesi sadece geçici "
        "dosyaya yazılıyor; Streamlit Cloud uygulamayı uyuttuğunda silinir.",
        icon="⚠️")
    with st.expander("🔑 Kalıcılığı açmak için 3 adım (2 dakika)", expanded=False):
        st.markdown("""
**1) GitHub'da jeton üretin**

`github.com` → sağ üst profil → **Settings** → en altta **Developer settings**
→ **Personal access tokens** → **Fine-grained tokens** → **Generate new token**

- *Repository access*: **Only select repositories** → bu uygulamanın deposu
- *Permissions* → *Repository permissions* → **Contents: Read and write**
  (başka hiçbir izne gerek yok)
- *Expiration*: uzun bir süre seçin; süresi dolunca kayıt sessizce durmaz,
  uygulama hata gösterir.

Üretilen `github_pat_...` değerini kopyalayın — sayfadan çıkınca bir daha
gösterilmez.

**2) Streamlit Cloud'a yapıştırın**

Uygulama sayfası → sağ alt **Manage app** → **⋮** → **Settings** → **Secrets**
sekmesine aşağıdakini yapıştırıp **Save** deyin:
""")
        st.code('[github]\n'
                'token  = "github_pat_BURAYA_JETONUNUZ"\n'
                'repo   = "kullanici-adiniz/depo-adiniz"\n'
                'branch = "main"\n'
                'path   = "apex_watchlist.json"\n', language="toml")
        st.markdown("""
`repo` alanı **`kullanıcı/depo`** biçimindedir — tam URL değil.
`branch` deponuzun ana dalı (`main` ya da `master`).
`path` deponun içinde oluşturulacak dosyanın adı; elle oluşturmanıza gerek yok,
ilk kayıtta kendisi commit edilir.

**3) Kaydedin ve uygulamayı yeniden başlatın**

Secrets kaydedilince Streamlit uygulamayı otomatik yeniden başlatır. Bu uyarı
kutusu kaybolur ve üstteki bilgi satırında **Kayıt: GitHub** yazar.

> Yerel bilgisayarda çalıştırıyorsanız aynı içeriği proje klasöründe
> `.streamlit/secrets.toml` dosyasına yazın ve bu dosyayı `.gitignore`'a
> ekleyin — jeton asla depoya girmemeli.
""")
        if st.button("🔌 Bağlantıyı test et", key="store_test"):
            probe = storage_from_secrets(getattr(st, "secrets", None),
                                         local_path="apex_watchlist.json")
            if not probe.enabled:
                st.error("Secrets içinde `[github]` bölümü görünmüyor. "
                         "Kaydettiyseniz uygulamayı **Reboot** edin — Secrets "
                         "değişikliği yeniden başlatmadan okunmaz.")
            else:
                try:
                    res = probe.load(default={})
                    st.success(f"Bağlantı çalışıyor → {probe.describe()}. "
                               + (res.message or "Mevcut kayıt okundu."))
                except StorageError as exc:
                    st.error(
                        f"{exc}\n\nSık görülen sebepler: jetonun bu depoya "
                        "erişimi yok · Contents izni 'Read and write' değil · "
                        "`repo` alanı `kullanıcı/depo` biçiminde değil · "
                        "`branch` adı yanlış.")
for e in M.errors:
    st.warning(e, icon="⚠️")

MARKET_REGIME_OK = M.scores.get("trend", 50) >= 50

TABS = st.tabs([
    "🧭 Karar Hunisi", "🔬 Hisse Analizi", "🌐 Makro & Rejim", "🔥 Tema Takibi", "🦅 ETF Radarı", "⚖️ Çarpan Uçurumu",
    "🦈 Haftalık", "🚨 4H Omni Swing", "🚀 Future Themes", "📅 Bilanço",
    "📄 Rapor",
])
(tab_funnel, tab_stock, tab_macro, tab_theme, tab_etf, tab_val, tab_week, tab_omni,
 tab_future, tab_earn, tab_report) = TABS


# ==========================================================================
# 0) KARAR HUNİSİ
# ==========================================================================
with tab_funnel:
    st.markdown(
        "Bir yatırım kararını **dört soruya** böler. Her adım bir öncekinin "
        "cevabına dayanır: önce piyasa risk almayı ödüllendiriyor mu, sonra "
        "para hangi bölgeye akıyor, sonra o bölgede hangi tema **erken** "
        "ivmeleniyor, en sonda o temada hangi hisse öne çıkıyor.")
    h1, h2 = st.columns([1, 4])
    if h1.button("🔄 Huniyi yenile", key="fn_ref", width="stretch"):
        bump("nonce_funnel")
        st.rerun()

    with st.spinner("Rotasyon oranları çekiliyor (15 yıllık günlük veri)…"):
        ROT, SER, ROT_FAILED, FILES = load_rotation(st.session_state.nonce_funnel)
    eksik = [k for k, v in FILES.items() if not v]
    h2.caption(
        "Kripto endeksleri: " + (FILES["kripto"] or "❌ dosya yok") + " · "
        "Fed likiditesi: " + (FILES["likidite"] or "❌ dosya yok")
        + (" · Çekilemeyen: " + ", ".join(ROT_FAILED) if ROT_FAILED else ""))
    if eksik:
        st.info("Kripto endeksleri (TOTAL, TOTAL3, BTC.D, OTHERS/BTC) ve Fed "
                "net likiditesi repodaki `data/` klasöründen okunur. GitHub'da "
                "**Actions → FINRA short hacim güncelle → Run workflow** ile "
                "bir kez çalıştırın; sonra her iş günü kendiliğinden güncellenir.")

    # ------------------------------------------------------------------ 1
    section("Adım 1 · Risk açık mı?")
    PIL = fnl.risk_pillars(M.scores, ROT)
    V = fnl.risk_verdict(PIL, M.regime)
    v1, v2 = st.columns([1, 2])
    v1.markdown(kpi("Piyasa durumu", V["etiket"],
                    f"skor {V['skor']:.0f}/100 · {V['boyut']}"
                    if np.isfinite(V["skor"]) else V["boyut"], V["renk"]),
                unsafe_allow_html=True)
    v2.markdown(f"**Ne demek?** {V['ozet']}")
    v2.caption(f"{V['n_ok']} sütun olumlu, {V['n_bad']} sütun olumsuz. "
               f"Makro sekmesindeki rejim etiketi: {M.regime}")
    pc = st.columns(6)
    for col, p in zip(pc, PIL):
        tone = ("pos" if p["durum"] == "✅" else "neg" if p["durum"] == "❌" else "")
        col.markdown(kpi(p["ad"], f"{p['durum']} {p['skor']:.0f}"
                         if np.isfinite(p["skor"]) else "➖", p["ne"], tone),
                     unsafe_allow_html=True)
    if any(x in M.regime for x in ("OPEX", "FOMC")):
        st.info(f"📅 **Takvim uyarısı — {M.regime}:** {M.regime_desc}")
    with st.expander("Risk açık / risk kapalı ne demek? Sütunlar nasıl okunur?"):
        st.markdown(
            "**Risk açık (risk-on):** Yatırımcılar getiri peşinde; para "
            "tahvil, altın ve nakitten hisseye, küçük şirketlere, büyüme "
            "hisselerine ve kriptoya akar. Alım sinyallerinin isabeti "
            "yüksektir.\n\n"
            "**Risk kapalı (risk-off):** Yatırımcılar korunma peşinde; para "
            "güvenli limana (hazine, altın, dolar, defansif sektörler) kaçar. "
            "İyi görünen alım sinyalleri bile sık başarısız olur.\n\n"
            "Karar altı sütunun ağırlıklı ortalamasıdır (Trend %25, diğerleri "
            "%15). Her sütun 0–100: **60+ ✅**, **40–60 ⚠️**, **40 altı ❌**.")
        for p in PIL:
            st.markdown(f"- **{p['ad']}** — {p['ne']} *{p['neden']}*")

    # ------------------------------------------------------------------ 2
    section("Adım 2 · Para nereye akıyor?")
    if ROT.empty:
        st.warning("Rotasyon verisi çekilemedi. Birkaç dakika sonra yenileyin.")
    else:
        for line in fnl.money_flow_summary(ROT):
            st.markdown(f"- {line}")

        SEG = fnl.segment_verdicts(ROT)
        s1, s2 = st.columns([1, 2])
        seg_pick = s1.selectbox("Alacağınız hisse hangi bölgede?",
                                list(fnl.SEGMENTS), key="fn_seg")
        row = SEG[SEG["Segment"] == seg_pick]
        if not row.empty:
            r = row.iloc[0]
            tone = ("pos" if r["Karar"].startswith("🟢")
                    else "neg" if r["Karar"].startswith("🔴") else "")
            s1.markdown(kpi(seg_pick, r["Karar"], f"skor {r['Skor']:+.0f}", tone),
                        unsafe_allow_html=True)
            s2.markdown(f"**Kapsam:** {r['Kapsam']}")
            s2.markdown("**Dayanak:** " + r["Dayanak"])
            uyum = ("✅ Piyasa risk açık ve bu bölgeye iştah var — ortam uygun."
                    if V["etiket"].startswith("🟢") and r["Skor"] >= 5 else
                    "⚠️ Bölgeye iştah var ama genel piyasa temkinli — küçük "
                    "pozisyon, sadece en güçlü kurulumlar."
                    if r["Skor"] >= 5 else
                    "🌱 Bölgede erken dönüş sinyali var — izleme listesine "
                    "alın, teyit bekleyin."
                    if r["Erken Sinyal"] else
                    "❌ Para bu bölgeye akmıyor — doğru zaman değil. Adım 2'de "
                    "iştahlı bölgelere bakın.")
            s2.markdown(f"**Sonuç:** {uyum}")

        st.dataframe(
            SEG[["Segment", "Karar", "Skor", "Erken Sinyal", "Dayanak"]],
            width="stretch", hide_index=True,
            column_config={
                "Skor": st.column_config.ProgressColumn(
                    format="%+.0f", min_value=-100, max_value=100,
                    help="Bölgeye ait oranların ortalama akış skoru"),
                "Dayanak": st.column_config.TextColumn(width="large")})

        st.markdown("**Bütün akış göstergeleri**")
        gtabs = st.tabs(["Hisse içi rotasyon", "Varlık sınıfları", "Kripto",
                         "Likidite / para basımı"])
        gmap = ["Hisse içi", "Varlık sınıfları", "Kripto", "Likidite"]
        for gt, g in zip(gtabs, gmap):
            with gt:
                sub = ROT[ROT["Grup"] == g].copy()
                if sub.empty:
                    st.caption("Veri yok" + (" — data/ klasöründeki dosya "
                                             "henüz oluşmamış." if g in
                                             ("Kripto", "Likidite") else "."))
                    continue
                sub["Çeyreklik"] = sub.apply(
                    lambda x: ("📐 TAZE KIRILIM" if x["Taze Çeyreklik Kırılım"]
                               else "✅ direnç üstü" if x["Çeyreklik Kırılım"]
                               else f"{x['Dirence Uzaklık %']:+.1f}% uzakta"
                               if np.isfinite(x["Dirence Uzaklık %"]) else "—"),
                    axis=1)
                st.dataframe(
                    sub[["Gösterge", "Akış", "Risk Okuması", "Faz", "20G %",
                         "63G %", "Çeyreklik", "Not"]],
                    width="stretch", hide_index=True,
                    column_config={
                        "20G %": st.column_config.NumberColumn(format="%+.1f%%"),
                        "63G %": st.column_config.NumberColumn(format="%+.1f%%"),
                        "Çeyreklik": st.column_config.TextColumn(
                            help="Çeyrek kapanışı, önceki 8 çeyreğin en yüksek "
                                 "kapanışını geçti mi?"),
                        "Not": st.column_config.TextColumn(width="large")})
                if g == "Kripto":
                    st.caption("TOTAL, TOTAL3, OTHERS ve BTC.D, ilk ~150 coinin "
                               "bugünkü arzı × günlük fiyatla hesaplanan "
                               "yaklaşık değerlerdir; yön ve kırılımlar "
                               "TradingView ile uyumludur, seviye birkaç puan "
                               "sapabilir.")

        # ---- oran grafiği (QQQ/SPY çeyreklik gibi)
        st.markdown("**Oran grafiği ve kırılım**")
        k1, k2 = st.columns([2, 1])
        keys = [k for k in ROT["Anahtar"]]
        ck = k1.selectbox("Gösterge", keys,
                          index=keys.index("QQQ/SPY") if "QQQ/SPY" in keys else 0,
                          format_func=lambda k: fnl.PAIR[k]["ad"], key="fn_ratio")
        per = k2.radio("Zaman dilimi", ["Çeyreklik", "Aylık", "Haftalık"],
                       horizontal=True, key="fn_per")
        sser = SER.get(ck)
        if sser is not None:
            rule, look = {"Çeyreklik": ("QE", 8), "Aylık": ("ME", 12),
                          "Haftalık": ("W-FRI", 26)}[per]
            o = sser.resample(rule).agg(["first", "max", "min", "last"]).dropna()
            o.columns = ["Open", "High", "Low", "Close"]
            o = o.tail({"QE": 48, "ME": 120, "W-FRI": 156}[rule])
            bi = fnl.breakout_info(sser, rule, look)
            fig = go.Figure(go.Candlestick(
                x=o.index, open=o["Open"], high=o["High"], low=o["Low"],
                close=o["Close"], increasing_line_color="#2fbe86",
                decreasing_line_color="#e66767", name=fnl.PAIR[ck]["ad"]))
            if np.isfinite(bi["direnc"]):
                fig.add_hline(y=bi["direnc"], line_dash="dot",
                              line_color="#c98500",
                              annotation_text=f"önceki {look} dönemin zirvesi",
                              annotation_font_color="#c98500")
            fig.update_layout(height=380, xaxis_rangeslider_visible=False,
                              showlegend=False, **CHART_LAYOUT)
            st.plotly_chart(fig, width="stretch")
            if bi["kirilim"]:
                st.success(f"📐 {fnl.PAIR[ck]['ad']} {per.lower()} kapanışta "
                           f"önceki {look} dönemin zirvesinin "
                           f"%{bi['uzaklik']:.1f} üstünde"
                           + (" — **bu dönem kırdı (taze)**." if bi.get("taze")
                              else "."))
            elif np.isfinite(bi["uzaklik"]):
                st.caption(f"Dirence uzaklık: {bi['uzaklik']:+.1f}%")
            st.caption(fnl.PAIR[ck]["not_"])

    # ------------------------------------------------------------------ 3
    section("Adım 3 · Hangi tema erken ivmeleniyor?")
    st.markdown(
        "Her tema S&P 500 ile kıyaslanır. **Yatay eksen** temanın endeksten "
        "güçlü mü zayıf mı olduğunu, **dikey eksen** bu gücün artıp "
        "azaldığını gösterir. Akıllı paranın erken izi, **🌱 İyileşiyor** "
        "bölgesinden **🚀 Lider** bölgesine geçiştir: tema henüz endeksi "
        "geçmemiştir ama göreli gücü artmaktadır.")
    with st.spinner("Tema ETF'leri karşılaştırılıyor…"):
        T, TAILS, E, IDX, T_FAILED = load_theme_rotation(
            st.session_state.nonce_funnel)
    if T.empty:
        st.warning("Tema verisi çekilemedi.")
    else:
        top = T.head(3)
        tc = st.columns(3)
        for col, (_, r) in zip(tc, top.iterrows()):
            neden = [r["Bölge"]]
            if r["Taze Lider"]:
                neden.append("liderliğe yeni geçti")
            if r["Hacim Onayı"]:
                neden.append(f"hacim {r['Hacim Oranı']:.1f}×")
            if np.isfinite(r["1H %"]):
                neden.append(f"1H {r['1H %']:+.1f}%")
            col.markdown(kpi(r["Tema"], f"{r['Erken Skor']}/100",
                             " · ".join(neden), "pos"), unsafe_allow_html=True)

        flt = st.radio("Göster", ["🌱 Erken akış", "🚀 Liderler", "Tümü"],
                       horizontal=True, key="fn_tflt")
        view = T
        if flt.startswith("🌱"):
            view = T[(T["_q"] == "iyilesen") | (T["Taze Lider"])]
        elif flt.startswith("🚀"):
            view = T[T["_q"] == "lider"]
        st.dataframe(
            view[["Tema", "Bölge", "Erken Skor", "1H %", "1A %", "3A %",
                  "Hacim Onayı", "RS-Oran", "RS-Mom", "ETF'ler"]],
            width="stretch", hide_index=True,
            column_config={
                "Erken Skor": st.column_config.ProgressColumn(
                    format="%d", min_value=0, max_value=100,
                    help="İyileşen bölge +45, liderliğe taze geçiş +25, göreli "
                         "güç artışı +10, hacim onayı +15, haftalık artı +5"),
                "1H %": st.column_config.NumberColumn(format="%+.1f%%"),
                "1A %": st.column_config.NumberColumn(format="%+.1f%%"),
                "3A %": st.column_config.NumberColumn(format="%+.1f%%"),
                "Hacim Onayı": st.column_config.CheckboxColumn(
                    help="Son 10 günün dolar hacmi önceki 60 günün 1.15 katından "
                         "fazla — fiyat hareketine para eşlik ediyor"),
                "RS-Oran": st.column_config.NumberColumn(format="%.1f"),
                "RS-Mom": st.column_config.NumberColumn(format="%.1f")})

        # ---- RRG grafiği
        pick = st.multiselect("Grafikte göster", list(T["Tema"]),
                              default=list(T["Tema"].head(10)), key="fn_rrg")
        if pick:
            fig = go.Figure()
            for i, tema in enumerate(pick):
                tl = TAILS.get(tema)
                if tl is None or tl.empty:
                    continue
                colr = SERIES[i % len(SERIES)]
                fig.add_trace(go.Scatter(
                    x=tl["RS-Oran"], y=tl["RS-Mom"], mode="lines+markers",
                    line=dict(color=colr, width=1.5), marker=dict(size=4),
                    name=tema, showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(
                    x=[tl["RS-Oran"].iloc[-1]], y=[tl["RS-Mom"].iloc[-1]],
                    mode="markers+text", marker=dict(size=11, color=colr),
                    text=[tema], textposition="top center",
                    textfont=dict(size=11, color=colr), name=tema,
                    hovertemplate=f"{tema}<br>RS-Oran %{{x:.1f}}<br>"
                                  f"RS-Mom %{{y:.1f}}<extra></extra>"))
            fig.add_vline(x=100, line_color="#3a3a46")
            fig.add_hline(y=100, line_color="#3a3a46")
            for x_, y_, t_ in ((1, 1, "🚀 Lider"), (0, 1, "🌱 İyileşiyor"),
                               (0, 0, "🩸 Geride"), (1, 0, "🌤️ Yoruluyor")):
                fig.add_annotation(xref="paper", yref="paper", x=0.02 + 0.96 * x_,
                                   y=0.02 + 0.96 * y_, text=t_, showarrow=False,
                                   xanchor="right" if x_ else "left",
                                   font=dict(size=12, color="#6e6e7a"))
            fig.update_layout(height=520, xaxis_title="RS-Oran (güç)",
                              yaxis_title="RS-Momentum (güçteki değişim)",
                              **CHART_LAYOUT)
            st.plotly_chart(fig, width="stretch")
            st.caption("Çizgiler son 6 haftanın izi, büyük nokta bugün. Saat "
                       "yönünün tersine dönüş (Geride → İyileşiyor → Lider) "
                       "sağlıklı rotasyondur.")
        with st.expander("ETF bazında aynı tablo"):
            if not E.empty:
                st.dataframe(
                    E[["ETF", "Bölge", "Erken Skor", "1H %", "1A %", "3A %",
                       "Hacim Onayı", "RS-Oran", "RS-Mom"]],
                    width="stretch", hide_index=True,
                    column_config={
                        "Erken Skor": st.column_config.ProgressColumn(
                            format="%d", min_value=0, max_value=100),
                        "1H %": st.column_config.NumberColumn(format="%+.1f%%"),
                        "1A %": st.column_config.NumberColumn(format="%+.1f%%"),
                        "3A %": st.column_config.NumberColumn(format="%+.1f%%"),
                        "RS-Oran": st.column_config.NumberColumn(format="%.1f"),
                        "RS-Mom": st.column_config.NumberColumn(format="%.1f")})

    # ------------------------------------------------------------------ 4
    section("Adım 4 · Bu temada hangi hisse?")
    tema_list = list(T["Tema"]) if not T.empty else list(uni.THEME_TRACKER)
    q1, q2 = st.columns([2, 1])
    tema = q1.selectbox("Tema", tema_list, key="fn_tema",
                        help="Liste Adım 3'teki erken skor sırasıyla gelir")
    min_liq = q2.number_input("Min. günlük işlem hacmi ($M)", 0.0, 500.0,
                              10.0, 1.0, key="fn_liq")
    hold = theme_holdings(tema)
    if not hold:
        st.info("Bu temadaki ETF'lerin bileşen listesi tanımlı değil. ETF "
                "Radarı sekmesinde bileşeni olan bir tema seçin.")
    else:
        st.caption(f"{len(hold)} hisse · kaynak: "
                   + ", ".join(f"`{e}`" for e in uni.THEME_TRACKER.get(tema, [])))
        S = scan_gate(f"funnel_{tema}", hold, "1d", "Tema hisselerini tara")
        if not S.empty:
            with st.spinner("Bilanço tarihleri ve değerleme çekiliyor…"):
                EARN_T = load_earnings(tuple(hold), st.session_state.nonce_earn)
            if not EARN_T.empty:
                EARN_T = fvm.add_fair_values(EARN_T)
            SVRAW_T, _, _ = load_short_volume(st.session_state.nonce_short)
            pchg = dict(zip(S["Sembol"], pd.to_numeric(S.get("1 Hafta %"),
                                                        errors="coerce")))
            SVT_T = svm.short_table(SVRAW_T, hold, pchg)
            TB = fnl.build_stock_table(S, IDX.get(tema), None, EARN_T, SVT_T,
                                       min_liq, swing_score=scr.swing_score)
            if TB.empty:
                st.warning("Tarama sonucu boş.")
            else:
                counts = TB["Durum"].value_counts()
                st.markdown(" ".join(
                    badge(f"{c} · {counts[c]}",
                          "pos" if c.startswith(("🎯", "🌱")) else
                          "neg" if c.startswith(("⛔", "💧")) else "")
                    for c in fnl.CLASS_ORDER if c in counts), unsafe_allow_html=True)
                show = st.multiselect(
                    "Gösterilecek durumlar", [c for c in fnl.CLASS_ORDER
                                              if c in counts],
                    default=[c for c in fnl.CLASS_ORDER[:3] if c in counts]
                    or [c for c in fnl.CLASS_ORDER if c in counts],
                    key=f"fn_show_{tema}")
                vv = TB[TB["Durum"].isin(show)] if show else TB
                cols = [c for c in [
                    "Durum", "Sembol", "Huni Skoru", "Sinyal", "Fiyat",
                    "Temaya Göre 1A", "Temaya Göre 1H", "1 Ay %", "Hacim ($M)",
                    "Kalan Gün", "Bilanço", "SV% 5G", "ΔSV pp", "WHALE",
                    "ΔWHALE 5B", "P/S Durum", "Neden"] if c in vv.columns]
                st.dataframe(
                    vv[cols].style.map(signal_style, subset=["Sinyal"]),
                    width="stretch", hide_index=True,
                    column_config={
                        "Huni Skoru": st.column_config.ProgressColumn(
                            format="%d", min_value=0, max_value=100,
                            help="Swing skoru (Pine sinyalleri) %55 + temaya "
                                 "göre güç + whale yönü − short artışı"),
                        "Fiyat": st.column_config.NumberColumn(format="$%.2f"),
                        "Temaya Göre 1A": st.column_config.NumberColumn(
                            format="%+.1f%%",
                            help="Hissenin 1 aylık getirisi − tema endeksinin"),
                        "Temaya Göre 1H": st.column_config.NumberColumn(
                            format="%+.1f%%"),
                        "1 Ay %": st.column_config.NumberColumn(format="%+.1f%%"),
                        "Hacim ($M)": st.column_config.NumberColumn(
                            "Günlük Ort. İşlem Hacmi ($M)", format="%.1f",
                            help="20 günlük ortalama dolar hacmi"),
                        "Kalan Gün": st.column_config.NumberColumn(
                            "Bilançoya Gün", format="%d"),
                        "SV% 5G": st.column_config.NumberColumn(
                            "Short Hacim %", format="%.1f%%"),
                        "ΔSV pp": st.column_config.NumberColumn(
                            "Short Δ (puan)", format="%+.1f"),
                        "WHALE": st.column_config.ProgressColumn(
                            format="%.0f", min_value=0, max_value=100),
                        "ΔWHALE 5B": st.column_config.NumberColumn(format="%+.1f"),
                        "Neden": st.column_config.TextColumn(width="large")})
                with st.expander("Durumlar ne anlama geliyor?"):
                    for c in fnl.CLASS_ORDER:
                        st.markdown(f"- **{c}** — {fnl.STOCK_CLASSES[c]}")

                # ---- tek paragraf sonuç
                al = TB[TB["Durum"] == "🎯 Alım adayı"]["Sembol"].head(5).tolist()
                ger = TB[TB["Durum"] == "🌱 Geride kaldı, toparlanıyor"][
                    "Sembol"].head(5).tolist()
                tr = T[T["Tema"] == tema].iloc[0] if not T.empty and (
                    T["Tema"] == tema).any() else None
                parca = [f"**Piyasa:** {V['etiket']}"]
                if tr is not None:
                    parca.append(f"**Tema ({tema}):** {tr['Bölge']}, erken skor "
                                 f"{tr['Erken Skor']}/100")
                parca.append("**Alım adayları:** "
                             + (", ".join(al) if al else "yok"))
                parca.append("**Toparlanan geride kalanlar:** "
                             + (", ".join(ger) if ger else "yok"))
                st.success(" · ".join(parca))


# ==========================================================================
# HİSSE ANALİZİ — tek hisse değerleme grafikleri
# ==========================================================================
with tab_stock:
    st.markdown(
        "Bir hisse seçin; **F/K geçmişini akranlarıyla** ve **fiyatı, "
        "hissenin kendi medyan çarpanından türetilen adil fiyat çizgisiyle** "
        "karşılaştırır. Fiyat yeşil çizginin çok üstündeyse hisse kendi "
        "tarihine göre pahalı, altındaysa ucuzdur.")
    a1, a2, a3 = st.columns([1, 1, 1])
    sym = a1.text_input("Sembol", value=st.session_state.get("sa_sym", "AVGO"),
                        key="sa_sym_in").strip().upper()
    basis = a2.radio("HBK tabanı", ["GAAP", "Düzeltilmiş"], horizontal=True,
                     key="sa_basis",
                     help="GAAP: resmi mali tablolardaki seyreltilmiş HBK "
                          "(yaklaşık 5 yıl). Düzeltilmiş: şirketin açıkladığı "
                          "non-GAAP HBK (yaklaşık 10 yıl, genelde daha yüksek, "
                          "F/K daha düşük çıkar).")
    if a3.button("🔄 Veriyi yenile", key="sa_ref", width="stretch"):
        bump("nonce_stock")
        st.rerun()
    st.session_state["sa_sym"] = sym

    if sym:
        with st.spinner(f"{sym} mali tabloları çekiliyor…"):
            RAW = load_stock_raw(sym, st.session_state.nonce_stock)
        info = RAW.get("info") or {}
        px_ = fch._price(RAW)
        if px_.empty:
            st.error(f"{sym} için fiyat verisi alınamadı. Sembolü kontrol edin "
                     "ya da birkaç dakika sonra yenileyin (Yahoo sınırlaması).")
        else:
            st.markdown(
                f"**{info.get('longName') or info.get('shortName') or sym}** · "
                f"{info.get('sector', '')} / {info.get('industry', '')} · "
                f"son fiyat **\\${px_.iloc[-1]:.2f}**")

            # ------------------------------------------------ F/K + akranlar
            section("F/K oranı (TTM) ve akran medyanı")
            sug = fch.suggest_peers(sym)
            peers_txt = st.text_input(
                "Akranlar (virgülle; varsayılan: aynı ETF'lerdeki en ağır şirketler)",
                value=", ".join(sug), key=f"sa_peers_{sym}")
            peers = [x.strip().upper() for x in peers_txt.split(",")
                     if x.strip() and x.strip().upper() != sym][:12]
            PE = fch.pe_history(RAW, basis)
            with st.spinner("Akranların F/K geçmişi hesaplanıyor…"):
                PEER = load_peer_pe(tuple(peers), basis,
                                    st.session_state.nonce_stock) if peers else {}
            MED, NPEER = fch.peer_median(PEER)
            if PE.empty:
                st.info("F/K hesaplanamadı (HBK verisi yok ya da şirket zararda).")
            else:
                own_med = float(PE.median())
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=PE.index, y=PE, mode="lines", name=f"{sym} F/K (TTM)",
                    line=dict(color="#3987e5", width=1.8)))
                if not MED.empty:
                    m2 = MED[MED.index >= PE.index.min()]
                    fig.add_trace(go.Scatter(
                        x=m2.index, y=m2, mode="lines+markers",
                        name=f"Akran medyanı ({len(PEER)} şirket)",
                        line=dict(color="#c98500", width=2),
                        marker=dict(size=4)))
                fig.add_hline(y=own_med, line_dash="dot", line_color="#6e6e7a",
                              annotation_text=f"{sym} kendi medyanı {own_med:.1f}",
                              annotation_font_color="#a0a0ab")
                fig.update_layout(height=380, legend=dict(orientation="h", y=1.08),
                                  yaxis_title="F/K", **CHART_LAYOUT)
                st.plotly_chart(fig, width="stretch")
                now = float(PE.iloc[-1])
                parts = [f"Şu an F/K **{now:.1f}**, kendi medyanı {own_med:.1f} "
                         f"({(now / own_med - 1) * 100:+.0f}%)."]
                if not MED.empty:
                    pm = float(MED.iloc[-1])
                    parts.append(f"Akran medyanı **{pm:.1f}** — hisse akranlarına "
                                 f"göre {(now / pm - 1) * 100:+.0f}% "
                                 + ("primli." if now > pm else "iskontolu."))
                st.markdown(" ".join(parts))
                if PEER:
                    eksik = [t for t in peers if t not in PEER or PEER[t].empty]
                    if eksik:
                        st.caption("F/K hesaplanamayan akranlar (zarar ya da veri "
                                   "yok): " + ", ".join(eksik))

            # ------------------------------------------------ adil fiyat çizgisi
            section("Fiyat ve medyan çarpan adil fiyatı")
            b1, b2 = st.columns([2, 1])
            mkey = b1.radio("Metrik", list(fch.METRICS),
                            format_func=lambda k: f"{fch.METRICS[k]['ad']} "
                                                  f"({fch.METRICS[k]['kisa']})",
                            horizontal=True, key="sa_metric")
            win = b2.selectbox("Medyan penceresi", [0, 3, 5],
                               format_func=lambda y: "Tüm geçmiş" if y == 0
                               else f"Son {y} yıl", key="sa_win")
            D = fch.fair_value_chart_data(RAW, mkey, win)
            if not D["ok"]:
                st.info(D["neden"])
            else:
                kp = st.columns(5)
                kp[0].markdown(kpi("Fiyat", f"${D['price'].iloc[-1]:.2f}",
                                   f"toplam {D['total']:+.0f}% · yıllık "
                                   f"{D['cagr']:+.1f}%"), unsafe_allow_html=True)
                kp[1].markdown(kpi(f"Medyan {D['metric']['kisa']}",
                                   f"{D['median']:.2f}",
                                   f"şu an {D['mult_now']:.2f}"),
                               unsafe_allow_html=True)
                kp[2].markdown(kpi("Medyan çarpanla adil fiyat",
                                   f"${D['fair_now']:.2f}", "son bilinen TTM"),
                               unsafe_allow_html=True)
                kp[3].markdown(kpi("Adile göre", f"{D['gap']:+.1f}%",
                                   "fiyat adilin üstünde" if D["gap"] > 0
                                   else "fiyat adilin altında",
                                   "neg" if D["gap"] > 15 else
                                   "pos" if D["gap"] < -10 else ""),
                               unsafe_allow_html=True)
                kp[4].markdown(kpi("Korelasyon", f"%{D['corr'] * 100:.0f}"
                                   if np.isfinite(D["corr"]) else "—",
                                   "fiyat ↔ metrik çizgisi"),
                               unsafe_allow_html=True)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=D["price"].index, y=D["price"], mode="lines", name="Fiyat",
                    line=dict(color="#3987e5", width=1.6)))
                fig.add_trace(go.Scatter(
                    x=D["fair"].index, y=D["fair"], mode="lines",
                    name=f"Medyan {D['metric']['kisa']} = {D['median']:.2f} ile fiyat",
                    line=dict(color="#2fbe86", width=2, shape="hv")))
                pts = D["points"][D["points"].index >= D["start"]]
                fig.add_trace(go.Scatter(
                    x=pts.index, y=pts, mode="markers", showlegend=False,
                    marker=dict(color="#2fbe86", size=7, symbol="diamond"),
                    hovertemplate="%{x|%m.%Y}: $%{y:.2f}<extra>dönem sonu</extra>"))
                if not D["est"].empty:
                    last_d = D["fair"].index[-1]
                    ex = pd.concat([pd.Series({last_d: D["fair_now"]}), D["est"]])
                    fig.add_trace(go.Scatter(
                        x=ex.index, y=ex, mode="lines+markers",
                        name="Tahmin", line=dict(color="#2fbe86", width=2,
                                                 dash="dash"),
                        marker=dict(size=8, symbol="diamond")))
                    fig.add_vrect(x0=last_d, x1=D["est"].index[-1],
                                  fillcolor="#c9b800", opacity=0.10, line_width=0,
                                  annotation_text="Tahminler",
                                  annotation_position="bottom right")
                fig.update_layout(height=460, legend=dict(orientation="h", y=1.08),
                                  yaxis_title="$", **CHART_LAYOUT)
                st.plotly_chart(fig, width="stretch")
                if not D["est"].empty:
                    e_last = D["est"]
                    st.caption(
                        "Tahmin noktaları: " + " · ".join(
                            f"{d:%m.%Y} → \\${v:.2f}" for d, v in e_last.items())
                        + f" — kaynak: {D['est_src']}.")
                st.caption(
                    "Yeşil çizgi = hisse başı TTM " + D["metric"]["ad"].lower()
                    + f" × hissenin {('tüm geçmişteki' if not win else f'son {win} yıldaki')} "
                    f"medyan {D['metric']['kisa']} çarpanı. Korelasyon yüksekse "
                    "(%70+) fiyat bu metriği takip ediyor demektir; o zaman "
                    "çizgiden uzaklaşmalar anlamlıdır.")

            # ------------------------------------------------ özet tablo
            section("Çarpan özeti")
            SUM = fch.multiples_summary(RAW, MED if not MED.empty else None)
            if SUM.empty:
                st.caption("Çarpan hesaplanamadı.")
            else:
                st.dataframe(SUM, width="stretch", hide_index=True,
                             column_config={
                                 c: st.column_config.NumberColumn(format="%.2f")
                                 for c in ("Şu an", "Kendi medyanı", "En düşük",
                                           "En yüksek", "Akran medyanı")} | {
                                 "Medyana göre %": st.column_config.NumberColumn(
                                     format="%+.0f%%")})
            if RAW.get("errors"):
                st.caption("Eksik veri: " + ", ".join(RAW["errors"]))
            with st.expander("Hesap yöntemi ve sınırlar"):
                st.markdown(
                    "- **TTM**: son 4 çeyreğin toplamı. Çeyreklik veri yetmeyen "
                    "eski dönemlerde mali yıl sonu değeri kullanılır.\n"
                    "- **Geçmiş uzunluğu**: Yahoo yıllık tabloları ~4–5 yıl, "
                    "çeyrekliği ~5–6 çeyrek verir; grafikler bu aralıktan "
                    "başlar. Düzeltilmiş HBK ~10 yıl gider.\n"
                    "- **Akran medyanı**: sektörün tamamı değil, seçtiğiniz "
                    "akranların aylık F/K medyanıdır (en az 3 akran gerekir).\n"
                    "- **Tahminler**: satış ve HBK için analist tahminleri "
                    "kullanılır. Yahoo nakit akışı tahmini vermediği için OCF/FCF "
                    "tahmini HBK büyümesiyle ölçeklenir — kaba bir yaklaşımdır.\n"
                    "- Zarardaki dönemlerde F/K ve çarpanlar tanımsızdır, "
                    "grafikte boş kalır.")


# ==========================================================================
# 1) MAKRO & REJİM
# ==========================================================================
with tab_macro:
    c1, c2, c3, c4 = st.columns(4)
    risk_tone = "pos" if M.risk_score >= 60 else "neg" if M.risk_score <= 40 else ""
    c1.markdown(kpi("Tespit Edilen Rejim", M.regime,
                    f"bileşik risk skoru {M.risk_score:.0f}/100", risk_tone),
                unsafe_allow_html=True)
    vix = M.get("VIX")
    c2.markdown(kpi("VIX", f"{vix:.1f}" if np.isfinite(vix) else "—",
                    M.readings["VIX"].detail if "VIX" in M.readings else "",
                    "neg" if np.isfinite(vix) and vix > 25 else "pos"),
                unsafe_allow_html=True)
    c3.markdown(kpi("OPEX'e Kalan", f"{M.opex_days} gün",
                    ("Üçlü cadı — etki güçlü" if M.opex_quad
                     else f"{M.opex_date}"),
                    "neg" if M.opex_days <= 2 else ""),
                unsafe_allow_html=True)
    c4.markdown(kpi("Piyasa Rejim Kapısı",
                    "AÇIK" if MARKET_REGIME_OK else "KAPALI",
                    "SPY 50 EMA üstünde — long sinyaller geçerli"
                    if MARKET_REGIME_OK else
                    "SPY 50 EMA altında — long sinyal kalitesi düşer",
                    "pos" if MARKET_REGIME_OK else "neg"),
                unsafe_allow_html=True)

    st.info(f"**{M.regime}** — {M.regime_desc}")

    left, right = st.columns([3, 2])

    with left:
        section("Canlı makro göstergeler")
        if M.readings:
            rd = pd.DataFrame([{
                "Gösterge": r.label,
                "Değer": r.value,
                "Değişim %": r.change_pct,
                "Yorum": r.detail,
            } for r in M.readings.values()])
            st.dataframe(
                rd, width="stretch", hide_index=True,
                column_config={
                    "Değer": st.column_config.NumberColumn(format="%.2f"),
                    "Değişim %": st.column_config.NumberColumn(format="%+.2f%%"),
                    "Yorum": st.column_config.TextColumn(width="large"),
                })

        section("Sermaye akış eğilimi")
        st.caption("Elle yazılmış senaryo sabitleri değil — yukarıdaki canlı "
                   "göstergelerden hesaplanır. Seviyenin yanındaki fark, aynı "
                   "formülün 1 gün / 1 hafta / 1 ay önceki veriyle yeniden "
                   "çalıştırılmasıyla bulunur.")

        manual_on = bool(st.session_state.manual_scenario)
        bat = (mac.MANUAL_SCENARIOS[st.session_state.manual_scenario]["battery"]
               if manual_on else M.battery)

        with st.spinner("Geçmiş rejim yeniden hesaplanıyor…"):
            BC, SC, BH, SHIFTS = macro_history(st.session_state.nonce_macro)

        delta_period = st.radio(
            "Karşılaştırma dönemi", list(mac.PERIOD_BARS),
            horizontal=True, index=1, key="bat_period",
            help="Bataryanın bu dönem önceki değerine göre farkı gösterilir.")
        dcol = f"Δ {delta_period}"
        deltas = ({} if manual_on else
                  {r["Varlık Sınıfı"]: r.get(dcol) for _, r in BC.iterrows()})

        keys = list(bat)
        prev_vals = [bat[k] - (deltas.get(k) or 0) for k in keys]
        bar = go.Figure()
        # Önceki seviye soluk gölge olarak arkada durur
        if not manual_on:
            bar.add_trace(go.Bar(
                x=prev_vals, y=keys, orientation="h", name=f"{delta_period} önce",
                marker=dict(color="rgba(255,255,255,.10)", cornerradius=4),
                hovertemplate="%{y}: %{x:.0f}/100 (" + delta_period
                              + " önce)<extra></extra>"))
        bar.add_trace(go.Bar(
            x=[bat[k] for k in keys], y=keys, orientation="h", name="şimdi",
            marker=dict(color=[SERIES[i % len(SERIES)] for i in range(len(keys))],
                        cornerradius=4),
            text=[f"{bat[k]}" + ("" if manual_on or deltas.get(k) is None
                                 or not np.isfinite(deltas.get(k, np.nan))
                                 else f"   {deltas[k]:+.0f}")
                  for k in keys],
            textposition="outside",
            hovertemplate="%{y}: %{x}/100<extra></extra>"))
        bar.update_layout(height=300, bargap=0.30, barmode="overlay",
                          showlegend=not manual_on,
                          legend=dict(orientation="h", y=1.18, x=0),
                          xaxis=dict(range=[0, 118], showgrid=False,
                                     showticklabels=False),
                          yaxis=dict(showgrid=False), **CHART_LAYOUT)
        st.plotly_chart(bar, width="stretch")

        if manual_on:
            st.caption("Elle senaryo seçiliyken geçmiş karşılaştırma kapalıdır — "
                       "senaryo sabitlerinin tarihçesi yoktur.")
        else:
            st.dataframe(
                BC, width="stretch", hide_index=True,
                column_config={
                    "Şimdi": st.column_config.ProgressColumn(
                        format="%.0f", min_value=0, max_value=100),
                    "1 gün önce": st.column_config.NumberColumn(format="%.0f"),
                    "1 hafta önce": st.column_config.NumberColumn(format="%.0f"),
                    "1 ay önce": st.column_config.NumberColumn(format="%.0f"),
                    "Δ 1 gün": st.column_config.NumberColumn(format="%+.0f"),
                    "Δ 1 hafta": st.column_config.NumberColumn(format="%+.0f"),
                    "Δ 1 ay": st.column_config.NumberColumn(format="%+.0f"),
                    "Besleyen": st.column_config.TextColumn(width="large"),
                })

            if not BH.empty:
                line = go.Figure()
                for i, k in enumerate(keys):
                    if k in BH.columns:
                        line.add_trace(go.Scatter(
                            x=BH.index, y=BH[k], name=k, mode="lines",
                            line=dict(color=SERIES[i % len(SERIES)], width=2)))
                line.add_trace(go.Scatter(
                    x=BH.index, y=BH["Risk Skoru"], name="Bileşik risk",
                    mode="lines", line=dict(color="#f2f2f6", width=2.5,
                                            dash="dot")))
                line.add_hline(y=50, line=dict(color="#3a3a48", width=1,
                                               dash="dot"))
                line.update_layout(height=330, **CHART_LAYOUT,
                                   yaxis=dict(range=[0, 100],
                                              gridcolor="#1b1b22"),
                                   xaxis=dict(gridcolor="#1b1b22"),
                                   legend=dict(orientation="h", y=1.15, x=0))
                st.plotly_chart(line, width="stretch")
                st.caption(f"Son {len(BH)} işlem günü. Her gün, bugünkü formülle "
                           "yeniden hesaplanır — formül değişse bile geçmiş "
                           "tutarlı kalır. 50 çizgisi nötr seviyedir.")

            if SHIFTS:
                st.markdown("**Bu pencerede rejim kaç kez değişti**")
                sh = pd.DataFrame(SHIFTS[-6:])
                sh["Tarih"] = pd.to_datetime(sh["Tarih"]).dt.strftime("%d.%m.%Y")
                st.dataframe(sh, width="stretch", hide_index=True)
                st.caption("Rejimin sık değişmesi kararsız piyasa demektir; "
                           "trend takip sistemleri bu pencerelerde kötü çalışır.")

        section("Rejimi ne itiyor, ne çekiyor")
        st.caption("Alt skorların dönemsel değişimi. Bileşik risk skoru bunların "
                   "ortalamasıdır; hangi bileşenin rejimi taşıdığı buradan "
                   "okunur.")
        st.dataframe(
            SC, width="stretch", hide_index=True,
            column_config={
                "Şimdi": st.column_config.ProgressColumn(
                    format="%.0f", min_value=0, max_value=100),
                "Δ 1 gün": st.column_config.NumberColumn(format="%+.1f"),
                "Δ 1 hafta": st.column_config.NumberColumn(format="%+.1f"),
                "Δ 1 ay": st.column_config.NumberColumn(format="%+.1f"),
                "Ne ölçüyor": st.column_config.TextColumn(width="large"),
            })

        section("Senaryo karşılaştırma (elle)")
        cols = st.columns(len(mac.MANUAL_SCENARIOS) + 1)
        if cols[0].button("Ölçülen", width="stretch"):
            st.session_state.manual_scenario = None
            st.rerun()
        for i, name in enumerate(mac.MANUAL_SCENARIOS, start=1):
            if cols[i].button(name, width="stretch"):
                st.session_state.manual_scenario = name
                st.rerun()
        if st.session_state.manual_scenario:
            sc = mac.MANUAL_SCENARIOS[st.session_state.manual_scenario]
            st.caption(f"**{st.session_state.manual_scenario}** — {sc['desc']}")

    with right:
        section("Kavram sözlüğü")
        for title, body in mac.REGIME_GLOSSARY:
            with st.expander(title):
                st.write(body)

    # ---------------- REJİM OYUN KİTABI ----------------
    section("Bu rejimde ne çalışır, ne çalışmaz")
    aktif = pb.drivers_for(M.regime)
    st.caption("Aşağıdaki kartlar, tespit edilen rejimi besleyen sürücüleri ve "
               "her birinin tarihsel olarak hangi tarafı vurduğunu gösterir. "
               "Rejim haberden değil ölçülen göstergelerden belirlenir; haberler "
               "sadece 'neden' sorusunu cevaplar.")

    for d in aktif:
        with st.expander(f"{d.icon} {d.label}", expanded=True):
            st.markdown(d.nedir)
            st.caption(f"**Veriden nasıl anlaşılır:** {d.veri_isareti}")

            lc, rc = st.columns(2)
            if d.lehte_etf or d.lehte_hisse:
                with lc:
                    st.markdown("**🟢 Lehte çalışan**")
                    if d.lehte_etf:
                        st.markdown("ETF: " + " ".join(f"`{x}`" for x in d.lehte_etf))
                    if d.lehte_hisse:
                        st.markdown("Hisse: " + " ".join(f"`{x}`" for x in d.lehte_hisse))
            if d.aleyhte_etf or d.aleyhte_hisse:
                with rc:
                    st.markdown("**🔴 Aleyhte çalışan**")
                    if d.aleyhte_etf:
                        st.markdown("ETF: " + " ".join(f"`{x}`" for x in d.aleyhte_etf))
                    if d.aleyhte_hisse:
                        st.markdown("Hisse: " + " ".join(f"`{x}`" for x in d.aleyhte_hisse))
            if d.islem_notu:
                st.info(f"**İşlem notu:** {d.islem_notu}")

    with st.expander("Diğer rejim sürücüleri (referans)"):
        for key, d in pb.DRIVERS.items():
            if d in aktif:
                continue
            st.markdown(f"**{d.icon} {d.label}** — {d.nedir}")
            if d.lehte_hisse:
                st.caption("Lehte: " + ", ".join(d.lehte_hisse[:8])
                           + (" · Aleyhte: " + ", ".join(d.aleyhte_hisse[:8])
                              if d.aleyhte_hisse else ""))
            st.markdown("---")

    section("Canlı haber akışı")
    nc1, nc2 = st.columns([4, 1])
    topics = nc1.multiselect("Konular", list(nws.FEEDS), default=list(nws.FEEDS),
                             label_visibility="collapsed")
    if nc2.button("🔄 Haberleri yenile", width="stretch"):
        bump("nonce_news")
        st.rerun()
    with st.spinner("Haber akışları taranıyor…"):
        items, nerr = load_news(tuple(topics), st.session_state.nonce_news)
    for e in nerr:
        st.caption(f"⚠️ {e}")
    if items:
        # Her haberi rejim sürücüsüne ve etkilediği sembollere bağla
        enriched = []
        driver_counts: dict[str, int] = {}
        for it in items:
            keys = pb.match_drivers(it["Başlık"])
            for k in keys:
                driver_counts[k] = driver_counts.get(k, 0) + 1
            imp = pb.impacted(keys)
            enriched.append({
                **it,
                "Rejim Sürücüsü": " ".join(
                    f"{pb.DRIVERS[k].icon}{pb.DRIVERS[k].label.split(' /')[0]}"
                    for k in keys) or "—",
                "🟢 Lehte": ", ".join((imp["lehte_etf"] + imp["lehte_hisse"])[:6]),
                "🔴 Aleyhte": ", ".join((imp["aleyhte_etf"] + imp["aleyhte_hisse"])[:6]),
            })

        if driver_counts:
            st.markdown("**Haber akışında şu an baskın olan sürücüler**")
            dc = st.columns(min(4, len(driver_counts)))
            for col, (k, n) in zip(dc, sorted(driver_counts.items(),
                                              key=lambda x: -x[1])):
                d = pb.DRIVERS[k]
                col.markdown(kpi(f"{d.icon} {d.label.split(' /')[0]}",
                                 f"{n} haber",
                                 "bu rejimi besliyor" if d in aktif
                                 else "rejimle eşleşmiyor",
                                 "pos" if d in aktif else ""),
                             unsafe_allow_html=True)
            st.caption("Haber sayısı bir rejimi KANITLAMAZ; ölçülen göstergelerle "
                       "aynı yönü gösteriyorsa teyit, göstermiyorsa erken uyarı "
                       "olarak okuyun.")

        st.session_state["rep_news"] = pd.DataFrame(enriched)
        st.dataframe(
            pd.DataFrame(enriched)[["Konu", "Tarih", "Başlık", "Rejim Sürücüsü",
                                    "İlgili", "🟢 Lehte", "🔴 Aleyhte", "Link"]],
            width="stretch", hide_index=True,
            column_config={
                "Link": st.column_config.LinkColumn("Link", width="small"),
                "Başlık": st.column_config.TextColumn(width="large"),
                "Rejim Sürücüsü": st.column_config.TextColumn(width="medium"),
                "🟢 Lehte": st.column_config.TextColumn(width="medium"),
                "🔴 Aleyhte": st.column_config.TextColumn(width="medium"),
            })


# ==========================================================================
# 2) TEMA TAKİBİ
# ==========================================================================
with tab_theme:
    tc1, tc2 = st.columns([4, 1])
    period = tc1.radio("Periyot", ["Bugün", "1H", "1A", "3A", "YBB"],
                       horizontal=True, index=2, label_visibility="collapsed")
    if tc2.button("🔄 Yenile", width="stretch", key="theme_refresh"):
        bump("nonce_theme")
        st.rerun()

    with st.expander("📐 Getiri ve İvme ne demek? (bir kez okuyun)", expanded=False):
        st.markdown(f"""
Her tema için **iki ayrı sayı** var ve bunlar farklı şeyler söyler:

| | Tanım | Örnek (**{period}** seçiliyken) |
|---|---|---|
| **Getiri %** | Seçilen dönemin yüzde değişimi | {thm.PERIOD_LABELS.get(period, period)} içindeki % değişim |
| **İvme** | Bu dönemin getirisi **eksi** bir önceki eşdeğer dönemin getirisi | (bu dönem) − ({thm.PERIOD_PREV.get(period, "önceki dönem")}) |

İvme pozitifse tema **hızlanıyor**, negatifse **yavaşlıyor**.

Neden ikisi de gerekli: bir tema %18 kazandırmış olabilir, ama önceki dönem
%39 kazandırdıysa ivme **−21**'dir — para hâlâ giriyor, fakat giriş hızı
yarıya inmiş; liderlik el değiştirmek üzere olabilir. Tersine −%4 getirili
bir tema önceki dönem −%15 yaptıysa ivmesi **+11**'dir ve dipten dönüş tam
buradan başlar.

Bu iki eksen dört çeyrek üretir; asıl karar çeyrekten çıkar:
""")
        for q in thm.QUADRANTS.values():
            st.markdown(f"- {q.icon} **{q.label}** — {q.aciklama}  \n"
                        f"  _Ne yapmalı:_ {q.aksiyon}")

    with st.spinner("Tema performansı hesaplanıyor…"):
        T = theme_performance(st.session_state.nonce_theme)

    if T.empty:
        st.warning("Tema verisi çekilemedi.", icon="⚠️")
    else:
        TQ = thm.build_table(T, period)
        st.session_state["rep_theme"] = (TQ, period)
        ornek = thm.worked_example(TQ, period)
        if ornek:
            st.info(ornek)

        section("Momentum × İvme haritası")
        st.caption("Yatay eksen: dönem getirisi. Dikey eksen: ivme (önceki "
                   "eşdeğer döneme göre hızlanma). Sağ üst çeyrek avlanma "
                   "sahası, sol üst çeyrek dipten dönüş adayları. Kalabalık "
                   "olmasın diye sadece merkeze en uzak 16 tema etiketlenir; "
                   "diğerlerinin adı için noktanın üzerine gelin.")
        # Etiket kalabalığını önlemek için sadece merkeze en uzak temalar
        # yazılır; kalanlar noktayla kalır ve üzerine gelince okunur.
        _mesafe = (TQ["Getiri %"].fillna(0) ** 2 + TQ["İvme"].fillna(0) ** 2) ** 0.5
        _etiketli = set(_mesafe.nlargest(16).index)

        qfig = go.Figure()
        for key, q in thm.QUADRANTS.items():
            sel = TQ[TQ["_q"] == key]
            if sel.empty:
                continue
            qfig.add_trace(go.Scatter(
                x=sel["Getiri %"], y=sel["İvme"], mode="markers+text",
                name=f"{q.icon} {q.label}",
                text=[t if t in _etiketli else "" for t in sel.index],
                customdata=list(sel.index),
                textposition="top center", textfont=dict(size=11, color="#d5d5de"),
                marker=dict(size=13, color=q.renk, line=dict(color="#050506",
                                                             width=1.5)),
                hovertemplate=("<b>%{customdata}</b><br>Getiri %{x:.2f}%"
                               "<br>İvme %{y:+.2f}<extra></extra>")))
        qfig.add_hline(y=0, line=dict(color="#3a3a48", width=1))
        qfig.add_vline(x=0, line=dict(color="#3a3a48", width=1))
        qfig.update_layout(
            height=560, legend=dict(orientation="h", y=-0.14),
            xaxis=dict(title="Dönem getirisi %", gridcolor="#1b1b22",
                       zeroline=False),
            yaxis=dict(title="İvme (hızlanma)", gridcolor="#1b1b22",
                       zeroline=False),
            **CHART_LAYOUT)
        st.plotly_chart(qfig, width="stretch")

        özet = thm.summary(TQ)
        if özet:
            cols = st.columns(len(özet))
            for col, (key, info) in zip(cols, özet.items()):
                q = info["quadrant"]
                col.markdown(kpi(f"{q.icon} {q.label}", f"{info['n']} tema",
                                 f"ort. getiri %{info['ort_getiri']:+.1f} · "
                                 f"ort. ivme {info['ort_ivme']:+.1f}",
                                 "pos" if key == "lider_hizlanan"
                                 else "neg" if key == "hizlanan_dusus" else ""),
                            unsafe_allow_html=True)

        section("Sıralı performans")
        srt = T.sort_values(period, ascending=True)
        labels, texts, bar_renk = [], [], []
        for tema, row in srt.iterrows():
            syms = str(row.get("Semboller", ""))
            short = syms if len(syms) < 26 else syms[:24] + "…"
            labels.append(f"{tema}  <span style='font-size:11px;color:#6e6e7a'>"
                          f"{short}</span>")
            val = row[period]
            prev = row.get(f"Prev_{period}", np.nan)
            delta = val - prev if np.isfinite(prev) else np.nan
            dtxt = ("" if not np.isfinite(delta)
                    else f"  (🔺+{delta:.1f})" if delta > 0
                    else f"  (🔻{delta:.1f})")
            texts.append(f"{val:+.2f}%{dtxt}")
            bar_renk.append("#3987e5" if val >= 0 else "#d55181")

        fig = go.Figure(go.Bar(
            y=labels, x=srt[period], orientation="h",
            marker=dict(color=bar_renk, cornerradius=3),
            text=texts, textposition="outside",
            textfont=dict(size=12), hovertemplate="%{x:.2f}%<extra></extra>"))
        rng = srt[period].max() - srt[period].min()
        pad = rng * 0.42 if rng else 5
        fig.update_layout(
            height=max(700, 22 * len(srt)),
            xaxis=dict(range=[srt[period].min() - pad, srt[period].max() + pad],
                       showgrid=False, zeroline=True, zerolinecolor="#2b2b36",
                       showticklabels=False),
            yaxis=dict(showgrid=False, tickfont=dict(size=12)),
            **CHART_LAYOUT)
        st.plotly_chart(fig, width="stretch", config={
            "toImageButtonOptions": {"format": "svg",
                                     "filename": f"TemaTakibi_{period}",
                                     "height": 1000, "width": 1400}})

        with st.expander("Tema tablosu — getiri, ivme ve çeyrek"):
            show = TQ.drop(columns=["_q"])
            st.dataframe(show, width="stretch",
                         column_config={
                             "Getiri %": st.column_config.NumberColumn(format="%+.2f%%"),
                             "Önceki %": st.column_config.NumberColumn(format="%+.2f%%"),
                             "İvme": st.column_config.NumberColumn(format="%+.2f"),
                             "Semboller": st.column_config.TextColumn(width="medium"),
                         })

    # --- Swing yorumu ---
    section("Swing işlem karakteri — canlı tarama")
    st.caption("Aşağıdaki gruplar sabit bir liste değil; ETF bileşenlerinin "
               "güncel ATR, likidite ve sinyal verisinden her yenilemede "
               "yeniden hesaplanır.")

    swing_universe = sorted({t for etf in
                             ["XLK", "SOXX", "SMH", "XLE", "XLI", "XLV", "XLU",
                              "IGV", "CIBR", "ARKX", "WGMI", "PAVE", "URA"]
                             for t in uni.holdings(etf)})
    S = scan_gate("swing_char", swing_universe, "1d",
                  "Swing karakter taramasını çalıştır")

    if not S.empty and "ATR %" in S.columns:
        rows = S[S["Sinyal"] != "⚫ VERİ YOK"].to_dict("records")
        recs = scr.build_recommendations(rows, scr.SwingFilters(), {},
                                         MARKET_REGIME_OK)
        cm = scr.swing_commentary(recs, M.regime, MARKET_REGIME_OK)

        for grup, kayitlar in cm["gruplar"].items():
            st.markdown(f"**{grup}**")
            names = ", ".join(f"`{r['Sembol']}` (ATR %{r['ATR %']:.1f}, "
                              f"skor {r['Skor']})" for r in kayitlar)
            st.markdown(names)
            st.caption(scr.character_note(kayitlar[0]["ATR %"]))

        st.markdown("**Sektör ETF'leri** — `XLE`, `XLV`, `SMH`, `XLI`, `XLU`: "
                    "tek hisse haber riskini seyreltir, çoklu-hisse "
                    "backtest'inde daha temiz istatistik üretir.")

        st.markdown("**Devrede olan pratik filtreler**")
        for ad, aciklama in cm["filtreler"]:
            st.markdown(f"- **{ad}** — {aciklama}")
        for n in cm["notlar"]:
            st.markdown(f"- {n}")


# ==========================================================================
# 3) ETF RADARI
# ==========================================================================
with tab_etf:
    kats = uni.etfs_by_category()
    ec1, ec2 = st.columns([4, 1])
    sec_kat = ec1.multiselect("Kategoriler", list(kats), default=list(kats))
    if ec2.button("🔄 Yenile", width="stretch", key="etf_refresh"):
        bump("nonce_scan")
        st.rerun()

    etf_list = sorted({e for k in sec_kat for e in kats[k]}
                      | set(uni.MAIN_SECTORS))
    E = scan_gate("etf", etf_list, "1d", "ETF sinyallerini tara", auto=True)

    if E.empty:
        st.warning("Veri çekilemedi.", icon="⚠️")
    else:
        E = E.copy()
        E["Kapsam"] = E["Sembol"].map(
            lambda s: uni.ETF.get(s, {}).get("name")
            or uni.MAIN_SECTORS.get(s, "—"))
        cols = ["Sembol", "Kapsam", "Sinyal", "Efor", "Fiyat", "1 Gün %",
                "1 Hafta %", "WHALE", "ΔWHALE", "Whale Yön", "PRO-RET",
                "ΔPRO-RET", "OMNI", "ΔOMNI", "OMNI Yön", "Boğa /6", "Ayı /6",
                "MAGNITUDE", "ΔMAG", "DIRECTION", "ΔDIR", "Hata"]
        cols = [c for c in cols if c in E.columns]
        st.session_state["rep_etf"] = E[cols]
        st.dataframe(
            E[cols].style.map(signal_style, subset=["Sinyal"]),
            width="stretch", hide_index=True,
            column_config={
                "Fiyat": st.column_config.NumberColumn(format="$%.2f"),
                "1 Gün %": st.column_config.NumberColumn(format="%+.2f%%"),
                "1 Hafta %": st.column_config.NumberColumn(format="%+.2f%%"),
                "WHALE": st.column_config.ProgressColumn(
                    format="%.0f", min_value=0, max_value=100),
                "ΔWHALE": st.column_config.NumberColumn(
                    format="%+.1f", help="Bir önceki bara göre WHALE değişimi"),
                "Whale Yön": st.column_config.TextColumn(
                    width="small",
                    help="1 barlık ve 5 barlık değişimin birleşimi"),
                "PRO-RET": st.column_config.NumberColumn(format="%+.1f"),
                "ΔPRO-RET": st.column_config.NumberColumn(format="%+.1f"),
                "OMNI": st.column_config.NumberColumn(format="%.0f"),
                "ΔOMNI": st.column_config.NumberColumn(format="%+.1f"),
                "OMNI Yön": st.column_config.TextColumn(width="small"),
                "ΔMAG": st.column_config.NumberColumn(format="%+.0f"),
                "ΔDIR": st.column_config.NumberColumn(format="%+.0f"),
                "Kapsam": st.column_config.TextColumn(width="medium"),
            })
        st.caption("**Δ sütunları** bir önceki bara göre değişimi gösterir; "
                   "**Yön** sütunu 1 barlık ve 5 barlık değişimi birleştirir: "
                   "⇈ güçleniyor · ↗ dönüyor · → yatay · ↘ soluklanıyor · "
                   "⇊ bozuluyor. WHALE 72 tek başına anlamsızdır — 60'tan mı "
                   "yükseldi yoksa 85'ten mi düştü, karar buna bağlıdır.")

    # --- ETF içeriği ve içerideki hisselerin göreli gücü ---
    section("ETF içi röntgen — kim taşıyor, kim geride kaldı")
    sel = st.selectbox("ETF seçin", uni.all_etfs(),
                       format_func=lambda s: f"{s} — {uni.ETF[s]['name']}")
    if sel:
        d = uni.ETF[sel]
        st.markdown(f"**{d['name']}** · {d['kategori']}")
        if d["aciklama"]:
            st.caption(d["aciklama"])

        hold = uni.holdings(sel)

        # ---------- 1) Bileşen taraması + göreli güç ----------
        if not hold:
            st.info("Bu ETF için tanımlı bileşen listesi yok.")
        else:
            pc1, pc2 = st.columns([1, 2])
            period_col = pc1.radio(
                "Kıyas penceresi", list(hld.PERIOD_COLS),
                format_func=lambda c: hld.PERIOD_COLS[c],
                horizontal=True, index=1, key="hold_period")
            pc2.caption(
                "Getiriler bu pencerede ölçülür. **Akrana Göre** = hissenin "
                "getirisi − aynı listedeki hisselerin medyanı; **Z** bu farkın "
                "medyan mutlak sapmaya bölünmüş hâlidir (tek bir dev bileşenin "
                "listeyi çarpıtmasını engeller). **ETF'e Göre** ise sepetin "
                "kendi getirisiyle kıyastır ve ağırlık etkisini içerir.")

            H = scan_gate(f"hold_{sel}", hold + [sel], "1d",
                          f"{sel} içindeki hisseleri tara",
                          auto=len(hold) <= 30)

            if not H.empty:
                etf_row = H[H["Sembol"] == sel]
                etf_row = etf_row.iloc[0].to_dict() if not etf_row.empty else None
                comp = H[(H["Sembol"] != sel) & (H["Sinyal"] != "⚫ VERİ YOK")]
                HT = hld.build_holdings_table(comp, sel, period_col, etf_row)

                if HT.empty:
                    st.warning("Bileşen verisi çekilemedi.", icon="⚠️")
                else:
                    anlati = hld.holdings_narrative(HT, sel, period_col)
                    st.session_state["rep_hold"] = {
                        "etf": sel, "period": period_col,
                        "table": HT, "narrative": anlati}
                    for line in anlati:
                        st.markdown(line)

                    cnt = hld.holdings_counts(HT)
                    kcols = st.columns(5)
                    for col, key in zip(kcols, ["lider", "lider_yorgun",
                                                "uyumlu", "geride_akis_var",
                                                "geride_akis_yok"]):
                        v = hld.VERDICTS[key]
                        tone = ("pos" if key == "lider" else
                                "neg" if key == "geride_akis_yok" else "")
                        col.markdown(
                            kpi(f"{v.icon} {v.label}", str(cnt.get(key, 0)),
                                "hisse", tone), unsafe_allow_html=True)

                    with st.expander("Bu beş durum nasıl belirleniyor?"):
                        for v in hld.VERDICTS.values():
                            st.markdown(
                                f"**{v.icon} {v.label}** — {v.aciklama}  \n"
                                f"↳ *{v.aksiyon}*")
                        st.caption(
                            f"Eşikler: akrana göre Z ≤ −{hld.Z_ESIK} ve fark "
                            f"≥ {hld.MIN_FARK_PP} puan ise 'geride', Z ≥ "
                            f"+{hld.Z_ESIK} ise 'lider'. Geride kalanlar, "
                            "kurumsal akış (WHALE/ΔWHALE, toplama-dağıtım, "
                            "likidite süpürmesi) yönüne göre ikiye ayrılır: "
                            "akış içerideyse yakalama adayı, dışarıdaysa tuzak.")

                    # ---------- 2) Ayrışma haritası ----------
                    fig = go.Figure()
                    for key, v in hld.VERDICTS.items():
                        sub = HT[HT["_durum"] == key]
                        if sub.empty:
                            continue
                        fig.add_trace(go.Scatter(
                            x=sub["Akrana Göre"], y=sub["WHALE"],
                            mode="markers+text", text=sub["Sembol"],
                            textposition="top center",
                            textfont=dict(size=10, color="#c2c2cc"),
                            name=f"{v.icon} {v.label}",
                            marker=dict(size=13, color=v.renk,
                                        line=dict(color="#050506", width=1.5)),
                            customdata=np.stack([
                                sub["Getiri %"].fillna(0),
                                sub["ΔWHALE"].fillna(0) if "ΔWHALE" in sub else
                                pd.Series(0, index=sub.index)], axis=-1),
                            hovertemplate=("<b>%{text}</b><br>Akrana göre "
                                           "%{x:+.2f} puan<br>WHALE %{y:.0f}"
                                           "<br>Getiri %{customdata[0]:+.2f}%"
                                           "<br>ΔWHALE %{customdata[1]:+.1f}"
                                           "<extra></extra>")))
                    fig.add_vline(x=0, line=dict(color="#3a3a48", width=1))
                    fig.add_hline(y=50, line=dict(color="#3a3a48", width=1,
                                                  dash="dot"))
                    fig.update_layout(
                        height=430, **CHART_LAYOUT,
                        xaxis_title="Akran medyanına göre fark (puan)",
                        yaxis_title="WHALE — kurumsal akış",
                        legend=dict(orientation="h", y=1.12, x=0))
                    fig.update_xaxes(gridcolor="#1b1b22", zeroline=False)
                    fig.update_yaxes(gridcolor="#1b1b22", zeroline=False)
                    st.plotly_chart(fig, width="stretch")
                    st.caption(
                        "**Sol üst köşe kritiktir:** fiyat akranlarının "
                        "gerisinde (sol) ama kurumsal akış yüksek (üst) — "
                        "yani hisse geri kaldı, para hâlâ içeride. Sol alt "
                        "köşe ise hem fiyatın hem akışın terk ettiği bacak.")

                    # ---------- 3) Tablo ----------
                    tcols = [c for c in [
                        "Sembol", "Durum", "Ağırlık %", "Getiri %", "ETF'e Göre",
                        "Akrana Göre", "Z", "Katkı pp", "Sinyal", "Efor",
                        "WHALE", "ΔWHALE", "Whale Yön", "OMNI", "ΔOMNI",
                        "MAGNITUDE", "DIRECTION", "ATR %", "RS Sıra", "Rejim",
                        "Neden"] if c in HT.columns]
                    st.dataframe(
                        HT[tcols].style.map(signal_style, subset=["Sinyal"]),
                        width="stretch", hide_index=True, height=460,
                        column_config={
                            "Ağırlık %": st.column_config.NumberColumn(
                                format="%.2f%%"),
                            "Getiri %": st.column_config.NumberColumn(
                                format="%+.2f%%"),
                            "ETF'e Göre": st.column_config.NumberColumn(
                                format="%+.2f", help="Hisse getirisi − ETF getirisi"),
                            "Akrana Göre": st.column_config.NumberColumn(
                                format="%+.2f",
                                help="Hisse getirisi − bileşen medyanı"),
                            "Z": st.column_config.NumberColumn(
                                format="%+.2f",
                                help="Akran farkının MAD ile standartlaştırılmışı"),
                            "Katkı pp": st.column_config.NumberColumn(
                                format="%+.3f",
                                help="Ağırlık × getiri — ETF getirisinin kaç "
                                     "puanını bu isim açıklıyor"),
                            "WHALE": st.column_config.ProgressColumn(
                                format="%.0f", min_value=0, max_value=100),
                            "ΔWHALE": st.column_config.NumberColumn(format="%+.1f"),
                            "OMNI": st.column_config.NumberColumn(format="%.0f"),
                            "ΔOMNI": st.column_config.NumberColumn(format="%+.1f"),
                            "MAGNITUDE": st.column_config.NumberColumn(format="%.0f"),
                            "DIRECTION": st.column_config.NumberColumn(format="%+.0f"),
                            "ATR %": st.column_config.NumberColumn(format="%.1f%%"),
                            "RS Sıra": st.column_config.NumberColumn(format="%.0f"),
                            "Whale Yön": st.column_config.TextColumn(width="small"),
                            "Durum": st.column_config.TextColumn(width="medium"),
                            "Neden": st.column_config.TextColumn(width="large"),
                        })

                    # ---------- 4) Hisse hisse teknik durum ----------
                    section(f"{sel} bileşenlerinin teknik durumu")
                    only_lag = st.checkbox(
                        "Sadece geride kalanları göster", value=False,
                        key=f"lagonly_{sel}")
                    show = HT[HT["_durum"].isin(["geride_akis_var",
                                                 "geride_akis_yok"])] \
                        if only_lag else HT
                    if show.empty:
                        st.info("Bu pencerede geride kalan bileşen yok.")
                    for _, r in show.iterrows():
                        v = hld.VERDICTS[r["_durum"]]
                        w = (f" · ağırlık %{r['Ağırlık %']:.2f}"
                             if pd.notna(r.get("Ağırlık %")) else "")
                        with st.expander(
                                f"{v.icon} {r['Sembol']} — {r['Sinyal']} · "
                                f"{hld.PERIOD_COLS[period_col]} "
                                f"%{r['Getiri %']:+.2f} "
                                f"(akrana göre {r['Akrana Göre']:+.2f}){w}"):
                            if r.get("Rol"):
                                st.caption(f"Roldeki yeri: {r['Rol']}")
                            st.markdown(f"**Durum:** {v.label} — {r['Neden']}")
                            st.markdown(r["Teknik Not"])
                            st.caption(f"Ne yapılır: {v.aksiyon}")
                            diger = [x for x in uni.etfs_containing(r["Sembol"])
                                     if x != sel]
                            if diger:
                                st.caption("Ayrıca şu temalarda: "
                                           + ", ".join(f"`{x}`" for x in diger))

        # ---------- 5) Ağırlık dağılımı (referans) ----------
        with st.expander("Ağırlık dağılımı ve grup listeleri"):
            if d["agirlik"]:
                wdf = pd.DataFrame([{
                    "Sembol": t,
                    "Ağırlık %": d["agirlik"].get(t),
                    "Rol": d["rol"].get(t, ""),
                    "Diğer ETF'ler": ", ".join(
                        x for x in uni.etfs_containing(t) if x != sel)[:60],
                } for t in hold])
                st.dataframe(wdf, width="stretch", hide_index=True,
                             column_config={
                                 "Ağırlık %": st.column_config.NumberColumn(
                                     format="%.2f%%"),
                                 "Rol": st.column_config.TextColumn(
                                     width="large")})
                top = [t for t in hold if d["agirlik"].get(t)][:10]
                if top:
                    pie = go.Figure(go.Pie(
                        labels=top, values=[d["agirlik"][t] for t in top],
                        hole=0.55, sort=False,
                        marker=dict(colors=[SERIES[i % len(SERIES)]
                                            for i in range(len(top))],
                                    line=dict(color="#050506", width=2)),
                        textinfo="label+percent", textposition="inside"))
                    pie.update_layout(height=360, showlegend=False, **CHART_LAYOUT)
                    st.plotly_chart(pie, width="stretch")
            if d["gruplar"]:
                for grup, lst in d["gruplar"].items():
                    st.markdown(f"**{grup}:** "
                                + ", ".join(f"`{t}`" for t in lst))


# ==========================================================================
# 4) ÇARPAN UÇURUMU
# ==========================================================================
with tab_val:
    st.markdown("Bir hissenin kaç farklı temada birden yer aldığını gösterir. "
                "Bu kesişim kümesi, paranın hangi alt temaya gittiğinden bağımsız "
                "olarak pastadan pay alan **altyapı sahiplerini** ortaya çıkarır.")

    section("Chokepoint çarpanları")
    for sym, info in uni.CHOKEPOINTS.items():
        etfs = uni.etfs_containing(sym)
        with st.expander(f"{sym} — {info['rol']}  ({len(etfs)} temada)"):
            st.markdown(f"**Beslendiği CapEx kaynağı:** {info['capex']}")
            st.markdown(info["mantik"])
            if etfs:
                st.caption("Bağladığı temalar: " + ", ".join(f"`{e}`" for e in etfs))

    section("Kesişim yoğunluğu — tüm evren")
    counts = [{"Sembol": t, "Tema Sayısı": len(uni.etfs_containing(t)),
               "Temalar": ", ".join(uni.etfs_containing(t))}
              for t in uni.all_stocks()]
    cdf = pd.DataFrame(counts).sort_values("Tema Sayısı", ascending=False)
    cdf = cdf[cdf["Tema Sayısı"] >= 2]
    st.dataframe(cdf.head(40), width="stretch", hide_index=True,
                 column_config={"Temalar": st.column_config.TextColumn(
                     width="large")})


# ==========================================================================
# 5) HAFTALIK MOMENTUM
# ==========================================================================
with tab_week:
    if st.button("🔄 Haftalık veriyi yenile", key="wk"):
        bump("nonce_scan")
        bump("nonce_short")
        st.rerun()
    universe = uni.all_stocks()
    W = scan_gate("weekly", universe, "1wk", "Haftalık momentumu tara")
    if not W.empty:
        W = W.copy()
        # Haftalık barlarda motorun "1 Gün %" sütunu = 1 bar = 1 HAFTA,
        # "1 Hafta %" sütunu = 5 bar = 5 HAFTA. Etiketler buna göre düzeltildi.
        W = W.rename(columns={"1 Gün %": "Haftalık %", "1 Hafta %": "5 Hafta %"})

        # ---------- FINRA short hacmi ----------
        section("📉 Short verisi")
        with st.spinner("FINRA short hacim dosyaları çekiliyor…"):
            SVRAW, sv_warn, sv_diag = load_short_volume(
                st.session_state.nonce_short)
        # Short interest her sembol için ayrı Yahoo isteği demek; yüzlerce
        # sembolde Yahoo geçici blok uygulayabildiği için isteğe bağlı.
        si_on = st.toggle(
            "Short interest (yfinance) sütunlarını ekle — yavaş, ilk "
            "çekimde 1–3 dk", key="si_on")
        SI = pd.DataFrame(columns=["Sembol"])
        if si_on:
            with st.spinner("Short interest (yfinance) çekiliyor…"):
                SI = load_short_interest(tuple(W["Sembol"].tolist()),
                                         st.session_state.nonce_short)
        n_ok = sum(1 for m in sv_diag.values() if m.startswith("ok"))
        n_repo = sum(1 for m in sv_diag.values() if "repo" in m)
        codes = sorted({m for m in sv_diag.values()
                        if not m.startswith("ok") and not m.startswith("HTTP 404")})
        pchg = dict(zip(W["Sembol"],
                        pd.to_numeric(W.get("Haftalık %"), errors="coerce")))
        SVT = svm.short_table(SVRAW, W["Sembol"].tolist(), pchg)
        n_match = int(SVT["ΔSV pp"].notna().sum()) if not SVT.empty else 0
        n_si = (int(SI["SI Δ% (ay)"].notna().sum())
                if "SI Δ% (ay)" in SI.columns else 0)
        durum = (f"**FINRA günlük short hacmi:** "
                 + (f"✅ {n_ok} gün ({n_repo} gün repodan), {n_match}/{len(W)} sembol eşleşti"
                    if n_ok else f"❌ alınamadı ({', '.join(codes) or 'yanıt yok'})")
                 + "  \n**yfinance short interest:** "
                 + (f"✅ {n_si}/{len(W)} sembol" if si_on and n_si
                    else "⏳ açık ama veri gelmedi — birkaç dk sonra yenileyin"
                    if si_on else "kapalı — yukarıdaki anahtarla açın"))
        (st.success if (n_match or n_si) else st.warning)(durum)
        for w_ in sv_warn:
            st.caption(f"⚠️ {w_}")
        W = W.merge(SVT, on="Sembol", how="left")
        # eşleşmeyen satırlarda NaN kalırsa çizgi grafik sütunu hata verir
        W["SV Seyri"] = W["SV Seyri"].apply(
            lambda x: x if isinstance(x, list) else None)
        W["SV Yön"] = W["SV Yön"].fillna("—")
        if not SI.empty:
            W = W.merge(SI, on="Sembol", how="left")

        # FINRA günlük verisi yoksa yorum, aylık short interest değişiminden
        def _si_yorum(r):
            d, p = r.get("SI Δ% (ay)"), r.get("Haftalık %")
            if pd.isna(d):
                return "Veri yok (OTC/kripto)"
            yon = ("🟠 Short interest artıyor" if d >= 10
                   else "🟢 Short interest azalıyor" if d <= -10
                   else "⚪ Short interest yatay")
            return f"{yon} ({d:+.0f}% / ay)"
        miss = W["Short Yorum"].isna()
        if "SI Δ% (ay)" in W.columns and miss.any():
            W.loc[miss, "Short Yorum"] = [_si_yorum(r) for r in
                                          W[miss].to_dict("records")]
        W["Short Yorum"] = W["Short Yorum"].fillna("Veri yok (OTC/kripto)")

        sv_filter = st.radio(
            "Short filtresi",
            ["Tümü", "⬆️ Short artanlar", "⬇️ Short azalanlar",
             "Aşırı uç (|Z| ≥ 2)"],
            horizontal=True, key="sv_filter")
        # FINRA günlük verisi varsa ΔSV pp, yoksa aylık SI değişimi kullanılır
        have_sv = W["ΔSV pp"].notna().any()
        fkey, fthr = (("ΔSV pp", svm.ESIK_PP) if have_sv
                      else ("SI Δ% (ay)", 10.0))
        if fkey not in W.columns:
            W[fkey] = np.nan
        view = W
        if sv_filter.startswith("⬆️"):
            view = W[W[fkey] >= fthr].sort_values(fkey, ascending=False)
        elif sv_filter.startswith("⬇️"):
            view = W[W[fkey] <= -fthr].sort_values(fkey)
        elif sv_filter.startswith("Aşırı"):
            view = W[W["SV% Z"].abs() >= 2].sort_values("SV% Z",
                                                         ascending=False)

        cols = [c for c in ["Sembol", "Sinyal", "Efor", "Fiyat", "Haftalık %",
                            "5 Hafta %", "WHALE", "ΔWHALE", "ΔWHALE 5B",
                            "Whale Yön", "SV% 5G", "ΔSV pp", "Short Hacim Δ%",
                            "SV% Z", "SV Yön", "SV Seyri", "SI Float %",
                            "SI Δ% (ay)", "Gün Kapama", "Short Yorum",
                            "PRO-RET", "ΔPRO-RET", "MAGNITUDE", "ΔMAG",
                            "DIRECTION", "ΔDIR", "OMNI", "ΔOMNI", "OMNI Yön",
                            "SV Son Gün", "Hata"] if c in view.columns]
        st.dataframe(
            view[cols].style.map(signal_style, subset=["Sinyal"]),
            width="stretch", hide_index=True,
            column_config={
                "Fiyat": st.column_config.NumberColumn(format="$%.2f"),
                "Haftalık %": st.column_config.NumberColumn(
                    format="%+.2f%%", help="Son haftalık barın değişimi"),
                "5 Hafta %": st.column_config.NumberColumn(format="%+.2f%%"),
                "WHALE": st.column_config.ProgressColumn(
                    format="%.0f", min_value=0, max_value=100),
                "ΔWHALE": st.column_config.NumberColumn(format="%+.1f"),
                "ΔWHALE 5B": st.column_config.NumberColumn(
                    format="%+.1f", help="5 bar önceki değere göre değişim"),
                "Whale Yön": st.column_config.TextColumn(width="small"),
                "SV% 5G": st.column_config.NumberColumn(
                    format="%.1f%%",
                    help="Son 5 işlem gününde short hacmin toplam hacme oranı "
                         "(hacim ağırlıklı)"),
                "ΔSV pp": st.column_config.NumberColumn(
                    format="%+.1f",
                    help="Son 5 gün oranı − önceki 5 gün oranı (yüzde puan)"),
                "Short Hacim Δ%": st.column_config.NumberColumn(
                    format="%+.0f%%",
                    help="Short hisse adedinin önceki 5 güne göre değişimi"),
                "SV% Z": st.column_config.NumberColumn(
                    format="%+.1f",
                    help="Son günün oranı, hissenin kendi 20 günlük "
                         "ortalamasından kaç σ uzakta"),
                "SV Yön": st.column_config.TextColumn(width="small"),
                "SV Seyri": st.column_config.LineChartColumn(
                    "SV% Seyri", y_min=0, y_max=100, width="small",
                    help="Son 15 işlem günü günlük short hacim oranı"),
                "SI Float %": st.column_config.NumberColumn(
                    format="%.1f%%",
                    help="Açık short pozisyonun halka açık hisseye oranı "
                         "(borsa verisi, ayda iki kez güncellenir)"),
                "SI Δ% (ay)": st.column_config.NumberColumn(
                    format="%+.0f%%",
                    help="Açık short pozisyonun bir önceki aya göre değişimi"),
                "Gün Kapama": st.column_config.NumberColumn(
                    format="%.1f",
                    help="Short'ların ortalama hacimle kaç günde kapanabileceği"),
                "Short Yorum": st.column_config.TextColumn(width="large"),
                "ΔPRO-RET": st.column_config.NumberColumn(format="%+.1f"),
                "ΔMAG": st.column_config.NumberColumn(format="%+.0f"),
                "ΔDIR": st.column_config.NumberColumn(format="%+.0f"),
                "ΔOMNI": st.column_config.NumberColumn(format="%+.1f"),
                "OMNI Yön": st.column_config.TextColumn(width="small")})
        st.caption("Haftalık barlarda Δ, bir hafta önceki değere göre "
                   "değişimdir; Δ5B ise beş hafta öncesine göre. Kurumsal "
                   "toplama (WHALE) yükselirken fiyatın yatay kalması, "
                   "scriptlerdeki 'stealth accumulation' durumudur.")

        # ---------- en büyük değişimler ----------
        section("Short hacminde en büyük haftalık değişimler")
        sv_ok = W.dropna(subset=[fkey])
        if sv_ok.empty:
            st.info("Short verisi eşleşmedi.")
        else:
            mini = (["Sembol", "SV% 5G", "ΔSV pp", "Haftalık %", "WHALE",
                     "Short Yorum"] if have_sv else
                    ["Sembol", "SI Float %", "SI Δ% (ay)", "Gün Kapama",
                     "Haftalık %", "WHALE", "Short Yorum"])
            mini = [c for c in mini if c in sv_ok.columns]
            mcfg = {"SV% 5G": st.column_config.NumberColumn(format="%.1f%%"),
                    "ΔSV pp": st.column_config.NumberColumn(format="%+.1f"),
                    "Haftalık %": st.column_config.NumberColumn(format="%+.2f%%"),
                    "WHALE": st.column_config.NumberColumn(format="%.0f"),
                    "SI Float %": st.column_config.NumberColumn(format="%.1f%%"),
                    "SI Δ% (ay)": st.column_config.NumberColumn(format="%+.0f%%"),
                    "Gün Kapama": st.column_config.NumberColumn(format="%.1f")}
            a, b = st.columns(2)
            a.markdown("**⬆️ Short en çok artan 10**")
            a.dataframe(sv_ok.nlargest(10, fkey)[mini], width="stretch",
                        hide_index=True, column_config=mcfg)
            b.markdown("**⬇️ Short en çok azalan 10**")
            b.dataframe(sv_ok.nsmallest(10, fkey)[mini], width="stretch",
                        hide_index=True, column_config=mcfg)

        with st.expander("Short hacmi nasıl okunur?"):
            st.markdown(f"""
- **Kaynak:** FINRA Reg SHO günlük dosyası (ücretsiz, her iş günü ABD
  akşamı yayımlanır). Son veri: **{SVT['SV Son Gün'].dropna().max()
  if not SVT.empty else '—'}**.
- **Short HACMİ ≠ short INTEREST.** Piyasa yapıcıların alıcıya hisse
  sağlamak için yaptığı gün içi açığa satışlar da buraya girer; çoğu
  hissede oran zaten %35–55'tir. Bu yüzden **seviyeye değil değişime**
  bakın: ΔSV pp (son 5 gün − önceki 5 gün) ve Z (hissenin kendi 20
  günlük ortalamasından sapma).
- Anlamlı değişim eşiği **±{svm.ESIK_PP:.0f} puan**; |Z| ≥ 2 aşırı uç.
- **Okuma matrisi:** short ↑ + fiyat ↓ = baskı · short ↑ + fiyat ↑ =
  dağıtım ya da squeeze yakıtı (WHALE'e bakın: WHALE yükseliyorsa squeeze
  tarafı) · short ↓ + fiyat ↑ = kapama desteği · short ↓ + fiyat ↓ =
  satış uzun pozisyonlardan geliyor.
- OTC semboller (FANUY, YASKY…) ve kripto FINRA konsolide dosyasında
  olmadığı için boş kalır.
- **SI sütunları** (yfinance): borsanın ayda iki kez yayımladığı *açık*
  short pozisyon. Daha yavaş ama daha anlamlıdır: SI Float % %10'un,
  Gün Kapama 5'in üstündeyse squeeze potansiyeli yüksektir. FINRA
  verisi çekilemezse filtreler ve listeler SI Δ% (ay) ile çalışır.
""")
            st.markdown("**FINRA bağlantı tanısı** (gün → sonuç)")
            st.dataframe(pd.DataFrame({"Gün": list(sv_diag),
                                       "Sonuç": list(sv_diag.values())}),
                         hide_index=True, width="stretch", height=220)


# ==========================================================================
# 6) 4H OMNI SWING
# ==========================================================================
with tab_omni:
    st.markdown("ETF bileşenleri arasından **swing işlem adayları**. Sıralama "
                "konfluans motorunun MAGNITUDE/DIRECTION skoruna, kurumsal akışa "
                "ve göreli güce göre yapılır; ardından pratik filtreler uygulanır.")

    f1, f2, f3 = st.columns(3)
    src_etfs = f1.multiselect(
        "Kaynak ETF'ler", uni.all_etfs(),
        default=["XLK", "SOXX", "SMH", "IGV", "XLU", "PAVE", "URA", "ARKX"])
    interval = f2.selectbox("Zaman dilimi", ["1d", "4h", "1wk"], index=0,
                            format_func=lambda x: {"1d": "Günlük",
                                                   "4h": "4 Saatlik",
                                                   "1wk": "Haftalık"}[x])
    min_score = f3.slider("Minimum skor", 0, 100, 45, 5)

    g1, g2, g3, g4 = st.columns(4)
    min_liq = g1.number_input("Min likidite ($M)", 0.0, 500.0, 5.0, 1.0)
    earn_buf = g2.number_input("Bilanço tamponu (gün)", 0, 10, 2, 1)
    use_regime = g3.toggle("Rejim kapısı", value=True,
                           help="SPY 50 EMA altındayken long sinyalleri ele")
    use_weekly = g4.toggle("Haftalık teyit", value=False)

    cand = sorted({t for e in src_etfs for t in uni.holdings(e)})
    st.caption(f"{len(cand)} aday sembol · kaynak: "
               + ", ".join(f"`{e}`" for e in src_etfs))

    if cand:
        R = scan_gate(f"omni_{interval}", cand, interval,
                      "Swing taramasını çalıştır")

        earn_map: dict[str, int] = {}
        if earn_buf > 0:
            with st.spinner("Bilanço takvimi kontrol ediliyor…"):
                edf = load_earnings(tuple(cand[:120]), st.session_state.nonce_earn)
            for _, r in edf.iterrows():
                if r.get("Kalan Gün") is not None and np.isfinite(
                        float(r["Kalan Gün"] or np.nan)):
                    earn_map[r["Hisse"]] = int(r["Kalan Gün"])

        if not R.empty:
            rows = R[R["Sinyal"] != "⚫ VERİ YOK"].to_dict("records")
            filt = scr.SwingFilters(min_dollar_vol_m=min_liq,
                                    earnings_buffer_days=int(earn_buf),
                                    require_regime=use_regime,
                                    require_weekly=use_weekly,
                                    min_score=min_score)
            RECS = scr.build_recommendations(rows, filt, earn_map, MARKET_REGIME_OK)
            st.session_state["rep_swing"] = RECS

            if RECS.empty:
                st.info("Aday bulunamadı.")
            else:
                # her adayın hangi ETF'ten geldiğini işaretle
                RECS["ETF"] = RECS["Sembol"].map(
                    lambda t: ", ".join(e for e in uni.etfs_containing(t)
                                        if e in src_etfs)[:24])
                uygun = RECS[RECS["Uygun"]]
                k1, k2, k3 = st.columns(3)
                k1.markdown(kpi("Filtreyi Geçen", str(len(uygun)),
                                f"{len(RECS)} aday tarandı",
                                "pos" if len(uygun) else ""), unsafe_allow_html=True)
                en_iyi = uygun.iloc[0] if len(uygun) else None
                k2.markdown(kpi("En Yüksek Skor",
                                en_iyi["Sembol"] if en_iyi is not None else "—",
                                (f"skor {en_iyi['Skor']} · {en_iyi['Sinyal']}"
                                 if en_iyi is not None else "")),
                            unsafe_allow_html=True)
                k3.markdown(kpi("Piyasa Rejimi",
                                "AÇIK" if MARKET_REGIME_OK else "KAPALI",
                                M.regime, "pos" if MARKET_REGIME_OK else "neg"),
                            unsafe_allow_html=True)

                section("Tavsiye edilen adaylar")
                show_cols = [c for c in
                             ["Sembol", "ETF", "Skor", "Sinyal", "Karakter", "Fiyat",
                              "ATR %", "Hacim ($M)", "WHALE", "ΔWHALE",
                              "Whale Yön", "OMNI", "ΔOMNI", "MAGNITUDE", "ΔMAG",
                              "DIRECTION", "RS Sıra", "Stop", "T1", "T2", "R (T2)",
                              "Risk %", "Bilanço Gün"] if c in uygun.columns]
                st.dataframe(
                    uygun[show_cols].style.map(signal_style, subset=["Sinyal"]),
                    width="stretch", hide_index=True,
                    column_config={
                        "Skor": st.column_config.ProgressColumn(
                            format="%d", min_value=0, max_value=100),
                        "Fiyat": st.column_config.NumberColumn(format="$%.2f"),
                        "Stop": st.column_config.NumberColumn(format="$%.2f"),
                        "T1": st.column_config.NumberColumn(format="$%.2f"),
                        "T2": st.column_config.NumberColumn(format="$%.2f"),
                        "ATR %": st.column_config.NumberColumn(format="%.2f%%"),
                        "Risk %": st.column_config.NumberColumn(format="%.2f%%"),
                        "R (T2)": st.column_config.NumberColumn(format="%.2fR"),
                        "Hacim ($M)": st.column_config.NumberColumn(format="%.0f"),
                        "RS Sıra": st.column_config.NumberColumn(format="%.0f"),
                        "ΔWHALE": st.column_config.NumberColumn(format="%+.1f"),
                        "ΔOMNI": st.column_config.NumberColumn(format="%+.1f"),
                        "ΔMAG": st.column_config.NumberColumn(format="%+.0f"),
                        "Whale Yön": st.column_config.TextColumn(width="small"),
                    })
                st.caption("**Stop** = iz süren ATR zırhı (low − 2×ATR, sadece yukarı "
                           "kayar, girişten %20 aşağıda sert taban). **T1/T2** = "
                           "1.8× ve 3.5× ATR, volatilite sıkışmasıyla ölçeklenir. "
                           "**R (T2)** = hedefe giden mesafenin riske oranı.")

                with st.expander(f"Filtreye takılanlar ({len(RECS) - len(uygun)})"):
                    red = RECS[~RECS["Uygun"]]
                    st.dataframe(
                        red[[c for c in ["Sembol", "Skor", "Sinyal", "Engel"]
                             if c in red.columns]],
                        width="stretch", hide_index=True,
                        column_config={"Engel": st.column_config.TextColumn(
                            width="large")})


# ==========================================================================
# 7) FUTURE THEMES
# ==========================================================================
with tab_future:
    wl = st.session_state.watchlist["future_themes"]

    st.markdown("Kendi ayıkladığınız hisse ve ETF'ler. Ekleme/çıkarma "
                + ("**GitHub deposuna kalıcı yazılır**." if store.backend == "github"
                   else "sadece bu oturumda tutulur (Secrets ekleyin)."))

    mc1, mc2 = st.columns([3, 1])
    tema_sec = mc1.selectbox("Tema", list(wl) + ["➕ Yeni tema oluştur…"])
    if tema_sec == "➕ Yeni tema oluştur…":
        yeni = mc2.text_input("Tema adı", key="yeni_tema")
        if st.button("Tema oluştur") and yeni.strip():
            wl[yeni.strip()] = {"hisse": [], "etf": []}
            if save_watchlist():
                st.success(f"'{yeni.strip()}' oluşturuldu.")
                st.rerun()
    else:
        tema = wl[tema_sec]
        a1, a2 = st.columns(2)
        with a1:
            st.markdown("**Hisse ekle**")
            yeni_h = st.text_input("Semboller (virgül/boşluk ile)",
                                   key="fh_add",
                                   placeholder="NVDA, AVGO CEG")
            if st.button("➕ Hisse ekle", width="stretch"):
                import re as _re
                yeni = [x.strip().upper() for x in _re.split(r"[,\s]+", yeni_h)
                        if x.strip()]
                tema["hisse"] = sorted(set(tema["hisse"]) | set(yeni))
                if save_watchlist():
                    st.success(f"{len(yeni)} sembol eklendi.")
                    st.rerun()
            cik_h = st.multiselect("Çıkarılacak hisseler", tema["hisse"],
                                   key="fh_del")
            if cik_h and st.button("🗑️ Seçilen hisseleri çıkar", width="stretch"):
                tema["hisse"] = [t for t in tema["hisse"] if t not in cik_h]
                if save_watchlist():
                    st.rerun()
        with a2:
            st.markdown("**ETF ekle**")
            yeni_e = st.multiselect("ETF seç", uni.all_etfs(), key="fe_add")
            ekstra_e = st.text_input("Listede olmayan ETF", key="fe_txt",
                                     placeholder="ITA, KWEB")
            if st.button("➕ ETF ekle", width="stretch"):
                import re as _re
                extra = [x.strip().upper() for x in _re.split(r"[,\s]+", ekstra_e)
                         if x.strip()]
                tema["etf"] = sorted(set(tema["etf"]) | set(yeni_e) | set(extra))
                if save_watchlist():
                    st.rerun()
            cik_e = st.multiselect("Çıkarılacak ETF'ler", tema["etf"], key="fe_del")
            if cik_e and st.button("🗑️ Seçilen ETF'leri çıkar", width="stretch"):
                tema["etf"] = [t for t in tema["etf"] if t not in cik_e]
                if save_watchlist():
                    st.rerun()

        if st.button(f"❌ '{tema_sec}' temasını tamamen sil"):
            wl.pop(tema_sec, None)
            if save_watchlist():
                st.rerun()

    section("Tüm Future Themes evreni")
    fut_tickers = sorted({t for v in wl.values()
                          for t in (v.get("hisse", []) + v.get("etf", []))})
    st.caption(f"{len(fut_tickers)} sembol · {len(wl)} tema")
    if fut_tickers:
        F = scan_gate("future", fut_tickers, "1d", "Future Themes tara",
                      auto=True)
        if not F.empty:
            F = F.copy()
            F["Tema"] = F["Sembol"].map(
                lambda t: ", ".join(k for k, v in wl.items()
                                    if t in v.get("hisse", []) + v.get("etf", [])))
            cols = [c for c in ["Sembol", "Tema", "Sinyal", "Efor", "Fiyat",
                                "1 Gün %", "WHALE", "OMNI", "MAGNITUDE",
                                "DIRECTION", "Hata"] if c in F.columns]
            st.dataframe(F[cols].style.map(signal_style, subset=["Sinyal"]),
                         width="stretch", hide_index=True,
                         column_config={
                             "Fiyat": st.column_config.NumberColumn(format="$%.2f"),
                             "1 Gün %": st.column_config.NumberColumn(
                                 format="%+.2f%%"),
                             "Tema": st.column_config.TextColumn(width="medium")})

    with st.expander("Yedek al / geri yükle"):
        st.download_button(
            "⬇️ İzleme listesini indir",
            json.dumps(st.session_state.watchlist, indent=2,
                       ensure_ascii=False).encode("utf-8"),
            "apex_watchlist.json", "application/json")
        up = st.file_uploader("Yedekten yükle", type="json")
        if up is not None and st.button("📥 Uygula"):
            try:
                st.session_state.watchlist = json.loads(
                    up.getvalue().decode("utf-8"))
                if save_watchlist():
                    st.success("Yüklendi.")
                    st.rerun()
            except Exception as exc:
                st.error(f"Okunamadı: {exc}")


# ==========================================================================
# 8) BİLANÇO
# ==========================================================================
with tab_earn:
    lst = st.session_state.watchlist["earnings"]
    e1, e2 = st.columns([4, 1])
    add_txt = e1.text_input("Sembol ekle (virgül/boşluk ile)",
                            placeholder="AAPL, TSLA MSFT")
    if e2.button("➕ Ekle", width="stretch"):
        import re as _re
        new = [x.strip().upper() for x in _re.split(r"[,\s]+", add_txt)
               if x.strip()]
        st.session_state.watchlist["earnings"] = sorted(set(lst) | set(new))
        if save_watchlist():
            st.rerun()

    with st.expander(f"Listeyi yönet ({len(lst)} sembol)"):
        rem = st.multiselect("Çıkarılacaklar", lst)
        if rem and st.button("🗑️ Çıkar"):
            st.session_state.watchlist["earnings"] = [t for t in lst if t not in rem]
            if save_watchlist():
                st.rerun()

    if st.button("🔄 Bilanço takvimini tara", type="primary"):
        bump("nonce_earn")
        st.rerun()

    with st.spinner("Bilanço tarihleri, analist hedefleri ve satış verisi çekiliyor…"):
        EARN = load_earnings(tuple(lst), st.session_state.nonce_earn)
    if lst and _earn_ok_ratio(EARN) < 0.5:
        st.warning("Yahoo Finance şu an istekleri sınırlıyor (bulut "
                   "sunucularında sık olur); verilerin çoğu boş geldi. Birkaç "
                   "dakika bekleyip **🔄 Bilanço takvimini tara** düğmesine "
                   "basın — boş sonuç önbelleğe alınmaz.")

    if not EARN.empty:
        # ---------- P/S adil değer ayarları ----------
        wlst = st.session_state.watchlist
        saved_mult = wlst.get("ps_multiples")
        ind_map = ({k: (None if v is None else float(v))
                    for k, v in saved_mult.items()}
                   if isinstance(saved_mult, dict) and saved_mult
                   else dict(fvm.INDUSTRY_PS))

        with st.expander("⚙️ P/S adil değer ayarları (çarpan tablosu, bant, "
                         "pahalı eşiği)"):
            st.markdown(
                "**Yöntem:** Adil orta = *endüstri çarpanı × gelecek yıl "
                "satış/hisse*; adil bant = orta ± bant. **PSG** = (fiyat / "
                "TTM satış-hisse) ÷ büyüme %. **Pahalı eşiği**, PSG'nin "
                "aşağıdaki değere ulaştığı fiyattır; ancak adil ortanın "
                "belirlenen katını geçemez. Bant üstünde ama pahalı "
                "eşiğinin altındaysa hisse *büyüme destekli adil* sayılır.")
            p1, p2, p3 = st.columns(3)
            band = p1.slider("Adil bant ±%", 5, 30,
                             int(fvm.DEFAULT_BAND * 100), 1) / 100
            psg_max = p2.slider("Pahalı eşiği (PSG)", 0.30, 1.50,
                                fvm.DEFAULT_PSG_MAX, 0.05)
            exp_cap = p3.slider(
                "Pahalı eşiği üst sınırı (× adil orta)", 1.2, 3.0,
                fvm.DEFAULT_EXP_CAP, 0.1,
                help="Büyüme ne kadar yüksek olursa olsun pahalı eşiği adil "
                     "ortanın bu katını geçmez.")
            st.caption("Çarpan tablosu endüstri adında anahtar kelime "
                       "arar; ilk eşleşen kazanır. Boş bırakılan çarpan = "
                       "P/S bu iş modelinde anlamsız (banka, sigorta, petrol, "
                       "GYO). Eşleşmeyen endüstri sektör varsayılanına düşer.")
            med = st.data_editor(
                pd.DataFrame({"Endüstri anahtarı": list(ind_map),
                              "İleri P/S çarpanı": list(ind_map.values())}),
                num_rows="dynamic", hide_index=True, width="stretch",
                key="ps_mult_editor",
                column_config={"İleri P/S çarpanı": st.column_config.NumberColumn(
                    min_value=0.1, max_value=60.0, step=0.5, format="%.1fx")})
            ind_map = {str(r["Endüstri anahtarı"]).strip():
                       (None if pd.isna(r["İleri P/S çarpanı"])
                        else float(r["İleri P/S çarpanı"]))
                       for _, r in med.iterrows()
                       if str(r["Endüstri anahtarı"]).strip()
                       and str(r["Endüstri anahtarı"]) != "nan"}
            s1, s2 = st.columns(2)
            if s1.button("💾 Çarpanları kaydet", width="stretch"):
                wlst["ps_multiples"] = ind_map
                if save_watchlist():
                    st.success("Çarpan tablosu kaydedildi.")
            if s2.button("↩️ Varsayılana dön", width="stretch"):
                wlst.pop("ps_multiples", None)
                save_watchlist()
                st.rerun()

        EARN = fvm.add_fair_values(EARN, ind_map=ind_map, band=band,
                                   psg_max=psg_max, exp_cap=exp_cap)

        # İki kaynağın (analist hedefi + P/S orta) aynı yönü gösterip
        # göstermediği — tek başına hiçbiri değerleme modeli değildir.
        def _uyum(r):
            p, h, m = r.get("Fiyat"), r.get("Hedef"), r.get("Adil Orta")
            if not p or pd.isna(h) or pd.isna(m):
                return "—"
            if h > p and m > p:
                return "🟢 İkisi de yukarı"
            if h < p and m < p:
                return "🔴 İkisi de aşağı"
            return "🟡 Ayrışıyor"
        EARN["Kaynak Uyumu"] = [_uyum(r) for r in EARN.to_dict("records")]

        def style_days(v):
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                return ""
            if v <= 2:
                return "background-color:#4a0d12;color:#fff;font-weight:700"
            if v <= 7:
                return "background-color:#3a2a05;color:#ffd54f;font-weight:700"
            if v <= 15:
                return "color:#ffd54f"
            return ""

        def style_pot(v):
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                return ""
            return ("color:#2fbe86;font-weight:700" if v > 0
                    else "color:#f0736f;font-weight:700")

        def style_ps(v):
            if not isinstance(v, str):
                return ""
            if v.startswith("🟢"):
                return "background-color:#0d3b2a;color:#4fd6a0;font-weight:700"
            if v.startswith("🔴"):
                return "background-color:#4a0d12;color:#ff8f8b;font-weight:700"
            if v.startswith("🟡"):
                return "background-color:#3a3205;color:#ffd54f;font-weight:700"
            return "color:#8a8a95"

        # Uzaklık: fiyat adil ortanın üstündeyse kırmızı, altındaysa yeşil
        def style_dist(v):
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                return ""
            return ("color:#f0736f;font-weight:700" if v > 0
                    else "color:#2fbe86;font-weight:700")

        ecols = [c for c in [
            "Hisse", "Bilanço", "Kalan Gün", "Fiyat", "P/S Durum", "Adil Alt",
            "Adil Orta", "Adil Üst", "Pahalı >", "Adile Uzaklık %", "Hedef",
            "Potansiyel %", "Kaynak Uyumu", "P/S", "İleri P/S", "PSG",
            "Büyüme %", "Çarpan", "Çarpan Kaynağı", "EV/Satış", "İleri F/K",
            "Piyasa Değ. ($B)", "52H Konum %", "Analist"] if c in EARN.columns]
        st.dataframe(
            EARN[ecols].style.map(style_days, subset=["Kalan Gün"])
                             .map(style_pot, subset=["Potansiyel %"])
                             .map(style_ps, subset=["P/S Durum"])
                             .map(style_dist, subset=["Adile Uzaklık %"]),
            width="stretch", hide_index=True,
            column_config={
                "Fiyat": st.column_config.NumberColumn(format="$%.2f"),
                "Hedef": st.column_config.NumberColumn(
                    format="$%.2f", help="Analist ortalama hedef fiyatı"),
                "Potansiyel %": st.column_config.NumberColumn(format="%+.1f%%"),
                "Kalan Gün": st.column_config.NumberColumn(format="%d"),
                "P/S Durum": st.column_config.TextColumn(width="medium"),
                "Adil Alt": st.column_config.NumberColumn(format="$%.2f"),
                "Adil Orta": st.column_config.NumberColumn(format="$%.2f"),
                "Adil Üst": st.column_config.NumberColumn(format="$%.2f"),
                "Pahalı >": st.column_config.NumberColumn(
                    format="$%.2f",
                    help="Bu fiyatın üstü satışlara göre pahalı. Büyüme "
                         "yüksekse esner, en fazla adil ortanın 1.8 katı "
                         "(ayarlardan değişir)."),
                "Adile Uzaklık %": st.column_config.NumberColumn(
                    format="%+.1f%%", help="Fiyatın adil ortaya göre konumu"),
                "P/S": st.column_config.NumberColumn(format="%.1fx"),
                "İleri P/S": st.column_config.NumberColumn(format="%.1fx"),
                "PSG": st.column_config.NumberColumn(format="%.2f"),
                "Büyüme %": st.column_config.NumberColumn(format="%.0f%%"),
                "Çarpan": st.column_config.NumberColumn(format="%.1fx"),
                "EV/Satış": st.column_config.NumberColumn(format="%.1fx"),
                "İleri F/K": st.column_config.NumberColumn(format="%.1fx"),
                "Piyasa Değ. ($B)": st.column_config.NumberColumn(format="%.1f"),
                "52H Konum %": st.column_config.ProgressColumn(
                    format="%.0f%%", min_value=0, max_value=100),
            })
        st.caption("Kırmızı satırlar bilanço tamponu içindedir — swing "
                   "taramasında bu hisseler otomatik elenir. **Hedef** analist "
                   "ortalamasıdır; **P/S adil değer** ise satış çarpanından "
                   "türetilen ikinci, bağımsız bir referanstır. İkisi de "
                   "değerleme modeli değil, bağlam göstergesidir — "
                   "**Kaynak Uyumu** ikisinin aynı yönü gösterip "
                   "göstermediğini özetler.")

        section("Hızlı bakış")
        pick = st.selectbox("Hisse", EARN["Hisse"].tolist(), key="glance_pick")
        if pick:
            r = EARN[EARN["Hisse"] == pick].iloc[0].to_dict()
            st.markdown(fvm.glance_text(r))
            if r.get("Satış Tahmin Kaynağı"):
                st.caption(f"Satış tahmini: {r['Satış Tahmin Kaynağı']} · "
                           f"endüstri: {r.get('Endüstri') or '—'}")


# ==========================================================================
# 9) PDF RAPOR
# ==========================================================================
with tab_report:
    st.markdown(
        "Ekrandaki tüm hesaplamaları tek bir PDF'e toplar: makro rejim ve "
        "sermaye akışı (geçmiş karşılaştırmasıyla), rejim oyun kitabı, tema "
        "haritası, ETF radarı, seçili ETF'in iç röntgeni, swing adayları ve "
        "haber akışı. Grafikler rapora yeniden çizilir — ekran görüntüsü değil, "
        "baskıya uygun vektörel sayfa düzeni üretilir.")

    have_theme = "rep_theme" in st.session_state
    have_etf = "rep_etf" in st.session_state
    have_hold = "rep_hold" in st.session_state
    have_swing = "rep_swing" in st.session_state
    have_news = "rep_news" in st.session_state

    section("Rapora nelerin gireceğini seçin")
    st.caption("Bir bölüm soluksa, o sekmedeki tarama henüz çalıştırılmamıştır. "
               "İlgili sekmeye gidip taramayı başlatın, sonra buraya dönün — "
               "sonuçlar oturumda saklanır.")

    r1, r2, r3 = st.columns(3)
    inc_macro = r1.checkbox("Makro & sermaye akışı", value=True)
    inc_hist = r1.checkbox("Geçmiş seyir grafiği ve rejim değişimleri", value=True)
    inc_play = r1.checkbox("Rejim oyun kitabı", value=True)
    inc_theme = r2.checkbox(
        f"Tema takibi{'' if have_theme else '  (tarama yok)'}",
        value=have_theme, disabled=not have_theme)
    inc_etf = r2.checkbox(
        f"ETF radarı{'' if have_etf else '  (tarama yok)'}",
        value=have_etf, disabled=not have_etf)
    inc_hold = r2.checkbox(
        (f"ETF içi röntgen — {st.session_state['rep_hold']['etf']}"
         if have_hold else "ETF içi röntgen  (tarama yok)"),
        value=have_hold, disabled=not have_hold)
    inc_swing = r3.checkbox(
        f"Swing adayları{'' if have_swing else '  (tarama yok)'}",
        value=have_swing, disabled=not have_swing)
    inc_news = r3.checkbox(
        f"Haber akışı{'' if have_news else '  (yüklenmedi)'}",
        value=have_news, disabled=not have_news)
    delta_p = r3.selectbox("Karşılaştırma dönemi", list(mac.PERIOD_BARS),
                           index=1, key="rep_delta")

    baslik = st.text_input("Rapor başlığı", value="AETHER APEX")

    if st.button("📄 PDF raporu oluştur", type="primary", width="stretch"):
        with st.spinner("Rapor hazırlanıyor — grafikler çiziliyor…"):
            try:
                BC = SC = BH = None
                SHIFTS: list = []
                if inc_macro or inc_hist:
                    BC, SC, BH, SHIFTS = macro_history(
                        st.session_state.nonce_macro)
                theme_tbl, theme_per = (st.session_state["rep_theme"]
                                        if (inc_theme and have_theme)
                                        else (None, ""))
                pdf_bytes = rep.build_report(
                    M,
                    battery_changes=BC if inc_macro else None,
                    score_changes=SC if inc_macro else None,
                    battery_history=BH if inc_hist else None,
                    regime_shifts=SHIFTS if inc_hist else [],
                    drivers=pb.drivers_for(M.regime) if inc_play else (),
                    theme_table=theme_tbl, theme_period=theme_per,
                    etf_table=(st.session_state["rep_etf"]
                               if (inc_etf and have_etf) else None),
                    holdings=(st.session_state["rep_hold"]
                              if (inc_hold and have_hold) else None),
                    swing=(st.session_state["rep_swing"]
                           if (inc_swing and have_swing) else None),
                    news=(st.session_state["rep_news"]
                          if (inc_news and have_news) else None),
                    delta_period=delta_p, baslik=baslik or "AETHER APEX")
                st.session_state["rep_pdf"] = pdf_bytes
                st.session_state["rep_pdf_name"] = rep.report_filename()
            except Exception as exc:                      # pragma: no cover
                st.session_state.pop("rep_pdf", None)
                st.error(f"Rapor üretilemedi: {exc}")
                logging.exception("PDF hatası")

    if st.session_state.get("rep_pdf"):
        boyut = len(st.session_state["rep_pdf"]) / 1024
        st.success(f"Rapor hazır — {boyut:.0f} KB. Aşağıdaki butonla indirin.")
        st.download_button(
            "⬇️ PDF'i indir", data=st.session_state["rep_pdf"],
            file_name=st.session_state.get("rep_pdf_name", "aether_apex.pdf"),
            mime="application/pdf", width="stretch", type="primary")
        if not rep.ensure_fonts():
            st.warning(
                "Sunucuda Türkçe karakter içeren bir TTF font bulunamadı; "
                "raporda harfler ASCII'ye çevrildi (ş→s, ğ→g). "
                "`requirements.txt` içinde `matplotlib` olduğundan emin olun — "
                "DejaVu fontu onunla birlikte gelir.", icon="ℹ️")

    with st.expander("Raporda ne var, nasıl okunur?"):
        st.markdown("""
| Bölüm | İçerik |
|---|---|
| **Kapak** | Tespit edilen rejim, bileşik risk skoru, VIX, rejim kapısı, OPEX/FOMC takvimi |
| **1. Makro göstergeler** | Tüm canlı okumalar ve yorumları; alt skorların 1 gün/1 hafta/1 ay değişimi (grafik + tablo) |
| **2. Sermaye akış eğilimi** | Varlık sınıfı bataryası, seçilen döneme göre farkı, 60 günlük seyir grafiği ve rejim değişim anları |
| **3. Oyun kitabı** | Aktif rejimi besleyen sürücüler; her biri için lehte/aleyhte ETF ve hisse listesi, işlem notu |
| **4. Tema takibi** | Momentum × ivme haritası ve çeyrek sınıflandırmalı tema tablosu |
| **5. ETF radarı** | Taranan ETF'lerin sinyal tablosu |
| **6. ETF içi röntgen** | Seçili ETF'in bileşen bazında göreli gücü, ayrışma haritası ve her hissenin teknik durumu |
| **7. Swing adayları** | Skor, karakter, stop/hedef, R katsayıları ve engel gerekçeleri |
| **8. Haber akışı** | Rejim sürücüsüne bağlanmış başlıklar |

Rapor açık zeminlidir; ekrandaki koyu tema baskıda mürekkep yakar ve telefonda
okunmaz. Grafikler PDF içine gömülü PNG olarak girer, metin ise gerçek metindir —
arama yapılabilir ve kopyalanabilir.
""")