# apex/playbook.py — AETHER APEX
from __future__ import annotations


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
