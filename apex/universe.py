# apex/universe.py — AETHER APEX
from __future__ import annotations


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
