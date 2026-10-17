# apex/explain.py — ⓘ SADE ANLATIM: ekonomiden ve terimlerden anlamayan birinin anlayacağı dil
#
# Kurallara dayalı (yapay zekâ metni değil): teknik okumanın her paragrafı (reading.read) ve her
# karar etiketi için içindeki sayılar okunur, günlük dille anlatılır. Ayrıca ~50 terimlik 📖 Sözlük.
from __future__ import annotations

import re

# ------------------------------------------------------------------ 📖 sözlük
GLOSSARY: dict[str, str] = {
    "Trend": "Fiyatın genel yönü. Tepeler ve dipler yükseliyorsa yükseliş trendi, alçalıyorsa "
             "düşüş trendi. 'Trend dostundur': yönün tersine işlem yapmak daha risklidir.",
    "EMA (üssel ortalama)": "Son X günün ortalama fiyatı; yeni günlere daha çok ağırlık verir. "
                            "EMA21 ≈ son 1 ay, EMA50 ≈ 2,5 ay, EMA200 ≈ 1 yıl. Fiyat ortalamanın "
                            "üstündeyse alıcılar o dönemde kârda demektir.",
    "EMA şeridi / dizilimi": "Beş ortalamanın (10·21·50·100·200) sırası. Kısalar uzunların üstünde ve "
                             "hepsi yukarı bakıyorsa 'tam boğa': en sağlıklı yükseliş. Tersi 'tam ayı'.",
    "SMA200": "Son 200 işlem gününün basit ortalaması (≈1 yıl). Büyük fonların en çok izlediği çizgi; "
              "altına inmek 'uzun vadeli trend bozuldu' diye okunur.",
    "Altın kesişim / ölüm kesişimi": "EMA50'nin EMA200'ü yukarı kesmesi altın kesişim (olumlu), "
                                     "aşağı kesmesi ölüm kesişimi (olumsuz). Geç gelen sinyallerdir.",
    "Yelpaze": "Kısa ve uzun ortalamalar arasındaki açıklık. Açılıyorsa trend hızlanıyor, "
               "kapanıyorsa yaşlanıyor; çok darsa (sıkışma) büyük bir hareket yakındır.",
    "ATR (ortalama gerçek aralık)": "Bir hissenin bir günde (ya da haftada) ortalama ne kadar "
                                    "oynadığı. ATR %3 → günde ortalama %3 iniş-çıkış normaldir. Stop "
                                    "mesafesi ATR'nin altındaysa gürültüyle patlar.",
    "RSI": "0–100 arası 'hız göstergesi'. 70 üstü: fazla hızlı yükseldi (ısınmış), 30 altı: fazla "
           "hızlı düştü (soğumuş). 50 civarı nötr.",
    "Momentum": "Fiyatın son haftalardaki yükseliş hızı. Güçlü momentumlu hisseler bir süre daha "
                "güçlü kalma eğilimindedir (testlerimizde orta vadede en güçlü etkenlerden biri).",
    "Konfluans": "Birçok göstergenin aynı yönü göstermesi. DIRECTION + ise çoğunluk yukarı, − ise "
                 "aşağı diyor.",
    "Omni / Fusion": "Kendi Pine göstergelerimizden momentum ölçüleri. 50 üstü / pozitif = alıcı "
                     "lehine.",
    "Destek": "Fiyatın düşerken daha önce durup döndüğü seviye — orada alıcılar bekliyor. Kırılırsa "
              "düşüş hızlanabilir.",
    "Direnç": "Fiyatın yükselirken daha önce takılıp döndüğü seviye — orada satıcılar bekliyor. "
              "Aşılırsa yol açılır, eski direnç desteğe dönüşür.",
    "Fibonacci geri çekilme": "Bir yükselişin ne kadarının geri verildiğini ölçen oranlar: 0.382, "
                              "0.5, 0.618, 0.786. Fiyat sık sık bu oranlarda durup döner. "
                              "0.382–0.786 arası 'indirim bölgesi' gibi düşünülebilir.",
    "Fibonacci uzantı (1.272 / 1.618)": "Önceki tepe aşılınca hedef olarak kullanılan oranlar.",
    "Ana yapı bacağı": "Son büyük yükseliş ya da düşüş hareketi (dipten tepeye). Fibonacci "
                       "oranları bu bacak üzerinde ölçülür.",
    "Elliott dalgaları": "Piyasanın 5 dalgada ilerleyip 3 dalgada (A-B-C) düzeltme yaptığı teorisi. "
                         "② bitti → ③ bekleniyor demek: en güçlü dalga başlayabilir. Yorumdur, "
                         "kesin değildir.",
    "Boşluk (Gap / FVG)": "Fiyatın çok hızlı geçtiği, arada işlem olmayan bölge. Fiyat çoğu zaman "
                          "bu boşluğu doldurmaya geri döner.",
    "Hacim": "O gün el değiştiren hisse sayısı. Yüksek hacimli hareket 'gerçek', hacimsiz hareket "
             "'zayıf' kabul edilir.",
    "VSA (hacim-fiyat analizi)": "Hacimle fiyat hareketinin uyumuna bakar. Çok hacim ama az hareket "
                                 "→ biri karşı tarafı emiyor. Hacimsiz düşüş → satıcı yok.",
    "CVD / akış dengesi": "Alıcı ve satıcı hacminin farkı. Yükseliyorsa alıcılar daha istekli.",
    "Balina (whale)": "Fiyatı tek başına hareket ettirebilecek kadar büyük yatırımcılar (fonlar, "
                      "kurumlar). Amacımız onların izini sürmek.",
    "Afterburner": "Balinaların sert ve hacimli girişi. 5 yılda 730 hissede sonraki 20 günde "
                   "piyasayı ortalama +1.65 puan geçti — kanıtlı en güçlü olayımız.",
    "Dip süpürmesi": "Fiyatın önceki dibin altına kısa süre sarkıp stopları patlatması, sonra "
                     "toparlanması. Büyükler ucuz mal toplar.",
    "Stealth IN": "Fiyat pek oynamadan gelen sessiz, hacimli alım — gizli toplama.",
    "Dağıtım": "Büyüklerin yükselişlere mal satması. Fiyat yükselse bile tehlike işaretidir.",
    "Whale tükenme": "Büyük alıcıların gücünün bittiği an; sonrasında fiyat zayıf kalma eğiliminde.",
    "Wyckoff (Spring / Upthrust)": "Spring: dibin altına sahte kırılım, sonra yukarı (alım fırsatı). "
                                   "Upthrust: tepenin üstüne sahte kırılım, sonra aşağı (satış).",
    "Likidite havuzu": "Çok sayıda stop emrinin biriktiği seviye. Büyükler fiyatı oraya iterek "
                       "emirleri tetikleyip mal toplar/satar.",
    "Q-dry": "Satıcıların kuruması: düşüşte hacim çok azalır, ardından alım gelir — dip işareti.",
    "Sıkışma": "Fiyat ve ortalamalar dar bir alana sıkıştı. Sıkışma uzun sürmez; ardından sert "
               "bir kırılım gelir, yönü kırılım belirler.",
    "Örgü": "Büyük yükselişlerden önceki 30 günde göstergelerin oluşturduğu ortak desen. Bugünkü "
            "desen ona benziyorsa büyük hareket olasılığı artar (kesinlik değil).",
    "Ölü kedi sıçraması": "Düşen bir hissede ani, kısa ömürlü sıçrama. Çoğu zaman düşüş devam eder.",
    "Stop (zarar kes)": "Önceden belirlenen çıkış fiyatı. Fiyat buraya inerse pozisyon kapatılır; "
                        "küçük zarar büyük zarara dönüşmez.",
    "Kâr al (TP1/TP2/TP3)": "Hedef fiyatlar. TP1'de pozisyonun bir kısmı satılır ve stop giriş "
                            "fiyatına çekilir; böylece işlem 'risksiz' hâle gelir.",
    "R:R (risk/ödül)": "Riske edilen her 1 birime karşı hedeflenen kazanç. 2R = 1 kaybetme "
                       "riskine karşı 2 kazanma hedefi.",
    "Limit emir": "'Fiyat şuraya gelirse al' emri. Fiyat gelmezse işlem olmaz.",
    "Fair value (adil değer)": "Analistlerin 12 aylık ortalama hedef fiyatı. Bilgi amaçlıdır; "
                               "analistler sık yanılır.",
    "F/K (fiyat/kazanç)": "Hisse fiyatının hisse başına yıllık kâra oranı: kaç yıllık kârı "
                          "ödüyorsun. İleri F/K gelecek yılın tahmini kârıyla hesaplanır.",
    "F/K sıkışması": "İleri F/K'nın geçmiş F/K'dan düşük olması: kâr fiyattan hızlı büyüyor — "
                     "fiyat yükselse bile hisse 'ucuzluyor'.",
    "Piyasa değeri": "Şirketin borsadaki toplam değeri (fiyat × hisse sayısı).",
    "ROE": "Özkaynak kârlılığı: şirketin ortakların parasıyla yılda yüzde kaç kâr ettiği. "
           "%10 üstü iyi kabul edilir.",
    "Serbest nakit akışı (FCF)": "Şirketin yatırımlar sonrası elinde kalan gerçek nakit. Pozitif "
                                 "olması sağlıklı işaret.",
    "Borç/özkaynak": "Şirketin borcunun ortak sermayesine oranı. 1.5'in altı makul.",
    "Tema / sektör rotasyonu": "Paranın bir sektörden diğerine geçmesi. Para giren temalardaki "
                               "hisseler rüzgârı arkasına alır.",
    "RRG": "Temaların endekse göre gücünü ve ivmesini gösteren dört bölgeli grafik: Geride → "
           "İyileşiyor → Lider → Zayıflıyor.",
    "Piyasa genişliği": "Endeksteki hisselerin kaçının yükseldiği. Endeks yükselirken çoğu hisse "
                        "düşüyorsa yükseliş 'dar' ve kırılgandır.",
    "Boğa / ayı piyasası": "Boğa: fiyatlar genel olarak yükseliyor. Ayı: zirveden %20+ düşüş, "
                           "genel düşüş.",
    "VIX (korku endeksi)": "S&P 500 opsiyonlarından hesaplanan, önümüzdeki 30 gün için beklenen "
                           "yıllık oynaklık. 20 altı sakin, 30 üstü korku.",
    "Beklenen hareket (VIX/16)": "VIX'i 16'ya bölünce S&P 500'ün bir günde ±% kaç oynamasının "
                                 "'normal' olduğu çıkar (yılda ~252 işlem günü, √252 ≈ 16). "
                                 "VIX 20 → günlük ±%1.25. Günlerin ~%68'i bu aralıkta kalır.",
    "σ (sigma)": "Normal oynaklık birimi. 1σ'lık hareket sıradan, 2σ nadir (≈ ayda bir), 3σ çok "
                 "nadir (≈ yılda bir) kabul edilir.",
    "VIX3M − VIX (vade yapısı)": "3 aylık korku ile 1 aylık korkunun farkı. Normalde pozitiftir "
                                 "(uzak gelecek daha belirsiz). Negatife dönerse (tersine yapı) "
                                 "piyasa ŞU AN panikte demektir; tersinme bitince çoğu zaman dip "
                                 "geride kalmıştır.",
    "Yüzdelik (persentil)": "Bugünkü değerin geçmiş 1 yıl içinde nerede olduğu. %10 → son bir yılın "
                            "en düşük %10'unda.",
    "Taban (karşılaştırma)": "Test sonuçlarında kıyas: aynı dönemde bütün hisselerin ortalaması. "
                             "Bir sinyal tabanı geçmiyorsa işe yaramıyor demektir.",
    "Test dışı (görülmemiş veri)": "Kuralları yazarken bakmadığımız dönem. Asıl güvenilir sonuç "
                                   "buradan gelir; kural yazılan dönemdeki başarı yanıltıcı olabilir.",
    "t değeri": "Sonucun şansla açıklanıp açıklanamayacağı. |t| ≥ 2 genelde 'muhtemelen gerçek' "
                "kabul edilir.",
}


def glossary_md(q: str = "") -> str:
    q = (q or "").strip().lower()
    items = [(k, v) for k, v in GLOSSARY.items() if not q or q in k.lower() or q in v.lower()]
    if not items:
        return "_Eşleşen terim yok._"
    return "\n\n".join(f"**{k}** — {v}" for k, v in sorted(items, key=lambda kv: kv[0].lower()))


# ------------------------------------------------------------------ karar etiketleri
KARAR: dict[str, str] = {
    "✅ GÜÇ + BALİNA": "Hisse güçlü yükseliş trendinde ve son günlerde büyük yatırımcı alımı "
                      "görüldü. Testlerde en tutarlı olumlu bileşim. Fiyattan, sıkı stopla alınabilir.",
    "✅ AFTERBURNER": "Son 5 günde balinalar sert ve hacimli girdi, hisse 1 yıllık ortalamasının "
                     "üstünde. 24 aylık testte görmediğimiz veride bile tabanı her vadede geçti.",
    "🟡 FİB BÖLGESİ + TEYİT": "Fiyat son yükselişin 'indirim bölgesinde' ve bugün dönüş mumu geldi. "
                             "Umut verici ama testte tutarsız — küçük pozisyon ya da bekle.",
    "🟡 FİB BÖLGESİ": "Fiyat indirim bölgesinde ama henüz dönüş işareti yok. Yeşil, hacimli ve dünkü "
                     "tepenin üstünde kapanan bir mum beklenir.",
    "🟡 LİMİT EMİR": "Fiyat henüz geri çekilmedi. 'Şuraya gelirse al' (limit) emri giriş bölgesine "
                    "konur; gelmezse işlem olmaz.",
    "⏳ DÜŞÜŞ BACAĞINDA TEPKİ": "Ana hareket düşüş; şu anki yükseliş bir tepki olabilir. Testte bu "
                               "durumda alım kaybettirdi → bekle.",
    "⏳ TEYİT BEKLE": "Fiyat yükselişin neredeyse tamamını geri verdi; önce toparlanma görülmeli.",
    "🔄 DİP TEYİDİ BEKLE": "Düşüş sürüyor, dip henüz belli değil.",
    "🟢 MOMENTUM": "Kısa vadede güçlü yükselen ve balina/örgü desteği olan hisse. Fiyattan, sıkı "
                  "stopla; hedef riskin 2–3 katı.",
    "💎 DEĞER DÖNÜŞÜ": "Endeks/tema dönüyor, fiyat analist değerinin çok altında ve şirket sağlam. "
                      "Orta-uzun vade için parça parça (kademeli) alınabilir.",
    "🌊 GENİŞLEME ADAYI": "Piyasa yükselişi birkaç devden geniş hisse grubuna yayılmaya başladı; bu "
                         "hisse zirvenin çok altında ama sağlam — yakalama potansiyeli.",
    "⛔ İZLE": "Ölü kedi, kırık bıçak ya da bozulmuş yapı. Taban oluşmadan alınmaz.",
    "⛔ UZAK DUR": "Kurallara göre şu an alım için uygun değil.",
    "✅ AL": "Kurallara göre alım bölgesinde.",
    "🟡 LİMİT": "Giriş bölgesine limit emir konabilir; fiyat gelmezse işlem olmaz.",
}
SUFFIX = {"endeks zayıf": "Endeks (piyasanın geneli) zayıf olduğu için ✅ yerine 🟡: testte endeks "
                          "zayıfken verilen alımların %75'i stop oldu.",
          "bilanço yakın": "Birkaç gün içinde bilanço açıklanacak; sonuç fiyatı sert oynatabilir."}


def karar(label: str) -> str:
    if not label:
        return ""
    lab = str(label)
    core = lab[1:].strip()                    # 🟡'ye çekilmiş ✅'ler de tanınsın
    best = ""
    for k, v in KARAR.items():
        if (lab.startswith(k) or core.startswith(k[1:].strip())) and len(k) > len(best):
            best = k
    out = KARAR.get(best, "")
    for k, v in SUFFIX.items():
        if k in lab:
            out += " " + v
    return out


SIG: dict[str, str] = {
    "🚀": "momentum: son 1 ve 3 ayda evrenin en hızlı yükselen %20'sinde",
    "🐋": "balina girişi: son 5 günde büyük yatırımcı alımı (Afterburner / dip süpürmesi / sessiz alım)",
    "🧬": "örgü: büyük yükselişlerden önceki desene benziyor",
    "📈": "endeks ya da tema bir süre düştükten sonra yukarı dönüyor",
    "💰": "fiyat analist hedefinin %35'ten fazla altında (ucuz)",
    "🏛️": "şirket temelleri sağlam (kârlı, büyüyen, nakit üreten, borcu makul)",
    "🧮": "ucuzlayan lider: fiyat yükselirken kâr daha hızlı büyüyor (F/K düşüyor)",
    "🌊": "genişleme adayı: zirvenin çok altında ama sağlam; piyasa genişlerse yakalar",
    "🔻": "200 günlük ortalamaya dokunuş: fiyat uzun süre sonra 1 yıllık ortalamaya indi",
}


def signals(*texts: str) -> list[str]:
    out = []
    for t in texts:
        for k, v in SIG.items():
            if t and k in str(t) and f"{k} {v}" not in out:
                out.append(f"{k} {v}")
    return out


# ------------------------------------------------------------------ teknik okuma paragrafları
def _num(pat: str, s: str, default=None):
    m = re.search(pat, s)
    if not m:
        return default
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return default


CIRC = {"①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5}
WAVE_NEXT = {1: "2. dalga (geri çekilme) bekleniyor",
             2: "3. dalga bekleniyor — genelde en uzun ve en güçlü dalga",
             3: "4. dalga (yatay/geri çekilme) bekleniyor",
             4: "5. ve son itiş dalgası bekleniyor — sonrasında düzeltme gelir",
             5: "5 dalga tamamlandı — A-B-C düzeltmesi beklenir"}


def heading(md: str) -> str:
    m = re.match(r"\s*\*\*([^*]+?)\*\*", md)
    return m.group(1).strip().rstrip(".") if m else ""


def paragraph(md: str) -> str:
    """Teknik okuma paragrafının sade anlatımı (boş dönerse ⓘ gösterilmez)."""
    s = md or ""
    h = heading(s).lower()
    out: list[str] = []
    if "hüküm" in h:
        sc = _num(r"skor\s*([+-]?\d+)", s)
        out.append("Bu, bütün göstergelerin ortak oyu. −100 ile +100 arası; +12 üstü alıcı lehine, "
                   "−12 altı satıcı lehine, arası kararsız.")
        if sc is not None:
            out.append(f"Şu an **{sc:+.0f}** → " + (
                "göstergelerin büyük çoğunluğu yükselişten yana." if sc >= 35 else
                "alıcılar hafif önde." if sc >= 12 else
                "göstergeler çelişiyor; yön belli olana kadar acele etmeyin." if sc > -12 else
                "satıcılar önde." if sc > -35 else "göstergelerin çoğu düşüşten yana."))
    elif h.startswith("trend ve mum"):
        out.append("**Trend**, fiyatın genel yönü. **Son mum**, son günün (haftanın) karakteri: "
                   "hacimli güçlü mum alıcıların istekli olduğunu, 'dağıtım izi' büyüklerin satış "
                   "yaptığını gösterebilir.")
        if "DAĞITIM" in s:
            out.append("Son mumda **dağıtım izi** var: yüksek hacme rağmen zayıf kapanış — büyükler "
                       "yükselişi satış fırsatı olarak kullanıyor olabilir.")
        if "hacimsiz geri çekilme" in s:
            out.append("**Hacimsiz geri çekilme** iyi bir işaret: düşüşe katılan satıcı az.")
        if "hacimli yükseliş" in s:
            out.append("**Hacimli yükseliş**: yükselişe gerçek para eşlik ediyor.")
    elif h.startswith("ema dizilimi"):
        out.append("Beş ortalama çizginin (1 haftalık → 1 yıllık) sırasına bakıyoruz. En sağlıklı "
                   "durum: fiyat en üstte, sonra kısa ortalamalar, en altta uzun ortalama ve hepsi "
                   "yukarı bakıyor.")
        for k, v in (("Tam boğa", "Şu an **tam boğa**: her şey yükseliş sırasında — trend güçlü."),
                     ("sağlıklı geri", "Uzun vade yukarı, fiyat kısa bir **mola** veriyor — "
                                       "trend içinde fırsat arayan bölge."),
                     ("derin düzeltme", "Uzun vade hâlâ yukarı ama düzeltme derinleşti — dikkat."),
                     ("kırılıyor", "Fiyat 1 yıllık ortalamanın altına indi — **trend bozuluyor**."),
                     ("Yorulma", "Yükseliş **yoruldu**: kısa ortalamalar aşağı döndü."),
                     ("Erken dönüş", "Düşüşten sonra **ilk toparlanma**; teyit için erken."),
                     ("Ayı tepkisi", "Ana trend aşağı; yükseliş bir **tepki** olabilir."),
                     ("Tam ayı", "Şu an **tam ayı**: her şey düşüş sırasında."),
                     ("Sıkışma", "Ortalamalar iç içe (**sıkışma**): yakında sert hareket gelir, yönü "
                                 "kırılım belirler."),
                     ("Karışık", "Ortalamalar **karışık**: net bir trend yok.")):
            if k in s:
                out.append(v)
                break
        uz = _num(r"EMA21'den\s*([+-]?\d+(?:\.\d+)?)\s*ATR", s)
        if uz is not None:
            out.append(f"Fiyat 1 aylık ortalamadan **{abs(uz):.1f} ATR** (ortalama bar oynaklığı) kadar "
                       f"{'yukarıda' if uz > 0 else 'aşağıda'}" + (
                           " — çok koşmuş, buradan almak kovalamak olur." if uz > 3 else
                           " — çok satılmış, kısa tepki gelebilir." if uz < -3 else
                           " — normal mesafe."))
        if "ölüm kesişimi" in s or "altın kesişim" in s:
            out.append("Altın kesişim (50 günlük 200 günlüğü yukarı keser) olumlu, ölüm kesişimi "
                       "olumsuz kabul edilir; ikisi de geç gelen sinyallerdir.")
    elif h.startswith("momentum"):
        rsi = _num(r"RSI\s*(\d+(?:\.\d+)?)", s)
        out.append("Momentum = fiyatın hızı. Araba gibi düşünün: hız artıyorsa yükseliş güçleniyor, "
                   "azalıyorsa yavaşlıyor.")
        if rsi is not None:
            out.append(f"**RSI {rsi:.0f}** (0–100 arası hız ölçer): " + (
                "70'in üstü — fazla ısınmış, kısa vadede soluklanma gelebilir." if rsi >= 70 else
                "güçlü ama aşırı değil." if rsi >= 55 else
                "nötr bölge — ne hızlı ne yavaş." if rsi >= 45 else
                "zayıf; satıcılar biraz önde." if rsi > 30 else
                "30'un altı — fazla soğumuş, tepki gelebilir."))
        if "Konfluans DIRECTION" in s:
            dr = _num(r"DIRECTION\s*([+-]?\d+)", s)
            if dr is not None:
                out.append(f"**Konfluans {dr:+.0f}**: göstergelerin oylaması; artı = çoğunluk yukarı, "
                           "eksi = aşağı.")
        if "tükenme" in s:
            out.append("⚠️ **Tükenme**: yükseliş nefessiz kalıyor; yeni alım için kötü zaman.")
        if "🔒" in s:
            out.append("🔒 **Sıkışma**: oynaklık çok düştü; sert bir hareket yakın.")
    elif h.startswith("yapı"):
        r = _num(r"\(oran\s*([+-]?\d+(?:\.\d+)?)\)", s)
        if "yükseliş;" in s and r is not None:
            out.append(f"Son büyük yükselişin **%{r * 100:.0f}'i geri verilmiş**. " + (
                "Henüz geri çekilme yok denecek kadar az." if r < 0.236 else
                "Sığ geri çekilme." if r < 0.382 else
                "Bu oran 0.382–0.786 arasında: alıcıların sık devreye girdiği 'indirim bölgesi'." if
                r <= 0.786 else "Çok derin — yükseliş yapısı zayıfladı."))
        elif "düşüş;" in s and r is not None:
            out.append(f"Son büyük düşüşün **%{r * 100:.0f}'i geri alınmış**. " + (
                "Dibe yakın, henüz toparlanma yok." if r < 0.236 else
                "Tepki yükselişi sürüyor ama ana yön hâlâ aşağı." if r <= 1 else
                "Düşüşün tamamı geri alındı — yeni yükseliş başlıyor olabilir."))
        m = re.search(r"Elliott:\s*(BOĞA|AYI)?\s*·?\s*([①②③④⑤ABC])\s*bitti", s)
        if m:
            yon = {"BOĞA": "yükseliş", "AYI": "düşüş"}.get(m.group(1) or "", "")
            w = m.group(2)
            if w in CIRC:
                out.append(f"**Elliott**: piyasa 5 dalgada ilerler, 3 dalgada düzeltir. Bu "
                           f"{yon} sayımında {CIRC[w]}. dalga bitti → {WAVE_NEXT[CIRC[w]]}.")
            else:
                out.append(f"**Elliott**: düzeltme dalgası {w} bitti.")
        elif "Elliott" in s and "bozuldu" in s:
            out.append("**Elliott** sayımı bozuldu: dalga senaryosu şu an güvenilir değil.")
        if "kâr alma bölgesi" in s:
            out.append("Fiyat dalga hedefine ulaştı — kâr almayı düşünme yeri.")
        if "Boşluklar" in s:
            out.append("**Boşluk**: fiyatın hızla geçtiği bölge; fiyat sık sık bu boşlukları "
                       "doldurmaya geri döner. Alttaki boşluk mıknatıs/destek, üstteki hedef olur.")
    elif h.startswith("akıllı para"):
        out.append("Bu bölüm 'büyük oyuncular ne yapıyor?' sorusunu cevaplar: hacmin yönü, kurumsal "
                   "alım-satım izleri, stopların biriktiği seviyeler.")
        if "TEYİTLİ YÜKSELİŞ" in s:
            out.append("**Teyitli yükseliş**: fiyat artarken alıcı hacmi de artıyor — sağlıklı.")
        if "TEYİTLİ DÜŞÜŞ" in s:
            out.append("**Teyitli düşüş**: fiyat düşerken satıcı hacmi artıyor — baskı gerçek.")
        w = _num(r"WHALE\s*(\d+)", s)
        if w is not None:
            out.append(f"**WHALE {w:.0f}**: büyük oyuncu aktivitesi göstergesi (0–100). Tek başına "
                       "yön göstermez; testlerimizde seviyesi değil ani olaylar (Afterburner, dip "
                       "süpürmesi) işe yaradı.")
        if "SPRING" in s:
            out.append("**Spring** bölgesi: dibin altına sahte kırılım yapılıp geri dönülen yer — "
                       "alıcıların bölgesi.")
        if "UPTHRUST" in s:
            out.append("**Upthrust** bölgesi: tepenin üstüne sahte kırılım yapılıp düşülen yer — "
                       "satıcıların bölgesi.")
        if "havuz" in s:
            out.append("**Likidite havuzu**: stop emirlerinin biriktiği seviye; fiyat çoğu zaman "
                       "buraya uğrar.")
        if "W-POC" in s:
            out.append("**W-POC**: son dönemde en çok işlem gören fiyat — mıknatıs gibi çalışır.")
    elif h.startswith("seviyeler"):
        atr = _num(r"\(ATR\)\s*[\d.,]+\s*\((\d+(?:\.\d+)?)%\)", s)
        out.append("**Direnç** = yükselirken satıcıların beklediği fiyat, **destek** = düşerken "
                   "alıcıların beklediği fiyat. Parantezdeki yüzde, bugünkü fiyata uzaklık.")
        if atr is not None:
            out.append(f"**ATR %{atr:.1f}**: bu varlık günde (ya da bu zaman diliminde barda) "
                       f"ortalama ±%{atr:.1f} oynar. Bundan dar bir stop sıradan bir dalgalanmayla "
                       "patlar.")
    elif "senaryo" in s.lower():
        out.append("İki ihtimal: destek korunursa yukarıdaki dirence doğru yükseliş (boğa), destek "
                   "kırılırsa bir alttaki desteğe doğru düşüş (ayı). Kapanış önemlidir — gün içi "
                   "sarkmalar değil, kapanış fiyatı belirler.")
    return " ".join(out)


def split(md: str) -> list[str]:
    """reading.read çıktısını paragraflara böler (senaryo listesi tek parça kalır)."""
    return [p for p in (md or "").split("\n\n") if p.strip()]
