# AETHER APEX

Makro, rotasyon, tema ve sinyal tarayıcı. Streamlit arayüzü + her akşam çalışan hesap motoru.

## Yapı

```
davinci_terminal.py      Streamlit arayüzü (yalnızca ekran) — Streamlit Cloud bunu çalıştırır
apex/                    Hesap kodu (Streamlit'siz, her yerden kullanılabilir)
  indicators.py          Temel göstergeler (EMA, RSI, ATR …)
  engine.py              Pine sinyal motoru (Whale/Retail, Omni, Şahane, Konfluans)
  decision.py            Endeks kararı: S&P 500 · Nasdaq · Kripto (G/H/A)
  funnel.py              Karar Hunisi: risk, rotasyon, tema RRG, hisse ayrımı
  fundchart.py           Hisse Analizi: F/K geçmişi, medyan çarpan adil fiyatı
  valuation.py           P/S adil değer
  shortvol.py            FINRA short hacmi, short interest
  compute.py             Arayüz ile akşam motorunun ORTAK hesap fonksiyonları
  snapshot.py            Akşam sonuçlarını kaydet / oku
  …                      macro, data, universe, themes, holdings, screener, news, report …
tools/
  build_snapshot.py      Akşam hesabı: makro, rotasyon, karar, tema, tarama, bilanço
  finra_update.py        FINRA günlük short hacmi
  macro_update.py        Kripto endeksleri (TOTAL, TOTAL3, BTC.D …) ve Fed likiditesi
data/                    Akşam hesabının çıktıları (GitHub Actions yazar)
  snapshot/              *.pkl.gz + _status.json (hangi adım ne zaman, başarılı mı)
.github/workflows/       Her iş günü akşam hesabı
.streamlit/config.toml   Koyu tema
```

## Nasıl çalışır

1. **Akşam hesabı** (GitHub Actions, iş günleri ~02:30 TSİ): bütün veri ve sinyaller bir kez
   hesaplanır, `data/` klasörüne yazılır.
2. **Arayüz yalnızca okur**: uygulama açılınca her şey akşam hesabından gelir; Yahoo'ya gidilmez.
   Bir sekmedeki 🔄 düğmesi yalnızca o kısmı canlı hesaplar.
3. Başlıktaki **“Akşam hesabı”** satırı sonuçların ne zaman üretildiğini, açılır kutu da
   hangi adımın başarısız olduğunu gösterir.

## Bilgisayarda çalıştırma

- Windows: `calistir_windows.bat` dosyasına çift tıklayın.
- macOS / Linux: `./calistir_mac.sh`
- İnternet yokken de son indirilen akşam hesabıyla açılır.
- Hesabı kendi bilgisayarınızda yapmak için: `calistir_windows.bat hesapla` / `./calistir_mac.sh hesapla`

## İsteğe bağlı: Fed likiditesi için FRED anahtarı

FRED bazen GitHub sunucularını reddeder. Ücretsiz anahtar alıp
(fredaccount.stlouisfed.org → API Keys) repo **Settings → Secrets and variables → Actions →
New repository secret** ile `FRED_API_KEY` adıyla eklerseniz resmî API kullanılır.
