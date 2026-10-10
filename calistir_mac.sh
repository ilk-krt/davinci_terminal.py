#!/usr/bin/env bash
# AETHER APEX — bilgisayarda çalıştırma (macOS / Linux)
# İlk seferde sanal ortam kurar; sonraki seferlerde doğrudan açar.
#   ./calistir_mac.sh            → son akşam hesabıyla aç
#   ./calistir_mac.sh hesapla    → önce hesabı bu bilgisayarda yap, sonra aç
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "Sanal ortam kuruluyor…"
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
else
  source .venv/bin/activate
fi
echo "GitHub'daki son akşam hesabı alınıyor (internet yoksa atlanır)…"
git pull --quiet 2>/dev/null || true
if [ "$1" = "hesapla" ]; then
  python tools/build_snapshot.py
fi
streamlit run davinci_terminal.py
