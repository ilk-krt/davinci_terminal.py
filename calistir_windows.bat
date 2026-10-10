@echo off
REM AETHER APEX — bilgisayarda calistirma (Windows)
REM Ilk seferde sanal ortam kurar; sonraki seferlerde dogrudan acar.
cd /d "%~dp0"
if not exist .venv (
  echo Sanal ortam kuruluyor...
  python -m venv .venv
  call .venv\Scripts\activate.bat
  python -m pip install --upgrade pip
  pip install -r requirements.txt
) else (
  call .venv\Scripts\activate.bat
)
echo GitHub'daki son aksam hesabi aliniyor (internet yoksa atlanir)...
git pull --quiet 2>nul
if "%1"=="hesapla" (
  echo Aksam hesabi bu bilgisayarda yapiliyor...
  python tools\build_snapshot.py
)
streamlit run davinci_terminal.py
