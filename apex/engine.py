# apex/engine.py — AETHER APEX
from __future__ import annotations

from apex.indicators import atr, barssince, cci, crossover, crossunder, ema, f_contrast, f_tanh, highest, leaky_reservoir, lowest, mfi, normal_cdf, percentile_lin, percentrank, ratcheting_atr_stop, roc, rsi, safe_bool, safe_last, sma, stdev, stoch, true_range, tsi, wma

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
