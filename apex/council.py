# apex/council.py — "SİNYAL KONSEYİ": bütün Pine/engine sinyallerini oylatan,
# her hissede ağırlıklarını GEÇMİŞTEN ÖĞRENEN (walk-forward) skor motoru.
#
# 1) Özellikler (hepsi nedensel — her bar yalnızca kendisine kadar olan veriyi görür):
#    1 saatlik  : APEX core (whale, pro/retail, yıldız, stealth, gerçek toplama/dağıtım,
#                 sıkışma patlaması), OMNI (HUD boğa/ayı, Fusion, Synergy, absorpsiyon,
#                 elmaslar, likidite süpürmesi, tükenme), Şahane (efor çizgisi kırılımı,
#                 Ultimate kanal, SMC al/sat), Konfluans (MAGNITUDE/DIRECTION, giriş /
#                 genişleme / risk durumları), V710 (Q-dry teyidi, ODI, efor kırılımı,
#                 efor gücü), VSA & Delta (soğurma dönüşü, baskınlık dönüşü, akış dengesi,
#                 Wyckoff soğurması), Mum Gücü (v666 sentez).
#    günlük     : aynı motorların günlük versiyonundan özet bağlam (bir gün kaydırılmış).
# 2) Öğrenme : her ~5 günde bir, YALNIZCA geçmiş barlarla her özelliğin ileri getiriyle
#    (varsayılan 14 saat ≈ 2 gün) korelasyonu ölçülür; t-istatistiği ağırlık olur
#    (|t| küçükse ağırlık 0 → gürültü sinyaller kendiliğinden susar).
# 3) Skor  : Σ ağırlık × standartlaştırılmış özellik. Giriş/çıkış eşikleri de aynı
#    eğitim penceresindeki skor dağılımının yüzdelikleridir (her hisseye uyarlanır).
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from apex import engine as eng
from apex import pine_fib as pfib
from apex import pine_v710 as pv710
from apex import pine_vsa as pvsa
from apex.indicators import ema

_log = logging.getLogger(__name__)

LABELS = {
    "c_whale_up": "Whale gücü artıyor", "c_pro_gt_ret": "PRO > RETAIL",
    "c_star": "APEX yıldız", "c_st_in": "Stealth giriş", "c_st_out": "Stealth çıkış",
    "c_acc": "Gerçek toplama", "c_dist": "Gerçek dağıtım", "c_cross_up": "Whale kesişim ↑",
    "c_cross_dn": "Whale kesişim ↓", "c_sqz_fire": "Sıkışma patlaması",
    "c_exh": "Tükenme", "o_hud_net": "OMNI HUD net (−6…+6)", "o_f_hist": "Fusion histogram",
    "o_s_speed": "Synergy hızı", "o_ab_bull": "Absorpsiyon (boğa)", "o_ab_bear": "Absorpsiyon (ayı)",
    "o_dia_buy": "Elmas al", "o_dia_sell": "Elmas sat", "o_sweep_low": "Dip süpürmesi",
    "o_sweep_high": "Tepe süpürmesi", "o_ew_bull": "Fusion kesişim ↑", "o_ew_bear": "Fusion kesişim ↓",
    "s_eff_pos": "Fiyat efor çizgisinin üstünde", "s_ult_up": "Ultimate kanal kırılımı",
    "s_smc_buy": "SMC al", "s_smc_sell": "SMC sat", "k_dir": "Konfluans DIRECTION",
    "k_mag": "Konfluans MAGNITUDE", "k_entry": "Konfluans giriş bölgesi",
    "k_expand": "Konfluans genişleme", "k_risk": "Konfluans risk", "v_qconf": "Q-dry teyidi",
    "v_odi_bull": "ODI boğa", "v_odi_bear": "ODI ayı", "v_eff_up": "Efor kırılımı ↑",
    "v_eff_dn": "Efor kırılımı ↓", "v_eff_score": "Efor gücü (0–8)", "a_rev_up": "VSA soğurma dönüşü ↑",
    "a_rev_dn": "VSA dağıtım dönüşü ↓", "a_flip_up": "Baskınlık alıcıya", "a_flip_dn": "Baskınlık satıcıya",
    "a_flow": "Akış dengesi (delta)", "a_abs_dn": "Düşüşte soğurma (talep)",
    "a_abs_up": "Yükselişte soğurma (arz)", "m_syn": "Mum Gücü sentezi",
    "d_dir": "GÜNLÜK Konfluans DIRECTION", "d_hud": "GÜNLÜK OMNI HUD", "d_regime": "GÜNLÜK EMA200 üstü",
    "d_whale": "GÜNLÜK whale gücü", "d_eff": "GÜNLÜK efor çizgisi üstü", "d_trend50": "GÜNLÜK EMA50 üstü",
}
EVENT_COLS = {"c_star", "c_st_in", "c_st_out", "c_acc", "c_dist", "c_cross_up", "c_cross_dn",
              "c_sqz_fire", "c_exh", "o_ab_bull", "o_ab_bear", "o_dia_buy", "o_dia_sell",
              "o_sweep_low", "o_sweep_high", "o_ew_bull", "o_ew_bear", "s_ult_up", "s_smc_buy",
              "s_smc_sell", "v_qconf", "v_odi_bull", "v_odi_bear", "v_eff_up", "v_eff_dn",
              "a_rev_up", "a_rev_dn", "a_flip_up", "a_flip_dn", "a_abs_dn", "a_abs_up"}
SYN_VAL = {4: 1.0, 2: 0.5, 3: -0.5, -2: 0.3, -3: -0.3, -4: -1.0}


def _b(x, idx) -> pd.Series:
    a = np.asarray(x)
    if len(a) != len(idx):
        return pd.Series(0.0, index=idx)
    return pd.Series(np.nan_to_num(a.astype(float)), index=idx)


def engine_features(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Bir OHLCV serisi için bütün motorlardan bar bar özellikler."""
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float).copy()
    df["Volume"] = df["Volume"].fillna(0.0)
    idx = df.index
    F = pd.DataFrame(index=idx)
    p = dict(eng.PARAMS)
    try:
        core = eng.apex_core(df, p)
        omni = eng.apex_omni(df, p)
        sah = eng.sahane_layer(df, core["w_pwr"], p)
        conf = eng.confluence(df, core, omni, sah, p)
        F["c_whale_up"] = (core["whale"].diff(3) > 0).astype(float)
        F["c_pro_gt_ret"] = (core["pro"] > core["ret_line"]).astype(float)
        for k, col in (("c_star", "star"), ("c_st_in", "st_in"), ("c_st_out", "st_out"),
                       ("c_acc", "real_acc"), ("c_dist", "real_dist"), ("c_cross_up", "cross_up"),
                       ("c_cross_dn", "cross_dn"), ("c_sqz_fire", "sqz_fire"), ("c_exh", "exhausted")):
            F[k] = _b(core[col].fillna(False), idx)
        F["o_hud_net"] = omni["hud_net"].astype(float) / 6
        F["o_f_hist"] = np.tanh(omni["f_hist"].fillna(0) / 10)
        F["o_s_speed"] = np.tanh(omni["s_speed"].fillna(0) / 30)
        for k, col in (("o_ab_bull", "ab_bull"), ("o_ab_bear", "ab_bear"), ("o_dia_buy", "dia_buy"),
                       ("o_dia_sell", "dia_sell"), ("o_sweep_low", "sweep_low"),
                       ("o_sweep_high", "sweep_high"), ("o_ew_bull", "ew_bull"),
                       ("o_ew_bear", "ew_bear")):
            F[k] = _b(omni[col].fillna(False), idx)
        F["s_eff_pos"] = (df["Close"] > sah["eff_price"]).astype(float)
        F["s_ult_up"] = _b(sah["ult_up_cross"].fillna(False), idx)
        F["s_smc_buy"] = _b(sah["smc_buy"].fillna(False), idx)
        F["s_smc_sell"] = _b(sah["smc_sell"].fillna(False), idx)
        F["k_dir"] = conf["direction"].astype(float) / 5
        F["k_mag"] = conf["magnitude"].astype(float) / 18
        F["k_entry"] = _b(conf["entry_state"].fillna(False), idx)
        F["k_expand"] = _b(conf["expand_state"].fillna(False), idx)
        F["k_risk"] = _b(conf["risk_state"].fillna(False), idx)
    except Exception as exc:                     # noqa: BLE001
        _log.info("council engine: %s", exc)
    try:
        r = pv710.compute(df, None, "Günlük")
        F["v_qconf"] = _b(r["q_conf"], idx)
        F["v_odi_bull"] = _b(r["odi_bull"], idx)
        F["v_odi_bear"] = _b(r["odi_bear"], idx)
        F["v_eff_up"] = _b(r["eff_up"], idx)
        F["v_eff_dn"] = _b(r["eff_dn"], idx)
        F["v_eff_score"] = _b(r["eff_score"], idx) / 8
    except Exception as exc:                     # noqa: BLE001
        _log.info("council v710: %s", exc)
    try:
        r = pvsa.compute(df)
        F["a_rev_up"] = _b(r["rev_up"], idx)
        F["a_rev_dn"] = _b(r["rev_dn"], idx)
        F["a_flip_up"] = _b(r["flip_up"], idx)
        F["a_flip_dn"] = _b(r["flip_dn"], idx)
        F["a_flow"] = _b(r["flow_imb"], idx)
        F["a_abs_dn"] = _b(r["absDn"], idx)
        F["a_abs_up"] = _b(r["absUp"], idx)
    except Exception as exc:                     # noqa: BLE001
        _log.info("council vsa: %s", exc)
    try:
        mg = pfib.mum_gucu(df, None, 300)
        F["m_syn"] = mg["syn"].map(lambda s: SYN_VAL.get(int(s), 0.0)).to_numpy()
    except Exception as exc:                     # noqa: BLE001
        _log.info("council mg: %s", exc)
    # olaylar 3 bar boyunca "taze" sayılır
    for c in F.columns:
        if c in EVENT_COLS:
            F[c] = F[c].rolling(3, min_periods=1).max()
    return F.add_prefix(prefix).fillna(0.0)


def daily_features(dfd: pd.DataFrame) -> pd.DataFrame:
    """Günlük bağlam — GÜN SONUNDA bilinen değerler (tarih = o günün tarihi)."""
    d = dfd[["Open", "High", "Low", "Close", "Volume"]].astype(float).dropna(subset=["Close"]).copy()
    p = dict(eng.PARAMS)
    out = pd.DataFrame(index=d.index)
    try:
        core = eng.apex_core(d, p)
        omni = eng.apex_omni(d, p)
        sah = eng.sahane_layer(d, core["w_pwr"], p)
        conf = eng.confluence(d, core, omni, sah, p)
        out["d_dir"] = conf["direction"].astype(float) / 5
        out["d_hud"] = omni["hud_net"].astype(float) / 6
        out["d_regime"] = (d["Close"] > ema(d["Close"], 200)).astype(float)
        out["d_whale"] = (core["w_pwr"].fillna(50) - 50) / 50
        out["d_eff"] = (d["Close"] > sah["eff_price"]).astype(float)
        out["d_trend50"] = (d["Close"] > ema(d["Close"], 50)).astype(float)
    except Exception as exc:                     # noqa: BLE001
        _log.info("council daily: %s", exc)
    return out.fillna(0.0)


def align_daily(h_index: pd.DatetimeIndex, dF: pd.DataFrame) -> pd.DataFrame:
    """Her 1 saatlik bara ÖNCEKİ günün günlük bağlamını eşler (geleceği görmez)."""
    if dF.empty:
        return pd.DataFrame(index=h_index)
    di = pd.to_datetime(dF.index).normalize()
    dF = dF.copy()
    dF.index = di
    dF = dF[~dF.index.duplicated(keep="last")].sort_index()
    days = pd.DatetimeIndex(h_index).normalize()
    pos = np.searchsorted(dF.index.values, days.values, side="left") - 1   # dünden önceki son gün
    out = pd.DataFrame(np.nan, index=h_index, columns=dF.columns)
    ok = pos >= 0
    out.loc[ok, :] = dF.to_numpy()[pos[ok]]
    return out.fillna(0.0)


def learn_scores(F: pd.DataFrame, close: np.ndarray, horizon: int = 14, retrain: int = 35,
                 min_train: int = 400, max_train: int = 2500, t_cut: float = 1.0,
                 q_in: float = 0.85, q_out: float = 0.35) -> dict[str, Any]:
    """Walk-forward: her 'retrain' barda bir yalnızca geçmişle ağırlık öğrenir."""
    X = F.to_numpy(float)
    n, k = X.shape
    fwd = np.full(n, np.nan)
    fwd[:-horizon] = close[horizon:] / close[:-horizon] - 1
    score = np.full(n, np.nan)
    thr_in = np.full(n, np.nan)
    thr_out = np.full(n, np.nan)
    W_hist: list[tuple[int, np.ndarray, np.ndarray]] = []
    w = mu = sd = None
    for t in range(min_train + horizon, n):
        if w is None or (t - (min_train + horizon)) % retrain == 0:
            hi_ = t - horizon                          # bu bara kadar ileri getirisi BİLİNEN barlar
            lo_ = max(0, hi_ - max_train)
            Xt, yt = X[lo_:hi_], fwd[lo_:hi_]
            m = np.isfinite(yt)
            Xt, yt = Xt[m], yt[m]
            if len(yt) < 100:
                continue
            mu = Xt.mean(axis=0)
            sd = Xt.std(axis=0)
            sd[sd < 1e-9] = np.nan
            Z = np.nan_to_num((Xt - mu) / sd)
            yz = (yt - yt.mean()) / (yt.std() or 1)
            r = (Z * yz[:, None]).mean(axis=0)
            nn = len(yt) / max(horizon / 2, 1)          # örtüşen getiriler → etkin örneklem
            tstat = r * np.sqrt(max(nn - 2, 1)) / np.sqrt(np.clip(1 - r ** 2, 1e-6, None))
            w = np.where(np.abs(tstat) >= t_cut, np.clip(tstat / 4, -1, 1), 0.0)
            S_tr = np.nansum(np.nan_to_num((Xt - mu) / sd) * w, axis=1)
            qi, qo = np.quantile(S_tr, q_in), np.quantile(S_tr, q_out)
            W_hist.append((t, w.copy(), tstat.copy()))
        if w is None:
            continue
        z = np.nan_to_num((X[t] - mu) / sd)
        score[t] = float(np.sum(z * w))
        thr_in[t], thr_out[t] = qi, qo
    return {"score": score, "thr_in": thr_in, "thr_out": thr_out, "W": W_hist,
            "cols": list(F.columns), "mu": mu, "sd": sd}


def top_contrib(F_row: pd.Series, w: np.ndarray, mu, sd, cols, n: int = 4) -> str:
    z = np.nan_to_num((F_row.to_numpy(float) - mu) / sd)
    c = z * w
    order = np.argsort(-np.abs(c))[:n]
    parts = [f"{LABELS.get(cols[j], cols[j])} {c[j]:+.2f}" for j in order if abs(c[j]) > 0.05]
    return " · ".join(parts)


def weights_table(L: dict[str, Any]) -> pd.DataFrame:
    if not L["W"]:
        return pd.DataFrame()
    t, w, ts = L["W"][-1]
    df = pd.DataFrame({"Sinyal": [LABELS.get(c, c) for c in L["cols"]], "Ağırlık": w,
                       "t-istatistiği": ts})
    df["Etki"] = np.where(df["Ağırlık"] > 0, "🟢 yükselişi öngördü",
                          np.where(df["Ağırlık"] < 0, "🔴 düşüşü öngördü", "⚪ gürültü (sustu)"))
    return df.reindex(df["Ağırlık"].abs().sort_values(ascending=False).index).reset_index(drop=True)
