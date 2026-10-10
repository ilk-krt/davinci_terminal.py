# apex/snapshot.py — akşam hesabının sonuçlarını kaydet / oku
#
# tools/build_snapshot.py her iş günü akşamı ağır hesapları yapar ve sonuçları
# data/snapshot/<ad>.pkl.gz olarak yazar; yanında _status.json neyin ne zaman
# hesaplandığını ve hata olup olmadığını tutar. Arayüz önce buradan okur:
# Yahoo'ya gitmez, anında açılır. Dosya yoksa ya da çok eskiyse canlı hesaplar.
from __future__ import annotations

import datetime as dt
import gzip
import json
import pickle
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent          # repo kökü
DATA = ROOT / "data"
SNAP = DATA / "snapshot"
STATUS = SNAP / "_status.json"
MAX_AGE_H = 72          # hafta sonu + tatil payı; daha eskiyse canlı hesap


def data_file(name: str) -> Path | None:
    """data/ altındaki bir dosya — repo kökünden ya da çalışma klasöründen."""
    for base in (DATA, Path.cwd() / "data"):
        p = base / name
        if p.exists():
            return p
    return None


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def read_status() -> dict[str, Any]:
    try:
        return json.loads(STATUS.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_status(name: str, ok: bool, secs: float, note: str = "") -> None:
    SNAP.mkdir(parents=True, exist_ok=True)
    st = read_status()
    st[name] = {"ok": ok, "ts": _now().isoformat(timespec="seconds"),
                "sure_sn": round(secs, 1), "not": note[:300]}
    STATUS.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")


def save(name: str, obj: Any) -> int:
    SNAP.mkdir(parents=True, exist_ok=True)
    p = SNAP / f"{name}.pkl.gz"
    tmp = p.with_suffix(".tmp")
    with gzip.open(tmp, "wb", compresslevel=6) as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(p)
    return p.stat().st_size


def age_hours(name: str) -> float | None:
    p = SNAP / f"{name}.pkl.gz"
    if not p.exists():
        return None
    ts = read_status().get(name, {}).get("ts")
    try:
        t = dt.datetime.fromisoformat(ts) if ts else \
            dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc)
    except Exception:
        t = dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc)
    return (_now() - t).total_seconds() / 3600


def load(name: str, max_age_h: float = MAX_AGE_H) -> Any | None:
    """Kayıtlı sonucu döner; yoksa, bozuksa ya da çok eskiyse None."""
    p = SNAP / f"{name}.pkl.gz"
    a = age_hours(name)
    if a is None or a > max_age_h:
        return None
    try:
        with gzip.open(p, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def summary() -> str:
    """Arayüz başlığı için tek satır: en son ne zaman hesaplandı."""
    st = read_status()
    ok = [v for v in st.values() if isinstance(v, dict) and v.get("ok")]
    if not ok:
        return ""
    last = max(dt.datetime.fromisoformat(v["ts"]) for v in ok)
    h = (_now() - last).total_seconds() / 3600
    ist = last.astimezone(dt.timezone(dt.timedelta(hours=3)))
    return f"{ist:%d.%m %H:%M} ({h:.0f} saat önce)"
