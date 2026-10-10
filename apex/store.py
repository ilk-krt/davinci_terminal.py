# apex/store.py — AETHER APEX
from __future__ import annotations


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
