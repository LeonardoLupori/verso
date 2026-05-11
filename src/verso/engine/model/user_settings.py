"""User-level display preferences, persisted across sessions."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _settings_path() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home()))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "verso" / "settings.json"


@dataclass
class UserSettings:
    """Display preferences stored per user, not per project."""

    cp_size: int = 10
    cp_shape: str = "Cross"
    cp_color: str = "Yellow"

    def to_dict(self) -> dict[str, Any]:
        return {
            "cp_size": self.cp_size,
            "cp_shape": self.cp_shape,
            "cp_color": self.cp_color,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> UserSettings:
        return cls(
            cp_size=int(d.get("cp_size", 10)),
            cp_shape=str(d.get("cp_shape", "Cross")),
            cp_color=str(d.get("cp_color", "Yellow")),
        )

    @classmethod
    def load(cls) -> UserSettings:
        """Load settings from disk, returning defaults if missing or corrupt."""
        path = _settings_path()
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return cls()

    def save(self) -> None:
        """Write settings to disk."""
        path = _settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
