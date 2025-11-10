from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Union

from scripts_projet04.brand.brand import (  # type: ignore[import-not-found]
    Theme,
    ThemeConfig,
    configure_brand,
    load_brand,
    make_diverging_cmap,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BRAND_PATH = ROOT_DIR / "scripts_projet04" / "brand" / "brand.yml"


def _resolve_path(path: Union[str, Path, None]) -> Path:
    if path is None:
        return DEFAULT_BRAND_PATH
    return Path(path).expanduser().resolve()


@lru_cache(maxsize=1)
def load_brand_config(path: Union[str, Path, None] = None) -> ThemeConfig:
    """Load the brand YAML once and return the parsed ThemeConfig."""
    cfg_path = _resolve_path(path)
    return load_brand(cfg_path)


@lru_cache(maxsize=1)
def apply_brand_theme(path: Union[str, Path, None] = None) -> ThemeConfig:
    """
    Apply the OpenClassrooms/TechNova brand theme globally.

    Returns the ThemeConfig so callers can inspect colors if needed.
    """
    cfg_path = _resolve_path(path)
    cfg = configure_brand(cfg_path)
    Theme.apply()
    return cfg


__all__ = [
    "Theme",
    "ThemeConfig",
    "apply_brand_theme",
    "load_brand_config",
    "make_diverging_cmap",
    "DEFAULT_BRAND_PATH",
]
