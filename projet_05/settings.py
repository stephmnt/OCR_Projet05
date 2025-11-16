from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import os
from pathlib import Path
from typing import Iterable

import yaml

DEFAULT_SETTINGS_PATH = Path(__file__).with_name("settings.yml")


@dataclass(frozen=True)
class Settings:
    random_state: int = 42
    path_sirh: Path = field(default_factory=lambda: Path("data/raw/sirh.csv"))
    path_eval: Path = field(default_factory=lambda: Path("data/raw/evaluation.csv"))
    path_sondage: Path = field(default_factory=lambda: Path("data/raw/sondage.csv"))
    col_id: str = "id_employee"
    target: str = "a_quitte_l_entreprise"
    num_cols: tuple[str, ...] = ()
    cat_cols: tuple[str, ...] = ()
    sat_cols: tuple[str, ...] = ()
    first_vars: tuple[str, ...] = ()
    subsample_frac: float = 1.0
    sql_file: Path = field(default_factory=lambda: Path("merge_sql.sql"))
    db_url: str | None = None
    db_schema: str | None = None

    def as_dict(self) -> dict:
        """Return a serializable representation (useful for logging/tests)."""
        return {
            "random_state": self.random_state,
            "path_sirh": str(self.path_sirh),
            "path_eval": str(self.path_eval),
            "path_sondage": str(self.path_sondage),
            "col_id": self.col_id,
            "target": self.target,
            "num_cols": list(self.num_cols),
            "cat_cols": list(self.cat_cols),
            "sat_cols": list(self.sat_cols),
            "first_vars": list(self.first_vars),
            "subsample_frac": self.subsample_frac,
            "sql_file": str(self.sql_file),
            "db_url": self.db_url,
            "db_schema": self.db_schema,
        }


def _ensure_iterable(values: Iterable[str] | None, *, field_name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        msg = f"'{field_name}' doit être une liste et non une chaîne isolée."
        raise TypeError(msg)
    return tuple(v for v in values if v)


def _resolve_path(candidate: str | os.PathLike[str] | None, *, base_dir: Path) -> Path:
    if not candidate:
        raise ValueError("Aucun chemin n'a été fourni dans le fichier de configuration.")
    resolved = Path(candidate)
    if not resolved.is_absolute():
        resolved = (base_dir / resolved).resolve()
    return resolved


def _load_raw_settings(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Le fichier de configuration {path} doit contenir un dictionnaire YAML.")
    return data


@lru_cache
def load_settings(custom_path: str | os.PathLike[str] | None = None) -> Settings:
    """
    Charger la configuration projet depuis un fichier YAML.

    L'ordre de recherche est :
    1. Argument `custom_path` si fourni.
    2. Variable d'environnement `PROJET05_SETTINGS`.
    3. Fichier par défaut `projet_05/settings.yml`.
    """

    env_path = os.environ.get("PROJET05_SETTINGS")
    raw_path = Path(custom_path or env_path or DEFAULT_SETTINGS_PATH)

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Fichier de configuration introuvable : {raw_path}. "
            "Initialisez-le depuis projet_05/settings.yml ou indiquez PROJET05_SETTINGS."
        )

    base_dir = raw_path.parent
    payload = _load_raw_settings(raw_path)
    paths_block = payload.get("paths", {})

    database_block = payload.get("database", {})
    env_db_url = os.environ.get("PROJET05_DATABASE_URL")
    db_url = env_db_url or database_block.get("url")
    db_schema = database_block.get("schema")

    settings = Settings(
        random_state=int(payload.get("random_state", Settings.random_state)),
        path_sirh=_resolve_path(paths_block.get("sirh", Settings().path_sirh), base_dir=base_dir),
        path_eval=_resolve_path(paths_block.get("evaluation", Settings().path_eval), base_dir=base_dir),
        path_sondage=_resolve_path(paths_block.get("sondage", Settings().path_sondage), base_dir=base_dir),
        col_id=payload.get("col_id", Settings.col_id),
        target=payload.get("target", Settings.target),
        num_cols=_ensure_iterable(payload.get("num_cols"), field_name="num_cols"),
        cat_cols=_ensure_iterable(payload.get("cat_cols"), field_name="cat_cols"),
        sat_cols=_ensure_iterable(payload.get("sat_cols"), field_name="sat_cols"),
        first_vars=_ensure_iterable(payload.get("first_vars"), field_name="first_vars"),
        subsample_frac=float(payload.get("subsample_frac", Settings.subsample_frac)),
        sql_file=_resolve_path(paths_block.get("sql_file", Settings().sql_file), base_dir=base_dir),
        db_url=db_url,
        db_schema=db_schema,
    )
    return settings
