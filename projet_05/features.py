from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
import typer

from projet_05.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from projet_05.settings import Settings, load_settings

app = typer.Typer(help="Génération des features et nettoyage de la cible.")

TARGET_MAPPING = {
    "1": 1,
    "0": 0,
    "oui": 1,
    "non": 0,
    "true": 1,
    "false": 0,
    "quitte": 1,
    "reste": 0,
    "yes": 1,
    "no": 0,
}


# ---------------------------------------------------------------------------
# Utilitaires cœur de pipeline
# ---------------------------------------------------------------------------
def _load_merged_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Le fichier fusionné {path} est introuvable. Lancez `python projet_05/dataset.py` d'abord."
        )
    logger.info("Chargement du dataset fusionné depuis {}", path)
    return pd.read_csv(path)


def _normalize_target(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    if settings.target not in df.columns:
        raise KeyError(f"La variable cible '{settings.target}' est absente du fichier.")

    normalized = (
        df[settings.target]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(TARGET_MAPPING)
    )
    df = df.copy()
    df[settings.target] = normalized
    before = len(df)
    df = df[df[settings.target].isin([0, 1])].copy()
    dropped = before - len(df)
    if dropped:
        logger.warning("Suppression de {} lignes avec une cible invalide.", dropped)
    df[settings.target] = df[settings.target].astype(int)
    return df


def _safe_ratio(df: pd.DataFrame, numerator: str, denominator: str, output: str) -> None:
    if numerator not in df.columns or denominator not in df.columns:
        return
    denominator_series = df[denominator].replace({0: np.nan})
    df[output] = df[numerator] / denominator_series


def _engineer_features(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    engineered = df.copy()

    col = "augementation_salaire_precedente"
    if col in engineered:
        engineered[col] = (
            engineered[col]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", ".", regex=False)
            .str.strip()
        )
        engineered[col] = pd.to_numeric(engineered[col], errors="coerce") / 100

    _safe_ratio(engineered, "augementation_salaire_precedente", "revenu_mensuel", "augmentation_par_revenu")
    _safe_ratio(engineered, "annees_dans_le_poste_actuel", "annee_experience_totale", "annee_sur_poste_par_experience")
    _safe_ratio(engineered, "nb_formations_suivies", "annee_experience_totale", "nb_formation_par_experience")
    _safe_ratio(
        engineered, "annees_depuis_la_derniere_promotion", "annee_experience_totale", "dern_promo_par_experience"
    )

    if settings.sat_cols:
        existing = [col for col in settings.sat_cols if col in engineered.columns]
        if existing:
            engineered["score_moyen_satisfaction"] = engineered[existing].mean(axis=1)

    if "note_evaluation_actuelle" in engineered.columns and "note_evaluation_precedente" in engineered.columns:
        engineered["evolution_note"] = (
            engineered["note_evaluation_actuelle"] - engineered["note_evaluation_precedente"]
        )

    return engineered


def build_features(settings: Settings, *, input_path: Path) -> pd.DataFrame:
    df = _load_merged_dataset(input_path)
    df = _normalize_target(df, settings)
    df = _engineer_features(df, settings)
    return df


def save_features(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.success("Dataset enrichi sauvegardé dans {}", output_path)


def save_schema(settings: Settings, output_path: Path) -> None:
    schema = {
        "target": settings.target,
        "col_id": settings.col_id,
        "numerical_features": list(settings.num_cols),
        "categorical_features": list(settings.cat_cols),
        "satisfaction_features": list(settings.sat_cols),
        "generated_at": datetime.utcnow().isoformat(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    logger.info("Schéma sauvegardé dans {}", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@app.command()
def main(
    settings_path: Path = typer.Option(
        None,
        "--settings",
        "-s",
        help="Chemin optionnel vers un fichier settings.yml personnalisé.",
    ),
    input_path: Path = typer.Option(
        INTERIM_DATA_DIR / "merged.csv",
        "--input",
        "-i",
        help="Chemin du fichier issu de la fusion.",
    ),
    output_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "dataset.csv",
        "--output",
        "-o",
        help="Chemin du fichier enrichi.",
    ),
    schema_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "schema.json",
        "--schema",
        help="Chemin de sauvegarde du schéma de features.",
    ),
):
    """Pipeline Typer pour préparer le dataset enrichi."""

    settings = load_settings(settings_path) if settings_path else load_settings()
    df = build_features(settings, input_path=input_path)
    save_features(df, output_path)
    save_schema(settings, schema_path)


if __name__ == "__main__":
    app()
