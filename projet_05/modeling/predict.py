from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from loguru import logger
import typer

from projet_05.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer(help="Inférence à partir du pipeline entraîné.")


def load_pipeline(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable: {model_path}")
    logger.info("Chargement du modèle {}", model_path)
    return load(model_path)


def load_metadata(metadata_path: Path) -> dict:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Fichier métadonnées introuvable: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def run_inference(
    df: pd.DataFrame,
    pipeline,
    threshold: float,
    drop_columns: list[str] | None = None,
    required_features: list[str] | None = None,
) -> pd.DataFrame:
    features = df.drop(columns=drop_columns or [], errors="ignore")
    if required_features:
        for col in required_features:
            if col not in features.columns:
                features[col] = np.nan
        features = features[required_features]
    proba = pipeline.predict_proba(features)[:, 1]
    predictions = (proba >= threshold).astype(int)
    output = df.copy()
    output["proba_depart"] = proba
    output["prediction"] = predictions
    return output


@app.command()
def main(
    model_path: Path = typer.Option(
        MODELS_DIR / "best_model.joblib",
        "--model-path",
        help="Pipeline entraîné sauvegardé via train.py",
    ),
    metadata_path: Path = typer.Option(
        MODELS_DIR / "best_model_meta.json",
        "--metadata-path",
        help="Fichier JSON contenant le seuil optimal.",
    ),
    features_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "dataset.csv",
        "--features",
        "-f",
        help="Jeu de features sur lequel produire des prédictions.",
    ),
    predictions_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "predictions.csv",
        "--output",
        "-o",
        help="Chemin de sauvegarde des prédictions.",
    ),
):
    """Entrypoint Typer pour générer un fichier de prédictions."""

    pipeline = load_pipeline(model_path)
    metadata = load_metadata(metadata_path)
    threshold = metadata.get("best_threshold", 0.5)
    features_cfg = metadata.get("features", {})
    required_features = (features_cfg.get("numerical") or []) + (features_cfg.get("categorical") or [])
    df = pd.read_csv(features_path)
    logger.info("Dataset chargé: {} lignes", len(df))

    target_col = metadata.get("target")
    predictions = run_inference(
        df,
        pipeline,
        threshold,
        drop_columns=[target_col] if target_col else None,
        required_features=required_features or None,
    )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_path, index=False)
    logger.success("Prédictions sauvegardées dans {}", predictions_path)


if __name__ == "__main__":
    app()
