from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
from loguru import logger

from projet_05.branding import apply_brand_theme
from projet_05.modeling.predict import load_metadata, load_pipeline, run_inference

MODEL_PATH = Path("models/best_model.joblib")
METADATA_PATH = Path("models/best_model_meta.json")
SCHEMA_PATH = Path("data/processed/schema.json")


def _load_schema(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_features(metadata: dict, schema: dict, pipeline) -> list[str]:
    if schema:
        candidates = schema.get("numerical_features", []) + schema.get("categorical_features", [])
        if candidates:
            return candidates
    features = metadata.get("features", {})
    explicit = (features.get("numerical") or []) + (features.get("categorical") or [])
    if explicit:
        return explicit
    if pipeline is not None and hasattr(pipeline, "feature_names_in_"):
        return list(pipeline.feature_names_in_)
    return []


def _convert_input(payload: Any, headers: list[str]) -> pd.DataFrame:
    if isinstance(payload, pd.DataFrame):
        df = payload.copy()
    elif payload is None:
        df = pd.DataFrame(columns=headers)
    else:
        df = pd.DataFrame(payload, columns=headers if headers else None)
    df = df.dropna(how="all")
    if df.empty:
        raise gr.Error("Merci de saisir au moins une ligne complète.")
    return df


def _ensure_model():
    if PIPELINE is None:
        raise gr.Error(
            "Aucun modèle entrainé n'a été trouvé. Lancez `python projet_05/modeling/train.py` puis relancez l'application."
        )


def score_table(table):
    _ensure_model()
    df = _convert_input(table, FEATURE_ORDER)
    drop_cols = [TARGET_COLUMN] if TARGET_COLUMN else None
    return run_inference(
        df,
        PIPELINE,
        THRESHOLD,
        drop_columns=drop_cols,
        required_features=FEATURE_ORDER or None,
    )


def score_csv(upload):
    _ensure_model()
    if upload is None:
        raise gr.Error("Veuillez déposer un fichier CSV.")
    df = pd.read_csv(upload.name)
    drop_cols = [TARGET_COLUMN] if TARGET_COLUMN else None
    return run_inference(
        df,
        PIPELINE,
        THRESHOLD,
        drop_columns=drop_cols,
        required_features=FEATURE_ORDER or None,
    )


def predict_from_form(*values):
    _ensure_model()
    if not FEATURE_ORDER:
        raise gr.Error("Impossible de générer le formulaire sans configuration des features.")
    payload = {feature: value for feature, value in zip(FEATURE_ORDER, values)}
    df = pd.DataFrame([payload])
    scored = run_inference(
        df,
        PIPELINE,
        THRESHOLD,
        required_features=FEATURE_ORDER or None,
    )
    row = scored.iloc[0]
    label = "Risque de départ" if int(row["prediction"]) == 1 else "Reste probable"
    return {
        "probability": round(float(row["proba_depart"]), 4),
        "decision": label,
        "threshold": THRESHOLD,
    }


# Chargement des artéfacts
apply_brand_theme()

PIPELINE = None
METADATA: dict[str, Any] = {}
THRESHOLD = 0.5
TARGET_COLUMN: str | None = None
SCHEMA = _load_schema(SCHEMA_PATH)

try:
    PIPELINE = load_pipeline(MODEL_PATH)
    METADATA = load_metadata(METADATA_PATH)
    THRESHOLD = float(METADATA.get("best_threshold", THRESHOLD))
    TARGET_COLUMN = METADATA.get("target")
except FileNotFoundError as exc:
    logger.warning("Artéfact manquant: {}", exc)

FEATURE_ORDER = _infer_features(METADATA, SCHEMA, PIPELINE)

with gr.Blocks(title="Prédicteur d'attrition") as demo:
    gr.Markdown("# API Gradio – Prédiction de départ employé")
    gr.Markdown(
        "Le modèle applique le pipeline entraîné hors-notebook pour fournir une probabilité de départ ainsi qu'une décision binaire."
    )

    if PIPELINE is None:
        gr.Markdown(
            "⚠️ **Aucun modèle disponible.** Lancez les scripts `dataset.py`, `features.py` puis `modeling/train.py`."
        )
    else:
        gr.Markdown(f"Seuil de décision actuel : **{THRESHOLD:.2f}**")

    with gr.Tab("Formulaire unitaire"):
        if not FEATURE_ORDER:
            gr.Markdown("Aucune configuration de features détectée. Utilisez l'onglet CSV pour scorer vos données.")
        else:
            form_inputs: list[gr.components.Component] = [] # type: ignore
            for feature in FEATURE_ORDER:
                form_inputs.append(
                    gr.Textbox(label=feature, placeholder=f"Saisir {feature.replace('_', ' ')}")
                )
            form_output = gr.JSON(label="Résultat")
            gr.Button("Prédire").click(
                fn=predict_from_form,
                inputs=form_inputs,
                outputs=form_output,
            )

    with gr.Tab("Tableau interactif"):
        table_input = gr.Dataframe(
            headers=FEATURE_ORDER if FEATURE_ORDER else None,
            row_count=(1, "dynamic"),
            col_count=(len(FEATURE_ORDER), "dynamic") if FEATURE_ORDER else (5, "dynamic"),
            type="pandas",
        )
        table_output = gr.Dataframe(label="Prédictions", type="pandas")
        gr.Button("Scorer les lignes").click(
            fn=score_table,
            inputs=table_input,
            outputs=table_output,
        )

    with gr.Tab("Fichier CSV"):
        file_input = gr.File(file_types=[".csv"], label="Déposez votre fichier CSV")
        file_output = gr.Dataframe(label="Résultats CSV", type="pandas")
        gr.Button("Scorer le fichier").click(
            fn=score_csv,
            inputs=file_input,
            outputs=file_output,
        )


if __name__ == "__main__":
    demo.launch()
