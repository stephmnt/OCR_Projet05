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
DERIVED_FEATURES = {
    "augmentation_par_revenu",
    "annee_sur_poste_par_experience",
    "nb_formation_par_experience",
    "score_moyen_satisfaction",
    "dern_promo_par_experience",
    "evolution_note",
}
SATISFACTION_COLUMNS = [
    "satisfaction_employee_environnement",
    "satisfaction_employee_nature_travail",
    "satisfaction_employee_equipe",
    "satisfaction_employee_equilibre_pro_perso",
]

# Configuration manuelle des champs d'entrée (label + placeholder).
FIELD_UI_CONFIG = [
    {"name": "age", "label": "Âge", "placeholder": "Âge en années (ex : 35)"},
    {"name": "revenu_mensuel", "label": "Revenu mensuel (€)", "placeholder": "Montant mensuel en euros (ex : 4500)"},
    {"name": "annees_dans_l_entreprise", "label": "Années dans l'entreprise", "placeholder": "Ancienneté totale (ex : 4.5)"},
    {"name": "annees_dans_le_poste_actuel", "label": "Années sur le poste actuel", "placeholder": "Durée dans le poste (ex : 2)"},
    {
        "name": "annees_depuis_la_derniere_promotion",
        "label": "Années depuis la dernière promotion",
        "placeholder": "Durée depuis la dernière promotion (ex : 1)",
    },
    {
        "name": "distance_domicile_travail",
        "label": "Distance domicile-travail (km)",
        "placeholder": "Distance en kilomètres (ex : 12)",
    },
    {
        "name": "nombre_participation_pee",
        "label": "Nombre de participations PEE",
        "placeholder": "Nombre de participations (entier)",
    },
    {
        "name": "note_evaluation_actuelle",
        "label": "Note d'évaluation actuelle",
        "placeholder": "Score actuel (1 à 5)",
    },
    {
        "name": "note_evaluation_precedente",
        "label": "Note d'évaluation précédente",
        "placeholder": "Score précédent (1 à 5)",
    },
    {
        "name": "annees_depuis_le_changement_deposte",
        "label": "Années depuis le dernier changement de poste",
        "placeholder": "Temps écoulé (ex : 0 si jamais)",
    },
    {
        "name": "annee_experience_totale",
        "label": "Années d'expérience totale",
        "placeholder": "Expérience cumulative (ex : 8)",
    },
    {
        "name": "nb_formations_suivies",
        "label": "Nombre de formations suivies",
        "placeholder": "Total des formations (entier)",
    },
    {
        "name": "satisfaction_employee_environnement",
        "label": "Satisfaction environnement",
        "placeholder": "Note de 1 (faible) à 5 (forte)",
        "info": "Valeur comprise entre 1 et 5",
    },
    {
        "name": "satisfaction_employee_nature_travail",
        "label": "Satisfaction nature du travail",
        "placeholder": "Note de 1 à 5",
        "info": "Valeur comprise entre 1 et 5",
    },
    {
        "name": "satisfaction_employee_equipe",
        "label": "Satisfaction équipe",
        "placeholder": "Note de 1 à 5",
        "info": "Valeur comprise entre 1 et 5",
    },
    {
        "name": "satisfaction_employee_equilibre_pro_perso",
        "label": "Satisfaction équilibre pro/perso",
        "placeholder": "Note de 1 à 5",
        "info": "Valeur comprise entre 1 et 5",
    },
    {
        "name": "genre",
        "label": "Genre",
        "component": "dropdown",
        "choices": ["Femme", "Homme"],
        "info": "Sélectionnez le genre",
    },
    {
        "name": "departement",
        "label": "Département",
        "component": "dropdown",
        "choices": ["Commercial", "Consulting", "Ressources Humaines"],
    },
    {
        "name": "frequence_deplacement",
        "label": "Fréquence des déplacements",
        "component": "dropdown",
        "choices": ["Aucun", "Occasionnel", "Frequent"],
    },
    {
        "name": "etat_civil",
        "label": "Statut marital",
        "component": "dropdown",
        "choices": ["Célibataire", "Marié(e)", "Divorcé(e)"],
    },
    {"name": "niveau_etudes", "label": "Niveau d'études", "placeholder": "Ex : Licence, Master"},
    {
        "name": "role",
        "label": "Poste occupé",
        "component": "dropdown",
        "choices": [
            "Cadre Commercial",
            "Assistant de Direction",
            "Consultant",
            "Tech Lead",
            "Manager",
            "Senior Manager",
            "Représentant Commercial",
            "Directeur Technique",
            "Ressources Humaines",
        ],
    },
    {"name": "type_contrat", "label": "Type de contrat", "placeholder": "Ex : CDI, CDD"},
]
FIELD_UI_LOOKUP = {cfg["name"]: cfg for cfg in FIELD_UI_CONFIG}
CATEGORICAL_NORMALIZERS: dict[str, dict[str, str]] = {
    "genre": {
        "f": "F",
        "femme": "F",
        "m": "M",
        "homme": "M",
    },
    "etat_civil": {
        "célibataire": "Célibataire",
        "celibataire": "Célibataire",
        "marié(e)": "Marié(e)",
        "marie(e)": "Marié(e)",
        "marie": "Marié(e)",
        "marié": "Marié(e)",
        "divorcé(e)": "Divorcé(e)",
        "divorce(e)": "Divorcé(e)",
    },
    "departement": {
        "commercial": "Commercial",
        "consulting": "Consulting",
        "ressources humaines": "Ressources Humaines",
    },
    "role": {
        "cadre commercial": "Cadre Commercial",
        "assistant de direction": "Assistant de Direction",
        "consultant": "Consultant",
        "tech lead": "Tech Lead",
        "manager": "Manager",
        "senior manager": "Senior Manager",
        "représentant commercial": "Représentant Commercial",
        "representant commercial": "Représentant Commercial",
        "directeur technique": "Directeur Technique",
        "ressources humaines": "Ressources Humaines",
    },
}


def _load_schema(path: Path) -> dict[str, Any]:
    """Load the schema definition stored as JSON.

    Args:
        path: Path to the schema.json file.

    Returns:
        A dictionary describing the schema or an empty dict if the file is missing.
    """
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_features(metadata: dict, schema: dict, pipeline) -> list[str]:
    """Infer the ordered list of features expected by the model.

    Args:
        metadata: Metadata produced during training.
        schema: Schema derived from `features.py`.
        pipeline: Loaded sklearn pipeline (optional).

    Returns:
        List of feature names in the order expected by the model.
    """
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
    """Normalize any user input into a validated DataFrame.

    Args:
        payload: Raw table coming from Gradio (DataFrame, list, etc.).
        headers: Expected column names.

    Returns:
        A sanitized DataFrame.

    Raises:
        gr.Error: If no valid row is provided.
    """
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


def _resolve_field_ui(feature: str) -> tuple[str, str, str | None, str, dict[str, Any]]:
    """Return UI metadata (label, placeholder, info, component type, config)."""

    config = FIELD_UI_LOOKUP.get(feature, {})
    label = config.get("label") or feature.replace("_", " ").capitalize()
    placeholder = config.get("placeholder") or f"Saisir {label.lower()}"
    info = config.get("info")
    component = config.get("component", "textbox")
    return label, placeholder, info, component, config


def _build_input_component(feature: str) -> gr.components.Component: # type: ignore
    """Instantiate the appropriate Gradio component for a feature."""

    label, placeholder, info, component, config = _resolve_field_ui(feature)
    if component == "dropdown":
        choices = config.get("choices") or []
        default = config.get("default")
        allow_custom = config.get("allow_custom_value", False)
        return gr.Dropdown(
            label=label,
            choices=choices,
            value=default,
            info=info,
            allow_custom_value=allow_custom,
        )
    return gr.Textbox(label=label, placeholder=placeholder, info=info)


def _normalize_categorical_values(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize friendly categorical values into the codes used by the model."""

    normalized = df.copy()

    def _normalize_value(value, mapping: dict[str, str]):
        if pd.isna(value):
            return value
        if isinstance(value, str):
            cleaned = value.strip()
            lowered = cleaned.lower()
            return mapping.get(lowered, cleaned)
        return mapping.get(value, value)

    for column, mapping in CATEGORICAL_NORMALIZERS.items():
        if column not in normalized.columns:
            continue
        normalized[column] = normalized[column].apply(lambda v, m=mapping: _normalize_value(v, m))
    return normalized


def _apply_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute engineered ratios so end-users do not have to provide them."""

    enriched = _normalize_categorical_values(df)

    def _safe_ratio(numerator: str, denominator: str, output: str) -> None:
        if numerator not in enriched.columns or denominator not in enriched.columns:
            return
        numerator_series = pd.to_numeric(enriched[numerator], errors="coerce")
        denominator_series = pd.to_numeric(enriched[denominator], errors="coerce").replace(0, pd.NA)
        enriched[output] = numerator_series / denominator_series

    prev_raise_col = "augementation_salaire_precedente"
    if prev_raise_col in enriched:
        normalized = (
            enriched[prev_raise_col]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", ".", regex=False)
            .str.strip()
        )
        enriched[prev_raise_col] = pd.to_numeric(normalized, errors="coerce") / 100

    _safe_ratio("augementation_salaire_precedente", "revenu_mensuel", "augmentation_par_revenu")
    _safe_ratio("annees_dans_le_poste_actuel", "annee_experience_totale", "annee_sur_poste_par_experience")
    _safe_ratio("nb_formations_suivies", "annee_experience_totale", "nb_formation_par_experience")
    _safe_ratio("annees_depuis_la_derniere_promotion", "annee_experience_totale", "dern_promo_par_experience")

    existing_sats = [col for col in SATISFACTION_COLUMNS if col in enriched.columns]
    if existing_sats:
        enriched["score_moyen_satisfaction"] = pd.DataFrame(
            {col: pd.to_numeric(enriched[col], errors="coerce") for col in existing_sats}
        ).mean(axis=1)

    if {"note_evaluation_actuelle", "note_evaluation_precedente"}.issubset(enriched.columns):
        enriched["evolution_note"] = pd.to_numeric(
            enriched["note_evaluation_actuelle"], errors="coerce"
        ) - pd.to_numeric(enriched["note_evaluation_precedente"], errors="coerce")

    return enriched


def _ensure_model():
    """Ensure that a pipeline has been loaded before inference."""
    if PIPELINE is None:
        raise gr.Error(
            "Aucun modèle entrainé n'a été trouvé. Lancez `python projet_05/modeling/train.py` puis relancez l'application."
        )


def score_table(table):
    """Score data entered via the interactive table."""
    _ensure_model()
    df = _convert_input(table, INPUT_FEATURES)
    df = _apply_derived_features(df)
    drop_cols = [TARGET_COLUMN] if TARGET_COLUMN else None
    return run_inference(
        df,
        PIPELINE,
        THRESHOLD,
        drop_columns=drop_cols,
        required_features=FEATURE_ORDER or None,
    )


def score_csv(upload):
    
    """Score a CSV uploaded by the user."""
    _ensure_model()
    if upload is None:
        raise gr.Error("Veuillez déposer un fichier CSV.")
    df = pd.read_csv(upload.name)
    df = _apply_derived_features(df)
    drop_cols = [TARGET_COLUMN] if TARGET_COLUMN else None
    return run_inference(
        df,
        PIPELINE,
        THRESHOLD,
        drop_columns=drop_cols,
        required_features=FEATURE_ORDER or None,
    )


def predict_from_form(*values):
    """Score a single row coming from the form tab."""
    _ensure_model()
    if not INPUT_FEATURES:
        raise gr.Error("Impossible de générer le formulaire sans configuration des features.")
    payload = {feature: value for feature, value in zip(INPUT_FEATURES, values)}
    df = pd.DataFrame([payload])
    df = _apply_derived_features(df)
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
INPUT_FEATURES = [feature for feature in FEATURE_ORDER if feature not in DERIVED_FEATURES]
if not INPUT_FEATURES:
    INPUT_FEATURES = FEATURE_ORDER

with gr.Blocks(title="Prédicteur d'attrition") as demo:
    gr.Markdown("# OCR Projet 5 – Prédiction de départ employé")
    gr.Markdown(
        "Le modèle fournit une probabilité de départ ainsi qu'une décision binaire."
    )

    if PIPELINE is None:
        gr.Markdown(
            "**Aucun modèle disponible.** Lancez les scripts `dataset.py`, `features.py` puis `modeling/train.py`."
        )
    else:
        gr.Markdown(f"Seuil de décision actuel : **{THRESHOLD:.2f}**")

    with gr.Tab("Formulaire unitaire"):
        if not INPUT_FEATURES:
            gr.Markdown("Aucune configuration de features détectée. Utilisez l'onglet CSV pour scorer vos données.")
        else:
            form_inputs: list[gr.components.Component] = [] # type: ignore
            for feature in INPUT_FEATURES:
                form_inputs.append(_build_input_component(feature))
            form_output = gr.JSON(label="Résultat")
            gr.Button("Prédire").click(
                fn=predict_from_form,
                inputs=form_inputs,
                outputs=form_output,
            )

    with gr.Tab("Tableau interactif"):
        table_input = gr.Dataframe(
            headers=INPUT_FEATURES if INPUT_FEATURES else None,
            row_count=(1, "dynamic"),
            col_count=(len(INPUT_FEATURES), "dynamic") if INPUT_FEATURES else (5, "dynamic"),
            type="pandas",
        )
        table_output = gr.Dataframe(label="Prédictions", type="pandas")
        gr.Button("Scorer les lignes").click(
            fn=score_table,
            inputs=table_input,
            outputs=table_output,
        )

    with gr.Tab("Fichier CSV"):
        gr.Markdown("Un exemple complet de fichier à importer est disponible dans le dépôt github : [`references/sample_employees.csv`](https://github.com/stephmnt/OCR_Projet05/blob/main/references/sample_employees.csv)")
        file_input = gr.File(file_types=[".csv"], label="Déposez votre fichier CSV")
        file_output = gr.Dataframe(label="Résultats CSV", type="pandas")
        gr.Button("Scorer le fichier").click(
            fn=score_csv,
            inputs=file_input,
            outputs=file_output,
        )


if __name__ == "__main__":
    demo.launch()
