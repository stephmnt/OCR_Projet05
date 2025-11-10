from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from joblib import dump
from loguru import logger
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import typer

from projet_05.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from projet_05.explainability import (
    compute_shap_summary,
    export_local_explanations,
    save_shap_summary,
)
from projet_05.settings import Settings, load_settings

app = typer.Typer(help="Entraînement et sélection du meilleur modèle.")


def _clean_values(payload: dict) -> dict:
    def _convert(value):
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        return value

    return {key: _convert(value) for key, value in payload.items()}


@dataclass
class ModelResult:
    name: str
    best_estimator: ImbPipeline
    best_params: dict
    best_threshold: float
    metrics: Dict[str, float]


def load_processed_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset traité introuvable ({path}). Lancez `python projet_05/features.py`."
        )
    logger.info("Chargement du dataset préparé depuis {}", path)
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame, settings: Settings) -> Tuple[pd.DataFrame, pd.Series]:
    if settings.target not in df.columns:
        raise KeyError(f"La cible {settings.target} est absente du dataset.")
    y = df[settings.target].astype(int)
    drop_cols = [settings.target]
    if settings.col_id in df.columns:
        drop_cols.append(settings.col_id)
    X = df.drop(columns=drop_cols, errors="ignore")
    return X, y


def build_preprocessor(settings: Settings, X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = [col for col in settings.num_cols if col in X.columns]
    categorical_features = [col for col in settings.cat_cols if col in X.columns]
    if not numeric_features:
        numeric_features = X.select_dtypes(include="number").columns.tolist()
    if not categorical_features:
        categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_transformer, categorical_features))
    if not transformers:
        raise ValueError("Aucune feature disponible pour l'entraînement.")
    return ColumnTransformer(transformers=transformers)


def get_models(random_state: int):
    return {
        "LogReg_balanced": (
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=random_state,
            ),
            [
                {
                    "clf__solver": ["lbfgs"],
                    "clf__penalty": ["l2"],
                    "clf__C": [0.1, 1.0, 10.0],
                },
                {
                    "clf__solver": ["liblinear"],
                    "clf__penalty": ["l1", "l2"],
                    "clf__C": [0.1, 1.0, 10.0],
                },
            ],
        ),
        "RF_balanced": (
            RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                random_state=random_state,
            ),
            {
                "clf__n_estimators": [200, 300, 500],
                "clf__max_depth": [6, 8, 10],
                "clf__min_samples_split": [5, 10, 15],
                "clf__min_samples_leaf": [2, 5, 8],
            },
        ),
    }


def _compute_best_threshold(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.nanargmax(f1_scores)
    if thresholds.size == 0:
        return 0.5
    best_idx = min(best_idx, thresholds.size - 1)
    return thresholds[best_idx]


def evaluate_models(X, y, settings: Settings, preprocessor: ColumnTransformer) -> list[ModelResult]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=settings.random_state)
    results: list[ModelResult] = []

    for name, (model, grid) in get_models(settings.random_state).items():
        logger.info("Entraînement du modèle {}", name)
        pipe = ImbPipeline(
            steps=[
                ("prep", preprocessor),
                ("smote", SMOTE(random_state=settings.random_state)),
                ("clf", model),
            ]
        )
        search = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            cv=cv,
            scoring="f1",
            n_jobs=-1,
        )
        search.fit(X, y)
        best_pipe = search.best_estimator_

        y_proba = cross_val_predict(best_pipe, X, y, cv=cv, method="predict_proba")[:, 1]
        threshold = _compute_best_threshold(y, y_proba)
        y_pred = (y_proba >= threshold).astype(int)

        metrics = {
            "f1": f1_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "precision": precision_score(y, y_pred),
            "roc_auc": roc_auc_score(y, y_proba),
        }
        logger.info("Scores {} -> {}", name, metrics)
        results.append(
            ModelResult(
                name=name,
                best_estimator=best_pipe,
                best_params=search.best_params_,
                best_threshold=threshold,
                metrics=metrics,
            )
        )
    return results


def compute_dummy_baseline(y: pd.Series) -> dict:
    majority = int(y.mode().iloc[0])
    y_pred = np.full_like(y, fill_value=majority)
    return {
        "strategy": "most_frequent",
        "majority_class": majority,
        "f1": f1_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "roc_auc": 0.5,
    }


def fit_final_pipeline(
    best_result: ModelResult,
    X: pd.DataFrame,
    y: pd.Series,
    settings: Settings,
):
    tuned_pipeline = clone(best_result.best_estimator)
    tuned_pipeline.fit(X, y)
    final_preprocessor = tuned_pipeline.named_steps["prep"]
    clf = tuned_pipeline.named_steps["clf"]
    final_pipe = Pipeline(
        steps=[
            ("prep", final_preprocessor),
            ("clf", clf),
        ]
    )
    logger.success(
        "Modèle {} ré-entraîné sur {} lignes (avec rééchantillonnage interne).",
        best_result.name,
        len(X),
    )
    return final_pipe


def save_artifacts(
    pipeline: Pipeline,
    results: list[ModelResult],
    best_result: ModelResult,
    baseline: dict,
    settings: Settings,
    model_path: Path,
    metadata_path: Path,
    shap_path: Path,
    X: pd.DataFrame,
    y: pd.Series,
):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(pipeline, model_path)
    logger.success("Pipeline sauvegardé dans {}", model_path)

    metadata = {
        "best_model": best_result.name,
        "best_threshold": float(best_result.best_threshold),
        "best_params": best_result.best_params,
        "metrics": _clean_values(best_result.metrics),
        "all_results": [
            {
                "model": r.name,
                "metrics": _clean_values(r.metrics),
                "best_threshold": float(r.best_threshold),
                "best_params": r.best_params,
            }
            for r in results
        ],
        "baseline": _clean_values(baseline),
        "features": {
            "numerical": list(settings.num_cols),
            "categorical": list(settings.cat_cols),
        },
        "target": settings.target,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Métadonnées sauvegardées dans {}", metadata_path)

    shap_summary, shap_values, shap_sample = compute_shap_summary(pipeline, X, y)
    if shap_summary is not None:
        save_shap_summary(shap_summary, shap_path)
        export_local_explanations(pipeline, shap_values, shap_sample)


@app.command()
def main(
    settings_path: Path = typer.Option(None, "--settings", "-s", help="Chemin alternatif vers settings.yml."),
    input_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "dataset.csv",
        "--input",
        "-i",
        help="Dataset enrichi issu de projet_05/features.py",
    ),
    model_path: Path = typer.Option(
        MODELS_DIR / "best_model.joblib",
        "--model-path",
        help="Chemin de sauvegarde du pipeline entraîné.",
    ),
    metadata_path: Path = typer.Option(
        MODELS_DIR / "best_model_meta.json",
        "--metadata-path",
        help="Chemin de sauvegarde des métriques et métadonnées.",
    ),
    shap_path: Path = typer.Option(
        REPORTS_DIR / "shap_summary.csv",
        "--shap-path",
        help="Chemin de sortie du résumé SHAP.",
    ),
):
    """Script principal pour lancer l'entraînement complet."""

    settings = load_settings(settings_path) if settings_path else load_settings()
    df = load_processed_dataset(input_path)
    X, y = split_features_target(df, settings)
    preprocessor = build_preprocessor(settings, X)
    results = evaluate_models(X, y, settings, preprocessor)
    if not results:
        raise RuntimeError("Aucun modèle évalué. Vérifiez la configuration.")
    best_result = max(results, key=lambda r: r.metrics["f1"])
    baseline = compute_dummy_baseline(y)
    logger.info("Baseline Dummy -> {}", baseline)

    final_pipeline = fit_final_pipeline(best_result, X, y, settings)
    save_artifacts(
        final_pipeline,
        results,
        best_result,
        baseline,
        settings,
        model_path,
        metadata_path,
        shap_path,
        X,
        y,
    )


if __name__ == "__main__":
    app()
