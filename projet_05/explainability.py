from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from loguru import logger

from projet_05.branding import Theme, apply_brand_theme, make_diverging_cmap
from scripts_projet04.manet_projet04.shap_generator import (  # type: ignore[import-not-found]
    shap_global,
    shap_local,
)

apply_brand_theme()


def _shape_array(values) -> np.ndarray:
    if hasattr(values, "values"):
        arr = np.array(values.values)
    else:
        arr = np.array(values)
    return np.nan_to_num(arr, copy=False)


def compute_shap_summary(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    max_samples: int = 500,
) -> Tuple[pd.DataFrame | None, object | None]:
    """
    Reuse the historical `shap_global` helper to build the plots and a tabular summary.

    Returns
    -------
    summary_df : pd.DataFrame | None
        Moyenne absolue des valeurs SHAP (ordre décroissant).
    shap_values : shap.Explanation | None
        Objet renvoyé par shap_global pour des analyses locales ultérieures.
    """
    cmap = make_diverging_cmap(Theme.PRIMARY, Theme.SECONDARY)
    shap_values, _, feature_names = shap_global(
        pipeline,
        X,
        y,
        sample_size=max_samples,
        cmap=cmap,
    )
    if shap_values is None or feature_names is None:
        logger.warning("Impossible de générer les résumés SHAP.")
        return None, None

    shap_array = _shape_array(shap_values)
    if shap_array.ndim == 1:
        shap_array = shap_array.reshape(-1, 1)
    mean_abs = np.abs(shap_array).mean(axis=0)
    summary = (
        pd.DataFrame({"feature": list(feature_names), "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return summary, shap_values


def save_shap_summary(summary: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    logger.info("Résumé SHAP sauvegardé dans {}", output_path)


def export_local_explanations(
    pipeline,
    shap_values,
    X: pd.DataFrame,
    custom_index: int | None = None,
) -> None:
    """
    Génère trois cas d'usage par défaut (impact max, risque max, risque min)
    et un indice custom optionnel pour la trace historique.
    """
    if shap_values is None:
        return

    shap_array = _shape_array(shap_values)
    idx_impact = int(np.argmax(np.sum(np.abs(shap_array), axis=1)))
    shap_local(idx_impact, shap_values)

    y_proba_all = pipeline.predict_proba(X)[:, 1]
    idx_highrisk = int(np.argmax(y_proba_all))
    shap_local(idx_highrisk, shap_values)

    idx_lowrisk = int(np.argmin(y_proba_all))
    shap_local(idx_lowrisk, shap_values, text_scale=0.6)

    if custom_index is not None:
        shap_local(custom_index, shap_values, max_display=8)


__all__ = ["compute_shap_summary", "save_shap_summary", "export_local_explanations"]
