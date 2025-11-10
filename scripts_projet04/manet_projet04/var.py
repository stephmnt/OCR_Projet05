"""
Chargement et nettoyage des paramètres du projet à partir d'un fichier YAML.
La fonction load_settings attend les variables "NUM_COLS", "CAT_COLS", "SAT_COLS", "FIRST_VARS"
sous forme de listes de chaînes.
"""

from pathlib import Path
from typing import Any, List, Optional, Dict
import yaml


# Dossier contenant ce module (utile pour retrouver settings.yml)
PKG_DIR = Path(__file__).resolve().parent


def _clean_list(lst: Optional[Any]) -> List[str]:
    """
    Nettoie une liste de chaînes provenant du YAML.

    - S'assure que la valeur renvoyée est une liste.
    - Supprime les guillemets parasites en début/fin.
    - Retire les apostrophes orphelines à la fin d'une valeur.

    Args:
        lst: Liste ou valeur unique à nettoyer.

    Returns:
        Liste nettoyée de chaînes.
    """
    if lst is None:
        return []
    if not isinstance(lst, (list, tuple)):
        return [str(lst)]

    cleaned = []
    for value in lst:
        s = str(value).strip()

        # Retire les guillemets doubles ou simples encadrants
        if (
            (s.startswith('"') and s.endswith('"'))
            or (s.startswith("'") and s.endswith("'"))
        ):
            s = s[1:-1]

        # Supprime un éventuel guillemet simple final
        if s.endswith("'"):
            s = s[:-1]

        cleaned.append(s)
    return cleaned


def load_settings(yaml_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Charge le fichier settings.yml et retourne un dictionnaire de paramètres.

    Si aucun chemin n'est fourni, le fichier par défaut est :
    `manet_projet04/settings.yml`.

    Args:
        yaml_path: Chemin explicite du fichier YAML (optionnel).

    Returns:
        dict: Dictionnaire des paramètres du projet.
    """
    path = Path(yaml_path) if yaml_path else PKG_DIR / "settings.yml"

    with path.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file) or {}

    # Normalise certaines clés connues
    for key in ("NUM_COLS", "CAT_COLS", "SAT_COLS", "FIRST_VARS"):
        if key in cfg:
            cfg[key] = _clean_list(cfg[key])

    # Conversion sécurisée du sous-échantillon
    if "SUBSAMPLE_FRAC" in cfg:
        try:
            cfg["SUBSAMPLE_FRAC"] = float(cfg["SUBSAMPLE_FRAC"])
        except (TypeError, ValueError):
            pass

    return cfg