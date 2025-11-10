from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
import typer

from projet_05.config import INTERIM_DATA_DIR
from projet_05.settings import Settings, load_settings

app = typer.Typer(help="Préparation et fusion des données sources.")


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------
def safe_read_csv(path: Path, *, dtype=None) -> pd.DataFrame:
    """Read a CSV file and return an empty frame when it fails."""
    try:
        logger.info("Lecture du fichier {}", path)
        return pd.read_csv(path, dtype=dtype)
    except FileNotFoundError:
        logger.warning("Fichier absent: {}", path)
        return pd.DataFrame()
    except Exception as exc:  # pragma: no cover - log + empty dataframe
        logger.error("Impossible de lire {} ({})", path, exc)
        return pd.DataFrame()


def clean_text_values(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize textual values that often materialize missing values."""
    replace_tokens = [
        "",
        " ",
        "  ",
        "   ",
        "nan",
        "NaN",
        "NAN",
        "None",
        "JE ne sais pas",
        "je ne sais pas",
        "Je ne sais pas",
        "Unknow",
        "Unknown",
        "non pertinent",
        "Non pertinent",
        "NON PERTINENT",
    ]
    normalized = df.copy()
    normalized = normalized.replace(replace_tokens, np.nan)

    for column in normalized.select_dtypes(include="object"):
        normalized[column] = (
            normalized[column].replace(replace_tokens, np.nan).astype("string").str.strip()
        )
    return normalized


def _harmonize_id_column(df: pd.DataFrame, column: str, *, digits_only: bool = True) -> pd.DataFrame:
    data = df.copy()
    if column not in data.columns:
        return data

    if digits_only:
        extracted = data[column].astype(str).str.extract(r"(\\d+)")
        data[column] = pd.to_numeric(extracted[0], errors="coerce")
    data[column] = pd.to_numeric(data[column], errors="coerce").astype("Int64")
    return data


def _rename_column(df: pd.DataFrame, source: str, target: str) -> pd.DataFrame:
    if source not in df.columns:
        return df
    return df.rename(columns={source: target})


def _log_id_diagnostics(df: pd.DataFrame, *, name: str, col_id: str) -> None:
    if col_id not in df.columns:
        logger.warning("La colonne {} est absente du fichier {}.", col_id, name)
        return
    total = len(df)
    uniques = df[col_id].nunique(dropna=True)
    duplicates = total - uniques
    logger.info(
        "{name}: {total} lignes | {uniques} identifiants uniques | {duplicates} doublons",
        name=name,
        total=total,
        uniques=uniques,
        duplicates=duplicates,
    )


def _persist_sql_trace(df_dict: dict[str, pd.DataFrame], settings: Settings) -> pd.DataFrame:
    """
    Reproduire la fusion SQL décrite dans le notebook.

    Chaque DataFrame est stocké dans une base SQLite éphémère pour
    conserver une traçabilité de la requête exécutée.
    """
    db_path = settings.db_file
    sql_path = settings.sql_file

    db_path.parent.mkdir(parents=True, exist_ok=True)
    sql_path.parent.mkdir(parents=True, exist_ok=True)

    if db_path.exists():
        db_path.unlink()

    query = f"""
    SELECT *
    FROM sirh
    INNER JOIN evaluation USING ({settings.col_id})
    INNER JOIN sond USING ({settings.col_id});
    """.strip()

    with db_path.open("wb") as _:
        pass  # just ensure the file exists for sqlite on some platforms

    with sqlite3.connect(db_path) as conn:
        for name, frame in df_dict.items():
            frame.to_sql(name, conn, index=False, if_exists="replace")
        merged = pd.read_sql_query(query, conn)

    sql_path.write_text(query, encoding="utf-8")
    return merged


def build_dataset(settings: Settings) -> pd.DataFrame:
    """Load, clean, harmonize and merge the three raw sources."""
    sirh = clean_text_values(
        safe_read_csv(settings.path_sirh).pipe(
            _harmonize_id_column, settings.col_id, digits_only=True
        )
    )
    evaluation = clean_text_values(
        safe_read_csv(settings.path_eval)
        .pipe(_rename_column, "eval_number", settings.col_id)
        .pipe(_harmonize_id_column, settings.col_id, digits_only=True)
    )
    sond = clean_text_values(
        safe_read_csv(settings.path_sondage)
        .pipe(_rename_column, "code_sondage", settings.col_id)
        .pipe(_harmonize_id_column, settings.col_id, digits_only=True)
    )

    for name, frame in {"sirh": sirh, "evaluation": evaluation, "sond": sond}.items():
        _log_id_diagnostics(frame, name=name, col_id=settings.col_id)

    frames = {
        "sirh": sirh,
        "evaluation": evaluation,
        "sond": sond,
    }
    merged = _persist_sql_trace(frames, settings)

    missing_cols = [settings.col_id] if settings.col_id not in merged.columns else []
    if missing_cols:
        raise KeyError(
            f"La colonne {settings.col_id} est absente de la fusion finale. "
            "Vérifiez vos fichiers sources."
        )

    logger.success("Fusion réalisée: {} lignes / {} colonnes", *merged.shape)
    return merged


def save_dataset(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.success("Fichier fusionné sauvegardé dans {}", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@app.command()
def main(
    settings_path: Path = typer.Option(
        None,
        "--settings",
        "-s",
        help="Chemin vers un fichier settings.yml personnalisé.",
    ),
    output_path: Path = typer.Option(
        INTERIM_DATA_DIR / "merged.csv",
        "--output",
        "-o",
        help="Chemin de sortie du dataset fusionné.",
    ),
):
    """Entrypoint Typer pour reproduire la fusion des données brutes."""

    settings = load_settings(settings_path) if settings_path else load_settings()
    df = build_dataset(settings)
    save_dataset(df, output_path)


if __name__ == "__main__":
    app()
