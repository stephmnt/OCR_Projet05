"""Scripts utilitaires pour créer et remplir la base PostgreSQL."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    text,
)
import typer

from projet_05 import dataset as ds
from projet_05.settings import Settings, load_settings

app = typer.Typer(help="Initialisation complète de la base PostgreSQL.")


def _build_metadata(settings: Settings) -> MetaData:
    """Définir le schéma SQLAlchemy (tables et colonnes) aligné sur nos CSV."""

    metadata = MetaData(schema=settings.db_schema)

    Table(
        "sirh",
        metadata,
        Column("id_employee", Integer, primary_key=True),
        Column("age", Float),
        Column("genre", String(16)),
        Column("revenu_mensuel", Float),
        Column("statut_marital", String(32)),
        Column("departement", String(64)),
        Column("poste", String(64)),
        Column("nombre_experiences_precedentes", Float),
        Column("nombre_heures_travailless", Float),
        Column("annee_experience_totale", Float),
        Column("annees_dans_l_entreprise", Float),
        Column("annees_dans_le_poste_actuel", Float),
    )

    Table(
        "evaluation",
        metadata,
        Column("id_employee", Integer, primary_key=True),
        Column("satisfaction_employee_environnement", Float),
        Column("note_evaluation_precedente", Float),
        Column("niveau_hierarchique_poste", Float),
        Column("satisfaction_employee_nature_travail", Float),
        Column("satisfaction_employee_equipe", Float),
        Column("satisfaction_employee_equilibre_pro_perso", Float),
        Column("eval_number", String(64)),
        Column("note_evaluation_actuelle", Float),
        Column("heure_supplementaires", String(8)),
        Column("augementation_salaire_precedente", String(32)),
    )

    Table(
        "sond",
        metadata,
        Column("id_employee", Integer, primary_key=True),
        Column("a_quitte_l_entreprise", String(8)),
        Column("nombre_participation_pee", Float),
        Column("nb_formations_suivies", Float),
        Column("nombre_employee_sous_responsabilite", Float),
        Column("code_sondage", String(64)),
        Column("distance_domicile_travail", Float),
        Column("niveau_education", Float),
        Column("domaine_etude", String(64)),
        Column("ayant_enfants", String(8)),
        Column("frequence_deplacement", String(32)),
        Column("annees_depuis_la_derniere_promotion", Float),
        Column("annes_sous_responsable_actuel", Float),
    )

    Table(
        "prediction_logs",
        metadata,
        Column("log_id", Integer, primary_key=True, autoincrement=True),
        Column("created_at", DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP")),
        Column("id_employee", Integer),
        Column("source", String(32)),
        Column("probability", Float),
        Column("decision", Integer),
        Column("threshold", Float),
        Column("payload", Text),
    )

    return metadata


def _load_frames(settings: Settings) -> dict[str, pd.DataFrame]:
    """Charger les trois CSV bruts (sirh, évaluation, sondage) déjà nettoyés."""

    sirh = ds.clean_text_values(
        ds.safe_read_csv(settings.path_sirh).pipe(ds._harmonize_id_column, settings.col_id, digits_only=True)
    )
    evaluation = ds.clean_text_values(
        ds.safe_read_csv(settings.path_eval)
        .pipe(ds._rename_column, "eval_number", settings.col_id)
        .pipe(ds._harmonize_id_column, settings.col_id, digits_only=True)
    )
    sond = ds.clean_text_values(
        ds.safe_read_csv(settings.path_sondage)
        .pipe(ds._rename_column, "code_sondage", settings.col_id)
        .pipe(ds._harmonize_id_column, settings.col_id, digits_only=True)
    )
    return {"sirh": sirh, "evaluation": evaluation, "sond": sond}


@app.command()
def main(
    settings_path: Path = typer.Option(
        None,
        "--settings",
        "-s",
        help="Chemin vers un fichier settings.yml personnalisé.",
    )
):
    """Créer les tables PostgreSQL et charger les données d'exemple."""

    Path("logs").mkdir(parents=True, exist_ok=True)
    settings = load_settings(settings_path) if settings_path else load_settings()
    if not settings.db_url:
        raise typer.BadParameter(
            "Aucune URL de base de données fournie. Configurez `database.url` dans settings.yml."
        )

    engine = create_engine(settings.db_url, future=True)
    metadata = _build_metadata(settings)

    with engine.begin() as conn:
        if settings.db_schema:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {settings.db_schema}"))
        metadata.drop_all(conn, checkfirst=True)
        metadata.create_all(conn, checkfirst=True)

    frames = _load_frames(settings)
    with engine.begin() as conn:
        for table_name, frame in frames.items():
            logger.info("Insertion de {} lignes dans la table {}", len(frame), table_name)
            frame.to_sql(
                table_name,
                conn,
                schema=settings.db_schema,
                index=False,
                if_exists="append",
                method="multi",
            )

    logger.success("Initialisation PostgreSQL terminée.")


if __name__ == "__main__":
    app()
