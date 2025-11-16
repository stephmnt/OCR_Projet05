from __future__ import annotations

import importlib
import os

import pandas as pd
import pytest
from pandas.api.types import is_integer_dtype
from sqlalchemy import create_engine, inspect, text

import app as app_module
from projet_05.dataset import build_dataset
from projet_05.settings import load_settings


def test_tables_structure_and_counts(initialized_db, raw_rowcount):
    engine = create_engine(initialized_db, future=True)
    try:
        insp = inspect(engine)
        tables = set(insp.get_table_names())
        assert {"sirh", "evaluation", "sond", "prediction_logs"}.issubset(tables)

        sirh_cols = {col["name"] for col in insp.get_columns("sirh")}
        assert {"id_employee", "age", "revenu_mensuel", "poste"}.issubset(sirh_cols)

        with engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM sirh")).scalar_one()
        assert count == raw_rowcount
    finally:
        engine.dispose()


def test_dataset_build_preserves_data_integrity(initialized_db, raw_rowcount):
    os.environ["PROJET05_DATABASE_URL"] = initialized_db
    load_settings.cache_clear()
    settings = load_settings()
    df = build_dataset(settings)
    assert len(df) == raw_rowcount
    assert settings.col_id in df.columns
    assert df[settings.col_id].notna().all()
    assert is_integer_dtype(df[settings.col_id])
    assert df[settings.target].isin(["Oui", "Non", 0, 1]).all()


def test_prediction_logs_record_interactions(initialized_db):
    os.environ["PROJET05_DATABASE_URL"] = initialized_db
    load_settings.cache_clear()
    app = importlib.reload(app_module)

    raw = pd.DataFrame([{"id_employee": 9999, "age": 35}], index=[0])
    scored = pd.DataFrame(
        [
            {
                "id_employee": 9999,
                "proba_depart": 0.42,
                "prediction": 1,
            }
        ],
        index=[0],
    )
    app._log_predictions("pytest", raw, scored)

    engine = create_engine(initialized_db, future=True)
    try:
        with engine.begin() as conn:
            count = conn.execute(
                text("SELECT COUNT(*) FROM prediction_logs WHERE source = 'pytest'")
            ).scalar_one()
            assert count >= 1
            conn.execute(text("DELETE FROM prediction_logs WHERE source = 'pytest'"))
    finally:
        engine.dispose()
