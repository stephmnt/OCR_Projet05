from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Generator

import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError

import scripts.init_db as init_db
from projet_05.settings import load_settings


@pytest.fixture(scope="session")
def test_db_url() -> str:
    """URL de connexion PostgreSQL utilisée pendant les tests."""

    return os.getenv(
        "PROJET05_TEST_DATABASE_URL",
        "postgresql+psycopg://postgres:postgres@localhost:5432/projet05_test",
    )


@pytest.fixture(scope="session")
def initialized_db(test_db_url: str) -> Generator[str, None, None]:
    """Crée une base dédiée aux tests, exécute scripts.init_db puis la met à disposition."""

    url = make_url(test_db_url)
    admin_url = url.set(database="postgres")
    try:
        admin_engine = create_engine(admin_url, future=True)
        with admin_engine.execution_options(isolation_level="AUTOCOMMIT").connect() as conn:
            conn.execute(text(f"DROP DATABASE IF EXISTS {url.database}"))
            conn.execute(text(f"CREATE DATABASE {url.database}"))
    except OperationalError as exc:
        pytest.skip(f"Base PostgreSQL de test indisponible: {exc}")
    os.environ["PROJET05_DATABASE_URL"] = test_db_url
    load_settings.cache_clear()
    init_db.main(settings_path=None)
    yield test_db_url
    load_settings.cache_clear()
    with admin_engine.execution_options(isolation_level="AUTOCOMMIT").connect() as conn:
        conn.execute(
            text(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = :db AND pid <> pg_backend_pid()"
            ),
            {"db": url.database},
        )
        conn.execute(text(f"DROP DATABASE IF EXISTS {url.database}"))


@pytest.fixture(scope="session")
def raw_rowcount() -> int:
    """Nombre de lignes attendues dans les CSV bruts."""

    source = Path("data/raw/extrait_sirh.csv")
    return pd.read_csv(source).shape[0]
