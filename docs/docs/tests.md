# Tests et couverture

Ce projet embarque désormais une démarche de tests structurée autour de **Pytest** et de **pytest-cov** afin de valider l’intégrité du pipeline, en particulier la partie base de données PostgreSQL.

## Objectifs

- Garantir que la fusion et les manipulations SQL respectent la structure attendue (tables, colonnes, volumes).
- Vérifier que les transformations amont (`dataset.py`) et la journalisation des prédictions (`app.py`) restent robustes face aux évolutions.
- Offrir un rapport de couverture exploitable (terminal + `coverage.xml`) pour le suivi de la qualité.

## Organisation

- **Dépendances** : `pyproject.toml` inclut `pytest`, `pytest-cov` et la configuration des options (`--cov`, `--cov-report`).
- **Dossier `tests/`** :
  - `tests/conftest.py` : prépare une base PostgreSQL dédiée aux tests (`projet05_test`) en créant/ supprimant dynamiquement la base, en lançant `scripts.init_db`, puis en nettoyant les connexions via `pg_terminate_backend`.
  - `tests/test_database.py` :
    - vérifie la présence des tables (`sirh`, `evaluation`, `sond`, `prediction_logs`) et leur volumétrie par rapport aux CSV bruts ;
    - s’assure que la fusion `build_dataset` préserve typage, clé primaire et valeurs de la cible ;
    - teste `_log_predictions` pour confirmer que chaque interaction utilisateur est bien enregistrée.
  - `tests/test_data.py` : couvre les utilitaires de nettoyage (`clean_text_values`) et d’harmonisation des identifiants (`_harmonize_id_column`).

## Exécution

```bash
make test
```

Cette commande :
1. Démarre la base de test, initialise les tables avec les CSV (`scripts.init_db`).
2. Exécute l’ensemble des tests.
3. Génère les rapports de couverture (`term-missing` et `coverage.xml`).
4. Sauvegarde la sortie complète de Pytest dans `logs/tests_logs/<timestamp>.log` (créé automatiquement via `make test`).

Le rôle `postgres`/base `projet05_test` sont créés à la volée. Les tests nécessitent un serveur PostgreSQL local accessible via `postgresql+psycopg://postgres:postgres@localhost:5432`. Vous pouvez ajuster l’URL via `PROJET05_TEST_DATABASE_URL`.

## Pourquoi cette approche ?

- **Reproductibilité** : chaque test s’appuie sur des données d’exemple réalistes sans impacter l’environnement de prod.
- **Robustesse** : failover pandas dans `dataset.py` est testé implicitement, et la journalisation garantit la traçabilité des interactions.
- **Transparence** : la couverture met immédiatement en évidence les modules non testés (ex. `features.py`, `train.py`) pour prioriser les efforts, et les journaux sont regroupés sous `logs/tests_logs/`.