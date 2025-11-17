# Guide de démarrage rapide

Ce document détaille toutes les étapes pour exécuter le projet OCR_Projet05 (préparation des données, entraînement, application Gradio et CI/CD). Les instructions sont valables sur macOS, Linux et Windows (PowerShell ou WSL).

## 1. Prérequis

| Outil | Version recommandée | Notes |
| --- | --- | --- |
| Python | 3.11 | Utilise `pyenv`, `conda` ou l’installateur officiel. |
| PostgreSQL | ≥ 14 (tests) / 17 (prod locale) | Le rôle `postgres/postgres` doit disposer des droits `CREATE DATABASE`. |
| Git | ≥ 2.30 | Pour cloner le dépôt et pousser vers GitHub/Hugging Face. |
| Make (optionnel) | ≥ 4 | Simplifie l’exécution de `make test`, `make data`, etc. |
| Poetry (optionnel) | 1.8.x | Alternative à `pip install -e .`. |

> Sur Windows, privilégier WSL2 ou Docker pour PostgreSQL. Pour un serveur distant, adaptez simplement l’URL `postgresql+psycopg://`.

## 2. Récupérer le dépôt

```bash
git clone https://github.com/stephmnt/OCR_Projet05.git
cd OCR_Projet05
```

Pour suivre les bonnes pratiques décrites dans `docs/docs/choix-techniques.md`, crée une branche de travail :

```bash
git checkout -b feature/<mon-sujet>
```

## 3. Préparer l’environnement Python

### 3.1. Virtualenv via `venv`

```bash
python3.11 -m venv .venv
source .venv/bin/activate            # sous Windows : .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e .                     # dépendances dev (pyproject)
pip install -r requirements.txt      # dépendances runtime Hugging Face
```

### 3.2. Option Poetry

```bash
poetry env use 3.11
poetry install
```

Les scripts `make` utiliseront automatiquement l’interpréteur Python actif. Pense à activer l’environnement à chaque session.

## 4. Configuration du projet

1. Copie `projet_05/settings.yml` si besoin vers un fichier spécifique et renseigne les chemins de fichiers bruts (`data/raw/` par défaut).
2. Exporte les variables d’environnement si nécessaire :

```bash
export PROJET05_DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/projet05"
export PROJET05_TEST_DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/projet05_test"
export PROJET05_SETTINGS="/chemin/vers/mon_settings.yml"   # optionnel
```

Pour la prod (Hugging Face, CI/CD), configure également `HF_TOKEN` (secret GitHub Actions) et les variables `PROJET05_DATABASE_URL`, `PROJET05_SETTINGS` côté Space.

## 5. Initialiser PostgreSQL

1. Création du rôle et de la base locale (exemple Homebrew macOS) :

```bash
/opt/homebrew/opt/postgresql@17/bin/createuser -s postgres
/opt/homebrew/opt/postgresql@17/bin/psql -d postgres -c "ALTER USER postgres WITH PASSWORD 'postgres';"
/opt/homebrew/opt/postgresql@17/bin/createdb -O postgres projet05
```

2. Peupler les tables avec les CSV `data/raw/extrait_*.csv` :

```bash
python -m scripts.init_db
# ou
make data
```

Le script crée les tables `sirh`, `evaluation`, `sond`, `prediction_logs` puis insère les données brutes. Vérifie les logs générés dans `logs/pipeline_logs`.

## 6. Pipeline data & entraînement

### 6.1. Pipeline complet

```bash
python main.py
```

Cette commande :

1. Recrée la base (`scripts.init_db`).
2. Fusionne les sources via `projet_05/dataset.py`.
3. Applique les features (`projet_05/features.py`) et exporte `data/processed/dataset.csv` + `schema.json`.
4. Entraîne le modèle (`projet_05/modeling/train.py`), sauvegarde `models/best_model.joblib` et `models/best_model_meta.json`.

### 6.2. Étapes unitaires

```bash
python -m projet_05.dataset          # génère data/interim/merged.csv
python -m projet_05.features         # génère data/processed/dataset.csv
python -m projet_05.modeling.train   # met à jour models/
python -m projet_05.modeling.predict --features data/processed/dataset.csv
```

Le module `predict` produit un CSV `data/processed/predictions.csv` utile pour valider l’inférence CLI.

## 7. Lancer l’application Gradio

```bash
python app.py
```

- La console affiche l’URL locale (ex. `http://127.0.0.1:7860`).
- Les logs utilisateur sont enregistrés dans la table `prediction_logs` si la variable `PROJET05_DATABASE_URL` est définie.
- Pour exécuter sur Hugging Face Spaces :
  1. Crée une Space “Gradio”.
  2. Pousse `app.py`, `models/`, `data/processed/schema.json`, `requirements.txt`.
  3. Renseigne `HF_TOKEN` si tu utilises GitHub Actions (`.github/workflows/deploy.yml`).

## 8. Tests et qualité

### 8.1. Tests Pytest

```bash
make test
# ou
pytest
```

- La fixture `tests/conftest.py` crée la base `projet05_test`.
- Les logs pytest sont stockés dans `logs/tests_logs/<timestamp>.log`.
- Un rapport de couverture `coverage.xml` est généré (paramétré dans `pyproject.toml`).

### 8.2. Lint / formatage

```bash
make lint
make format
```

Ces commandes utilisent Ruff pour vérifier le style et formater automatiquement.

## 9. Documentation MkDocs

La documentation (celle que tu lis) est générée par MkDocs :

```bash
pip install mkdocs mkdocs-mermaid2-plugin
mkdocs serve
```

Le site statique est publié via `.github/workflows/static.yml` (GitHub Pages). Ajoute les nouveaux fichiers sous `docs/docs/` et mets à jour `docs/mkdocs.yml` si nécessaire.

## 10. CI/CD & déploiement Hugging Face

1. Configure les secrets GitHub `HF_TOKEN` (espace de destination) et éventuellement `PROJET05_DATABASE_URL`.
2. Push sur `main` → déclenchement du workflow `deploy.yml` :
   - `pip install -r requirements.txt` + `pip install -e .`
   - `python main.py`
   - Synchronisation du dépôt vers `https://huggingface.co/spaces/stephmnt/projet_05`
3. La documentation est publiée via `static.yml`.

## 11. Dépannage rapide

| Symptom | Résolution |
| --- | --- |
| `OperationalError: connection refused` | Vérifie que PostgreSQL écoute sur `localhost:5432` et que l’URL dans `settings.yml` est correcte. |
| `FileNotFoundError: data/raw/...` | Les CSV ne sont pas présents : récupère les extraits fournis ou mets à jour `paths` dans `settings.yml`. |
| `Gradio Error: Configuration introuvable` | Définit `PROJET05_SETTINGS` ou place `projet_05/settings.yml` dans le dépôt lors du déploiement HF. |
| `pytest` skip la base | Assure-toi que `PROJET05_TEST_DATABASE_URL` est accessible et que l’utilisateur peut créer/supprimer des bases. |

---

Tu es maintenant prêt·e à préparer les données, entraîner le modèle, lancer l’UI Gradio et déployer automatiquement sur Hugging Face. Pense à consulter `docs/docs/choix-techniques.md` pour expliquer tes choix lors de la soutenance et `correction.md` pour suivre l’état d’avancement des livrables.
