---
title: OCR_Projet05
emoji: 🔥
colorFrom: purple
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: true
short_description: Projet 05 formation Openclassrooms
python_version: 3.11
---

# OCR Projet 05 – Prédiction d’attrition

[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/stephmnt/OCR_Projet05/deploy.yml)](https://github.com/stephmnt/OCR_Projet05/actions/workflows/deploy.yml)
[![GitHub Release Date](https://img.shields.io/github/release-date/stephmnt/OCR_Projet05?display_date=published_at&style=flat-square)](https://github.com/stephmnt/OCR_Projet05/releases)
[![project_license](https://img.shields.io/github/license/stephmnt/OCR_projet05.svg)](https://github.com/stephmnt/OCR_Projet05/blob/main/LICENSE)
[![MkDocs](https://img.shields.io/badge/MkDocs-526CFE?logo=materialformkdocs&logoColor=fff)](https://stephmnt.github.io/OCR_Projet05/)
[![Cookie Cutter](https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter)](https://cookiecutter-data-science.drivendata.org/)

Ce dépôt contient le projet OCR_Projet05. Il s’agit d’une application Gradio déployable sur Hugging Face Spaces, alimentée par un pipeline de préparation de données, un entraînement automatique et des services d’inférence orientés RH (prédiction de départ d’employés).  

Ce document décrit :

- la **présentation fonctionnelle** ;
- les **instructions d’installation, d’utilisation et de déploiement** (local + Hugging Face) ;
- le **processus de stockage/gestion des données** (PostgreSQL + journaux) ;
- les **besoins analytiques** (tableaux de bord, métriques clés).

---

## 1. Vue d’ensemble du projet

- **Objectif métier** : détecter les employés à risque de départ en exploitant 3 sources brutes (SIRH, évaluation, sondage interne).
- **Technologie** : pipeline Python (Typer, pandas, scikit-learn, SQLAlchemy) + application Gradio (`app.py`) déployée sur Hugging Face.
- **Modèle** : pipeline scikit-learn (prétraitement + classifieur) sérialisé dans `models/best_model.joblib`, paramétré avec un seuil de décision optimisé (visible dans l’UI).
- **Journaux** : deux sous-dossiers `logs/pipeline_logs` et `logs/tests_logs` contiennent respectivement les traces du pipeline `main.py` et les sorties Pytest.

Arborescence clé :

```
├── projet_05/                # Package Python principal
├── app.py                    # Interface Gradio (déploiement HF)
├── scripts/init_db.py        # Création/initialisation PostgreSQL
├── main.py                   # Orchestrateur du pipeline local
├── docs/                     # Documentation MkDocs + tests.md
├── tests/                    # Suite Pytest (DB + intégration)
└── requirements.txt          # Dépendances runtime (HF)
```

---

## 2. Installation locale

### 2.1. Prérequis

1. Python 3.11 (virtualenv ou Poetry recommandé).
2. PostgreSQL (>= 17) accessible localement (cf. instructions DB plus bas).
3. Outils optionnels : `make`, `pip`, `pytest`.

### 2.2. Étapes

```bash
git clone https://github.com/stephmnt/OCR_Projet05.git
cd OCR_Projet05
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # pour HF
pip install -e .                    # pour le développement local (pyproject)
```

### 2.3. Configuration PostgreSQL

```bash
/opt/homebrew/opt/postgresql@17/bin/createuser -s postgres
/opt/homebrew/opt/postgresql@17/bin/psql -d postgres -c "ALTER USER postgres WITH PASSWORD 'postgres';"
/opt/homebrew/opt/postgresql@17/bin/createdb -O postgres projet05
```

Puis dans `projet_05/settings.yml` :

```yaml
database:
  url: postgresql+psycopg://postgres:postgres@localhost:5432/projet05
  schema: public
```

> Sur une autre infrastructure, adaptez l’URL ou utilisez `PROJET05_DATABASE_URL`.

---

## 3. Utilisation du pipeline

### 3.1. Initialiser la base

```bash
python -m scripts.init_db
```

Création des tables `sirh`, `evaluation`, `sond`, `prediction_logs` + insertion des CSV bruts situés dans `data/raw`.

### 3.2. Pipeline complet

```bash
python main.py
```

- `main.py` enchaîne :
  1. Initialisation base PostgreSQL (`scripts.init_db`)
  2. Préparation des données (`projet_05.dataset`)
  3. Feature engineering (`projet_05.features`)
  4. Entraînement (`projet_05.modeling.train`)
- Les logs sont regroupés dans `logs/pipeline_logs/<timestamp>.log`.

### 3.3. Application Gradio locale

```bash
python app.py
```

L’interface propose :

- onglet **Formulaire** ;
- **Tableau interactif** ;
- **Upload CSV** ;
- **Fichiers non-mergés** (chargement des trois CSV bruts, fusion automatique).
Un appel à `_log_predictions` trace chaque prédiction dans PostgreSQL (`prediction_logs`).

---

## 4. Déploiement sur Hugging Face

### 4.1. Dépendances

`requirements.txt` contient toutes les bibliothèques nécessaires à la Space (Gradio, scikit-learn, pandas, SQLAlchemy, psycopg…).

### 4.2. Étapes

1. Créer une Space Gradio (Python 3.11).
2. Copier `app.py`, `requirements.txt`, `models/`, `data/processed/schema.json`.
3. Configurer les secrets HF (si besoin de variables d’environnement).
4. Optionnel : définir `HUGGINGFACEHUB_API_TOKEN` pour automatiser les déploiements via GitHub Actions.

### 4.3. Spécificités Space

- Hugging Face n’expose pas PostgreSQL. L’application Gradio bascule alors sur le mode **pandas fallback** (fusion locale) grâce à la gestion d’erreur de `dataset.py`.
- Les journaux restants sont ceux générés par l’appli (pas d’écriture dans `logs/` côté Space).

---

## 5. Processus de stockage & gestion des données

### 5.1. Sources

- `data/raw/extrait_sirh.csv`
- `data/raw/extrait_eval.csv`
- `data/raw/extrait_sondage.csv`

### 5.2. Base relationnelle

Tables PostgreSQL créées par `scripts.init_db` :

| Table | Rôle | Colonnes clés |
| --- | --- | --- |
| `sirh` | Profil RH structuré | `id_employee`, `age`, `revenu_mensuel`, `poste`, etc. |
| `evaluation` | Historique d’évaluations | `id_employee`, `note_evaluation_actuelle`, `niveau_hierarchique_poste`, `satisfaction_*` |
| `sond` | Sondage + cible | `id_employee`, `a_quitte_l_entreprise`, `distance_domicile_travail`, `domaine_etude`, etc. |
| `prediction_logs` | Journal d’inférence | `log_id`, `created_at`, `id_employee`, `source`, `probability`, `decision`, `payload` JSON |

`projet_05.dataset` fusionne `sirh ∩ evaluation ∩ sond` via SQL ; en cas d’indisponibilité DB, la fusion pandas est utilisée en repli.

### 5.3. Journaux et tracing

- `logs/pipeline_logs` : sorties `main.py`
- `logs/tests_logs` : sorties Pytest (`make test`)
- `prediction_logs` : base PostgreSQL, indispensable pour l’audit des décisions ML.

---

## 6. Tests et couverture

### 6.1. Exécution

```bash
pytest
```

- La fixture `initialized_db` crée une base `projet05_test`, lance `scripts.init_db`, puis la supprime.
- Les logs Pytest sont stockés dans `logs/tests_logs/<timestamp>.log`.

### 6.2. Couverture

- Rapports `term-missing` + `coverage.xml`.
- Zones non couvertes : `features.py`, `modeling/train.py`, `explainability.py` (à prioriser si besoin).

---

## 7. Besoins analytiques / tableaux de bord

- **Dashboard RH** basé sur les journaux `prediction_logs` :
  - Volume de prédictions par source (Formulaire / CSV / Raw).
  - Répartition des scores (`proba_depart`) / seuil de décision.
  - Historique des décisions (tendance du taux de risque).
  - Drill-down par attributs (`departement`, `poste`, `genre`…).
- **Monitoring modèle** :
  - Taux d’utilisation (logs journaliers).
  - Drift potentiel : comparer les distributions des features avec `docs/` (notebooks d’analyse) ou via un outil externe.
- **KPI Data/IT** :
  - Latence d’inférence (calculable via timestamps, si ajoutés).
  - Suivi des erreurs (logs pipeline/tests).

---

## 8. Choix techniques et justification

Ce projet combine une interface Gradio, une base PostgreSQL et un pipeline CI/CD GitHub Actions. Les décisions d’architecture détaillant le pourquoi/du comment (Gradio vs FastAPI, choix de PostgreSQL, automatisations) sont regroupées dans [`docs/docs/choix-techniques.md`](docs/docs/choix-techniques.md). Cette section sert de support de soutenance pour rappeler :

- pourquoi Gradio a été privilégié pour la démonstration Hugging Face ;
- comment PostgreSQL sécurise la fusion des trois sources et la journalisation ;
- en quoi les workflows GitHub Actions garantissent un déploiement fiable.
- comment les environnements sont configurés : `main.py` est exécuté en environnement **test** (base `projet05_test`, variables `PROJET05_TEST_DATABASE_URL`) pour valider le pipeline complet, tandis que `app.py` tourne en **production** (Space Hugging Face, variable `PROJET05_DATABASE_URL`/fallback pandas) afin de servir les utilisateurs finaux.

## 9. Instructions rapides

| Action | Commande |
| --- | --- |
| Init DB + pipeline complet | `python main.py` |
| Lancer Gradio local | `python app.py` |
| Initialiser la base seule | `python -m scripts.init_db` |
| Lancer les tests + logs | `make test` |
| Déployer sur Hugging Face | Pousser `app.py`, `requirements.txt`, `models/`, config Space |

---

## 10. Licence / références

Ce projet est fourni dans le cadre de la formation OpenClassrooms.  
La documentation complémentaire est disponible dans `docs/` (MkDocs + `docs/docs/tests.md` pour les tests).  
Pour toute question : [LinkedIn](https://linkedin.com/in/stephanemanet).
