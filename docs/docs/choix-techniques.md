# Choix d’architecture et justification

Ce document résume les décisions structurantes du projet. Il complète les sections techniques du `README.md` et sert de support lors de la soutenance pour expliquer *pourquoi* chaque brique a été sélectionnée, comment elle répond aux contraintes métier et quelles alternatives ont été écartées.

## 1. Interface utilisateur : Gradio vs FastAPI

- **Besoin métier** : fournir rapidement une interface interactive aux RH (formulaire, tableau éditable, import CSV et fusion automatique des trois fichiers bruts).
- **Choix retenu — Gradio** (`app.py`):
  - Offre des composants prêts à l’emploi (tabs, `Gradio.DataFrame`, upload) indispensables pour démontrer le produit aux sponsors sans développement front spécifique.
  - Gère l’hébergement Hugging Face Spaces en natif (fichier `README.md:131-149`, workflow `deploy.yml`). L’authentification côté Space peut être activée via les paramètres HF (mot de passe, private Space) en attendant une couche custom.
  - Documentation intégrée directement dans l’UI et via MkDocs (`docs/docs/app.md`).
- **Alternative étudiée — FastAPI** :
  - Avantage : maîtrise fine des endpoints REST, intégration simple avec des frontends custom ou des SI internes.
  - Inconvénient pour ce projet : nécessité de développer un front supplémentaire pour la démo finale et un effort plus important pour reproduire l’expérience multi-onglets.
  - **Stratégie retenue** : garder Gradio pour la soutenance/HF et prévoir une exposition REST via FastAPI uniquement si un client SI la demande (le module `projet_05/modeling/predict.py` est déjà découplé et réutilisable dans un service FastAPI ou Celery).

## 2. Stockage et orchestration des données : PostgreSQL

- **Contraintes** : fusion de trois exports SIRH/évaluation/sondage, besoin d’historiser les interactions du modèle, typage strict pour préserver la qualité des features.
- **Choix retenu — PostgreSQL** :
  - Supporte nativement les contraintes d’intégrité et les jointures complexes exploitées dans `projet_05/dataset.py` et `scripts/init_db.py`.
  - Compatible SQLAlchemy, ce qui permet d’utiliser le même code côté tests (`tests/conftest.py`, `tests/test_database.py`) et côté application (`app.py` pour `prediction_logs`).
  - Installation reproductible décrite dans `README.md:72-101`, y compris pour la base de test `projet05_test`.
- **Alternatives** :
  - SQLite : insuffisant pour tester les scénarios multi-utilisateurs et la journalisation (pas de parallélisme, pas de rôle/permission).
  - NoSQL (ex. MongoDB) : moins adaptées au besoin d’analytics structurés et aux jointures fortes entre les trois tables.
- **Ouverture** : le module `dataset.py` implémente un fallback pandas lorsque la connexion PostgreSQL est indisponible (utile sur Hugging Face Spaces), tout en gardant la trace SQL dans `reports/merge_sql.sql`.

## 3. CI/CD et automatisation

- **Objectifs** : garantir la reproductibilité (pipeline d’entraînement, génération des artefacts) et automatiser les déploiements vers Hugging Face + GitHub Pages.
- **Pipeline retenu** :
  - `main.py` orchestre localement les étapes data → features → entraînement → packaging.
  - Workflow `.github/workflows/deploy.yml` :
    1. Installe les dépendances (`requirements.txt` + `pip install -e .`).
    2. Lance `python main.py` pour régénérer base, features et modèle.
    3. Synchronise le dépôt vers la Space `stephmnt/projet_05` via `HF_TOKEN`.
  - Workflow `.github/workflows/static.yml` publie la documentation MkDocs (`docs/site`) sur GitHub Pages.
- **Justification** :
  - GitHub Actions évite l’hébergement d’un runner custom et est aligné avec le dépôt public.
  - La préparation des données avant déploiement garantit que les artefacts (`models/`, `data/processed/schema.json`) sont cohérents avec l’application Gradio, limitant les erreurs en production.
  - Les logs générés (`logs/pipeline_logs`, `logs/tests_logs`, `prediction_logs`) facilitent les audits et nourrissent les tableaux de bord analytiques (`README.md:198-211`).
  - Séparation claire des environnements : `main.py` s’exécute dans un contexte **test** (base `projet05_test`, variables `PROJET05_TEST_DATABASE_URL`) pour valider l’ensemble du pipeline, tandis que `app.py` tourne en **production** (Space Hugging Face + `PROJET05_DATABASE_URL` ou fallback pandas) afin de servir les utilisateurs finaux.
- **Améliorations prévues** :
  - Ajouter un smoke test API (ex. appel Gradio via `gradio_client`) pour valider l’URL Hugging Face après chaque déploiement.
  - Coupler les secrets GitHub Actions/Hugging Face avec une stratégie de rotation documentée dans la future section “Sécurité & Authentification”.

## 4. Stratégie Git

- **Organisation des branches** :
  - `main` : branche stable, protégée par les workflows `deploy.yml` et `static.yml`.
  - Trois branches fonctionnelles ont été créées au fil de la mission, chacune dédiée à une étape majeure (`tests`, `postgresql`, `doc`). Chacune a été mergée dans `main` une fois la fonctionnalité validée.
  - Des branches ponctuelles `feature/*` ou `fix/*` peuvent encore être ouvertes pour de futures évolutions (ex. `feature/api-security`, `fix/logging-timezone`).
- **Conventions de nommage** :
  - Préfixes explicites (`feature/`, `fix/`, `docs/`) suivis d’un identifiant court décrivant l’objet du travail.
  - Messages de commit au format `type(scope): résumé` (ex. `feat(app): log prediction payload`), facilitant la lecture du `git log --oneline`.
- **Releases & tags** :
  - Chaque jalon correspond à une présentation (mentorat OpenClassrooms) ou à la soutenance finale. La release actuelle (`v1.1.1`) marque la présentation intermédiaire ; la prochaine accompagnera la soutenance.
  - Les tags `vX.Y.Z` suivent les fonctionnalités livrées (intégration PostgreSQL, campagne de tests, documentation). Chaque release génère les artefacts (modèle, doc) et déclenche le déploiement HF, ce qui facilite le retour arrière en cas de regression.

## Synthèse

- L’ensemble de ces choix permet de livrer rapidement une démo fidèle aux attentes métier, tout en conservant des briques suffisamment modulaires pour évoluer (exposition REST possible, moteur de base interchangeable via SQLAlchemy).
- Ces arguments peuvent être repris lors de la soutenance pour justifier la cohérence entre contraintes (time-to-market, transparence, exigences data) et solutions retenues.
