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

# projet_05 : Déployez un modèle de Machine Learning

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

[![mkdocs-shield]][mkdocs-url]


## Organisation du projet

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         projet_05 and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── projet_05   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes projet_05 a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

## Code hérité réutilisé

- `scripts_projet04/brand` : charte graphique OpenClassrooms (classe `Theme`, palettes, YAML). Le module `projet_05/branding.py` en est la porte d'entrée et applique automatiquement le thème.
- `scripts_projet04/manet_projet04/shap_generator.py` : fonctions `shap_global` / `shap_local` utilisées par `projet_05/modeling/train.py` pour reproduire les visualisations SHAP.

## Base de données PostgreSQL

Depuis la branche `postgresql`, toute la fusion des fichiers bruts repose sur une base PostgreSQL accessible via SQLAlchemy.

1. Installez PostgreSQL (Homebrew, package officiel, etc.).
2. Créez un rôle et la base attendue :

> Exemple pour MacOS

   ```bash
   /opt/homebrew/opt/postgresql@17/bin/createuser -s postgres
   /opt/homebrew/opt/postgresql@17/bin/psql -d postgres -c "ALTER USER postgres WITH PASSWORD 'postgres';"
   /opt/homebrew/opt/postgresql@17/bin/createdb -O postgres projet05
   ```

   Adaptez les chemins/versions selon votre environnement.
3. Renseignez la chaîne de connexion dans `projet_05/settings.yml` :

   ```yaml
   database:
     url: postgresql+psycopg://user:password@host:5432/projet05
     schema: public
   ```

   Il est également possible de définir `PROJET05_DATABASE_URL` dans l'environnement.

4. Initialisez la base (création des tables + insertion des CSV d'exemple) avec :

   ```bash
   python -m scripts.init_db
   ```

5. Assurez-vous que l'utilisateur possède les droits `CREATE/DROP TABLE` dans le schéma ciblé : les tables `sirh`, `evaluation`, `sond` ainsi que `prediction_logs` seront créées ou recréées à chaque ré-exécution.

6. Lancez ensuite `python -m projet_05.dataset` comme auparavant (ou `python main.py` pour exécuter toutes les étapes). La requête SQL utilisée est toujours exportée dans `reports/merge_sql.sql` pour audit.

> Les interactions utilisateur/modèle (qu'elles proviennent du formulaire, du tableau ou d'un upload) sont automatiquement journalisées dans la table `prediction_logs`, ce qui permet de tracer les usages et de constituer un dataset réel pour le monitoring.

## Tests & couverture

Une batterie de tests Pytest valident l’intégrité de la base PostgreSQL, la fusion des données et la journalisation des prédictions.

1. Démarrez PostgreSQL (cf. section précédente) et créez un utilisateur ayant les droits `CREATE/DROP DATABASE`.
2. Facultatif : définissez `PROJET05_TEST_DATABASE_URL` si vous souhaitez utiliser une URL différente de `postgresql+psycopg://postgres:postgres@localhost:5432/projet05_test`.
3. Exécutez les tests et générez le rapport de couverture :

   ```bash
   pytest
   ```

   La configuration Pytest produit à la fois un rapport terminal (`--cov-report=term-missing`) et un fichier `coverage.xml` exploitable par vos outils CI/CD.
   Les sorties complètes sont sauvegardées dans `logs/tests_logs/<timestamp>.log`.

Les tests vérifient notamment :

- la création des tables `sirh`, `evaluation`, `sond`, `prediction_logs` et la cohérence du nombre de lignes insérées ;
- l’intégrité du DataFrame fusionné (typage, absence de valeurs nulles sur la clé primaire, cohérence de la cible) ;
- la robustesse du script de log des prédictions (insertion d’entrées dans `prediction_logs` et nettoyage) ;
- la génération des logs de pipeline, regroupés dans `logs/pipeline_logs/<timestamp>.log`.

--------

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Python][python]][python]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![project_license][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/stephmnt/OCR_Projet05/deploy.yml)

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/github_username/repo_name">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">project_title</h3>

  <p align="center">
    project_description
    <br />
    <a href="https://github.com/github_username/repo_name"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/github_username/repo_name">View Demo</a>
    &middot;
    <a href="https://github.com/github_username/repo_name/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/github_username/repo_name/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

Here's a blank template to get started. To avoid retyping too much info, do a search and replace with your text editor for the following: `github_username`, `repo_name`, `twitter_handle`, `linkedin_username`, `email_client`, `email`, `project_title`, `project_description`, `project_license`

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* [![Python][Python]][Python-url]
* [![SQL][SQL]][SQL-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

pip install -r requirements.txt
uvicorn app.main:app --reload

1. Clone the repo
   ```sh
   git clone https://github.com/stephmnt/OCR_Projet05.git
   ```
2. Install NPM packages
   ```sh
   npm install
   ```
3. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```
4. Change git remote url to avoid accidental pushes to base project
   ```sh
   git remote set-url origin github_username/repo_name
   git remote -v # confirm the changes
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [ ] Feature 1
- [ ] Feature 2
- [ ] Feature 3
    - [ ] Nested Feature

See the [open issues](https://github.com/stephmnt/OCR_projet05/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Top contributors:

<a href="https://github.com/github_username/repo_name/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=github_username/repo_name" alt="contrib.rocks image" />
</a>



<!-- LICENSE -->
## License

Distributed under the project_license. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email@email_client.com

Project Link: [https://github.com/github_username/repo_name](https://github.com/github_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/stephmnt/OCR_projet05.svg?style=for-the-badge
[contributors-url]: https://github.com/stephmnt/OCR_projet05/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/stephmnt/OCR_projet05.svg?style=for-the-badge
[forks-url]: https://github.com/stephmnt/OCR_projet05/network/members
[stars-shield]: https://img.shields.io/github/stars/stephmnt/OCR_projet05.svg?style=for-the-badge
[stars-url]: https://github.com/stephmnt/OCR_projet05/stargazers
[issues-shield]: https://img.shields.io/github/issues/stephmnt/OCR_projet05.svg?style=for-the-badge
[issues-url]: https://github.com/stephmnt/OCR_projet05/issues
[product-screenshot]: images/screenshot.png
[Noobie]: https://img.shields.io/badge/Data%20Science%20for%20Beginners-84CC16?style=for-the-badge&labelColor=E5E7EB&color=84CC16
<!-- Shields.io badges. You can a comprehensive list with many more badges at: https://github.com/inttter/md-badges -->
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
<!-- OK -->
[license-shield]: https://img.shields.io/github/license/stephmnt/OCR_projet05.svg?style=for-the-badge
[license-url]: https://github.com/stephmnt/OCR_Projet05/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/stephanemanet
<!-- TODO: -->
[postgres-shield]: https://img.shields.io/badge/Postgres-%23316192.svg?logo=postgresql&logoColor=white
[python-shield]: https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)
[mkdocs-shield]: https://img.shields.io/badge/MkDocs-526CFE?logo=materialformkdocs&logoColor=fff
[mkdocs-url]: https://stephmnt.github.io/OCR_Projet05/
[NumPy]: https://img.shields.io/badge/NumPy-4DABCF?logo=numpy&logoColor=fff
[![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=fff)](#)

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/stephmnt/OCR_Projet05/deploy.yml)
[![https://img.shields.io/badge/MkDocs-526CFE?logo=materialformkdocs&logoColor=fff]][[mkdocs-url](https://stephmnt.github.io/OCR_Projet05/)]
![GitHub Release Date](https://img.shields.io/github/release-date/stephmnt/OCR_Projet05?display_date=published_at&style=flat-square)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/stephmnt/OCR_Projet05/deploy.yml)
