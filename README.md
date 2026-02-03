# 💼 Prédicteur de Salaires - Métiers de la Data

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green.svg)](https://xgboost.readthedocs.io/)
[![Tests](https://img.shields.io/badge/Tests-99%20passing-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/Coverage-70%25-yellow.svg)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> Ce projet analyse 5 868 offres d’emploi Data collectées sur HelloWork en janvier 2026, afin d’estimer le salaire annuel brut à partir de 29 variables explicatives. Le modèle XGBoost retenu atteint un R² de 0.337 et une erreur absolue moyenne de 5 163 €. 

Accéder à l’application en ligne:
🔗 Application Streamlit : https://projetmachinelearning.streamlit.app/

![Page d'accueil](images/gift_acceuil.gif)

---

## 📋 Table des matières

- [Description du projet](#-description-du-projet)
- [Contexte et problématique](#-contexte-et-problématique)
- [Démarche et objectifs](#-démarche-et-objectifs)
- [Méthodologie](#-méthodologie)
- [Résultats du Machine Learning](#-résultats-du-machine-learning)
- [Utilisation de l'application](#-utilisation-de-lapplication)
- [Installation](#-installation)
- [Structure du projet](#-structure-du-projet)
- [Technologies utilisées](#-technologies-utilisées)
- [Axes d'amélioration](#-axes-damélioration)
- [Auteurs](#-auteurs)
- [License](#-license)

---

##  Description du projet

Ce projet vise à **prédire les salaires** des métiers de la Data en France en exploitant des données  d'offres d'emploi non structurées. À travers une application web interactive développée avec **Streamlit**, il offre trois fonctionnalités principales :

1. ** Prédiction de salaire** - Estimation salariale pour un profil donné,
2. ** Analyse de marché** - Vue d'ensemble des tendances observées dans les données,
3. ** Conseil carrière** - simulation de l’impact de certains choix professionnels.

Les résultats reflètent les tendances présentes dans le dataset. Le modèle a un R² de 0.337 et une erreur absolue moyenne de 5 163 €. Ces limites sont affichées pour contextualiser les estimations proposées.

> Ce modèle ne prétend pas déterminer la "valeur" d’un candidat, mais reflète les tendances observées dans un corpus spécifique d’offres d’emploi. Il ne doit pas être utilisé comme unique référence pour des décisions de recrutement ou de négociation salariale.

### Question centrale de recherche

> **Quels facteurs observables dans les offres d’emploi expliquent la variance des salaires dans le domaine Data en France ?**

---

##  Contexte et problématique

Le marché de l'emploi dans la Data se distingue par une **évolution rapide des technologies** et une **transformation profonde des métiers**. Aujourd'hui, un profil "Data" ne se définit plus uniquement par son intitulé de poste ou son parcours académique, mais par une **combinaison précise de compétences techniques**, allant de la maîtrise de langages de programmation aux environnements Cloud.

### Défis identifiés

Cette complexité structurelle crée un marché de l'emploi très actif où :

-  **Les références de rémunération fluctuent en permanence**
-  **L'orientation professionnelle devient complexe** pour les futurs diplômés
-  **Les informations sont dispersées** dans des milliers d'annonces hétérogènes
-  **L'évaluation de sa propre valeur sur le marché** est particulièrement délicate

### Nécessité du projet

Dans ce contexte de **forte volatilité**, il existe une difficulté réelle à évaluer l'impact concret de chaque critère sur le salaire proposé. Il devient donc nécessaire de **transformer ces données brutes en informations structurées** afin de comprendre comment les recruteurs valorisent réellement :

- Une expertise technique (Python, SQL, Cloud, ML/DL)
- Des facteurs traditionnels (localisation, expérience, formation)
- Les synergies entre compétences

---

##  Démarche et objectifs

### Objectifs de l'étude

Ce travail est structuré autour des axes suivants :

1. ** Vérifier la capacité de prédiction**
   - Tester si l'extraction de variables techniques permet de faire converger un modèle vers une estimation salariale cohérente

2. ** Analyser la hiérarchie des signaux**
   - Mesurer le poids relatif de l'expertise technique face aux déterminants géographiques traditionnels

3. ** Étudier les synergies entre compétences**
   - Identifier les combinaisons créant des sauts de valeur non-linéaires

4. ** Cartographier la distribution des opportunités**
   - À partir de données extraites de HelloWork.com

5. ** Fournir des indicateurs concrets**
   - Pour l'orientation et l'évaluation de profil des futurs candidats

### Périmètre de l'étude

L'analyse porte sur un échantillon de **5,868 offres d'emploi** collectées via web scraping.

**Métiers ciblés** :
- Data Scientist
- Data Engineer
- Data Analyst
- BI Analyst
- ML Engineer

**Variables explicatives** :
- Stack technique (Python, SQL, R, Cloud, BI tools)
- Expérience requise
- Localisation géographique
- Secteur d'activité
- Niveau de formation
- Avantages sociaux

---

##  Méthodologie

### 1. Web Scraping

**Source** : HelloWork.com  
**Période** : Janvier 2026  
**Volume** : 5,868 offres d'emploi

```python
# Données collectées
- Titre du poste
- Description complète
- Fourchette salariale
- Localisation
- Compétences requises
- Avantages sociaux
```

**Outils** : Botright, Requests, Pandas

### 2. Nettoyage et Feature Engineering

**Pipeline de traitement** :
-  Nettoyage des descriptions 
-  Extraction d'expérience (patterns regex + correction des valeurs extrêmes)
-  Parsing de compétences (12+ technologies détectées)
-  Normalisation des salaires (support k€ et €)
- Création de 101 colonnes au total, dont 29 utilisées comme features d’entrée

**Architecture** :
```python
ExperienceExtractor      # Extraction  d'expérience
CompanyExtractor        # Détection d'entreprises
LocationExtractor       # Géolocalisation
JobTypeClassifier       # Classification de postes
DescriptionCleaner      # Nettoyage de texte
```

### 3. Modélisation Machine Learning

**Approche** :
- Split stratifié (80% train / 20% test)
- Cross-validation 5-fold
- GridSearchCV pour l'optimisation
- Prévention stricte de l'overfitting

**Modèles testés** :
1. Ridge / Lasso / ElasticNet (régularisation L1/L2)
2. Random Forest (min_samples_leaf=20)
3. Gradient Boosting (subsample=0.8)
4. **XGBoost**  (modèle retenu)
5. LightGBM (feature_fraction=0.7)

---

##  Résultats du Machine Learning

### Modèle retenu : XGBoost

**Performances** :

| Métrique | Train | Test | Cross-Validation |
|----------|-------|------|------------------|
| **R²** | 0.451 | **0.337** | 0.315 (±0.028) |
| **MAE** | 4,328€ | **5,163€** | 5,421€ (±315€) |
| **RMSE** | 6,547€ | **7,854€** | 8,012€ (±421€) |

**Taux d'erreur moyen** : ~11% du salaire prédit

### Précision par marge d'erreur

| Marge | % de prédictions correctes |
|-------|---------------------------|
| **±5%** | 23.4% |
| **±10%** | 47.8% |
| **±15%** | 68.2% |
| **±20%** | 82.5% |

### Features les plus importantes

**Top 10 variables explicatives** :

1. **Expérience** (années) - 18.2%
2. **Score technique global** - 14.7%
3. **Nombre de compétences** - 12.3%
4. **Localisation** (Paris vs Province) - 9.8%
5. **Secteur d'activité** - 8.4%
6. **Compétences Cloud** (AWS/Azure/GCP) - 7.2%
7. **Séniorité** (Junior/Mid/Senior) - 6.9%
8. **Machine Learning / Deep Learning** - 5.8%
9. **Type de contrat** (CDI/CDD/Freelance) - 4.6%
10. **Formation** (Bac+3/5/8) - 4.1%

### Diagnostic d'overfitting

**ΔR² (Train - Test)** : 0.114 → ✅ **Overfitting maîtrisé**

**Stratégies de régularisation appliquées** :
- `max_depth=3` (limitation de profondeur)
- `min_child_weight=10` (samples minimum par feuille)
- `reg_alpha=5.0` (régularisation L1)
- `reg_lambda=10.0` (régularisation L2)
- `subsample=0.7` (bagging d'échantillons)
- `colsample_bytree=0.7` (bagging de features)

---

##  Utilisation de l'application

### Page d'accueil

> _« Synthèse des tendances observées dans les 5 868 offres analysées, avec métriques du modèle et navigation vers les fonctionnalités principales. »_

![Accueil](images/gift_acceuil.gif)

**Fonctionnalités** : 

- 📊 Vue d'ensemble du marché (4,253 postes Data analysés)
- 💰 Salaire médian par type de poste
- 🔥 Top compétences les plus demandées
- 📈 Répartition géographique des offres

---

### 1. 🔮 Module Prédiction

> _«Estimation salariale basée sur le modèle XGBoost, accompagnée d’une fourchette réaliste (±7 417 €) et d’un positionnement par rapport au marché réel.»_

![Prédiction](images/gift_pred.gif)

**Fonctionnalités** :

- **Formulaire de profil** : Type de poste, expérience, compétences, localisation
- **Prédiction instantanée** : Salaire estimé avec intervalle de confiance (±MAE)
- **Explicabilité SHAP** : Contribution de chaque variable à la prédiction
- **Positionnement marché** : Votre salaire vs la distribution du marché

![Prédiction détaillée](images/gift_pred_02.gif)

**Comparaisons avancées** :
-  **Par secteur** : Tech, Banque, ESN, Assurance, etc.
-  **Par ville** : Paris, Lyon, Toulouse, Bordeaux, etc.
- ⏱ **Projection carrière** : Évolution salariale sur 10 ans
-  **Impact des compétences** : Gain salarial par skill (+Python, +AWS, +ML/DL)

---

### 2. Module Marché

 > _«Exploration interactive des données : filtres par poste, ville, secteur ou compétences, avec visualisations des distributions et impacts salariaux observés.»_

![Marché](images/gift_marche.gif)

**Onglets d'analyse** :

** Vue d'ensemble**
- Distribution des salaires (histogramme + boxplot)
- Salaire médian par type de contrat
- Évolution salaire vs expérience

** Jobs & Secteurs**
- Top 10 métiers Data les mieux payés
- Salaires par secteur d'activité
- Multiplicateurs sectoriels (Tech: +8%, Banque: +12%)

** Géographie**
- Top 10 villes par salaire moyen
- Heatmap France (salaires moyens par région)
- Multiplicateurs géographiques (Paris: +15%)

** Compétences**
- Fréquence des compétences (Python: 68%, SQL: 72%)
- Impact salarial par compétence (+Python: +3.2k€, +AWS: +5.8k€)

** Combinaisons**
- Stacks techniques populaires (Python+SQL+Cloud, etc.)
- ROI des combinaisons (gains salariaux)

---

### 3. Module Carrière

> _« Simulation de l’impact de décisions concrètes (ex. : localisation, acquisition de compétences) et projection de trajectoire fondée sur la distribution réelle des salaires par expérience.. »_

![Carrière](images/gift_carriere.gif)

**Fonctionnalités** :

**📊 Diagnostic de positionnement**
- Score d'employabilité (0-100)
- Positionnement vs marché (percentile)
- Gain de compétence optimal

**🗺️ Roadmap de compétences**
- Identification des compétences manquantes
- Calcul de l'impact salarial (+Python: +3.2k€)
- Matrice Effort/Impact pour prioriser

**🔄 Transitions de carrière**
- Top 3 transitions possibles (ex: Data Analyst → Data Scientist)
- Compétences requises pour chaque transition
- Gain salarial estimé (+12k€ en moyenne)

**📈 Projection salariale**
- Évolution sur 10 ans (3 scénarios)
- Graphique interactif

---

##  Installation

### Prérequis

- **Python** 3.13+
- **pip** (gestionnaire de paquets)

### Installation rapide

```bash
# 1. Cloner le repository
git clone https://github.com/votre-username/predicteur-salaires-data.git
cd predicteur-salaires-data

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv

# Sur Windows
venv\Scripts\activate

# Sur macOS/Linux
source venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'application
streamlit run 01_Accueil.py
```

L'application sera accessible à l'adresse : **http://localhost:8501**

---

##  Structure du projet

```
Projet_Salaires_Data/
│
├── 01_Accueil.py                    #  Page d'accueil Streamlit
│
├── pages/                            #  Pages de l'application
│   ├── 01_Prediction.py             #  Module prédiction
│   ├── 02_Marche.py                 #  Module marché
│   └── 03_Carriere.py               #  Module carrière
│
├── internal/                         #   Implémentations internes
│   ├── prediction_display_impl.py
│   ├── prediction_comparisons_impl.py
│   ├── prediction_actions_impl.py
│   ├── career_*.py
│   └── market_*.py
│
├── utils/                            #  Utilitaires
│   ├── config.py
│   ├── model_utils.py
│   └── feature_engineer.py
│
├── models/                           #  Modèles ML
│   ├── best_model_XGBoost.pkl
│   └── modeling_report_v7.json
│
├── scripts/                          #  Scripts de traitement
│   ├── data_cleaning_refactored_part1.py
│   ├── data_cleaning_refactored_part2.py
│   └── modeling_refactored.py
│
├── tests/                            #  Tests (99 tests)
│   ├── test_model_utils.py
│   ├── test_modeling_refactored.py
│   └── test_simplified.py
│
├── images/                           #   GIFs de démonstration
│   ├── gift_acceuil.gif
│   ├── gift_pred.gif
│   ├── gift_marche.gif
│   └── gift_carriere.gif
│
└── requirements.txt                  #  Dépendances
```

---

##  Technologies utilisées

### Frontend
- **Streamlit** 1.31 - Interface web
- **Plotly** 5.18 - Graphiques interactifs
- **Matplotlib/Seaborn** - Visualisations

### Machine Learning
- **XGBoost** 2.0 - Modèle principal
- **LightGBM** 4.1 - Modèle alternatif
- **scikit-learn** 1.3 - Preprocessing
- **SHAP** 0.44 - Explicabilité

### Data Processing
- **Pandas** 2.1 - Manipulation de données
- **NumPy** 1.26 - Calculs numériques

### Testing
- **pytest** 7.4 - Tests unitaires
- **pytest-cov** 4.1 - Couverture

---

##  Axes d'amélioration

### Court terme
- [ ] Ajout de sources de données (Indeed, LinkedIn)
- [ ] Amélioration du modèle (R² >0.40)
- [ ] Mode sombre et export PDF

### Moyen terme
- [ ] API REST pour prédictions
- [ ] Système de recommandations de formations
- [ ] Déploiement cloud (AWS/Streamlit Cloud)

### Long terme
- [ ] NLP avancé (BERT, GPT)
- [ ] Prédiction d'évolution du marché
- [ ] Plateforme collaborative

---

## 👥 Auteurs

**Emmanuel Paguiel**
- 🎓 Etudiant en Economie de l'entreprise


---

## 🙏 Remerciements

- **HelloWork.com** pour les données
- **Communauté Streamlit**
- **Anthropic Claude** pour l'assistance

---

## 📄 License

MIT License - Copyright (c) 2026 Emmanuel

---
