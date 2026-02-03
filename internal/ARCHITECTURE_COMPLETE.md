# 📘 Guide Complet du Projet - Prédicteur de Salaires Data Jobs

> **Version 2.1** - Février 2026  
> Application Streamlit d'estimation salariale basée sur 5,868 offres HelloWork

---

## 📋 Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Architecture du projet](#architecture-du-projet)
3. [Structure des fichiers](#structure-des-fichiers)
4. [Modules principaux](#modules-principaux)
5. [Configuration et déploiement](#configuration-et-déploiement)
6. [Guide de développement](#guide-de-développement)
7. [Dépannage](#dépannage)
8. [Feuille de route](#feuille-de-route)

---

## 🎯 Vue d'ensemble

### Objectif du projet

Application web interactive permettant d'estimer les salaires dans les métiers de la Data à partir d'un profil utilisateur, basée sur l'analyse de **5,868 offres d'emploi** collectées sur HelloWork en janvier 2026.

### Fonctionnalités principales

1. **🔮 Prédiction salariale** : Estimation personnalisée basée sur profil, localisation, compétences
2. **📊 Analyse du marché** : Visualisations interactives, tendances, comparaisons
3. **🎓 Feuille de route carrière** : Roadmap personnalisée, projections salariales, transitions de rôle
4. **💡 Insights dynamiques** : Multiplicateurs salariaux calculés en temps réel depuis les données

### Technologies utilisées

| Catégorie | Technologies |
|-----------|-------------|
| **Frontend** | Streamlit 1.31.0, Plotly 5.18.0, Matplotlib 3.8.2 |
| **ML/Data** | XGBoost 2.0.3, Scikit-learn 1.3.2, Pandas 2.1.4, NumPy 1.26.2 |
| **Viz avancée** | Seaborn 0.13.0, SHAP 0.44.0 |
| **Tests** | Pytest 7.4.3, Coverage 4.1.0 |
| **Cloud** | Streamlit Cloud, GitHub |

### Métriques du modèle

```
Modèle       : XGBoost v7 optimisé
Dataset      : 2,681 échantillons Data (train), 5,868 total
R²           : 0.337
MAE          : 5,163 €
RMSE         : 6,969 €
Précision    : 73.7% (±15%), 83.8% (±20%)
Stabilité    : 0.995 (cross-validation)
Overfitting  : 0.140 (contrôlé)
```

---

## 🏗️ Architecture du projet

### Architecture globale

```
┌─────────────────────────────────────────────────────────────┐
│                     STREAMLIT FRONTEND                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Accueil    │  │  Prédiction  │  │    Marché    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Carrière   │  │    Debug     │  │   (Autres)   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    COUCHE LOGIQUE                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   internal/  │  │    utils/    │  │   scripts/   │     │
│  │  (modules)   │  │  (config)    │  │  (nettoyage) │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    COUCHE DONNÉES                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  output/     │  │   models/    │  │    data/     │     │
│  │  (dataset)   │  │  (XGBoost)   │  │  (scripts)   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Flux de données

```
Données brutes (HelloWork)
         │
         ▼
Nettoyage (scripts/data_cleaning*.py)
         │
         ▼
Dataset enrichi (output/hellowork_cleaned_complete.csv)
         │
         ▼
Feature Engineering (utils/feature_engineer.py)
         │
         ▼
Modèle XGBoost (models/best_model_XGBoost_fixed.pkl)
         │
         ▼
API Prédiction (utils/model_utils.py)
         │
         ▼
Interface Streamlit (pages/*.py)
         │
         ▼
Utilisateur final
```

---

## 📁 Structure des fichiers

### Arborescence complète

```
Projet_machine_learning/
│
├── 01_Accueil.py                      # Page d'accueil principale ⭐
│
├── pages/                             # Pages Streamlit
│   ├── 01_Prediction.py              # Prédiction salariale
│   ├── 02_Marche.py                  # Analyse du marché
│   ├── 03_Carriere.py                # Feuille de route carrière
│   └── 99_Debug.py                   # Outils de debug
│
├── internal/                          # Modules de logique métier
│   ├── prediction_display_impl.py    # Affichage prédiction
│   ├── prediction_action_impl.py     # Actions prédiction
│   ├── prediction_comparisons.py     # Comparaisons
│   ├── market_analysis_impl.py       # Analyses marché
│   ├── market_filters_impl.py        # Filtres marché
│   ├── market_export_impl.py         # Export données
│   ├── career_analysis.py            # Analyse carrière
│   ├── career_roadmap.py             # Roadmap compétences
│   ├── career_transitions.py         # Transitions rôles
│   └── career_export.py              # Export carrière
│
├── utils/                             # Utilitaires
│   ├── config.py                     # Configuration centrale ⚙️
│   ├── model_utils.py                # Gestion modèle ML
│   └── feature_engineer.py           # Feature engineering
│
├── models/                            # Modèles ML
│   ├── best_model_XGBoost_fixed.pkl  # Modèle XGBoost v7
│   ├── test_data.pkl                 # Données de test
│   └── modeling_report_v7.json       # Rapport performance
│
├── output/                            # Données nettoyées
│   ├── hellowork_cleaned_complete.csv # Dataset principal (52 MB)
│   ├── test_data.pkl                 # Données test
│   └── analysis_complete/            # Analyses complètes
│       └── modeling_v7_improved/
│           └── modeling_report_v7.json
│
├── data/                              # Scripts de collecte
│   └── hellowork_scraper.py          # Web scraper
│
├── scripts/                           # Scripts d'analyse
│   ├── data_cleaning_step*.py        # Nettoyage (5 étapes)
│   └── modeling_refactored.py        # Entraînement modèle
│
├── tests/                             # Tests unitaires
│   └── test_*.py                     # 99 tests (73% coverage)
│
├── docs/                              # Documentation
│   ├── ARCHITECTURE.md               # Ce fichier
│   ├── GUIDE_GITHUB.md               # Guide Git/GitHub
│   └── GUIDE_FIX_STREAMLIT_CLOUD.md  # Dépannage déploiement
│
├── images/                            # Assets visuels
│   ├── gift_accueil.gif
│   ├── gift_pred.gif
│   ├── gift_pred_02.gif
│   ├── gift_marche.gif
│   └── gift_carriere.gif
│
├── .streamlit/                        # Config Streamlit
│   └── config.toml
│
├── requirements.txt                   # Dépendances Python
├── .gitignore                         # Exclusions Git
├── .gitattributes                     # Attributs Git (LFS)
├── README.md                          # Documentation principale
└── DEBUG_PATHS.py                     # Outil de debug chemins
```

### Statistiques du projet

```
Fichiers Python        : ~45
Lignes de code         : ~9,500
Tests                  : 99 (73% coverage)
Taille dataset         : 52 MB (5,868 offres)
Taille modèle          : 67 KB
Pages Streamlit        : 4
Modules internes       : 10
```

---

## 🧩 Modules principaux

### 1. Configuration (`utils/config.py`)

**Rôle** : Configuration centralisée de l'application

```python
class Config:
    # Chemins dynamiques
    BASE_DIR = Path(__file__).parent.parent
    DATA_PATH = BASE_DIR / "output" / "hellowork_cleaned_complete.csv"
    MODEL_PATH = BASE_DIR / "models" / "best_model_XGBoost_fixed.pkl"
    
    # Métriques modèle
    MODEL_INFO = {
        'r2_score': 0.337,
        'mae': 5163,
        'precision_15': 73.7
    }
    
    # Énumérations
    JOB_TYPES = ["Data Analyst", "Data Scientist", ...]
    CITIES = ["Paris", "Lyon", "Toulouse", ...]
    SECTORS = ["Tech", "Banque", "Finance", ...]
    
    # Multiplicateurs dynamiques
    @classmethod
    def get_city_multiplier(cls, city: str) -> float:
        """Calcule multiplicateur salarial par ville"""
        ...
```

**Fonctionnalités clés** :
- ✅ Chemins de fichiers dynamiques (compatible Streamlit Cloud)
- ✅ Multiplicateurs calculés depuis le dataset en temps réel
- ✅ Cache des valeurs pour performances
- ✅ Exports pour tous les modules

---

### 2. Utilitaires ML (`utils/model_utils.py`)

**Rôle** : Gestion du modèle XGBoost et calculs ML

```python
class ModelUtils:
    """Gestionnaire du modèle XGBoost"""
    
    def predict(self, profile: Dict) -> Dict:
        """Prédiction salariale"""
        features = self._prepare_features(profile)
        prediction = self.model.predict(features)
        return {
            'prediction': float(prediction),
            'confidence': self._calculate_confidence(features),
            'shap_values': self._get_shap_values(features)
        }
    
    def get_real_market_data(self) -> np.ndarray:
        """Données salariales du marché réel"""
        ...

class CalculationUtils:
    """Calculs statistiques et utilitaires"""
    
    @staticmethod
    def get_percentile_real(salary: float, market_data: np.ndarray) -> float:
        """Calcule le percentile d'un salaire"""
        ...
    
    @staticmethod
    def calculate_skills_count_from_profile(skills: Dict) -> int:
        """Compte les compétences d'un profil"""
        ...

class ChartUtils:
    """Création de graphiques Plotly"""
    
    @staticmethod
    def create_salary_gauge(prediction: float, median: float, ...) -> go.Figure:
        """Jauge de positionnement salarial"""
        ...
```

**Fonctionnalités** :
- ✅ Chargement et gestion du modèle XGBoost
- ✅ Prédictions avec intervalles de confiance
- ✅ Analyse SHAP (explicabilité)
- ✅ Calculs de percentiles et statistiques
- ✅ Génération de graphiques Plotly

---

### 3. Feature Engineering (`utils/feature_engineer.py`)

**Rôle** : Transformation des données brutes en features ML

```python
class FeatureEngineer:
    """Ingénierie des features pour le modèle"""
    
    def prepare_features(self, raw_profile: Dict) -> np.ndarray:
        """
        Transforme un profil utilisateur en features ML
        
        Steps:
        1. Extraction features de base
        2. Encoding catégorielles (one-hot)
        3. Scaling numériques (robust scaler)
        4. Features dérivées
        """
        features = self._extract_base_features(raw_profile)
        features = self._encode_categorical(features)
        features = self._scale_numerical(features)
        features = self._add_derived_features(features)
        return features
    
    def _extract_base_features(self, profile: Dict) -> Dict:
        """Extraction des features de base"""
        return {
            'experience_final': profile['experience'],
            'location_final': profile['location'],
            'sector_clean': profile['sector'],
            'skills_count': self._count_skills(profile),
            'technical_score': self._calculate_tech_score(profile),
            ...
        }
```

**Features gérées** (29 au total) :
- Numériques : expérience, salaire, nombre de compétences
- Catégorielles : type de poste, ville, secteur, niveau d'études
- Binaires : télétravail, avantages, compétences spécifiques
- Dérivées : score technique, complexité, mots-clés

---

### 4. Pages Streamlit

#### 📄 **01_Accueil.py**

```python
def main():
    """Page d'accueil avec métriques et navigation"""
    
    # Initialisation
    config, model_utils = initialize_app()
    data = load_application_data()
    
    # Sidebar
    render_sidebar(data, config)
    
    # Hero section avec CTA
    render_hero_section(config)
    
    # Métriques clés (4 colonnes)
    render_key_metrics(data, config)
    
    # Méthodologie (4 étapes)
    render_methodology_section()
    
    # Visualisations
    render_salary_distribution(data['test_salaries'])
    render_top_jobs(data['dataset'])
    
    # Navigation (3 cards)
    render_navigation_cards()
```

**Widgets avec clés uniques** :
- `sidebar_btn_report` : Bouton rapport
- `hero_btn_prediction` : CTA principal
- `nav_btn_prediction`, `nav_btn_market`, `nav_btn_career` : Navigation

---

#### 🔮 **pages/01_Prediction.py**

```python
def main():
    """Page de prédiction salariale"""
    
    # Initialisation
    model_utils, real_market_data, market_stats = initialize_prediction_page()
    
    # Formulaire de profil
    profile_data = render_prediction_form()
    
    if profile_data:
        # Prédiction
        result = model_utils.predict(profile_data)
        
        # Affichage résultats
        render_results(model_utils, real_market_data, market_stats)
        
        # Sections :
        # 1. Résultat principal + confiance
        # 2. Positionnement marché (jauge + percentile)
        # 3. Distribution marché
        # 4. Analyse SHAP (top 10 features)
        # 5. Comparaisons (secteur, expérience, ville)
        # 6. Impact des compétences
```

**Modules utilisés** :
- `prediction_display_impl.py` : Affichage
- `prediction_action_impl.py` : Actions (export, reset)
- `prediction_comparisons.py` : Comparaisons

---

#### 📊 **pages/02_Marche.py**

```python
def main():
    """Page d'analyse du marché Data"""
    
    # Chargement données
    market_data = load_market_data()
    
    # Filtres sidebar
    filtered_data, filters_info = render_sidebar_filters(market_data)
    
    # Insights (3 colonnes)
    render_key_insights(filtered_data)
    
    # Onglets d'analyse (6 tabs)
    tabs = st.tabs([
        "🔍 Vue d'ensemble",
        "💼 Postes & Secteurs",
        "🌍 Géographie",
        "🛠️ Compétences",
        "🔗 Combinaisons",
        "📊 Benchmark"
    ])
    
    # Export et navigation
    render_export_and_navigation(filtered_data, total_size, filters_info)
```

**Modules utilisés** :
- `market_filters_impl.py` : 8 filtres avec clés uniques
- `market_analysis_impl.py` : Graphiques et analyses
- `market_export_impl.py` : Export CSV/JSON + 4 boutons navigation

---

#### 🎓 **pages/03_Carriere.py**

```python
def main():
    """Feuille de route carrière personnalisée"""
    
    # Initialisation
    model_utils, df_final, real_market_data, market_median = initialize_career_page()
    
    # Formulaire profil (18 widgets avec clés)
    profile_data = render_profile_form()
    
    if profile_data:
        # Prédiction de base
        base_salary, percentile, base_pred = process_career_profile(...)
        
        # Analyses (8 sections)
        render_scorecard(...)                    # Scorecard 4 métriques
        render_positioning_diagnosis(...)        # Diagnostic positionnement
        render_roadmap_section(...)              # Roadmap compétences
        render_effort_impact_matrix(...)         # Matrice effort/impact
        render_transitions_analysis(...)         # Transitions de rôle
        render_salary_projection(...)            # Projection 10 ans (3 scénarios)
        render_negotiation_simulator(...)        # Simulateur négociation
        render_export_section(...)               # Export PDF/JSON
```

**Modules utilisés** :
- `career_analysis.py` : Scorecard et diagnostic
- `career_roadmap.py` : Roadmap et matrice
- `career_transitions.py` : Transitions et projections
- `career_export.py` : Négociation et export

**Clés uniques des widgets** (18 total) :
```python
# Section professionnelle (6)
"career_job_type", "career_experience", "career_location"
"career_sector", "career_education", "career_telework"

# Compétences (11)
"career_skill_python", "career_skill_sql", "career_skill_r"
"career_skill_tableau", "career_skill_powerbi", "career_skill_aws"
"career_skill_azure", "career_skill_spark", "career_skill_ml"
"career_skill_dl", "career_skill_etl"

# Submit (pas de clé - géré automatiquement)
```

---

## ⚙️ Configuration et déploiement

### Configuration locale

#### 1. Installation

```bash
# Cloner le repository
git clone https://github.com/Paguy-Stream/Projet_machine_learning.git
cd Projet_machine_learning

# Créer environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Installer dépendances
pip install -r requirements.txt
```

#### 2. Structure des données requise

```
Projet_machine_learning/
├── models/
│   └── best_model_XGBoost_fixed.pkl  ← OBLIGATOIRE (67 KB)
└── output/
    └── hellowork_cleaned_complete.csv ← OBLIGATOIRE (52 MB)
```

#### 3. Lancer l'application

```bash
streamlit run 01_Accueil.py
```

L'application s'ouvre sur `http://localhost:8501`

---

### Déploiement Streamlit Cloud

#### 1. Préparation du repository

```bash
# Vérifier que les fichiers critiques sont trackés
git ls-files | grep -E "(\.pkl|hellowork_cleaned_complete\.csv)"

# Si manquants, les ajouter (même si dans .gitignore)
git add -f models/best_model_XGBoost_fixed.pkl
git add -f output/hellowork_cleaned_complete.csv
git add -f output/test_data.pkl

git commit -m "Add critical data files for deployment"
git push origin main
```

#### 2. Configuration Streamlit Cloud

1. Aller sur https://share.streamlit.io/
2. Cliquer "New app"
3. Sélectionner :
   - Repository : `Paguy-Stream/Projet_machine_learning`
   - Branch : `main`
   - Main file : `01_Accueil.py`
4. Advanced settings :
   - Python version : `3.13`
5. Deploy !

#### 3. Fichiers de configuration

**`.streamlit/config.toml`** :
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
enableCORS = false
enableXsrfProtection = true
```

**`requirements.txt`** :
```txt
streamlit==1.31.0
pandas==2.1.4
numpy==1.26.2
scikit-learn==1.3.2
xgboost==2.0.3
lightgbm==4.1.0
shap==0.44.0
plotly==5.18.0
matplotlib==3.8.2
seaborn==0.13.0
python-dateutil==2.8.2
openpyxl==3.1.2
joblib==1.3.2
beautifulsoup4==4.12.2
requests==2.31.0
pytest==7.4.3
pytest-cov==4.1.0
pytest-mock==3.12.0
```

**Important** : 
- ❌ Ne PAS inclure `scipy` (Streamlit Cloud l'installe automatiquement)
- ❌ Ne PAS inclure `statsmodels` (incompatibilité)

---

### Variables d'environnement

Aucune variable d'environnement requise. Tous les chemins sont dynamiques via `Path(__file__).parent.parent`.

---

## 👨‍💻 Guide de développement

### Convention de nommage

#### Fichiers
```
Pages Streamlit    : 01_Accueil.py, 02_Marche.py (PascalCase)
Modules internes   : market_analysis_impl.py (snake_case)
Utilitaires        : config.py, model_utils.py (snake_case)
Tests              : test_model_utils.py (préfixe test_)
```

#### Code Python
```python
# Classes : PascalCase
class ModelUtils:
    pass

# Fonctions : snake_case
def calculate_percentile():
    pass

# Constantes : SCREAMING_SNAKE_CASE
MAX_SALARY = 150000

# Variables : snake_case
user_profile = {}

# Clés Streamlit : {page}_{section}_{type}_{purpose}
key="market_filter_salary"
key="career_skill_python"
key="nav_btn_prediction"
```

---

### Bonnes pratiques Streamlit

#### 1. Clés uniques obligatoires

```python
# ✅ BON
st.selectbox("Ville", options, key="market_filter_city")
st.button("Analyser", key="career_btn_analyze")

# ❌ MAUVAIS (cause des erreurs removeChild)
st.selectbox("Ville", options)  # Pas de clé
st.button("Analyser")  # Pas de clé
```

#### 2. Formulaires

```python
# ✅ BON
with st.form("my_form", clear_on_submit=False):
    city = st.selectbox("Ville", options, key="form_city")
    submit = st.form_submit_button("Soumettre")  # PAS de key ici
    
# ❌ MAUVAIS
with st.form("my_form"):
    submit = st.form_submit_button("Soumettre", key="btn_submit")  # Erreur !
```

#### 3. Cache

```python
# Cache données
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

# Cache ressources (modèles)
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

# Clear cache
st.cache_data.clear()
```

#### 4. Session state

```python
# Initialisation
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

# Utilisation
if st.button("Prédire"):
    st.session_state.prediction_made = True
    st.session_state.last_result = result
```

---

### Tests

#### Lancer les tests

```bash
# Tous les tests
pytest

# Avec coverage
pytest --cov=. --cov-report=html

# Test spécifique
pytest tests/test_model_utils.py

# Verbose
pytest -v
```

#### Écrire un test

```python
import pytest
from utils.model_utils import CalculationUtils

def test_calculate_percentile():
    """Test du calcul de percentile"""
    market_data = np.array([30000, 40000, 50000, 60000, 70000])
    result = CalculationUtils.get_percentile_real(50000, market_data)
    assert 45 <= result <= 55  # Tolérance

def test_predict_with_mock(mocker):
    """Test avec mock"""
    mocker.patch('utils.model_utils.ModelUtils.predict', return_value={
        'prediction': 50000,
        'confidence': 0.85
    })
    # ... rest of test
```

---

### Git workflow

```bash
# 1. Créer une branche
git checkout -b feature/nouvelle-fonctionnalite

# 2. Développer + commit fréquents
git add .
git commit -m "feat: Add new feature"

# 3. Push
git push origin feature/nouvelle-fonctionnalite

# 4. Créer Pull Request sur GitHub

# 5. Merge dans main
git checkout main
git pull origin main
git merge feature/nouvelle-fonctionnalite
git push origin main

# 6. Delete branch
git branch -d feature/nouvelle-fonctionnalite
```

#### Messages de commit

```bash
# Format : <type>: <description>

feat: Add salary projection feature
fix: Correct percentile calculation
docs: Update README with deployment guide
style: Format code with black
refactor: Reorganize market analysis module
test: Add tests for career module
chore: Update dependencies
```

---

## 🔧 Dépannage

### Erreurs courantes

#### 1. `ModuleNotFoundError: No module named 'internal.xxx'`

**Cause** : Module manquant dans `internal/`

**Solution** :
```bash
# Créer un placeholder
cat > internal/xxx.py << EOF
"""Placeholder module"""
import streamlit as st

def render_xxx(*args, **kwargs):
    st.info("Fonctionnalité en développement")
EOF

git add internal/xxx.py
git commit -m "Add placeholder for xxx module"
git push
```

---

#### 2. `ValueError: Invalid property 'weight' for Font`

**Cause** : Plotly n'accepte pas la propriété `weight` pour les fonts

**Solution** : Dans `utils/model_utils.py`, ligne ~1040
```python
# ❌ AVANT
number={'font': {'size': 32, 'weight': 'bold'}}

# ✅ APRÈS
number={'font': {'size': 32}}  # Supprimer 'weight'
```

---

#### 3. `File not found: /mount/src/.../models/xxx.pkl`

**Cause** : Chemins en dur au lieu de dynamiques

**Solution** : Dans `utils/config.py`
```python
# ❌ AVANT
DATA_PATH = Path("data/file.csv")

# ✅ APRÈS
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "output" / "file.csv"
```

---

#### 4. `TypeError: form_submit_button() got an unexpected keyword argument 'key'`

**Cause** : `st.form_submit_button()` ne prend PAS de paramètre `key`

**Solution** :
```python
# ❌ AVANT
st.form_submit_button("Submit", key="btn_submit")

# ✅ APRÈS
st.form_submit_button("Submit")  # Pas de key
```

---

#### 5. `ImportError: cannot import name '_lazywhere' from 'scipy._lib._util'`

**Cause** : Incompatibilité statsmodels/scipy

**Solutions** :
1. **Supprimer scipy du requirements.txt** (Streamlit Cloud l'installe automatiquement)
2. **Supprimer `trendline='lowess'`** des graphiques scatter :

```python
# ❌ AVANT
fig = px.scatter(data, x='x', y='y', trendline='lowess')

# ✅ APRÈS
fig = px.scatter(data, x='x', y='y')  # Pas de trendline
```

---

#### 6. Erreur `removeChild` dans la console

**Cause** : Widgets Streamlit sans clés uniques

**Solution** : Ajouter des clés à TOUS les widgets
```python
# ❌ AVANT
st.button("Analyser")
st.selectbox("Ville", options)

# ✅ APRÈS
st.button("Analyser", key="page_btn_analyze")
st.selectbox("Ville", options, key="page_select_city")
```

**Pattern de nommage** : `{page}_{section}_{type}_{purpose}`

---

### Debug sur Streamlit Cloud

#### Accéder aux logs

1. Aller sur l'app Streamlit Cloud
2. Cliquer "Manage app" (coin inférieur droit)
3. Onglet "Logs"
4. Chercher les erreurs (stack traces en rouge)

#### Ajouter du debug

```python
# Afficher les chemins
st.sidebar.write(f"BASE_DIR: {Config.BASE_DIR}")
st.sidebar.write(f"DATA exists: {Config.DATA_PATH.exists()}")

# Afficher les variables
st.write(f"Profile: {profile_data}")
st.write(f"Prediction: {result}")
```

#### Créer une page debug dédiée

Voir `pages/99_Debug.py` pour un exemple complet de page de debug avec :
- Affichage des chemins
- Vérification des fichiers
- Test des imports
- Recherche récursive de fichiers

---

## 🗺️ Feuille de route

### Version 2.2 (Court terme - 1-2 mois)

- [ ] **Multi-sources de données** : Intégrer LinkedIn, Indeed, Glassdoor
- [ ] **Amélioration modèle** : R² > 0.40 avec ensemble methods
- [ ] **Dark mode** : Thème sombre pour l'UI
- [ ] **Export amélioré** : PDF avec graphiques, rapport complet
- [ ] **Comparateur de profils** : Comparer 2+ profils côte à côte
- [ ] **Alertes salariales** : Notifications si salaire change
- [ ] **Plus de visualisations** : Heatmaps géographiques, network graphs

### Version 3.0 (Moyen terme - 3-6 mois)

- [ ] **API REST** : Endpoint `/predict` pour intégrations externes
- [ ] **Recommandations formations** : Coursera, Udemy, OpenClassrooms
- [ ] **Déploiement cloud** : AWS Lambda ou Google Cloud Run
- [ ] **Authentification** : Comptes utilisateurs avec historique
- [ ] **Tableau de bord personnel** : Suivi évolution carrière
- [ ] **Intégration calendrier** : Suivi objectifs professionnels
- [ ] **Notifications** : Email/SMS pour opportunités

### Version 4.0 (Long terme - 6-12 mois)

- [ ] **NLP avancé** : BERT/GPT pour analyse descriptions de poste
- [ ] **Prédiction évolution marché** : Tendances sur 1-2 ans
- [ ] **Plateforme collaborative** : Communauté, forum, partage d'expériences
- [ ] **Matching offres/candidats** : Algorithme de recommandation
- [ ] **Mobile app** : iOS et Android natives

---

## 📞 Support et contribution

### Contribuer

1. Fork le projet
2. Créer une branche (`git checkout -b feature/AmazingFeature`)
3. Commit (`git commit -m 'Add some AmazingFeature'`)
4. Push (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

### Signaler un bug

Ouvrir une issue sur GitHub avec :
- Description du bug
- Étapes pour reproduire
- Comportement attendu vs réel
- Environnement (OS, Python version, navigateur)
- Logs/screenshots si possible

### Contact

- **GitHub** : https://github.com/Paguy-Stream/Projet_machine_learning
- **Email** : [Votre email]
- **LinkedIn** : [Votre LinkedIn]

---

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

---

## 🙏 Remerciements

- **HelloWork** : Pour les données d'offres d'emploi
- **Streamlit** : Pour le framework web
- **Anthropic Claude** : Pour l'assistance au développement
- **Communauté Data** : Pour les retours et suggestions

---

**Version du guide** : 2.1  
**Dernière mise à jour** : Février 2026  
**Auteur** : Emmanuel / Data Team

---

## 📚 Annexes

### A. Glossaire

| Terme | Définition |
|-------|------------|
| **MAE** | Mean Absolute Error - Erreur moyenne absolue |
| **R²** | Coefficient de détermination - Qualité de l'ajustement |
| **RMSE** | Root Mean Square Error - Erreur quadratique moyenne |
| **SHAP** | SHapley Additive exPlanations - Explicabilité ML |
| **XGBoost** | eXtreme Gradient Boosting - Algorithme ML |
| **Percentile** | Position relative dans une distribution (0-100) |
| **Multiplicateur** | Coefficient d'ajustement salarial (ville/secteur) |

### B. Références utiles

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Documentation](https://plotly.com/python/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

### C. Commandes utiles

```bash
# Développement
streamlit run 01_Accueil.py --server.port 8502  # Port custom
streamlit run 01_Accueil.py --server.headless true  # Sans browser

# Tests
pytest --maxfail=1  # Stop au premier échec
pytest -k "test_model"  # Tests contenant "test_model"
pytest --pdb  # Debugger interactif

# Git
git log --oneline --graph --all  # Historique graphique
git diff HEAD~1  # Diff avec commit précédent
git stash  # Sauvegarder changements temporairement

# Python
python -m pip list --outdated  # Packages à mettre à jour
python -m pip install -U <package>  # Upgrade package
```

---

**Fin du guide** 🎉

Ce document sera mis à jour au fur et à mesure de l'évolution du projet.