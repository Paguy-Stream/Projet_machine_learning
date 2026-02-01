"""
Configuration de l'application Prédicteur de Salaires Data Jobs.

Ce module centralise :
- Les paramètres de l'application
- Les chemins des fichiers et modèles
- Les énumérations (métiers, villes, secteurs, compétences)
- Les multiplicateurs salariaux calculés dynamiquement
- Les métriques du modèle XGBoost
- La configuration de l'interface Streamlit
"""

import streamlit as st
from pathlib import Path
import json
import pandas as pd
from typing import Dict, Tuple, Optional, Any


# ============================================================================
# CLASSE PRINCIPALE DE CONFIGURATION
# ============================================================================

class Config:
    """
    Configuration centralisée de l'application avec calculs dynamiques.
    """
    
    # ========================================================================
    # CHEMINS DES FICHIERS - CORRIGÉS SELON DEBUG
    # ========================================================================
    
    BASE_DIR = Path(__file__).parent.parent  # Dynamique : /mount/src/projet_machine_learning
    MODEL_PATH = BASE_DIR / "models" / "best_model_XGBoost_fixed.pkl"
    DATA_PATH = BASE_DIR / "output" / "hellowork_cleaned_complete.csv"  # FIX: Chemin réel
    TEST_DATA_PATH = BASE_DIR / "output" / "test_data.pkl"  # FIX: Chemin réel
    REPORT_PATH = BASE_DIR / "output" / "analysis_complete" / "modeling_v7_improved" / "modeling_report_v7.json"
    
    # ========================================================================
    # BRANDING
    # ========================================================================
    
    APP_TITLE = "Prédicteur de Salaires Data Jobs"
    APP_SUBTITLE = "Analyse de 5 868 offres HelloWork"
    APP_ICON = "💼"
    
    # ========================================================================
    # ÉNUMÉRATIONS MÉTIERS
    # ========================================================================
    
    JOB_TYPES = [
        "Data Analyst",
        "Data Scientist",
        "Data Engineer",
        "BI/Analytics",
        "Spécialiste IA/ML",
        "Data Consultant",
        "Data Management",
        "Autre rôle en données",
        "Autre"
    ]
    
    SENIORITY_LEVELS = [
        "Stage/Alternance",
        "Junior",
        "Intermédiaire",
        "Senior",
        "Lead / Manager",
        "Freelance/Consultant"
    ]
    
    EDUCATION_LEVELS = [
        "Bac",
        "Bac+2",
        "Bac+3",
        "Bac+4",
        "Bac+5",
        "Non spécifié"
    ]
    
    # ========================================================================
    # ÉNUMÉRATIONS GÉOGRAPHIQUES ET SECTORIELLES
    # ========================================================================
    
    CITIES = [
        "Paris", "Lyon", "Marseille", "Toulouse", "Bordeaux",
        "Lille", "Nantes", "Nice", "Rennes", "Strasbourg",
        "Montpellier", "Grenoble", "Reims", "Saint-Étienne",
        "Toulon", "Autre", "Non spécifié"
    ]
    
    SECTORS = [
        "Tech", "Banque", "Finance", "Startup", "Conseil",
        "Assurance", "ESN", "E-commerce", "Industrie", "Santé",
        "Retail", "Autre", "Non spécifié"
    ]
    
    CONTRACT_TYPES = ["CDI", "CDD", "Stage", "Alternance"]
    
    # ========================================================================
    # COMPÉTENCES TECHNIQUES
    # ========================================================================
    
    LANGUAGES = ["Python", "R", "SQL", "Java", "Scala", "PySpark"]
    VIZ_TOOLS = ["Tableau", "Power BI", "Qlik", "Looker", "Superset"]
    CLOUD = ["AWS", "Azure", "GCP", "Google Cloud", "Snowflake", "Databricks"]
    BIG_DATA = ["Spark", "Hadoop", "Kafka", "Airflow", "Dbt", "Presto"]
    ML_TOOLS = [
        "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch",
        "Scikit-learn", "MLflow", "NLP", "Computer Vision"
    ]
    OTHER_TECH = ["Git", "Docker", "Kubernetes", "CI/CD", "Jenkins", "Terraform"]
    
    # ========================================================================
    # MÉTRIQUES DU MODÈLE
    # ========================================================================
    
    MODEL_INFO = {
        'r2_score': 0.337,
        'mae': 5163,
        'rmse': 6969,
        'precision_10': 57.9,
        'precision_15': 73.7,
        'precision_20': 83.8,
        'model_name': 'XGBoost',
        'cv_mae': 5188,
        'stability': 0.995,
        'overfitting_score': 0.140
    }
    
    MARKET_MEDIAN = 49450
    MARKET_MEAN = 48914
    MARKET_STD = 13056
    
    # ========================================================================
    # CACHE POUR DONNÉES DYNAMIQUES
    # ========================================================================
    
    _dynamic_cache: Dict[str, Any] = {}
    
    # ========================================================================
    # INITIALISATION
    # ========================================================================
    
    def __init__(self):
        """Initialise la configuration et charge les données du rapport."""
        self._load_config_from_report()
    
    def _load_config_from_report(self) -> None:
        """Charge la configuration depuis le rapport de modélisation JSON."""
        try:
            if not self.REPORT_PATH.exists():
                st.sidebar.info("ℹ️ Rapport non trouvé - Valeurs par défaut utilisées")
                return
            
            with open(self.REPORT_PATH, 'r', encoding='utf-8') as f:
                report = json.load(f)
            
            perf_metrics = report.get('performance_metrics', {})
            Config.MODEL_INFO.update({
                'r2_score': perf_metrics.get('test_r2', 0.337),
                'mae': perf_metrics.get('test_mae', 5163),
                'rmse': perf_metrics.get('test_rmse', 6969),
                'model_name': report.get('best_model', {}).get('name', 'XGBoost'),
                'cv_mae': perf_metrics.get('cv_mae_mean', 5188),
                'stability': perf_metrics.get('model_stability', 0.995),
                'overfitting_score': perf_metrics.get('overfitting_score', 0.140)
            })
            
            data_info = report.get('data_info', {})
            Config.MARKET_MEDIAN = data_info.get('target_median', 49450)
            Config.MARKET_MEAN = data_info.get('target_mean', 48914)
            Config.MARKET_STD = data_info.get('target_std', 13056)
            
            st.sidebar.success("✅ Configuration chargée depuis le rapport")
            
        except Exception as e:
            st.sidebar.warning(f"⚠️ Erreur chargement rapport : {str(e)[:80]}...")
    
    # ========================================================================
    # CHARGEMENT DES DONNÉES DYNAMIQUES
    # ========================================================================
    
    @classmethod
    def _ensure_dynamic_loaded(cls) -> None:
        """Charge toutes les valeurs dynamiques une seule fois (lazy loading)."""
        if cls._dynamic_cache:
            return
        
        try:
            df = pd.read_csv(
                cls.DATA_PATH,
                usecols=['location_final', 'sector_clean', 'salary_mid', 'job_type_with_desc', 'experience_final'],
                encoding='utf-8'
            )
            
            df['salary_mid'] = pd.to_numeric(df['salary_mid'], errors='coerce')
            df = df.dropna(subset=['salary_mid'])
            
            if df.empty:
                raise ValueError("Dataset vide après nettoyage")
            
            global_median = df['salary_mid'].median()
            
            city_medians = df.groupby('location_final')['salary_mid'].median()
            city_multipliers = (city_medians / global_median).to_dict()
            
            sector_medians = df.groupby('sector_clean')['salary_mid'].median()
            sector_multipliers = (sector_medians / global_median).to_dict()
            
            job_ranges = {}
            for job in df['job_type_with_desc'].unique():
                job_salaries = df[df['job_type_with_desc'] == job]['salary_mid']
                if len(job_salaries) >= 10:
                    job_ranges[job] = (
                        int(job_salaries.quantile(0.25)),
                        int(job_salaries.quantile(0.75))
                    )
            
            exp_ranges = {}
            exp_bins = [(0, 1), (1, 3), (3, 5), (5, 8), (8, 12), (12, 30)]
            for min_exp, max_exp in exp_bins:
                exp_salaries = df[
                    (df['experience_final'] >= min_exp) &
                    (df['experience_final'] < max_exp)
                ]['salary_mid']
                
                if len(exp_salaries) >= 10:
                    exp_ranges[(min_exp, max_exp)] = (
                        int(exp_salaries.quantile(0.25)),
                        int(exp_salaries.quantile(0.75))
                    )
            
            cls._dynamic_cache = {
                'city_multipliers': city_multipliers,
                'sector_multipliers': sector_multipliers,
                'job_ranges': job_ranges,
                'exp_ranges': exp_ranges,
                'loaded': True,
                'n_samples': len(df)
            }
            
            st.success(f"✅ Multiplicateurs calculés depuis {len(df):,} offres")
            
        except Exception as e:
            st.warning(f"⚠️ Calcul dynamique échoué : {str(e)[:80]}...")
            cls._dynamic_cache = cls._get_default_multipliers()
    
    @staticmethod
    def _get_default_multipliers() -> Dict[str, Any]:
        """Retourne les multiplicateurs par défaut."""
        return {
            'city_multipliers': {
                "Paris": 1.20, "Lyon": 1.10, "Marseille": 1.05,
                "Toulouse": 1.00, "Bordeaux": 1.00, "Lille": 0.95,
                "Nantes": 0.95, "Nice": 1.05, "Rennes": 0.95,
                "Strasbourg": 0.95, "Montpellier": 0.95, "Grenoble": 0.95,
                "Reims": 0.90, "Saint-Étienne": 0.90, "Toulon": 0.90,
                "Non spécifié": 1.00, "Autre": 1.00
            },
            'sector_multipliers': {
                "Banque": 1.25, "Finance": 1.20, "Tech": 1.15,
                "Startup": 1.15, "Conseil": 1.10, "Assurance": 1.10,
                "ESN": 1.05, "E-commerce": 1.00, "Industrie": 0.95,
                "Santé": 0.95, "Retail": 0.90, "Non spécifié": 1.00,
                "Autre": 1.00
            },
            'job_ranges': {
                "Data Scientist": (40000, 65000),
                "Data Analyst": (35000, 60000),
                "Data Engineer": (40000, 65000),
                "BI/Analytics": (35000, 55000),
                "Spécialiste IA/ML": (45000, 75000)
            },
            'exp_ranges': {
                (0, 1): (30000, 45000), (1, 3): (35000, 50000),
                (3, 5): (40000, 55000), (5, 8): (45000, 65000),
                (8, 12): (50000, 75000), (12, 30): (60000, 90000)
            },
            'loaded': False,
            'n_samples': 0
        }
    
    # ========================================================================
    # ACCESSEURS PUBLICS
    # ========================================================================
    
    @classmethod
    def get_city_multiplier(cls, city: str) -> float:
        """Retourne le multiplicateur salarial pour une ville donnée."""
        cls._ensure_dynamic_loaded()
        return cls._dynamic_cache['city_multipliers'].get(city, 1.0)
    
    @classmethod
    def get_sector_multiplier(cls, sector: str) -> float:
        """Retourne le multiplicateur salarial pour un secteur donné."""
        cls._ensure_dynamic_loaded()
        return cls._dynamic_cache['sector_multipliers'].get(sector, 1.0)
    
    @classmethod
    def get_salary_range_by_job(cls, job_type: str) -> Tuple[int, int]:
        """Retourne la fourchette salariale (P25-P75) pour un type de poste."""
        cls._ensure_dynamic_loaded()
        return cls._dynamic_cache['job_ranges'].get(job_type, (35000, 60000))
    
    @classmethod
    def get_salary_range_by_experience(cls, experience_years: float) -> Tuple[int, int]:
        """Retourne la fourchette salariale selon le niveau d'expérience."""
        cls._ensure_dynamic_loaded()
        exp_ranges = cls._dynamic_cache['exp_ranges']
        for (min_exp, max_exp), salary_range in exp_ranges.items():
            if min_exp <= experience_years < max_exp:
                return salary_range
        return exp_ranges.get((12, 30), (60000, 90000))
    
    @classmethod
    def get_all_city_multipliers(cls) -> Dict[str, float]:
        """Retourne tous les multiplicateurs de villes."""
        cls._ensure_dynamic_loaded()
        return cls._dynamic_cache['city_multipliers']
    
    @classmethod
    def get_all_sector_multipliers(cls) -> Dict[str, float]:
        """Retourne tous les multiplicateurs de secteurs."""
        cls._ensure_dynamic_loaded()
        return cls._dynamic_cache['sector_multipliers']
    
    @classmethod
    def reload_dynamic_data(cls) -> None:
        """Force le rechargement de toutes les données dynamiques."""
        cls._dynamic_cache = {}
        cls._ensure_dynamic_loaded()
        st.success("🔄 Données dynamiques rechargées")


# ============================================================================
# INITIALISATION DE SESSION STATE
# ============================================================================

def init_session_state() -> None:
    """Initialise les variables de session Streamlit."""
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    
    if 'current_profile' not in st.session_state:
        st.session_state.current_profile = {
            'job_type': 'Data Analyst',
            'seniority': 'Intermédiaire',
            'experience_years': 4,
            'education': 'Bac+5',
            'city': 'Paris',
            'sector': 'Tech',
            'contract_type': 'CDI',
            'technical_score': 3,
            'has_python': True,
            'has_sql': True,
            'has_tableau': True,
            'has_aws': False,
            'telework_numeric': 0.5,
            'benefits_score': 2
        }
    
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = None
    
    if 'model_metrics' not in st.session_state:
        config = Config()
        st.session_state.model_metrics = Config.MODEL_INFO


# ============================================================================
# CONFIGURATION STREAMLIT
# ============================================================================

def setup_page(title: Optional[str] = None, icon: str = "💼") -> None:
    """Configure la page Streamlit avec header personnalisé."""
    if title is None:
        title = Config.APP_TITLE
    
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/Paguy-Stream/Projet_machine_learning',
            'Report a bug': 'https://github.com/Paguy-Stream/Projet_machine_learning/issues',
            'About': f"""
            ## {Config.APP_TITLE}
            
            Application d'estimation salariale basée sur **5 868 offres**.
            
            **Modèle** : {Config.MODEL_INFO['model_name']}
            **R²** : {Config.MODEL_INFO['r2_score']:.3f}
            **MAE** : {Config.MODEL_INFO['mae']:,}€
            **Précision** : {Config.MODEL_INFO['precision_15']:.0f}% (±15%)
            """
        }
    )
    
    _apply_custom_css()
    _render_page_header(title)


def _apply_custom_css() -> None:
    """Applique les styles CSS personnalisés."""
    st.markdown("""
    <style>
    .stButton>button {
        background: linear-gradient(135deg, #1f77b4 0%, #0d5a9e 100%);
        color: white;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 28px;
        color: #1f77b4;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .block-container {
        padding-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)


def _render_page_header(title: str) -> None:
    """Affiche le header principal de la page."""
    st.markdown(f"""
    <div style='
        text-align: center;
        padding: 25px 20px;
        background: linear-gradient(135deg, #1f77b4 0%, #0d5a9e 100%);
        border-radius: 12px;
        margin-bottom: 30px;
    '>
        <h1 style='color: white; font-size: 36px; margin: 0;'>💼 {title}</h1>
        <p style='color: rgba(255,255,255,0.95); margin: 8px 0;'>
            5 868 offres • XGBoost v7 • R² = {Config.MODEL_INFO['r2_score']:.3f}
        </p>
        <p style='color: rgba(255,255,255,0.85); font-size: 14px; margin: 0;'>
            ✅ MAE = {Config.MODEL_INFO['mae']:,}€ • 
            ✅ Précision {Config.MODEL_INFO['precision_15']:.0f}% (±15%)
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'Config',
    'init_session_state',
    'setup_page',
    # FIX: Export des chemins pour debug
    'BASE_DIR',
    'DATA_PATH',
    'MODEL_PATH',
    'TEST_DATA_PATH',
    'REPORT_PATH'
]

# Exports de niveau module pour compatibilité
BASE_DIR = Config.BASE_DIR
DATA_PATH = Config.DATA_PATH
MODEL_PATH = Config.MODEL_PATH
TEST_DATA_PATH = Config.TEST_DATA_PATH
REPORT_PATH = Config.REPORT_PATH
