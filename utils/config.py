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
    
    Cette classe gère :
    - Les chemins des fichiers (modèle, données, rapports)
    - Les énumérations statiques (villes, métiers, secteurs)
    - Les multiplicateurs salariaux calculés depuis le dataset
    - Les métriques de performance du modèle
    - Le cache des valeurs dynamiques
    
    Attributes:
        BASE_DIR (Path): Répertoire racine de l'application
        MODEL_PATH (Path): Chemin vers le modèle XGBoost
        DATA_PATH (Path): Chemin vers le dataset nettoyé
        REPORT_PATH (Path): Chemin vers le rapport de modélisation
        MODEL_INFO (dict): Métriques de performance du modèle
        MARKET_MEDIAN (float): Salaire médian du marché Data
        
    Examples:
        >>> config = Config()
        >>> multiplier = config.get_city_multiplier("Paris")
        >>> print(f"Multiplicateur Paris: {multiplier}")
        Multiplicateur Paris: 1.20
        
        >>> salary_range = config.get_salary_range_by_job("Data Scientist")
        >>> print(f"Fourchette: {salary_range}")
        Fourchette: (40000, 65000)
    """
    
    # ========================================================================
    # CHEMINS DES FICHIERS
    # ========================================================================
    
    BASE_DIR = Path(__file__).parent.parent  # FIX: Chemin dynamique
    MODEL_PATH = BASE_DIR / "models" / "best_model_XGBoost_fixed.pkl"
    DATA_PATH = BASE_DIR / "data" / "hellowork_cleaned_complete.csv"
    TEST_DATA_PATH = BASE_DIR / "models" / "test_data.pkl"
    REPORT_PATH = BASE_DIR / "models" / "modeling_report_v7.json"
    )
    
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
        "Paris",
        "Lyon",
        "Marseille",
        "Toulouse",
        "Bordeaux",
        "Lille",
        "Nantes",
        "Nice",
        "Rennes",
        "Strasbourg",
        "Montpellier",
        "Grenoble",
        "Reims",
        "Saint-Étienne",
        "Toulon",
        "Autre",
        "Non spécifié"
    ]
    
    SECTORS = [
        "Tech",
        "Banque",
        "Finance",
        "Startup",
        "Conseil",
        "Assurance",
        "ESN",
        "E-commerce",
        "Industrie",
        "Santé",
        "Retail",
        "Autre",
        "Non spécifié"
    ]
    
    CONTRACT_TYPES = [
        "CDI",
        "CDD",
        "Stage",
        "Alternance"
    ]
    
    # ========================================================================
    # COMPÉTENCES TECHNIQUES
    # ========================================================================
    
    LANGUAGES = [
        "Python",
        "R",
        "SQL",
        "Java",
        "Scala",
        "PySpark"
    ]
    
    VIZ_TOOLS = [
        "Tableau",
        "Power BI",
        "Qlik",
        "Looker",
        "Superset"
    ]
    
    CLOUD = [
        "AWS",
        "Azure",
        "GCP",
        "Google Cloud",
        "Snowflake",
        "Databricks"
    ]
    
    BIG_DATA = [
        "Spark",
        "Hadoop",
        "Kafka",
        "Airflow",
        "Dbt",
        "Presto"
    ]
    
    ML_TOOLS = [
        "Machine Learning",
        "Deep Learning",
        "TensorFlow",
        "PyTorch",
        "Scikit-learn",
        "MLflow",
        "NLP",
        "Computer Vision"
    ]
    
    OTHER_TECH = [
        "Git",
        "Docker",
        "Kubernetes",
        "CI/CD",
        "Jenkins",
        "Terraform"
    ]
    
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
    
    TOP_FEATURES = [
        "Paris (localisation)",
        "Secteur Banque",
        "Expérience d'entreprise non spécifiée",
        "Secteur à haut salaire",
        "Toulouse (localisation)"
    ]
    
    # ========================================================================
    # CACHE POUR DONNÉES DYNAMIQUES
    # ========================================================================
    
    _dynamic_cache: Dict[str, Any] = {}
    
    # ========================================================================
    # INITIALISATION
    # ========================================================================
    
    def __init__(self):
        """
        Initialise la configuration et charge les données du rapport.
        
        Tente de charger les métriques réelles depuis le rapport de
        modélisation. En cas d'échec, utilise les valeurs par défaut.
        """
        self._load_config_from_report()
    
    def _load_config_from_report(self) -> None:
        """
        Charge la configuration depuis le rapport de modélisation JSON.
        
        Met à jour :
        - Les métriques de performance (R², MAE, RMSE)
        - Les statistiques du marché (médiane, moyenne, écart-type)
        - Le nom du meilleur modèle
        
        En cas d'erreur, conserve les valeurs par défaut et affiche un
        avertissement dans la sidebar.
        """
        try:
            if not self.REPORT_PATH.exists():
                st.sidebar.info("ℹ️ Rapport non trouvé - Valeurs par défaut utilisées")
                return
            
            with open(self.REPORT_PATH, 'r', encoding='utf-8') as f:
                report = json.load(f)
            
            # Extraction des métriques de performance
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
            
            # Extraction des statistiques du marché
            data_info = report.get('data_info', {})
            Config.MARKET_MEDIAN = data_info.get('target_median', 49450)
            Config.MARKET_MEAN = data_info.get('target_mean', 48914)
            Config.MARKET_STD = data_info.get('target_std', 13056)
            
            st.sidebar.success("✅ Configuration chargée depuis le rapport")
            
        except Exception as e:
            st.sidebar.warning(
                f"⚠️ Erreur chargement rapport : {str(e)[:80]}...\n"
                "Utilisation des valeurs par défaut."
            )
    
    # ========================================================================
    # CHARGEMENT DES DONNÉES DYNAMIQUES
    # ========================================================================
    
    @classmethod
    def _ensure_dynamic_loaded(cls) -> None:
        """
        Charge toutes les valeurs dynamiques une seule fois (lazy loading).
        
        Calcule depuis le dataset :
        - Multiplicateurs salariaux par ville (médiane_ville / médiane_globale)
        - Multiplicateurs salariaux par secteur (médiane_secteur / médiane_globale)
        - Fourchettes salariales par type de poste (P25-P75)
        - Fourchettes salariales par niveau d'expérience (P25-P75)
        
        Les résultats sont mis en cache dans cls._dynamic_cache.
        En cas d'erreur, utilise des valeurs par défaut raisonnables.
        
        Notes:
            Cette méthode est thread-safe grâce au cache classe.
            Elle n'est exécutée qu'une seule fois par session.
        """
        if cls._dynamic_cache:
            return  # Déjà chargé
        
        try:
            # Chargement du dataset
            df = pd.read_csv(
                cls.DATA_PATH,
                usecols=[
                    'location_final',
                    'sector_clean',
                    'salary_mid',
                    'job_type_with_desc',
                    'experience_final'
                ],
                encoding='utf-8'
            )
            
            # Nettoyage des données salariales
            df['salary_mid'] = pd.to_numeric(df['salary_mid'], errors='coerce')
            df = df.dropna(subset=['salary_mid'])
            
            if df.empty:
                raise ValueError("Dataset vide après nettoyage")
            
            global_median = df['salary_mid'].median()
            
            # Calcul des multiplicateurs de villes
            city_medians = df.groupby('location_final')['salary_mid'].median()
            city_multipliers = (city_medians / global_median).to_dict()
            
            # Calcul des multiplicateurs de secteurs
            sector_medians = df.groupby('sector_clean')['salary_mid'].median()
            sector_multipliers = (sector_medians / global_median).to_dict()
            
            # Calcul des fourchettes par type de poste (P25-P75)
            job_ranges = {}
            for job in df['job_type_with_desc'].unique():
                job_salaries = df[df['job_type_with_desc'] == job]['salary_mid']
                
                if len(job_salaries) >= 10:  # Minimum 10 observations
                    job_ranges[job] = (
                        int(job_salaries.quantile(0.25)),
                        int(job_salaries.quantile(0.75))
                    )
            
            # Calcul des fourchettes par expérience (P25-P75)
            exp_ranges = {}
            exp_bins = [
                (0, 1),    # Débutant
                (1, 3),    # Junior
                (3, 5),    # Confirmé
                (5, 8),    # Senior
                (8, 12),   # Expert
                (12, 30)   # Lead/Directeur
            ]
            
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
            
            # Stockage dans le cache
            cls._dynamic_cache = {
                'city_multipliers': city_multipliers,
                'sector_multipliers': sector_multipliers,
                'job_ranges': job_ranges,
                'exp_ranges': exp_ranges,
                'loaded': True,
                'n_samples': len(df)
            }
            
            st.success(
                f"✅ Multiplicateurs calculés depuis {len(df):,} offres"
            )
            
        except Exception as e:
            # Fallback sur valeurs par défaut
            st.warning(
                f"⚠️ Calcul dynamique échoué : {str(e)[:80]}...\n"
                "Utilisation des valeurs par défaut."
            )
            
            cls._dynamic_cache = cls._get_default_multipliers()
    
    @staticmethod
    def _get_default_multipliers() -> Dict[str, Any]:
        """
        Retourne les multiplicateurs et fourchettes par défaut.
        
        Utilisé en fallback si le calcul dynamique échoue.
        
        Returns:
            Dict contenant les multiplicateurs et fourchettes par défaut
        """
        return {
            'city_multipliers': {
                "Paris": 1.20,
                "Lyon": 1.10,
                "Marseille": 1.05,
                "Toulouse": 1.00,
                "Bordeaux": 1.00,
                "Lille": 0.95,
                "Nantes": 0.95,
                "Nice": 1.05,
                "Rennes": 0.95,
                "Strasbourg": 0.95,
                "Montpellier": 0.95,
                "Grenoble": 0.95,
                "Reims": 0.90,
                "Saint-Étienne": 0.90,
                "Toulon": 0.90,
                "Non spécifié": 1.00,
                "Autre": 1.00
            },
            'sector_multipliers': {
                "Banque": 1.25,
                "Finance": 1.20,
                "Tech": 1.15,
                "Startup": 1.15,
                "Conseil": 1.10,
                "Assurance": 1.10,
                "ESN": 1.05,
                "E-commerce": 1.00,
                "Industrie": 0.95,
                "Santé": 0.95,
                "Retail": 0.90,
                "Non spécifié": 1.00,
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
                (0, 1): (30000, 45000),
                (1, 3): (35000, 50000),
                (3, 5): (40000, 55000),
                (5, 8): (45000, 65000),
                (8, 12): (50000, 75000),
                (12, 30): (60000, 90000)
            },
            'loaded': False,
            'n_samples': 0
        }
    
    # ========================================================================
    # ACCESSEURS PUBLICS (API)
    # ========================================================================
    
    @classmethod
    def get_city_multiplier(cls, city: str) -> float:
        """
        Retourne le multiplicateur salarial pour une ville donnée.
        
        Le multiplicateur est calculé comme :
        médiane_salaire_ville / médiane_salaire_globale
        
        Args:
            city: Nom de la ville
            
        Returns:
            Multiplicateur (ex: 1.20 pour Paris = +20%)
            
        Examples:
            >>> Config.get_city_multiplier("Paris")
            1.20
            >>> Config.get_city_multiplier("Lille")
            0.95
        """
        cls._ensure_dynamic_loaded()
        return cls._dynamic_cache['city_multipliers'].get(city, 1.0)
    
    @classmethod
    def get_sector_multiplier(cls, sector: str) -> float:
        """
        Retourne le multiplicateur salarial pour un secteur donné.
        
        Le multiplicateur est calculé comme :
        médiane_salaire_secteur / médiane_salaire_globale
        
        Args:
            sector: Nom du secteur
            
        Returns:
            Multiplicateur (ex: 1.25 pour Banque = +25%)
            
        Examples:
            >>> Config.get_sector_multiplier("Banque")
            1.25
            >>> Config.get_sector_multiplier("Retail")
            0.90
        """
        cls._ensure_dynamic_loaded()
        return cls._dynamic_cache['sector_multipliers'].get(sector, 1.0)
    
    @classmethod
    def get_salary_range_by_job(cls, job_type: str) -> Tuple[int, int]:
        """
        Retourne la fourchette salariale (P25-P75) pour un type de poste.
        
        Args:
            job_type: Type de poste (ex: "Data Scientist")
            
        Returns:
            Tuple (salaire_bas, salaire_haut) en euros annuels bruts
            
        Examples:
            >>> Config.get_salary_range_by_job("Data Scientist")
            (40000, 65000)
        """
        cls._ensure_dynamic_loaded()
        job_ranges = cls._dynamic_cache['job_ranges']
        return job_ranges.get(job_type, (35000, 60000))
    
    @classmethod
    def get_salary_range_by_experience(
        cls,
        experience_years: float
    ) -> Tuple[int, int]:
        """
        Retourne la fourchette salariale selon le niveau d'expérience.
        
        Args:
            experience_years: Nombre d'années d'expérience
            
        Returns:
            Tuple (salaire_bas, salaire_haut) en euros annuels bruts
            
        Examples:
            >>> Config.get_salary_range_by_experience(4)
            (40000, 55000)
            >>> Config.get_salary_range_by_experience(10)
            (50000, 75000)
        """
        cls._ensure_dynamic_loaded()
        exp_ranges = cls._dynamic_cache['exp_ranges']
        
        # Trouver la bonne tranche d'expérience
        for (min_exp, max_exp), salary_range in exp_ranges.items():
            if min_exp <= experience_years < max_exp:
                return salary_range
        
        # Fallback pour expérience très élevée (>12 ans)
        return exp_ranges.get((12, 30), (60000, 90000))
    
    @classmethod
    def get_all_city_multipliers(cls) -> Dict[str, float]:
        """
        Retourne tous les multiplicateurs de villes.
        
        Utile pour affichage, debug ou analyse comparative.
        
        Returns:
            Dict {ville: multiplicateur}
        """
        cls._ensure_dynamic_loaded()
        return cls._dynamic_cache['city_multipliers']
    
    @classmethod
    def get_all_sector_multipliers(cls) -> Dict[str, float]:
        """
        Retourne tous les multiplicateurs de secteurs.
        
        Utile pour affichage, debug ou analyse comparative.
        
        Returns:
            Dict {secteur: multiplicateur}
        """
        cls._ensure_dynamic_loaded()
        return cls._dynamic_cache['sector_multipliers']
    
    @classmethod
    def reload_dynamic_data(cls) -> None:
        """
        Force le rechargement de toutes les données dynamiques.
        
        Utile si le dataset a été modifié et qu'on veut recalculer
        les multiplicateurs sans redémarrer l'application.
        
        Warning:
            Cette opération peut prendre quelques secondes.
        """
        cls._dynamic_cache = {}
        cls._ensure_dynamic_loaded()
        st.success("🔄 Données dynamiques rechargées")


# ============================================================================
# INITIALISATION DE SESSION STATE
# ============================================================================

def init_session_state() -> None:
    """
    Initialise les variables de session Streamlit.
    
    Crée les variables suivantes si elles n'existent pas :
    - model_loaded : Indicateur de chargement du modèle
    - current_profile : Profil utilisateur par défaut
    - prediction_made : Flag de prédiction effectuée
    - last_prediction : Dernière prédiction réalisée
    - model_metrics : Métriques du modèle
    
    Notes:
        Cette fonction doit être appelée au début de chaque page.
        Les valeurs ne sont initialisées qu'une seule fois par session.
    """
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
    """
    Configure la page Streamlit avec header personnalisé et style CSS.
    
    Args:
        title: Titre de la page (défaut: Config.APP_TITLE)
        icon: Icône de la page (défaut: "💼")
    
    Notes:
        Cette fonction doit être appelée en premier dans chaque page.
        Elle configure :
        - Les métadonnées de la page
        - Le CSS personnalisé
        - Le header avec gradient
    """
    if title is None:
        title = Config.APP_TITLE
    
    # Configuration de base
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/your-repo',
            'Report a bug': 'https://github.com/your-repo/issues',
            'About': f"""
            ## {Config.APP_TITLE}
            
            Application d'estimation salariale basée sur l'analyse de 
            **5 868 offres d'emploi** du dataset HelloWork.
            
            **Modèle** : {Config.MODEL_INFO['model_name']}
            **R²** : {Config.MODEL_INFO['r2_score']:.3f}
            **MAE** : {Config.MODEL_INFO['mae']:,}€
            **Précision** : {Config.MODEL_INFO['precision_15']:.0f}% (±15%)
            
            ✅ Multiplicateurs calculés dynamiquement
            ✅ Fourchettes salariales auto-ajustées
            
            Données collectées en janvier 2026.
            """
        }
    )
    
    # CSS personnalisé
    _apply_custom_css()
    
    # Header unique et optimisé
    _render_page_header(title)


def _apply_custom_css() -> None:
    """
    Applique les styles CSS personnalisés à l'application.
    
    Styles appliqués :
    - Boutons avec gradient et effets hover
    - Cartes de métriques avec ombres
    - Sidebar avec fond dégradé
    - Badges colorés
    - Alertes personnalisées
    """
    st.markdown("""
    <style>
    /* Boutons principaux */
    .stButton>button {
        background: linear-gradient(135deg, #1f77b4 0%, #0d5a9e 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 16px;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #0d5a9e 0%, #1f77b4 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Cartes de métriques */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #1f77b4;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Suppression du padding supérieur excessif */
    .block-container {
        padding-top: 2rem;
    }
    
    /* Espacement des sections */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 2px solid #e9ecef;
    }
    
    /* Amélioration des expanders */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)


def _render_page_header(title: str) -> None:
    """
    Affiche le header principal de la page avec gradient.
    
    Args:
        title: Titre à afficher dans le header
    """
    st.markdown(f"""
    <div style='
        text-align: center;
        padding: 25px 20px;
        background: linear-gradient(135deg, #1f77b4 0%, #0d5a9e 100%);
        border-radius: 12px;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    '>
        <h1 style='
            color: white;
            font-size: 36px;
            margin: 0 0 8px 0;
            font-weight: 700;
        '>
            💼 {title}
        </h1>
        <p style='
            color: rgba(255,255,255,0.95);
            font-size: 16px;
            margin: 5px 0;
            font-weight: 500;
        '>
            5 868 offres HelloWork • XGBoost v7 • R² = {Config.MODEL_INFO['r2_score']:.3f}
        </p>
        <p style='
            color: rgba(255,255,255,0.85);
            font-size: 14px;
            margin: 8px 0 0 0;
        '>
            ✅ MAE = {Config.MODEL_INFO['mae']:,}€ • 
            ✅ Précision {Config.MODEL_INFO['precision_15']:.0f}% (±15%) • 
            ✅ Calculs dynamiques
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'Config',
    'init_session_state',
    'setup_page'
]
