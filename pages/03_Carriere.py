"""
Page de feuille de route carrière Data - VERSION CORRIGÉE.

Cette page permet de :
- Définir son profil professionnel actuel
- Obtenir une analyse de positionnement
- Recevoir une roadmap personnalisée de compétences
- Analyser les transitions de rôle possibles
- Projeter sa progression salariale (3 scénarios)
- Simuler une négociation salariale

FIX: Ajout de clés uniques pour éviter l'erreur removeChild

Architecture:
    - Module principal : Orchestration et formulaire
    - career_analysis : Scorecard et diagnostic
    - career_roadmap : Roadmap pédagogique et matrice effort/impact
    - career_transitions : Transitions et projections
    - career_export : Négociation, export et navigation
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Any

from utils.config import Config, init_session_state, setup_page
from utils.model_utils import (
    init_model_utils,
    CalculationUtils,
    DataDistributions
)


# ============================================================================
# CONFIGURATION
# ============================================================================

def initialize_career_page() -> Tuple[Any, pd.DataFrame, np.ndarray, float]:
    """
    Initialise la page carrière.
    
    Returns:
        Tuple contenant :
            - model_utils: Gestionnaire du modèle
            - df_final: Dataset complet
            - real_market_data: Données salariales du marché
            - market_median: Médiane du marché
            
    Raises:
        SystemExit: Si les données ne peuvent pas être chargées
    """
    setup_page("Feuille de Route Carrière", "🎓")
    init_session_state()
    
    # Initialiser le modèle
    if 'model_utils' not in st.session_state:
        model_utils = init_model_utils()
        st.session_state.model_utils = model_utils
    else:
        model_utils = st.session_state.model_utils
    
    # Charger le dataset
    df_final = load_full_dataset()
    if df_final is None:
        st.stop()
    
    # Récupérer les données du marché
    real_market_data = model_utils.get_real_market_data()
    if real_market_data is None:
        st.error("❌ Données du marché non disponibles")
        st.stop()
    
    market_median = np.median(real_market_data)
    
    return model_utils, df_final, real_market_data, market_median


@st.cache_data
def load_full_dataset() -> Optional[pd.DataFrame]:
    """
    Charge le dataset complet avec gestion d'erreurs robuste.
    
    Returns:
        DataFrame complet ou None si erreur
        
    Notes:
        Vérifie la présence des colonnes critiques et effectue
        le nettoyage de base.
    """
    try:
        df = pd.read_csv(Config.DATA_PATH, encoding='utf-8')
        
        # Vérifier colonnes critiques
        required_cols = [
            'job_type_with_desc', 'location_final', 'sector_clean',
            'salary_mid', 'experience_final'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"❌ Colonnes manquantes : {', '.join(missing_cols)}")
            return None
        
        # Nettoyage
        df['salary_mid'] = pd.to_numeric(df['salary_mid'], errors='coerce')
        df = df.dropna(subset=['salary_mid', 'experience_final'])
        
        return df
    
    except Exception as e:
        st.error(f"❌ Erreur de chargement : {str(e)}")
        return None


# ============================================================================
# HEADER
# ============================================================================

def render_career_header(df_size: int, market_median: float) -> None:
    """
    Affiche l'en-tête de la page carrière.
    
    Args:
        df_size: Taille du dataset
        market_median: Médiane salariale du marché
    """
    st.title("🎓 Votre feuille de route carrière personnalisée")
    
    st.markdown(f"""
    Optimisez votre évolution professionnelle avec une analyse basée sur 
    **{df_size:,} offres réelles** et le modèle **XGBoost v7**.
    
    Cette page vous permet de :
    - 📊 Évaluer votre positionnement actuel
    - 🗺️ Obtenir une roadmap de compétences personnalisée
    - 🔄 Explorer les transitions de rôle possibles
    - 📈 Projeter votre évolution salariale sur 10 ans
    - 💬 Préparer votre prochaine négociation
    """)
    
    st.markdown("---")


# ============================================================================
# FORMULAIRE DE PROFIL
# ============================================================================

def render_profile_form() -> Optional[Dict[str, Any]]:
    """
    Affiche le formulaire de saisie du profil utilisateur.
    
    Returns:
        Dict du profil complet si soumis, None sinon
        
    Notes:
        Le formulaire est divisé en 2 sections :
        1. Informations professionnelles
        2. Stack technique actuelle
        
    FIX: Ajout de clés uniques et clear_on_submit=False
    """
    st.markdown("### 👤 Étape 1 : Définissez votre profil actuel")
    
    with st.expander("ℹ️ Pourquoi ces informations ?", expanded=False):
        st.markdown("""
        - **Poste/Expérience** : Base de calcul du salaire actuel
        - **Localisation/Secteur** : Multiplicateurs dynamiques
        - **Compétences** : Identification des gaps et opportunités
        - **Toutes les données restent privées et ne sont jamais stockées**
        """)
    
    # FIX: Ajout de clear_on_submit=False pour éviter les rerenders multiples
    with st.form("career_profile_form", clear_on_submit=False):
        # Section 1 : Informations professionnelles
        professional_data = _render_professional_section()
        
        st.markdown("---")
        
        # Section 2 : Stack technique
        skills_data = _render_skills_section()
        
        st.markdown("---")
        
        # FIX: Ajout d'une clé unique au bouton submit
        submitted = st.form_submit_button(
            "🚀 Générer ma feuille de route",
            type="primary",
            use_container_width=True,
            key="career_submit_btn"
        )
        
        if submitted:
            return _build_complete_profile(professional_data, skills_data)
    
    return None


def _render_professional_section() -> Dict[str, Any]:
    """
    Affiche la section informations professionnelles.
    
    FIX: Ajout de clés uniques à tous les widgets
    """
    st.markdown("#### 💼 Informations professionnelles")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        job_type = st.selectbox(
            "Type de poste actuel",
            Config.JOB_TYPES,
            index=0,
            help="Votre rôle principal en ce moment",
            key="career_job_type"  # FIX: Clé unique
        )
        experience = st.number_input(
            "Années d'expérience",
            min_value=0.0,
            max_value=30.0,
            value=4.0,
            step=0.5,
            help="Expérience totale dans la Data",
            key="career_experience"  # FIX: Clé unique
        )
    
    with col2:
        location = st.selectbox(
            "Ville actuelle",
            Config.CITIES,
            index=0,
            help="Votre lieu de travail principal",
            key="career_location"  # FIX: Clé unique
        )
        sector = st.selectbox(
            "Secteur actuel",
            Config.SECTORS,
            index=1,
            help="Industrie de votre entreprise",
            key="career_sector"  # FIX: Clé unique
        )
    
    with col3:
        education = st.selectbox(
            "Niveau d'études",
            Config.EDUCATION_LEVELS,
            index=4,
            help="Diplôme le plus élevé obtenu",
            key="career_education"  # FIX: Clé unique
        )
        telework = st.slider(
            "Télétravail (j/sem)",
            0, 5, 2,
            help="Nombre de jours de télétravail par semaine",
            key="career_telework"  # FIX: Clé unique
        )
    
    return {
        'job_type': job_type,
        'experience': experience,
        'location': location,
        'sector': sector,
        'education': education,
        'telework': telework
    }


def _render_skills_section() -> Dict[str, bool]:
    """
    Affiche la section stack technique.
    
    FIX: Ajout de clés uniques à toutes les checkboxes
    """
    st.markdown("#### 🛠️ Votre stack technique actuelle")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Langages & Outils**")
        python = st.checkbox("Python", value=True, key="career_skill_python")  # FIX
        sql = st.checkbox("SQL", value=True, key="career_skill_sql")  # FIX
        r = st.checkbox("R", value=False, key="career_skill_r")  # FIX
    
    with col2:
        st.markdown("**Visualisation & BI**")
        tableau = st.checkbox("Tableau", value=False, key="career_skill_tableau")  # FIX
        power_bi = st.checkbox("Power BI", value=False, key="career_skill_powerbi")  # FIX
    
    with col3:
        st.markdown("**Cloud & Big Data**")
        aws = st.checkbox("AWS", value=False, key="career_skill_aws")  # FIX
        azure = st.checkbox("Azure", value=False, key="career_skill_azure")  # FIX
        spark = st.checkbox("Spark", value=False, key="career_skill_spark")  # FIX
    
    col4, col5 = st.columns(2)
    
    with col4:
        st.markdown("**Intelligence Artificielle**")
        ml = st.checkbox("Machine Learning", value=False, key="career_skill_ml")  # FIX
        dl = st.checkbox("Deep Learning", value=False, key="career_skill_dl")  # FIX
    
    with col5:
        st.markdown("**Data Engineering**")
        etl = st.checkbox("ETL / Pipelines", value=False, key="career_skill_etl")  # FIX
    
    return {
        'contient_python': python,
        'contient_sql': sql,
        'contient_r': r,
        'contient_tableau': tableau,
        'contient_power_bi': power_bi,
        'contient_aws': aws,
        'contient_azure': azure,
        'contient_spark': spark,
        'contient_machine_learning': ml,
        'contient_deep_learning': dl,
        'contient_etl': etl
    }


def _build_complete_profile(
    professional: Dict[str, Any],
    skills: Dict[str, bool]
) -> Dict[str, Any]:
    """
    Construit le profil complet avec calculs dynamiques.
    
    Args:
        professional: Données professionnelles
        skills: Compétences techniques
        
    Returns:
        Dict du profil complet prêt pour la prédiction
    """
    # Déduire le seniority
    seniority = _deduce_seniority(professional['experience'])
    
    # Profil de base
    profile = {
        'job_type': professional['job_type'],
        'seniority': seniority,
        'experience_final': float(professional['experience']),
        'contract_type_clean': 'CDI',
        'education_clean': professional['education'],
        'location_final': professional['location'],
        'sector_clean': professional['sector'],
        'telework_numeric': professional['telework'] / 5.0,
        **skills,
        'has_teletravail': professional['telework'] > 0,
        'has_mutuelle': False,
        'has_tickets': False,
        'has_prime': False
    }
    
    # Calculs dynamiques des scores
    profile['skills_count'] = CalculationUtils.calculate_skills_count_from_profile(skills)
    profile['technical_score'] = CalculationUtils.calculate_technical_score_from_profile(skills)
    profile['benefits_score'] = int(professional['telework'] > 0)
    
    # Estimations dynamiques
    profile['description_word_count'] = (
        CalculationUtils.estimate_description_complexity(profile)
    )
    profile['nb_mots_cles_techniques'] = (
        CalculationUtils.estimate_technical_keywords(profile)
    )
    
    return profile


def _deduce_seniority(experience: float) -> str:
    """Déduit le niveau de séniorité selon l'expérience."""
    if experience < 1:
        return "Stage/Alternance"
    elif experience <= 3:
        return "Junior (1-3 ans)"
    elif experience <= 5:
        return "Mid-level"
    elif experience <= 8:
        return "Senior (5-8 ans)"
    else:
        return "Expert (8-12 ans)"


# ============================================================================
# WELCOME PAGE
# ============================================================================

def render_welcome_page(df_size: int, market_median: float, df_final: pd.DataFrame) -> None:
    """
    Affiche la page d'accueil avant soumission du formulaire.
    
    Args:
        df_size: Taille du dataset
        market_median: Médiane du marché
        df_final: DataFrame complet
    """
    st.markdown(f"""
    <div style='text-align: center; padding: 40px; background: #f0f2f6; 
                border-radius: 10px; margin: 20px 0;'>
        <div style='font-size: 60px; margin-bottom: 15px;'>🎯</div>
        <h3 style='color: #1f77b4; margin-bottom: 10px;'>
            Prêt à optimiser votre carrière ?
        </h3>
        <p style='color: #666; font-size: 16px;'>
            Remplissez le formulaire ci-dessus pour obtenir votre roadmap personnalisée<br>
            basée sur l'analyse de <strong>{df_size:,} offres réelles</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Statistiques du marché
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_exp = df_final['experience_final'].mean()
        st.metric("Expérience moyenne marché", f"{avg_exp:.1f} ans")
    
    with col2:
        avg_skills = df_final['skills_count'].mean() if 'skills_count' in df_final.columns else 0
        st.metric("Nombre moyen de compétences", f"{avg_skills:.1f}")
    
    with col3:
        st.metric("Salaire médian marché", f"{market_median:,.0f}€")


# ============================================================================
# TRAITEMENT DU PROFIL
# ============================================================================

def process_career_profile(
    profile: Dict[str, Any],
    model_utils: Any,
    real_market_data: np.ndarray
) -> Tuple[float, float, Dict]:
    """
    Traite le profil et effectue la prédiction de base.
    
    Args:
        profile: Profil complet de l'utilisateur
        model_utils: Gestionnaire du modèle
        real_market_data: Données du marché
        
    Returns:
        Tuple (base_salary, percentile, prediction_result)
        
    Raises:
        SystemExit: Si la prédiction échoue
    """
    st.success("✅ Profil enregistré ! Génération de votre feuille de route...")
    
    # Prédiction de base
    with st.spinner("🔄 Calcul de votre estimation salariale..."):
        base_pred = model_utils.predict(profile)
    
    if not base_pred:
        st.error("❌ Impossible de calculer votre estimation. Vérifiez votre profil.")
        st.stop()
    
    base_salary = base_pred['prediction']
    percentile = CalculationUtils.get_percentile_real(base_salary, real_market_data)
    
    return base_salary, percentile, base_pred


# ============================================================================
# MAIN - ORCHESTRATION
# ============================================================================

def main() -> None:
    """
    Fonction principale orchestrant l'affichage de la page carrière.
    
    Workflow:
        1. Initialisation (modèle, données, stats)
        2. Affichage du header
        3. Affichage du formulaire
        4. Traitement de la soumission
        5. Affichage des analyses (si profil soumis)
    """
    # Initialisation
    model_utils, df_final, real_market_data, market_median = initialize_career_page()
    
    # Header
    render_career_header(len(df_final), market_median)
    
    # Formulaire
    profile_data = render_profile_form()
    
    # Si pas de soumission, afficher welcome page
    if profile_data is None:
        render_welcome_page(len(df_final), market_median, df_final)
        st.stop()
    
    # Traitement du profil
    base_salary, percentile, base_pred = process_career_profile(
        profile_data,
        model_utils,
        real_market_data
    )
    
    # Stocker en session pour les autres modules
    st.session_state.career_profile_data = profile_data
    st.session_state.career_salary_data = base_salary
    st.session_state.career_percentile_data = percentile
    
    # Import et affichage des analyses
    from career_analysis import render_scorecard, render_positioning_diagnosis
    from career_roadmap import render_roadmap_section, render_effort_impact_matrix
    from career_transitions import render_transitions_analysis, render_salary_projection
    from career_export import render_negotiation_simulator, render_export_section
    
    st.markdown("---")
    
    # Analyses principales
    render_scorecard(profile_data, base_salary, percentile, df_final, model_utils)
    st.markdown("---")
    
    render_positioning_diagnosis(base_salary, percentile, market_median, real_market_data)
    st.markdown("---")
    
    render_roadmap_section(profile_data, base_salary, df_final, model_utils)
    st.markdown("---")
    
    render_effort_impact_matrix(profile_data, base_salary, df_final, model_utils)
    st.markdown("---")
    
    render_transitions_analysis(profile_data, base_salary, df_final, model_utils)
    st.markdown("---")
    
    render_salary_projection(profile_data, base_salary, model_utils)
    st.markdown("---")
    
    render_negotiation_simulator(profile_data, base_salary, df_final)
    st.markdown("---")
    
    render_export_section(profile_data, base_salary, percentile, df_final, market_median)


if __name__ == "__main__":
    main()