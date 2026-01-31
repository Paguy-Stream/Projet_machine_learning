"""
Page de prédiction de salaires pour les métiers Data.

Cette page permet aux utilisateurs de :
- Remplir un formulaire détaillé sur leur profil professionnel
- Obtenir une estimation salariale basée sur le modèle XGBoost
- Visualiser leur positionnement sur le marché
- Analyser les facteurs d'influence (SHAP)
- Comparer différents scénarios (secteurs, ML/DL, expérience)
- Exporter leurs résultats

Architecture:
    - Section 1: Configuration et initialisation
    - Section 2: Formulaire de saisie (sidebar)
    - Section 3: Affichage des résultats principaux
    - Section 4: Analyses SHAP et explications
    - Section 5: Comparaisons et projections
    - Section 6: Actions et export
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional, Any

from utils.config import Config, init_session_state, setup_page
from utils.model_utils import (
    init_model_utils,
    ChartUtils,
    CalculationUtils,
    DataDistributions
)


# ============================================================================
# CONFIGURATION ET INITIALISATION
# ============================================================================

def initialize_page() -> Tuple[Any, np.ndarray, Dict[str, float]]:
    """
    Initialise la page et charge les ressources nécessaires.
    
    Returns:
        Tuple contenant :
            - model_utils: Gestionnaire du modèle
            - real_market_data: Données salariales du marché
            - market_stats: Statistiques du marché (médiane, quartiles, etc.)
    
    Raises:
        SystemExit: Si les données du marché ne peuvent pas être chargées
    """
    setup_page("Estimation de Salaire", "🔮")
    init_session_state()
    
    # Initialisation du modèle (avec cache)
    if 'model_utils' not in st.session_state:
        model_utils = init_model_utils()
        st.session_state.model_utils = model_utils
    else:
        model_utils = st.session_state.model_utils
    
    # Chargement des données de marché
    real_market_data = model_utils.get_real_market_data()
    
    if real_market_data is None:
        st.error("❌ Données salariales non chargées – vérifiez `test_data.pkl`")
        st.stop()
    
    # Calcul des statistiques du marché
    market_stats = {
        'median': float(np.median(real_market_data)),
        'q1': float(np.percentile(real_market_data, 25)),
        'q3': float(np.percentile(real_market_data, 75)),
        'gauge_min': float(max(0, np.percentile(real_market_data, 1))),
        'gauge_max': float(np.percentile(real_market_data, 99))
    }
    
    return model_utils, real_market_data, market_stats


def render_page_header() -> None:
    """Affiche l'en-tête de la page avec titre et description."""
    st.title("🔮 Estimation de votre salaire")
    
    total_offers = DataDistributions.get_total_offers()
    
    st.markdown(f"""
    Remplissez le formulaire ci-dessous pour obtenir une estimation personnalisée basée sur 
    **{total_offers:,} offres réelles** avec calculs automatiques.
    """)


# ============================================================================
# FORMULAIRE DE SAISIE
# ============================================================================

def auto_detect_seniority(experience: float) -> str:
    """
    Déduit automatiquement le niveau de séniorité selon l'expérience.
    
    Args:
        experience: Nombre d'années d'expérience
        
    Returns:
        Niveau de séniorité correspondant
        
    Examples:
        >>> auto_detect_seniority(2.5)
        'Junior (1-3 ans)'
        >>> auto_detect_seniority(6)
        'Senior (5-8 ans)'
    """
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


def render_profile_form() -> Optional[Dict[str, Any]]:
    """
    Affiche le formulaire de saisie du profil dans la sidebar.
    
    Returns:
        Dict du profil complet si soumis, None sinon
        
    Notes:
        Le formulaire est divisé en 4 sections :
        1. Poste et expérience
        2. Localisation et formation
        3. Compétences techniques
        4. Avantages sociaux
    """
    with st.sidebar:
        st.header("📋 Votre profil")
        
        # Options avancées
        _render_advanced_options()
        
        # Récupération de la corrélation ML/DL pour l'info-bulle
        try:
            corr_val = DataDistributions.get_ml_dl_correlation()
            corr_info = f"(Corrélation ML/DL : {corr_val:.1%})"
        except:
            corr_info = ""
        
        with st.form("profile_form"):
            # Section 1: Poste et expérience
            job_data = _render_job_section()
            
            st.markdown("---")
            
            # Section 2: Localisation et formation
            location_data = _render_location_section()
            
            st.markdown("---")
            
            # Section 3: Compétences techniques
            skills_data = _render_skills_section(corr_info)
            
            st.markdown("---")
            
            # Section 4: Avantages
            benefits_data = _render_benefits_section()
            
            st.markdown("---")
            
            # Bouton de soumission
            submitted = st.form_submit_button(
                "🔮 Estimer mon salaire",
                type="primary",
                use_container_width=True
            )
            
            if submitted:
                return _build_complete_profile(
                    job_data,
                    location_data,
                    skills_data,
                    benefits_data
                )
    
    return None


def _render_advanced_options() -> None:
    """Affiche les options  (rechargement des stats)."""
    with st.expander("⚙️ Options "):
        if st.button("🔄 Recharger statistiques dataset"):
            DataDistributions.reload()
            st.success("✅ Statistiques rechargées !")
            st.rerun()


def _render_job_section() -> Dict[str, Any]:
    """
    Affiche la section Poste et expérience du formulaire.
    
    Returns:
        Dict contenant job_type, experience, seniority, contract
    """
    st.markdown("### 💼 Poste")
    
    job_type = st.selectbox("Type de poste", Config.JOB_TYPES)
    
    experience = st.number_input(
        "Années d'expérience",
        min_value=0.0,
        max_value=30.0,
        value=4.0,
        step=0.5,
        help="Le niveau hiérarchique sera déduit automatiquement"
    )
    
    # Déduction automatique du seniority
    seniority = auto_detect_seniority(experience)
    st.info(f"**Niveau déduit** : {seniority}")
    
    contract = st.radio(
        "Type de contrat",
        Config.CONTRACT_TYPES,
        index=0,
        horizontal=True,
        help="Note : 97% des offres sont en CDI, l'impact sera faible pour les autres contrats"
    )
    
    return {
        'job_type': job_type,
        'experience': experience,
        'seniority': seniority,
        'contract': contract
    }


def _render_location_section() -> Dict[str, Any]:
    """
    Affiche la section Localisation & Formation du formulaire.
    
    Returns:
        Dict contenant education, location, telework, sector
    """
    st.markdown("### 📍 Localisation & Formation")
    
    education = st.selectbox("Niveau d'études", Config.EDUCATION_LEVELS, index=4)
    location = st.selectbox("Ville", Config.CITIES, index=0)
    
    telework = st.select_slider(
        "Télétravail",
        ["Présentiel", "Hybride (1-3j)", "Full remote"],
        value="Hybride (1-3j)"
    )
    
    sector = st.selectbox(
        "Secteur d'activité",
        Config.SECTORS,
        key="sector_input"
    )
    
    return {
        'education': education,
        'location': location,
        'telework': telework,
        'sector': sector
    }


def _render_skills_section(corr_info: str) -> Dict[str, bool]:
    """
    Affiche la section Compétences techniques du formulaire.
    
    Args:
        corr_info: Information sur la corrélation ML/DL
        
    Returns:
        Dict avec toutes les compétences (booléens)
    """
    st.markdown("### 🛠️ Compétences techniques")
    
    # Langages & Outils
    st.markdown("**Langages & Outils**")
    col1, col2 = st.columns(2)
    
    with col1:
        python = st.checkbox("Python", value=True)
        sql = st.checkbox("SQL", value=True)
        r = st.checkbox("R")
    
    with col2:
        tableau = st.checkbox("Tableau")
        power_bi = st.checkbox("Power BI")
    
    # Cloud & Big Data
    st.markdown("**Cloud & Big Data**")
    col3, col4 = st.columns(2)
    
    with col3:
        aws = st.checkbox("AWS")
        azure = st.checkbox("Azure")
    
    with col4:
        gcp = st.checkbox("GCP")
        spark = st.checkbox("Spark")
    
    # Intelligence Artificielle
    st.markdown("**Intelligence Artificielle**")
    
    ml = st.checkbox(
        "Machine Learning",
        help="Algorithmes ML classiques : régression, classification, clustering"
    )
    
    dl = st.checkbox(
        "Deep Learning",
        help=f"Réseaux de neurones : CNN, RNN, Transformers {corr_info}"
    )
    
    # Data Engineering
    st.markdown("**Data Engineering**")
    etl = st.checkbox("ETL / Pipelines de données")
    
    return {
        'contient_python': python,
        'contient_sql': sql,
        'contient_r': r,
        'contient_tableau': tableau,
        'contient_power_bi': power_bi,
        'contient_aws': aws,
        'contient_azure': azure,
        'contient_gcp': gcp,
        'contient_spark': spark,
        'contient_machine_learning': ml,
        'contient_deep_learning': dl,
        'contient_etl': etl
    }


def _render_benefits_section() -> Dict[str, bool]:
    """
    Affiche la section Avantages du formulaire.
    
    Returns:
        Dict avec les avantages (booléens)
    """
    st.markdown("### 🎁 Avantages")
    
    col5, col6 = st.columns(2)
    
    with col5:
        teletravail_benefit = st.checkbox("Télétravail", value=True)
        mutuelle = st.checkbox("Mutuelle")
    
    with col6:
        tickets = st.checkbox("Tickets resto")
        prime = st.checkbox("Prime")
    
    return {
        'has_teletravail': teletravail_benefit,
        'has_mutuelle': mutuelle,
        'has_tickets': tickets,
        'has_prime': prime
    }


def _build_complete_profile(
    job_data: Dict,
    location_data: Dict,
    skills_data: Dict,
    benefits_data: Dict
) -> Dict[str, Any]:
    """
    Construit le profil complet avec calculs dynamiques.
    
    Args:
        job_data: Données métier
        location_data: Données localisation
        skills_data: Compétences techniques
        benefits_data: Avantages sociaux
        
    Returns:
        Dict du profil complet prêt pour la prédiction
    """
    # Construction du profil partiel pour calculs dynamiques
    profile_partial = {
        'job_type': job_data['job_type'],
        'seniority': job_data['seniority'],
        'experience_final': float(job_data['experience']),
        'sector_clean': location_data['sector'],
        'location_final': location_data['location'],
        **skills_data
    }
    
    # Calcul du nombre de compétences
    profile_partial['skills_count'] = (
        CalculationUtils.calculate_skills_count_from_profile(profile_partial)
    )
    
    # Calculs dynamiques
    desc_word_count = CalculationUtils.estimate_description_complexity(
        profile_partial
    )
    tech_keywords = CalculationUtils.estimate_technical_keywords(
        profile_partial
    )
    
    # Conversion du télétravail en valeur numérique
    telework_map = {
        "Présentiel": 0.0,
        "Hybride (1-3j)": 0.5,
        "Full remote": 1.0
    }
    telework_numeric = telework_map.get(location_data['telework'], 0.5)
    
    # Profil complet
    return {
        'job_type': job_data['job_type'],
        'seniority': job_data['seniority'],
        'experience_final': float(job_data['experience']),
        'contract_type_clean': job_data['contract'],
        'education_clean': location_data['education'],
        'location_final': location_data['location'],
        'sector_clean': location_data['sector'],
        'telework_numeric': telework_numeric,
        **skills_data,
        **benefits_data,
        'skills_count': profile_partial['skills_count'],
        'technical_score': CalculationUtils.calculate_technical_score_from_profile(
            profile_partial
        ),
        'benefits_score': CalculationUtils.calculate_benefits_score_from_profile(
            benefits_data
        ),
        'description_word_count': desc_word_count,
        'nb_mots_cles_techniques': tech_keywords
    }


# ============================================================================
# TRAITEMENT DE LA PRÉDICTION
# ============================================================================

def process_prediction(
    profile_data: Dict[str, Any],
    model_utils: Any
) -> bool:
    """
    Effectue la prédiction et stocke les résultats en session.
    
    Args:
        profile_data: Profil complet de l'utilisateur
        model_utils: Gestionnaire du modèle
        
    Returns:
        True si la prédiction a réussi, False sinon
    """
    with st.spinner("🔄 Calcul de l'estimation et analyse SHAP..."):
        prediction_result = model_utils.predict(profile_data)
        shap_explanation = model_utils.explain_prediction(profile_data)
    
    if prediction_result:
        # Stockage en session
        st.session_state.prediction_made = True
        st.session_state.last_prediction = prediction_result
        st.session_state.current_profile = profile_data
        st.session_state.shap_explanation = shap_explanation
        
        st.balloons()
        st.success("✅ Estimation terminée !")
        
        return True
    
    return False


# ============================================================================
# MAIN - ORCHESTRATION
# ============================================================================

def main() -> None:
    """
    Fonction principale orchestrant l'affichage de la page.
    
    Workflow:
        1. Initialisation (modèle, données, stats)
        2. Affichage du header
        3. Affichage du formulaire
        4. Traitement de la soumission
        5. Affichage des résultats (si disponibles)
        6. Page d'accueil (si pas de résultats)
    """
    # Initialisation
    model_utils, real_market_data, market_stats = initialize_page()
    
    # Header
    render_page_header()
    
    # Formulaire
    profile_data = render_profile_form()
    
    # Traitement de la soumission
    if profile_data is not None:
        process_prediction(profile_data, model_utils)
    
    # Affichage des résultats ou page d'accueil
    if st.session_state.get('prediction_made'):
        # Import des fonctions d'affichage (partie 2)
        from prediction_display import render_results
        render_results(model_utils, real_market_data, market_stats)
    else:
        from prediction_display import render_welcome_page
        render_welcome_page()


if __name__ == "__main__":
    main()
