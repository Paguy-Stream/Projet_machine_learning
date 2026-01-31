"""
Page d'analyse du marché Data.

Cette page permet d'analyser le marché des métiers Data avec :
- Statistiques clés et KPIs
- Analyses par (postes, secteurs, géographie, compétences)
- Combinaisons de compétences
- comparateur de profils
- Export des données

Architecture:
    - Module principal :  chargement données
    - market_filters : Gestion des filtres sidebar
    - market_overview : Vue d'ensemble et statistiques
    - market_analysis : Analyses dpar onglet
    - market_benchmark : Benchmark et comparaisons
    - market_export : Export données et navigation


"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List
from pathlib import Path

from utils.config import Config, init_session_state, setup_page


# ============================================================================
# CONFIGURATION
# ============================================================================

def initialize_market_page() -> None:
    """
    Initialise la page du marché.
    
    Configure :
    - Le titre et l'icône de la page
    - L'état de session
    """
    setup_page("Analyse du Marché", "📊")
    init_session_state()


# ============================================================================
# CHARGEMENT DES DONNÉES
# ============================================================================

@st.cache_data
def load_market_data() -> Tuple[Optional[pd.DataFrame], int]:
    """
    Charge les données du marché depuis le dataset HelloWork.
    
    Effectue :
    - Chargement sélectif des colonnes nécessaires
    - Nettoyage des valeurs manquantes
    - Simplification des types de postes
    - Création de variables dérivées (stacks techniques)
    
    Returns:
        Tuple contenant :
            - DataFrame des données (ou None si erreur)
            - Nombre total d'offres dans le dataset
            
    Examples:
        >>> df, total = load_market_data()
        >>> print(f"Chargé {len(df)} offres sur {total}")
        Chargé 5868 offres sur 5868
        
    Notes:
        Utilise st.cache_data pour éviter les rechargements.
        En cas d'erreur, affiche un message et retourne (None, 0).
    """
    data_path = Config.DATA_PATH
    
    if not data_path.exists():
        st.error(f"❌ Fichier non trouvé : {data_path}")
        return None, 0
    
    try:
        # Colonnes à charger
        columns_to_load = _get_columns_to_load()
        
        # Chargement
        df = pd.read_csv(
            data_path,
            encoding='utf-8',
            usecols=columns_to_load
        )
        
        # Nettoyage
        df = _clean_market_data(df)
        
        # Features dérivées
        df = _create_derived_features(df)
        
        return df, len(df)
    
    except Exception as e:
        st.error(f"❌ Erreur chargement : {str(e)[:100]}")
        return None, 0


def _get_columns_to_load() -> List[str]:
    """
    Retourne la liste des colonnes nécessaires.
    
    Returns:
        Liste des noms de colonnes à charger
    """
    return [
        # Informations de base
        'job_type_with_desc', 'seniority', 'salary_mid',
        'location_final', 'sector_clean', 'experience_final',
        'contract_type_clean', 'telework_numeric',
        
        # Compétences techniques
        'contient_sql', 'contient_python', 'contient_r',
        'contient_tableau', 'contient_power_bi',
        'contient_aws', 'contient_azure', 'contient_gcp',
        'contient_spark', 'contient_machine_learning',
        'contient_deep_learning',
        
        # Avantages
        'has_teletravail', 'has_mutuelle', 'has_tickets', 'has_prime',
        
        # Scores
        'skills_count', 'technical_score', 'benefits_score'
    ]


def _clean_market_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les données du marché.
    
    Args:
        df: DataFrame brut
        
    Returns:
        DataFrame nettoyé
    """
    # Supprimer les lignes sans poste ou salaire
    df = df.dropna(subset=['job_type_with_desc', 'salary_mid'], how='all')
    
    # Simplifier les types de postes
    df['job_type_simplified'] = df['job_type_with_desc'].apply(_simplify_job_type)
    
    # Nettoyer localisation et secteur
    df['location_clean'] = df['location_final'].fillna('Non spécifié')
    df['sector_clean'] = df['sector_clean'].fillna('Non spécifié')
    
    return df


def _simplify_job_type(job: str) -> str:
    """
    Simplifie le type de poste pour l'analyse.
    
    Args:
        job: Type de poste complet
        
    Returns:
        Type de poste simplifié
        
    Examples:
        >>> _simplify_job_type("Data Engineer (Senior)")
        'Data Engineer'
        >>> _simplify_job_type("Unknown role")
        'Autre Data Role'
    """
    if pd.isna(job):
        return 'Autre'
    
    # Mapping des types de postes
    job_mapping = {
        'Data Engineer': 'Data Engineer',
        'Data Scientist': 'Data Scientist',
        'Data Analyst': 'Data Analyst',
        'BI/Analytics': 'BI/Analytics',
        'Data Management': 'Data Management',
        'AI/ML': 'AI/ML Specialist',
        'Data Consultant': 'Data Consultant'
    }
    
    for key, value in job_mapping.items():
        if key in job:
            return value
    
    return 'Autre Data Role'


def _create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des features dérivées pour l'analyse.
    
    Args:
        df: DataFrame nettoyé
        
    Returns:
        DataFrame avec features additionnelles
        
    Notes:
        Crée 3 indicateurs de stack technique :
        - has_modern_stack : Python + Cloud + Spark
        - has_ds_stack : Python + ML + Cloud
        - has_bi_stack : SQL + (Tableau OU Power BI)
    """
    # Stack technique moderne
    df['has_modern_stack'] = (
        (df['contient_python'] == 1) &
        ((df['contient_aws'] == 1) | 
         (df['contient_azure'] == 1) | 
         (df['contient_gcp'] == 1)) &
        (df['contient_spark'] == 1)
    ).astype(int)
    
    # Stack Data Scientist
    df['has_ds_stack'] = (
        (df['contient_python'] == 1) & 
        (df['contient_machine_learning'] == 1) & 
        ((df['contient_aws'] == 1) | 
         (df['contient_azure'] == 1) | 
         (df['contient_gcp'] == 1))
    ).astype(int)
    
    # Stack BI
    df['has_bi_stack'] = (
        (df['contient_sql'] == 1) & 
        ((df['contient_tableau'] == 1) | 
         (df['contient_power_bi'] == 1))
    ).astype(int)
    
    return df


# ============================================================================
# HEADER ET STATISTIQUES
# ============================================================================

def render_market_header(
    filtered_size: int,
    total_size: int,
    filters_info: Dict[str, int]
) -> None:
    """
    Affiche l'en-tête de la page avec statistiques.
    
    Args:
        filtered_size: Nombre d'offres après filtres
        total_size: Nombre total d'offres
        filters_info: Info sur les filtres actifs
    """
    st.title(" Analyse du marché Data")
    
    st.markdown(f"""
    Exploration de **{filtered_size:,}** offres sur **{total_size:,}** du dataset HelloWork  
    _Filtres actifs : {filters_info['jobs']} postes, {filters_info['locations']} villes, 
    {filters_info['sectors']} secteurs_
    """)
    
    st.markdown("---")


def render_kpi_metrics(
    filtered_data: pd.DataFrame,
    market_data: pd.DataFrame
) -> None:
    """
    Affiche les KPIs principaux du marché.
    
    Args:
        filtered_data: Données filtrées
        market_data: Données complètes du marché
        
    Notes:
        Affiche 5 métriques :
        - Nombre d'offres
        - Salaire médian
        - Salaire moyen
        - Nombre moyen de compétences
        - % avec télétravail
    """
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Métrique 1 : Nombre d'offres
    with col1:
        total_size = len(market_data)
        filtered_size = len(filtered_data)
        pct = (filtered_size / total_size * 100) if total_size > 0 else 0
        
        st.metric(
            "📋 Offres",
            f"{filtered_size:,}",
            delta=f"{pct:.1f}% du total"
        )
    
    # Métrique 2 : Salaire médian
    with col2:
        median_salary = filtered_data['salary_mid'].median()
        global_median = market_data['salary_mid'].median()
        delta_median = median_salary - global_median
        
        st.metric(
            "💰 Médiane",
            f"{median_salary:,.0f} €",
            delta=f"{delta_median:+,.0f}€ vs global"
        )
    
    # Métrique 3 : Salaire moyen
    with col3:
        mean_salary = filtered_data['salary_mid'].mean()
        st.metric("📊 Moyenne", f"{mean_salary:,.0f} €")
    
    # Métrique 4 : Compétences moyennes
    with col4:
        if 'skills_count' in filtered_data.columns:
            avg_skills = filtered_data['skills_count'].mean()
            st.metric("🛠️ Skills moy.", f"{avg_skills:.1f}")
        else:
            st.metric("🛠️ Skills moy.", "N/A")
    
    # Métrique 5 : Télétravail
    with col5:
        if 'telework_numeric' in filtered_data.columns:
            telework_pct = (filtered_data['telework_numeric'] > 0).mean() * 100
            st.metric("🏠 Télétravail", f"{telework_pct:.0f}%")
        else:
            st.metric("🏠 Télétravail", "N/A")
    
    st.markdown("---")


# ============================================================================
# MAIN - ORCHESTRATION
# ============================================================================

def main() -> None:
    """
    Fonction principale orchestrant l'affichage de la page marché.
    
    Workflow:
        1. Initialisation de la page
        2. Chargement des données
        3. Affichage de la sidebar avec filtres
        4. Application des filtres
        5. Affichage du header et KPIs
        6. Affichage des insights actionnables
        7. Affichage des onglets d'analyse
        8. Affichage de l'export et navigation
    """
    # Initialisation
    initialize_market_page()
    
    # Chargement des données
    market_data, total_size = load_market_data()
    
    if market_data is None:
        st.stop()
    
    # Import des modules d'affichage
    from market_filters import render_sidebar_filters
    from market_overview import render_insights_section
    from market_analysis import render_analysis_tabs
    from market_export import render_export_and_navigation
    
    # Affichage de la sidebar avec filtres
    filtered_data, filters_info = render_sidebar_filters(market_data)
    
    # Vérification des données filtrées
    if len(filtered_data) == 0:
        st.warning("⚠️ Aucune donnée disponible avec les filtres actuels")
        st.stop()
    
    # Header et KPIs
    render_market_header(len(filtered_data), total_size, filters_info)
    render_kpi_metrics(filtered_data, market_data)
    
    # Insights actionnables
    render_insights_section(filtered_data)
    
    st.markdown("---")
    
    # Onglets d'analyse détaillée
    render_analysis_tabs(filtered_data, market_data)
    
    st.markdown("---")
    
    # Export et navigation
    render_export_and_navigation(filtered_data, total_size, filters_info)


if __name__ == "__main__":
    main()
