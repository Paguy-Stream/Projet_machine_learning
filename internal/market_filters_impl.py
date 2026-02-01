"""
Module de gestion des filtres de la page Marché

Ce module contient toutes les fonctions pour :
- Afficher les filtres dans la sidebar
- Gérer l'état des filtres
- Appliquer les filtres aux données
- Réinitialiser les filtres

"""

import streamlit as st
import pandas as pd
from typing import Tuple, Dict, List


# ============================================================================
# FILTRES SIDEBAR
# ============================================================================

def render_sidebar_filters(
    market_data: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Affiche les filtres dans la sidebar et applique les filtres aux données.
    
    Args:
        market_data: DataFrame complet du marché
        
    Returns:
        Tuple contenant :
            - DataFrame filtré
            - Dict avec info sur les filtres actifs
            
    Examples:
        >>> filtered_df, info = render_sidebar_filters(market_data)
        >>> print(f"{len(filtered_df)} offres filtrées")
    """
    with st.sidebar:
        st.header("🎛️ Filtres")
        
        # Section 1 : Filtres catégoriels
        job_filter = _render_job_filter(market_data)
        location_filter = _render_location_filter(market_data)
        sector_filter = _render_sector_filter(market_data)
        
        st.markdown("---")
        
        # Section 2 : Filtres numériques
        salary_min, salary_max = _render_salary_filter(market_data)
        exp_min, exp_max = _render_experience_filter(market_data)
        
        st.markdown("---")
        
        # Section 3 : Filtres techniques
        tech_filter = _render_tech_stack_filter()
        
        st.markdown("---")
        
        # Section 4 : Boutons d'action
        _render_filter_actions(market_data)
    
    # Application des filtres
    filtered_data = _apply_all_filters(
        market_data,
        job_filter,
        location_filter,
        sector_filter,
        salary_min,
        salary_max,
        exp_min,
        exp_max,
        tech_filter
    )
    
    # Info sur les filtres actifs
    filters_info = {
        'jobs': len(job_filter),
        'locations': len(location_filter),
        'sectors': len(sector_filter)
    }
    
    return filtered_data, filters_info


# ============================================================================
# FILTRES INDIVIDUELS
# ============================================================================

def _render_job_filter(market_data: pd.DataFrame) -> List[str]:
    """
    Affiche le filtre des types de postes.
    
    Args:
        market_data: DataFrame du marché
        
    Returns:
        Liste des postes sélectionnés
        
    FIX: La clé est déjà "job_filter" donc OK, mais on vérifie l'initialisation
    """
    job_options = sorted(market_data['job_type_simplified'].dropna().unique())
    
    # Initialisation de la valeur par défaut
    if 'job_filter' not in st.session_state:
        st.session_state.job_filter = job_options[:3]
    
    job_filter = st.multiselect(
        "Type de poste",
        job_options,
        default=st.session_state.job_filter,
        key='market_filter_job',  # FIX: Clé explicite et unique
        help="Sélectionnez un ou plusieurs types de postes"
    )
    
    return job_filter


def _render_location_filter(market_data: pd.DataFrame) -> List[str]:
    """
    Affiche le filtre des villes.
    
    Args:
        market_data: DataFrame du marché
        
    Returns:
        Liste des villes sélectionnées
        
    FIX: Clé unique ajoutée
    """
    location_options = sorted(market_data['location_clean'].dropna().unique())
    
    # Initialisation
    if 'location_filter' not in st.session_state:
        default_cities = ['Paris', 'Lyon', 'Toulouse']
        st.session_state.location_filter = [
            c for c in default_cities if c in location_options
        ]
    
    location_filter = st.multiselect(
        "Ville",
        location_options,
        default=st.session_state.location_filter,
        key='market_filter_location',  # FIX: Clé explicite et unique
        help="Sélectionnez une ou plusieurs villes"
    )
    
    return location_filter


def _render_sector_filter(market_data: pd.DataFrame) -> List[str]:
    """
    Affiche le filtre des secteurs.
    
    Args:
        market_data: DataFrame du marché
        
    Returns:
        Liste des secteurs sélectionnés
        
    FIX: Clé unique ajoutée
    """
    sector_options = sorted(market_data['sector_clean'].dropna().unique())
    
    # Initialisation
    if 'sector_filter' not in st.session_state:
        default_sectors = ['Tech', 'Banque', 'ESN']
        st.session_state.sector_filter = [
            s for s in default_sectors if s in sector_options
        ]
    
    sector_filter = st.multiselect(
        "Secteur",
        sector_options,
        default=st.session_state.sector_filter,
        key='market_filter_sector',  # FIX: Clé explicite et unique
        help="Sélectionnez un ou plusieurs secteurs"
    )
    
    return sector_filter


def _render_salary_filter(market_data: pd.DataFrame) -> Tuple[float, float]:
    """
    Affiche le filtre de fourchette salariale.
    
    Args:
        market_data: DataFrame du marché
        
    Returns:
        Tuple (salaire_min, salaire_max)
        
    FIX: Clé unique ajoutée
    """
    salary_min_val = int(market_data['salary_mid'].min())
    salary_max_val = int(market_data['salary_mid'].max())
    
    salary_min, salary_max = st.slider(
        "Fourchette de salaire (€)",
        min_value=salary_min_val,
        max_value=salary_max_val,
        value=(salary_min_val, salary_max_val),
        step=1000,
        key='market_filter_salary',  # FIX: Clé unique
        help="Ajustez la fourchette salariale"
    )
    
    return salary_min, salary_max


def _render_experience_filter(market_data: pd.DataFrame) -> Tuple[float, float]:
    """
    Affiche le filtre d'expérience.
    
    Args:
        market_data: DataFrame du marché
        
    Returns:
        Tuple (exp_min, exp_max)
        
    FIX: Clé unique ajoutée
    """
    if 'experience_final' not in market_data.columns:
        return 0.0, 30.0
    
    exp_vals = market_data['experience_final'].dropna()
    exp_min_val = float(exp_vals.min())
    exp_max_val = float(exp_vals.max())
    
    exp_min, exp_max = st.slider(
        "Années d'expérience",
        min_value=exp_min_val,
        max_value=exp_max_val,
        value=(exp_min_val, exp_max_val),
        step=0.5,
        key='market_filter_experience',  # FIX: Clé unique
        help="Filtrez par niveau d'expérience"
    )
    
    return exp_min, exp_max


def _render_tech_stack_filter() -> List[str]:
    """
    Affiche le filtre des stacks techniques.
    
    Returns:
        Liste des stacks sélectionnés
        
    FIX: Clé unique ajoutée
    """
    tech_options = [
        'Python + Cloud + Spark',
        'BI Tools',
        'Machine Learning',
        'SQL',
        'Deep Learning'
    ]
    
    tech_filter = st.multiselect(
        "Stack technique",
        tech_options,
        key='market_filter_tech',  # FIX: Clé unique
        help="Filtrez par stack technique (ET logique)"
    )
    
    return tech_filter


# ============================================================================
# BOUTONS D'ACTION
# ============================================================================

def _render_filter_actions(market_data: pd.DataFrame) -> None:
    """
    Affiche les boutons d'action des filtres.
    
    Args:
        market_data: DataFrame du marché
        
    FIX: Clés uniques ajoutées aux boutons
    """
    col1, col2 = st.columns(2)
    
    with col1:
        # FIX: Clé unique
        if st.button("🔄 Réinitialiser", use_container_width=True, key='market_btn_reset'):
            _reset_filters()
    
    with col2:
        # FIX: Clé unique
        if st.button("📊 Tout afficher", use_container_width=True, key='market_btn_show_all'):
            _show_all_filters(market_data)


def _reset_filters() -> None:
    """Réinitialise tous les filtres aux valeurs par défaut."""
    keys_to_delete = ['job_filter', 'location_filter', 'sector_filter']
    
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]
    
    st.rerun()


def _show_all_filters(market_data: pd.DataFrame) -> None:
    """
    Active tous les filtres (affiche toutes les données).
    
    Args:
        market_data: DataFrame du marché
    """
    all_jobs = sorted(
        market_data['job_type_simplified'].dropna().unique()
    )
    all_locations = sorted(
        market_data['location_clean'].dropna().unique()
    )
    all_sectors = sorted(
        market_data['sector_clean'].dropna().unique()
    )
    
    st.session_state.job_filter = all_jobs
    st.session_state.location_filter = all_locations
    st.session_state.sector_filter = all_sectors
    
    st.rerun()


# ============================================================================
# APPLICATION DES FILTRES
# ============================================================================

def _apply_all_filters(
    market_data: pd.DataFrame,
    job_filter: List[str],
    location_filter: List[str],
    sector_filter: List[str],
    salary_min: float,
    salary_max: float,
    exp_min: float,
    exp_max: float,
    tech_filter: List[str]
) -> pd.DataFrame:
    """
    Applique tous les filtres aux données du marché.
    
    Args:
        market_data: DataFrame complet
        job_filter: Postes sélectionnés
        location_filter: Villes sélectionnées
        sector_filter: Secteurs sélectionnés
        salary_min: Salaire minimum
        salary_max: Salaire maximum
        exp_min: Expérience minimum
        exp_max: Expérience maximum
        tech_filter: Stacks techniques sélectionnés
        
    Returns:
        DataFrame filtré
    """
    filtered_data = market_data.copy()
    
    # Filtres catégoriels
    if job_filter:
        filtered_data = filtered_data[
            filtered_data['job_type_simplified'].isin(job_filter)
        ]
    
    if location_filter:
        filtered_data = filtered_data[
            filtered_data['location_clean'].isin(location_filter)
        ]
    
    if sector_filter:
        filtered_data = filtered_data[
            filtered_data['sector_clean'].isin(sector_filter)
        ]
    
    # Filtre salarial
    filtered_data = filtered_data[
        (filtered_data['salary_mid'] >= salary_min) &
        (filtered_data['salary_mid'] <= salary_max)
    ]
    
    # Filtre expérience
    if 'experience_final' in filtered_data.columns:
        filtered_data = filtered_data[
            (filtered_data['experience_final'] >= exp_min) &
            (filtered_data['experience_final'] <= exp_max)
        ]
    
    # Filtres techniques (ET logique)
    if tech_filter:
        filtered_data = _apply_tech_filters(filtered_data, tech_filter)
    
    return filtered_data


def _apply_tech_filters(
    data: pd.DataFrame,
    tech_filter: List[str]
) -> pd.DataFrame:
    """
    Applique les filtres de stack technique.
    
    Args:
        data: DataFrame à filtrer
        tech_filter: Liste des stacks sélectionnés
        
    Returns:
        DataFrame filtré
    """
    for tech in tech_filter:
        if tech == 'Python + Cloud + Spark':
            data = data[data['has_modern_stack'] == 1]
        
        elif tech == 'BI Tools':
            data = data[
                (data['contient_tableau'] == 1) | 
                (data['contient_power_bi'] == 1)
            ]
        
        elif tech == 'Machine Learning':
            data = data[data['contient_machine_learning'] == 1]
        
        elif tech == 'Deep Learning':
            if 'contient_deep_learning' in data.columns:
                data = data[data['contient_deep_learning'] == 1]
        
        elif tech == 'SQL':
            data = data[data['contient_sql'] == 1]
    
    return data


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'render_sidebar_filters'
]