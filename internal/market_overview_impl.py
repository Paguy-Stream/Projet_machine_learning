"""
Module des insights et vue d'ensemble du marché.

Ce module contient les fonctions pour afficher :
- Insights actionnables (top compétences, villes, secteurs)
- Vue d'ensemble avec statistiques clés
- Métriques comparatives


"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

from utils.config import Config


# ============================================================================
# INSIGHTS ACTIONNABLES
# ============================================================================

def render_insights_section(filtered_data: pd.DataFrame) -> None:
    """
    Affiche la section des insights actionnables du marché.
    
    Args:
        filtered_data: DataFrame des données filtrées
        
    Notes:
        Affiche 3 colonnes d'insights :
        - Top compétences rentables
        - Meilleures villes
        - Secteurs les plus généreux
    """
    st.markdown("## 💡 Insights du marché")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        _render_top_skills(filtered_data)
    
    with col2:
        _render_best_cities(filtered_data)
    
    with col3:
        _render_top_sectors(filtered_data)


# ============================================================================
# TOP COMPÉTENCES
# ============================================================================

def _render_top_skills(filtered_data: pd.DataFrame) -> None:
    """
    Affiche les compétences les plus rentables.
    
    Args:
        filtered_data: DataFrame filtré
    """
    st.markdown("### 🎯 Top compétences rentables")
    
    skill_impacts = _calculate_skill_impacts(filtered_data)
    
    if skill_impacts:
        sorted_impacts = sorted(
            skill_impacts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        for skill, impact in sorted_impacts:
            st.success(f"**{skill}** : +{impact:,.0f}€")
    else:
        st.info("Pas assez de données")


def _calculate_skill_impacts(data: pd.DataFrame) -> Dict[str, float]:
    """
    Calcule l'impact salarial de chaque compétence.
    
    Args:
        data: DataFrame des données
        
    Returns:
        Dict {compétence: impact_salarial}
    """
    skills_to_analyze = {
        'Python': 'contient_python',
        'SQL': 'contient_sql',
        'Machine Learning': 'contient_machine_learning',
        'Deep Learning': 'contient_deep_learning',
        'AWS': 'contient_aws',
        'Spark': 'contient_spark',
        'Tableau': 'contient_tableau'
    }
    
    skill_impacts = {}
    
    for skill_name, col_name in skills_to_analyze.items():
        if col_name not in data.columns:
            continue
        
        with_skill = data[data[col_name] == 1]['salary_mid'].median()
        without_skill = data[data[col_name] == 0]['salary_mid'].median()
        
        impact = with_skill - without_skill
        
        if not np.isnan(impact) and impact != 0:
            skill_impacts[skill_name] = impact
    
    return skill_impacts


# ============================================================================
# MEILLEURES VILLES
# ============================================================================

def _render_best_cities(filtered_data: pd.DataFrame) -> None:
    """
    Affiche les 3 meilleures villes par salaire médian.
    
    Args:
        filtered_data: DataFrame filtré
    """
    st.markdown("### 🏙️ Meilleures villes")
    
    city_salaries = _calculate_city_salaries(filtered_data)
    
    if len(city_salaries) > 0:
        # Top 3 villes
        top_cities = city_salaries.sort_values(
            'median',
            ascending=False
        ).head(3)
        
        for city, row in top_cities.iterrows():
            # Récupérer le multiplicateur dynamique
            city_mult = Config.get_city_multiplier(city)
            st.info(f"**{city}** : {row['median']:,.0f}€ (×{city_mult:.2f})")
    else:
        st.info("Pas assez de données")


def _calculate_city_salaries(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les statistiques salariales par ville.
    
    Args:
        data: DataFrame des données
        
    Returns:
        DataFrame avec médiane et count par ville
    """
    city_salaries = data.groupby('location_clean')['salary_mid'].agg([
        'median', 'count'
    ])
    
    # Filtrer les villes avec au moins 5 offres
    city_salaries = city_salaries[city_salaries['count'] >= 5]
    
    return city_salaries


# ============================================================================
# TOP SECTEURS
# ============================================================================

def _render_top_sectors(filtered_data: pd.DataFrame) -> None:
    """
    Affiche les 3 secteurs les plus généreux.
    
    Args:
        filtered_data: DataFrame filtré
    """
    st.markdown("### 💼 Secteurs les plus généreux")
    
    sector_salaries = _calculate_sector_salaries(filtered_data)
    
    if len(sector_salaries) > 0:
        # Top 3 secteurs
        top_sectors = sector_salaries.sort_values(
            'median',
            ascending=False
        ).head(3)
        
        for sector, row in top_sectors.iterrows():
            # Récupérer le multiplicateur dynamique
            sector_mult = Config.get_sector_multiplier(sector)
            st.warning(f"**{sector}** : {row['median']:,.0f}€ (×{sector_mult:.2f})")
    else:
        st.info("Pas assez de données")


def _calculate_sector_salaries(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les statistiques salariales par secteur.
    
    Args:
        data: DataFrame des données
        
    Returns:
        DataFrame avec médiane et count par secteur
    """
    sector_salaries = data.groupby('sector_clean')['salary_mid'].agg([
        'median', 'count'
    ])
    
    # Filtrer les secteurs avec au moins 5 offres
    sector_salaries = sector_salaries[sector_salaries['count'] >= 5]
    
    return sector_salaries


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'render_insights_section'
]
