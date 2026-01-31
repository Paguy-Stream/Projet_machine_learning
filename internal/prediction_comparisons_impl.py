"""
Module d'analyses comparatives pour les prédictions salariales.

Ce module contient toutes les analyses comparatives avancées :
- Comparaison par secteur d'activité
- Projection de carrière selon l'expérience
- Comparaison par localisation
- Impact des compétences
- Simulations de scénarios
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional, Any

from utils.model_utils import CalculationUtils, DataDistributions


# ============================================================================
# COMPARAISON PAR SECTEUR
# ============================================================================

def render_sector_comparison(
    profile: Dict[str, Any],
    model_utils: Any
) -> None:
    """
    Affiche la comparaison salariale par secteur d'activité.
    
    Compare le salaire du profil dans différents secteurs clés pour
    identifier les opportunités d'augmentation.
    
    Args:
        profile: Profil complet de l'utilisateur
        model_utils: Gestionnaire du modèle pour les prédictions
        
    Examples:
        >>> render_sector_comparison(user_profile, model_utils)
        # Affiche un graphique comparatif avec insights
    """
    with st.expander("📊 Comparaison salariale par secteur"):
        st.markdown("#### Impact du secteur sur votre salaire")
        
        # Secteurs clés à comparer
        key_sectors = [
            'Tech', 'Finance', 'Banque', 'Conseil', 
            'ESN', 'Startup', 'E-commerce', 'Industrie'
        ]
        
        # Prédictions pour chaque secteur
        sector_predictions = _calculate_sector_predictions(
            profile,
            key_sectors,
            model_utils
        )
        
        if sector_predictions:
            # Graphique comparatif
            _display_sector_comparison_chart(
                sector_predictions,
                profile['sector_clean']
            )
            
            # Insights et recommandations
            _display_sector_insights(
                sector_predictions,
                profile['sector_clean']
            )


def _calculate_sector_predictions(
    profile: Dict[str, Any],
    sectors: List[str],
    model_utils: Any
) -> Dict[str, float]:
    """
    Calcule les prédictions pour différents secteurs.
    
    Args:
        profile: Profil de base
        sectors: Liste des secteurs à comparer
        model_utils: Gestionnaire du modèle
        
    Returns:
        Dict {secteur: salaire_prédit}
    """
    sector_predictions = {}
    
    with st.spinner("Calcul des comparaisons par secteur..."):
        for sector in sectors:
            profile_sector = profile.copy()
            profile_sector['sector_clean'] = sector
            
            # Recalculer les features dépendantes du secteur
            high_paying_sectors = DataDistributions.get_high_paying_sectors()
            profile_sector['is_high_paying_sector'] = int(
                sector in high_paying_sectors
            )
            
            # Recalculer la description (ajustement sectoriel)
            profile_sector['description_word_count'] = (
                CalculationUtils.estimate_description_complexity(profile_sector)
            )
            
            # Prédiction
            pred = model_utils.predict(profile_sector)
            if pred:
                sector_predictions[sector] = pred['prediction']
    
    return sector_predictions


def _display_sector_comparison_chart(
    sector_predictions: Dict[str, float],
    current_sector: str
) -> None:
    """
    Affiche le graphique de comparaison par secteur.
    
    Args:
        sector_predictions: Dict des prédictions par secteur
        current_sector: Secteur actuel de l'utilisateur
    """
    # Tri par salaire décroissant
    sectors_sorted = sorted(
        sector_predictions.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    sector_names = [s[0] for s in sectors_sorted]
    sector_values = [s[1] for s in sectors_sorted]
    
    # Coloration : secteur actuel en orange, autres en bleu
    colors = [
        '#ff7f0e' if s == current_sector else '#1f77b4'
        for s in sector_names
    ]
    
    # Création du graphique
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=sector_names,
        x=sector_values,
        orientation='h',
        marker_color=colors,
        text=[f"{v:,.0f}€" for v in sector_values],
        textposition='outside',
        hovertemplate=(
            '<b>%{y}</b><br>' +
            'Salaire estimé: %{x:,.0f}€<br>' +
            '<extra></extra>'
        )
    ))
    
    fig.update_layout(
        title=f"💼 Votre profil dans différents secteurs (Actuel : {current_sector})",
        xaxis_title="Salaire estimé (€)",
        yaxis_title="",
        height=450,
        showlegend=False,
        plot_bgcolor='white',
        yaxis=dict(autorange='reversed')
    )
    
    fig.update_xaxes(gridcolor='lightgray')
    
    st.plotly_chart(fig, use_container_width=True)


def _display_sector_insights(
    sector_predictions: Dict[str, float],
    current_sector: str
) -> None:
    """
    Affiche les insights sur les secteurs.
    
    Args:
        sector_predictions: Dict des prédictions par secteur
        current_sector: Secteur actuel
    """
    # Meilleur secteur
    best_sector = max(sector_predictions.items(), key=lambda x: x[1])
    worst_sector = min(sector_predictions.items(), key=lambda x: x[1])
    
    # Calcul des écarts
    if current_sector in sector_predictions:
        current_salary = sector_predictions[current_sector]
        potential_gain = best_sector[1] - current_salary
        
        if potential_gain > 1000:  # Gain significatif
            st.info(f"""
            💡 **Opportunité détectée** :
            
            En changeant pour le secteur **{best_sector[0]}**, vous pourriez gagner 
            environ **{potential_gain:+,.0f}€** de plus (estimation : **{best_sector[1]:,.0f}€**)
            
            Écart min-max : **{best_sector[1] - worst_sector[1]:,.0f}€** entre 
            {best_sector[0]} et {worst_sector[0]}
            """)
        else:
            st.success(
                "✅ Vous êtes déjà dans l'un des secteurs les mieux rémunérés !"
            )
    else:
        st.info(f"""
        📊 **Analyse sectorielle** :
        
        - **Meilleur secteur** : {best_sector[0]} ({best_sector[1]:,.0f}€)
        - **Secteur le moins rémunérateur** : {worst_sector[0]} ({worst_sector[1]:,.0f}€)
        - **Écart** : {best_sector[1] - worst_sector[1]:,.0f}€
        """)


# ============================================================================
# PROJECTION DE CARRIÈRE
# ============================================================================

def render_experience_projection(
    profile: Dict[str, Any],
    model_utils: Any
) -> None:
    """
    Affiche la projection de carrière selon l'expérience.
    
    Simule l'évolution du salaire avec l'augmentation de l'expérience
    pour donner une vision de la progression de carrière.
    
    Args:
        profile: Profil complet de l'utilisateur
        model_utils: Gestionnaire du modèle
    """
    with st.expander("📈 Évolution salariale selon l'expérience"):
        st.markdown("#### Projection de carrière")
        
        # Niveaux d'expérience à projeter
        exp_levels = [0.5, 1, 2, 3, 5, 7, 10, 12, 15, 20]
        
        # Calcul des prédictions
        exp_predictions = _calculate_experience_predictions(
            profile,
            exp_levels,
            model_utils
        )
        
        if exp_predictions:
            # Graphique de projection
            _display_experience_projection_chart(
                exp_predictions,
                profile['experience_final']
            )
            
            # Métriques et insights
            _display_experience_insights(
                exp_predictions,
                profile['experience_final']
            )


def _calculate_experience_predictions(
    profile: Dict[str, Any],
    exp_levels: List[float],
    model_utils: Any
) -> List[Tuple[float, float]]:
    """
    Calcule les prédictions pour différents niveaux d'expérience.
    
    Args:
        profile: Profil de base
        exp_levels: Liste des années d'expérience à simuler
        model_utils: Gestionnaire du modèle
        
    Returns:
        Liste de tuples (expérience, salaire_prédit)
    """
    exp_predictions = []
    
    with st.spinner("Calcul de la projection de carrière..."):
        for exp in exp_levels:
            profile_exp = profile.copy()
            profile_exp['experience_final'] = float(exp)
            
            # Réajuster le seniority
            profile_exp['seniority'] = _get_seniority_for_experience(exp)
            
            # Recalculer les features dépendantes
            profile_exp['description_word_count'] = (
                CalculationUtils.estimate_description_complexity(profile_exp)
            )
            profile_exp['nb_mots_cles_techniques'] = (
                CalculationUtils.estimate_technical_keywords(profile_exp)
            )
            
            # Prédiction
            pred = model_utils.predict(profile_exp)
            if pred:
                exp_predictions.append((exp, pred['prediction']))
    
    return exp_predictions


def _get_seniority_for_experience(exp: float) -> str:
    """
    Détermine le niveau de séniorité selon l'expérience.
    
    Args:
        exp: Années d'expérience
        
    Returns:
        Niveau de séniorité
    """
    if exp < 1:
        return "Stage/Alternance"
    elif exp <= 3:
        return "Junior (1-3 ans)"
    elif exp <= 5:
        return "Mid-level"
    elif exp <= 8:
        return "Senior (5-8 ans)"
    elif exp <= 12:
        return "Expert (8-12 ans)"
    else:
        return "Lead/Manager (12-20 ans)"


def _display_experience_projection_chart(
    exp_predictions: List[Tuple[float, float]],
    current_exp: float
) -> None:
    """
    Affiche le graphique de projection de carrière.
    
    Args:
        exp_predictions: Liste de (expérience, salaire)
        current_exp: Expérience actuelle de l'utilisateur
    """
    exp_years = [e[0] for e in exp_predictions]
    exp_salaries = [e[1] for e in exp_predictions]
    
    # Trouver le salaire actuel
    current_pred = None
    for exp, sal in exp_predictions:
        if abs(exp - current_exp) < 0.5:
            current_pred = sal
            break
    
    # Si pas trouvé, interpoler
    if current_pred is None:
        current_pred = np.interp(current_exp, exp_years, exp_salaries)
    
    # Création du graphique
    fig = go.Figure()
    
    # Courbe d'évolution
    fig.add_trace(go.Scatter(
        x=exp_years,
        y=exp_salaries,
        mode='lines+markers',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=10, color='#1f77b4'),
        name='Salaire estimé',
        hovertemplate=(
            '<b>Expérience: %{x:.1f} ans</b><br>' +
            'Salaire: %{y:,.0f}€<br>' +
            '<extra></extra>'
        )
    ))
    
    # Marquer la position actuelle
    fig.add_trace(go.Scatter(
        x=[current_exp],
        y=[current_pred],
        mode='markers',
        marker=dict(
            size=20,
            color='red',
            symbol='star',
            line=dict(color='darkred', width=2)
        ),
        name='Vous êtes ici',
        hovertemplate=(
            '<b>Position actuelle</b><br>' +
            'Expérience: %{x:.1f} ans<br>' +
            'Salaire: %{y:,.0f}€<br>' +
            '<extra></extra>'
        )
    ))
    
    fig.update_layout(
        title="📈 Évolution salariale estimée selon l'expérience",
        xaxis_title="Années d'expérience",
        yaxis_title="Salaire annuel brut (€)",
        height=450,
        hovermode='x unified',
        plot_bgcolor='white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(gridcolor='lightgray')
    fig.update_yaxes(gridcolor='lightgray')
    
    st.plotly_chart(fig, use_container_width=True)


def _display_experience_insights(
    exp_predictions: List[Tuple[float, float]],
    current_exp: float
) -> None:
    """
    Affiche les insights sur la projection de carrière.
    
    Args:
        exp_predictions: Liste de (expérience, salaire)
        current_exp: Expérience actuelle
    """
    if len(exp_predictions) < 2:
        return
    
    # Calculs de croissance
    exp_years = [e[0] for e in exp_predictions]
    exp_salaries = [e[1] for e in exp_predictions]
    
    # Croissance totale
    total_growth = exp_predictions[-1][1] - exp_predictions[0][1]
    years_span = exp_predictions[-1][0] - exp_predictions[0][0]
    avg_annual_growth = total_growth / years_span if years_span > 0 else 0
    
    # Salaire actuel et projection 5 ans
    current_salary = np.interp(current_exp, exp_years, exp_salaries)
    future_exp = min(current_exp + 5, exp_years[-1])
    future_salary = np.interp(future_exp, exp_years, exp_salaries)
    five_year_growth = future_salary - current_salary
    
    st.info(f"""
    📊 **Projection de carrière** :
    
    - **Croissance moyenne** : **{avg_annual_growth:+,.0f}€/an** 
      (de {exp_predictions[0][0]:.0f} à {exp_predictions[-1][0]:.0f} ans)
    - **Progression totale estimée** : **+{total_growth:,.0f}€** sur {years_span:.0f} ans
    - **Dans 5 ans** ({future_exp:.0f} ans d'exp.) : **~{future_salary:,.0f}€** 
      (gain estimé : **+{five_year_growth:,.0f}€**)
    
    💡 _Cette projection suppose que votre profil reste stable (compétences, secteur, localisation)_
    """)


# ============================================================================
# COMPARAISON PAR LOCALISATION
# ============================================================================

def render_location_comparison(
    profile: Dict[str, Any],
    model_utils: Any
) -> None:
    """
    Affiche la comparaison salariale par ville.
    
    Args:
        profile: Profil complet de l'utilisateur
        model_utils: Gestionnaire du modèle
    """
    with st.expander("📍 Comparaison salariale par ville"):
        st.markdown("#### Impact de la localisation sur votre salaire")
        
        # Villes clés à comparer
        key_cities = [
            'Paris', 'Lyon', 'Toulouse', 'Bordeaux',
            'Nantes', 'Lille', 'Marseille', 'Rennes'
        ]
        
        # Prédictions par ville
        city_predictions = _calculate_city_predictions(
            profile,
            key_cities,
            model_utils
        )
        
        if city_predictions:
            # Graphique
            _display_city_comparison_chart(
                city_predictions,
                profile['location_final']
            )
            
            # Insights
            _display_city_insights(
                city_predictions,
                profile['location_final']
            )


def _calculate_city_predictions(
    profile: Dict[str, Any],
    cities: List[str],
    model_utils: Any
) -> Dict[str, float]:
    """Calcule les prédictions pour différentes villes."""
    city_predictions = {}
    grandes_villes = DataDistributions.get_grandes_villes()
    
    with st.spinner("Calcul des comparaisons par ville..."):
        for city in cities:
            profile_city = profile.copy()
            profile_city['location_final'] = city
            
            # Recalculer les features dépendantes
            profile_city['is_grande_ville'] = int(city in grandes_villes)
            profile_city['is_paris_region'] = int('Paris' in city)
            
            pred = model_utils.predict(profile_city)
            if pred:
                city_predictions[city] = pred['prediction']
    
    return city_predictions


def _display_city_comparison_chart(
    city_predictions: Dict[str, float],
    current_city: str
) -> None:
    """Affiche le graphique de comparaison par ville."""
    cities_sorted = sorted(
        city_predictions.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    city_names = [c[0] for c in cities_sorted]
    city_values = [c[1] for c in cities_sorted]
    
    colors = [
        '#ff7f0e' if c == current_city else '#1f77b4'
        for c in city_names
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=city_names,
        x=city_values,
        orientation='h',
        marker_color=colors,
        text=[f"{v:,.0f}€" for v in city_values],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f"📍 Votre profil dans différentes villes (Actuel : {current_city})",
        xaxis_title="Salaire estimé (€)",
        height=400,
        showlegend=False,
        yaxis=dict(autorange='reversed')
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _display_city_insights(
    city_predictions: Dict[str, float],
    current_city: str
) -> None:
    """Affiche les insights sur les villes."""
    best_city = max(city_predictions.items(), key=lambda x: x[1])
    
    if current_city in city_predictions:
        current_salary = city_predictions[current_city]
        potential_gain = best_city[1] - current_salary
        
        if potential_gain > 2000:
            st.info(f"""
            💡 **Opportunité géographique** :
            
            En vous installant à **{best_city[0]}**, vous pourriez gagner 
            environ **{potential_gain:+,.0f}€** de plus.
            
            _Note : Pensez au coût de la vie différentiel_
            """)
        else:
            st.success("✅ Vous êtes dans une ville bien rémunératrice !")


# ============================================================================
# IMPACT DES COMPÉTENCES
# ============================================================================

def render_skills_impact_analysis(
    profile: Dict[str, Any],
    model_utils: Any
) -> None:
    """
    Analyse l'impact individuel de chaque compétence.
    
    Args:
        profile: Profil complet
        model_utils: Gestionnaire du modèle
    """
    with st.expander("🛠️ Impact individuel de vos compétences"):
        st.markdown("#### Valeur ajoutée de chaque compétence")
        
        # Liste des compétences à analyser
        skills_to_analyze = {
            'contient_python': '🐍 Python',
            'contient_sql': '🗃️ SQL',
            'contient_machine_learning': '🤖 Machine Learning',
            'contient_deep_learning': '🧠 Deep Learning',
            'contient_aws': '☁️ AWS',
            'contient_spark': '🔥 Spark',
            'contient_tableau': '📊 Tableau'
        }
        
        # Calcul de l'impact de chaque compétence
        skills_impact = _calculate_skills_individual_impact(
            profile,
            skills_to_analyze,
            model_utils
        )
        
        if skills_impact:
            _display_skills_impact_chart(skills_impact)
            _display_skills_recommendations(skills_impact, profile)


def _calculate_skills_individual_impact(
    profile: Dict[str, Any],
    skills: Dict[str, str],
    model_utils: Any
) -> Dict[str, float]:
    """
    Calcule l'impact individuel de chaque compétence.
    
    Args:
        profile: Profil de base
        skills: Dict {skill_key: skill_label}
        model_utils: Gestionnaire du modèle
        
    Returns:
        Dict {skill_label: impact_salary}
    """
    # Profil de base (sans la compétence)
    base_profile = profile.copy()
    base_pred = model_utils.predict(base_profile)
    
    if not base_pred:
        return {}
    
    base_salary = base_pred['prediction']
    skills_impact = {}
    
    with st.spinner("Analyse de l'impact des compétences..."):
        for skill_key, skill_label in skills.items():
            # Profil avec la compétence activée
            skill_profile = base_profile.copy()
            skill_profile[skill_key] = True
            
            # Recalculer les scores
            skill_profile['skills_count'] = (
                CalculationUtils.calculate_skills_count_from_profile(skill_profile)
            )
            skill_profile['technical_score'] = (
                CalculationUtils.calculate_technical_score_from_profile(skill_profile)
            )
            
            # Prédiction
            pred = model_utils.predict(skill_profile)
            if pred:
                impact = pred['prediction'] - base_salary
                skills_impact[skill_label] = impact
    
    return skills_impact


def _display_skills_impact_chart(skills_impact: Dict[str, float]) -> None:
    """Affiche le graphique d'impact des compétences."""
    # Tri par impact décroissant
    sorted_skills = sorted(
        skills_impact.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    skill_labels = [s[0] for s in sorted_skills]
    skill_values = [s[1] for s in sorted_skills]
    
    # Coloration : vert si positif, rouge si négatif
    colors = ['#2ca02c' if v > 0 else '#d62728' for v in skill_values]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=skill_labels,
        x=skill_values,
        orientation='h',
        marker_color=colors,
        text=[f"{v:+,.0f}€" for v in skill_values],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="💰 Impact salarial de chaque compétence",
        xaxis_title="Impact sur le salaire (€)",
        height=400,
        showlegend=False,
        yaxis=dict(autorange='reversed')
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _display_skills_recommendations(
    skills_impact: Dict[str, float],
    profile: Dict[str, Any]
) -> None:
    """Affiche les recommandations basées sur l'impact des compétences."""
    # Trouver les compétences les plus rentables non maîtrisées
    best_skills = sorted(
        skills_impact.items(),
        key=lambda x: x[1],
        reverse=True
    )[:3]
    
    st.info(f"""
    📈 **Top 3 des compétences les plus valorisées** :
    
    1. **{best_skills[0][0]}** : +{best_skills[0][1]:,.0f}€
    2. **{best_skills[1][0]}** : +{best_skills[1][1]:,.0f}€
    3. **{best_skills[2][0]}** : +{best_skills[2][1]:,.0f}€
    
    💡 _Se former à ces compétences peut significativement augmenter votre rémunération_
    """)


# ============================================================================
# EXPORT DES FONCTIONS
# ============================================================================

__all__ = [
    'render_sector_comparison',
    'render_experience_projection',
    'render_location_comparison',
    'render_skills_impact_analysis'
]
