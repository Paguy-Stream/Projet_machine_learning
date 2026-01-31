"""
Module d'analyse de carrière (Scorecard et Diagnostic).

Ce module contient :
- Scorecard de résumé avec 4 KPIs visuels
- Calcul du score d'employabilité
- Diagnostic de positionnement détaillé
- Graphique de positionnement sur le marché
"""
import streamlit as st 
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Tuple, Any, List

from utils.model_utils import CalculationUtils


# ============================================================================
# SCORECARD DE RÉSUMÉ
# ============================================================================

def render_scorecard(
    profile: Dict[str, Any],
    base_salary: float,
    percentile: float,
    df_final: pd.DataFrame,
    model_utils: Any
) -> None:
    """
    Affiche la scorecard de résumé avec 4 KPIs visuels.
    
    Args:
        profile: Profil complet de l'utilisateur
        base_salary: Salaire estimé actuel
        percentile: Percentile sur le marché
        df_final: DataFrame complet du marché
        model_utils: Gestionnaire du modèle
        
    Notes:
        4 KPIs affichés :
        1. Salaire actuel (gradient violet)
        2. Percentile marché (coloré selon niveau)
        3. Employabilité (match avec offres)
        4. Potentiel boost (meilleure compétence)
    """
    st.markdown("## 📊 Votre carte d'identité Data")
    
    # Calcul des métriques
    resilience_score = _calculate_employability_score(profile, df_final)
    best_gain = _calculate_best_skill_gain(profile, base_salary, model_utils)
    
    # Affichage des 4 KPIs
    _render_kpi_cards(
        base_salary,
        profile['seniority'],
        percentile,
        resilience_score,
        best_gain
    )
    
    st.markdown("---")


def _calculate_employability_score(
    profile: Dict[str, Any],
    df_final: pd.DataFrame
) -> float:
    """
    Calcule le score d'employabilité (match avec offres du marché).
    
    Args:
        profile: Profil de l'utilisateur
        df_final: DataFrame complet
        
    Returns:
        Score d'employabilité (0-1)
        
    Notes:
        Calcule le % d'offres où l'utilisateur a ≥70% des compétences requises.
    """
    # Identifier les colonnes de compétences disponibles
    skills_cols = [
        'contient_python', 'contient_sql', 'contient_r',
        'contient_tableau', 'contient_power_bi',
        'contient_aws', 'contient_azure', 'contient_spark',
        'contient_machine_learning', 'contient_deep_learning',
        'contient_etl'
    ]
    
    # Filtrer les colonnes existantes
    available_skills = [col for col in skills_cols if col in df_final.columns]
    
    if not available_skills:
        return 0.5  # Valeur par défaut
    
    # Vecteur des compétences de l'utilisateur
    user_skills_vector = [
        1 if profile.get(k, False) else 0
        for k in available_skills
    ]
    
    # Fonction de match avec chaque offre
    def calculate_match(row):
        offer_vector = row[available_skills].values
        intersection = np.logical_and(user_skills_vector, offer_vector).sum()
        union = offer_vector.sum()
        return intersection / union if union > 0 else 1.0
    
    # Calculer le match pour toutes les offres
    df_final['match_score'] = df_final.apply(calculate_match, axis=1)
    
    # Score = % d'offres avec match ≥ 70%
    resilience_score = (df_final['match_score'] >= 0.7).mean()
    
    return resilience_score


def _calculate_best_skill_gain(
    profile: Dict[str, Any],
    base_salary: float,
    model_utils: Any
) -> float:
    """
    Calcule le gain potentiel de la meilleure compétence manquante.
    
    Args:
        profile: Profil actuel
        base_salary: Salaire de base
        model_utils: Gestionnaire du modèle
        
    Returns:
        Gain maximum possible (€)
    """
    # Liste de toutes les compétences
    all_skills = [
        ('Python', 'contient_python'),
        ('SQL', 'contient_sql'),
        ('R', 'contient_r'),
        ('Tableau', 'contient_tableau'),
        ('Power BI', 'contient_power_bi'),
        ('AWS', 'contient_aws'),
        ('Azure', 'contient_azure'),
        ('Spark', 'contient_spark'),
        ('Machine Learning', 'contient_machine_learning'),
        ('Deep Learning', 'contient_deep_learning'),
        ('ETL', 'contient_etl')
    ]
    
    # Filtrer les compétences manquantes
    missing_skills = [
        (name, key) for name, key in all_skills
        if not profile.get(key, False)
    ]
    
    if not missing_skills:
        return 0.0
    
    best_gain = 0.0
    
    # Tester les 3 premières compétences manquantes
    for name, key in missing_skills[:3]:
        scenario = profile.copy()
        scenario[key] = True
        
        # Recalculer les scores
        scenario['skills_count'] = CalculationUtils.calculate_skills_count_from_profile(
            {k: scenario.get(k, False) for _, k in all_skills}
        )
        scenario['technical_score'] = CalculationUtils.calculate_technical_score_from_profile(
            {k: scenario.get(k, False) for _, k in all_skills}
        )
        
        # Prédiction
        pred = model_utils.predict(scenario)
        if pred:
            gain = pred['prediction'] - base_salary
            if gain > best_gain:
                best_gain = gain
    
    return best_gain


def _render_kpi_cards(
    base_salary: float,
    seniority: str,
    percentile: float,
    resilience_score: float,
    best_gain: float
) -> None:
    """
    Affiche les 4 cards KPI avec style.
    
    Args:
        base_salary: Salaire actuel
        seniority: Niveau de séniorité
        percentile: Percentile sur le marché
        resilience_score: Score d'employabilité
        best_gain: Potentiel de gain max
    """
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    # KPI 1 : Salaire actuel
    with kpi1:
        st.markdown(f"""
        <div style='text-align: center; padding: 20px; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <p style='color: white; font-size: 14px; margin: 0;'>Salaire Actuel</p>
            <h2 style='color: white; margin: 10px 0; font-size: 32px;'>
                {base_salary:,.0f}€
            </h2>
            <p style='color: rgba(255,255,255,0.8); font-size: 12px; margin: 0;'>
                {seniority}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # KPI 2 : Percentile
    with kpi2:
        percentile_color = "#2ecc71" if percentile >= 50 else "#e74c3c"
        status_text = (
            "🌟 Top profil" if percentile >= 75 
            else "✅ Au-dessus médiane" if percentile >= 50 
            else "💡 À optimiser"
        )
        
        st.markdown(f"""
        <div style='text-align: center; padding: 20px; background: {percentile_color}; 
                    border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <p style='color: white; font-size: 14px; margin: 0;'>Percentile Marché</p>
            <h2 style='color: white; margin: 10px 0; font-size: 32px;'>
                {percentile:.0f}%
            </h2>
            <p style='color: rgba(255,255,255,0.8); font-size: 12px; margin: 0;'>
                {status_text}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # KPI 3 : Employabilité
    with kpi3:
        employability_color = "#3498db" if resilience_score >= 0.5 else "#e67e22"
        
        st.markdown(f"""
        <div style='text-align: center; padding: 20px; background: {employability_color}; 
                    border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <p style='color: white; font-size: 14px; margin: 0;'>Employabilité</p>
            <h2 style='color: white; margin: 10px 0; font-size: 32px;'>
                {resilience_score:.0%}
            </h2>
            <p style='color: rgba(255,255,255,0.8); font-size: 12px; margin: 0;'>
                Match > 70%
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # KPI 4 : Potentiel boost
    with kpi4:
        st.markdown(f"""
        <div style='text-align: center; padding: 20px; 
                    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <p style='color: white; font-size: 14px; margin: 0;'>Potentiel Boost</p>
            <h2 style='color: white; margin: 10px 0; font-size: 32px;'>
                +{best_gain:,.0f}€
            </h2>
            <p style='color: rgba(255,255,255,0.8); font-size: 12px; margin: 0;'>
                {"Via formation" if best_gain > 0 else "Optimisé"}
            </p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# DIAGNOSTIC DE POSITIONNEMENT
# ============================================================================

def render_positioning_diagnosis(
    base_salary: float,
    percentile: float,
    market_median: float,
    real_market_data: np.ndarray
) -> None:
    """
    Affiche le diagnostic de positionnement détaillé.
    
    Args:
        base_salary: Salaire estimé
        percentile: Percentile sur le marché
        market_median: Médiane du marché
        real_market_data: Distribution complète des salaires
    """
    st.markdown("## 🔍 Diagnostic de positionnement")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        _render_positioning_chart(
            base_salary,
            market_median,
            real_market_data
        )
    
    with col2:
        _render_diagnosis_interpretation(
            base_salary,
            percentile,
            market_median
        )


def _render_positioning_chart(
    base_salary: float,
    market_median: float,
    real_market_data: np.ndarray
) -> None:
    """
    Affiche le graphique de positionnement.
    
    Args:
        base_salary: Salaire de l'utilisateur
        market_median: Médiane du marché
        real_market_data: Distribution des salaires
    """
    fig = go.Figure()
    
    # Histogramme de la distribution du marché
    fig.add_trace(go.Histogram(
        x=real_market_data,
        name='Distribution marché',
        nbinsx=50,
        marker_color='rgba(31, 119, 180, 0.3)',
        showlegend=True
    ))
    
    # Position de l'utilisateur
    fig.add_vline(
        x=base_salary,
        line_dash="dash",
        line_color="red",
        line_width=3,
        annotation_text=f"Vous : {base_salary:,.0f}€",
        annotation_position="top"
    )
    
    # Médiane du marché
    fig.add_vline(
        x=market_median,
        line_dash="dot",
        line_color="green",
        line_width=2,
        annotation_text=f"Médiane : {market_median:,.0f}€",
        annotation_position="bottom"
    )
    
    fig.update_layout(
        title="Votre positionnement sur le marché",
        xaxis_title="Salaire annuel (€)",
        yaxis_title="Nombre d'offres",
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_diagnosis_interpretation(
    base_salary: float,
    percentile: float,
    market_median: float
) -> None:
    """
    Affiche l'interprétation du diagnostic.
    
    Args:
        base_salary: Salaire de l'utilisateur
        percentile: Percentile
        market_median: Médiane du marché
    """
    st.markdown("### 📈 Analyse")
    
    diff_median = base_salary - market_median
    
    if percentile >= 90:
        st.success(f"""
        🌟 **Excellent positionnement !**
        
        Vous êtes dans le **top 10%** du marché.
        
        - Écart vs médiane : **{diff_median:+,.0f}€**
        - Percentile : **{percentile:.0f}%**
        
        💡 **Focus** : Maximiser votre valeur via spécialisation 
        ou leadership technique.
        """)
    
    elif percentile >= 75:
        st.success(f"""
        ✅ **Très bon positionnement**
        
        Vous êtes dans le **top 25%**.
        
        - Écart vs médiane : **{diff_median:+,.0f}€**
        - Percentile : **{percentile:.0f}%**
        
        💡 **Opportunité** : Viser le top 10% via compétences 
        premium (ML, Cloud, DL).
        """)
    
    elif percentile >= 50:
        st.info(f"""
        👍 **Au-dessus de la médiane**
        
        Position solide sur le marché.
        
        - Écart vs médiane : **{diff_median:+,.0f}€**
        - Percentile : **{percentile:.0f}%**
        
        💡 **Marge de progression** : +{market_median * 0.25:,.0f}€ 
        atteignables via formation ciblée.
        """)
    
    elif percentile >= 25:
        st.warning(f"""
        💡 **Potentiel d'amélioration**
        
        Sous la médiane du marché.
        
        - Écart vs médiane : **{diff_median:+,.0f}€**
        - Percentile : **{percentile:.0f}%**
        
        💡 **Action** : Suivre la roadmap pour rattraper 
        rapidement le marché.
        """)
    
    else:
        st.error(f"""
        ⚠️ **En dessous du marché**
        
        Opportunité de forte progression.
        
        - Écart vs médiane : **{diff_median:+,.0f}€**
        - Percentile : **{percentile:.0f}%**
        
        💡 **Priorité** : Formation intensive + optimisation CV.
        Gain potentiel : +{market_median - base_salary:,.0f}€ 
        pour atteindre la médiane.
        """)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'render_scorecard',
    'render_positioning_diagnosis'
]
