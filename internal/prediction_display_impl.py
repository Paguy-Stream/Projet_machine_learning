"""
Module d'affichage des résultats de prédiction.

Ce module contient toutes les fonctions pour afficher :
- Les résultats de prédiction principaux
- Les visualisations SHAP
- Les analyses comparatives
- Les projections de carrière
- Les actions utilisateur

"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional, Any

from utils.config import Config
from utils.model_utils import ChartUtils, CalculationUtils, DataDistributions
from internal.prediction_comparisons_impl import (
    render_sector_comparison,
    render_experience_projection,
    render_location_comparison,
    render_skills_impact_analysis
)


# ============================================================================
# AFFICHAGE DES RÉSULTATS PRINCIPAUX
# ============================================================================

def render_main_prediction_result(
    result: Dict[str, float],
    profile: Dict[str, Any]
) -> None:
    """
    Affiche le résultat principal de la prédiction.
    
    Args:
        result: Dict contenant prediction, bounds, errors
        profile: Profil complet de l'utilisateur
    """
    st.markdown("## 💰 Votre estimation salariale")
    
    prediction = result['prediction']
    lower_bound = result['lower_bound']
    upper_bound = result['upper_bound']
    mae_error = result['mae_error']
    std_error = result.get('std_error', 183)
    
    # Affichage principal dans une card stylisée
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"""
        <div style='
            text-align: center;
            padding: 30px;
            background: linear-gradient(135deg, #1f77b4 0%, #0d5a9e 100%);
            border-radius: 15px;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        '>
            <p style='color: white; font-size: 18px; margin-bottom: 10px;'>
                Salaire annuel brut estimé
            </p>
            <h1 style='color: white; font-size: 60px; margin: 0; font-weight: bold;'>
                {prediction:,.0f} €
            </h1>
            <p style='color: rgba(255,255,255,0.9); font-size: 16px; margin-top: 10px;'>
                Erreur moyenne : ±{mae_error:,.0f}€ | Confiance : ±{std_error:,.0f}€
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Fourchette réaliste
    st.info(f"""
    📊 **Fourchette réaliste** (P75 des erreurs absolues) :  
    **{lower_bound:,.0f}€** ← **{prediction:,.0f}€** → **{upper_bound:,.0f}€**
    """)
    
    # Indicateur de calculs dynamiques
    _display_dynamic_calculation_info(profile)
    
    st.markdown("---")


def _display_dynamic_calculation_info(profile: Dict[str, Any]) -> None:
    """Affiche les informations sur les calculs dynamiques."""
    desc_stats = DataDistributions.get_desc_words()
    tech_stats = DataDistributions.get_tech_keywords()
    
    st.caption(f"""
    ℹ️ **Paramètres estimés automatiquement** : 
    Description ({profile['description_word_count']} mots, 
    P25-P90: {desc_stats['p25']}-{desc_stats['p90']}) • 
    Mots-clés techniques ({profile['nb_mots_cles_techniques']}, 
    médiane: {tech_stats['median']})
    """)


# ============================================================================
# POSITIONNEMENT SUR LE MARCHÉ
# ============================================================================

def render_market_positioning(
    prediction: float,
    real_market_data: np.ndarray,
    market_stats: Dict[str, float]
) -> None:
    """
    Affiche le positionnement de l'utilisateur sur le marché.
    
    Args:
        prediction: Salaire prédit
        real_market_data: Distribution des salaires du marché
        market_stats: Statistiques du marché
    """
    st.markdown("### 📊 Votre positionnement sur le marché")
    
    col1, col2 = st.columns([2, 1])
    
    # Jauge de positionnement
    with col1:
        gauge_fig = ChartUtils.create_salary_gauge(
            prediction,
            market_stats['median'],
            market_stats['q1'],
            market_stats['q3'],
            market_stats['gauge_min'],
            market_stats['gauge_max']
        )
        st.plotly_chart(gauge_fig, use_container_width=True)
    
    # Métriques et interprétation
    with col2:
        percentile = CalculationUtils.get_percentile_real(
            prediction,
            real_market_data
        )
        
        st.metric(
            "Percentile",
            f"{percentile:.0f}%",
            help="Votre position par rapport aux autres offres"
        )
        
        diff_median = prediction - market_stats['median']
        st.metric(
            f"vs Médiane ({market_stats['median']:,.0f}€)",
            f"{diff_median:+,.0f}€",
            delta_color="normal" if diff_median >= 0 else "inverse"
        )
        
        # Interprétation qualitative
        if percentile >= 75:
            st.success("🌟 Excellent positionnement !")
        elif percentile >= 50:
            st.info("✅ Au-dessus de la moyenne")
        else:
            st.warning("⚠️ En dessous de la médiane")
    
    st.markdown("---")


# ============================================================================
# DISTRIBUTION DU MARCHÉ
# ============================================================================

def render_market_distribution(
    prediction: float,
    real_market_data: np.ndarray,
    market_stats: Dict[str, float]
) -> None:
    """
    Affiche la distribution salariale du marché.
    
    Args:
        prediction: Salaire prédit
        real_market_data: Distribution des salaires
        market_stats: Statistiques du marché
    """
    st.markdown("### 📈 Distribution salariale du marché")
    
    comparison_fig = ChartUtils.create_market_comparison(
        prediction,
        real_market_data,
        market_stats['median']
    )
    st.plotly_chart(comparison_fig, use_container_width=True)
    
    # Statistiques détaillées
    col_stats1, col_stats2, col_stats3 = st.columns(3)
    
    with col_stats1:
        st.metric("Q1 (25%)", f"{market_stats['q1']:,.0f}€")
    
    with col_stats2:
        st.metric("Médiane (50%)", f"{market_stats['median']:,.0f}€")
    
    with col_stats3:
        st.metric("Q3 (75%)", f"{market_stats['q3']:,.0f}€")
    
    st.markdown("---")


# ============================================================================
# EXPLICATIONS SHAP
# ============================================================================

def render_shap_explanations(
    shap_exp: Optional[Dict],
    prediction: float
) -> None:
    """
    Affiche les explications SHAP avec visualisations.
    
    Args:
        shap_exp: Dictionnaire d'explication SHAP
        prediction: Salaire prédit
    """
    if not shap_exp:
        st.warning("⚠️ Explications SHAP non disponibles")
        return
    
    st.markdown("### 🔍 Pourquoi cette estimation ? (Analyse SHAP)")
    
    # Traduction des features
    feature_labels = _get_feature_translation_dict()
    
    base_val = shap_exp['base_value']
    total_pred = shap_exp['prediction']
    
    st.info(f"""
    **🎯 Base du modèle** : {base_val:,.0f}€  
    _Salaire moyen prédit sans tenir compte de vos caractéristiques spécifiques_
    """)
    
    # Graphique waterfall
    waterfall_fig = ChartUtils.create_shap_waterfall(
        shap_exp,
        feature_translation=feature_labels,
        max_display=12
    )
    
    if waterfall_fig:
        st.plotly_chart(waterfall_fig, use_container_width=True)
    
    # Analyse flash des impacts
    _render_impact_flash_analysis(shap_exp, feature_labels)
    
    st.markdown("---")
    
    # Suggestion d'amélioration
    _render_salary_boost_suggestion(total_pred)
    
    # Top facteurs textuels
    _render_top_factors(shap_exp, feature_labels, total_pred)
    
    # Graphique détaillé (dans expander)
    with st.expander("📊 Voir le graphique d'importance détaillé"):
        importance_fig = ChartUtils.create_feature_importance_bar(
            shap_exp,
            top_n=15
        )
        if importance_fig:
            st.plotly_chart(importance_fig, use_container_width=True)
    
    st.markdown("---")


def _get_feature_translation_dict() -> Dict[str, str]:
    """Retourne le dictionnaire de traduction des features."""
    return {
        'location_final': '📍 Localisation',
        'sector_clean': '🏦 Secteur d\'activité',
        'experience_final': '🎓 Expérience',
        'education_clean': '📜 Niveau d\'études',
        'contient_machine_learning': '🤖 Machine Learning',
        'contient_deep_learning': '🧠 Deep Learning',
        'is_paris_region': '🗼 Région Parisienne',
        'technical_score': '⚡ Score Technique',
        'skills_count': '🧰 Nombre de compétences',
        'description_word_count': '📝 Détails de l\'annonce',
        'nb_mots_cles_techniques': '🔧 Mots-clés Tech',
        'telework_numeric': '🏠 Télétravail',
        'is_high_paying_sector': '💰 Secteur Premium',
        'contient_python': '🐍 Python',
        'contient_sql': '🗃️ SQL',
        'contient_aws': '☁️ AWS',
        'contient_azure': '☁️ Azure',
        'contient_gcp': '☁️ GCP',
        'contient_spark': '🔥 Spark',
        'job_type_with_desc': '💼 Type de poste',
        'seniority': '📈 Niveau hiérarchique',
        'has_modern_stack': '🧰 Stack moderne'
    }


def _render_impact_flash_analysis(
    shap_exp: Dict,
    feature_labels: Dict[str, str]
) -> None:
    """Affiche l'analyse flash des principaux impacts."""
    names = shap_exp['feature_names']
    values = shap_exp['shap_values']
    
    # Création des impacts traduits
    impacts = [
        (feature_labels.get(n, n), v)
        for n, v in zip(names, values)
    ]
    
    # Tri par impact
    impacts.sort(key=lambda x: x[1], reverse=True)
    
    boosters = [i for i in impacts if i[1] > 0][:2]
    freins = [i for i in impacts if i[1] < 0]
    principal_frein = freins[-1] if freins else None
    
    st.markdown("---")
    st.markdown("#### 💡 Analyse flash de votre profil")
    
    col_b, col_f = st.columns(2)
    
    with col_b:
        if len(boosters) >= 2:
            st.success(f"""
            **🚀 Vos principaux leviers :**
            1. **{boosters[0][0]}** (+{boosters[0][1]:,.0f}€)
            2. **{boosters[1][0]}** (+{boosters[1][1]:,.0f}€)
            """)
    
    with col_f:
        if principal_frein:
            st.warning(f"""
            **⚖️ Point de vigilance :**
            * **{principal_frein[0]}** ({principal_frein[1]:,.0f}€)  
            _C'est le facteur qui limite actuellement le plus votre estimation._
            """)


def _render_salary_boost_suggestion(total_pred: float) -> None:
    """Suggère une compétence à ajouter pour booster le salaire."""
    st.markdown("### 🎯 Comment booster votre salaire ?")
    
    competences_cibles = {
        'contient_machine_learning': 'Machine Learning',
        'contient_deep_learning': 'Deep Learning',
        'contient_aws': 'Cloud AWS',
        'contient_spark': 'Big Data (Spark)',
        'contient_gcp': 'Google Cloud (GCP)',
        'contient_azure': 'Azure',
        'contient_sql': 'SQL (Expert)'
    }
    
    current_profile = st.session_state.current_profile
    manquantes = [
        feat for feat in competences_cibles
        if not current_profile.get(feat, False)
    ]
    
    if manquantes:
        target_feat = manquantes[0]
        target_label = competences_cibles[target_feat]
        
        # Simulation avec la compétence ajoutée
        boosted_profile = current_profile.copy()
        boosted_profile[target_feat] = True
        boosted_profile['skills_count'] = (
            CalculationUtils.calculate_skills_count_from_profile(boosted_profile)
        )
        boosted_profile['technical_score'] = (
            CalculationUtils.calculate_technical_score_from_profile(boosted_profile)
        )
        
        model_utils = st.session_state.model_utils
        new_pred_data = model_utils.predict(boosted_profile)
        
        if new_pred_data:
            gain_potentiel = new_pred_data['prediction'] - total_pred
            
            st.write(
                f"Si vous ajoutez la compétence **{target_label}** à votre profil :"
            )
            
            col_metric, col_text = st.columns([1, 2])
            
            with col_metric:
                st.metric(
                    "Gain estimé",
                    f"+{gain_potentiel:,.0f}€",
                    delta_color="normal"
                )
            
            with col_text:
                st.info(
                    f"Votre estimation passerait de **{total_pred:,.0f}€** "
                    f"à **{new_pred_data['prediction']:,.0f}€**."
                )
    else:
        st.balloons()
        st.success(
            "Félicitations ! Votre stack technique est déjà optimale selon nos critères."
        )


def _render_top_factors(
    shap_exp: Dict,
    feature_translation: Dict[str, str],
    total_pred: float
) -> None:
    """Affiche les principaux facteurs d'influence."""
    st.markdown("#### 📊 Principaux facteurs d'influence")
    
    # Filtrer et trier les contributions
    filtered_contributions = []
    
    for feat, val in zip(shap_exp['feature_names'], shap_exp['shap_values']):
        if abs(val) < 100:  # Seuil de significativité
            continue
        
        readable = feature_translation.get(
            feat,
            feat.replace('_', ' ').title()
        )
        filtered_contributions.append((readable, val))
    
    filtered_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    top_contributions = filtered_contributions[:10]
    
    # Affichage en deux colonnes
    col_left, col_right = st.columns(2)
    
    for i, (readable, val) in enumerate(top_contributions):
        target_col = col_left if i % 2 == 0 else col_right
        
        with target_col:
            color = "🟢" if val > 0 else "🔴"
            sign = "+" if val >= 0 else ""
            st.markdown(f"{color} **{readable}** : {sign}{val:,.0f}€")
    
    st.success(f"**💰 Total estimé** : {total_pred:,.0f}€")


# ============================================================================
# ANALYSES COMPARATIVES
# ============================================================================

def render_ml_dl_comparison(profile: Dict[str, Any]) -> None:
    """
    Affiche la comparaison ML vs DL si applicable.
    
    Args:
        profile: Profil complet de l'utilisateur
    """
    if not (profile.get('contient_machine_learning') or 
            profile.get('contient_deep_learning')):
        return
    
    st.markdown("### 🤖 Analyse comparative : Machine Learning vs Deep Learning")
    
    model_utils = st.session_state.model_utils
    
    # Création des profils hypothétiques
    profiles = _create_ml_dl_comparison_profiles(profile)
    
    # Prédictions
    with st.spinner("Calcul des comparaisons ML/DL..."):
        predictions = {
            name: model_utils.predict(prof)
            for name, prof in profiles.items()
        }
    
    if all(predictions.values()):
        _display_ml_dl_comparison_chart(predictions)
        _display_ml_dl_metrics(predictions)
        _display_ml_dl_insights(predictions)
    
    st.markdown("---")


def _create_ml_dl_comparison_profiles(
    profile: Dict[str, Any]
) -> Dict[str, Dict]:
    """Crée les 4 profils pour la comparaison ML/DL."""
    profiles = {}
    
    # Sans ML/DL
    profile_none = profile.copy()
    profile_none['contient_machine_learning'] = False
    profile_none['contient_deep_learning'] = False
    profiles['none'] = profile_none
    
    # ML uniquement
    profile_ml = profile.copy()
    profile_ml['contient_machine_learning'] = True
    profile_ml['contient_deep_learning'] = False
    profiles['ml'] = profile_ml
    
    # DL uniquement
    profile_dl = profile.copy()
    profile_dl['contient_machine_learning'] = False
    profile_dl['contient_deep_learning'] = True
    profiles['dl'] = profile_dl
    
    # Les deux
    profile_both = profile.copy()
    profile_both['contient_machine_learning'] = True
    profile_both['contient_deep_learning'] = True
    profiles['both'] = profile_both
    
    # Recalculer les scores pour chaque profil
    for prof in profiles.values():
        prof['technical_score'] = (
            CalculationUtils.calculate_technical_score_from_profile(prof)
        )
        prof['skills_count'] = (
            CalculationUtils.calculate_skills_count_from_profile(prof)
        )
    
    return profiles


def _display_ml_dl_comparison_chart(predictions: Dict[str, Dict]) -> None:
    """Affiche le graphique de comparaison ML/DL."""
    categories = ['Sans ML/DL', 'ML uniquement', 'DL uniquement', 'ML + DL']
    values = [
        predictions['none']['prediction'],
        predictions['ml']['prediction'],
        predictions['dl']['prediction'],
        predictions['both']['prediction']
    ]
    colors = ['#cccccc', '#ff7f0e', '#1f77b4', '#2ca02c']
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=[f"{v:,.0f}€" for v in values],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="💰 Impact salarial : ML vs DL vs Combinaison",
        yaxis_title="Salaire estimé (€)",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _display_ml_dl_metrics(predictions: Dict[str, Dict]) -> None:
    """Affiche les métriques de comparaison ML/DL."""
    col1, col2, col3, col4 = st.columns(4)
    
    base_pred = predictions['none']['prediction']
    
    with col1:
        st.metric("Sans ML/DL", f"{base_pred:,.0f}€")
    
    with col2:
        delta_ml = predictions['ml']['prediction'] - base_pred
        st.metric(
            "ML uniquement",
            f"{predictions['ml']['prediction']:,.0f}€",
            f"+{delta_ml:,.0f}€"
        )
    
    with col3:
        delta_dl = predictions['dl']['prediction'] - base_pred
        st.metric(
            "DL uniquement",
            f"{predictions['dl']['prediction']:,.0f}€",
            f"+{delta_dl:,.0f}€"
        )
    
    with col4:
        delta_both = predictions['both']['prediction'] - base_pred
        st.metric(
            "ML + DL",
            f"{predictions['both']['prediction']:,.0f}€",
            f"+{delta_both:,.0f}€"
        )


def _display_ml_dl_insights(predictions: Dict[str, Dict]) -> None:
    """Affiche les insights ML/DL."""
    base_pred = predictions['none']['prediction']
    delta_ml = predictions['ml']['prediction'] - base_pred
    delta_dl = predictions['dl']['prediction'] - base_pred
    delta_both = predictions['both']['prediction'] - base_pred
    
    ml_dl_correlation = DataDistributions.get_ml_dl_correlation()
    total_offers = DataDistributions.get_total_offers()
    
    st.info(f"""
    📈 **Insights** :
    - Maîtriser **ML seul** ajoute environ **{delta_ml:+,.0f}€** au salaire de base
    - Maîtriser **DL seul** ajoute environ **{delta_dl:+,.0f}€** au salaire de base
    - Maîtriser **les deux** ajoute environ **{delta_both:+,.0f}€** au salaire de base
    ---
    💡 **Avis** : Le cumul des deux compétences montre un effet de **"rendement décroissant"**. 
    Le marché valorise la spécialisation, mais considère qu'un expert en Deep Learning 
    possède déjà les fondamentaux du Machine Learning.
    
    - _Note : Corrélation ML-DL = {ml_dl_correlation:.2%} 
    (modérément liées, calculée depuis {total_offers:,} offres)_
    """)


# ============================================================================
# PAGE D'ACCUEIL (PAS DE RÉSULTATS)
# ============================================================================

def render_welcome_page() -> None:
    """Affiche la page d'accueil quand aucune prédiction n'a été faite."""
    st.markdown(f"""
    <div style='text-align: center; padding: 60px 20px;'>
        <div style='font-size: 80px; margin-bottom: 20px;'>🔮</div>
        <h2 style='color: #1f77b4;'>Obtenez une estimation de votre salaire</h2>
        <p style='font-size: 18px; color: #666; margin: 20px 0;'>
            Basée sur l'analyse de <strong>{DataDistributions.get_total_offers():,} offres HelloWork</strong> 
            avec calculs dynamiques
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Statistiques du modèle
    _render_model_stats()
    
    st.markdown("---")
    
    
    # Informations techniques
    _render_technical_info()


def _render_model_stats() -> None:
    """Affiche les statistiques du modèle."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background: #f0f2f6; border-radius: 10px;'>
            <div style='font-size: 32px; font-weight: bold; color: #1f77b4;'>2 681</div>
            <div style='color: #666;'>postes Data modélisés</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background: #f0f2f6; border-radius: 10px;'>
            <div style='font-size: 32px; font-weight: bold; color: #1f77b4;'>29</div>
            <div style='color: #666;'>features extraites</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background: #f0f2f6; border-radius: 10px;'>
            <div style='font-size: 32px; font-weight: bold; color: #1f77b4;'>5 163€</div>
            <div style='color: #666;'>erreur moyenne (MAE)</div>
        </div>
        """, unsafe_allow_html=True)


def _render_technical_info() -> None:
    """Affiche les informations techniques."""
    with st.expander("🔧 Informations techniques"):
        desc_stats = DataDistributions.get_desc_words()
        tech_stats = DataDistributions.get_tech_keywords()
        
        st.markdown(f"""
        **📊 Statistiques du dataset (mises à jour automatiquement)** :
        
        - **Nombre total d'offres** : {DataDistributions.get_total_offers():,}
        - **Description word count** : P25={desc_stats['p25']}, 
          Médiane={desc_stats['median']}, P75={desc_stats['p75']}, 
          P90={desc_stats['p90']} (n={desc_stats['count']:,})
        - **Mots-clés techniques** : P25={tech_stats['p25']}, 
          Médiane={tech_stats['median']}, P75={tech_stats['p75']}, 
          P90={tech_stats['p90']} (n={tech_stats['count']:,})
        - **Corrélation ML/DL** : {DataDistributions.get_ml_dl_correlation():.2%}
        
        **🤖 Modèle XGBoost v7** :
        - Architecture : Pipeline (FeatureEngineering → Preprocessing → XGBRegressor)
        - Features : 29 variables dont engineered features
        - Validation : Cross-validation 5-fold
        - Explainability : SHAP TreeExplainer
        
        **♻️ Mise à jour des données** :
        - Les statistiques sont recalculées à chaque démarrage
        - Option de rechargement manuel disponible dans la sidebar
        - Fallback automatique sur valeurs par défaut si erreur
        """)


# ============================================================================
# FONCTION PRINCIPALE D'AFFICHAGE DES RÉSULTATS
# ============================================================================

def render_results(
    model_utils: Any,
    real_market_data: np.ndarray,
    market_stats: Dict[str, float]
) -> None:
    """
    Affiche tous les résultats de la prédiction.
    
    Args:
        model_utils: Gestionnaire du modèle
        real_market_data: Données du marché
        market_stats: Statistiques du marché
    """
    result = st.session_state.last_prediction
    profile = st.session_state.current_profile
    shap_exp = st.session_state.get('shap_explanation')
    
    prediction = result['prediction']
    
    # 1. Résultat principal
    render_main_prediction_result(result, profile)
    
    # 2. Positionnement sur le marché
    render_market_positioning(prediction, real_market_data, market_stats)
    
    # 3. Distribution du marché
    render_market_distribution(prediction, real_market_data, market_stats)
    
    # 4. Explications SHAP
    render_shap_explanations(shap_exp, prediction)
    
    # 5. Comparaison ML/DL
    render_ml_dl_comparison(profile)
    
    # 6. Analyses comparatives avancées
    render_sector_comparison(profile, model_utils)
    render_experience_projection(profile, model_utils)
    render_location_comparison(profile, model_utils)
    render_skills_impact_analysis(profile, model_utils)
    
    # 7. Autres analyses (à implémenter dans prediction_actions.py)
    # render_warnings_and_debug(profile)
    
    # 8. Actions finales
    # render_actions_section(result, profile, shap_exp)
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px 0;'>
        <p>© 2026 Prédicteur de salaires Data Jobs v2.0 • 
        Données : HelloWork (janvier 2026) • Modèle : XGBoost v7</p>
        <p style='font-size: 12px;'>
        Avec calculs dynamiques, visualisations SHAP avancées et analyses comparatives
        </p>
    </div>
    """, unsafe_allow_html=True)
