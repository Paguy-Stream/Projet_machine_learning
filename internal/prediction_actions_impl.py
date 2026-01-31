"""
Module d'actions finales et utilitaires pour les prédictions.

Ce module contient :
- Warnings et avertissements contextuels
- Mode debug et inspection technique
- Actions finales (nouvelle estimation, navigation, export)
- Export des résultats (JSON, PDF)
- Affichage des performances du modèle

"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import json
from typing import Dict, List, Optional, Any

from utils.config import Config
from utils.model_utils import CalculationUtils, DataDistributions


# ============================================================================
# WARNINGS ET AVERTISSEMENTS
# ============================================================================

def render_contextual_warnings(profile: Dict[str, Any]) -> None:
    """
    Affiche des avertissements contextuels selon le profil.
    
    Analyse le profil pour identifier les situations particulières
    et afficher des warnings pertinents :
    - Type de contrat inhabituel (non-CDI)
    - Combinaison ML + DL
    - Secteur avec peu de données
    - Expérience extrême (très junior ou très senior)
    
    Args:
        profile: Profil complet de l'utilisateur
        
    Examples:
        >>> render_contextual_warnings(user_profile)
        # Affiche les warnings applicables
    """
    warnings_displayed = 0
    
    # Warning 1 : Type de contrat
    if profile.get('contract_type_clean') != 'CDI':
        st.warning(f"""
        ⚠️ **À propos du type de contrat ({profile['contract_type_clean']})** :
        
        Votre choix a **peu d'impact** sur la prédiction car **97% des offres 
        dans le dataset sont en CDI**. Le modèle ne dispose pas de suffisamment 
        d'exemples pour estimer un effet significatif des autres types de contrats.
        
        💡 _Les salaires réels pour {profile['contract_type_clean']} peuvent 
        différer significativement de cette estimation._
        """)
        warnings_displayed += 1
    
    # Warning 2 : ML + DL combinés
    if (profile.get('contient_machine_learning') and 
        profile.get('contient_deep_learning')):
        ml_dl_corr = DataDistributions.get_ml_dl_correlation()
        total_offers = DataDistributions.get_total_offers()
        
        st.info(f"""
        ℹ️ **Machine Learning & Deep Learning** :
        
        Vous avez coché les deux compétences. Notez que leur corrélation est de 
        **{ml_dl_corr:.1%}** dans le dataset ({total_offers:,} offres), ce qui 
        signifie qu'elles sont partiellement liées mais distinctes.
        
        Le modèle peut appliquer un effet de **rendement décroissant** : 
        maîtriser les deux apporte moins que 2× l'impact d'une seule compétence.
        """)
        warnings_displayed += 1
    
    # Warning 3 : Expérience très faible
    if profile.get('experience_final', 0) < 0.5:
        st.info(f"""
        ℹ️ **Profil débutant** :
        
        Avec moins de 6 mois d'expérience, les prédictions peuvent être moins 
        précises. Le marché pour les profils très juniors est plus volatil et 
        dépend fortement du type de formation et des stages effectués.
        
        💡 _Considérez cette estimation comme une fourchette indicative large._
        """)
        warnings_displayed += 1
    
    # Warning 4 : Expérience très élevée
    if profile.get('experience_final', 0) >= 15:
        st.info(f"""
        ℹ️ **Profil senior/expert** :
        
        Avec {profile['experience_final']:.0f} ans d'expérience, vous êtes dans 
        le haut du spectre. Les salaires à ce niveau dépendent fortement de :
        - Votre expertise spécifique
        - Votre réseau professionnel
        - Vos responsabilités managériales
        - Votre réputation dans le domaine
        
        💡 _Cette estimation peut être sous-évaluée pour des profils experts 
        avec une forte valeur ajoutée._
        """)
        warnings_displayed += 1
    
    # Warning 5 : Secteur non spécifié
    if profile.get('sector_clean') == 'Non spécifié':
        st.warning(f"""
        ⚠️ **Secteur non spécifié** :
        
        Vous n'avez pas précisé votre secteur d'activité. Le modèle utilise 
        une valeur neutre, mais le secteur peut avoir un impact significatif :
        
        - **Banque/Finance** : +15% à +25%
        - **Tech/Startup** : +10% à +15%
        - **Retail** : -10%
        
        💡 _Précisez votre secteur pour une estimation plus précise._
        """)
        warnings_displayed += 1
    
    # Si aucun warning, afficher un message positif
    if warnings_displayed == 0:
        st.success("""
        ✅ **Profil standard** : Votre profil ne présente pas de particularités 
        qui pourraient affecter la précision de la prédiction.
        """)


# ============================================================================
# MODE DEBUG
# ============================================================================

def render_debug_section(profile: Dict[str, Any], model_utils: Any) -> None:
    """
    Affiche la section debug avec inspection technique.
    
    Permet aux utilisateurs avancés de :
    - Voir le résumé du profil
    - Inspecter les features envoyées au modèle
    - Vérifier l'encodage one-hot
    - Examiner les features actives
    
    Args:
        profile: Profil complet de l'utilisateur
        model_utils: Gestionnaire du modèle
    """
    with st.expander("🔬 Mode Debug (Vérification technique)"):
        st.markdown("### Debug : Vérification du profil et des features")
        
        # Section 1 : Résumé du profil
        _render_profile_summary(profile)
        
        st.markdown("---")
        
        # Section 2 : Features brutes
        _render_raw_features(profile, model_utils)
        
        st.markdown("---")
        
        # Section 3 : Vérification encodage
        _render_encoding_verification(profile, model_utils)


def _render_profile_summary(profile: Dict[str, Any]) -> None:
    """Affiche le résumé du profil."""
    st.markdown("#### 📋 Résumé de votre profil")
    
    profile_summary = CalculationUtils.create_profile_summary(profile)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Poste** : {profile_summary['job_info']}")
        st.write(f"**Localisation** : {profile_summary['location_sector']}")
        st.write(f"**Formation** : {profile_summary['education_exp']}")
        st.write(f"**Télétravail** : {profile_summary['telework']}")
    
    with col2:
        st.write(f"**Compétences** : {profile_summary['skills_count']} "
                 f"({profile_summary['key_skills']})")
        st.write(f"**Score technique** : {profile_summary['tech_score']}/15")
        st.write(f"**Avantages** : {profile_summary['benefits_score']}/4")
        st.write(f"**Description** : {profile.get('description_word_count', 0)} mots")


def _render_raw_features(profile: Dict[str, Any], model_utils: Any) -> None:
    """Affiche les features brutes envoyées au modèle."""
    st.markdown("#### 🔍 Features envoyées au modèle")
    
    try:
        df_raw = model_utils._prepare_features_for_real_model(profile)
        
        # Sélection des colonnes clés
        key_cols = [
            'sector_clean', 'location_final', 'experience_final',
            'skills_count', 'technical_score', 'description_word_count',
            'nb_mots_cles_techniques', 'is_high_paying_sector',
            'is_paris_region', 'has_modern_stack', 'hierarchy_score',
            'tech_exp_interaction', 'advanced_data_score'
        ]
        
        # Filtrer les colonnes existantes
        available_cols = [col for col in key_cols if col in df_raw.columns]
        
        # Affichage en JSON
        st.json(df_raw[available_cols].iloc[0].to_dict())
        
        # Compteur de features
        st.caption(f"📊 Total : {len(df_raw.columns)} features préparées")
        
    except Exception as e:
        st.error(f"❌ Erreur lors de la préparation des features : {str(e)}")


def _render_encoding_verification(
    profile: Dict[str, Any],
    model_utils: Any
) -> None:
    """Vérifie l'encodage one-hot des variables catégorielles."""
    st.markdown("#### 🧬 Vérification de l'encodage OneHot")
    
    try:
        # Préparation des données
        df_raw = model_utils._prepare_features_for_real_model(profile)
        fe = model_utils.model.named_steps['feature_eng']
        preprocessor = model_utils.model.named_steps['preprocessor']
        
        df_eng = fe.transform(df_raw)
        df_transformed = preprocessor.transform(df_eng)
        
        feature_names = model_utils._get_feature_names()
        
        # Secteur
        st.markdown("**🏦 Secteur** :")
        sector_features = [name for name in feature_names if 'sector_clean' in name]
        sector_active = []
        
        for name in sector_features:
            idx = feature_names.index(name)
            value = df_transformed[0][idx]
            if value == 1.0:
                sector_active.append(name)
                st.markdown(f"🟢 **{name}** = {value:.0f} ← ACTIVÉ")
        
        if not sector_active:
            st.caption("Aucune variable secteur activée (valeur par défaut)")
        
        st.markdown("")
        
        # Localisation
        st.markdown("**📍 Localisation** :")
        location_features = [name for name in feature_names if 'location_final' in name]
        location_active = []
        
        for name in location_features:
            idx = feature_names.index(name)
            value = df_transformed[0][idx]
            if value == 1.0:
                location_active.append(name)
                st.markdown(f"🟢 **{name}** = {value:.0f} ← ACTIVÉ")
        
        if not location_active:
            st.caption("Aucune variable localisation activée (valeur par défaut)")
        
        st.markdown("")
        
        # Compétences actives
        st.markdown("**🛠️ Compétences actives** :")
        bool_features = [
            name for name in feature_names 
            if name.startswith(('contient_', 'has_', 'is_'))
        ]
        active_features = []
        
        for name in bool_features:
            idx = feature_names.index(name)
            value = df_transformed[0][idx]
            if value == 1.0:
                active_features.append(name)
        
        if active_features:
            # Affichage en colonnes
            col1, col2 = st.columns(2)
            mid = len(active_features) // 2
            
            with col1:
                for feat in active_features[:mid]:
                    st.markdown(f"✅ {feat}")
            
            with col2:
                for feat in active_features[mid:]:
                    st.markdown(f"✅ {feat}")
        else:
            st.caption("Aucune compétence activée")
        
    except Exception as e:
        st.error(f"❌ Erreur lors de l'inspection : {str(e)}")


# ============================================================================
# PERFORMANCE DU MODÈLE
# ============================================================================

def render_model_performance_section(model_utils: Any) -> None:
    """
    Affiche les métriques de performance du modèle.
    
    Args:
        model_utils: Gestionnaire du modèle
    """
    with st.expander("📊 Performance du modèle XGBoost v7"):
        st.markdown("### Métriques de performance")
        
        model_perf = model_utils.get_model_performance()
        
        # Métriques principales
        _render_performance_metrics(model_perf)
        
        st.markdown("---")
        
        # Graphique de précision
        _render_precision_chart(model_perf)
        
        # Interprétation
        _render_performance_interpretation(model_perf)


def _render_performance_metrics(model_perf: Dict[str, float]) -> None:
    """Affiche les métriques principales du modèle."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "R² Score",
            f"{model_perf.get('test_r2', 0.337):.3f}",
            help="Coefficient de détermination (0-1, plus proche de 1 = meilleur)"
        )
    
    with col2:
        st.metric(
            "MAE Test",
            f"{model_perf.get('test_mae', 5163):,.0f}€",
            help="Erreur absolue moyenne sur le jeu de test"
        )
    
    with col3:
        st.metric(
            "CV MAE",
            f"{model_perf.get('cv_mae_mean', 5188):,.0f}€",
            help="Erreur moyenne en validation croisée"
        )
    
    with col4:
        st.metric(
            "Stabilité",
            f"{model_perf.get('stability', 0.995):.1%}",
            help="Cohérence des prédictions entre les folds"
        )


def _render_precision_chart(model_perf: Dict[str, float]) -> None:
    """Affiche le graphique de précision du modèle."""
    st.markdown("#### 🎯 Précision des prédictions")
    
    precision_data = {
        "Marge d'erreur": ["±5%", "±10%", "±15%", "±20%"],
        "% de prédictions": [
            Config.MODEL_INFO.get('precision_5', 25),
            Config.MODEL_INFO.get('precision_10', 45),
            Config.MODEL_INFO.get('precision_15', 65),
            Config.MODEL_INFO.get('precision_20', 80)
        ]
    }
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=precision_data["Marge d'erreur"],
        y=precision_data["% de prédictions"],
        marker_color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4'],
        text=[f"{v}%" for v in precision_data["% de prédictions"]],
        textposition='outside',
        hovertemplate=(
            '<b>%{x}</b><br>' +
            '%{y}% des prédictions<br>' +
            '<extra></extra>'
        )
    ))
    
    fig.update_layout(
        title="Précision : % de prédictions dans chaque marge d'erreur",
        yaxis_title="% de prédictions correctes",
        xaxis_title="Marge d'erreur",
        height=350,
        showlegend=False,
        plot_bgcolor='white'
    )
    
    fig.update_xaxes(gridcolor='lightgray')
    fig.update_yaxes(gridcolor='lightgray')
    
    st.plotly_chart(fig, use_container_width=True)


def _render_performance_interpretation(model_perf: Dict[str, float]) -> None:
    """Affiche l'interprétation des performances."""
    total_offers = DataDistributions.get_total_offers()
    
    st.info(f"""
    📌 **Interprétation** :
    
    - **{Config.MODEL_INFO.get('precision_15', 65)}%** des prédictions sont 
      dans une marge de ±15%
    - L'erreur P75 est de **{model_perf.get('error_75_percentile', 7417):,.0f}€**, 
      ce qui signifie que 75% des prédictions ont une erreur inférieure à cette valeur
    - Le modèle a été entraîné sur **2 681 postes Data** issus de 
      **{total_offers:,} offres HelloWork**
    - La stabilité de **{model_perf.get('stability', 0.995):.1%}** indique 
      une très bonne cohérence entre les différentes validations
    
    💡 _Le modèle est plus précis pour les profils standards (3-8 ans d'expérience, 
    secteurs bien représentés)_
    """)


# ============================================================================
# ACTIONS FINALES
# ============================================================================

def render_action_buttons(
    result: Dict[str, float],
    profile: Dict[str, Any],
    shap_exp: Optional[Dict]
) -> None:
    """
    Affiche les boutons d'action finaux.
    
    Args:
        result: Résultat de la prédiction
        profile: Profil complet
        shap_exp: Explications SHAP (optionnel)
    """
    st.markdown("### 🎯 Actions")
    
    col1, col2, col3 = st.columns(3)
    
    # Action 1 : Nouvelle estimation
    with col1:
        if st.button(
            "🔄 Nouvelle estimation",
            use_container_width=True,
            type="primary",
            help="Réinitialiser le formulaire pour une nouvelle prédiction"
        ):
            _reset_prediction_session()
    
    # Action 2 : Explorer le marché
    with col2:
        if st.button(
            "📊 Explorer le marché",
            use_container_width=True,
            help="Accéder à l'analyse du marché Data"
        ):
            st.switch_page("pages/2_📊_Marché.py")
    
    # Action 3 : Télécharger les résultats
    with col3:
        export_data = _prepare_export_data(result, profile, shap_exp)
        json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
        
        st.download_button(
            "📥 Télécharger résultat",
            data=json_str,
            file_name=f"estimation_salaire_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
            help="Exporter les résultats au format JSON"
        )


def _reset_prediction_session() -> None:
    """Réinitialise la session pour une nouvelle prédiction."""
    keys_to_delete = [
        'prediction_made',
        'last_prediction',
        'current_profile',
        'shap_explanation'
    ]
    
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]
    
    st.rerun()


def _prepare_export_data(
    result: Dict[str, float],
    profile: Dict[str, Any],
    shap_exp: Optional[Dict]
) -> Dict[str, Any]:
    """
    Prépare les données pour l'export JSON.
    
    Args:
        result: Résultat de la prédiction
        profile: Profil complet
        shap_exp: Explications SHAP
        
    Returns:
        Dict complet prêt pour l'export
    """
    # Statistiques du marché
    model_utils = st.session_state.model_utils
    real_market_data = model_utils.get_real_market_data()
    
    market_stats = {}
    if real_market_data is not None:
        market_stats = {
            'median': float(np.median(real_market_data)),
            'q1': float(np.percentile(real_market_data, 25)),
            'q3': float(np.percentile(real_market_data, 75)),
            'percentile': float(CalculationUtils.get_percentile_real(
                result['prediction'],
                real_market_data
            ))
        }
    
    # Construction du JSON d'export
    export_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'app_version': '2.0',
            'model_version': 'XGBoost_v7'
        },
        'profile': {
            k: (str(v) if not isinstance(v, (int, float, bool)) else v)
            for k, v in profile.items()
        },
        'prediction': result,
        'market_stats': market_stats,
        'shap_analysis': {
            'top_features': [
                {
                    'feature': name,
                    'impact': float(val)
                }
                for name, val in zip(
                    shap_exp['feature_names'][:10],
                    shap_exp['shap_values'][:10]
                )
            ] if shap_exp else []
        },
        'dataset_info': {
            'total_offers': DataDistributions.get_total_offers(),
            'desc_words_stats': DataDistributions.get_desc_words(),
            'tech_keywords_stats': DataDistributions.get_tech_keywords(),
            'ml_dl_correlation': DataDistributions.get_ml_dl_correlation()
        }
    }
    
    return export_data


# ============================================================================
# INFORMATIONS SUR LES CALCULS DYNAMIQUES
# ============================================================================

def render_dynamic_calculations_info(profile: Dict[str, Any]) -> None:
    """
    Affiche les informations sur les calculs automatiques.
    
    Args:
        profile: Profil complet de l'utilisateur
    """
    with st.expander("ℹ️ À propos des calculs automatiques"):
        desc_stats = DataDistributions.get_desc_words()
        tech_stats = DataDistributions.get_tech_keywords()
        total_offers = DataDistributions.get_total_offers()
        
        st.markdown(f"""
        Cette estimation utilise des **calculs dynamiques** basés sur 
        **{total_offers:,} offres réelles** :
        
        ---
        
        #### 📝 Complexité de la description : {profile['description_word_count']} mots
        
        **Comment est-ce calculé ?**
        - Basé sur votre expérience ({profile['experience_final']:.1f} ans)
        - Ajusté selon votre secteur ({profile['sector_clean']})
        - Modulé par votre nombre de compétences ({profile['skills_count']})
        
        **Distribution réelle du dataset :**
        - P10 : {desc_stats['p10']} mots
        - P25 : {desc_stats['p25']} mots
        - **Médiane : {desc_stats['median']} mots** ← Valeur centrale
        - P75 : {desc_stats['p75']} mots
        - P90 : {desc_stats['p90']} mots
        
        _Échantillon : {desc_stats['count']:,} offres analysées_
        
        ---
        
        #### 🔧 Mots-clés techniques : {profile['nb_mots_cles_techniques']}
        
        **Comment est-ce calculé ?**
        - Basé sur vos compétences cochées
        - Ajusté selon votre niveau d'expérience
        - Bonus pour les compétences avancées (ML, DL, Cloud)
        
        **Distribution réelle du dataset :**
        - P25 : {tech_stats['p25']}
        - **Médiane : {tech_stats['median']}** ← Valeur centrale
        - P75 : {tech_stats['p75']}
        - P90 : {tech_stats['p90']}
        - Moyenne : {tech_stats['mean']:.1f}
        
        _Échantillon : {tech_stats['count']:,} offres analysées_
        
        ---
        
        #### ♻️ Actualisation automatique
        
        Ces statistiques sont **recalculées automatiquement** depuis le dataset 
        à chaque démarrage de l'application.
        
        Si le dataset est mis à jour avec plus d'offres ou de nouvelles données, 
        les distributions s'ajustent automatiquement **sans modification du code**.
        
        💡 _Vous pouvez forcer le rechargement dans "Options avancées" de la sidebar_
        """)


# ============================================================================
# ORCHESTRATION FINALE
# ============================================================================

def render_all_actions_and_info(
    result: Dict[str, float],
    profile: Dict[str, Any],
    shap_exp: Optional[Dict],
    model_utils: Any
) -> None:
    """
    Orchestrate l'affichage de toutes les actions et informations finales.
    
    Cette fonction regroupe :
    - Warnings contextuels
    - Informations sur les calculs dynamiques
    - Mode debug
    - Performance du modèle
    - Actions finales
    
    Args:
        result: Résultat de la prédiction
        profile: Profil complet
        shap_exp: Explications SHAP
        model_utils: Gestionnaire du modèle
    """
    st.markdown("---")
    
    # 1. Warnings contextuels
    render_contextual_warnings(profile)
    
    # 2. Informations calculs dynamiques
    render_dynamic_calculations_info(profile)
    
    # 3. Mode debug
    render_debug_section(profile, model_utils)
    
    # 4. Performance du modèle
    render_model_performance_section(model_utils)
    
    st.markdown("---")
    
    # 5. Actions finales
    render_action_buttons(result, profile, shap_exp)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'render_contextual_warnings',
    'render_debug_section',
    'render_model_performance_section',
    'render_action_buttons',
    'render_dynamic_calculations_info',
    'render_all_actions_and_info'
]
