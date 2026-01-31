"""
Module d'export et négociation pour la page Carrière.

Ce module contient :
- Simulateur de négociation salariale
- Export JSON complet de la feuille de route
- Export CSV simplifié de la roadmap
- Navigation entre pages
- Footer avec informations

"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple


# ============================================================================
# SIMULATEUR DE NÉGOCIATION
# ============================================================================

def render_negotiation_simulator(
    profile: Dict[str, Any],
    base_salary: float,
    df_final: pd.DataFrame
) -> None:
    """
    Affiche le simulateur de négociation salariale.
    
    Args:
        profile: Profil de l'utilisateur
        base_salary: Salaire estimé actuel
        df_final: DataFrame complet du marché
        
    Notes:
        Calcule les fourchettes du marché pour le poste et la ville.
        Fournit des stratégies et phrases types pour négocier.
    """
    st.markdown("## 💬 Simulateur de négociation salariale")
    
    st.info("""
    💡 **Préparez votre négociation** : Basé sur les fourchettes réelles 
    des offres similaires, nous vous donnons des arguments chiffrés pour 
    votre prochaine discussion salariale.
    """)
    
    # Filtrer les offres similaires
    similar_offers = _filter_similar_offers(profile, df_final)
    
    if len(similar_offers) < 5:
        _render_insufficient_data_warning(profile)
        return
    
    # Calculer les fourchettes
    market_ranges = _calculate_market_ranges(similar_offers)
    
    # Calculer la fourchette personnalisée
    personal_range = _calculate_personal_range(base_salary, market_ranges)
    
    # Afficher les fourchettes
    _render_market_ranges(
        market_ranges,
        personal_range,
        base_salary,
        profile,
        len(similar_offers)
    )
    
    # Phrases types pour négocier
    _render_negotiation_phrases(profile, base_salary, market_ranges, personal_range)


def _filter_similar_offers(
    profile: Dict[str, Any],
    df_final: pd.DataFrame
) -> pd.DataFrame:
    """
    Filtre les offres similaires (même poste + même ville).
    
    Args:
        profile: Profil de l'utilisateur
        df_final: DataFrame complet
        
    Returns:
        DataFrame des offres similaires
    """
    job_type = profile.get('job_type', '')
    location = profile.get('location_final', '')
    
    similar_offers = df_final[
        (df_final['job_type_with_desc'] == job_type) &
        (df_final['location_final'] == location)
    ]
    
    return similar_offers


def _calculate_market_ranges(similar_offers: pd.DataFrame) -> Dict[str, float]:
    """
    Calcule les fourchettes du marché.
    
    Args:
        similar_offers: Offres similaires
        
    Returns:
        Dict avec min, max, median
    """
    # Vérifier si les colonnes min/max existent
    has_ranges = (
        'salary_min_clean' in similar_offers.columns and
        'salary_max_clean' in similar_offers.columns
    )
    
    if has_ranges:
        min_observed = similar_offers['salary_min_clean'].median()
        max_observed = similar_offers['salary_max_clean'].median()
    else:
        # Utiliser salary_mid avec une marge
        median = similar_offers['salary_mid'].median()
        min_observed = median * 0.9
        max_observed = median * 1.1
    
    median_salary = similar_offers['salary_mid'].median()
    
    return {
        'min': float(min_observed),
        'max': float(max_observed),
        'median': float(median_salary)
    }


def _calculate_personal_range(
    base_salary: float,
    market_ranges: Dict[str, float]
) -> Dict[str, float]:
    """
    Calcule la fourchette de négociation personnalisée.
    
    Args:
        base_salary: Salaire estimé
        market_ranges: Fourchettes du marché
        
    Returns:
        Dict avec lower, target, upper
    """
    # Erreur P75 du modèle (basé sur les métriques)
    error_p75 = 7417
    
    lower = max(base_salary - error_p75, market_ranges['min'])
    upper = min(base_salary + error_p75, market_ranges['max'])
    
    return {
        'lower': float(lower),
        'target': float(base_salary),
        'upper': float(upper)
    }


def _render_market_ranges(
    market_ranges: Dict[str, float],
    personal_range: Dict[str, float],
    base_salary: float,
    profile: Dict[str, Any],
    offers_count: int
) -> None:
    """
    Affiche les fourchettes du marché et personnalisées.
    
    Args:
        market_ranges: Fourchettes du marché
        personal_range: Fourchette personnalisée
        base_salary: Salaire estimé
        profile: Profil de l'utilisateur
        offers_count: Nombre d'offres similaires
    """
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        ### 📊 Fourchettes du marché
        
        **Basé sur {offers_count} offres** pour **{profile.get('job_type', 'votre poste')}** 
        à **{profile.get('location_final', 'votre ville')}** :
        
        - **Minimum typique** : {market_ranges['min']:,.0f}€
        - **Maximum typique** : {market_ranges['max']:,.0f}€
        - **Médiane** : {market_ranges['median']:,.0f}€
        """)
        
        st.markdown(f"""
        ### 🎯 Votre fourchette de négociation
        
        **Plancher raisonnable** : {personal_range['lower']:,.0f}€  
        **Cible médiane** : {personal_range['target']:,.0f}€  
        **Plafond justifiable** : {personal_range['upper']:,.0f}€
        """)
    
    with col2:
        st.markdown("### 💡 Stratégie")
        
        if base_salary < market_ranges['min']:
            st.warning(f"""
            ⚠️ **Sous-évalué**
            
            Votre estimation est **en dessous du minimum du marché**.
            
            → Demandez au moins **{market_ranges['min']:,.0f}€**
            """)
        elif base_salary > market_ranges['max']:
            st.info(f"""
            🌟 **Au-dessus du marché**
            
            Votre profil est **premium**.
            
            → Justifiez votre demande par votre expertise unique
            """)
        else:
            st.success(f"""
            ✅ **Dans la fourchette**
            
            Votre estimation est **cohérente avec le marché**.
            
            → Négociez sereinement autour de **{base_salary:,.0f}€**
            """)


def _render_insufficient_data_warning(profile: Dict[str, Any]) -> None:
    """
    Affiche un avertissement si données insuffisantes.
    
    Args:
        profile: Profil de l'utilisateur
    """
    st.warning(f"""
    ⚠️ **Données insuffisantes** pour générer des fourchettes de négociation précises.
    
    **Raison possible** :
    - Moins de 5 offres similaires pour **{profile.get('job_type', 'votre poste')}** 
      à **{profile.get('location_final', 'votre ville')}**
    - Combinaison poste/ville rare dans le dataset
    
    💡 **Alternative** : Consultez l'onglet **📊 Marché** pour explorer les salaires 
    par ville et secteur de manière plus générale.
    """)


def _render_negotiation_phrases(
    profile: Dict[str, Any],
    base_salary: float,
    market_ranges: Dict[str, float],
    personal_range: Dict[str, float]
) -> None:
    """
    Affiche les phrases types pour négocier.
    
    Args:
        profile: Profil de l'utilisateur
        base_salary: Salaire estimé
        market_ranges: Fourchettes du marché
        personal_range: Fourchette personnalisée
    """
    with st.expander("📝 Phrases types pour négocier", expanded=False):
        job_type = profile.get('job_type', 'ce poste')
        location = profile.get('location_final', 'cette ville')
        skills_count = profile.get('skills_count', 0)
        
        # Lister quelques compétences clés
        key_skills = []
        skill_mapping = [
            ('Python', 'contient_python'),
            ('SQL', 'contient_sql'),
            ('Machine Learning', 'contient_machine_learning'),
            ('AWS', 'contient_aws'),
            ('Spark', 'contient_spark')
        ]
        
        for name, key in skill_mapping:
            if profile.get(key, False):
                key_skills.append(name)
        
        skills_str = ', '.join(key_skills[:3]) if key_skills else 'mes compétences clés'
        
        st.markdown(f"""
        **Exemple 1 : Argumenter avec les données**
        > "Selon mon analyse du marché basée sur des offres similaires, 
        > la fourchette pour un {job_type} à {location} se situe entre 
        > {market_ranges['min']:,.0f}€ et {market_ranges['max']:,.0f}€. 
        > Compte tenu de mon expérience et de mes compétences, je vise une 
        > rémunération de {base_salary:,.0f}€."
        
        **Exemple 2 : Justifier sa valeur**
        > "Mon profil combine {skills_count} compétences clés dont {skills_str}, 
        > ce qui me permet d'apporter une valeur ajoutée significative à votre équipe."
        
        **Exemple 3 : Demander une fourchette**
        > "Quelle est la fourchette salariale prévue pour ce poste ? 
        > De mon côté, sur la base de mes recherches, j'envisage une rémunération 
        > entre {personal_range['lower']:,.0f}€ et {personal_range['upper']:,.0f}€."
        
        **Exemple 4 : Négocier à la hausse**
        > "Je suis très intéressé par ce poste. Cependant, compte tenu du marché 
        > et de mon expertise en {skills_str}, je souhaiterais discuter d'une 
        > rémunération plus proche de {personal_range['upper']:,.0f}€."
        
        **Exemple 5 : Demander des avantages**
        > "Si la fourchette salariale est fixe, serions-nous en mesure de discuter 
        > d'avantages complémentaires comme le télétravail, la formation continue, 
        > ou des primes sur objectifs ?"
        """)


# ============================================================================
# EXPORT DE LA FEUILLE DE ROUTE
# ============================================================================

def render_export_section(
    profile: Dict[str, Any],
    base_salary: float,
    percentile: float,
    df_final: pd.DataFrame,
    market_median: float
) -> None:
    """
    Affiche la section d'export de la feuille de route.
    
    Args:
        profile: Profil complet
        base_salary: Salaire estimé
        percentile: Percentile sur le marché
        df_final: DataFrame du marché
        market_median: Médiane du marché
    """
    st.markdown("## 📥 Télécharger votre feuille de route")
    
    col1, col2 = st.columns(2)
    
    with col1:
        _render_json_export(profile, base_salary, percentile, df_final, market_median)
    
    with col2:
        _render_csv_export(profile, base_salary)


def _render_json_export(
    profile: Dict[str, Any],
    base_salary: float,
    percentile: float,
    df_final: pd.DataFrame,
    market_median: float
) -> None:
    """
    Bouton d'export JSON complet.
    
    Args:
        profile: Profil complet
        base_salary: Salaire estimé
        percentile: Percentile
        df_final: DataFrame du marché
        market_median: Médiane du marché
    """
    career_plan = _prepare_career_plan_json(
        profile,
        base_salary,
        percentile,
        df_final,
        market_median
    )
    
    json_str = json.dumps(career_plan, indent=2, ensure_ascii=False)
    
    st.download_button(
        "📄 Télécharger feuille de route (JSON)",
        data=json_str,
        file_name=f"feuille_route_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True,
        help="Export complet avec profil, roadmap, transitions et projections"
    )


def _prepare_career_plan_json(
    profile: Dict[str, Any],
    base_salary: float,
    percentile: float,
    df_final: pd.DataFrame,
    market_median: float
) -> Dict[str, Any]:
    """
    Prépare le JSON complet de la feuille de route.
    
    Args:
        profile: Profil complet
        base_salary: Salaire estimé
        percentile: Percentile
        df_final: DataFrame du marché
        market_median: Médiane du marché
        
    Returns:
        Dict complet pour export
    """
    # Récupérer les données de session si disponibles
    skill_impacts = st.session_state.get('skill_impacts', {})
    transitions = st.session_state.get('transitions', {})
    scenarios = st.session_state.get('salary_scenarios', {})
    
    # Lister les compétences
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
    
    present_skills = [name for name, key in all_skills if profile.get(key, False)]
    missing_skills = [name for name, key in all_skills if not profile.get(key, False)]
    
    # Préparer le TOP 3 compétences
    top_skills = []
    if skill_impacts:
        sorted_impacts = sorted(
            skill_impacts.items(),
            key=lambda x: x.get('roi', 0),
            reverse=True
        )[:3]
        
        for skill, metrics in sorted_impacts:
            top_skills.append({
                'competence': skill,
                'gain_euro': float(metrics.get('gain', 0)),
                'roi': float(metrics.get('roi', 0)),
                'frequence_marche': float(metrics.get('frequency', 0))
            })
    
    # Préparer les transitions
    top_transitions = []
    if transitions:
        sorted_transitions = sorted(
            transitions.items(),
            key=lambda x: x[1].get('gain', 0),
            reverse=True
        )[:3]
        
        for role, data in sorted_transitions:
            top_transitions.append({
                'role': role,
                'gain_euro': float(data.get('gain', 0)),
                'temps_apprentissage_mois': int(data.get('learning_time', 0))
            })
    
    # Préparer les projections
    projection_5_ans = {}
    if scenarios:
        for scenario, values in scenarios.items():
            if len(values) >= 3:  # Au moins 3 points
                projection_5_ans[scenario] = float(values[2])  # Index 2 = 4 ans
    
    return {
        'metadata': {
            'date_generation': datetime.now().isoformat(),
            'dataset_size': len(df_final),
            'model_version': 'XGBoost_v7'
        },
        'profil_actuel': {
            'poste': profile.get('job_type', ''),
            'experience_annees': float(profile.get('experience_final', 0)),
            'ville': profile.get('location_final', ''),
            'secteur': profile.get('sector_clean', ''),
            'competences_presentes': present_skills,
            'competences_manquantes': missing_skills,
            'skills_count': int(profile.get('skills_count', 0))
        },
        'estimation_salariale': {
            'salaire_actuel_euro': float(base_salary),
            'percentile_marche': float(percentile),
            'mediane_marche_euro': float(market_median),
            'ecart_vs_mediane_euro': float(base_salary - market_median)
        },
        'roadmap_competences': {
            'top_3_priorites': top_skills
        },
        'transitions_recommandees': top_transitions,
        'projection_5_ans': projection_5_ans
    }


def _render_csv_export(
    profile: Dict[str, Any],
    base_salary: float
) -> None:
    """
    Bouton d'export CSV simplifié.
    
    Args:
        profile: Profil complet
        base_salary: Salaire estimé
    """
    # Récupérer les impacts de compétences si disponibles
    skill_impacts = st.session_state.get('skill_impacts', {})
    
    if not skill_impacts:
        st.info("Aucune roadmap de compétences à exporter")
        return
    
    # Créer le DataFrame
    roadmap_data = []
    for skill, metrics in sorted(
        skill_impacts.items(),
        key=lambda x: x[1].get('roi', 0),
        reverse=True
    ):
        roadmap_data.append({
            'Compétence': skill,
            'Gain (€)': metrics.get('gain', 0),
            'Fréquence marché (%)': metrics.get('frequency', 0) * 100,
            'ROI': metrics.get('roi', 0),
            'Score rareté': metrics.get('rarity_score', 0)
        })
    
    roadmap_df = pd.DataFrame(roadmap_data)
    csv = roadmap_df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        "📊 Télécharger roadmap (CSV)",
        data=csv,
        file_name=f"roadmap_competences_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True,
        help="Export simplifié de la roadmap de compétences"
    )


# ============================================================================
# NAVIGATION ET FOOTER
# ============================================================================

def render_navigation_footer(df_size: int, market_median: float) -> None:
    """
    Affiche la navigation et le footer.
    
    Args:
        df_size: Taille du dataset
        market_median: Médiane du marché
    """
    st.markdown("---")
    
    # Navigation
    st.markdown("### 🧭 Navigation rapide")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🏠 Accueil", use_container_width=True):
            st.switch_page("01_Accueil.py")
    
    with col2:
        if st.button("🔮 Prédiction", use_container_width=True):
            st.switch_page("pages/1_🔮_Prédiction.py")
    
    with col3:
        if st.button("📊 Marché", use_container_width=True):
            st.switch_page("pages/2_📊_Marché.py")
    
    with col4:
        if st.button("🔄 Nouvelle analyse", use_container_width=True):
            # Réinitialiser les états sauf model_utils
            for key in list(st.session_state.keys()):
                if key not in ['model_utils']:
                    del st.session_state[key]
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #666; padding: 20px 0;'>
        <p>💡 <strong>Note</strong> : Toutes les recommandations sont basées sur 
        l'analyse de <strong>{df_size:,} offres réelles</strong>.</p>
        <p>Aucune donnée externe ni hypothèse artificielle. 
        Modèle : <strong>XGBoost v7</strong> • 
        Médiane marché : <strong>{market_median:,.0f}€</strong></p>
        <p style='font-size: 12px; margin-top: 10px;'>
            © 2026 Prédicteur de salaires Data Jobs • Données HelloWork (janvier 2026)
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'render_negotiation_simulator',
    'render_export_section',
    'render_navigation_footer'
]
