"""
Module de transitions de rôle et projections salariales.

Ce module contient :
- Analyse des transitions de rôle possibles
- Calcul des compétences manquantes par rôle
- Matching avec profils réels similaires
- Projection salariale à 10 ans (3 scénarios)
- Comparaison des trajectoires

"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional, Any

from utils.config import Config
from utils.model_utils import CalculationUtils


# ============================================================================
# DÉFINITION DES DIFFICULTÉS D'APPRENTISSAGE
# ============================================================================

LEARNING_DIFFICULTY = {
    'SQL': 3,
    'Tableau': 4,
    'Power BI': 4,
    'Python': 6,
    'R': 6,
    'ETL': 7,
    'AWS': 8,
    'Azure': 8,
    'Spark': 9,
    'Machine Learning': 12,
    'Deep Learning': 15
}


# ============================================================================
# TRANSITIONS DE RÔLE
# ============================================================================

def render_transitions_analysis(
    profile: Dict[str, Any],
    base_salary: float,
    df_final: pd.DataFrame,
    model_utils: Any
) -> None:
    """
    Analyse les transitions de rôle possibles.
    
    Args:
        profile: Profil actuel de l'utilisateur
        base_salary: Salaire actuel
        df_final: DataFrame complet du marché
        model_utils: Gestionnaire du modèle
    """
    st.markdown("## 🔄 Transitions de rôle envisageables")
    
    st.info("""
    💡 **Analyse prédictive** : Pour chaque type de poste Data, nous calculons :
    - Le salaire estimé si vous changiez de rôle aujourd'hui
    - Les compétences manquantes pour ce rôle (basé sur les offres réelles)
    - Le temps d'apprentissage estimé et le gain potentiel
    """)
    
    # Identifier les rôles cibles (tous sauf le rôle actuel)
    current_role = profile['job_type']
    target_roles = [role for role in Config.JOB_TYPES if role != current_role]
    
    if not target_roles:
        st.warning("Aucune transition disponible")
        return
    
    # Calculer les transitions possibles
    transitions = _calculate_role_transitions(
        profile,
        target_roles,
        base_salary,
        df_final,
        model_utils
    )
    
    if not transitions:
        st.warning("⚠️ Impossible de calculer les transitions")
        return
    
    # Trier par gain décroissant
    sorted_transitions = sorted(
        transitions.items(),
        key=lambda x: x[1]['gain'],
        reverse=True
    )
    
    # Afficher le TOP 3
    _render_top3_transitions(sorted_transitions, base_salary)
    
    st.markdown("---")
    
    # Graphique comparatif
    _render_transitions_chart(sorted_transitions, base_salary)


def _calculate_role_transitions(
    profile: Dict[str, Any],
    target_roles: List[str],
    base_salary: float,
    df_final: pd.DataFrame,
    model_utils: Any
) -> Dict[str, Dict[str, Any]]:
    """
    Calcule les métriques pour chaque transition de rôle.
    
    Args:
        profile: Profil actuel
        target_roles: Liste des rôles cibles
        base_salary: Salaire actuel
        df_final: DataFrame du marché
        model_utils: Gestionnaire du modèle
        
    Returns:
        Dict {role: {salary, gain, missing_skills, ...}}
    """
    transitions = {}
    all_skills = _get_all_skills()
    
    with st.spinner("Analyse des transitions possibles..."):
        for role in target_roles:
            # Créer un scénario avec le nouveau rôle
            scenario = profile.copy()
            scenario['job_type'] = role
            
            # Ajuster les estimations dynamiques
            scenario['description_word_count'] = (
                CalculationUtils.estimate_description_complexity(scenario)
            )
            scenario['nb_mots_cles_techniques'] = (
                CalculationUtils.estimate_technical_keywords(scenario)
            )
            
            # Prédiction
            pred = model_utils.predict(scenario)
            
            if not pred:
                continue
            
            gain = pred['prediction'] - base_salary
            
            # Analyser les compétences requises pour ce rôle
            role_offers = df_final[df_final['job_type_with_desc'] == role]
            
            if len(role_offers) >= 10:
                req_skills = _identify_required_skills(
                    role_offers,
                    profile,
                    all_skills
                )
                
                # Calculer le temps total d'apprentissage
                total_months = sum(
                    LEARNING_DIFFICULTY.get(skill, 6)
                    for skill in req_skills.keys()
                )
                
                roi_monthly = gain / total_months if total_months > 0 else 0
                
                transitions[role] = {
                    'salary': pred['prediction'],
                    'gain': gain,
                    'missing_skills': req_skills,
                    'offer_count': len(role_offers),
                    'learning_time': total_months,
                    'roi_monthly': roi_monthly
                }
    
    return transitions


def _get_all_skills() -> List[Tuple[str, str]]:
    """Retourne la liste complète des compétences."""
    return [
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


def _identify_required_skills(
    role_offers: pd.DataFrame,
    profile: Dict[str, Any],
    all_skills: List[Tuple[str, str]]
) -> Dict[str, float]:
    """
    Identifie les compétences requises pour un rôle.
    
    Args:
        role_offers: DataFrame des offres pour ce rôle
        profile: Profil actuel
        all_skills: Liste de toutes les compétences
        
    Returns:
        Dict {skill_name: frequency} pour les compétences manquantes
    """
    req_skills = {}
    
    for name, key in all_skills:
        # Vérifier si la compétence est déjà possédée
        if profile.get(key, False):
            continue
        
        # Vérifier si la colonne existe
        if key not in role_offers.columns:
            continue
        
        # Calculer la fréquence de la compétence dans les offres
        req_rate = role_offers[key].mean()
        
        # Garder seulement si demandée dans >30% des offres
        if req_rate > 0.3:
            req_skills[name] = req_rate
    
    return req_skills


def _render_top3_transitions(
    sorted_transitions: List[Tuple[str, Dict]],
    base_salary: float
) -> None:
    """
    Affiche le TOP 3 des transitions les plus rentables.
    
    Args:
        sorted_transitions: Liste triée des transitions
        base_salary: Salaire actuel
    """
    st.markdown("### 🎯 TOP 3 des transitions les plus rentables")
    
    medals = ["🥇", "🥈", "🥉"]
    
    for i, (role, data) in enumerate(sorted_transitions[:3], 1):
        with st.expander(
            f"{medals[i-1]} {i}. {role} (+{data['gain']:,.0f}€)",
            expanded=(i == 1)
        ):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"""
                **💰 Salaire estimé :** {data['salary']:,.0f}€  
                **📈 Gain vs actuel :** +{data['gain']:,.0f}€ 
                ({data['gain']/base_salary*100:+.1f}%)  
                **📊 Basé sur :** {data['offer_count']} offres réelles
                """)
                
                if data['missing_skills']:
                    st.markdown("**🎓 Compétences à acquérir :**")
                    for skill, rate in sorted(
                        data['missing_skills'].items(),
                        key=lambda x: x[1],
                        reverse=True
                    ):
                        months = LEARNING_DIFFICULTY.get(skill, 6)
                        st.markdown(
                            f"- **{skill}** (demandé dans {rate:.0%} "
                            f"des offres) → ~{months} mois"
                        )
                else:
                    st.success(
                        "✅ Aucune compétence manquante ! "
                        "Vous êtes prêt pour cette transition."
                    )
            
            with col2:
                st.metric("Temps d'apprentissage", f"{data['learning_time']} mois")
                st.metric("ROI mensuel", f"{data['roi_monthly']:,.0f}€/mois")
                
                if data['learning_time'] > 0:
                    st.caption(f"Rentabilisé en {data['learning_time']:.0f} mois")


def _render_transitions_chart(
    sorted_transitions: List[Tuple[str, Dict]],
    base_salary: float
) -> None:
    """
    Affiche le graphique comparatif des transitions.
    
    Args:
        sorted_transitions: Liste des transitions
        base_salary: Salaire actuel
    """
    st.markdown("### 📊 Comparaison visuelle des transitions")
    
    roles = [role for role, _ in sorted_transitions]
    salaries = [data['salary'] for _, data in sorted_transitions]
    
    fig = go.Figure()
    
    # Barre du salaire actuel
    fig.add_trace(go.Bar(
        name='Salaire actuel',
        x=roles,
        y=[base_salary] * len(roles),
        marker_color='rgba(31, 119, 180, 0.3)'
    ))
    
    # Barres des salaires après transition
    fig.add_trace(go.Bar(
        name='Salaire après transition',
        x=roles,
        y=salaries,
        marker_color='rgba(255, 127, 14, 0.8)'
    ))
    
    fig.update_layout(
        title="Comparaison salariale : Rôle actuel vs Transitions possibles",
        yaxis_title="Salaire annuel (€)",
        barmode='overlay',
        height=400,
        xaxis_tickangle=-45
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Recommandation
    if sorted_transitions:
        best_transition = sorted_transitions[0]
        st.success(f"""
        🎯 **Recommandation principale** : Transition vers **{best_transition[0]}**
        
        - Gain potentiel : **+{best_transition[1]['gain']:,.0f}€**
        - Temps d'apprentissage : **{best_transition[1]['learning_time']} mois**
        - ROI mensuel : **{best_transition[1]['roi_monthly']:,.0f}€/mois**
        """)


# ============================================================================
# PROFILS SIMILAIRES
# ============================================================================

def render_similar_profiles(
    profile: Dict[str, Any],
    df_final: pd.DataFrame
) -> None:
    """
    Affiche les profils réels similaires au profil utilisateur.
    
    Args:
        profile: Profil de l'utilisateur
        df_final: DataFrame complet
    """
    st.markdown("## 👥 Profils réels similaires au vôtre")
    
    st.info("""
    💡 **Benchmark par similarité** : Nous identifions les 5 offres du marché 
    qui correspondent le mieux à votre profil actuel (expérience, secteur, ville, stack).
    """)
    
    # Calculer le score de similarité
    similar_profiles = _calculate_similarity_scores(profile, df_final)
    
    if similar_profiles.empty:
        st.warning("⚠️ Aucun profil similaire trouvé")
        return
    
    # Afficher le tableau
    _render_similar_profiles_table(similar_profiles)
    
    # Statistiques comparatives
    _render_similarity_stats(similar_profiles, profile)


def _calculate_similarity_scores(
    profile: Dict[str, Any],
    df_final: pd.DataFrame
) -> pd.DataFrame:
    """
    Calcule les scores de similarité avec les offres du marché.
    
    Args:
        profile: Profil de l'utilisateur
        df_final: DataFrame complet
        
    Returns:
        DataFrame des 5 profils les plus similaires
    """
    df = df_final.copy()
    
    # Calculer la stack size
    skills_cols = [
        'contient_python', 'contient_sql', 'contient_r',
        'contient_tableau', 'contient_power_bi',
        'contient_aws', 'contient_azure', 'contient_spark',
        'contient_machine_learning', 'contient_deep_learning'
    ]
    
    available_cols = [col for col in skills_cols if col in df.columns]
    
    if available_cols:
        df['stack_score'] = df[available_cols].sum(axis=1)
    else:
        df['stack_score'] = 0
    
    # Distances
    df['dist_exp'] = (df['experience_final'] - profile['experience_final']).abs()
    df['dist_stack'] = (df['stack_score'] - profile['skills_count']).abs()
    df['same_sector'] = (df['sector_clean'] == profile['sector_clean']).astype(int)
    df['same_city'] = (df['location_final'] == profile['location_final']).astype(int)
    
    # Score de similarité pondéré
    df['similarity'] = (
        -df['dist_exp'] * 0.3
        - df['dist_stack'] * 0.2
        + df['same_sector'] * 1.0
        + df['same_city'] * 0.8
    )
    
    # Top 5
    top_matches = df.nlargest(5, 'similarity')[[
        'job_type_with_desc', 'location_final', 'experience_final',
        'sector_clean', 'salary_mid', 'stack_score'
    ]].copy()
    
    return top_matches


def _render_similar_profiles_table(similar_profiles: pd.DataFrame) -> None:
    """
    Affiche le tableau des profils similaires.
    
    Args:
        similar_profiles: DataFrame des profils
    """
    # Renommer les colonnes
    display_df = similar_profiles.rename(columns={
        'job_type_with_desc': 'Poste',
        'experience_final': 'Exp (ans)',
        'location_final': 'Ville',
        'sector_clean': 'Secteur',
        'salary_mid': 'Salaire (€)',
        'stack_score': 'Compétences'
    })
    
    st.dataframe(
        display_df.style.format({'Salaire (€)': '{:,.0f}€'}),
        use_container_width=True
    )


def _render_similarity_stats(
    similar_profiles: pd.DataFrame,
    profile: Dict[str, Any]
) -> None:
    """
    Affiche les statistiques sur les profils similaires.
    
    Args:
        similar_profiles: DataFrame des profils similaires
        profile: Profil de l'utilisateur
    """
    avg_similar_salary = similar_profiles['salary_mid'].mean()
    user_salary = st.session_state.get('career_salary', 0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Salaire moyen profils similaires",
            f"{avg_similar_salary:,.0f}€",
            delta=f"{avg_similar_salary - user_salary:+,.0f}€ vs vous"
        )
    
    with col2:
        percentile_similar = (
            (similar_profiles['salary_mid'] < user_salary).sum() / 
            len(similar_profiles) * 100
        )
        st.metric(
            "Votre position parmi ces profils",
            f"{percentile_similar:.0f}%",
            help="Percentile dans ce groupe de profils similaires"
        )


# ============================================================================
# PROJECTION SALARIALE À 10 ANS
# ============================================================================

def render_salary_projection(
    profile: Dict[str, Any],
    base_salary: float,
    model_utils: Any
) -> None:
    """
    Affiche la projection salariale à 10 ans selon 3 scénarios.
    
    Args:
        profile: Profil actuel
        base_salary: Salaire de base
        model_utils: Gestionnaire du modèle
    """
    st.markdown("## 📈 Projection salariale à 10 ans")
    
    st.info("""
    💡 **Simulation de trajectoires** : Nous modélisons 3 scénarios d'évolution :
    - **Passif** : Vous ne changez rien, seule l'expérience augmente
    - **Actif léger** : Vous ajoutez 1 compétence tous les 2 ans
    - **Actif intensif** : Vous complétez une stack moderne (Python + Cloud + ML + Spark)
    """)
    
    # Simuler les 3 scénarios
    scenarios = _simulate_salary_scenarios(profile, base_salary, model_utils)
    
    # Graphique de projection
    _render_projection_chart(scenarios, base_salary)
    
    # Comparaison des scénarios
    _render_scenarios_comparison(scenarios, base_salary)


def _simulate_salary_scenarios(
    profile: Dict[str, Any],
    base_salary: float,
    model_utils: Any
) -> Dict[str, List[float]]:
    """
    Simule les 3 scénarios sur 10 ans.
    
    Args:
        profile: Profil actuel
        base_salary: Salaire de base
        model_utils: Gestionnaire du modèle
        
    Returns:
        Dict {scenario_name: [salaries_by_year]}
    """
    years = np.arange(0, 11, 2)  # Tous les 2 ans
    scenarios = {
        'Passif (expérience seule)': [],
        'Actif léger (+1 compétence/2 ans)': [],
        'Actif intensif (stack moderne)': []
    }
    
    all_skills = _get_all_skills()
    current_experience = profile['experience_final']
    
    for y in years:
        exp = current_experience + y
        
        # Ajuster le seniority
        seniority = _get_seniority_for_experience(exp)
        
        # SCÉNARIO 1 : Passif
        passive_profile = profile.copy()
        passive_profile['experience_final'] = exp
        passive_profile['seniority'] = seniority
        passive_profile['description_word_count'] = (
            CalculationUtils.estimate_description_complexity(passive_profile)
        )
        passive_profile['nb_mots_cles_techniques'] = (
            CalculationUtils.estimate_technical_keywords(passive_profile)
        )
        
        pred_passive = model_utils.predict(passive_profile)
        scenarios['Passif (expérience seule)'].append(
            pred_passive['prediction'] if pred_passive else base_salary
        )
        
        # SCÉNARIO 2 : Actif léger (+1 skill tous les 2 ans)
        active_profile = passive_profile.copy()
        added = 0
        for name, key in all_skills:
            if not profile.get(key, False) and added < (y // 2):
                active_profile[key] = True
                added += 1
        
        if added > 0:
            skills_dict = {k: active_profile.get(k, False) for _, k in all_skills}
            active_profile['skills_count'] = (
                CalculationUtils.calculate_skills_count_from_profile(skills_dict)
            )
            active_profile['technical_score'] = (
                CalculationUtils.calculate_technical_score_from_profile(skills_dict)
            )
        
        pred_active = model_utils.predict(active_profile)
        scenarios['Actif léger (+1 compétence/2 ans)'].append(
            pred_active['prediction'] if pred_active else base_salary
        )
        
        # SCÉNARIO 3 : Actif intensif (stack complète)
        intensive_profile = passive_profile.copy()
        intensive_profile.update({
            'contient_python': True,
            'contient_sql': True,
            'contient_aws': True,
            'contient_spark': True,
            'contient_machine_learning': True
        })
        
        intensive_skills = {
            'contient_python': True,
            'contient_sql': True,
            'contient_aws': True,
            'contient_spark': True,
            'contient_machine_learning': True
        }
        intensive_profile['skills_count'] = 5
        intensive_profile['technical_score'] = (
            CalculationUtils.calculate_technical_score_from_profile(intensive_skills)
        )
        
        pred_intensive = model_utils.predict(intensive_profile)
        scenarios['Actif intensif (stack moderne)'].append(
            pred_intensive['prediction'] if pred_intensive else base_salary
        )
    
    return scenarios


def _get_seniority_for_experience(experience: float) -> str:
    """Détermine le seniority selon l'expérience."""
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


def _render_projection_chart(
    scenarios: Dict[str, List[float]],
    base_salary: float
) -> None:
    """
    Affiche le graphique de projection à 10 ans.
    
    Args:
        scenarios: Dict des scénarios
        base_salary: Salaire actuel
    """
    years = np.arange(0, 11, 2)
    
    fig = go.Figure()
    
    colors = {
        'Passif (expérience seule)': '#e74c3c',
        'Actif léger (+1 compétence/2 ans)': '#3498db',
        'Actif intensif (stack moderne)': '#2ecc71'
    }
    
    for label, values in scenarios.items():
        fig.add_trace(go.Scatter(
            x=years,
            y=values,
            mode='lines+markers',
            name=label,
            line=dict(color=colors[label], width=3),
            marker=dict(size=10)
        ))
    
    fig.update_layout(
        title="Projection salariale à 10 ans selon 3 stratégies",
        xaxis_title="Années dans le futur",
        yaxis_title="Salaire estimé (€)",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_scenarios_comparison(
    scenarios: Dict[str, List[float]],
    base_salary: float
) -> None:
    """
    Affiche la comparaison des 3 scénarios.
    
    Args:
        scenarios: Dict des scénarios
        base_salary: Salaire actuel
    """
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gain_passive = scenarios['Passif (expérience seule)'][-1] - base_salary
        st.metric(
            "Scénario Passif (10 ans)",
            f"{scenarios['Passif (expérience seule)'][-1]:,.0f}€",
            delta=f"+{gain_passive:,.0f}€"
        )
    
    with col2:
        gain_active = scenarios['Actif léger (+1 compétence/2 ans)'][-1] - base_salary
        st.metric(
            "Scénario Actif léger (10 ans)",
            f"{scenarios['Actif léger (+1 compétence/2 ans)'][-1]:,.0f}€",
            delta=f"+{gain_active:,.0f}€"
        )
    
    with col3:
        gain_intensive = scenarios['Actif intensif (stack moderne)'][-1] - base_salary
        st.metric(
            "Scénario Actif intensif (10 ans)",
            f"{scenarios['Actif intensif (stack moderne)'][-1]:,.0f}€",
            delta=f"+{gain_intensive:,.0f}€"
        )
    
    st.success(f"""
    💡 **Conclusion** : En adoptant une stratégie active de montée en compétences, 
    vous pourriez gagner **{gain_intensive - gain_passive:,.0f}€ de plus** sur 10 ans 
    par rapport à un scénario passif.
    """)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'render_transitions_analysis',
    'render_similar_profiles',
    'render_salary_projection'
]
