"""
Module de roadmap pédagogique et matrice effort/impact.

Ce module contient :
- Roadmap pédagogique optimisée avec calcul ROI
- TOP 3 des compétences à acquérir
- Matrice Impact vs Fréquence (scatter plot)
- Matrice Effort vs Impact avec identification des Quick Wins
- Recommandations stratégiques

"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Any

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
# ROADMAP PÉDAGOGIQUE
# ============================================================================

def render_roadmap_section(
    profile: Dict[str, Any],
    base_salary: float,
    df_final: pd.DataFrame,
    model_utils: Any
) -> None:
    """
    Affiche la roadmap pédagogique optimisée avec ROI.
    
    Args:
        profile: Profil complet de l'utilisateur
        base_salary: Salaire actuel estimé
        df_final: DataFrame complet du marché
        model_utils: Gestionnaire du modèle
        
    Notes:
        ROI = Impact_salarial / (1 - Fréquence_marché)
        Priorise les compétences rares ET impactantes
    """
    st.markdown("## 🗺️ Roadmap pédagogique")
    
    st.info("""
    💡 **Méthodologie** : Chaque compétence manquante est évaluée selon :
    - **Impact salarial** : Gain estimé par le modèle
    - **Rareté** : Moins une compétence est fréquente, plus elle est valorisée
    - **ROI = Impact / (1 - Fréquence)** → Priorise les compétences rares ET impactantes
    """)
    
    # Identifier les compétences manquantes
    all_skills = _get_all_skills_mapping()
    missing_skills = _identify_missing_skills(profile, all_skills)
    
    if not missing_skills:
        st.success("""
        🎉 **Félicitations !** Votre stack technique est complète selon nos critères.
        
        **Focus** : Approfondissement et spécialisation dans vos domaines d'expertise.
        """)
        return
    
    # Calculer les impacts pour chaque compétence manquante
    skill_impacts = _calculate_skills_impacts(
        profile,
        missing_skills,
        base_salary,
        df_final,
        model_utils
    )
    
    if not skill_impacts:
        st.warning("⚠️ Impossible de calculer les impacts des compétences")
        return
    
    # Trier par ROI décroissant
    sorted_skills = sorted(
        skill_impacts.items(),
        key=lambda x: x[1]['roi'],
        reverse=True
    )
    
    # Afficher le TOP 3
    _render_top3_skills(sorted_skills)
    
    st.markdown("---")
    
    # Matrice complète
    _render_full_roadmap_matrix(sorted_skills)


def _get_all_skills_mapping() -> List[Tuple[str, str]]:
    """
    Retourne la liste complète des compétences.
    
    Returns:
        Liste de tuples (nom_affiché, clé_profil)
    """
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


def _identify_missing_skills(
    profile: Dict[str, Any],
    all_skills: List[Tuple[str, str]]
) -> List[Tuple[str, str]]:
    """
    Identifie les compétences manquantes du profil.
    
    Args:
        profile: Profil de l'utilisateur
        all_skills: Liste complète des compétences
        
    Returns:
        Liste de tuples (nom, clé) pour les compétences manquantes
    """
    return [
        (name, key) for name, key in all_skills
        if not profile.get(key, False)
    ]


def _calculate_skills_impacts(
    profile: Dict[str, Any],
    missing_skills: List[Tuple[str, str]],
    base_salary: float,
    df_final: pd.DataFrame,
    model_utils: Any
) -> Dict[str, Dict[str, float]]:
    """
    Calcule l'impact, la fréquence et le ROI de chaque compétence.
    
    Args:
        profile: Profil actuel
        missing_skills: Compétences manquantes
        base_salary: Salaire de base
        df_final: DataFrame du marché
        model_utils: Gestionnaire du modèle
        
    Returns:
        Dict {skill_name: {gain, frequency, roi, rarity_score}}
    """
    skill_impacts = {}
    all_skills = _get_all_skills_mapping()
    
    with st.spinner("Calcul des impacts pour chaque compétence manquante..."):
        for name, key in missing_skills:
            # Créer un scénario avec la compétence ajoutée
            scenario = profile.copy()
            scenario[key] = True
            
            # Recalculer les scores
            skills_dict = {k: scenario.get(k, False) for _, k in all_skills}
            scenario['skills_count'] = (
                CalculationUtils.calculate_skills_count_from_profile(skills_dict)
            )
            scenario['technical_score'] = (
                CalculationUtils.calculate_technical_score_from_profile(skills_dict)
            )
            
            # Prédiction
            pred = model_utils.predict(scenario)
            
            if pred:
                gain = pred['prediction'] - base_salary
                
                # Calculer la fréquence dans le marché
                freq = df_final[key].mean() if key in df_final.columns else 0.5
                
                # Calculer le ROI (éviter division par zéro)
                roi = gain / (1 - freq + 0.01)
                
                skill_impacts[name] = {
                    'gain': gain,
                    'frequency': freq,
                    'roi': roi,
                    'rarity_score': (1 - freq) * 100
                }
    
    return skill_impacts


def _render_top3_skills(sorted_skills: List[Tuple[str, Dict]]) -> None:
    """
    Affiche le TOP 3 des compétences à acquérir.
    
    Args:
        sorted_skills: Liste triée par ROI décroissant
    """
    st.markdown("### 🏆 TOP 3 des compétences à acquérir")
    
    medals = ["🥇", "🥈", "🥉"]
    colors = ["#FFD700", "#C0C0C0", "#CD7F32"]
    
    for i, (skill, metrics) in enumerate(sorted_skills[:3], 1):
        col_rank, col_details = st.columns([3, 1])
        
        with col_rank:
            st.markdown(f"""
            <div style='padding: 15px; background: #f0f2f6; border-radius: 10px; 
                        margin-bottom: 10px; border-left: 5px solid {colors[i-1]};'>
                <h4 style='margin: 0; color: #1f77b4;'>{medals[i-1]} {i}. {skill}</h4>
                <p style='margin: 5px 0;'>
                    <strong>Gain estimé :</strong> +{metrics['gain']:,.0f}€/an
                </p>
                <p style='margin: 5px 0;'>
                    <strong>Présence marché :</strong> {metrics['frequency']:.0%} des offres
                </p>
                <p style='margin: 5px 0; color: #666;'>
                    <em>Score de rareté : {metrics['rarity_score']:.0f}/100</em>
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_details:
            st.metric(
                "ROI",
                f"{metrics['roi']:,.0f}",
                delta="Priorité" if i == 1 else None
            )


def _render_full_roadmap_matrix(sorted_skills: List[Tuple[str, Dict]]) -> None:
    """
    Affiche la matrice complète Impact vs Fréquence.
    
    Args:
        sorted_skills: Liste de toutes les compétences triées
    """
    st.markdown("### 📊 Vue complète : Toutes les compétences manquantes")
    
    # Préparer les données
    skills_df = pd.DataFrame([
        {
            'Compétence': name,
            'Gain (€)': metrics['gain'],
            'Fréquence (%)': metrics['frequency'] * 100,
            'ROI': metrics['roi']
        }
        for name, metrics in sorted_skills
    ])
    
    # Assurer des valeurs positives pour la taille
    skills_df['ROI_size'] = skills_df['ROI'].clip(lower=1)
    
    # Créer le scatter plot
    fig = px.scatter(
        skills_df,
        x='Fréquence (%)',
        y='Gain (€)',
        size='ROI_size',
        color='ROI',
        text='Compétence',
        title="Matrice Impact vs Fréquence (Taille = ROI)",
        color_continuous_scale='RdYlGn',
        size_max=30,
        hover_data={
            'Compétence': True,
            'Gain (€)': ':,.0f',
            'Fréquence (%)': ':.1f',
            'ROI': ':,.0f',
            'ROI_size': False
        }
    )
    
    fig.update_traces(textposition='top center')
    fig.update_layout(
        height=500,
        xaxis_title="Fréquence sur le marché (%)",
        yaxis_title="Gain salarial estimé (€)"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.success("""
    💡 **Interprétation** :
    - **En haut à gauche** (faible fréquence, fort impact) = Compétences rares 
      et très valorisées → **TOP priorité**
    - **En haut à droite** (forte fréquence, fort impact) = Compétences courantes 
      mais toujours impactantes
    - **En bas** = Compétences moins prioritaires pour votre profil
    """)


# ============================================================================
# MATRICE EFFORT / IMPACT
# ============================================================================

def render_effort_impact_matrix(
    profile: Dict[str, Any],
    base_salary: float,
    df_final: pd.DataFrame,
    model_utils: Any
) -> None:
    """
    Affiche la matrice Effort vs Impact avec identification des Quick Wins.
    
    Args:
        profile: Profil de l'utilisateur
        base_salary: Salaire actuel
        df_final: DataFrame du marché
        model_utils: Gestionnaire du modèle
    """
    st.markdown("## ⚡ Matrice Effort / Impact")
    
    st.info("""
    💡 **Aide à la décision** : Cette matrice croise l'effort d'apprentissage avec 
    le gain salarial potentiel. Objectif : identifier les **quick wins** 
    (faible effort, fort impact) vs investissements long terme.
    """)
    
    # Identifier les compétences manquantes
    all_skills = _get_all_skills_mapping()
    missing_skills = _identify_missing_skills(profile, all_skills)
    
    if not missing_skills:
        st.info("Votre stack est complète. Pas de matrice effort/impact à afficher.")
        return
    
    # Calculer les impacts
    skill_impacts = _calculate_skills_impacts(
        profile,
        missing_skills,
        base_salary,
        df_final,
        model_utils
    )
    
    if not skill_impacts:
        return
    
    # Préparer les données effort/impact
    effort_impact_data = _prepare_effort_impact_data(skill_impacts)
    
    if not effort_impact_data:
        st.warning("⚠️ Aucune compétence avec effort défini")
        return
    
    # Afficher la matrice
    _render_effort_matrix(effort_impact_data)
    
    # Recommandations stratégiques
    _render_strategic_recommendations(effort_impact_data)


def _prepare_effort_impact_data(
    skill_impacts: Dict[str, Dict[str, float]]
) -> List[Dict[str, Any]]:
    """
    Prépare les données pour la matrice effort/impact.
    
    Args:
        skill_impacts: Dict des impacts par compétence
        
    Returns:
        Liste de dicts avec effort, impact, ROI et catégorie
    """
    effort_impact_data = []
    
    for skill, metrics in skill_impacts.items():
        if skill not in LEARNING_DIFFICULTY:
            continue
        
        effort = LEARNING_DIFFICULTY[skill]
        impact = metrics['gain']
        roi_monthly = impact / effort if effort > 0 else 0
        
        # Catégoriser
        if effort <= 6 and impact >= 3000:
            category = 'Quick Win'
        elif effort > 9:
            category = 'Investissement'
        else:
            category = 'Équilibré'
        
        effort_impact_data.append({
            'Compétence': skill,
            'Effort (mois)': effort,
            'Impact (€)': impact,
            'ROI (€/mois)': roi_monthly,
            'Catégorie': category
        })
    
    return effort_impact_data


def _render_effort_matrix(effort_impact_data: List[Dict]) -> None:
    """
    Affiche le graphique de la matrice effort/impact.
    
    Args:
        effort_impact_data: Liste des données préparées
    """
    effort_df = pd.DataFrame(effort_impact_data)
    
    # Assurer des valeurs positives pour la taille
    effort_df['ROI_size'] = effort_df['ROI (€/mois)'].fillna(0).clip(lower=1)
    
    # Créer le scatter plot
    fig = px.scatter(
        effort_df,
        x='Effort (mois)',
        y='Impact (€)',
        size='ROI_size',
        color='Catégorie',
        text='Compétence',
        hover_data={
            'Compétence': True,
            'ROI (€/mois)': ':,.0f',
            'Effort (mois)': True,
            'Impact (€)': ':,.0f',
            'Catégorie': True,
            'ROI_size': False
        },
        title="Matrice Effort vs Impact (Taille = ROI mensuel)",
        color_discrete_map={
            'Quick Win': '#2ecc71',
            'Équilibré': '#3498db',
            'Investissement': '#e74c3c'
        },
        size_max=30
    )
    
    # Ajouter des lignes de référence
    if not effort_df.empty:
        fig.add_hline(
            y=effort_df['Impact (€)'].median(),
            line_dash="dash",
            line_color="gray",
            annotation_text="Impact médian",
            annotation_position="right"
        )
        
        fig.add_vline(
            x=effort_df['Effort (mois)'].median(),
            line_dash="dash",
            line_color="gray",
            annotation_text="Effort médian",
            annotation_position="top"
        )
    
    fig.update_traces(textposition='top center')
    fig.update_layout(
        height=500,
        xaxis_title="Effort d'apprentissage (mois)",
        yaxis_title="Gain salarial estimé (€)"
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_strategic_recommendations(
    effort_impact_data: List[Dict]
) -> None:
    """
    Affiche les recommandations stratégiques basées sur la matrice.
    
    Args:
        effort_impact_data: Liste des données
    """
    effort_df = pd.DataFrame(effort_impact_data)
    
    col1, col2 = st.columns(2)
    
    # Quick Wins
    with col1:
        quick_wins = effort_df[effort_df['Catégorie'] == 'Quick Win']
        
        if not quick_wins.empty:
            st.success(f"""
            🚀 **Quick Wins identifiés** ({len(quick_wins)})
            
            Compétences à fort ROI et faible effort :
            """)
            
            for _, row in quick_wins.iterrows():
                st.markdown(
                    f"- **{row['Compétence']}** : {row['Effort (mois)']} mois "
                    f"→ +{row['Impact (€)']:,.0f}€"
                )
        else:
            st.info("""
            Aucun quick win identifié. 
            
            **Focus** : Formation progressive avec les compétences équilibrées.
            """)
    
    # Investissements long terme
    with col2:
        investments = effort_df[effort_df['Catégorie'] == 'Investissement']
        
        if not investments.empty:
            st.warning(f"""
            📚 **Investissements long terme** ({len(investments)})
            
            Compétences premium nécessitant plus de temps :
            """)
            
            for _, row in investments.iterrows():
                st.markdown(
                    f"- **{row['Compétence']}** : {row['Effort (mois)']} mois "
                    f"→ +{row['Impact (€)']:,.0f}€"
                )
        else:
            st.info("Aucun investissement long terme dans votre roadmap.")
    
    # Recommandation globale
    if not effort_df.empty:
        best_roi = effort_df.nlargest(1, 'ROI (€/mois)').iloc[0]
        
        st.markdown("---")
        st.info(f"""
        💡 **Recommandation stratégique** :
        
        Commencez par **{best_roi['Compétence']}** qui offre le meilleur ROI mensuel 
        ({best_roi['ROI (€/mois)']:,.0f}€/mois) avec un effort de 
        {best_roi['Effort (mois)']} mois pour un gain de {best_roi['Impact (€)']:,.0f}€.
        """)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'render_roadmap_section',
    'render_effort_impact_matrix'
]
