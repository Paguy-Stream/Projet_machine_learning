"""
Module d'analyses détaillées du marché.

Ce module contient les 6 onglets d'analyse :
1. Vue d'ensemble (distributions, scatter)
2. Postes & Secteurs (box plots, multiplicateurs)
3. Géographie (heatmap, top villes)
4. Compétences (fréquence, impact)
5. Combinaisons (stacks gagnants, ROI)
6. Benchmark personnel (jauge, comparateur)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional

from utils.config import Config


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def render_analysis_tabs(
    filtered_data: pd.DataFrame,
    market_data: pd.DataFrame
) -> None:
    """
    Affiche les 6 onglets d'analyse détaillée.
    
    Args:
        filtered_data: Données après application des filtres
        market_data: Données complètes du marché
    """
    tabs = st.tabs([
        "📊 Vue d'ensemble",
        "💼 Postes & Secteurs",
        "🗺️ Géographie",
        "🛠️ Compétences",
        "🔗 Combinaisons",
        "🆚 Benchmark"
    ])
    
    with tabs[0]:
        render_overview_tab(filtered_data)
    
    with tabs[1]:
        render_jobs_sectors_tab(filtered_data)
    
    with tabs[2]:
        render_geography_tab(filtered_data)
    
    with tabs[3]:
        render_skills_tab(filtered_data)
    
    with tabs[4]:
        render_combinations_tab(filtered_data)
    
    with tabs[5]:
        render_benchmark_tab(filtered_data, market_data)


# ============================================================================
# TAB 1 : VUE D'ENSEMBLE
# ============================================================================

def render_overview_tab(filtered_data: pd.DataFrame) -> None:
    """
    Affiche la vue d'ensemble avec distributions et tendances.
    
    Args:
        filtered_data: Données filtrées
    """
    st.markdown("### Distribution des salaires")
    
    col1, col2 = st.columns(2)
    
    with col1:
        _render_salary_histogram(filtered_data)
    
    with col2:
        _render_contract_boxplot(filtered_data)
    
    st.markdown("---")
    
    # Salaire vs Expérience
    if 'experience_final' in filtered_data.columns:
        _render_salary_vs_experience(filtered_data)


def _render_salary_histogram(data: pd.DataFrame) -> None:
    """Histogramme de distribution des salaires."""
    median_salary = data['salary_mid'].median()
    q25 = data['salary_mid'].quantile(0.25)
    q75 = data['salary_mid'].quantile(0.75)
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=data['salary_mid'],
        nbinsx=40,
        marker_color='#1f77b4',
        name='Distribution'
    ))
    
    # Médiane
    fig.add_vline(
        x=median_salary,
        line_dash="dash",
        line_color="#ff7f0e",
        annotation_text=f"Médiane : {median_salary:,.0f}€",
        annotation_position="top"
    )
    
    # IQR
    fig.add_vrect(
        x0=q25, x1=q75,
        fillcolor="rgba(255, 127, 14, 0.1)",
        layer="below", line_width=0,
        annotation_text="IQR (50% central)",
        annotation_position="top left"
    )
    
    fig.update_layout(
        title="Distribution des salaires",
        xaxis_title="Salaire annuel (€)",
        yaxis_title="Nombre d'offres",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_contract_boxplot(data: pd.DataFrame) -> None:
    """Box plot par type de contrat."""
    if 'contract_type_clean' not in data.columns:
        st.info("Données de contrat non disponibles")
        return
    
    fig = px.box(
        data,
        x='contract_type_clean',
        y='salary_mid',
        color='contract_type_clean',
        title="Salaire par type de contrat",
        labels={
            'salary_mid': 'Salaire (€)',
            'contract_type_clean': 'Type de contrat'
        }
    )
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def _render_salary_vs_experience(data: pd.DataFrame) -> None:
    """Graphique salaire vs expérience."""
    st.markdown("### 📈 Évolution salariale selon l'expérience")
    
    fig = px.scatter(
        data,
        x='experience_final',
        y='salary_mid',
        color='job_type_simplified',
        size='skills_count' if 'skills_count' in data.columns else None,
        title="Salaire vs Expérience (taille = nb de compétences)",
        labels={
            'experience_final': 'Années d\'expérience',
            'salary_mid': 'Salaire annuel (€)',
            'job_type_simplified': 'Type de poste'
        },
        trendline="lowess",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistiques par tranche
    _render_experience_stats(data)


def _render_experience_stats(data: pd.DataFrame) -> None:
    """Tableau des statistiques par tranche d'expérience."""
    exp_bins = [0, 1, 3, 5, 8, 12, 30]
    exp_labels = ['<1 an', '1-3 ans', '3-5 ans', '5-8 ans', '8-12 ans', '12+ ans']
    
    data_copy = data.copy()
    data_copy['exp_range'] = pd.cut(
        data_copy['experience_final'],
        bins=exp_bins,
        labels=exp_labels,
        include_lowest=True
    )
    
    exp_stats = data_copy.groupby('exp_range', observed=True)['salary_mid'].agg([
        'median', 'mean', 'count'
    ])
    
    st.markdown("**📊 Statistiques par tranche d'expérience**")
    st.dataframe(
        exp_stats.style.format({
            'median': '{:,.0f}€',
            'mean': '{:,.0f}€',
            'count': '{:.0f} offres'
        }),
        use_container_width=True
    )


# ============================================================================
# TAB 2 : POSTES & SECTEURS
# ============================================================================

def render_jobs_sectors_tab(filtered_data: pd.DataFrame) -> None:
    """
    Analyse des postes et secteurs.
    
    Args:
        filtered_data: Données filtrées
    """
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💼 Salaire par type de poste")
        _render_jobs_boxplot(filtered_data)
    
    with col2:
        st.markdown("### 🏦 Salaire par secteur")
        _render_sectors_boxplot(filtered_data)
    
    st.markdown("---")
    
    # Multiplicateurs sectoriels
    _render_sector_multipliers(filtered_data)


def _render_jobs_boxplot(data: pd.DataFrame) -> None:
    """Box plot par type de poste."""
    fig = px.box(
        data,
        x='job_type_simplified',
        y='salary_mid',
        color='job_type_simplified',
        title="Distribution salariale par poste"
    )
    fig.update_layout(showlegend=False, height=400)
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)


def _render_sectors_boxplot(data: pd.DataFrame) -> None:
    """Box plot par secteur."""
    fig = px.box(
        data,
        x='sector_clean',
        y='salary_mid',
        color='sector_clean',
        title="Distribution salariale par secteur"
    )
    fig.update_layout(showlegend=False, height=400)
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)


def _render_sector_multipliers(data: pd.DataFrame) -> None:
    """Affiche les multiplicateurs sectoriels dynamiques."""
    st.markdown("### 🔢 Multiplicateurs sectoriels (calculés dynamiquement)")
    
    sector_mults = Config.get_all_sector_multipliers()
    active_sectors = data['sector_clean'].unique()
    filtered_mults = {
        k: v for k, v in sector_mults.items()
        if k in active_sectors
    }
    
    if not filtered_mults:
        st.info("Aucun multiplicateur disponible")
        return
    
    mult_df = pd.DataFrame({
        'Secteur': list(filtered_mults.keys()),
        'Multiplicateur': list(filtered_mults.values())
    }).sort_values('Multiplicateur', ascending=False)
    
    fig = px.bar(
        mult_df,
        x='Multiplicateur',
        y='Secteur',
        orientation='h',
        title="Impact sectoriel sur le salaire (vs médiane globale)",
        color='Multiplicateur',
        color_continuous_scale='RdYlGn',
        labels={'Multiplicateur': 'Multiplicateur salarial'}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    💡 **Interprétation** : Un multiplicateur de **1.25** signifie que le secteur paie 
    en moyenne **25% de plus** que la médiane globale du marché.
    """)


# ============================================================================
# TAB 3 : GÉOGRAPHIE
# ============================================================================

def render_geography_tab(filtered_data: pd.DataFrame) -> None:
    """
    Analyse géographique du marché.
    
    Args:
        filtered_data: Données filtrées
    """
    st.markdown("### 🗺️ Analyse géographique")
    
    city_stats = _calculate_city_stats(filtered_data)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        _render_top_cities_chart(city_stats)
    
    with col2:
        _render_city_multipliers(city_stats)
    
    st.markdown("---")
    
    # Heatmap ville × secteur
    _render_geography_heatmap(filtered_data, city_stats)


def _calculate_city_stats(data: pd.DataFrame) -> pd.DataFrame:
    """Calcule les statistiques par ville."""
    city_stats = data.groupby('location_clean')['salary_mid'].agg([
        'median', 'mean', 'count'
    ])
    return city_stats[city_stats['count'] >= 3].sort_values(
        'median',
        ascending=False
    )


def _render_top_cities_chart(city_stats: pd.DataFrame) -> None:
    """Graphique des top villes."""
    fig = px.bar(
        city_stats.head(15),
        x=city_stats.head(15).index,
        y='median',
        title="Top 15 villes par salaire médian",
        labels={'median': 'Salaire médian (€)', 'location_clean': 'Ville'},
        color='median',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=450, showlegend=False)
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)


def _render_city_multipliers(city_stats: pd.DataFrame) -> None:
    """Affiche les multiplicateurs géographiques."""
    st.markdown("**🔢 Multiplicateurs géographiques**")
    
    city_mults = Config.get_all_city_multipliers()
    top_cities = city_stats.head(10).index.tolist()
    
    for city in top_cities:
        mult = city_mults.get(city, 1.0)
        salary = city_stats.loc[city, 'median']
        
        color = "🟢" if mult >= 1.1 else "🟡" if mult >= 1.0 else "🔴"
        st.markdown(f"{color} **{city}** : ×{mult:.2f}")
        st.caption(f"Médiane : {salary:,.0f}€")


def _render_geography_heatmap(
    data: pd.DataFrame,
    city_stats: pd.DataFrame
) -> None:
    """Heatmap ville × secteur."""
    st.markdown("### 🌡️ Heatmap : Salaire médian par Ville × Secteur")
    
    top_cities = city_stats.head(10).index.tolist()
    top_sectors = data['sector_clean'].value_counts().head(8).index.tolist()
    
    heatmap_data = data[
        data['location_clean'].isin(top_cities) &
        data['sector_clean'].isin(top_sectors)
    ]
    
    if len(heatmap_data) < 20:
        st.warning("⚠️ Pas assez de données pour la heatmap")
        return
    
    pivot_table = heatmap_data.pivot_table(
        values='salary_mid',
        index='location_clean',
        columns='sector_clean',
        aggfunc='median'
    )
    
    fig = px.imshow(
        pivot_table,
        labels=dict(x="Secteur", y="Ville", color="Salaire médian (€)"),
        title="Heatmap des salaires médians (Top villes × Top secteurs)",
        color_continuous_scale='RdYlGn',
        aspect="auto"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.success("""
    💡 **Opportunités** : Cherchez les cases **vertes** (salaires élevés) dans des 
    secteurs/villes où vous avez de l'expérience !
    """)


# ============================================================================
# TAB 4 : COMPÉTENCES
# ============================================================================

def render_skills_tab(filtered_data: pd.DataFrame) -> None:
    """
    Analyse des compétences techniques.
    
    Args:
        filtered_data: Données filtrées
    """
    st.markdown("### 🛠️ Analyse des compétences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Fréquence des compétences**")
        _render_skills_frequency(filtered_data)
    
    with col2:
        st.markdown("**💰 Impact salarial des compétences**")
        _render_skills_impact(filtered_data)


def _render_skills_frequency(data: pd.DataFrame) -> None:
    """Graphique de fréquence des compétences."""
    skills = {
        'Python': data['contient_python'].mean() * 100,
        'SQL': data['contient_sql'].mean() * 100,
        'R': data['contient_r'].mean() * 100,
        'Tableau': data['contient_tableau'].mean() * 100,
        'Power BI': data['contient_power_bi'].mean() * 100,
        'AWS': data['contient_aws'].mean() * 100,
        'Azure': data['contient_azure'].mean() * 100,
        'GCP': data.get('contient_gcp', pd.Series([0])).mean() * 100,
        'Spark': data['contient_spark'].mean() * 100,
        'ML': data['contient_machine_learning'].mean() * 100,
        'Deep Learning': data.get('contient_deep_learning', pd.Series([0])).mean() * 100
    }
    
    skills_df = pd.DataFrame({
        'Compétence': list(skills.keys()),
        'Fréquence (%)': list(skills.values())
    }).sort_values('Fréquence (%)', ascending=False)
    
    fig = px.bar(
        skills_df,
        x='Fréquence (%)',
        y='Compétence',
        orientation='h',
        title="Fréquence des compétences dans les offres",
        color='Fréquence (%)',
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def _render_skills_impact(data: pd.DataFrame) -> None:
    """Graphique d'impact salarial des compétences."""
    skills_mapping = [
        ('Python', 'contient_python'),
        ('SQL', 'contient_sql'),
        ('R', 'contient_r'),
        ('Tableau', 'contient_tableau'),
        ('Power BI', 'contient_power_bi'),
        ('AWS', 'contient_aws'),
        ('Azure', 'contient_azure'),
        ('GCP', 'contient_gcp'),
        ('Spark', 'contient_spark'),
        ('ML', 'contient_machine_learning'),
        ('Deep Learning', 'contient_deep_learning')
    ]
    
    impacts = {}
    
    for skill_name, col_name in skills_mapping:
        if col_name not in data.columns:
            continue
        
        if data[col_name].sum() < 5:
            continue
        
        with_skill = data[data[col_name] == 1]['salary_mid'].median()
        without_skill = data[data[col_name] == 0]['salary_mid'].median()
        impact = with_skill - without_skill
        
        if not np.isnan(impact):
            impacts[skill_name] = impact
    
    if not impacts:
        st.info("Pas assez de données")
        return
    
    impact_df = pd.DataFrame({
        'Compétence': list(impacts.keys()),
        'Impact (€)': list(impacts.values())
    }).sort_values('Impact (€)', key=abs, ascending=False)
    
    fig = px.bar(
        impact_df,
        x='Impact (€)',
        y='Compétence',
        orientation='h',
        title="Impact salarial (Avec vs Sans la compétence)",
        color='Impact (€)',
        color_continuous_scale='RdYlGn'
    )
    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    ℹ️ **Rappel** : L'impact des compétences est **réel mais modeste** comparé 
    aux facteurs comme la **localisation**, le **secteur** ou l'**expérience**.
    """)


# ============================================================================
# TAB 5 : COMBINAISONS
# ============================================================================

def render_combinations_tab(filtered_data: pd.DataFrame) -> None:
    """
    Analyse des combinaisons de compétences.
    
    Args:
        filtered_data: Données filtrées
    """
    st.markdown("### 🔗 Combinaisons de compétences gagnantes")
    
    st.info("""
    💡 **Analyse des stacks complets** : Découvrez quelles combinaisons de compétences 
    sont les plus rémunératrices sur le marché.
    """)
    
    stacks = _define_tech_stacks(filtered_data)
    stack_results = _calculate_stack_statistics(filtered_data, stacks)
    
    if not stack_results:
        st.warning("⚠️ Pas assez de données pour analyser les stacks")
        return
    
    _render_stacks_chart(stack_results)
    _render_stacks_table(stack_results)
    _render_roi_analysis(filtered_data)


def _define_tech_stacks(data: pd.DataFrame) -> Dict[str, pd.Series]:
    """Définit les stacks techniques à analyser."""
    return {
        'Data Scientist (Full Stack)': (
            (data['contient_python'] == 1) &
            (data['contient_machine_learning'] == 1) &
            ((data['contient_aws'] == 1) |
             (data['contient_azure'] == 1) |
             (data.get('contient_gcp', 0) == 1))
        ),
        'Data Engineer (Cloud)': (
            (data['contient_python'] == 1) &
            (data['contient_spark'] == 1) &
            ((data['contient_aws'] == 1) | (data['contient_azure'] == 1))
        ),
        'BI Analyst': (
            (data['contient_sql'] == 1) &
            ((data['contient_tableau'] == 1) | (data['contient_power_bi'] == 1))
        ),
        'ML Engineer': (
            (data['contient_python'] == 1) &
            (data['contient_machine_learning'] == 1) &
            (data['contient_spark'] == 1)
        ),
        'Full Stack Data': (
            (data['contient_python'] == 1) &
            (data['contient_sql'] == 1) &
            ((data['contient_aws'] == 1) |
             (data['contient_azure'] == 1) |
             (data.get('contient_gcp', 0) == 1))
        )
    }


def _calculate_stack_statistics(
    data: pd.DataFrame,
    stacks: Dict[str, pd.Series]
) -> List[Dict]:
    """Calcule les statistiques pour chaque stack."""
    results = []
    
    for stack_name, mask in stacks.items():
        count = mask.sum()
        
        if count < 5:
            continue
        
        results.append({
            'Stack': stack_name,
            'Médiane (€)': data[mask]['salary_mid'].median(),
            'Moyenne (€)': data[mask]['salary_mid'].mean(),
            'Q25 (€)': data[mask]['salary_mid'].quantile(0.25),
            'Q75 (€)': data[mask]['salary_mid'].quantile(0.75),
            'Nombre d\'offres': count
        })
    
    return results


def _render_stacks_chart(stack_results: List[Dict]) -> None:
    """Graphique des salaires par stack."""
    stack_df = pd.DataFrame(stack_results).sort_values('Médiane (€)', ascending=False)
    
    fig = px.bar(
        stack_df,
        x='Médiane (€)',
        y='Stack',
        orientation='h',
        title="💰 Salaire médian par combinaison de compétences",
        color='Médiane (€)',
        color_continuous_scale='Viridis',
        text='Médiane (€)'
    )
    fig.update_traces(texttemplate='%{text:,.0f}€', textposition='outside')
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def _render_stacks_table(stack_results: List[Dict]) -> None:
    """Tableau détaillé des stacks."""
    st.markdown("#### 📊 Détails par stack")
    
    stack_df = pd.DataFrame(stack_results).sort_values('Médiane (€)', ascending=False)
    
    st.dataframe(
        stack_df.style.format({
            'Médiane (€)': '{:,.0f}€',
            'Moyenne (€)': '{:,.0f}€',
            'Q25 (€)': '{:,.0f}€',
            'Q75 (€)': '{:,.0f}€',
            'Nombre d\'offres': '{:.0f}'
        }),
        use_container_width=True
    )
    
    best_stack = stack_df.iloc[0]
    st.success(f"""
    🌟 **Stack la plus rémunératrice** : **{best_stack['Stack']}**  
    - Salaire médian : **{best_stack['Médiane (€)']:,.0f}€**  
    - Basé sur **{best_stack['Nombre d\'offres']:.0f} offres**
    """)


def _render_roi_analysis(data: pd.DataFrame) -> None:
    """Analyse ROI des compétences."""
    st.markdown("---")
    st.markdown("#### 💡 ROI des compétences (Retour sur Investissement)")
    
    st.info("""
    **Méthodologie** : Impact salarial divisé par la difficulté d'apprentissage estimée.
    
    - **Facile** (SQL, Tableau) : 3-6 mois  
    - **Moyen** (Python, Power BI) : 6-12 mois  
    - **Avancé** (ML, Cloud, Spark) : 12-24 mois
    """)
    
    # Calculer impacts et ROI
    roi_data = _calculate_roi_data(data)
    
    if not roi_data:
        return
    
    roi_df = pd.DataFrame(roi_data).sort_values('ROI (€/mois)', ascending=False)
    roi_df['size_display'] = roi_df['ROI (€/mois)'].clip(lower=0.1)
    
    fig = px.scatter(
        roi_df,
        x='Difficulté (mois)',
        y='Impact (€)',
        size='size_display',
        color='ROI (€/mois)',
        text='Compétence',
        title="ROI des compétences : Impact vs Effort d'apprentissage",
        color_continuous_scale='RdYlGn',
        size_max=30
    )
    fig.update_traces(textposition='top center')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.success("💡 **En haut à gauche** = Meilleur ROI (fort impact, peu d'effort)")


def _calculate_roi_data(data: pd.DataFrame) -> List[Dict]:
    """Calcule les données ROI pour chaque compétence."""
    learning_difficulty = {
        'SQL': 4, 'Tableau': 4, 'Power BI': 5,
        'Python': 6, 'R': 6,
        'AWS': 8, 'Azure': 8, 'GCP': 8,
        'Spark': 9, 'ML': 10, 'Deep Learning': 12
    }
    
    skills_mapping = {
        'Python': 'contient_python',
        'SQL': 'contient_sql',
        'R': 'contient_r',
        'Tableau': 'contient_tableau',
        'Power BI': 'contient_power_bi',
        'AWS': 'contient_aws',
        'Azure': 'contient_azure',
        'Spark': 'contient_spark',
        'ML': 'contient_machine_learning',
        'Deep Learning': 'contient_deep_learning'
    }
    
    roi_data = []
    
    for skill, col in skills_mapping.items():
        if col not in data.columns or data[col].sum() < 5:
            continue
        
        if skill not in learning_difficulty:
            continue
        
        with_skill = data[data[col] == 1]['salary_mid'].median()
        without_skill = data[data[col] == 0]['salary_mid'].median()
        impact = with_skill - without_skill
        
        if np.isnan(impact):
            continue
        
        difficulty = learning_difficulty[skill]
        roi = impact / difficulty
        
        roi_data.append({
            'Compétence': skill,
            'Impact (€)': impact,
            'Difficulté (mois)': difficulty,
            'ROI (€/mois)': roi
        })
    
    return roi_data


# ============================================================================
# TAB 6 : BENCHMARK
# ============================================================================

def render_benchmark_tab(
    filtered_data: pd.DataFrame,
    market_data: pd.DataFrame
) -> None:
    """
    Benchmark personnel et comparateur de profils.
    
    Args:
        filtered_data: Données filtrées
        market_data: Données complètes du marché
    """
    st.markdown("### 📊 Où vous situez-vous sur le marché ?")
    
    st.info("""
    💡 **Benchmark personnel** : Comparez votre profil au marché pour identifier 
    vos points forts et opportunités d'amélioration.
    """)
    
    # Section 1 : Benchmark salarial
    _render_salary_benchmark(filtered_data)
    
    st.markdown("---")
    
    # Section 2 : Comparateur de profils
    _render_profile_comparator(filtered_data)


def _render_salary_benchmark(data: pd.DataFrame) -> None:
    """Benchmark salarial avec jauge."""
    st.markdown("#### 💰 Benchmark salarial")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_salary = st.number_input(
            "Votre salaire actuel (€/an)",
            min_value=20000,
            max_value=150000,
            value=45000,
            step=1000
        )
        
        percentile = (data['salary_mid'] < user_salary).sum() / len(data) * 100
        median_salary = data['salary_mid'].median()
        
        # Jauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=percentile,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Votre percentile sur le marché"},
            delta={'reference': 50, 'valueformat': '.0f'},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#1f77b4"},
                'steps': [
                    {'range': [0, 25], 'color': "#ffcccc"},
                    {'range': [25, 50], 'color': "#ffffcc"},
                    {'range': [50, 75], 'color': "#ccffcc"},
                    {'range': [75, 100], 'color': "#99ff99"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': percentile
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.metric("Votre percentile", f"{percentile:.0f}%")
        st.metric("Médiane du marché", f"{median_salary:,.0f}€")
        
        diff = user_salary - median_salary
        st.metric("Écart vs médiane", f"{diff:+,.0f}€")
        
        # Interprétation
        if percentile >= 90:
            st.success("🌟 Top 10% du marché !")
        elif percentile >= 75:
            st.success("✅ Excellent positionnement")
        elif percentile >= 50:
            st.info("👍 Au-dessus de la médiane")
        elif percentile >= 25:
            st.warning("💡 Potentiel d'amélioration")
        else:
            st.error("⚠️ En dessous du marché")


def _render_profile_comparator(data: pd.DataFrame) -> None:
    """Comparateur de profils détaillé."""
    st.markdown("#### 🆚 Comparateur de profils")
    
    job_options = sorted(data['job_type_simplified'].unique())
    location_options = sorted(data['location_clean'].unique())
    sector_options = sorted(data['sector_clean'].unique())
    
    col1, col2 = st.columns(2)
    
    # Profil A
    with col1:
        st.markdown("**👤 Profil A**")
        prof_a = _render_profile_form("a", job_options, location_options, sector_options)
    
    # Profil B
    with col2:
        st.markdown("**👤 Profil B**")
        prof_b = _render_profile_form("b", job_options, location_options, sector_options)
    
    # Estimation et comparaison
    salary_a = _estimate_profile_salary(data, prof_a)
    salary_b = _estimate_profile_salary(data, prof_b)
    
    _render_comparison_results(salary_a, salary_b, prof_a, prof_b)


def _render_profile_form(
    prefix: str,
    jobs: List[str],
    locations: List[str],
    sectors: List[str]
) -> Dict:
    """Formulaire de saisie d'un profil."""
    idx_offset = 0 if prefix == "a" else 1
    
    job = st.selectbox(
        "Type de poste",
        jobs,
        index=idx_offset,
        key=f"prof_{prefix}_job"
    )
    
    exp = st.slider(
        "Expérience (années)",
        0.0, 15.0,
        3.0 if prefix == "a" else 7.0,
        0.5,
        key=f"prof_{prefix}_exp"
    )
    
    city = st.selectbox(
        "Ville",
        locations,
        index=idx_offset,
        key=f"prof_{prefix}_city"
    )
    
    sector = st.selectbox(
        "Secteur",
        sectors,
        index=idx_offset,
        key=f"prof_{prefix}_sector"
    )
    
    skills = st.multiselect(
        "Compétences",
        ['Python', 'SQL', 'R', 'ML', 'Deep Learning', 'AWS', 'Spark', 'Tableau'],
        default=['Python', 'SQL'] if prefix == "a" else ['Python', 'SQL', 'ML', 'AWS'],
        key=f"prof_{prefix}_skills"
    )
    
    return {
        'job': job,
        'exp': exp,
        'city': city,
        'sector': sector,
        'skills': skills
    }


def _estimate_profile_salary(data: pd.DataFrame, profile: Dict) -> float:
    """Estime le salaire pour un profil donné."""
    mask = (
        (data['job_type_simplified'] == profile['job']) &
        (data['experience_final'].between(profile['exp'] - 1, profile['exp'] + 1)) &
        (data['location_clean'] == profile['city']) &
        (data['sector_clean'] == profile['sector'])
    )
    
    similar_data = data[mask]
    
    if len(similar_data) >= 3:
        base_salary = similar_data['salary_mid'].median()
    else:
        job_data = data[data['job_type_simplified'] == profile['job']]
        base_salary = job_data['salary_mid'].median() if len(job_data) > 0 else data['salary_mid'].median()
        
        city_mult = Config.get_city_multiplier(profile['city'])
        sector_mult = Config.get_sector_multiplier(profile['sector'])
        base_salary *= city_mult * sector_mult
    
    skill_bonus = len(profile['skills']) * 1000
    
    return base_salary + skill_bonus


def _render_comparison_results(
    salary_a: float,
    salary_b: float,
    prof_a: Dict,
    prof_b: Dict
) -> None:
    """Affiche les résultats de la comparaison."""
    st.markdown("---")
    st.markdown("#### 📊 Résultats de la comparaison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("💰 Profil A", f"{salary_a:,.0f}€")
        st.caption(f"Skills : {len(prof_a['skills'])} | Exp : {prof_a['exp']:.0f} ans")
    
    with col2:
        diff = salary_b - salary_a
        st.metric("📊 Écart", f"{diff:+,.0f}€", delta_color="off")
        
        if abs(diff) < 2000:
            st.info("≈ Profils équivalents")
        elif diff > 0:
            st.success("Profil B mieux payé")
        else:
            st.warning("Profil A mieux payé")
    
    with col3:
        st.metric("💰 Profil B", f"{salary_b:,.0f}€")
        st.caption(f"Skills : {len(prof_b['skills'])} | Exp : {prof_b['exp']:.0f} ans")
    
    # Radar chart
    _render_radar_comparison(prof_a, prof_b)


def _render_radar_comparison(prof_a: Dict, prof_b: Dict) -> None:
    """Radar chart de comparaison des profils."""
    categories = ['Expérience', 'Ville (mult.)', 'Secteur (mult.)', 'Compétences']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=[
            prof_a['exp'] / 15 * 100,
            Config.get_city_multiplier(prof_a['city']) * 100,
            Config.get_sector_multiplier(prof_a['sector']) * 100,
            len(prof_a['skills']) / 8 * 100
        ],
        theta=categories,
        fill='toself',
        name='Profil A'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=[
            prof_b['exp'] / 15 * 100,
            Config.get_city_multiplier(prof_b['city']) * 100,
            Config.get_sector_multiplier(prof_b['sector']) * 100,
            len(prof_b['skills']) / 8 * 100
        ],
        theta=categories,
        fill='toself',
        name='Profil B'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="Comparaison des profils (normalisé sur 100)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'render_analysis_tabs'
]
