"""
PAGE D'ACCUEIL – PRÉDICTEUR DE SALAIRES DATA JOBS

Application Streamlit pour l'estimation de salaires dans les métiers de la Data.
Basé sur 5 868 offres HelloWork collectées en janvier 2026.
Modèle : XGBoost v7 optimisé.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import hashlib
from typing import Dict, Optional, Tuple

from utils.config import Config, init_session_state, setup_page
from utils.model_utils import init_model_utils


# ============================================================================
# CONFIGURATION INITIALE
# ============================================================================

def initialize_app() -> Tuple[Config, object]:
    """
    Initialize l'application Streamlit et ses composants.
    
    Returns:
        Tuple[Config, object]: Configuration et utilitaires du modèle
    """
    setup_page()
    init_session_state()
    config = Config()
    model_utils = init_model_utils()
    
    return config, model_utils


# ============================================================================
# CHARGEMENT DES DONNÉES
# ============================================================================

@st.cache_data
def load_application_data() -> Dict[str, Optional[pd.DataFrame]]:
    """
    Charge l'ensemble des données nécessaires à l'application.
    
    Returns:
        Dict contenant :
            - dataset: DataFrame des offres d'emploi
            - report: Dict du rapport de modélisation
            - test_salaries: Array des salaires de test
            
    Notes:
        Utilise st.cache_data pour optimiser les performances.
        Gère les erreurs de chargement de manière gracieuse.
    """
    data = {
        'dataset': None,
        'report': None,
        'test_salaries': None
    }
    
    # Chargement du dataset principal
    data['dataset'] = _load_main_dataset()
    
    # Chargement du rapport de modélisation
    data['report'] = _load_modeling_report()
    
    # Chargement des données de test
    data['test_salaries'] = _load_test_data()
    
    return data


def _load_main_dataset() -> Optional[pd.DataFrame]:
    """
    Charge le dataset nettoyé des offres d'emploi.
    
    Returns:
        DataFrame ou None si erreur de chargement
    """
    data_path = Config.DATA_PATH
    
    if not data_path.exists():
        st.warning(f"⚠️ Dataset introuvable : {data_path}")
        return None
    
    try:
        df = pd.read_csv(
            data_path,
            encoding='utf-8',
            usecols=[
                'job_type_with_desc',
                'seniority',
                'salary_mid',
                'location_final',
                'sector_clean',
                'experience_final'
            ]
        )
        return df
        
    except Exception as e:
        st.error(f"❌ Erreur chargement dataset : {str(e)}")
        return None


def _load_modeling_report() -> Optional[Dict]:
    """
    Charge le rapport de modélisation JSON.
    
    Returns:
        Dict du rapport ou None si erreur
    """
    report_path = Config.REPORT_PATH
    
    if not report_path.exists():
        return None
    
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"⚠️ Impossible de charger le rapport : {str(e)}")
        return None


def _load_test_data() -> Optional[np.ndarray]:
    """
    Charge les données de test pour les visualisations.
    
    Returns:
        Array numpy des salaires de test ou None
    """
    test_path = (
        Config.BASE_DIR / "output" / "analysis_complete" / 
        "modeling_v7_improved" / "models" / "test_data.pkl"
    )
    
    if not test_path.exists():
        return None
    
    try:
        import pickle
        with open(test_path, 'rb') as f:
            test_data = pickle.load(f)
            return test_data.get('y_test')
    except Exception as e:
        st.warning(f"⚠️ Données de test non disponibles : {str(e)}")
        return None


# ============================================================================
# INTERFACE SIDEBAR
# ============================================================================

def render_sidebar(data: Dict, config: Config) -> None:
    """
    Affiche la barre latérale avec informations et navigation.
    
    Args:
        data: Dictionnaire des données chargées
        config: Configuration de l'application
        
    FIX: Ajout de clés uniques aux boutons
    """
    with st.sidebar:
        _render_sidebar_header()
        st.markdown("---")
        
        _render_model_info(data, config)
        st.markdown("---")
        
        _render_daily_insight()
        st.markdown("---")
        
        _render_sidebar_actions(data)


def _render_sidebar_header() -> None:
    """Affiche l'en-tête de la sidebar."""
    st.image(
        "https://img.icons8.com/color/96/000000/data-configuration.png",
        width=96
    )


def _render_model_info(data: Dict, config: Config) -> None:
    """
    Affiche les informations du modèle.
    
    Args:
        data: Données de l'application
        config: Configuration
    """
    st.markdown("### 📊 À propos")
    
    if data.get('report'):
        perf = data['report'].get('performance_metrics', {})
        st.info(f"""
        **Modèle entraîné sur 2 681 offres Data**
        
        • **Algorithme** : XGBoost (v7)
        • **R²** : {perf.get('test_r2', 0.337):.1%}
        • **MAE** : {perf.get('test_mae', 5163):,.0f} €
        • **Stabilité** : {perf.get('model_stability', 0.995):.1%}
        """)
    else:
        st.info(f"""
        **Base : 5 868 offres HelloWork**
        
        • **Modèle** : XGBoost optimisé
        • **R²** : {config.MODEL_INFO['r2_score']:.3f}
        • **MAE** : {config.MODEL_INFO['mae']:,.0f} €
        • **Précision** : {config.MODEL_INFO['precision_15']:.0f}% (±15%)
        """)


def _render_daily_insight() -> None:
    """Affiche l'insight du jour basé sur la date."""
    insights = [
        "Paris représente **36.9%** des offres (+20% de salaire)",
        "Le secteur **Banque** paie **25%** de plus que la moyenne",
        "**Python** présent dans **22.3%** des offres",
        "Data Scientist : **52 920 €** en moyenne",
        "Télétravail : **+5 000 €** en moyenne",
        "Expérience 5–8 ans : salaire médian **50 000 €**",
        "Les compétences ML/AI augmentent le salaire de **15%**"
    ]
    
    # Sélection déterministe basée sur la date
    today = pd.Timestamp.now().date().strftime('%Y-%m-%d')
    idx = int(hashlib.sha256(today.encode()).hexdigest(), 16) % len(insights)
    
    st.success(f"💡 **Insight du jour**\n\n{insights[idx]}")


def _render_sidebar_actions(data: Dict) -> None:
    """
    Affiche les actions disponibles dans la sidebar.
    
    Args:
        data: Données de l'application
        
    FIX: Ajout de clé unique au bouton
    """
    # FIX: Ajout de key unique
    if st.button("📄 Rapport complet", use_container_width=True, key="sidebar_btn_report"):
        if data.get('report'):
            with st.expander("📊 Rapport de modélisation", expanded=True):
                st.json(data['report'])
        else:
            st.warning("Rapport non disponible")


# ============================================================================
# HEADER PRINCIPAL
# ============================================================================

def render_hero_section(config: Config) -> None:
    """
    Affiche la section hero avec le CTA principal.
    
    Args:
        config: Configuration de l'application
    
    Note:
        Le titre principal est géré par setup_page() dans utils/config.py
        pour éviter la duplication.
        
    FIX: Ajout de clé unique au bouton principal
    """
    # CTA principal
    st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # FIX: Ajout de key unique
        if st.button(
            "🚀 Obtenir une estimation salariale",
            type="primary",
            use_container_width=True,
            help="Accédez au formulaire de prédiction personnalisée",
            key="hero_btn_prediction"
        ):
            st.switch_page("pages/01_Prediction.py")


# ============================================================================
# MÉTRIQUES CLÉS
# ============================================================================

def render_key_metrics(data: Dict, config: Config) -> None:
    """
    Affiche les métriques clés du dataset et du modèle.
    
    Args:
        data: Données de l'application
        config: Configuration
    """
    st.markdown("### 📊 Synthèse du marché Data")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Métrique 1 : Nombre d'offres
    with col1:
        st.metric(
            label="Offres analysées",
            value="5 868",
            help="Total des offres collectées en janvier 2026"
        )
    
    # Métrique 2 : Précision
    with col2:
        st.metric(
            label="Précision ±15%",
            value=f"{config.MODEL_INFO['precision_15']:.0f}%",
            help="Pourcentage de prédictions dans ±15% de la valeur réelle"
        )
    
    # Métrique 3 : Salaire médian
    with col3:
        median_salary = (
            data['dataset']['salary_mid'].median()
            if data.get('dataset') is not None
            else config.MARKET_MEDIAN
        )
        st.metric(
            label="Salaire médian",
            value=f"{median_salary:,.0f} €",
            help="Salaire annuel brut médian pour les postes Data"
        )
    
    # Métrique 4 : Performance modèle
    with col4:
        st.metric(
            label="R² Score",
            value=f"{config.MODEL_INFO['r2_score']:.3f}",
            help=f"Coefficient de détermination • MAE : {config.MODEL_INFO['mae']:,.0f} €"
        )


# ============================================================================
# MÉTHODOLOGIE
# ============================================================================

def render_methodology_section() -> None:
    """Affiche la section méthodologie avec les étapes clés."""
    st.markdown("### 🔍 Méthodologie")
    
    process_steps = [
        {
            'icon': '📥',
            'title': 'Collecte',
            'description': '5 868 offres HelloWork',
            'details': 'Scraping automatisé + nettoyage'
        },
        {
            'icon': '🔧',
            'title': 'Feature Engineering',
            'description': '29 variables extraites',
            'details': 'NLP + encodage + normalisation'
        },
        {
            'icon': '🤖',
            'title': 'Modélisation',
            'description': '7 algorithmes comparés',
            'details': 'XGBoost sélectionné (meilleur R²)'
        },
        {
            'icon': '✅',
            'title': 'Validation',
            'description': '2 681 échantillons Data',
            'details': 'Test set + cross-validation'
        }
    ]
    
    cols = st.columns(4)
    
    for col, step in zip(cols, process_steps):
        with col:
            st.markdown(f"""
            <div style='
                text-align: center;
                padding: 25px 15px;
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                border-radius: 12px;
                border-left: 4px solid #1f77b4;
                height: 180px;
                display: flex;
                flex-direction: column;
                justify-content: center;
            '>
                <div style='font-size: 36px; margin-bottom: 10px;'>{step['icon']}</div>
                <h4 style='margin: 10px 0; color: #1f77b4;'>{step['title']}</h4>
                <p style='font-size: 14px; color: #666; margin: 5px 0;'>{step['description']}</p>
                <p style='font-size: 12px; color: #999; margin-top: 8px;'>{step['details']}</p>
            </div>
            """, unsafe_allow_html=True)


# ============================================================================
# VISUALISATIONS
# ============================================================================

def render_salary_distribution(salaries: np.ndarray) -> None:
    """
    Affiche la distribution des salaires.
    
    Args:
        salaries: Array des salaires de test
    """
    st.markdown("### 📈 Distribution des salaires")
    
    # Calcul des statistiques
    mean_sal = np.mean(salaries)
    median_sal = np.median(salaries)
    std_sal = np.std(salaries)
    q1, q3 = np.percentile(salaries, [25, 75])
    
    col1, col2 = st.columns([2.5, 1])
    
    # Graphique
    with col1:
        fig = go.Figure()
        
        # Histogramme
        fig.add_trace(go.Histogram(
            x=salaries,
            nbinsx=35,
            marker_color='steelblue',
            opacity=0.75,
            name='Distribution'
        ))
        
        # Ligne médiane
        fig.add_vline(
            x=median_sal,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text=f"Médiane : {median_sal:,.0f} €",
            annotation_position="top"
        )
        
        # Ligne moyenne
        fig.add_vline(
            x=mean_sal,
            line_dash="dot",
            line_color="green",
            line_width=2,
            annotation_text=f"Moyenne : {mean_sal:,.0f} €",
            annotation_position="bottom"
        )
        
        fig.update_layout(
            title={
                'text': "Distribution salariale (échantillon test)",
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title="Salaire annuel brut (€)",
            yaxis_title="Nombre d'offres",
            height=450,
            plot_bgcolor='white',
            showlegend=False,
            hovermode='x unified'
        )
        
        fig.update_xaxes(gridcolor='lightgray')
        fig.update_yaxes(gridcolor='lightgray')
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistiques
    with col2:
        st.markdown("#### 📊 Statistiques")
        
        st.metric("Moyenne", f"{mean_sal:,.0f} €")
        st.metric("Médiane", f"{median_sal:,.0f} €")
        st.metric("Écart-type", f"{std_sal:,.0f} €")
        
        st.markdown("---")
        st.markdown("**Quartiles**")
        st.write(f"• **Q1** (25%) : {q1:,.0f} €")
        st.write(f"• **Q3** (75%) : {q3:,.0f} €")
        st.write(f"• **IQR** : {q3-q1:,.0f} €")
        
        st.markdown("---")
        st.markdown("**Plage**")
        st.write(f"• **Min** : {np.min(salaries):,.0f} €")
        st.write(f"• **Max** : {np.max(salaries):,.0f} €")


def render_top_jobs(dataset: pd.DataFrame) -> None:
    """
    Affiche les postes les plus fréquents.
    
    Args:
        dataset: DataFrame des offres
    """
    st.markdown("### 💼 Top 10 des postes Data")
    
    job_counts = dataset['job_type_with_desc'].value_counts().head(10)
    
    # Calcul des salaires moyens par poste
    avg_salaries = (
        dataset.groupby('job_type_with_desc')['salary_mid']
        .mean()
        .reindex(job_counts.index)
    )
    
    # Création du graphique
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=job_counts.index,
        x=job_counts.values,
        orientation='h',
        marker_color='steelblue',
        text=job_counts.values,
        textposition='outside',
        hovertemplate=(
            '<b>%{y}</b><br>' +
            'Offres: %{x}<br>' +
            'Salaire moyen: %{customdata:,.0f} €<br>' +
            '<extra></extra>'
        ),
        customdata=avg_salaries.values
    ))
    
    fig.update_layout(
        title={
            'text': "Nombre d'offres par type de poste",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="Nombre d'offres",
        yaxis_title="",
        height=450,
        yaxis={'autorange': 'reversed'},
        plot_bgcolor='white',
        showlegend=False
    )
    
    fig.update_xaxes(gridcolor='lightgray')
    
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# NAVIGATION PRINCIPALE
# ============================================================================

def render_navigation_cards() -> None:
    """
    Affiche les cartes de navigation principale.
    
    FIX: Ajout de clés uniques à tous les boutons de navigation
    """
    st.markdown("""
    <div style='
        text-align: center;
        padding: 40px 20px;
        background: linear-gradient(135deg, #1f77b4 0%, #0d5a9e 100%);
        border-radius: 15px;
        color: white;
        margin: 30px 0;
    '>
        <h2 style='color: white; margin-bottom: 15px;'>
            🗺️ Explorer le marché Data
        </h2>
        <p style='font-size: 18px; margin-bottom: 10px;'>
            Analyse approfondie • Estimation personnalisée • Insights carrière
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Cards de navigation
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='text-align: center; padding: 15px;'>
            <h3>🔮 Estimation</h3>
            <p style='font-size: 14px; color: #666;'>
                Obtenez une estimation précise de votre salaire
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # FIX: Ajout de key unique
        if st.button("Accéder", key="nav_btn_prediction", use_container_width=True):
            st.switch_page("pages/01_Prediction.py")
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 15px;'>
            <h3>📊 Marché</h3>
            <p style='font-size: 14px; color: #666;'>
                Analysez les tendances du marché Data
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # FIX: Ajout de key unique
        if st.button("Accéder", key="nav_btn_market", use_container_width=True):
            st.switch_page("pages/02_Marche.py")
    
    with col3:
        st.markdown("""
        <div style='text-align: center; padding: 15px;'>
            <h3>🎓 Carrière</h3>
            <p style='font-size: 14px; color: #666;'>
                Planifiez votre évolution professionnelle
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # FIX: Ajout de key unique
        if st.button("Accéder", key="nav_btn_career", use_container_width=True):
            st.switch_page("pages/03_Carriere.py")


# ============================================================================
# FOOTER
# ============================================================================

def render_footer() -> None:
    """Affiche le footer de l'application."""
    st.markdown("---")
    st.markdown("""
    <div style='
        text-align: center;
        color: #666;
        padding: 30px 0;
        font-size: 14px;
    '>
        <p style='margin-bottom: 10px;'>
            <strong>© 2026 Prédicteur de Salaires Data Jobs</strong>
        </p>
        <p style='font-size: 12px; color: #999;'>
            Données : HelloWork (janvier 2026) • 
            Modèle : XGBoost v7 • 
            Développé avec ❤️ et Python
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """
    Fonction principale de l'application.
    
    Orchestre le rendu de tous les composants de la page d'accueil.
    """
    # Initialisation
    config, model_utils = initialize_app()
    
    # Chargement des données
    data = load_application_data()
    
    # Sidebar
    render_sidebar(data, config)
    
    # Header
    render_hero_section(config)
    st.markdown("---")
    
    # Métriques clés
    render_key_metrics(data, config)
    st.markdown("---")
    
    # Méthodologie
    render_methodology_section()
    st.markdown("---")
    
    # Visualisations
    if data.get('test_salaries') is not None:
        render_salary_distribution(data['test_salaries'])
        st.markdown("---")
    else:
        st.warning("⚠️ Données salariales non disponibles pour les visualisations")
    
    if data.get('dataset') is not None:
        render_top_jobs(data['dataset'])
        st.markdown("---")
    
    # Navigation
    render_navigation_cards()
    
    # Footer
    render_footer()


if __name__ == "__main__":
    main()