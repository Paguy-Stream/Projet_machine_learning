"""
Module d'export et navigation pour la page Marché.

Ce module contient les fonctions pour :
- Export des données filtrées (CSV)
- Export des statistiques (JSON)
- Navigation entre les pages
- Boutons d'aide et actualisation

"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List

from utils.config import Config


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def render_export_and_navigation(
    filtered_data: pd.DataFrame,
    total_size: int,
    filters_info: Dict[str, int]
) -> None:
    """
    Affiche les options d'export et de navigation.
    
    Args:
        filtered_data: Données après filtres
        total_size: Nombre total d'offres
        filters_info: Info sur les filtres actifs
    """
    st.markdown("### 📥 Export des données")
    
    col1, col2 = st.columns(2)
    
    with col1:
        _render_csv_export(filtered_data)
    
    with col2:
        _render_json_export(
            filtered_data,
            total_size,
            filters_info
        )
    
    st.markdown("---")
    
    # Navigation
    _render_navigation_buttons()
    
    # Footer
    _render_footer(total_size)


# ============================================================================
# EXPORT CSV
# ============================================================================

def _render_csv_export(filtered_data: pd.DataFrame) -> None:
    """
    Bouton d'export CSV des données filtrées.
    
    Args:
        filtered_data: Données à exporter
    """
    export_cols = [
        'job_type_simplified', 'seniority', 'salary_mid',
        'location_clean', 'sector_clean', 'experience_final',
        'contract_type_clean', 'telework_numeric',
        'skills_count', 'technical_score'
    ]
    
    # Filtrer les colonnes existantes
    available_cols = [col for col in export_cols if col in filtered_data.columns]
    
    export_data = filtered_data[available_cols].copy()
    csv = export_data.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        "📥 Télécharger données filtrées (CSV)",
        data=csv,
        file_name=f"marche_data_jobs_{len(filtered_data)}_offres.csv",
        mime="text/csv",
        use_container_width=True,
        help="Exporte les données filtrées au format CSV"
    )


# ============================================================================
# EXPORT JSON
# ============================================================================

def _render_json_export(
    filtered_data: pd.DataFrame,
    total_size: int,
    filters_info: Dict[str, int]
) -> None:
    """
    Bouton d'export JSON avec statistiques complètes.
    
    Args:
        filtered_data: Données filtrées
        total_size: Nombre total d'offres
        filters_info: Info sur les filtres
    """
    export_json = _prepare_json_export(
        filtered_data,
        total_size,
        filters_info
    )
    
    json_str = json.dumps(export_json, indent=2, ensure_ascii=False)
    
    st.download_button(
        "📥 Télécharger statistiques (JSON)",
        data=json_str,
        file_name=f"stats_marche_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True,
        help="Exporte les statistiques et multiplicateurs au format JSON"
    )


def _prepare_json_export(
    filtered_data: pd.DataFrame,
    total_size: int,
    filters_info: Dict[str, int]
) -> Dict:
    """
    Prépare les données JSON pour l'export.
    
    Args:
        filtered_data: Données filtrées
        total_size: Nombre total d'offres
        filters_info: Info sur les filtres
        
    Returns:
        Dict complet pour export JSON
    """
    # Récupérer les filtres actifs depuis session_state
    job_filter = st.session_state.get('job_filter', [])
    location_filter = st.session_state.get('location_filter', [])
    sector_filter = st.session_state.get('sector_filter', [])
    
    # Statistiques principales
    stats = {
        'salaire_median': float(filtered_data['salary_mid'].median()),
        'salaire_moyen': float(filtered_data['salary_mid'].mean()),
        'salaire_q25': float(filtered_data['salary_mid'].quantile(0.25)),
        'salaire_q75': float(filtered_data['salary_mid'].quantile(0.75)),
        'salaire_min': float(filtered_data['salary_mid'].min()),
        'salaire_max': float(filtered_data['salary_mid'].max())
    }
    
    # Multiplicateurs dynamiques
    all_city_mults = Config.get_all_city_multipliers()
    all_sector_mults = Config.get_all_sector_multipliers()
    
    multiplicateurs = {
        'villes': {
            k: float(v)
            for k, v in all_city_mults.items()
            if k in location_filter
        },
        'secteurs': {
            k: float(v)
            for k, v in all_sector_mults.items()
            if k in sector_filter
        }
    }
    
    # Construction du JSON complet
    return {
        'metadata': {
            'date_export': datetime.now().isoformat(),
            'app_version': '2.0',
            'total_offres': total_size,
            'offres_filtrees': len(filtered_data),
            'filtres_appliques': {
                'postes': job_filter,
                'villes': location_filter,
                'secteurs': sector_filter
            }
        },
        'statistiques': stats,
        'multiplicateurs_dynamiques': multiplicateurs,
        'competences_impact': _calculate_skills_impact_for_export(filtered_data),
        'top_combinaisons': _calculate_top_stacks_for_export(filtered_data)
    }


def _calculate_skills_impact_for_export(data: pd.DataFrame) -> Dict[str, float]:
    """Calcule l'impact des compétences pour l'export."""
    skills_mapping = {
        'Python': 'contient_python',
        'SQL': 'contient_sql',
        'ML': 'contient_machine_learning',
        'AWS': 'contient_aws',
        'Spark': 'contient_spark'
    }
    
    impacts = {}
    
    for skill_name, col_name in skills_mapping.items():
        if col_name not in data.columns or data[col_name].sum() < 5:
            continue
        
        with_skill = data[data[col_name] == 1]['salary_mid'].median()
        without_skill = data[data[col_name] == 0]['salary_mid'].median()
        impact = with_skill - without_skill
        
        if not pd.isna(impact):
            impacts[skill_name] = float(impact)
    
    return impacts


def _calculate_top_stacks_for_export(data: pd.DataFrame) -> List[Dict]:
    """Calcule les top stacks pour l'export."""
    stacks = {
        'Data Scientist': (
            (data['contient_python'] == 1) &
            (data['contient_machine_learning'] == 1) &
            ((data['contient_aws'] == 1) | (data['contient_azure'] == 1))
        ),
        'Data Engineer': (
            (data['contient_python'] == 1) &
            (data['contient_spark'] == 1)
        ),
        'BI Analyst': (
            (data['contient_sql'] == 1) &
            ((data['contient_tableau'] == 1) | (data['contient_power_bi'] == 1))
        )
    }
    
    results = []
    
    for stack_name, mask in stacks.items():
        count = mask.sum()
        if count >= 5:
            results.append({
                'stack': stack_name,
                'median_salary': float(data[mask]['salary_mid'].median()),
                'count': int(count)
            })
    
    return sorted(results, key=lambda x: x['median_salary'], reverse=True)


# ============================================================================
# NAVIGATION
# ============================================================================

def _render_navigation_buttons() -> None:
    """Affiche les boutons de navigation entre pages."""
    st.markdown("### 🧭 Navigation")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🏠 Accueil", use_container_width=True):
            st.switch_page("01_Accueil.py")
    
    with col2:
        if st.button("🔮 Prédiction", use_container_width=True):
            st.switch_page("pages/01_Prediction.py")
    
    with col3:
        if st.button("🔄 Actualiser", use_container_width=True):
            st.cache_data.clear()
            Config.reload_dynamic_data()
            st.rerun()
    
    with col4:
        if st.button("💡 Aide", use_container_width=True):
            _show_help_modal()


def _show_help_modal() -> None:
    """Affiche une modal d'aide."""
    st.info("""
    **📖 Guide d'utilisation** :
    
    1. **Filtres (sidebar)** : Personnalisez votre recherche
       - Types de postes, villes, secteurs
       - Fourchette salariale et expérience
       - Stacks techniques
    
    2. **Insights** : 3 colonnes d'opportunités
       - Top compétences rentables
       - Meilleures villes
       - Secteurs généreux
    
    3. **Onglets d'analyse** :
       -  Vue d'ensemble : distributions, tendances
       -  Postes & Secteurs : comparaisons, multiplicateurs
       -  Géographie : heatmap, top villes
       -  Compétences : fréquence, impact
       -  Combinaisons : stacks gagnants, ROI
       -  Benchmark : positionnement, comparateur
    
    4. **Export** : Téléchargez vos analyses
       - CSV : données brutes filtrées
       - JSON : statistiques complètes + multiplicateurs
    
    ---
    
    💡 **Astuce** : Les multiplicateurs sont **calculés dynamiquement** 
    depuis le dataset pour refléter le marché réel !
    """)


# ============================================================================
# FOOTER
# ============================================================================

def _render_footer(total_size: int) -> None:
    """
    Affiche le footer de la page.
    
    Args:
        total_size: Nombre total d'offres
    """
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #666; padding: 20px 0;'>
        <p>© 2026 Prédicteur de salaires Data Jobs v2.0 • 
        Données : HelloWork ({total_size:,} offres, janvier 2026)</p>
        <p style='font-size: 12px;'>
        ✅ Multiplicateurs s • Insights • Benchmark personnel
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'render_export_and_navigation'
]
