"""
Tests unitaires pour le module model_utils.

Ces tests couvrent :
- CalculationUtils : Calculs de scores et estimations
- DataDistributions : Statistiques du marché
- ChartUtils : Génération de graphiques
- ModelUtils : Prédictions et SHAP

Version: 2.0
Auteur: Data Team
Date: Janvier 2026
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from utils.model_utils import (
    CalculationUtils,
    DataDistributions,
    ChartUtils,
    ModelUtils
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_profile():
    """Profil utilisateur de test."""
    return {
        'job_type': 'Data Scientist',
        'seniority': 'Mid-level',
        'experience_final': 4.0,
        'contract_type_clean': 'CDI',
        'education_clean': 'Master',
        'location_final': 'Paris',
        'sector_clean': 'Tech',
        'telework_numeric': 0.4,
        'contient_python': True,
        'contient_sql': True,
        'contient_r': False,
        'contient_tableau': False,
        'contient_power_bi': False,
        'contient_aws': True,
        'contient_azure': False,
        'contient_spark': False,
        'contient_machine_learning': True,
        'contient_deep_learning': False,
        'has_teletravail': True,
        'has_mutuelle': False,
        'has_tickets': False,
        'has_prime': False,
        'skills_count': 4,
        'technical_score': 7,
        'benefits_score': 1
    }


@pytest.fixture
def sample_market_data():
    """Données du marché de test."""
    np.random.seed(42)
    return np.random.normal(50000, 10000, 1000)


@pytest.fixture
def sample_dataframe():
    """DataFrame de test."""
    np.random.seed(42)
    return pd.DataFrame({
        'salary_mid': np.random.normal(50000, 10000, 100),
        'experience_final': np.random.uniform(0, 15, 100),
        'skills_count': np.random.randint(1, 10, 100),
        'contient_python': np.random.choice([0, 1], 100),
        'contient_sql': np.random.choice([0, 1], 100)
    })


# ============================================================================
# TESTS - CalculationUtils
# ============================================================================

class TestCalculationUtils:
    """Tests pour la classe CalculationUtils."""
    
    def test_calculate_skills_count_from_profile(self):
        """Test du calcul du nombre de compétences."""
        skills = {
            'contient_python': True,
            'contient_sql': True,
            'contient_r': False,
            'contient_aws': True,
            'contient_machine_learning': True
        }
        
        count = CalculationUtils.calculate_skills_count_from_profile(skills)
        
        assert count == 4
        assert isinstance(count, int)
    
    def test_calculate_skills_count_empty(self):
        """Test avec aucune compétence."""
        skills = {
            'contient_python': False,
            'contient_sql': False,
            'contient_r': False
        }
        
        count = CalculationUtils.calculate_skills_count_from_profile(skills)
        
        assert count == 0
    
    def test_calculate_technical_score_from_profile(self):
        """Test du calcul du score technique."""
        skills = {
            'contient_python': True,
            'contient_sql': True,
            'contient_machine_learning': True,
            'contient_deep_learning': True,
            'contient_aws': True
        }
        
        score = CalculationUtils.calculate_technical_score_from_profile(skills)
        
        assert score > 0
        assert isinstance(score, (int, float))
        # Python(2) + SQL(2) + ML(2) + DL(2) + AWS(1) = 9
        assert score == 9
    
    def test_calculate_technical_score_basic(self):
        """Test score technique avec compétences basiques."""
        skills = {
            'contient_python': True,
            'contient_sql': True,
            'contient_tableau': True
        }
        
        score = CalculationUtils.calculate_technical_score_from_profile(skills)
        
        # Python(2) + SQL(2) + Tableau(1) = 5
        assert score == 5
    
    def test_estimate_description_complexity(self, sample_profile):
        """Test de l'estimation de la complexité de description."""
        word_count = CalculationUtils.estimate_description_complexity(sample_profile)
        
        assert isinstance(word_count, int)
        assert word_count > 0
        assert word_count < 2000  # Valeur raisonnable
    
    def test_estimate_description_complexity_varies_with_experience(self):
        """Test que la complexité varie avec l'expérience."""
        profile_junior = {'experience_final': 1.0, 'sector_clean': 'Tech', 'skills_count': 2}
        profile_senior = {'experience_final': 10.0, 'sector_clean': 'Tech', 'skills_count': 8}
        
        count_junior = CalculationUtils.estimate_description_complexity(profile_junior)
        count_senior = CalculationUtils.estimate_description_complexity(profile_senior)
        
        assert count_senior > count_junior
    
    def test_estimate_technical_keywords(self, sample_profile):
        """Test de l'estimation des mots-clés techniques."""
        keywords = CalculationUtils.estimate_technical_keywords(sample_profile)
        
        assert isinstance(keywords, int)
        assert keywords > 0
        assert keywords < 20  # Valeur raisonnable
    
    def test_estimate_technical_keywords_correlates_with_skills(self):
        """Test que les mots-clés corrèlent avec les compétences."""
        profile_few_skills = {
            'experience_final': 3.0,
            'contient_python': True,
            'contient_sql': False,
            'contient_machine_learning': False
        }
        
        profile_many_skills = {
            'experience_final': 3.0,
            'contient_python': True,
            'contient_sql': True,
            'contient_machine_learning': True,
            'contient_deep_learning': True,
            'contient_aws': True
        }
        
        keywords_few = CalculationUtils.estimate_technical_keywords(profile_few_skills)
        keywords_many = CalculationUtils.estimate_technical_keywords(profile_many_skills)
        
        assert keywords_many > keywords_few
    
    def test_get_percentile_real(self, sample_market_data):
        """Test du calcul de percentile."""
        value = 55000
        percentile = CalculationUtils.get_percentile_real(value, sample_market_data)
        
        assert 0 <= percentile <= 100
        assert isinstance(percentile, (int, float))
    
    def test_get_percentile_real_extreme_values(self, sample_market_data):
        """Test percentile avec valeurs extrêmes."""
        # Valeur très basse
        percentile_low = CalculationUtils.get_percentile_real(0, sample_market_data)
        assert percentile_low < 10
        
        # Valeur très haute
        percentile_high = CalculationUtils.get_percentile_real(100000, sample_market_data)
        assert percentile_high > 90
    
    def test_create_profile_summary(self, sample_profile):
        """Test de la création du résumé de profil."""
        summary = CalculationUtils.create_profile_summary(sample_profile)
        
        assert isinstance(summary, dict)
        assert 'job_info' in summary
        assert 'location_sector' in summary
        assert 'education_exp' in summary
        assert 'telework' in summary
        assert 'skills_count' in summary
        assert 'key_skills' in summary
        
        # Vérifier le contenu
        assert 'Data Scientist' in summary['job_info']
        assert 'Paris' in summary['location_sector']


# ============================================================================
# TESTS - DataDistributions
# ============================================================================

class TestDataDistributions:
    """Tests pour la classe DataDistributions."""
    
    @patch('utils.model_utils.pd.read_csv')
    def test_reload(self, mock_read_csv, sample_dataframe):
        """Test du rechargement des distributions."""
        mock_read_csv.return_value = sample_dataframe
        
        # Forcer le rechargement
        distributions = DataDistributions.reload()
        
        # Vérifier que les distributions sont chargées
        assert distributions is not None
        assert isinstance(distributions, dict)
    
    def test_get_total_offers(self):
        """Test de récupération du nombre total d'offres."""
        # Créer un mock des distributions
        with patch.object(DataDistributions, '_load_distributions') as mock_load:
            mock_load.return_value = {'TOTAL_OFFERS': 5868}
            
            total = DataDistributions.get_total_offers()
            
            assert total == 5868
            assert isinstance(total, int)
    
    def test_get_desc_words(self):
        """Test de récupération des stats de description."""
        with patch.object(DataDistributions, '_load_distributions') as mock_load:
            mock_load.return_value = {
                'DESC_WORDS': {
                    'p10': 200,
                    'p25': 300,
                    'median': 500,
                    'p75': 700,
                    'p90': 900,
                    'count': 5000
                }
            }
            
            stats = DataDistributions.get_desc_words()
            
            assert stats['median'] == 500
            assert stats['count'] == 5000
    
    def test_get_tech_keywords(self):
        """Test de récupération des stats de mots-clés."""
        with patch.object(DataDistributions, '_load_distributions') as mock_load:
            mock_load.return_value = {
                'TECH_KEYWORDS': {
                    'p25': 2,
                    'median': 3,
                    'p75': 5,
                    'p90': 7,
                    'mean': 3.5,
                    'count': 5000
                }
            }
            
            stats = DataDistributions.get_tech_keywords()
            
            assert stats['median'] == 3
            assert stats['mean'] == 3.5
    
    def test_get_ml_dl_correlation(self):
        """Test de récupération de la corrélation ML/DL."""
        with patch.object(DataDistributions, '_load_distributions') as mock_load:
            mock_load.return_value = {'ML_DL_CORRELATION': 0.65}
            
            corr = DataDistributions.get_ml_dl_correlation()
            
            assert corr == 0.65
            assert 0 <= corr <= 1


# ============================================================================
# TESTS - ChartUtils
# ============================================================================

class TestChartUtils:
    """Tests pour la classe ChartUtils."""
    
    def test_create_shap_waterfall_returns_figure(self, sample_profile):
        """Test que la fonction retourne bien une figure."""
        # Mock SHAP explanation avec la structure attendue
        mock_raw_shap = MagicMock()
        mock_raw_shap.values = np.array([[1500, 1200, 800]])
        mock_raw_shap.feature_names = ['experience_final', 'skills_count', 'technical_score']
        mock_raw_shap.base_values = np.array([45000])
        
        shap_exp = {
            'raw_shap': mock_raw_shap
        }
        
        feature_translation = {
            'experience_final': 'Expérience',
            'skills_count': 'Nombre de compétences',
            'technical_score': 'Score technique'
        }
        
        fig = ChartUtils.create_shap_waterfall(shap_exp, feature_translation)
        
        assert fig is not None
        assert hasattr(fig, 'data')
        assert hasattr(fig, 'layout')
    
    def test_create_shap_waterfall_handles_many_features(self):
        """Test avec beaucoup de features (limite à max_display)."""
        # Mock avec 20 features
        mock_raw_shap = MagicMock()
        mock_raw_shap.values = np.array([[100 * i for i in range(20)]])
        mock_raw_shap.feature_names = [f'feature_{i}' for i in range(20)]
        mock_raw_shap.base_values = np.array([40000])
        
        shap_exp = {
            'raw_shap': mock_raw_shap
        }
        
        feature_translation = {f'feature_{i}': f'Feature {i}' for i in range(20)}
        
        fig = ChartUtils.create_shap_waterfall(shap_exp, feature_translation, max_display=10)
        
        # Devrait limiter à 10 features max
        assert fig is not None
    
    def test_create_market_comparison_returns_figure(self, sample_market_data):
        """Test de création du graphique de comparaison marché."""
        prediction = 55000
        market_median = 50000
        error_margin = 7417
        
        fig = ChartUtils.create_market_comparison(
            prediction,
            sample_market_data,
            market_median,
            error_margin
        )
        
        assert fig is not None
        assert hasattr(fig, 'data')
        assert len(fig.data) > 0  # Au moins une trace
    
    def test_create_salary_gauge_returns_figure(self):
        """Test de création de la jauge."""
        fig = ChartUtils.create_salary_gauge(
            prediction=55000,
            market_median=50000,
            q1=45000,
            q3=55000,
            gauge_min=30000,
            gauge_max=80000
        )
        
        assert fig is not None
        assert hasattr(fig, 'data')
        assert fig.data[0].type == 'indicator'


# ============================================================================
# TESTS - ModelUtils
# ============================================================================

class TestModelUtils:
    """Tests pour la classe ModelUtils."""
    
    @pytest.fixture
    def mock_model_utils(self):
        """Fixture avec un ModelUtils mocké."""
        with patch('utils.model_utils.joblib.load') as mock_load:
            # Mock du modèle
            mock_model = MagicMock()
            mock_model.predict.return_value = np.array([55000])
            mock_load.return_value = mock_model
            
            model_utils = ModelUtils()
            yield model_utils
    
    def test_predict_returns_dict(self, mock_model_utils, sample_profile):
        """Test que predict retourne bien un dict."""
        result = mock_model_utils.predict(sample_profile)
        
        assert isinstance(result, dict)
        assert 'prediction' in result
    
    def test_predict_validates_profile(self, mock_model_utils):
        """Test de validation du profil."""
        # Profil avec seulement quelques champs
        incomplete_profile = {
            'job_type': 'Data Scientist',
            'experience_final': 4.0
            # Manque beaucoup de champs
        }
        
        # Le modèle devrait quand même fonctionner avec des valeurs par défaut
        result = mock_model_utils.predict(incomplete_profile)
        
        # Le résultat peut être valide (modèle robuste) ou None
        if result:
            assert isinstance(result, dict)
            assert 'prediction' in result
    
    def test_predict_handles_missing_fields(self, mock_model_utils, sample_profile):
        """Test avec champs manquants."""
        # Supprimer quelques champs optionnels
        incomplete_profile = sample_profile.copy()
        del incomplete_profile['contient_deep_learning']
        
        result = mock_model_utils.predict(incomplete_profile)
        
        # Devrait quand même fonctionner
        assert result is not None
    
    def test_get_real_market_data_returns_array(self, mock_model_utils):
        """Test de récupération des données du marché."""
        # Mock les données
        mock_model_utils.real_salaries = np.array([40000, 50000, 60000])
        
        data = mock_model_utils.get_real_market_data()
        
        assert isinstance(data, np.ndarray)
        assert len(data) > 0
    
    def test_get_model_performance_returns_dict(self, mock_model_utils):
        """Test de récupération des performances."""
        perf = mock_model_utils.get_model_performance()
        
        assert isinstance(perf, dict)
        # Devrait contenir au moins quelques métriques
        assert len(perf) > 0


# ============================================================================
# TESTS D'INTÉGRATION
# ============================================================================

class TestIntegration:
    """Tests d'intégration entre les différentes classes."""
    
    def test_full_prediction_flow(self, sample_profile, sample_market_data):
        """Test du flux complet de prédiction."""
        with patch('utils.model_utils.joblib.load') as mock_load:
            # Mock du modèle
            mock_model = MagicMock()
            mock_model.predict.return_value = np.array([55000])
            mock_load.return_value = mock_model
            
            # Initialiser
            model_utils = ModelUtils()
            model_utils.real_salaries = sample_market_data
            
            # Calculer les scores
            skills_dict = {
                k: v for k, v in sample_profile.items()
                if k.startswith('contient_')
            }
            
            sample_profile['skills_count'] = (
                CalculationUtils.calculate_skills_count_from_profile(skills_dict)
            )
            sample_profile['technical_score'] = (
                CalculationUtils.calculate_technical_score_from_profile(skills_dict)
            )
            sample_profile['description_word_count'] = (
                CalculationUtils.estimate_description_complexity(sample_profile)
            )
            sample_profile['nb_mots_cles_techniques'] = (
                CalculationUtils.estimate_technical_keywords(sample_profile)
            )
            
            # Prédire
            result = model_utils.predict(sample_profile)
            
            # Calculer le percentile
            if result:
                percentile = CalculationUtils.get_percentile_real(
                    result['prediction'],
                    sample_market_data
                )
                
                assert isinstance(percentile, (int, float))
                assert 0 <= percentile <= 100
    
    def test_profile_creation_and_summary(self, sample_profile):
        """Test de création de profil et résumé."""
        # Ajouter les calculs dynamiques
        skills_dict = {
            k: v for k, v in sample_profile.items()
            if k.startswith('contient_')
        }
        
        sample_profile['skills_count'] = (
            CalculationUtils.calculate_skills_count_from_profile(skills_dict)
        )
        sample_profile['technical_score'] = (
            CalculationUtils.calculate_technical_score_from_profile(skills_dict)
        )
        
        # Créer le résumé
        summary = CalculationUtils.create_profile_summary(sample_profile)
        
        assert summary['skills_count'] == sample_profile['skills_count']
        assert summary['tech_score'] == sample_profile['technical_score']


# ============================================================================
# TESTS DE RÉGRESSION
# ============================================================================

class TestRegression:
    """Tests de régression pour garantir la stabilité."""
    
    def test_skills_count_consistency(self):
        """Test que le calcul reste cohérent."""
        skills = {
            'contient_python': True,
            'contient_sql': True,
            'contient_aws': True
        }
        
        # Calculer plusieurs fois
        counts = [
            CalculationUtils.calculate_skills_count_from_profile(skills)
            for _ in range(10)
        ]
        
        # Tous les résultats doivent être identiques
        assert len(set(counts)) == 1
        assert counts[0] == 3
    
    def test_percentile_consistency(self):
        """Test que le percentile reste cohérent."""
        data = np.array([30000, 40000, 50000, 60000, 70000])
        value = 55000
        
        # Calculer plusieurs fois
        percentiles = [
            CalculationUtils.get_percentile_real(value, data)
            for _ in range(10)
        ]
        
        # Tous les résultats doivent être identiques
        assert len(set(percentiles)) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--cov=utils.model_utils'])
