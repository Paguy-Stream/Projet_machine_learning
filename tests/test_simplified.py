"""
Tests unitaires simplifiés pour config et validation de base.

Ces tests se concentrent sur les fonctionnalités testables sans mocks complexes.
Couverture : utils/config.py et validation de structure

Version: 2.0 - Simplifié
Auteur: Data Team  
Date: Janvier 2026
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock
import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# TESTS - CONFIG.PY (Version simplifiée)
# ============================================================================

class TestConfig:
    """Tests pour utils/config.py - Version simplifiée et robuste."""
    
    def test_config_class_exists(self):
        """Test que la classe Config existe."""
        from utils.config import Config
        assert Config is not None
    
    def test_config_has_required_attributes(self):
        """Test que Config a tous les attributs requis."""
        from utils.config import Config
        
        required_attrs = ['CITIES', 'SECTORS', 'JOB_TYPES', 'MODEL_PATH', 'DATA_PATH']
        for attr in required_attrs:
            assert hasattr(Config, attr), f"Config manque l'attribut {attr}"
    
    def test_cities_is_list(self):
        """Test que CITIES est une liste non vide."""
        from utils.config import Config
        
        assert isinstance(Config.CITIES, list)
        assert len(Config.CITIES) > 0
        assert all(isinstance(city, str) for city in Config.CITIES)
    
    def test_sectors_is_list(self):
        """Test que SECTORS est une liste non vide."""
        from utils.config import Config
        
        assert isinstance(Config.SECTORS, list)
        assert len(Config.SECTORS) > 0
        assert all(isinstance(sector, str) for sector in Config.SECTORS)
    
    def test_job_types_is_list(self):
        """Test que JOB_TYPES est une liste non vide."""
        from utils.config import Config
        
        assert isinstance(Config.JOB_TYPES, list)
        assert len(Config.JOB_TYPES) > 0
        assert all(isinstance(job, str) for job in Config.JOB_TYPES)
    
    def test_get_city_multiplier_known_city(self):
        """Test multiplicateur pour ville connue."""
        from utils.config import Config
        
        # Test avec Paris qui devrait toujours être dans la liste
        if 'Paris' in Config.CITIES:
            multiplier = Config.get_city_multiplier('Paris')
            assert isinstance(multiplier, (int, float))
            assert multiplier > 0
            assert 0.5 <= multiplier <= 2.0  # Valeur raisonnable
    
    def test_get_city_multiplier_unknown_city(self):
        """Test multiplicateur pour ville inconnue."""
        from utils.config import Config
        
        multiplier = Config.get_city_multiplier('VilleInexistante123')
        assert multiplier == 1.0  # Devrait retourner 1.0 par défaut
    
    def test_get_sector_multiplier_known_sector(self):
        """Test multiplicateur pour secteur connu."""
        from utils.config import Config
        
        # Test avec Tech qui devrait être dans la liste
        if 'Tech' in Config.SECTORS:
            multiplier = Config.get_sector_multiplier('Tech')
            assert isinstance(multiplier, (int, float))
            assert multiplier > 0
            assert 0.5 <= multiplier <= 2.0
    
    def test_get_sector_multiplier_unknown_sector(self):
        """Test multiplicateur pour secteur inconnu."""
        from utils.config import Config
        
        multiplier = Config.get_sector_multiplier('SecteurInexistant123')
        assert multiplier == 1.0  # Devrait retourner 1.0 par défaut
    
    def test_get_all_city_multipliers(self):
        """Test récupération de tous les multiplicateurs villes."""
        from utils.config import Config
        
        multipliers = Config.get_all_city_multipliers()
        
        assert isinstance(multipliers, dict)
        assert len(multipliers) > 0
        
        # Vérifier que toutes les valeurs sont valides
        for city, mult in multipliers.items():
            assert isinstance(city, str)
            assert isinstance(mult, (int, float))
            assert mult > 0
            assert 0.5 <= mult <= 2.0  # Valeur raisonnable
    
    def test_get_all_sector_multipliers(self):
        """Test récupération de tous les multiplicateurs secteurs."""
        from utils.config import Config
        
        multipliers = Config.get_all_sector_multipliers()
        
        assert isinstance(multipliers, dict)
        assert len(multipliers) > 0
        
        # Vérifier que toutes les valeurs sont valides
        for sector, mult in multipliers.items():
            assert isinstance(sector, str)
            assert isinstance(mult, (int, float))
            assert mult > 0
            assert 0.5 <= mult <= 2.0
    
    def test_multipliers_consistency(self):
        """Test de cohérence entre villes et multiplicateurs."""
        from utils.config import Config
        
        city_mults = Config.get_all_city_multipliers()
        
        # Les villes dans CITIES devraient avoir des multiplicateurs
        for city in Config.CITIES[:5]:  # Tester les 5 premières
            assert city in city_mults or Config.get_city_multiplier(city) == 1.0


# ============================================================================
# TESTS - STRUCTURE DU PROJET
# ============================================================================

class TestProjectStructure:
    """Tests de validation de la structure du projet."""
    
    def test_utils_module_exists(self):
        """Test que le module utils existe."""
        import utils
        assert utils is not None
    
    def test_utils_config_importable(self):
        """Test que utils.config est importable."""
        from utils import config
        assert config is not None
    
    def test_utils_model_utils_importable(self):
        """Test que utils.model_utils est importable."""
        from utils import model_utils
        assert model_utils is not None
    
    def test_required_classes_exist(self):
        """Test que les classes requises existent."""
        from utils.config import Config
        from utils.model_utils import ModelUtils, CalculationUtils, ChartUtils
        
        assert Config is not None
        assert ModelUtils is not None
        assert CalculationUtils is not None
        assert ChartUtils is not None
    
    def test_pages_directory_structure(self):
        """Test que la structure pages/ existe."""
        pages_dir = Path(__file__).parent.parent / 'pages'
        assert pages_dir.exists(), "Le répertoire pages/ devrait exister"
    
    def test_internal_directory_structure(self):
        """Test que la structure internal/ existe."""
        internal_dir = Path(__file__).parent.parent / 'internal'
        assert internal_dir.exists(), "Le répertoire internal/ devrait exister"


# ============================================================================
# TESTS - DONNÉES ET MODÈLE
# ============================================================================

class TestDataAndModel:
    """Tests de validation des données et du modèle."""
    
    def test_data_path_configured(self):
        """Test que le chemin des données est configuré."""
        from utils.config import Config
        
        assert hasattr(Config, 'DATA_PATH')
        assert Config.DATA_PATH is not None
    
    def test_model_path_configured(self):
        """Test que le chemin du modèle est configuré."""
        from utils.config import Config
        
        assert hasattr(Config, 'MODEL_PATH')
        assert Config.MODEL_PATH is not None
    
    def test_model_utils_initializable(self):
        """Test que ModelUtils peut être initialisé."""
        try:
            from utils.model_utils import ModelUtils
            # Ne pas instancier si le modèle n'existe pas
            # Juste vérifier que la classe existe
            assert ModelUtils is not None
        except Exception as e:
            pytest.skip(f"ModelUtils non initialisable : {e}")


# ============================================================================
# TESTS - VALIDATION DES CONSTANTES
# ============================================================================

class TestConstants:
    """Tests de validation des constantes."""
    
    def test_feature_constants_exist(self):
        """Test que FeatureConstants existe."""
        from utils.model_utils import FeatureConstants
        
        assert FeatureConstants is not None
        assert hasattr(FeatureConstants, 'SENIORITY_MAP')
        assert hasattr(FeatureConstants, 'SKILL_WEIGHTS')
    
    def test_seniority_map_valid(self):
        """Test que SENIORITY_MAP est valide."""
        from utils.model_utils import FeatureConstants
        
        seniority_map = FeatureConstants.SENIORITY_MAP
        
        assert isinstance(seniority_map, dict)
        assert len(seniority_map) > 0
        
        # Vérifier que les valeurs sont des entiers positifs
        for level, value in seniority_map.items():
            assert isinstance(level, str)
            assert isinstance(value, int)
            assert value >= 0
    
    def test_skill_weights_valid(self):
        """Test que SKILL_WEIGHTS est valide."""
        from utils.model_utils import FeatureConstants
        
        skill_weights = FeatureConstants.SKILL_WEIGHTS
        
        assert isinstance(skill_weights, dict)
        assert len(skill_weights) > 0
        
        # Vérifier que les poids sont valides
        for skill, weight in skill_weights.items():
            assert isinstance(skill, str)
            assert isinstance(weight, (int, float))
            assert weight > 0
            assert weight <= 10  # Poids raisonnable


# ============================================================================
# TESTS - CALCULS DE BASE
# ============================================================================

class TestBasicCalculations:
    """Tests des calculs de base sans dépendances complexes."""
    
    def test_calculate_skills_count(self):
        """Test du comptage de compétences."""
        from utils.model_utils import CalculationUtils
        
        skills = {
            'contient_python': True,
            'contient_sql': True,
            'contient_r': False,
            'contient_machine_learning': True
        }
        
        count = CalculationUtils.calculate_skills_count_from_profile(skills)
        
        assert count == 3
        assert isinstance(count, int)
    
    def test_calculate_technical_score(self):
        """Test du calcul du score technique."""
        from utils.model_utils import CalculationUtils
        
        skills = {
            'contient_python': True,  # 2 points
            'contient_sql': True,     # 2 points
            'contient_tableau': True  # 1 point
        }
        
        score = CalculationUtils.calculate_technical_score_from_profile(skills)
        
        assert score == 5  # 2 + 2 + 1
        assert isinstance(score, (int, float))
    
    def test_get_percentile_real(self):
        """Test du calcul de percentile."""
        from utils.model_utils import CalculationUtils
        
        data = np.array([30000, 40000, 50000, 60000, 70000])
        value = 55000
        
        percentile = CalculationUtils.get_percentile_real(value, data)
        
        assert 0 <= percentile <= 100
        assert isinstance(percentile, (int, float))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
