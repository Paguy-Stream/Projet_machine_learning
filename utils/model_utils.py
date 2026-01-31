"""
Utilitaires pour le modèle de prédiction salariale.

Ce module contient toutes les classes et fonctions nécessaires pour :
- Gérer le modèle XGBoost (chargement, prédiction, explications SHAP)
- Calculer les statistiques dynamiques depuis le dataset
- Générer les visualisations (jauges, histogrammes, waterfall)
- Préparer les features pour le modèle
- Effectuer des calculs intelligents basés sur les distributions réelles

Architecture:
    - FeatureConstants: Constantes du modèle (encodages, poids)
    - DataDistributions: Statistiques calculées dynamiquement
    - ModelUtils: Gestion du modèle XGBoost et SHAP
    - ChartUtils: Création de graphiques Plotly
    - CalculationUtils: Calculs et estimations intelligentes
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import json
from datetime import datetime
from typing import Dict, Optional, List, Tuple, Any
from pathlib import Path

import plotly.graph_objects as go
import plotly.express as px

from utils.config import Config


# ============================================================================
# CONSTANTES DU MODÈLE
# ============================================================================

class FeatureConstants:
    """
    Constantes définies lors de l'entraînement du modèle XGBoost.
    
    Ces valeurs sont FIGÉES car le modèle a été entraîné avec.
    Ne JAMAIS les modifier sans réentraîner le modèle.
    
    Attributes:
        SENIORITY_MAP (dict): Encodage ordinal des niveaux de séniorité
        SKILL_WEIGHTS (dict): Poids des compétences pour le score technique
        
    Warning:
        Toute modification de ces constantes invalidera le modèle entraîné.
        
    Examples:
        >>> FeatureConstants.SENIORITY_MAP['Senior']
        5
        >>> FeatureConstants.SKILL_WEIGHTS['contient_python']
        2
    """
    
    # Encodage ordinal de la hiérarchie professionnelle
    SENIORITY_MAP = {
        'Stage/Alternance': 1,
        'Débutant (<1 an)': 1,
        'Junior (1-3 ans)': 2,
        'Mid-level': 4,
        'Mid confirmé (3-5 ans)': 3,
        'Senior (5-8 ans)': 4,
        'Senior': 5,
        'Expert (8-12 ans)': 5,
        'Lead/Manager (12-20 ans)': 6,
        'Lead/Manager': 7,
        'Directeur/VP (>20 ans)': 7,
        'Freelance/Consultant': 3,
        'Non spécifié': 0
    }
    
    # Poids des compétences pour le score technique
    SKILL_WEIGHTS = {
        'contient_python': 2,
        'contient_sql': 2,
        'contient_machine_learning': 2,
        'contient_deep_learning': 2,
        'contient_r': 1,
        'contient_aws': 1,
        'contient_azure': 1,
        'contient_gcp': 1,
        'contient_spark': 1,
        'contient_tableau': 1,
        'contient_power_bi': 1
    }


# ============================================================================
# DISTRIBUTIONS STATISTIQUES DYNAMIQUES
# ============================================================================

class DataDistributions:
    """
    Statistiques calculées dynamiquement depuis le dataset.
    
    Cette classe charge et met en cache toutes les distributions statistiques
    utilisées pour les calculs intelligents :
    - Distribution des longueurs de descriptions
    - Distribution des mots-clés techniques
    - Corrélations entre variables
    - Secteurs bien payés (P75)
    - Grandes villes (top 3)
    - Impacts des variables sur la description
    
    Le cache est partagé entre toutes les instances (class-level).
    
    Attributes:
        _distributions_cache (dict): Cache des statistiques calculées
        
    Examples:
        >>> desc_stats = DataDistributions.get_desc_words()
        >>> print(desc_stats['median'])
        585
        
        >>> sectors = DataDistributions.get_high_paying_sectors()
        >>> print(sectors)
        ['Banque', 'Finance', 'Tech']
    """
    
    _distributions_cache: Optional[Dict[str, Any]] = None
    
    @classmethod
    def _load_distributions(cls) -> Dict[str, Any]:
        """
        Charge et calcule toutes les distributions depuis le dataset.
        
        Cette méthode :
        1. Charge le dataset complet
        2. Calcule les statistiques descriptives (percentiles, moyennes)
        3. Identifie les secteurs bien payés (>P75)
        4. Identifie les grandes villes (top 3 fréquence)
        5. Calcule les impacts des variables (régressions)
        6. Met en cache tous les résultats
        
        Returns:
            Dict contenant toutes les statistiques calculées
            
        Raises:
            Exception: En cas d'erreur, retourne des valeurs par défaut
            
        Notes:
            Les calculs ne sont effectués qu'une seule fois grâce au cache.
            Pour forcer un rechargement, utiliser reload().
        """
        if cls._distributions_cache is not None:
            return cls._distributions_cache
        
        try:
            # Chargement du dataset
            df = pd.read_csv(Config.DATA_PATH, encoding='utf-8', sep=',')
            
            # Nettoyage des colonnes numériques
            numeric_cols = ['description_word_count', 'nb_mots_cles_techniques']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Nettoyage des colonnes binaires
            binary_cols = ['contient_machine_learning', 'contient_deep_learning']
            for col in binary_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            
            # === 1. DISTRIBUTION DES LONGUEURS DE DESCRIPTION ===
            desc_words = df['description_word_count'].dropna()
            desc_stats = {
                'p10': int(desc_words.quantile(0.10)),
                'p25': int(desc_words.quantile(0.25)),
                'median': int(desc_words.median()),
                'p75': int(desc_words.quantile(0.75)),
                'p90': int(desc_words.quantile(0.90)),
                'mean': int(desc_words.mean()),
                'count': len(desc_words)
            }
            
            # === 2. DISTRIBUTION DES MOTS-CLÉS TECHNIQUES ===
            tech_kw = df['nb_mots_cles_techniques'].dropna()
            tech_stats = {
                'p25': int(tech_kw.quantile(0.25)),
                'median': int(tech_kw.median()),
                'p75': int(tech_kw.quantile(0.75)),
                'p90': int(tech_kw.quantile(0.90)),
                'mean': float(tech_kw.mean()),
                'count': len(tech_kw)
            }
            
            # === 3. CORRÉLATION ML/DL ===
            ml_dl_corr = df['contient_deep_learning'].corr(
                df['contient_machine_learning']
            )
            
            # === 4. SECTEURS BIEN PAYÉS (P75) ===
            high_paying_sectors = cls._identify_high_paying_sectors(df)
            
            # === 5. GRANDES VILLES (TOP 3) ===
            grandes_villes = cls._identify_major_cities(df)
            
            # === 6. IMPACT DES COMPÉTENCES SUR LA DESCRIPTION ===
            skills_word_impact = cls._calculate_skills_impact(df)
            
            # === 7. IMPACT DE L'EXPÉRIENCE SUR LA DESCRIPTION ===
            exp_word_impact = cls._calculate_experience_impact(df)
            
            # === 8. AJUSTEMENTS PAR SECTEUR ===
            sector_adjustments = cls._calculate_sector_adjustments(
                df, desc_stats['median']
            )
            
            # Mise en cache de tous les résultats
            cls._distributions_cache = {
                'DESC_WORDS': desc_stats,
                'TECH_KEYWORDS': tech_stats,
                'ML_DL_CORRELATION': float(ml_dl_corr),
                'TOTAL_OFFERS': len(df),
                'HIGH_PAYING_SECTORS': high_paying_sectors,
                'GRANDES_VILLES': grandes_villes,
                'SKILLS_WORD_IMPACT': skills_word_impact,
                'EXP_WORD_IMPACT': exp_word_impact,
                'SECTOR_ADJUSTMENTS': sector_adjustments
            }
            
            st.success(
                f"✅ Statistiques dynamiques chargées depuis {len(df):,} offres"
            )
            return cls._distributions_cache
            
        except Exception as e:
            st.warning(
                f"⚠️ Calcul dynamique échoué : {str(e)[:150]}\n"
                "Utilisation des valeurs par défaut."
            )
            return cls._get_default_distributions()
    
    @staticmethod
    def _identify_high_paying_sectors(df: pd.DataFrame) -> List[str]:
        """
        Identifie les secteurs bien payés (>P75 des salaires médians).
        
        Args:
            df: DataFrame du dataset
            
        Returns:
            Liste des secteurs au-dessus du 75e percentile
        """
        if 'salary' not in df.columns or 'sector_clean' not in df.columns:
            return ['Banque', 'Finance', 'Tech']  # Fallback
        
        sector_medians = df.groupby('sector_clean')['salary'].median()
        threshold_75 = sector_medians.quantile(0.75)
        high_paying = sector_medians[sector_medians >= threshold_75]
        
        return high_paying.index.tolist()
    
    @staticmethod
    def _identify_major_cities(df: pd.DataFrame) -> List[str]:
        """
        Identifie les 3 villes les plus fréquentes dans le dataset.
        
        Args:
            df: DataFrame du dataset
            
        Returns:
            Liste des 3 villes principales
        """
        if 'location_final' not in df.columns:
            return ['Paris', 'Lyon', 'Marseille']  # Fallback
        
        location_counts = df['location_final'].value_counts()
        return location_counts.head(3).index.tolist()
    
    @staticmethod
    def _calculate_skills_impact(df: pd.DataFrame) -> float:
        """
        Calcule l'impact du nombre de compétences sur la longueur de description.
        
        Utilise une régression linéaire simple.
        
        Args:
            df: DataFrame du dataset
            
        Returns:
            Coefficient de la régression (slope)
        """
        if 'skills_count' not in df.columns:
            return 15.0  # Fallback
        
        try:
            from scipy.stats import linregress
            
            valid_data = df[['skills_count', 'description_word_count']].dropna()
            
            if len(valid_data) < 10:
                return 15.0
            
            slope, intercept, _, _, _ = linregress(
                valid_data['skills_count'],
                valid_data['description_word_count']
            )
            
            return float(slope)
            
        except Exception:
            return 15.0
    
    @staticmethod
    def _calculate_experience_impact(df: pd.DataFrame) -> float:
        """
        Calcule l'impact de l'expérience sur la longueur de description.
        
        Utilise une régression linéaire simple.
        
        Args:
            df: DataFrame du dataset
            
        Returns:
            Coefficient de la régression (slope)
        """
        if 'experience_final' not in df.columns:
            return 10.0  # Fallback
        
        try:
            from scipy.stats import linregress
            
            valid_data = df[['experience_final', 'description_word_count']].dropna()
            
            if len(valid_data) < 10:
                return 10.0
            
            slope, _, _, _, _ = linregress(
                valid_data['experience_final'],
                valid_data['description_word_count']
            )
            
            return float(slope)
            
        except Exception:
            return 10.0
    
    @staticmethod
    def _calculate_sector_adjustments(
        df: pd.DataFrame,
        global_median: float
    ) -> Dict[str, int]:
        """
        Calcule les ajustements de description par secteur.
        
        Args:
            df: DataFrame du dataset
            global_median: Médiane globale des longueurs de description
            
        Returns:
            Dict {secteur: ajustement en nombre de mots}
        """
        if 'sector_clean' not in df.columns:
            return {}
        
        sector_adjustments = {}
        
        for sector in df['sector_clean'].unique():
            sector_data = df[df['sector_clean'] == sector]['description_word_count']
            sector_median = sector_data.median()
            
            if not pd.isna(sector_median):
                adjustment = int(sector_median - global_median)
                sector_adjustments[sector] = adjustment
        
        return sector_adjustments
    
    @staticmethod
    def _get_default_distributions() -> Dict[str, Any]:
        """
        Retourne les distributions par défaut en cas d'erreur.
        
        Returns:
            Dict avec valeurs par défaut calculées sur le dataset initial
        """
        return {
            'DESC_WORDS': {
                'p10': 300,
                'p25': 430,
                'median': 585,
                'p75': 710,
                'p90': 748,
                'mean': 554,
                'count': 5868
            },
            'TECH_KEYWORDS': {
                'p25': 1,
                'median': 1,
                'p75': 3,
                'p90': 5,
                'mean': 2.0,
                'count': 5868
            },
            'ML_DL_CORRELATION': 0.4050,
            'TOTAL_OFFERS': 5868,
            'HIGH_PAYING_SECTORS': ['Banque', 'Finance', 'Tech'],
            'GRANDES_VILLES': ['Paris', 'Lyon', 'Marseille'],
            'SKILLS_WORD_IMPACT': 15.0,
            'EXP_WORD_IMPACT': 10.0,
            'SECTOR_ADJUSTMENTS': {}
        }
    
    # ========================================================================
    # ACCESSEURS PUBLICS
    # ========================================================================
    
    @classmethod
    def get_desc_words(cls) -> Dict[str, int]:
        """Retourne les statistiques de longueur des descriptions."""
        return cls._load_distributions()['DESC_WORDS']
    
    @classmethod
    def get_tech_keywords(cls) -> Dict[str, float]:
        """Retourne les statistiques des mots-clés techniques."""
        return cls._load_distributions()['TECH_KEYWORDS']
    
    @classmethod
    def get_ml_dl_correlation(cls) -> float:
        """Retourne la corrélation entre ML et DL."""
        return cls._load_distributions()['ML_DL_CORRELATION']
    
    @classmethod
    def get_total_offers(cls) -> int:
        """Retourne le nombre total d'offres dans le dataset."""
        return cls._load_distributions()['TOTAL_OFFERS']
    
    @classmethod
    def get_high_paying_sectors(cls) -> List[str]:
        """Retourne la liste des secteurs bien payés (>P75)."""
        return cls._load_distributions()['HIGH_PAYING_SECTORS']
    
    @classmethod
    def get_grandes_villes(cls) -> List[str]:
        """Retourne la liste des 3 villes principales."""
        return cls._load_distributions()['GRANDES_VILLES']
    
    @classmethod
    def get_skills_word_impact(cls) -> float:
        """Retourne l'impact des compétences sur la longueur de description."""
        return cls._load_distributions()['SKILLS_WORD_IMPACT']
    
    @classmethod
    def get_exp_word_impact(cls) -> float:
        """Retourne l'impact de l'expérience sur la longueur de description."""
        return cls._load_distributions()['EXP_WORD_IMPACT']
    
    @classmethod
    def get_sector_adjustments(cls) -> Dict[str, int]:
        """Retourne les ajustements de description par secteur."""
        return cls._load_distributions()['SECTOR_ADJUSTMENTS']
    
    @classmethod
    def reload(cls) -> Dict[str, Any]:
        """
        Force le rechargement de toutes les distributions.
        
        Returns:
            Nouvelles distributions calculées
        """
        cls._distributions_cache = None
        return cls._load_distributions()


# ============================================================================
# GESTION DU MODÈLE XGBOOST
# ============================================================================

class ModelUtils:
    """
    Gestionnaire du modèle XGBoost et des explications SHAP.
    
    Cette classe centralise :
    - Le chargement du modèle et des ressources associées
    - Les prédictions salariales
    - Les explications SHAP (feature importance)
    - La préparation des features pour le modèle
    - L'accès aux métriques de performance
    
    Attributes:
        model: Pipeline XGBoost chargé
        report (dict): Rapport de modélisation JSON
        test_data (dict): Données de test pour comparaisons
        explainer: Explainer SHAP (si initialisé)
        
    Examples:
        >>> utils = ModelUtils()
        >>> profile = {'job_type': 'Data Scientist', 'experience_final': 5}
        >>> result = utils.predict(profile)
        >>> print(result['prediction'])
        52000
    """
    
    def __init__(self):
        """
        Initialise le gestionnaire de modèle.
        
        Charge :
        - Le modèle XGBoost
        - Le rapport de modélisation
        - Les données de test
        - L'explainer SHAP (si possible)
        """
        resources = self._load_all_resources()
        self.model = resources['model']
        self.report = resources['report']
        self.test_data = resources['test_data']
        self.explainer = None
        self._init_shap_explainer()
    
    @staticmethod
    @st.cache_resource
    def _load_all_resources() -> Dict[str, Any]:
        """
        Charge toutes les ressources nécessaires (modèle, rapport, données).
        
        Utilise le cache Streamlit pour éviter les rechargements.
        
        Returns:
            Dict contenant model, report et test_data
            
        Notes:
            Le décorateur @st.cache_resource assure que le chargement
            n'est fait qu'une seule fois par session.
        """
        resources = {
            'model': None,
            'report': None,
            'test_data': None
        }
        
        # Chargement du modèle
        model_path = Config.MODEL_PATH
        if model_path.exists():
            try:
                resources['model'] = joblib.load(model_path)
                st.success("✅ Modèle XGBoost v7 chargé")
            except Exception as e:
                st.error(f"❌ Erreur chargement modèle : {str(e)[:100]}")
        else:
            st.error(f"❌ Modèle introuvable : {model_path}")
        
        # Chargement du rapport
        report_path = Config.REPORT_PATH
        if report_path.exists():
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    resources['report'] = json.load(f)
            except Exception as e:
                st.warning(f"⚠️ Rapport non chargé : {str(e)[:80]}")
        
        # Chargement des données de test
        test_data_path = (
            Config.BASE_DIR / "output" / "analysis_complete" /
            "modeling_v7_improved" / "models" / "test_data.pkl"
        )
        
        if test_data_path.exists():
            try:
                with open(test_data_path, 'rb') as f:
                    resources['test_data'] = pickle.load(f)
            except Exception as e:
                st.warning(f"⚠️ Données test non chargées : {str(e)[:80]}")
        
        return resources
    
    def _init_shap_explainer(self) -> None:
        """
        Initialise l'explainer SHAP pour les feature importance.
        
        Nécessite :
        - Le modèle XGBoost chargé
        - Les données d'entraînement (train_data.pkl)
        
        En cas d'erreur, l'explainer reste None et les explications
        SHAP ne seront pas disponibles.
        """
        if self.model is None:
            return
        
        try:
            import shap
            
            # Chargement des données d'entraînement
            x_train_path = (
                Config.BASE_DIR / "output" / "analysis_complete" /
                "modeling_v7_improved" / "models" / "train_data.pkl"
            )
            
            if not x_train_path.exists():
                st.warning("⚠️ Données d'entraînement introuvables - SHAP non disponible")
                return
            
            with open(x_train_path, 'rb') as f:
                train_data = pickle.load(f)
                X_train_raw = train_data['X_train']
            
            # Transformation des données
            fe = self.model.named_steps['feature_eng']
            preprocessor = self.model.named_steps['preprocessor']
            xgb_model = self.model.named_steps['regressor']
            
            X_transformed = preprocessor.transform(fe.transform(X_train_raw))
            X_df = pd.DataFrame(
                X_transformed,
                columns=self._get_feature_names()
            )
            
            # Échantillonnage pour SHAP (100 observations)
            X_sample = X_df.sample(
                n=min(100, len(X_df)),
                random_state=42
            )
            
            # Initialisation de l'explainer
            self.explainer = shap.TreeExplainer(xgb_model, X_sample)
            st.success("✅ SHAP initialisé")
            
        except Exception as e:
            st.error(f"❌ Initialisation SHAP échouée : {str(e)[:100]}")
    
    def _get_feature_names(self) -> List[str]:
        """
        Extrait les noms de toutes les features après preprocessing.
        
        Returns:
            Liste des noms de features (numériques + catégorielles encodées)
        """
        preprocessor = self.model.named_steps['preprocessor']
        
        # Features numériques
        num_features = preprocessor.transformers_[0][2]
        
        # Features catégorielles (one-hot encoded)
        cat_encoder = preprocessor.transformers_[1][1].named_steps['onehot']
        cat_features = preprocessor.transformers_[1][2]
        cat_feature_names = cat_encoder.get_feature_names_out(cat_features)
        
        return list(num_features) + list(cat_feature_names)
    
    def _aggregate_shap_by_original_feature(
        self,
        shap_values: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[List[str], List[float]]:
        """
        Agrège les valeurs SHAP par feature originale.
        
        Regroupe les features one-hot encodées (ex: job_type_Data_Scientist,
        job_type_Data_Engineer) en une seule feature (job_type).
        
        Args:
            shap_values: Valeurs SHAP pour chaque feature
            feature_names: Noms des features
            
        Returns:
            Tuple (noms_agrégés, valeurs_agrégées) triés par importance
        """
        aggregated = {}
        
        for val, name in zip(shap_values, feature_names):
            # Déterminer le nom de base de la feature
            if name.startswith(('contient_', 'has_', 'is_')):
                base_name = name
            elif '_' in name:
                parts = name.split('_')
                prefixes = ['job', 'contract', 'education', 'sector', 'location']
                
                if parts[0] in prefixes:
                    base_name = '_'.join(parts[:2])
                else:
                    base_name = parts[0]
            else:
                base_name = name
            
            # Agréger les valeurs
            aggregated[base_name] = aggregated.get(base_name, 0.0) + val
        
        # Trier par importance absolue
        sorted_items = sorted(
            aggregated.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        names = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]
        
        return names, values
    
    def predict(self, profile_data: Dict) -> Optional[Dict[str, Any]]:
        """
        Effectue une prédiction salariale pour un profil donné.
        
        Args:
            profile_data: Dictionnaire contenant les caractéristiques du profil
            
        Returns:
            Dict contenant :
                - prediction: Salaire prédit
                - lower_bound: Borne inférieure (P75 erreur)
                - upper_bound: Borne supérieure (P75 erreur)
                - mae_error: MAE du modèle
                - std_error: Écart-type de l'erreur
                - prediction_date: Date de la prédiction
                - model_version: Version du modèle
                
            None si erreur
            
        Examples:
            >>> utils = ModelUtils()
            >>> profile = {
            ...     'job_type': 'Data Scientist',
            ...     'experience_final': 5,
            ...     'location_final': 'Paris'
            ... }
            >>> result = utils.predict(profile)
            >>> print(f"Salaire: {result['prediction']:,.0f}€")
            Salaire: 52,000€
        """
        if self.model is None:
            st.error("❌ Modèle non chargé")
            return None
        
        try:
            # Préparation des features
            df = self._prepare_features_for_real_model(profile_data)
            
            # Prédiction
            prediction = self.model.predict(df)[0]
            
            # Récupération des métriques de performance
            perf = self.get_model_performance()
            error_p75 = perf.get('error_75_percentile', 7417)
            
            return {
                'prediction': float(prediction),
                'lower_bound': float(max(0, prediction - error_p75)),
                'upper_bound': float(prediction + error_p75),
                'mae_error': float(perf.get('test_mae', 5163)),
                'std_error': float(perf.get('cv_mae_std', 183)),
                'prediction_date': datetime.now().isoformat(),
                'model_version': 'XGBoost_v7'
            }
            
        except Exception as e:
            st.error(f"❌ Erreur prédiction : {str(e)[:100]}")
            return None
    
    def explain_prediction(
        self,
        profile_data: Dict
    ) -> Optional[Dict[str, Any]]:
        """
        Génère une explication SHAP pour une prédiction.
        
        Args:
            profile_data: Dictionnaire du profil à expliquer
            
        Returns:
            Dict contenant :
                - shap_values: Valeurs SHAP agrégées
                - feature_names: Noms des features agrégées
                - base_value: Valeur de base (moyenne des prédictions)
                - prediction: Prédiction finale
                - raw_shap: Objet SHAP brut (pour waterfall)
                
            None si SHAP non disponible ou erreur
        """
        if self.explainer is None:
            st.warning("⚠️ SHAP non initialisé - Explications non disponibles")
            return None
        
        try:
            # Préparation et transformation des features
            df = self._prepare_features_for_real_model(profile_data)
            df_eng = self.model.named_steps['feature_eng'].transform(df)
            df_transformed = self.model.named_steps['preprocessor'].transform(df_eng)
            df_final = pd.DataFrame(
                df_transformed,
                columns=self._get_feature_names()
            )
            
            # Calcul des valeurs SHAP
            shap_values = self.explainer(df_final)
            
            # Agrégation par feature originale
            agg_names, agg_values = self._aggregate_shap_by_original_feature(
                shap_values.values[0],
                shap_values.feature_names
            )
            
            return {
                'shap_values': agg_values,
                'feature_names': agg_names,
                'base_value': float(shap_values.base_values[0]),
                'prediction': float(
                    shap_values.values[0].sum() + shap_values.base_values[0]
                ),
                'raw_shap': shap_values
            }
            
        except Exception as e:
            st.error(f"❌ Erreur SHAP : {str(e)}")
            return None
    
    def _prepare_features_for_real_model(
        self,
        profile_data: Dict
    ) -> pd.DataFrame:
        """
        Prépare les features pour le modèle avec valeurs par défaut dynamiques.
        
        Cette méthode :
        1. Extrait les features de base du profil
        2. Applique les valeurs par défaut dynamiques
        3. Calcule les features dérivées (interactions, encodages)
        4. Retourne un DataFrame prêt pour la prédiction
        
        Args:
            profile_data: Dictionnaire du profil utilisateur
            
        Returns:
            DataFrame avec toutes les features nécessaires
        """
        # === FEATURES DE BASE ===
        features = {
            # Informations métier
            'job_type_with_desc': profile_data.get('job_type', 'Data Analyst'),
            'seniority': profile_data.get('seniority', 'Mid-level'),
            'contract_type_clean': profile_data.get('contract_type_clean', 'CDI'),
            'education_clean': profile_data.get('education_clean', 'Bac+5'),
            'experience_final': float(profile_data.get('experience_final', 4.0)),
            
            # Localisation et secteur
            'location_final': profile_data.get('location_final', 'Paris'),
            'sector_clean': profile_data.get('sector_clean', 'Non spécifié'),
            
            # Compétences techniques (langages)
            'contient_sql': int(profile_data.get('contient_sql', True)),
            'contient_python': int(profile_data.get('contient_python', True)),
            'contient_r': int(profile_data.get('contient_r', False)),
            
            # Compétences techniques (outils)
            'contient_tableau': int(profile_data.get('contient_tableau', False)),
            'contient_power_bi': int(profile_data.get('contient_power_bi', False)),
            
            # Compétences techniques (cloud)
            'contient_aws': int(profile_data.get('contient_aws', False)),
            'contient_azure': int(profile_data.get('contient_azure', False)),
            'contient_gcp': int(profile_data.get('contient_gcp', False)),
            
            # Compétences techniques (big data)
            'contient_spark': int(profile_data.get('contient_spark', False)),
            'contient_etl': int(profile_data.get('contient_etl', False)),
            
            # Compétences techniques (ML/AI)
            'contient_machine_learning': int(
                profile_data.get('contient_machine_learning', False)
            ),
            
            # Scores agrégés
            'skills_count': int(profile_data.get('skills_count', 3)),
            'technical_score': int(profile_data.get('technical_score', 2)),
            'benefits_score': int(profile_data.get('benefits_score', 1)),
            
            # Avantages
            'has_teletravail': int(profile_data.get('has_teletravail', True)),
            'has_mutuelle': int(profile_data.get('has_mutuelle', False)),
            'has_tickets': int(profile_data.get('has_tickets', False)),
            'has_prime': int(profile_data.get('has_prime', False)),
            'telework_numeric': float(profile_data.get('telework_numeric', 0.5)),
            
            # Features de description (valeurs dynamiques)
            'is_grande_ville': int(
                profile_data.get('location_final', 'Paris')
                in DataDistributions.get_grandes_villes()
            ),
            'description_word_count': int(
                profile_data.get(
                    'description_word_count',
                    DataDistributions.get_desc_words()['median']
                )
            ),
            'nb_mots_cles_techniques': int(
                profile_data.get(
                    'nb_mots_cles_techniques',
                    DataDistributions.get_tech_keywords()['median']
                )
            )
        }
        
        # === FEATURE ENGINEERING ===
        
        # Gestion de l'expérience manquante
        exp_value = features['experience_final']
        features['experience_missing'] = 1 if pd.isna(exp_value) else 0
        features['experience_final'] = exp_value if not pd.isna(exp_value) else 8.0
        
        # Interaction tech × expérience
        features['tech_exp_interaction'] = (
            features['technical_score'] * np.log1p(features['experience_final'])
        )
        
        # Indicateur région parisienne
        features['is_paris_region'] = int(
            'Paris' in str(features.get('location_final', ''))
        )
        
        # Score de compétences avancées
        advanced_skills = [
            'contient_machine_learning',
            'contient_spark',
            'contient_aws'
        ]
        features['advanced_data_score'] = sum([
            features.get(skill, 0) for skill in advanced_skills
        ])
        
        # Indicateur secteur bien payé
        features['is_high_paying_sector'] = int(
            features.get('sector_clean', '')
            in DataDistributions.get_high_paying_sectors()
        )
        
        # Encodage de la complexité technique
        nb_kw = features.get('nb_mots_cles_techniques', 0)
        if nb_kw <= 2:
            features['tech_complexity_encoded'] = 0
        elif nb_kw <= 5:
            features['tech_complexity_encoded'] = 1
        elif nb_kw <= 10:
            features['tech_complexity_encoded'] = 2
        else:
            features['tech_complexity_encoded'] = 3
        
        # Indicateur stack moderne
        features['has_modern_stack'] = int(
            features.get('contient_python', 0) > 0 and
            any([
                features.get(f'contient_{cloud}', 0) > 0
                for cloud in ['aws', 'azure', 'gcp']
            ]) and
            features.get('contient_spark', 0) > 0
        )
        
        # Score hiérarchique
        features['hierarchy_score'] = FeatureConstants.SENIORITY_MAP.get(
            features.get('seniority', ''),
            0
        )
        
        return pd.DataFrame([features])
    
    def get_real_market_data(self) -> Optional[np.ndarray]:
        """
        Retourne les salaires réels du jeu de test.
        
        Returns:
            Array des salaires ou None si non disponible
        """
        if self.test_data and 'y_test' in self.test_data:
            return self.test_data['y_test']
        return None
    
    def get_model_performance(self) -> Dict[str, float]:
        """
        Retourne les métriques de performance du modèle.
        
        Returns:
            Dict avec R², MAE, RMSE, stabilité, etc.
        """
        if self.report:
            return self.report.get('performance_metrics', {})
        
        # Valeurs par défaut si rapport non disponible
        return {
            'test_r2': 0.337,
            'test_mae': 5163,
            'cv_mae_mean': 5188,
            'cv_mae_std': 183,
            'stability': 0.995,
            'error_75_percentile': 7417
        }


# ============================================================================
# UTILITAIRES DE VISUALISATION
# ============================================================================

class ChartUtils:
    """
    Utilitaires pour créer des graphiques Plotly.
    
    Contient des méthodes statiques pour générer :
    - Jauges de positionnement salarial
    - Histogrammes de comparaison au marché
    - Graphiques waterfall SHAP
    - Barres d'importance des features
    """
    
    @staticmethod
    def create_salary_gauge(
        prediction: float,
        market_median: float,
        q1: float,
        q3: float,
        gauge_min: float,
        gauge_max: float
    ) -> go.Figure:
        """
        Crée une jauge de positionnement salarial.
        
        Args:
            prediction: Salaire prédit
            market_median: Médiane du marché
            q1: Premier quartile
            q3: Troisième quartile
            gauge_min: Minimum de la jauge
            gauge_max: Maximum de la jauge
            
        Returns:
            Figure Plotly avec la jauge
        """
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prediction,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={
                'text': "Positionnement salarial",
                'font': {'size': 22, 'weight': 'bold'}
            },
            delta={
                'reference': market_median,
                'valueformat': '.0f',
                'suffix': '€'
            },
            number={
                'suffix': " €",
                'font': {'size': 42, 'weight': 'bold'},
                'valueformat': ',.0f'
            },
            gauge={
                'axis': {
                    'range': [gauge_min, gauge_max],
                    'tickformat': ',.0f',
                    'tickprefix': '€'
                },
                'bar': {
                    'color': "#1f77b4",
                    'thickness': 0.75
                },
                'steps': [
                    {'range': [gauge_min, q1], 'color': '#f0f0f0'},
                    {'range': [q1, market_median], 'color': '#c7e9c0'},
                    {'range': [market_median, q3], 'color': '#74c476'},
                    {'range': [q3, gauge_max], 'color': '#fdd835'}
                ],
                'threshold': {
                    'line': {'color': "#ff7f0e", 'width': 3},
                    'thickness': 0.8,
                    'value': prediction
                }
            }
        ))
        
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=60, b=20),
            paper_bgcolor="white"
        )
        
        return fig
    
    @staticmethod
    def create_market_comparison(
        prediction: float,
        market_data: np.ndarray,
        market_median: float,
        error_margin: float = 7417
    ) -> go.Figure:
        """
        Crée un histogramme de comparaison au marché.
        
        Args:
            prediction: Salaire prédit
            market_data: Distribution des salaires du marché
            market_median: Médiane du marché
            error_margin: Marge d'erreur (P75)
            
        Returns:
            Figure Plotly avec l'histogramme
        """
        fig = go.Figure()
        
        # Histogramme du marché
        fig.add_trace(go.Histogram(
            x=market_data,
            nbinsx=25,
            name='Marché',
            marker_color='#1f77b4',
            opacity=0.7
        ))
        
        # Ligne de prédiction
        fig.add_vline(
            x=prediction,
            line_dash="solid",
            line_color="#ff7f0e",
            line_width=3,
            annotation_text="Votre prédiction",
            annotation_position="top"
        )
        
        # Ligne médiane du marché
        fig.add_vline(
            x=market_median,
            line_dash="dash",
            line_color="#2ca02c",
            line_width=2,
            annotation_text="Médiane marché",
            annotation_position="bottom"
        )
        
        # Zone d'incertitude
        fig.add_vrect(
            x0=prediction - error_margin,
            x1=prediction + error_margin,
            fillcolor="rgba(255,127,14,0.1)",
            line_width=0,
            annotation_text="Intervalle de confiance",
            annotation_position="top left"
        )
        
        fig.update_layout(
            title="📊 Position sur le marché",
            xaxis_title="Salaire annuel brut (€)",
            yaxis_title="Nombre d'offres",
            height=400,
            showlegend=True,
            plot_bgcolor='white'
        )
        
        return fig
    
    @staticmethod
    def create_shap_waterfall(
        shap_exp: Dict,
        feature_translation: Dict[str, str],
        max_display: int = 10
    ) -> Optional[go.Figure]:
        """
        Crée un graphique waterfall des valeurs SHAP.
        
        Args:
            shap_exp: Dictionnaire d'explication SHAP
            feature_translation: Traduction des noms de features
            max_display: Nombre maximum de features à afficher
            
        Returns:
            Figure Plotly ou None si erreur
        """
        try:
            import shap
            
            raw = shap_exp.get('raw_shap')
            if not raw:
                return None
            
            vals = raw.values[0]
            feats = [
                feature_translation.get(f, f.replace('_', ' ').title())
                for f in raw.feature_names
            ]
            base = raw.base_values[0]
            
            # Sélection des top features par importance
            idx = np.argsort(np.abs(vals))[::-1][:max_display]
            
            # Construction du waterfall
            x_labels = ['Base']
            y_values = [base]
            cumsum = base
            
            for i in idx:
                cumsum += vals[i]
                x_labels.append(feats[i][:25])  # Tronquer à 25 caractères
                y_values.append(cumsum)
            
            x_labels.append('Prédiction')
            y_values.append(cumsum)
            
            # Création du graphique
            fig = go.Figure(go.Waterfall(
                x=x_labels,
                y=y_values,
                connector={"line": {"color": "rgb(63,63,63)"}},
                increasing={"marker": {"color": "#2ca02c"}},
                decreasing={"marker": {"color": "#d62728"}}
            ))
            
            fig.update_layout(
                title="🔍 Impact de vos caractéristiques sur le salaire",
                xaxis_title="Caractéristiques",
                yaxis_title="Salaire (€)",
                height=500,
                yaxis_tickformat=',.0f'
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"⚠️ Impossible de créer le waterfall : {str(e)[:80]}")
            return None
    
    @staticmethod
    def create_feature_importance_bar(
        shap_exp: Dict,
        top_n: int = 15
    ) -> go.Figure:
        """
        Crée un graphique en barres des features importantes.
        
        Args:
            shap_exp: Dictionnaire d'explication SHAP
            top_n: Nombre de features à afficher
            
        Returns:
            Figure Plotly avec les barres
        """
        names = shap_exp['feature_names'][:top_n]
        vals = shap_exp['shap_values'][:top_n]
        
        fig = go.Figure(go.Bar(
            x=vals,
            y=names,
            orientation='h',
            marker=dict(
                color=vals,
                colorscale='RdBu',
                cmid=0
            ),
            text=[f"{v:+,.0f}€" for v in vals],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="📊 Top facteurs influençant le salaire",
            xaxis_title="Impact sur le salaire (€)",
            yaxis_title="",
            height=600,
            yaxis=dict(autorange="reversed"),
            plot_bgcolor='white'
        )
        
        return fig


# ============================================================================
# CALCULS ET ESTIMATIONS INTELLIGENTES
# ============================================================================

class CalculationUtils:
    """
    Utilitaires pour calculs dynamiques et estimations intelligentes.
    
    Contient des méthodes pour :
    - Estimer la complexité des descriptions
    - Calculer les scores de compétences
    - Déterminer les percentiles
    - Créer des résumés de profil
    """
    
    @staticmethod
    def _interpolate(
        value: float,
        ranges: List[float],
        outputs: List[int]
    ) -> int:
        """
        Effectue une interpolation linéaire entre des ranges.
        
        Args:
            value: Valeur à interpoler
            ranges: Liste des bornes de ranges
            outputs: Liste des valeurs correspondantes
            
        Returns:
            Valeur interpolée
            
        Examples:
            >>> CalculationUtils._interpolate(3, [0, 2, 5], [10, 20, 30])
            26
        """
        for i in range(len(ranges) - 1):
            if ranges[i] <= value < ranges[i + 1]:
                ratio = (value - ranges[i]) / (ranges[i + 1] - ranges[i])
                interpolated = outputs[i] + ratio * (outputs[i + 1] - outputs[i])
                return int(interpolated)
        
        # Si hors limites, retourner la première ou dernière valeur
        return outputs[-1] if value >= ranges[-1] else outputs[0]
    
    @staticmethod
    def estimate_description_complexity(profile_data: Dict) -> int:
        """
        Estime la longueur de description basée sur le profil.
        
        Utilise :
        - L'expérience (interpolation sur percentiles)
        - Le nombre de compétences (régression linéaire)
        - Le secteur (ajustements sectoriels)
        - Le type de poste (heuristiques métier)
        
        Args:
            profile_data: Dictionnaire du profil
            
        Returns:
            Nombre estimé de mots dans la description
            
        Examples:
            >>> profile = {'experience_final': 5, 'skills_count': 4}
            >>> CalculationUtils.estimate_description_complexity(profile)
            620
        """
        desc_stats = DataDistributions.get_desc_words()
        exp = profile_data.get('experience_final', 4.0)
        
        # === 1. BASE : INTERPOLATION SUR L'EXPÉRIENCE ===
        exp_ranges = [0, 1, 3, 5, 8, 10, 30]
        exp_outputs = [
            desc_stats['p25'],
            desc_stats['p25'],
            desc_stats['median'],
            int((desc_stats['median'] + desc_stats['p75']) / 2),
            desc_stats['p75'],
            desc_stats['p90'],
            desc_stats['p90']
        ]
        
        base_words = CalculationUtils._interpolate(exp, exp_ranges, exp_outputs)
        
        # === 2. AJUSTEMENT COMPÉTENCES (RÉGRESSION) ===
        skills = profile_data.get('skills_count', 3)
        skills_impact = DataDistributions.get_skills_word_impact()
        base_words += int((skills - 3) * skills_impact)  # 3 = baseline
        
        # === 3. AJUSTEMENT SECTORIEL ===
        sector = profile_data.get('sector_clean', '')
        sector_adjustments = DataDistributions.get_sector_adjustments()
        
        if sector in sector_adjustments:
            base_words += sector_adjustments[sector]
        
        # === 4. AJUSTEMENT TYPE DE POSTE ===
        job = profile_data.get('job_type', '')
        
        if any(keyword in job for keyword in ['Lead', 'Manager', 'Architect']):
            base_words += 80
        elif any(keyword in job for keyword in ['Scientist', 'Engineer']):
            base_words += 40
        
        # === 5. CLIPPING ===
        min_words = desc_stats['p10']
        max_words = desc_stats['p90'] + 100
        
        return int(np.clip(base_words, min_words, max_words))
    
    @staticmethod
    def estimate_technical_keywords(profile_data: Dict) -> int:
        """
        Estime le nombre de mots-clés techniques basé sur le profil.
        
        Args:
            profile_data: Dictionnaire du profil
            
        Returns:
            Nombre estimé de mots-clés techniques
        """
        tech_stats = DataDistributions.get_tech_keywords()
        skills = profile_data.get('skills_count', 3)
        
        # === 1. BASE : INTERPOLATION SUR COMPÉTENCES ===
        skill_ranges = [0, 2, 4, 6, 8, 15]
        skill_outputs = [
            tech_stats['median'],
            tech_stats['median'],
            tech_stats['p75'],
            int((tech_stats['p75'] + tech_stats['p90']) / 2),
            tech_stats['p90'],
            tech_stats['p90']
        ]
        
        keywords = CalculationUtils._interpolate(
            skills,
            skill_ranges,
            skill_outputs
        )
        
        # === 2. BONUS COMPÉTENCES AVANCÉES ===
        advanced_count = sum([
            profile_data.get('contient_machine_learning', False),
            profile_data.get('contient_deep_learning', False),
            profile_data.get('contient_spark', False),
            any([
                profile_data.get(f'contient_{cloud}', False)
                for cloud in ['aws', 'azure', 'gcp']
            ])
        ])
        
        if advanced_count >= 3:
            keywords += 1
        
        # === 3. BONUS EXPÉRIENCE ===
        if profile_data.get('experience_final', 4.0) >= 8:
            keywords += 1
        
        # === 4. CLIPPING ===
        return int(np.clip(
            keywords,
            tech_stats['median'],
            tech_stats['p90']
        ))
    
    @staticmethod
    def calculate_skills_count_from_profile(profile: Dict) -> int:
        """
        Calcule le nombre total de compétences dans le profil.
        
        Args:
            profile: Dictionnaire du profil
            
        Returns:
            Nombre de compétences activées
        """
        skill_columns = [
            'contient_sql',
            'contient_python',
            'contient_r',
            'contient_tableau',
            'contient_power_bi',
            'contient_aws',
            'contient_azure',
            'contient_gcp',
            'contient_spark',
            'contient_machine_learning',
            'contient_deep_learning',
            'contient_etl'
        ]
        
        return sum(1 for col in skill_columns if profile.get(col, False))
    
    @staticmethod
    def calculate_technical_score_from_profile(profile: Dict) -> int:
        """
        Calcule le score technique pondéré.
        
        Utilise les poids définis dans FeatureConstants.SKILL_WEIGHTS.
        
        Args:
            profile: Dictionnaire du profil
            
        Returns:
            Score technique (max 15)
        """
        score = 0
        
        for skill, weight in FeatureConstants.SKILL_WEIGHTS.items():
            if profile.get(skill, False):
                score += weight
        
        return min(score, 15)
    
    @staticmethod
    def calculate_benefits_score_from_profile(profile: Dict) -> int:
        """
        Calcule le score d'avantages sociaux.
        
        Args:
            profile: Dictionnaire du profil
            
        Returns:
            Nombre d'avantages (0-4)
        """
        benefit_columns = [
            'has_teletravail',
            'has_mutuelle',
            'has_tickets',
            'has_prime'
        ]
        
        return sum(1 for col in benefit_columns if profile.get(col, False))
    
    @staticmethod
    def get_percentile_real(salary: float, market_data: np.ndarray) -> float:
        """
        Calcule le percentile d'un salaire dans la distribution du marché.
        
        Args:
            salary: Salaire à évaluer
            market_data: Distribution des salaires du marché
            
        Returns:
            Percentile (0-100)
            
        Examples:
            >>> market = np.array([30000, 40000, 50000, 60000, 70000])
            >>> CalculationUtils.get_percentile_real(55000, market)
            60.0
        """
        if len(market_data) == 0:
            return 50.0
        
        percentile = (market_data < salary).sum() / len(market_data) * 100
        
        return float(np.clip(percentile, 0, 100))
    
    @staticmethod
    def create_profile_summary(profile: Dict) -> Dict[str, Any]:
        """
        Crée un résumé formaté du profil pour affichage.
        
        Args:
            profile: Dictionnaire du profil complet
            
        Returns:
            Dict avec informations formatées
            
        Examples:
            >>> profile = {
            ...     'job_type': 'Data Scientist',
            ...     'seniority': 'Senior',
            ...     'location_final': 'Paris'
            ... }
            >>> summary = CalculationUtils.create_profile_summary(profile)
            >>> print(summary['job_info'])
            'Data Scientist - Senior'
        """
        # Extraction des compétences clés
        key_skills_map = [
            ("Python", 'contient_python'),
            ("SQL", 'contient_sql'),
            ("ML", 'contient_machine_learning'),
            ("DL", 'contient_deep_learning'),
            ("AWS", 'contient_aws')
        ]
        
        key_skills = [
            name for name, key in key_skills_map
            if profile.get(key, False)
        ]
        
        return {
            'job_info': (
                f"{profile.get('job_type', 'N/A')} - "
                f"{profile.get('seniority', 'N/A')}"
            ),
            'location_sector': (
                f"{profile.get('location_final', 'N/A')} "
                f"({profile.get('sector_clean', 'N/A')})"
            ),
            'education_exp': (
                f"{profile.get('education_clean', 'N/A')} - "
                f"{profile.get('experience_final', 0):.1f} ans"
            ),
            'skills_count': CalculationUtils.calculate_skills_count_from_profile(
                profile
            ),
            'tech_score': CalculationUtils.calculate_technical_score_from_profile(
                profile
            ),
            'benefits_score': CalculationUtils.calculate_benefits_score_from_profile(
                profile
            ),
            'telework': (
                "Oui" if profile.get('telework_numeric', 0) > 0 else "Non"
            ),
            'key_skills': ", ".join(key_skills) if key_skills else "N/A"
        }


# ============================================================================
# INITIALISATION
# ============================================================================

def init_model_utils() -> ModelUtils:
    """
    Initialise les utilitaires du modèle et affiche les métriques.
    
    Cette fonction :
    1. Crée une instance de ModelUtils
    2. Affiche les performances dans la sidebar
    3. Affiche les secteurs bien payés
    
    Returns:
        Instance de ModelUtils initialisée
        
    Examples:
        >>> utils = init_model_utils()
        >>> prediction = utils.predict({'job_type': 'Data Scientist'})
    """
    utils = ModelUtils()
    
    if utils.model:
        perf = utils.get_model_performance()
        
        with st.sidebar.expander("📊 Performance du modèle", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("R²", f"{perf.get('test_r2', 0):.3f}")
                st.metric("MAE", f"{perf.get('test_mae', 0):,.0f} €")
            
            with col2:
                st.metric("Stabilité", f"{perf.get('stability', 0):.1%}")
                st.metric("CV Std", f"{perf.get('cv_mae_std', 0):,.0f} €")
            
            # Informations additionnelles
            st.markdown("---")
            st.caption(
                f"**Secteurs bien payés (P75)** : "
                f"{', '.join(DataDistributions.get_high_paying_sectors()[:3])}"
            )
    
    return utils


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'ModelUtils',
    'ChartUtils',
    'CalculationUtils',
    'DataDistributions',
    'FeatureConstants',
    'init_model_utils'
]
