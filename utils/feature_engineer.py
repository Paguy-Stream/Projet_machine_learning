"""
utils/feature_engineer.py
Transformateur personnalisé pour le feature engineering
Évite la fuite de données en apprenant UNIQUEMENT sur le train set
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class SafeFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformateur qui calcule les features UNIQUEMENT sur le train set
    et applique les mêmes transformations au test set
    
    Prévient la fuite de données (data leakage) en séparant fit() et transform()
    """
    
    def __init__(self):
        """Initialiser les paramètres qui seront appris pendant fit()"""
        self.exp_medians_ = {}
        self.global_exp_median_ = None
        self.paris_codes_ = ['75', '77', '78', '91', '92', '93', '94', '95']
        self.high_paying_sectors_ = ['Banque', 'Finance', 'Tech', 'Consulting', 'Assurance', 'Pharma']

    def fit(self, X, y=None):
        """
        Apprend les paramètres UNIQUEMENT sur le train set
        
        Args:
            X: DataFrame d'entraînement
            y: Target (non utilisé mais requis par l'interface sklearn)
            
        Returns:
            self: pour le chaînage de méthodes
        """
        X_copy = X.copy()

        # Calculer la médiane globale d'expérience
        if 'experience_final' in X_copy.columns:
            self.global_exp_median_ = X_copy['experience_final'].median()

            # Calculer médiane par seniority (sans fuite)
            if 'seniority' in X_copy.columns:
                seniority_mapping = {
                    'Débutant (<1 an)': 1, 
                    'Junior (1-3 ans)': 2, 
                    'Mid confirmé (3-5 ans)': 3,
                    'Senior (5-8 ans)': 4, 
                    'Expert (8-12 ans)': 5, 
                    'Lead/Manager (12-20 ans)': 6,
                    'Directeur/VP (>20 ans)': 7, 
                    'Non spécifié': 0, 
                    'Stage/Alternance': 1,
                    'Freelance/Consultant': 3
                }
                X_copy['seniority_numeric'] = X_copy['seniority'].map(
                    lambda x: seniority_mapping.get(x, 0)
                )

                # Calculer médianes par groupe
                for sen in X_copy['seniority_numeric'].unique():
                    mask = X_copy['seniority_numeric'] == sen
                    exp_values = X_copy.loc[mask, 'experience_final'].dropna()
                    if len(exp_values) > 0:
                        self.exp_medians_[sen] = exp_values.median()

        return self

    def transform(self, X):
        """
        Applique les transformations avec les paramètres appris sur le train set
        
        Args:
            X: DataFrame à transformer (train ou test)
            
        Returns:
            X_transformed: DataFrame transformé
        """
        X_transformed = X.copy()

        # =============================================
        # 1. CONVERTIR LES BOOLÉENS EN INT
        # =============================================
        bool_cols = [col for col in X_transformed.columns if X_transformed[col].dtype == 'bool']
        for col in bool_cols:
            X_transformed[col] = X_transformed[col].astype(int)

        # =============================================
        # 2. FEATURE ENGINEERING SUR L'EXPÉRIENCE
        # =============================================
        if 'experience_final' in X_transformed.columns:
            # Indicateur de valeur manquante
            X_transformed['experience_missing'] = X_transformed['experience_final'].isna().astype(int)

            if 'seniority' in X_transformed.columns:
                # Mapper seniority en numérique
                seniority_mapping = {
                    'Débutant (<1 an)': 1, 
                    'Junior (1-3 ans)': 2, 
                    'Mid confirmé (3-5 ans)': 3,
                    'Senior (5-8 ans)': 4, 
                    'Expert (8-12 ans)': 5, 
                    'Lead/Manager (12-20 ans)': 6,
                    'Directeur/VP (>20 ans)': 7, 
                    'Non spécifié': 0, 
                    'Stage/Alternance': 1,
                    'Freelance/Consultant': 3
                }
                X_transformed['seniority_numeric'] = X_transformed['seniority'].map(
                    lambda x: seniority_mapping.get(x, 0)
                )

                # Imputer avec médianes du train set (apprises pendant fit)
                X_transformed['experience_final_imputed'] = X_transformed['experience_final'].copy()
                for sen, median_val in self.exp_medians_.items():
                    mask = (X_transformed['seniority_numeric'] == sen) & (X_transformed['experience_final'].isna())
                    X_transformed.loc[mask, 'experience_final_imputed'] = median_val

                # Fallback avec médiane globale pour les cas non couverts
                still_missing = X_transformed['experience_final_imputed'].isna()
                X_transformed.loc[still_missing, 'experience_final_imputed'] = self.global_exp_median_
            else:
                # Si pas de seniority, utiliser directement la médiane globale
                X_transformed['experience_final_imputed'] = X_transformed['experience_final'].fillna(
                    self.global_exp_median_
                )

        # =============================================
        # 3. INTERACTION TECHNIQUE-EXPÉRIENCE
        # =============================================
        if 'technical_score' in X_transformed.columns and 'experience_final_imputed' in X_transformed.columns:
            X_transformed['tech_exp_interaction'] = (
                X_transformed['technical_score'] * np.log1p(X_transformed['experience_final_imputed'])
            )

        # =============================================
        # 4. NETTOYER NB_MOTS_CLES_TECHNIQUES
        # =============================================
        if 'nb_mots_cles_techniques' in X_transformed.columns:
            if not pd.api.types.is_numeric_dtype(X_transformed['nb_mots_cles_techniques']):
                X_transformed['nb_mots_cles_techniques'] = pd.to_numeric(
                    X_transformed['nb_mots_cles_techniques'], errors='coerce'
                )
            X_transformed['nb_mots_cles_techniques'] = X_transformed['nb_mots_cles_techniques'].fillna(0)

        # =============================================
        # 5. RÉGION ÉCONOMIQUE (ÎLE-DE-FRANCE)
        # =============================================
        if 'location_final' in X_transformed.columns:
            X_transformed['is_paris_region'] = X_transformed['location_final'].apply(
                lambda x: 1 if any(code in str(x) for code in self.paris_codes_) else 0
            )

        # =============================================
        # 6. SCORE DE COMPÉTENCES DATA AVANCÉES
        # =============================================
        advanced_skills = ['contient_machine_learning', 'contient_spark', 'contient_aws']
        if all(col in X_transformed.columns for col in advanced_skills):
            X_transformed['advanced_data_score'] = X_transformed[advanced_skills].sum(axis=1)

        # =============================================
        # 7. TYPE D'ENTREPRISE (SECTEUR À HAUT SALAIRE)
        # =============================================
        if 'sector_clean' in X_transformed.columns:
            X_transformed['is_high_paying_sector'] = X_transformed['sector_clean'].apply(
                lambda x: 1 if x in self.high_paying_sectors_ else 0
            )

        # =============================================
        # 8. COMPLEXITÉ TECHNIQUE DU POSTE
        # =============================================
        if 'nb_mots_cles_techniques' in X_transformed.columns:
            # Créer des niveaux de complexité
            X_transformed['tech_complexity'] = pd.cut(
                X_transformed['nb_mots_cles_techniques'],
                bins=[-1, 2, 5, 10, float('inf')],
                labels=['Faible', 'Moyenne', 'Élevée', 'Très élevée']
            )

        # =============================================
        # 9. STACK MODERNE (PYTHON + CLOUD + BIG DATA)
        # =============================================
        modern_stack = ['contient_python', 'contient_aws', 'contient_spark']
        if all(col in X_transformed.columns for col in modern_stack):
            # Vérifier si présence de au moins un cloud provider
            has_cloud = False
            if 'contient_aws' in X_transformed.columns:
                has_cloud = X_transformed['contient_aws']
            if 'contient_azure' in X_transformed.columns:
                has_cloud = has_cloud | X_transformed['contient_azure']
            if 'contient_gcp' in X_transformed.columns:
                has_cloud = has_cloud | X_transformed['contient_gcp']
            
            X_transformed['has_modern_stack'] = (
                X_transformed['contient_python'] & 
                has_cloud &
                X_transformed['contient_spark']
            ).astype(int)

        # =============================================
        # 10. SCORE HIÉRARCHIQUE AMÉLIORÉ
        # =============================================
        if 'seniority' in X_transformed.columns:
            hierarchy_mapping = {
                'Stage/Alternance': 1, 
                'Débutant (<1 an)': 2, 
                'Junior (1-3 ans)': 3,
                'Mid confirmé (3-5 ans)': 4, 
                'Senior (5-8 ans)': 5, 
                'Expert (8-12 ans)': 6,
                'Lead/Manager (12-20 ans)': 7, 
                'Directeur/VP (>20 ans)': 8
            }
            X_transformed['hierarchy_score'] = X_transformed['seniority'].map(
                lambda x: hierarchy_mapping.get(x, 0)
            )

        return X_transformed

    def get_feature_names_out(self, input_features=None):
        """
        Retourne les noms des features après transformation
        (Requis pour compatibilité sklearn >= 1.0)
        """
        if input_features is None:
            return None
        
        # Liste des nouvelles features créées
        new_features = [
            'experience_missing',
            'seniority_numeric', 
            'experience_final_imputed',
            'tech_exp_interaction',
            'is_paris_region',
            'advanced_data_score',
            'is_high_paying_sector',
            'tech_complexity',
            'has_modern_stack',
            'hierarchy_score'
        ]
        
        # Combiner features originales + nouvelles
        all_features = list(input_features) + [f for f in new_features if f not in input_features]
        
        return np.array(all_features)


# ============================================================================
# FONCTIONS UTILITAIRES POUR TESTER LE TRANSFORMATEUR
# ============================================================================

def test_safe_feature_engineer():
    """
    Fonction de test pour vérifier le bon fonctionnement du transformateur
    """
    print("🧪 Test de SafeFeatureEngineer")
    print("=" * 60)
    
    # Créer des données de test
    test_data = pd.DataFrame({
        'experience_final': [2.0, np.nan, 5.0, np.nan, 10.0],
        'seniority': ['Junior (1-3 ans)', 'Mid confirmé (3-5 ans)', 'Senior (5-8 ans)', 
                     'Junior (1-3 ans)', 'Expert (8-12 ans)'],
        'technical_score': [2, 3, 5, 2, 7],
        'location_final': ['Paris', 'Lyon', '75001', 'Marseille', '92100'],
        'sector_clean': ['Banque', 'Tech', 'Retail', 'Finance', 'Industrie'],
        'nb_mots_cles_techniques': [3, 5, 2, 8, 12],
        'contient_python': [True, True, False, True, True],
        'contient_aws': [False, True, False, True, True],
        'contient_azure': [True, False, False, False, False],
        'contient_spark': [False, True, False, False, True],
        'contient_machine_learning': [True, True, False, True, True]
    })
    
    # Initialiser le transformateur
    fe = SafeFeatureEngineer()
    
    # Fit sur les données
    print("\n1️⃣ Fit du transformateur...")
    fe.fit(test_data)
    print(f"   ✓ Médiane globale exp: {fe.global_exp_median_:.2f}")
    print(f"   ✓ Médianes par seniority: {fe.exp_medians_}")
    
    # Transform
    print("\n2️⃣ Transform des données...")
    transformed = fe.transform(test_data)
    print(f"   ✓ Shape: {transformed.shape}")
    print(f"   ✓ Nouvelles colonnes créées:")
    
    new_cols = set(transformed.columns) - set(test_data.columns)
    for col in sorted(new_cols):
        print(f"      - {col}")
    
    # Afficher quelques résultats
    print("\n3️⃣ Aperçu des transformations:")
    cols_to_show = ['experience_final', 'experience_final_imputed', 
                    'is_paris_region', 'is_high_paying_sector']
    print(transformed[cols_to_show].head())
    
    print("\n✅ Test terminé avec succès!")
    
    return fe, transformed


if __name__ == "__main__":
    # Exécuter les tests si le fichier est lancé directement
    test_safe_feature_engineer()