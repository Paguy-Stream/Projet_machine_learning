"""
Script de nettoyage et feature engineering - Partie 2.

Ce fichier contient la classe principale DataCleaner et les transformateurs.
"""

import pandas as pd
import numpy as np
import ast
import re
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any

from data_cleaning_refactored_part1 import (
    CleaningConfig, ExperienceExtractor, CompanyExtractor,
    LocationExtractor, JobTypeClassifier, SeniorityExtractor
)


# ============================================================================
# TRANSFORMATEURS DE DONNÉES
# ============================================================================

class DataTransformers:
    """
    Collection de transformateurs pour les données.
    
    Cette classe fournit des méthodes statiques pour transformer
    différents types de colonnes.
    """
    
    @staticmethod
    def parse_python_list(value: Any, default: Optional[List] = None) -> List:
        """
        Parse une chaîne représentant une liste Python.
        
        Args:
            value: Valeur à parser
            default: Valeur par défaut si parsing échoue
            
        Returns:
            Liste parsée ou valeur par défaut
            
        Example:
            >>> parse_python_list("['Python', 'SQL']")
            ['Python', 'SQL']
        """
        if pd.isna(value):
            return default if default is not None else []
        
        if isinstance(value, list):
            return value
        
        value_str = str(value).strip()
        
        if value_str.startswith('[') and value_str.endswith(']'):
            try:
                parsed = ast.literal_eval(value_str)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except (SyntaxError, ValueError):
                pass
        
        if value_str in ['', '[]', 'nan', 'NaN', 'None']:
            return []
        
        return [value_str]
    
    @staticmethod
    def clean_salary(value: Any) -> Optional[float]:
        """
        Nettoie les valeurs de salaires (supporte k€ et €).
        
        Args:
            value: Valeur de salaire à nettoyer
            
        Returns:
            Salaire en euros ou None
            
        Example:
            >>> clean_salary("45k€")
            45000.0
        """
        if pd.isna(value):
            return None
        
        value_str = str(value).lower().replace(' ', '')
        
        # Format k€
        if 'k€' in value_str or 'k' in value_str:
            match = re.search(r'(\d+\.?\d*)\s*k', value_str)
            if match:
                try:
                    return float(match.group(1)) * 1000
                except:
                    return None
        
        # Format standard
        numbers = re.findall(r'[\d\s]+', value_str.replace(' ', ''))
        if numbers:
            try:
                return int(numbers[0])
            except ValueError:
                return None
        
        return None
    
    @staticmethod
    def normalize_education(value: Any) -> str:
        """
        Normalise les niveaux d'éducation.
        
        Args:
            value: Niveau d'éducation brut
            
        Returns:
            Niveau normalisé
            
        Example:
            >>> normalize_education("master 2")
            "Bac+5"
        """
        if pd.isna(value):
            return 'Non spécifié'
        
        value_str = str(value).strip().lower()
        
        mapping = {
            'bac+5': 'Bac+5', 'master': 'Bac+5', 'ingénieur': 'Bac+5',
            'bac+3': 'Bac+3', 'licence': 'Bac+3', 'bachelor': 'Bac+3',
            'bac+4': 'Bac+4', 'master 1': 'Bac+4',
            'bac+2': 'Bac+2', 'dut': 'Bac+2', 'bts': 'Bac+2',
            'bac': 'Bac',
            'doctorat': 'Doctorat', 'phd': 'Doctorat', 'bac+8': 'Doctorat',
        }
        
        for key, normalized in mapping.items():
            if key in value_str:
                return normalized
        
        return value_str.title()
    
    @staticmethod
    def telework_to_numeric(value: Any) -> float:
        """
        Convertit le télétravail en valeur numérique.
        
        Args:
            value: Valeur de télétravail (texte)
            
        Returns:
            Valeur entre 0 et 1
            
        Example:
            >>> telework_to_numeric("Télétravail partiel")
            0.5
        """
        if pd.isna(value):
            return 0.0
        
        value_str = str(value).lower()
        
        if '100%' in value_str or 'full remote' in value_str:
            return 1.0
        elif 'partiel' in value_str or 'hybride' in value_str:
            return 0.5
        elif 'non' in value_str or 'présentiel' in value_str:
            return 0.0
        
        return 0.0


class DescriptionCleaner:
    """
    Nettoyeur de descriptions d'offres d'emploi.
    
    Supprime les headers répétitifs, les footers et normalise le texte.
    
    Example:
        >>> cleaner = DescriptionCleaner()
        >>> text_clean = cleaner.clean(description)
    """
    
    def __init__(self):
        """Initialise avec les patterns à supprimer."""
        self.patterns_to_remove = [
            r'Trouver mon\s*j\s*ob.*?Mon compte\s*',
            r'Créez votre compte.*?candidature',
            r'Ces offres pourraient.*?intéresser',
            r'L\'entreprise\s*Qui sommes-nous\?.*?(?=Le job|Détail|$)',
            r'Content missing',
            r'Publiée le \d{2}/\d{2}/\d{4}.*',
            r'Réf\s*:.*',
            r'Job ID:.*',
        ]
    
    def clean(self, description: str) -> str:
        """
        Nettoie une description.
        
        Args:
            description: Texte brut de la description
            
        Returns:
            Texte nettoyé
        """
        if pd.isna(description):
            return ""
        
        text = str(description)
        
        # Supprimer les patterns
        for pattern in self.patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Normaliser les espaces
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def extract_info(self, description: str) -> Dict[str, Any]:
        """
        Extrait les informations clés de la description.
        
        Args:
            description: Description nettoyée
            
        Returns:
            Dictionnaire avec les informations extraites
        """
        if not description:
            return {}
        
        text_lower = description.lower()
        
        # Détection de compétences
        info = {
            'contient_sql': 'sql' in text_lower,
            'contient_python': 'python' in text_lower,
            'contient_r': ' r ' in text_lower or 'r,' in text_lower,
            'contient_tableau': 'tableau' in text_lower,
            'contient_power_bi': 'power bi' in text_lower or 'powerbi' in text_lower,
            'contient_aws': 'aws' in text_lower,
            'contient_azure': 'azure' in text_lower,
            'contient_gcp': 'gcp' in text_lower or 'google cloud' in text_lower,
            'contient_spark': 'spark' in text_lower,
            'contient_machine_learning': 'machine learning' in text_lower or 'ml ' in text_lower,
            'contient_deep_learning': 'deep learning' in text_lower,
            'contient_etl': 'etl' in text_lower or 'elt' in text_lower,
        }
        
        # Métriques de texte
        info['longueur_description'] = len(description)
        info['mots_description'] = len(description.split())
        
        # Compter mots-clés techniques
        keywords = ['sql', 'python', 'r', 'tableau', 'power bi', 'aws', 
                   'azure', 'gcp', 'spark', 'hadoop', 'kafka', 'docker']
        info['nb_mots_cles_techniques'] = sum(1 for kw in keywords if kw in text_lower)
        
        return info


# ============================================================================
# CLASSE PRINCIPALE DE NETTOYAGE
# ============================================================================

class DataCleaner:
    """
    Classe principale pour le nettoyage complet des données.
    
    Cette classe orchestre tout le pipeline de nettoyage et 
    feature engineering.
    
    Attributes:
        input_path (Path): Chemin du fichier d'entrée
        output_path (Path): Chemin du fichier de sortie
        config (CleaningConfig): Configuration
        df (pd.DataFrame): DataFrame original
        df_clean (pd.DataFrame): DataFrame nettoyé
        
    Example:
        >>> cleaner = DataCleaner(input_path, output_path)
        >>> df_clean = cleaner.run_full_pipeline()
        >>> cleaner.save_results()
    """
    
    def __init__(self, input_path: Path, output_path: Path):
        """
        Initialise le cleaner.
        
        Args:
            input_path: Chemin vers le CSV brut
            output_path: Chemin vers le CSV nettoyé
        """
        self.input_path = input_path
        self.output_path = output_path
        self.config = CleaningConfig()
        
        # Initialiser les extracteurs
        self.exp_extractor = ExperienceExtractor()
        self.company_extractor = CompanyExtractor()
        self.location_extractor = LocationExtractor()
        self.job_classifier = JobTypeClassifier()
        self.seniority_extractor = SeniorityExtractor()
        self.desc_cleaner = DescriptionCleaner()
        
        self.df = None
        self.df_clean = None
        self.quality_report = {}
    
    def load_data(self) -> pd.DataFrame:
        """
        Charge les données brutes.
        
        Returns:
            DataFrame original
            
        Raises:
            FileNotFoundError: Si le fichier n'existe pas
        """
        print(f"📂 Chargement des données depuis {self.input_path}")
        
        if not self.input_path.exists():
            raise FileNotFoundError(f"Fichier non trouvé: {self.input_path}")
        
        self.df = pd.read_csv(self.input_path, encoding='utf-8')
        print(f"✅ {len(self.df):,} lignes × {len(self.df.columns)} colonnes")
        
        return self.df
    
    def clean_columns(self) -> None:
        """Supprime les colonnes inutiles."""
        print("\n🗑️  Suppression des colonnes inutiles...")
        
        cols_to_drop = [col for col in self.config.COLUMNS_TO_DROP 
                       if col in self.df_clean.columns]
        
        if cols_to_drop:
            self.df_clean = self.df_clean.drop(columns=cols_to_drop)
            print(f"   • {len(cols_to_drop)} colonnes supprimées")
    
    def parse_lists(self) -> None:
        """Parse les colonnes au format liste Python."""
        print("\n🔧 Parsing des listes Python...")
        
        # Skills
        if 'skills' in self.df_clean.columns:
            self.df_clean['skills_parsed'] = self.df_clean['skills'].apply(
                DataTransformers.parse_python_list
            )
            self.df_clean['skills_count'] = self.df_clean['skills_parsed'].apply(len)
        
        # Benefits
        if 'benefits' in self.df_clean.columns:
            self.df_clean['benefits_parsed'] = self.df_clean['benefits'].apply(
                DataTransformers.parse_python_list
            )
            self.df_clean['benefits_count'] = self.df_clean['benefits_parsed'].apply(len)
    
    def clean_salaries(self) -> None:
        """Nettoie et calcule les salaires."""
        print("\n💰 Nettoyage des salaires...")
        
        if 'salary_min' in self.df_clean.columns:
            self.df_clean['salary_min_clean'] = self.df_clean['salary_min'].apply(
                DataTransformers.clean_salary
            )
        
        if 'salary_max' in self.df_clean.columns:
            self.df_clean['salary_max_clean'] = self.df_clean['salary_max'].apply(
                DataTransformers.clean_salary
            )
        
        # Calculer le salaire moyen
        if all(col in self.df_clean.columns for col in ['salary_min_clean', 'salary_max_clean']):
            self.df_clean['salary_mid'] = (
                self.df_clean['salary_min_clean'] + 
                self.df_clean['salary_max_clean']
            ) / 2
            self.df_clean['salary_range'] = (
                self.df_clean['salary_max_clean'] - 
                self.df_clean['salary_min_clean']
            )
    
    def extract_experience(self) -> None:
        """Extrait et corrige l'expérience."""
        print("\n📅 Extraction de l'expérience...")
        
        # Extraire depuis toutes les sources
        self.df_clean['experience_extracted'] = self.df_clean.apply(
            self.exp_extractor.extract_from_all_sources, axis=1
        )
        
        # Corriger les valeurs extrêmes
        self.df_clean['experience_final'] = self.df_clean['experience_extracted'].apply(
            self.exp_extractor.correct_extreme_values
        )
        
        # Statistiques
        exp_count = self.df_clean['experience_final'].notna().sum()
        exp_pct = exp_count / len(self.df_clean) * 100
        print(f"   • Expérience extraite : {exp_count:,} ({exp_pct:.1f}%)")
    
    def extract_from_title(self) -> None:
        """Extrait les informations depuis le titre."""
        print("\n🎯 Extraction depuis le titre...")
        
        if 'title' not in self.df_clean.columns:
            return
        
        # Entreprise
        self.df_clean['company_extracted'] = self.df_clean['title'].apply(
            self.company_extractor.extract
        )
        
        # Type de poste
        self.df_clean['job_type'] = self.df_clean['title'].apply(
            self.job_classifier.classify
        )
        
        # Séniorité
        self.df_clean['seniority'] = self.df_clean['title'].apply(
            self.seniority_extractor.extract
        )
        
        # Détection poste Data
        self.df_clean['is_data_job'] = self.df_clean['job_type'].apply(
            lambda x: x != 'Autre'
        )
        
        print(f"   • {self.df_clean['company_extracted'].notna().sum():,} entreprises extraites")
        print(f"   • {self.df_clean['is_data_job'].sum():,} postes Data identifiés")
    
    def process_description(self) -> None:
        """Nettoie et analyse les descriptions."""
        print("\n📝 Traitement des descriptions...")
        
        if 'description' not in self.df_clean.columns:
            return
        
        # Nettoyer
        self.df_clean['description_cleaned'] = self.df_clean['description'].apply(
            self.desc_cleaner.clean
        )
        
        # Extraire informations
        desc_info = self.df_clean['description_cleaned'].apply(
            self.desc_cleaner.extract_info
        )
        
        # Créer colonnes
        for key in ['contient_sql', 'contient_python', 'contient_r', 
                   'contient_tableau', 'contient_power_bi', 'contient_aws',
                   'nb_mots_cles_techniques']:
            if desc_info.size > 0 and key in desc_info.iloc[0]:
                self.df_clean[key] = desc_info.apply(lambda x: x.get(key, False))
        
        # Longueur
        self.df_clean['description_word_count'] = desc_info.apply(
            lambda x: x.get('mots_description', 0)
        )
    
    def create_features(self) -> None:
        """Crée les features supplémentaires."""
        print("\n✨ Création de features...")
        
        # Normaliser éducation
        if 'education' in self.df_clean.columns:
            self.df_clean['education_clean'] = self.df_clean['education'].apply(
                DataTransformers.normalize_education
            )
        
        # Télétravail numérique
        if 'telework' in self.df_clean.columns:
            self.df_clean['telework_numeric'] = self.df_clean['telework'].apply(
                DataTransformers.telework_to_numeric
            )
        
        # Localisation finale
        if 'search_location' in self.df_clean.columns:
            self.df_clean['location_final'] = self.df_clean['search_location'].fillna(
                'Non spécifié'
            )
            
            # Grandes villes
            self.df_clean['is_grande_ville'] = self.df_clean['location_final'].apply(
                lambda x: any(ville in str(x) for ville in self.config.MAJOR_CITIES) 
                if pd.notna(x) else False
            )
    
    def calculate_quality_scores(self) -> None:
        """Calcule les scores de qualité."""
        print("\n📊 Calcul des scores de qualité...")
        
        # Score de complétude
        completeness_cols = ['salary_mid', 'experience_final', 
                            'skills_count', 'company_extracted']
        existing_cols = [col for col in completeness_cols 
                        if col in self.df_clean.columns]
        
        if existing_cols:
            self.df_clean['profile_completeness'] = (
                self.df_clean[existing_cols].notna().sum(axis=1) / 
                len(existing_cols) * 100
            )
    
    def run_full_pipeline(self) -> pd.DataFrame:
        """
        Exécute le pipeline complet de nettoyage.
        
        Returns:
            DataFrame nettoyé
        """
        print("="*80)
        print("🧹 PIPELINE DE NETTOYAGE COMPLET")
        print("="*80)
        
        # 1. Charger
        self.load_data()
        self.df_clean = self.df.copy()
        
        # 2. Nettoyer
        self.clean_columns()
        self.parse_lists()
        self.clean_salaries()
        
        # 3. Extraire
        self.extract_experience()
        self.extract_from_title()
        self.process_description()
        
        # 4. Features
        self.create_features()
        self.calculate_quality_scores()
        
        print("\n✅ Pipeline terminé !")
        return self.df_clean
    
    def save_results(self) -> None:
        """Sauvegarde les résultats."""
        print("\n💾 Sauvegarde des résultats...")
        
        self.df_clean.to_csv(self.output_path, index=False, encoding='utf-8')
        print(f"✅ Données sauvegardées : {self.output_path}")
    
    def generate_quality_report(self) -> Dict:
        """
        Génère un rapport de qualité.
        
        Returns:
            Dictionnaire avec les métriques de qualité
        """
        print("\n📋 Génération du rapport de qualité...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_stats': {
                'original_rows': len(self.df),
                'final_rows': len(self.df_clean),
                'final_columns': len(self.df_clean.columns),
            }
        }
        
        # Taux de remplissage
        key_cols = ['job_type', 'salary_mid', 'experience_final', 
                   'company_extracted', 'location_final']
        
        for col in key_cols:
            if col in self.df_clean.columns:
                fill_rate = self.df_clean[col].notna().sum() / len(self.df_clean) * 100
                report[f'fill_rate_{col}'] = float(fill_rate)
        
        # Sauvegarder
        report_path = self.config.REPORT_PATH
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Rapport sauvegardé : {report_path}")
        return report


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

def main():
    """Point d'entrée principal du script."""
    config = CleaningConfig()
    
    # Créer le cleaner
    cleaner = DataCleaner(config.DATA_PATH, config.OUTPUT_PATH)
    
    # Exécuter le pipeline
    df_clean = cleaner.run_full_pipeline()
    
    # Sauvegarder
    cleaner.save_results()
    
    # Rapport
    report = cleaner.generate_quality_report()
    
    print("\n" + "="*80)
    print("✅ NETTOYAGE TERMINÉ AVEC SUCCÈS")
    print("="*80)
    
    return df_clean, report


if __name__ == "__main__":
    main()
