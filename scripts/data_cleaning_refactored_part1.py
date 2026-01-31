"""
Script de nettoyage et feature engineering pour données de jobs Data.

Ce script transforme les données brutes scrapées en un dataset propre et enrichi,
prêt pour l'analyse et la modélisation. Il effectue :
- Nettoyage et parsing des données
- Extraction intelligente d'informations (expérience, entreprise, localisation)
- Feature engineering avancé
- Création de métriques de qualité

Architecture :
    - Configuration centralisée
    - Fonctions modulaires et réutilisables
    - Pipeline de transformation claire
    - Validation et rapports de qualité


Example:
    >>> cleaner = DataCleaner(input_path, output_path)
    >>> df_clean = cleaner.run_full_pipeline()
    >>> cleaner.generate_quality_report()
"""

import pandas as pd
import numpy as np
import ast
import re
from pathlib import Path
import json
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

class CleaningConfig:
    """Configuration centralisée pour le nettoyage des données."""
    
    # Dossier des données
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = BASE_DIR / "output"
    ANALYSIS_DIR = OUTPUT_DIR / "analysis_complete"

# Fichiers
    DATA_PATH = DATA_DIR / "hellowork_ultra_20260111_105253.csv"
    OUTPUT_PATH = OUTPUT_DIR / "hellowork_cleaned_improved.csv"
    REPORT_PATH = ANALYSIS_DIR / "etape2_rapport_amelioré.json"

    
    # Colonnes à supprimer
    COLUMNS_TO_DROP = [
        'company', 'location', 'salary', 'date_posted',
        'missions', 'profile', 'advantages', 'hash', 'contract'
    ]
    
    # Grandes villes françaises
    MAJOR_CITIES = [
        'Paris', 'Lyon', 'Marseille', 'Toulouse', 'Bordeaux',
        'Lille', 'Nice', 'Nantes', 'Strasbourg', 'Rennes',
        'Grenoble', 'Toulon', 'Montpellier'
    ]
    
    # Seuils de correction d'expérience
    EXPERIENCE_THRESHOLDS = {
        'max_realistic': 25.0,
        'extreme_high': 50.0,
        'very_high': 40.0,
        'high': 30.0,
        'senior': 20.0
    }


# ============================================================================
# EXTRACTEURS D'INFORMATION
# ============================================================================

class ExperienceExtractor:
    """
    Extracteur intelligent d'expérience depuis différentes sources de texte.
    
    Cette classe implémente des patterns avancés pour détecter et extraire
    les années d'expérience depuis des textes variés (titres, descriptions, etc.).
    
    Attributes:
        seniority_map (Dict[str, float]): Mapping des niveaux vers des années
        
    Example:
        >>> extractor = ExperienceExtractor()
        >>> exp = extractor.extract("5 ans d'expérience minimum")
        >>> print(exp)  # 5.0
    """
    
    def __init__(self):
        """Initialise l'extracteur avec les mappings de niveaux."""
        self.seniority_map = {
            'junior': 1.0,
            'débutant': 1.0,
            'jeune diplômé': 1.0,
            'entry level': 1.0,
            'confirmé': 5.0,
            'expérimenté': 8.0,
            'senior': 8.0,
            'sénior': 8.0,
            'expert': 10.0,
            'lead': 10.0,
            'manager': 10.0,
            'chef': 10.0,
            'responsable': 10.0,
        }
    
    def extract(self, text: str) -> Optional[float]:
        """
        Extrait l'expérience depuis n'importe quel texte.
        
        Args:
            text: Texte source (titre, description, etc.)
            
        Returns:
            Nombre d'années d'expérience ou None si non trouvé
            
        Notes:
            Utilise plusieurs stratégies :
            1. Patterns spécifiques (ex: "3-5 ans d'expérience")
            2. Niveaux qualitatifs (ex: "junior", "senior")
            3. Patterns généraux
        """
        if pd.isna(text) or not text or str(text).lower() == 'nan':
            return None
        
        text_clean = self._clean_text(text)
        
        # Stratégie 1: Patterns spécifiques
        exp = self._extract_from_specific_patterns(text_clean)
        if exp is not None:
            return exp
        
        # Stratégie 2: Niveaux qualitatifs
        exp = self._extract_from_seniority_levels(text_clean)
        if exp is not None:
            return exp
        
        # Stratégie 3: Patterns généraux
        exp = self._extract_from_general_patterns(text_clean)
        return exp
    
    def _clean_text(self, text: str) -> str:
        """Nettoie le texte pour extraction."""
        text_clean = str(text).lower()
        text_clean = re.sub(r'\s+', ' ', text_clean.replace('\n', ' ').replace('\r', ' '))
        return text_clean.strip()
    
    def _extract_from_specific_patterns(self, text: str) -> Optional[float]:
        
        """Extrait depuis patterns spécifiques d'expérience."""
        patterns = [
            # Formats avec "à" ou "-"
            (r"(\d+)\s*[àa-]\s*(\d+)\s*ans?\s*d['’]?exp[eé]rience", "range"),
            (r"(\d+)\s*[àa-]\s*(\d+)\s*ans?\s*d['’]?exp", "range"),

        # Formats simples
            (r"(\d+)\s*ans?\s*d['’]?exp[eé]rience", "single"),
            (r"(\d+)ans?\s*d['’]?exp", "single"),

        # Formats avec minimum/requis
            (r"minimum\s*(\d+)\s*ans?", "single"),
            (r"(\d+)\s*ans?\s*minimum", "single"),
            (r"exp[eé]rience\s*:\s*(\d+)\s*ans?", "single"),
        ]
        
        for pattern, pattern_type in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    if pattern_type == 'range':
                        min_val, max_val = int(match.group(1)), int(match.group(2))
                        if self._is_valid_range(min_val, max_val):
                            return np.mean([min_val, max_val])
                    elif pattern_type == 'single':
                        val = int(match.group(1))
                        if self._is_valid_value(val):
                            return float(val)
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def _extract_from_seniority_levels(self, text: str) -> Optional[float]:
        """Extrait depuis niveaux de séniorité."""
        for level, years in self.seniority_map.items():
            if re.search(rf'\b{level}\b', text, re.IGNORECASE):
                return years
        return None
    
    def _extract_from_general_patterns(self, text: str) -> Optional[float]:
        """Extrait depuis patterns généraux."""
        patterns = [
            (r'(\d+)\s*à\s*(\d+)\s*ans?', 'range'),
            (r'(\d+)\s*-\s*(\d+)\s*ans?', 'range'),
            (r'(\d+)\s*ans?', 'single'),
        ]
        
        for pattern, pattern_type in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    if pattern_type == 'range':
                        min_val, max_val = int(match.group(1)), int(match.group(2))
                        if self._is_valid_range(min_val, max_val):
                            return np.mean([min_val, max_val])
                    elif pattern_type == 'single':
                        val = int(match.group(1))
                        if self._is_valid_value(val):
                            return float(val)
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def _is_valid_value(self, value: int) -> bool:
        """Vérifie si une valeur d'expérience est valide."""
        return 0 <= value <= 100
    
    def _is_valid_range(self, min_val: int, max_val: int) -> bool:
        """Vérifie si un range d'expérience est valide."""
        return (0 <= min_val <= 100 and 
                0 <= max_val <= 100 and 
                min_val <= max_val)
    
    def extract_from_all_sources(self, row: pd.Series) -> Optional[float]:
        """
        Extrait l'expérience depuis toutes les sources disponibles.
        
        Args:
            row: Ligne du DataFrame avec plusieurs colonnes possibles
            
        Returns:
            Moyenne des expériences trouvées ou None
            
        Notes:
            Cherche dans : experience, description, title, description_cleaned
        """
        sources = []
        source_columns = ['experience', 'description', 'title', 'description_cleaned']
        
        for col in source_columns:
            if col in row and not pd.isna(row[col]):
                exp = self.extract(row[col])
                if exp is not None:
                    sources.append(exp)
        
        return np.mean(sources) if sources else None
    
    def correct_extreme_values(self, value: Optional[float]) -> Optional[float]:
        """
        Corrige les valeurs d'expérience extrêmes.
        
        Args:
            value: Valeur d'expérience à corriger
            
        Returns:
            Valeur corrigée ou None
            
        Notes:
            - >100 ans: invalide (None)
            - >50 ans: probablement erreur (réduction à max 25 ans)
            - >40 ans: erreur probable (division par 2, max 25 ans)
            - >30 ans: rare mais possible (limitation à 25 ans)
            - <=30 ans: conservation
        """
        if pd.isna(value):
            return None
        
        thresholds = CleaningConfig.EXPERIENCE_THRESHOLDS
        
        if value > 100:
            return None
        elif value > thresholds['extreme_high']:
            return min(thresholds['max_realistic'], value / 4)
        elif value > thresholds['very_high']:
            return min(thresholds['max_realistic'], value / 2)
        elif value > thresholds['high']:
            return min(thresholds['max_realistic'], value)
        else:
            return value


class CompanyExtractor:
    """
    Extracteur de noms d'entreprises depuis les titres d'offres.
    
    Utilise des patterns regex pour détecter les noms d'entreprises
    dans différents formats de titres.
    
    Example:
        >>> extractor = CompanyExtractor()
        >>> company = extractor.extract("Data Analyst H/F Accenture")
        >>> print(company)  # "Accenture"
    """
    
    def __init__(self):
        """Initialise l'extracteur avec les patterns."""
        self.patterns = [
            r'H/F\s+([A-ZÀ-Ÿ][A-Za-zÀ-Ÿ0-9\s&\.\-]+)(?:\s*Siège|\s*Groupe|$)',
            r'F/H\s+([A-ZÀ-Ÿ][A-Za-zÀ-Ÿ0-9\s&\.\-]+)(?:\s*Siège|\s*Groupe|$)',
            r'chez\s+([A-ZÀ-Ÿ][A-Za-zÀ-Ÿ0-9\s&\.\-]+)(?:\s*Siège|$)',
            r'-\s+([A-ZÀ-Ÿ][A-Za-zÀ-Ÿ0-9\s&\.\-]+)\s*$',
            r'H/F(?:.*H/F)?\s*([A-ZÀ-Ÿ][A-Za-zÀ-Ÿ0-9\s&\.\-]+)\s*$',
            r'\(([A-ZÀ-Ÿ][A-Za-zÀ-Ÿ0-9\s&\.\-]+)\)',
        ]
    
    def extract(self, title: str) -> Optional[str]:
        """
        Extrait le nom d'entreprise depuis un titre.
        
        Args:
            title: Titre de l'offre d'emploi
            
        Returns:
            Nom de l'entreprise ou None si non trouvé
        """
        if pd.isna(title):
            return None
        
        for pattern in self.patterns:
            match = re.search(pattern, str(title), re.IGNORECASE)
            if match:
                company = match.group(1).strip()
                company = self._clean_company_name(company)
                if company and len(company) > 2:
                    return company
        
        return None
    
    def _clean_company_name(self, company: str) -> str:
        """Nettoie le nom d'entreprise extrait."""
        # Supprimer les termes indésirables
        company = re.sub(
            r'\s*(?:Siège|Groupe|ESN|Cabinet|France|Company|Corp|Ltd)$',
            '', company, flags=re.IGNORECASE
        )
        return company.strip()


class LocationExtractor:
    """
    Extracteur de localisation depuis descriptions et textes.
    
    Détecte les villes françaises et codes postaux dans les textes.
    
    Attributes:
        city_keywords (Dict[str, List[str]]): Mapping villes vers mots-clés
        
    Example:
        >>> extractor = LocationExtractor()
        >>> loc = extractor.extract("Poste basé à Paris 75008")
        >>> print(loc)  # "Paris"
    """
    
    def __init__(self):
        """Initialise avec les mapping de villes."""
        self.city_keywords = {
            'Paris': ['paris', '750', '75 ', 'île-de-france'],
            'Lyon': ['lyon', '69000', '69 ', 'rhône'],
            'Marseille': ['marseille', '13000', '13 '],
            'Toulouse': ['toulouse', '31000', '31 '],
            'Bordeaux': ['bordeaux', '33000', '33 '],
            'Lille': ['lille', '59000', '59 ', 'nord'],
            'Nice': ['nice', '06000'],
            'Nantes': ['nantes', '44000'],
            'Strasbourg': ['strasbourg', '67000'],
            'Rennes': ['rennes', '35000'],
            'Grenoble': ['grenoble', '38000'],
            'Montpellier': ['montpellier', '34000'],
        }
        
        self.postal_code_map = {
            '75': 'Paris', '69': 'Lyon', '13': 'Marseille',
            '31': 'Toulouse', '33': 'Bordeaux', '59': 'Lille',
            '06': 'Nice', '44': 'Nantes', '67': 'Strasbourg',
            '35': 'Rennes', '38': 'Grenoble', '34': 'Montpellier'
        }
    
    def extract(self, text: str) -> Optional[str]:
        """
        Extrait la localisation depuis un texte.
        
        Args:
            text: Texte source (description, etc.)
            
        Returns:
            Nom de la ville ou None
        """
        if pd.isna(text) or not text:
            return None
        
        text_lower = str(text).lower()
        
        # Chercher par mots-clés
        for city, keywords in self.city_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return city
        
        # Chercher par code postal
        cp_match = re.search(r'\b(\d{5})\b', text)
        if cp_match:
            cp_prefix = cp_match.group(1)[:2]
            if cp_prefix in self.postal_code_map:
                return self.postal_code_map[cp_prefix]
        
        return None


class JobTypeClassifier:
    """
    Classificateur de types de postes Data.
    
    Catégorise les offres d'emploi en différents types basés sur
    le titre et la description.
    
    Attributes:
        categories (Dict[str, List[str]]): Catégories et leurs patterns
        
    Example:
        >>> classifier = JobTypeClassifier()
        >>> job_type = classifier.classify("Senior Data Scientist")
        >>> print(job_type)  # "Data Scientist"
    """
    
    def __init__(self):
        """Initialise avec les catégories et patterns."""
        self.categories = {
            'Data Analyst': [
                r'data.*analyst', r'analyste.*data', r'business.*data.*analyst',
                r'growth.*analyst', r'product.*analyst'
            ],
            'Data Scientist': [
                r'data.*scientist', r'scientist.*data', r'machine.*learning.*scientist',
                r'ai.*scientist', r'research.*scientist'
            ],
            'Data Engineer': [
                r'data.*engineer', r'ingénieur.*data', r'big.*data.*engineer',
                r'etl.*developer', r'data.*pipeline'
            ],
            'BI/Analytics': [
                r'bi.*developer', r'business.*intelligence', r'power.*bi',
                r'tableau', r'analytics.*engineer'
            ],
            'Data Management': [
                r'data.*manager', r'chief.*data.*officer', r'head.*of.*data',
                r'data.*governance', r'data.*architect'
            ],
            'AI/ML Specialist': [
                r'ai.*specialist', r'ml.*specialist', r'llm.*engineer',
                r'nlp.*engineer', r'computer.*vision'
            ],
        }
    
    def classify(self, title: str, description: str = "") -> str:
        """
        Classifie un poste selon son titre et description.
        
        Args:
            title: Titre du poste
            description: Description du poste (optionnel)
            
        Returns:
            Catégorie du poste ou "Autre" si non identifié
        """
        if pd.isna(title):
            return 'Autre'
        
        title_lower = str(title).lower()
        desc_lower = str(description).lower() if description else ""
        
        # Chercher dans le titre d'abord
        for category, patterns in self.categories.items():
            for pattern in patterns:
                if re.search(pattern, title_lower, re.IGNORECASE):
                    return category
        
        # Chercher dans la description si fournie
        if desc_lower:
            for category, patterns in self.categories.items():
                for pattern in patterns:
                    if re.search(pattern, desc_lower, re.IGNORECASE):
                        return f"{category} (via description)"
        
        return 'Autre'


class SeniorityExtractor:
    """
    Extracteur de niveau de séniorité.
    
    Détecte le niveau d'expérience depuis le titre du poste.
    
    Example:
        >>> extractor = SeniorityExtractor()
        >>> level = extractor.extract("Senior Data Scientist")
        >>> print(level)  # "Senior"
    """
    
    def __init__(self):
        """Initialise avec les mots-clés de séniorité."""
        self.keywords = {
            'Junior': ['junior', 'jr', 'débutant', 'entry level', 'graduate'],
            'Senior': ['senior', 'sr', 'expérimenté', 'confirmé', 'expert'],
            'Lead/Manager': ['lead', 'principal', 'chef', 'responsable', 
                           'manager', 'head of', 'director'],
            'Stage/Alternance': ['stage', 'alternance', 'intern', 'apprenti'],
            'Freelance/Consultant': ['freelance', 'consultant', 'independent'],
        }
    
    def extract(self, title: str) -> str:
        """
        Extrait le niveau de séniorité.
        
        Args:
            title: Titre du poste
            
        Returns:
            Niveau de séniorité ou "Mid-level" par défaut
        """
        if pd.isna(title):
            return 'Non spécifié'
        
        title_lower = str(title).lower()
        
        for level, keywords in self.keywords.items():
            for keyword in keywords:
                if keyword in title_lower:
                    return level
        
        return 'Mid-level'


# ============================================================================
# SUITE DU CODE DANS PARTIE 2
# ============================================================================
