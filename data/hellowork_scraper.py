"""

Extraction  avec gestion mémoire optimisée pour le scraping d'offres d'emploi.

Ce module permet de scraper le site HelloWork.fr pour collecter des offres d'emploi
dans le domaine de la data science, data engineering, data analysis et autres métiers
liés aux données. Il inclut une gestion des ressources, une déduplication
des offres et une extraction complète des détails.

Fonctionnalités principales:
- Recherche par mots-clés et localisations
- Extraction des offres depuis les pages de résultats
- Extraction détaillée de chaque offre (description, salaire, compétences, etc.)
- Gestion de la concurrence et des redémarrages périodiques
- Sauvegarde progressive des résultats
- Support de pandas pour l'export CSV ou fallback vers csv standard

"""

import gc
import asyncio
from botright import Botright

# Gestion de l'import conditionnel de pandas
try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    import csv
    from collections import Counter
    HAS_PANDAS = False

    def dicts_to_csv(path, rows, columns=None):
        """
        Convertit une liste de dictionnaires en fichier CSV (fallback sans pandas).
        
        Args:
            path (str): Chemin du fichier CSV de sortie
            rows (list): Liste de dictionnaires à exporter
            columns (list, optional): Liste des colonnes à inclure. Si None, 
                                     déduit des clés des dictionnaires.
        """
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            if rows:
                if not columns:
                    cols = []
                    for r in rows:
                        for k in r.keys():
                            if k not in cols:
                                cols.append(k)
                else:
                    cols = columns
                writer = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(rows)
            else:
                f.write('')

import os
from datetime import datetime
import random
import hashlib
import re

# ===== CONFIGURATION =====
# ————— STRATÉGIE 1 : MOTS-CLÉS -ÉLARGIS —————
BASE_KEYWORDS = [
    # Core Data
    "Data", "Données", "Analytics", "Intelligence", "Insights",
    
    # Data Science
    "Data Scientist", "Data Science", "Scientist Data", "Scientifique données",
    "Junior Data Scientist", "Consultant Data Scientist", "Consultant Confirmé en Data", "Consultant Data gouvernance", "Senior Data Scientist", "Lead Data Scientist",
    "Chief Data Scientist", "Principal Data Scientist",
    
    # Data Analysis
    "Data Analyst", "Data Analysis", "Analyste données", "Analyste de données",
    "Junior Data Analyst", "Senior Data Analyst", "Lead Data Analyst",
    "Business Data Analyst", "Marketing Data Analyst", "Financial Data Analyst",
    "Product Data Analyst", "Commercial Data Analyst", "Customer Data Analyst",
    
    # Data Engineering
    "Data Engineer", "Ingénieur Data", "Ingénieur données", "Data Engineering",
    "Junior Data Engineer", "Senior Data Engineer", "Lead Data Engineer",
    "Cloud Data Engineer", "Data Engineer aws", "Consultant Data Engineer", "Big Data Engineer", "Data Platform Engineer",
    "Staff Data Engineer", "Principal Data Engineer", "Ingénieur big data",
    
    # Machine Learning & AI
    "Machine Learning", "ML Engineer", "MLOps", "AI Engineer", "IA",
    "Artificial Intelligence", "Deep Learning", "Computer Vision",
    "NLP Engineer", "Natural Language Processing", "Research Scientist",
    "Applied Scientist", "Research Engineer", "AI Research",
    "Machine Learning Scientist", "Apprentissage automatique",
    
    # Business Intelligence
    "Business Intelligence", "BI", "Intelligence affaires",
    "BI Analyst", "BI Developer", "BI Engineer", "BI Consultant",
    "Business Analyst Data", "Analyste BI", "Intégrateur Data",
    
    # Architecture & Management
    "Data Architect", "Architecte données", "Data Architecture",
    "Chief Data Officer", "CDO", "Data Manager", "Data Lead",
    "Data Product Manager", "Data Product Owner", "Product Manager Data",
    "Data Governance", "Data specialist", "Data Steward", "CTO", "CDO", "IT Manager", "Data Quality",
    
    # Analytics spécialisés
    "Analytics Engineer", "Analytics", "Predictive Analytics",
    "Web Analytics", "Digital Analytics", "Product Analyst",
    "Growth Analyst", "Performance Analyst", "Insights Analyst",
    "Revenue Analyst", "Consultant Data","Operations Analyst Data",
    
    # Techniques
    "Python Data", "R Developer", "SQL", "ETL Developer",
    "Data Integration", "Data Warehouse", "Data Lake",
    "DataOps", "Data Pipeline", "Data Platform",
    
    # Visualisation
    "Data Visualization", "Data Viz", "Tableau", "Power BI",
    "Tableau Developer", "Power BI Developer", "Looker", "Qlik",
    "Dashboard", "Reporting Analyst",
    
    # Émergents & IA Générative
    "Prompt Engineer", "LLM", "GenAI", "Generative AI",
    "ChatGPT", "OpenAI", "Large Language Model",
    "Feature Engineer", "Feature Engineering",
    
    # Alternance & Stage
    "Alternance Data", "Stage Data", "Apprenti Data", "Apprentissage Data",
    "Stagiaire Data", "Alternant Data", "Contrat apprentissage Data",
    "Stage Data Scientist", "Stage Data Analyst", "Stage Data Engineer",
    "Alternance Data Scientist", "Alternance Data Analyst",
    
    # Secteurs spécifiques
    "Data Finance", "Data Marketing", "Data RH", "Data Sales",
    "Data Healthcare", "Data Retail", "Data E-commerce",
    "Data Bancaire", "Data Assurance", "Data Logistique",
    
    # Technologies cloud
    "AWS Data", "Azure Data", "GCP Data", "Snowflake",
    "Databricks", "Spark", "Hadoop", "Kafka",
    
    # Rôles hybrides
    "Full Stack Data", "Data Developer", "Software Engineer Data",
    "Backend Data", "API Data", "Microservices Data",
]

FRENCH_CITIES = [
    "", "Paris", "Lyon", "Marseille", "Toulouse", "Lille",
    "Bordeaux", "Nantes", "Strasbourg", "Montpellier", "Nice",
    "Rennes", "Reims", "Saint-Étienne", "Toulon", "Le Havre",
    "Grenoble", "Dijon", "Angers", "Nîmes", "Villeurbanne",
    "Saint-Denis", "Clermont-Ferrand", "Aix-en-Provence",

    # Autres grandes villes
    "Brest", "Limoges", "Tours", "Amiens", "Perpignan",
    "Metz", "Nancy", "Besançon", "Orléans", "Rouen",
    "Mulhouse", "Caen", "Boulogne-Billancourt", "Argenteuil",
    "Montreuil", "Roubaix", "Tourcoing", "Nanterre",
    "Avignon", "Poitiers"
]

TARGET_OFFERS = 8000
MAX_CONCURRENT_DETAIL_PAGES = 2
PROGRESSIVE_SAVE_INTERVAL = 100
USE_GEOGRAPHIC_FILTERS = True
MAX_KEYWORDS_TO_TRY = 100
PAGE_TIMEOUT = 45000
DETAIL_TIMEOUT = 20000
RESTART_INTERVAL = 15


class ResourceManager:
    """
    Gestionnaire de ressources pour contrôler l'accès concurrentiel aux pages.
    
    Cette classe utilise un sémaphore pour limiter le nombre de pages actives
    simultanément et gère la fermeture propre des pages.
    
    Attributes:
        max_concurrent (int): Nombre maximum de pages concurrentes
        active_pages (list): Liste des pages actuellement actives
        semaphore (asyncio.Semaphore): Sémaphore pour contrôler l'accès
    """
    
    def __init__(self, max_concurrent=2):
        """
        Initialise le gestionnaire de ressources.
        
        Args:
            max_concurrent (int): Nombre maximum de pages concurrentes (défaut: 2)
        """
        self.max_concurrent = max_concurrent
        self.active_pages = []
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def acquire_slot(self):
        """Acquiert un slot pour ouvrir une nouvelle page (bloquant si limite atteinte)."""
        await self.semaphore.acquire()
    
    def release_slot(self):
        """Libère un slot pour permettre à d'autres pages de s'ouvrir."""
        self.semaphore.release()
    
    def register_page(self, page):
        """
        Enregistre une page comme active.
        
        Args:
            page: L'objet page à enregistrer
        """
        if page not in self.active_pages:
            self.active_pages.append(page)
    
    async def close_page(self, page):
        """
        Ferme proprement une page et la retire de la liste des pages actives.
        
        Args:
            page: La page à fermer
        """
        try:
            if page and not page.is_closed():
                await page.close()
            if page in self.active_pages:
                self.active_pages.remove(page)
        except:
            pass
    
    async def close_all(self):
        """Ferme toutes les pages actives."""
        for page in self.active_pages[:]:
            await self.close_page(page)


def generate_keyword_combinations():
    """
    Génère des combinaisons de mots-clés et localisations pour la recherche.
    
    Returns:
        list: Liste de tuples (mot-clé, localisation) mélangée aléatoirement
    """
    combinations = []
    combinations.extend([(kw, "") for kw in BASE_KEYWORDS])
    
    if USE_GEOGRAPHIC_FILTERS:
        for city in FRENCH_CITIES[1:8]:
            for kw in BASE_KEYWORDS[:15]:
                combinations.append((kw, city))
    
    random.shuffle(combinations)
    return combinations[:MAX_KEYWORDS_TO_TRY]


async def build_search_url(keyword, location="", page=1):
    """
    Construit l'URL de recherche HelloWork à partir des paramètres.
    
    Args:
        keyword (str): Mot-clé de recherche
        location (str, optional): Localisation pour filtrer. Défaut: ""
        page (int, optional): Numéro de page. Défaut: 1
    
    Returns:
        str: URL complète de recherche
    """
    keyword_clean = keyword.lower().replace(' ', '%20')
    location_clean = location.lower().replace(' ', '%20').replace('-', '%20') if location else ""
    
    url = f"https://www.hellowork.com/fr-fr/emploi/recherche.html?k={keyword_clean}"
    
    if location_clean:
        url += f"&l={location_clean}"
    
    if page > 1:
        url += f"&p={page}"
    
    return url


def create_offer_hash(offer):
    """
    Crée un hash unique pour une offre afin de détecter les doublons.
    
    Args:
        offer (dict): Dictionnaire contenant les informations de l'offre
    
    Returns:
        str: Hash MD5 de la concaténation des champs clés de l'offre
    """
    key = f"{offer.get('title', '')}_{offer.get('company', '')}_{offer.get('location', '')}_{offer.get('id', '')}"
    return hashlib.md5(key.encode('utf-8')).hexdigest()


async def extract_offers_from_page(page):
    """
    Extrait les offres d'une page de résultats de recherche.
    
    Args:
        page: L'objet page Browser contenant les résultats
    
    Returns:
        list: Liste de dictionnaires, chaque dictionnaire contenant les informations
              de base d'une offre (titre, entreprise, localisation, etc.)
    """
    try:
        return await page.evaluate("""() => {
            const offers = [];
            const selectors = ['a[href*="/emplois/"]', '[data-testid*="job"] a', '.job-card a', 'article a'];
            
            let links = [];
            for (const sel of selectors) {
                const found = document.querySelectorAll(sel);
                if (found.length > 0) {
                    links = Array.from(found);
                    break;
                }
            }
            
            for (const link of links) {
                const href = link.getAttribute('href');
                if (!href || !/\\/emplois\\/\\d+\\.html/.test(href)) continue;
                
                const card = link.closest('li, div[data-testid*="job"], article, .job-card') || link.parentElement;
                if (!card) continue;
                
                const cardText = card.innerText || '';
                const lines = cardText.split('\\n').map(l => l.trim()).filter(l => l);
                
                let title = '';
                const titleElem = card.querySelector('h2, h3, [data-testid*="title"], .job-title');
                if (titleElem) {
                    title = titleElem.innerText.trim();
                } else {
                    title = link.innerText.trim();
                }
                
                if (!title || title.length < 5 || title.length > 200) continue;
                
                let company = 'Non spécifié';
                const companyElem = card.querySelector('[data-testid*="company"], .company, .company-name');
                if (companyElem) {
                    company = companyElem.innerText.trim();
                } else {
                    const titleIdx = lines.findIndex(l => l === title);
                    if (titleIdx !== -1 && titleIdx + 1 < lines.length) {
                        const nextLine = lines[titleIdx + 1];
                        if (nextLine && nextLine.length < 60 && 
                            !/\\d{2,5}|CDI|CDD|Stage|Alternance|€|Paris|Lyon|Marseille/i.test(nextLine)) {
                            company = nextLine;
                        }
                    }
                }
                
                let location = 'Non spécifié';
                const locMatch = cardText.match(/([A-ZÀ-Ÿ][a-zà-ÿ\\s-]+(?:\\s*-\\s*|\\s+)\\d{2,5})/);
                if (locMatch) {
                    location = locMatch[0].trim();
                } else {
                    const cityMatch = cardText.match(/Paris|Lyon|Marseille|Toulouse|Lille|Bordeaux|Nantes|Nice|Strasbourg|Montpellier|Rennes|Île-de-France/i);
                    if (cityMatch) location = cityMatch[0];
                }
                
                let salary = '';
                const salMatch = cardText.match(/(\\d{2}[\\s]?\\d{3}(?:[\\s-]+\\d{2}[\\s]?\\d{3})?[\\s]*€(?:\\/an|\\/mois)?)/);
                if (salMatch) salary = salMatch[0].trim();
                
                let contract = '';
                if (/\\bCDI\\b/i.test(cardText)) contract = 'CDI';
                else if (/\\bCDD\\b/i.test(cardText)) contract = 'CDD';
                else if (/\\bStage\\b/i.test(cardText)) contract = 'Stage';
                else if (/\\bAlternance\\b/i.test(cardText)) contract = 'Alternance';
                
                let date = '';
                const dateMatch = cardText.match(/(?:il y a|Publiée? il y a|Publiée?)\\s+([^\\n]+)/i);
                if (dateMatch) date = dateMatch[0].trim();
                
                const idMatch = href.match(/\\/emplois\\/(\\d+)\\.html/);
                const id = idMatch ? idMatch[1] : '';
                
                const fullUrl = href.startsWith('http') ? href : 'https://www.hellowork.com' + href;
                
                offers.push({title, company, location, salary, contract, date_posted: date, url: fullUrl, id});
            }
            
            return offers;
        }""")
    except Exception as e:
        print(f"   ⚠️ Erreur extraction: {str(e)[:50]}")
        return []


async def extract_offer_details(browser, offer_url, resource_manager):
    """
    Extrait les détails complets d'une offre d'emploi.
    
    Args:
        browser: Instance du navigateur
        offer_url (str): URL de l'offre à scraper
        resource_manager (ResourceManager): Gestionnaire de ressources pour le contrôle de concurrence
    
    Returns:
        dict: Dictionnaire contenant tous les détails extraits de l'offre
    """
    detail_page = None
    
    try:
        await resource_manager.acquire_slot()
        detail_page = await browser.new_page()
        resource_manager.register_page(detail_page)
        
        await detail_page.set_extra_http_headers({
            'Accept-Language': 'fr-FR,fr;q=0.9',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        try:
            await detail_page.route("**/*.{png,jpg,jpeg,gif,webp,svg,css,woff,woff2}", lambda route: route.abort())
        except:
            pass
        
        await detail_page.goto(offer_url, timeout=DETAIL_TIMEOUT, wait_until='domcontentloaded')
        await asyncio.sleep(0.8)
        
        details = await detail_page.evaluate("""() => {
            const data = {
                description: '', missions: '', profile_sought: '',
                experience_required: '', education_level: '', skills: [], languages: [],
                contract_type: '', contract_duration: '', salary_min: '', salary_max: '', salary_text: '',
                telework: '', work_schedule: '', start_date: '',
                company_name: '', company_size: '', company_sector: '', company_description: '',
                benefits: [], advantages: '',
                location_full: '', department: '', region: '',
                application_count: '', publication_date: '', deadline: '', reference: '', positions_available: ''
            };
            
            const text = document.body.innerText || '';
            
            const descSelectors = ['[data-testid="job-description"]', '.job-description', '[class*="description"]', 'main'];
            for (const sel of descSelectors) {
                const elem = document.querySelector(sel);
                if (elem && elem.innerText.length > 100) {
                    data.description = elem.innerText.trim().substring(0, 5000);
                    break;
                }
            }
            if (!data.description) data.description = text.substring(0, 5000);
            
            const missionsMatch = text.match(/(?:Missions?|Vos missions|Responsabilités)[:\\s]+([^]*?)(?=\\n\\n|Profil|Compétences|$)/i);
            if (missionsMatch) data.missions = missionsMatch[1].trim().substring(0, 2000);
            
            const profileMatch = text.match(/(?:Profil recherché|Profil|Vous êtes)[:\\s]+([^]*?)(?=\\n\\n|Nous offrons|Avantages|$)/i);
            if (profileMatch) data.profile_sought = profileMatch[1].trim().substring(0, 2000);
            
            const sizePatterns = [
                /(\\d+)\\s*(?:à|-)\\s*(\\d+)\\s*(?:salariés?|employés?|collaborateurs?)/i,
                /(\\d+)\\s*(?:salariés?|employés?)/i,
                /(?:TPE|PME|ETI|Grand[e]? entreprise|Startup|Scale-up)/i
            ];
            for (const pattern of sizePatterns) {
                const match = text.match(pattern);
                if (match) {
                    data.company_size = match[0].trim();
                    break;
                }
            }
            
            const sectorPatterns = [
                /(?:Banque|Finance|Assurance|E-commerce|Tech|Conseil|Industrie|Santé|Énergie|Transport|Logistique|Retail|Agroalimentaire|Pharmaceutique)/i
            ];
            for (const pattern of sectorPatterns) {
                const match = text.match(pattern);
                if (match) {
                    data.company_sector = match[0].trim();
                    break;
                }
            }
            
            const contractPatterns = [
                { regex: /\\bCDI\\b/i, value: 'CDI' },
                { regex: /\\bCDD\\b/i, value: 'CDD' },
                { regex: /\\bStage\\b/i, value: 'Stage' },
                { regex: /\\bAlternance\\b/i, value: 'Alternance' }
            ];
            for (const pattern of contractPatterns) {
                if (pattern.regex.test(text)) {
                    data.contract_type = pattern.value;
                    break;
                }
            }
            
            const salaryPatterns = [
                /(\\d{2}[\\s]?\\d{3})[\\s-]+(\\d{2}[\\s]?\\d{3})[\\s]*€/i,
                /(\\d{2}[\\s]?\\d{3})[\\s]*€/i
            ];
            for (const pattern of salaryPatterns) {
                const match = text.match(pattern);
                if (match) {
                    if (match[2]) {
                        data.salary_min = match[1].replace(/\\s/g, '') + '€';
                        data.salary_max = match[2].replace(/\\s/g, '') + '€';
                    } else {
                        data.salary_min = match[1].replace(/\\s/g, '') + '€';
                    }
                    data.salary_text = match[0];
                    break;
                }
            }
            
            const expPatterns = [
                /(\\d+)\\s*(?:à|-)\\s*(\\d+)\\s*ans?/i,
                /(\\d+)\\s*ans?\\s*(?:d')?(?:expérience)/i,
                /(?:Junior|Confirmé|Senior|Expert)/i
            ];
            for (const pattern of expPatterns) {
                const match = text.match(pattern);
                if (match) {
                    data.experience_required = match[0];
                    break;
                }
            }
            
            const eduPatterns = [/Bac\\s*\\+\\s*(\\d+)/i, /(?:Master|M1|M2|Doctorat|Ingénieur)/i];
            for (const pattern of eduPatterns) {
                const match = text.match(pattern);
                if (match) {
                    data.education_level = match[0];
                    break;
                }
            }
            
            const techSkills = ['Python', 'SQL', 'R', 'Spark', 'TensorFlow', 'PyTorch', 'AWS', 'Azure', 'GCP', 'Tableau', 'Power BI', 'Machine Learning', 'Deep Learning', 'Docker', 'Kubernetes', 'Airflow', 'Kafka', 'Snowflake', 'Git', 'PostgreSQL', 'MongoDB'];
            data.skills = techSkills.filter(skill => text.includes(skill) || text.toLowerCase().includes(skill.toLowerCase()));
            
            if (/télétravail|remote|hybride/i.test(text)) {
                const teleworkMatch = text.match(/(?:télétravail|remote|hybride)[\\s:]*([^\\n.]+)/i);
                data.telework = teleworkMatch ? teleworkMatch[0].trim() : 'Oui';
            } else {
                data.telework = 'Non spécifié';
            }
            
            const benefitsKeywords = ['Tickets restaurant', 'Mutuelle', 'RTT', 'Prime', 'Bonus', 'Intéressement', 'Télétravail', 'Formation', 'CSE', '13ème mois', 'Stock options'];
            for (const benefit of benefitsKeywords) {
                if (new RegExp(benefit, 'i').test(text)) {
                    data.benefits.push(benefit);
                }
            }
            
            const locationMatch = text.match(/([A-ZÀ-Ÿ][a-zà-ÿ\\s-]+(?:\\s*-\\s*|\\s+)\\d{2,5})/);
            if (locationMatch) {
                data.location_full = locationMatch[0].trim();
                const deptMatch = locationMatch[0].match(/\\d{2,5}/);
                if (deptMatch) data.department = deptMatch[0].substring(0, 2);
            }
            
            const pubDateMatch = text.match(/(?:Publiée?|Postée?)\\s*(?:le|il y a)?\\s*([^\\n]+)/i);
            if (pubDateMatch) data.publication_date = pubDateMatch[1].trim();
            
            return data;
        }""")
        
        return details
        
    except Exception as e:
        return {}
    finally:
        if detail_page:
            await resource_manager.close_page(detail_page)
        resource_manager.release_slot()


async def scrape_ultra():
    """
    Fonction principale de scraping HelloWork.
    
    Cette fonction prends en compte tout le processus de scraping:
    1. Initialisation du navigateur et des ressources
    2. Génération des combinaisons de recherche
    3. Parcours des pages de résultats
    4. Extraction des offres et de leurs détails
    5. Sauvegarde progressive et finale des résultats
    6. Nettoyage des ressources
    
    Gestion des redémarrages périodiques pour éviter les blocages et fuites mémoire.
    """
    print(" SCRAPER HELLOWORK COMPLET v3.3")
    print("="*70)
    print(f" Objectif: {TARGET_OFFERS}+ offres")
    print(f" Redémarrage: tous les {RESTART_INTERVAL} mots-clés")
    print("="*70)
    
    start = asyncio.get_event_loop().time()
    
    botright_client = await Botright()
    browser = await botright_client.new_browser()
    resource_manager = ResourceManager(max_concurrent=MAX_CONCURRENT_DETAIL_PAGES)
    
    main_page = await browser.new_page()
    await main_page.set_extra_http_headers({
        'Accept-Language': 'fr-FR,fr;q=0.9',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    try:
        await main_page.route("**/*.{png,jpg,jpeg,gif,webp,svg,css,woff,woff2}", lambda route: route.abort())
    except:
        pass
    
    all_results = []
    seen_offers = set()
    total_details = 0
    failed_keywords = []
    keywords_processed = 0
    
    keyword_combinations = generate_keyword_combinations()
    print(f"✅ {len(keyword_combinations)} combinaisons générées\n")
    
    for idx, (keyword, location) in enumerate(keyword_combinations, 1):
        
        if keywords_processed > 0 and keywords_processed % RESTART_INTERVAL == 0:
            print(f"\n🔄 Redémarrage du navigateur (après {keywords_processed} mots-clés)")
            
            await resource_manager.close_all()
            await main_page.close()
            await browser.close()
            gc.collect()
            await asyncio.sleep(2)
            
            browser = await botright_client.new_browser()
            resource_manager = ResourceManager(max_concurrent=MAX_CONCURRENT_DETAIL_PAGES)
            
            main_page = await browser.new_page()
            await main_page.set_extra_http_headers({
                'Accept-Language': 'fr-FR,fr;q=0.9',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            try:
                await main_page.route("**/*.{png,jpg,jpeg,gif,webp,svg,css,woff,woff2}", lambda route: route.abort())
            except:
                pass
            
            await asyncio.sleep(3)
            print(f"✅ Navigateur redémarré\n")
        
        keywords_processed += 1
        
        if keywords_processed % 10 == 0:
            gc.collect()
        
        if len(all_results) >= TARGET_OFFERS:
            print(f"\n🎉 OBJECTIF {TARGET_OFFERS} ATTEINT !")
            break
        
        search_label = f"{keyword}" + (f" @ {location}" if location else "")
        print(f"\n{'='*70}")
        print(f" [{idx}/{len(keyword_combinations)}] {search_label}")
        print(f" {len(all_results)}/{TARGET_OFFERS} offres")
        
        page_num = 1
        empty_pages = 0
        consecutive_errors = 0
        
        while len(all_results) < TARGET_OFFERS and empty_pages < 2 and consecutive_errors < 3:
            
            url = await build_search_url(keyword, location, page_num)
            
            try:
                await main_page.goto(url, timeout=PAGE_TIMEOUT, wait_until='domcontentloaded')
                await asyncio.sleep(random.uniform(1.5, 3))
                
                await main_page.evaluate("""
                    async () => {
                        await new Promise(resolve => {
                            let totalHeight = 0;
                            const distance = 200;
                            const timer = setInterval(() => {
                                window.scrollBy(0, distance);
                                totalHeight += distance;
                                if(totalHeight >= document.body.scrollHeight){
                                    clearInterval(timer);
                                    resolve();
                                }
                            }, 100);
                        });
                    }
                """)
                
                await asyncio.sleep(1)
                
                text = await main_page.evaluate("() => document.body.innerText")
                if "Aucune offre" in text or "0 offre" in text or "Aucun résultat" in text:
                    print(f"   📄 Page {page_num} - ℹ️ Aucun résultat")
                    break
                
                offers = await extract_offers_from_page(main_page)
                
                if not offers or len(offers) == 0:
                    empty_pages += 1
                    print(f"   📄 Page {page_num} - ⚠️ Vide ({empty_pages}/2)")
                    page_num += 1
                    continue
                
                empty_pages = 0
                consecutive_errors = 0
                new_count = 0
                
                for offer in offers:
                    if len(all_results) >= TARGET_OFFERS:
                        break
                    
                    offer_hash = create_offer_hash(offer)
                    
                    if offer_hash in seen_offers:
                        continue
                    
                    seen_offers.add(offer_hash)
                    
                    if offer.get('url'):
                        details = await extract_offer_details(browser, offer['url'], resource_manager)
                        offer.update(details)
                        total_details += 1
                        await asyncio.sleep(random.uniform(0.5, 1.5))
                    
                    offer['search_keyword'] = keyword
                    offer['search_location'] = location
                    offer['page'] = page_num
                    offer['scraped_at'] = datetime.now().isoformat()
                    offer['offer_hash'] = offer_hash
                    
                    all_results.append(offer)
                    new_count += 1
                    
                    if len(all_results) % PROGRESSIVE_SAVE_INTERVAL == 0:
                        os.makedirs("output/progressive", exist_ok=True)
                        if HAS_PANDAS:
                            pd.DataFrame(all_results).to_csv(
                                f"output/progressive/progress_{len(all_results)}.csv",
                                index=False, encoding="utf-8-sig"
                            )
                        else:
                            dicts_to_csv(f"output/progressive/progress_{len(all_results)}.csv", all_results)
                
                print(f"   📄 Page {page_num} - ✅ {len(offers)} offres → +{new_count} (Total: {len(all_results)})")
                page_num += 1
                
            except asyncio.TimeoutError:
                consecutive_errors += 1
                print(f"   📄 Page {page_num} - ❌ Timeout ({consecutive_errors}/3)")
                page_num += 1
                await asyncio.sleep(3)
                
            except Exception as e:
                consecutive_errors += 1
                error_msg = str(e)[:60]
                print(f"   📄 Page {page_num} - ❌ Erreur: {error_msg}")
                if consecutive_errors >= 3:
                    failed_keywords.append(search_label)
                    break
                page_num += 1
                await asyncio.sleep(2)
    
    print("\n🔒 Fermeture...")
    await resource_manager.close_all()
    await main_page.close()
    await browser.close()
    await botright_client.close()
    
    if all_results:
        os.makedirs("output", exist_ok=True)
        
        if HAS_PANDAS:
            df = pd.DataFrame(all_results)
            
            initial = len(df)
            df = df.drop_duplicates(subset=['offer_hash'], keep='first')
            df = df[df['title'].str.len() > 5]
            final = len(df)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"output/hellowork_complete_{timestamp}.csv"
            
            cols = [
                'title', 'id', 'reference', 'offer_hash',
                'company', 'company_name', 'company_size', 'company_sector', 'company_description',
                'location', 'location_full', 'department', 'region',
                'contract', 'contract_type', 'contract_duration',
                'salary', 'salary_min', 'salary_max', 'salary_text',
                'telework', 'work_schedule', 'start_date',
                'experience_required', 'education_level', 'skills', 'languages',
                'benefits', 'advantages',
                'description', 'missions', 'profile_sought',
                'date_posted', 'publication_date', 'deadline',
                'application_count', 'positions_available',
                'search_keyword', 'search_location', 'page', 'url', 'scraped_at'
            ]
            
            existing = [c for c in cols if c in df.columns]
            df = df[existing + [c for c in df.columns if c not in existing]]
            
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
            
            df.to_csv(filename, index=False, encoding="utf-8-sig")
            
            elapsed = asyncio.get_event_loop().time() - start
            
            print(f"\n{'='*70}")
            print(f"🎉 TERMINÉ !")
            print(f"{'='*70}")
            print(f"📊 Offres collectées: {final}")
            print(f"🗑️ Doublons: {initial - final}")
            print(f"💾 Fichier: {filename}")
            print(f"🔍 Détails: {total_details}")
            print(f"⏱️ Temps: {elapsed/60:.1f} min")
            print(f"⚡ Vitesse: {final/(elapsed/60):.1f} offres/min")
            
            if failed_keywords:
                print(f"\n⚠️ Mots-clés échoués ({len(failed_keywords)}):")
                for kw in failed_keywords[:10]:
                    print(f"   • {kw}")
            
            print(f"\n📈 TOP 20 Mots-clés:")
            for kw, cnt in df['search_keyword'].value_counts().head(20).items():
                print(f"   • {kw:40s}: {cnt:4d}")
    else:
        print("\n❌ Aucune offre collectée")


if __name__ == "__main__":
    """
    Point d'entrée principal du script.
    Gère les interruptions clavier et les erreurs globales.
    """
    try:
        asyncio.run(scrape_ultra())
    except KeyboardInterrupt:
        print("\n⏹️ Arrêt utilisateur")
    except Exception as e:
        print(f"\n💥 Erreur: {e}")
        import traceback
        traceback.print_exc()