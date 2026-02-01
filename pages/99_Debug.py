"""
SCRIPT DE DEBUG - IDENTIFIER LES CHEMINS EN DUR

Ce script affiche tous les chemins utilisés dans l'application
pour identifier où les chemins en dur subsistent.
"""

import streamlit as st
from pathlib import Path
import os

st.title("🔍 Debug - Chemins de l'application")

# ============================================================================
# INFORMATIONS SYSTÈME
# ============================================================================

st.header("1️⃣ Informations système")

col1, col2 = st.columns(2)

with col1:
    st.metric("Répertoire courant", os.getcwd())
    st.metric("Fichier actuel", __file__)

with col2:
    st.metric("Path.cwd()", str(Path.cwd()))
    st.metric("BASE_DIR détecté", str(Path(__file__).parent))

# ============================================================================
# CONTENU DES DOSSIERS
# ============================================================================

st.header("2️⃣ Contenu des dossiers")

base_dir = Path.cwd()

# Models
st.subheader("📁 Dossier models/")
models_dir = base_dir / "models"
if models_dir.exists():
    st.success(f"✅ Dossier trouvé : {models_dir}")
    files = list(models_dir.iterdir())
    st.write(f"**{len(files)} fichiers trouvés** :")
    for f in files:
        st.write(f"  • {f.name} ({f.stat().st_size / 1024:.1f} KB)")
else:
    st.error(f"❌ Dossier introuvable : {models_dir}")

# Data
st.subheader("📁 Dossier data/")
data_dir = base_dir / "data"
if data_dir.exists():
    st.success(f"✅ Dossier trouvé : {data_dir}")
    files = list(data_dir.iterdir())
    st.write(f"**{len(files)} fichiers trouvés** :")
    for f in files:
        st.write(f"  • {f.name} ({f.stat().st_size / 1024:.1f} KB)")
else:
    st.error(f"❌ Dossier introuvable : {data_dir}")

# Output
st.subheader("📁 Dossier output/")
output_dir = base_dir / "output"
if output_dir.exists():
    st.success(f"✅ Dossier trouvé : {output_dir}")
    files = list(output_dir.iterdir())
    st.write(f"**{len(files)} fichiers/dossiers trouvés** :")
    for f in files:
        if f.is_dir():
            st.write(f"  📁 {f.name}/")
        else:
            st.write(f"  • {f.name} ({f.stat().st_size / 1024:.1f} KB)")
else:
    st.error(f"❌ Dossier introuvable : {output_dir}")

# ============================================================================
# CHEMINS DEPUIS CONFIG.PY
# ============================================================================

st.header("3️⃣ Chemins définis dans utils/config.py")

try:
    from utils.config import Config, BASE_DIR, DATA_PATH, MODEL_PATH, TEST_DATA_PATH, REPORT_PATH
    
    st.success("✅ Module config importé avec succès")
    
    paths = {
        "BASE_DIR": BASE_DIR,
        "DATA_PATH": DATA_PATH,
        "MODEL_PATH": MODEL_PATH,
        "TEST_DATA_PATH": TEST_DATA_PATH,
        "REPORT_PATH": REPORT_PATH
    }
    
    for name, path in paths.items():
        exists = path.exists() if hasattr(path, 'exists') else Path(path).exists()
        status = "✅" if exists else "❌"
        st.write(f"{status} **{name}** : `{path}`")
        
except Exception as e:
    st.error(f"❌ Erreur lors de l'import de config : {e}")
    st.code(str(e))

# ============================================================================
# RECHERCHE DE FICHIERS DANS TOUT LE PROJET
# ============================================================================

st.header("4️⃣ Recherche des fichiers critiques")

st.subheader("🔍 Recherche de 'best_model_XGBoost_fixed.pkl'")

def find_file(filename, search_path):
    """Recherche récursive d'un fichier."""
    results = []
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            results.append(Path(root) / filename)
    return results

model_file = "best_model_XGBoost_fixed.pkl"
model_locations = find_file(model_file, base_dir)

if model_locations:
    st.success(f"✅ Fichier trouvé à {len(model_locations)} emplacement(s) :")
    for loc in model_locations:
        st.write(f"  • {loc}")
        st.write(f"    Taille : {loc.stat().st_size / (1024*1024):.2f} MB")
else:
    st.error("❌ Fichier introuvable dans tout le projet")

st.subheader("🔍 Recherche de 'hellowork_cleaned_complete.csv'")

csv_file = "hellowork_cleaned_complete.csv"
csv_locations = find_file(csv_file, base_dir)

if csv_locations:
    st.success(f"✅ Fichier trouvé à {len(csv_locations)} emplacement(s) :")
    for loc in csv_locations:
        st.write(f"  • {loc}")
        st.write(f"    Taille : {loc.stat().st_size / (1024*1024):.2f} MB")
else:
    st.error("❌ Fichier introuvable dans tout le projet")

# ============================================================================
# VÉRIFICATION DES IMPORTS
# ============================================================================

st.header("5️⃣ Vérification des imports")

st.subheader("📦 model_utils.py")

try:
    from utils import model_utils
    st.success("✅ model_utils importé")
    
    # Afficher les attributs
    attrs = [a for a in dir(model_utils) if not a.startswith('_')]
    st.write(f"**{len(attrs)} attributs/fonctions** :")
    st.code(", ".join(attrs[:10]) + ("..." if len(attrs) > 10 else ""))
    
except Exception as e:
    st.error(f"❌ Erreur : {e}")

# ============================================================================
# RECOMMANDATIONS
# ============================================================================

st.header("6️⃣ Recommandations")

st.info("""
**Si le fichier est introuvable** :

1. **Vérifier .gitignore** : Assurez-vous que `*.pkl` et `*.csv` ne sont PAS bloqués
2. **Vérifier Git LFS** : Si fichiers > 100MB, Git LFS doit être activé
3. **Vérifier le commit** : `git ls-files | grep -E "(pkl|csv)"`
4. **Reboot l'app** : Sur Streamlit Cloud, cliquez "Reboot app"

**Si les chemins sont incorrects** :

Vérifiez que `utils/config.py` utilise :
```python
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "hellowork_cleaned_complete.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model_XGBoost_fixed.pkl"
```

ET PAS :
```python
DATA_PATH = Path("output/hellowork_cleaned_complete.csv")  # ❌
```
""")

# ============================================================================
# CODE POUR COPIER-COLLER
# ============================================================================

st.header("7️⃣ Code corrigé à copier-coller")

st.code("""
# Dans utils/config.py (lignes 20-30 environ)

from pathlib import Path

# BASE_DIR dynamique
BASE_DIR = Path(__file__).parent.parent

# Chemins relatifs
DATA_PATH = BASE_DIR / "data" / "hellowork_cleaned_complete.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model_XGBoost_fixed.pkl"
TEST_DATA_PATH = BASE_DIR / "models" / "test_data.pkl"
REPORT_PATH = BASE_DIR / "models" / "modeling_report_v7.json"
""", language="python")
