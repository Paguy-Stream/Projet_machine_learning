import os
import ast

# Racine du projet (à ajuster si besoin)
ROOT_DIR = os.path.abspath(".")

def inspect_python_file(filepath):
    """Retourne les classes et fonctions définies dans un fichier Python."""
    classes = []
    functions = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=filepath)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
    except Exception as e:
        print(f"Erreur lecture {filepath}: {e}")
    return classes, functions

def inspect_project(root_dir):
    print(f"Structure du projet {root_dir}\n{'='*50}")
    for dirpath, dirnames, filenames in os.walk(root_dir):
        indent_level = dirpath.replace(root_dir, "").count(os.sep)
        indent = "    " * indent_level
        print(f"{indent}{os.path.basename(dirpath)}/")
        for filename in filenames:
            if filename.endswith(".py"):
                file_path = os.path.join(dirpath, filename)
                classes, functions = inspect_python_file(file_path)
                print(f"{indent}    {filename}")
                if classes:
                    print(f"{indent}        Classes: {classes}")
                if functions:
                    print(f"{indent}        Fonctions: {functions}")

if __name__ == "__main__":
    inspect_project(ROOT_DIR)
