"""
inspect_excel.py — Diagnostique les colonnes d'un fichier Excel de requirements.

Lance ce script AVANT run_full_extraction.py pour verifier que l'auto-detection
des colonnes fonctionne correctement sur ton vrai fichier Excel.

Usage:
    python inspect_excel.py "OT Bill Generation Solution Requirement v1.xls"
    python inspect_excel.py requirements.xlsx

Sortie :
  - Nom de toutes les colonnes detectees (ligne 1)
  - Mapping automatique propose (critere / categorie / obligatoire / id)
  - Apercu des 5 premieres lignes de donnees
  - Nombre total de lignes
  - Liste des valeurs uniques dans la colonne "Categorie" et "Obligatoire"
    (pour verifier qu'elles correspondent aux valeurs attendues)
"""

import sys, io
from pathlib import Path

# ── Couleurs ANSI ────────────────────────────────────────────────────────────────
RESET = "\033[0m"; BOLD = "\033[1m"; GREEN = "\033[32m"
YELLOW = "\033[33m"; RED = "\033[31m"; CYAN = "\033[36m"; DIM = "\033[2m"
def clr(c, t): return f"{c}{t}{RESET}"
def ok(t):  return clr(GREEN, t)
def warn(t): return clr(YELLOW, t)
def err(t):  return clr(RED, t)

VALID_CATS = {
    "Functional", "Technical", "Operational",
    "Security", "Infrastructure", "Administrative", "Financial"
}

# Mots-cles pour l'auto-detection (meme liste que run_full_extraction.py)
KEYWORDS = {
    "criterion": ["requirement", "criterion", "description", "exigence",
                  "critere", "specification", "text", "wording"],
    "category":  ["category", "categorie", "type", "area", "domain",
                  "domaine", "functional area"],
    "mandatory": ["mandatory", "obligatoire", "priority", "required",
                  "compliance", "must", "priorite"],
    "id":        ["id", "ref", "no.", "num", "req_id", "req id"],
    "source":    ["source", "section", "chapter", "chapitre"],
}


def find_col(headers, *kws):
    for kw in kws:
        for h in headers:
            if kw.lower() in h.lower():
                return h
    return None


def load_excel(filepath: Path):
    """Charge le fichier et retourne (headers, rows)."""
    ext = filepath.suffix.lower()
    content = filepath.read_bytes()

    if ext in (".xlsx", ".xlsm"):
        import openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(content), data_only=True)
        ws = wb.active
        raw = list(ws.iter_rows(values_only=True))
        if not raw:
            return [], []
        headers = [str(h).strip() if h is not None else f"col_{i}"
                   for i, h in enumerate(raw[0])]
        rows = []
        for row in raw[1:]:
            if all(v is None for v in row): continue
            rows.append([str(v).strip() if v is not None else "" for v in row])
        return headers, rows

    elif ext == ".xls":
        import xlrd
        wb = xlrd.open_workbook(file_contents=content)
        ws = wb.sheet_by_index(0)
        headers = [str(ws.cell_value(0, c)).strip() for c in range(ws.ncols)]
        rows = []
        for r in range(1, ws.nrows):
            row = [str(ws.cell_value(r, c)).strip() for c in range(ws.ncols)]
            if any(v for v in row):
                rows.append(row)
        return headers, rows

    else:
        print(err(f"Format non supporte : {ext}")); sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print(err("Usage : python inspect_excel.py <fichier.xls|xlsx>"))
        sys.exit(1)

    filepath = Path(sys.argv[1])
    if not filepath.exists():
        print(err(f"Fichier introuvable : {filepath}")); sys.exit(1)

    print()
    print(clr(BOLD + CYAN, "=== DIAGNOSTIC EXCEL ==="))
    print(clr(DIM, f"  Fichier : {filepath.resolve()}"))
    print(clr(DIM, f"  Taille  : {filepath.stat().st_size // 1024} Ko"))
    print()

    # ── Chargement ───────────────────────────────────────────────────────────
    try:
        headers, rows = load_excel(filepath)
    except ImportError as e:
        print(err(f"  [X] Dependance manquante : {e}"))
        print(warn("      Installez avec : uv add xlrd openpyxl"))
        sys.exit(1)

    if not headers:
        print(err("  [X] Fichier vide ou illisible")); sys.exit(1)

    print(clr(BOLD, f"  {len(rows)} lignes de donnees  |  {len(headers)} colonnes"))
    print()

    # ── Affichage des colonnes ────────────────────────────────────────────────
    print(clr(BOLD, "  Colonnes detectees (ligne 1) :"))
    for i, h in enumerate(headers):
        print(f"    [{i:2d}]  {h}")
    print()

    # ── Auto-detection du mapping ─────────────────────────────────────────────
    mapping = {role: find_col(headers, *kws) for role, kws in KEYWORDS.items()}

    print(clr(BOLD, "  Mapping automatique propose :"))
    roles_fr = {
        "criterion": "Critere (requis)",
        "category":  "Categorie",
        "mandatory": "Obligatoire",
        "id":        "ID / Reference",
        "source":    "Section / Source",
    }
    all_ok = True
    for role, col in mapping.items():
        label = roles_fr[role]
        if col:
            status = ok(f"[OK]  -> colonne '{col}'")
        else:
            status = warn(f"[?]   -> non detecte") if role != "criterion" else err(f"[!!] -> NON DETECTE (critique !)")
            if role == "criterion": all_ok = False
        print(f"    {label:<30} {status}")

    if not all_ok:
        print()
        print(err("  [!!] La colonne 'Critere' n'a pas ete detectee automatiquement."))
        print(err("       Verifiez les noms de colonnes ci-dessus et ajoutez le mot-cle"))
        print(err("       manquant dans run_full_extraction.py (fonction find() dans ingest_excel)."))
    print()

    # ── Apercu des 5 premieres lignes ─────────────────────────────────────────
    col_crit = mapping["criterion"] or headers[0]
    col_cat  = mapping["category"]
    col_mand = mapping["mandatory"]
    col_id   = mapping["id"]

    idx_crit = headers.index(col_crit)
    idx_cat  = headers.index(col_cat)  if col_cat  and col_cat  in headers else None
    idx_mand = headers.index(col_mand) if col_mand and col_mand in headers else None
    idx_id   = headers.index(col_id)   if col_id   and col_id   in headers else None

    print(clr(BOLD, "  Apercu des 5 premieres lignes :"))
    print(clr(DIM, f"  {'ID':<10}  {'Categorie':<18}  {'Obligatoire':<12}  Critere"))
    print(clr(DIM, "  " + "-" * 80))

    for row in rows[:5]:
        rid   = row[idx_id][:10]   if idx_id   is not None else "—"
        cat   = row[idx_cat][:18]  if idx_cat  is not None else "—"
        mand  = row[idx_mand][:12] if idx_mand is not None else "—"
        crit  = row[idx_crit][:55]
        print(f"  {rid:<10}  {cat:<18}  {mand:<12}  {crit}")
    print()

    # ── Valeurs uniques : Categorie ───────────────────────────────────────────
    if idx_cat is not None:
        cats_found = sorted(set(row[idx_cat].strip() for row in rows if row[idx_cat].strip()))
        print(clr(BOLD, f"  Valeurs uniques dans '{col_cat}' ({len(cats_found)}) :"))
        for c in cats_found:
            norm = next((v for v in VALID_CATS if v.lower() == c.lower() or v.lower() in c.lower()), None)
            if norm:
                print(f"    {ok('[OK]')}  {c}  ->  sera mappe vers '{norm}'")
            else:
                print(f"    {warn('[?] ')}  {c}  ->  sera classe en 'Functional' (defaut)")
        print()

    # ── Valeurs uniques : Obligatoire ─────────────────────────────────────────
    if idx_mand is not None:
        mands_found = sorted(set(row[idx_mand].strip().upper() for row in rows if row[idx_mand].strip()))
        YES = {"YES","OUI","MANDATORY","OBLIGATOIRE","MUST","SHALL","M","1","TRUE"}
        NO  = {"NO","NON","OPTIONAL","OPTIONNEL","SHOULD","O","0","FALSE"}
        print(clr(BOLD, f"  Valeurs uniques dans '{col_mand}' ({len(mands_found)}) :"))
        for v in mands_found:
            if v in YES:   print(f"    {ok('[YES]')}  {v}")
            elif v in NO:  print(f"    {warn('[NO] ')}  {v}")
            else:           print(f"    {err('[?]  ')}  {v}  -> sera interprete comme 'Yes' (defaut)")
        print()

    # ── Estimation de l'extraction ────────────────────────────────────────────
    non_empty = sum(1 for row in rows if row[idx_crit].strip())
    print(clr(BOLD, f"  Estimation de l'extraction :"))
    print(f"    Lignes totales avec contenu    : {non_empty}")
    print(f"    Criteres attendus apres filtre : ~{max(0, non_empty - 5)} "
          f"(lignes vides / trop courtes ignorees)")
    print()

    # ── Verdict ───────────────────────────────────────────────────────────────
    if non_empty >= 200:
        print(ok(f"  [OK] Ce fichier devrait produire >= 200 criteres."))
        print(ok(f"       Vous pouvez lancer :"))
        print(clr(DIM, f"       python run_full_extraction.py rfp.txt --excel \"{filepath.name}\" --out output/"))
    elif non_empty >= 50:
        print(warn(f"  [~]  {non_empty} lignes — complement utile, mais insuffisant seul."))
        print(warn(f"       Combinez avec le PDF narratif pour atteindre 200+ criteres."))
    else:
        print(err(f"  [!!] Seulement {non_empty} lignes — ce fichier est peut-etre un mauvais onglet."))
        print(err(f"       Verifiez que vous lisez le bon onglet (onglet actif = premier onglet)."))
    print()


if __name__ == "__main__":
    main()
