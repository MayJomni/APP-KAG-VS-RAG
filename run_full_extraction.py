"""
run_full_extraction.py — Extraction hybride RFP : Excel direct + LLM narratif.

Deux modes d'ingestion combines :
  A) Fichiers Excel de requirements (.xls/.xlsx) -> ingestion directe, 0 appel LLM
  B) Fichiers narratifs (PDF/DOCX/TXT)           -> extraction LLM (build_kag_from_docs)

Usage:
  # Extraction LLM seule (texte narratif)
  python run_full_extraction.py rfp.txt

  # Excel direct seul (matrice deja structuree)
  python run_full_extraction.py --excel requirements.xls

  # Les deux combinee (recommande pour 200-300 criteres)
  python run_full_extraction.py rfp.txt --excel requirements.xls

  # Plusieurs fichiers + dossier de sortie
  python run_full_extraction.py rfp.txt annexe.pdf --excel req.xls --out output/

Options:
  --excel <fichier.xls>   Fichier Excel de requirements (ingestion directe, sans LLM)
  --out   <dossier/>      Dossier de sortie pour l'export Excel (defaut: .)
"""

import os, sys, time, difflib, io, argparse
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# ── Couleurs ANSI ────────────────────────────────────────────────────────────────
RESET = "\033[0m"; BOLD = "\033[1m"; GREEN = "\033[32m"
YELLOW = "\033[33m"; RED = "\033[31m"; CYAN = "\033[36m"; DIM = "\033[2m"

def clr(c, t): return f"{c}{t}{RESET}"
def ok(t):     return clr(GREEN, t)
def warn(t):   return clr(YELLOW, t)
def err(t):    return clr(RED, t)
def hdr(t):    print(clr(BOLD + CYAN, t))

# ── Config ───────────────────────────────────────────────────────────────────────
load_dotenv()
GROQ_KEY      = os.getenv("GROQ_API_KEY", "")
MODEL_EXTRACT = "openai/gpt-oss-20b"
MODEL_ANSWER  = "openai/gpt-oss-120b"   # fallback
SLEEP_BETWEEN = 2
DUP_THRESHOLD = 0.85
TARGET_MIN    = 200

VALID_CATS = {
    "Functional", "Technical", "Operational",
    "Security", "Infrastructure", "Administrative", "Financial"
}

PROMPT_FILTER = {
    "shall respond within 2 seconds", "unified xml file", "future bill design",
    "iso 27001 is recommended", "criterion", "category", "mandatory",
    "example", "format", "note:", "header", "where:", "output only",
}

EXTRACT_SYS = (
    "You are a requirements extraction expert analyzing an RFP (Request for Proposal) document.\n"
    "Read the text and extract EVERY distinct requirement or criterion mentioned, "
    "no matter how small. Do NOT summarize or merge multiple requirements into one line.\n"
    "Each bullet point, each sentence stating an obligation, capability, or constraint "
    "is a SEPARATE criterion.\n"
    "Output ONLY lines in this exact format:\n"
    "Category|Criterion|Mandatory\n"
    "Where:\n"
    "- Category is one of: Functional, Technical, Operational, Security, Infrastructure, "
    "Administrative, Financial\n"
    "- Criterion is the requirement itself, in a short precise sentence (max 20 words)\n"
    "- Mandatory is Yes or No (Yes if the text uses 'must', 'shall', 'required'; "
    "No if it uses 'should', 'recommended', 'optional')\n"
    "No JSON, no numbering, no explanation, no headers, no introduction.\n"
    "Example:\n"
    "Technical|Solution must produce unified XML file for each bill|Yes\n"
    "Functional|Vendor should support future bill design changes|No\n"
    "Security|Vendor certified ISO 27001 is recommended|No"
)

# ── Utilitaires communs ──────────────────────────────────────────────────────────

def _chunk_by_clause(title, text, min_words=6):
    import re as _r
    chunks, cur = [], ""
    pat = _r.compile(
        r"^(\d+[\.\)\-]\s|[a-zA-Z][\.\)\-]\s|[*\-\u2022\u2013]\s"
        r"|[A-Z]{2,}\s*:|Section\s+\d|Article\s+\d|Clause\s+\d"
        r"|Requirement\s+\d|REQ-\d)",
        _r.IGNORECASE
    )
    def flush(b):
        b = b.strip()
        if len(b.split()) >= min_words:
            chunks.append({"title": title, "text": b, "id": f"{title}_cl{len(chunks)}"})
    for line in text.splitlines():
        s = line.strip()
        if not s:          flush(cur); cur = ""
        elif pat.match(s) and cur: flush(cur); cur = s
        else:              cur = (cur + " " + s).strip() if cur else s
    flush(cur)
    if len(chunks) < 3:
        chunks = [{"title": title, "text": p.strip(), "id": f"{title}_p{i}"}
                  for i, p in enumerate(_r.split(r"\n\s*\n", text)) if len(p.split()) >= min_words]
    if len(chunks) < 2:
        w = text.split()
        chunks = [{"title": title, "text": " ".join(w[i:i+80]), "id": f"{title}_w{i}"}
                  for i in range(0, len(w), 80) if len(w[i:i+80]) >= min_words]
    return chunks


def _is_dup(a, b):
    return difflib.SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio() >= DUP_THRESHOLD


def _norm_cat(raw):
    n = next((c for c in VALID_CATS if c.lower() == raw.lower()), None)
    if not n:
        n = next((c for c in VALID_CATS if c.lower() in raw.lower()), "Functional")
    return n


# ── Mode A : Ingestion directe Excel (0 appel LLM) ──────────────────────────────

def ingest_excel(filepath: Path, all_criteria: list, seen_texts: list) -> dict:
    """Lit chaque ligne du fichier Excel comme un critere pre-structure."""
    ext = filepath.suffix.lower()
    rows, headers = [], []
    content = filepath.read_bytes()

    try:
        if ext in (".xlsx", ".xlsm"):
            import openpyxl
            wb = openpyxl.load_workbook(io.BytesIO(content), data_only=True)
            ws = wb.active
            raw_rows = list(ws.iter_rows(values_only=True))
            if not raw_rows:
                return {"error": "Feuille vide", "ingested": 0}
            headers = [str(h).strip() if h is not None else f"col_{i}"
                       for i, h in enumerate(raw_rows[0])]
            for row in raw_rows[1:]:
                if all(v is None for v in row): continue
                rows.append(dict(zip(headers,
                    [str(v).strip() if v is not None else "" for v in row])))

        elif ext == ".xls":
            try:
                import xlrd
                wb = xlrd.open_workbook(file_contents=content)
                ws = wb.sheet_by_index(0)
                headers = [str(ws.cell_value(0, c)).strip() for c in range(ws.ncols)]
                for r in range(1, ws.nrows):
                    row = {headers[c]: str(ws.cell_value(r, c)).strip() for c in range(ws.ncols)}
                    if any(v for v in row.values()): rows.append(row)
            except ImportError:
                return {"error": "xlrd manquant — pip install xlrd", "ingested": 0}
        else:
            return {"error": f"Format non supporte : {ext}", "ingested": 0}

    except Exception as e:
        return {"error": str(e), "ingested": 0}

    # Auto-detection des colonnes
    def find(hdrs, *kws):
        for kw in kws:
            for h in hdrs:
                if kw.lower() in h.lower(): return h
        return None

    col_crit = find(headers, "requirement", "criterion", "description", "exigence",
                    "critere", "specification", "text", "wording") or headers[0]
    col_cat  = find(headers, "category", "categorie", "type", "area", "domain",
                    "domaine", "functional area")
    col_mand = find(headers, "mandatory", "obligatoire", "priority", "required",
                    "compliance", "must", "priorite")
    col_id   = find(headers, "id", "ref", "no.", "num", "req_id", "req id")
    col_src  = find(headers, "source", "section", "chapter", "chapitre")

    print(clr(DIM, f"    En-tetes detectes : {headers[:8]}{'...' if len(headers)>8 else ''}"))
    print(clr(DIM, f"    Mapping  critere={col_crit} | categorie={col_cat} "
                   f"| obligatoire={col_mand} | id={col_id}"))

    ingested = dup = skipped = 0

    for row in rows:
        criterion = row.get(col_crit, "").strip()
        if not criterion or len(criterion) < 4: skipped += 1; continue

        raw_cat  = row.get(col_cat, "").strip() if col_cat else ""
        raw_mand = row.get(col_mand, "").strip().upper() if col_mand else ""
        ext_id   = row.get(col_id, "").strip() if col_id else ""
        int_src  = row.get(col_src, "").strip() if col_src else ""

        cat_norm = _norm_cat(raw_cat) if raw_cat else "Functional"

        YES_VALS = {"YES", "OUI", "MANDATORY", "OBLIGATOIRE", "MUST", "SHALL", "M", "1", "TRUE"}
        NO_VALS  = {"NO", "NON", "OPTIONAL", "OPTIONNEL", "SHOULD", "O", "0", "FALSE"}
        mandatory = "Yes" if raw_mand in YES_VALS else ("No" if raw_mand in NO_VALS else "Yes")

        # Deduplication inter-sources
        if any(_is_dup(criterion, s) for s in seen_texts):
            dup += 1; continue
        seen_texts.append(criterion)

        all_criteria.append({
            "id":                f"CR-{len(all_criteria)+1:03d}",
            "category":          cat_norm,
            "criterion":         criterion,
            "mandatory":         mandatory,
            "source":            filepath.name,
            "chunk_id":          ext_id or int_src or f"row_{ingested+1}",
            "compliance_status": "To Be Assessed",
            "source_type":       "Excel (direct)",
            "ext_id":            ext_id,
        })
        ingested += 1

    return {
        "ingested": ingested, "skipped": skipped, "dup": dup,
        "total_rows": len(rows), "headers": headers,
        "mapping": {"criterion": col_crit, "category": col_cat,
                    "mandatory": col_mand, "id": col_id}
    }


# ── Mode B : Extraction LLM depuis texte narratif ────────────────────────────────

def _parse_file_text(filepath: Path) -> str:
    """Extrait le texte brut depuis TXT, PDF ou DOCX."""
    ext = filepath.suffix.lower()
    content = filepath.read_bytes()
    if ext in (".txt", ".md"):
        return content.decode("utf-8", errors="ignore")
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(content))
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        except ImportError:
            print(warn("  [!] pypdf manquant — pip install pypdf")); return ""
    if ext == ".docx":
        try:
            from docx import Document as D
            doc = D(io.BytesIO(content))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError:
            print(warn("  [!] python-docx manquant — pip install python-docx")); return ""
    return content.decode("utf-8", errors="ignore")


def llm_call(user_prompt):
    client = Groq(api_key=GROQ_KEY)
    for model in [MODEL_EXTRACT, MODEL_ANSWER]:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": EXTRACT_SYS},
                          {"role": "user",   "content": user_prompt}],
                temperature=0.0, max_tokens=3000,
            )
            raw = resp.choices[0].message.content.strip()
            if raw: return raw, model
        except Exception as e:
            print(warn(f"  [!] {model} : {e}"))
    return "", ""


def parse_llm_output(raw, doc_title, chunk_id):
    criteria, malformed = [], 0
    for line in raw.split("\n"):
        line = line.strip().lstrip("-*>.0123456789.) ")
        if not line: continue
        if "|" not in line:
            if len(line) > 10: malformed += 1
            continue
        parts = [x.strip() for x in line.split("|")]
        if len(parts) < 3: malformed += 1; continue
        cat, criterion, mand_raw = parts[0], parts[1], parts[2].upper()
        mandatory = ("Yes" if mand_raw in ("YES","OUI") else
                     "No"  if mand_raw in ("NO","NON") else
                     "Yes" if any(w in mand_raw.lower()
                                  for w in ["yes","oui","must","shall","required"]) else "No")
        cat_norm = _norm_cat(cat)
        if any(b in criterion.lower() for b in PROMPT_FILTER): continue
        if not criterion or len(criterion) < 5: continue
        criteria.append({
            "category": cat_norm, "criterion": criterion,
            "mandatory": mandatory, "source": doc_title, "chunk_id": chunk_id,
        })
    return criteria, malformed


def extract_llm(narrative_files: list, all_criteria: list, seen_texts: list) -> dict:
    """Lance l'extraction LLM sur une liste de fichiers narratifs."""
    total_chunks = total_raw = total_malf = dup = 0

    for fpath in narrative_files:
        text  = _parse_file_text(fpath)
        if not text.strip():
            print(err(f"  [X] Impossible d'extraire le texte de {fpath.name}")); continue
        chunks = _chunk_by_clause(fpath.stem, text)
        print(clr(DIM, f"    {fpath.name} -> {len(chunks)} chunks"))
        total_chunks += len(chunks)

        for bi, chunk in enumerate(chunks):
            n_words = len(chunk["text"].split())
            pct = int((bi + 1) / len(chunks) * 100)
            bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
            print(f"\r    [{bar}] {pct:3d}%  chunk {bi+1}/{len(chunks)} "
                  f"({n_words}w)  uniques={len(all_criteria)}  dup={dup}   ",
                  end="", flush=True)

            user_prompt = f"[CLAUSE {bi+1} - {chunk['title']}]\n{chunk['text']}\n"
            raw, _ = llm_call(user_prompt)

            if raw:
                crits, malf = parse_llm_output(raw, fpath.name, chunk["id"])
                total_raw  += len(crits)
                total_malf += malf
                for c in crits:
                    if any(_is_dup(c["criterion"], s) for s in seen_texts):
                        dup += 1; continue
                    seen_texts.append(c["criterion"])
                    all_criteria.append({
                        "id":                f"CR-{len(all_criteria)+1:03d}",
                        "category":          c["category"],
                        "criterion":         c["criterion"],
                        "mandatory":         c["mandatory"],
                        "source":            c["source"],
                        "chunk_id":          c["chunk_id"],
                        "compliance_status": "To Be Assessed",
                        "source_type":       "LLM extraction",
                        "ext_id":            "",
                    })

            if bi < len(chunks) - 1:
                time.sleep(SLEEP_BETWEEN)

        print()  # fin barre de progression

    return {"chunks": total_chunks, "raw": total_raw, "malf": total_malf, "dup": dup}


# ── Export Excel ─────────────────────────────────────────────────────────────────

def export_excel(criteria: list, output_path: Path) -> bool:
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment
    except ImportError:
        print(err("  [X] openpyxl manquant — pip install openpyxl")); return False

    # Colonnes : on ajoute "Source Document" et "Type d'extraction"
    COLS = ["id", "category", "criterion", "mandatory",
            "source", "chunk_id", "source_type", "compliance_status"]
    HDRS = ["ID", "Categorie", "Critere", "Obligatoire",
            "Document Source", "Ref / Chunk", "Type d'extraction", "Statut de Conformite"]

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Compliance Matrix"

    fill_hdr = PatternFill("solid", fgColor="1F3A5F")
    font_hdr = Font(bold=True, color="FFFFFF")
    for ci, h in enumerate(HDRS, 1):
        cell = ws.cell(row=1, column=ci, value=h)
        cell.fill = fill_hdr; cell.font = font_hdr
        cell.alignment = Alignment(horizontal="center")

    fill_yes  = PatternFill("solid", fgColor="C6EFCE")
    fill_no   = PatternFill("solid", fgColor="FFEB9C")
    fill_xls  = PatternFill("solid", fgColor="DDEEFF")  # bleu clair = Excel direct
    fill_llm  = PatternFill("solid", fgColor="FFF2CC")  # jaune clair = LLM
    font_yes  = Font(color="276221", bold=True)
    font_no   = Font(color="9C6500")

    for ri, c in enumerate(criteria, 2):
        for ci, key in enumerate(COLS, 1):
            ws.cell(row=ri, column=ci, value=c.get(key, ""))
        # Couleur colonne Obligatoire
        mc = ws.cell(row=ri, column=4)
        if c.get("mandatory") == "Yes": mc.fill = fill_yes; mc.font = font_yes
        else:                           mc.fill = fill_no;  mc.font = font_no
        # Couleur colonne Type d'extraction
        tc = ws.cell(row=ri, column=7)
        if c.get("source_type") == "Excel (direct)": tc.fill = fill_xls
        else:                                          tc.fill = fill_llm

    for col, w in zip("ABCDEFGH", [12, 20, 65, 14, 35, 28, 20, 30]):
        ws.column_dimensions[col].width = w

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    wb.save(output_path)
    return True


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extraction hybride RFP : Excel direct + LLM narratif")
    parser.add_argument("narrative_files", nargs="*",
                        help="Fichiers narratifs RFP (TXT, PDF, DOCX)")
    parser.add_argument("--excel", nargs="+", metavar="FILE",
                        help="Fichier(s) Excel de requirements (ingestion directe)")
    parser.add_argument("--out", default=".", metavar="DIR",
                        help="Dossier de sortie (defaut: .)")
    args = parser.parse_args()

    narrative_paths = [Path(f) for f in args.narrative_files]
    excel_paths     = [Path(f) for f in (args.excel or [])]
    output_dir      = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not narrative_paths and not excel_paths:
        # Defaut : utilise test_ooredoo_rfp.txt si present
        default = Path("test_ooredoo_rfp.txt")
        if default.exists():
            narrative_paths = [default]
            print(warn(f"  Aucun fichier specifie — utilisation de {default}"))
        else:
            print(err("  Aucun fichier specifie.")); parser.print_help(); sys.exit(1)

    for p in narrative_paths + excel_paths:
        if not p.exists():
            print(err(f"  [X] Fichier introuvable : {p}")); sys.exit(1)

    if narrative_paths and not GROQ_KEY:
        print(err("  [X] GROQ_API_KEY manquante — verifiez .env")); sys.exit(1)

    all_criteria = []
    seen_texts   = []
    stats_excel  = {"ingested": 0, "skipped": 0, "dup": 0, "files": 0}
    stats_llm    = {"chunks": 0, "raw": 0, "malf": 0, "dup": 0, "files": 0}

    print()
    hdr("=== EXTRACTION HYBRIDE RFP ===")
    print()

    # ── Phase A : Excel direct ───────────────────────────────────────────────
    if excel_paths:
        hdr(f"--- PHASE A : Ingestion Excel directe ({len(excel_paths)} fichier(s)) ---")
        t0 = time.time()
        for xp in excel_paths:
            print(f"\n  {clr(BOLD, xp.name)}")
            res = ingest_excel(xp, all_criteria, seen_texts)
            if "error" in res:
                print(err(f"  [X] {res['error']}")); continue
            print(ok(f"  [OK] {res['ingested']} criteres ingeres "
                     f"| {res['skipped']} ignores | {res['dup']} doublons"))
            stats_excel["ingested"] += res["ingested"]
            stats_excel["skipped"]  += res["skipped"]
            stats_excel["dup"]      += res["dup"]
            stats_excel["files"]    += 1
        print(f"\n  Sous-total Phase A : {clr(GREEN + BOLD, str(stats_excel['ingested']))} criteres directs"
              f" en {round(time.time()-t0, 1)}s")
        print()

    # ── Phase B : Extraction LLM ─────────────────────────────────────────────
    if narrative_paths:
        n_chunks_est = sum(len(_chunk_by_clause(p.stem,
            p.read_text(encoding="utf-8", errors="ignore")))
            for p in narrative_paths)
        eta = n_chunks_est * (SLEEP_BETWEEN + 2)
        hdr(f"--- PHASE B : Extraction LLM ({len(narrative_paths)} fichier(s), "
            f"~{n_chunks_est} chunks, ETA ~{eta//60}m{eta%60:02d}s) ---")
        t0 = time.time()
        res = extract_llm(narrative_paths, all_criteria, seen_texts)
        print(f"\n  Sous-total Phase B : {clr(GREEN + BOLD, str(res['raw']))} bruts "
              f"-> {clr(GREEN, str(len([c for c in all_criteria if c.get('source_type')=='LLM extraction'])))} uniques"
              f" | {res['malf']} malformes | {res['dup']} doublons"
              f" en {round(time.time()-t0, 1)}s")
        stats_llm.update(res); stats_llm["files"] = len(narrative_paths)
        print()

    # ── Reassigner les IDs sequentiels ──────────────────────────────────────
    for i, c in enumerate(all_criteria):
        c["id"] = f"CR-{i+1:03d}"

    # ── Statistiques finales ─────────────────────────────────────────────────
    xls_count = sum(1 for c in all_criteria if c.get("source_type") == "Excel (direct)")
    llm_count = sum(1 for c in all_criteria if c.get("source_type") == "LLM extraction")
    mand_c    = sum(1 for c in all_criteria if c["mandatory"] == "Yes")
    opt_c     = sum(1 for c in all_criteria if c["mandatory"] == "No")
    cats      = {}
    for c in all_criteria: cats[c["category"]] = cats.get(c["category"], 0) + 1

    print()
    hdr("=== RESULTATS FINAUX ===")
    print()
    print(f"  {clr(BOLD,'Criteres UNIQUES     ')} : "
          + clr(GREEN + BOLD, str(len(all_criteria)))
          + f"  ({mand_c} obligatoires | {opt_c} optionnels)")
    print(f"  {clr(BOLD,'  dont Excel direct  ')} : {clr(CYAN, str(xls_count))}")
    print(f"  {clr(BOLD,'  dont LLM extrait   ')} : {clr(YELLOW, str(llm_count))}")
    print()

    print(clr(BOLD, "  Repartition par categorie :"))
    for cat in sorted(VALID_CATS):
        n   = cats.get(cat, 0)
        pct = round(n / max(len(all_criteria), 1) * 100)
        bar = clr(GREEN if n > 5 else DIM, "#" * min(n, 50))
        print(f"    {cat:<22} {n:4d}  {pct:3d}%  {bar}")
    print()

    # ── Alerte seuil < 200 ───────────────────────────────────────────────────
    if len(all_criteria) < TARGET_MIN:
        print(clr(RED + BOLD,
            f"  [!] SEUIL NON ATTEINT : {len(all_criteria)} criteres < {TARGET_MIN} vises"))
        print(warn("  Le document source ne couvre pas assez de clauses distinctes."))
        print(warn("  Suggestions :"))
        print(warn("  1. Ajoutez le fichier Excel de requirements via --excel"))
        print(warn("     (ex: --excel 'OT Bill Generation Solution Requirement v1.xls')"))
        print(warn("  2. Ajoutez le PDF fonctionnel : --excel annexe.pdf"))
        print(warn("  3. Relancez : python run_full_extraction.py rfp.txt --excel req.xls"))
        print()
    else:
        print(ok(f"  [OK] {len(all_criteria)} criteres — seuil de {TARGET_MIN} atteint."))
        print()

    # ── Export Excel ─────────────────────────────────────────────────────────
    if not all_criteria:
        print(err("  [X] Aucun critere extrait — export annule.")); sys.exit(1)

    excel_out = output_dir / "compliance_matrix_RFP_bill_generation.xlsx"
    print(clr(DIM, f"  Export -> {excel_out}"))
    if export_excel(all_criteria, excel_out):
        size_kb = excel_out.stat().st_size // 1024
        print(ok(f"  [OK] {excel_out.name}  ({size_kb} Ko | {len(all_criteria)} lignes | 8 colonnes)"))
        print(clr(DIM, "       Colonnes : ID | Categorie | Critere | Obligatoire |"))
        print(clr(DIM, "                  Document Source | Ref/Chunk | Type extraction | Statut"))
        print(clr(DIM, "       Couleurs : bleu=Excel direct | jaune=LLM | vert=Oui | orange=Non"))
    print()
    hdr("=== TERMINE ===")
    print(f"  {xls_count} Excel direct + {llm_count} LLM = "
          + clr(GREEN + BOLD, f"{len(all_criteria)} criteres uniques"))
    print()


if __name__ == "__main__":
    main()
