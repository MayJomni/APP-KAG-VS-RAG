import re, sys

SRC = 'fast_server.py'
with open(SRC, encoding='utf-8') as f:
    code = f.read()
orig = len(code)

# ============================================================
# PATCH A : Ajouter imports difflib, csv (si absents)
# ============================================================
OLD_A = 'import os, sys, time, json, logging, re, string, io'
NEW_A = 'import os, sys, time, json, logging, re, string, io, csv, difflib'
assert OLD_A in code, 'IMPORT NOT FOUND'
code = code.replace(OLD_A, NEW_A, 1)
print('A OK : imports')

# ============================================================
# PATCH B : StreamingResponse dans fastapi.responses
# ============================================================
OLD_B = 'from fastapi.responses import HTMLResponse, JSONResponse'
NEW_B = 'from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse'
assert OLD_B in code, 'RESPONSE IMPORT NOT FOUND'
code = code.replace(OLD_B, NEW_B, 1)
print('B OK : StreamingResponse import')

# ============================================================
# PATCH C : Ajouter compliance_criteria global
# ============================================================
OLD_C = 'uploaded_raw_texts = []   # [{title, text}] texte brut KAG'
NEW_C = ('uploaded_raw_texts  = []   # [{title, text}] texte brut KAG\n'
         'compliance_criteria = []   # [{id, category, criterion, mandatory, source, chunk_id}]')
assert OLD_C in code, 'RAW_TEXTS GLOBAL NOT FOUND'
code = code.replace(OLD_C, NEW_C, 1)
print('C OK : compliance_criteria global')

# ============================================================
# PATCH D : Nouvelle fonction _chunk_for_compliance
#           apres _chunk_by_clause, avant Routing Agent
# ============================================================
INSERT_D = '''
def _chunk_for_compliance(title, text, chunk_size=400, overlap=50):
    """Decoupe le texte en chunks de ~300-500 mots pour extraction compliance.
    Un chunk trop large fait que le LLM resume au lieu de lister exhaustivement.
    """
    words = text.split()
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        w = words[i:i + chunk_size]
        if len(w) < 10:
            continue
        chunks.append({
            "title":    title,
            "text":     " ".join(w),
            "id":       f"{title}_cc{i}",
            "chunk_idx": len(chunks)
        })
    return chunks

def _deduplicate_criteria(criteria, threshold=0.85):
    """Deduplique les criteres avec similarite de texte > threshold.
    Utilise difflib.SequenceMatcher. Retourne (kept, removed_count).
    """
    kept = []
    removed = 0
    for crit in criteria:
        text_new = crit["criterion"].lower().strip()
        is_dup = False
        for existing in kept:
            ratio = difflib.SequenceMatcher(
                None, text_new, existing["criterion"].lower().strip()
            ).ratio()
            if ratio >= threshold:
                is_dup = True
                removed += 1
                break
        if not is_dup:
            kept.append(crit)
    return kept, removed

'''
ANCHOR_D = '# Signaux linguistiques pour la classification de questions'
assert ANCHOR_D in code, 'ANCHOR_D NOT FOUND'
code = code.replace(ANCHOR_D, INSERT_D + ANCHOR_D, 1)
print('D OK : _chunk_for_compliance + _deduplicate_criteria')

# ============================================================
# PATCH E : Nouveau EXTRACT_SYS (remplace l ancien)
# ============================================================
# Trouver et remplacer EXTRACT_SYS complet
start_tag = 'EXTRACT_SYS = ('
end_tag   = '\n)\n'
idx_start = code.index(start_tag)
idx_end   = code.index(end_tag, idx_start) + len(end_tag)

NEW_E = '''EXTRACT_SYS = (
    "You are a requirements extraction expert analyzing an RFP (Request for Proposal) document.\\n"
    "Read the text and extract EVERY distinct requirement or criterion mentioned, "
    "no matter how small. Do NOT summarize or merge multiple requirements into one line.\\n"
    "Each bullet point, each sentence stating an obligation, capability, or constraint "
    "is a SEPARATE criterion.\\n"
    "Output ONLY lines in this exact format:\\n"
    "Category|Criterion|Mandatory\\n"
    "Where:\\n"
    "- Category is one of: Functional, Technical, Operational, Security, Infrastructure, "
    "Administrative, Financial\\n"
    "- Criterion is the requirement itself, in a short precise sentence (max 20 words)\\n"
    "- Mandatory is Yes or No (Yes if the text uses must, shall, required; "
    "No if it uses should, recommended, optional)\\n"
    "No JSON, no numbering, no explanation, no headers, no introduction.\\n"
    "Example:\\n"
    "Technical|Solution must produce unified XML file for each bill|Yes\\n"
    "Functional|Vendor should support future bill design changes|No\\n"
    "Security|Vendor certified ISO 27001 is recommended|No"
)
'''
code = code[:idx_start] + NEW_E + code[idx_end:]
print('E OK : EXTRACT_SYS remplace')

# ============================================================
# PATCH F : Nouvelle build_kag_from_docs
# ============================================================
m1 = code.index('def build_kag_from_docs(docs):')
m2_tag = '\ndef build_kag_from_precomputed'
m2 = code.index(m2_tag, m1)

NEW_F = '''def build_kag_from_docs(docs):
    """Extrait les criteres de conformite (compliance requirements) depuis les docs.
    Format: Category|Criterion|Mandatory (200-300 criteres par doc RFP)
    - batch_size=1 : un chunk a la fois pour ne pas diluer l attention du LLM
    - max_tokens=3000 : budget suffisant pour lister tous les criteres d un chunk
    - chunk_size=400 mots : ~300-500 mots par chunk pour forcer l exhaustivite
    - deduplication automatique a 85% de similarite textuelle
    - time.sleep(2) entre batches pour respecter le rate limit Groq (6000 TPM)
    """
    global kag_graph, compliance_criteria
    kag_graph           = {"nodes": {}, "relations": {}}
    compliance_criteria = []

    # Chunking compliance (300-500 mots) depuis le texte brut
    if uploaded_raw_texts:
        all_chunks = []
        for src in uploaded_raw_texts:
            all_chunks.extend(_chunk_for_compliance(src["title"], src["text"],
                                                    chunk_size=400, overlap=50))
        print(f"[KAG] Compliance chunking: {len(uploaded_raw_texts)} fichiers -> {len(all_chunks)} chunks")
    else:
        all_chunks = docs
        print(f"[KAG] Fallback chunks RAG: {len(all_chunks)}")

    if not all_chunks:
        print("[KAG] Aucun document a traiter.")
        return

    # max_docs dynamique : couvrir tout le document sans plafonner
    max_docs = len(all_chunks)   # pas de cap fixe
    doc_list = all_chunks[:max_docs]

    # batch_size = 1 : un chunk a la fois pour exhaustivite maximale
    batches = [[doc] for doc in doc_list]
    print(f"[KAG] {len(doc_list)} chunks, batch_size=1, {len(batches)} appels LLM")

    raw_criteria = []   # accumulation avant deduplication

    for bi, batch in enumerate(batches):
        doc = batch[0]
        title     = doc.get("title", "doc")
        chunk_idx = doc.get("chunk_idx", bi)
        text      = doc.get("text", "")   # texte COMPLET, pas de [:400]
        combined  = f"[CHUNK {bi+1}/{len(batches)} - {title}]\\n{text}\\n"

        raw = ""
        for mdl in [MODEL_EXTRACT, MODEL_ANSWER]:
            r = llm(EXTRACT_SYS, combined, max_tokens=3000, model=mdl)
            if r and not r.startswith("ERREUR:"):
                raw = r; break
            print(f"[KAG] Batch {bi+1} {mdl} echoue: {r[:80]}")

        if not raw:
            print(f"[KAG] Batch {bi+1} ignore")
        else:
            print(f"[KAG] Batch {bi+1}/{len(batches)}: {repr(raw[:150])}")
            # Parser Category|Criterion|Mandatory
            for line in raw.split("\\n"):
                line = line.strip().lstrip("-*>.0123456789.) ")
                if not line or "|" not in line or len(line) < 6:
                    continue
                p = [x.strip() for x in line.split("|")]
                if len(p) < 3:
                    continue
                cat  = p[0]
                crit = p[1]
                mand = p[2].strip()

                # Normaliser Mandatory
                mand_bool = mand.lower() in ("yes","oui","true","1","mandatory","required","shall","must")
                mand_str  = "Yes" if mand_bool else "No"

                # Filtrer lignes header/exemple
                if any(b in cat.lower() for b in ["category","categorie","format","example","note"]): continue
                if not cat or not crit or len(crit) < 5: continue

                entry = {
                    "id":          f"REQ-{len(raw_criteria)+1:04d}",
                    "category":    cat,
                    "criterion":   crit,
                    "mandatory":   mand_str,
                    "source":      title,
                    "chunk_id":    f"{title}_chunk{chunk_idx}",
                    "compliance_status": ""   # a remplir manuellement
                }
                raw_criteria.append(entry)

                # Aussi dans kag_graph pour compatibilite avec /kag/graph
                key = f"{crit[:80]}||{cat}||{mand_str}"
                kag_graph["nodes"][crit[:80]] = {"doc": title, "categorie": cat, "obligatoire": mand_str}
                kag_graph["nodes"][cat]       = {"doc": title, "type": "category"}
                kag_graph["relations"][key]   = {"obligatoire": mand_str, "description": crit, "doc": title}

        # Rate-limit protection : 2s entre batches (Groq 6000 TPM free tier)
        if bi < len(batches) - 1:
            time.sleep(2)

    # Deduplication a 85% de similarite textuelle
    before = len(raw_criteria)
    compliance_criteria, removed = _deduplicate_criteria(raw_criteria, threshold=0.85)
    # Reassigner les IDs apres deduplication
    for i, c in enumerate(compliance_criteria):
        c["id"] = f"REQ-{i+1:04d}"

    print(f"[KAG] Extraction: {before} criteres bruts -> {removed} doublons supprimes -> {len(compliance_criteria)} uniques")
    print(f"[KAG] FINAL: {len(kag_graph['nodes'])} noeuds KAG, {len(compliance_criteria)} criteres conformite")

'''
code = code[:m1] + NEW_F + code[m2+1:]
print('F OK : build_kag_from_docs remplace')

# ============================================================
# PATCH G : kag_search compatible nouveau format dict
# ============================================================
OLD_G_s = 'def kag_search(question):\n    """Cherche exigences. Compatible bool(True) et dict meta."""'
assert OLD_G_s in code, 'KAG_SEARCH HEADER NOT FOUND'
# Remplace aussi le label format --[rel]--> par description
OLD_G_label = '"label":f"{src} --[{rel}]--> {desc} {badge}".strip()}'
NEW_G_label = '"label":f"{src} | {desc} [{badge}]".strip()}'
if OLD_G_label in code:
    code = code.replace(OLD_G_label, NEW_G_label, 1)
    print('G OK : kag_search label update')
else:
    print('G SKIP : label already updated or different')

# ============================================================
# PATCH H : /upload/clear reset compliance_criteria
# ============================================================
OLD_H = ('    global rag_index, documents, kag_graph, uploaded_files, uploaded_raw_texts\n'
         '    documents          = []\n'
         '    uploaded_files     = []\n'
         '    uploaded_raw_texts = []\n'
         '    rag_index          = None\n'
         '    kag_graph          = {"nodes": {}, "relations": {}}')
NEW_H = ('    global rag_index, documents, kag_graph, uploaded_files, uploaded_raw_texts, compliance_criteria\n'
         '    documents           = []\n'
         '    uploaded_files      = []\n'
         '    uploaded_raw_texts  = []\n'
         '    compliance_criteria = []\n'
         '    rag_index           = None\n'
         '    kag_graph           = {"nodes": {}, "relations": {}}')
assert OLD_H in code, 'CLEAR ENDPOINT NOT FOUND'
code = code.replace(OLD_H, NEW_H, 1)
print('H OK : /upload/clear reset compliance_criteria')

# ============================================================
# PATCH I : Nouveaux endpoints /compliance/list et /compliance/export
# ============================================================
# Trouver l endroit juste avant @app.get("/mlops/stats")
ANCHOR_I = '@app.get("/mlops/stats")'
assert ANCHOR_I in code, 'MLOPS STATS NOT FOUND'

NEW_ENDPOINTS = '''
# ── Compliance Matrix Endpoints ────────────────────────────────────────────────

@app.get("/compliance/list")
async def compliance_list(
    category: str = Query(None, description="Filtrer par categorie"),
    mandatory: str = Query(None, description="Yes ou No"),
    limit: int = Query(500, description="Nombre max de criteres retournes")
):
    """Liste tous les criteres de conformite extraits.
    Supporte filtrage par category et mandatory.
    """
    results = compliance_criteria
    if category:
        results = [c for c in results if c["category"].lower() == category.lower()]
    if mandatory:
        results = [c for c in results if c["mandatory"].lower() == mandatory.lower()]
    return {
        "total": len(compliance_criteria),
        "filtered": len(results),
        "criteria": results[:limit],
        "categories": list({c["category"] for c in compliance_criteria}),
        "mandatory_count":  sum(1 for c in compliance_criteria if c["mandatory"] == "Yes"),
        "optional_count":   sum(1 for c in compliance_criteria if c["mandatory"] == "No"),
    }


@app.get("/compliance/export")
async def compliance_export(fmt: str = Query("csv", description="Format: csv ou excel")):
    """Exporte la Compliance Matrix en CSV ou Excel.
    Colonnes : ID | Category | Criterion | Mandatory | Source | Compliance Status
    """
    if not compliance_criteria:
        raise HTTPException(status_code=404, detail="Aucun critere extrait. Uploadez un document RFP d abord.")

    cols = ["id", "category", "criterion", "mandatory", "source", "chunk_id", "compliance_status"]
    headers_fr = ["ID", "Categorie", "Critere", "Obligatoire", "Source", "Chunk", "Statut de Conformite"]

    if fmt.lower() == "excel":
        try:
            import openpyxl
            from openpyxl.styles import Font, PatternFill, Alignment
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Compliance Matrix"

            # Header row style
            header_fill = PatternFill("solid", fgColor="1F3A5F")
            header_font = Font(bold=True, color="FFFFFF")
            for col_idx, h in enumerate(headers_fr, 1):
                cell = ws.cell(row=1, column=col_idx, value=h)
                cell.fill  = header_fill
                cell.font  = header_font
                cell.alignment = Alignment(horizontal="center")

            # Data rows
            for row_idx, crit in enumerate(compliance_criteria, 2):
                for col_idx, key in enumerate(cols, 1):
                    ws.cell(row=row_idx, column=col_idx, value=crit.get(key, ""))
                # Color Yes/No
                mand_cell = ws.cell(row=row_idx, column=4)
                if crit.get("mandatory") == "Yes":
                    mand_cell.fill = PatternFill("solid", fgColor="C6EFCE")
                    mand_cell.font = Font(color="276221", bold=True)
                else:
                    mand_cell.fill = PatternFill("solid", fgColor="FFEB9C")
                    mand_cell.font = Font(color="9C6500")

            # Column widths
            ws.column_dimensions["A"].width = 12
            ws.column_dimensions["B"].width = 18
            ws.column_dimensions["C"].width = 55
            ws.column_dimensions["D"].width = 14
            ws.column_dimensions["E"].width = 28
            ws.column_dimensions["F"].width = 24
            ws.column_dimensions["G"].width = 28

            buf = io.BytesIO()
            wb.save(buf)
            buf.seek(0)
            return StreamingResponse(
                buf,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": "attachment; filename=compliance_matrix.xlsx"}
            )
        except ImportError:
            # Fallback CSV si openpyxl absent
            pass

    # CSV export (defaut)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    writer.writerow(dict(zip(cols, headers_fr)))   # header FR
    for crit in compliance_criteria:
        writer.writerow(crit)
    buf.seek(0)
    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode("utf-8-sig")),   # BOM pour Excel FR
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=compliance_matrix.csv"}
    )


@app.get("/compliance/stats")
async def compliance_stats():
    """Statistiques sur les criteres extraits."""
    by_cat = {}
    for c in compliance_criteria:
        by_cat.setdefault(c["category"], {"total": 0, "mandatory": 0, "optional": 0})
        by_cat[c["category"]]["total"] += 1
        if c["mandatory"] == "Yes":
            by_cat[c["category"]]["mandatory"] += 1
        else:
            by_cat[c["category"]]["optional"] += 1
    return {
        "total_criteria":    len(compliance_criteria),
        "mandatory_total":   sum(1 for c in compliance_criteria if c["mandatory"] == "Yes"),
        "optional_total":    sum(1 for c in compliance_criteria if c["mandatory"] == "No"),
        "by_category":       by_cat,
        "sources":           list({c["source"] for c in compliance_criteria}),
        "kag_nodes":         len(kag_graph["nodes"]),
        "kag_relations":     len(kag_graph["relations"])
    }


'''
code = code.replace(ANCHOR_I, NEW_ENDPOINTS + ANCHOR_I, 1)
print('I OK : /compliance/list, /export, /stats endpoints')

# ============================================================
# Ecriture finale
# ============================================================
with open(SRC, 'w', encoding='utf-8') as f:
    f.write(code)
print(f'DONE: {orig} -> {len(code)} bytes (+{len(code)-orig})')
