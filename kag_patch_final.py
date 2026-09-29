import re, sys

src = 'fast_server.py'
with open(src, encoding='utf-8') as f:
    code = f.read()
orig = len(code)

# P1: global
code = code.replace(
    'uploaded_files = []   # [{name, size_kb, chunks, status}]',
    'uploaded_files     = []   # [{name, size_kb, chunks, status}]\nuploaded_raw_texts = []   # [{title, text}] texte brut KAG',
    1)
print('P1 OK')

# P2: _chunk_by_clause
CLAUSE = '''
def _chunk_by_clause(title, text, min_words=6):
    import re as _r
    chunks = []
    lines = text.splitlines()
    cur = ""
    pat = _r.compile(
        r"^(\\d+[\\.)\\-]\\s|[a-zA-Z][\\.)\\-]\\s|[*\\-\\u2022\\u2013]\\s"
        r"|[A-Z]{2,}\\s*:|Section\\s+\\d|Article\\s+\\d|Clause\\s+\\d|Requirement\\s+\\d|REQ-\\d)",
        _r.IGNORECASE)
    def flush(b):
        b = b.strip()
        if len(b.split()) >= min_words:
            chunks.append({"title": title, "text": b, "id": f"{title}_cl{len(chunks)}"})
    for line in lines:
        s = line.strip()
        if not s: flush(cur); cur = ""
        elif pat.match(s) and cur: flush(cur); cur = s
        else: cur = (cur + " " + s).strip() if cur else s
    flush(cur)
    if len(chunks) < 3:
        chunks = [{"title": title, "text": p.strip(), "id": f"{title}_p{i}"}
                  for i, p in enumerate(_r.split(r"\\n\\s*\\n", text)) if len(p.split()) >= min_words]
    if len(chunks) < 2:
        w = text.split()
        chunks = [{"title": title, "text": " ".join(w[i:i+80]), "id": f"{title}_w{i}"}
                  for i in range(0, len(w), 80) if len(w[i:i+80]) >= min_words]
    return chunks
'''
ANCHOR = '# Signaux linguistiques pour la classification de questions'
assert ANCHOR in code, 'ANCHOR NOT FOUND'
code = code.replace(ANCHOR, CLAUSE + ANCHOR, 1)
print('P2 OK')

# P3: EXTRACT_SYS
old3 = '''EXTRACT_SYS = (
    "You are a knowledge graph extractor. Read the document text and extract factual triplets.\\n"
    "Rules:\\n"
    "- Output ONLY lines in format: Subject|Predicate|Object\\n"
    "- Use short, precise labels (2-5 words max per element)\\n"
    "- Extract 3 to 8 triplets per passage\\n"
    "- No JSON, no numbering, no explanation, no headers\\n"
    "- Examples of valid triplets:\\n"
    "  Ooredoo Tunisia|is|telecom company\\n"
    "  Contract|requires|NDA signature\\n"
    "  Article 5|states|payment in 30 days\\n"
    "  Author|published|research paper 2023\\n"
    "  Product A|costs|200 USD\\n"
    "- Extract from ANY document type: legal, technical, academic, commercial"
)'''
new3 = '''EXTRACT_SYS = (
    "You are a requirements extraction engine for compliance matrices.\\n"
    "For EACH distinct requirement, criterion, obligation, or specification, output ONE line.\\n"
    "Format: Critere|Categorie|Description|Obligatoire\\n"
    "- Critere: label 2-6 words (no pipe)\\n"
    "- Categorie: Technical|Financial|Legal|Functional|Security|Administrative|Other\\n"
    "- Description: verbatim wording max 25 words (no pipe)\\n"
    "- Obligatoire: OUI (must/shall/required) or NON (should/may/optional)\\n"
    "Extract EVERY requirement as separate line. No merge. No summary. No JSON.\\n"
    "NDA Signature|Legal|Vendor must sign Non-Disclosure Agreement|OUI\\n"
    "Response Time|Technical|System shall respond within 2 seconds|OUI\\n"
    "Multi-language|Functional|Solution should support Arabic and French|NON\\n"
    "Data Encryption|Security|All data encrypted at rest AES-256|OUI"
)'''
assert old3 in code, 'EXTRACT_SYS NOT FOUND'
code = code.replace(old3, new3, 1)
print('P3 OK')

# P4: build_kag_from_docs
m1 = code.index('def build_kag_from_docs(docs):')
m2 = code.index('\ndef build_kag_from_precomputed')
new4 = '''def build_kag_from_docs(docs):
    """KAG: clause-chunking + format Critere|Categorie|Description|Obligatoire."""
    global kag_graph
    kag_graph = {"nodes": {}, "relations": {}}
    clause_chunks = []
    if uploaded_raw_texts:
        for src in uploaded_raw_texts:
            clause_chunks.extend(_chunk_by_clause(src["title"], src["text"]))
        print(f"[KAG] Clause-chunking: {len(uploaded_raw_texts)} fichiers -> {len(clause_chunks)} clauses")
    else:
        clause_chunks = docs
        print(f"[KAG] Fallback: {len(clause_chunks)} chunks")
    if not clause_chunks:
        print("[KAG] Aucun doc."); return
    n = len(clause_chunks)
    if n > 100: bsz, mx = 5, 300
    elif n > 30: bsz, mx = 3, 150
    else: bsz, mx = 1, 80
    dl = clause_chunks[:mx]
    batches = [dl[i:i+bsz] for i in range(0, len(dl), bsz)]
    print(f"[KAG] {len(dl)} clauses bsz={bsz} -> {len(batches)} appels LLM")
    for bi, batch in enumerate(batches):
        combined = ""
        for ix, doc in enumerate(batch):
            combined += f"[CLAUSE {bi*bsz+ix+1} - {doc.get('title','doc')}]\\n{doc.get('text','')}\\n\\n"
        raw = ""
        for mdl in [MODEL_EXTRACT, MODEL_ANSWER]:
            r = llm(EXTRACT_SYS, combined, max_tokens=2000, model=mdl)
            if r and not r.startswith("ERREUR:"): raw = r; break
            print(f"[KAG] Batch {bi+1} {mdl}: {r[:80]}")
        if not raw: print(f"[KAG] Batch {bi+1} ignore"); continue
        print(f"[KAG] Batch {bi+1}/{len(batches)}: {repr(raw[:150])}")
        found = 0
        for line in raw.split("\\n"):
            line = line.strip().lstrip("-*>.0123456789.) ")
            if not line or "|" not in line or len(line) < 8: continue
            p = [x.strip() for x in line.split("|")]
            if len(p) < 3: continue
            crit=p[0]; cat=p[1] if len(p)>1 else "Other"
            desc=p[2] if len(p)>2 else ""; obl=p[3].upper() if len(p)>3 else "?"
            if obl not in ("OUI","NON"):
                obl = "OUI" if any(w in obl.lower() for w in ["oui","yes","must","shall","mandatory"]) else "NON"
            if any(b in crit.lower() for b in ["critere","format","entity","example","note:","header","field","subject"]): continue
            if not crit or len(crit) < 3: continue
            dt = batch[0].get("title","")
            kag_graph["nodes"][crit] = {"doc": dt, "categorie": cat, "obligatoire": obl}
            kag_graph["nodes"][cat]  = {"doc": dt, "type": "category"}
            kag_graph["relations"][f"{crit}||{cat}||{desc[:120]}"] = {"obligatoire": obl, "description": desc, "doc": dt}
            found += 1
        print(f"[KAG] Batch {bi+1} -> {found} exigences")
    print(f"[KAG] FINAL: {len(kag_graph['nodes'])} noeuds, {len(kag_graph['relations'])} relations")
'''
code = code[:m1] + new4 + code[m2+1:]
print('P4 OK')

# P5: kag_search
old5_s = 'def kag_search(question):\n    """Cherche les triplets pertinents.'
old5_e = '    return relevant[:20]\n\ndef kag_answer'
assert old5_s in code and old5_e in code, 'KAG_SEARCH NOT FOUND'
s5 = code.index(old5_s); e5 = code.index(old5_e)+len(old5_e)
new5 = '''def kag_search(question):
    """Cherche exigences. Compatible bool(True) et dict meta."""
    q_words = [w for w in question.lower().split() if len(w) > 3]
    relevant = []
    for key, meta in kag_graph["relations"].items():
        parts = key.split("||")
        if len(parts) != 3: continue
        src, rel, tgt = parts
        desc = meta.get("description", tgt) if isinstance(meta, dict) else tgt
        if not any(w in f"{src} {rel} {tgt} {desc}".lower() for w in q_words): continue
        obl = meta.get("obligatoire","?") if isinstance(meta, dict) else "?"
        badge = "[OUI]" if obl=="OUI" else ("[NON]" if obl=="NON" else "")
        relevant.append({"source":src,"relation":rel,"target":tgt,
                         "description":desc,"obligatoire":obl,
                         "label":f"{src} --[{rel}]--> {desc} {badge}".strip()})
    if not relevant and kag_graph["relations"]:
        for key, meta in list(kag_graph["relations"].items())[:30]:
            parts = key.split("||")
            if len(parts) != 3: continue
            src, rel, tgt = parts
            desc = meta.get("description", tgt) if isinstance(meta, dict) else tgt
            obl = meta.get("obligatoire","?") if isinstance(meta, dict) else "?"
            relevant.append({"source":src,"relation":rel,"target":tgt,
                             "description":desc,"obligatoire":obl,
                             "label":f"{src} --[{rel}]--> {desc}"})
    return relevant[:30]

def kag_answer'''
code = code[:s5] + new5 + code[e5:]
print('P5 OK')

# P6a: upload global
code = code.replace(
    '    global rag_index, documents, uploaded_files\n\n    ALLOWED',
    '    global rag_index, documents, uploaded_files, uploaded_raw_texts\n\n    ALLOWED', 1)
print('P6a OK')

# P6b: raw text storage
old6b = '        chunks = _chunk_text(text, file.filename)\n        new_chunks.extend(chunks)\n\n        # M\u00e9ta dans la liste des fichiers upload\u00e9s'
new6b = '        chunks = _chunk_text(text, file.filename)\n        new_chunks.extend(chunks)\n        uploaded_raw_texts.append({"title": file.filename, "text": text})\n\n        # Meta dans la liste des fichiers uploades'
assert old6b in code, f'UPLOAD_META NOT FOUND: {repr(old6b[:60])}'
code = code.replace(old6b, new6b, 1)
print('P6b OK')

# P6c: clear
old6c = '    global rag_index, documents, kag_graph, uploaded_files\n    documents      = []\n    uploaded_files = []\n    rag_index      = None'
new6c = '    global rag_index, documents, kag_graph, uploaded_files, uploaded_raw_texts\n    documents          = []\n    uploaded_files     = []\n    uploaded_raw_texts = []\n    rag_index          = None'
assert old6c in code, 'CLEAR NOT FOUND'
code = code.replace(old6c, new6c, 1)
print('P6c OK')

with open(src, 'w', encoding='utf-8') as f:
    f.write(code)
print(f'DONE: {orig} -> {len(code)} bytes (+{len(code)-orig})')
