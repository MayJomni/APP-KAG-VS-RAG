import re, sys

SRC = "fast_server.py"
with open(SRC, encoding="utf-8") as f:
    code = f.read()
original_len = len(code)

# ── PATCH 2 : inserer _chunk_by_clause avant "# Signaux linguistiques" ──────
CLAUSE_FN = r"""
def _chunk_by_clause(title, text, min_words=6):
    """Decoupe le texte par section/clause pour KAG (1 chunk = 1 clause)."""
    import re as _re
    chunks = []
    lines = text.splitlines()
    current = ""
    BULLET = _re.compile(
        r"^(\d+[\.\)\-]\s|[a-zA-Z][\.\)\-]\s|[*\-\u2022\u2013]\s"
        r"|[A-Z]{2,}\s*:|Section\s+\d|Article\s+\d|Clause\s+\d"
        r"|Requirement\s+\d|REQ-\d)", _re.IGNORECASE)
    def flush(buf):
        buf = buf.strip()
        if len(buf.split()) >= min_words:
            chunks.append({"title": title, "text": buf, "id": f"{title}_cl{len(chunks)}"})
    for line in lines:
        s = line.strip()
        if not s:
            flush(current); current = ""
        elif BULLET.match(s) and current:
            flush(current); current = s
        else:
            current = (current + " " + s).strip() if current else s
    flush(current)
    if len(chunks) < 3:
        chunks = [{"title": title, "text": p.strip(), "id": f"{title}_p{i}"}
                  for i, p in enumerate(_re.split(r"\n\s*\n", text)) if len(p.split()) >= min_words]
    if len(chunks) < 2:
        w = text.split()
        chunks = [{"title": title, "text": " ".join(w[i:i+80]), "id": f"{title}_w{i}"}
                  for i in range(0, len(w), 80) if len(w[i:i+80]) >= min_words]
    return chunks

"""

OLD2 = "# Signaux linguistiques pour la classification de questions"
assert OLD2 in code, f"PATCH2 ANCHOR NOT FOUND"
code = code.replace(OLD2, CLAUSE_FN + OLD2, 1)
print("PATCH2 OK")

# ── PATCH 3 : EXTRACT_SYS ─────────────────────────────────────────────────────
OLD3_START = 'EXTRACT_SYS = (\n    "You are a knowledge graph extractor.'
OLD3_END   = '    "- Extract from ANY document type: legal, technical, academic, commercial"\n)'
assert OLD3_START in code, "PATCH3 START NOT FOUND"
assert OLD3_END   in code, "PATCH3 END NOT FOUND"
i0 = code.index(OLD3_START)
i1 = code.index(OLD3_END) + len(OLD3_END)
NEW3 = r"""EXTRACT_SYS = (
    "You are a requirements extraction engine for compliance matrices.\n"
    "For EACH distinct requirement, criterion, obligation, or specification in the text, output ONE line.\n"
    "Format: Critere|Categorie|Description|Obligatoire\n"
    "Fields:\n"
    "- Critere    : short label 2-6 words (no pipe)\n"
    "- Categorie  : Technical | Financial | Legal | Functional | Security | Administrative | Other\n"
    "- Description: exact verbatim wording max 25 words (no pipe)\n"
    "- Obligatoire: OUI (must/shall/required/mandatory) NON (should/may/preferred/optional)\n"
    "Rules:\n"
    "- Extract EVERY distinct requirement/criterion as a separate line\n"
    "- Do NOT summarize, do NOT merge multiple clauses into one line\n"
    "- Do NOT skip any requirement even if similar to a previous one\n"
    "- No JSON, no headers, no numbering, no explanation, no blank lines\n"
    "Examples:\n"
    "NDA Signature|Legal|The selected vendor must sign a Non-Disclosure Agreement|OUI\n"
    "Response Time|Technical|System shall respond within 2 seconds under normal load|OUI\n"
    "Multi-language Support|Functional|Solution should support Arabic and French|NON\n"
    "Budget Estimate|Financial|Vendor must provide detailed budgetary cost estimates|OUI\n"
    "Data Encryption|Security|All data must be encrypted at rest and in transit AES-256|OUI\n"
    "24/7 Support|Administrative|Vendor should provide round-the-clock technical support|NON"
)"""
code = code[:i0] + NEW3 + code[i1:]
print("PATCH3 OK")

# ── PATCH 4 : build_kag_from_docs entiere ─────────────────────────────────────
OLD4_START = 'def build_kag_from_docs(docs):\n    """Construit le graphe KAG depuis les documents charges.'
m_end = re.search(r'\ndef build_kag_from_precomputed', code)
assert OLD4_START in code and m_end, "PATCH4 NOT FOUND"
fn_start = code.index(OLD4_START)
fn_end   = m_end.start() + 1
NEW4 = r"""def build_kag_from_docs(docs):
    """Construit le graphe KAG. Utilise uploaded_raw_texts + clause-chunking.
    Format: Critere|Categorie|Description|Obligatoire
    """
    global kag_graph
    kag_graph = {"nodes": {}, "relations": {}}
    clause_chunks = []
    if uploaded_raw_texts:
        for src in uploaded_raw_texts:
            clause_chunks.extend(_chunk_by_clause(src["title"], src["text"]))
        print(f"[KAG] Clause-chunking: {len(uploaded_raw_texts)} fichiers -> {len(clause_chunks)} clauses")
    else:
        clause_chunks = docs
        print(f"[KAG] Fallback chunks RAG: {len(clause_chunks)}")
    if not clause_chunks:
        print("[KAG] Aucun document."); return
    n = len(clause_chunks)
    if n > 100:   batch_size, max_docs = 5, 300
    elif n > 30:  batch_size, max_docs = 3, 150
    else:         batch_size, max_docs = 1, 80
    doc_list = clause_chunks[:max_docs]
    batches  = [doc_list[i:i+batch_size] for i in range(0, len(doc_list), batch_size)]
    print(f"[KAG] {len(doc_list)} clauses, batch_size={batch_size}, {len(batches)} appels LLM")
    for b_idx, batch in enumerate(batches):
        combined = ""
        for idx, doc in enumerate(batch):
            title = doc.get("title", f"doc_{idx}")
            text  = doc.get("text", "")   # texte COMPLET - pas de [:400]
            combined += f"[CLAUSE {b_idx*batch_size+idx+1} - {title}]\n{text}\n\n"
        raw = ""
        for attempt_model in [MODEL_EXTRACT, MODEL_ANSWER]:
            result = llm(EXTRACT_SYS, combined, max_tokens=2000, model=attempt_model)
            if result and not result.startswith("ERREUR:"):
                raw = result; break
            print(f"[KAG] Batch {b_idx+1} {attempt_model} echoue: {result[:100]}")
        if not raw:
            print(f"[KAG] Batch {b_idx+1} ignore."); continue
        print(f"[KAG] Batch {b_idx+1}/{len(batches)}: {repr(raw[:200])}")
        found = 0
        for line in raw.split("\n"):
            line = line.strip().lstrip("-*>.0123456789.) ")
            if not line or "|" not in line or len(line) < 8: continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3: continue
            critere     = parts[0]
            categorie   = parts[1] if len(parts) > 1 else "Other"
            description = parts[2] if len(parts) > 2 else ""
            obligatoire = parts[3].upper() if len(parts) > 3 else "?"
            if obligatoire not in ("OUI","NON"):
                low = obligatoire.lower()
                obligatoire = "OUI" if any(w in low for w in ["oui","yes","must","shall","mandatory","required"]) else "NON"
            if any(b in critere.lower() for b in ["critere","format","entity","example","note:","header","field","subject"]): continue
            if not critere or len(critere) < 3: continue
            doc_title = batch[0].get("title","")
            kag_graph["nodes"][critere]   = {"doc": doc_title, "categorie": categorie, "obligatoire": obligatoire}
            kag_graph["nodes"][categorie] = {"doc": doc_title, "type": "category"}
            key = f"{critere}||{categorie}||{description[:120]}"
            kag_graph["relations"][key] = {"obligatoire": obligatoire, "description": description, "doc": doc_title}
            found += 1
        print(f"[KAG] Batch {b_idx+1} -> {found} exigences")
    print(f"[KAG] FINAL: {len(kag_graph['nodes'])} noeuds, {len(kag_graph['relations'])} relations")

"""
code = code[:fn_start] + NEW4 + code[fn_end:]
print("PATCH4 OK")

# ── PATCH 5 : kag_search ──────────────────────────────────────────────────────
OLD5_START = 'def kag_search(question):\n    """Cherche les triplets pertinents.'
OLD5_END   = '    return relevant[:20]\n\ndef kag_answer'
assert OLD5_START in code and OLD5_END in code, "PATCH5 NOT FOUND"
s5 = code.index(OLD5_START); e5 = code.index(OLD5_END) + len(OLD5_END)
NEW5 = r"""def kag_search(question):
    """Cherche les exigences. Compatible ancien format (True) et nouveau (dict)."""
    q_words = [w for w in question.lower().split() if len(w) > 3]
    relevant = []
    for key, meta in kag_graph["relations"].items():
        parts = key.split("||")
        if len(parts) != 3: continue
        src, rel, tgt = parts
        description = meta.get("description", tgt) if isinstance(meta, dict) else tgt
        if not any(w in f"{src} {rel} {tgt} {description}".lower() for w in q_words): continue
        obligatoire = meta.get("obligatoire", "?") if isinstance(meta, dict) else "?"
        badge = "[OUI]" if obligatoire == "OUI" else ("[NON]" if obligatoire == "NON" else "")
        label = f"{src} --[{rel}]--> {description} {badge}".strip()
        relevant.append({"source": src, "relation": rel, "target": tgt,
                         "description": description, "obligatoire": obligatoire, "label": label})
    if not relevant and kag_graph["relations"]:
        for key, meta in list(kag_graph["relations"].items())[:30]:
            parts = key.split("||")
            if len(parts) != 3: continue
            src, rel, tgt = parts
            description = meta.get("description", tgt) if isinstance(meta, dict) else tgt
            obligatoire = meta.get("obligatoire", "?") if isinstance(meta, dict) else "?"
            label = f"{src} --[{rel}]--> {description}"
            relevant.append({"source": src, "relation": rel, "target": tgt,
                             "description": description, "obligatoire": obligatoire, "label": label})
    return relevant[:30]

def kag_answer"""
code = code[:s5] + NEW5 + code[e5:]
print("PATCH5 OK")

# ── PATCH 6a : upload global ──────────────────────────────────────────────────
OLD6a = '    global rag_index, documents, uploaded_files\n\n    ALLOWED'
code = code.replace(OLD6a, '    global rag_index, documents, uploaded_files, uploaded_raw_texts\n\n    ALLOWED', 1)
print("PATCH6a OK")

# ── PATCH 6b : stocker raw text ───────────────────────────────────────────────
OLD6b = '        chunks = _chunk_text(text, file.filename)\n        new_chunks.extend(chunks)\n\n        # Meta dans la liste des fichiers uploades'
NEW6b = '        chunks = _chunk_text(text, file.filename)\n        new_chunks.extend(chunks)\n        uploaded_raw_texts.append({"title": file.filename, "text": text})\n\n        # Meta dans la liste des fichiers uploades'
assert OLD6b in code, "PATCH6b NOT FOUND"
code = code.replace(OLD6b, NEW6b, 1)
print("PATCH6b OK")

# ── PATCH 6c : clear ──────────────────────────────────────────────────────────
OLD6c = '    global rag_index, documents, kag_graph, uploaded_files\n    documents      = []\n    uploaded_files = []\n    rag_index      = None'
NEW6c = '    global rag_index, documents, kag_graph, uploaded_files, uploaded_raw_texts\n    documents          = []\n    uploaded_files     = []\n    uploaded_raw_texts = []\n    rag_index          = None'
assert OLD6c in code, "PATCH6c NOT FOUND"
code = code.replace(OLD6c, NEW6c, 1)
print("PATCH6c OK")

with open(SRC, "w", encoding="utf-8") as f:
    f.write(code)
print(f"DONE: {original_len} -> {len(code)} bytes (+{len(code)-original_len})")
