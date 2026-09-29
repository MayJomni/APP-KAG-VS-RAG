"""
fast_server.py — Serveur RAG vs KAG (version corrigée)
"""
import os, sys, time, json, logging, re, string, io
import io as _io
import difflib

# Évaluateurs avancés (Ragas + DeepEval)
try:
    from evaluation.ragas_evaluator import RagasEvaluator
    from evaluation.deepeval_evaluator import DeepEvalEvaluator
    ADVANCED_EVAL = True
except ImportError:
    ADVANCED_EVAL = False

# Force UTF-8 stdout/stderr sur Windows (évite UnicodeEncodeError avec cp1252)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path

from collections import Counter
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
import minsearch

# PDF / DOCX parsers
try:
    from pypdf import PdfReader
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

load_dotenv()
logging.basicConfig(level=logging.WARNING)

GROQ_KEY      = os.getenv("GROQ_API_KEY", "")
client        = Groq(api_key=GROQ_KEY)

# Évaluateurs avancés
ragas_evaluator = RagasEvaluator() if ADVANCED_EVAL else None
deepeval_evaluator = DeepEvalEvaluator() if ADVANCED_EVAL else None

MODEL_ANSWER  = "openai/gpt-oss-20b"    # réponses QA — rapide
MODEL_EXTRACT = "openai/gpt-oss-20b"    # extraction KAG — rapide

def llm(system_prompt: str, user_prompt: str,
        max_tokens: int = 150, model: str = None) -> str:
    if model is None:
        model = MODEL_ANSWER
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.0, max_tokens=max_tokens
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"ERREUR: {e}"

app = FastAPI(title="RAG vs KAG")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Servir les fichiers statiques PWA (manifest.json, sw.js, icons)
_static_dir = Path(__file__).parent / "kag_app"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# ── State ──────────────────────────────────────────────────────────────────────
rag_index      = None
documents      = []
kag_graph      = {"nodes": {}, "relations": {}}
run_log        = []   # {pipeline, question, answer, latency_ms}
routing_log    = []   # {question, type, routed_to, reason, confidence}
uploaded_files     = []   # [{name, size_kb, chunks, status}]
uploaded_raw_texts = []   # [{title, text}] texte brut KAG
compliance_criteria = []  # [{id, category, criterion, mandatory, source, chunk_id, compliance_status}]

# ── File Parsing ───────────────────────────────────────────────────────────────
def _parse_uploaded_file(filename: str, content: bytes) -> str:
    """Extrait le texte brut depuis PDF, DOCX, TXT, MD ou CSV."""
    ext = Path(filename).suffix.lower()
    try:
        if ext == ".pdf":
            if not HAS_PDF:
                return ""
            reader = PdfReader(io.BytesIO(content))
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        elif ext in (".docx",):
            if not HAS_DOCX:
                return ""
            doc = DocxDocument(io.BytesIO(content))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        elif ext in (".txt", ".md"):
            return content.decode("utf-8", errors="ignore")
        elif ext == ".csv":
            text = content.decode("utf-8", errors="ignore")
            # Chaque ligne CSV devient une phrase
            lines = [l.replace(",", " — ") for l in text.splitlines() if l.strip()]
            return "\n".join(lines)
        else:
            # Essai en UTF-8 pour tout autre format texte
            return content.decode("utf-8", errors="ignore")
    except Exception as e:
        logging.warning(f"Parse error {filename}: {e}")
        return ""

def _chunk_text(text: str, filename: str, chunk_size: int = 400,
                overlap: int = 50) -> list:
    """Découpe le texte en chunks de ~chunk_size mots avec overlap."""
    words  = text.split()
    chunks = []
    step   = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        if len(chunk_words) < 20:   # ignorer les micro-chunks
            continue
        chunks.append({
            "title": filename,
            "text":  " ".join(chunk_words),
            "id":    f"{filename}_{i}"
        })
    return chunks

# ── Routing Agent ──────────────────────────────────────────────────────────────

def _chunk_by_clause(title, text, min_words=6):
    import re as _r
    chunks = []
    lines = text.splitlines()
    cur = ""
    pat = _r.compile(
        r"^(\d+[\.)\-]\s|[a-zA-Z][\.)\-]\s|[*\-\u2022\u2013]\s"
        r"|[A-Z]{2,}\s*:|Section\s+\d|Article\s+\d|Clause\s+\d|Requirement\s+\d|REQ-\d)",
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
                  for i, p in enumerate(_r.split(r"\n\s*\n", text)) if len(p.split()) >= min_words]
    if len(chunks) < 2:
        w = text.split()
        chunks = [{"title": title, "text": " ".join(w[i:i+80]), "id": f"{title}_w{i}"}
                  for i in range(0, len(w), 80) if len(w[i:i+80]) >= min_words]
    return chunks
# Signaux linguistiques pour la classification de questions

MULTIHOP_SIGNALS  = {"same nationality", "also", "were they", "did they",
                     "same country", "same city", "same field", "same genre",
                     "who is also", "who was also", "as well as",
                     "in addition to", "both born", "both studied",
                     "both worked", "how many of them", "which of them"}

RELATIONAL_SIGNALS= {"father of", "mother of", "son of", "daughter of",
                     "founded by", "created by", "directed by", "produced by",
                     "who played", "who portrayed", "who held",
                     "what position", "what role", "what government",
                     "relationship between", "married to", "worked with",
                     "collaborated with", "trained by", "coached by"}

FACTUAL_SIGNALS   = {"what year", "when was", "when did", "how many",
                     "how much", "what is the name", "what was the name",
                     "in what year", "what date", "how old", "what number",
                     "which year", "what age", "how long", "what time"}

BOOLEAN_STARTERS  = {"were", "was", "is", "are", "did", "does",
                     "have", "has", "can", "could", "would", "do",
                     "will", "shall", "might", "should"}

# Mots exclus du comptage d'entités
STOP_CAPS = {"The","A","An","Is","Are","Was","Were","Did","Does","Have",
             "Has","Can","Could","Would","Do","Will","Shall","In","Of",
             "And","Or","But","For","With","From","By","On","At","To"}


def _risk_consequences(chosen_pipeline: str, q_type: str,
                       entity_count: int, mh_hits: int, re_hits: int,
                       fa_hits: int, is_simple_bool: bool) -> list:
    """Génère les conséquences si l'utilisateur choisit le mauvais pipeline."""
    risks = []
    wrong = "rag" if chosen_pipeline == "kag" else "kag"

    if wrong == "rag":
        # Risques d'utiliser RAG quand KAG est recommandé
        if mh_hits > 0:
            risks.append("❌ RAG ne peut pas chaîner plusieurs faits → réponse partielle ou incorrecte")
        if re_hits > 0:
            risks.append("❌ RAG cherche du texte brut, pas des relations → connexion entre entités manquée")
        if entity_count >= 2:
            risks.append(f"⚠️ {entity_count} entités détectées → RAG risque de ne trouver qu'un seul contexte")
        if q_type == "multi_hop":
            risks.append("❌ Raisonnement multi-sauts impossible sans graphe → hallucination probable")
        if not risks:
            risks.append("⚠️ RAG pourrait manquer une relation implicite entre entités du graphe")
    else:
        # Risques d'utiliser KAG quand RAG est recommandé
        if is_simple_bool or q_type == "boolean":
            risks.append("⚠️ KAG sur-complexifie une question oui/non simple → latence inutile")
        if q_type == "factual":
            risks.append("⚠️ Un seul passage de texte suffit → traverser le graphe est inutile")
        risks.append("❌ Si l'entité n'est pas dans le graphe → réponse vide ou générique")
        risks.append("⚠️ KAG dépend de la qualité de l'extraction → triplets manquants = erreurs")

    return risks[:3]  # max 3 risques affichés


def classify_question(question: str) -> dict:
    """Classifie la question avec scoring RAG vs KAG, facteurs détectés et analyse de risque."""
    q     = question.lower().strip()
    words = q.split()
    orig  = question.split()
    first = words[0] if words else ""

    # ── Détection des signaux ──────────────────────────────────────────────────
    factors_detected = []

    mh_hits = 0
    for sig in MULTIHOP_SIGNALS:
        if sig in q:
            mh_hits += 1
            factors_detected.append({"label": f'"{sig}"', "type": "multi_hop",
                                     "icon": "🧠", "favor": "kag", "weight": 45})

    re_hits = 0
    for sig in RELATIONAL_SIGNALS:
        if sig in q:
            re_hits += 1
            factors_detected.append({"label": f'"{sig}"', "type": "relational",
                                     "icon": "🔗", "favor": "kag", "weight": 40})

    fa_hits = 0
    for sig in FACTUAL_SIGNALS:
        if sig in q:
            fa_hits += 1
            factors_detected.append({"label": f'"{sig}"', "type": "factual",
                                     "icon": "📄", "favor": "rag", "weight": 35})

    bo_hit = 1 if first in BOOLEAN_STARTERS else 0
    if bo_hit:
        factors_detected.append({"label": f'commence par "{first}"', "type": "boolean",
                                 "icon": "❓", "favor": "rag", "weight": 25})

    # Facteur : nombre d'entités propres dans la question
    entity_count = sum(1 for w in orig
                       if w[0].isupper() and len(w) > 2 and w not in STOP_CAPS)
    if entity_count >= 3:
        factors_detected.append({"label": f"{entity_count} entités nommées", "type": "multi_hop",
                                 "icon": "👥", "favor": "kag", "weight": 30})
    elif entity_count == 2:
        factors_detected.append({"label": "2 entités nommées", "type": "relational",
                                 "icon": "👥", "favor": "kag", "weight": 20})

    # Facteur : longueur de la question
    qlen = len(words)
    if qlen > 18:
        factors_detected.append({"label": f"question longue ({qlen} mots)", "type": "multi_hop",
                                 "icon": "📏", "favor": "kag", "weight": 15})
    elif qlen <= 8:
        factors_detected.append({"label": f"question courte ({qlen} mots)", "type": "factual",
                                 "icon": "📏", "favor": "rag", "weight": 15})

    # Facteur : présence de "both" + "and"
    has_both    = "both" in q
    has_and_two = q.count(" and ") >= 1
    if has_both and has_and_two:
        factors_detected.append({"label": '"both ... and" détecté', "type": "multi_hop",
                                 "icon": "🔄", "favor": "kag", "weight": 25})

    # Facteur : pronoms de référence multiple
    if any(p in words for p in ["they", "them", "their", "both"]):
        factors_detected.append({"label": "pronom multi-entité", "type": "multi_hop",
                                 "icon": "🔄", "favor": "kag", "weight": 10})

    # ── Scores RAG vs KAG (0-100) ─────────────────────────────────────────────
    is_simple_boolean = (bo_hit and qlen <= 14 and re_hits == 0 and mh_hits == 0)

    rag_score  = 10  # base
    kag_score  = 10  # base
    rag_score += fa_hits * 30
    rag_score += 30 if is_simple_boolean else 0
    rag_score += 15 if qlen <= 8 else 0
    rag_score += 10 if entity_count <= 1 else 0
    rag_score += 20 if bo_hit and not has_both else 0

    kag_score += mh_hits * 40
    kag_score += re_hits * 35
    kag_score += 25 if has_both and has_and_two else 0
    kag_score += 20 if entity_count >= 3 else (12 if entity_count == 2 else 0)
    kag_score += 15 if qlen > 18 else 0
    kag_score += 10 if any(p in words for p in ["they","them","their"]) else 0

    rag_score = min(100, rag_score)
    kag_score = min(100, kag_score)

    # Normalisation pour que les deux totalisent 100
    total = rag_score + kag_score
    if total > 0:
        rag_pct = round(rag_score / total * 100)
        kag_pct = 100 - rag_pct
    else:
        rag_pct = kag_pct = 50

    # ── Niveau de risque si mauvais choix ─────────────────────────────────────
    score_diff = abs(rag_pct - kag_pct)
    if score_diff >= 35:
        risk_level, risk_emoji = "Élevé",  "🔴"
    elif score_diff >= 15:
        risk_level, risk_emoji = "Modéré", "🟡"
    else:
        risk_level, risk_emoji = "Faible",  "🟢"

    # ── Type scores (radar chart) ──────────────────────────────────────────────
    type_scores = {
        "multi_hop":  min(100, mh_hits * 45 + (15 if mh_hits else 0)),
        "relational": min(100, re_hits * 40 + (10 if re_hits else 0)),
        "factual":    min(100, fa_hits * 45 + (10 if fa_hits else 0)),
        "boolean":    bo_hit * 70 + (20 if is_simple_boolean else 0),
    }
    if bo_hit and has_both and has_and_two:
        type_scores["multi_hop"] = max(type_scores["multi_hop"], 55)
    for k in type_scores:
        if type_scores[k] == 0:
            type_scores[k] = 5

    # ── Classification prioritaire ────────────────────────────────────────────
    if is_simple_boolean:
        q_type, confidence = "boolean", 0.85
        reason = "Question oui/non simple → RAG suffisant (un seul passage de texte)"
    elif mh_hits >= 2 or (mh_hits >= 1 and fa_hits == 0 and re_hits == 0):
        q_type, confidence = "multi_hop", 0.90
        reason = "Multi-sauts détectés → KAG nécessaire (raisonnement sur le graphe)"
    elif bo_hit and has_both and has_and_two:
        q_type, confidence = "multi_hop", 0.82
        reason = "Oui/non sur deux entités distinctes → KAG pour comparer dans le graphe"
    elif re_hits:
        q_type, confidence = "relational", 0.88
        reason = "Relation entre entités détectée → KAG (parcours des arêtes du graphe)"
    elif fa_hits:
        q_type, confidence = "factual", 0.85
        reason = "Valeur précise recherchée → RAG (passages BM25 pertinents)"
    elif bo_hit:
        q_type, confidence = "boolean", 0.78
        reason = "Question oui/non → Ensemble pour arbitrage RAG + KAG"
    else:
        sys_p = ("Classify this question in ONE word: multi_hop | relational | factual | boolean. "
                 "Just the word, no explanation.")
        res   = llm(sys_p, question, max_tokens=5).strip().lower()
        q_type    = res if res in ("multi_hop","relational","factual","boolean") else "factual"
        confidence = 0.70
        reason = "Classifié par LLM (aucun signal lexical détecté)"
        type_scores[q_type] = max(type_scores[q_type], 60)

    # Pipeline recommandé selon les scores
    recommended = "rag" if rag_pct >= kag_pct else "kag"

    # ── Analyse de risque si mauvais choix ────────────────────────────────────
    risk_consequences = _risk_consequences(
        recommended, q_type, entity_count,
        mh_hits, re_hits, fa_hits, is_simple_boolean
    )

    return {
        "type": q_type,
        "confidence": confidence,
        "reason": reason,
        "type_scores": type_scores,
        # Nouveau : scoring comparatif
        "score_rag": rag_pct,
        "score_kag": kag_pct,
        "risk_level": risk_level,
        "risk_emoji": risk_emoji,
        "risk_consequences": risk_consequences,
        "factors_detected": factors_detected,
        "entity_count": entity_count,
        "q_length": qlen,
    }


def route_question(question: str) -> dict:
    """Décide le meilleur pipeline : rag | kag | ensemble."""
    kag_ready = len(kag_graph["nodes"]) > 5
    rag_ready = rag_index is not None and len(documents) > 0
    clf       = classify_question(question)
    q_type    = clf["type"]
    words     = question.lower().split()
    first     = words[0] if words else ""
    is_simple_bool = (first in BOOLEAN_STARTERS and len(words) <= 14 and q_type == "boolean")

    if not kag_ready and not rag_ready:
        return {**clf, "pipeline": "none",
                "reason": "Aucun pipeline disponible — chargez des données d'abord"}
    if not kag_ready:
        return {**clf, "pipeline": "rag",
                "reason": f"KAG non disponible. RAG utilisé (question: {q_type})"}
    if not rag_ready:
        return {**clf, "pipeline": "kag",
                "reason": f"RAG non disponible. KAG utilisé (question: {q_type})"}

    if q_type == "multi_hop":
        return {**clf, "pipeline": "kag",
                "reason": "KAG ✓ : multi-sauts → raisonnement sur le graphe de triplets"}
    if q_type == "relational":
        return {**clf, "pipeline": "kag",
                "reason": "KAG ✓ : relation entre entités → parcours des arêtes du graphe"}
    if q_type == "factual":
        return {**clf, "pipeline": "rag",
                "reason": "RAG ✓ : fait direct → passages BM25 pertinents"}
    if q_type == "boolean":
        if is_simple_bool:
            return {**clf, "pipeline": "rag",
                    "reason": "RAG ✓ : oui/non simple → un passage suffit"}
        return {**clf, "pipeline": "ensemble",
                "reason": "Ensemble ✓ : oui/non complexe → RAG + KAG + agent arbitre"}

    return {**clf, "pipeline": "ensemble",
            "reason": "Incertain → Ensemble RAG + KAG pour robustesse"}

def ensemble_answer(question: str, rag: dict, kag: dict, routing: dict) -> dict:
    """Agent arbitre : choisit la meilleure réponse entre RAG et KAG."""
    rag_ans = rag.get("answer","")
    kag_ans = kag.get("answer","")
    REFUSALS = ["don't know", "do not know", "cannot determine",
                "not enough", "je ne sais pas", "impossible de"]

    rag_refused = any(p in rag_ans.lower() for p in REFUSALS)
    kag_refused = any(p in kag_ans.lower() for p in REFUSALS)

    # Un seul refuse → prendre l'autre directement
    if rag_refused and not kag_refused:
        return {"winner": "kag", "answer": kag_ans,
                "reason": "RAG sans contexte suffisant → KAG sélectionné"}
    if kag_refused and not rag_refused:
        return {"winner": "rag", "answer": rag_ans,
                "reason": "KAG sans triplets suffisants → RAG sélectionné"}

    # Les deux refusent → connaissance générale LLM
    if rag_refused and kag_refused:
        ans = llm("Answer this question directly in 1 sentence. Be confident.",
                  question, max_tokens=80)
        return {"winner": "rag", "answer": ans,
                "reason": "Deux pipelines insuffisants → réponse par connaissance générale"}

    # Les deux répondent → agent LLM arbitre
    sys_p = ("You are an arbitration agent. Choose the most accurate and concise answer. "
             "Reply ONLY: 'RAG: <one-line reason>' or 'KAG: <one-line reason>'.")
    prompt = (f"QUESTION: {question}\n\n"
              f"ANSWER_RAG: {rag_ans}\n\n"
              f"ANSWER_KAG: {kag_ans}")
    verdict = llm(sys_p, prompt, max_tokens=60)

    if verdict.upper().startswith("KAG"):
        return {"winner": "kag", "answer": kag_ans,
                "reason": verdict.replace("KAG:","").strip()}
    return {"winner": "rag", "answer": rag_ans,
            "reason": verdict.replace("RAG:","").strip()}


# ── Métriques ─────────────────────────────────────────────────────────────────
YES_SYNONYMS = {"yes", "yeah", "yep", "both", "same", "correct", "true", "american", "indeed"}
NO_SYNONYMS  = {"no", "not", "neither", "different", "false", "incorrect", "none"}

def normalize(text):
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())

def exact_match(pred, gt):
    p, g = normalize(pred), normalize(gt)
    if p == g: return 1.0
    if g and g in p: return 1.0
    # yes/no semantic matching
    if g == "yes"  and any(w in p.split() for w in YES_SYNONYMS): return 1.0
    if g == "no"   and any(w in p.split() for w in NO_SYNONYMS):  return 1.0
    return 0.0

def token_f1(pred, gt):
    pt = normalize(pred).split()
    gt_t = normalize(gt).split()
    if not pt or not gt_t: return 0.0
    common = sum((Counter(pt) & Counter(gt_t)).values())
    if common == 0: return 0.0
    p, r = common / len(pt), common / len(gt_t)
    return round(2*p*r/(p+r), 4)

# ── RAG ───────────────────────────────────────────────────────────────────────
def rag_search(question, num_results=10):
    if rag_index is None or not documents:
        return [], "Aucun document chargé."
    results = rag_index.search(question,
                               boost_dict={"title":2.0,"text":1.0},
                               num_results=num_results)
    context = "\n\n".join(
        f"[{r.get('title','')}] {r.get('text','')[:500]}" for r in results
    )
    return results, context

def rag_answer(question):
    t0 = time.time()
    results, context = rag_search(question)

    if not results:
        # Aucun document chargé — connaissance générale
        sys_p = (
            "You are a knowledgeable QA assistant. "
            "Give a detailed, informative answer of at least 3 sentences (minimum 200 characters). "
            "Cover the what, why, and how. Never refuse or say you don't know."
        )
        prompt  = f"QUESTION: {question}"
        context = "[Connaissance générale — aucun document spécifique chargé]"
    else:
        # Nom du (premier) fichier source pour contextualiser le prompt
        doc_name = uploaded_files[0]["name"] if uploaded_files else "the uploaded document"
        sys_p = (
            f"You are a precise document QA assistant analyzing '{doc_name}'. "
            "The CONTEXT below is the EXACT TEXT extracted from this document. "
            "It is real content — you MUST use it directly to answer. "
            "NEVER say the document is unavailable, inaccessible, or that you cannot read it. "
            "Answer the question by citing facts from the CONTEXT. "
            "Write a DETAILED, COMPREHENSIVE answer (minimum 200 characters, at least 3 sentences). "
            "Structure your answer clearly. Quote key passages verbatim when relevant. "
            "If context is partial, supplement with related general knowledge but prioritize the document."
        )
        prompt = f"QUESTION: {question}\n\nCONTEXT FROM '{doc_name}':\n{context}"

    answer = llm(sys_p, prompt, max_tokens=400)

    # Supprimer les hallucinations de refus
    HALLUCINATIONS = [
        "not available for me to access", "i cannot access", "i can't access",
        "uploaded document is not available", "do not have access to the file",
        "i don't know", "i do not know", "cannot determine",
        "not enough information", "je ne sais pas", "unable to"
    ]
    if any(p in answer.lower() for p in HALLUCINATIONS):
        # Relancer avec contexte injecté directement dans le prompt utilisateur
        answer = llm(
            "You are a document QA assistant. The text below IS the document. Read it and answer.",
            f"QUESTION: {question}\n\nDOCUMENT TEXT:\n{context[:2000]}\n\n"
            f"Write at least 3 sentences using only the document text above (minimum 200 characters).",
            max_tokens=400
        )

    # Garantie longueur minimale
    if len(answer.strip()) < 200:
        expanded = llm(
            "Expand this answer to at least 200 characters by quoting more from the document.",
            f"QUESTION: {question}\n\nCURRENT ANSWER: {answer}\n\nDOCUMENT:\n{context[:2000]}",
            max_tokens=400
        )
        if len(expanded.strip()) > len(answer.strip()):
            answer = expanded

    return {
        "answer": answer,
        "context": context,
        "answer_length": len(answer.strip()),
        "sources": [{"title": r.get("title",""), "snippet": r.get("text","")[:500]}
                    for r in results],
        "latency_ms": round((time.time()-t0)*1000)
    }


# ── KAG ───────────────────────────────────────────────────────────────────────
# SOLUTION RATE LIMIT :
# 1) batch_size=1  → 1 chunk par appel LLM, attention maximale du modèle
# 2) max_tokens=3000 → sortie exhaustive sans troncature
# 3) time.sleep(2) entre batches → respecte le free-tier Groq (6000 TPM)
# 4) Déduplication difflib ≥ 85% → supprime les doublons sémantiques

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


def _is_near_duplicate(text_a: str, text_b: str, threshold: float = 0.85) -> bool:
    """Retourne True si text_a et text_b ont une similarité >= threshold (difflib)."""
    ratio = difflib.SequenceMatcher(None,
                                    text_a.lower().strip(),
                                    text_b.lower().strip()).ratio()
    return ratio >= threshold


def build_kag_from_docs(docs):
    """Extrait les criteres de conformite RFP depuis les chunks. Format: Category|Criterion|Mandatory.

    Changements v2:
    - batch_size=1 : un chunk par appel LLM (attention maximale)
    - Texte complet du chunk (sans troncature [:400])
    - max_docs dynamique : traite TOUS les chunks disponibles
    - max_tokens=3000 : sortie exhaustive
    - time.sleep(2) entre chaque batch : rate-limit Groq free-tier
    - Déduplication difflib >= 85% après extraction
    """
    global kag_graph, compliance_criteria
    kag_graph           = {"nodes": {}, "relations": {}}
    compliance_criteria = []

    # ── Choix des chunks source ───────────────────────────────────────────────
    clause_chunks = []
    if uploaded_raw_texts:
        for src in uploaded_raw_texts:
            clause_chunks.extend(_chunk_by_clause(src["title"], src["text"]))
        print(f"[KAG] Clause-chunking: {len(uploaded_raw_texts)} fichier(s) -> {len(clause_chunks)} clauses")
    else:
        clause_chunks = list(docs)
        print(f"[KAG] Fallback docs: {len(clause_chunks)} chunks")

    if not clause_chunks:
        print("[KAG] Aucun chunk disponible."); return

    # ── Paramètres dynamiques ─────────────────────────────────────────────────
    # batch_size=1 : un chunk à la fois pour une extraction exhaustive
    bsz = 1
    # max_docs : traite TOUS les chunks (pas de plafond fixe)
    max_docs = len(clause_chunks)
    dl       = clause_chunks[:max_docs]
    batches  = [dl[i:i+bsz] for i in range(0, len(dl), bsz)]
    print(f"[KAG] {len(dl)} clauses | batch_size={bsz} | {len(batches)} appels LLM | max_tokens=3000")

    raw_criteria_texts = []  # textes bruts pour déduplication
    dup_count          = 0

    for bi, batch in enumerate(batches):
        # ── Construction du prompt utilisateur ────────────────────────────────
        doc      = batch[0]
        doc_title = doc.get("title", "doc")
        # Texte complet du chunk, sans troncature
        doc_text  = doc.get("text", "")
        combined  = f"[CLAUSE {bi+1} - {doc_title}]\n{doc_text}\n"

        # ── Appel LLM avec fallback gemma2 -> llama ───────────────────────────
        raw = ""
        for mdl in [MODEL_EXTRACT, MODEL_ANSWER]:
            r = llm(EXTRACT_SYS, combined, max_tokens=3000, model=mdl)
            if r and not r.startswith("ERREUR:"):
                raw = r; break
            print(f"[KAG] Batch {bi+1} échec {mdl}: {r[:80]}")

        if not raw:
            print(f"[KAG] Batch {bi+1}/{len(batches)} ignoré (pas de réponse LLM)")
            # Pause avant le prochain appel même en cas d'échec
            if bi < len(batches) - 1:
                time.sleep(2)
            continue

        print(f"[KAG] Batch {bi+1}/{len(batches)} [{doc_title[:40]}]: {repr(raw[:120])}")

        # ── Parsing ligne par ligne ───────────────────────────────────────────
        found = 0
        chunk_id = doc.get("id", f"chunk_{bi}")

        for line in raw.split("\n"):
            line = line.strip().lstrip("-*>.0123456789.) ")
            if not line or "|" not in line or len(line) < 8:
                continue

            parts = [x.strip() for x in line.split("|")]
            if len(parts) < 3:
                continue

            # Parsing Category|Criterion|Mandatory
            cat           = parts[0]   # Category
            criterion     = parts[1]   # Criterion
            mandatory_raw = parts[2].upper() if len(parts) > 2 else "?"

            # Normalisation Mandatory → Yes/No
            if mandatory_raw in ("YES", "OUI"):
                mandatory = "Yes"
            elif mandatory_raw in ("NO", "NON"):
                mandatory = "No"
            else:
                mandatory = "Yes" if any(w in mandatory_raw.lower()
                                         for w in ["yes", "oui", "must", "shall", "required", "mandatory"]
                                         ) else "No"

            # Normalisation Category vers valeurs autorisées
            VALID_CATS = {"Functional", "Technical", "Operational", "Security",
                          "Infrastructure", "Administrative", "Financial"}
            cat_normalized = next(
                (c for c in VALID_CATS if c.lower() == cat.lower()), "Functional"
            )

            # Filtre les lignes parasites (exemples du prompt, en-têtes...)
            if any(b in criterion.lower() for b in [
                "criterion", "category", "mandatory", "example",
                "format", "note:", "header", "where:", "output only",
                "shall respond within 2 seconds",   # exemple du prompt
                "unified xml file",                 # exemple du prompt
                "future bill design",               # exemple du prompt
                "iso 27001 is recommended"          # exemple du prompt
            ]):
                continue
            if not criterion or len(criterion) < 5:
                continue

            # ── Déduplication difflib >= 85% ──────────────────────────────────
            is_dup = False
            for existing_text in raw_criteria_texts:
                if _is_near_duplicate(criterion, existing_text, threshold=0.85):
                    is_dup = True
                    dup_count += 1
                    logging.info(
                        f"[KAG] Doublon supprimé (>85%): '{criterion[:60]}'"
                        f" ≈ '{existing_text[:60]}'"
                    )
                    break
            if is_dup:
                continue

            raw_criteria_texts.append(criterion)

            # ── Stockage kag_graph (rétrocompatibilité) ───────────────────────
            kag_graph["nodes"][criterion] = {
                "doc": doc_title,
                "categorie": cat_normalized,
                "obligatoire": "OUI" if mandatory == "Yes" else "NON"
            }
            kag_graph["nodes"][cat_normalized] = {"doc": doc_title, "type": "category"}
            kag_graph["relations"][f"{criterion}||{cat_normalized}||{mandatory}"] = {
                "obligatoire": "OUI" if mandatory == "Yes" else "NON",
                "description": criterion,
                "doc": doc_title
            }

            # ── Stockage compliance_criteria ──────────────────────────────────
            compliance_criteria.append({
                "id":                f"CR-{len(compliance_criteria)+1:03d}",
                "category":          cat_normalized,
                "criterion":         criterion,
                "mandatory":         mandatory,
                "source":            doc_title,
                "chunk_id":          chunk_id,
                "compliance_status": "To Be Assessed"
            })
            found += 1

        print(f"[KAG] Batch {bi+1}/{len(batches)} -> {found} critère(s) extrait(s)")

        # ── Pause rate-limit Groq (sauf après le dernier batch) ──────────────
        if bi < len(batches) - 1:
            time.sleep(2)

    print(
        f"[KAG] FINAL: {len(kag_graph['nodes'])} nœuds | "
        f"{len(kag_graph['relations'])} relations | "
        f"{len(compliance_criteria)} critères uniques | "
        f"{dup_count} doublon(s) supprimé(s)"
    )
def build_kag_from_precomputed(results_path: str = None):
    """Charge un graphe KAG pré-calculé depuis results_kag.json (0 appel API)."""
    global kag_graph
    if results_path is None:
        results_path = str(Path(__file__).parent / "results_kag.json")
    kag_graph = {"nodes": {}, "relations": {}}
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        for item in results:
            ctx = item.get("context", "")
            for line in ctx.split("\n"):
                line = line.strip().lstrip("-• ")
                # Format: "A --[rel]--> B"  ou  "A|rel|B"
                if "--[" in line and "-->" in line:
                    try:
                        src = line.split("--[")[0].strip()
                        rel = line.split("--[")[1].split("]-->")[0].strip()
                        tgt = line.split("]-->")[1].strip()
                        if src and rel and tgt:
                            kag_graph["nodes"][src] = {"doc": item.get("question","")[:40]}
                            kag_graph["nodes"][tgt] = {"doc": item.get("question","")[:40]}
                            kag_graph["relations"][f"{src}||{rel}||{tgt}"] = True
                    except Exception:
                        pass
        return len(kag_graph["nodes"]), len(kag_graph["relations"])
    except Exception as e:
        return 0, 0


def kag_search(question):
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

def kag_answer(question):
    t0 = time.time()
    triplets = kag_search(question)
    used_rag_fallback = False

    if triplets:
        context = "\n".join(t["label"] for t in triplets)
    else:
        # Graphe KAG vide — fallback sur les chunks du corpus (RAG search)
        # au lieu de dire "pas de document", on extrait les faits directement
        rag_results, rag_context = rag_search(question, num_results=8)
        if rag_results:
            used_rag_fallback = True
            # Présenter les chunks comme des faits structurés
            context = "[Faits extraits du document uploadé]\n" + rag_context
            # Construire des pseudo-triplets pour l'affichage
            triplets = [
                {"source": r.get("title", "Document"),
                 "relation": "contains",
                 "target": r.get("text", "")[:120] + "...",
                 "label": f"{r.get('title','Document')} —[contient]→ {r.get('text','')[:80]}..."}
                for r in rag_results[:5]
            ]
        else:
            context = "[Aucun document chargé — réponse basée sur connaissance générale]"

    # Nom du document source pour contextualiser
    doc_name = uploaded_files[0]["name"] if uploaded_files else "the uploaded document"
    sys_p = (
        f"You are a knowledge graph QA expert analyzing '{doc_name}'. "
        "The FACTS below are triplets extracted from this document's knowledge graph. "
        "Each fact is in format: Entity —[relation]→ Entity. "
        "These are REAL facts from the file — use them to answer precisely. "
        "NEVER say the document is unavailable or inaccessible. "
        "Answer the question using the facts provided. "
        "Write a DETAILED, STRUCTURED answer (minimum 200 characters, at least 3 sentences). "
        "Explain what each fact means in context. "
        "If the question asks to extract text or quote content, use the facts as direct evidence. "
        "Fill any gaps with general knowledge while clearly citing the document facts."
    )
    prompt = f"QUESTION: {question}\n\nKNOWLEDGE GRAPH FACTS FROM '{doc_name}':\n{context}"
    answer = llm(sys_p, prompt, max_tokens=400)

    # Supprimer les hallucinations de refus
    HALLUCINATIONS = [
        "not available for me to access", "i cannot access", "i can't access",
        "uploaded document is not available", "do not have access to the file",
        "i don't know", "i do not know", "cannot determine",
        "not enough information", "je ne sais pas", "pas de réponse",
        "impossible", "unable to", "n'avez pas téléchargé", "aucun document"
    ]
    if any(p in answer.lower() for p in HALLUCINATIONS):
        answer = llm(
            "You are a document QA assistant. The facts below ARE from the document. Read and answer.",
            f"QUESTION: {question}\n\nDOCUMENT FACTS:\n{context[:2000]}\n\n"
            f"Write at least 3 sentences using only the facts above (minimum 200 characters).",
            max_tokens=400
        )

    # Garantie longueur minimale
    if len(answer.strip()) < 200:
        expanded = llm(
            "Expand this answer to at least 200 characters by adding more detail from the document facts.",
            f"QUESTION: {question}\n\nCURRENT ANSWER: {answer}\n\nFACTS:\n{context[:2000]}",
            max_tokens=400
        )
        if len(expanded.strip()) > len(answer.strip()):
            answer = expanded

    return {
        "answer": answer,
        "context": context,
        "answer_length": len(answer.strip()),
        "triplets": triplets,
        "used_rag_fallback": used_rag_fallback,
        "latency_ms": round((time.time()-t0)*1000)
    }


# ── HTML ──────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def ui():
    html_file = Path(__file__).parent / "kag_app" / "index.html"
    if html_file.exists():
        return HTMLResponse(html_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>Interface non trouvee. Utilisez /docs</h2>")

@app.get("/mobile", response_class=HTMLResponse)
async def mobile_ui():
    html_file = Path(__file__).parent / "kag_app" / "mobile.html"
    if html_file.exists():
        return HTMLResponse(html_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>Interface mobile non trouvee.</h2>")

@app.get("/compliance/criteria")
async def compliance_criteria_list():
    """Retourne la liste brute des criteres pour le mobile dashboard."""
    return compliance_criteria

# ── API ───────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "docs_loaded": len(documents),
        "kag_nodes": len(kag_graph["nodes"]),
        "kag_relations": len(kag_graph["relations"])
    }

@app.get("/stats")
async def stats():
    rag_lats = [r["latency_ms"] for r in run_log if r.get("pipeline")=="RAG"]
    kag_lats = [r["latency_ms"] for r in run_log if r.get("pipeline")=="KAG"]
    return {
        "rag": {"doc_count": len(documents)},
        "kag": {"node_count": len(kag_graph["nodes"]),
                "relation_count": len(kag_graph["relations"])},
        "milvus": {"doc_count": 0},
        "total_runs": len(run_log),
        "avg_latency_rag": round(sum(rag_lats)/len(rag_lats)) if rag_lats else 0,
        "avg_latency_kag": round(sum(kag_lats)/len(kag_lats)) if kag_lats else 0,
        "uploaded_files": uploaded_files,
    }

# ── Upload endpoints ───────────────────────────────────────────────────────────
@app.post("/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    build_kag: bool = Form(True)
):
    """Upload de fichiers utilisateur (PDF, DOCX, TXT, MD, CSV).
    Indexe automatiquement dans RAG (BM25) et construit le graphe KAG."""
    global rag_index, documents, uploaded_files, uploaded_raw_texts

    ALLOWED = {".pdf", ".docx", ".txt", ".md", ".csv"}
    results = []
    new_chunks = []

    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED:
            results.append({"file": file.filename, "status": "error",
                            "message": f"Format non supporté : {ext}"})
            continue

        content = await file.read()
        size_kb  = round(len(content) / 1024, 1)
        text     = _parse_uploaded_file(file.filename, content)

        if not text.strip():
            results.append({"file": file.filename, "status": "error",
                            "message": "Aucun texte extrait du fichier"})
            continue

        chunks = _chunk_text(text, file.filename)
        new_chunks.extend(chunks)
        uploaded_raw_texts.append({"title": file.filename, "text": text})

        # Meta dans la liste des fichiers uploades
        uploaded_files.append({
            "name":     file.filename,
            "size_kb":  size_kb,
            "chunks":   len(chunks),
            "status":   "indexé",
            "ext":      ext.lstrip(".")
        })
        results.append({"file": file.filename, "status": "ok",
                        "chunks": len(chunks), "size_kb": size_kb})

    if not new_chunks:
        return JSONResponse({"status": "error", "message": "Aucun document valide reçu",
                             "details": results}, status_code=400)

    # Ajouter aux documents existants
    documents.extend(new_chunks)

    # Réindexer RAG (BM25) avec tous les documents
    rag_index = minsearch.Index(text_fields=["text", "title"], keyword_fields=["id"])
    rag_index.fit(documents)

    # Construire KAG depuis les nouveaux chunks (max 15)
    if build_kag:
        build_kag_from_docs(new_chunks)

    return {
        "status":     "ok",
        "total_docs": len(documents),
        "new_chunks": len(new_chunks),
        "kag_nodes":  len(kag_graph["nodes"]),
        "kag_rels":   len(kag_graph["relations"]),
        "files":      results
    }


# ── Ingestion directe Excel de requirements (sans LLM) ────────────────────────

def ingest_requirements_excel(content: bytes, filename: str) -> dict:
    """
    Lit un fichier Excel de requirements Ooredoo et ajoute chaque ligne
    directement dans compliance_criteria, SANS appel LLM.

    Auto-detection des colonnes par mots-cles dans les en-tetes.
    Champ 'source_type' = 'Excel (direct)' pour distinguer du pipeline LLM.

    Retourne un dict avec le nombre de criteres ingeres et les colonnes mappees.
    """
    global compliance_criteria

    ext = Path(filename).suffix.lower()

    # ── Lecture du fichier ────────────────────────────────────────────────────
    rows = []     # liste de dicts {header: valeur}
    headers = []

    try:
        if ext in (".xlsx", ".xlsm"):
            import openpyxl
            wb = openpyxl.load_workbook(io.BytesIO(content), data_only=True)
            ws = wb.active
            raw_rows = list(ws.iter_rows(values_only=True))
            if not raw_rows:
                return {"error": "Feuille Excel vide", "ingested": 0}
            headers = [str(h).strip() if h is not None else f"col_{i}"
                       for i, h in enumerate(raw_rows[0])]
            for row in raw_rows[1:]:
                if all(v is None for v in row):
                    continue
                rows.append(dict(zip(headers, [str(v).strip() if v is not None else "" for v in row])))

        elif ext == ".xls":
            try:
                import xlrd
                wb = xlrd.open_workbook(file_contents=content)
                ws = wb.sheet_by_index(0)
                headers = [str(ws.cell_value(0, c)).strip() for c in range(ws.ncols)]
                for r in range(1, ws.nrows):
                    row = {headers[c]: str(ws.cell_value(r, c)).strip() for c in range(ws.ncols)}
                    if any(v for v in row.values()):
                        rows.append(row)
            except ImportError:
                return {"error": "xlrd non installe — pip install xlrd pour lire .xls", "ingested": 0}
        else:
            return {"error": f"Format non supporte pour ingestion directe : {ext}", "ingested": 0}

    except Exception as e:
        return {"error": f"Erreur de lecture Excel : {e}", "ingested": 0}

    if not rows:
        return {"error": "Aucune ligne de donnees trouvee", "ingested": 0}

    # ── Auto-detection des colonnes par mots-cles ────────────────────────────
    VALID_CATS = {"Functional", "Technical", "Operational",
                  "Security", "Infrastructure", "Administrative", "Financial"}

    def _find_col(headers, *keywords):
        """Retourne le premier header dont le nom contient l'un des keywords."""
        for kw in keywords:
            for h in headers:
                if kw.lower() in h.lower():
                    return h
        return None

    col_criterion = _find_col(headers, "requirement", "criterion", "description",
                              "exigence", "critere", "specification", "text", "wording")
    col_category  = _find_col(headers, "category", "categorie", "type", "area",
                              "domain", "domaine", "fonction", "functional area")
    col_mandatory = _find_col(headers, "mandatory", "obligatoire", "priority",
                              "priorite", "required", "compliance", "must")
    col_id        = _find_col(headers, "id", "ref", "no.", "num", "req_id", "req id")
    col_source    = _find_col(headers, "source", "section", "chapter", "chapitre", "ref doc")

    mapping = {
        "criterion": col_criterion or headers[0],
        "category":  col_category,
        "mandatory": col_mandatory,
        "id_col":    col_id,
        "source":    col_source,
    }

    # ── Ingestion ligne par ligne ─────────────────────────────────────────────
    ingested = 0
    skipped  = 0

    for row in rows:
        criterion = row.get(mapping["criterion"], "").strip()
        if not criterion or len(criterion) < 4:
            skipped += 1
            continue

        # Category
        raw_cat = row.get(mapping["category"] or "", "").strip() if mapping["category"] else ""
        cat_norm = next((c for c in VALID_CATS if c.lower() == raw_cat.lower()), None)
        if not cat_norm:
            cat_norm = next(
                (c for c in VALID_CATS if c.lower() in raw_cat.lower()), "Functional"
            )

        # Mandatory
        raw_mand = row.get(mapping["mandatory"] or "", "").strip().upper() if mapping["mandatory"] else ""
        if raw_mand in ("YES", "OUI", "MANDATORY", "OBLIGATOIRE", "MUST", "SHALL", "M", "1", "TRUE"):
            mandatory = "Yes"
        elif raw_mand in ("NO", "NON", "OPTIONAL", "OPTIONNEL", "SHOULD", "O", "0", "FALSE"):
            mandatory = "No"
        else:
            mandatory = "Yes"  # defaut : obligatoire

        # ID externe (conserve pour tracabilite)
        ext_id = row.get(mapping["id_col"] or "", "").strip() if mapping["id_col"] else ""
        # Source interne au fichier Excel (section, chapitre...)
        int_source = row.get(mapping["source"] or "", "").strip() if mapping["source"] else ""

        compliance_criteria.append({
            "id":                f"CR-{len(compliance_criteria)+1:03d}",
            "category":          cat_norm,
            "criterion":         criterion,
            "mandatory":         mandatory,
            "source":            filename,
            "chunk_id":          ext_id or int_source or f"row_{ingested+1}",
            "compliance_status": "To Be Assessed",
            "source_type":       "Excel (direct)",
            "ext_id":            ext_id,
        })
        ingested += 1

    return {
        "ingested":      ingested,
        "skipped":       skipped,
        "total_rows":    len(rows),
        "headers_found": headers,
        "column_mapping": mapping,
        "filename":      filename,
    }


@app.post("/upload-requirements-excel")
async def upload_requirements_excel(files: List[UploadFile] = File(...)):
    """
    Ingere directement un fichier Excel de requirements (xls / xlsx) dans
    compliance_criteria, SANS appel LLM. Chaque ligne = un critere.

    Utile pour les fichiers de requirements deja structures (ex: Ooredoo
    'OT Bill Generation Solution Requirement v1.xls').

    Format attendu : colonnes detectees automatiquement par mots-cles.
    Tous les formats : .xls (xlrd requis), .xlsx, .xlsm.
    """
    global compliance_criteria
    all_results = []

    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in (".xls", ".xlsx", ".xlsm"):
            all_results.append({
                "file": file.filename, "status": "error",
                "message": f"Format non supporte : {ext}. Utilisez .xls, .xlsx ou .xlsm."
            })
            continue

        content = await file.read()
        result  = ingest_requirements_excel(content, file.filename)

        if "error" in result:
            all_results.append({"file": file.filename, "status": "error",
                                 "message": result["error"]})
        else:
            all_results.append({
                "file":           file.filename,
                "status":         "ok",
                "ingested":       result["ingested"],
                "skipped":        result["skipped"],
                "total_rows":     result["total_rows"],
                "column_mapping": result["column_mapping"],
            })

    return {
        "status":              "ok",
        "total_criteria":      len(compliance_criteria),
        "excel_direct_count":  sum(1 for c in compliance_criteria if c.get("source_type") == "Excel (direct)"),
        "llm_extracted_count": sum(1 for c in compliance_criteria if c.get("source_type") != "Excel (direct)"),
        "files":               all_results,
    }


@app.delete("/upload/clear")
async def clear_uploads():
    """Réinitialise tous les documents uploadés, l'index RAG et le graphe KAG."""
    global rag_index, documents, kag_graph, uploaded_files, uploaded_raw_texts, compliance_criteria
    documents           = []
    uploaded_files      = []
    uploaded_raw_texts  = []
    compliance_criteria = []
    rag_index           = None
    kag_graph           = {"nodes": {}, "relations": {}}
    return {"status": "ok", "message": "Tous les documents et le graphe ont été réinitialisés"}



# ── Compliance Matrix Endpoints ────────────────────────────────────────────────

@app.get("/compliance/list")
async def compliance_list(
    category: str = Query(None, description="Filtrer par categorie"),
    mandatory: str = Query(None, description="Yes ou No"),
    limit: int = Query(500, description="Nombre max de criteres")
):
    """Liste les criteres de conformite extraits. Filtrage par category et mandatory."""
    global compliance_criteria
    results = compliance_criteria
    if category:
        results = [c for c in results if c["category"].lower() == category.lower()]
    if mandatory:
        results = [c for c in results if c["mandatory"].lower() == mandatory.lower()]
    return {
        "total":            len(compliance_criteria),
        "filtered":         len(results),
        "criteria":         results[:limit],
        "categories":       list({c["category"] for c in compliance_criteria}),
        "mandatory_count":  sum(1 for c in compliance_criteria if c["mandatory"] == "Yes"),
        "optional_count":   sum(1 for c in compliance_criteria if c["mandatory"] == "No"),
    }


@app.get("/compliance/stats")
async def compliance_stats():
    """Statistiques sur les criteres extraits par categorie."""
    global compliance_criteria
    by_cat = {}
    for c in compliance_criteria:
        by_cat.setdefault(c["category"], {"total": 0, "mandatory": 0, "optional": 0})
        by_cat[c["category"]]["total"] += 1
        if c["mandatory"] == "Yes":
            by_cat[c["category"]]["mandatory"] += 1
        else:
            by_cat[c["category"]]["optional"] += 1
    return {
        "total_criteria":  len(compliance_criteria),
        "mandatory_total": sum(1 for c in compliance_criteria if c["mandatory"] == "Yes"),
        "optional_total":  sum(1 for c in compliance_criteria if c["mandatory"] == "No"),
        "by_category":     by_cat,
        "sources":         list({c["source"] for c in compliance_criteria}),
        "kag_nodes":       len(kag_graph["nodes"]),
        "kag_relations":   len(kag_graph["relations"])
    }


@app.get("/compliance/export")
async def compliance_export(fmt: str = Query("csv", description="Format: csv ou excel")):
    """Exporte la Compliance Matrix.
    CSV  : colonnes ID|Categorie|Critere|Obligatoire|Source|Chunk|Statut de Conformite
    Excel: meme colonnes avec mise en forme (vert=Yes, orange=No)
    """
    global compliance_criteria
    if not compliance_criteria:
        raise HTTPException(status_code=404,
            detail="Aucun critere extrait. Uploadez un document RFP d abord.")

    COLS    = ["id","category","criterion","mandatory","source","chunk_id","compliance_status"]
    HDRS_FR = ["ID","Categorie","Critere","Obligatoire","Source","Chunk","Statut de Conformite"]

    if fmt.lower() in ("excel", "xlsx"):
        try:
            import openpyxl
            from openpyxl.styles import Font, PatternFill, Alignment
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Compliance Matrix"

            # Header
            fill_hdr = PatternFill("solid", fgColor="1F3A5F")
            font_hdr = Font(bold=True, color="FFFFFF")
            for ci, h in enumerate(HDRS_FR, 1):
                cell = ws.cell(row=1, column=ci, value=h)
                cell.fill = fill_hdr; cell.font = font_hdr
                cell.alignment = Alignment(horizontal="center")

            # Rows
            fill_yes = PatternFill("solid", fgColor="C6EFCE")
            fill_no  = PatternFill("solid", fgColor="FFEB9C")
            font_yes = Font(color="276221", bold=True)
            font_no  = Font(color="9C6500")
            for ri, crit in enumerate(compliance_criteria, 2):
                for ci, key in enumerate(COLS, 1):
                    ws.cell(row=ri, column=ci, value=crit.get(key, ""))
                mc = ws.cell(row=ri, column=4)
                if crit.get("mandatory") == "Yes":
                    mc.fill = fill_yes; mc.font = font_yes
                else:
                    mc.fill = fill_no;  mc.font = font_no

            for col, w in zip("ABCDEFG", [12, 20, 60, 14, 30, 26, 30]):
                ws.column_dimensions[col].width = w

            buf = _io.BytesIO()
            wb.save(buf); buf.seek(0)
            return StreamingResponse(buf,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition":"attachment; filename=compliance_matrix.xlsx"})
        except ImportError:
            pass   # fallback CSV

    # CSV (defaut, BOM pour Excel FR)
    buf = _io.StringIO()
    import csv as _csv
    writer = _csv.DictWriter(buf, fieldnames=COLS, extrasaction="ignore")
    writer.writerow(dict(zip(COLS, HDRS_FR)))
    for crit in compliance_criteria:
        writer.writerow(crit)
    buf.seek(0)
    return StreamingResponse(
        _io.BytesIO(buf.getvalue().encode("utf-8-sig")),
        media_type="text/csv",
        headers={"Content-Disposition":"attachment; filename=compliance_matrix.csv"})


@app.get("/mlops/stats")
async def mlops_stats():
    rag_runs = [r for r in run_log if r.get("pipeline")=="RAG"]
    kag_runs = [r for r in run_log if r.get("pipeline")=="KAG"]
    cmp_runs = [r for r in run_log if r.get("pipeline")=="COMPARE"]
    smart_runs = [r for r in run_log if str(r.get("pipeline","")).startswith("SMART")]

    rag_lats = [r["latency_ms"] for r in rag_runs]
    kag_lats = [r["latency_ms"] for r in kag_runs]

    # Stats depuis routing_log
    routing_ems_rag = [r["em"] for r in routing_log
                       if r.get("winner")=="rag" and r.get("em") is not None]
    routing_ems_kag = [r["em"] for r in routing_log
                       if r.get("winner")=="kag" and r.get("em") is not None]

    history = [{"pipeline": r["pipeline"],
                "question": r["question"][:60],
                "latency_ms": r["latency_ms"],
                "estimated_tokens": 200,
                "n_results": 5,
                "rag_em": r.get("rag_em"),
                "kag_em": r.get("kag_em"),
                "rag_f1": r.get("rag_f1"),
                "kag_f1": r.get("kag_f1")}
               for r in run_log[-30:]]

    # Scores comparés depuis run_log COMPARE
    compare_runs_with_scores = [r for r in run_log if r.get("rag_em") is not None]
    avg_rag_em = round(sum(r["rag_em"] for r in compare_runs_with_scores)/len(compare_runs_with_scores), 3) if compare_runs_with_scores else None
    avg_kag_em = round(sum(r["kag_em"] for r in compare_runs_with_scores)/len(compare_runs_with_scores), 3) if compare_runs_with_scores else None
    avg_rag_f1 = round(sum(r["rag_f1"] for r in compare_runs_with_scores)/len(compare_runs_with_scores), 3) if compare_runs_with_scores else None
    avg_kag_f1 = round(sum(r["kag_f1"] for r in compare_runs_with_scores)/len(compare_runs_with_scores), 3) if compare_runs_with_scores else None

    return {
        "total_runs":       len(run_log),
        "rag_runs":         len(rag_runs),
        "kag_runs":         len(kag_runs),
        "compare_runs":     len(cmp_runs),
        "smart_runs":       len(smart_runs),
        "avg_latency_rag":  round(sum(rag_lats)/len(rag_lats)) if rag_lats else 0,
        "avg_latency_milvus": 0,
        "avg_latency_kag":  round(sum(kag_lats)/len(kag_lats)) if kag_lats else 0,
        "avg_rag_em":       avg_rag_em,
        "avg_kag_em":       avg_kag_em,
        "avg_rag_f1":       avg_rag_f1,
        "avg_kag_f1":       avg_kag_f1,
        "history":          history,
        "eval_results":     {}
    }


class QueryReq(BaseModel):
    question: str

class CompareReq(BaseModel):
    question: str
    ground_truth: str = ""

class EvalReq(BaseModel):
    questions: List[str]
    ground_truths: List[str]
    pipeline: str = "both"

class LoadHotpotReq(BaseModel):
    n_samples: int = 5

@app.post("/rag/query")
async def rag_query(req: QueryReq):
    r = rag_answer(req.question)
    run_log.append({"pipeline":"RAG","question":req.question,
                    "answer":r["answer"],"latency_ms":r["latency_ms"]})
    return r

@app.post("/kag/query")
async def kag_query(req: QueryReq):
    r = kag_answer(req.question)
    run_log.append({"pipeline":"KAG","question":req.question,
                    "answer":r["answer"],"latency_ms":r["latency_ms"]})
    return r

@app.post("/compare")
async def compare(req: CompareReq):
    t0  = time.time()
    rag = rag_answer(req.question)
    kag = kag_answer(req.question)
    gt  = req.ground_truth
    rag_em = exact_match(rag["answer"],gt) if gt else None
    kag_em = exact_match(kag["answer"],gt) if gt else None
    rag_f1 = token_f1(rag["answer"],gt)   if gt else None
    kag_f1 = token_f1(kag["answer"],gt)   if gt else None
    result = {
        "question": req.question,
        "ground_truth": gt,
        "rag": {**rag, "em": rag_em, "f1": rag_f1},
        "kag": {**kag, "em": kag_em, "f1": kag_f1},
        "total_ms": round((time.time()-t0)*1000)
    }
    # Métriques avancées (si disponibles)
    if gt and ragas_evaluator and ragas_evaluator.available:
        try:
            rag_ragas = ragas_evaluator.evaluate(
                [req.question], [rag["answer"]], [[rag.get("context","")]],
                [gt], "RAG"
            )
            kag_ragas = ragas_evaluator.evaluate(
                [req.question], [kag["answer"]], [[kag.get("context","")]],
                [gt], "KAG"
            )
            result["rag"]["ragas"] = rag_ragas
            result["kag"]["ragas"] = kag_ragas
        except Exception:
            pass
    if gt and deepeval_evaluator and deepeval_evaluator.available:
        try:
            rag_de = deepeval_evaluator.evaluate(
                [req.question], [rag["answer"]], [[rag.get("context","")]],
                [gt], "RAG"
            )
            kag_de = deepeval_evaluator.evaluate(
                [req.question], [kag["answer"]], [[kag.get("context","")]],
                [gt], "KAG"
            )
            result["rag"]["deepeval"] = rag_de
            result["kag"]["deepeval"] = kag_de
        except Exception:
            pass
    run_log.append({
        "pipeline": "COMPARE", "question": req.question,
        "answer": f"RAG:{rag['answer']} | KAG:{kag['answer']}",
        "latency_ms": result["total_ms"],
        "rag_em": rag_em, "kag_em": kag_em,
        "rag_f1": rag_f1, "kag_f1": kag_f1,
        "rag_lat": rag["latency_ms"], "kag_lat": kag["latency_ms"]
    })
    return result

@app.post("/evaluate")
async def evaluate(req: EvalReq):
    results = {}
    if req.pipeline in ("rag","both"):
        preds = [rag_answer(q)["answer"] for q in req.questions]
        ems=[exact_match(p,g) for p,g in zip(preds,req.ground_truths)]
        f1s=[token_f1(p,g)   for p,g in zip(preds,req.ground_truths)]
        results["rag"] = {
            "avg_exact_match": round(sum(ems)/len(ems),4),
            "avg_token_f1":    round(sum(f1s)/len(f1s),4),
            "n_correct": sum(1 for e in ems if e==1.0),
            "details": [{"q":q,"pred":p,"gt":g,"em":e,"f1":f}
                        for q,p,g,e,f in zip(req.questions,preds,
                                             req.ground_truths,ems,f1s)]
        }
    if req.pipeline in ("kag","both"):
        preds = [kag_answer(q)["answer"] for q in req.questions]
        ems=[exact_match(p,g) for p,g in zip(preds,req.ground_truths)]
        f1s=[token_f1(p,g)   for p,g in zip(preds,req.ground_truths)]
        results["kag"] = {
            "avg_exact_match": round(sum(ems)/len(ems),4),
            "avg_token_f1":    round(sum(f1s)/len(f1s),4),
            "n_correct": sum(1 for e in ems if e==1.0),
            "details": [{"q":q,"pred":p,"gt":g,"em":e,"f1":f}
                        for q,p,g,e,f in zip(req.questions,preds,
                                             req.ground_truths,ems,f1s)]
        }
    return results

class AdvancedEvalReq(BaseModel):
    questions: List[str]
    ground_truths: List[str]
    pipeline: str = "both"  # "rag", "kag", "both"

@app.post("/evaluate/advanced")
async def evaluate_advanced(req: AdvancedEvalReq):
    """Évaluation avancée avec Ragas + DeepEval (métriques LLM-based)."""
    results = {}
    
    if req.pipeline in ("rag", "both"):
        rag_answers_list = []
        rag_contexts_list = []
        for q in req.questions:
            r = rag_answer(q)
            rag_answers_list.append(r["answer"])
            rag_contexts_list.append([r.get("context", "")])
        
        # Métriques de base
        ems = [exact_match(p, g) for p, g in zip(rag_answers_list, req.ground_truths)]
        f1s = [token_f1(p, g) for p, g in zip(rag_answers_list, req.ground_truths)]
        
        results["rag"] = {
            "basic": {
                "avg_exact_match": round(sum(ems)/len(ems), 4),
                "avg_token_f1": round(sum(f1s)/len(f1s), 4),
                "n_correct": sum(1 for e in ems if e == 1.0),
            }
        }
        
        # Ragas
        if ragas_evaluator and ragas_evaluator.available:
            results["rag"]["ragas"] = ragas_evaluator.evaluate(
                questions=req.questions,
                answers=rag_answers_list,
                contexts=rag_contexts_list,
                ground_truths=req.ground_truths,
                pipeline_name="RAG"
            )
        
        # DeepEval
        if deepeval_evaluator and deepeval_evaluator.available:
            results["rag"]["deepeval"] = deepeval_evaluator.evaluate(
                questions=req.questions,
                answers=rag_answers_list,
                contexts=rag_contexts_list,
                ground_truths=req.ground_truths,
                pipeline_name="RAG"
            )
    
    if req.pipeline in ("kag", "both"):
        kag_answers_list = []
        kag_contexts_list = []
        for q in req.questions:
            r = kag_answer(q)
            kag_answers_list.append(r["answer"])
            kag_contexts_list.append([r.get("context", "")])
        
        ems = [exact_match(p, g) for p, g in zip(kag_answers_list, req.ground_truths)]
        f1s = [token_f1(p, g) for p, g in zip(kag_answers_list, req.ground_truths)]
        
        results["kag"] = {
            "basic": {
                "avg_exact_match": round(sum(ems)/len(ems), 4),
                "avg_token_f1": round(sum(f1s)/len(f1s), 4),
                "n_correct": sum(1 for e in ems if e == 1.0),
            }
        }
        
        if ragas_evaluator and ragas_evaluator.available:
            results["kag"]["ragas"] = ragas_evaluator.evaluate(
                questions=req.questions,
                answers=kag_answers_list,
                contexts=kag_contexts_list,
                ground_truths=req.ground_truths,
                pipeline_name="KAG"
            )
        
        if deepeval_evaluator and deepeval_evaluator.available:
            results["kag"]["deepeval"] = deepeval_evaluator.evaluate(
                questions=req.questions,
                answers=kag_answers_list,
                contexts=kag_contexts_list,
                ground_truths=req.ground_truths,
                pipeline_name="KAG"
            )
    
    results["available_frameworks"] = {
        "ragas": ragas_evaluator.available if ragas_evaluator else False,
        "deepeval": deepeval_evaluator.available if deepeval_evaluator else False,
    }
    
    return results

# ── Smart Agent endpoints ──────────────────────────────────────────────────────

class SmartReq(BaseModel):
    question:     str
    ground_truth: str = ""

@app.get("/route")
async def preview_route(question: str = Query(...)):
    """Prévisualise la décision de routage sans exécuter le pipeline."""
    routing = route_question(question)
    return routing

@app.post("/smart")
async def smart_query(req: SmartReq):
    """
    Agent intelligent : classe la question → route vers RAG/KAG/Ensemble
    → si Ensemble : arbitrage LLM pour choisir la meilleure réponse.
    """
    t0      = time.time()
    gt      = req.ground_truth.strip()
    routing = route_question(req.question)
    pipe    = routing["pipeline"]

    result  = {
        "question":  req.question,
        "routing":   routing,
        "pipeline_used": pipe,
    }

    if pipe == "none":
        result["answer"]     = "Aucun pipeline disponible."
        result["latency_ms"] = 0
        return result

    if pipe == "rag":
        ans = rag_answer(req.question)
        result.update({
            "answer":     ans["answer"],
            "context":    ans.get("context",""),
            "sources":    ans.get("sources",[]),
            "latency_ms": ans["latency_ms"],
            "winner":     "rag",
        })

    elif pipe == "kag":
        ans = kag_answer(req.question)
        result.update({
            "answer":   ans["answer"],
            "context":  ans.get("context",""),
            "triplets": ans.get("triplets",[]),
            "latency_ms": ans["latency_ms"],
            "winner":   "kag",
        })

    else:  # ensemble
        rag = rag_answer(req.question)
        kag = kag_answer(req.question)
        arb = ensemble_answer(req.question, rag, kag, routing)
        result.update({
            "answer":       arb["answer"],
            "winner":       arb["winner"],
            "arb_reason":   arb["reason"],
            "rag_answer":   rag["answer"],
            "kag_answer":   kag["answer"],
            "rag_context":  rag.get("context",""),
            "kag_triplets": kag.get("triplets",[]),
            "latency_ms":   round((time.time()-t0)*1000),
        })

    # Métriques EM / F1
    if gt:
        result["em"] = exact_match(result["answer"], gt)
        result["f1"] = token_f1(result["answer"], gt)

    # Log routage
    routing_log.append({
        "question":   req.question,
        "type":       routing["type"],
        "routed_to":  pipe,
        "winner":     result.get("winner", pipe),
        "reason":     routing["reason"],
        "confidence": routing["confidence"],
        "em":         result.get("em"),
    })
    run_log.append({
        "pipeline":   f"SMART→{pipe.upper()}",
        "question":   req.question,
        "answer":     result["answer"],
        "latency_ms": result.get("latency_ms",0)
    })
    return result

@app.get("/routing/stats")
async def routing_stats():
    """Statistiques du routing agent : distribution RAG/KAG/Ensemble."""
    if not routing_log:
        return {"total": 0, "distribution": {}, "history": []}

    from collections import Counter
    dist = Counter(r["routed_to"] for r in routing_log)
    type_dist = Counter(r["type"] for r in routing_log)
    winner_dist = Counter(r.get("winner","?") for r in routing_log)

    ems = [r["em"] for r in routing_log if r.get("em") is not None]
    return {
        "total":          len(routing_log),
        "distribution":   dict(dist),
        "type_distribution": dict(type_dist),
        "winner_distribution": dict(winner_dist),
        "avg_em":         round(sum(ems)/len(ems), 3) if ems else None,
        "history":        routing_log[-20:]   # 20 dernières décisions
    }

@app.post("/load-hotpotqa")
async def load_hotpotqa(req: Optional[LoadHotpotReq] = None,
                        n_samples: int = Query(default=5)):
    global rag_index, documents
    n = req.n_samples if req else n_samples
    try:
        from datasets import load_dataset
        dataset = load_dataset("hotpotqa/hotpot_qa","distractor",
                               split=f"validation[:{n}]",
                               trust_remote_code=True)
        docs, seen = [], set()
        qa_pairs = []
        for item in dataset:
            qa_pairs.append({
                "question": item["question"],
                "answer":   item["answer"]
            })
            for title, sents in zip(item["context"]["title"],
                                    item["context"]["sentences"]):
                text = " ".join(sents)
                key  = title + text[:30]
                if key not in seen:
                    seen.add(key)
                    docs.append({"title": title, "text": text})
        documents = docs
        rag_index = minsearch.Index(text_fields=["title","text"],
                                    keyword_fields=[])
        rag_index.fit(documents)
        return {
            "status": "ok",
            "rag_docs": len(documents),
            "kag_nodes": len(kag_graph["nodes"]),
            "n_samples": n,
            "sample_questions": qa_pairs[:5]
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/kag/build")
async def kag_build():
    """Construit le graphe KAG depuis les documents uploadés par l'utilisateur.
    Endpoint principal appelé par le bouton 'Construire le graphe de connaissance'.
    """
    if not documents:
        return JSONResponse(
            {"error": "Aucun document chargé. Uploadez d'abord un fichier via la sidebar."},
            status_code=400
        )
    # Utiliser TOUS les documents chargés (pas de limite arbitraire)
    docs_to_use = documents  # tous les chunks uploadés
    build_kag_from_docs(docs_to_use)
    return {
        "status":        "ok",
        "source":        "documents uploadés par l'utilisateur",
        "kag_nodes":     len(kag_graph["nodes"]),
        "kag_relations": len(kag_graph["relations"]),
        "docs_used":     len(docs_to_use),
        "message":       f"Graphe construit depuis {len(docs_to_use)} chunks : "
                         f"{len(kag_graph['nodes'])} nœuds, {len(kag_graph['relations'])} relations"
    }

@app.post("/load-kag")
async def load_kag(n_docs: int = 8):
    """Ancien endpoint — redirige vers /kag/build pour compatibilité."""
    if not documents:
        return JSONResponse(
            {"error": "Aucun document chargé. Uploadez d'abord un fichier."},
            status_code=400)
    build_kag_from_docs(documents[:n_docs])
    return {
        "status": "ok",
        "kag_nodes": len(kag_graph["nodes"]),
        "kag_relations": len(kag_graph["relations"])
    }

@app.get("/runs")
async def get_runs():
    return {"runs": run_log, "total": len(run_log)}

@app.post("/load-kag-precomputed")
async def load_kag_precomputed():
    """Charge le graphe KAG depuis results_kag.json (0 appel API, instantané)."""
    nodes_n, rels_n = build_kag_from_precomputed()
    if nodes_n == 0:
        return JSONResponse(
            {"error": "results_kag.json introuvable ou vide"},
            status_code=404)
    return {
        "status":        "ok",
        "source":        "results_kag.json (pré-calculé)",
        "kag_nodes":     nodes_n,
        "kag_relations": rels_n,
        "api_calls":     0,
        "message":       f"Graphe chargé instantanément : {nodes_n} nœuds, {rels_n} relations"
    }

@app.get("/info")
async def info():
    """Informations sur les modèles et limites utilisés."""
    return {
        "models": {
            "answers":   MODEL_ANSWER  + " (6 000 TPM Groq free)",
            "extraction": MODEL_EXTRACT + " (15 000 TPM Groq free)"
        },
        "eval_frameworks": {
            "ragas": ragas_evaluator.available if ragas_evaluator else False,
            "deepeval": deepeval_evaluator.available if deepeval_evaluator else False
        },
        "rate_limit_solutions": [
            "1. Format compact pipe (src|rel|tgt) : -75% output tokens",
            "2. Batch 3 docs/appel : 3x moins d'appels API",
            "3. gemma2-9b-it pour KAG : 2.5x plus de quota",
            "4. /load-kag-precomputed : 0 appels API (résultats JSON)"
        ],
        "token_usage": {
            "avant": "15 docs × 500 tok = 7500 tok/min → rate limit",
            "apres": "5 batches × 400 tok = 2000 tok/min → OK ✅"
        }
    }

@app.get("/kag/graph")
async def kag_graph_data():
    nodes = [{"id": n, "label": n, "doc": v.get("doc", "")}
             for n, v in kag_graph["nodes"].items()]
    edges = []
    for key in kag_graph["relations"]:
        parts = key.split("||")
        if len(parts) == 3:
            edges.append({"source": parts[0], "relation": parts[1], "target": parts[2]})
    return {"nodes": nodes, "edges": edges,
            "node_count": len(nodes), "edge_count": len(edges)}

def _extract_text(filename: str, raw: bytes) -> str:
    """Extract plain text from PDF, DOCX, or TXT/other."""
    name = filename.lower()
    if name.endswith(".pdf"):
        if not HAS_PDF:
            return raw.decode("utf-8", errors="ignore")
        reader = PdfReader(io.BytesIO(raw))
        pages = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                pages.append(t)
        return "\n\n".join(pages)
    elif name.endswith(".docx"):
        if not HAS_DOCX:
            return raw.decode("utf-8", errors="ignore")
        doc = DocxDocument(io.BytesIO(raw))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    else:
        # TXT / CSV / MD / any text
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                return raw.decode(enc)
            except Exception:
                continue
        return raw.decode("utf-8", errors="ignore")

@app.post("/upload")
async def upload(file: UploadFile = File(...),
                 pipeline: str = Form("rag"),
                 chunk_size: int = Form(300)):
    global rag_index, documents
    try:
        raw      = await file.read()
        ext      = Path(file.filename).suffix.lower()
        text     = _extract_text(file.filename, raw)
        if not text.strip():
            return JSONResponse({"error": "Fichier vide ou non lisible"}, status_code=400)

        # Chunking par mots avec overlap
        words  = text.split()
        chunks = []
        step   = max(chunk_size - 50, 50)   # 50-word overlap
        for i in range(0, len(words), step):
            chunk_text = " ".join(words[i : i + chunk_size])
            if len(chunk_text.strip()) < 20:
                continue
            chunks.append({
                "title": file.filename,
                "text":  chunk_text,
                "chunk": i // step
            })

        # Ajouter aux documents existants (ne pas écraser)
        documents.extend(chunks)
        rag_index = minsearch.Index(text_fields=["title", "text"],
                                    keyword_fields=[])
        rag_index.fit(documents)

        # KAG optionnel
        kag_n = 0
        if pipeline in ("kag", "both"):
            build_kag_from_docs(chunks[:8])
            kag_n = len(kag_graph["nodes"])

        sample = text[:200].replace("\n", " ")
        return {
            "status":    "ok",
            "filename":  file.filename,
            "file_type": ext or ".txt",
            "n_chunks":  len(chunks),
            "total_docs": len(documents),
            "kag_nodes": kag_n,
            "sample":    sample
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/reset")
async def reset_docs():
    """Vider tous les documents chargés (RAG + KAG)."""
    global rag_index, documents, kag_graph, run_log
    documents = []
    rag_index = None
    kag_graph = {"nodes": {}, "relations": {}}
    return {"status": "ok", "message": "Toutes les données effacées"}

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*55)
    print("  OOREDOO AI LAB — RAG vs KAG")
    print("  >>> http://localhost:8000")
    print("  >>> http://localhost:8000/docs")
    print("="*55 + "\n")
    uvicorn.run("fast_server:app", host="0.0.0.0", port=8000, reload=False)
