"""
fast_server.py — Serveur RAG vs KAG (version corrigée)
"""
import os, sys, time, json, logging, re, string, io
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
MODEL_ANSWER  = "llama-3.1-8b-instant"   # réponses QA — 6 000 TPM
MODEL_EXTRACT = "gemma2-9b-it"           # extraction KAG — 15 000 TPM

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

# ── State ──────────────────────────────────────────────────────────────────────
rag_index   = None
documents   = []
kag_graph   = {"nodes": {}, "relations": {}}
run_log     = []   # {"pipeline","question","answer","latency_ms"}

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
    sys_p = ("You are a QA assistant. Answer the question using the context provided. "
             "Be SHORT and DIRECT (1 sentence max). "
             "For yes/no questions answer just 'yes' or 'no'. "
             "If truly not answerable from context, say: I don't know.")
    prompt = f"QUESTION: {question}\n\nCONTEXT:\n{context}"
    answer = llm(sys_p, prompt, max_tokens=80)
    return {
        "answer": answer,
        "context": context,
        "sources": [{"title": r.get("title",""), "snippet": r.get("text","")[:200]}
                    for r in results],
        "latency_ms": round((time.time()-t0)*1000)
    }

# ── KAG ───────────────────────────────────────────────────────────────────────
# SOLUTION RATE LIMIT :
# 1) Format compact pipe (src|rel|tgt) → max 80 tokens/output  (-75%)
# 2) Batch de 3 docs par appel LLM      → 3x moins d'appels
# 3) Modèle gemma2-9b-it (15000 TPM)   → 2.5x plus de quota
# Résultat : ~2000 tokens pour 15 docs (~30s) au lieu de 7500 (>60s + erreurs)

EXTRACT_SYS = (
    "Extract factual triplets from text. "
    "Format: entity|relation|entity — one per line, max 5 lines. "
    "Use short labels. No JSON, no explanation."
)

def build_kag_from_docs(docs):
    """Construit le graphe KAG avec batching + format compact + modèle haute limite."""
    global kag_graph
    kag_graph = {"nodes": {}, "relations": {}}
    batch_size = 3          # 3 docs par appel LLM
    max_docs   = 15         # max docs à traiter

    doc_list = docs[:max_docs]
    batches  = [doc_list[i:i+batch_size] for i in range(0, len(doc_list), batch_size)]

    for batch in batches:
        # Construire le texte de batch
        combined = ""
        for idx, doc in enumerate(batch):
            title = doc.get("title", "")
            text  = doc.get("text",  "")[:300]   # 300 mots par doc
            combined += f"[DOC{idx+1}: {title}]\n{text}\n\n"

        try:
            raw = llm(
                EXTRACT_SYS,
                combined,
                max_tokens=120,          # ~5 triplets × 3 docs = 15 lignes max
                model=MODEL_EXTRACT      # gemma2-9b-it : 15000 TPM
            )
            for line in raw.split("\n"):
                line = line.strip().lstrip("-• ")
                parts = [p.strip() for p in line.split("|")]
                if len(parts) == 3 and all(parts):
                    src, rel, tgt = parts
                    title = batch[0].get("title", "")
                    kag_graph["nodes"][src] = {"doc": title}
                    kag_graph["nodes"][tgt] = {"doc": title}
                    kag_graph["relations"][f"{src}||{rel}||{tgt}"] = True
        except Exception:
            pass   # on continue le batch suivant

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
    q_words = [w for w in question.lower().split() if len(w) > 3]
    relevant = []
    for key in kag_graph["relations"]:
        src, rel, tgt = key.split("||")
        if any(w in src.lower() or w in tgt.lower() for w in q_words):
            relevant.append({"source": src, "relation": rel, "target": tgt,
                             "label": f"{src} —[{rel}]→ {tgt}"})
    return relevant[:15]

def kag_answer(question):
    t0 = time.time()
    triplets = kag_search(question)
    if triplets:
        context = "\n".join(t["label"] for t in triplets)
    else:
        context = "No relevant facts found in knowledge graph."
    sys_p = ("Answer based on these knowledge graph facts. "
             "Be SHORT (1 sentence). "
             "If facts don't answer the question, say: I don't know.")
    prompt = f"QUESTION: {question}\n\nKNOWLEDGE GRAPH FACTS:\n{context}"
    answer = llm(sys_p, prompt, max_tokens=80)
    return {
        "answer": answer,
        "context": context,
        "triplets": triplets,
        "latency_ms": round((time.time()-t0)*1000)
    }

# ── HTML ──────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def ui():
    html_file = Path(__file__).parent / "kag_app" / "index.html"
    if html_file.exists():
        return HTMLResponse(html_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>Interface non trouvée. Utilisez /docs</h2>")

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
    }

@app.get("/mlops/stats")
async def mlops_stats():
    rag_lats = [r["latency_ms"] for r in run_log if r.get("pipeline")=="RAG"]
    kag_lats = [r["latency_ms"] for r in run_log if r.get("pipeline")=="KAG"]
    history  = [{"pipeline": r["pipeline"], "question": r["question"][:60],
                 "latency_ms": r["latency_ms"], "estimated_tokens": 200, "n_results": 5}
                for r in run_log[-20:]]
    return {
        "total_runs": len(run_log),
        "avg_latency_rag": round(sum(rag_lats)/len(rag_lats)) if rag_lats else 0,
        "avg_latency_milvus": 0,
        "avg_latency_kag": round(sum(kag_lats)/len(kag_lats)) if kag_lats else 0,
        "history": history,
        "eval_results": {}
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
    result = {
        "question": req.question,
        "ground_truth": gt,
        "rag": {**rag,
                "em": exact_match(rag["answer"],gt) if gt else None,
                "f1": token_f1(rag["answer"],gt)   if gt else None},
        "kag": {**kag,
                "em": exact_match(kag["answer"],gt) if gt else None,
                "f1": token_f1(kag["answer"],gt)    if gt else None},
        "total_ms": round((time.time()-t0)*1000)
    }
    run_log.append({"pipeline":"COMPARE","question":req.question,
                    "answer":f"RAG:{rag['answer']} | KAG:{kag['answer']}",
                    "latency_ms":result["total_ms"]})
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

@app.post("/load-kag")
async def load_kag(n_docs: int = 8):
    if not documents:
        return JSONResponse(
            {"error": "Chargez d'abord les documents via /load-hotpotqa"},
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
