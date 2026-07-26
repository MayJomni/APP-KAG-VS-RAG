"""
kag_server.py — Plateforme unifiée RAG + KAG (Ooredoo AI Lab)
Permet à l'utilisateur d'uploader ses propres documents et de
comparer RAG vs KAG avec métriques MLOps intégrées.
"""

import os, sys, time, logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from kag_pipeline.graph_builder import (
    build_graph_from_docs, get_neo4j_driver, get_graph_summary, InMemoryGraphDriver
)
from kag_pipeline.kag_helper import KAGBase
from rag_pipeline.rag_hotpotqa import RAGHotpotQA
from mlops.tracker import MLOpsTracker
from document_processor import extract_text_from_file, chunk_text, build_documents_from_chunks
from evaluation.metrics import evaluate_batch

load_dotenv()
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ooredoo AI Lab — RAG vs KAG")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── State global ──────────────────────────────────────────────────────────────
driver = get_neo4j_driver()
kag_agent = KAGBase(driver=driver)
rag_agent = RAGHotpotQA()
tracker   = MLOpsTracker(experiment_name="RAG_vs_KAG")

# Stockage en mémoire des documents uploadés et des évaluations
uploaded_docs: list = []
eval_results: dict  = {}


# ── Modèles ───────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str

class CompareRequest(BaseModel):
    question: str

class EvalRequest(BaseModel):
    questions: list[str]
    ground_truths: list[str]
    pipeline: str = "both"   # "rag", "kag", "both"

class HotpotRequest(BaseModel):
    n_samples: int = 5


# ── Serve UI ──────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html = Path(__file__).parent / "kag_app" / "index.html"
    return HTMLResponse(html.read_text(encoding="utf-8"))

@app.get("/health")
async def health():
    return {"status": "ok", "docs_loaded": len(uploaded_docs)}


# ── Stats ─────────────────────────────────────────────────────────────────────
@app.get("/stats")
async def get_stats():
    try:
        kag_s = get_graph_summary(driver)
    except Exception:
        kag_s = {"node_count": 0, "relation_count": 0}
    return {
        "kag":   kag_s,
        "rag":   {"doc_count": len(rag_agent.documents)},
        "mlops": tracker.get_stats(),
        "docs_loaded": len(uploaded_docs)
    }


# ── Upload document ───────────────────────────────────────────────────────────
@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    pipeline: str = Form("both"),
    chunk_size: int = Form(400)
):
    """
    Upload un document (PDF/TXT/MD/DOCX) et le charge dans RAG, KAG ou les deux.
    """
    global uploaded_docs, kag_agent, rag_agent, driver

    content = await file.read()
    filename = file.filename or "document.txt"

    logger.warning(f"Traitement de '{filename}' ({len(content)} octets)...")
    text = extract_text_from_file(filename, content)

    if not text.strip():
        return JSONResponse({"error": "Impossible d'extraire du texte de ce fichier."}, status_code=400)

    chunks = chunk_text(text, chunk_size=chunk_size)
    if not chunks:
        return JSONResponse({"error": "Le document est vide ou trop court."}, status_code=400)

    docs = build_documents_from_chunks(chunks, source_name=filename)
    uploaded_docs = docs  # Remplace les docs précédents

    kag_nodes = kag_rels = rag_docs = 0

    # Charger dans RAG
    if pipeline in ("rag", "both"):
        rag_agent.documents = docs
        from minsearch import Index
        rag_agent.index = Index(text_fields=["title", "text"], keyword_fields=[])
        rag_agent.index.fit(docs)
        rag_docs = len(docs)

    # Charger dans KAG
    if pipeline in ("kag", "both"):
        from kag_pipeline.graph_builder import build_graph_from_docs
        summary = build_graph_from_docs(docs, driver=driver, clear_existing=True)
        kag_agent = KAGBase(driver=driver)
        kag_nodes = summary.get("node_count", 0)
        kag_rels  = summary.get("relation_count", 0)

    return {
        "status":     "success",
        "filename":   filename,
        "n_chunks":   len(chunks),
        "pipeline":   pipeline,
        "rag_docs":   rag_docs,
        "kag_nodes":  kag_nodes,
        "kag_rels":   kag_rels,
        "preview":    chunks[0]["text"][:300] + "..." if chunks else ""
    }


# ── HotpotQA loader (fallback) ────────────────────────────────────────────────
@app.post("/load-hotpotqa")
async def load_hotpotqa_endpoint(req: HotpotRequest):
    """Charge des exemples HotpotQA comme alternative à l'upload."""
    global uploaded_docs, kag_agent, rag_agent, driver
    from data_loader import load_hotpotqa
    from kag_pipeline.graph_builder import build_graph as kg_build

    examples = load_hotpotqa(n_samples=req.n_samples)

    # Build RAG index
    docs = []
    for ex in examples:
        for d in ex.get("documents", []):
            docs.append({"title": d.get("title",""), "text": d.get("text",""), "source": "hotpotqa", "example_id": ex.get("id","")})
    uploaded_docs = docs
    rag_agent.documents = docs
    from minsearch import Index
    rag_agent.index = Index(text_fields=["title","text"], keyword_fields=[])
    rag_agent.index.fit(docs)

    # Build KAG graph
    kag_s = kg_build(examples, driver=driver, clear_existing=True)
    kag_agent = KAGBase(driver=driver)

    return {"status": "success", "n_samples": req.n_samples,
            "rag_docs": len(docs), "kag_nodes": kag_s["node_count"],
            "kag_rels": kag_s["relation_count"]}


# ── Requêtes RAG / KAG / Compare ─────────────────────────────────────────────
@app.post("/rag/query")
async def rag_query(req: QueryRequest):
    t0 = time.time()
    try:
        result = rag_agent.rag(req.question)
        ms = round((time.time()-t0)*1000, 1)
        tracker.track("RAG", req.question, result["answer"], result["context"],
                      ms, len(result["search_results"]))
        return {**result, "latency_ms": ms}
    except Exception as e:
        return {"answer": f"Erreur RAG: {e}", "context": "", "search_results": [], "latency_ms": 0}


@app.post("/kag/query")
async def kag_query(req: QueryRequest):
    t0 = time.time()
    try:
        result = kag_agent.kag(req.question)
        ms = round((time.time()-t0)*1000, 1)
        tracker.track("KAG", req.question, result["answer"], result["context"],
                      ms, len(result.get("search_results", [])))
        return {**result, "latency_ms": ms}
    except Exception as e:
        return {"answer": f"Erreur KAG: {e}", "context": "", "search_results": [], "latency_ms": 0}


@app.post("/compare")
async def compare_query(req: CompareRequest):
    """Lance RAG et KAG en parallèle et retourne les deux résultats."""
    import asyncio

    async def run_rag():
        t0 = time.time()
        try:
            r = rag_agent.rag(req.question)
            ms = round((time.time()-t0)*1000, 1)
            tracker.track("RAG", req.question, r["answer"], r["context"], ms, len(r["search_results"]))
            return {**r, "latency_ms": ms}
        except Exception as e:
            return {"answer": f"Erreur RAG: {e}", "context": "", "search_results": [], "latency_ms": 0}

    async def run_kag():
        t0 = time.time()
        try:
            r = kag_agent.kag(req.question)
            ms = round((time.time()-t0)*1000, 1)
            tracker.track("KAG", req.question, r["answer"], r["context"], ms, len(r.get("search_results", [])))
            return {**r, "latency_ms": ms}
        except Exception as e:
            return {"answer": f"Erreur KAG: {e}", "context": "", "search_results": [], "latency_ms": 0}

    rag_result, kag_result = await asyncio.gather(run_rag(), run_kag())
    return {"rag": rag_result, "kag": kag_result}


# ── Évaluation avec métriques ─────────────────────────────────────────────────
@app.post("/evaluate")
async def evaluate(req: EvalRequest):
    """
    Évalue RAG vs KAG sur une liste de questions avec vérités terrain.
    Retourne EM, F1, et détails par question.
    """
    global eval_results

    results = {}

    if req.pipeline in ("rag", "both"):
        preds_rag = []
        for q in req.questions:
            try:
                r = rag_agent.rag(q)
                preds_rag.append(r["answer"])
            except Exception as e:
                preds_rag.append(f"ERREUR: {e}")
        results["rag"] = evaluate_batch(req.questions, req.ground_truths, preds_rag)

    if req.pipeline in ("kag", "both"):
        preds_kag = []
        for q in req.questions:
            try:
                r = kag_agent.kag(q)
                preds_kag.append(r["answer"])
            except Exception as e:
                preds_kag.append(f"ERREUR: {e}")
        results["kag"] = evaluate_batch(req.questions, req.ground_truths, preds_kag)

    eval_results = results

    # Log dans MLflow
    if tracker.mlflow_enabled:
        try:
            import mlflow
            with mlflow.start_run(run_name="evaluation_batch"):
                for pipe, res in results.items():
                    mlflow.log_metric(f"{pipe}_exact_match", res.get("avg_exact_match", 0))
                    mlflow.log_metric(f"{pipe}_token_f1", res.get("avg_token_f1", 0))
                    mlflow.log_metric(f"{pipe}_accuracy", res.get("accuracy", 0))
        except Exception:
            pass

    return results


@app.get("/mlops/stats")
async def mlops_stats():
    return {**tracker.get_stats(), "eval_results": eval_results}


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("\n" + "="*60)
    print("  [OOREDOO AI LAB] RAG vs KAG Explorer")
    print("  >>> http://localhost:8000")
    print("="*60 + "\n")
    uvicorn.run("kag_server:app", host="0.0.0.0", port=8000, reload=False)
