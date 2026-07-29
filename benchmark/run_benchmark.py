"""
benchmark/run_benchmark.py
Protocole de benchmark standardisé — Sujet 03.
Lance une évaluation complète RAG (minsearch) vs RAG (Milvus) vs KAG
sur un jeu de questions avec vérités terrain.
Produit un rapport JSON + affichage console.

Usage:
    uv run python benchmark/run_benchmark.py --questions 5 --output results/benchmark_report.json
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def load_benchmark_data(n_questions: int = 5) -> List[Dict]:
    """Charge les questions HotpotQA avec vérités terrain."""
    from data_loader import load_hotpotqa
    examples = load_hotpotqa(n_samples=n_questions)
    return [
        {
            "id": ex.get("id", f"q{i}"),
            "question": ex.get("question", ""),
            "ground_truth": ex.get("answer", ""),
            "documents": ex.get("documents", [])
        }
        for i, ex in enumerate(examples)
        if ex.get("question") and ex.get("answer")
    ]


def run_rag_minsearch(data: List[Dict], groq_api_key: str) -> Dict[str, Any]:
    """Benchmark du pipeline RAG minsearch (BM25)."""
    from rag_pipeline.rag_hotpotqa import RAGHotpotQA

    # Construction index
    rag = RAGHotpotQA()
    examples_format = [{"id": d["id"], "question": d["question"],
                         "answer": d["ground_truth"], "documents": d["documents"]}
                        for d in data]
    rag.build_index(examples_format)

    questions, answers, contexts, ground_truths = [], [], [], []
    latencies = []

    for item in data:
        t0 = time.time()
        result = rag.rag(item["question"])
        lat = (time.time() - t0) * 1000
        questions.append(item["question"])
        answers.append(result["answer"])
        ground_truths.append(item["ground_truth"])
        contexts.append([r.get("text", "") for r in result.get("search_results", [])])
        latencies.append(round(lat, 1))

    return {"questions": questions, "answers": answers, "contexts": contexts,
            "ground_truths": ground_truths, "latencies": latencies, "pipeline": "RAG-minsearch"}


def run_rag_milvus(data: List[Dict], groq_api_key: str) -> Dict[str, Any]:
    """Benchmark du pipeline RAG Milvus (dense vectors)."""
    try:
        from rag_pipeline.rag_milvus import RAGMilvus
        rag = RAGMilvus()

        docs = []
        for item in data:
            for doc in item["documents"]:
                docs.append({"title": doc.get("title", ""), "text": doc.get("text", ""),
                             "source": "hotpotqa"})
        rag.build_index(docs)

        questions, answers, contexts, ground_truths, latencies = [], [], [], [], []
        for item in data:
            t0 = time.time()
            result = rag.rag(item["question"])
            lat = (time.time() - t0) * 1000
            questions.append(item["question"])
            answers.append(result["answer"])
            ground_truths.append(item["ground_truth"])
            contexts.append([r.get("text", "") for r in result.get("search_results", [])])
            latencies.append(round(lat, 1))

        return {"questions": questions, "answers": answers, "contexts": contexts,
                "ground_truths": ground_truths, "latencies": latencies, "pipeline": "RAG-Milvus"}
    except Exception as e:
        logger.warning(f"RAG Milvus non disponible: {e}")
        return None


def run_kag(data: List[Dict], groq_api_key: str) -> Dict[str, Any]:
    """Benchmark du pipeline KAG."""
    from kag_pipeline.graph_builder import get_neo4j_driver, build_graph
    from kag_pipeline.kag_helper import KAGBase

    driver = get_neo4j_driver()
    examples_format = [{"id": d["id"], "question": d["question"],
                         "answer": d["ground_truth"], "documents": d["documents"]}
                        for d in data]
    build_graph(examples_format, driver=driver, clear_existing=True)
    kag = KAGBase(driver=driver)

    questions, answers, contexts, ground_truths, latencies = [], [], [], [], []
    for item in data:
        t0 = time.time()
        result = kag.kag(item["question"])
        lat = (time.time() - t0) * 1000
        questions.append(item["question"])
        answers.append(result["answer"])
        ground_truths.append(item["ground_truth"])
        contexts.append([f"{r.get('source','')} → {r.get('relation','')} → {r.get('target','')}"
                         for r in result.get("search_results", [])])
        latencies.append(round(lat, 1))

    return {"questions": questions, "answers": answers, "contexts": contexts,
            "ground_truths": ground_truths, "latencies": latencies, "pipeline": "KAG"}


def compute_baseline_metrics(pipeline_result: Dict) -> Dict:
    """Calcule EM + F1 (métriques sans LLM juge)."""
    from evaluation.metrics import evaluate_batch
    result = evaluate_batch(
        pipeline_result["questions"],
        pipeline_result["ground_truths"],
        pipeline_result["answers"]
    )
    result["avg_latency_ms"] = round(
        sum(pipeline_result["latencies"]) / len(pipeline_result["latencies"]), 1
    )
    return result


def compute_ragas_metrics(pipeline_result: Dict) -> Dict:
    """Calcule les métriques Ragas (faithfulness, relevancy, etc.)."""
    try:
        from evaluation.ragas_evaluator import RagasEvaluator
        evaluator = RagasEvaluator()
        if not evaluator.available:
            return {"error": "Ragas non disponible"}
        return evaluator.evaluate(
            pipeline_result["questions"],
            pipeline_result["answers"],
            pipeline_result["contexts"],
            pipeline_result["ground_truths"],
            pipeline_name=pipeline_result["pipeline"]
        )
    except Exception as e:
        return {"error": str(e)}


def generate_recommendations(results: Dict) -> List[str]:
    """Génère des recommandations d'architecture basées sur les résultats."""
    recs = []
    pipelines = {k: v for k, v in results.items() if "baseline" in v}

    # Trouver le meilleur F1
    best_f1 = max(pipelines.items(), key=lambda x: x[1]["baseline"].get("avg_token_f1", 0), default=None)
    # Trouver le plus rapide
    fastest = min(pipelines.items(), key=lambda x: x[1]["baseline"].get("avg_latency_ms", 9999), default=None)

    if best_f1:
        recs.append(f"✅ Meilleure précision (Token F1) : {best_f1[0]} "
                    f"({best_f1[1]['baseline'].get('avg_token_f1', 0)*100:.1f}%)")
    if fastest:
        recs.append(f"⚡ Pipeline le plus rapide : {fastest[0]} "
                    f"({fastest[1]['baseline'].get('avg_latency_ms', 0):.0f}ms/requête)")

    # Recommandations contextuelles
    recs.append("💡 Recommandation : utilisez KAG pour les requêtes multi-hop nécessitant "
                "du raisonnement sur des relations entre entités.")
    recs.append("💡 Recommandation : utilisez RAG-Milvus pour les corpus larges (>10k docs) "
                "grâce à la recherche sémantique dense.")
    recs.append("💡 Recommandation : utilisez RAG-minsearch pour les prototypes rapides "
                "et les corpus <5k documents.")
    return recs


def print_report(report: Dict):
    """Affiche le rapport benchmark en console."""
    print("\n" + "="*65)
    print("  RAPPORT BENCHMARK — RAG vs KAG")
    print(f"  Date : {report['timestamp']}")
    print(f"  Questions testées : {report['n_questions']}")
    print("="*65)

    for pipe_name, pipe_data in report["results"].items():
        b = pipe_data.get("baseline", {})
        r = pipe_data.get("ragas", {})
        print(f"\n{'─'*50}")
        print(f"  {pipe_name}")
        print(f"{'─'*50}")
        print(f"  Exact Match   : {b.get('avg_exact_match', 0)*100:.1f}%")
        print(f"  Token F1      : {b.get('avg_token_f1', 0)*100:.1f}%")
        print(f"  Latence moy.  : {b.get('avg_latency_ms', 0):.0f} ms")
        if not r.get("error"):
            print(f"  Faithfulness  : {r.get('faithfulness', 0)*100:.1f}%")
            print(f"  Ans. Relevancy: {r.get('answer_relevancy', 0)*100:.1f}%")
            print(f"  Ctx Precision : {r.get('context_precision', 0)*100:.1f}%")
            print(f"  Ctx Recall    : {r.get('context_recall', 0)*100:.1f}%")

    print(f"\n{'─'*50}")
    print("  RECOMMANDATIONS")
    print(f"{'─'*50}")
    for rec in report.get("recommendations", []):
        print(f"  {rec}")
    print("="*65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark standardisé RAG vs KAG")
    parser.add_argument("--questions", type=int, default=5, help="Nombre de questions (défaut: 5)")
    parser.add_argument("--output", type=str, default="results/benchmark_report.json")
    parser.add_argument("--skip-ragas", action="store_true", help="Ignorer Ragas (plus rapide)")
    parser.add_argument("--pipelines", nargs="+", default=["rag", "milvus", "kag"],
                        help="Pipelines à tester: rag milvus kag")
    args = parser.parse_args()

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("❌ GROQ_API_KEY manquant dans .env")
        sys.exit(1)

    print(f"\n🚀 Démarrage du benchmark ({args.questions} questions)...")
    data = load_benchmark_data(args.questions)
    print(f"✅ {len(data)} questions chargées depuis HotpotQA")

    report = {
        "timestamp": datetime.now().isoformat(),
        "n_questions": len(data),
        "questions": [d["question"] for d in data],
        "ground_truths": [d["ground_truth"] for d in data],
        "results": {},
        "recommendations": []
    }

    runners = {
        "RAG-minsearch": (run_rag_minsearch, "rag" in args.pipelines),
        "RAG-Milvus":    (run_rag_milvus,    "milvus" in args.pipelines),
        "KAG":           (run_kag,           "kag" in args.pipelines),
    }

    for pipe_name, (runner, enabled) in runners.items():
        if not enabled:
            continue
        print(f"\n⏳ Exécution {pipe_name}...")
        try:
            pipe_result = runner(data, groq_key)
            if pipe_result is None:
                print(f"  ⚠️  {pipe_name} non disponible, ignoré.")
                continue

            baseline = compute_baseline_metrics(pipe_result)
            ragas_scores = {} if args.skip_ragas else compute_ragas_metrics(pipe_result)

            report["results"][pipe_name] = {
                "baseline": baseline,
                "ragas": ragas_scores,
                "answers": pipe_result["answers"],
                "latencies": pipe_result["latencies"]
            }
            print(f"  ✅ {pipe_name} — EM: {baseline.get('avg_exact_match',0)*100:.1f}%"
                  f" | F1: {baseline.get('avg_token_f1',0)*100:.1f}%"
                  f" | {baseline.get('avg_latency_ms',0):.0f}ms")
        except Exception as e:
            print(f"  ❌ Erreur {pipe_name}: {e}")
            report["results"][pipe_name] = {"error": str(e)}

    report["recommendations"] = generate_recommendations(report["results"])

    # Sauvegarde
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print_report(report)
    print(f"📄 Rapport sauvegardé : {out_path.resolve()}")
    return report


if __name__ == "__main__":
    main()
