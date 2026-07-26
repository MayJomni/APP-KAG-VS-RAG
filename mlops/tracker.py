"""
mlops/tracker.py
Module MLOps : tracking des expérimentations RAG vs KAG avec MLflow.
Enregistre automatiquement latence, tokens estimés, pipeline utilisé et réponse.
"""

import os
import time
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Tentative d'import MLflow (optionnel)
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow non installé. Le tracking sera désactivé.")

# Tentative d'import LangFuse (optionnel)
try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False


class MLOpsTracker:
    """
    Tracker MLOps unifié pour RAG et KAG.
    - MLflow : tracking des runs, paramètres et métriques
    - LangFuse : observabilité des traces LLM (optionnel)
    """

    def __init__(self, experiment_name: str = "RAG_vs_KAG",
                 mlflow_tracking_uri: str = "./mlruns",
                 langfuse_public_key: Optional[str] = None,
                 langfuse_secret_key: Optional[str] = None):

        self.experiment_name = experiment_name
        self.mlflow_enabled = MLFLOW_AVAILABLE
        self.langfuse_enabled = False
        self.run_history = []  # Historique en mémoire pour l'interface

        # Initialisation MLflow
        if self.mlflow_enabled:
            try:
                os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
                mlflow.set_tracking_uri(mlflow_tracking_uri)
                mlflow.set_experiment(experiment_name)
                logger.info(f"MLflow initialisé — Expérience : {experiment_name}")
            except Exception as e:
                logger.warning(f"MLflow non disponible : {e}")
                self.mlflow_enabled = False

        # Initialisation LangFuse (optionnel)
        pk = langfuse_public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        sk = langfuse_secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        if LANGFUSE_AVAILABLE and pk and sk:
            try:
                self.langfuse = Langfuse(public_key=pk, secret_key=sk)
                self.langfuse_enabled = True
                logger.info("LangFuse initialisé avec succès.")
            except Exception as e:
                logger.warning(f"LangFuse non disponible : {e}")

    def track(self, pipeline: str, question: str, answer: str,
              context: str, latency_ms: float,
              n_results: int = 0, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Enregistre un run dans MLflow et LangFuse.

        Args:
            pipeline: "RAG" ou "KAG"
            question: Question posée
            answer: Réponse générée
            context: Contexte utilisé (documents ou triplets)
            latency_ms: Temps de réponse en millisecondes
            n_results: Nombre de résultats retournés (documents ou triplets)
            params: Paramètres supplémentaires (top_k, model, etc.)
        """
        # Estimation naïve du nombre de tokens
        estimated_tokens = (len(question) + len(context) + len(answer)) // 4

        run_data = {
            "pipeline": pipeline,
            "question": question[:100] + "..." if len(question) > 100 else question,
            "answer": answer[:150] + "..." if len(answer) > 150 else answer,
            "latency_ms": round(latency_ms, 2),
            "n_results": n_results,
            "estimated_tokens": estimated_tokens,
        }

        # Sauvegarde en mémoire pour l'API
        self.run_history.append(run_data)
        if len(self.run_history) > 100:
            self.run_history.pop(0)

        # Tracking MLflow
        if self.mlflow_enabled:
            try:
                with mlflow.start_run(run_name=f"{pipeline}_{int(time.time())}"):
                    mlflow.log_param("pipeline", pipeline)
                    mlflow.log_param("question_preview", question[:80])
                    mlflow.log_param("model", params.get("model", "llama-3.1-8b-instant") if params else "llama-3.1-8b-instant")
                    if params:
                        for k, v in params.items():
                            mlflow.log_param(k, v)
                    mlflow.log_metric("latency_ms", latency_ms)
                    mlflow.log_metric("estimated_tokens", estimated_tokens)
                    mlflow.log_metric("n_results", n_results)
            except Exception as e:
                logger.warning(f"Erreur MLflow track : {e}")

        # Tracking LangFuse
        if self.langfuse_enabled:
            try:
                trace = self.langfuse.trace(
                    name=f"{pipeline.lower()}_query",
                    input={"question": question},
                    output={"answer": answer},
                    metadata={
                        "pipeline": pipeline,
                        "latency_ms": latency_ms,
                        "estimated_tokens": estimated_tokens
                    }
                )
            except Exception as e:
                logger.warning(f"Erreur LangFuse track : {e}")

        return run_data

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques agrégées des runs pour l'interface."""
        if not self.run_history:
            return {"total_runs": 0, "rag_runs": 0, "kag_runs": 0,
                    "avg_latency_rag": 0, "avg_latency_kag": 0,
                    "avg_tokens_rag": 0, "avg_tokens_kag": 0,
                    "history": []}

        rag_runs = [r for r in self.run_history if r["pipeline"] == "RAG"]
        kag_runs = [r for r in self.run_history if r["pipeline"] == "KAG"]

        def safe_avg(lst, key):
            return round(sum(r[key] for r in lst) / len(lst), 1) if lst else 0

        return {
            "total_runs": len(self.run_history),
            "rag_runs": len(rag_runs),
            "kag_runs": len(kag_runs),
            "avg_latency_rag": safe_avg(rag_runs, "latency_ms"),
            "avg_latency_kag": safe_avg(kag_runs, "latency_ms"),
            "avg_tokens_rag": safe_avg(rag_runs, "estimated_tokens"),
            "avg_tokens_kag": safe_avg(kag_runs, "estimated_tokens"),
            "history": list(reversed(self.run_history[-20:]))
        }
