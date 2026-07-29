"""
evaluation/ragas_evaluator.py
Évaluation LLM-based avec Ragas.
Métriques : faithfulness, answer_relevancy, context_precision, context_recall.
Ces métriques vont bien au-delà du simple Exact Match :
- faithfulness      : la réponse est-elle fidèle au contexte fourni ? (anti-hallucination)
- answer_relevancy  : la réponse répond-elle bien à la question ?
- context_precision : les documents récupérés sont-ils pertinents ?
- context_recall    : tous les faits nécessaires sont-ils dans le contexte ?
"""

import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Ragas avec LangChain-OpenAI pointant sur l'API Groq (compatible OpenAI)
try:
    from langchain_openai import ChatOpenAI
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError as e:
    RAGAS_AVAILABLE = False
    logger.warning(f"Ragas non disponible : {e}")


class RagasEvaluator:
    """
    Évaluateur Ragas utilisant Groq comme LLM juge.
    Évalue les pipelines RAG et KAG selon 4 métriques LLM-based.
    """

    def __init__(self, groq_api_key: Optional[str] = None):
        self.available = RAGAS_AVAILABLE
        if not self.available:
            logger.warning("Ragas non disponible. Installez : uv add ragas langchain-groq")
            return

        api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        try:
            # LangChain-OpenAI configuré pour l'API Groq (compatible OpenAI)
            llm = ChatOpenAI(
                model="llama-3.1-8b-instant",
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
                temperature=0
            )
            self.ragas_llm = LangchainLLMWrapper(llm)

            # Embeddings HuggingFace pour context_precision / recall
            hf_embed = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            self.ragas_embeddings = LangchainEmbeddingsWrapper(hf_embed)

            self.metrics = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ]
            logger.info("RagasEvaluator initialisé avec Groq + HuggingFace embeddings")
        except Exception as e:
            logger.warning(f"Erreur init Ragas: {e}")
            self.available = False

    def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str],
        pipeline_name: str = "RAG"
    ) -> Dict[str, Any]:
        """
        Lance l'évaluation Ragas sur un batch de Q&A.

        Args:
            questions     : liste de questions
            answers       : réponses générées par le pipeline
            contexts      : liste de listes de passages contexte (un par question)
            ground_truths : vérités terrain
            pipeline_name : "RAG" ou "KAG"

        Returns:
            dict avec scores moyens par métrique + détails
        """
        if not self.available:
            return {"error": "Ragas non disponible", "pipeline": pipeline_name}

        if not questions:
            return {"error": "Aucune question fournie"}

        try:
            # Ragas attend un Dataset HuggingFace
            data = {
                "question":    questions,
                "answer":      answers,
                "contexts":    contexts,
                "ground_truth": ground_truths,
            }
            dataset = Dataset.from_dict(data)

            # Configuration des métriques avec notre LLM
            for metric in self.metrics:
                metric.llm = self.ragas_llm
                if hasattr(metric, "embeddings"):
                    metric.embeddings = self.ragas_embeddings

            result = evaluate(dataset, metrics=self.metrics)
            scores = result.to_pandas().mean().to_dict()

            return {
                "pipeline": pipeline_name,
                "n_questions": len(questions),
                "faithfulness":       round(float(scores.get("faithfulness", 0)), 4),
                "answer_relevancy":   round(float(scores.get("answer_relevancy", 0)), 4),
                "context_precision":  round(float(scores.get("context_precision", 0)), 4),
                "context_recall":     round(float(scores.get("context_recall", 0)), 4),
                "avg_score": round(sum([
                    float(scores.get("faithfulness", 0)),
                    float(scores.get("answer_relevancy", 0)),
                    float(scores.get("context_precision", 0)),
                    float(scores.get("context_recall", 0)),
                ]) / 4, 4)
            }

        except Exception as e:
            logger.error(f"Erreur évaluation Ragas: {e}")
            return {"error": str(e), "pipeline": pipeline_name}
