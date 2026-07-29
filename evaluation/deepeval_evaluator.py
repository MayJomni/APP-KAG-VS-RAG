"""
evaluation/deepeval_evaluator.py
Évaluation avec DeepEval.
Métriques : GEval (correctness), Hallucination, Contextual Relevancy.
DeepEval utilise aussi un LLM juge (Groq) pour scorer les réponses.
"""

import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    from deepeval import evaluate as deval_evaluate
    from deepeval.metrics import (
        GEval,
        HallucinationMetric,
        ContextualRelevancyMetric,
    )
    from deepeval.models.gpt_model import GPTModel
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    DEEPEVAL_AVAILABLE = True
except ImportError as e:
    DEEPEVAL_AVAILABLE = False
    logger.warning(f"DeepEval non disponible : {e}")


def _safe_score(metric_obj) -> float:
    """Extrait le score de façon sécurisée."""
    try:
        return round(float(metric_obj.score or 0), 4)
    except Exception:
        return 0.0


class DeepEvalEvaluator:
    """
    Évaluateur DeepEval avec métriques LLM-based.
    Utilise Groq (via API OpenAI-compatible) comme juge.
    """

    def __init__(self, groq_api_key: Optional[str] = None):
        self.available = DEEPEVAL_AVAILABLE
        if not self.available:
            logger.warning("DeepEval non disponible. Installez : uv add deepeval")
            return

        api_key = groq_api_key or os.getenv("GROQ_API_KEY")

        # Configuration de DeepEval pour utiliser Groq
        os.environ["OPENAI_API_KEY"] = api_key or "dummy"
        os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"

        try:
            self.correctness_metric = GEval(
                name="Correctness",
                criteria="The answer correctly addresses the question based on the ground truth.",
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.EXPECTED_OUTPUT,
                ],
                model="llama-3.1-8b-instant",
            )
            self.hallucination_metric = HallucinationMetric(
                threshold=0.5,
                model="llama-3.1-8b-instant",
            )
            self.relevancy_metric = ContextualRelevancyMetric(
                threshold=0.5,
                model="llama-3.1-8b-instant",
            )
            logger.info("DeepEvalEvaluator initialisé avec Groq")
        except Exception as e:
            logger.warning(f"Erreur init DeepEval: {e}")
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
        Évalue un batch avec DeepEval.
        Retourne correctness, hallucination_score, contextual_relevancy.
        """
        if not self.available:
            return {"error": "DeepEval non disponible", "pipeline": pipeline_name}

        correctness_scores = []
        hallucination_scores = []
        relevancy_scores = []

        for q, ans, ctx, gt in zip(questions, answers, contexts, ground_truths):
            ctx_str = ctx if isinstance(ctx, list) else [ctx]
            try:
                test_case = LLMTestCase(
                    input=q,
                    actual_output=ans,
                    expected_output=gt,
                    context=ctx_str,
                    retrieval_context=ctx_str,
                )

                # Correctness
                try:
                    self.correctness_metric.measure(test_case)
                    correctness_scores.append(_safe_score(self.correctness_metric))
                except Exception:
                    correctness_scores.append(0.0)

                # Hallucination
                try:
                    self.hallucination_metric.measure(test_case)
                    # Score hallu : 1 - hallucination (1 = pas d'hallucination)
                    h = _safe_score(self.hallucination_metric)
                    hallucination_scores.append(round(1 - h, 4))
                except Exception:
                    hallucination_scores.append(0.0)

                # Contextual relevancy
                try:
                    self.relevancy_metric.measure(test_case)
                    relevancy_scores.append(_safe_score(self.relevancy_metric))
                except Exception:
                    relevancy_scores.append(0.0)

            except Exception as e:
                logger.warning(f"Erreur DeepEval test case: {e}")
                correctness_scores.append(0.0)
                hallucination_scores.append(0.0)
                relevancy_scores.append(0.0)

        def avg(lst):
            return round(sum(lst) / len(lst), 4) if lst else 0.0

        return {
            "pipeline": pipeline_name,
            "n_questions": len(questions),
            "correctness":          avg(correctness_scores),
            "anti_hallucination":   avg(hallucination_scores),
            "contextual_relevancy": avg(relevancy_scores),
            "avg_score": avg([
                avg(correctness_scores),
                avg(hallucination_scores),
                avg(relevancy_scores),
            ])
        }
