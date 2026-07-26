"""
evaluation/metrics.py
Métriques d'évaluation standard pour comparer RAG vs KAG sur HotpotQA.

- Exact Match (EM) : la réponse contient-elle exactement la vérité terrain ?
- Token F1         : chevauchement de tokens entre réponse prédite et vérité terrain
- Ragas-style      : faithfulness et answer_relevancy (via Groq, optionnel)

Basé sur le script officiel HotpotQA : https://hotpotqa.github.io/
"""

import re
import string
import logging
from typing import Dict, List, Optional, Any
from collections import Counter

logger = logging.getLogger(__name__)


# ── Nettoyage du texte ────────────────────────────────────────────────────────

def normalize_answer(text: str) -> str:
    """Normalise une réponse : minuscules, sans ponctuation, sans articles."""
    def remove_articles(t):
        return re.sub(r'\b(a|an|the)\b', ' ', t)

    def white_space_fix(t):
        return ' '.join(t.split())

    def remove_punc(t):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in t if ch not in exclude)

    def lower(t):
        return t.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def get_tokens(text: str) -> List[str]:
    """Tokenise un texte normalisé."""
    return normalize_answer(text).split()


# ── Métriques de base ─────────────────────────────────────────────────────────

def exact_match(prediction: str, ground_truth: str) -> float:
    """
    Exact Match : 1.0 si les réponses normalisées sont identiques, 0.0 sinon.
    Version souple : vérifie aussi si la vérité est contenue dans la prédiction.
    """
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)
    if pred_norm == gt_norm:
        return 1.0
    # Version souple : la réponse contient-elle la vérité ?
    if gt_norm and gt_norm in pred_norm:
        return 1.0
    return 0.0


def token_f1(prediction: str, ground_truth: str) -> float:
    """
    Token F1 : chevauchement de tokens entre prédiction et vérité terrain.
    Métrique principale utilisée dans le leaderboard HotpotQA.
    """
    pred_tokens = get_tokens(prediction)
    gt_tokens = get_tokens(ground_truth)

    if not pred_tokens or not gt_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return round(f1, 4)


# ── Évaluation d'un batch ─────────────────────────────────────────────────────

def evaluate_batch(
    questions: List[str],
    ground_truths: List[str],
    predictions: List[str],
    contexts: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Évalue un batch de prédictions RAG ou KAG.

    Returns:
        dict avec EM moyen, F1 moyen, et détails par question
    """
    if not questions or len(questions) != len(predictions) or len(questions) != len(ground_truths):
        return {"error": "Listes de longueurs incompatibles"}

    results = []
    for i, (q, gt, pred) in enumerate(zip(questions, ground_truths, predictions)):
        em = exact_match(pred, gt)
        f1 = token_f1(pred, gt)
        results.append({
            "question": q,
            "ground_truth": gt,
            "prediction": pred[:200] + "..." if len(pred) > 200 else pred,
            "exact_match": em,
            "token_f1": f1,
            "correct": em == 1.0
        })

    avg_em = round(sum(r["exact_match"] for r in results) / len(results), 4)
    avg_f1 = round(sum(r["token_f1"] for r in results) / len(results), 4)
    n_correct = sum(1 for r in results if r["correct"])

    return {
        "n_questions": len(results),
        "n_correct": n_correct,
        "accuracy": round(n_correct / len(results), 4),
        "avg_exact_match": avg_em,
        "avg_token_f1": avg_f1,
        "details": results
    }
