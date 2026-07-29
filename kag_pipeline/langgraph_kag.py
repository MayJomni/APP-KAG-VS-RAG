"""
kag_pipeline/langgraph_kag.py
Pipeline KAG orchestré avec LangGraph.
Implémente un graphe d'état avec nœuds :
  1. Extraction de la question
  2. Recherche dans le graphe de connaissances
  3. Raisonnement multi-hop
  4. Génération de la réponse finale
Montre l'utilisation de LangGraph pour orchestrer le pipeline KAG
comme demandé dans le sujet (stack LangGraph).
"""

import os
import logging
from typing import TypedDict, List, Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning("LangGraph non disponible : uv add langgraph")

try:
    from langchain_openai import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain non disponible")


# ── État du graphe LangGraph ──────────────────────────
class KAGState(TypedDict):
    question: str
    triplets: List[Dict]
    reasoning_steps: List[str]
    answer: str
    latency_ms: int
    pipeline: str


def build_kag_langgraph(kag_agent, groq_api_key: Optional[str] = None):
    """
    Construit le graphe LangGraph pour le pipeline KAG.
    
    Nœuds :
    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────┐
    │  retrieve   │───▶│   reason     │───▶│  generate   │───▶│  END     │
    └─────────────┘    └──────────────┘    └─────────────┘    └──────────┘
    """
    if not LANGGRAPH_AVAILABLE:
        return None

    api_key = groq_api_key or os.getenv("GROQ_API_KEY")

    if LANGCHAIN_AVAILABLE and api_key:
        llm = ChatOpenAI(
            model="llama-3.1-8b-instant",
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=0
        )
    else:
        llm = None

    # ── Nœud 1 : Retrieval du graphe KAG ──────────────
    def retrieve_node(state: KAGState) -> KAGState:
        """Cherche les triplets pertinents dans le graphe de connaissances."""
        results = kag_agent.search(state["question"])
        state["triplets"] = results
        state["reasoning_steps"] = [
            f"Recherche KAG: {len(results)} triplets trouvés pour '{state['question'][:60]}...'"
        ]
        return state

    # ── Nœud 2 : Raisonnement multi-hop ──────────────
    def reasoning_node(state: KAGState) -> KAGState:
        """Identifie les chaînes de raisonnement dans les triplets."""
        triplets = state["triplets"]
        steps = list(state["reasoning_steps"])

        if not triplets:
            steps.append("Aucun triplet trouvé — réponse directe.")
        else:
            # Construire un résumé des relations trouvées
            entities = set()
            relations = []
            for t in triplets[:10]:
                entities.add(t.get("source", ""))
                entities.add(t.get("target", ""))
                relations.append(f"{t.get('source','')} → {t.get('relation','')} → {t.get('target','')}")

            steps.append(f"Entités trouvées: {', '.join(list(entities)[:5])}")
            steps.append(f"Relations clés: {' | '.join(relations[:3])}")

        state["reasoning_steps"] = steps
        return state

    # ── Nœud 3 : Génération de la réponse ──────────────
    def generate_node(state: KAGState) -> KAGState:
        """Génère la réponse finale basée sur les triplets et le raisonnement."""
        triplets = state["triplets"]
        reasoning = state["reasoning_steps"]

        context_parts = []
        for t in triplets[:15]:
            context_parts.append(
                f"{t.get('source', '')} → [{t.get('relation', '')}] → {t.get('target', '')}"
            )
        context = "\n".join(context_parts) if context_parts else "Aucune information trouvée."

        reasoning_text = "\n".join(f"  • {s}" for s in reasoning)

        if llm:
            prompt = f"""Vous avez effectué une recherche dans un graphe de connaissances.

Question : {state['question']}

Triplets du graphe de connaissances :
{context}

Étapes de raisonnement :
{reasoning_text}

Répondez à la question de manière concise et factuelle en vous basant sur les triplets.
Si l'information n'est pas dans le graphe, dites "Je ne sais pas"."""

            try:
                response = llm.invoke([HumanMessage(content=prompt)])
                state["answer"] = response.content
            except Exception as e:
                # Fallback: réponse simple basée sur les triplets
                state["answer"] = _fallback_answer(state["question"], triplets)
        else:
            state["answer"] = _fallback_answer(state["question"], triplets)

        state["pipeline"] = "KAG-LangGraph"
        return state

    def _fallback_answer(question: str, triplets: List[Dict]) -> str:
        if not triplets:
            return "Aucune information trouvée dans le graphe."
        top = triplets[0]
        return f"Basé sur le graphe : {top.get('source','')} {top.get('relation','')} {top.get('target','')}."

    # ── Construction du graphe ──────────────────────────
    if not LANGGRAPH_AVAILABLE:
        return None

    graph = StateGraph(KAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "reasoning")
    graph.add_edge("reasoning", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


class KAGLangGraph:
    """
    Wrapper pour le pipeline KAG orchestré avec LangGraph.
    Expose la même interface que KAGBase (.kag() method).
    """

    def __init__(self, kag_agent, groq_api_key: Optional[str] = None):
        self.kag_agent = kag_agent
        self.graph = build_kag_langgraph(kag_agent, groq_api_key)
        self.available = self.graph is not None

    def kag(self, question: str) -> Dict[str, Any]:
        """Lance le pipeline KAG LangGraph."""
        import time
        t0 = time.time()

        if not self.available:
            # Fallback vers le KAG basique
            result = self.kag_agent.kag(question)
            result["pipeline"] = "KAG-basic (LangGraph non disponible)"
            return result

        initial_state = KAGState(
            question=question,
            triplets=[],
            reasoning_steps=[],
            answer="",
            latency_ms=0,
            pipeline="KAG-LangGraph"
        )

        try:
            final_state = self.graph.invoke(initial_state)
            lat = round((time.time() - t0) * 1000)
            return {
                "answer": final_state["answer"],
                "search_results": final_state["triplets"],
                "reasoning_steps": final_state["reasoning_steps"],
                "latency_ms": lat,
                "pipeline": "KAG-LangGraph"
            }
        except Exception as e:
            logger.error(f"Erreur KAG LangGraph: {e}")
            result = self.kag_agent.kag(question)
            result["pipeline"] = "KAG-fallback"
            return result
