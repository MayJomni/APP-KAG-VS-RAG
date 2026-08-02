"""
rag_pipeline/rag_hotpotqa.py
Pipeline RAG adapté pour HotpotQA en utilisant minsearch comme moteur de recherche.
Compatible avec la même interface que KAGBase pour faciliter la comparaison.
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional

from minsearch import Index
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAG_INSTRUCTIONS = """You are an expert assistant that answers questions based on provided document context.
Use ONLY the information in the CONTEXT section to answer.
If the answer is not found in the context, respond with "I don't know."
Be concise and factual.
"""

RAG_PROMPT_TEMPLATE = """QUESTION: {question}

CONTEXT:
{context}
""".strip()


class RAGHotpotQA:
    """
    Pipeline RAG pour HotpotQA.
    Utilise minsearch (recherche BM25 en mémoire) pour retrouver les documents pertinents
    et Groq LLM pour générer la réponse.
    Même interface que KAGBase pour une comparaison directe.
    """

    def __init__(self, llm_client: Optional[Groq] = None,
                 instructions: str = RAG_INSTRUCTIONS,
                 prompt_template: str = RAG_PROMPT_TEMPLATE,
                 model: str = "llama-3.1-8b-instant",
                 num_results: int = 5):
        if llm_client is None:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY manquant dans les variables d'environnement.")
            self.llm_client = Groq(api_key=api_key)
        else:
            self.llm_client = llm_client

        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model
        self.num_results = num_results
        self.index = None
        self.documents = []

    def build_index(self, examples: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Construit l'index minsearch à partir des exemples HotpotQA.
        Chaque document du corpus devient un document de l'index.
        """
        self.documents = []
        seen = set()

        for example in examples:
            q_id = example.get("id", "")
            for doc in example.get("documents", []):
                title = doc.get("title", "")
                text = doc.get("text", "")
                key = f"{title}:{text[:50]}"
                if key in seen:
                    continue
                seen.add(key)
                self.documents.append({
                    "title": title,
                    "text": text,
                    "example_id": q_id
                })

        # Construction de l'index minsearch
        self.index = Index(
            text_fields=["title", "text"],
            keyword_fields=[]
        )
        self.index.fit(self.documents)

        logger.info(f"Index RAG construit avec {len(self.documents)} documents.")
        return {"doc_count": len(self.documents)}

    def search(self, query: str) -> List[Dict[str, Any]]:
        """Recherche les documents pertinents via minsearch."""
        if self.index is None:
            logger.warning("Index RAG non construit. Retour d'une liste vide.")
            return []
        try:
            boost = {"title": 2.0, "text": 1.0}
            results = self.index.search(query, boost_dict=boost, num_results=self.num_results)
            return results
        except Exception as e:
            logger.error(f"Erreur lors de la recherche RAG: {e}")
            return []

    def build_context(self, search_results: List[Dict[str, Any]]) -> str:
        """Formate les documents retrouvés en un contexte textuel pour le LLM."""
        if not search_results:
            return "Aucun document trouvé."
        lines = []
        for i, doc in enumerate(search_results, 1):
            lines.append(f"[Document {i}] {doc.get('title', '')}")
            lines.append(doc.get('text', ''))
            lines.append("")
        return "\n".join(lines).strip()

    def build_prompt(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """Construit le prompt final."""
        context = self.build_context(search_results)
        return self.prompt_template.format(question=query, context=context)

    def llm(self, prompt: str) -> str:
        """Appelle le LLM Groq et retourne la réponse."""
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.instructions},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content

    def rag(self, query: str) -> Dict[str, Any]:
        """
        Exécute le pipeline RAG complet.
        Retourne un dictionnaire compatible avec kag() de KAGBase.
        """
        search_results = self.search(query)
        context = self.build_context(search_results)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return {
            "answer": answer,
            "context": context,
            "search_results": search_results
        }
