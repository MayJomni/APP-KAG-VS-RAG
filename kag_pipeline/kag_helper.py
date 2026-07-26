import os
import logging
from typing import List, Dict, Any, Optional
from groq import Groq
from neo4j import Driver
from dotenv import load_dotenv
from .graph_query import query_graph
from .graph_builder import get_neo4j_driver

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KAG_INSTRUCTIONS = """Vous êtes un assistant basé sur un graphe de connaissances (Knowledge-Augmented Generation / KAG).
Votre rôle est de répondre aux questions uniquement à partir des faits extraits du graphe sous forme de triplets (Sujet -- Relation --> Objet).

Règles de réponse :
1. Utilisez exclusivement les faits fournis dans le CONTEXTE sous forme de triplets.
2. Si les faits dans le contexte ne permettent pas de répondre avec certitude, répondez "I don't know."
3. Soyez concis, précis et factuel.
"""

KAG_PROMPT_TEMPLATE = """QUESTION: {question}

CONTEXTE (Faits du graphe de connaissances) :
{context}
""".strip()

class KAGBase:
    """
    Classe représentant l'orchestrateur du pipeline KAG.
    Conserve la même interface que RAGBase pour faciliter la comparaison.
    """

    def __init__(self, driver: Optional[Driver] = None, llm_client: Optional[Groq] = None,
                 instructions: str = KAG_INSTRUCTIONS, prompt_template: str = KAG_PROMPT_TEMPLATE,
                 model: str = "llama-3.1-8b-instant"):
        """
        Initialise l'instance KAGBase.
        """
        self.driver = driver if driver is not None else get_neo4j_driver()
        
        if llm_client is None:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("La variable d'environnement GROQ_API_KEY est manquante.")
            self.llm_client = Groq(api_key=api_key)
        else:
            self.llm_client = llm_client

        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query: str) -> List[Dict[str, str]]:
        """
        Interroge le graphe Neo4j pour extraire les triplets pertinents.
        """
        return query_graph(query, driver=self.driver)

    def build_context(self, search_results: List[Dict[str, str]]) -> str:
        """
        Transforme la liste de triplets (search_results) en un texte lisible pour le LLM.
        """
        if not search_results:
            return "Aucun fait trouvé dans le graphe."
            
        lines = []
        for triplet in search_results:
            src = triplet.get("source", "")
            rel = triplet.get("relation", "")
            tgt = triplet.get("target", "")
            lines.append(f"- {src} --[{rel}]--> {tgt}")
            
        return "\n".join(lines).strip()

    def build_prompt(self, query: str, search_results: List[Dict[str, str]]) -> str:
        """
        Construit le prompt utilisateur final en incorporant le contexte structuré du graphe.
        """
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )

    def llm(self, prompt: str) -> str:
        """
        Envoie le prompt au LLM Groq et retourne la réponse texte générée.
        """
        input_messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=input_messages,
            temperature=0.0
        )
        return response.choices[0].message.content

    def kag(self, query: str) -> Dict[str, Any]:
        """
        Exécute le pipeline KAG complet (search -> build_prompt -> llm -> answer).
        
        Returns:
            Dict[str, Any]: {"answer": answer_text, "context": context_text, "search_results": search_results}
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
