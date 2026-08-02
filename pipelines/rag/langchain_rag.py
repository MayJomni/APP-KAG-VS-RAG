"""
rag_pipeline/langchain_rag.py
Pipeline RAG avec LangChain (LCEL — LangChain Expression Language).
Implémente la chaîne : retrieve → format_docs → prompt → llm → parse
comme demandé dans le sujet (stack LangChain).
Utilise LlamaIndex-style retriever wrappé dans LangChain.
"""

import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    from langchain_openai import ChatOpenAI
    from langchain.schema import Document
    from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema.output_parser import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain non disponible : uv add langchain langchain-openai")


RAG_PROMPT_TEMPLATE = """You are an expert assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""


class RAGLangChain:
    """
    Pipeline RAG implémenté avec LangChain LCEL.
    
    Chaîne LCEL :
    retriever | format_docs | prompt | llm | StrOutputParser
    
    Différence vs RAGHotpotQA :
    - RAGHotpotQA = implémentation maison avec minsearch
    - RAGLangChain = implémentation standard LangChain (production-ready)
    """

    def __init__(self, rag_base_agent, groq_api_key: Optional[str] = None):
        """
        Args:
            rag_base_agent : instance de RAGHotpotQA ou RAGMilvus
                             qui expose une méthode .search(query) -> List[Dict]
        """
        self.rag_agent = rag_base_agent
        self.available = LANGCHAIN_AVAILABLE

        if not LANGCHAIN_AVAILABLE:
            return

        api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.llm = ChatOpenAI(
            model="llama-3.1-8b-instant",
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=0
        )

        self.prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

        # Construction de la chaîne LCEL
        self.chain = (
            {
                "context": RunnableLambda(self._retrieve_and_format),
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        self._last_results = []
        logger.info("RAGLangChain initialisé avec LangChain LCEL + Groq")

    def _retrieve_and_format(self, question: str) -> str:
        """Récupère les documents et les formate pour le prompt."""
        results = self.rag_agent.search(question)
        self._last_results = results
        docs = []
        for i, doc in enumerate(results, 1):
            docs.append(f"[{i}] {doc.get('title', '')}\n{doc.get('text', '')}")
        return "\n\n".join(docs) if docs else "Aucun document trouvé."

    def rag(self, question: str) -> Dict[str, Any]:
        """Lance le pipeline RAG LangChain."""
        import time
        t0 = time.time()

        if not self.available:
            # Fallback vers le RAG basique
            result = self.rag_agent.rag(question)
            result["pipeline"] = "RAG-basic (LangChain non disponible)"
            return result

        try:
            answer = self.chain.invoke(question)
            lat = round((time.time() - t0) * 1000)
            return {
                "answer": answer,
                "search_results": self._last_results,
                "latency_ms": lat,
                "pipeline": "RAG-LangChain"
            }
        except Exception as e:
            logger.error(f"Erreur RAG LangChain: {e}")
            result = self.rag_agent.rag(question)
            result["pipeline"] = "RAG-fallback"
            return result
