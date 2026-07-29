"""
rag_pipeline/rag_milvus.py
Pipeline RAG avec Milvus comme base vectorielle (2ème base vectorielle du sujet).
Utilise les embeddings HuggingFace (sentence-transformers/all-MiniLM-L6-v2).
Compare avec minsearch pour montrer la différence entre BM25 et dense retrieval.
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional

from groq import Groq
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Import Milvus (milvus-lite = mode embarqué, pas de serveur requis)
try:
    from pymilvus import MilvusClient
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    logger.warning("pymilvus non installé : uv add milvus-lite")

# Import sentence-transformers pour les embeddings
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    logger.warning("sentence-transformers non installé")

RAG_INSTRUCTIONS = """You are an expert assistant. Answer the question using ONLY the provided context.
If the answer is not in the context, say "I don't know."
Be concise and factual."""

RAG_PROMPT = "QUESTION: {question}\n\nCONTEXT:\n{context}"


class RAGMilvus:
    """
    Pipeline RAG avec Milvus (dense vector search) + HuggingFace embeddings.
    
    Différence vs minsearch (BM25):
    - minsearch = recherche par mots-clés (TF-IDF/BM25) → rapide mais lexicale
    - Milvus    = recherche par similarité sémantique → comprend le sens des phrases
    
    Exemple : "voiture" vs "automobile" → Milvus les reconnaît comme similaires, BM25 non.
    """

    DB_PATH = "./milvus_rag.db"
    COLLECTION = "rag_documents"
    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, llm_client: Optional[Groq] = None,
                 model: str = "llama-3.1-8b-instant",
                 num_results: int = 5):
        self.model = model
        self.num_results = num_results
        self.documents: List[Dict] = []
        self.client: Optional[Any] = None
        self.embedder: Optional[Any] = None
        self.embed_dim: int = 384  # all-MiniLM-L6-v2

        if llm_client is None:
            api_key = os.getenv("GROQ_API_KEY")
            self.llm = Groq(api_key=api_key) if api_key else None
        else:
            self.llm = llm_client

        if not MILVUS_AVAILABLE:
            raise ImportError("Installez pymilvus : uv add milvus-lite")

        if not ST_AVAILABLE:
            raise ImportError("Installez sentence-transformers : uv add sentence-transformers")

        self._init_embedder()
        self._init_milvus()

    def _init_embedder(self):
        """Charge le modèle d'embeddings HuggingFace."""
        logger.info(f"Chargement embeddings : {self.EMBED_MODEL}")
        self.embedder = SentenceTransformer(self.EMBED_MODEL)
        # Test de la dimension
        test = self.embedder.encode(["test"])
        self.embed_dim = len(test[0])
        logger.info(f"Embeddings chargés — dimension : {self.embed_dim}")

    def _init_milvus(self):
        """Initialise Milvus Lite (fichier local, pas de serveur)."""
        self.client = MilvusClient(self.DB_PATH)
        logger.info(f"Milvus Lite initialisé : {self.DB_PATH}")

    def _drop_collection(self):
        """Supprime la collection si elle existe."""
        if self.client.has_collection(self.COLLECTION):
            self.client.drop_collection(self.COLLECTION)

    def build_index(self, documents: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Encode les documents avec HuggingFace et les insère dans Milvus.
        
        Args:
            documents: liste de dicts avec 'title' et 'text'
        
        Returns:
            dict avec nombre de documents indexés
        """
        if not documents:
            return {"doc_count": 0}

        self._drop_collection()
        self.documents = documents

        # Création de la collection Milvus
        self.client.create_collection(
            collection_name=self.COLLECTION,
            dimension=self.embed_dim,
            metric_type="COSINE",
        )

        # Encodage des textes
        texts = [f"{d.get('title', '')} {d.get('text', '')}" for d in documents]
        logger.info(f"Encodage de {len(texts)} documents avec {self.EMBED_MODEL}...")
        vectors = self.embedder.encode(texts, batch_size=32, show_progress_bar=False)

        # Insertion dans Milvus
        data = [
            {
                "id": i,
                "vector": vectors[i].tolist(),
                "title": documents[i].get("title", ""),
                "text": documents[i].get("text", "")[:2000],  # limite 2000 chars
                "source": documents[i].get("source", "")
            }
            for i in range(len(documents))
        ]
        self.client.insert(collection_name=self.COLLECTION, data=data)
        logger.info(f"Milvus index : {len(documents)} documents insérés.")
        return {"doc_count": len(documents)}

    def search(self, query: str) -> List[Dict[str, Any]]:
        """Recherche sémantique dans Milvus."""
        if not self.client or not self.client.has_collection(self.COLLECTION):
            return []
        try:
            q_vec = self.embedder.encode([query])[0].tolist()
            results = self.client.search(
                collection_name=self.COLLECTION,
                data=[q_vec],
                limit=self.num_results,
                output_fields=["title", "text", "source"],
            )
            docs = []
            for hit in results[0]:
                entity = hit.get("entity", hit)
                docs.append({
                    "title":  entity.get("title", ""),
                    "text":   entity.get("text", ""),
                    "source": entity.get("source", ""),
                    "score":  round(float(hit.get("distance", 0)), 4)
                })
            return docs
        except Exception as e:
            logger.error(f"Erreur recherche Milvus: {e}")
            return []

    def build_context(self, results: List[Dict]) -> str:
        if not results:
            return "Aucun document trouvé."
        lines = []
        for i, doc in enumerate(results, 1):
            lines.append(f"[Document {i}] {doc.get('title', '')}")
            lines.append(doc.get("text", ""))
            lines.append("")
        return "\n".join(lines).strip()

    def rag(self, query: str) -> Dict[str, Any]:
        """Pipeline RAG Milvus complet."""
        results = self.search(query)
        context = self.build_context(results)
        prompt = RAG_PROMPT.format(question=query, context=context)

        if not self.llm:
            return {"answer": "GROQ_API_KEY manquant", "context": context, "search_results": results}

        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": RAG_INSTRUCTIONS},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.0
        )
        answer = response.choices[0].message.content
        return {"answer": answer, "context": context, "search_results": results}
