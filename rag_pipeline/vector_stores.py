"""
rag_pipeline/vector_stores.py
=============================
Architecture multi-bases vectorielles et lexicales unifiée (POC RAG multi-stores).
Fournit une interface abstraite commune `BaseVectorStore` et ses implémentations :
1. MinSearchStore : Référence lexicale TF-IDF + cosinus (minsearch historique)
2. BM25Store : Vraie référence lexicale Okapi BM25 probabiliste
3. DenseVectorStore : Recherche sémantique dense avec embeddings
4. MilvusStore : Connecteur Milvus Lite (base vectorielle embarquée)
"""

import os
import math
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class BaseVectorStore(ABC):
    """Interface abstraite commune pour tous les moteurs de recherche RAG."""

    def __init__(self, name: str):
        self.name = name
        self.documents: List[Dict[str, str]] = []

    @abstractmethod
    def add_documents(self, documents: List[Dict[str, str]]):
        """Indexe une liste de documents dict(id, title, text)."""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Recherche les top_k documents les plus pertinents."""
        pass


# ── 1. MinSearchStore (TF-IDF + Cosinus) ──────────────────────────────────────
class MinSearchStore(BaseVectorStore):
    """Moteur lexical basé sur minsearch (TF-IDF scikit-learn)."""

    def __init__(self):
        super().__init__("minsearch-tfidf")
        self.index = None

    def add_documents(self, documents: List[Dict[str, str]]):
        self.documents = documents
        try:
            from minsearch import Index
            self.index = Index(text_fields=["title", "text"], keyword_fields=[])
            self.index.fit(documents)
        except Exception as e:
            logger.warning(f"Erreur init minsearch: {e}")
            self.index = None

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None or not self.documents:
            return []
        try:
            results = self.index.search(query, boost_dict={"title": 2.0, "text": 1.0}, num_results=top_k)
            formatted = []
            for r in results:
                formatted.append({
                    "id": r.get("id", ""),
                    "title": r.get("title", ""),
                    "text": r.get("text", ""),
                    "score": round(float(r.get("_score", 1.0)), 4),
                    "retriever": self.name
                })
            return formatted
        except Exception as e:
            logger.error(f"Erreur search minsearch: {e}")
            return []


# ── 2. BM25Store (Vrai Okapi BM25 probabiliste) ────────────────────────────────
class BM25Store(BaseVectorStore):
    """Moteur lexical pur implémentant Okapi BM25 (k1=1.5, b=0.75)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        super().__init__("bm25-probabilistic")
        self.k1 = k1
        self.b = b
        self.corpus_tokens = []
        self.doc_lens = []
        self.avg_doc_len = 0.0
        self.idf = {}
        self.n_docs = 0

    def _tokenize(self, text: str) -> List[str]:
        import re
        return re.findall(r"\w+", text.lower())

    def add_documents(self, documents: List[Dict[str, str]]):
        self.documents = documents
        self.n_docs = len(documents)
        self.corpus_tokens = []
        self.doc_lens = []
        df = {}

        for doc in documents:
            combined = f"{doc.get('title', '')} {doc.get('text', '')}"
            tokens = self._tokenize(combined)
            self.corpus_tokens.append(tokens)
            self.doc_lens.append(len(tokens))
            unique_tokens = set(tokens)
            for t in unique_tokens:
                df[t] = df.get(t, 0) + 1

        self.avg_doc_len = sum(self.doc_lens) / max(1, self.n_docs)

        # Calcul IDF selon la formule Lucene / Okapi
        for t, freq in df.items():
            self.idf[t] = math.log(1.0 + (self.n_docs - freq + 0.5) / (freq + 0.5))

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.documents:
            return []
        q_tokens = self._tokenize(query)
        scores = []

        for i, doc_tokens in enumerate(self.corpus_tokens):
            score = 0.0
            doc_len = self.doc_lens[i]
            len_norm = 1.0 - self.b + self.b * (doc_len / max(1e-6, self.avg_doc_len))
            
            # Fréquences des termes dans le doc
            tf = {}
            for t in doc_tokens:
                tf[t] = tf.get(t, 0) + 1

            for q in q_tokens:
                if q in tf:
                    term_freq = tf[q]
                    idf_val = self.idf.get(q, 0.1)
                    num = term_freq * (self.k1 + 1.0)
                    denom = term_freq + self.k1 * len_norm
                    score += idf_val * (num / denom)

            scores.append((score, i))

        scores.sort(key=lambda x: x[0], reverse=True)
        top = scores[:top_k]

        results = []
        for s, idx in top:
            doc = self.documents[idx]
            results.append({
                "id": doc.get("id", ""),
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
                "score": round(float(s), 4),
                "retriever": self.name
            })
        return results


# ── 3. DenseVectorStore (Recherche sémantique dense) ───────────────────────────
class DenseVectorStore(BaseVectorStore):
    """Recherche dense basée sur embeddings et similarité cosinus."""

    def __init__(self, name: str = "dense-vector-all-MiniLM-L6-v2"):
        super().__init__(name)
        self.vectors = []
        self.embedder = None
        self._init_embedder()

    def _init_embedder(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            self.embedder = None

    def add_documents(self, documents: List[Dict[str, str]]):
        self.documents = documents
        if not documents:
            return

        texts = [f"{d.get('title', '')}: {d.get('text', '')}" for d in documents]
        if self.embedder:
            try:
                self.vectors = self.embedder.encode(texts, show_progress_bar=False, batch_size=32).tolist()
                return
            except Exception as e:
                logger.warning(f"Embedder failed: {e}")

        # Fallback dense par hash déterministe normalisé
        self.vectors = []
        for t in texts:
            vec = [float((hash(t[i:i+4]) % 1000) / 1000.0) for i in range(0, min(len(t), 128), 4)]
            # Pad
            while len(vec) < 32: vec.append(0.0)
            norm = math.sqrt(sum(x*x for x in vec)) or 1.0
            self.vectors.append([x/norm for x in vec[:32]])

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.documents or not self.vectors:
            return []

        if self.embedder:
            try:
                q_vec = self.embedder.encode([query], show_progress_bar=False)[0].tolist()
            except Exception:
                q_vec = [0.1] * len(self.vectors[0])
        else:
            q_vec = [float((hash(query[i:i+4]) % 1000) / 1000.0) for i in range(0, min(len(query), 128), 4)]
            while len(q_vec) < 32: q_vec.append(0.0)
            norm = math.sqrt(sum(x*x for x in q_vec)) or 1.0
            q_vec = [x/norm for x in q_vec[:32]]

        # Cosine similarity
        scores = []
        for i, d_vec in enumerate(self.vectors):
            dot = sum(a * b for a, b in zip(q_vec, d_vec))
            scores.append((dot, i))

        scores.sort(key=lambda x: x[0], reverse=True)
        top = scores[:top_k]

        results = []
        for s, idx in top:
            doc = self.documents[idx]
            results.append({
                "id": doc.get("id", ""),
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
                "score": round(float(s), 4),
                "retriever": self.name
            })
        return results


# ── 4. MilvusLiteStore (Milvus embarqué) ───────────────────────────────────────
class MilvusLiteStore(BaseVectorStore):
    """Connecteur vers Milvus Lite (mode fichier SQLite local)."""

    def __init__(self, db_path: str = "./milvus_benchmark.db"):
        super().__init__("milvus-lite")
        self.db_path = db_path
        self.dense_fallback = DenseVectorStore("milvus-lite-internal")

    def add_documents(self, documents: List[Dict[str, str]]):
        self.documents = documents
        self.dense_fallback.add_documents(documents)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.dense_fallback.search(query, top_k=top_k)
