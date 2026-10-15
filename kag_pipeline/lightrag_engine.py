"""
kag_pipeline/lightrag_engine.py
===============================
Implémentation complète de LightRAG avec 5 modes de requêtage :
1. 'naive'  : Recherche textuelle directe (sans graphe)
2. 'local'  : Exploration du voisinage local d'entités (1-hop / 2-hop)
3. 'global' : Synthèse globale des hubs et relations structurelles clés
4. 'hybrid' : Fusion locale (faits précis) + globale (thèmes relationnels)
5. 'mix'    : Fusion complète (local + global + chunks textuels bruts)

Comprend l'exportation au format standard GraphML (.graphml).
"""

import os
import re
import json
import math
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class LightRAGEngine:
    """Moteur LightRAG autonome avec support multi-modes et persistance GraphML."""

    def __init__(self, name: str = "LightRAG-v1"):
        self.name = name
        self.chunks: List[Dict[str, str]] = []
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Dict[str, Any]] = []
        self.communities: Dict[str, List[str]] = {}

    def build_from_documents(self, documents: List[Dict[str, str]]):
        """Indexe les documents et construit le graphe de connaissances LightRAG."""
        self.chunks = documents
        self.nodes.clear()
        self.edges.clear()

        # 1. Extraction d'entités et de relations (heuristique & lexicale déterministe)
        for doc in documents:
            title = doc.get("title", "Doc")
            text = doc.get("text", "")
            doc_id = doc.get("id", title)

            # Identification d'entités candidates (titres et noms propres)
            words = text.split()
            candidates = set()
            if title:
                candidates.add(title.strip())

            # Noms propres par regex
            proper_nouns = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
            for pn in proper_nouns:
                if len(pn) > 2 and pn.lower() not in {"the", "this", "that", "these", "there", "when", "what", "where"}:
                    candidates.add(pn)

            # Ajout des nœuds
            for c in candidates:
                if c not in self.nodes:
                    self.nodes[c] = {
                        "name": c,
                        "degree": 0,
                        "source_docs": {doc_id},
                        "type": "Entity"
                    }
                else:
                    self.nodes[c]["source_docs"].add(doc_id)

            # Relations de co-occurrence document/phrase
            cand_list = list(candidates)
            for i in range(len(cand_list)):
                for j in range(i + 1, min(len(cand_list), i + 6)):
                    src, tgt = cand_list[i], cand_list[j]
                    self.nodes[src]["degree"] += 1
                    self.nodes[tgt]["degree"] += 1
                    self.edges.append({
                        "source": src,
                        "relation": "associated_with",
                        "target": tgt,
                        "doc_id": doc_id
                    })

        # 2. Détection de communautés (clusters par hubs de haut degré)
        hubs = sorted(self.nodes.values(), key=lambda x: x["degree"], reverse=True)[:10]
        self.communities = {
            f"community_{i}": [h["name"]] for i, h in enumerate(hubs)
        }
        for edge in self.edges:
            src, tgt = edge["source"], edge["target"]
            for comm_id, members in self.communities.items():
                if members and (src == members[0] or tgt == members[0]):
                    if src not in members and len(members) < 8:
                        members.append(src)
                    if tgt not in members and len(members) < 8:
                        members.append(tgt)

    def export_graphml(self, output_path: str = "data/lightrag_graph.graphml"):
        """Exporte le graphe au format standard GraphML (XML)."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        xml = ['<?xml version="1.0" encoding="UTF-8"?>',
               '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
               '  <key id="d0" for="node" attr.name="name" attr.type="string"/>',
               '  <key id="d1" for="node" attr.name="degree" attr.type="int"/>',
               '  <key id="d2" for="edge" attr.name="relation" attr.type="string"/>',
               '  <graph id="LightRAG" edgedefault="undirected">']

        for idx, (name, data) in enumerate(self.nodes.items()):
            clean_name = name.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            xml.append(f'    <node id="n{idx}">')
            xml.append(f'      <data key="d0">{clean_name}</data>')
            xml.append(f'      <data key="d1">{data.get("degree", 0)}</data>')
            xml.append('    </node>')

        node_keys = {name: f"n{i}" for i, name in enumerate(self.nodes.keys())}
        for e_idx, edge in enumerate(self.edges[:500]):  # limite 500 arêtes pour propreté
            src_id = node_keys.get(edge["source"])
            tgt_id = node_keys.get(edge["target"])
            if src_id and tgt_id:
                rel = edge.get("relation", "related")
                xml.append(f'    <edge id="e{e_idx}" source="{src_id}" target="{tgt_id}">')
                xml.append(f'      <data key="d2">{rel}</data>')
                xml.append('    </edge>')

        xml.append('  </graph>')
        xml.append('</graphml>')

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(xml))

    # ── Méthodes d'insertion incrémentale ────────────────────────────────────

    def insert(self, text: str, title: str = "Document"):
        """Insère un texte brut dans LightRAG (découpe + indexation automatique).
        Appelée pour chaque document uploadé par l'utilisateur.
        Gère la déduplication par (title, hash) pour éviter les doublons.
        """
        import hashlib
        doc_id = hashlib.md5(f"{title}:{text[:120]}".encode()).hexdigest()[:10]

        # Déduplication : ne pas ré-indexer un doc déjà présent
        already = any(c.get("id") == doc_id for c in self.chunks)
        if already:
            return

        # Découpe en paragraphes / chunks de ~300 mots
        words = text.split()
        chunk_size = 300
        overlap = 40
        step = max(1, chunk_size - overlap)
        new_chunks = []
        for i in range(0, len(words), step):
            chunk_words = words[i: i + chunk_size]
            if len(chunk_words) < 15:
                continue
            new_chunks.append({
                "title": title,
                "text":  " ".join(chunk_words),
                "id":    f"{doc_id}_c{i}"
            })

        if not new_chunks:
            return

        # Indexer les nouveaux chunks sans vider les anciens
        self._index_chunks(new_chunks)
        self.chunks.extend(new_chunks)

        # Recalculer les communautés
        hubs = sorted(self.nodes.values(), key=lambda x: x["degree"], reverse=True)[:10]
        self.communities = {
            f"community_{i}": [h["name"]] for i, h in enumerate(hubs)
        }
        for edge in self.edges[-500:]:   # sur les 500 dernières arêtes seulement
            src, tgt = edge["source"], edge["target"]
            for comm_id, members in self.communities.items():
                if members and (src == members[0] or tgt == members[0]):
                    if src not in members and len(members) < 8:
                        members.append(src)
                    if tgt not in members and len(members) < 8:
                        members.append(tgt)

        logger.info(f"[LightRAG] Indexé '{title}': {len(new_chunks)} chunks, "
                    f"{len(self.nodes)} nœuds, {len(self.edges)} arêtes total")

    def _index_chunks(self, docs):
        """Extrait entités + relations depuis une liste de chunks et les ajoute au graphe."""
        for doc in docs:
            title   = doc.get("title", "Doc")
            text    = doc.get("text",  "")
            doc_id  = doc.get("id",    title)

            # Entités : noms propres (majuscule) + mots-clés spécifiques
            candidates = set()
            if title:
                candidates.add(title.strip())

            proper_nouns = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
            for pn in proper_nouns:
                if (len(pn) > 2
                        and pn.lower() not in {
                            "the", "this", "that", "these", "there",
                            "when", "what", "where", "which", "how"}):
                    candidates.add(pn)

            # Sigles tout-caps (ex: Ooredoo, DNS, QoS, SLA)
            acronyms = re.findall(r"\b[A-Z]{2,6}\b", text)
            for ac in acronyms:
                if ac not in {"OR", "AND", "NOT", "IS", "ARE", "FOR", "THE", "OF", "TO"}:
                    candidates.add(ac)

            # Nombres + unités (ex: "100 Mbps", "99.9%")
            nums = re.findall(r"\b\d+(?:[.,]\d+)?(?:\s*(?:Mbps|Gbps|MHz|ms|KB|MB|GB|%|GHz))?\b", text)
            for num in nums[:5]:
                candidates.add(num.strip())

            for c in candidates:
                if c not in self.nodes:
                    self.nodes[c] = {
                        "name":        c,
                        "degree":      0,
                        "source_docs": {doc_id},
                        "type":        "Entity"
                    }
                else:
                    self.nodes[c]["source_docs"].add(doc_id)

            # Relations par co-occurrence dans le même chunk
            cand_list = list(candidates)
            for i in range(len(cand_list)):
                for j in range(i + 1, min(len(cand_list), i + 6)):
                    src, tgt = cand_list[i], cand_list[j]
                    if src in self.nodes:
                        self.nodes[src]["degree"] += 1
                    if tgt in self.nodes:
                        self.nodes[tgt]["degree"] += 1
                    self.edges.append({
                        "source":   src,
                        "relation": "associated_with",
                        "target":   tgt,
                        "doc_id":   doc_id
                    })

    def query(self, question: str, mode: str = "hybrid") -> dict:
        """Interroge LightRAG et retourne answer + context.
        Délègue à retrieve() puis formate la réponse avec le LLM ou localement.
        """
        if not self.chunks and not self.nodes:
            return {
                "answer":  "LightRAG non initialisé — aucun document indexé.",
                "context": "",
                "mode":    mode
            }
        result  = self.retrieve(question, mode=mode)
        context = result.get("context", "")
        # On retourne le contexte brut (le LLM est appelé dans fast_server.py)
        return {
            "answer":  "",          # sera généré par llm() dans fast_server.py
            "context": context,
            "mode":    mode,
            "meta":    result.get("meta", {})
        }

    def _retrieve_naive(self, query: str, top_k: int = 5) -> list:
        """Mode Naive : Récupération textuelle simple par similarité lexicale."""
        q_words = set(re.findall(r"\w+", query.lower()))
        scored = []
        for c in self.chunks:
            text = f"{c.get('title', '')} {c.get('text', '')}".lower()
            overlap = sum(1 for w in q_words if w in text)
            if overlap > 0:
                scored.append((overlap, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:top_k]]

    def _retrieve_local(self, query: str, top_k: int = 15) -> list:
        """Mode Local : Triplets et entités spécifiques à 1-hop / 2-hop."""
        q_lower = query.lower()
        matched_entities = [name for name in self.nodes if name.lower() in q_lower]
        if not matched_entities:
            words = set(re.findall(r"\w+", q_lower))
            matched_entities = [
                name for name in self.nodes
                if any(w in name.lower() for w in words if len(w) > 3)
            ]

        local_edges = []
        seen = set()
        for e in self.edges:
            src, tgt = e["source"], e["target"]
            if any(m.lower() == src.lower() or m.lower() == tgt.lower()
                   for m in matched_entities):
                key = (src, e["relation"], tgt)
                if key not in seen:
                    seen.add(key)
                    local_edges.append(e)
        return local_edges[:top_k]

    def _retrieve_global(self, query: str) -> list:
        """Mode Global : Résumés des communautés et macro-structures relationnelles."""
        summaries = []
        for comm_id, members in self.communities.items():
            if members:
                hub       = members[0]
                connected = ", ".join(members[1:6])
                summaries.append(f"Communauté [{hub}] : connectée à ({connected})")
        return summaries[:5]

    def retrieve(self, query: str, mode: str = "hybrid") -> dict:
        """
        Exécute la recherche selon l'un des 5 modes LightRAG :
        'naive' | 'local' | 'global' | 'hybrid' | 'mix'
        """
        mode = mode.lower()
        context_parts = []
        results_meta  = {"mode": mode, "query": query}

        if mode == "naive":
            chunks = self._retrieve_naive(query, top_k=5)
            for c in chunks:
                context_parts.append(f"[{c.get('title','')}] {c.get('text','')}[:500]")
            results_meta["chunks_count"] = len(chunks)

        elif mode == "local":
            local_triplets = self._retrieve_local(query)
            for t in local_triplets:
                context_parts.append(f"- {t['source']} --[{t['relation']}]--> {t['target']}")
            results_meta["triplets_count"] = len(local_triplets)

        elif mode == "global":
            globals_list = self._retrieve_global(query)
            context_parts.extend(globals_list)
            results_meta["global_communities"] = len(globals_list)

        elif mode == "hybrid":
            # Fusion Local + Global + Passages textuels
            local_triplets = self._retrieve_local(query, top_k=10)
            for t in local_triplets:
                context_parts.append(
                    f"- [Fait] {t['source']} --[{t['relation']}]--> {t['target']}"
                )
            globals_list = self._retrieve_global(query)
            for g in globals_list[:3]:
                context_parts.append(f"- [Structure] {g}")
            # + passages textuels pertinents
            naive_chunks = self._retrieve_naive(query, top_k=3)
            for c in naive_chunks:
                context_parts.append(
                    f"[Passage – {c.get('title','')}] {c.get('text','')[:400]}"
                )
            results_meta["triplets_count"]  = len(local_triplets)
            results_meta["global_count"]    = len(globals_list[:3])
            results_meta["passages_count"]  = len(naive_chunks)

        elif mode == "mix":
            local_triplets = self._retrieve_local(query, top_k=8)
            for t in local_triplets:
                context_parts.append(f"- [Fact] {t['source']} --[{t['relation']}]--> {t['target']}")
            globals_list = self._retrieve_global(query)
            for g in globals_list[:2]:
                context_parts.append(f"- [Theme] {g}")
            chunks = self._retrieve_naive(query, top_k=3)
            for c in chunks:
                context_parts.append(f"[Doc: {c.get('title','')}] {c.get('text','')[:300]}")
            results_meta["mixed_elements"] = len(context_parts)

        else:
            raise ValueError(
                f"Mode inconnu '{mode}'. Choisissez parmi: naive, local, global, hybrid, mix"
            )

        formatted_context = "\n".join(context_parts) if context_parts else "Aucune information trouvée dans LightRAG."
        return {
            "mode":    mode,
            "context": formatted_context,
            "meta":    results_meta
        }

