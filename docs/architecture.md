# Architecture — RAG vs KAG Platform

## Vue d'ensemble

Ce document décrit l'architecture technique de la plateforme de comparaison RAG vs KAG.

## Composants principaux

### 1. Serveur backend (`fast_server.py`)

Point d'entrée unique. Contient :
- Tous les endpoints FastAPI
- La logique RAG (BM25 via minsearch)
- La logique KAG (extraction de triplets + graphe in-memory)
- Les métriques EM et Token F1
- Le service de l'interface HTML

### 2. Interface frontend (`kag_app/index.html`)

Single-page application en HTML/JS vanilla. Trois onglets :
- **💬 Comparer** : questions avec réponses RAG et KAG côte à côte
- **🕸️ Graphe KAG** : visualisation canvas force-directed du graphe
- **📊 Résultats** : métriques agrégées et historique

### 3. Pipeline RAG

```
Question
  → BM25 search (minsearch) sur les documents indexés
  → Top-10 passages récupérés
  → Prompt : [SYSTEM: QA assistant] + [USER: question + contexte]
  → LLaMA 3.1 8B (Groq) génère la réponse
  → EM + F1 calculés si ground_truth fourni
```

### 4. Pipeline KAG

```
Documents
  → LLM extrait des triplets (Sujet, Relation, Objet) par chunk
  → Triplets stockés dans un graphe in-memory

Question
  → Recherche par mots-clés dans les nœuds du graphe
  → Triplets pertinents récupérés
  → Prompt : [SYSTEM: QA assistant] + [USER: question + triplets]
  → LLaMA 3.1 8B (Groq) génère la réponse
```

## Flux de données

```
HotpotQA (HF)              Documents custom
     │                           │
     ▼                           ▼
POST /load-hotpotqa      POST /upload (PDF/DOCX/TXT)
     │                           │
     └───────────────────────────┘
                 │
         ┌───────┴────────┐
         ▼                ▼
   RAG: minsearch    KAG: graphe
   BM25 index        in-memory
         │                │
         └───────┬─────────┘
                 ▼
         POST /compare
                 │
         Groq API (LLaMA)
                 │
         réponse + EM + F1
```

## Limites connues

| Limite | Cause | Solution possible |
|---|---|---|
| Rate limit Groq 6000 TPM | Construction KAG via LLM | API payante ou GPT-4o |
| Graphe non persistant | In-memory | Neo4j AuraDB |
| RAG lexical seulement | BM25 | Embeddings dense (MiniLM) |
