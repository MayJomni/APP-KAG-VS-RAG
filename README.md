# APP-KAG-VS-RAG

Plateforme comparative **RAG vs KAG** avec interface web Ooredoo, MLOps intégré et support de documents personnalisés.

## Description

Cette application permet de comparer deux approches de Question-Answering basées sur LLM :

- **RAG (Retrieval-Augmented Generation)** : recherche vectorielle via `minsearch` + LLM Groq
- **KAG (Knowledge-Augmented Generation)** : graphe de connaissances (Neo4j / in-memory) + LLM Groq

## Fonctionnalités

- 📄 **Upload de documents** : PDF, TXT, MD, DOCX
- ⚡ **Mode Comparer** : réponses RAG et KAG côte à côte
- 📊 **MLOps Dashboard** : latence, tokens, métriques (Exact Match, Token F1) via MLflow
- 🔗 **Visualisation** : triplets KAG et documents RAG utilisés pour répondre
- 🎨 **Interface Ooredoo** : design premium rouge et noir

## Stack technique

| Composant | Technologie |
|-----------|-------------|
| LLM | Groq (`llama-3.1-8b-instant`) |
| RAG Index | minsearch (BM25 in-memory) |
| KAG Graph | Neo4j / InMemoryGraphDriver |
| Backend | FastAPI + uvicorn |
| MLOps | MLflow + LangFuse (optionnel) |
| Évaluation | Exact Match + Token F1 (HotpotQA standard) |
| Frontend | HTML/CSS/JS vanilla |

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/MayJomni/APP-KAG-VS-RAG.git
cd APP-KAG-VS-RAG

# Installer les dépendances
uv sync

# Configurer les variables d'environnement
cp .env.example .env
# Éditer .env et ajouter votre GROQ_API_KEY

# Lancer le serveur
uv run python kag_server.py
```

## Configuration `.env`

```env
GROQ_API_KEY=your_groq_api_key
NEO4J_URI=bolt://localhost:7687      # optionnel
NEO4J_USERNAME=neo4j                 # optionnel
NEO4J_PASSWORD=your_password         # optionnel
LANGFUSE_PUBLIC_KEY=...              # optionnel
LANGFUSE_SECRET_KEY=...              # optionnel
```

## Utilisation

1. Ouvrir **http://localhost:8000**
2. Uploader un document (PDF/TXT/MD) ou charger des exemples HotpotQA
3. Cliquer **"Construire les pipelines"**
4. Poser des questions en mode **⚡ Comparer**
5. Consulter le dashboard **📊 MLOps** pour les métriques

## Structure du projet

```
├── kag_server.py          # Serveur FastAPI unifié
├── document_processor.py  # Parsing PDF/TXT/MD/DOCX + chunking
├── data_loader.py         # Chargement HotpotQA
├── kag_pipeline/          # Pipeline KAG (extraction entités + graphe)
├── rag_pipeline/          # Pipeline RAG (minsearch + Groq)
├── mlops/                 # Tracking MLflow + LangFuse
├── evaluation/            # Métriques EM + Token F1
└── kag_app/               # Interface web Ooredoo
```

## Auteur

**MayJomni** — Projet de fin d'études (Sujet 3 : RAG vs KAG)