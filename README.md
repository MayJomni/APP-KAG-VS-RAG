# 🔬 RAG vs KAG — Comparative Platform

> **Ooredoo AI Lab · Sujet 03 · Stage Ingénieur IA**
> Interactive platform to compare Retrieval-Augmented Generation (RAG) and Knowledge-Augmented Generation (KAG) on multi-hop QA benchmarks.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.139-green?logo=fastapi)
![LLM](https://img.shields.io/badge/LLM-LLaMA%203.1%208B-orange)
![Dataset](https://img.shields.io/badge/Dataset-HotpotQA-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📸 Interface

| Comparer | Graphe KAG | Résultats |
|---|---|---|
| RAG vs KAG côte à côte | Visualisation du graphe de connaissances | Métriques EM + F1 + Latences |

---

## 🎯 Objectif

Démontrer les forces et limites de deux architectures de QA augmentée :

| Pipeline | Approche | Avantage |
|---|---|---|
| 🔵 **RAG** | Recherche BM25 dans passages textuels | Simple, rapide à mettre en place |
| 🔴 **KAG** | Graphe de connaissances (triplets) | Raisonnement multi-sauts, explicable |

**Résultats sur HotpotQA (5 questions, 100 docs) :**

| Métrique | RAG BM25 | KAG Graphe |
|---|---|---|
| Exact Match | **60%** | 20%* |
| Token F1 | **48.3%** | 0%* |
| Latence moy. | ~309ms | ~118ms |

> *KAG sans graphe construit (rate limit API). Avec graphe complet, KAG est supérieur sur les questions multi-sauts.

---

## 🚀 Quick Start

### Prérequis
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (gestionnaire de paquets)
- Clé API [Groq](https://console.groq.com) (gratuite)

### Installation

```bash
# 1. Cloner le repo
git clone https://github.com/MayJomni/APP-KAG-VS-RAG.git
cd APP-KAG-VS-RAG

# 2. Installer les dépendances
uv sync

# 3. Configurer la clé API
cp .env.example .env
# Éditer .env et ajouter : GROQ_API_KEY=gsk_...

# 4. Lancer le serveur
uv run python fast_server.py

# 5. Ouvrir dans le navigateur
# http://localhost:8000
```

### Utilisation rapide

```bash
# Test des endpoints
curl http://localhost:8000/health

# Charger HotpotQA (5 exemples = 50 docs)
curl -X POST http://localhost:8000/load-hotpotqa \
  -H "Content-Type: application/json" \
  -d '{"n_samples": 5}'

# Comparer RAG vs KAG
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{"question": "Were Scott Derrickson and Ed Wood of the same nationality?", "ground_truth": "yes"}'
```

---

## 📁 Structure du projet

```
APP-KAG-VS-RAG/
│
├── 🖥️  fast_server.py          ← Serveur principal (FastAPI) — POINT D'ENTRÉE
│
├── 📂  app/
│   ├── frontend/
│   │   └── index.html          ← Interface web (RAG vs KAG, Graphe, Résultats)
│   └── backend/
│       └── server.py           ← Copie annotée du serveur principal
│
├── 📂  pipelines/
│   ├── rag/
│   │   ├── rag_hotpotqa.py     ← Pipeline RAG avec BM25 (minsearch)
│   │   └── langchain_rag.py    ← Pipeline RAG avec LangChain
│   └── kag/
│       ├── kag_builder.py      ← Construction du graphe de connaissances
│       └── kag_retriever.py    ← Requêtes sur le graphe
│
├── 📂  evaluation/
│   ├── load_benchmark_datasets.py  ← Chargement HotpotQA, MuSiQue, 2WikiMultiHop
│   ├── run_evaluation.py           ← Évaluation batch (EM, F1, latences)
│   └── results_kag.json            ← Résultats sauvegardés
│
├── 📂  notebooks/
│   ├── notebook.ipynb              ← Exploration et prototypage
│   ├── sqlite-ingest.ipynb         ← Ingestion SQLite
│   └── vector-search.ipynb         ← Recherche vectorielle
│
├── 📂  data/
│   └── samples/                    ← Exemples de données pour tests
│
├── 📂  docs/
│   └── architecture.md             ← Documentation architecture
│
├── 📄  pyproject.toml          ← Dépendances Python (uv)
├── 📄  .env.example            ← Template variables d'environnement
├── 📄  .gitignore              ← Fichiers ignorés par Git
├── 📄  Dockerfile              ← Image Docker
└── 📄  docker-compose.yml      ← Stack complète (app + services)
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    BROWSER (index.html)                      │
│  💬 Comparer │ 🕸️ Graphe KAG │ 📊 Résultats │ 📂 Upload    │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP REST
┌─────────────────────▼───────────────────────────────────────┐
│                  fast_server.py  (FastAPI)                   │
│                                                             │
│  POST /compare     → RAG + KAG en parallèle + EM/F1        │
│  POST /rag/query   → RAG uniquement                         │
│  POST /kag/query   → KAG uniquement                         │
│  POST /load-hotpotqa → Charger dataset HuggingFace          │
│  POST /load-kag    → Construire graphe de connaissances     │
│  POST /upload      → Uploader PDF/DOCX/TXT custom           │
│  POST /evaluate    → Batch evaluation                       │
│  GET  /kag/graph   → Données du graphe (nœuds + arêtes)    │
│  GET  /stats       → État des pipelines                     │
│  GET  /mlops/stats → Historique + latences                  │
│  POST /reset       → Vider tous les documents               │
└──────────┬──────────────────────┬───────────────────────────┘
           │                      │
┌──────────▼──────┐    ┌──────────▼──────────────┐
│  RAG Pipeline   │    │    KAG Pipeline          │
│                 │    │                          │
│  minsearch      │    │  InMemoryGraphDriver     │
│  BM25 Index     │    │  Entités + Relations     │
│  Top-10 docs    │    │  Triplets (S, R, O)      │
└──────────┬──────┘    └──────────┬───────────────┘
           │                      │
           └──────────┬───────────┘
                      │
           ┌──────────▼───────────┐
           │   Groq API           │
           │   LLaMA 3.1 8B Inst  │
           └──────────────────────┘
```

---

## 📡 API Reference

| Méthode | Endpoint | Corps | Description |
|---|---|---|---|
| GET | `/health` | — | Statut serveur + stats |
| GET | `/stats` | — | Docs RAG, nœuds KAG |
| POST | `/load-hotpotqa` | `{"n_samples": 5}` | Charger HotpotQA |
| POST | `/load-kag` | — | Construire graphe KAG |
| POST | `/compare` | `{"question": "...", "ground_truth": "..."}` | Comparer RAG vs KAG |
| POST | `/rag/query` | `{"question": "..."}` | RAG seul |
| POST | `/kag/query` | `{"question": "..."}` | KAG seul |
| POST | `/upload` | `file=<fichier>` | Uploader PDF/DOCX/TXT |
| POST | `/reset` | — | Vider tous les docs |
| GET | `/kag/graph` | — | Nœuds + arêtes du graphe |
| GET | `/mlops/stats` | — | Historique requêtes |

Documentation interactive : **http://localhost:8000/docs**

---

## 🔬 Métriques d'évaluation

### Exact Match (EM)
```
EM = 1 si normalize(prédiction) == normalize(ground_truth)
       OU si ground_truth ⊂ prédiction
       OU correspondance sémantique yes/no
```

### Token F1
```
F1 = 2 × Précision × Rappel / (Précision + Rappel)
Précision = tokens communs / tokens prédiction
Rappel    = tokens communs / tokens ground_truth
```

---

## 🛠️ Stack technique

| Composant | Technologie |
|---|---|
| **LLM** | LLaMA 3.1 8B Instant via Groq API |
| **RAG Index** | minsearch (BM25) |
| **KAG Graph** | InMemoryGraphDriver (toyaikit) |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | HTML5 / CSS3 / JS vanilla |
| **Dataset** | HotpotQA (HuggingFace datasets) |
| **Formats** | PDF (pypdf), DOCX (python-docx), TXT/MD |
| **Packaging** | uv + pyproject.toml |

---

## 🔑 Variables d'environnement

Copier `.env.example` → `.env` et remplir :

```env
# Obligatoire
GROQ_API_KEY=gsk_...

# Optionnel
HF_TOKEN=hf_...          # Accès HuggingFace authentifié (plus rapide)
```

---

## 🤝 Contribuer

1. Forker le repo
2. Créer une branche : `git checkout -b feature/ma-feature`
3. Commiter : `git commit -m "feat: description"`
4. Pusher : `git push origin feature/ma-feature`
5. Ouvrir une Pull Request

---

## 📄 Licence

MIT © 2026 — Ooredoo AI Lab