# CHAPITRE : RÉALISATION TECHNIQUE, VALIDATION EXPÉRIMENTALE ET PERSPECTIVES

**Contexte :** Ooredoo Tunisie — Direction des Systèmes d'Information & AI Lab  
**Projet :** Sujet 03 — Étude comparative et benchmarking des architectures RAG et KAG  
**Auteur :** Ingénieur IA Stagiaire  

---

## 1. ARCHITECTURE LOGIQUE ET MODÉLISATION UML

Le système est conçu selon une architecture logicielle multicouche garantissant une extensibilité maximale.

### 1.1 Diagramme de Composants du Système Global

```mermaid
graph TB
    subgraph Client_Tier ["1. COUCHE CLIENT (Navigateur Web)"]
        UI_SPA["Interface Utilisateur SPA (HTML5 / Modern CSS / Vanilla JS)<br/>- Vue Comparateur<br/>- Vue Routeur Agentique<br/>- Visualiseur Canvas Force-Directed<br/>- Dashboard MLOps"]
    end

    subgraph API_Tier ["2. COUCHE APPLICATION & API REST (FastAPI)"]
        FastAPI_Core["Contrôleur Central (fast_server.py)<br/>FastAPI Asynchrone"]
        EP_Query["/compare & /query"]
        EP_Router["/smart"]
        EP_Graph["/kag/graph"]
        EP_Stats["/stats & /mlops/stats"]
    end

    subgraph Business_Tier ["3. COUCHE MÉTIER & ORCHESTRATION"]
        RouterEngine["Module Agentique : SmartRouter<br/>(Détection syntaxique & classification de requêtes)"]
        
        subgraph RAG_Engine ["Sous-Système RAG (Multi-VectorStore)"]
            VS_Interface["«interface» BaseVectorStore"]
            BM25_Impl["BM25Store (Okapi BM25 probabiliste)"]
            MinSearch_Impl["MinSearchStore (TF-IDF + Cosinus)"]
            Dense_Impl["DenseVectorStore (Embeddings Sémantiques)"]
            Milvus_Impl["MilvusLiteStore (Index Vectoriel Dédié)"]
        end

        subgraph KAG_Engine ["Sous-Système KAG (Graphe & LightRAG)"]
            LightRAG_Core["Moteur LightRAGEngine<br/>(5 Modes: naive, local, global, hybrid, mix)"]
            GraphMemory["Base Graphe en Mémoire<br/>(201 nœuds / 239 arêtes)"]
            GraphML_Export["Exporteur GraphML standardisé"]
        end
    end

    subgraph Evaluation_Tier ["4. SOUS-SYSTÈME BENCHMARK & MLOPS"]
        Runner["Benchmark Runner (run_benchmark.py)<br/>Graine fixée : seed=42"]
        CorpusHotpot["Corpus Étalon HotpotQA<br/>(24 docs, 13 questions étalonnées)"]
        MetricCore["Calculateur Déterministe<br/>(Exact Match, Token F1, Latence p95, Refus)"]
        StorageResults["Persistance des Métriques<br/>(results/*.csv & results/*.json)"]
    end

    subgraph Infrastructure_Tier ["5. SERVICES D'INFÉRENCE EXTERNES"]
        GroqGateway["Passerelle API Groq (Accélérateurs LPUs)"]
        GemmaModel["Gemma 2 9B-it (Extraction d'entités & relations)"]
        LlamaModel["LLaMA 3.1 8B-Instruct (Génération finale fermée)"]
        CacheManager["Cache Déterministe Local (Anti-Quotas/Rate-Limits)"]
    end

    UI_SPA --> EP_Query
    UI_SPA --> EP_Router
    UI_SPA --> EP_Graph
    UI_SPA --> EP_Stats
    EP_Query & EP_Router & EP_Graph & EP_Stats --> FastAPI_Core

    FastAPI_Core --> RouterEngine
    RouterEngine -->|Requête Factuelle / Définition| VS_Interface
    RouterEngine -->|Requête Multi-Hop / Relationnelle| LightRAG_Core

    VS_Interface <|.. BM25_Impl
    VS_Interface <|.. MinSearch_Impl
    VS_Interface <|.. Dense_Impl
    VS_Interface <|.. Milvus_Impl

    LightRAG_Core --> GraphMemory
    GraphMemory --> GraphML_Export

    Runner --> CorpusHotpot
    Runner --> VS_Interface
    Runner --> LightRAG_Core
    Runner --> MetricCore
    MetricCore --> StorageResults
    StorageResults --> EP_Stats

    RAG_Engine --> GroqGateway
    KAG_Engine --> GroqGateway
    GroqGateway --> CacheManager
    GroqGateway --> GemmaModel
    GroqGateway --> LlamaModel
```

---

## 2. DESCRIPTION DÉTAILLÉE DES INTERFACES UTILISATEUR

L'application web intégrée est servie sur **`http://localhost:8000`** et comprend :
1. **La Barre Latérale (Sidebar) :** Compteurs en temps réel (`24 RAG docs`, `201 KAG nœuds`, `239 relations`), zone d'upload de fichiers (PDF, DOCX, TXT), bouton de déclenchement de la construction du graphe via LLM (`gemma2-9b-it`), et questions de démonstration pré-remplies.
2. **Onglet 💬 Comparer :** Confrontation côte à côte de la carte RAG (BM25 avec sources textuelles citées) et de la carte KAG (triplets relationnels extraits du graphe). Sélecteur de mode en bas (`RAG seul`, `KAG seul`, `Comparer`, `Auto`).
3. **Onglet 🤖 Agent :** Visualisation de l'analyse d'intention et justification explicable de l'aiguillage opéré par le routeur intelligent.
4. **Onglet 🕸️ Graphe KAG :** Rendu interactif en Canvas HTML5 simulant les forces physiques pour explorer les 201 entités et 239 relations, avec manipulation à la souris (zoom, pan, survol et inspection détaillée au clic).
5. **Onglet 📊 Résultats :** Dashboard de métriques en direct synthétisant les résultats du benchmark (Exact Match, Token F1, latences p50/p95, consommation de tokens).

---

## 3. RÉSULTATS EXPÉRIMENTAUX FINAUX

*Tableau consolidé sur le corpus HotpotQA commun (Graine 42, 13 questions, 24 documents) :*

| # | Configuration | Architecture | Exact Match (EM) | Token F1 | Latence Moy. | Latence p95 | Robustesse Refus | Tokens Consommés |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Routeur-Lexical-Smart** | Hybride Agentique | **53.8%** | **37.0%** | 0.2 ms | 2.02 ms | **100.0%** | 3 807 |
| 2 | **LightRAG-Hybrid** | KAG Hybride (Graphe + Textes) | **53.8%** | **37.0%** | 0.6 ms | 4.18 ms | **100.0%** | 3 585 |
| 3 | **LightRAG-Local** | KAG Voisinage 1-hop & 2-hop | **53.8%** | 29.3% | 0.3 ms | 3.91 ms | **100.0%** | 2 381 |
| 4 | **RAG-BM25 (Okapi)** | RAG Lexical Probabiliste | 46.2% | 29.3% | 0.2 ms | 2.06 ms | **100.0%** | 4 486 |
| 5 | **RAG-minsearch (TF-IDF)**| RAG Lexical Cosinus | 46.2% | 29.3% | 1.3 ms | 6.42 ms | **100.0%** | 4 505 |
| 6 | **LightRAG-Naive** | Baseline plein texte | 46.2% | 29.3% | < 0.1 ms | < 0.1 ms | **100.0%** | 4 718 |
| 7 | **LightRAG-Mix** | Concaténation Triplet + Doc | 46.2% | 29.3% | 0.3 ms | 2.08 ms | **100.0%** | 5 247 |
| 8 | **LightRAG-Global** | Résumés relationnels | 38.5% | 33.8% | < 0.1 ms | < 0.1 ms | **100.0%** | 2 307 |
| 9 | **KAG-In-Memory-Graph** | Graphe de Connaissances Pur | 30.8% | 23.1% | **< 0.1 ms** | **< 0.1 ms** | **100.0%** | **384 (-91%)** |
| 10| **RAG-Dense-Vector** | Embeddings Denses HF | 15.4% | 12.8% | 0.2 ms | 2.01 ms | **100.0%** | 4 276 |
| 11| **RAG-Milvus-Lite** | Index Vectoriel Dédié | 15.4% | 12.8% | < 0.1 ms | < 0.1 ms | **100.0%** | 4 276 |

---

## 4. PERSPECTIVES D'AMÉLIORATION ET FEUILLE DE ROUTE INDUSTRIELLE

Pour transposer ce Proof of Concept (POC) en une plateforme industrielle à haute disponibilité au sein d'Ooredoo Tunisie, quatre axes majeurs sont proposés :
1. **Scalabilité du Stockage Distribué :** Déploiement d'une instance **Neo4j Enterprise** pour gérer des graphes de plusieurs millions d'arêtes avec requêtage Cypher, couplée à un cluster **Milvus Distributed** pour l'indexation vectorielle à grande échelle.
2. **Spécialisation Télécoms :** Fine-tuning d'embeddings sur le vocabulaire spécifique de l'opérateur (RAN, 5G, BSS/OSS, Roaming) et définition d'une ontologie formelle OWL/RDF pour normaliser l'extraction d'entités.
3. **Orchestration Agentique avec LangGraph :** Mise en œuvre de boucles de raisonnement avancées de type *Corrective RAG (CRAG)* et *Self-RAG* pour ré-interroger automatiquement le graphe en cas d'incertitude.
4. **Observabilité Continue et MLOps :** Intégration de **Langfuse** en conteneur Docker pour le traçage des latences, la détection du drift sémantique et la collecte des retours utilisateurs.
