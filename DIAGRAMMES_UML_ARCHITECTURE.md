# DIAGRAMMES UML D'ARCHITECTURE ET DES INTERFACES

**Projet :** Plateforme de Benchmarking RAG vs KAG  
**Contexte :** Ooredoo Tunisie — Sujet 03  
**Auteur :** Ingénieur IA Stagiaire  

---

# 1. DIAGRAMME D'ARCHITECTURE GLOBALE DU SYSTÈME (COMPOSANTS)

```mermaid
graph TB
    subgraph UI_Layer ["1. COUCHE PRÉSENTATION (Frontend Web)"]
        UI_Compare["🖥️ Vue Comparer<br/>(Cartes RAG vs KAG)"]
        UI_Agent["🤖 Vue Agent<br/>(Explication de Routage)"]
        UI_Graph["🕸️ Visualiseur Canvas<br/>(201 nœuds / 239 arêtes)"]
        UI_Results["📊 Tableau de Bord MLOps<br/>(EM, F1, Latences)"]
    end

    subgraph API_Layer ["2. COUCHE CONTRÔLE & EXPOSITION (FastAPI)"]
        FastAPI_App["Serveur REST FastAPI<br/>(fast_server.py)"]
        EP_Compare["/compare & /query"]
        EP_Smart["/smart (Routage)"]
        EP_Graph["/kag/graph"]
        EP_Stats["/stats & /mlops/stats"]
    end

    subgraph Router_Layer ["3. COUCHE AGENTIQUE & ROUTAGE"]
        SmartRouter["Routeur Intelligent (SmartRouter)<br/>Analyse syntaxique & relationnelle"]
    end

    subgraph RAG_Subsystem ["4. SOUS-SYSTÈME RAG (Multi-VectorStore)"]
        BaseVS["«interface»<br/>BaseVectorStore"]
        BM25["BM25Store<br/>(Okapi BM25 pur)"]
        MinSearch["MinSearchStore<br/>(TF-IDF + Cosinus)"]
        Dense["DenseVectorStore<br/>(Hugging Face)"]
        Milvus["MilvusLiteStore<br/>(Index Vectoriel)"]
    end

    subgraph KAG_Subsystem ["5. SOUS-SYSTÈME KAG & LIGHTRAG"]
        LightRAG["Moteur LightRAG<br/>(5 Modes: naive, local, global, hybrid, mix)"]
        InMemKG["Base Graphe en Mémoire<br/>(kg_graph.json)"]
        GraphMLExport["Export GraphML<br/>(lightrag_graph.graphml)"]
    end

    subgraph Benchmark_Subsystem ["6. SOUS-SYSTÈME ÉVALUATION & MLOPS"]
        BenchRunner["Benchmark Runner<br/>(run_benchmark.py)"]
        CorpusDB["Corpus HotpotQA Commun<br/>(24 docs, 13 questions étalonnées)"]
        MetricEngine["Moteur Métriques<br/>(Exact Match, Token F1, Latence p95, Refus)"]
        ResultsCSV["results/*.csv & results/*.json"]
    end

    subgraph External_Inference ["7. COUCHE INFÉRENCE & MODÈLES EXTERNES"]
        GroqGateway["Passerelle Groq LPUs"]
        LLM_Gemma["Gemma 2 9B<br/>(Extraction triplets & entités)"]
        LLM_Llama["LLaMA 3.1 8B<br/>(Génération finale conditionnée)"]
        LocalCache["Cache Local des Réponses (Anti-Rate-Limit)"]
    end

    UI_Compare --> EP_Compare
    UI_Agent --> EP_Smart
    UI_Graph --> EP_Graph
    UI_Results --> EP_Stats
    EP_Compare & EP_Smart & EP_Graph & EP_Stats --> FastAPI_App

    FastAPI_App --> SmartRouter
    SmartRouter -->|Requête Factuelle / Définition| BaseVS
    SmartRouter -->|Requête Multi-Hop / Relationnelle| LightRAG

    BaseVS <|-- BM25
    BaseVS <|-- MinSearch
    BaseVS <|-- Dense
    BaseVS <|-- Milvus

    LightRAG --> InMemKG
    InMemKG --> GraphMLExport

    BenchRunner --> CorpusDB
    BenchRunner --> BaseVS
    BenchRunner --> LightRAG
    BenchRunner --> MetricEngine
    MetricEngine --> ResultsCSV
    ResultsCSV --> EP_Stats

    RAG_Subsystem --> GroqGateway
    KAG_Subsystem --> GroqGateway
    GroqGateway --> LocalCache
    GroqGateway --> LLM_Gemma
    GroqGateway --> LLM_Llama
```

---

# 2. DIAGRAMME DE CLASSES GLOBAL DU CŒUR MÉTIER

```mermaid
classDiagram
    direction TB

    class BaseVectorStore {
        <<abstract>>
        +String name
        +List~dict~ documents
        +index_documents(documents: List~dict~) void*
        +search(query: String, top_k: int) List~dict~*
    }

    class BM25Store {
        +float k1
        +float b
        +dict idf
        +List~int~ doc_len
        +float avgdl
        +index_documents(documents: List~dict~) void
        +search(query: String, top_k: int) List~dict~
        -_tokenize(text: String) List~String~
    }

    class MinSearchStore {
        +Index text_index
        +index_documents(documents: List~dict~) void
        +search(query: String, top_k: int) List~dict~
    }

    class DenseVectorStore {
        +String model_name
        +List~float~ embeddings
        +index_documents(documents: List~dict~) void
        +search(query: String, top_k: int) List~dict~
    }

    class MilvusLiteStore {
        +String collection_name
        +index_documents(documents: List~dict~) void
        +search(query: String, top_k: int) List~dict~
    }

    class LightRAGEngine {
        +dict graph_data
        +List~dict~ documents
        +load_graph(json_path: String) void
        +query(question: String, mode: String, top_k: int) dict
        +_extract_entities(text: String) List~String~
        +_get_local_neighborhood(entities: List~String~, hops: int) List~dict~
        +_bm25_search(query: String, top_k: int) String
    }

    class SmartRouter {
        +route(question: String) dict
        +_detect_multi_hop_patterns(question: String) bool
        +_extract_named_entities(question: String) List~String~
    }

    BaseVectorStore <|-- BM25Store
    BaseVectorStore <|-- MinSearchStore
    BaseVectorStore <|-- DenseVectorStore
    BaseVectorStore <|-- MilvusLiteStore

    SmartRouter --> BaseVectorStore : RAG
    SmartRouter --> LightRAGEngine : KAG
```

---

# 3. DIAGRAMMES UML PAR INTERFACE

## 3.1 Interface 1 : Vue `💬 Comparer` (Diagramme de Séquence)

```mermaid
sequenceDiagram
    autonumber
    actor User as Collaborateur Ooredoo
    participant View as Interface [💬 Comparer]
    participant Server as Contrôleur FastAPI
    participant RAG as Pipeline RAG (BM25)
    participant KAG as Pipeline KAG (LightRAG)
    participant Groq as LLM Groq (LPUs)

    User->>View: Saisit question & clique "🚀 Lancer la Comparaison"
    activate View
    View->>Server: POST /compare {"question": "...", "mode": "compare"}
    activate Server

    par Exécution Asynchrone RAG
        Server->>RAG: search(query, top_k=5)
        activate RAG
        RAG-->>Server: Top 5 passages + scores
        deactivate RAG
        Server->>Groq: Générer réponse RAG (Prompt + Passages)
        activate Groq
        Groq-->>Server: Texte réponse RAG
        deactivate Groq
    and Exécution Asynchrone KAG
        Server->>KAG: query(query, mode="hybrid")
        activate KAG
        KAG->>KAG: Résolution d'entités & triplets 1-hop/2-hop
        KAG-->>Server: Triplets structurés + contexte local
        deactivate KAG
        Server->>Groq: Générer réponse KAG (Prompt + Triplets)
        activate Groq
        Groq-->>Server: Texte réponse KAG
        deactivate Groq
    end

    Server-->>View: Payload JSON {rag: {...}, kag: {...}}
    deactivate Server
    View-->>User: Rendu côte à côte (Carte RAG vs Carte KAG)
    deactivate View
```

## 3.2 Interface 2 : Vue `🤖 Agent` (Diagramme d'Activité)

```mermaid
stateDiagram-v2
    [*] --> ReceptionQuestion : Saisie question
    ReceptionQuestion --> AnalyseSyntaxique : Détection conjonctions & mots-clés
    AnalyseSyntaxique --> EvaluationComplexite
    
    state EvaluationComplexite <<choice>>
    EvaluationComplexite --> BrancheKAG : Si multi-hop / relations composées
    EvaluationComplexite --> BrancheRAG : Si fait direct / définition simple

    state BrancheKAG {
        TraverserRelations --> SyntheseTriplets
    }
    state BrancheRAG {
        ScoringBM25 --> ExtractionChunks
    }

    BrancheKAG --> ProductionReponse
    BrancheRAG --> ProductionReponse
    ProductionReponse --> [*] : Affichage avec justification
```

## 3.3 Interface 3 : Vue `🕸️ Graphe KAG` (Diagramme de Séquence)

```mermaid
sequenceDiagram
    autonumber
    actor User as Utilisateur
    participant Canvas as Vue [🕸️ Graphe KAG] (Canvas HTML5)
    participant Server as Serveur FastAPI (/kag/graph)

    User->>Canvas: Clique onglet "🕸️ Graphe KAG"
    activate Canvas
    Canvas->>Server: GET /kag/graph
    activate Server
    Server-->>Canvas: JSON {nodes: 201, edges: 239}
    deactivate Server
    Canvas->>Canvas: Simulation de forces (Force-Directed Graph)
    Canvas-->>User: Affichage du graphe interactif (zoom/pan/inspection)
    deactivate Canvas
```

## 3.4 Interface 4 : Vue `📊 Résultats` (Diagramme de Séquence)

```mermaid
sequenceDiagram
    autonumber
    actor Admin as Évaluateur
    participant View as Interface [📊 Résultats]
    participant Server as Serveur FastAPI (/mlops/stats)
    participant DB as Fichiers Résultats (results/*.csv, *.json)

    Admin->>View: Ouvre onglet "📊 Résultats"
    activate View
    View->>Server: GET /mlops/stats
    activate Server
    Server->>DB: Lecture des 11 configurations
    DB-->>Server: Métriques (EM, F1, Latences p95, Tokens)
    Server-->>View: Payload JSON agrégé
    deactivate Server
    View-->>Admin: Dashboard de métriques en direct
    deactivate View
```
