# 📋 AUDIT TECHNIQUE & ARCHITECTURAL — PROJET APP-KAG-VS-RAG
**Projet :** Étude comparative et benchmarking des architectures RAG et KAG  
**Contexte :** Ooredoo AI Lab · Stage Ingénieur IA (Sujet 03)  
**Date d'audit :** 30 septembre 2026  
**Auditeur :** Ingénieur IA Senior  

---

## 1. Arborescence du dépôt et cartographie des composants

### 1.1 Arborescence actuelle

```
APP-KAG-VS-RAG/
├── app/
│   ├── backend/
│   │   └── server.py                   # Copie antérieure annotée de fast_server.py
│   └── frontend/
│       └── index.html                  # Interface UI principale (Onglets Comparer, Graphe, Résultats)
├── benchmark/
│   ├── benchmark_config.yaml           # Configuration déclarative du benchmark
│   ├── run_benchmark.py                # Script d'exécution du benchmark standardisé
│   └── __init__.py
├── docs/
│   └── architecture.md                 # Documentation d'architecture sommaire
├── evaluation/
│   ├── deepeval_evaluator.py           # Évaluateur DeepEval (Correctness, Hallucination, Relevancy)
│   ├── load_benchmark_datasets.py      # Chargeur multi-datasets (HotpotQA, MuSiQue, 2Wiki, etc.)
│   ├── metrics.py                      # Métriques canoniques (Exact Match souple, Token F1)
│   ├── ragas_evaluator.py              # Évaluateur Ragas (Faithfulness, Relevancy, Precision, Recall)
│   ├── results_kag.json                # Résultats précalculés (5 questions HotpotQA)
│   ├── run_evaluation.py               # Script d'évaluation standalone RAG vs KAG
│   └── __init__.py
├── kag_app/
│   ├── icon-512.jpg, splash.jpg        # Assets PWA
│   ├── index.html                      # UI alternative servie par kag_server.py
│   ├── manifest.json, sw.js            # Configuration Progressive Web App
│   └── mobile.html                     # Vue responsive mobile
├── kag_pipeline/
│   ├── entity_extraction.py            # Extraction d'entités/relations via Groq (Pydantic)
│   ├── graph_builder.py                # Construction graphe (Neo4j driver + fallback InMemoryGraphDriver)
│   ├── graph_query.py                  # Requêtes Cypher 1-hop & 2-hop (multi-hop)
│   ├── kag_helper.py                   # Orchestrateur KAGBase (interface unifiée)
│   ├── langgraph_kag.py                # Pipeline KAG orchestré avec LangGraph
│   └── __init__.py
├── mlops/
│   ├── tracker.py                      # Tracker unifié MLflow + Langfuse + historique mémoire
│   └── __init__.py
├── notebooks/
│   ├── notebook.ipynb                  # Prototypage et exploration
│   ├── sqlite-ingest.ipynb             # Ingestion FAQ / SQLite
│   ├── sqlite-query.ipynb              # Requêtes SQLite
│   └── vector-search.ipynb             # Recherche vectorielle exploratoire
├── output/
│   └── compliance_matrix_RFP_bill_generation.xlsx # Fichier de sortie matrice RFP (DONNÉE INTERNE)
├── pipelines/
│   ├── kag/                            # Doublon exact de kag_pipeline/
│   └── rag/                            # Doublon exact de rag_pipeline/ (sauf rag_milvus.py absent ici)
├── rag_pipeline/
│   ├── langchain_rag.py                # Pipeline RAG LCEL (LangChain Expression Language)
│   ├── rag_hotpotqa.py                 # Pipeline RAG minsearch (TF-IDF étiqueté BM25)
│   ├── rag_milvus.py                   # Pipeline RAG vectoriel Milvus Lite + all-MiniLM-L6-v2
│   └── __init__.py
├── .env.example                        # Template de configuration (GROQ_API_KEY, HF_TOKEN)
├── .gitignore                          # Règles d'exclusion Git
├── .python-version                     # 3.12
├── compliance_patch.py                 # Script de patch monkey-patching pour dérive RFP
├── compliance_patch2.py                # Script de patch monkey-patching pour dérive RFP
├── data_loader.py                      # Chargeur minimal HotpotQA (validation[:n_samples])
├── docker-compose.yml                  # Services app, neo4j (5.13), mlflow (2.13.0)
├── Dockerfile                          # Image Python 3.12-slim basée sur uv, lance kag_server.py
├── document_processor.py               # Découpage et extraction PDF, DOCX, TXT
├── faq.db, faq.db-shm, faq.db-wal      # Base SQLite locale
├── fast_server.py                      # Serveur FastAPI v1 (dérive RFP + routeur lexical)
├── fix_globals.py                      # Script utilitaire de correction de variables globales
├── ingest.py                           # Ingestion documents Zoomcamp / FAQ
├── inspect_excel.py                    # Inspection du fichier Excel de conformité
├── kag_main.py                         # Point d'entrée KAG
├── kag_patch_final.py                  # Script de patch pour KAG
├── kag_server.py                       # Serveur FastAPI v2 (orienté benchmark & Docker)
├── load_benchmark_datasets.py          # Doublon racine de evaluation/load_benchmark_datasets.py
├── main.py                             # Stub minimaliste ("Hello from rag-project!")
├── patch_kag.py, patch_kag2.py         # Scripts de patch textuels
├── pyproject.toml                      # Dépendances du projet (uv)
├── rag_helper.py                       # Ancien helper RAG (hérité de LLM Zoomcamp)
├── rag_milvus.py                       # Ancien snippet Milvus de 35 lignes (doublon dégradé)
├── README.md                           # Documentation d'accueil du dépôt
├── results_kag.json                    # Doublon racine de evaluation/results_kag.json
├── run_evaluation.py                   # Doublon racine de evaluation/run_evaluation.py
├── run_full_extraction.py              # Script d'extraction RFP batch
├── show_questions.py                   # Affichage rapide de questions HotpotQA
├── test_extraction.py                  # Test d'extraction sur RFP
├── test_generic_doc.txt                # Document de test générique
├── test_ooredoo_rfp.txt                # RFP confidentiel Ooredoo (DONNÉE INTERNE)
├── test_rfp.txt                        # Extrait RFP (DONNÉE INTERNE)
└── uv.lock                             # Lockfile uv des dépendances résolues
```

### 1.2 Points d'entrée identifiés

1. **`kag_server.py`** : Point d'entrée de production configuré dans le `Dockerfile` (`CMD ["uv", "run", "python", "kag_server.py"]`). C'est le serveur orienté benchmark HotpotQA, RAG Milvus, KAG, Ragas, DeepEval et MLOps.
2. **`fast_server.py`** : Serveur initial ayant divergé vers une plateforme d'extraction de conformité RFP Ooredoo, incluant un routeur lexical et un mode ensemble.
3. **`benchmark/run_benchmark.py`** : Point d'entrée CLI pour lancer le protocole de benchmark standardisé.
4. **`evaluation/run_evaluation.py`** : Point d'entrée CLI alternatif chargeant `results_kag.json` et évaluant RAG minsearch.
5. **`main.py`** : Point d'entrée inopérant (affiche uniquement un message de bienvenue).

---

## 2. Routes FastAPI réelles (Méthode, Chemin, Entrée, Sortie)

Deux serveurs distincts coexistent dans le dépôt :

### 2.1 Serveur de production Docker : `kag_server.py`

| Méthode | Chemin | Entrée (Body / Query / Form) | Sortie (Payload) | Description |
|---|---|---|---|---|
| `GET` | `/` | — | HTML (`kag_app/index.html`) | Interface Web PWA |
| `GET` | `/health` | — | `{"status": "ok", "docs_loaded": int}` | Vérification de santé |
| `GET` | `/stats` | — | `{"kag": dict, "rag": dict, "mlops": dict, "docs_loaded": int}` | Statistiques des pipelines et MLOps |
| `POST` | `/upload` | Form-data: `file` (UploadFile), `pipeline` ("rag"\|"kag"\|"both"), `chunk_size` (int=400) | `{"status": "success", "filename": str, "n_chunks": int, "rag_docs": int, "kag_nodes": int, "kag_rels": int, "preview": str}` | Upload et indexation de fichier |
| `POST` | `/load-hotpotqa` | JSON: `{"n_samples": int}` (défaut 5) | `{"status": "success", "n_samples": int, "rag_docs": int, "kag_nodes": int, "kag_rels": int}` | Chargement HotpotQA via HuggingFace |
| `POST` | `/rag/query` | JSON: `{"question": str}` | `{"answer": str, "context": str, "search_results": list, "latency_ms": float}` | Requête RAG minsearch |
| `POST` | `/kag/query` | JSON: `{"question": str}` | `{"answer": str, "context": str, "search_results": list, "latency_ms": float}` | Requête KAG (graphe) |
| `POST` | `/compare` | JSON: `{"question": str}` | `{"rag": dict, "kag": dict}` | Exécution parallèle RAG vs KAG |
| `POST` | `/evaluate` | JSON: `{"questions": list[str], "ground_truths": list[str], "pipeline": "rag"\|"kag"\|"both"}` | `{"rag"?: dict, "kag"?: dict}` (EM, F1, accuracy) | Évaluation batch sur métriques de base |
| `POST` | `/rag/milvus/query` | JSON: `{"question": str, "top_k": int}` | `{"answer": str, "context": str, "search_results": list, "latency_ms": int, "pipeline": "RAG-Milvus"}` | Requête RAG vectoriel Milvus Lite |
| `POST` | `/evaluate/ragas` | JSON: `{"questions": list, "answers": list, "contexts": list[list[str]], "ground_truths": list, "pipeline": str}` | `{"pipeline": str, "faithfulness": float, "answer_relevancy": float, "context_precision": float, "context_recall": float, "avg_score": float}` | Évaluation via Ragas (Groq LLM) |
| `POST` | `/evaluate/deepeval` | JSON: `{"questions": list, "answers": list, "contexts": list[list[str]], "ground_truths": list, "pipeline": str}` | `{"pipeline": str, "correctness": float, "anti_hallucination": float, "contextual_relevancy": float, "avg_score": float}` | Évaluation via DeepEval (Groq LLM) |
| `POST` | `/benchmark/run` | JSON: `{"n_questions": int, "pipelines": list[str], "skip_ragas": bool}` | `{"status": str, "report": dict, "stdout": str}` | Exécution du sous-processus `run_benchmark.py` |
| `GET` | `/benchmark/last` | — | `dict` (contenu de `results/benchmark_report.json`) | Dernier rapport généré |
| `GET` | `/mlops/stats` | — | `{"total_runs": int, "history": list, "eval_results": dict}` | Télémétrie MLOps |

### 2.2 Serveur legacy / RFP : `fast_server.py`

| Méthode | Chemin | Entrée | Sortie | Rôle |
|---|---|---|---|---|
| `GET` | `/` | — | HTML | UI principale |
| `GET` | `/mobile` | — | HTML | UI mobile |
| `GET` | `/health`, `/stats`, `/info` | — | JSON | Santé et métriques internes |
| `POST` | `/upload` | `file` (UploadFile) | JSON | Upload document RFP |
| `POST` | `/upload-requirements-excel` | `files: list[UploadFile]` | JSON | Ingestion Excel matrice d'exigences |
| `DELETE` | `/upload/clear` | — | JSON | Nettoyage uploads |
| `GET` | `/compliance/criteria`, `/compliance/list`, `/compliance/stats`, `/compliance/export` | Paramètres filtres / format | JSON / CSV / Excel | Gestion de la conformité RFP Ooredoo |
| `POST` | `/rag/query`, `/kag/query`, `/compare` | JSON `QueryReq` / `CompareReq` | JSON | Réponses RAG / KAG / Comparaison |
| `POST` | `/smart` | JSON `{"question": str}` | JSON (`rag` ou `kag` ou `ensemble`) | Requête routée dynamiquement |
| `GET` | `/route` | Query `question: str` | JSON | Diagnostic et classification de la question |
| `GET` | `/routing/stats` | — | JSON | Statistiques du routeur |
| `POST` | `/load-hotpotqa`, `/load-kag`, `/load-kag-precomputed`, `/reset` | JSON / Query | JSON | Gestion des données |
| `GET` | `/kag/graph` | — | `{"nodes": list, "edges": list}` | Visualisation Canvas D3/Cytoscape |

---

## 3. Algorithme de recherche réel dans le RAG (minsearch)

### 3.1 La vérité sur minsearch : TF-IDF et non BM25
Le README, la documentation et les commentaires du code indiquent de façon erronée :
> *"minsearch (BM25 Index)"*

**Constat technique formel :**
La classe `minsearch.Index` (développée par Alexey Grigorev) implémente en réalité :
1. La vectorisation **TF-IDF** via `sklearn.feature_extraction.text.TfidfVectorizer`.
2. Le calcul de score via le produit scalaire / similarité cosinus (`sklearn.metrics.pairwise.linear_kernel`).
3. Elle n'applique **pas** la formule Okapi BM25 (saturation du terme $k_1$ ni normalisation de longueur $b$).

### 3.2 Paramètres exacts dans le code
Dans `rag_pipeline/rag_hotpotqa.py` :
```python
# Initialisation de l'index
self.index = Index(
    text_fields=["title", "text"],
    keyword_fields=[]
)
self.index.fit(self.documents)

# Requête de recherche
boost = {"title": 2.0, "text": 1.0}
results = self.index.search(query, boost_dict=boost, num_results=self.num_results) # num_results=5 par défaut
```
- Champs indexés : `title` (pondération / boost $\times 2.0$), `text` (pondération $\times 1.0$).
- Filtrage par mot-clé : aucun (`keyword_fields=[]`).
- Nombre de passages retournés : $k = 5$ (dans `rag_hotpotqa.py`) ou $k = 10$ (dans `fast_server.py`).

---

## 4. Schéma du graphe en mémoire (KAG)

### 4.1 Structure des nœuds et arêtes
Dans `kag_pipeline/graph_builder.py` (`InMemoryGraphDriver` / `InMemorySession`) :
- **Nœuds :**
  - Clé : `name` (chaîne normalisée)
  - Données : `{"name": str, "type": str, "source_docs": set[str]}`
  - Types de nœuds : Déterminés dynamiquement par le prompt du LLM (`Personne`, `Lieu`, `Organisation`, `Concept`, `Event`), valeur par défaut : `"Concept"`.
- **Arêtes :**
  - Clé de relation : tuple `(source_name, relation_type, target_name)`
  - Données : `{"source": str, "relation": str, "target": str, "source_docs": set[str]}`
  - Type de relation : Chaîne libre extraite par le LLM (ex: `"est né à"`, `"réalisateur"`, `"acteur"`, `"RELATED_TO"`).

### 4.2 Prompt d'extraction d'entités et de relations
Défini dans `kag_pipeline/entity_extraction.py` :
```text
SYSTEM: Vous êtes un expert en extraction d'informations et en graphes de connaissances.
Votre tâche est d'analyser le texte fourni et d'en extraire :
1. Une liste d'entités clés (personnes, lieux, organisations, œuvres, événements, concepts, etc.).
2. Une liste de relations sous forme de triplets (entité_source, relation, entité_cible).
Format JSON strict:
{
  "entities": [{"name": "...", "type": "..."}],
  "relations": [{"source": "...", "relation": "...", "target": "..."}]
}
```
- Modèle utilisé : `llama-3.1-8b-instant` via Groq (température : 0.1, `response_format={"type": "json_object"}`).
- Validation : Schéma Pydantic `ExtractionResult`.

### 4.3 Règles de fusion (Merge)
- **Nœuds :** Correspondance exacte de chaîne sur `name.strip()`. Si le nœud existe déjà, seul le titre du document source `doc_title` est ajouté à l'ensemble `source_docs`. Aucun mécanisme de désambiguïsation, lemmatisation ou résolution d'entités (ex: "Ed Wood" et "Edward Davis Wood Jr." restent deux nœuds séparés).
- **Arêtes :** Correspondance exacte sur le triplet `(source, relation, target)`. Si identique, union de `source_docs`.

### 4.4 Moteur de parcours (Traversée)
Dans `graph_query.py` :
1. Extraction préalable des entités de la question via Groq (`extract_entities_from_question`).
2. Requêtes simulées (ou Cypher si Neo4j connecté) :
   - **1-hop :** `WHERE toLower(e.name) CONTAINS toLower($entity_name)` $\to$ `MATCH (e)-[r:RELATED]-(target) LIMIT 25`
   - **2-hop :** `MATCH (e)-[r1:RELATED]-(m)-[r2:RELATED]-(target) WHERE target <> e LIMIT 25`

---

## 5. Logique exacte du routeur lexical

Implémenté dans `fast_server.py` (`classify_question` et `route_question`) :

### 5.1 Signaux lexicaux et pondérations
Le classifieur analyse les lexèmes et expressions régulières :
1. `MULTIHOP_SIGNALS` (+45 pts KAG par occurrence) : `"same nationality"`, `"also"`, `"were they"`, `"did they"`, `"same country"`, `"same city"`, `"who is also"`, `"both born"`, etc.
2. `RELATIONAL_SIGNALS` (+40 pts KAG par occurrence) : `"father of"`, `"mother of"`, `"founded by"`, `"created by"`, `"directed by"`, `"relationship between"`, `"married to"`, etc.
3. `FACTUAL_SIGNALS` (+35 pts RAG par occurrence) : `"what year"`, `"when was"`, `"how many"`, `"what is the name"`, `"in what year"`, `"how old"`, etc.
4. `BOOLEAN_STARTERS` (+25 pts RAG) : `"were"`, `"was"`, `"is"`, `"are"`, `"did"`, `"does"`, `"have"`, `"can"`, etc.
5. **Nombre d'entités nommées** (mots capitalisés hors mots d'arrêt `STOP_CAPS`) :
   - $\ge 3$ entités : +30 pts KAG
   - 2 entités : +20 pts KAG
   - $\le 1$ entité : +10 pts RAG
6. **Longueur de la question** :
   - $> 18$ mots : +15 pts KAG
   - $\le 8$ mots : +15 pts RAG
7. **Opérateur comparatif "both ... and"** : +25 pts KAG.

### 5.2 Règle d'arbitrage
- Normalisation des scores sur 100 : $RAG_{pct} = \frac{Score_{RAG}}{Score_{RAG} + Score_{KAG}} \times 100$.
- Si signal factuel direct $\to$ Aiguillage vers **RAG**.
- Si multi-hop ou relationnel $\to$ Aiguillage vers **KAG**.
- Si booléen complexe ou scores équilibrés $\to$ Aiguillage vers **Ensemble** (les deux pipelines sont interrogés, et un prompt LLM arbitre retient la meilleure réponse).
- En l'absence de tout signal lexical $\to$ Fallback sur un appel LLM prompté pour classifier en un mot (`multi_hop`, `relational`, `factual`, `boolean`).

> **Observation critique :** Ce routeur est codé uniquement dans `fast_server.py`. Il n'est pas présent dans `kag_server.py` ni intégré dans `benchmark/run_benchmark.py`.

---

## 6. Configuration LightRAG (Vérification dans le code)

- **Statut dans le code :** **TOTALEMENT ABSENT**.
- **Preuve :** Une recherche exhaustive insensible à la casse sur l'intégralité des fichiers du dépôt (`lightrag`, `nano-vectordb`, `graphml`, modes `naive`, `local`, `global`, `hybrid`, `mix`) ne renvoie **strictement aucune occurrence**.
- Aucune bibliothèque correspondante n'est présente dans `pyproject.toml` ou `uv.lock`.

---

## 7. Configuration Ragas et DeepEval

### 7.1 Ragas (`evaluation/ragas_evaluator.py`)
- **Version installée :** `ragas>=0.4.3`.
- **LLM Juge :** Groq `llama-3.1-8b-instant` interfacé via `langchain_openai.ChatOpenAI(base_url="https://api.groq.com/openai/v1")` enveloppé dans `LangchainLLMWrapper`.
- **Embeddings :** `sentence-transformers/all-MiniLM-L6-v2` via `HuggingFaceEmbeddings` + `LangchainEmbeddingsWrapper`.
- **Métriques configurées :**
  - `faithfulness` (Fidélité au contexte)
  - `answer_relevancy` (Pertinence de la réponse à la question)
  - `context_precision` (Précision des passages récupérés)
  - `context_recall` (Complétude des faits par rapport à la vérité terrain)
- **Patch de contournement :** Présence d'un mock injecté dynamiquement dans `sys.modules` (lignes 20-36) pour court-circuiter des dépendances manquantes relatives à Google Cloud VertexAI requises accidentellement par `langchain_community`.

### 7.2 DeepEval (`evaluation/deepeval_evaluator.py`)
- **Version installée :** `deepeval>=4.1.4`.
- **LLM Juge :** Groq `llama-3.1-8b-instant` configuré via variables d'environnement OpenAI (`OPENAI_API_BASE=https://api.groq.com/openai/v1`).
- **Métriques configurées :**
  - `GEval(name="Correctness")` : Évalue si la réponse traite correctement la question au vu de la vérité terrain.
  - `HallucinationMetric(threshold=0.5)` : Score d'anti-hallucination.
  - `ContextualRelevancyMetric(threshold=0.5)` : Pertinence contextuelle des passages.

---

## 8. Données HotpotQA, taille et représentativité du corpus

### 8.1 Partition et sous-ensembles
- Dans `data_loader.py` :
  ```python
  dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split=f"validation[:{n_samples}]")
  ```
- **Configuration par défaut :** Prend les $N$ premiers exemples du split `validation` de manière séquentielle (`[:5]` ou `[:200]`).
- **Absence de graine aléatoire :** Aucun `seed` n'est fixé lors de l'extraction, et aucun sous-ensemble figé n'est sérialisé sur disque dans un format neutre (JSONL / Parquet).
- **Corpus commun :** Les documents sont extraits à la volée depuis le champ `context` de chaque item HotpotQA (10 paragraphes par question : 2 paragraphes d'or + 8 distracteurs).
- **Questions négatives :** Aucune question sans réponse ("information absente" / unanswerable) n'est incluse dans les scripts actuels.

---

## 9. Inventaire de la stack prévue : Présent vs Absent

| Technologie | Statut dans le dépôt | Preuve / Localisation dans le code |
|---|---|---|
| **Python** | **Présent** | Python 3.12 (`.python-version`, `pyproject.toml`) |
| **FastAPI** | **Présent** | `fast_server.py`, `kag_server.py` |
| **Docker / Compose** | **Présent** | `Dockerfile`, `docker-compose.yml` (services app, neo4j, mlflow) |
| **Groq (LLaMA 3.1 8B)** | **Présent** | Appels API Groq dans tous les pipelines |
| **Hugging Face** | **Présent** | `datasets`, `sentence-transformers/all-MiniLM-L6-v2` |
| **Milvus** | **Partiel** | `milvus-lite>=3.1.1`, fichier `rag_pipeline/rag_milvus.py` utilisant `MilvusClient("./milvus_rag.db")`. Aucun serveur Milvus standalone dans Docker. |
| **Neo4j** | **Partiel** | Défini dans `docker-compose.yml` (`neo4j:5.13-community`), mais désactivé volontairement dans `kag_server.py` (`driver = InMemoryGraphDriver()`). |
| **Langfuse** | **Partiel** | Dépendance `langfuse>=4.14.1` présente, wrapper dans `mlops/tracker.py`, mais aucun conteneur Langfuse dans `docker-compose.yml`. |
| **MLflow** | **Présent** | Dépendance `mlflow>=3.14.0`, conteneur dans `docker-compose.yml`, instrumenté dans `mlops/tracker.py`. |
| **LangGraph** | **Partiel** | `langgraph>=1.2.9` dans `pyproject.toml`, prototype 3 nœuds dans `kag_pipeline/langgraph_kag.py`, mais non relié aux routes API ni au benchmark. |
| **LangChain** | **Présent** | `rag_pipeline/langchain_rag.py` (chaîne LCEL) |
| **Ragas** | **Présent** | `evaluation/ragas_evaluator.py` |
| **DeepEval** | **Présent** | `evaluation/deepeval_evaluator.py` |
| **ChromaDB** | **Absent** | Aucune mention, non présent dans `pyproject.toml`. |
| **Qdrant** | **Absent** | Aucune mention, non présent dans `pyproject.toml`. |
| **LlamaIndex** | **Absent** | Aucune mention fonctionnelle, non présent dans `pyproject.toml`. |
| **LightRAG** | **Absent** | Introuvable dans l'ensemble du dépôt. |

---

## 10. Résultats existants et stockage

Les seuls résultats mesurés et persistés se trouvent dans `results_kag.json` (et sa copie `evaluation/results_kag.json`) :
- Échantillon : **5 questions** issues de HotpotQA.
- Contenu : Question, vérité terrain, prédiction KAG, triplets de contexte.
- **Bilan des scores réels enregistrés :**
  - Q1 (Nationalité Scott Derrickson / Ed Wood) : KAG répond *"I don't know"* (Vérité : *"yes"*) $\to$ **EM = 0.0, F1 = 0.0**.
  - Q2 (Position Shirley Temple) : KAG répond *"ambassadeur des États-Unis"* (Vérité : *"Chief of Protocol"*) $\to$ **EM = 0.0, F1 = 0.08**.
  - Q3 (Série Animorphs) : KAG répond *"Dune... I don't know"* (Vérité : *"Animorphs"*) $\to$ **EM = 0.0, F1 = 0.0**.
  - Q4 (Laleli Mosque / Esma Sultan) : KAG répond *"Oui, situés dans le même quartier"* (Vérité : *"no"*) $\to$ **EM = 0.0, F1 = 0.0**.
  - Q5 (Big Stone Gap) : KAG répond *"I don't know"* (Vérité : *"Greenwich Village, New York City"*) $\to$ **EM = 0.0, F1 = 0.0**.
- **Constat sur les chiffres du README :**
  Le README mentionne *Exact Match KAG: 20%*, obtenu uniquement via une fonction de normalisation permissive dans `fast_server.py`. Le dossier `results/` prévu pour archiver les runs CSV n'existe pas encore.

---

## 11. Problèmes identifiés, dette technique et confidentialité

### 11.1 Risques de confidentialité et fuite de données
1. **Documents internes Ooredoo présents :**
   - `test_ooredoo_rfp.txt` : Document interne portant explicitement la mention *"All information in connection with this RFP and project is strictly confidential"*.
   - `test_rfp.txt` : Extrait direct de l'appel d'offres.
   - `output/compliance_matrix_RFP_bill_generation.xlsx` : Matrice de conformité interne générée.
   > **Action requise :** Ces fichiers doivent être exclus de toute expérimentation, ignorés ou purgés pour garantir l'étanchéité stricte sur HotpotQA.

### 11.2 Biais et fabrications algorithmiques
1. **Overfitting manifeste dans les métriques :**
   Dans `fast_server.py` (ligne 485) :
   ```python
   YES_SYNONYMS = {"yes", "yeah", "yep", "both", "same", "correct", "true", "american", "indeed"}
   ```
   L'injection du mot `"american"` parmi les synonymes de `"yes"` a été introduite spécifiquement pour faire passer la réponse à la question Scott Derrickson ("both American" $\to$ "yes") à 1.0 en Exact Match.
2. **Allégation mensongère "BM25" :** minsearch est un TF-IDF cosinus scikit-learn, pas un algorithme BM25 probabiliste.

### 11.3 Dette architecturale
1. **Dédoublement anarchique des fichiers :** Doublons stricts entre racine et sous-dossiers (`run_evaluation.py`, `results_kag.json`, `pipelines/` vs `rag_pipeline/` & `kag_pipeline/`).
2. **Gestion fragile du Rate Limit Groq :** De simples `time.sleep(2)` qui ne protègent pas contre les dépassements de quota en cas de batching important. Aucun système de mise en cache locale (ex: SQLite / pickle) des requêtes Groq.
3. **Bypass de Neo4j :** Le serveur principal `kag_server.py` instancie directement `InMemoryGraphDriver()` pour éviter des lenteurs de connexion TCP, rendant le conteneur Neo4j inactif en pratique.
4. **Environnement d'exécution local :** L'outil `git` n'est pas configuré dans le PATH du shell Windows hôte, et WSL2 n'est pas activé sur la machine.

---

## 12. Tableau de conformité : Travaux de la fiche vs Réalité

| Travaux de la fiche de stage | Statut | Preuve dans le code |
|---|---|---|
| **1. Étude approfondie des architectures RAG et KAG** | **Partiel** | `docs/architecture.md` (très succinct, 80 lignes), code implémenté dans `rag_pipeline/` et `kag_pipeline/`. |
| **2. Avantages, limites et cas d'usage de chaque approche** | **Partiel** | Mentionné dans README et `classify_question()` (`_risk_consequences`), mais pas de document d'analyse formel. |
| **3. Corpus de données commun aux expérimentations** | **Manquant** | Chargement à la volée `validation[:n]`, pas de sous-ensemble commun sérialisé, pas de graine fixe, pas de négatifs. |
| **4. POC RAG basé sur plusieurs bases vectorielles** | **Partiel** | Présence de minsearch (lexical TF-IDF) et Milvus Lite (`rag_pipeline/rag_milvus.py`). ChromaDB et Qdrant sont totalement absents. Pas de classe abstraite `VectorStore`. |
| **5. POC KAG basé sur un graphe de connaissances** | **Fait** | Extraction LLM Pydantic (`entity_extraction.py`), requêtes 1-hop & 2-hop (`graph_query.py`), stockage mémoire et Neo4j (`graph_builder.py`). |
| **6. Comparaison de différentes technologies de stockage et de recherche** | **Partiel** | Ébauche dans `benchmark/run_benchmark.py` (minsearch vs Milvus Lite vs Graphe mémoire). Neo4j non benchmarké activement. |
| **7. Protocole de benchmark standardisé** | **Partiel** | `benchmark/run_benchmark.py` existe mais ne gère que 3 configurations, sans sauvegarde CSV détaillée ni graine aléatoire. |
| **8. Mesure des performances selon plusieurs critères** | **Partiel** | EM, F1 et latences implémentés. Ragas et DeepEval écrits mais pas intégrés de manière robuste et automatisée. Pas de mesure CPU/RAM ni tokens réels. |
| **9. Recommandations d'architecture adaptées aux besoins métier** | **Manquant** | Aucune recommandation étayée par des résultats chiffrés rigoureux à ce jour (`RECOMMANDATIONS.md` absent). |

---

## 13. Plan d'action priorisé pour la Phase 2

Voici la feuille de route méthodique recommandée, strictement alignée avec le cahier des charges :

```
[Phase 2.A] Corpus commun HotpotQA figé & standardisé
     │
[Phase 2.B] Abstraction VectorStore unifiée (minsearch TF-IDF, BM25 réel, ChromaDB, Milvus)
     │
[Phase 2.C] Consolidation KAG & Intégration LightRAG / Neo4j
     │
[Phase 2.D] Moteur de Benchmark standardisé multi-configurations & Cache LLM
     │
[Phase 2.E] Évaluation complète (EM, F1, Ragas, DeepEval, Latences, Ressources)
     │
[Phase 2.F] Observabilité & Traces (Langfuse / Logs structurés)
     │
[Phase 2.G] Visualisation du Graphe de connaissances & Export
     │
[Phase 2.H] API Unifiée & Endpoint de comparaison côte à côte
     │
[Livrables] ECARTS.md, BENCHMARK.md, RECOMMANDATIONS.md, results/*.csv
```

### Détail des actions :
1. **Étape A — Corpus commun HotpotQA :** Créer un script reproductible extrayant un jeu figé (ex. 100 questions avec seed fixe), incluant 10% de questions "information absente" pour évaluer la robustesse au refus d'hallucination, sauvegardé en JSONL/CSV partagé par toutes les configurations.
2. **Étape B — Architecture RAG multi-stores :** Définir une interface abstraite `BaseVectorStore` avec 3 implémentations comparables (minsearch TF-IDF, vrai BM25 `rank_bm25`, ChromaDB ou Qdrant, et Milvus Lite), partageant strictement le même modèle d'embedding (`all-MiniLM-L6-v2`) et la même taille de chunk.
3. **Étape C — Consolidation KAG & Clarification LightRAG :**
   - Maintenir le graphe en mémoire + Neo4j.
   - Intégrer l'implémentation de LightRAG (nano-vectordb + modes de parcours naive/local/global/hybrid) ou acter formellement l'écart dans `ECARTS.md`.
4. **Étape D — Script unique de benchmark (`run_benchmark`) :**
   - Mise en cache SQLite des appels LLM Groq avec backoff exponentiel pour éviter tout blocage de rate limit (429).
   - Exécution séquentielle de chaque stratégie seule, puis avec le routeur d'arbitrage.
   - Sauvegarde systématique dans `results/` au format CSV horodaté avec paramètres de configuration et graine.
5. **Étape E — Métriques étendues :** Calcul automatique d'Exact Match strict, Token F1, Ragas, DeepEval, latence p50/p95, tokens estimés et consommation mémoire.
6. **Étape F — Nettoyage & Sécurité :** Isoler et archiver les fichiers de patchs RFP internes hors du flux benchmark.
7. **Étape G — Documentation et Recommandations :** Rédaction de `ECARTS.md`, `BENCHMARK.md`, et `RECOMMANDATIONS.md` uniquement à partir des données réellement collectées.
