# ⚖️ RAPPORT DES ÉCARTS — PRÉVU vs RÉALISÉ (ECARTS.md)
**Sujet 03 :** Étude Comparative et Benchmarking des Architectures RAG et KAG  
**Organisme :** Ooredoo AI Lab  
**Date :** 30 septembre 2026  

---

## 1. Synthèse globale des écarts

Le tableau ci-dessous dresse l'état précis entre les technologies prévues dans la fiche de stage officielle et leur réalisation concrète dans la plateforme de benchmarking.

| Composant de la Fiche | Prévu | Réalisé | Justification technique & Choix d'ingénierie |
|---|---|---|---|
| **Dataset d'expérimentation** | Non spécifié (Corpus commun) | **HotpotQA** (validation distractor, 50 questions + 5 négatifs "information absente") | HotpotQA est le benchmark académique de référence pour tester le raisonnement multi-sauts (multi-hop) et les relations entre entités. Ajout de questions sans réponse pour mesurer la robustesse au refus d'hallucination. |
| **RAG - Moteur Lexical** | "Recherche documentaire" | **minsearch (TF-IDF)** et **Okapi BM25 probabiliste** | minsearch utilisait un TF-IDF cosinus scikit-learn faussement documenté comme BM25. Ajout d'une véritable classe `BM25Store` pour une comparaison loyale TF-IDF vs BM25. |
| **RAG - Moteur Vectoriel** | Milvus | **Milvus Lite** + **Dense Vector Store** | Milvus Lite permet une exécution embarquée native locale (sur fichier SQLite) sans exiger le déploiement lourd d'un cluster Milvus Standalone sous Docker. |
| **KAG - Graphe de connaissances** | Neo4j | **InMemoryGraphDriver** (avec connecteur Neo4j Cypher prêt) | Le driver en mémoire élimine la latence réseau et les timeouts TCP de Neo4j en phase de prototypage rapide, tout en conservant la sémantique de traversée Cypher 1-hop et 2-hop. |
| **LightRAG** | "Architectures KAG" | **LightRAGEngine (5 modes : naive, local, global, hybrid, mix)** + export GraphML | Développé pour comparer les différents modes d'exploration de graphe combinés à la recherche vectorielle. Export au format standard GraphML (.graphml). |
| **Orchestration d'Agents** | LangGraph, LangChain | **KAGLangGraph** (3 nœuds) et **RAGLangChain** (chaîne LCEL) | Implémentés dans `kag_pipeline/langgraph_kag.py` et `rag_pipeline/langchain_rag.py`. |
| **LlamaIndex** | Prévu dans la stack | **Remplacé par l'interface commune `BaseVectorStore`** | LlamaIndex introduisait une lourdeur de dépendances (conflits de versions avec LangChain). Remplacé par une abstraction `BaseVectorStore` standardisée garantissant le même $k$, le même chunking et les mêmes métriques pour tous les moteurs. |
| **Évaluation LLM-based** | Ragas, DeepEval | **Modules Ragas & DeepEval intégrés** + métriques canoniques EM/F1 | Implémentés avec Groq `llama-3.1-8b-instant`. Ragas et DeepEval nécessitent une clé API active pour les appels aux juges LLM ; les métriques déterministes (Exact Match, Token F1, latence p50/p95, robustesse refus) ont été automatisées pour garantir une reproductibilité immédiate sans dépendre des quotas d'API. |
| **Observabilité / MLOps** | Langfuse, MLflow | **MLflow + Tracker unifié + Télémétrie intégrée** | MLflow configuré dans `docker-compose.yml`. Le module `MLOpsTracker` permet de tracer les runs avec estimation de tokens et latences. En l'absence de serveur distant Langfuse, les logs structurés locaux assurent l'auditabilité complète. |
| **Dérive RFP Ooredoo** | Hors sujet initial | **Isolée et écartée du benchmark HotpotQA** | Des scripts de patchs internes RFP (`test_ooredoo_rfp.txt`, `compliance_patch.py`) avaient altéré le serveur principal. Le benchmark a été recentré exclusivement sur HotpotQA afin de respecter la stricte confidentialité des données internes. |

---

## 2. Analyse détaillée des arbitrages d'ingénierie

### 2.1 Pourquoi Milvus Lite plutôt qu'un cluster Milvus Docker ?
Un cluster Milvus distribué requiert etcd, MinIO et Pulsar, consommant plusieurs gigaoctets de mémoire RAM. Milvus Lite (`pymilvus`) offre la même API cliente avec un moteur local embarqué. Cela garantit une portabilité totale sur PC portable, WSL2 ou conteneur léger.

### 2.2 Pourquoi InMemoryGraphDriver en complément de Neo4j ?
L'accès à une instance Neo4j nécessite un service actif sur le port 7687 avec authentification. Lorsque l'environnement d'exécution n'a pas démarré le démon Docker Neo4j, le fallback automatique `InMemoryGraphDriver` prend le relais instantanément, empêchant tout plantage de l'application.

### 2.3 Clarification sur minsearch vs BM25
Le projet initial qualifiait minsearch de "BM25". Techniquement, `minsearch` utilise un modèle TF-IDF avec similarité cosinus. Nous avons formellement introduit un moteur Okapi BM25 (`BM25Store`) afin de comparer rigoureusement un modèle vectoriel TF-IDF et un modèle probabiliste BM25 avec saturation du terme et normalisation de longueur.
