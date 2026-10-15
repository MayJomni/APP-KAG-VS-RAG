# Protocole de Benchmark Standardisé — RAG vs KAG

**Projet :** Plateforme de Benchmarking RAG vs KAG  
**Contexte :** Stage Ooredoo Tunisie — Sujet 03  
**Date d'exécution :** 2026-09-30  
**Graine aléatoire (seed) :** 42  
**Corpus de test :** HotpotQA multi-hop (24 documents sources, 13 questions étalonnées)

---

## 1. Objectifs & Cadre Méthodologique

Ce protocole standardisé a pour but de mesurer rigoureusement et de comparer les performances des architectures **RAG** (Retrieval-Augmented Generation) et **KAG** (Knowledge-Augmented Generation), ainsi que des approches hybrides et routées, sur un jeu d'évaluation exigeant des raisonnements multi-sauts (*multi-hop reasoning*).

### Règles d'or du benchmark :
1. **Zéro fabrication :** Toutes les mesures proviennent d'exécutions effectives enregistrées dans `results/benchmark_runs_20260930_232533.csv` et `results/benchmark_report.json`.
2. **Corpus commun :** Les 11 configurations sont évaluées exactement sur les mêmes 24 documents et 13 questions.
3. **Reproductibilité totale :** Graine fixée à 42, partition figée dans `data/corpus_documents.json` et `data/benchmark_questions.json`.
4. **Confidentialité absolue :** Données 100% publiques (HotpotQA), aucune donnée interne, aucune clé API dans les journaux ou le code.

---

## 2. Jeu de Données Étalon (Benchmark Dataset)

Le jeu de test est constitué de **13 questions** représentatives :
- **10 questions multi-hop HotpotQA** combinant :
  - Type *Bridge* (ex: trouver l'entité intermédiaire liant deux faits, ex: réalisateurs, dates de sortie, relations familiales).
  - Type *Comparison* (ex: comparer des faits ou caractéristiques entre deux entités).
- **3 questions négatives (hors corpus)** : questions conçues pour vérifier la capacité du système à refuser d'halluciner lorsque l'information est absente (*"Information absente du corpus"*).

Le corpus contient **24 documents textuels** extraits du dataset HotpotQA, couvrant le cinéma, la musique, l'histoire et la géographie.

---

## 3. Configurations Testées (11 Architectures)

| # | Configuration | Catégorie | Mécanisme de Recherche / Stockage |
|---|---|---|---|
| 1 | **RAG-minsearch** | RAG Lexical | Vectorisation TF-IDF + similarité cosinus (`minsearch`) |
| 2 | **RAG-BM25** | RAG Lexical | Algorithme probabiliste Okapi BM25 standard ($k_1=1.5, b=0.75$) |
| 3 | **RAG-Dense-Vector** | RAG Vectoriel | Embeddings denses sémantiques (Hugging Face / DenseStore) |
| 4 | **RAG-Milvus-Lite** | RAG Vectoriel | Index vectoriel compatible Milvus Lite |
| 5 | **KAG-In-Memory-Graph** | KAG Graphe | Graphe de connaissances en mémoire (201 nœuds, 239 arêtes, Gemma 2 9B) |
| 6 | **LightRAG-Naive** | Hybride (Baseline) | Recherche documentaire plein texte pure |
| 7 | **LightRAG-Local** | KAG Local | Voisinage de graphe à 1-hop et 2-hop autour des entités détectées |
| 8 | **LightRAG-Global** | KAG Global | Résumés relationnels globaux et communautés d'entités |
| 9 | **LightRAG-Hybrid** | KAG Hybride | Fusion pondérée Graphe Local (entités/relations) + Chunks textuels |
| 10 | **LightRAG-Mix** | KAG Triplets | Concaténation explicite triplets de graphe + documents sources |
| 11 | **Routeur-Lexical-Smart**| Agentique | Routage par règles analysant la structure de la question (graphe vs texte) |

---

## 4. Métriques d'Évaluation

1. **Exact Match (EM) :** Mesure binaire stricte (après normalisation : minuscules, suppression ponctuation et articles). 1.0 si la réponse prédite contient exactement la réponse de référence.
2. **Token F1-Score :** Moyenne harmonique de la précision et du rappel calculée sur l'intersection des sacs de mots (tokens) entre prédiction et vérité terrain.
3. **Latence d'Exécution :**
   - **Moyenne :** temps de traitement moyen par requête (retrieval + extraction).
   - **p50 (Médiane) :** percentile 50.
   - **p95 :** percentile 95 pour identifier les requêtes lentes.
4. **Robustesse au Refus (Taux d'abstention correct) :** Pourcentage de questions négatives correctement identifiées comme non documentées dans le corpus.
5. **Consommation de Tokens :** Nombre total de tokens consommés (contexte injecté + réponse) sur l'ensemble du benchmark.

---

## 5. Résultats Obtenus (Données Réelles et Vérifiées)

*Exécution du benchmark le 2026-09-30 à 23:25:33 (Graine 42) :*

| Configuration | Exact Match (EM) | Token F1 | Latence Moyenne | Latence p50 | Latence p95 | Robustesse Refus | Tokens Consommés |
|---|---|---|---|---|---|---|---|
| **Routeur-Lexical-Smart** | **53.8%** | **37.0%** | 0.2 ms | 0.0 ms | 2.02 ms | **100.0%** | 3 807 |
| **LightRAG-Hybrid** | **53.8%** | **37.0%** | 0.6 ms | 0.0 ms | 4.18 ms | **100.0%** | 3 585 |
| **LightRAG-Local** | **53.8%** | 29.3% | 0.3 ms | 0.0 ms | 3.91 ms | **100.0%** | 2 381 |
| **RAG-BM25 (Okapi)** | 46.2% | 29.3% | 0.2 ms | 0.0 ms | 2.06 ms | **100.0%** | 4 486 |
| **RAG-minsearch (TF-IDF)** | 46.2% | 29.3% | 1.3 ms | 0.69 ms | 6.42 ms | **100.0%** | 4 505 |
| **LightRAG-Naive** | 46.2% | 29.3% | 0.0 ms | 0.0 ms | 0.0 ms | **100.0%** | 4 718 |
| **LightRAG-Mix** | 46.2% | 29.3% | 0.3 ms | 0.0 ms | 2.08 ms | **100.0%** | 5 247 |
| **LightRAG-Global** | 38.5% | 33.8% | 0.0 ms | 0.0 ms | 0.0 ms | **100.0%** | 2 307 |
| **KAG-In-Memory-Graph** | 30.8% | 23.1% | **0.0 ms** | 0.0 ms | 0.0 ms | **100.0%** | **384** |
| **RAG-Dense-Vector** | 15.4% | 12.8% | 0.2 ms | 0.0 ms | 2.01 ms | **100.0%** | 4 276 |
| **RAG-Milvus-Lite** | 15.4% | 12.8% | 0.0 ms | 0.0 ms | 0.0 ms | **100.0%** | 4 276 |

---

## 6. Analyse des Résultats

### 6.1 Supériorité de l'Hybridation KAG + RAG
- **LightRAG-Hybrid** et le **Routeur Smart** obtiennent les meilleurs scores d'Exact Match (**53.8%**) et de Token F1 (**37.0%**).
- Sur les questions *multi-hop*, la recherche purement lexicale ou vectorielle échoue souvent à récupérer le second document si les mots-clés de la question ne mentionnent pas directement la deuxième entité.
- Le graphe de connaissances relie explicitement les entités par des arêtes relationnelles (ex: `[Film] --directed_by--> [Personne] --born_in--> [Lieu]`), permettant à **LightRAG-Local** d'atteindre **53.8% EM** avec seulement 2 381 tokens (contre 4 505 tokens pour le RAG lexical).

### 6.2 Efficacité Énergétique et Économie de Contexte
- Le graphe de connaissances pur (**KAG-In-Memory-Graph**) consomme seulement **384 tokens** au total, soit **11.7 fois moins** que le RAG TF-IDF (4 505 tokens).
- Cependant, sans documents textuels complémentaires, le graphe pur souffre d'un manque de couverture textuelle fine (30.8% EM).

### 6.3 Robustesse aux Hallucinations (Questions Négatives)
- Les 11 configurations atteignent **100.0% de robustesse au refus**.
- Lorsque les scores de pertinence ou les entités ne sont pas trouvés dans la base, le système renvoie rigoureusement l'indication d'absence d'information, évitant ainsi toute hallucination.

---

## 7. Reproductibilité & Commandes

Pour rejouer l'ensemble du protocole et générer les rapports :

```bash
# 1. Activation de l'environnement virtuel
.venv\Scripts\activate

# 2. Exécution du benchmark complet (graine 42)
python benchmark/run_benchmark.py --seed 42

# 3. Les résultats sont automatiquement écrits dans :
# - results/benchmark_runs_<TIMESTAMP>.csv (détail question par question)
# - results/benchmark_report.json (synthèse JSON)
# - results/benchmark_summary.md (tableau markdown)
# - results/lightrag_graph.graphml (graphe exporté)
```
