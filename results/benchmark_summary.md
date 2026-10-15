# 📊 Tableau Comparatif Standardisé — RAG vs KAG
**Date :** 20260930_232533 | **Questions :** 13 | **Corpus :** 24 docs

| Configuration | Exact Match (EM) | Token F1 | Latence Moyenne | Latence p50 | Latence p95 | Robustesse Refus | Tokens |
|---|---|---|---|---|---|---|---|
| **RAG-minsearch (TF-IDF)** | 46.2% | 29.3% | 1.3 ms | 0.69 ms | 6.42 ms | 100.0% | 4505 |
| **RAG-BM25 (Okapi)** | 46.2% | 29.3% | 0.2 ms | 0.0 ms | 2.06 ms | 100.0% | 4486 |
| **RAG-Dense-Vector** | 15.4% | 12.8% | 0.2 ms | 0.0 ms | 2.01 ms | 100.0% | 4276 |
| **RAG-Milvus-Lite** | 15.4% | 12.8% | 0.0 ms | 0.0 ms | 0.0 ms | 100.0% | 4276 |
| **KAG-In-Memory-Graph** | 30.8% | 23.1% | 0.0 ms | 0.0 ms | 0.0 ms | 100.0% | 384 |
| **LightRAG-Naive** | 46.2% | 29.3% | 0.0 ms | 0.0 ms | 0.0 ms | 100.0% | 4718 |
| **LightRAG-Local** | 53.8% | 29.3% | 0.3 ms | 0.0 ms | 3.91 ms | 100.0% | 2381 |
| **LightRAG-Global** | 38.5% | 33.8% | 0.0 ms | 0.0 ms | 0.0 ms | 100.0% | 2307 |
| **LightRAG-Hybrid** | 53.8% | 37.0% | 0.6 ms | 0.0 ms | 4.18 ms | 100.0% | 3585 |
| **LightRAG-Mix** | 46.2% | 29.3% | 0.3 ms | 0.0 ms | 2.08 ms | 100.0% | 5247 |
| **Routeur-Lexical-Smart** | 53.8% | 37.0% | 0.2 ms | 0.0 ms | 2.02 ms | 100.0% | 3807 |