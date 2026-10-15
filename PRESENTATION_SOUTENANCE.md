# SUPPORT DE PRÉSENTATION POUR SOUTENANCE DE STAGE (15 SLIDES)

**Sujet 03 :** Étude comparative et benchmarking des architectures RAG et KAG  
**Organisme d'accueil :** Ooredoo Tunisie — Direction des Systèmes d'Information & AI Lab  
**Durée conseillée :** 20 minutes d'exposé + 10 minutes de questions  

---

## 📽️ SLIDE 1 : Titre et Contexte
- **Titre :** Étude Comparative et Benchmarking des Architectures RAG et KAG
- **Sous-titre :** Conception de POCs, Évaluation Standardisée sur HotpotQA et Recommandations Métier
- **Candidat :** Ingénieur Stagiaire en Intelligence Artificielle
- **Encadrants :** Responsables AI Lab Ooredoo Tunisie
- **Note de présentation (Oral) :**
  > *"Bonjour à tous. Dans le cadre de mon stage au sein de l'AI Lab d'Ooredoo Tunisie, j'ai mené une étude comparative et expérimentale approfondie pour répondre à un défi stratégique : comment exploiter au mieux la puissance des LLMs en entreprise sans subir leurs limites d'hallucination et d'amnésie ? Aujourd'hui, je vais vous présenter notre démarche, nos POCs opérationnels et les résultats concrets de notre benchmark opposant RAG et KAG."*

---

## 📽️ SLIDE 2 : Contexte Télécoms & Problématique
- **Les enjeux Ooredoo :** Gestion documentaire massive (appels d'offres RFP, contrats partenaires, support technique 5G, procédures RH).
- **Les 3 limites critiques des LLMs généralistes :**
  1. *Hallucinations :* Réponses fausses mais convaincantes (risque opérationnel majeur).
  2. *Connaissances figées :* Incapacité d'intégrer les données propriétaires en temps réel.
  3. *Angle mort du raisonnement multi-sauts (Multi-hop) :* Incapacité à relier des faits dispersés.
- **La question de recherche :** RAG textuel ou KAG relationnel : quelle architecture déployer pour chaque cas d'usage chez l'opérateur ?

---

## 📽️ SLIDE 3 : RAG vs KAG — Deux Paradigmes Complémentaires
- **RAG (Retrieval-Augmented Generation) :**
  - Principe : Découpage en passages textuels (*chunks*), indexation vectorielle, recherche par similarité.
  - Avantages : Déploiement rapide, instantané à l'ingestion, idéal pour faits directs.
  - Limite : Incapable de déduire des liens implicites entre documents séparés.
- **KAG (Knowledge-Augmented Generation) :**
  - Principe : Extraction de triplets sémantiques `(Sujet, Relation, Objet)` formant un graphe de connaissances.
  - Avantages : Raisonnement multi-sauts déterministe, explicabilité, économie massive de contexte.
  - Limite : Coût d'extraction initial via LLM.

---

## 📽️ SLIDE 4 : Cahier des Charges & Travaux Réalisés
| Travaux de la Fiche de Stage | Statut | Réalisation Concrète |
|---|:---:|---|
| 1. Étude comparative approfondie | ✅ Fait | Rapports `AUDIT.md` & `RAPPORT_PFE_STRUCTURE.md` |
| 2. Avantages, limites et cas d'usage | ✅ Fait | Arbre décisionnel dans `RECOMMANDATIONS.md` |
| 3. Corpus commun étalonné | ✅ Fait | 24 documents & 13 questions HotpotQA (`seed=42`) |
| 4. POC RAG multi-bases vectorielles | ✅ Fait | Interface `BaseVectorStore` (BM25, minsearch, Dense, Milvus) |
| 5. POC KAG basé sur un graphe | ✅ Fait | Graphe en mémoire (201 nœuds, 239 arêtes) + export GraphML |
| 6. Comparaison des stockages | ✅ Fait | Évaluation combinée lexicale, vectorielle et graphe |
| 7. Protocole de benchmark standardisé | ✅ Fait | Script reproductible `run_benchmark.py` |
| 8. Mesure des performances | ✅ Fait | Exact Match, Token F1, Latence p95, Robustesse Refus, Tokens |
| 9. Recommandations d'architecture | ✅ Fait | Matrice décisionnelle pour Ooredoo |

---

## 📽️ SLIDE 5 : Le Protocole Expérimental (HotpotQA)
- **Pourquoi HotpotQA ?** Dataset de référence conçu spécifiquement pour le raisonnement multi-sauts (contrairement à SQuAD qui est purement lexical).
- **Structure du jeu de données :**
  - **24 documents textuels de référence** (extraits équilibrés cinéma, histoire, géographie, télécoms).
  - **10 questions multi-hop :** Types *Bridge* (entité intermédiaire) et *Comparison*.
  - **3 questions négatives (pièges hors-corpus) :** Validation formelle du taux d'abstention ("Information absente du corpus").
- **Règle absolue :** Zéro fabrication de données, graine aléatoire fixe (`seed=42`).

---

## 📽️ SLIDE 6 : Architecture Globale du Système
- *Afficher le Diagramme de Composants UML en 5 couches :*
  1. *Couche Présentation :* Interface Web SPA responsive (HTML5 / Canvas).
  2. *Couche Contrôle :* API REST asynchrone FastAPI (`fast_server.py`).
  3. *Couche Routage :* Routeur Intelligent adaptatif (analyse d'intention).
  4. *Couche Moteurs :* Sous-système RAG multi-vectoriel + Moteur LightRAG (5 modes).
  5. *Couche Inférence :* Inférence accélérée Groq (Gemma 2 9B / LLaMA 3.1) + Synthétiseur local autonome.

---

## 📽️ SLIDE 7 : POC RAG Multi-Bases Vectorielles
- Utilisation du patron de conception **Stratégie (Strategy Pattern)** via `BaseVectorStore`.
- **4 implémentations concrètes évaluées :**
  - `BM25Store` : Algorithme probabiliste conforme Okapi BM25 ($k_1=1.5, b=0.75$).
  - `MinSearchStore` : TF-IDF avec produit scalaire / similarité cosinus.
  - `DenseVectorStore` : Plongements denses sémantiques (Hugging Face).
  - `MilvusLiteStore` : Indexation vectorielle compatible Milvus.

---

## 📽️ SLIDE 8 : POC KAG & LightRAG (5 Modes de Requêtage)
- **Graphe de Connaissances :** 201 entités nommées, 239 relations sémantiques.
- **Les 5 Modes d'Interrogation LightRAG :**
  1. *Naive :* Recherche plein texte (baseline).
  2. *Local :* Extraction du voisinage de graphe 1-hop et 2-hop autour des entités.
  3. *Global :* Résumés des communautés et clusters de relations.
  4. *Hybrid :* Fusion pondérée Graphe Local + Extraits textuels documentaires.
  5. *Mix :* Concaténation explicite triplets + passages bruts.

---

## 📽️ SLIDE 9 : Démonstration de la Plateforme Web (`http://localhost:8000`)
- **Onglet 💬 Comparer :** Comparaison côte à côte RAG vs KAG avec citation directe des sources et temps d'exécution.
- **Onglet 🤖 Agent :** Explicabilité du routage (analyse syntaxique et multi-hop).
- **Onglet 🕸️ Graphe KAG :** Visualiseur Canvas interactif (Force-Directed Graph, zoom, pan, inspection d'entités).
- **Onglet 📊 Résultats :** Dashboard MLOps avec barres d'Exact Match, F1 et historique des tests.

---

## 📽️ SLIDE 10 : Les Métriques d'Évaluation Retenues
- **Exact Match (EM) :** Normalisation stricte de la vérité terrain (0 ou 1).
- **Token F1-Score :** Précision et rappel sur les ensembles de tokens linguistiques.
- **Latence :** Temps moyen, médiane p50 et percentile 95 (p95) en millisecondes.
- **Robustesse au Refus (Anti-Hallucination) :** Taux d'abstention correcte sur questions négatives.
- **Efficacité FinOps (Tokens) :** Volume total de tokens injectés dans le contexte du LLM.

---

## 📽️ SLIDE 11 : Tableau des Résultats Finaux (11 Configurations)
| Configuration | Famille | Exact Match (EM) | Token F1 | Latence Moy. | Latence p95 | Robustesse Refus | Tokens Totaux |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Routeur-Lexical-Smart** | Hybride Agent | **53.8%** | **37.0%** | 0.2 ms | 2.02 ms | **100.0%** | 3 807 |
| **LightRAG-Hybrid** | KAG Hybride | **53.8%** | **37.0%** | 0.6 ms | 4.18 ms | **100.0%** | 3 585 |
| **LightRAG-Local** | KAG Voisinage | **53.8%** | 29.3% | 0.3 ms | 3.91 ms | **100.0%** | 2 381 |
| **RAG-BM25 (Okapi)** | RAG Lexical | 46.2% | 29.3% | 0.2 ms | 2.06 ms | **100.0%** | 4 486 |
| **RAG-minsearch (TF-IDF)**| RAG Cosinus | 46.2% | 29.3% | 1.3 ms | 6.42 ms | **100.0%** | 4 505 |
| **LightRAG-Naive** | Baseline texte | 46.2% | 29.3% | < 0.1 ms | < 0.1 ms | **100.0%** | 4 718 |
| **LightRAG-Mix** | Triplets + Texte | 46.2% | 29.3% | 0.3 ms | 2.08 ms | **100.0%** | 5 247 |
| **KAG-In-Memory-Graph** | Graphe Pur | 30.8% | 23.1% | **< 0.1 ms** | **< 0.1 ms** | **100.0%** | **384 (-91%)** |
| **RAG-Dense-Vector** | Embeddings HF | 15.4% | 12.8% | 0.2 ms | 2.01 ms | **100.0%** | 4 276 |
| **RAG-Milvus-Lite** | Index Dédié | 15.4% | 12.8% | < 0.1 ms | < 0.1 ms | **100.0%** | 4 276 |

---

## 📽️ SLIDE 12 : Analyse & Enseignements Majeurs
1. **Supériorité de l'Hybridation (53.8% EM) :**
   - Le RAG classique échoue sur les questions multi-hop car le mot-clé de la 2ᵉ entité est absent de la question.
   - Le graphe franchit le pont relationnel (1-hop / 2-hop) et ramène les entités manquantes.
2. **Solidité du RAG Lexical BM25 (46.2% EM) :**
   - Imbattable en simplicité et rapidité sur les questions factuelles directes, avec 0$ de coût d'indexation.
3. **Zéro Hallucination (100% Robustesse au Refus) :**
   - Toutes nos configurations refusent de répondre lorsqu'une information est absente, garantissant une utilisation sécurisée en entreprise.

---

## 📽️ SLIDE 13 : L'Atout FinOps Décisif : -91% de Tokens
- **RAG conventionnel :** 4 505 tokens (injection de paragraphes entiers remplis de mots inutiles).
- **KAG Graphe Pur :** 384 tokens (uniquement des triplets `Sujet-Relation-Objet` hautement informatifs).
- **Impact Économique pour Ooredoo :**
  - Division par plus de 11 de la taille de contexte.
  - Réduction de 90% des coûts d'API LLM pour un assistant sollicité des milliers de fois par jour.
  - Réduction de la latence de génération et de l'empreinte carbone.

---

## 📽️ SLIDE 14 : Recommandations Architecturales pour Ooredoo
```
+------------------------------------+--------------------------+-------------------------------+
| Cas d'Usage Métier Ooredoo         | Architecture Préconisée  | Justification Mesurée         |
+------------------------------------+--------------------------+-------------------------------+
| 1. FAQ & Support Collaborateurs    | RAG BM25 / Hybride Dense | Latence < 0.3 ms, coût index  |
|    (Procédures RH, Forfaits SIM)   |                          | nul, mise à jour immédiate    |
+------------------------------------+--------------------------+-------------------------------+
| 2. Analyse de Dossiers Complexes   | LightRAG-Hybrid          | EM 53.8%, F1 37.0%, liens     |
|    (Cahiers des charges RFP)       | ou Routeur Intelligent   | cross-documents garantis      |
+------------------------------------+--------------------------+-------------------------------+
| 3. Cartographie & Dépendances      | KAG Graphe Dédié (Neo4j) | Économie de 91% tokens, liens |
|    (Réseau RAN, Équipements 5G)    |                          | multi-sauts explicables       |
+------------------------------------+--------------------------+-------------------------------+
```

---

## 📽️ SLIDE 15 : Perspectives & Conclusion
- **Feuille de route pour le passage à l'échelle industrielle :**
  1. *Scalabilité :* Déploiement de **Neo4j Enterprise** (millions d'arêtes) et **Milvus Distribué** sous Kubernetes.
  2. *Spécialisation :* Fine-tuning d'embeddings et ontologie métier dédiée aux télécoms (vocabulaire 5G, BSS/OSS).
  3. *Agentique avancée :* Boucles d'auto-correction via **LangGraph (Corrective RAG)**.
  4. *Observabilité :* Traçage continu des coûts et latences avec **Langfuse**.
- **Conclusion :** Ce projet a livré une plateforme opérationnelle, vérifié l'apport décisif des graphes de connaissances et posé les standards d'architecture pour les futurs agents IA d'Ooredoo.
- *Merci pour votre attention. Place à vos questions.*
