"""
load_benchmark_datasets.py
==========================
Charge 5 sources HuggingFace, tag par catégorie et fusionne dans un CSV unique
compatible Ragas et DeepEval.

Catégories :
  - single_hop        : réponse trouvable dans un seul passage
  - multi_hop         : nécessite chaîner 2+ passages/documents
  - relational_complex: inférence relationnelle / comparaison / superlative

Colonnes CSV de sortie :
  id | question | ground_truth | contexts (JSON list) | context_str |
  category | hop_count | source_dataset | answerable

Usage :
  uv run python load_benchmark_datasets.py
  ou
  pip install datasets && python load_benchmark_datasets.py
"""

import json
import csv
import sys
import os
from pathlib import Path
from typing import Optional

# ── Vérification de la lib datasets ───────────────────────────────────────────
try:
    from datasets import load_dataset
    print("✅ datasets disponible")
except ImportError:
    print("❌ datasets non installé. Exécutez : pip install datasets")
    sys.exit(1)

OUTPUT_FILE = "rag_kag_benchmark.csv"
MAX_PER_DATASET = 200  # Nombre d'exemples max par dataset (modifiable)

rows = []   # Liste finale de toutes les lignes

# ─────────────────────────────────────────────────────────────────────────────
# HELPER : normaliser les contextes en liste de strings
# ─────────────────────────────────────────────────────────────────────────────
def make_contexts(passages) -> list[str]:
    """Convertit n'importe quel format de passages en liste de strings."""
    if passages is None:
        return []
    if isinstance(passages, str):
        return [passages]
    if isinstance(passages, list):
        result = []
        for p in passages:
            if isinstance(p, str):
                result.append(p)
            elif isinstance(p, dict):
                # HotpotQA : {"title": ..., "sentences": [...]}
                text = p.get("text") or " ".join(p.get("sentences", []))
                title = p.get("title", "")
                result.append(f"[{title}] {text}".strip() if title else text)
            elif isinstance(p, list):
                result.append(" ".join(str(x) for x in p))
        return result
    return [str(passages)]

def build_row(uid, question, ground_truth, contexts_raw,
              category, hop_count, source, answerable=True):
    ctx_list = make_contexts(contexts_raw)
    ctx_str  = " | ".join(ctx_list)[:2000]   # Tronqué pour lisibilité CSV
    return {
        "id":             uid,
        "question":       question.strip(),
        "ground_truth":   str(ground_truth).strip(),
        "contexts":       json.dumps(ctx_list, ensure_ascii=False),
        "context_str":    ctx_str,
        "category":       category,
        "hop_count":      hop_count,
        "source_dataset": source,
        "answerable":     answerable,
        # Colonnes vides pour Ragas/DeepEval (à remplir après inférence)
        "rag_answer":     "",
        "kag_answer":     "",
        "rag_em":         "",
        "kag_em":         "",
        "rag_f1":         "",
        "kag_f1":         "",
    }

# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 1 — HotpotQA (multi_hop, 2 sauts)
# https://huggingface.co/datasets/hotpotqa/hotpot_qa
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("📦 [1/5] HotpotQA — distractor (multi_hop)")
print("="*55)
try:
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor",
                      split=f"validation[:{MAX_PER_DATASET}]",
                      trust_remote_code=True)
    count = 0
    for i, ex in enumerate(ds):
        # Construire contextes : titles + sentences
        ctx_list = []
        for title, sents in zip(ex["context"]["title"],
                                ex["context"]["sentences"]):
            text = " ".join(sents)
            ctx_list.append(f"[{title}] {text}")

        rows.append(build_row(
            uid         = f"hotpot_{i}",
            question    = ex["question"],
            ground_truth= ex["answer"],
            contexts_raw= ctx_list,
            category    = "multi_hop",
            hop_count   = 2,
            source      = "hotpotqa/hotpot_qa",
        ))
        count += 1
    print(f"  ✅ {count} exemples chargés")
except Exception as e:
    print(f"  ❌ ÉCHEC HotpotQA : {e}")
    print("  → Chercher 'hotpotqa' sur https://huggingface.co/datasets")

# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 2 — MuSiQue (multi_hop, 2-4 sauts)
# Essai avec plusieurs noms connus
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("📦 [2/5] MuSiQue — (multi_hop, 2-4 sauts)")
print("="*55)
MUSIQUE_CANDIDATES = [
    ("dgslibisey/MuSiQue",    "train",      None),
    ("musique",                "train",      None),
    ("multi_hop_reasoning/musique", "train", None),
    ("HuggingFaceH4/MuSiQue", "train",      None),
]
musique_loaded = False
for ds_name, split, config in MUSIQUE_CANDIDATES:
    try:
        print(f"  ↳ Tentative : {ds_name} ...")
        kwargs = {"split": f"{split}[:{MAX_PER_DATASET}]", "trust_remote_code": True}
        if config:
            kwargs["name"] = config
        ds = load_dataset(ds_name, **kwargs)
        count = 0
        for i, ex in enumerate(ds):
            question     = ex.get("question") or ex.get("input") or ""
            ground_truth = ex.get("answer") or ex.get("output") or ex.get("answers", [""])[0]
            if isinstance(ground_truth, list):
                ground_truth = ground_truth[0]

            # Récupérer les paragraphes sources
            paras = ex.get("paragraphs") or ex.get("context") or []
            ctx_list = []
            if isinstance(paras, list):
                for p in paras:
                    if isinstance(p, dict):
                        title = p.get("title", "")
                        body  = p.get("paragraph_text") or p.get("text", "")
                        ctx_list.append(f"[{title}] {body}".strip())
                    elif isinstance(p, str):
                        ctx_list.append(p)

            # Compter les sauts
            bridges = ex.get("question_decomposition") or []
            hop_n   = len(bridges) if bridges else 2

            rows.append(build_row(
                uid         = f"musique_{i}",
                question    = question,
                ground_truth= ground_truth,
                contexts_raw= ctx_list,
                category    = "multi_hop",
                hop_count   = hop_n,
                source      = ds_name,
                answerable  = ex.get("answerable", True),
            ))
            count += 1
        print(f"  ✅ {count} exemples chargés depuis {ds_name}")
        musique_loaded = True
        break
    except Exception as e:
        print(f"  ⚠️  {ds_name} échoué : {e}")

if not musique_loaded:
    print("  ❌ Tous les candidats MuSiQue ont échoué.")
    print("  → Chercher 'musique' sur https://huggingface.co/datasets")

# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 3 — 2WikiMultihopQA (multi_hop, comparaison/bridge)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("📦 [3/5] 2WikiMultihopQA — (multi_hop + relational_complex)")
print("="*55)
WIKI2_CANDIDATES = [
    ("xanhho/2WikiMultihopQA",         "train", None),
    ("2wikimultihopqa",                "train", None),
    ("wiki_hop",                       "train", "original"),
    ("bigbio/2wikimultihop_qa",        "train", "2wikimultihop_qa_source"),
]
wiki2_loaded = False
for ds_name, split, config in WIKI2_CANDIDATES:
    try:
        print(f"  ↳ Tentative : {ds_name} ...")
        kwargs = {"split": f"{split}[:{MAX_PER_DATASET}]", "trust_remote_code": True}
        if config:
            kwargs["name"] = config
        ds = load_dataset(ds_name, **kwargs)
        count = 0
        for i, ex in enumerate(ds):
            question     = ex.get("question") or ""
            ground_truth = ex.get("answer") or ex.get("answers", [""])[0]
            if isinstance(ground_truth, list):
                ground_truth = ground_truth[0]

            # Détecter le type pour la catégorie
            q_type   = ex.get("type", "bridge")  # bridge / comparison / inference
            category = "relational_complex" if q_type in ("comparison", "inference") else "multi_hop"

            # Contextes
            ctx_raw = (ex.get("context") or
                       ex.get("passages") or
                       ex.get("supporting_facts") or [])
            ctx_list = make_contexts(ctx_raw)

            rows.append(build_row(
                uid         = f"wiki2_{i}",
                question    = question,
                ground_truth= ground_truth,
                contexts_raw= ctx_list,
                category    = category,
                hop_count   = 2,
                source      = ds_name,
            ))
            count += 1
        print(f"  ✅ {count} exemples chargés depuis {ds_name}")
        wiki2_loaded = True
        break
    except Exception as e:
        print(f"  ⚠️  {ds_name} échoué : {e}")

if not wiki2_loaded:
    print("  ❌ Tous les candidats 2WikiMultihopQA ont échoué.")
    print("  → Chercher '2wikimultihopqa' sur https://huggingface.co/datasets")

# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 4 — TriviaQA (single_hop, fact retrieval)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("📦 [4/5] TriviaQA — rc (single_hop)")
print("="*55)
TRIVIA_CANDIDATES = [
    ("trivia_qa", "validation", "rc"),
    ("trivia_qa", "validation", "unfiltered"),
    ("mandarjoshi/trivia_qa", "validation", "rc"),
]
trivia_loaded = False
for ds_name, split, config in TRIVIA_CANDIDATES:
    try:
        print(f"  ↳ Tentative : {ds_name} (config={config}) ...")
        ds = load_dataset(ds_name, config,
                          split=f"{split}[:{MAX_PER_DATASET}]",
                          trust_remote_code=True)
        count = 0
        for i, ex in enumerate(ds):
            question = ex.get("question", "")
            aliases  = ex.get("answer", {})
            if isinstance(aliases, dict):
                ground_truth = aliases.get("value") or (aliases.get("aliases") or [""])[0]
            else:
                ground_truth = str(aliases)

            # Contextes depuis les résultats web/wiki
            ctx_list = []
            for src in (ex.get("search_results") or {}).get("search_context", []):
                if src:
                    ctx_list.append(str(src)[:500])
            for src in (ex.get("entity_pages") or {}).get("wiki_context", []):
                if src:
                    ctx_list.append(str(src)[:500])
            if not ctx_list:
                ctx_list = ["[No context provided]"]

            rows.append(build_row(
                uid         = f"trivia_{i}",
                question    = question,
                ground_truth= ground_truth,
                contexts_raw= ctx_list,
                category    = "single_hop",
                hop_count   = 1,
                source      = ds_name,
            ))
            count += 1
        print(f"  ✅ {count} exemples chargés depuis {ds_name}")
        trivia_loaded = True
        break
    except Exception as e:
        print(f"  ⚠️  {ds_name} (config={config}) échoué : {e}")

if not trivia_loaded:
    print("  ❌ TriviaQA non chargé.")
    print("  → Chercher 'trivia_qa' sur https://huggingface.co/datasets")

# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 5 — SQuAD v2 (single_hop + unanswerable = relational_complex)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("📦 [5/5] SQuAD v2 — (single_hop + relational_complex)")
print("="*55)
SQUAD_CANDIDATES = [
    ("rajpurkar/squad_v2", "validation", None),
    ("squad_v2",           "validation", None),
    ("squad",              "validation", None),
    ("rajpurkar/squad",    "validation", None),
]
squad_loaded = False
for ds_name, split, config in SQUAD_CANDIDATES:
    try:
        print(f"  ↳ Tentative : {ds_name} ...")
        kwargs = {"split": f"{split}[:{MAX_PER_DATASET}]", "trust_remote_code": True}
        if config:
            kwargs["name"] = config
        ds = load_dataset(ds_name, **kwargs)
        count = 0
        for i, ex in enumerate(ds):
            question = ex.get("question", "")
            answers  = ex.get("answers", {})

            if isinstance(answers, dict):
                ans_texts = answers.get("text", [])
            else:
                ans_texts = []

            answerable   = len(ans_texts) > 0
            ground_truth = ans_texts[0] if ans_texts else "unanswerable"

            # Catégorie : unanswerable = relational_complex (nécessite raisonnement)
            category = "relational_complex" if not answerable else "single_hop"

            context = ex.get("context", "")
            rows.append(build_row(
                uid         = f"squad_{i}",
                question    = question,
                ground_truth= ground_truth,
                contexts_raw= [context],
                category    = category,
                hop_count   = 1,
                source      = ds_name,
                answerable  = answerable,
            ))
            count += 1
        print(f"  ✅ {count} exemples chargés depuis {ds_name}")
        squad_loaded = True
        break
    except Exception as e:
        print(f"  ⚠️  {ds_name} échoué : {e}")

if not squad_loaded:
    print("  ❌ SQuAD non chargé.")
    print("  → Chercher 'squad_v2' sur https://huggingface.co/datasets")

# ─────────────────────────────────────────────────────────────────────────────
# FUSION ET EXPORT CSV
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("💾 Fusion et export CSV...")
print("="*55)

if not rows:
    print("❌ Aucune donnée chargée. Vérifiez votre connexion HuggingFace.")
    sys.exit(1)

FIELDNAMES = [
    "id", "question", "ground_truth", "contexts", "context_str",
    "category", "hop_count", "source_dataset", "answerable",
    "rag_answer", "kag_answer", "rag_em", "kag_em", "rag_f1", "kag_f1",
]

output_path = Path(OUTPUT_FILE)
with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)

# ─────────────────────────────────────────────────────────────────────────────
# RAPPORT FINAL
# ─────────────────────────────────────────────────────────────────────────────
from collections import Counter
cat_counts = Counter(r["category"]      for r in rows)
src_counts = Counter(r["source_dataset"] for r in rows)
hop_counts = Counter(r["hop_count"]     for r in rows)

print(f"\n{'='*55}")
print(f"  📊 RAPPORT FINAL — {OUTPUT_FILE}")
print(f"{'='*55}")
print(f"  Total lignes  : {len(rows)}")
print(f"\n  Par catégorie :")
for cat, n in sorted(cat_counts.items()):
    pct = n / len(rows) * 100
    bar = "█" * int(pct / 3)
    print(f"    {cat:<22} {n:>4} ({pct:5.1f}%) {bar}")
print(f"\n  Par dataset source :")
for src, n in sorted(src_counts.items(), key=lambda x: -x[1]):
    print(f"    {src:<40} {n:>4}")
print(f"\n  Par nombre de sauts :")
for hop, n in sorted(hop_counts.items()):
    print(f"    hop={hop}  →  {n} questions")
print(f"\n  ✅ Fichier : {output_path.resolve()}")
print(f"{'='*55}")
print("""
  Utilisation avec Ragas :
    from ragas import evaluate
    from datasets import Dataset
    import pandas as pd, json

    df = pd.read_csv("rag_kag_benchmark.csv")
    df["contexts"] = df["contexts"].apply(json.loads)
    dataset = Dataset.from_pandas(df[["question","contexts","ground_truth"]])

  Utilisation avec DeepEval :
    from deepeval.test_case import LLMTestCase
    import pandas as pd, json

    df = pd.read_csv("rag_kag_benchmark.csv")
    test_cases = [
        LLMTestCase(
            input=row.question,
            actual_output=row.rag_answer,
            expected_output=row.ground_truth,
            retrieval_context=json.loads(row.contexts),
        )
        for _, row in df.iterrows()
    ]
""")
