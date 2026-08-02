"""
run_evaluation.py
=================
Script d'évaluation standalone RAG vs KAG.
- Charge les résultats KAG depuis results_kag.json (déjà calculés)
- Charge HotpotQA et construit un index RAG minsearch (sans LLM)
- Lance le LLM Groq sur RAG uniquement (questions déjà filtrées)
- Calcule Exact Match + Token F1 pour les deux pipelines
- Sauvegarde le rapport JSON + affiche le résumé dans le terminal
"""

import os, sys, json, time, logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.WARNING)

# ── Vérification GROQ_API_KEY ─────────────────────────────────────────────────
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("❌ GROQ_API_KEY manquante dans .env")
    sys.exit(1)

print("\n" + "="*60)
print("  EVALUATION FINALE — RAG vs KAG")
print("  Dataset : HotpotQA | LLM : llama-3.1-8b-instant")
print("="*60)

# ── 1. Charger les résultats KAG existants ────────────────────────────────────
kag_path = Path("results_kag.json")
if not kag_path.exists():
    print("❌ results_kag.json introuvable")
    sys.exit(1)

with open(kag_path, encoding="utf-8") as f:
    kag_data = json.load(f)

# Filtrer les entrées valides
kag_entries = [e for e in kag_data if e.get("question") and e.get("ground_truth") and e.get("predicted_answer")]
N = len(kag_entries)
print(f"\n✅ {N} résultats KAG chargés depuis results_kag.json")

questions    = [e["question"]        for e in kag_entries]
ground_truths= [e["ground_truth"]    for e in kag_entries]
kag_preds    = [e["predicted_answer"] for e in kag_entries]

# ── 2. Métriques d'évaluation ─────────────────────────────────────────────────
import re, string
from collections import Counter

def normalize_answer(text: str) -> str:
    def remove_articles(t): return re.sub(r'\b(a|an|the)\b', ' ', t)
    def white_space_fix(t): return ' '.join(t.split())
    def remove_punc(t):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in t if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(text.lower())))

def exact_match(pred, gt):
    p, g = normalize_answer(pred), normalize_answer(gt)
    return 1.0 if p == g or (g and g in p) else 0.0

def token_f1(pred, gt):
    pt = normalize_answer(pred).split()
    gt_t = normalize_answer(gt).split()
    if not pt or not gt_t: return 0.0
    common = sum((Counter(pt) & Counter(gt_t)).values())
    if common == 0: return 0.0
    p = common / len(pt)
    r = common / len(gt_t)
    return round(2*p*r/(p+r), 4)

# ── 3. Charger HotpotQA + construire index RAG ────────────────────────────────
print(f"\n📥 Chargement HotpotQA ({N} exemples) pour RAG...")
try:
    from datasets import load_dataset
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split=f"validation[:{N}]")
    docs = []
    seen = set()
    for item in dataset:
        for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
            text = " ".join(sentences)
            key = f"{title}:{text[:40]}"
            if key not in seen:
                seen.add(key)
                docs.append({"title": title, "text": text})
    print(f"✅ {len(docs)} documents RAG indexés")
except Exception as e:
    print(f"⚠️  HotpotQA non disponible: {e}")
    docs = []

# Construire index minsearch
rag_index = None
if docs:
    try:
        from minsearch import Index
        rag_index = Index(text_fields=["title", "text"], keyword_fields=[])
        rag_index.fit(docs)
        print("✅ Index minsearch construit")
    except Exception as e:
        print(f"⚠️  minsearch: {e}")

# ── 4. Lancer RAG sur les questions ──────────────────────────────────────────
from groq import Groq
client = Groq(api_key=api_key)

RAG_SYSTEM = """You are an expert QA assistant. Answer the question using ONLY the provided context.
Be very concise — answer with just the key fact (1-5 words when possible).
If the answer is not in the context, say: I don't know."""

def rag_query(question, index, num_results=5):
    results = []
    if index:
        try:
            results = index.search(question, boost_dict={"title": 2.0, "text": 1.0}, num_results=num_results)
        except Exception:
            pass
    context = "\n".join(f"[{r.get('title','')}] {r.get('text','')}" for r in results) or "No context found."
    prompt = f"QUESTION: {question}\n\nCONTEXT:\n{context}"
    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": RAG_SYSTEM},
                      {"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=100
        )
        return resp.choices[0].message.content.strip(), context
    except Exception as e:
        return f"ERREUR: {e}", context

print(f"\n🚀 Évaluation RAG sur {N} questions...")
rag_preds = []
rag_latencies = []
for i, q in enumerate(questions, 1):
    t0 = time.time()
    ans, ctx = rag_query(q, rag_index)
    lat = round((time.time()-t0)*1000)
    rag_preds.append(ans)
    rag_latencies.append(lat)
    em  = exact_match(ans, ground_truths[i-1])
    f1  = token_f1(ans, ground_truths[i-1])
    status = "✅" if em == 1.0 else ("⚠️" if f1 > 0.3 else "❌")
    print(f"  [{i}/{N}] {status} EM={em:.1f} F1={f1:.2f} ({lat}ms) | Q: {q[:60]}...")
    time.sleep(0.5)  # éviter rate limit

# ── 5. Calcul des métriques finales ──────────────────────────────────────────
def compute_metrics(preds, gts, name):
    ems  = [exact_match(p, g) for p, g in zip(preds, gts)]
    f1s  = [token_f1(p, g)   for p, g in zip(preds, gts)]
    n_correct = sum(1 for e in ems if e == 1.0)
    details = []
    for i, (q, gt, pred, em, f1) in enumerate(zip(questions, gts, preds, ems, f1s)):
        details.append({
            "idx": i+1, "question": q, "ground_truth": gt,
            f"answer_{name.lower()}": pred,
            "exact_match": em, "token_f1": f1, "correct": em == 1.0
        })
    return {
        "pipeline": name,
        "n_questions": len(preds),
        "n_correct": n_correct,
        "accuracy": round(n_correct / len(preds), 4),
        "avg_exact_match": round(sum(ems) / len(ems), 4),
        "avg_token_f1":    round(sum(f1s) / len(f1s), 4),
        "details": details
    }

rag_metrics = compute_metrics(rag_preds,  ground_truths, "RAG")
kag_metrics = compute_metrics(kag_preds,  ground_truths, "KAG")

# ── 6. Affichage du rapport ────────────────────────────────────────────────────
print("\n" + "="*60)
print("  📊 RÉSULTATS FINAUX — BENCHMARK RAG vs KAG")
print("="*60)
print(f"\n{'Métrique':<25} {'RAG (minsearch)':<20} {'KAG (Graphe)':<20}")
print("-"*65)
print(f"{'Exact Match (EM)':<25} {rag_metrics['avg_exact_match']:<20.4f} {kag_metrics['avg_exact_match']:<20.4f}")
print(f"{'Token F1':<25} {rag_metrics['avg_token_f1']:<20.4f} {kag_metrics['avg_token_f1']:<20.4f}")
print(f"{'Accuracy':<25} {rag_metrics['accuracy']:<20.4f} {kag_metrics['accuracy']:<20.4f}")
print(f"{'Réponses correctes':<25} {rag_metrics['n_correct']}/{N:<17} {kag_metrics['n_correct']}/{N:<17}")
print(f"{'Latence moy. (ms)':<25} {round(sum(rag_latencies)/len(rag_latencies)):<20} {'N/A (pré-calculé)':<20}")
print("-"*65)

winner_em = "RAG" if rag_metrics['avg_exact_match'] >= kag_metrics['avg_exact_match'] else "KAG"
winner_f1 = "RAG" if rag_metrics['avg_token_f1'] >= kag_metrics['avg_token_f1'] else "KAG"
print(f"\n🏆 Meilleur EM    : {winner_em}")
print(f"🏆 Meilleur F1    : {winner_f1}")

print("\n📋 Détail par question :")
print(f"{'#':<4} {'Vérité':<20} {'RAG':<30} {'KAG':<30} {'EM_R':<6} {'EM_K':<6}")
print("-"*96)
for i, (gt, rp, kp) in enumerate(zip(ground_truths, rag_preds, kag_preds), 1):
    em_r = exact_match(rp, gt)
    em_k = exact_match(kp, gt)
    r_icon = "✅" if em_r else "❌"
    k_icon = "✅" if em_k else "❌"
    print(f"{i:<4} {gt[:18]:<20} {r_icon} {rp[:26]:<28} {k_icon} {kp[:26]:<28} {em_r:<6.1f} {em_k:<6.1f}")

# ── 7. Sauvegarde JSON ────────────────────────────────────────────────────────
report = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "dataset": "HotpotQA distractor",
    "n_questions": N,
    "llm": "llama-3.1-8b-instant (Groq)",
    "rag_engine": "minsearch BM25",
    "kag_engine": "InMemory Knowledge Graph",
    "results": {
        "RAG": {k: v for k, v in rag_metrics.items() if k != "details"},
        "KAG": {k: v for k, v in kag_metrics.items() if k != "details"}
    },
    "details": [
        {
            "idx": i+1,
            "question": q,
            "ground_truth": gt,
            "rag_answer": rp,
            "kag_answer": kp,
            "rag_em": exact_match(rp, gt),
            "kag_em": exact_match(kp, gt),
            "rag_f1": token_f1(rp, gt),
            "kag_f1": token_f1(kp, gt),
        }
        for i, (q, gt, rp, kp) in enumerate(zip(questions, ground_truths, rag_preds, kag_preds))
    ]
}

out_path = Path("results/benchmark_final.json")
out_path.parent.mkdir(exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\n✅ Rapport sauvegardé : {out_path}")
print("\n" + "="*60)
print("  🌐 Lancer le serveur : uv run python kag_server.py")
print("  📂 Rapport JSON      : results/benchmark_final.json")
print("="*60 + "\n")
