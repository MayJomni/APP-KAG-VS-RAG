"""
test_extraction.py — Dry-run d'extraction RFP sur 3 chunks seulement.

Usage:
    python test_extraction.py                       # utilise test_ooredoo_rfp.txt
    python test_extraction.py mon_rfp.txt           # utilise un autre fichier
    python test_extraction.py mon_rfp.txt 5         # teste sur 5 chunks au lieu de 3

Objectif : valider le format de sortie du LLM et le pipeline de parsing
AVANT de lancer l'extraction complète (qui consomme tout le quota Groq).
"""

import os
import sys
import time
import difflib
import logging
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# ── Couleurs ANSI (compatibles Windows 10+ et tous les terminaux UNIX) ─────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
RED     = "\033[31m"
CYAN    = "\033[36m"
DIM     = "\033[2m"

def clr(color: str, text: str) -> str:
    return f"{color}{text}{RESET}"

# ── Configuration ───────────────────────────────────────────────────────────────
load_dotenv()
GROQ_KEY      = os.getenv("GROQ_API_KEY", "")
MODEL_EXTRACT = "openai/gpt-oss-20b"          # quota rapide
MODEL_ANSWER  = "openai/gpt-oss-120b"  # fallback

VALID_CATS = {
    "Functional", "Technical", "Operational",
    "Security", "Infrastructure", "Administrative", "Financial"
}

# ── Prompt identique à fast_server.py ──────────────────────────────────────────
EXTRACT_SYS = (
    "You are a requirements extraction expert analyzing an RFP (Request for Proposal) document.\n"
    "Read the text and extract EVERY distinct requirement or criterion mentioned, "
    "no matter how small. Do NOT summarize or merge multiple requirements into one line.\n"
    "Each bullet point, each sentence stating an obligation, capability, or constraint "
    "is a SEPARATE criterion.\n"
    "Output ONLY lines in this exact format:\n"
    "Category|Criterion|Mandatory\n"
    "Where:\n"
    "- Category is one of: Functional, Technical, Operational, Security, Infrastructure, "
    "Administrative, Financial\n"
    "- Criterion is the requirement itself, in a short precise sentence (max 20 words)\n"
    "- Mandatory is Yes or No (Yes if the text uses 'must', 'shall', 'required'; "
    "No if it uses 'should', 'recommended', 'optional')\n"
    "No JSON, no numbering, no explanation, no headers, no introduction.\n"
    "Example:\n"
    "Technical|Solution must produce unified XML file for each bill|Yes\n"
    "Functional|Vendor should support future bill design changes|No\n"
    "Security|Vendor certified ISO 27001 is recommended|No"
)

# Mots-clés parasites issus du prompt lui-meme → filtrer ces lignes
PROMPT_EXAMPLES = {
    "shall respond within 2 seconds",
    "unified xml file",
    "future bill design",
    "iso 27001 is recommended",
    "criterion", "category", "mandatory",
    "example", "format", "note:", "header", "where:", "output only",
}

# ── Fonctions utilitaires (identiques a fast_server.py) ────────────────────────

def _chunk_by_clause(title: str, text: str, min_words: int = 6) -> list:
    """Decoupe le texte par section/clause/puce. Meme logique que fast_server.py."""
    import re as _r
    chunks = []
    lines  = text.splitlines()
    cur    = ""
    pat    = _r.compile(
        r"^(\d+[\.\)\-]\s|[a-zA-Z][\.\)\-]\s|[*\-\u2022\u2013]\s"
        r"|[A-Z]{2,}\s*:|Section\s+\d|Article\s+\d|Clause\s+\d"
        r"|Requirement\s+\d|REQ-\d)",
        _r.IGNORECASE
    )

    def flush(b):
        b = b.strip()
        if len(b.split()) >= min_words:
            chunks.append({"title": title, "text": b, "id": f"{title}_cl{len(chunks)}"})

    for line in lines:
        s = line.strip()
        if not s:
            flush(cur); cur = ""
        elif pat.match(s) and cur:
            flush(cur); cur = s
        else:
            cur = (cur + " " + s).strip() if cur else s
    flush(cur)

    # Fallback 1 : decoupe par paragraphes vides
    if len(chunks) < 3:
        chunks = [
            {"title": title, "text": p.strip(), "id": f"{title}_p{i}"}
            for i, p in enumerate(_r.split(r"\n\s*\n", text))
            if len(p.split()) >= min_words
        ]
    # Fallback 2 : fenetres fixes de 80 mots
    if len(chunks) < 2:
        w = text.split()
        chunks = [
            {"title": title, "text": " ".join(w[i:i+80]), "id": f"{title}_w{i}"}
            for i in range(0, len(w), 80) if len(w[i:i+80]) >= min_words
        ]
    return chunks


def _is_near_duplicate(a: str, b: str, threshold: float = 0.85) -> bool:
    return difflib.SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio() >= threshold


def llm_call(user_prompt: str, max_tokens: int = 3000) -> tuple:
    """Appelle le LLM avec fallback gemma2 -> llama. Retourne (raw_text, model_used)."""
    client = Groq(api_key=GROQ_KEY)
    for model in [MODEL_EXTRACT, MODEL_ANSWER]:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": EXTRACT_SYS},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            raw = resp.choices[0].message.content.strip()
            if raw:
                return raw, model
        except Exception as e:
            print(clr(YELLOW, f"  [!] {model} echoue : {e}"))
    return "", ""


def parse_lines(raw: str, doc_title: str, chunk_id: str) -> tuple:
    """
    Parse le texte brut du LLM ligne par ligne.
    Retourne (criteria_list, malformed_lines).
      criteria_list  : [{category, cat_valid, criterion, mandatory, source, chunk_id}]
      malformed_lines: [str]  lignes qui n'ont pas pu etre parsees
    """
    criteria  = []
    malformed = []

    for line in raw.split("\n"):
        line = line.strip().lstrip("-*>.0123456789.) ")
        if not line:
            continue

        # Ligne sans separateur "|"
        if "|" not in line:
            if len(line) > 10:
                malformed.append(line)
            continue

        parts = [x.strip() for x in line.split("|")]

        # Pas assez de champs
        if len(parts) < 3:
            malformed.append(line)
            continue

        cat           = parts[0]
        criterion     = parts[1]
        mandatory_raw = parts[2].upper()

        # Trop de champs (ancien format 4 colonnes) -> signaler
        if len(parts) > 3:
            malformed.append(f"[EXTRA FIELDS] {line}")

        # Normalisation Mandatory
        if mandatory_raw in ("YES", "OUI"):
            mandatory = "Yes"
        elif mandatory_raw in ("NO", "NON"):
            mandatory = "No"
        else:
            mandatory = "Yes" if any(
                w in mandatory_raw.lower()
                for w in ["yes", "oui", "must", "shall", "required", "mandatory"]
            ) else "No"

        # Normalisation Category
        cat_normalized = next(
            (c for c in VALID_CATS if c.lower() == cat.lower()), None
        )

        # Filtre les lignes parasites du prompt
        if any(b in criterion.lower() for b in PROMPT_EXAMPLES):
            continue
        if not criterion or len(criterion) < 5:
            continue

        criteria.append({
            "category":  cat_normalized if cat_normalized else cat,
            "cat_valid": cat_normalized is not None,
            "criterion": criterion,
            "mandatory": mandatory,
            "source":    doc_title,
            "chunk_id":  chunk_id,
        })

    return criteria, malformed


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    # Arguments CLI
    rfp_path = sys.argv[1] if len(sys.argv) > 1 else "test_ooredoo_rfp.txt"
    n_chunks = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    rfp_file = Path(rfp_path)
    if not rfp_file.exists():
        print(clr(RED, f"[X] Fichier introuvable : {rfp_path}"))
        print(clr(DIM,  "    Usage : python test_extraction.py [fichier_rfp.txt] [n_chunks]"))
        sys.exit(1)

    if not GROQ_KEY:
        print(clr(RED, "[X] GROQ_API_KEY manquante — verifiez votre fichier .env"))
        sys.exit(1)

    # ── Chargement + chunking ────────────────────────────────────────────────
    text   = rfp_file.read_text(encoding="utf-8", errors="ignore")
    title  = rfp_file.stem
    chunks = _chunk_by_clause(title, text)

    print()
    print(clr(BOLD + CYAN, "=== TEST D'EXTRACTION RFP ==="))
    print(clr(DIM, f"  Fichier              : {rfp_file.resolve()}"))
    print(clr(DIM, f"  Chunks totaux        : {len(chunks)}"))
    print(clr(BOLD, f"  Chunks testes        : {n_chunks} (sur {len(chunks)})"))
    print(clr(DIM, f"  Modele primaire      : {MODEL_EXTRACT}"))
    print(clr(DIM, f"  Fallback             : {MODEL_ANSWER}"))
    print()

    sample_chunks  = chunks[:n_chunks]
    all_criteria   = []
    all_malformed  = []
    raw_texts_seen = []  # pour deduplication inter-chunks
    dup_count      = 0

    for ci, chunk in enumerate(sample_chunks):
        doc_title = chunk["title"]
        chunk_id  = chunk["id"]
        text_len  = len(chunk["text"].split())

        print(clr(BOLD, f"+-- Chunk {ci+1}/{n_chunks}  [{chunk_id}]"))
        print(clr(DIM,  f"|   {text_len} mots | texte complet envoye au LLM"))

        user_prompt = f"[CLAUSE {ci+1} - {doc_title}]\n{chunk['text']}\n"

        t0 = time.time()
        raw, model_used = llm_call(user_prompt, max_tokens=3000)
        elapsed = round((time.time() - t0) * 1000)

        if not raw:
            print(clr(RED, "|   [X] Pas de reponse LLM pour ce chunk"))
            print("+" + "-" * 60)
            if ci < len(sample_chunks) - 1:
                time.sleep(2)
            continue

        print(clr(DIM, f"|   [OK] {model_used} en {elapsed}ms"))
        print(clr(DIM, f"|   Apercu brut : {repr(raw[:100])}"))
        print("|")

        criteria, malformed = parse_lines(raw, doc_title, chunk_id)

        # Deduplication inter-chunks
        unique_criteria = []
        for c in criteria:
            is_dup = False
            for seen in raw_texts_seen:
                if _is_near_duplicate(c["criterion"], seen):
                    is_dup = True
                    dup_count += 1
                    break
            if not is_dup:
                raw_texts_seen.append(c["criterion"])
                unique_criteria.append(c)

        # Affichage des criteres
        for c in unique_criteria:
            mand_color = GREEN if c["mandatory"] == "Yes" else YELLOW
            cat_color  = RESET if c["cat_valid"] else RED
            cat_str    = f"[{c['category']:<17}]"
            mand_str   = f"({c['mandatory']:3})"
            print(
                f"|   {clr(cat_color + BOLD, cat_str)}  "
                f"{clr(mand_color, mand_str)}  "
                f"{c['criterion']}"
            )
            print(clr(DIM, f"|              source: {c['source']} | chunk: {c['chunk_id']}"))


        # Affichage des lignes malformees
        if malformed:
            print("|")
            for bad in malformed:
                print(clr(RED, f"|   [!] MALFORMED -> {bad}"))

        print("|")
        print(clr(DIM, f"|   -> {len(unique_criteria)} critere(s) unique(s) | "
                       f"{len(malformed)} ligne(s) malformee(s)"))
        print("+" + "-" * 60)
        print()

        all_criteria.extend(unique_criteria)
        all_malformed.extend(malformed)

        if ci < len(sample_chunks) - 1:
            print(clr(DIM, "  [sleep 2s — rate-limit Groq]"))
            time.sleep(2)
            print()

    # ── Resume final ─────────────────────────────────────────────────────────
    cats_found    = sorted({c["category"] for c in all_criteria})
    cats_invalid  = [c for c in cats_found if c not in VALID_CATS]
    mand_count    = sum(1 for c in all_criteria if c["mandatory"] == "Yes")
    opt_count     = sum(1 for c in all_criteria if c["mandatory"] == "No")

    print()
    print(clr(BOLD + CYAN, "=== RESUME DE L'EXTRACTION ==="))
    print()
    print(f"  {clr(BOLD, 'Criteres extraits  ')} : "
          f"{clr(GREEN + BOLD, str(len(all_criteria)))}  "
          f"({mand_count} obligatoires  |  {opt_count} optionnels)")
    print(f"  {clr(BOLD, 'Doublons supprimes ')} : {clr(YELLOW, str(dup_count))}")
    print(f"  {clr(BOLD, 'Lignes malformees  ')} : "
          + (clr(RED, str(len(all_malformed))) if all_malformed else clr(GREEN, "0 [OK]")))
    print()

    print(clr(BOLD, "  Categories rencontrees :"))
    for cat in cats_found:
        if cat in VALID_CATS:
            print(f"    [OK]  {cat}")
        else:
            print(clr(RED, f"    [!!]  {cat}  <- HORS LISTE AUTORISEE"))

    print()

    if cats_invalid:
        print(clr(RED + BOLD, "  [!!] CATEGORIES INVALIDES :"))
        for c in cats_invalid:
            print(clr(RED, f"       - {c}"))
        print(clr(DIM, "       -> Ajustez le prompt EXTRACT_SYS ou la normalisation."))
        print()

    if all_malformed:
        print(clr(RED + BOLD, "  [!!] LIGNES MALFORMEES :"))
        for line in all_malformed:
            print(clr(RED, f"       {line}"))
        print()

    # Verdict
    if not cats_invalid and not all_malformed:
        print(clr(GREEN + BOLD,
            "  [OK] Format valide — vous pouvez lancer l'extraction complete."))
    else:
        print(clr(YELLOW + BOLD,
            "  [!]  Des ajustements sont necessaires avant l'extraction complete."))
    print()


if __name__ == "__main__":
    main()
