import os
import time
import json
import argparse
import logging
from dotenv import load_dotenv

from data_loader import load_hotpotqa
from kag_pipeline.graph_builder import build_graph, get_neo4j_driver
from kag_pipeline.kag_helper import KAGBase

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Pipeline KAG (Knowledge-Augmented Generation)")
    parser.add_argument("--n-samples", type=int, default=10, help="Nombre d'exemples HotpotQA à charger (défaut: 10 pour test rapide, 200 pour complet)")
    parser.add_argument("--test-questions", type=int, default=5, help="Nombre de questions à tester et enregistrer (défaut: 5)")
    args = parser.parse_args()

    print("\n==================================================")
    print(f"  DÉMARRAGE DU PIPELINE KAG ({args.n_samples} exemples)")
    print("==================================================\n")

    # 1. Chargement des données HotpotQA
    logger.info(f"Chargement de {args.n_samples} exemples du dataset HotpotQA...")
    examples = load_hotpotqa(n_samples=args.n_samples)
    logger.info(f"{len(examples)} exemples chargés avec succès.")

    # 2. Obtenir le Driver Neo4j (ou Fallback en mémoire si Neo4j hors ligne)
    driver = get_neo4j_driver()

    start_time = time.time()
    logger.info("Début de la construction du graphe...")
    summary = build_graph(examples, driver=driver, clear_existing=True)
    build_duration = time.time() - start_time
    logger.info(f"Graphe construit en {build_duration:.2f} secondes.")

    # 3. Instanciation de KAGBase
    logger.info("Initialisation de l'orchestrateur KAGBase...")
    kag_agent = KAGBase(driver=driver)

    # 4. Évaluation sur les N premières questions
    eval_count = min(args.test_questions, len(examples))
    logger.info(f"Test de {eval_count} questions...")

    results = []

    print("\n==================================================")
    print("  RÉSULTATS DE TEST SUR LES QUESTIONS")
    print("==================================================\n")

    for idx, item in enumerate(examples[:eval_count], start=1):
        q = item["question"]
        ground_truth = item["answer"]

        logger.info(f"Traitement Q{idx}/{eval_count} : {q}")
        kag_output = kag_agent.kag(q)

        predicted_answer = kag_output["answer"]
        context_used = kag_output["context"]

        print(f"--- Question {idx} ---")
        print(f"Q: {q}")
        print(f"Ground Truth : {ground_truth}")
        print(f"KAG Predicted: {predicted_answer}")
        print(f"Contexte Faits: {context_used[:200]}..." if len(context_used) > 200 else f"Contexte Faits: {context_used}")
        print("-" * 50 + "\n")

        results.append({
            "question": q,
            "ground_truth": ground_truth,
            "predicted_answer": predicted_answer,
            "context": context_used
        })

    # 5. Sauvegarde des résultats dans results_kag.json
    output_filepath = "results_kag.json"
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Résultats sauvegardés avec succès dans {output_filepath}")
    print(f"\n[SUCCÈS] Pipeline KAG terminé. Résultats enregistrés dans '{output_filepath}'.\n")

    driver.close()

if __name__ == "__main__":
    main()
