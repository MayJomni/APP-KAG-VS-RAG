import os
import logging
from typing import List, Dict, Any, Optional
from neo4j import Driver
from groq import Groq
from dotenv import load_dotenv
from .graph_builder import get_neo4j_driver

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT_QUESTION_ENTITIES = """Vous êtes un assistant spécialisé dans l'analyse de questions.
Votre objectif est d'extraire la liste des entités clés (noms propres, lieux, concepts, organisations, titres) mentionnées dans la question posée.

Répondez STRICTEMENT avec un objet JSON au format :
{
  "entities": ["Entité 1", "Entité 2"]
}
"""

def extract_entities_from_question(question: str, client: Optional[Groq] = None, model: str = "llama-3.1-8b-instant") -> List[str]:
    """
    Extrait la liste des noms d'entités mentionnées dans une question via Groq LLM.
    """
    if not question or not question.strip():
        return []

    if client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return []
        client = Groq(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_QUESTION_ENTITIES},
                {"role": "user", "content": f"Question : {question}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        content = response.choices[0].message.content
        import json
        data = json.loads(content)
        return data.get("entities", [])
    except Exception as e:
        logger.warning(f"Impossible d'extraire les entités de la question '{question}': {e}")
        return []

def query_graph(question: str, driver: Optional[Driver] = None, top_k_entities: int = 5) -> List[Dict[str, str]]:
    """
    Interroge le graphe Neo4j pour trouver les sous-graphes pertinents (1-hop et 2-hop)
    liés aux entités mentionnées dans la question.
    
    Args:
        question (str): Question posée par l'utilisateur.
        driver (Driver, optional): Instance Driver Neo4j.
        top_k_entities (int): Nombre maximum d'entités à faire correspondre.
        
    Returns:
        List[Dict[str, str]]: Liste de triplets sous forme de dictionnaires {"source": ..., "relation": ..., "target": ...}
    """
    close_driver = False
    if driver is None:
        driver = get_neo4j_driver()
        close_driver = True

    try:
        # a. Extrait les entités de la question via le LLM
        question_entities = extract_entities_from_question(question)
        if not question_entities:
            logger.info("Aucune entité extraite de la question.")
            return []

        triplets = []
        seen_triplets = set()

        with driver.session() as session:
            for entity in question_entities:
                if not entity.strip():
                    continue
                    
                # b & c. Recherche d'entités identiques ou similaires et exploration 1-hop + 2-hop (multi-hop)
                # Cypher query à 1 saut
                cypher_1hop = """
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($entity_name)
                MATCH (e)-[r:RELATED]-(target:Entity)
                RETURN e.name AS source, r.type AS relation, target.name AS target
                LIMIT 25
                """
                results_1hop = session.run(cypher_1hop, entity_name=entity)
                for record in results_1hop:
                    t_key = (record["source"], record["relation"], record["target"])
                    if t_key not in seen_triplets:
                        seen_triplets.add(t_key)
                        triplets.append({
                            "source": record["source"],
                            "relation": record["relation"],
                            "target": record["target"]
                        })

                # Cypher query à 2 sauts (multi-hop)
                cypher_2hop = """
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($entity_name)
                MATCH (e)-[r1:RELATED]-(m:Entity)-[r2:RELATED]-(target:Entity)
                WHERE target <> e
                RETURN e.name AS s1, r1.type AS rel1, m.name AS t1,
                       m.name AS s2, r2.type AS rel2, target.name AS t2
                LIMIT 25
                """
                results_2hop = session.run(cypher_2hop, entity_name=entity)
                for record in results_2hop:
                    t_key1 = (record["s1"], record["rel1"], record["t1"])
                    if t_key1 not in seen_triplets:
                        seen_triplets.add(t_key1)
                        triplets.append({
                            "source": record["s1"],
                            "relation": record["rel1"],
                            "target": record["t1"]
                        })
                    t_key2 = (record["s2"], record["rel2"], record["t2"])
                    if t_key2 not in seen_triplets:
                        seen_triplets.add(t_key2)
                        triplets.append({
                            "source": record["s2"],
                            "relation": record["rel2"],
                            "target": record["t2"]
                        })

        return triplets

    except Exception as e:
        logger.error(f"Erreur lors de la requête du graphe pour la question '{question}': {e}")
        # D'après l'exigence : Doit gérer le cas où aucune entité n'est trouvée (retourner liste vide, pas d'exception)
        return []
    finally:
        if close_driver:
            driver.close()
