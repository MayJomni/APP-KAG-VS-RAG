import os
import logging
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from neo4j import GraphDatabase, Driver
from dotenv import load_dotenv
from .entity_extraction import extract_entities_relations

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InMemoryRecord(dict):
    def __getitem__(self, item):
        return super().__getitem__(item)

class InMemorySession:
    def __init__(self, db):
        self.db = db

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def run(self, query, **params):
        query_str = query.strip()
        if "DETACH DELETE" in query_str:
            self.db["nodes"].clear()
            self.db["relations"].clear()
            return []

        if "MERGE (e:Entity" in query_str:
            name = params.get("name", "").strip()
            e_type = params.get("type", "Concept").strip()
            doc_title = params.get("doc_title", "")
            if name:
                if name not in self.db["nodes"]:
                    self.db["nodes"][name] = {"name": name, "type": e_type, "source_docs": {doc_title}}
                else:
                    self.db["nodes"][name]["source_docs"].add(doc_title)
            return []

        if "MERGE (a:Entity" in query_str:
            src = params.get("source", "").strip()
            tgt = params.get("target", "").strip()
            rel_type = params.get("rel_type", "RELATED_TO").strip()
            doc_title = params.get("doc_title", "")
            if src and tgt:
                if src not in self.db["nodes"]:
                    self.db["nodes"][src] = {"name": src, "type": "Concept", "source_docs": {doc_title}}
                if tgt not in self.db["nodes"]:
                    self.db["nodes"][tgt] = {"name": tgt, "type": "Concept", "source_docs": {doc_title}}
                
                rel_key = (src, rel_type, tgt)
                if rel_key not in self.db["relations"]:
                    self.db["relations"][rel_key] = {"source": src, "relation": rel_type, "target": tgt, "source_docs": {doc_title}}
                else:
                    self.db["relations"][rel_key]["source_docs"].add(doc_title)
            return []

        if "count(n) AS node_count" in query_str:
            return [InMemoryRecord({"node_count": len(self.db["nodes"])})]

        if "count(r) AS rel_count" in query_str:
            return [InMemoryRecord({"rel_count": len(self.db["relations"])})]

        if "MATCH (e)-[r:RELATED]-(target:Entity)" in query_str:
            entity_name = params.get("entity_name", "").lower()
            records = []
            for (src, rel, tgt) in self.db["relations"].keys():
                if entity_name in src.lower():
                    records.append(InMemoryRecord({"source": src, "relation": rel, "target": tgt}))
                elif entity_name in tgt.lower():
                    records.append(InMemoryRecord({"source": tgt, "relation": rel, "target": src}))
            return records[:25]

        if "MATCH (e)-[r1:RELATED]-(m:Entity)-[r2:RELATED]-(target:Entity)" in query_str:
            entity_name = params.get("entity_name", "").lower()
            records = []
            first_hop = []
            for (src, rel, tgt) in self.db["relations"].keys():
                if entity_name in src.lower():
                    first_hop.append((src, rel, tgt))
                elif entity_name in tgt.lower():
                    first_hop.append((tgt, rel, src))

            for s1, r1, t1 in first_hop:
                m_node = t1
                for (src2, r2, tgt2) in self.db["relations"].keys():
                    if src2.lower() == m_node.lower() and tgt2.lower() != s1.lower():
                        records.append(InMemoryRecord({
                            "s1": s1, "rel1": r1, "t1": t1,
                            "s2": m_node, "rel2": r2, "t2": tgt2
                        }))
                    elif tgt2.lower() == m_node.lower() and src2.lower() != s1.lower():
                        records.append(InMemoryRecord({
                            "s1": s1, "rel1": r1, "t1": t1,
                            "s2": m_node, "rel2": r2, "t2": src2
                        }))
            return records[:25]

        return []

class InMemoryGraphDriver:
    """
    Driver de secours en mémoire si l'instance Neo4j locale/Docker n'est pas accessible.
    """
    def __init__(self):
        self.db = {"nodes": {}, "relations": {}}

    def verify_connectivity(self):
        pass

    def session(self):
        return InMemorySession(self.db)

    def close(self):
        pass

def get_neo4j_driver(uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None):
    """
    Tente d'initialiser le driver Neo4j réel. En cas d'impossibilité de connexion,
    rebascule vers le driver en mémoire InMemoryGraphDriver.
    """
    uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = user or os.getenv("NEO4J_USER", "neo4j")
    password = password or os.getenv("NEO4J_PASSWORD", "password123")
    
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        logger.info("Connexion établie à Neo4j.")
        return driver
    except Exception as e:
        logger.warning(f"Neo4j non disponible ({e}). Basculement vers le graphe en mémoire (InMemoryGraphDriver).")
        return InMemoryGraphDriver()

def clear_graph(driver):
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    logger.info("Graphe nettoyé avec succès.")

def insert_extraction_results(session, extraction_data: Dict[str, Any], doc_title: str):
    entities = extraction_data.get("entities", [])
    relations = extraction_data.get("relations", [])

    for entity in entities:
        name = entity.get("name", "").strip()
        e_type = entity.get("type", "Concept").strip()
        if not name:
            continue
            
        cypher_entity = """
        MERGE (e:Entity {name: $name})
        ON CREATE SET e.type = $type, e.source_docs = [$doc_title]
        ON MATCH SET e.source_docs = CASE 
            WHEN NOT $doc_title IN e.source_docs THEN e.source_docs + $doc_title 
            ELSE e.source_docs 
        END
        """
        session.run(cypher_entity, name=name, type=e_type, doc_title=doc_title)

    for rel in relations:
        src = rel.get("source", "").strip()
        rel_type = rel.get("relation", "RELATED_TO").strip()
        tgt = rel.get("target", "").strip()
        if not src or not tgt:
            continue

        cypher_relation = """
        MERGE (a:Entity {name: $source})
        MERGE (b:Entity {name: $target})
        MERGE (a)-[r:RELATED {type: $rel_type}]->(b)
        ON CREATE SET r.source_docs = [$doc_title]
        ON MATCH SET r.source_docs = CASE 
            WHEN NOT $doc_title IN r.source_docs THEN r.source_docs + $doc_title 
            ELSE r.source_docs 
        END
        """
        session.run(cypher_relation, source=src, target=tgt, rel_type=rel_type, doc_title=doc_title)

def get_graph_summary(driver) -> Dict[str, int]:
    with driver.session() as session:
        node_res = session.run("MATCH (n:Entity) RETURN count(n) AS node_count")
        node_count = list(node_res)[0]["node_count"]

        rel_res = session.run("MATCH ()-[r:RELATED]->() RETURN count(r) AS rel_count")
        rel_count = list(rel_res)[0]["rel_count"]

    return {"node_count": node_count, "relation_count": rel_count}

def build_graph(examples: List[Dict[str, Any]], driver=None, clear_existing: bool = True) -> Dict[str, int]:
    close_driver = False
    if driver is None:
        driver = get_neo4j_driver()
        close_driver = True

    try:
        if clear_existing:
            clear_graph(driver)

        processed_docs = set()

        with driver.session() as session:
            for example in tqdm(examples, desc="Extraction et construction du graphe"):
                docs = example.get("documents", [])
                for doc in docs:
                    title = doc.get("title", "Sans titre")
                    text = doc.get("text", "")
                    
                    doc_key = f"{title}:{text[:100]}"
                    if doc_key in processed_docs:
                        continue
                    processed_docs.add(doc_key)

                    extraction_data = extract_entities_relations(text)
                    insert_extraction_results(session, extraction_data, doc_title=title)

        summary = get_graph_summary(driver)
        logger.info(f"Construction du graphe terminée : {summary['node_count']} nœuds, {summary['relation_count']} relations.")
        print(f"\n--- RÉSUMÉ DU GRAPHE ---")
        print(f"Nœuds (Entités) créés : {summary['node_count']}")
        print(f"Relations créées      : {summary['relation_count']}")
        print("------------------------\n")
        return summary

    finally:
        if close_driver:
            driver.close()


def build_graph_from_docs(docs: List[Dict[str, Any]], driver=None, clear_existing: bool = True) -> Dict[str, int]:
    """
    Construit le graphe KAG à partir d'une liste de documents pré-traités.
    Chaque doc doit avoir 'title' et 'text'.
    Compatible avec les chunks produits par document_processor.py.
    """
    close_driver = False
    if driver is None:
        driver = get_neo4j_driver()
        close_driver = True

    try:
        if clear_existing:
            clear_graph(driver)

        with driver.session() as session:
            for doc in tqdm(docs, desc="Construction graphe KAG"):
                title = doc.get("title", "Document")
                text  = doc.get("text", "")
                if not text.strip():
                    continue
                extraction_data = extract_entities_relations(text)
                insert_extraction_results(session, extraction_data, doc_title=title)

        summary = get_graph_summary(driver)
        logger.warning(f"Graphe construit : {summary['node_count']} entités, {summary['relation_count']} relations.")
        return summary

    finally:
        if close_driver:
            driver.close()
