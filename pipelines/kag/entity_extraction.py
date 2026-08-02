import os
import json
import time
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modèles Pydantic pour valider la structure JSON extraite du LLM
class Entity(BaseModel):
    name: str = Field(description="Nom exact de l'entité")
    type: str = Field(description="Type de l'entité (ex: Personne, Lieu, Organisation, Concept, Event, etc.)")

class Relation(BaseModel):
    source: str = Field(description="Nom de l'entité source")
    relation: str = Field(description="Type ou libellé de la relation (ex: est né à, membre de, a créé)")
    target: str = Field(description="Nom de l'entité cible")

class ExtractionResult(BaseModel):
    entities: List[Entity] = Field(default_factory=list, description="Liste des entités identifiées")
    relations: List[Relation] = Field(default_factory=list, description="Liste des relations identifiées")

SYSTEM_PROMPT = """Vous êtes un expert en extraction d'informations et en graphes de connaissances.
Votre tâche est d'analyser le texte fourni et d'en extraire :
1. Une liste d'entités clés (personnes, lieux, organisations, œuvres, événements, concepts, etc.).
2. Une liste de relations sous forme de triplets (entité_source, relation, entité_cible).

IMPORTANT: Vous devez répondre STRICTEMENT avec un objet JSON respectant le schéma suivant :
{
  "entities": [
    {"name": "Nom de l'entité", "type": "Type de l'entité"}
  ],
  "relations": [
    {"source": "Entité A", "relation": "type_de_relation", "target": "Entité B"}
  ]
}

Ne fournissez AUCUN texte explicatif en dehors du bloc JSON.
"""

def extract_entities_relations(text: str, client: Optional[Groq] = None, model: str = "llama-3.1-8b-instant") -> Dict[str, Any]:
    """
    Extrait les entités et relations d'un texte via le LLM Groq.
    
    Args:
        text (str): Texte source à analyser.
        client (Groq, optional): Client Groq pré-initialisé. Si None, crée une nouvelle instance.
        model (str): Modèle LLM Groq à utiliser.
        
    Returns:
        Dict[str, Any]: Dictionnaire au format {"entities": [...], "relations": [...]}
    """
    if not text or not text.strip():
        return {"entities": [], "relations": []}
        
    if client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("La variable d'environnement GROQ_API_KEY est manquante.")
        client = Groq(api_key=api_key)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Texte à analyser :\n{text}"}
    ]

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.1
            )
            raw_content = response.choices[0].message.content
            
            # Parsing et validation Pydantic
            parsed_data = ExtractionResult.model_validate_json(raw_content)
            return parsed_data.model_dump()
            
        except Exception as e:
            logger.warning(f"Tentative {attempt + 1}/{max_retries} échouée pour l'extraction : {e}")
            # Si échec de parsing lors de la première tentative, on réessaie en réinsistant dans les prompts
            if attempt < max_retries - 1:
                # Pause pour gérer le Rate Limiting (429) le cas échéant
                time.sleep(2 ** attempt)
            else:
                logger.error(f"Échec définitif d'extraction d'entités pour le texte. Erreur : {e}")
                return {"entities": [], "relations": []}
