"""
document_processor.py
Parse et découpe en chunks les documents uploadés par l'utilisateur.
Supporte : PDF, TXT, MD, DOCX, et texte brut.
"""

import io
import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def extract_text_from_file(filename: str, content: bytes) -> str:
    """Extrait le texte brut d'un fichier selon son extension."""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else "txt"

    if ext == "pdf":
        return _extract_pdf(content)
    elif ext in ("txt", "md", "rst", "csv"):
        return _extract_text(content)
    elif ext in ("docx",):
        return _extract_docx(content)
    else:
        # Tentative décodage brut
        try:
            return content.decode("utf-8", errors="replace")
        except Exception:
            return ""


def _extract_pdf(content: bytes) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(content))
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text.strip())
        return "\n\n".join(pages)
    except ImportError:
        logger.error("pypdf non installé. Lancez : uv add pypdf")
        return ""
    except Exception as e:
        logger.error(f"Erreur lecture PDF: {e}")
        return ""


def _extract_text(content: bytes) -> str:
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return content.decode(enc)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def _extract_docx(content: bytes) -> str:
    try:
        from docx import Document
        doc = Document(io.BytesIO(content))
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except ImportError:
        logger.error("python-docx non installé. Lancez : uv add python-docx")
        return ""
    except Exception as e:
        logger.error(f"Erreur lecture DOCX: {e}")
        return ""


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> List[Dict[str, Any]]:
    """
    Découpe un texte en chunks avec chevauchement.
    Essaie de couper aux fins de paragraphes/phrases.
    """
    # Nettoyage
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    if not text:
        return []

    # Découpage par paragraphes d'abord
    paragraphs = [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]

    chunks = []
    current = ""
    chunk_idx = 0

    for para in paragraphs:
        # Si le paragraphe seul dépasse chunk_size, on le découpe par phrases
        if len(para) > chunk_size * 2:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                if len(current) + len(sent) + 1 > chunk_size and current:
                    chunks.append({
                        "chunk_id": chunk_idx,
                        "text": current.strip(),
                        "title": f"Passage {chunk_idx + 1}"
                    })
                    chunk_idx += 1
                    # Chevauchement : garder les N derniers mots
                    words = current.split()
                    current = " ".join(words[-overlap // 6:]) + " " + sent
                else:
                    current = (current + " " + sent).strip()
        else:
            if len(current) + len(para) + 2 > chunk_size and current:
                chunks.append({
                    "chunk_id": chunk_idx,
                    "text": current.strip(),
                    "title": f"Passage {chunk_idx + 1}"
                })
                chunk_idx += 1
                # Chevauchement
                words = current.split()
                current = " ".join(words[-overlap // 6:]) + "\n\n" + para
            else:
                current = (current + "\n\n" + para).strip() if current else para

    if current.strip():
        chunks.append({
            "chunk_id": chunk_idx,
            "text": current.strip(),
            "title": f"Passage {chunk_idx + 1}"
        })

    logger.info(f"Document découpé en {len(chunks)} chunks (taille ~{chunk_size} mots).")
    return chunks


def build_documents_from_chunks(chunks: List[Dict], source_name: str = "document") -> List[Dict]:
    """Formate les chunks comme des documents compatibles RAG et KAG."""
    docs = []
    for c in chunks:
        docs.append({
            "title": c["title"],
            "text": c["text"],
            "source": source_name,
            "example_id": f"{source_name}_{c['chunk_id']}"
        })
    return docs
