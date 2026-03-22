# experiments/models.py
 
from dataclasses import dataclass, field
from typing import List, Optional
 
 
@dataclass
class RAGConfig:
    """
    Décrit une configuration d'expérience RAG complète.
    Chaque champ correspond à une variable du système qu'on veut tester.
    """
    name: str                            # Identifiant unique (ex: "single_k8")
    strategy: str                        # "single" | "multi"
    queries: List[str]                   # 1 requête (single) ou N (multi)
    k: int                               # Nombre de chunks à récupérer par requête
    prompt_template: str                 # Template LCEL avec {context}
    description: str = ""               # Note humaine sur ce qu'on teste
 
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "strategy": self.strategy,
            "queries": self.queries,
            "k": self.k,
            "prompt_template": self.prompt_template,
            "description": self.description,
        }
 
 
@dataclass
class ChunkInfo:
    """Métadonnées d'un chunk récupéré depuis ChromaDB."""
    page: int
    chunk_index: int
    preview: str                         # 120 premiers caractères
 
 
@dataclass
class ExperimentResult:
    """
    Résultat complet d'une expérience.
    Sérialisable en JSON pour stockage et réutilisation.
    """
    config_name: str
    config: RAGConfig
    summary: str                         # Texte généré par le LLM
    chunks: List[ChunkInfo]              # Chunks récupérés (avec métadonnées)
    tokens_prompt: int
    tokens_completion: int
    tokens_total: int
    duration_seconds: float
    error: Optional[str] = None
    context: str = ""   # Chunks RAG concaténés — source de vérité pour l'évaluation,← les chunks RAG utilisés pour générer le summary,il faut stocker le contexte RAG
 
    # Calculés à la création
    chunks_retrieved: int = 0
    unique_pages: List[int] = field(default_factory=list)
    summary_word_count: int = 0
 
    def __post_init__(self):
        self.chunks_retrieved = len(self.chunks)
        self.unique_pages = sorted(set(c.page for c in self.chunks))
        self.summary_word_count = len(self.summary.split()) if self.summary else 0
 
    def to_dict(self) -> dict:
        return {
            "config_name": self.config_name,
            "config": self.config.to_dict(),
            "summary": self.summary,
            "chunks": [
                {"page": c.page, "chunk_index": c.chunk_index, "preview": c.preview}
                for c in self.chunks
            ],
            "tokens_prompt": self.tokens_prompt,
            "tokens_completion": self.tokens_completion,
            "tokens_total": self.tokens_total,
            "duration_seconds": round(self.duration_seconds, 2),
            "chunks_retrieved": self.chunks_retrieved,
            "unique_pages": self.unique_pages,
            "summary_word_count": self.summary_word_count,
            "context": self.context,
            "error": self.error,
        }
 
    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentResult":
        """Recharge un résultat depuis un dict JSON."""
        config = RAGConfig(**data["config"])
        chunks = [ChunkInfo(**c) for c in data["chunks"]]
        return cls(
            config_name=data["config_name"],
            config=config,
            summary=data["summary"],
            chunks=chunks,
            tokens_prompt=data["tokens_prompt"],
            tokens_completion=data["tokens_completion"],
            tokens_total=data["tokens_total"],
            duration_seconds=data["duration_seconds"],
            context=data.get("context", ""),
            error=data.get("error"),
        )
 