"""
search_engine.py
Builds a semantic index over PageChunk objects using Sentence Transformers
and NumPy cosine similarity. No external vector DB required.
"""

from typing import List, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from pdf_processor import PageChunk

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class SemanticSearchEngine:
    """
    Encodes all page chunks into embeddings and answers nearest-neighbour
    queries via cosine similarity.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.chunks: List[PageChunk] = []
        self.embeddings: Optional[np.ndarray] = None  # shape: (N, D)

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build_index(self, chunks: List[PageChunk], batch_size: int = 32) -> None:
        """
        Encode all chunk texts and store the normalised embedding matrix.

        Args:
            chunks:     PageChunk objects to index.
            batch_size: Sentences processed per forward pass.
        """
        if not chunks:
            raise ValueError("Cannot build an index from an empty chunk list.")

        self.chunks = list(chunks)
        texts = [c.text for c in self.chunks]

        raw = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # unit-norm: dot product == cosine similarity
        )
        self.embeddings = raw.astype(np.float32)

    def add_chunks(self, new_chunks: List[PageChunk], batch_size: int = 32) -> None:
        """
        Incrementally index additional chunks without re-encoding the entire dataset.
        """
        if not new_chunks:
            return

        if not self.is_ready or self.embeddings is None:
            self.build_index(new_chunks, batch_size=batch_size)
            return

        texts = [c.text for c in new_chunks]
        raw_new = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        self.chunks.extend(new_chunks)
        self.embeddings = np.vstack([self.embeddings, raw_new])

    @property
    def is_ready(self) -> bool:
        return self.embeddings is not None and len(self.chunks) > 0

    @property
    def document_names(self) -> List[str]:
        """Returns unique sorted list of document names indexed."""
        return sorted(list({c.doc_name for c in self.chunks}))

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_filter: Optional[str] = None,
        min_score: float = 0.0,
    ) -> List[Tuple[PageChunk, float]]:
        """
        Return the top-k most relevant chunks for *query*.

        Args:
            query: Natural-language question or keyword string.
            top_k: Number of results to return.
            doc_filter: Optional document name to restrict results to.
            min_score: Minimum cosine similarity score threshold (0.0 to 1.0).

        Returns:
            List of (PageChunk, score) tuples, highest score first.
        """
        if not self.is_ready or self.embeddings is None:
            return []

        if not query.strip():
            return []

        q_vec = self.model.encode(
            [query.strip()],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        # Dot product of unit vectors == cosine similarity
        scores: np.ndarray = (self.embeddings @ q_vec.T).squeeze()
        if scores.ndim == 0:
            scores = np.array([float(scores)])

        # Apply optional document filtering
        candidate_indices = list(range(len(self.chunks)))
        if doc_filter and doc_filter != "All Documents":
            candidate_indices = [
                i for i in candidate_indices if self.chunks[i].doc_name == doc_filter
            ]

        if not candidate_indices:
            return []

        # Filter by minimum score
        ranked_candidates = []
        for i in candidate_indices:
            score_val = float(scores[i])
            if score_val >= min_score:
                ranked_candidates.append((self.chunks[i], score_val))

        # Sort descending by score
        ranked_candidates.sort(key=lambda x: x[1], reverse=True)

        return ranked_candidates[:top_k]
