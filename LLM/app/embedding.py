from __future__ import annotations
import os
from typing import List
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

class Embedder:
    """
    Thin wrapper over OpenAI embeddings.
    """

    def __init__(self, model: str | None = None):
        self.model = model or DEFAULT_EMBED_MODEL
        self.client = OpenAI()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts and return vectors.
        """
        # OpenAI API supports batching; we send all at once when reasonable
        resp = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [d.embedding for d in resp.data]

    def embed_text(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]
