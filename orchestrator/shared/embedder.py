"""Embedding via Azure OpenAI models."""

from __future__ import annotations

import os

import numpy as np
from numpy.typing import NDArray
from openai import AzureOpenAI


class Embedder:
    """Embedding via Azure OpenAI text-embedding-ada-002."""

    def __init__(
        self,
        model: str = "text-embedding-ada-002",
        *,
        azure_endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
    ) -> None:
        self._model = model
        self._client = AzureOpenAI(
            azure_endpoint=azure_endpoint or os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=api_key or os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version=api_version or os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01"),
        )
        self.total_calls: int = 0
        self.total_tokens: int = 0

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        """Encode a batch of texts into embeddings."""
        response = self._client.embeddings.create(input=texts, model=self._model)
        self.total_calls += 1
        if response.usage:
            self.total_tokens += response.usage.total_tokens
        return np.array(
            [item.embedding for item in response.data], dtype=np.float32,
        )

    def reset_usage(self) -> None:
        """Reset call and token counters."""
        self.total_calls = 0
        self.total_tokens = 0
