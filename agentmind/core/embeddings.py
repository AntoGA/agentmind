"""
Embedding Models
Support for different embedding models.
"""

import os
from typing import List, Optional
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential


class EmbeddingModel:
    """Model for creating embeddings"""
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name
        
        if model_name.startswith("text-embedding"):
            self._init_openai()
        elif "sentence-transformers" in model_name or model_name.startswith("all-"):
            self._init_sentence_transformers()
        else:
            raise ValueError(f"Unsupported embedding model: {model_name}")
    
    def _init_openai(self):
        """Initialize OpenAI embeddings"""
        try:
            from openai import OpenAI
            self.client = OpenAI()
            self.provider = "openai"
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
    
    def _init_sentence_transformers(self):
        """Initialize Sentence Transformers"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.provider = "sentence_transformers"
        except ImportError:
            raise ImportError("Sentence Transformers not installed. Run: pip install sentence-transformers")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def encode(self, text: str) -> List[float]:
        """
        Create embedding for text.
        
        Args:
            text: Input text
        
        Returns:
            Embedding vector
        
        Example:
            >>> embedding = model.encode("Hello, world!")
            >>> print(len(embedding))  # 1536 for text-embedding-3-small
        """
        if self.provider == "openai":
            return self._encode_openai(text)
        elif self.provider == "sentence_transformers":
            return self._encode_sentence_transformers(text)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _encode_openai(self, text: str) -> List[float]:
        """Create embedding via OpenAI API"""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text
        )
        return response.data[0].embedding
    
    def _encode_sentence_transformers(self, text: str) -> List[float]:
        """Create embedding via Sentence Transformers"""
        embedding = self.model.encode(text)
        return embedding.tolist()
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for list of texts.
        
        Args:
            texts: List of texts
        
        Returns:
            List of embedding vectors
        """
        if self.provider == "openai":
            return self._encode_batch_openai(texts)
        elif self.provider == "sentence_transformers":
            return self._encode_batch_sentence_transformers(texts)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _encode_batch_openai(self, texts: List[str]) -> List[List[float]]:
        """Batch encoding via OpenAI"""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        return [item.embedding for item in response.data]
    
    def _encode_batch_sentence_transformers(self, texts: List[str]) -> List[List[float]]:
        """Batch encoding via Sentence Transformers"""
        embeddings = self.model.encode(texts)
        return [emb.tolist() for emb in embeddings]
