"""Offline deterministic vectors for storage tests, not embedding quality evidence."""
import pytest
import agentmind.core.memory as module


class FakeEmbeddings:
    def __init__(self, name):
        pass

    def encode(self, text):
        text = text.lower()
        if any(word in text for word in ('sport', 'outdoor', 'basketball', 'soccer', 'hiking')):
            return [1.0, 0.0, 0.0]
        if any(word in text for word in ('python', 'programming', 'preferences')):
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


@pytest.fixture(autouse=True)
def offline_embeddings(monkeypatch):
    monkeypatch.setattr(module, 'EmbeddingModel', FakeEmbeddings)
