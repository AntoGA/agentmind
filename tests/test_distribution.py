"""Run against the installed distribution, including from outside the checkout."""
from importlib.metadata import metadata, version
from agentmind import Memory, __version__


class Embeddings:
    def encode(self, text):
        return [1.0, 0.0]


def test_installed_metadata():
    assert version("agentmind") == __version__
    data = metadata("agentmind")
    assert data["Requires-Python"] == ">=3.10"
    core = [r for r in data.get_all("Requires-Dist") if "extra ==" not in r]
    assert {r.split(">=")[0] for r in core} == {"numpy", "pydantic", "tenacity"}


def test_custom_provider_roundtrip(tmp_path):
    memory = Memory(db_path=str(tmp_path / "memory.db"), embeddings=Embeddings())
    mid = memory.remember("test", "packaged memory")
    result = memory.recall("test", "memory")
    assert result[0].id == mid
    assert result[0].content == "packaged memory"
    memory.forget("test")
    assert memory.recall("test") == []


def test_missing_provider_has_actionable_error():
    import subprocess
    import sys
    check = subprocess.run(
        [sys.executable, "-I", "-c",
         "import sys; sys.modules['openai'] = None; "
         "from agentmind.core.embeddings import EmbeddingModel; EmbeddingModel()"],
        capture_output=True, text=True,
    )
    assert check.returncode != 0
    assert "agentmind[openai]" in check.stderr
