from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="agentmind",
    version="0.2.0",
    author="AntoGA",
    author_email="your.email@example.com",
    description="Persistent memory for AI agents that actually works",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AntoGA/agentmind",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "openai>=1.0.0",
        "chromadb>=0.4.0",
        "sqlalchemy>=2.0.0",
        "tqdm>=4.65.0",
        "tenacity>=8.2.0",
    ],
    extras_require={
        "langchain": ["langchain>=0.3.0", "langchain-openai>=0.2.0"],
        "redis": ["redis>=5.0.0"],
        "postgresql": ["psycopg2-binary>=2.9.0"],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "agentmind=agentmind.cli:main",
        ],
    },
)
