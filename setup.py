import os
from pathlib import Path
from setuptools import setup

setup(
    name = "cecil",
    version = "1.0.0",
    install_requires = [
        "IPython",
        "langchain",
        "langgraph",
        "langchain_community",
        "ollama",
        "omegaconf",
        "pydantic",
        "uvicorn",
    ],
    author = "Dylan Miller",
    author_email = "dylanamiller3@gmail.com",
    description = ("jupyter ai assistant"),
    keywords = "jupyter ai assistant",
    url = "http://packages.python.org/an_example_pypi_project",
    classifiers=[
        "Development Status :: 3 - Alpha",
    ],
)
