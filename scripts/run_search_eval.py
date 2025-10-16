#!/usr/bin/env python3
"""Run the FAISS benchmark using Hydra configuration files."""

from img_search.search.evaluate import app


if __name__ == "__main__":
    app()
