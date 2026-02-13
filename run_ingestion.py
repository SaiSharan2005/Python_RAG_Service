"""Entry point for the ingestion pipeline.

Usage:
    python run_ingestion.py
    python run_ingestion.py --json-path ../lucene-service/chunk-exports
    python run_ingestion.py --json-path /path/to/single-file.json
"""
from __future__ import annotations

import argparse

from app.config import settings, log
from app.ingestion import run_ingestion


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest chunked JSON into Qdrant")
    parser.add_argument(
        "--json-path",
        type=str,
        default=None,
        help="Path to a .json file or directory of .json files (overrides .env)",
    )
    args = parser.parse_args()

    if args.json_path:
        settings.ingestion_json_path = args.json_path

    log.info("JSON source: %s", settings.ingestion_json_path)
    run_ingestion()


if __name__ == "__main__":
    main()
