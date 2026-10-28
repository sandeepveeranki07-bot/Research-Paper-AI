"""Command-line document ingestion script for batch PDF processing and index management.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import get_settings
from ingestion.pipeline import IngestionPipeline
from storage.database import init_database
from storage.document_repository import DocumentRepository


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Research Intelligence Assistant — PDF Ingestion CLI"
    )
    parser.add_argument(
        "--path",
        type=str,
        help="Path to a PDF file or directory containing PDFs to ingest"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Recursively scan subdirectories for PDFs (default: True)"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Display current indexed document and chunk statistics"
    )
    parser.add_argument(
        "--delete",
        type=str,
        metavar="DOC_ID",
        help="Delete a document and purge its vectors from FAISS index by document ID"
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild the FAISS index from stored chunks in SQLite"
    )

    args = parser.parse_args()

    init_database()
    doc_repo = DocumentRepository()

    if args.stats:
        stats = doc_repo.get_document_stats()
        print("\n" + "=" * 50)
        print("RESEARCH ASSISTANT — INDEX STATISTICS")
        print("=" * 50)
        print(f"Total Indexed Documents: {stats['total_documents']}")
        print(f"Total Pages Processed:   {stats['total_pages']}")
        print(f"Total Chunks in FAISS:   {stats['total_chunks']}")
        print(f"Ingestion Failures:      {stats['ingestion_failures']}")
        print("=" * 50 + "\n")
        return

    if args.delete:
        pipeline = IngestionPipeline()
        success = pipeline.delete_document(args.delete)
        if success:
            print(f"Successfully deleted document '{args.delete}' and updated FAISS index.")
        else:
            print(f"Document ID '{args.delete}' not found.")
        return

    if args.rebuild_index:
        pipeline = IngestionPipeline()
        total = pipeline.rebuild_index()
        print(f"Successfully reconstructed FAISS index with {total} vectors.")
        return

    if not args.path:
        parser.print_help()
        print("\nError: Please provide a --path, --stats, or --rebuild-index argument.")
        sys.exit(1)

    target_path = Path(args.path)
    if not target_path.exists():
        print(f"Error: Path does not exist: {target_path}")
        sys.exit(1)

    pipeline = IngestionPipeline()

    def print_progress(current: int, total: int, msg: str) -> None:
        print(f"[{current}/{total}] {msg}")

    print(f"\nInitiating ingestion for: {target_path}")
    if target_path.is_file():
        res = pipeline.ingest_files([target_path], progress_callback=print_progress)
    else:
        res = pipeline.ingest_directory(target_path, recursive=args.recursive, progress_callback=print_progress)

    print("\n" + "=" * 50)
    print("INGESTION COMPLETED")
    print("=" * 50)
    print(res.summary())
    if res.errors:
        print("\nEncountered Errors:")
        for err in res.errors:
            print(f"  - {err['file']}: {err['error']}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
