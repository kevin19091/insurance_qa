"""CLI to inspect raw Chroma DB entries.

Usage:
    python -m src.chroma_inspect [--path data/chroma]
"""

import argparse
import sys

import chromadb


def dump_collection(chroma_path: str, collection_name: str = "insurance_policy") -> list[dict]:
    client = chromadb.PersistentClient(path=chroma_path)
    try:
        col = client.get_collection(collection_name)
    except ValueError:
        print(f"Collection '{collection_name}' not found at {chroma_path}", file=sys.stderr)
        sys.exit(1)

    result = col.get()
    entries = []
    for i, doc_id in enumerate(result["ids"]):
        entries.append(
            {
                "id": doc_id,
                "text": (result["documents"][i] or "")[:500],
                "metadata": result["metadatas"][i] if result["metadatas"] else {},
            }
        )
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Chroma DB entries")
    parser.add_argument("--path", default="data/chroma", help="Path to Chroma DB directory")
    args = parser.parse_args()

    entries = dump_collection(args.path)
    print(f"Found {len(entries)} entries in Chroma DB at {args.path}")
    print()
    for entry in entries:
        print(f"--- {entry['id']} ---")
        print(f"Metadata: {entry['metadata']}")
        print(f"Text: {entry['text'][:200]}...")
        print()


if __name__ == "__main__":
    main()
