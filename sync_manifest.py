"""
Sync manifest.json to match only PDF files present in data/raw.

Use this after manually curating data/raw: keeps manifest entries only for
sources whose filename exists in that directory. Run ``rag_pipeline ingest --force`` afterward.
"""

import json
from pathlib import Path
from typing import Optional

from config import get_path_config


def sync_manifest_from_raw(
    manifest_path: Optional[Path] = None,
    raw_dir: Optional[Path] = None,
) -> dict:
    """
    Filter manifest sources to only those with files in data/raw.

    Returns:
        {"kept": N, "removed": N, "manifest_path": Path}
    """
    paths = get_path_config()
    manifest_path = manifest_path or paths.manifest_path
    raw_dir = raw_dir or paths.raw_dir

    raw_files = {f.name for f in raw_dir.iterdir() if f.suffix.lower() == ".pdf"}
    if not raw_files:
        raise FileNotFoundError(f"No PDFs found in {raw_dir}")

    with open(manifest_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    sources = data.get("sources", [])
    original_count = len(sources)

    kept = [s for s in sources if s.get("filename", "") in raw_files]
    removed_count = original_count - len(kept)

    data["sources"] = kept
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)

    return {
        "kept": len(kept),
        "removed": removed_count,
        "manifest_path": str(manifest_path),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sync manifest to data/raw contents")
    args = parser.parse_args()
    result = sync_manifest_from_raw()
    print(f"Manifest synced: {result['kept']} sources kept, {result['removed']} removed")
    print(f"Updated: {result['manifest_path']}")
