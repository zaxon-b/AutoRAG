"""Backfill the experimental alpha-on-white image vector on existing ES docs.

This command deliberately does not parse the PDF, rebuild icon groups, create
chunks, or alter PostgreSQL. It updates one additional vector field on each
existing searchable visual document.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from service.es_index_service import (
    update_image_embedding_v2,
    visual_documents_for_image_reembedding,
)
from service.image_embedding_service import (
    IMAGE_EMBEDDING_PROFILE_ALPHA_WHITE,
    embed_images,
)
from service.manual_storage_service import storage_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill image_embedding_vector_v2 without rebuilding chunks.")
    parser.add_argument("--manual-id", default="demo_manual")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0, help="0 processes every current visual ES document.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def local_image_path(value: str | None) -> Path | None:
    if not value:
        return None
    if value.startswith("/storage/"):
        return storage_root() / value.removeprefix("/storage/")
    path = Path(value)
    return path if path.is_absolute() else storage_root() / path


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    documents = visual_documents_for_image_reembedding(args.manual_id)
    if args.limit:
        documents = documents[: args.limit]

    usable: list[tuple[str, Path]] = []
    missing: list[str] = []
    for document in documents:
        path = local_image_path(document.get("image_path"))
        if path is None or not path.exists():
            missing.append(str(document.get("_id") or ""))
            continue
        usable.append((str(document["_id"]), path))

    print(
        f"manual={args.manual_id} documents={len(documents)} "
        f"usable={len(usable)} missing={len(missing)} dry_run={args.dry_run}",
        flush=True,
    )
    if args.dry_run:
        return

    succeeded = 0
    failed = 0
    for start in range(0, len(usable), args.batch_size):
        batch = usable[start : start + args.batch_size]
        vectors = embed_images(
            [path for _, path in batch],
            preprocessing=IMAGE_EMBEDDING_PROFILE_ALPHA_WHITE,
        )
        updates = [
            (document_id, vector)
            for (document_id, _), vector in zip(batch, vectors)
            if vector is not None
        ]
        update_image_embedding_v2(updates)
        succeeded += len(updates)
        failed += len(batch) - len(updates)
        print(
            f"processed={min(start + len(batch), len(usable))}/{len(usable)} "
            f"succeeded={succeeded} failed={failed}",
            flush=True,
        )

    if failed:
        raise RuntimeError(f"Image v2 backfill incomplete: succeeded={succeeded} failed={failed}")


if __name__ == "__main__":
    main()
