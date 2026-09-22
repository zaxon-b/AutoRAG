"""Synchronous PDF ingestion for the four-chunk hierarchical RAG schema."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import pdfplumber
from sqlalchemy import delete
from sqlalchemy.orm import Session

from models.base import Base
from models.manual import (
    Manual,
    ManualChunk,
    ManualIconGroup,
    ManualIconPrototype,
    ManualImageAsset,
    ManualImageOccurrence,
    ManualPage,
)
from service.bookmark_outline_service import extract_bookmark_outline, section_at_position
from service.embedding_service import embed_texts
from service.es_index_service import clear_manual, index_chunks, prepare_index_for_ingestion
from service.hierarchical_chunk_service import build_hierarchical_text_chunks, stable_chunk_id
from service.icon_asset_grouping_service import build_native_icon_groups
from service.icon_catalog_service import clear_icon_group_cache
from service.image_embedding_service import (
    IMAGE_EMBEDDING_FIELD,
    IMAGE_EMBEDDING_PROFILE_ALPHA_WHITE,
    IMAGE_EMBEDDING_V2_FIELD,
    embed_images,
)
from service.layout_analysis_service import analyze_pdf_layout
from service.manual_storage_service import DEFAULT_MANUAL_ID, save_manual_pdf, storage_root
from service.multimodal_embedding_service import MULTIMODAL_EMBEDDING_FIELD, embed_multimodal_chunks
from service.native_pdf_image_service import extract_native_image_catalog
from service.page_render_service import render_pdf_pages
from service.visual_content_association_service import associate_icon_content, associate_operation_content
from service.visual_asset_classification_service import (
    CLASSIFICATION_SCHEMA_VERSION,
    classify_visual_assets,
)


logger = logging.getLogger(__name__)


def ensure_tables(db: Session) -> None:
    Base.metadata.create_all(bind=db.get_bind())


def ingest_manual(db: Session, file_obj, filename: str, manual_id: str = DEFAULT_MANUAL_ID) -> dict[str, Any]:
    """Replace one manual and build its complete PostgreSQL/ES evidence graph."""

    ensure_tables(db)
    if not filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF owner manuals are supported in this demo.")

    classification_cache = _load_visual_classification_cache(db, manual_id)
    pdf_path = save_manual_pdf(file_obj, filename, manual_id)
    page_count, pages = _extract_pdf_pages(pdf_path)
    layout_by_page = analyze_pdf_layout(pdf_path)
    bookmark_nodes, outline_warnings = extract_bookmark_outline(pdf_path, layout_by_page)
    for warning in outline_warnings:
        logger.warning("PDF outline: %s", warning)
    rendered_pages = render_pdf_pages(pdf_path, pdf_path.parent / "pages")

    prepare_index_for_ingestion()
    _delete_active_manual(db, manual_id)
    clear_manual(manual_id)

    manual = Manual(
        manual_id=manual_id,
        vehicle_model=os.getenv("DEMO_VEHICLE_MODEL", "demo_ev"),
        manual_version=os.getenv("DEMO_MANUAL_VERSION", "public_manual"),
        file_name=filename,
        file_path=str(pdf_path),
        page_count=page_count,
        status="processing",
    )
    db.add(manual)
    for page_info in pages:
        rendered = rendered_pages.get(page_info["page_number"], {})
        db.add(
            ManualPage(
                manual_id=manual_id,
                page_number=page_info["page_number"],
                image_path=rendered.get("image_path"),
                width=rendered.get("width") or page_info["width"],
                height=rendered.get("height") or page_info["height"],
                ocr_text=page_info["text"],
            )
        )
    # Commit the processing marker before the slower embedding/VLM stages, so
    # the status endpoint reflects that ingestion is alive.
    db.commit()

    text_payloads = build_hierarchical_text_chunks(manual_id, bookmark_nodes, layout_by_page)
    visual_catalog = _build_native_visual_catalog(
        manual_id=manual_id,
        pdf_path=pdf_path,
        pages=pages,
        rendered_pages=rendered_pages,
        layout_by_page=layout_by_page,
        bookmark_nodes=bookmark_nodes,
        classification_cache=classification_cache,
    )
    visual_payloads = visual_catalog["visual_chunks"]

    db.add_all([_asset_to_row(asset) for asset in visual_catalog["assets"]])
    db.flush()
    db.add_all(
        [
            ManualIconGroup(
                icon_group_id=group["icon_group_id"],
                manual_id=manual_id,
                canonical_icon_name=group.get("canonical_icon_name"),
                representative_chunk_id=group["group_chunk_id"],
                member_count=int(group.get("member_count") or 0),
            )
            for group in visual_catalog["groups"]
        ]
    )
    db.flush()
    db.add_all([_payload_to_row(payload) for payload in [*text_payloads, *visual_payloads]])
    db.flush()
    db.add_all(
        [_occurrence_to_row(occurrence) for occurrence in visual_catalog["occurrences"]]
    )
    db.add_all(
        [
            _prototype_to_row(manual_id, prototype)
            for prototype in visual_catalog["prototypes"]
        ]
    )
    db.commit()

    leaves = [payload for payload in text_payloads if payload.get("is_leaf") is True]
    leaf_inputs = [f"{payload.get('section_title') or ''}\n{payload.get('content_text') or ''}" for payload in leaves]
    for payload, vector in zip(leaves, embed_texts(leaf_inputs)):
        payload["text_vector"] = vector

    visual_index_payloads = visual_catalog["visual_index_payloads"]
    # The native image vectors above remain the grouping signal. Retrieval v2
    # is computed only after groups/prototypes are fixed, so changing retrieval
    # preprocessing cannot silently regroup the manual.
    for payload, vector in zip(
        visual_index_payloads,
        embed_images(
            [payload.get("image_path") for payload in visual_index_payloads],
            preprocessing=IMAGE_EMBEDDING_PROFILE_ALPHA_WHITE,
        ),
    ):
        if vector is not None:
            payload[IMAGE_EMBEDDING_V2_FIELD] = vector
    for payload, vector in zip(
        visual_index_payloads,
        embed_multimodal_chunks(visual_index_payloads),
    ):
        if vector is not None:
            payload[MULTIMODAL_EMBEDDING_FIELD] = vector

    index_chunks([*leaves, *visual_index_payloads])
    manual.status = "ready"
    db.commit()
    clear_icon_group_cache()

    text_parent_count = sum(payload.get("is_leaf") is False for payload in text_payloads)
    return {
        "manual_id": manual_id,
        "file_name": filename,
        "page_count": page_count,
        "text_chunk_count": len(leaves),
        "text_parent_count": text_parent_count,
        "visual_chunk_count": len(visual_payloads),
        "icon_group_count": len(visual_catalog["groups"]),
        "image_asset_count": len(visual_catalog["assets"]),
        "image_occurrence_count": len(visual_catalog["occurrences"]),
        "icon_prototype_count": len(visual_catalog["prototypes"]),
        "status": "ready",
    }


def get_manual_status(db: Session, manual_id: str = DEFAULT_MANUAL_ID) -> dict[str, Any]:
    ensure_tables(db)
    manual = db.query(Manual).filter(Manual.manual_id == manual_id).first()
    if not manual:
        return {
            "manual_id": None,
            "file_name": None,
            "page_count": 0,
            "text_chunk_count": 0,
            "text_parent_count": 0,
            "visual_chunk_count": 0,
            "icon_group_count": 0,
            "image_asset_count": 0,
            "image_occurrence_count": 0,
            "icon_prototype_count": 0,
            "status": "empty",
        }
    text_leaf_count = (
        db.query(ManualChunk)
        .filter(
            ManualChunk.manual_id == manual_id,
            ManualChunk.modality == "text",
            ManualChunk.is_leaf.is_(True),
        )
        .count()
    )
    text_parent_count = (
        db.query(ManualChunk)
        .filter(
            ManualChunk.manual_id == manual_id,
            ManualChunk.modality == "text",
            ManualChunk.is_leaf.is_(False),
        )
        .count()
    )
    visual_count = (
        db.query(ManualChunk)
        .filter(
            ManualChunk.manual_id == manual_id,
            ManualChunk.modality.in_(("icon", "operation_figure", "page_snapshot")),
        )
        .count()
    )
    icon_group_count = db.query(ManualIconGroup).filter(ManualIconGroup.manual_id == manual_id).count()
    image_asset_count = (
        db.query(ManualImageAsset)
        .filter(ManualImageAsset.manual_id == manual_id)
        .count()
    )
    image_occurrence_count = (
        db.query(ManualImageOccurrence)
        .filter(ManualImageOccurrence.manual_id == manual_id)
        .count()
    )
    icon_prototype_count = (
        db.query(ManualIconPrototype)
        .filter(ManualIconPrototype.manual_id == manual_id)
        .count()
    )
    return {
        "manual_id": manual.manual_id,
        "file_name": manual.file_name,
        "page_count": manual.page_count,
        "text_chunk_count": text_leaf_count,
        "text_parent_count": text_parent_count,
        "visual_chunk_count": visual_count,
        "icon_group_count": icon_group_count,
        "image_asset_count": image_asset_count,
        "image_occurrence_count": image_occurrence_count,
        "icon_prototype_count": icon_prototype_count,
        "status": manual.status,
    }


def _build_native_visual_catalog(
    *,
    manual_id: str,
    pdf_path: Path,
    pages: list[dict[str, Any]],
    rendered_pages: dict[int, dict[str, Any]],
    layout_by_page: dict[int, list[dict[str, Any]]],
    bookmark_nodes: list[dict[str, Any]],
    classification_cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    page_snapshots: list[dict[str, Any]] = []
    for page_number, rendered in sorted(rendered_pages.items()):
        page_section = _page_snapshot_section(bookmark_nodes, page_number)
        page_snapshots.append(
            _visual_payload(
                chunk_id=stable_chunk_id(manual_id, "page_snapshot", page_number),
                manual_id=manual_id,
                page=page_number,
                modality="page_snapshot",
                section_title=page_section.get("path") if page_section else None,
                object_name=None,
                content_text=None,
                bbox=None,
                image_path=rendered.get("image_path"),
                thumbnail_path=rendered.get("image_path"),
                risk_level=None,
            )
        )

    assets, occurrences = extract_native_image_catalog(
        pdf_path=pdf_path,
        output_dir=pdf_path.parent / "native_assets",
        manual_id=manual_id,
        pages=pages,
        bookmark_nodes=bookmark_nodes,
    )
    logger.info(
        "Native PDF image extraction complete: assets=%s occurrences=%s noise=%s",
        len(assets),
        len(occurrences),
        sum(bool(asset.get("noise_reason")) for asset in assets),
    )
    cache = {
        **_load_visual_classification_file_cache(manual_id),
        **(classification_cache or {}),
    }
    for asset in assets:
        cached = cache.get(str(asset.get("digest") or ""))
        if (
            not cached
            or asset.get("noise_reason")
            or not str(cached.get("classification_reason") or "").startswith(
                f"{CLASSIFICATION_SCHEMA_VERSION}:"
            )
        ):
            continue
        asset.update(cached)
        asset["_classification_cached"] = True
    classify_visual_assets(
        assets,
        occurrences,
        variant_dir=pdf_path.parent / "native_assets" / "variants",
    )
    _save_visual_classification_file_cache(manual_id, assets)
    logger.info(
        "Native visual classification complete: icon=%s operation_figure=%s ignore=%s",
        sum(asset.get("classification") == "icon" for asset in assets),
        sum(asset.get("classification") == "operation_figure" for asset in assets),
        sum(asset.get("classification") == "ignore" for asset in assets),
    )
    asset_by_id = {str(asset["asset_id"]): asset for asset in assets}

    searchable_assets = [
        asset
        for asset in assets
        if asset.get("classification") in {"icon", "operation_figure"}
        and asset.get("native_image_path")
    ]
    image_vectors = embed_images(
        [asset.get("native_image_path") for asset in searchable_assets]
    )
    image_vectors_by_asset: dict[str, list[float] | None] = {}
    for asset, vector in zip(searchable_assets, image_vectors):
        image_vectors_by_asset[str(asset["asset_id"])] = vector
    logger.info(
        "Unique native asset image embeddings complete: requested=%s succeeded=%s",
        len(searchable_assets),
        sum(vector is not None for vector in image_vectors),
    )

    groups = build_native_icon_groups(
        manual_id=manual_id,
        assets=assets,
        occurrences=occurrences,
        image_vectors=image_vectors_by_asset,
        variant_dir=pdf_path.parent / "native_assets" / "variants",
    )
    logger.info(
        "Native icon grouping complete: groups=%s occurrences=%s",
        len(groups),
        sum(int(group.get("member_count") or 0) for group in groups),
    )

    icon_chunks: list[dict[str, Any]] = []
    icon_index_payloads: list[dict[str, Any]] = []
    prototypes: list[dict[str, Any]] = []
    for group in groups:
        group_id = str(group["icon_group_id"])
        group_chunk_id = stable_chunk_id(manual_id, "icon_group_chunk", group_id)
        group["group_chunk_id"] = group_chunk_id
        representative = group.get("representative_occurrence") or {}
        primary_prototype = (group.get("prototypes") or [{}])[0]

        for occurrence in group.get("occurrences") or []:
            metadata = associate_icon_content(
                {
                    "page": occurrence["page_number"],
                    "bbox": occurrence["bbox"],
                    "label_candidates": occurrence.get("name_candidates") or [],
                    "nearby_texts": occurrence.get("nearby_text") or [],
                },
                bookmark_nodes,
                layout_by_page,
            )
            occurrence["section_title"] = metadata.get("section_title")
            occurrence["content_text"] = metadata.get("content_text")
            occurrence["name_candidates"] = metadata.get("name_candidates") or occurrence.get(
                "name_candidates"
            ) or []
            occurrence["icon_group_id"] = group_id
            occurrence["visual_chunk_id"] = group_chunk_id

        icon_payload = _visual_payload(
            chunk_id=group_chunk_id,
            manual_id=manual_id,
            page=int(representative.get("page_number") or 1),
            modality="icon",
            section_title=representative.get("section_title"),
            object_name=group.get("canonical_icon_name"),
            # A group can occur under several unrelated sections. Local
            # explanations stay on occurrence rows; the group chunk carries
            # only stable visual identity.
            content_text=None,
            bbox=representative.get("bbox"),
            image_path=primary_prototype.get("image_path"),
            thumbnail_path=primary_prototype.get("image_path"),
            risk_level=_risk_level(representative.get("content_text")),
            icon_group_id=group_id,
            _es_doc_id=primary_prototype.get("prototype_id"),
        )
        icon_chunks.append(icon_payload)

        for prototype in group.get("prototypes") or []:
            persisted = {
                **prototype,
                "manual_id": manual_id,
                "icon_group_id": group_id,
                "group_chunk_id": group_chunk_id,
            }
            prototypes.append(persisted)
            index_payload = {
                **icon_payload,
                "prototype_id": prototype["prototype_id"],
                "asset_id": prototype["asset_id"],
                "image_path": prototype["image_path"],
                "thumbnail_path": prototype["image_path"],
                "_es_doc_id": prototype["prototype_id"],
            }
            if prototype.get("image_vector") is not None:
                index_payload[IMAGE_EMBEDDING_FIELD] = prototype["image_vector"]
            icon_index_payloads.append(index_payload)

    operation_occurrences = [
        occurrence
        for occurrence in occurrences
        if asset_by_id[str(occurrence["asset_id"])].get("classification")
        == "operation_figure"
    ]
    operation_figures = [
        {
            "layout_id": occurrence["occurrence_id"],
            "page": occurrence["page_number"],
            "bbox": occurrence["bbox"],
        }
        for occurrence in operation_occurrences
    ]
    operation_chunks: list[dict[str, Any]] = []
    for occurrence, figure in zip(operation_occurrences, operation_figures):
        asset = asset_by_id[str(occurrence["asset_id"])]
        image_path = Path(str(asset["native_image_path"]))
        metadata = associate_operation_content(
            figure,
            operation_figures,
            bookmark_nodes,
            layout_by_page,
            image_path,
        )
        occurrence["section_title"] = metadata.get("section_title")
        occurrence["content_text"] = metadata.get("content_text")
        occurrence["name_candidates"] = metadata.get("name_candidates") or occurrence.get(
            "name_candidates"
        ) or []
        chunk_id = stable_chunk_id(
            manual_id,
            "operation_figure",
            occurrence["occurrence_id"],
        )
        occurrence["visual_chunk_id"] = chunk_id
        payload = _visual_payload(
            chunk_id=chunk_id,
            manual_id=manual_id,
            page=int(occurrence["page_number"]),
            modality="operation_figure",
            section_title=metadata.get("section_title"),
            object_name=_operation_object_name(asset, occurrence),
            content_text=metadata.get("content_text"),
            bbox=occurrence["bbox"],
            image_path=str(image_path),
            thumbnail_path=str(image_path),
            risk_level=_risk_level(metadata.get("content_text")),
            asset_id=asset["asset_id"],
            occurrence_id=occurrence["occurrence_id"],
        )
        vector = image_vectors_by_asset.get(str(asset["asset_id"]))
        if vector is not None:
            payload[IMAGE_EMBEDDING_FIELD] = vector
        operation_chunks.append(payload)

    snapshot_vectors = embed_images(
        [payload.get("image_path") for payload in page_snapshots]
    )
    for payload, vector in zip(page_snapshots, snapshot_vectors):
        if vector is not None:
            payload[IMAGE_EMBEDDING_FIELD] = vector

    visual_chunks = [*page_snapshots, *icon_chunks, *operation_chunks]
    visual_index_payloads = [
        *page_snapshots,
        *icon_index_payloads,
        *operation_chunks,
    ]
    return {
        "assets": assets,
        "occurrences": occurrences,
        "groups": groups,
        "prototypes": prototypes,
        "visual_chunks": visual_chunks,
        "visual_index_payloads": visual_index_payloads,
    }


def _visual_payload(**values: Any) -> dict[str, Any]:
    payload = dict(values)
    payload.setdefault("icon_group_id", None)
    payload.setdefault("parent_chunk_id", None)
    payload.setdefault("is_leaf", None)
    payload.setdefault("token_count", None)
    return payload


def _payload_to_row(payload: dict[str, Any]) -> ManualChunk:
    indexed = payload.get("modality") != "text" or payload.get("is_leaf") is True
    return ManualChunk(
        chunk_id=payload["chunk_id"],
        manual_id=payload["manual_id"],
        page_number=payload["page"],
        modality=payload["modality"],
        section_title=payload.get("section_title"),
        content_text=payload.get("content_text"),
        object_name=payload.get("object_name"),
        icon_group_id=payload.get("icon_group_id"),
        parent_chunk_id=payload.get("parent_chunk_id"),
        is_leaf=payload.get("is_leaf"),
        token_count=payload.get("token_count"),
        risk_level=payload.get("risk_level"),
        bbox=payload.get("bbox"),
        image_path=payload.get("image_path"),
        thumbnail_path=payload.get("thumbnail_path"),
        es_doc_id=(payload.get("_es_doc_id") or payload["chunk_id"]) if indexed else None,
    )


def _delete_active_manual(db: Session, manual_id: str) -> None:
    db.execute(
        delete(ManualIconPrototype).where(ManualIconPrototype.manual_id == manual_id)
    )
    db.execute(
        delete(ManualImageOccurrence).where(
            ManualImageOccurrence.manual_id == manual_id
        )
    )
    db.execute(delete(ManualChunk).where(ManualChunk.manual_id == manual_id))
    db.execute(delete(ManualIconGroup).where(ManualIconGroup.manual_id == manual_id))
    db.execute(
        delete(ManualImageAsset).where(ManualImageAsset.manual_id == manual_id)
    )
    db.execute(delete(ManualPage).where(ManualPage.manual_id == manual_id))
    db.execute(delete(Manual).where(Manual.manual_id == manual_id))
    db.commit()


def _load_visual_classification_cache(
    db: Session,
    manual_id: str,
) -> dict[str, dict[str, Any]]:
    """Reuse VLM type decisions when the same PDF assets are rebuilt."""

    rows = (
        db.query(ManualImageAsset)
        .filter(
            ManualImageAsset.manual_id == manual_id,
            ManualImageAsset.digest.isnot(None),
        )
        .all()
    )
    operation_names_by_path = {
        str(row.image_path): row.object_name
        for row in db.query(ManualChunk)
        .filter(
            ManualChunk.manual_id == manual_id,
            ManualChunk.modality == "operation_figure",
            ManualChunk.image_path.isnot(None),
            ManualChunk.object_name.isnot(None),
        )
        .all()
    }
    return {
        str(row.digest): {
            "classification": row.classification,
            "classification_confidence": row.classification_confidence,
            "classification_reason": row.classification_reason,
            "operation_object_name": operation_names_by_path.get(
                str(row.native_image_path)
            ),
        }
        for row in rows
        if row.classification in {"icon", "operation_figure", "ignore"}
    }


def _visual_classification_cache_path(manual_id: str) -> Path:
    safe_manual_id = "".join(
        character
        for character in str(manual_id)
        if character.isalnum() or character in {"-", "_"}
    ) or "manual"
    return storage_root() / "ingestion_cache" / f"{safe_manual_id}_visual_classification.json"


def _load_visual_classification_file_cache(
    manual_id: str,
) -> dict[str, dict[str, Any]]:
    path = _visual_classification_cache_path(manual_id)
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {
        str(digest): value
        for digest, value in parsed.items()
        if digest and isinstance(value, dict)
    }


def _save_visual_classification_file_cache(
    manual_id: str,
    assets: list[dict[str, Any]],
) -> None:
    payload = {
        str(asset["digest"]): {
            "classification": asset.get("classification"),
            "classification_confidence": asset.get("classification_confidence"),
            "classification_reason": asset.get("classification_reason"),
            "operation_object_name": asset.get("operation_object_name"),
        }
        for asset in assets
        if asset.get("digest")
        and str(asset.get("classification_reason") or "").startswith(
            f"{CLASSIFICATION_SCHEMA_VERSION}:"
        )
        and asset.get("classification") in {"icon", "operation_figure", "ignore"}
    }
    path = _visual_classification_cache_path(manual_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary.replace(path)


def _operation_object_name(
    asset: dict[str, Any],
    occurrence: dict[str, Any],
) -> str | None:
    name = " ".join(str(asset.get("operation_object_name") or "").split()).strip()
    if name:
        return name[:255]
    # Rebuilds can reuse the prior VLM type without repeating classification.
    # The most local PDF bookmark is still a literal manual label and is a
    # safer fallback than generating a new figure name.
    path = str(occurrence.get("section_title") or "")
    local_title = path.split(" > ")[-1].strip()
    return local_title[:255] if 2 <= len(local_title) <= 80 else None


def _asset_to_row(asset: dict[str, Any]) -> ManualImageAsset:
    return ManualImageAsset(
        asset_id=asset["asset_id"],
        manual_id=asset["manual_id"],
        source_xrefs=asset.get("source_xrefs") or [],
        source_smask_xrefs=asset.get("source_smask_xrefs") or [],
        digest=asset.get("digest"),
        pixel_width=asset.get("pixel_width"),
        pixel_height=asset.get("pixel_height"),
        native_image_path=asset.get("native_image_path"),
        classification=asset.get("classification") or "ignore",
        classification_confidence=asset.get("classification_confidence"),
        classification_reason=asset.get("classification_reason"),
        noise_reason=asset.get("noise_reason"),
        phash=asset.get("phash"),
        dominant_colour=asset.get("dominant_colour"),
    )


def _occurrence_to_row(
    occurrence: dict[str, Any],
) -> ManualImageOccurrence:
    return ManualImageOccurrence(
        occurrence_id=occurrence["occurrence_id"],
        manual_id=occurrence["manual_id"],
        asset_id=occurrence["asset_id"],
        page_number=occurrence["page_number"],
        bbox=occurrence["bbox"],
        transform=occurrence.get("transform"),
        orientation=occurrence.get("orientation") or "r0",
        section_title=occurrence.get("section_title"),
        nearby_text=occurrence.get("nearby_text") or [],
        name_candidates=occurrence.get("name_candidates") or [],
        content_text=occurrence.get("content_text"),
        icon_group_id=occurrence.get("icon_group_id"),
        visual_chunk_id=occurrence.get("visual_chunk_id"),
    )


def _prototype_to_row(
    manual_id: str,
    prototype: dict[str, Any],
) -> ManualIconPrototype:
    return ManualIconPrototype(
        prototype_id=prototype["prototype_id"],
        manual_id=manual_id,
        icon_group_id=prototype["icon_group_id"],
        asset_id=prototype["asset_id"],
        group_chunk_id=prototype["group_chunk_id"],
        prototype_rank=prototype["prototype_rank"],
        image_path=prototype["image_path"],
        phash=prototype.get("phash"),
        dominant_colour=prototype.get("dominant_colour"),
        quality_score=prototype.get("quality_score"),
    )


def _extract_pdf_pages(pdf_path: Path) -> tuple[int, list[dict[str, Any]]]:
    pages: list[dict[str, Any]] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for index, page in enumerate(pdf.pages, start=1):
            pages.append(
                {
                    "page_number": index,
                    "text": page.extract_text() or "",
                    "words": page.extract_words(x_tolerance=2, y_tolerance=3) or [],
                    "width": float(page.width or 0),
                    "height": float(page.height or 0),
                }
            )
        return len(pdf.pages), pages


def _risk_level(text: str | None) -> str | None:
    value = str(text or "")
    if any(term in value for term in ("危险", "严禁", "请勿")):
        return "high"
    if any(term in value for term in ("警告", "注意", "务必", "故障")):
        return "medium"
    return None


def _page_snapshot_section(
    bookmark_nodes: list[dict[str, Any]],
    page_number: int,
) -> dict[str, Any] | None:
    starts_here = [
        node for node in bookmark_nodes if int(node.get("page") or 0) == page_number
    ]
    if starts_here:
        return min(
            starts_here,
            key=lambda node: (float(node.get("y") or 0.0), -int(node.get("level") or 0)),
        )
    return section_at_position(bookmark_nodes, page_number, 0.0)
