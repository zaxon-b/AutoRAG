"""Elasticsearch storage for text leaves and the three visual chunk types."""

from __future__ import annotations

import os
from typing import Any

from elasticsearch import Elasticsearch

from service.embedding_service import EMBEDDING_DIM
from service.image_embedding_service import (
    IMAGE_EMBEDDING_DIM,
    IMAGE_EMBEDDING_FIELD,
    IMAGE_EMBEDDING_V2_FIELD,
)
from service.multimodal_embedding_service import MULTIMODAL_EMBEDDING_DIM, MULTIMODAL_EMBEDDING_FIELD


INDEX_NAME = os.getenv("MANUAL_INDEX_NAME", "manual_chunks_demo")
INDEX_SCHEMA_VERSION = "rag_chunk_v3_native_visual_assets"
DEFAULT_TEXT_SEARCH_FIELDS = ["section_title^4", "content_text^3"]
DEFAULT_IMAGE_SEARCH_MODALITIES = ["icon", "operation_figure", "page_snapshot"]
_VECTOR_FIELDS = {
    "text_vector",
    IMAGE_EMBEDDING_FIELD,
    IMAGE_EMBEDDING_V2_FIELD,
    MULTIMODAL_EMBEDDING_FIELD,
}
_DOCUMENT_FIELDS = {
    "manual_id",
    "chunk_id",
    "modality",
    "page",
    "section_title",
    "content_text",
    "object_name",
    "parent_chunk_id",
    "is_leaf",
    "token_count",
    "icon_group_id",
    "prototype_id",
    "asset_id",
    "risk_level",
    "bbox",
    "image_path",
    "thumbnail_path",
    *_VECTOR_FIELDS,
}


def _es_client() -> Elasticsearch:
    url = os.getenv("ELASTICSEARCH_URL") or os.getenv("ES_HOST", "http://localhost:9200")
    password = os.getenv("ELASTIC_PASSWORD")
    kwargs = {"verify_certs": False, "request_timeout": 60}
    if password:
        kwargs["basic_auth"] = ("elastic", password)
    return Elasticsearch([url], **kwargs)


def ping_es() -> bool:
    try:
        return bool(_es_client().ping())
    except Exception:
        return False


def ensure_index() -> None:
    es = _es_client()
    if es.indices.exists(index=INDEX_NAME):
        mapping = es.indices.get_mapping(index=INDEX_NAME)
        metadata = mapping.get(INDEX_NAME, {}).get("mappings", {}).get("_meta", {})
        if metadata.get("schema_version") != INDEX_SCHEMA_VERSION:
            raise RuntimeError(
                f"Elasticsearch index {INDEX_NAME!r} uses the old chunk schema; "
                "delete it once before rebuilding the manual."
            )
        return
    es.indices.create(index=INDEX_NAME, body=_index_mapping())


def prepare_index_for_ingestion() -> None:
    """Create the current index, replacing only an incompatible old schema.

    Uploading a PDF is already a full rebuild operation. This keeps the schema
    reset inside that operation instead of requiring a separate migration
    command from the user.
    """

    es = _es_client()
    if es.indices.exists(index=INDEX_NAME):
        mapping = es.indices.get_mapping(index=INDEX_NAME)
        metadata = mapping.get(INDEX_NAME, {}).get("mappings", {}).get("_meta", {})
        if metadata.get("schema_version") != INDEX_SCHEMA_VERSION:
            es.indices.delete(index=INDEX_NAME)
    if not es.indices.exists(index=INDEX_NAME):
        es.indices.create(index=INDEX_NAME, body=_index_mapping())


def delete_index() -> None:
    es = _es_client()
    if es.indices.exists(index=INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)


def clear_manual(manual_id: str) -> None:
    ensure_index()
    _es_client().delete_by_query(
        index=INDEX_NAME,
        body={"query": {"term": {"manual_id": manual_id}}},
        conflicts="proceed",
        refresh=True,
    )


def index_chunks(chunks: list[dict[str, Any]]) -> None:
    documents = [_sanitize_document(chunk) for chunk in chunks if _should_index(chunk)]
    if not documents:
        return
    ensure_index()
    operations: list[dict[str, Any]] = []
    source_chunks = [chunk for chunk in chunks if _should_index(chunk)]
    for chunk, document in zip(source_chunks, documents):
        es_doc_id = (
            chunk.get("_es_doc_id")
            or chunk.get("prototype_id")
            or document["chunk_id"]
        )
        operations.append({"index": {"_index": INDEX_NAME, "_id": es_doc_id}})
        operations.append(document)
    _es_client().bulk(operations=operations, refresh=True)


def search_keyword_chunks(
    manual_id: str,
    query: str,
    top_k: int = 8,
    modality_filter: list[str] | None = None,
    chunk_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    del modality_filter  # Text retrieval is intentionally isolated to leaves.
    ensure_index()
    body = {
        "query": {
            "bool": {
                "filter": _text_leaf_filter(manual_id, chunk_ids),
                "must": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": DEFAULT_TEXT_SEARCH_FIELDS,
                            "type": "best_fields",
                            "operator": "or",
                        }
                    }
                ],
            }
        },
        "_source": {"excludes": sorted(_VECTOR_FIELDS)},
        "size": top_k,
    }
    return _hits_to_sources(_es_client().search(index=INDEX_NAME, body=body), "bm25")


def search_text_vector_chunks(
    manual_id: str,
    query_vector: list[float],
    top_k: int = 8,
    modality_filter: list[str] | None = None,
    chunk_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    del modality_filter
    ensure_index()
    body = {
        "query": {
            "script_score": {
                "query": {
                    "bool": {
                        "filter": [
                            *_text_leaf_filter(manual_id, chunk_ids),
                            {"exists": {"field": "text_vector"}},
                        ]
                    }
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'text_vector') + 1.0",
                    "params": {"query_vector": query_vector},
                },
            }
        },
        "_source": {"excludes": sorted(_VECTOR_FIELDS)},
        "size": top_k,
    }
    return _hits_to_sources(_es_client().search(index=INDEX_NAME, body=body), "text_embedding")


def search_chunks(
    manual_id: str,
    query: str,
    query_vector: list[float] | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    results = search_keyword_chunks(manual_id, query, top_k=top_k)
    if query_vector:
        results.extend(search_text_vector_chunks(manual_id, query_vector, top_k=top_k))
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for item in sorted(results, key=lambda value: float(value.get("score") or 0), reverse=True):
        chunk_id = str(item.get("chunk_id") or "")
        if chunk_id and chunk_id not in seen:
            seen.add(chunk_id)
            merged.append(item)
    return merged[:top_k]


def search_multimodal_chunks(
    manual_id: str,
    query_vector: list[float],
    top_k: int = 8,
    modality_filter: list[str] | None = None,
) -> list[dict[str, Any]]:
    modalities = _visual_modalities(modality_filter)
    return _search_vector_field(
        manual_id,
        query_vector,
        MULTIMODAL_EMBEDDING_FIELD,
        modalities,
        top_k,
        "multimodal_embedding",
    )


def search_image_chunks(
    manual_id: str,
    image_vector: list[float],
    top_k: int = 5,
    modalities: list[str] | None = None,
    vector_field: str = IMAGE_EMBEDDING_FIELD,
) -> list[dict[str, Any]]:
    if vector_field not in {IMAGE_EMBEDDING_FIELD, IMAGE_EMBEDDING_V2_FIELD}:
        raise ValueError(f"Unsupported image vector field: {vector_field}")
    if vector_field == IMAGE_EMBEDDING_V2_FIELD:
        ensure_image_embedding_v2_mapping()
    return _search_vector_field(
        manual_id,
        image_vector,
        vector_field,
        _visual_modalities(modalities),
        top_k,
        "image_embedding_v2" if vector_field == IMAGE_EMBEDDING_V2_FIELD else "image_embedding",
    )


def _search_vector_field(
    manual_id: str,
    query_vector: list[float],
    field: str,
    modalities: list[str],
    top_k: int,
    method: str,
) -> list[dict[str, Any]]:
    ensure_index()
    body = {
        "query": {
            "script_score": {
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"manual_id": manual_id}},
                            {"terms": {"modality": modalities}},
                            {"exists": {"field": field}},
                        ]
                    }
                },
                "script": {
                    "source": f"cosineSimilarity(params.query_vector, '{field}') + 1.0",
                    "params": {"query_vector": query_vector},
                },
            }
        },
        "_source": {"excludes": sorted(_VECTOR_FIELDS)},
        "size": top_k,
    }
    return _hits_to_sources(_es_client().search(index=INDEX_NAME, body=body), method)


def _index_mapping() -> dict[str, Any]:
    return {
        "mappings": {
            "dynamic": "strict",
            "_meta": {"schema_version": INDEX_SCHEMA_VERSION},
            "properties": {
                "manual_id": {"type": "keyword"},
                "chunk_id": {"type": "keyword"},
                "modality": {"type": "keyword"},
                "page": {"type": "integer"},
                "section_title": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 1024}}},
                "content_text": {"type": "text", "analyzer": "standard"},
                "object_name": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 255}}},
                "parent_chunk_id": {"type": "keyword"},
                "is_leaf": {"type": "boolean"},
                "token_count": {"type": "integer"},
                "icon_group_id": {"type": "keyword"},
                "prototype_id": {"type": "keyword"},
                "asset_id": {"type": "keyword"},
                "risk_level": {"type": "keyword"},
                "bbox": {"type": "float"},
                "image_path": {"type": "keyword", "ignore_above": 1024},
                "thumbnail_path": {"type": "keyword", "ignore_above": 1024},
                "text_vector": {"type": "dense_vector", "dims": EMBEDDING_DIM, "index": True, "similarity": "cosine"},
                IMAGE_EMBEDDING_FIELD: {
                    "type": "dense_vector",
                    "dims": IMAGE_EMBEDDING_DIM,
                    "index": True,
                    "similarity": "cosine",
                },
                IMAGE_EMBEDDING_V2_FIELD: {
                    "type": "dense_vector",
                    "dims": IMAGE_EMBEDDING_DIM,
                    "index": True,
                    "similarity": "cosine",
                },
                MULTIMODAL_EMBEDDING_FIELD: {
                    "type": "dense_vector",
                    "dims": MULTIMODAL_EMBEDDING_DIM,
                    "index": True,
                    "similarity": "cosine",
                },
            },
        }
    }


def ensure_image_embedding_v2_mapping() -> None:
    """Add the experimental vector field without recreating the ES index."""

    ensure_index()
    es = _es_client()
    mapping = es.indices.get_mapping(index=INDEX_NAME)
    properties = mapping.get(INDEX_NAME, {}).get("mappings", {}).get("properties", {})
    if IMAGE_EMBEDDING_V2_FIELD in properties:
        return
    es.indices.put_mapping(
        index=INDEX_NAME,
        properties={
            IMAGE_EMBEDDING_V2_FIELD: {
                "type": "dense_vector",
                "dims": IMAGE_EMBEDDING_DIM,
                "index": True,
                "similarity": "cosine",
            }
        },
    )


def visual_documents_for_image_reembedding(manual_id: str) -> list[dict[str, Any]]:
    """Read current searchable visual documents without changing any chunk."""

    ensure_index()
    response = _es_client().search(
        index=INDEX_NAME,
        body={
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"manual_id": manual_id}},
                        {"terms": {"modality": DEFAULT_IMAGE_SEARCH_MODALITIES}},
                        {"exists": {"field": "image_path"}},
                    ]
                }
            },
            "_source": ["chunk_id", "icon_group_id", "prototype_id", "modality", "image_path"],
            "size": 10_000,
        },
    )
    return [
        {"_id": hit.get("_id"), **dict(hit.get("_source") or {})}
        for hit in response.get("hits", {}).get("hits", [])
        if hit.get("_id")
    ]


def update_image_embedding_v2(updates: list[tuple[str, list[float]]]) -> None:
    """Attach v2 vectors to existing ES documents; no duplicate docs are made."""

    if not updates:
        return
    ensure_image_embedding_v2_mapping()
    operations: list[dict[str, Any]] = []
    for document_id, vector in updates:
        operations.append({"update": {"_index": INDEX_NAME, "_id": document_id}})
        operations.append({"doc": {IMAGE_EMBEDDING_V2_FIELD: vector}})
    _es_client().bulk(operations=operations, refresh=True)


def _should_index(chunk: dict[str, Any]) -> bool:
    modality = chunk.get("modality")
    if modality == "text":
        return chunk.get("is_leaf") is True
    if modality == "icon":
        return bool(chunk.get("prototype_id"))
    return modality in DEFAULT_IMAGE_SEARCH_MODALITIES


def _sanitize_document(chunk: dict[str, Any]) -> dict[str, Any]:
    document = {key: value for key, value in chunk.items() if key in _DOCUMENT_FIELDS and value is not None}
    if document.get("modality") == "text":
        for field in (
            IMAGE_EMBEDDING_FIELD,
            IMAGE_EMBEDDING_V2_FIELD,
            MULTIMODAL_EMBEDDING_FIELD,
        ):
            document.pop(field, None)
    else:
        document.pop("text_vector", None)
    return document


def _text_leaf_filter(manual_id: str, chunk_ids: list[str] | None = None) -> list[dict[str, Any]]:
    filters: list[dict[str, Any]] = [
        {"term": {"manual_id": manual_id}},
        {"term": {"modality": "text"}},
        {"term": {"is_leaf": True}},
    ]
    if chunk_ids:
        filters.append({"terms": {"chunk_id": chunk_ids}})
    return filters


def _visual_modalities(values: list[str] | None) -> list[str]:
    requested = values or DEFAULT_IMAGE_SEARCH_MODALITIES
    allowed = [value for value in requested if value in DEFAULT_IMAGE_SEARCH_MODALITIES]
    return allowed or list(DEFAULT_IMAGE_SEARCH_MODALITIES)


def _hits_to_sources(response: dict[str, Any], match_method: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for hit in response.get("hits", {}).get("hits", []):
        source = dict(hit.get("_source") or {})
        source["score"] = hit.get("_score")
        source["match_method"] = match_method
        result.append(source)
    return result
