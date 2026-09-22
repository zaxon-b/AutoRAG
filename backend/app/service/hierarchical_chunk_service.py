"""Build searchable text leaves and PostgreSQL-only parent chunks."""

from __future__ import annotations

import hashlib
import os
import re
from functools import lru_cache
from typing import Any

from service.bookmark_outline_service import section_at_position


LEAF_TOKEN_LIMIT = int(os.getenv("MANUAL_TEXT_LEAF_TOKEN_LIMIT", "500"))


def build_hierarchical_text_chunks(
    manual_id: str,
    bookmark_nodes: list[dict[str, Any]],
    blocks_by_page: dict[int, list[dict[str, Any]]],
    *,
    leaf_token_limit: int = LEAF_TOKEN_LIMIT,
) -> list[dict[str, Any]]:
    """Create the complete text tree; only returned rows with is_leaf=True enter ES."""

    if not bookmark_nodes:
        return _build_page_fallback(manual_id, blocks_by_page, leaf_token_limit)

    node_by_id = {str(node["node_id"]): node for node in bookmark_nodes}
    direct_blocks: dict[str, list[tuple[int, str]]] = {node_id: [] for node_id in node_by_id}
    full_blocks: dict[str, list[tuple[int, str]]] = {node_id: [] for node_id in node_by_id}
    index_pages = {
        page
        for page, blocks in blocks_by_page.items()
        if _looks_like_index_page(page, blocks, bookmark_nodes)
    }

    for page, blocks in sorted(blocks_by_page.items()):
        if page in index_pages:
            continue
        for block in blocks:
            if block.get("kind") != "text":
                continue
            if _is_page_decoration(block):
                continue
            node = section_at_position(bookmark_nodes, page, float(block["bbox"][1]))
            if not node:
                continue
            text = _clean_body_text(str(block.get("text") or ""))
            text = _remove_bookmark_label(text, block, node)
            if not text:
                continue
            node_id = str(node["node_id"])
            direct_blocks[node_id].append((page, text))
            current: dict[str, Any] | None = node
            while current:
                full_blocks[str(current["node_id"])].append((page, text))
                parent_id = current.get("parent_id")
                current = node_by_id.get(str(parent_id)) if parent_id else None

    parent_chunk_ids: dict[str, str] = {}
    for node in bookmark_nodes:
        node_id = str(node["node_id"])
        full_text = _join_blocks(full_blocks[node_id])
        must_be_parent = bool(node.get("children_ids")) or count_tokens(full_text) > leaf_token_limit
        if must_be_parent and full_text:
            parent_chunk_ids[node_id] = stable_chunk_id(manual_id, "text", node_id, "parent")

    chunks: list[dict[str, Any]] = []
    for node in bookmark_nodes:
        node_id = str(node["node_id"])
        path = node.get("path")
        full_text = _join_blocks(full_blocks[node_id])
        direct_text = _join_blocks(direct_blocks[node_id])
        parent_chunk_id = parent_chunk_ids.get(node_id)
        ancestor_parent_id = _nearest_parent_chunk_id(node, node_by_id, parent_chunk_ids)

        if parent_chunk_id and full_text:
            chunks.append(
                _text_payload(
                    chunk_id=parent_chunk_id,
                    manual_id=manual_id,
                    page=int(node["page"]),
                    section_title=path,
                    content_text=full_text,
                    parent_chunk_id=ancestor_parent_id,
                    is_leaf=False,
                )
            )

        if node.get("children_ids"):
            # Text between a parent heading and its first child must remain directly searchable.
            if direct_text:
                chunks.extend(
                    _leaf_payloads(
                        manual_id,
                        node_id,
                        direct_blocks[node_id],
                        path,
                        parent_chunk_id,
                        leaf_token_limit,
                        leaf_role="intro",
                    )
                )
            continue

        if not full_text:
            continue
        leaf_parent_id = parent_chunk_id or ancestor_parent_id
        chunks.extend(
            _leaf_payloads(
                manual_id,
                node_id,
                full_blocks[node_id],
                path,
                leaf_parent_id,
                leaf_token_limit,
                leaf_role="leaf",
            )
        )
    return chunks


def stable_chunk_id(manual_id: str, *parts: object) -> str:
    """Generate a reproducible ID from document structure instead of a random UUID."""

    raw = "|".join([manual_id, *(str(part) for part in parts)])
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    safe_manual_id = re.sub(r"[^A-Za-z0-9_-]+", "-", manual_id).strip("-")[:48] or "manual"
    return f"{safe_manual_id}_{digest}"


@lru_cache(maxsize=1)
def _token_encoder():
    import tiktoken

    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str | None) -> int:
    value = str(text or "")
    if not value:
        return 0
    try:
        return len(_token_encoder().encode(value))
    except Exception:
        # Conservative fallback for Chinese-heavy manuals.
        return len(value)


def split_text_by_tokens(text: str, max_tokens: int) -> list[str]:
    """Split on paragraph/sentence boundaries, using hard character cuts only last."""

    clean = _clean_body_text(text)
    if not clean:
        return []
    if count_tokens(clean) <= max_tokens:
        return [clean]

    units: list[str] = []
    for paragraph in re.split(r"\n+", clean):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if count_tokens(paragraph) <= max_tokens:
            units.append(paragraph)
            continue
        sentences = [item.strip() for item in re.split(r"(?<=[。！？!?；;\.])", paragraph) if item.strip()]
        for sentence in sentences or [paragraph]:
            if count_tokens(sentence) <= max_tokens:
                units.append(sentence)
            else:
                units.extend(_hard_split(sentence, max_tokens))

    chunks: list[str] = []
    current: list[str] = []
    for unit in units:
        candidate = "\n".join([*current, unit])
        if current and count_tokens(candidate) > max_tokens:
            chunks.append("\n".join(current))
            current = [unit]
        else:
            current.append(unit)
    if current:
        chunks.append("\n".join(current))
    return chunks


def _build_page_fallback(
    manual_id: str,
    blocks_by_page: dict[int, list[dict[str, Any]]],
    leaf_token_limit: int,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for page, blocks in sorted(blocks_by_page.items()):
        if _looks_like_index_page(page, blocks, []):
            continue
        body_blocks = [
            (page, _clean_body_text(str(block.get("text") or "")))
            for block in blocks
            if block.get("kind") == "text" and _clean_body_text(str(block.get("text") or ""))
        ]
        full_text = _join_blocks(body_blocks)
        if not full_text:
            continue
        parent_id = stable_chunk_id(manual_id, "text", "page", page, "parent")
        chunks.append(
            _text_payload(
                chunk_id=parent_id,
                manual_id=manual_id,
                page=page,
                section_title=None,
                content_text=full_text,
                parent_chunk_id=None,
                is_leaf=False,
            )
        )
        chunks.extend(
            _leaf_payloads(
                manual_id,
                f"page_{page:04d}",
                body_blocks,
                None,
                parent_id,
                leaf_token_limit,
                leaf_role="fallback",
            )
        )
    return chunks


def _leaf_payloads(
    manual_id: str,
    node_id: str,
    blocks: list[tuple[int, str]],
    section_title: str | None,
    parent_chunk_id: str | None,
    leaf_token_limit: int,
    *,
    leaf_role: str,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for index, (page, text) in enumerate(
        _split_blocks_by_tokens(blocks, leaf_token_limit),
        start=1,
    ):
        payloads.append(
            _text_payload(
                chunk_id=stable_chunk_id(manual_id, "text", node_id, leaf_role, index),
                manual_id=manual_id,
                page=page,
                section_title=section_title,
                content_text=text,
                parent_chunk_id=parent_chunk_id,
                is_leaf=True,
            )
        )
    return payloads


def _text_payload(
    *,
    chunk_id: str,
    manual_id: str,
    page: int,
    section_title: str | None,
    content_text: str,
    parent_chunk_id: str | None,
    is_leaf: bool,
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "manual_id": manual_id,
        "page": page,
        "modality": "text",
        "section_title": section_title,
        "content_text": content_text,
        "object_name": None,
        "icon_group_id": None,
        "parent_chunk_id": parent_chunk_id,
        "is_leaf": is_leaf,
        "token_count": count_tokens(content_text),
        "risk_level": None,
        "bbox": None,
        "image_path": None,
        "thumbnail_path": None,
    }


def _nearest_parent_chunk_id(
    node: dict[str, Any],
    node_by_id: dict[str, dict[str, Any]],
    parent_chunk_ids: dict[str, str],
) -> str | None:
    parent_id = node.get("parent_id")
    while parent_id:
        parent_key = str(parent_id)
        if parent_key in parent_chunk_ids:
            return parent_chunk_ids[parent_key]
        parent = node_by_id.get(parent_key)
        parent_id = parent.get("parent_id") if parent else None
    return None


def _remove_bookmark_label(
    text: str,
    block: dict[str, Any],
    node: dict[str, Any],
) -> str:
    if int(block.get("page") or 0) != int(node["page"]):
        return text
    if abs(float(block["bbox"][1]) - float(node["y"])) > 40.0:
        return text
    block_text = _normalize_heading(text)
    title = _normalize_heading(str(node.get("title") or ""))
    if not title or not block_text.startswith(title):
        return text
    raw_title = " ".join(str(node.get("title") or "").split())
    compact_text = " ".join(text.split())
    if compact_text.startswith(raw_title):
        return compact_text[len(raw_title) :].lstrip(" ：:，,。.-")
    # Formatting spaces can differ between the bookmark and PDF text.  Remove
    # only the first line when it is the matched label, never the entire block.
    lines = text.splitlines()
    if lines and _normalize_heading(lines[0]).startswith(title):
        return "\n".join(lines[1:]).strip()
    return text


def _normalize_heading(value: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value.lower())


def _is_page_decoration(block: dict[str, Any]) -> bool:
    x0, y0, x1, y1 = [float(value) for value in block.get("bbox") or [0, 0, 0, 0]]
    page_width = float(block.get("page_width") or 0)
    page_height = float(block.get("page_height") or 0)
    if y1 <= 55.0 or (page_height and y0 >= page_height * 0.95):
        return True
    if page_width and (x0 >= page_width * 0.92 or x1 <= page_width * 0.08):
        return True
    return False


def _looks_like_index_page(
    page: int,
    blocks: list[dict[str, Any]],
    bookmark_nodes: list[dict[str, Any]],
) -> bool:
    """Reject document/chapter contents pages without hard-coding page numbers."""

    text_blocks = [
        block
        for block in blocks
        if block.get("kind") == "text" and not _is_page_decoration(block)
    ]
    active_index_node = any(
        int(node.get("page") or 0) <= page <= int(node.get("end_page") or page)
        and _normalize_heading(str(node.get("title") or "")) in {"目录", "contents", "tableofcontents"}
        for node in bookmark_nodes
    )
    if active_index_node:
        return True
    for block in text_blocks:
        section = section_at_position(bookmark_nodes, page, float(block.get("bbox", [0, 0])[1])) if bookmark_nodes else None
        title = _normalize_heading(str((section or {}).get("title") or ""))
        if title in {"目录", "contents", "tableofcontents"}:
            return True

    lines = [
        line.strip()
        for block in text_blocks
        for line in str(block.get("text") or "").splitlines()
        if line.strip()
    ]
    leader_lines = sum(bool(re.search(r"\.{3,}\s*\d+\s*$", line)) for line in lines)
    return leader_lines >= 3 and leader_lines / max(1, len(lines)) >= 0.25


def _hard_split(text: str, max_tokens: int) -> list[str]:
    chunks: list[str] = []
    remaining = text.strip()
    while remaining:
        if count_tokens(remaining) <= max_tokens:
            chunks.append(remaining)
            break
        low, high = 1, len(remaining)
        while low < high:
            middle = (low + high + 1) // 2
            if count_tokens(remaining[:middle]) <= max_tokens:
                low = middle
            else:
                high = middle - 1
        cut = max(1, low)
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    return [item for item in chunks if item]


def _split_blocks_by_tokens(
    blocks: list[tuple[int, str]],
    max_tokens: int,
) -> list[tuple[int, str]]:
    """Split while preserving the first source page of each resulting leaf."""

    units: list[tuple[int, str]] = []
    for page, text in blocks:
        units.extend((page, part) for part in split_text_by_tokens(text, max_tokens))

    result: list[tuple[int, str]] = []
    current_page: int | None = None
    current: list[str] = []
    for page, unit in units:
        candidate = "\n".join([*current, unit])
        if current and count_tokens(candidate) > max_tokens:
            result.append((int(current_page or page), "\n".join(current)))
            current_page = page
            current = [unit]
        else:
            if current_page is None:
                current_page = page
            current.append(unit)
    if current:
        result.append((int(current_page or blocks[0][0]), "\n".join(current)))
    return result


def _join_blocks(blocks: list[tuple[int, str]]) -> str:
    return "\n".join(text for _, text in blocks if text).strip()


def _clean_body_text(value: str) -> str:
    lines = [" ".join(line.split()) for line in value.replace("\u00a0", " ").splitlines()]
    return "\n".join(line for line in lines if line).strip()
