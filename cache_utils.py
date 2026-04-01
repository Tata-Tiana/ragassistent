"""Простые функции для файлового кэша проекта."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
METADATA_PATH = CACHE_DIR / "metadata.json"
QUERY_CACHE_PATH = CACHE_DIR / "query_cache.json"


def calculate_text_hash(text: str) -> str:
    """Возвращает SHA-256 хэш для текста."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """Читает JSON-файл. Если файла нет, возвращает пустой словарь."""
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data: dict[str, Any]) -> None:
    """Сохраняет словарь в JSON-файл."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def load_metadata() -> dict[str, Any]:
    """Загружает метаданные индексации."""
    return load_json(METADATA_PATH)


def save_metadata(file_hash: str, chunks_count: int) -> None:
    """Сохраняет метаданные после успешной индексации."""
    payload = {
        "file_hash": file_hash,
        "indexed_at": datetime.now().isoformat(timespec="seconds"),
        "chunks_count": chunks_count,
    }
    save_json(METADATA_PATH, payload)


def load_query_cache() -> dict[str, Any]:
    """Загружает кэш вопросов."""
    return load_json(QUERY_CACHE_PATH)


def save_query_cache(data: dict[str, Any]) -> None:
    """Сохраняет кэш вопросов."""
    save_json(QUERY_CACHE_PATH, data)


def clear_cache_files() -> list[str]:
    """Удаляет файлы кэша и возвращает список удалённых путей."""
    removed_files: list[str] = []

    for path in [METADATA_PATH, QUERY_CACHE_PATH]:
        if path.exists():
            path.unlink()
            removed_files.append(str(path.relative_to(BASE_DIR)))

    return removed_files
