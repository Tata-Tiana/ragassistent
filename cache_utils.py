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
DIALOG_HISTORIES_PATH = CACHE_DIR / "dialog_histories.json"


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


def save_metadata(file_hash: str, chunks_count: int, index_version: int = 1) -> None:
    """Сохраняет метаданные после успешной индексации."""
    payload = {
        "file_hash": file_hash,
        "index_version": index_version,
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


def load_dialog_histories() -> dict[str, Any]:
    """Загружает истории диалогов."""
    return load_json(DIALOG_HISTORIES_PATH)


def save_dialog_histories(data: dict[str, Any]) -> None:
    """Сохраняет истории диалогов."""
    save_json(DIALOG_HISTORIES_PATH, data)


def load_dialog_history(dialog_id: str) -> list[dict[str, str]]:
    """Возвращает историю конкретного диалога."""
    histories = load_dialog_histories()
    history = histories.get(dialog_id, [])
    return history if isinstance(history, list) else []


def save_dialog_history(dialog_id: str, history: list[dict[str, str]]) -> None:
    """Сохраняет историю конкретного диалога."""
    histories = load_dialog_histories()
    histories[dialog_id] = history
    save_dialog_histories(histories)


def clear_dialog_history(dialog_id: str) -> bool:
    """Удаляет историю одного диалога."""
    histories = load_dialog_histories()
    if dialog_id not in histories:
        return False

    del histories[dialog_id]
    save_dialog_histories(histories)
    return True


def clear_cache_files() -> list[str]:
    """Удаляет файлы кэша и возвращает список удалённых путей."""
    removed_files: list[str] = []

    for path in [METADATA_PATH, QUERY_CACHE_PATH, DIALOG_HISTORIES_PATH]:
        if path.exists():
            path.unlink()
            removed_files.append(str(path.relative_to(BASE_DIR)))

    return removed_files
