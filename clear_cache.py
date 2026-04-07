"""Простая команда для очистки кэша проекта."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from cache_utils import BASE_DIR, CACHE_DIR, clear_cache_files


CHROMA_DIR = BASE_DIR / "chroma_db"


def parse_args() -> argparse.Namespace:
    """Разбирает параметры CLI."""
    parser = argparse.ArgumentParser(description="Очистка кэша RAG-проекта.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Очистить не только cache/, но и локальную базу chroma_db/.",
    )
    return parser.parse_args()


def main() -> None:
    """Удаляет файлы кэша и при необходимости локальную индексацию."""
    args = parse_args()

    print("Очистка кэша проекта...")
    print("Что делает команда:")
    print("- python clear_cache.py — удаляет файловый кэш")
    print("- python clear_cache.py --all — удаляет файловый кэш и локальный индекс ChromaDB")
    print("")
    removed_files = clear_cache_files()

    if removed_files:
        print("Удалены файлы кэша:")
        for item in removed_files:
            print(f"- {item}")
    else:
        print("Файлы кэша уже были пустыми или отсутствовали.")

    if args.all:
        if CHROMA_DIR.exists():
            shutil.rmtree(CHROMA_DIR)
            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            print("- chroma_db/ очищена")
        else:
            print("- chroma_db/ не найдена")

    if not CACHE_DIR.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("Готово.")


if __name__ == "__main__":
    main()
