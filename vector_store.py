"""Работа с несколькими документами, embeddings, ChromaDB и файловым кэшем."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import chromadb
from openai import OpenAI

from cache_utils import (
    calculate_text_hash,
    load_metadata,
    load_query_cache,
    save_metadata,
    save_query_cache,
)
from config import CHUNK_MIN_SIZE, CHUNK_OVERLAP, CHUNK_SIZE, TOP_K


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "local_rag_knowledge_base"
EMBEDDING_MODEL = "text-embedding-3-small"


class VectorStore:
    """Минимальная обертка для индексации документов и поиска по ним."""

    def __init__(self, openai_client: OpenAI) -> None:
        self.openai_client = openai_client
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        self.collection = self.chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    def load_documents(self) -> list[dict[str, str]]:
        """Читает все txt-файлы из папки data/."""
        documents: list[dict[str, str]] = []

        for path in sorted(DATA_DIR.glob("*.txt")):
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            documents.append({"source": path.name, "text": text})

        return documents

    def calculate_documents_hash(self, documents: list[dict[str, str]]) -> str:
        """Считает общий хэш по именам и содержимому всех txt-файлов."""
        combined_text = ""
        for document in documents:
            combined_text += f"{document['source']}\n{document['text']}\n"
        return calculate_text_hash(combined_text)

    def normalize_text(self, text: str) -> str:
        """Нормализует лишние пробелы, сохраняя структуру абзацев."""
        lines = [line.rstrip() for line in text.splitlines()]
        cleaned_lines: list[str] = []

        for line in lines:
            stripped = line.strip()
            if stripped:
                cleaned_lines.append(stripped)
            elif cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")

        return "\n".join(cleaned_lines).strip()

    def find_chunk_end(self, text: str, raw_end: int) -> int:
        """Сдвигает конец чанка к удобной границе, чтобы не резать слово."""
        if raw_end >= len(text):
            return len(text)

        window_end = min(raw_end + 80, len(text))
        for index in range(raw_end, window_end):
            if text[index].isspace():
                return index
        return raw_end

    def find_next_start(self, text: str, raw_start: int) -> int:
        """Сдвигает начало следующего чанка к началу слова."""
        if raw_start <= 0:
            return 0

        while raw_start < len(text) and not text[raw_start].isspace():
            raw_start += 1
        while raw_start < len(text) and text[raw_start].isspace():
            raw_start += 1
        return raw_start

    def split_long_block_with_window(self, text: str) -> list[str]:
        """Режет только слишком длинный блок скользящим окном."""
        normalized_text = self.normalize_text(text)
        if not normalized_text:
            return []

        if len(normalized_text) <= CHUNK_SIZE:
            return [normalized_text]

        overlap_size = int(CHUNK_SIZE * CHUNK_OVERLAP)
        step = CHUNK_SIZE - overlap_size
        chunks: list[str] = []
        start = 0
        text_length = len(normalized_text)

        # Overlap сохраняет часть соседнего текста.
        # Это помогает не потерять важное уточнение на границе двух чанков.
        # Размер 500 символов оставляем как понятный компромисс:
        # чанк уже содержит законченную мысль, но еще не слишком широкий.
        while start < text_length:
            raw_end = min(start + CHUNK_SIZE, text_length)
            end = self.find_chunk_end(normalized_text, raw_end)
            chunk = normalized_text[start:end].strip()

            if len(chunk) < CHUNK_MIN_SIZE and chunks:
                chunks[-1] = f"{chunks[-1]}\n{chunk}".strip()
                break

            if chunk:
                chunks.append(chunk)

            if end >= text_length:
                break

            raw_next_start = max(end - overlap_size, start + step)
            start = self.find_next_start(normalized_text, raw_next_start)

        return chunks

    def split_price_document(self, source: str, text: str) -> list[str]:
        """Сначала режет прайс по логическим блокам и строкам прайса."""
        normalized_text = self.normalize_text(text)
        lines = [line for line in normalized_text.splitlines() if line.strip()]
        if not lines:
            return []

        title = lines[0] if lines[0].startswith("#") else source
        intro_lines: list[str] = []
        price_lines: list[str] = []

        for line in lines[1:]:
            if " — от " in line or " - от " in line:
                price_lines.append(line)
            else:
                intro_lines.append(line)

        blocks: list[str] = []
        if intro_lines:
            blocks.append("\n".join([title, *intro_lines]))

        current_group: list[str] = [title]
        for price_line in price_lines:
            current_group.append(price_line)
            joined = "\n".join(current_group)

            # Для прайса важно держать рядом несколько соседних строк,
            # чтобы retrieval находил конкретную позицию вместе с контекстом раздела.
            if len(joined) >= CHUNK_MIN_SIZE or len(current_group) >= 4:
                blocks.append(joined)
                current_group = [title]

        if len(current_group) > 1:
            blocks.append("\n".join(current_group))

        final_chunks: list[str] = []
        for block in blocks:
            final_chunks.extend(self.split_long_block_with_window(block))
        return final_chunks

    def split_section_document(self, source: str, text: str) -> list[str]:
        """Сначала режет FAQ и сервисные тексты по смысловым секциям."""
        normalized_text = self.normalize_text(text)
        if not normalized_text:
            return []

        lines = normalized_text.splitlines()
        blocks: list[str] = []
        current_block: list[str] = []

        for line in lines:
            stripped = line.strip()

            if stripped == "---":
                if current_block:
                    blocks.append("\n".join(current_block).strip())
                    current_block = []
                continue

            if stripped.startswith("#"):
                if current_block:
                    blocks.append("\n".join(current_block).strip())
                current_block = [stripped]
                continue

            if stripped == "":
                if current_block:
                    current_block.append("")
                continue

            current_block.append(stripped)

        if current_block:
            blocks.append("\n".join(current_block).strip())

        if not blocks:
            blocks = [normalized_text]

        final_chunks: list[str] = []
        for block in blocks:
            final_chunks.extend(self.split_long_block_with_window(block))
        return final_chunks

    def split_document_into_chunks(self, source: str, text: str) -> list[str]:
        """Выбирает подходящий чанкинг по типу файла."""
        if source.startswith("price_"):
            return self.split_price_document(source, text)

        if source.startswith("faq_") or source.startswith("services_") or source.startswith("company_info_"):
            return self.split_section_document(source, text)

        return self.split_long_block_with_window(text)

    def create_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Создает embeddings для списка текстов через OpenAI API."""
        response = self.openai_client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return [item.embedding for item in response.data]

    def clear_collection(self) -> None:
        """Полностью пересоздает коллекцию, если нужно переиндексировать базу."""
        try:
            self.chroma_client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            pass

        self.collection = self.chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    def ensure_index(self) -> dict[str, Any]:
        """Проверяет кэш и при необходимости переиндексирует все документы."""
        documents = self.load_documents()
        file_hash = self.calculate_documents_hash(documents)
        metadata = load_metadata()
        collection_count = self.collection.count()

        if metadata.get("file_hash") == file_hash and collection_count > 0:
            return {
                "status": "cached",
                "chunks_count": collection_count,
                "files_count": len(documents),
            }

        all_chunks: list[str] = []
        all_metadatas: list[dict[str, Any]] = []

        for document in documents:
            chunks = self.split_document_into_chunks(document["source"], document["text"])
            for chunk_index, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadatas.append(
                    {
                        "source": document["source"],
                        "chunk_index": chunk_index,
                    }
                )

        embeddings = self.create_embeddings(all_chunks)

        self.clear_collection()
        ids = [f"chunk_{index}" for index in range(len(all_chunks))]
        self.collection.add(
            ids=ids,
            documents=all_chunks,
            embeddings=embeddings,
            metadatas=all_metadatas,
        )

        save_metadata(file_hash=file_hash, chunks_count=len(all_chunks))
        save_query_cache({})

        return {
            "status": "reindexed",
            "chunks_count": len(all_chunks),
            "files_count": len(documents),
        }

    def get_question_embedding(self, question: str) -> list[float]:
        """Берет embedding вопроса из кэша или запрашивает заново."""
        query_cache = load_query_cache()
        cached_item = query_cache.get(question, {})

        if cached_item.get("question_embedding"):
            return cached_item["question_embedding"]

        response = self.openai_client.embeddings.create(model=EMBEDDING_MODEL, input=question)
        embedding = response.data[0].embedding

        query_cache[question] = {
            **cached_item,
            "question_embedding": embedding,
        }
        save_query_cache(query_cache)
        return embedding

    def build_retrieval_debug(self, contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Строит короткое представление retrieval для терминала."""
        debug_rows: list[dict[str, Any]] = []
        for rank, context in enumerate(contexts[:TOP_K], start=1):
            preview = " ".join(context["text"].split())
            if len(preview) > 140:
                preview = f"{preview[:140].rstrip()}..."
            debug_rows.append(
                {
                    "rank": rank,
                    "source": context.get("source", "источник не указан"),
                    "chunk_index": context.get("chunk_index", -1),
                    "preview": preview,
                }
            )
        return debug_rows

    def search(self, question: str) -> list[dict[str, Any]]:
        """Ищет TOP_K наиболее релевантных чанков и возвращает текст вместе с source."""
        question_embedding = self.get_question_embedding(question)
        results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=TOP_K,
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        contexts: list[dict[str, Any]] = []
        for rank, (document, metadata) in enumerate(zip(documents, metadatas), start=1):
            contexts.append(
                {
                    "rank": rank,
                    "text": document,
                    "source": metadata.get("source", "unknown"),
                    "chunk_index": metadata.get("chunk_index", -1),
                }
            )
        return contexts[:TOP_K]

    def get_cached_answer(self, question: str) -> Optional[dict[str, Any]]:
        """Возвращает закэшированный ответ и контексты, если они уже есть."""
        query_cache = load_query_cache()
        cached_item = query_cache.get(question)

        if not cached_item:
            return None

        if cached_item.get("answer") and cached_item.get("retrieved_contexts"):
            return {
                "answer": cached_item["answer"],
                "contexts": cached_item["retrieved_contexts"][:TOP_K],
            }
        return None

    def save_query_result(self, question: str, answer: str, contexts: list[dict[str, Any]]) -> None:
        """Сохраняет результат ответа на вопрос в файловый кэш."""
        query_cache = load_query_cache()
        cached_item = query_cache.get(question, {})

        query_cache[question] = {
            **cached_item,
            "answer": answer,
            "retrieved_contexts": contexts[:TOP_K],
        }
        save_query_cache(query_cache)
