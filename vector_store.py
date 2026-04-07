"""Работа с несколькими документами, embeddings, ChromaDB и файловым кэшем."""

from __future__ import annotations

import re
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
from config import (
    CHUNK_MIN_SIZE,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    ENABLE_RERANKER,
    FAQ_EMBEDDING_WEIGHT,
    FAQ_KEYWORD_WEIGHT,
    PRICE_RETRIEVAL_K,
    PRICE_EMBEDDING_WEIGHT,
    PRICE_KEYWORD_WEIGHT,
    SERVICE_RETRIEVAL_K,
    SERVICE_EMBEDDING_WEIGHT,
    SERVICE_KEYWORD_WEIGHT,
    TOP_K,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "local_rag_knowledge_base"
EMBEDDING_MODEL = "text-embedding-3-small"
INDEX_VERSION = 2

PRICE_QUERY_MARKERS = [
    "сколько стоит",
    "цен",
    "стоимост",
    "по цене",
    "от ",
]

SERVICE_QUERY_MARKERS = [
    "что входит",
    "что не входит",
    "какие работ",
    "подходит",
    "не подходит",
    "формат мастер на час",
    "капитальн",
]

FAQ_QUERY_MARKERS = [
    "гарант",
    "оплат",
    "материал",
    "район",
    "районы",
    "москва",
    "инструмент",
    "выезд",
]

LIST_MARKERS = ["- ", "•", ":"]

STOPWORDS = {
    "и",
    "в",
    "во",
    "на",
    "с",
    "со",
    "по",
    "для",
    "от",
    "до",
    "или",
    "ли",
    "а",
    "но",
    "что",
    "это",
    "как",
    "у",
    "к",
    "из",
    "за",
    "не",
    "нет",
    "при",
    "если",
    "так",
    "же",
    "уже",
    "мне",
    "вам",
}


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

    def get_doc_type(self, source: str) -> str:
        """Определяет тип документа по имени файла."""
        if source.startswith("price_"):
            return "price"
        if source.startswith("faq_"):
            return "faq"
        if source.startswith("services_"):
            return "service"
        if source.startswith("company_info_"):
            return "company_info"
        return "general"

    def normalize_query(self, text: str) -> str:
        """Нормализует пользовательский запрос для keyword search."""
        text = text.lower().replace("ё", "е")
        text = re.sub(r"[\.,:;!\?\(\)\[\]\"'«»/\\\-]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def tokenize_query(self, text: str) -> list[str]:
        """Разбивает запрос на полезные слова без коротких стоп-слов."""
        normalized = self.normalize_query(text)
        tokens = re.findall(r"[a-zа-я0-9]+", normalized)
        return [token for token in tokens if len(token) > 2 and token not in STOPWORDS]

    def tokenize_text(self, text: str) -> list[str]:
        """Общая токенизация текста для lexical search."""
        return self.tokenize_query(text)

    def tokens_soft_match(self, left_token: str, right_token: str) -> bool:
        """Мягко сравнивает токены, чтобы переживать обычные словоформы."""
        if left_token == right_token:
            return True
        if left_token in right_token or right_token in left_token:
            return True
        if len(left_token) >= 5 and len(right_token) >= 5:
            return left_token[:5] == right_token[:5]
        return False

    def detect_query_type(self, query: str) -> str:
        """Просто определяет тип запроса для retrieval."""
        normalized_query = self.normalize_query(query)

        if any(marker in normalized_query for marker in PRICE_QUERY_MARKERS):
            return "price"
        if any(marker in normalized_query for marker in SERVICE_QUERY_MARKERS):
            return "service"
        if any(marker in normalized_query for marker in FAQ_QUERY_MARKERS):
            return "faq"
        return "general"

    def lexical_overlap_score(self, query: str, text: str, query_type: str) -> float:
        """Считает простой lexical score для keyword retrieval."""
        query_tokens = self.tokenize_text(query)
        text_tokens = self.tokenize_text(text)
        if not query_tokens or not text_tokens:
            return 0.0

        score = 0.0
        for token in query_tokens:
            if any(self.tokens_soft_match(token, text_token) for text_token in text_tokens):
                # Более длинные и редкие по виду слова дают чуть больший вклад.
                score += 1.0 + min(len(token) / 10, 0.5)

        normalized_query = self.normalize_query(query)
        normalized_text = self.normalize_query(text)

        if normalized_query and normalized_query in normalized_text:
            score += 3.0

        # Для service-вопросов списки полезнее общих абзацев.
        if query_type == "service" and any(marker in text for marker in LIST_MARKERS):
            score += 1.2

        return round(score, 4)

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

        if (
            metadata.get("file_hash") == file_hash
            and metadata.get("index_version") == INDEX_VERSION
            and collection_count > 0
        ):
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
                        "doc_type": self.get_doc_type(document["source"]),
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

        save_metadata(file_hash=file_hash, chunks_count=len(all_chunks), index_version=INDEX_VERSION)
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

    def build_context(
        self,
        document: str,
        metadata: dict[str, Any],
        rank: int,
        embedding_distance: float | None = None,
        keyword_score: int = 0,
    ) -> dict[str, Any]:
        """Собирает единый формат context-объекта."""
        embedding_score = 0.0
        if embedding_distance is not None:
            embedding_score = 1 / (1 + max(embedding_distance, 0))

        return {
            "rank": rank,
            "text": document,
            "source": metadata.get("source", "unknown"),
            "doc_type": metadata.get("doc_type", "general"),
            "chunk_index": metadata.get("chunk_index", -1),
            "embedding_distance": embedding_distance,
            "embedding_score": round(embedding_score, 4),
            "keyword_score": keyword_score,
        }

    def run_chroma_query(
        self,
        query_embedding: list[float],
        n_results: int,
        doc_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Выполняет один Chroma query и форматирует результат."""
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if doc_type:
            query_kwargs["where"] = {"doc_type": doc_type}

        results = self.collection.query(**query_kwargs)
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        contexts: list[dict[str, Any]] = []
        for rank, (document, metadata, distance) in enumerate(zip(documents, metadatas, distances), start=1):
            contexts.append(
                self.build_context(
                    document=document,
                    metadata=metadata,
                    rank=rank,
                    embedding_distance=distance,
                )
            )
        return contexts

    def embedding_search(
        self,
        question: str,
        n_results: int,
        doc_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Ищет по embeddings, при необходимости только в нужных типах документов."""
        question_embedding = self.get_question_embedding(question)

        if not doc_types:
            return self.run_chroma_query(question_embedding, n_results=n_results)

        merged: dict[tuple[str, int], dict[str, Any]] = {}
        for doc_type in doc_types:
            for item in self.run_chroma_query(question_embedding, n_results=n_results, doc_type=doc_type):
                key = (item["source"], item["chunk_index"])
                existing = merged.get(key)
                if not existing or item["embedding_score"] > existing["embedding_score"]:
                    merged[key] = item

        results = sorted(
            merged.values(),
            key=lambda item: (-item["embedding_score"], item["source"], item["chunk_index"]),
        )
        for rank, item in enumerate(results, start=1):
            item["rank"] = rank
        return results

    def search_price_by_keywords(self, query: str, k: int = PRICE_RETRIEVAL_K) -> list[dict[str, Any]]:
        """Ищет price-чанки по простому совпадению слов из запроса."""
        return self.search_by_keywords(query=query, top_k=k, doc_types=["price"], query_type="price")

    def search_by_keywords(
        self,
        query: str,
        chunks: list[dict[str, Any]] | None = None,
        top_k: int = 10,
        doc_types: list[str] | None = None,
        query_type: str = "general",
    ) -> list[dict[str, Any]]:
        """Делает простой lexical search по токенам и точным фразам."""
        if chunks is None:
            get_kwargs: dict[str, Any] = {"include": ["documents", "metadatas"]}
            if doc_types and len(doc_types) == 1:
                get_kwargs["where"] = {"doc_type": doc_types[0]}
            raw_results = self.collection.get(**get_kwargs)
            documents = raw_results.get("documents", [])
            metadatas = raw_results.get("metadatas", [])
            chunks = [
                self.build_context(document=document, metadata=metadata, rank=0)
                for document, metadata in zip(documents, metadatas)
                if not doc_types or metadata.get("doc_type", "general") in doc_types
            ]

        keyword_rows: list[dict[str, Any]] = []
        for chunk in chunks:
            keyword_score = self.lexical_overlap_score(query, chunk["text"], query_type=query_type)
            if keyword_score <= 0:
                continue

            item = {**chunk}
            item["keyword_score"] = keyword_score
            keyword_rows.append(item)

        keyword_rows.sort(
            key=lambda item: (-item["keyword_score"], item["source"], item["chunk_index"]),
        )
        for rank, item in enumerate(keyword_rows, start=1):
            item["rank"] = rank
        return keyword_rows[:top_k]

    def extract_best_price_line(self, query: str, text: str) -> str:
        """Оставляет одну самую релевантную строку с ценой вместо списка соседних цен."""
        query_tokens = set(self.tokenize_query(query))
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        price_lines = [line for line in lines if not line.startswith("#")]
        if not price_lines:
            price_lines = lines

        best_line = price_lines[0] if price_lines else text.strip()
        best_score = -1

        for line in price_lines:
            line_tokens = set(self.tokenize_query(line))
            score = len(query_tokens & line_tokens)
            if score > best_score:
                best_score = score
                best_line = line

        return best_line

    def deduplicate_contexts(self, contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Удаляет точные дубликаты retrieved чанков."""
        unique_contexts: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, int]] = set()

        for context in contexts:
            key = (context["source"], context["chunk_index"])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique_contexts.append(context)

        return unique_contexts

    def texts_are_too_similar(self, left_text: str, right_text: str) -> bool:
        """Проверяет, что два чанка почти одинаковые по смыслу и словам."""
        left_tokens = set(self.tokenize_query(left_text))
        right_tokens = set(self.tokenize_query(right_text))
        if not left_tokens or not right_tokens:
            return False

        overlap = len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens))
        return overlap >= 0.75

    def keep_diverse_contexts(self, contexts: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        """Оставляет несколько разных по смыслу чанков без почти одинаковых дублей."""
        selected: list[dict[str, Any]] = []

        for context in contexts:
            if any(self.texts_are_too_similar(context["text"], item["text"]) for item in selected):
                continue
            selected.append(context)
            if len(selected) >= limit:
                break

        return selected

    def doc_types_for_query(self, query_type: str) -> list[str] | None:
        """Возвращает приоритетные типы документов для query_type."""
        if query_type == "price":
            return ["price"]
        if query_type == "service":
            return ["service", "faq"]
        if query_type == "faq":
            return ["faq", "service"]
        return None

    def get_hybrid_weights(self, query_type: str) -> tuple[float, float]:
        """Возвращает веса keyword и embedding score для hybrid retrieval."""
        if query_type == "price":
            return PRICE_KEYWORD_WEIGHT, PRICE_EMBEDDING_WEIGHT
        if query_type == "service":
            return SERVICE_KEYWORD_WEIGHT, SERVICE_EMBEDDING_WEIGHT
        if query_type == "faq":
            return FAQ_KEYWORD_WEIGHT, FAQ_EMBEDDING_WEIGHT
        return 0.5, 0.5

    def apply_hybrid_score(self, contexts: list[dict[str, Any]], query_type: str) -> list[dict[str, Any]]:
        """Добавляет hybrid_score к результатам retrieval."""
        keyword_weight, embedding_weight = self.get_hybrid_weights(query_type)
        scored: list[dict[str, Any]] = []
        for item in contexts:
            hybrid_score = item.get("keyword_score", 0.0) * keyword_weight + item.get("embedding_score", 0.0) * embedding_weight
            scored.append({**item, "hybrid_score": round(hybrid_score, 4)})
        return scored

    def rerank_candidates(self, query: str, query_type: str, contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Легкий rule-based reranker без новых зависимостей."""
        normalized_query = self.normalize_query(query)
        query_tokens = set(self.tokenize_text(query))
        reranked: list[dict[str, Any]] = []

        for item in contexts:
            text = item["text"]
            normalized_text = self.normalize_query(text)
            text_tokens = set(self.tokenize_text(text))
            bonus = 0.0

            if normalized_query and normalized_query in normalized_text:
                bonus += 2.5
            if query_tokens and query_tokens.issubset(text_tokens):
                bonus += 1.5
            if query_type == "service" and any(marker in text for marker in LIST_MARKERS):
                bonus += 1.0
            if query_type == "price" and len(normalized_text) > 220:
                bonus -= 1.0
            if query_type == "price" and " — от " in text:
                bonus += 1.0

            reranked.append(
                {
                    **item,
                    "rerank_bonus": round(bonus, 4),
                    "rerank_score": round(item.get("hybrid_score", 0.0) + bonus, 4),
                }
            )

        reranked.sort(
            key=lambda item: (-item.get("rerank_score", item.get("hybrid_score", 0.0)), item["source"], item["chunk_index"]),
        )
        for rank, item in enumerate(reranked, start=1):
            item["rank"] = rank
        return reranked

    def merge_price_results(
        self,
        query: str,
        embedding_results: list[dict[str, Any]],
        keyword_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Объединяет embedding и keyword результаты для ценовых вопросов."""
        merged: dict[tuple[str, int], dict[str, Any]] = {}

        for item in embedding_results:
            key = (item["source"], item["chunk_index"])
            merged[key] = {**item}

        for item in keyword_results:
            key = (item["source"], item["chunk_index"])
            existing = merged.get(key)
            if existing:
                existing["keyword_score"] = max(existing.get("keyword_score", 0), item["keyword_score"])
            else:
                merged[key] = {**item}

        results = self.apply_hybrid_score(list(merged.values()), query_type="price")
        results.sort(
            key=lambda item: (-item.get("hybrid_score", 0), -item.get("keyword_score", 0), item["source"], item["chunk_index"]),
        )

        for rank, item in enumerate(results, start=1):
            item["rank"] = rank
            item["text"] = self.extract_best_price_line(query, item["text"])
        return results

    def hybrid_search(
        self,
        query: str,
        query_type: str,
        top_k: int = 5,
        doc_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Объединяет embedding search и keyword search в один hybrid retrieval."""
        embedding_results = self.embedding_search(
            question=query,
            n_results=max(top_k, TOP_K),
            doc_types=doc_types,
        )
        keyword_results = self.search_by_keywords(
            query=query,
            top_k=max(top_k, TOP_K),
            doc_types=doc_types,
            query_type=query_type,
        )

        merged: dict[tuple[str, int], dict[str, Any]] = {}
        for item in embedding_results:
            key = (item["source"], item["chunk_index"])
            merged[key] = {**item}

        for item in keyword_results:
            key = (item["source"], item["chunk_index"])
            existing = merged.get(key)
            if existing:
                existing["keyword_score"] = max(existing.get("keyword_score", 0.0), item.get("keyword_score", 0.0))
            else:
                merged[key] = {**item}

        results = self.apply_hybrid_score(list(merged.values()), query_type=query_type)
        results.sort(
            key=lambda item: (-item.get("hybrid_score", 0), -item.get("keyword_score", 0), -item.get("embedding_score", 0), item["source"], item["chunk_index"]),
        )

        if ENABLE_RERANKER:
            results = self.rerank_candidates(query=query, query_type=query_type, contexts=results[: max(top_k, 8)])

        for rank, item in enumerate(results, start=1):
            item["rank"] = rank
        return results[:top_k]

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
                    "doc_type": context.get("doc_type", "unknown"),
                    "chunk_index": context.get("chunk_index", -1),
                    "keyword_score": context.get("keyword_score", 0),
                    "embedding_score": context.get("embedding_score", 0.0),
                    "hybrid_score": context.get("hybrid_score", 0.0),
                    "rerank_score": context.get("rerank_score", 0.0),
                    "preview": preview,
                }
            )
        return debug_rows

    def search(
        self,
        question: str,
        query_type: str = "general",
        original_question: str | None = None,
    ) -> list[dict[str, Any]]:
        """Ищет чанки с учетом типа запроса: price, service, faq или general."""
        source_question = original_question or question
        doc_types = self.doc_types_for_query(query_type)

        if query_type == "price":
            embedding_results = self.embedding_search(question=question, n_results=PRICE_RETRIEVAL_K, doc_types=doc_types)
            keyword_results = self.search_price_by_keywords(source_question, k=PRICE_RETRIEVAL_K)
            merged_results = self.merge_price_results(source_question, embedding_results, keyword_results)
            if ENABLE_RERANKER:
                merged_results = self.rerank_candidates(source_question, query_type="price", contexts=merged_results)
            contexts = self.deduplicate_contexts(merged_results)
            return contexts[:1]

        if query_type == "service":
            hybrid_results = self.hybrid_search(
                query=source_question,
                query_type="service",
                top_k=SERVICE_RETRIEVAL_K,
                doc_types=doc_types,
            )
            deduplicated = self.deduplicate_contexts(hybrid_results)
            diverse = self.keep_diverse_contexts(deduplicated, limit=3)
            return diverse[:3]

        if query_type == "faq":
            hybrid_results = self.hybrid_search(
                query=source_question,
                query_type="faq",
                top_k=TOP_K + 2,
                doc_types=doc_types,
            )
            deduplicated = self.deduplicate_contexts(hybrid_results)
            diverse = self.keep_diverse_contexts(deduplicated, limit=3)
            return diverse[:3]

        hybrid_results = self.hybrid_search(
            query=source_question,
            query_type="general",
            top_k=TOP_K,
            doc_types=doc_types,
        )
        return self.deduplicate_contexts(hybrid_results)[:TOP_K]

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
