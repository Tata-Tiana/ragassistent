"""Консольный интерфейс для локального RAG-ассистента."""

from __future__ import annotations

import argparse

from config import TOP_K
from rag_pipeline import RAGPipeline


def preview_text(text: str, limit: int = 120) -> str:
    """Делает короткое превью чанка для печати в консоли."""
    normalized = " ".join(text.split())
    return normalized if len(normalized) <= limit else f"{normalized[:limit].rstrip()}..."


def parse_args() -> argparse.Namespace:
    """Разбирает параметры CLI для консольного приложения."""
    parser = argparse.ArgumentParser(description="Консольный RAG-ассистент 'Мастер на час'.")
    parser.add_argument(
        "--dialog-id",
        type=str,
        default="console",
        help="Идентификатор диалога для сохранения истории между запусками.",
    )
    parser.add_argument(
        "--debug-retrieval",
        action="store_true",
        help="Показывать найденные чанки с порядком retrieval, source и preview.",
    )
    return parser.parse_args()


def main() -> None:
    """Запускает цикл вопросов в терминале."""
    args = parse_args()
    print("Запуск RAG-ассистента...")
    print("\n💡 Шпаргалка по командам:")
    print("python app.py — обычный запуск ассистента")
    print("python app.py --dialog-id avito_chat_42 — запуск с конкретным id диалога")
    print("python app.py --debug-retrieval — запуск с подробным выводом retrieval")
    print("python clear_cache.py — очистить файловый кэш")
    print("python clear_cache.py --all — очистить кэш и локальный индекс\n")
    pipeline = RAGPipeline(dialog_id=args.dialog_id)
    print(f"Статус индексации: {pipeline.index_status['status']}, чанков: {pipeline.index_status['chunks_count']}")
    print(f"ID диалога: {pipeline.dialog_id}")
    print("Введите вопрос. Для выхода напишите: exit")
    print("Чтобы сбросить память текущего диалога, напишите: reset\n")

    while True:
        question = input("Ваш вопрос: ").strip()

        if question.lower() == "exit":
            print("Работа завершена.")
            break

        if question.lower() == "reset":
            pipeline.reset_history()
            print("История диалога очищена.\n")
            continue

        if not question:
            print("Пожалуйста, введите непустой вопрос.\n")
            continue

        result = pipeline.ask(question)

        mode_label = "тёплый клиент" if result.get("warm_lead") else "обычный вопрос"
        print(f"\nРежим ответа: {mode_label}")
        print("\nОтвет:")
        print(result["answer"])
        print("\nНайденные чанки:")
        for index, chunk in enumerate(result["contexts"], start=1):
            print(f"{index}. source: {chunk['source']}")
            print(f"   preview: {preview_text(chunk['text'])}")

        if args.debug_retrieval:
            print("\nDebug retrieval:")
            print(f"🔍 Retrieval TOP_K = {TOP_K}")
            print(f"🧭 Query type = {result.get('query_type', 'general')}")
            print(f"🗂️ Doc types filter = {', '.join(result.get('doc_types_filter') or ['all'])}")
            for item in pipeline.vector_store.build_retrieval_debug(result["contexts"]):
                print(f"{item['rank']}. source: {item['source']}")
                print(f"   doc_type: {item['doc_type']}")
                print(f"   chunk_index: {item['chunk_index']}")
                print(f"   keyword_score: {item['keyword_score']}")
                print(f"   embedding_score: {item['embedding_score']}")
                print(f"   hybrid_score: {item['hybrid_score']}")
                if item['rerank_score']:
                    print(f"   rerank_score: {item['rerank_score']}")
                print(f"   preview: {item['preview']}")
        print()


if __name__ == "__main__":
    main()
