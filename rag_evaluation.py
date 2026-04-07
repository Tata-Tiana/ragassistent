"""CLI-инструмент для оценки качества RAG через RAGAS."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from datasets import Dataset
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from ragas import evaluate
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms import llm_factory
from ragas.metrics import AnswerCorrectness, AnswerRelevancy, Faithfulness, SemanticSimilarity

from config import CHUNK_MIN_SIZE, CHUNK_OVERLAP, CHUNK_SIZE, TOP_K
from rag_pipeline import RAGPipeline
from test_questions import RETRIEVAL_DEBUG_TESTS, TEST_DATASET, WARM_LEAD_TESTS


EMBEDDING_MODEL = "text-embedding-3-small"
EVALUATION_MODEL = "gpt-4o-mini"


class OpenAIEmbeddingAdapter(BaseRagasEmbeddings):
    """Адаптер OpenAI embeddings для legacy- и async-режимов RAGAS."""

    def __init__(self, sync_client: OpenAI, async_client: AsyncOpenAI, model: str) -> None:
        super().__init__()
        self.sync_client = sync_client
        self.async_client = async_client
        self.model = model
        self.provider_name = "OpenAIEmbeddingAdapter"

    def embed_query(self, text: str) -> list[float]:
        """Синхронный embedding одного текста для legacy-метрик."""
        response = self.sync_client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Синхронный embedding списка текстов для legacy-метрик."""
        response = self.sync_client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    async def aembed_query(self, text: str) -> list[float]:
        """Асинхронный embedding одного текста для async-метрик."""
        response = await self.async_client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Асинхронный embedding списка текстов для async-метрик."""
        response = await self.async_client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    def describe(self) -> str:
        """Короткое текстовое описание адаптера для CLI."""
        return f"{self.provider_name}(model={self.model}, client=AsyncOpenAI)"


class Reporter:
    """Простая обертка над print, чтобы при желании сохранить отчет в файл."""

    def __init__(self, save_report_path: str | None = None) -> None:
        self.save_report_path = Path(save_report_path) if save_report_path else None
        self.lines: list[str] = []

    def write(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)

    def save(self) -> None:
        """Сохраняет весь накопленный текстовый отчет в файл."""
        if not self.save_report_path:
            return

        self.save_report_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_report_path.write_text("\n".join(self.lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Человекочитаемая оценка качества RAG-ассистента 'Мастер на час'.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Ограничить количество тестов. Пример: --limit 5")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["main", "warm-lead", "retrieval-debug"],
        default="main",
        help="Выбрать набор тестов: основной, warm lead или retrieval-debug.",
    )
    parser.add_argument("--show-context", action="store_true", help="Показывать найденные чанки и источники по каждому тесту.")
    parser.add_argument("--verbose", action="store_true", help="Показывать расширенный вывод по каждому тесту.")
    parser.add_argument("--only-failed", action="store_true", help="Показывать только слабые тесты, где основная доступная метрика < 0.7.")
    parser.add_argument("--fast", action="store_true", help="Запустить безопасный укороченный режим с меньшим набором метрик.")
    parser.add_argument(
        "--save-report",
        type=str,
        default=None,
        help="Сохранить человекочитаемый отчет в текстовый файл. Пример: --save-report reports/last_eval.txt",
    )
    return parser.parse_args()


def print_header(reporter: Reporter) -> None:
    """Печатает вступительный блок и шпаргалку по командам."""
    reporter.write("==================================================")
    reporter.write("🚀 RAG Evaluation started")
    reporter.write("==================================================")
    reporter.write("")
    reporter.write('Проект: RAG-ассистент "Мастер на час"')
    reporter.write("")
    reporter.write("Что сейчас произойдёт:")
    reporter.write("1. Проверю базу знаний")
    reporter.write("2. Проверю индекс и кэш")
    reporter.write("3. Запущу тесты")
    reporter.write("4. Покажу метрики по каждому вопросу")
    reporter.write("5. В конце выведу общий итог")
    reporter.write("")
    reporter.write("💡 Шпаргалка по командам:")
    reporter.write("python rag_evaluation.py — полный запуск оценки по всем тестам")
    reporter.write("python rag_evaluation.py --limit 5 — короткий прогон на первых 5 тестах")
    reporter.write("python rag_evaluation.py --verbose — подробный вывод по каждому тесту")
    reporter.write("python rag_evaluation.py --dataset warm-lead — прогон по warm lead сценариям")
    reporter.write("python rag_evaluation.py --show-context — показать найденные чанки и источники")
    reporter.write("python rag_evaluation.py --limit 5 --verbose --show-context — короткий, но подробный прогон с контекстом")
    reporter.write("python rag_evaluation.py --only-failed — показать только слабые тесты")
    reporter.write("python rag_evaluation.py --fast — безопасный режим с меньшим набором метрик")
    reporter.write("python clear_cache.py — очистить файловый кэш")
    reporter.write("python clear_cache.py --all — очистить кэш и локальную базу ChromaDB")
    reporter.write("")
    reporter.write("==================================================")
    reporter.write("")


def print_step(reporter: Reporter, title: str) -> None:
    """Печатает короткий шаг сценария."""
    reporter.write(title)


def preview_text(text: str, limit: int = 140) -> str:
    """Делает короткое превью текста для терминала."""
    normalized = " ".join(text.split())
    return normalized if len(normalized) <= limit else f"{normalized[:limit].rstrip()}..."


def format_metric(value: Any) -> str:
    """Аккуратно форматирует значение метрики."""
    if value is None:
        return "n/a"
    try:
        if value != value:
            return "n/a"
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "n/a"


def metric_value(row: dict[str, Any], name: str) -> float | None:
    """Возвращает float-значение метрики или None, если метрика недоступна."""
    value = row.get(name)
    try:
        if value is None or value != value:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def choose_quality_metric(row: dict[str, Any]) -> tuple[str, float | None]:
    """Выбирает основную метрику для интерпретации качества."""
    for metric_name in ["answer_correctness", "semantic_similarity", "answer_relevancy", "faithfulness"]:
        value = metric_value(row, metric_name)
        if value is not None:
            return metric_name, value
    return "n/a", None


def build_comment_from_metrics(row: dict[str, Any]) -> str:
    """Строит короткий комментарий по качеству ответа."""
    _, quality_value = choose_quality_metric(row)
    answer_relevancy_value = metric_value(row, "answer_relevancy")
    faithfulness_value = metric_value(row, "faithfulness")

    if quality_value is None:
        comment = "Метрики доступны не полностью, поэтому итог лучше оценивать вручную по ответу и контексту."
    elif quality_value >= 0.85:
        comment = "Ответ хороший: система нашла нужный контекст и ответ близок к эталону."
    elif quality_value >= 0.65:
        comment = "Ответ частично корректный: смысл в целом близкий, но есть неточности или неполнота."
    else:
        comment = "Ответ слабый: система либо нашла не тот контекст, либо ответила неполно или неверно."

    hints: list[str] = []
    if answer_relevancy_value is not None and answer_relevancy_value < 0.6:
        hints.append("Ответ не очень точно попадает в формулировку вопроса.")
    if faithfulness_value is not None and faithfulness_value < 0.6:
        hints.append("Есть риск, что ответ слабо опирается на найденный контекст.")

    if hints:
        comment = f"{comment} {' '.join(hints)}"
    return comment


def build_summary_comment(avg_correctness: float | None) -> str:
    """Возвращает общую интерпретацию итогового качества системы."""
    if avg_correctness is None:
        return "Не все ключевые метрики доступны, поэтому итог лучше оценивать вместе с примерами ответов."
    if avg_correctness >= 0.85:
        return "Система работает хорошо. Можно переходить к точечной доработке качества."
    if avg_correctness >= 0.65:
        return "Система уже рабочая, но есть заметные слабые места. Стоит улучшить retrieval, prompt или базу знаний."
    return "Система работает нестабильно. Нужно проверить retrieval, prompt, embeddings и тестовый набор."


def build_summary_tips(averages: dict[str, float | None]) -> list[str]:
    """Формирует короткие советы на основе средних метрик."""
    tips: list[str] = []
    if averages.get("faithfulness") is not None and averages["faithfulness"] < 0.7:
        tips.append("Совет: ответы не всегда уверенно опираются на найденный контекст. Проверь prompt и качество retrieval.")
    if averages.get("answer_relevancy") is not None and averages["answer_relevancy"] < 0.7:
        tips.append("Совет: ответы не всегда достаточно точно попадают в вопрос. Проверь retrieval и формулировку запроса к модели.")
    if averages.get("answer_correctness") is not None and averages["answer_correctness"] < 0.7:
        tips.append("Совет: ответ заметно отличается от эталона. Проверь prompt и качество найденных источников.")
    return tips


def print_test_result(
    reporter: Reporter,
    item: dict[str, Any],
    row: dict[str, Any],
    test_number: int,
    total_tests: int,
    show_context: bool,
    verbose: bool,
) -> None:
    """Печатает человекочитаемый блок по одному тесту."""
    reporter.write("--------------------------------------------------")
    reporter.write(f"🧪 Тест {test_number} из {total_tests}")
    reporter.write("--------------------------------------------------")
    reporter.write("")
    reporter.write("❓ Вопрос:")
    reporter.write(item["question"])
    if item.get("type"):
        reporter.write(f"Тип теста: {item['type']}")
    reporter.write("")
    reporter.write("🤖 Ответ RAG:")
    reporter.write(item["answer"])
    reporter.write("")
    reporter.write("✅ Эталонный ответ:")
    reporter.write(item["ground_truth"])
    reporter.write("")
    reporter.write("📊 Метрики:")
    reporter.write(f"- Faithfulness: {format_metric(row.get('faithfulness'))}")
    reporter.write(f"- Answer relevance: {format_metric(row.get('answer_relevancy'))}")
    reporter.write(f"- Answer correctness: {format_metric(row.get('answer_correctness'))}")
    reporter.write(f"- Semantic similarity: {format_metric(row.get('semantic_similarity'))}")
    reporter.write("")
    reporter.write("💡 Комментарий:")
    reporter.write(build_comment_from_metrics(row))

    if show_context or verbose:
        reporter.write("")
        reporter.write("📚 Порядок retrieval:")
        contexts = item.get("contexts", [])
        if not contexts:
            reporter.write("Источники не найдены.")
        for index, context in enumerate(contexts, start=1):
            source = context.get("source") or "источник не указан"
            chunk_index = context.get("chunk_index", -1)
            reporter.write(f"{index}. {source}")
            reporter.write(f"   chunk_index: {chunk_index}")
            reporter.write(f"   Превью: {preview_text(context.get('text', ''))}")

    if verbose:
        reporter.write("")
        reporter.write("🔎 Техническая заметка:")
        reporter.write(f"Основная метрика для интерпретации: {choose_quality_metric(row)[0]}")

    reporter.write("")


def get_selected_dataset(dataset_name: str) -> list[dict[str, Any]]:
    """Возвращает выбранный набор тестов."""
    if dataset_name == "warm-lead":
        return WARM_LEAD_TESTS
    if dataset_name == "retrieval-debug":
        return RETRIEVAL_DEBUG_TESTS
    return TEST_DATASET


def collect_test_records(
    pipeline: RAGPipeline,
    dataset_name: str,
    limit: int | None,
    reporter: Reporter,
) -> list[dict[str, Any]]:
    """Прогоняет тесты через RAG pipeline и собирает данные для RAGAS."""
    dataset = get_selected_dataset(dataset_name)
    selected_tests = dataset[:limit] if limit else dataset
    records: list[dict[str, Any]] = []

    print_step(reporter, "🧪 Шаг 3. Запускаю тесты...")
    reporter.write(f"Всего тестов к запуску: {len(selected_tests)}")
    reporter.write("")

    for index, item in enumerate(selected_tests, start=1):
        pipeline.reset_history()
        reporter.write(f"   [{index}/{len(selected_tests)}] Обрабатываю вопрос: {item['question']}")
        result = pipeline.ask(item["question"])
        records.append(
            {
                "question": item["question"],
                "answer": result["answer"],
                "contexts": result["contexts"],
                "ground_truth": item["ground_truth"],
                "type": item.get("type", "n/a"),
                "user_input": item["question"],
                "response": result["answer"],
                "retrieved_contexts": [context["text"] for context in result["contexts"]],
                "reference": item["ground_truth"],
            }
        )

    reporter.write("")
    return records


def build_ragas_dataset(records: list[dict[str, Any]]) -> Dataset:
    """Преобразует собранные записи в Dataset для RAGAS."""
    dataset_rows = []
    for item in records:
        dataset_rows.append(
            {
                "question": item["question"],
                "answer": item["answer"],
                "contexts": [context["text"] for context in item["contexts"]],
                "ground_truth": item["ground_truth"],
                "user_input": item["question"],
                "response": item["answer"],
                "retrieved_contexts": [context["text"] for context in item["contexts"]],
                "reference": item["ground_truth"],
            }
        )
    return Dataset.from_list(dataset_rows)


def merge_records_with_scores(records: list[dict[str, Any]], scores: object) -> list[dict[str, Any]]:
    """Объединяет ответы по тестам и таблицу метрик в один список."""
    frame = scores.to_pandas()
    merged: list[dict[str, Any]] = []

    for index, item in enumerate(records):
        merged.append({**item, "metrics": frame.iloc[index].to_dict()})
    return merged


def should_show_test(item: dict[str, Any], only_failed: bool) -> bool:
    """Решает, нужно ли печатать тест с учетом фильтра only-failed."""
    if not only_failed:
        return True
    _, metric = choose_quality_metric(item["metrics"])
    if metric is None:
        return True
    return metric < 0.7


def calculate_averages(results: list[dict[str, Any]]) -> dict[str, float | None]:
    """Считает средние значения по всем доступным метрикам."""
    averages: dict[str, float | None] = {}
    for metric_name in ["faithfulness", "answer_relevancy", "answer_correctness", "semantic_similarity"]:
        values = [metric_value(item["metrics"], metric_name) for item in results if metric_value(item["metrics"], metric_name) is not None]
        averages[metric_name] = sum(values) / len(values) if values else None
    return averages


def print_summary(reporter: Reporter, results: list[dict[str, Any]]) -> None:
    """Печатает итоговый summary по всем тестам."""
    averages = calculate_averages(results)
    problematic_tests = [
        item for item in results if (choose_quality_metric(item["metrics"])[1] is not None and choose_quality_metric(item["metrics"])[1] < 0.7)
    ]
    avg_quality = averages.get("answer_correctness")
    if avg_quality is None:
        avg_quality = averages.get("semantic_similarity")
    if avg_quality is None:
        avg_quality = averages.get("answer_relevancy")
    if avg_quality is None:
        avg_quality = averages.get("faithfulness")

    reporter.write("==================================================")
    reporter.write("📈 SUMMARY")
    reporter.write("==================================================")
    reporter.write("")
    reporter.write(f"Всего тестов: {len(results)}")
    reporter.write(f"Успешно обработано: {len(results)}")
    reporter.write(f"Проблемных тестов: {len(problematic_tests)}")
    reporter.write("")
    reporter.write("Средние метрики:")
    reporter.write(f"- Faithfulness avg: {format_metric(averages.get('faithfulness'))}")
    reporter.write(f"- Answer relevance avg: {format_metric(averages.get('answer_relevancy'))}")
    reporter.write(f"- Answer correctness avg: {format_metric(averages.get('answer_correctness'))}")
    reporter.write(f"- Semantic similarity avg: {format_metric(averages.get('semantic_similarity'))}")
    reporter.write("")
    reporter.write("💡 Общая оценка:")
    reporter.write(build_summary_comment(avg_quality))

    tips = build_summary_tips(averages)
    if tips:
        reporter.write("")
        for tip in tips:
            reporter.write(tip)

    reporter.write("")
    reporter.write("💡 Быстрые команды:")
    reporter.write("- python rag_evaluation.py — полный прогон оценки")
    reporter.write("- python rag_evaluation.py --limit 5 — быстрый прогон на 5 тестах")
    reporter.write("- python rag_evaluation.py --verbose --show-context — подробный режим с retrieval")
    reporter.write("- python rag_evaluation.py --only-failed — показать только слабые тесты")
    reporter.write("- python rag_evaluation.py --fast — безопасный режим с меньшим набором метрик")
    reporter.write("- python clear_cache.py — очистить файловый кэш")
    reporter.write("- python clear_cache.py --all — очистить кэш и индекс")
    reporter.write("")


def build_runtime_components() -> tuple[Any, OpenAIEmbeddingAdapter]:
    """Создает совместимые с RAGAS LLM и embeddings-компоненты."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Не найден OPENAI_API_KEY. Создайте .env по примеру из .env.example.")

    sync_client = OpenAI(api_key=api_key)
    async_client = AsyncOpenAI(api_key=api_key)
    llm = llm_factory(EVALUATION_MODEL, client=async_client)
    embeddings = OpenAIEmbeddingAdapter(sync_client=sync_client, async_client=async_client, model=EMBEDDING_MODEL)
    return llm, embeddings


def get_enabled_metrics(args: argparse.Namespace, llm: Any, embeddings: OpenAIEmbeddingAdapter) -> list[Any]:
    """Возвращает набор метрик для выбранного режима запуска."""
    metrics: list[Any] = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
    ]

    if args.fast:
        return metrics

    metrics.append(AnswerCorrectness(llm=llm, embeddings=embeddings))
    metrics.append(SemanticSimilarity(embeddings=embeddings))
    return metrics


def evaluate_rag(args: argparse.Namespace) -> None:
    """Запускает полный CLI-сценарий оценки RAG."""
    reporter = Reporter(save_report_path=args.save_report)
    print_header(reporter)

    print_step(reporter, "📂 Шаг 1. Проверяю папку data/...")
    pipeline = RAGPipeline(dialog_id=f"evaluation_{args.dataset}")
    pipeline.reset_history()
    documents = pipeline.vector_store.load_documents()
    reporter.write(f"📦 Найдено файлов базы знаний: {len(documents)}")
    reporter.write("")
    reporter.write("Список файлов:")
    for document in documents:
        reporter.write(f"- {document['source']}")
    reporter.write("")

    print_step(reporter, "🔍 Шаг 2. Проверяю индекс и кэш...")
    if pipeline.index_status["status"] == "cached":
        reporter.write("✅ Найден существующий индекс")
        reporter.write("✅ Использую кэш, повторная индексация не требуется")
    else:
        reporter.write("⚠️ Индекс не найден")
        reporter.write("🧠 Создаю новый индекс...")
        reporter.write("✅ Индексация завершена")

    reporter.write(f"🧩 Всего чанков в базе: {pipeline.index_status['chunks_count']}")
    reporter.write("⚙️ Параметры RAG:")
    reporter.write(f"- CHUNK_SIZE = {CHUNK_SIZE}")
    reporter.write(f"- CHUNK_MIN_SIZE = {CHUNK_MIN_SIZE}")
    reporter.write(f"- CHUNK_OVERLAP = {CHUNK_OVERLAP}")
    reporter.write(f"- TOP_K = {TOP_K}")
    reporter.write(f"- DATASET = {args.dataset}")
    reporter.write("")

    llm, embeddings = build_runtime_components()
    metrics = get_enabled_metrics(args, llm, embeddings)

    if args.fast:
        reporter.write("⚠️ Запущен упрощённый режим (меньше метрик, быстрее запуск)")
        reporter.write("")

    reporter.write(f"🔌 Embeddings provider: {embeddings.describe()}")
    reporter.write(f"🧪 Метрики запуска: {', '.join(metric.name for metric in metrics)}")
    reporter.write("")

    records = collect_test_records(pipeline, args.dataset, args.limit, reporter)
    dataset = build_ragas_dataset(records)

    reporter.write("🧠 Шаг 4. Считаю метрики RAGAS...")
    scores = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
    )
    reporter.write("✅ Метрики рассчитаны")
    reporter.write("")

    results = merge_records_with_scores(records, scores)
    shown_results = [item for item in results if should_show_test(item, args.only_failed)]

    if args.only_failed:
        reporter.write("🧹 Включен фильтр only-failed: показываю только слабые тесты")
        reporter.write(f"Показано тестов: {len(shown_results)} из {len(results)}")
        reporter.write("")

    print_step(reporter, "📋 Шаг 5. Показываю результаты по тестам...")
    for index, item in enumerate(shown_results, start=1):
        print_test_result(
            reporter=reporter,
            item=item,
            row=item["metrics"],
            test_number=index,
            total_tests=len(shown_results),
            show_context=args.show_context,
            verbose=args.verbose,
        )

    if args.only_failed and not shown_results:
        reporter.write("Слабых тестов по выбранному порогу не найдено.")
        reporter.write("")

    print_summary(reporter, results)

    if args.save_report:
        reporter.write(f"📝 Отчет сохранен в: {args.save_report}")
    reporter.save()


if __name__ == "__main__":
    evaluate_rag(parse_args())
