"""Отдельная продакшн-оценка persona-ответов бота Тиграна."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from rag_pipeline import RAGPipeline


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "eval_dataset_tigran.json"

FACT_WEIGHT = 0.5
PERSONA_WEIGHT = 0.2
STYLE_WEIGHT = 0.2
BEHAVIOR_WEIGHT = 0.1
FAILED_THRESHOLD = 0.65

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
    "вам",
}

FIRST_PERSON_MARKERS = [
    "я ",
    "я\n",
    "могу",
    "сделаю",
    "посмотрю",
    "подскажу",
    "напишу",
    "уточню",
    "смогу",
    "берусь",
]

GLOBAL_FORBIDDEN_PERSONA_PHRASES = [
    "как ассистент",
    "как бот",
    "я как ии",
    "как ии",
    "в базе знаний",
    "согласно контексту",
    "согласно базе знаний",
]

DRY_STYLE_PHRASES = [
    "в рамках услуги",
    "согласно контексту",
    "в базе знаний нет точного ответа",
    "предоставляется",
    "осуществляется",
    "выполняется",
]

HUMAN_STYLE_MARKERS = [
    "с ходу",
    "нужно уточнить",
    "пришлите фото",
    "посмотрю",
    "скажу точнее",
    "могу подъехать",
    "напишу чуть позже",
    "лучше на месте глянуть",
    "по тому что у меня есть",
    "по тому что есть",
    "точнее зависит от ситуации",
    "обычно от",
]

STRONG_CTA_MARKERS = [
    "пришлите фото",
    "скиньте фото",
    "можете прислать фото",
    "напишите адрес",
    "напишите район",
    "опишите задачу",
    "опишите проблему",
    "перечислите задачи",
    "полный список работ",
    "договориться на выезд",
    "договоримся на выезд",
    "созвониться",
    "замер",
]

SOFT_CTA_MARKERS = [
    "если хотите",
    "можем обсудить",
    "могу подсказать",
    "могу уточнить",
    "дайте знать",
    "спрашивайте",
    "можно обсудить",
    "уточните",
    "напишите",
]

BEHAVIOR_MARKERS = {
    "можно сказать, что все зависит от загрузки": ["зависит от загрузки", "если будет окно", "если по времени получится", "по загрузке"],
    "можно предложить написать адрес или фото задачи": ["адрес", "пришлите фото", "скиньте фото", "напишите адрес"],
    "можно предложить прислать список работ или фото": ["список работ", "пришлите фото", "скиньте фото"],
    "можно честно сказать, что с ходу точную цену не назвать": ["с ходу", "точно не скажу", "без осмотра точно не скажу"],
    "можно предложить подсказать, что лучше купить": ["подскажу, что лучше купить", "подскажу что лучше купить", "могу подсказать", "скажу, что лучше взять"],
    "можно предложить написать список задач": ["список задач", "напишите список задач", "перечислите задачи"],
    "можно пояснить, что для такого нужна полноценная бригада": ["нужна бригада", "лучше бригадой", "полноценная бригада"],
    "можно попросить адрес или район": ["какой район", "напишите район", "адрес"],
    "можно коротко и без лишнего": ["наличными", "переводом"],
    "можно ответить коротко и спокойно": ["да, гарантия есть", "гарантия есть", "зависит от типа работ"],
    "желательно предложить прислать фото": ["пришлите фото", "скиньте фото", "можете прислать фото"],
    "предложить прислать список задач": ["список задач", "полный список работ", "перечислите задачи"],
    "можно добавить, что точнее зависит от задачи": ["зависит от задачи", "зависит от ситуации", "точнее скажу"],
    "можно предложить прислать фото стены или карниза": ["фото стены", "фото карниза", "пришлите фото"],
    "можно предложить прислать модель телевизора или фото места": ["модель телевизора", "фото места", "пришлите фото"],
    "можно попросить фото или модель шкафа": ["фото шкафа", "модель шкафа", "пришлите фото"],
    "можно уточнить, что зависит от конкретной позиции": ["зависит от конкретной позиции", "какая именно позиция", "какая именно мебель"],
    "можно сказать, что точнее зависит от модели и условий установки": ["зависит от модели", "условий установки", "точнее зависит"],
    "можно уточнить, какой именно унитаз и нужен ли демонтаж": ["какой именно унитаз", "нужен ли демонтаж", "какой унитаз"],
    "можно предложить прислать фото": ["фото", "пришлите фото", "скиньте фото"],
    "можно сказать, что цена зависит от деталей": ["зависит", "нужно уточнить", "точнее смогу сказать", "после осмотра"],
    "можно предложить уточнить, кто будет покупать материалы": ["кто будет покупать", "если хотите", "можем согласовать"],
    "можно предложить прислать фото или коротко описать задачу": ["пришлите фото", "опишите задачу", "коротко опишите"],
    "можно попросить полный список задач": ["полный список", "список работ", "перечислите задачи"],
    "можно предложить прислать фото": ["фото", "пришлите фото", "скиньте фото"],
    "можно честно сказать, что с ходу точного ответа нет": ["с ходу", "точно не скажу", "нужно уточнить"],
    "честно сказать, что нужно уточнить": ["нужно уточнить", "с ходу не скажу", "точнее скажу позже"],
    "аккуратно объяснить, что это не формат мастер на час": ["не формат мастер на час", "лучше бригадой", "другой формат работ"],
    "можно сказать, что цена зависит от типа шкафа": ["зависит от типа шкафа", "какой именно шкаф", "тип шкафа"],
    "можно сказать, что нужно уточнить детали подключения": ["нужно уточнить", "какое подключение", "детали подключения"],
    "аккуратно объяснить, что это уже другой формат работ": ["другой формат работ", "не формат мастер на час", "лучше бригадой"],
    "можно уточнить, есть ли уже подводка": ["есть ли подводка", "уже подведено", "подводка"],
    "можно уточнить высоту потолка или тип крепления": ["высота потолка", "тип крепления", "какое крепление"],
    "можно коротко объяснить это без жесткости": ["оплачивается только выезд", "без жесткости", "считается только выезд"],
    "допустима фраза вроде 'с ходу точно не скажу'": ["с ходу", "точно не скажу", "без осмотра точно не скажу"],
    "можно предложить фото или описание задачи": ["пришлите фото", "опишите задачу", "скиньте фото"],
    "можно пояснить, что это уже формат полноценного ремонта или бригады": ["полноценный ремонт", "нужна бригада", "это уже не мастер на час"],
    "допустима мягкая человеческая формулировка вроде 'по Москве сейчас не подскажу, работаю в СПб'": ["по москве сейчас не подскажу", "работаю в спб", "я в спб работаю"],
    "допустима фраза 'сейчас точно не скажу, но обычно работаю с 8 до 22'": ["сейчас точно не скажу", "обычно работаю с 8 до 22", "с 8 до 22"],
    "можно предложить написать проблему": ["напишите проблему", "опишите проблему", "в чем проблема"],
}


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы CLI."""
    parser = argparse.ArgumentParser(
        description="Отдельная persona-оценка ответов бота Тиграна.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Ограничить количество тестов. Пример: --limit 5")
    parser.add_argument("--only-failed", action="store_true", help=f"Показывать только слабые ответы с total_score < {FAILED_THRESHOLD}")
    parser.add_argument("--verbose", action="store_true", help="Показывать расширенный вывод с контекстом и тех. заметками.")
    return parser.parse_args()


def print_header() -> None:
    """Печатает вступительный блок."""
    print("==================================================")
    print("🛠️ Tigran Persona Evaluation")
    print("==================================================")
    print("")
    print("Что проверяется:")
    print("1. Переданы ли ключевые факты")
    print("2. Выдержана ли persona живого мастера")
    print("3. Нормально ли звучит стиль ответа")
    print("4. Есть ли полезное следующее действие, если оно уместно")
    print("")
    print("💡 Команды:")
    print("python rag_evaluation_tigran.py")
    print("python rag_evaluation_tigran.py --limit 5")
    print("python rag_evaluation_tigran.py --only-failed")
    print("python rag_evaluation_tigran.py --verbose")
    print("")


def load_dataset() -> list[dict[str, Any]]:
    """Загружает dataset из JSON."""
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize_text(text: str) -> str:
    """Нормализует текст для мягкого сравнения."""
    text = text.lower().replace("ё", "е")
    text = text.replace("₽", " рублей ")
    text = re.sub(r"[\.,:;!\?\(\)\[\]\"'«»/\\\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Разбивает текст на токены без коротких стоп-слов."""
    normalized = normalize_text(text)
    tokens = re.findall(r"[a-zа-я0-9]+", normalized)
    return [token for token in tokens if len(token) > 2 and token not in STOPWORDS]


def soft_match_score(expected: str, answer: str) -> float:
    """Считает мягкое совпадение фразы с ответом."""
    normalized_expected = normalize_text(expected)
    normalized_answer = normalize_text(answer)

    if not normalized_expected:
        return 1.0
    if normalized_expected in normalized_answer:
        return 1.0

    expected_tokens = set(tokenize(expected))
    answer_tokens = set(tokenize(answer))
    if not expected_tokens:
        return 0.0

    overlap = len(expected_tokens & answer_tokens) / len(expected_tokens)
    return min(1.0, overlap)


def contains_phrase(text: str, phrase: str) -> bool:
    """Проверяет наличие фразы после нормализации."""
    return normalize_text(phrase) in normalize_text(text)


def match_behavior(answer: str, behavior: str) -> bool:
    """Проверяет, выполнено ли желательное поведение."""
    markers = BEHAVIOR_MARKERS.get(behavior, [])
    normalized_answer = normalize_text(answer)
    return any(normalize_text(marker) in normalized_answer for marker in markers)


def detect_general_cta(answer: str) -> float:
    """Оценивает общий call-to-action, даже если он не совпал с точным шаблоном."""
    normalized_answer = normalize_text(answer)
    if any(marker in normalized_answer for marker in STRONG_CTA_MARKERS):
        return 1.0
    if any(marker in normalized_answer for marker in SOFT_CTA_MARKERS):
        return 0.5
    return 0.0


def score_facts(case: dict[str, Any], answer: str) -> tuple[float, list[str], list[str]]:
    """Оценивает передачу обязательных фактов и отсутствие запрещенных утверждений."""
    gold_facts = case.get("gold_facts", [])
    forbidden_claims = case.get("forbidden_claims", [])

    fact_scores = [soft_match_score(fact, answer) for fact in gold_facts]
    avg_fact_score = sum(fact_scores) / len(fact_scores) if fact_scores else 1.0

    matched_facts = [fact for fact, score in zip(gold_facts, fact_scores) if score >= 0.55]
    missing_facts = [fact for fact, score in zip(gold_facts, fact_scores) if score < 0.55]

    forbidden_hits = [claim for claim in forbidden_claims if contains_phrase(answer, claim)]
    penalty = 0.35 * len(forbidden_hits)
    total = max(0.0, min(1.0, avg_fact_score - penalty))

    comments: list[str] = []
    if matched_facts:
        comments.append("факты в целом переданы")
    if missing_facts:
        comments.append("не все ключевые факты отражены")
    if forbidden_hits:
        comments.append("есть недопустимое утверждение")
    return total, comments, missing_facts + forbidden_hits


def score_persona(case: dict[str, Any], answer: str) -> tuple[float, list[str], list[str]]:
    """Оценивает выдержанность persona живого мастера."""
    forbidden_phrases = case.get("forbidden_phrases", []) + GLOBAL_FORBIDDEN_PERSONA_PHRASES
    forbidden_hits = [phrase for phrase in forbidden_phrases if contains_phrase(answer, phrase)]
    normalized_answer = normalize_text(answer)

    has_first_person = any(marker in normalized_answer for marker in FIRST_PERSON_MARKERS)
    has_human_marker = any(marker in normalized_answer for marker in HUMAN_STYLE_MARKERS)

    score = 1.0
    if forbidden_hits:
        score -= 0.45 * len(set(forbidden_hits))
    if not has_first_person:
        score -= 0.35
    if not has_human_marker:
        score -= 0.1

    score = max(0.0, min(1.0, score))

    comments: list[str] = []
    issues: list[str] = []
    if has_first_person:
        comments.append("persona выдержана")
    else:
        comments.append("не хватает первого лица")
        issues.append("нет первого лица")
    if forbidden_hits:
        comments.append("бот палится как ассистент")
        issues.append("бот палится как ассистент")
    return score, comments, issues


def score_style(answer: str) -> tuple[float, list[str], list[str]]:
    """Оценивает живость и удобочитаемость ответа."""
    normalized_answer = normalize_text(answer)
    answer_length = len(answer.strip())
    sentence_count = max(1, len(re.findall(r"[.!?]+", answer)))
    dry_hits = [phrase for phrase in DRY_STYLE_PHRASES if phrase in normalized_answer]
    human_hits = [phrase for phrase in HUMAN_STYLE_MARKERS if phrase in normalized_answer]

    score = 1.0
    if answer_length > 550:
        score -= 0.35
    elif answer_length > 380:
        score -= 0.2
    if sentence_count > 6:
        score -= 0.15
    if dry_hits:
        score -= 0.2 * len(dry_hits)
    if human_hits:
        score += 0.1

    score = max(0.0, min(1.0, score))

    comments: list[str] = []
    issues: list[str] = []
    if score >= 0.8:
        comments.append("стиль звучит по-человечески")
    if answer_length > 380:
        comments.append("ответ немного длинноват")
        issues.append("ответ слишком длинный")
    if dry_hits:
        comments.append("ответ немного сухой")
        issues.append("слишком сухой ответ")
    return score, comments, issues


def score_behavior(case: dict[str, Any], answer: str) -> tuple[float, list[str], list[str]]:
    """Оценивает наличие полезного следующего действия, если оно уместно."""
    preferred_behaviors = case.get("preferred_behaviors", [])
    if not preferred_behaviors:
        return 1.0, ["поведение достаточное"], []

    matched = [behavior for behavior in preferred_behaviors if match_behavior(answer, behavior)]
    preferred_score = len(matched) / len(preferred_behaviors) if preferred_behaviors else 1.0
    general_cta_score = detect_general_cta(answer)

    if matched:
        score = max(preferred_score, general_cta_score, 0.75)
    else:
        score = general_cta_score

    comments: list[str] = []
    issues: list[str] = []
    if score >= 0.75:
        comments.append("есть полезное следующее действие")
    elif score >= 0.4:
        comments.append("есть слабый, но полезный следующий шаг")
    else:
        comments.append("не хватает полезного следующего шага")
        issues.append("не хватает CTA")
    return score, comments, issues


def calculate_total_score(fact_score: float, persona_score: float, style_score: float, behavior_score: float) -> float:
    """Собирает итоговый weighted score."""
    total = (
        fact_score * FACT_WEIGHT
        + persona_score * PERSONA_WEIGHT
        + style_score * STYLE_WEIGHT
        + behavior_score * BEHAVIOR_WEIGHT
    )
    return round(total, 4)


def format_score(score: float) -> str:
    """Форматирует score для терминала."""
    return f"{score:.2f}"


def build_comment(
    fact_comments: list[str],
    persona_comments: list[str],
    style_comments: list[str],
    behavior_comments: list[str],
) -> list[str]:
    """Собирает короткий комментарий по тесту."""
    seen: set[str] = set()
    comments: list[str] = []
    for group in [fact_comments, persona_comments, style_comments, behavior_comments]:
        for comment in group:
            if comment not in seen:
                seen.add(comment)
                comments.append(comment)
    return comments


def evaluate_case(case: dict[str, Any], pipeline: RAGPipeline, index: int) -> dict[str, Any]:
    """Прогоняет один тест и возвращает все scores."""
    pipeline.reset_history()
    result = pipeline.ask(case["question"])
    answer = result["answer"]

    fact_score, fact_comments, fact_issues = score_facts(case, answer)
    persona_score, persona_comments, persona_issues = score_persona(case, answer)
    style_score, style_comments, style_issues = score_style(answer)
    behavior_score, behavior_comments, behavior_issues = score_behavior(case, answer)
    total_score = calculate_total_score(fact_score, persona_score, style_score, behavior_score)

    comments = build_comment(fact_comments, persona_comments, style_comments, behavior_comments)
    issues = fact_issues + persona_issues + style_issues + behavior_issues

    return {
        "index": index,
        "question": case["question"],
        "answer": answer,
        "fact_score": fact_score,
        "persona_score": persona_score,
        "style_score": style_score,
        "behavior_score": behavior_score,
        "total_score": total_score,
        "comments": comments,
        "issues": issues,
        "contexts": result["contexts"],
        "notes": case.get("notes", ""),
    }


def print_test_result(result: dict[str, Any], verbose: bool) -> None:
    """Печатает результат одного теста."""
    print("--------------------------------------------------")
    print(f"Тест {result['index']}")
    print("--------------------------------------------------")
    print(f"Вопрос: {result['question']}")
    print(f"Ответ: {result['answer']}")
    print("")
    print(f"FACT: {format_score(result['fact_score'])}")
    print(f"PERSONA: {format_score(result['persona_score'])}")
    print(f"STYLE: {format_score(result['style_score'])}")
    print(f"BEHAVIOR: {format_score(result['behavior_score'])}")
    print(f"TOTAL: {format_score(result['total_score'])}")
    print("")
    print("Комментарий:")
    for comment in result["comments"]:
        print(f"- {comment}")

    if verbose:
        if result["notes"]:
            print(f"- notes: {result['notes']}")
        if result["issues"]:
            print("- проблемы:")
            for issue in result["issues"]:
                print(f"  - {issue}")
        if result["contexts"]:
            print("- retrieval:")
            for context in result["contexts"]:
                preview = " ".join(context["text"].split())
                if len(preview) > 120:
                    preview = f"{preview[:120].rstrip()}..."
                print(f"  - {context['source']}: {preview}")
    print("")


def average(values: list[float]) -> float:
    """Считает среднее значение."""
    return sum(values) / len(values) if values else 0.0


def print_summary(results: list[dict[str, Any]]) -> None:
    """Печатает итоговую сводку."""
    fact_avg = average([item["fact_score"] for item in results])
    persona_avg = average([item["persona_score"] for item in results])
    style_avg = average([item["style_score"] for item in results])
    behavior_avg = average([item["behavior_score"] for item in results])
    total_avg = average([item["total_score"] for item in results])
    weak_results = [item for item in results if item["total_score"] < FAILED_THRESHOLD]

    issues_counter: Counter[str] = Counter()
    for item in results:
        for issue in item["issues"]:
            issues_counter[issue] += 1

    print("==================================================")
    print("📈 SUMMARY")
    print("==================================================")
    print("")
    print(f"Всего тестов: {len(results)}")
    print(f"Средний total_score: {format_score(total_avg)}")
    print(f"Средний fact_score: {format_score(fact_avg)}")
    print(f"Средний persona_score: {format_score(persona_avg)}")
    print(f"Средний style_score: {format_score(style_avg)}")
    print(f"Средний behavior_score: {format_score(behavior_avg)}")
    print(f"Слабых ответов (total_score < {FAILED_THRESHOLD}): {len(weak_results)}")
    print("")
    print("Частые проблемы:")
    if not issues_counter:
        print("- явных типовых проблем не найдено")
    else:
        for issue, count in issues_counter.most_common(5):
            print(f"- {issue}: {count}")
    print("")
    print("💡 Полезные команды:")
    print("- python clear_cache.py — очистить только файловый кэш")
    print("- python clear_cache.py --all — очистить кэш и локальный индекс ChromaDB")
    print("- python app.py --debug-retrieval — заново проверить, какие чанки реально выбираются")
    print("- python rag_evaluation_tigran.py --limit 5 --verbose — быстрый подробный прогон на 5 тестах")
    print("")


def main() -> None:
    """Запускает persona-оценку."""
    args = parse_args()
    print_header()

    dataset = load_dataset()
    selected_cases = dataset[: args.limit] if args.limit else dataset

    print(f"Загружено тест-кейсов: {len(dataset)}")
    print(f"К запуску выбрано: {len(selected_cases)}")
    print("")

    pipeline = RAGPipeline(dialog_id="tigran_persona_eval")
    pipeline.reset_history()

    results: list[dict[str, Any]] = []
    for index, case in enumerate(selected_cases, start=1):
        print(f"[{index}/{len(selected_cases)}] Проверяю: {case['question']}")
        result = evaluate_case(case, pipeline, index)
        results.append(result)

    print("")
    shown_results = results
    if args.only_failed:
        shown_results = [item for item in results if item["total_score"] < FAILED_THRESHOLD]
        print(f"Фильтр only-failed включен: показываю {len(shown_results)} из {len(results)}")
        print("")

    for item in shown_results:
        print_test_result(item, verbose=args.verbose)

    print_summary(results)


if __name__ == "__main__":
    main()
