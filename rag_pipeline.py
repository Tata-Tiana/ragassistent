"""Простой RAG pipeline для консольного ассистента."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from config import ENABLE_RERANKER, HISTORY_TURNS, WARM_LEAD_HISTORY_WINDOW
from cache_utils import clear_dialog_history, load_dialog_history, save_dialog_history
from vector_store import VectorStore


CHAT_MODEL = "gpt-4o-mini"
WARM_LEAD_KEYWORDS = [
    "сколько стоит",
    "цена",
    "цене",
    "стоимость",
    "сориентируете",
    "сориентировать",
    "по цене",
    "когда сможете",
    "когда можете",
    "когда подъехать",
    "можно сегодня",
    "можно завтра",
    "выезд",
    "замер",
    "созвон",
    "договоримся",
    "куда прислать фото",
    "пришлю фото",
    "пришлите номер",
    "можете сделать",
    "сделаете",
    "беретесь",
]
SYSTEM_MESSAGE = """
Ты — Тигран, частный мастер по ремонту квартир с опытом более 20 лет.

Ты работаешь сам, без посредников и компаний. Общайся с клиентами напрямую, как живой человек в переписке на Авито или в мессенджере.

Твоя задача:
- вызвать доверие;
- показать опыт;
- довести человека до уточнения деталей, фото, созвона или замера.

Правила ответа:
- Не говори, что ты ассистент, бот или ИИ.
- Всегда отвечай от первого лица: "сделаю", "посмотрю", "подскажу", "нужно уточнить".
- Пиши просто, по-человечески, без канцелярита и длинных текстов.
- Не придумывай факты, цены, условия и услуги, которых нет в контексте.
- Если вопрос про цену, сначала дай ориентир, если он есть в контексте, а потом коротко скажи, от чего зависит итог.
- Если вопрос слишком общий, можешь перечислить 1–3 самых релевантных варианта и сказать, что точнее скажешь после уточнения.
- Если по задаче полезно фото, попроси фото.
- Если информации недостаточно, честно скажи, что нужно уточнить, и предложи следующий шаг.
- Если работа не подходит под формат "мастер на час", скажи это по-человечески и аккуратно объясни, что там уже нужен другой формат работ.
- Не перегружай ответ, не перечисляй всё подряд и не используй сложные термины без необходимости.
- Не используй фразы вроде "согласно базе знаний", "в контексте указано", "как ассистент".
""".strip()


class RAGPipeline:
    """Связывает поиск контекста, сбор prompt и генерацию ответа."""

    def __init__(self, dialog_id: str = "default") -> None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Не найден OPENAI_API_KEY. Создайте .env по примеру из .env.example.")

        self.dialog_id = dialog_id
        self.openai_client = OpenAI(api_key=api_key)
        self.vector_store = VectorStore(self.openai_client)
        self.index_status = self.vector_store.ensure_index()
        self.dialog_history = load_dialog_history(self.dialog_id)

    def reset_history(self) -> None:
        """Очищает историю текущего диалога."""
        self.dialog_history = []
        clear_dialog_history(self.dialog_id)

    def add_to_history(self, role: str, content: str) -> None:
        """Сохраняет последние сообщения диалога в памяти сессии."""
        self.dialog_history.append({"role": role, "content": content.strip()})
        max_messages = HISTORY_TURNS * 2
        self.dialog_history = self.dialog_history[-max_messages:]
        save_dialog_history(self.dialog_id, self.dialog_history)

    def build_history_block(self) -> str:
        """Собирает короткий блок недавней истории для prompt."""
        if not self.dialog_history:
            return "История диалога пока пустая."

        role_labels = {
            "user": "Клиент",
            "assistant": "Ты",
        }
        history_lines = []
        for item in self.dialog_history:
            label = role_labels.get(item["role"], item["role"])
            history_lines.append(f"{label}: {item['content']}")
        return "\n".join(history_lines)

    def build_search_question(self, question: str) -> str:
        """Расширяет поисковый запрос недавними вопросами клиента."""
        recent_user_messages = [
            item["content"]
            for item in self.dialog_history
            if item["role"] == "user"
        ][-2:]

        if not recent_user_messages:
            return question

        history_part = "\n".join(recent_user_messages)
        return f"Предыдущие вопросы клиента:\n{history_part}\n\nТекущий вопрос:\n{question}"

    def is_warm_lead(self, question: str) -> bool:
        """Понимает, что клиент уже близок к следующему шагу."""
        recent_user_messages = [
            item["content"].lower()
            for item in self.dialog_history
            if item["role"] == "user"
        ][-WARM_LEAD_HISTORY_WINDOW:]

        combined_text = " ".join([*recent_user_messages, question.lower()])
        return any(keyword in combined_text for keyword in WARM_LEAD_KEYWORDS)

    def build_mode_block(self, is_warm_lead: bool) -> str:
        """Добавляет в prompt режим ответа: обычный или тёплый лид."""
        if is_warm_lead:
            return """
Режим ответа: тёплый клиент.

Если человек уже близок к заказу, отвечай особенно практично:
- после короткого ответа мягко предложи следующий шаг;
- можно предложить прислать фото, уточнить детали, созвониться или договориться на выезд;
- не дави и не продавай агрессивно;
- не обещай того, чего нет в контексте.
""".strip()

        return """
Режим ответа: обычный.

Просто ответь по делу и помоги клиенту сориентироваться.
""".strip()

    def build_prompt(self, question: str, contexts: list[dict[str, Any]], is_warm_lead: bool) -> str:
        """Собирает prompt для модели с найденным контекстом."""
        context_parts = []
        for context in contexts:
            context_parts.append(
                f"Источник: {context['source']}\nФрагмент: {context['text']}"
            )
        context_block = "\n\n---\n\n".join(context_parts) if context_parts else "Контекст не найден."
        history_block = self.build_history_block()
        mode_block = self.build_mode_block(is_warm_lead)

        return f"""
Используй только контекст ниже.

Недавняя история диалога:
{history_block}

{mode_block}

Как отвечать:
1. Сначала дай прямой ответ на вопрос клиента.
2. Если вопрос о стоимости и в контексте есть несколько близких услуг, перечисли 1–3 самых релевантных варианта с ценами.
3. Если в найденном контексте есть важные уточнения, не опускай их.
4. Старайся давать полный, но короткий ответ без лишней информации.
5. Не добавляй информацию, если она не относится напрямую к вопросу.
6. Если точный ответ зависит от деталей, коротко укажи это.
7. Если по вопросу полезно фото или уточнение, предложи прислать фото, описать проблему, созвониться или договориться на замер.
8. Не выдумывай и не додумывай ничего сверх контекста.

Отвечай максимально точно по найденному контексту.

Важно:
- Если в контексте есть конкретный список или перечисление, обязательно используй его в ответе.
- Не заменяй конкретные пункты общими словами.
- Если ответ состоит из нескольких частей из разных фрагментов контекста, объедини их в один ответ.
- Не упрощай ответ, если в контексте есть более точная формулировка.
- Не добавляй свои пояснения или выводы, если их нет в контексте.
- Если человек близок к заказу, мягко предложи следующий шаг: фото, уточнение, созвон или выезд.
- Для ценовых вопросов используй естественные формулировки вроде "обычно такие работы от ... ₽", только если такая цена есть в контексте.
- Если текущий вопрос короткий или уточняющий, учитывай недавнюю историю диалога, чтобы понять, о чём речь.
- Если задача не подходит под формат "мастер на час", не отвечай резко. Коротко объясни, что это уже другой формат работ, и не обещай то, чего нет в контексте.
- Для задач "не твой формат" используй спокойные формулировки вроде "Это уже не формат мастер на час" или "Такую работу лучше делать бригадой", только если это подтверждается контекстом.

Если в контексте нет прямого ответа, но есть связанная информация, сначала честно скажи, что точно с ходу не скажешь, а затем добавь только релевантные факты из контекста.
Если в контексте совсем не хватает информации, честно скажи, что нужно уточнить детали.

Контекст:
{context_block}

Вопрос клиента:
{question}
""".strip()

    def ask(self, question: str) -> dict[str, Any]:
        """Возвращает ответ модели и найденные чанки."""
        query_type = self.vector_store.detect_query_type(question)
        doc_types_filter = self.vector_store.doc_types_for_query(query_type)

        # Кэш готовых ответов безопасно использовать только без истории диалога.
        # Иначе одна и та же фраза может означать разное в разных переписках.
        if not self.dialog_history:
            cached_result = self.vector_store.get_cached_answer(question)
            if cached_result:
                self.add_to_history("user", question)
                self.add_to_history("assistant", cached_result["answer"])
                cached_result["warm_lead"] = self.is_warm_lead(question)
                cached_result["query_type"] = query_type
                cached_result["doc_types_filter"] = doc_types_filter
                cached_result["reranker_enabled"] = ENABLE_RERANKER
                return cached_result

        is_warm_lead = self.is_warm_lead(question)
        search_question = self.build_search_question(question)
        contexts = self.vector_store.search(
            question=search_question,
            query_type=query_type,
            original_question=question,
        )
        prompt = self.build_prompt(question, contexts, is_warm_lead)

        response = self.openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_MESSAGE,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            # Низкая температура используется для RAG, чтобы уменьшить
            # генерацию лишней информации и повысить точность ответа.
            temperature=0,
        )
        answer = response.choices[0].message.content.strip()

        if not self.dialog_history:
            self.vector_store.save_query_result(question, answer, contexts)

        self.add_to_history("user", question)
        self.add_to_history("assistant", answer)
        return {
            "answer": answer,
            "contexts": contexts,
            "warm_lead": is_warm_lead,
            "query_type": query_type,
            "doc_types_filter": doc_types_filter,
            "reranker_enabled": ENABLE_RERANKER,
        }
