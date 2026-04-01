"""Простой RAG pipeline для консольного ассистента."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from vector_store import VectorStore


CHAT_MODEL = "gpt-4o-mini"
SYSTEM_MESSAGE = """
Ты ассистент сервиса «Мастер на час» в Санкт-Петербурге.

Твоя задача — отвечать клиенту только на основе найденного контекста из базы знаний.

Правила ответа:
- Не придумывай факты, цены, условия и услуги.
- Если в контексте нет точного ответа, честно скажи: "В базе знаний нет точного ответа на этот вопрос."
- Отвечай понятным, простым и вежливым русским языком.
- Отвечай кратко, но не опускай важные уточнения из контекста.
- Не добавляй информацию, которая не относится к вопросу.
- Если вопрос клиента слишком общий, а в контексте есть несколько близких вариантов, перечисли их коротко и скажи, что точная стоимость зависит от конкретной задачи.
- Если вопрос о цене, используй только цены из контекста и указывай их как "от ... ₽".
- Если для точного ответа нужно уточнение, в конце коротко предложи уточнить проблему, объём работ или прислать фото.
- Не используй формулировки вроде "возможно" или "наверное", если этого нет в контексте.
- Не пиши длинные вступления и лишнюю вежливость.
""".strip()


class RAGPipeline:
    """Связывает поиск контекста, сбор prompt и генерацию ответа."""

    def __init__(self) -> None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Не найден OPENAI_API_KEY. Создайте .env по примеру из .env.example.")

        self.openai_client = OpenAI(api_key=api_key)
        self.vector_store = VectorStore(self.openai_client)
        self.index_status = self.vector_store.ensure_index()

    def build_prompt(self, question: str, contexts: list[dict[str, Any]]) -> str:
        """Собирает prompt для модели с найденным контекстом."""
        context_parts = []
        for context in contexts:
            context_parts.append(
                f"Источник: {context['source']}\nФрагмент: {context['text']}"
            )
        context_block = "\n\n---\n\n".join(context_parts) if context_parts else "Контекст не найден."

        return f"""
Используй только контекст ниже.

Как отвечать:
1. Сначала дай прямой ответ на вопрос клиента.
2. Если вопрос о стоимости и в контексте есть несколько близких услуг, перечисли 1–3 самых релевантных варианта с ценами.
3. Если в найденном контексте есть важные уточнения, не опускай их.
4. Старайся давать полный, но короткий ответ без лишней информации.
5. Не добавляй информацию, если она не относится напрямую к вопросу.
6. Если точный ответ зависит от деталей, коротко укажи это.
7. Если информации в контексте недостаточно, скажи: "В базе знаний нет точного ответа на этот вопрос."
8. Не выдумывай и не додумывай ничего сверх контекста.

Контекст:
{context_block}

Вопрос клиента:
{question}
""".strip()

    def ask(self, question: str) -> dict[str, Any]:
        """Возвращает ответ модели и найденные чанки."""
        cached_result = self.vector_store.get_cached_answer(question)
        if cached_result:
            return cached_result

        contexts = self.vector_store.search(question)
        prompt = self.build_prompt(question, contexts)

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
            temperature=0,
        )
        answer = response.choices[0].message.content.strip()

        self.vector_store.save_query_result(question, answer, contexts)
        return {
            "answer": answer,
            "contexts": contexts,
        }
