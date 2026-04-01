<h1 align="center">RAG Assistant</h1>

<p align="center">
  Локальный RAG-ассистент на Python для вопросов по базе знаний
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/OpenAI-gpt--4o--mini-10A37F?style=for-the-badge&logo=openai&logoColor=white" alt="OpenAI GPT-4o-mini">
  <img src="https://img.shields.io/badge/Embeddings-text--embedding--3--small-0A66C2?style=for-the-badge" alt="Embeddings">
  <img src="https://img.shields.io/badge/ChromaDB-Vector%20Store-F97316?style=for-the-badge" alt="ChromaDB">
  <img src="https://img.shields.io/badge/RAGAS-Evaluation-7C3AED?style=for-the-badge" alt="RAGAS">
  <img src="https://img.shields.io/badge/CLI-Friendly-22C55E?style=for-the-badge" alt="CLI Friendly">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Language-Русский-2563EB?style=flat-square" alt="Language">
  <img src="https://img.shields.io/badge/Retrieval-Debug%20Mode-14B8A6?style=flat-square" alt="Retrieval Debug">
  <img src="https://img.shields.io/badge/Cache-File%20Cache-F59E0B?style=flat-square" alt="Cache">
  <img src="https://img.shields.io/badge/GitHub-Ready-111827?style=flat-square&logo=github&logoColor=white" alt="GitHub Ready">
</p>

# Минимальный RAG-ассистент на Python

Это простой учебный проект RAG-ассистента. Он читает все `.txt`-файлы из папки `data/`, разбивает текст на чанки, сохраняет их в ChromaDB и отвечает на вопросы пользователя на основе найденного контекста.

Проект специально сделан без лишней архитектурной сложности, чтобы его было удобно разбирать новичку.

## Структура проекта

- `data/` - база знаний из нескольких `.txt`-файлов.
- `chroma_db/` - локальная база ChromaDB.
- `cache/` - файловый кэш индексации и повторных запросов.
- `config.py` - параметры чанков и поиска.
- `app.py` - консольный интерфейс.
- `rag_pipeline.py` - основной RAG pipeline.
- `vector_store.py` - индексация, embeddings, поиск и работа с ChromaDB.
- `cache_utils.py` - функции для JSON-кэша и хэшей.
- `test_questions.py` - тестовый набор вопросов и эталонных ответов.
- `rag_evaluation.py` - оценка качества через RAGAS.
- `requirements.txt` - зависимости проекта.
- `.env.example` - пример переменных окружения.

## Требования

- Python 3.11+
- OpenAI API key

## Как создать окружение

Рекомендуемый вариант:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Если `python3.11` пока не установлен, можно временно использовать:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Для VS Code в проект уже добавлены настройки, чтобы терминал автоматически активировал `.venv`.

## Как установить зависимости

После активации окружения выполните:

```bash
pip install -r requirements.txt
```

## Как создать .env

Создайте файл `.env` рядом с `.env.example` и добавьте ключ:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Файл `.env` не нужно загружать на GitHub. В репозитории должен оставаться только `.env.example`.

## Как запустить проект

Запуск консольного приложения:

```bash
python app.py                    # обычный запуск ассистента
python app.py --debug-retrieval  # запуск с подробным выводом retrieval
python clear_cache.py            # очистить файловый кэш
python clear_cache.py --all      # очистить кэш и локальный индекс
```

Для выхода используйте команду `exit`.

## Как подготовить проект к GitHub

В проект уже добавлен `.gitignore`, поэтому в репозиторий не попадут:

- `.env`
- `.venv/`
- `__pycache__/`
- локальный кэш из `cache/`
- локальная база `chroma_db/`

Базовые команды для первой публикации:

```bash
git init
git add .
git commit -m "Initial commit: local RAG assistant"
git branch -M main
git remote add origin https://github.com/USERNAME/REPOSITORY.git
git push -u origin main
```

Перед публикацией полезно проверить:

- в `.env` нет секретов, которые случайно попали в коммит
- в репозитории не лежат файлы из `cache/` и `chroma_db/`
- README отражает актуальные команды запуска

## Как проверить работу RAG

1. Убедитесь, что в `.env` задан `OPENAI_API_KEY`.
2. Запустите `python app.py`.
3. Задайте несколько вопросов по содержимому файлов в папке `data/`.
4. После каждого ответа приложение покажет найденные чанки, их источник и короткое превью.

## Параметры RAG

Параметры вынесены в файл `config.py`.

- `CHUNK_SIZE = 500` - максимальный размер чанка в символах.
- `CHUNK_MIN_SIZE = 100` - минимальный размер чанка. Если остаток текста слишком маленький, он приклеивается к предыдущему чанку.
- `CHUNK_OVERLAP = 0.2` - overlap между соседними чанками, то есть 20% или 100 символов. Это помогает не потерять фразу на границе двух фрагментов.
- `TOP_K = 3` - сколько наиболее релевантных чанков возвращается из ChromaDB при поиске.
- Параметр `TOP_K` определяет, сколько чанков извлекается из базы знаний перед генерацией ответа.

## Оценка качества (RAGAS с эталонными ответами)

Для оценки качества используется файл `test_questions.py`, где хранится `TEST_DATASET` с вопросами и `ground_truth`.

Запуск оценки:

```bash
python rag_evaluation.py
```

Скрипт:

- задает все тестовые вопросы через `RAGPipeline`
- собирает ответы и найденные контексты
- сравнивает их с эталонными ответами
- считает метрики `faithfulness`, `answer_relevancy`, `answer_correctness`, `semantic_similarity`
- печатает значения по каждому вопросу и средние значения

## CLI для оценки RAG

Файл `rag_evaluation.py` запускает человекочитаемую оценку качества RAG через терминал. Скрипт показывает пошаговый прогресс, список файлов базы знаний, статус индекса и кэша, результаты по каждому тесту и общий итог в конце.

Поддерживаемые параметры CLI:

- `--limit N` - ограничить количество тестов, например `--limit 5`
- `--show-context` - показывать найденные чанки и источники по каждому тесту
- `--verbose` - показывать расширенный вывод: вопрос, ответ, эталон, источники, чанки, метрики и комментарий
- `--only-failed` - показывать только слабые тесты, где основная доступная метрика ниже `0.7`
- `--fast` - запустить безопасный укороченный режим с метриками `faithfulness` и `answer_relevancy`
- `--save-report reports/last_eval.txt` - сохранить весь человекочитаемый отчет в текстовый файл

Как читать вывод:

- в начале скрипт показывает, что он собирается делать
- затем печатает шаги проверки базы знаний, индекса и тестов
- по каждому тесту выводятся вопрос, ответ RAG, эталонный ответ и метрики
- в конце показывается summary со средними значениями и советами по улучшению

Что означают основные метрики:

- `Faithfulness` - насколько ответ опирается на найденный контекст
- `Answer relevance` - насколько ответ вообще относится к вопросу

В обычном режиме CLI использует `AsyncOpenAI` и совместимый embeddings adapter, поэтому доступны и embedding-based метрики. Режим `--fast` оставляет только более короткий набор метрик для быстрого прогона.

Как интерпретировать итог:

- `Ответ хороший` - система нашла нужный контекст и ответ близок к эталону
- `Ответ частично корректный` - смысл близкий, но есть неточности или неполнота
- `Ответ слабый` - найден не тот контекст или ответ заметно отличается от эталона

## Пример запуска

```bash
python rag_evaluation.py                      # полный запуск оценки по всем тестам
python rag_evaluation.py --limit 5           # короткий прогон на первых 5 тестах
python rag_evaluation.py --verbose           # подробный вывод по каждому тесту
python rag_evaluation.py --show-context      # показать найденные чанки и источники
python rag_evaluation.py --verbose --show-context   # подробный прогон с retrieval
python rag_evaluation.py --only-failed       # показать только слабые тесты
python rag_evaluation.py --fast              # безопасный упрощённый режим
python clear_cache.py                        # очистить файловый кэш
python clear_cache.py --all                  # очистить кэш и локальный индекс
```

## Кэширование

В проекте есть простой файловый кэш в папке `cache/`.

Что кэшируется:

- общий хэш всех `.txt`-файлов из папки `data/`
- дата последней индексации
- количество чанков
- embedding вопроса
- уже полученный ответ
- найденные контексты

Зачем это нужно:

- не пересчитывать embeddings документов без необходимости
- не запрашивать повторно embedding одного и того же вопроса
- быстрее отвечать на одинаковые вопросы

Когда кэш пересоздается:

- если изменился хотя бы один `.txt`-файл в папке `data/`
- если коллекция ChromaDB отсутствует или пуста

Как быстро очистить кэш при тестировании:

```bash
python clear_cache.py
```

Если нужно очистить и файловый кэш, и локальный индекс ChromaDB:

```bash
python clear_cache.py --all
```

## Как теперь работает индексация

1. Приложение находит все `.txt`-файлы в папке `data/`.
2. Каждый файл читается отдельно.
3. Текст каждого файла разбивается на чанки скользящим окном.
4. Для каждого чанка в ChromaDB сохраняются:
   - текст чанка
   - `source` с именем файла
   - `chunk_index`
5. При поиске система возвращает не только текст, но и источник, из которого взят найденный фрагмент.
