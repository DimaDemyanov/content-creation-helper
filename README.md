# Content Creation Helper

Инструмент для сбора постов о яхтинге из Telegram-каналов и Instagram, хранения и поиска по ним через Telegram-бота.

---

## Контекст для Claude

Вставляй этот файл в начало нового чата, чтобы я сразу понял проект.

---

## Цель проекта

Собирать посты на тему яхтинга из Telegram-каналов и Instagram, хранить их локально и искать по текстам — для анализа и помощи в написании похожих постов.

---

## Ключевые решения

### Язык и платформа
- **Node.js** — основной язык
- Причина: опытный разработчик, привычный стек

### Источники данных
- **Telegram** — основной источник, официальный MTProto API через GramJS
- **Instagram** — через Apify (`apify/instagram-scraper`), Node.js SDK (`apify-client`)

### Хранилище
- **JSON-файлы** — локально, структура: `data/posts/`
- Причина: простота, без зависимостей от БД

Структура одного поста:
```json
{
  "id": "channel_name_12345",
  "source": "telegram",
  "channel": "yacht_russia",
  "url": "https://t.me/yacht_russia/12345",
  "date": "2026-03-20T14:32:00Z",
  "text": "Оригинальный текст с эмодзи 🌊 и #хэштегами",
  "textClean": "Очищенный текст для индексации",
  "ocrText": "Текст извлечённый с картинок",
  "media": [{ "type": "photo" }],
  "stats": { "views": 4200, "likes": 142, "comments": 18 },
  "collectedAt": "2026-03-27T08:00:00Z"
}
```

Структура `data/state.json` (per-channel):
```json
{
  "seapinta": {
    "source": "telegram",
    "lastCollectedAt": "2026-03-27T09:00:00Z",
    "allPostsDownloaded": true,
    "totalPosts": 342,
    "downloadedDateFrom": "2023-01-15T00:00:00Z",
    "downloadedDateTo": "2026-03-27T00:00:00Z",
    "channelTotalPosts": 850,
    "channelDateFrom": "2020-05-01T00:00:00Z",
    "channelDateTo": "2026-03-27T00:00:00Z"
  }
}
```

### Сбор данных
- **Расписание** — автоматически 1 раз в день через `node-cron`
- **Первый запуск:** Telegram — все посты (бесплатно), Instagram — последние 500 постов на канал (~$0.12 на канал)
- **Ежедневный сбор** — только новые посты с момента `lastCollectedAt` (дельта)
- **Сохранение батчами** — Telegram каждые 100 постов, Instagram каждые 50, прогресс не теряется при падении
- Состояние хранится в `data/state.json` отдельно для каждого канала
- **Задержка между OCR** — 1.5 сек между постами при сборе Instagram, чтобы не превышать TPM-лимит OpenAI

### Поиск

Рекомендуемый метод: **hybridSearchRRF** — объединяет три сигнала через Reciprocal Rank Fusion.

| Метод | Описание |
|---|---|
| `search()` | BM25 + расширение синонимами |
| `vectorSearch()` | Косинусное сходство по эмбеддингу переписанного запроса |
| `hybridSearch()` | `0.3×BM25 + 0.7×vector` |
| `vectorSearchHyDE()` | Векторный поиск по гипотетическому документу (HyDE) |
| `hybridSearchRRF()` | **RRF по трём ранкингам: BM25 + vector + HyDE** |

- **BM25** — `wink-bm25-text-search` с русским стеммером (`@nlpjs/lang-ru`) и русскими стоп-словами. "Мель" и "мели" дают один токен.
- **Расширение запроса** — `gpt-4o-mini` генерирует синонимы перед BM25-поиском
- **Query rewriting** — перед векторным поиском `gpt-4o-mini` преобразует вопрос в ключевые понятия ("Как выйти замуж на яхте" → "свадьба на яхте, венчание")
- **HyDE** — GPT генерирует гипотетический пост, его эмбеддинг сравнивается с корпусом. Закрывает лексический разрыв между запросом и постами.
- **RRF** — `Σ 1/(60+rank)` по трём сигналам, без нормализации скоров
- Эмбеддинги: `text-embedding-3-small` (1536 измерений), ~67 MB для 11k постов
- Кэш синонимов / HyDE / rewrite в памяти процесса
- Доступ через **Telegram-бота** (не веб-интерфейс)

### Обработка картинок (OCR)
- **OpenAI Vision** (`gpt-4o-mini`) — извлечение текста с изображений через API
- Логика: скачиваем фото в память → отправляем как base64 в OpenAI Vision → `ocrText` сохраняется в пост
- Если текста нет — GPT отвечает «нет» и `ocrText = null`
- Поддерживает Telegram (buffer → base64) и Instagram (публичный URL → base64)
- Пропускает изображения > 3.9 MB
- **Rate limit** — при 429 экспоненциальный backoff: 1s → 2s → 4s → 8s → 16s → 32s → 60s → ...

### AI-интеграция
- **OpenAI API** (`gpt-4o-mini`) — используется в пяти местах:
  1. Расширение поисковых запросов синонимами (перед BM25-поиском)
  2. Query rewriting — преобразование вопроса в ключевые понятия (перед эмбеддингом)
  3. HyDE — генерация гипотетического поста (для vectorSearchHyDE / hybridSearchRRF)
  4. Извлечение текста с картинок (Vision API при сборе)
  5. Генерация постов по запросу пользователя

### Поиск — детали реализации
- Токенизатор использует Unicode regex `[\p{L}\p{N}]+` — корректно обрабатывает кириллицу
- Стемминг через `StemmerRu` (`@nlpjs/lang-ru`) — русские падежи/формы приводятся к основе ("мели" → "мел", "яхтой" → "яхт")
- Все кэши (синонимы, HyDE, rewrite) живут в памяти процесса — при рестарте сбрасываются

### Известные ограничения
- **Dense retrieval** — один вектор на пост усредняет всё содержимое. Chunking реализован, эффект минимален для коротких постов (≤60 слов).
- **Instagram CDN-ссылки** — URL картинок имеют срок жизни (~24-48ч). OCR-бэкфилл нужно запускать вскоре после сбора.
- **Память** — 11k × 1536 float32 ≈ 67 MB. При 100k+ постах нужна векторная БД (Qdrant, Chroma).

---

## Архитектура (3 модуля)

```
Telegram каналы       Instagram аккаунты
      │                        │
      │                 [Apify Scraper]
      │                        │
      ▼                        ▼
[Модуль 1: Collector]  ←── Планировщик (node-cron / ручной запуск)
  GramJS / MTProto + apify-client
  Сохранение батчами + per-channel state
      │
      ▼
[JSON-файлы]  data/posts/*.json + data/state.json
      │
      ▼
[Модуль 2: Search]
  BM25 + лемматизация + расширение запроса через OpenAI
      │
      ▼
[Модуль 3: Telegram Bot]  ←──▶  Пользователь
  node-telegram-bot-api
      │
      ▼
[OpenAI API]
  Синонимы для поиска + OCR Vision + генерация постов
```

---

## Статус модулей

| Модуль | Статус | Описание |
|--------|--------|----------|
| Collector (Telegram) | ✅ Готов | GramJS, батчи по 100, per-channel state |
| Collector (Instagram) | ✅ Готов | Apify, батчи по 50, per-channel state |
| Search | ✅ Готов | BM25 (рус. стеммер) + Vector + HyDE + RRF |
| Bot | ✅ Готов | Команды: search, generate, collect, status, addchannel |

---

## Команды бота

| Команда | Описание |
|---------|----------|
| `/search <запрос>` | Найти посты по теме (BM25 + синонимы OpenAI) |
| `/generate <запрос>` | Найти посты и сгенерировать похожий текст |
| `/collect` | Запустить сбор новых постов вручную |
| `/status` | Статистика: скачано/всего постов, диапазоны дат, архив |
| `/addchannel telegram <username>` | Добавить Telegram-канал и запустить первичный сбор |
| `/addchannel instagram <username>` | Добавить Instagram-аккаунт и запустить первичный сбор |
| `/removechannel telegram <username>` | Удалить канал, все его посты и эмбеддинги |
| `/removechannel instagram <username>` | Удалить аккаунт, все его посты и эмбеддинги |

---

## Необходимые credentials

| Ключ | Где получить |
|------|-------------|
| `TELEGRAM_API_ID` | my.telegram.org |
| `TELEGRAM_API_HASH` | my.telegram.org |
| `TELEGRAM_BOT_TOKEN` | @BotFather |
| `OPENAI_API_KEY` | platform.openai.com |
| `APIFY_API_TOKEN` | console.apify.com |

---

## Структура проекта

```
content-creation-helper/
├── README.md
├── .env                        # credentials (не в git!)
├── config.json                 # список каналов и настройки
├── auth/
│   └── telegram.js             # одноразовая авторизация GramJS
├── collector/
│   ├── index.js                # оркестратор + cron
│   ├── telegram.js             # GramJS сбор
│   └── instagram.js            # Apify сбор
├── search/
│   ├── index.js                # BM25 + vector + HyDE + RRF
│   ├── embeddings.js           # генерация и хранение эмбеддингов
│   └── chunking.js             # разбивка постов на чанки
├── bot/
│   └── index.js                # Telegram бот
├── utils/
│   ├── textClean.js            # очистка текста от эмодзи/хэштегов
│   └── ocr.js                  # OpenAI Vision OCR с backoff
├── scripts/
│   ├── backfill-ocr.js         # обновление ocrText для существующих постов
│   ├── backfill-embeddings.js  # генерация full-post и chunk-эмбеддингов
│   └── create-fixtures.js      # создание тестовых фикстур (50 постов/канал)
├── docs/
│   └── vector-search.md        # план реализации векторного поиска
├── tests/
│   ├── textClean.test.js
│   ├── search.test.js
│   └── collector.test.js
└── data/
    ├── state.json              # per-channel состояние сбора
    ├── posts/                  # JSON файлы с постами по каналам
    ├── embeddings/             # full-post эмбеддинги по каналам
    └── chunk-embeddings/       # chunk-эмбеддинги по каналам
```

---

## Запуск

```bash
# 1. Установить зависимости
npm install

# 2. Авторизоваться в Telegram (один раз)
node auth/telegram.js

# 3. Запустить бота
node bot/index.js

# 4. Запустить сбор вручную (опционально)
node collector/index.js --once

# 5. Обновить OCR для существующих постов
node --env-file=.env scripts/backfill-ocr.js --source instagram
node --env-file=.env scripts/backfill-ocr.js --source telegram
node --env-file=.env scripts/backfill-ocr.js --channel seapinta  # один канал

# 6. Сгенерировать эмбеддинги (для vectorSearch / hybridSearchRRF)
node --env-file=.env scripts/backfill-embeddings.js
node --env-file=.env scripts/backfill-embeddings.js --chunked  # опционально

# 7. Запустить тесты
npm test                    # юнит-тесты
npm run test:integration    # интеграционные (требует OPENAI_API_KEY + эмбеддинги)
```
