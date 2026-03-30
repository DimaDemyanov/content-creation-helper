import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import winkBM25 from 'wink-bm25-text-search';
import winkNLP from 'wink-nlp-utils';
import OpenAI from 'openai';
import { StemmerRu, StopwordsRu } from '@nlpjs/lang-ru';
import { generateEmbedding, loadAllEmbeddings, loadAllChunkEmbeddings, cosineSimilarity } from './embeddings.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_POSTS_DIR = path.join(__dirname, '../data/posts');
const QUERY_CACHE_DIR = path.join(__dirname, '../.cache/queries');

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Русский стеммер и стоп-слова
const ruStemmer = new StemmerRu();
const ruStopwordsDict = new StopwordsRu().dictionary;

// In-memory кэши (работают внутри процесса)
const synonymsCache = new Map();
const paraphrasesCache = new Map();
const hydeCache = new Map();
const rewriteCache = new Map();

// Диск-кэш для дорогих GPT-вызовов (паrafrazы, HyDE, rewrite, synonyms)
// Ключ: тип + запрос. Позволяет переиспользовать между запусками.
async function diskCacheGet(type, query) {
  try {
    const file = path.join(QUERY_CACHE_DIR, `${type}_${Buffer.from(query).toString('base64url').slice(0, 64)}.json`);
    const raw = await fs.readFile(file, 'utf-8');
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

async function diskCacheSet(type, query, value) {
  try {
    await fs.mkdir(QUERY_CACHE_DIR, { recursive: true });
    const file = path.join(QUERY_CACHE_DIR, `${type}_${Buffer.from(query).toString('base64url').slice(0, 64)}.json`);
    await fs.writeFile(file, JSON.stringify(value));
  } catch {}
}

export async function search(query, topK = 10, { postsDir } = {}) {
  const allPosts = await loadAllPosts(postsDir);
  if (allPosts.length === 0) return [];

  const expandedTerms = await expandQuery(query);
  const expandedQuery = expandedTerms.join(' ');

  const engine = buildIndex(allPosts);
  const results = engine.search(expandedQuery);

  return results
    .slice(0, topK)
    .map(([id, score]) => {
      const post = allPosts.find(p => p.id === id);
      return { ...post, score };
    });
}

export async function vectorSearch(query, topK = 10, { postsDir, embeddingsDir } = {}) {
  const allPosts = await loadAllPosts(postsDir);
  if (allPosts.length === 0) return [];

  const embeddings = await loadAllEmbeddings(embeddingsDir);
  if (embeddings.size === 0) return [];

  const rewritten = await rewriteQueryForEmbedding(query);
  const queryVector = await generateEmbedding(rewritten);

  const scored = allPosts
    .filter(p => embeddings.has(p.id))
    .map(p => ({ post: p, score: cosineSimilarity(queryVector, embeddings.get(p.id)) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);

  return scored.map(({ post, score }) => ({ ...post, score }));
}

/**
 * Векторный поиск по чанкам.
 *
 * Каждый пост разбит на чанки (фрагменты) при генерации эмбеддингов.
 * Поиск находит наиболее похожие чанки, затем возвращает уникальные посты
 * по наилучшему чанку. Это позволяет находить посты, где только часть текста
 * релевантна запросу — без "разбавления" вектором остального контента.
 */
export async function vectorSearchChunked(query, topK = 10, { postsDir, chunkEmbeddingsDir } = {}) {
  const allPosts = await loadAllPosts(postsDir);
  if (allPosts.length === 0) return [];

  const chunkMap = await loadAllChunkEmbeddings(chunkEmbeddingsDir);
  if (chunkMap.size === 0) return [];

  const rewritten = await rewriteQueryForEmbedding(query);
  const queryVector = await generateEmbedding(rewritten);
  const postMap = new Map(allPosts.map(p => [p.id, p]));

  // Для каждого поста берём максимальный скор среди всех его чанков
  const scored = [];
  for (const [postId, chunks] of chunkMap) {
    if (!postMap.has(postId)) continue;
    const bestScore = Math.max(...chunks.map(c => cosineSimilarity(queryVector, c.vector)));
    scored.push({ post: postMap.get(postId), score: bestScore });
  }

  return scored
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
    .map(({ post, score }) => ({ ...post, score }));
}

/**
 * Гибридный поиск с чанкингом.
 * BM25 по полному тексту + векторный поиск по чанкам.
 */
export async function hybridSearchChunked(query, topK = 10, { postsDir, chunkEmbeddingsDir, alpha = 0.3 } = {}) {
  const allPosts = await loadAllPosts(postsDir);
  if (allPosts.length === 0) return [];

  const chunkMap = await loadAllChunkEmbeddings(chunkEmbeddingsDir);

  const expandedTerms = await expandQuery(query);
  const expandedQuery = expandedTerms.join(' ');

  // BM25
  const engine = buildIndex(allPosts);
  const bm25Raw = engine.search(expandedQuery);
  const bm25Max = bm25Raw.length > 0 ? bm25Raw[0][1] : 1;
  const bm25Map = new Map(bm25Raw.map(([id, score]) => [id, score / bm25Max]));

  // Vector (по чанкам)
  let vectorMap = new Map();
  if (chunkMap.size > 0) {
    const rewritten = await rewriteQueryForEmbedding(query);
    const queryVector = await generateEmbedding(rewritten);
    for (const [postId, chunks] of chunkMap) {
      // cosine similarity диапазон -1..1, нормализуем в 0..1
      const best = Math.max(...chunks.map(c => cosineSimilarity(queryVector, c.vector)));
      vectorMap.set(postId, (best + 1) / 2);
    }
  }

  const postMap = new Map(allPosts.map(p => [p.id, p]));
  const allIds = new Set([...bm25Map.keys(), ...vectorMap.keys()]);

  return Array.from(allIds)
    .map(id => {
      const bm25 = bm25Map.get(id) ?? 0;
      const vec = vectorMap.get(id) ?? 0;
      return { id, score: alpha * bm25 + (1 - alpha) * vec };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
    .map(({ id, score }) => ({ ...postMap.get(id), score }));
}

export async function hybridSearch(query, topK = 10, { postsDir, embeddingsDir, alpha = 0.3 } = {}) {
  const allPosts = await loadAllPosts(postsDir);
  if (allPosts.length === 0) return [];

  const embeddings = await loadAllEmbeddings(embeddingsDir);

  const expandedTerms = await expandQuery(query);
  const expandedQuery = expandedTerms.join(' ');

  // BM25
  const engine = buildIndex(allPosts);
  const bm25Raw = engine.search(expandedQuery);

  // Нормализуем BM25: максимальный скор = 1
  const bm25Max = bm25Raw.length > 0 ? bm25Raw[0][1] : 1;
  const bm25Map = new Map(bm25Raw.map(([id, score]) => [id, score / bm25Max]));

  // Vector
  let vectorMap = new Map();
  if (embeddings.size > 0) {
    const rewritten = await rewriteQueryForEmbedding(query);
    const queryVector = await generateEmbedding(rewritten);
    for (const post of allPosts) {
      if (embeddings.has(post.id)) {
        // cosine similarity диапазон -1..1, нормализуем в 0..1
        const sim = (cosineSimilarity(queryVector, embeddings.get(post.id)) + 1) / 2;
        vectorMap.set(post.id, sim);
      }
    }
  }

  // Гибридный скор
  const postMap = new Map(allPosts.map(p => [p.id, p]));
  const allIds = new Set([...bm25Map.keys(), ...vectorMap.keys()]);

  const scored = Array.from(allIds)
    .map(id => {
      const bm25 = bm25Map.get(id) ?? 0;
      const vec = vectorMap.get(id) ?? 0;
      const score = alpha * bm25 + (1 - alpha) * vec;
      return { id, score };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);

  return scored.map(({ id, score }) => ({ ...postMap.get(id), score }));
}

/**
 * HyDE (Hypothetical Document Embeddings).
 *
 * Проблема обычного векторного поиска: эмбеддинг запроса "Как выйти замуж на яхте"
 * семантически далёк от поста "устраиваю свадьбы и венчания" — разные фреймы,
 * разные роли (спрашивающий vs организатор), разный стиль.
 *
 * Идея HyDE: вместо эмбеддинга запроса — просим GPT написать гипотетический пост
 * который отвечает на этот запрос, и эмбеддим его. Гипотетический пост уже содержит
 * "свадьба", "яхта", "море" в том же стиле что и реальные посты — его вектор
 * гораздо ближе к релевантным документам.
 *
 * Ограничение: +1 вызов GPT на запрос (gpt-4o-mini, ~$0.0001), кэшируется.
 */
export async function vectorSearchHyDE(query, topK = 10, { postsDir, embeddingsDir } = {}) {
  const allPosts = await loadAllPosts(postsDir);
  if (allPosts.length === 0) return [];

  const embeddings = await loadAllEmbeddings(embeddingsDir);
  if (embeddings.size === 0) return [];

  const hypothetical = await generateHypotheticalDocument(query);
  const queryVector = await generateEmbedding(hypothetical);

  return allPosts
    .filter(p => embeddings.has(p.id))
    .map(p => ({ post: p, score: cosineSimilarity(queryVector, embeddings.get(p.id)) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
    .map(({ post, score }) => ({ ...post, score }));
}

/**
 * Гибридный HyDE: BM25 по оригинальному запросу + векторный поиск по гипотетическому документу.
 */
export async function hybridSearchHyDE(query, topK = 10, { postsDir, embeddingsDir, alpha = 0.3 } = {}) {
  const allPosts = await loadAllPosts(postsDir);
  if (allPosts.length === 0) return [];

  const embeddings = await loadAllEmbeddings(embeddingsDir);

  // BM25 по оригинальному запросу (с расширением синонимами)
  const expandedTerms = await expandQuery(query);
  const engine = buildIndex(allPosts);
  const bm25Raw = engine.search(expandedTerms.join(' '));
  const bm25Max = bm25Raw.length > 0 ? bm25Raw[0][1] : 1;
  const bm25Map = new Map(bm25Raw.map(([id, score]) => [id, score / bm25Max]));

  // Векторный поиск по гипотетическому документу
  let vectorMap = new Map();
  if (embeddings.size > 0) {
    const hypothetical = await generateHypotheticalDocument(query);
    const queryVector = await generateEmbedding(hypothetical);
    for (const post of allPosts) {
      if (embeddings.has(post.id)) {
        const sim = (cosineSimilarity(queryVector, embeddings.get(post.id)) + 1) / 2;
        vectorMap.set(post.id, sim);
      }
    }
  }

  const postMap = new Map(allPosts.map(p => [p.id, p]));
  const allIds = new Set([...bm25Map.keys(), ...vectorMap.keys()]);

  return Array.from(allIds)
    .map(id => ({
      id,
      score: alpha * (bm25Map.get(id) ?? 0) + (1 - alpha) * (vectorMap.get(id) ?? 0),
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
    .map(({ id, score }) => ({ ...postMap.get(id), score }));
}

/**
 * Трёхсигнальный гибридный поиск: BM25 + стандартный вектор + HyDE вектор.
 *
 * Почему три сигнала:
 * - BM25 точен для конкретных терминов ("предпродажа", "регата")
 * - Vector (прямой эмбеддинг запроса) хорош для общей семантики
 * - HyDE устраняет лексический/фреймовый разрыв (разные роли, разный стиль)
 *
 * По умолчанию alpha=0.2 (BM25), beta=0.4 (vector), gamma=0.4 (HyDE).
 */
export async function hybridSearchFull(query, topK = 10, { postsDir, embeddingsDir, alpha = 0.2, beta = 0.4, gamma = 0.4 } = {}) {
  const allPosts = await loadAllPosts(postsDir);
  if (allPosts.length === 0) return [];

  const embeddings = await loadAllEmbeddings(embeddingsDir);

  // BM25
  const expandedTerms = await expandQuery(query);
  const engine = buildIndex(allPosts);
  const bm25Raw = engine.search(expandedTerms.join(' '));
  const bm25Max = bm25Raw.length > 0 ? bm25Raw[0][1] : 1;
  const bm25Map = new Map(bm25Raw.map(([id, score]) => [id, score / bm25Max]));

  // Два вектора параллельно
  const [rewritten, hypothetical] = await Promise.all([
    rewriteQueryForEmbedding(query),
    generateHypotheticalDocument(query),
  ]);
  const queryVector = await generateEmbedding(rewritten);
  const hydeVector = await generateEmbedding(hypothetical);

  const vecMap = new Map();
  const hydeMap = new Map();
  for (const post of allPosts) {
    if (!embeddings.has(post.id)) continue;
    const postVec = embeddings.get(post.id);
    vecMap.set(post.id, (cosineSimilarity(queryVector, postVec) + 1) / 2);
    hydeMap.set(post.id, (cosineSimilarity(hydeVector, postVec) + 1) / 2);
  }

  const postMap = new Map(allPosts.map(p => [p.id, p]));
  const allIds = new Set([...bm25Map.keys(), ...vecMap.keys()]);

  return Array.from(allIds)
    .map(id => ({
      id,
      score: alpha * (bm25Map.get(id) ?? 0)
           + beta  * (vecMap.get(id)  ?? 0)
           + gamma * (hydeMap.get(id) ?? 0),
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
    .map(({ id, score }) => ({ ...postMap.get(id), score }));
}

/**
 * RRF (Reciprocal Rank Fusion) поиск.
 *
 * Проблема взвешенного суммирования скоров (hybridSearchFull):
 * скоры BM25, vector и HyDE имеют разные масштабы и распределения,
 * поэтому один сигнал может подавлять другой при неправильных весах.
 *
 * RRF решает это через ранги вместо скоров:
 *   rrf_score = Σ 1/(k + rank_i)  для каждого сигнала
 * k=60 — стандартный параметр, сглаживает влияние топ-1.
 *
 * Преимущества:
 * - Не требует нормализации скоров
 * - Равноправно объединяет сигналы разных масштабов
 * - Хорошо работает при неоднородных корпусах
 */
export async function hybridSearchRRF(query, topK = 10, { postsDir, embeddingsDir, k = 60 } = {}) {
  const allPosts = await loadAllPosts(postsDir);
  if (allPosts.length === 0) return [];

  const embeddings = await loadAllEmbeddings(embeddingsDir);
  const postMap = new Map(allPosts.map(p => [p.id, p]));

  // Параллельно: синонимы для BM25, переписанный запрос, гипотетический документ
  const [expandedTerms, rewritten, hypothetical] = await Promise.all([
    expandQuery(query),
    rewriteQueryForEmbedding(query),
    generateHypotheticalDocument(query),
  ]);
  const [queryVector, hydeVector] = await Promise.all([
    generateEmbedding(rewritten),
    generateEmbedding(hypothetical),
  ]);

  // BM25 ranking
  const engine = buildIndex(allPosts);
  const bm25Raw = engine.search(expandedTerms.join(' '));
  const bm25Ranks = new Map(bm25Raw.map(([id], i) => [id, i]));

  // Vector ranking
  const vecScored = allPosts
    .filter(p => embeddings.has(p.id))
    .map(p => ({ id: p.id, score: cosineSimilarity(queryVector, embeddings.get(p.id)) }))
    .sort((a, b) => b.score - a.score);
  const vecRanks = new Map(vecScored.map(({ id }, i) => [id, i]));

  // HyDE ranking
  const hydeScored = allPosts
    .filter(p => embeddings.has(p.id))
    .map(p => ({ id: p.id, score: cosineSimilarity(hydeVector, embeddings.get(p.id)) }))
    .sort((a, b) => b.score - a.score);
  const hydeRanks = new Map(hydeScored.map(({ id }, i) => [id, i]));

  // RRF fusion
  const allIds = new Set([...bm25Ranks.keys(), ...vecRanks.keys()]);
  const scored = Array.from(allIds).map(id => {
    const rrf = (ranks, fallback) => 1 / (k + (ranks.has(id) ? ranks.get(id) : fallback));
    const score = rrf(bm25Ranks, allPosts.length)
                + rrf(vecRanks,  allPosts.length)
                + rrf(hydeRanks, allPosts.length);
    return { id, score };
  });

  return scored
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
    .map(({ id, score }) => ({ ...postMap.get(id), score }));
}

/**
 * Переписывает запрос для эмбеддинга: убирает вопросительную структуру,
 * оставляет ключевые понятия в виде существительных/прилагательных.
 *
 * Проблема: запрос "Как выйти замуж на яхте" эмбеддится с паттерном "Как X на яхте",
 * и вектор ближе к другим "Как X на яхте"-постам, чем к постам о свадьбе.
 * Переписанный запрос "свадьба на яхте, замужество, венчание" эмбеддится семантически точнее.
 */
async function rewriteQueryForEmbedding(query) {
  const cacheKey = query.toLowerCase().trim();
  if (rewriteCache.has(cacheKey)) return rewriteCache.get(cacheKey);

  const cached = await diskCacheGet('rewrite', cacheKey);
  if (cached) { rewriteCache.set(cacheKey, cached); return cached; }

  const response = await openai.chat.completions.create({
    model: 'gpt-4o-mini',
    max_tokens: 60,
    messages: [{
      role: 'user',
      content: `Перепиши этот поисковый запрос как ключевые понятия (существительные и прилагательные), убрав вопросительную или повелительную структуру. Только слова и словосочетания через запятую, без пояснений. Запрос: "${query}"`,
    }],
  });

  const rewritten = response.choices[0].message.content.trim();
  rewriteCache.set(cacheKey, rewritten);
  await diskCacheSet('rewrite', cacheKey, rewritten);
  return rewritten;
}

async function generateHypotheticalDocument(query) {
  const cacheKey = query.toLowerCase().trim();
  if (hydeCache.has(cacheKey)) return hydeCache.get(cacheKey);

  const cached = await diskCacheGet('hyde', cacheKey);
  if (cached) { hydeCache.set(cacheKey, cached); return cached; }

  const response = await openai.chat.completions.create({
    model: 'gpt-4o-mini',
    max_tokens: 200,
    messages: [{
      role: 'user',
      content: `Напиши короткий пост для Telegram-канала о яхтинге (50-150 слов, живой разговорный стиль, без заголовков), который максимально точно соответствует теме: "${query}".`,
    }],
  });

  const doc = response.choices[0].message.content;
  hydeCache.set(cacheKey, doc);
  await diskCacheSet('hyde', cacheKey, doc);
  return doc;
}

/**
 * Multi-query BM25 с RRF-фьюжном.
 *
 * Проблема одного BM25-запроса: если релевантные посты используют другой словарь
 * ("кайф", "свобода", "инвестиция"), чем запрос ("яхтенные путешествия", "лучший отдых") —
 * BM25 их не находит.
 *
 * Решение: генерируем 4 паrafразы запроса в стиле русских соцсетей (разные слова,
 * тот же смысл), прогоняем BM25 по каждой, объединяем 5 ранкингов через RRF.
 * Каждая паrafраза ловит другое подмножество постов с разными словами.
 */
export async function multiQueryRRF(query, topK = 10, { postsDir, k = 60 } = {}) {
  const allPosts = await loadAllPosts(postsDir);
  if (allPosts.length === 0) return [];

  const postMap = new Map(allPosts.map(p => [p.id, p]));
  const engine = buildIndex(allPosts);

  // Генерируем паrafразы + оригинальный запрос
  const paraphrases = await generateParaphrases(query);
  const queries = [query, ...paraphrases];

  // BM25 ранкинг для каждого запроса
  const rankings = [];
  for (const q of queries) {
    const results = engine.search(q);
    rankings.push(new Map(results.map(([id], i) => [id, i])));
  }

  // RRF fusion по всем ранкингам
  const allIds = new Set(rankings.flatMap(r => [...r.keys()]));
  const scored = Array.from(allIds).map(id => {
    let score = 0;
    for (const ranks of rankings) {
      const rank = ranks.has(id) ? ranks.get(id) : allPosts.length;
      score += 1 / (k + rank);
    }
    return { id, score };
  });

  return scored
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
    .map(({ id, score }) => ({ ...postMap.get(id), score }));
}

/**
 * Комбинированный поиск: Multi-query BM25 + Vector + HyDE → RRF.
 *
 * Объединяет все сигналы:
 * - BM25 по оригинальному запросу + 4 парафразам (ловит разный словарь)
 * - Вектор переписанного запроса (семантика без вопросительной структуры)
 * - HyDE вектор (устраняет фреймовый разрыв — разные роли, разный стиль)
 *
 * Итого 7 сигналов в RRF. Каждый ловит то, что другие пропускают.
 */
export async function mqHybridRRF(query, topK = 10, { postsDir, embeddingsDir, k = 60 } = {}) {
  const allPosts = await loadAllPosts(postsDir);
  if (allPosts.length === 0) return [];

  const embeddings = await loadAllEmbeddings(embeddingsDir);
  const postMap = new Map(allPosts.map(p => [p.id, p]));

  // Параллельно: парафразы, переписанный запрос, HyDE документ
  const [paraphrases, rewritten, hypothetical] = await Promise.all([
    generateParaphrases(query),
    rewriteQueryForEmbedding(query),
    generateHypotheticalDocument(query),
  ]);

  // Два вектора параллельно
  const [queryVector, hydeVector] = await Promise.all([
    generateEmbedding(rewritten),
    generateEmbedding(hypothetical),
  ]);

  // BM25 ранкинги: оригинал + 4 парафразы
  const engine = buildIndex(allPosts);
  const bm25Queries = [query, ...paraphrases];
  const bm25Rankings = bm25Queries.map(q => {
    const results = engine.search(q);
    return new Map(results.map(([id], i) => [id, i]));
  });

  // Векторные ранкинги
  const vecScored = allPosts
    .filter(p => embeddings.has(p.id))
    .map(p => ({ id: p.id, score: cosineSimilarity(queryVector, embeddings.get(p.id)) }))
    .sort((a, b) => b.score - a.score);
  const vecRanks = new Map(vecScored.map(({ id }, i) => [id, i]));

  const hydeScored = allPosts
    .filter(p => embeddings.has(p.id))
    .map(p => ({ id: p.id, score: cosineSimilarity(hydeVector, embeddings.get(p.id)) }))
    .sort((a, b) => b.score - a.score);
  const hydeRanks = new Map(hydeScored.map(({ id }, i) => [id, i]));

  // Все ранкинги вместе
  const allRankings = [...bm25Rankings, vecRanks, hydeRanks];

  // RRF fusion
  const allIds = new Set(allRankings.flatMap(r => [...r.keys()]));
  const scored = Array.from(allIds).map(id => {
    let score = 0;
    for (const ranks of allRankings) {
      const rank = ranks.has(id) ? ranks.get(id) : allPosts.length;
      score += 1 / (k + rank);
    }
    return { id, score };
  });

  return scored
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
    .map(({ id, score }) => ({ ...postMap.get(id), score }));
}

/**
 * Поиск всех релевантных постов без фиксированного лимита.
 *
 * Вместо "выдай топ-N" просим LLM отфильтровать все релевантные посты из candidateK
 * кандидатов. Возвращает переменное количество постов — столько, сколько LLM
 * сочтёт полезными для темы.
 *
 * Подходит когда нужно собрать весь материал по теме, а не только топ-15.
 */
export async function searchRelevant(query, { postsDir, embeddingsDir, candidateK = 100 } = {}) {
  const candidates = await mqHybridRRF(query, candidateK, { postsDir, embeddingsDir });
  if (candidates.length === 0) return [];
  return await llmFilterAll(query, candidates);
}

async function llmFilterAll(query, candidates) {
  const postList = candidates
    .map((p, i) => {
      const text = p.textClean || p.ocrText || '';
      const snippet = text.length > 600
        ? text.slice(0, 350) + ' … ' + text.slice(-150)
        : text.slice(0, 500);
      return `[${i + 1}] ${snippet}`;
    })
    .join('\n\n');

  const response = await openai.chat.completions.create({
    model: 'gpt-4o-mini',
    max_tokens: 600,
    messages: [
      {
        role: 'system',
        content: 'Ты эксперт по яхтингу и контент-маркетингу. Пост считается релевантным если он может дать идеи, факты, истории, эмоции или формулировки для создания контента на заданную тему — даже если прямых совпадений слов нет.',
      },
      {
        role: 'user',
        content: `Тема контент-плана: "${query}"\n\nВерни номера ВСЕХ постов которые могут быть полезны для этой темы, от наиболее к наименее релевантному. Не ограничивай себя числом — верни все подходящие. Посты совсем не по теме не включай.\n\nТолько номера через запятую, без пояснений.\n\n${postList}`,
      },
    ],
  });

  const text = response.choices[0].message.content;
  const indices = (text.match(/\d+/g) || [])
    .map(n => parseInt(n) - 1)
    .filter((i, pos, arr) => i >= 0 && i < candidates.length && arr.indexOf(i) === pos);

  return indices.map(i => candidates[i]);
}

/**
 * Pointwise re-ranking: оценивает каждый кандидат независимо по шкале 1-10,
 * возвращает посты со скором >= minScore. Нет позиционного bias — каждый пост
 * оценивается отдельным параллельным запросом.
 */
export async function searchWithRerank(query, topK = 10, { postsDir, embeddingsDir, candidateK = 50, minScore = 7 } = {}) {
  const candidates = await mqHybridRRF(query, candidateK, { postsDir, embeddingsDir });
  if (candidates.length === 0) return [];

  const scores = await llmRerankPointwise(query, candidates);

  // Посты со скором >= minScore, отсортированные по убыванию скора
  return scores
    .filter(({ score }) => score >= minScore)
    .sort((a, b) => b.score - a.score)
    .map(({ post }) => post);
}

async function llmRerankPointwise(query, candidates) {
  const results = await Promise.all(
    candidates.map(async (post) => {
      const text = post.textClean || post.ocrText || '';
      const snippet = text.length > 600
        ? text.slice(0, 350) + ' … ' + text.slice(-150)
        : text.slice(0, 500);

      const response = await openai.chat.completions.create({
        model: 'gpt-4o-mini',
        max_tokens: 5,
        messages: [
          {
            role: 'system',
            content: 'Ты эксперт по яхтингу и контент-маркетингу. Оцени насколько пост полезен для создания контента на заданную тему. Пост полезен если даёт идеи, факты, истории, эмоции или формулировки — даже без прямых совпадений слов. Ответь одним числом от 1 до 10.',
          },
          {
            role: 'user',
            content: `Тема: "${query}"\n\nПост:\n${snippet}`,
          },
        ],
      });

      const raw = response.choices[0].message.content.trim();
      const score = parseInt(raw.match(/\d+/)?.[0] ?? '0');
      return { post, score: Math.min(10, Math.max(1, score)) };
    })
  );

  return results;
}

/**
 * Listwise re-ranking (устаревший вариант, оставлен для сравнения).
 */
async function searchWithRerankListwise(query, topK = 10, { postsDir, embeddingsDir, candidateK = 50 } = {}) {
  const candidates = await mqHybridRRF(query, candidateK, { postsDir, embeddingsDir });
  if (candidates.length === 0) return [];

  const reranked = await llmRerank(query, candidates, topK * 2);
  const mqHybridTop = candidates.slice(0, topK);

  const k = 60;
  const rerankRanks = new Map(reranked.map((p, i) => [p.id, i]));
  const mqRanks = new Map(mqHybridTop.map((p, i) => [p.id, i]));

  const allIds = new Set([...rerankRanks.keys(), ...mqRanks.keys()]);
  const postMap = new Map(candidates.map(p => [p.id, p]));

  return Array.from(allIds)
    .map(id => ({
      id,
      score: 1 / (k + (rerankRanks.get(id) ?? reranked.length))
           + 1 / (k + (mqRanks.get(id) ?? topK)),
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
    .map(({ id }) => postMap.get(id));
}

async function llmRerank(query, candidates, topK) {
  const postList = candidates
    .map((p, i) => {
      const text = p.textClean || p.ocrText || '';
      // Для длинных постов берём начало и конец — тема может раскрываться в любом месте
      const snippet = text.length > 600
        ? text.slice(0, 350) + ' … ' + text.slice(-150)
        : text.slice(0, 500);
      return `[${i + 1}] ${snippet}`;
    })
    .join('\n\n');

  const response = await openai.chat.completions.create({
    model: 'gpt-4o-mini',
    max_tokens: 300,
    messages: [
      {
        role: 'system',
        content: 'Ты эксперт по яхтингу и контент-маркетингу. Ты умеешь находить смысловые связи между постами и темами, даже если совпадений слов нет. Пост считается релевантным если он может дать идеи, факты, истории, эмоции или формулировки для создания контента на заданную тему.',
      },
      {
        role: 'user',
        content: `Тема контент-плана: "${query}"\n\nИз постов ниже выбери ${topK} наиболее релевантных. Ищи семантическую связь — пост может описывать тему другими словами, через личный опыт или конкретные детали.\n\nВерни только номера постов через запятую, от наиболее к наименее релевантному. Без пояснений.\n\n${postList}`,
      },
    ],
  });

  const text = response.choices[0].message.content;
  const indices = (text.match(/\d+/g) || [])
    .map(n => parseInt(n) - 1)
    .filter((i, pos, arr) => i >= 0 && i < candidates.length && arr.indexOf(i) === pos);

  const seen = new Set(indices);
  const rest = candidates.map((_, i) => i).filter(i => !seen.has(i));
  return [...indices, ...rest].slice(0, topK).map(i => candidates[i]);
}

async function generateParaphrases(query) {
  const cacheKey = query.toLowerCase().trim();
  if (paraphrasesCache.has(cacheKey)) return paraphrasesCache.get(cacheKey);

  const cached = await diskCacheGet('para8', cacheKey);
  if (cached) { paraphrasesCache.set(cacheKey, cached); return cached; }

  const response = await openai.chat.completions.create({
    model: 'gpt-4o-mini',
    max_tokens: 400,
    messages: [{
      role: 'user',
      content: `Перефразируй этот запрос 8 разными способами, используя максимально разные слова и выражения: разговорный стиль соцсетей, нарративные ключевые слова, синонимы темы. Каждая парафраза на новой строке, без нумерации и пояснений. Запрос: "${query}"`,
    }],
  });

  const paraphrases = response.choices[0].message.content
    .split('\n')
    .map(s => s.trim())
    .filter(Boolean)
    .slice(0, 8);

  paraphrasesCache.set(cacheKey, paraphrases);
  await diskCacheSet('para8', cacheKey, paraphrases);
  return paraphrases;
}

async function loadAllPosts(postsDir = DEFAULT_POSTS_DIR) {
  const posts = [];
  let files;
  try {
    files = await fs.readdir(postsDir);
  } catch {
    return [];
  }
  for (const file of files) {
    if (!file.endsWith('.json')) continue;
    try {
      const raw = await fs.readFile(path.join(postsDir, file), 'utf-8');
      posts.push(...JSON.parse(raw));
    } catch {}
  }
  return posts;
}

function buildIndex(posts) {
  const engine = winkBM25();
  engine.defineConfig({ fldWeights: { textClean: 3, ocrText: 1 } });
  engine.definePrepTasks([
    winkNLP.string.lowerCase,
    winkNLP.string.removeExtraSpaces,
    (s) => (s.match(/[\p{L}\p{N}]+/gu) || []),
    (tokens) => tokens.filter(t => !ruStopwordsDict[t]),
    (tokens) => ruStemmer.stem(tokens),
  ]);
  for (const post of posts) {
    engine.addDoc(
      { textClean: post.textClean || '', ocrText: post.ocrText || '' },
      post.id
    );
  }
  engine.consolidate();
  return engine;
}

async function expandQuery(query) {
  const cacheKey = query.toLowerCase().trim();
  if (synonymsCache.has(cacheKey)) {
    return synonymsCache.get(cacheKey);
  }

  try {
    const response = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      max_tokens: 100,
      messages: [
        {
          role: 'user',
          content: `Ты помогаешь искать посты о яхтинге. Дай список из 5-7 синонимов и связанных терминов для запроса: "${query}". Только слова через запятую, без пояснений.`,
        },
      ],
    });

    const raw = response.choices[0].message.content;
    const synonyms = raw.split(',').map(s => s.trim()).filter(Boolean);
    const terms = [query, ...synonyms];
    synonymsCache.set(cacheKey, terms);
    return terms;
  } catch (err) {
    throw new Error(`OpenAI недоступен: ${err.message}`);
  }
}

export async function getStats() {
  const files = await fs.readdir(DEFAULT_POSTS_DIR).catch(() => []);
  const stats = {};
  for (const file of files) {
    if (!file.endsWith('.json')) continue;
    try {
      const raw = await fs.readFile(path.join(DEFAULT_POSTS_DIR, file), 'utf-8');
      const posts = JSON.parse(raw);
      stats[file.replace('.json', '')] = posts.length;
    } catch {}
  }
  return stats;
}
