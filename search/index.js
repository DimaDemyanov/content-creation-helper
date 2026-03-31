import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import winkBM25 from 'wink-bm25-text-search';
import winkNLP from 'wink-nlp-utils';
import { StemmerRu, StopwordsRu } from '@nlpjs/lang-ru';
import { generateEmbedding, loadAllEmbeddings, loadAllChunkEmbeddings, cosineSimilarity } from './embeddings.js';
import { llmClient as openai, LLM_MODEL } from '../llm.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_POSTS_DIR = path.join(__dirname, '../data/posts');
const QUERY_CACHE_DIR = path.join(__dirname, '../.cache/queries');

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function parseRetryAfterMs(err) {
  const retryAfterMs = Number(
    err?.headers?.['retry-after-ms']
    ?? err?.response?.headers?.['retry-after-ms']
  );
  if (Number.isFinite(retryAfterMs) && retryAfterMs > 0) return retryAfterMs;

  const retryAfterSec = Number(
    err?.headers?.['retry-after']
    ?? err?.response?.headers?.['retry-after']
  );
  if (Number.isFinite(retryAfterSec) && retryAfterSec > 0) return retryAfterSec * 1000;
  return 0;
}

function isRateLimitError(err) {
  const message = String(err?.message || '').toLowerCase();
  return err?.status === 429
    || err?.code === 'rate_limit_exceeded'
    || err?.error?.code === 'rate_limit_exceeded'
    || message.includes('rate limit')
    || message.includes(' 429')
    || message.startsWith('429 ');
}

// Экспоненциальный ретрай для rate limit ошибок (до 3 минут суммарно)
async function withRetry(fn, { maxTotalMs = 180_000, initialDelayMs = 800 } = {}) {
  const startMs = Date.now();
  let delayMs = initialDelayMs;
  while (true) {
    try {
      return await fn();
    } catch (err) {
      if (!isRateLimitError(err)) throw err;

      const elapsed = Date.now() - startMs;
      const retryAfterMs = parseRetryAfterMs(err);
      const jitterMs = Math.floor(Math.random() * 250);
      const waitMs = Math.max(delayMs, retryAfterMs) + jitterMs;
      if (elapsed + waitMs > maxTotalMs) throw err;

      await sleep(waitMs);
      delayMs = Math.min(Math.round(delayMs * 1.7), 8_000);
    }
  }
}

// Русский стеммер и стоп-слова
const ruStemmer = new StemmerRu();
const ruStopwordsDict = new StopwordsRu().dictionary;

// In-memory кэши (работают внутри процесса)
const synonymsCache = new Map();
const paraphrasesCache = new Map();
const hydeCache = new Map();
const rewriteCache = new Map();

// Основные режимы поиска для продукта:
// - fast: быстрый и дешёвый BM25
// - balanced: лучший компромисс по качеству (mqHybridRRF)
// - highRecall: алиас balanced для обратной совместимости
export const PRIMARY_SEARCH_METHODS = Object.freeze({
  fast: 'search',
  balanced: 'mqHybridRRF',
  highRecall: 'mqHybridRRF',
});

const MQ_HYBRID_DEFAULT_WEIGHTS = Object.freeze({
  bm25Raw: 1.05,
  bm25Expanded: 1.35,
  bm25Paraphrase: 0.75,
  bm25Rewrite: 0.9,
  bm25Feedback: 0.45,
  vecRewrite: 1.15,
  vecHyde: 1.05,
  vecRaw: 0.55,
  vecRewriteChunk: 0.95,
  vecHydeChunk: 0.85,
  vecCentroid: 0.55,
  channelPrior: 0.2,
  intentAlignment: 1.65,
  directMatch: 0.25,
});

// Документированная последовательность шагов mqHybrid.
export const MQ_HYBRID_STEPS = Object.freeze([
  'prepare_query_signals',
  'build_lexical_rankings',
  'build_vector_rankings',
  'expand_with_pseudo_relevance_feedback',
  'build_intent_alignment_ranking',
  'fuse_rankings_with_weighted_rrf',
  'llm_relevance_refinement',
  'return_top_k',
]);

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

  const response = await withRetry(() => openai.chat.completions.create({
    model: LLM_MODEL,
    temperature: 0,
    max_tokens: 60,
    messages: [{
      role: 'user',
      content: `Перепиши этот поисковый запрос как ключевые понятия (существительные и прилагательные), убрав вопросительную или повелительную структуру. Только слова и словосочетания через запятую, без пояснений. Запрос: "${query}"`,
    }],
  }));

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

  const response = await withRetry(() => openai.chat.completions.create({
    model: LLM_MODEL,
    temperature: 0,
    max_tokens: 200,
    messages: [{
      role: 'user',
      content: `Напиши короткий пост для Telegram-канала о яхтинге (50-150 слов, живой разговорный стиль, без заголовков), который максимально точно соответствует теме: "${query}".`,
    }],
  }));

  const doc = response.choices[0].message.content;
  hydeCache.set(cacheKey, doc);
  await diskCacheSet('hyde', cacheKey, doc);
  return doc;
}

function buildVectorRanks(allPosts, embeddings, vector) {
  const scored = allPosts
    .filter(p => embeddings.has(p.id))
    .map(p => ({ id: p.id, score: cosineSimilarity(vector, embeddings.get(p.id)) }))
    .sort((a, b) => b.score - a.score);
  return new Map(scored.map(({ id }, i) => [id, i]));
}

function buildChunkVectorRanks(chunkMap, vector) {
  const scored = [];
  for (const [postId, chunks] of chunkMap) {
    if (!chunks || chunks.length === 0) continue;
    const bestScore = Math.max(...chunks.map(c => cosineSimilarity(vector, c.vector)));
    scored.push({ id: postId, score: bestScore });
  }
  scored.sort((a, b) => b.score - a.score);
  return new Map(scored.map(({ id }, i) => [id, i]));
}

function computeWeightedRrfScores(signals, k, fallbackRank) {
  const allIds = new Set(signals.flatMap(s => [...s.ranks.keys()]));
  return Array.from(allIds).map(id => {
    let score = 0;
    for (const { ranks, weight } of signals) {
      const rank = ranks.has(id) ? ranks.get(id) : fallbackRank;
      score += weight * (1 / (k + rank));
    }
    return { id, score };
  });
}

function averageVectors(vectors) {
  if (!vectors || vectors.length === 0) return null;
  const size = vectors[0].length;
  const acc = new Array(size).fill(0);
  for (const vec of vectors) {
    for (let i = 0; i < size; i++) acc[i] += vec[i];
  }
  for (let i = 0; i < size; i++) acc[i] /= vectors.length;
  return acc;
}

function tokenizeAndStem(text) {
  const tokens = (text.toLowerCase().match(/[\p{L}\p{N}]+/gu) || [])
    .filter(t => !ruStopwordsDict[t]);
  return ruStemmer.stem(tokens);
}

function toStemSet(text) {
  return new Set(tokenizeAndStem(text));
}

function buildStemBigrams(stems) {
  const pairs = [];
  for (let i = 0; i < stems.length - 1; i++) {
    if (stems[i] && stems[i + 1]) pairs.push(`${stems[i]}::${stems[i + 1]}`);
  }
  return pairs;
}

function conceptCoverage(postStemSet, conceptStemSet) {
  if (conceptStemSet.size === 0) return 0;
  let hit = 0;
  for (const stem of conceptStemSet) {
    if (postStemSet.has(stem)) hit++;
  }
  return hit / conceptStemSet.size;
}

function computeDirectMatchScore(postStemSet, queryStemSet, queryBigrams = []) {
  if (queryStemSet.size === 0) return 0;

  let overlapHits = 0;
  for (const stem of queryStemSet) {
    if (postStemSet.has(stem)) overlapHits++;
  }
  const overlap = overlapHits / queryStemSet.size;

  let bigramHits = 0;
  for (const pair of queryBigrams) {
    const [a, b] = pair.split('::');
    if (postStemSet.has(a) && postStemSet.has(b)) bigramHits++;
  }
  const bigramCoverage = queryBigrams.length > 0 ? bigramHits / queryBigrams.length : 0;

  return (0.75 * overlap) + (0.25 * bigramCoverage);
}

function buildDirectMatchRanks(posts, query, { corpusForDf = posts } = {}) {
  const rawQueryStems = tokenizeAndStem(query).filter(stem => stem.length >= 3);
  if (rawQueryStems.length === 0) {
    return { ranks: new Map(), scoreMap: new Map() };
  }

  // Убираем слишком частые query-термы (низкая дискриминативность по корпусу).
  const df = new Map();
  for (const post of corpusForDf) {
    const stemSet = toStemSet(`${post.textClean || ''} ${post.ocrText || ''}`);
    for (const stem of new Set(rawQueryStems)) {
      if (stemSet.has(stem)) df.set(stem, (df.get(stem) || 0) + 1);
    }
  }

  const corpusSize = Math.max(1, corpusForDf.length);
  const discriminative = rawQueryStems.filter(stem => ((df.get(stem) || 0) / corpusSize) <= 0.45);
  const queryStems = discriminative.length > 0 ? discriminative : rawQueryStems;
  const queryStemSet = new Set(queryStems);
  const queryBigrams = buildStemBigrams(queryStems);

  const ranked = posts
    .map((post) => {
      const postStemSet = toStemSet(`${post.textClean || ''} ${post.ocrText || ''}`);
      const score = computeDirectMatchScore(postStemSet, queryStemSet, queryBigrams);
      return { id: post.id, score };
    })
    .sort((a, b) => b.score - a.score);

  return {
    ranks: new Map(ranked.map(({ id }, i) => [id, i])),
    scoreMap: new Map(ranked.map(({ id, score }) => [id, score])),
  };
}

function buildIntentProfile(query, expandedTerms = []) {
  const rawTokens = (query.toLowerCase().match(/[\p{L}\p{N}]+/gu) || [])
    .filter(t => !ruStopwordsDict[t] && t.length >= 3);

  const mustConcepts = Array.from(new Set(rawTokens)).slice(0, 6);
  const supportingConcepts = expandedTerms
    .map(t => String(t || '').toLowerCase().trim())
    .filter(Boolean)
    .filter(t => !mustConcepts.includes(t))
    .slice(0, 8);

  return {
    queryType: 'other',
    mustConcepts,
    supportingConcepts,
  };
}

function buildIntentAlignment(allPosts, query, intentProfile) {
  const queryStems = toStemSet(query);
  const mustStemSets = intentProfile.mustConcepts.map(c => toStemSet(c)).filter(s => s.size > 0);
  const supportStemSets = intentProfile.supportingConcepts.map(c => toStemSet(c)).filter(s => s.size > 0);

  const scored = allPosts.map((post) => {
    const text = `${post.textClean || ''} ${post.ocrText || ''}`.trim();
    const postStemSet = toStemSet(text);

    const mustCoverage = mustStemSets.length > 0
      ? mustStemSets.filter(concept => conceptCoverage(postStemSet, concept) >= 0.6).length / mustStemSets.length
      : 0;
    const supportCoverage = supportStemSets.length > 0
      ? supportStemSets.filter(concept => conceptCoverage(postStemSet, concept) >= 0.5).length / supportStemSets.length
      : 0;

    let queryOverlapHits = 0;
    for (const stem of queryStems) {
      if (postStemSet.has(stem)) queryOverlapHits++;
    }
    const overlap = queryStems.size > 0 ? queryOverlapHits / queryStems.size : 0;

    const score = (0.65 * mustCoverage) + (0.25 * supportCoverage) + (0.10 * overlap);
    return { id: post.id, score, mustCoverage };
  });

  const ranks = new Map(
    scored
      .sort((a, b) => {
        if (b.score !== a.score) return b.score - a.score;
        return b.mustCoverage - a.mustCoverage;
      })
      .map(({ id }, idx) => [id, idx])
  );

  const scoreMap = new Map(scored.map(({ id, score }) => [id, score]));
  return { ranks, scoreMap };
}

function buildPseudoRelevanceFeedback({
  allPosts,
  signals,
  k,
  engine,
  expandedQuery,
  query,
  embeddings,
  rrfWeights,
  seedK = 24,
}) {
  const feedbackSignals = [];
  if (signals.length === 0 || allPosts.length === 0) return feedbackSignals;

  const prelim = computeWeightedRrfScores(signals, k, allPosts.length)
    .sort((a, b) => b.score - a.score)
    .slice(0, Math.min(seedK, allPosts.length));
  const seedIds = prelim.map(x => x.id);
  if (seedIds.length === 0) return feedbackSignals;

  const postMap = new Map(allPosts.map(p => [p.id, p]));
  const queryStems = toStemSet(query);

  // 1) Lexical PRF: дополняем BM25-запрос термами из seed-документов.
  const stemFreq = new Map();
  for (const id of seedIds) {
    const post = postMap.get(id);
    if (!post) continue;
    const stemSet = toStemSet(`${post.textClean || ''} ${post.ocrText || ''}`);
    for (const stem of stemSet) {
      if (queryStems.has(stem)) continue;
      stemFreq.set(stem, (stemFreq.get(stem) || 0) + 1);
    }
  }

  const feedbackTerms = [...stemFreq.entries()]
    .filter(([, freq]) => freq >= 2)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
    .map(([stem]) => stem);

  if (feedbackTerms.length > 0) {
    const bm25FeedbackQuery = `${expandedQuery} ${feedbackTerms.join(' ')}`.trim();
    const ranks = new Map(engine.search(bm25FeedbackQuery).map(([id], i) => [id, i]));
    feedbackSignals.push({
      name: 'bm25Feedback',
      ranks,
      weight: rrfWeights.bm25Feedback,
    });
  }

  // 2) Vector PRF: добавляем ранкинг по centroid seed-эмбеддингов.
  if (embeddings.size > 0) {
    const seedVectors = seedIds
      .filter(id => embeddings.has(id))
      .map(id => embeddings.get(id));
    const centroid = averageVectors(seedVectors);
    if (centroid) {
      feedbackSignals.push({
        name: 'vecCentroid',
        ranks: buildVectorRanks(allPosts, embeddings, centroid),
        weight: rrfWeights.vecCentroid,
      });
    }
  }

  // 3) Channel prior: мягко повышаем каналы, подтверждённые seed-результатами.
  const channelWeights = new Map();
  seedIds.forEach((id, rank) => {
    const post = postMap.get(id);
    if (!post || !post.channel) return;
    channelWeights.set(post.channel, (channelWeights.get(post.channel) || 0) + (1 / (1 + rank)));
  });

  const channelEntries = [...channelWeights.entries()].sort((a, b) => b[1] - a[1]);
  const totalWeight = channelEntries.reduce((sum, [, w]) => sum + w, 0) || 1;
  const topShare = channelEntries.length > 0 ? channelEntries[0][1] / totalWeight : 0;

  if (channelEntries.length > 0 && topShare < 0.9) {
    const ranked = allPosts
      .map((post) => {
        const channelScore = channelWeights.get(post.channel) || 0;
        const overlap = conceptCoverage(toStemSet(`${post.textClean || ''} ${post.ocrText || ''}`), queryStems);
        return { id: post.id, score: channelScore + (0.35 * overlap) };
      })
      .sort((a, b) => b.score - a.score);

    feedbackSignals.push({
      name: 'channelPrior',
      ranks: new Map(ranked.map(({ id }, i) => [id, i])),
      weight: rrfWeights.channelPrior,
    });
  }

  return feedbackSignals;
}

function resolveChunkEmbeddingsDir(embeddingsDir, chunkEmbeddingsDir) {
  if (chunkEmbeddingsDir) return chunkEmbeddingsDir;
  if (embeddingsDir) return path.join(path.dirname(embeddingsDir), 'chunk-embeddings');
  return undefined;
}

function truncateText(text, maxChars) {
  if (text.length <= maxChars) return text;
  return text.slice(0, Math.max(0, maxChars - 1)).trimEnd() + '…';
}

function buildQueryAwareSnippet(text, query, maxChars = 500) {
  const normalized = (text || '').replace(/\s+/g, ' ').trim();
  if (!normalized) return '';
  if (normalized.length <= maxChars) return normalized;

  const queryStems = toStemSet(query);
  if (queryStems.size === 0) {
    return truncateText(normalized, maxChars);
  }

  const sentences = normalized.match(/[^.!?]+[.!?]*/g) ?? [normalized];
  const scored = sentences
    .map((s, idx) => {
      const stemSet = toStemSet(s);
      let hits = 0;
      for (const stem of queryStems) if (stemSet.has(stem)) hits++;
      return { idx, text: s.trim(), hits };
    })
    .filter(x => x.text.length > 0);

  scored.sort((a, b) => {
    if (b.hits !== a.hits) return b.hits - a.hits;
    return a.idx - b.idx;
  });

  if (scored.length === 0 || scored[0].hits === 0) {
    // fallback: "бутерброд" если совпадений не нашли
    const sandwich = normalized.length > 600
      ? normalized.slice(0, 350) + ' … ' + normalized.slice(-150)
      : normalized.slice(0, 500);
    return truncateText(sandwich, maxChars);
  }

  const selectedIdx = new Set([scored[0].idx]);
  for (let i = 1; i < scored.length && selectedIdx.size < 3; i++) {
    if (scored[i].hits === 0) break;
    selectedIdx.add(scored[i].idx);
  }
  // Добавляем первую фразу как контекст, если она не выбрана
  if (!selectedIdx.has(0)) selectedIdx.add(0);

  const ordered = [...selectedIdx].sort((a, b) => a - b).map(i => sentences[i].trim()).filter(Boolean);
  return truncateText(ordered.join(' … '), maxChars);
}

/**
 * Строит внутреннее состояние mqHybrid по шагам:
 * 1) подготавливает сигналы запроса (expand/paraphrase/rewrite/hyde)
 * 2) строит лексические ранкинги BM25
 * 3) строит векторные ранкинги (rewrite/HyDE/raw)
 * 4) расширяет сигналы через pseudo relevance feedback
 * 5) добавляет intent-alignment ранкинг
 * 6) объединяет сигналы через weighted RRF
 */
async function buildMqHybridState(
  query,
  {
    postsDir,
    embeddingsDir,
    chunkEmbeddingsDir,
    k = 60,
    paraphraseCount = 4,
    prfSeedK = 24,
    weights = {},
  } = {}
) {
  const allPosts = await loadAllPosts(postsDir);
  const postMap = new Map(allPosts.map(p => [p.id, p]));
  const rrfWeights = { ...MQ_HYBRID_DEFAULT_WEIGHTS, ...weights };

  if (allPosts.length === 0) {
    return {
      allPosts,
      postMap,
      artifacts: {
        expandedTerms: [],
        expandedQuery: '',
        paraphrases: [],
        rewritten: '',
        hypothetical: '',
      },
      signals: [],
      fusedScores: [],
      k,
      paraphraseCount,
      prfSeedK,
      rrfWeights,
    };
  }

  // Step 1: подготовка сигналов запроса
  const [expandedTerms, paraphrases, rewritten, hypothetical] = await Promise.all([
    expandQuery(query),
    generateParaphrases(query, paraphraseCount),
    rewriteQueryForEmbedding(query),
    generateHypotheticalDocument(query),
  ]);
  const expandedQuery = expandedTerms.join(' ');
  const intentProfile = buildIntentProfile(query, expandedTerms);

  // Step 2: лексические ранкинги
  const engine = buildIndex(allPosts);
  const signals = [];

  const bm25RawRanks = new Map(engine.search(query).map(([id], i) => [id, i]));
  signals.push({ name: 'bm25Raw', ranks: bm25RawRanks, weight: rrfWeights.bm25Raw });

  const bm25ExpandedRanks = new Map(engine.search(expandedQuery).map(([id], i) => [id, i]));
  signals.push({ name: 'bm25Expanded', ranks: bm25ExpandedRanks, weight: rrfWeights.bm25Expanded });

  const bm25RewriteRanks = new Map(engine.search(rewritten).map(([id], i) => [id, i]));
  signals.push({ name: 'bm25Rewrite', ranks: bm25RewriteRanks, weight: rrfWeights.bm25Rewrite });

  paraphrases.forEach((q, idx) => {
    const ranks = new Map(engine.search(q).map(([id], i) => [id, i]));
    signals.push({ name: `bm25Para${idx + 1}`, ranks, weight: rrfWeights.bm25Paraphrase });
  });

  // Step 3: векторные ранкинги
  const embeddings = await loadAllEmbeddings(embeddingsDir);
  if (embeddings.size > 0) {
    const [rewriteVec, hydeVec, rawVec] = await Promise.all([
      generateEmbedding(rewritten),
      generateEmbedding(hypothetical),
      generateEmbedding(query),
    ]);

    signals.push({
      name: 'vecRewrite',
      ranks: buildVectorRanks(allPosts, embeddings, rewriteVec),
      weight: rrfWeights.vecRewrite,
    });
    signals.push({
      name: 'vecHyDE',
      ranks: buildVectorRanks(allPosts, embeddings, hydeVec),
      weight: rrfWeights.vecHyde,
    });
    signals.push({
      name: 'vecRaw',
      ranks: buildVectorRanks(allPosts, embeddings, rawVec),
      weight: rrfWeights.vecRaw,
    });

    const chunkDir = resolveChunkEmbeddingsDir(embeddingsDir, chunkEmbeddingsDir);
    const chunkMap = await loadAllChunkEmbeddings(chunkDir);
    if (chunkMap.size > 0) {
      signals.push({
        name: 'vecRewriteChunk',
        ranks: buildChunkVectorRanks(chunkMap, rewriteVec),
        weight: rrfWeights.vecRewriteChunk,
      });
      signals.push({
        name: 'vecHyDEChunk',
        ranks: buildChunkVectorRanks(chunkMap, hydeVec),
        weight: rrfWeights.vecHydeChunk,
      });
    }
  }

  // Step 4: pseudo relevance feedback (lexical/vector/channel)
  const feedbackSignals = buildPseudoRelevanceFeedback({
    allPosts,
    signals,
    k,
    engine,
    expandedQuery,
    query,
    embeddings,
    rrfWeights,
    seedK: prfSeedK,
  });
  signals.push(...feedbackSignals);

  // Step 5: intent alignment ranking (структурное соответствие теме)
  const intentAlignment = buildIntentAlignment(allPosts, query, intentProfile);
  signals.push({
    name: 'intentAlignment',
    ranks: intentAlignment.ranks,
    weight: rrfWeights.intentAlignment,
  });

  const directMatch = buildDirectMatchRanks(allPosts, query);
  signals.push({
    name: 'directMatch',
    ranks: directMatch.ranks,
    weight: rrfWeights.directMatch,
  });

  // Step 6: weighted RRF fusion
  const fusedScores = computeWeightedRrfScores(signals, k, allPosts.length);

  return {
    allPosts,
    postMap,
    artifacts: {
      expandedTerms,
      expandedQuery,
      paraphrases,
      rewritten,
      hypothetical,
      intentProfile,
    },
    signals,
    fusedScores,
    k,
    paraphraseCount,
    prfSeedK,
    rrfWeights,
  };
}

/**
 * Debug API для интеграционных тестов mqHybrid.
 * Возвращает артефакты каждого шага и вклад каждого сигнала.
 */
export async function debugMqHybridSteps(
  query,
  topK = 10,
  {
    postsDir,
    embeddingsDir,
    chunkEmbeddingsDir,
    k = 60,
    paraphraseCount = 4,
    prfSeedK = 24,
    weights = {},
    llmRefine = true,
    refineCandidateK = 300,
    topSignalK = 5,
  } = {}
) {
  const state = await buildMqHybridState(query, { postsDir, embeddingsDir, chunkEmbeddingsDir, k, paraphraseCount, prfSeedK, weights });
  const fusedTop = [...state.fusedScores]
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);

  const baseCandidates = [...state.fusedScores]
    .sort((a, b) => b.score - a.score)
    .slice(0, Math.max(topK, refineCandidateK))
    .map(({ id, score }) => ({ ...state.postMap.get(id), __baseScore: score }));
  const preRefineTop = baseCandidates.slice(0, topK);

  let refined = [];
  let refineDebug = null;
  if (llmRefine) {
    try {
      const refinedPayload = await llmRefineCandidates(query, baseCandidates, topK, { debug: true });
      if (Array.isArray(refinedPayload)) {
        refined = refinedPayload;
      } else {
        refined = refinedPayload.results || [];
        refineDebug = refinedPayload.debugInfo || null;
      }
    } catch {
      refined = baseCandidates.slice(0, topK);
    }
  } else {
    refined = baseCandidates.slice(0, topK);
  }

  const signalStats = state.signals.map(({ name, weight, ranks }) => ({
    name,
    weight,
    size: ranks.size,
    topIds: [...ranks.entries()]
      .sort((a, b) => a[1] - b[1])
      .slice(0, topSignalK)
      .map(([id]) => id),
  }));

  return {
    query,
    steps: MQ_HYBRID_STEPS,
    config: {
      k: state.k,
      paraphraseCount: state.paraphraseCount,
      prfSeedK: state.prfSeedK,
      weights: state.rrfWeights,
    },
    artifacts: state.artifacts,
    signalStats,
    fusedTop,
    preRefineTop: preRefineTop.map(({ __baseScore, ...post }) => ({ ...post, score: __baseScore ?? 0 })),
    refineDebug,
    refinedTop: refined.map(({ __baseScore, ...post }) => ({ ...post, score: __baseScore ?? 0 })),
    results: refined.map(({ __baseScore, ...post }) => ({ ...post, score: __baseScore ?? 0 })),
  };
}

/**
 * Комбинированный поиск: Multi-query BM25 + Vector + HyDE → weighted RRF.
 */
export async function mqHybridRRF(
  query,
  topK = 10,
  {
    postsDir,
    embeddingsDir,
    chunkEmbeddingsDir,
    k = 60,
    paraphraseCount = 4,
    prfSeedK = 24,
    weights = {},
    llmRefine = true,
    refineCandidateK = 300,
  } = {}
) {
  const state = await buildMqHybridState(query, { postsDir, embeddingsDir, chunkEmbeddingsDir, k, paraphraseCount, prfSeedK, weights });
  const fusedSorted = state.fusedScores
    .sort((a, b) => b.score - a.score)
    .map(({ id, score }) => ({ post: state.postMap.get(id), score }));

  if (!llmRefine) {
    return fusedSorted
      .slice(0, topK)
      .map(({ post, score }) => ({ ...post, score }));
  }

  const candidates = fusedSorted
    .slice(0, Math.max(topK, refineCandidateK))
    .map(({ post, score }) => ({ ...post, __baseScore: score }));

  try {
    const reranked = await llmRefineCandidates(query, candidates, topK);
    return reranked.map(({ __baseScore, ...post }) => ({ ...post, score: __baseScore ?? 0 }));
  } catch {
    return fusedSorted
      .slice(0, topK)
      .map(({ post, score }) => ({ ...post, score }));
  }
}

async function llmFilterAll(query, candidates) {
  const postList = candidates
    .map((p, i) => {
      const text = p.textClean || p.ocrText || '';
      const snippet = buildQueryAwareSnippet(text, query, 500);
      return `[${i + 1}] ${snippet}`;
    })
    .join('\n\n');

  const response = await withRetry(() => openai.chat.completions.create({
    model: LLM_MODEL,
    temperature: 0,
    max_tokens: 600,
    messages: [
      {
        role: 'system',
        content: 'Ты эксперт по яхтингу и контент-маркетингу. Пост считается релевантным если он может дать идеи, факты, истории, эмоции или формулировки для создания контента на заданную тему — даже если прямых совпадений слов нет.',
      },
      {
        role: 'user',
        content: `Тема контент-плана: "${query}"\n\nВерни номера ВСЕХ постов которые могут быть полезны для этой темы, от наиболее к наименее релевантному. Не ограничивай себя числом — верни все подходящие. В приоритете прямые тематические попадания (конкретные истории/факты/сценарии), затем косвенные. Посты совсем не по теме не включай.\n\nТолько номера через запятую, без пояснений.\n\n${postList}`,
      },
    ],
  }));

  const text = response.choices[0].message.content;
  const indices = (text.match(/\d+/g) || [])
    .map(n => parseInt(n) - 1)
    .filter((i, pos, arr) => i >= 0 && i < candidates.length && arr.indexOf(i) === pos);

  return indices.map(i => candidates[i]);
}

async function llmRefineCandidates(query, candidates, topK, { batchSize = 50, debug = false } = {}) {
  if (candidates.length === 0) return [];
  if (candidates.length <= batchSize) {
    const results = await llmRerank(query, candidates, topK);
    if (!debug) return results;
    return {
      results,
      debugInfo: {
        mode: 'small_batch',
        candidates: candidates.map((p) => p.id),
        final: results.map((p) => p.id),
      },
    };
  }

  const picked = new Map(); // id -> { post, score }
  const batchSelections = [];
  for (let i = 0; i < candidates.length; i += batchSize) {
    const batch = candidates.slice(i, i + batchSize);
    const relevant = await llmFilterAll(query, batch);
    batchSelections.push({
      offset: i,
      size: batch.length,
      selectedIds: relevant.map((p) => p.id),
    });
    relevant.forEach((post, rank) => {
      const score = (1 / (1 + rank)) + ((post.__baseScore ?? 0) * 0.25);
      const prev = picked.get(post.id);
      if (!prev || score > prev.score) picked.set(post.id, { post, score });
    });
  }

  const shortlistLimit = Math.max(topK * 6, 90);
  const shortlist = Array.from(picked.values())
    .sort((a, b) => b.score - a.score)
    .slice(0, shortlistLimit)
    .map(x => x.post);

  const seen = new Set(shortlist.map(p => p.id));
  for (const candidate of candidates) {
    if (shortlist.length >= shortlistLimit) break;
    if (!seen.has(candidate.id)) {
      shortlist.push(candidate);
      seen.add(candidate.id);
    }
  }

  const strictTop = await llmRerank(query, shortlist, Math.max(topK * 2, 30), { mode: 'strict' });
  const broadTop = await llmRerank(query, shortlist, Math.max(topK * 2, 30), { mode: 'broad' });
  const directMatch = buildDirectMatchRanks(shortlist, query, { corpusForDf: candidates });
  const pickedRanks = new Map(
    Array.from(picked.values())
      .sort((a, b) => b.score - a.score)
      .map(({ post }, i) => [post.id, i])
  );

  const strictRanks = new Map(strictTop.map((p, i) => [p.id, i]));
  const broadRanks = new Map(broadTop.map((p, i) => [p.id, i]));
  const baseRanks = new Map(candidates.map((p, i) => [p.id, i]));
  const directRanks = directMatch.ranks;
  const directScores = directMatch.scoreMap;

  const byId = new Map(candidates.map(p => [p.id, p]));
  const allIds = new Set([
    ...strictRanks.keys(),
    ...broadRanks.keys(),
    ...baseRanks.keys(),
    ...directRanks.keys(),
    ...pickedRanks.keys(),
  ]);
  const k = 60;

  const fused = Array.from(allIds).map((id) => {
    const strict = 1 / (k + (strictRanks.get(id) ?? shortlist.length));
    const broad = 1 / (k + (broadRanks.get(id) ?? shortlist.length));
    const base = 1 / (k + (baseRanks.get(id) ?? candidates.length));
    const direct = 1 / (k + (directRanks.get(id) ?? shortlist.length));
    const picked = 1 / (k + (pickedRanks.get(id) ?? shortlist.length));
    const directBoost = (directScores.get(id) ?? 0) * 0.05;
    const score = (1.0 * strict) + (0.85 * broad) + (0.15 * base) + (0.25 * direct) + (0.45 * picked) + directBoost;
    return { id, score };
  });

  const fusedIds = fused
    .sort((a, b) => b.score - a.score)
    .map(({ id }) => id);
  const fusedTop = fusedIds.slice(0, Math.max(topK * 2, 30));
  const llmWindow = Math.max(topK * 2, 30);
  const strictHead = strictTop.slice(0, Math.max(topK, 15)).map((p) => p.id);
  const broadHead = broadTop.slice(0, Math.max(topK, 15)).map((p) => p.id);
  const broadHeadSet = new Set(broadHead);
  const consensus = strictHead.filter((id) => broadHeadSet.has(id)).slice(0, Math.max(5, Math.floor(topK / 2)));

  const eligible = new Set([
    ...strictTop.slice(0, llmWindow).map((p) => p.id),
    ...broadTop.slice(0, llmWindow).map((p) => p.id),
    ...candidates.slice(0, topK).map((p) => p.id),
  ]);

  const prioritized = fusedIds.filter((id) => eligible.has(id));
  const strictPriority = strictHead.slice(0, Math.min(6, strictHead.length));
  const broadPriority = broadHead.slice(0, Math.min(6, broadHead.length));

  const finalIds = [...consensus, ...strictPriority, ...broadPriority, ...prioritized, ...fusedIds]
    .filter((id, idx, arr) => arr.indexOf(id) === idx)
    .slice(0, topK);

  const results = finalIds
    .map((id) => byId.get(id))
    .filter(Boolean);

  if (!debug) return results;

  return {
    results,
    debugInfo: {
      mode: 'multi_batch',
      batchSelections,
      picked: Array.from(picked.keys()),
      shortlist: shortlist.map((p) => p.id),
      strictTop: strictTop.map((p) => p.id),
      broadTop: broadTop.map((p) => p.id),
      fusedTop,
      baseAnchorIds: [],
      final: results.map((p) => p.id),
    },
  };
}

async function llmRerank(query, candidates, topK, { mode = 'strict' } = {}) {
  const postList = candidates
    .map((p, i) => {
      const text = p.textClean || p.ocrText || '';
      const snippet = buildQueryAwareSnippet(text, query, 500);
      return `[${i + 1}] ${snippet}`;
    })
    .join('\n\n');

  const response = await withRetry(() => openai.chat.completions.create({
    model: LLM_MODEL,
    temperature: 0,
    max_tokens: 300,
    messages: [
      {
        role: 'system',
        content: 'Ты эксперт по яхтингу и контент-маркетингу. Ты умеешь находить смысловые связи между постами и темами, даже если совпадений слов нет. Пост считается релевантным если он может дать идеи, факты, истории, эмоции или формулировки для создания контента на заданную тему.',
      },
      {
        role: 'user',
        content: mode === 'strict'
          ? `Тема контент-плана: "${query}"\n\nИз постов ниже выбери ${topK} наиболее релевантных. Сначала выбирай прямые попадания по теме, затем сильные косвенные. Важны посты, которые реально можно использовать для написания контента на эту тему (факты, история, сценарий, конкретика).\n\nВерни только номера постов через запятую, от наиболее к наименее релевантному. Без пояснений.\n\n${postList}`
          : `Тема контент-плана: "${query}"\n\nИз постов ниже выбери ${topK} постов, которые могут быть полезны для раскрытия темы даже косвенно: опыт, эмоции, детали маршрутов, бытовые/организационные моменты, смежные истории. Прямые попадания тоже включай, но не игнорируй хорошие косвенные.\n\nВерни только номера постов через запятую, от наиболее к наименее релевантному. Без пояснений.\n\n${postList}`,
      },
    ],
  }));

  const text = response.choices[0].message.content;
  const indices = (text.match(/\d+/g) || [])
    .map(n => parseInt(n) - 1)
    .filter((i, pos, arr) => i >= 0 && i < candidates.length && arr.indexOf(i) === pos);

  const seen = new Set(indices);
  const rest = candidates.map((_, i) => i).filter(i => !seen.has(i));
  return [...indices, ...rest].slice(0, topK).map(i => candidates[i]);
}

async function generateParaphrases(query, count = 8) {
  const normalizedQuery = query.toLowerCase().trim();
  const cacheKey = `${count}:${normalizedQuery}`;
  if (paraphrasesCache.has(cacheKey)) return paraphrasesCache.get(cacheKey);

  const cached = await diskCacheGet(`para${count}`, normalizedQuery);
  if (cached) { paraphrasesCache.set(cacheKey, cached); return cached; }

  const response = await withRetry(() => openai.chat.completions.create({
    model: LLM_MODEL,
    temperature: 0,
    max_tokens: 400,
    messages: [{
      role: 'user',
      content: `Перефразируй этот запрос ${count} разными способами, используя максимально разные слова и выражения: разговорный стиль соцсетей, нарративные ключевые слова, синонимы темы. Каждая парафраза на новой строке, без нумерации и пояснений. Запрос: "${query}"`,
    }],
  }));

  const paraphrases = response.choices[0].message.content
    .split('\n')
    .map(s => s.trim())
    .filter(Boolean)
    .slice(0, count);

  paraphrasesCache.set(cacheKey, paraphrases);
  await diskCacheSet(`para${count}`, normalizedQuery, paraphrases);
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
    const response = await withRetry(() => openai.chat.completions.create({
      model: LLM_MODEL,
      temperature: 0,
      max_tokens: 100,
      messages: [
        {
          role: 'user',
          content: `Ты помогаешь искать посты о яхтинге. Дай список из 5-7 синонимов и связанных терминов для запроса: "${query}". Только слова через запятую, без пояснений.`,
        },
      ],
    }));

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
