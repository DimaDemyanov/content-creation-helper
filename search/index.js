import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import winkBM25 from 'wink-bm25-text-search';
import winkNLP from 'wink-nlp-utils';
import OpenAI from 'openai';
import { generateEmbedding, loadAllEmbeddings, loadAllChunkEmbeddings, cosineSimilarity } from './embeddings.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_POSTS_DIR = path.join(__dirname, '../data/posts');

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Кэш синонимов чтобы не вызывать OpenAI на одинаковые запросы
const synonymsCache = new Map();

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

  const queryVector = await generateEmbedding(query);

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

  const queryVector = await generateEmbedding(query);
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
    const queryVector = await generateEmbedding(query);
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
    const queryVector = await generateEmbedding(query);
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
    winkNLP.tokens.removeWords,
    winkNLP.tokens.stem,
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
