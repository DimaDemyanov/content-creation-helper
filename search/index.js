import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import winkBM25 from 'wink-bm25-text-search';
import winkNLP from 'wink-nlp-utils';
import OpenAI from 'openai';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const POSTS_DIR = path.join(__dirname, '../data/posts');

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Кэш синонимов чтобы не вызывать Claude на одинаковые запросы
const synonymsCache = new Map();

export async function search(query, topK = 10) {
  const allPosts = await loadAllPosts();
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

async function loadAllPosts() {
  const posts = [];

  let files;
  try {
    files = await fs.readdir(POSTS_DIR);
  } catch {
    return [];
  }

  for (const file of files) {
    if (!file.endsWith('.json')) continue;
    try {
      const raw = await fs.readFile(path.join(POSTS_DIR, file), 'utf-8');
      const parsed = JSON.parse(raw);
      posts.push(...parsed);
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
    winkNLP.string.tokenize0,
    winkNLP.tokens.removeWords,
    winkNLP.tokens.stem,
  ]);

  engine.consolidate();

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
    const synonyms = raw
      .split(',')
      .map(s => s.trim())
      .filter(Boolean);

    const terms = [query, ...synonyms];
    synonymsCache.set(cacheKey, terms);
    return terms;
  } catch {
    return [query];
  }
}

export async function getStats() {
  const files = await fs.readdir(POSTS_DIR).catch(() => []);
  const stats = {};

  for (const file of files) {
    if (!file.endsWith('.json')) continue;
    try {
      const raw = await fs.readFile(path.join(POSTS_DIR, file), 'utf-8');
      const posts = JSON.parse(raw);
      const channel = file.replace('.json', '');
      stats[channel] = posts.length;
    } catch {}
  }

  return stats;
}
