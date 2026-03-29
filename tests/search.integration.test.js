/**
 * Интеграционные тесты поиска — реальные вызовы OpenAI и реальные данные.
 *
 * Запуск:
 *   npm run test:integration
 *
 * Требует:
 *   - OPENAI_API_KEY в .env
 *   - Данные в data/posts/
 *   - Эмбеддинги в data/embeddings/ (для vectorSearch и hybridSearch)
 *     Если эмбеддинги не сгенерированы: node --env-file=.env scripts/backfill-embeddings.js
 */

import path from 'path';
import { fileURLToPath } from 'url';
import { describe, it, expect } from 'vitest';
import { search, vectorSearch, hybridSearch, vectorSearchHyDE, hybridSearchHyDE, hybridSearchFull, hybridSearchRRF } from '../search/index.js';
import { loadAllEmbeddings, loadAllChunkEmbeddings } from '../search/embeddings.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const POSTS_DIR = path.join(__dirname, 'fixtures/posts');
const EMBEDDINGS_DIR = path.join(__dirname, 'fixtures/embeddings');
const CHUNK_EMBEDDINGS_DIR = path.join(__dirname, 'fixtures/chunk-embeddings');

// Ground truth: полная разметка по каждой теме.
// Метрика Hit@5: хотя бы 1 из relevant попал в top-5.
const TOPICS = [
  {
    query: 'Почему яхтенные путешествия — один из лучших форматов отдыха',
    relevant: [
      'ig_anton_timk_DGibmz6v026', 'ig_anton_timk_DTNpIZXiCbG', 'ig_anton_timk_C8cKk1bC2to',
      'ig_anton_timk_DSK3_lRCGYv', 'ig_anton_timk_DWY0COXF93M',
      'ig_clevel.yacht_DVwJ3JUj9qo', 'ig_clevel.yacht_DUivkh2iqD3', 'ig_clevel.yacht_DOlH6fwijjn',
      'ig_clevel.yacht_DOvz8Xfiszo', 'ig_clevel.yacht_DRRtuieCl4k', 'ig_clevel.yacht_DFNka1_qMFk',
      'ig_clevel.yacht_DPMO2zACrOr', 'ig_clevel.yacht_DOOqCSuirFF', 'ig_clevel.yacht_DT_DbpTCgGm',
      'ig_clevel.yacht_DQ_oXPlD3Kg', 'ig_clevel.yacht_DUTaVm7D4kR', 'ig_clevel.yacht_DSc5KoijmDw',
      'ig_clevel.yacht_DLC3sJcqR7s',
      'meetingplace_news_176', 'meetingplace_news_28', 'meetingplace_news_96',
      'meetingplace_news_3', 'meetingplace_news_99',
      'silavetrasila_3938', 'silavetrasila_5403',
    ],
  },
  {
    query: 'До конца предпродажи осталось 6 дней',
    relevant: [
      'regataveka_35', 'regataveka_180',
      'ig_clevel.yacht_DOlH6fwijjn', 'ig_clevel.yacht_DIbHiPlK7al', 'ig_clevel.yacht_DNdN49UKGXn',
      'ig_clevel.yacht_DOvz8Xfiszo', 'ig_clevel.yacht_DU0tWdgio0F', 'ig_clevel.yacht_DPgk_DUDT8C',
      'ig_clevel.yacht_DVL1SPljfIH', 'ig_clevel.yacht_DRe1kpyCjbD', 'ig_clevel.yacht_DTxfTDojTmI',
      'ig_clevel.yacht_DSnQ7bxCjaQ',
      'ig_anton_timk_C1g6wmvrbte', 'ig_anton_timk_C1ttKHliZQJ',
      'silavetrasila_8156', 'silavetrasila_8184', 'silavetrasila_6542', 'silavetrasila_6596',
      'meetingplace_news_344',
    ],
  },
  {
    query: 'Как проходит один день на яхте',
    relevant: [
      'meetingplace_news_32', 'meetingplace_news_367', 'meetingplace_news_314',
      'regataveka_64', 'regataveka_83', 'regataveka_88', 'regataveka_95', 'regataveka_112', 'regataveka_120',
      'ig_clevel.yacht_DOG2eZLilZp', 'ig_clevel.yacht_DFNka1_qMFk', 'ig_clevel.yacht_DGgRXqaqMSO',
      'silavetrasila_6286',
    ],
  },
  {
    query: 'Как выйти замуж на яхте',
    relevant: [
      'seapinta_549', 'ig_clevel.yacht_DPvo6ACCtm7',
    ],
  },
  {
    query: 'Как люди обычно попадают на свою первую яхту',
    relevant: [
      'silavetrasila_7420', 'silavetrasila_6905', 'silavetrasila_6630', 'silavetrasila_5251', 'silavetrasila_4552',
      'ig_clevel.yacht_DRRtuieCl4k', 'ig_clevel.yacht_DOOqCSuirFF', 'ig_clevel.yacht_DHdhW5DKmOZ',
      'ig_clevel.yacht_DR7XTb8CqdW',
      'ig_anton_timk_DUz0RkgDy6J',
      'meetingplace_news_148',
    ],
  },
  {
    query: '5 вещей, которые люди не ожидают от яхтенных путешествий',
    relevant: [
      'LyubimovaEvgeniya_1510', 'LyubimovaEvgeniya_2122', 'LyubimovaEvgeniya_1656',
      'ig_clevel.yacht_DR7XTb8CqdW', 'ig_clevel.yacht_DUivkh2iqD3',
      'ig_clevel.yacht_DLz0Du5KCZv', 'ig_clevel.yacht_DPMO2zACrOr',
    ],
  },
  {
    query: 'Что будет на регате в Турции (5 лодок)',
    relevant: [
      'regataveka_40', 'regataveka_29', 'regataveka_64', 'regataveka_70', 'regataveka_69',
      'regataveka_83', 'regataveka_88', 'regataveka_95', 'regataveka_112', 'regataveka_120',
      'regataveka_73', 'regataveka_82', 'regataveka_63',
      'ig_clevel.yacht_DGgRXqaqMSO', 'ig_clevel.yacht_DSzdcLSDiVi', 'ig_clevel.yacht_DRe1kpyCjbD',
      'ig_clevel.yacht_DLC3sJcqR7s', 'ig_clevel.yacht_DEXaJnGKeQv', 'ig_clevel.yacht_DSc5KoijmDw',
      'ig_clevel.yacht_DVRPXpmiiZS', 'ig_clevel.yacht_DIWNtd3KIDo',
      'silavetrasila_7742',
    ],
  },
  {
    query: 'История: акула и камера',
    relevant: [
      'meetingplace_news_137',
      'ig_anton_timk_DQzdFtlExmi', 'ig_anton_timk_DUVqzSZD87t', 'ig_anton_timk_C2cpF7kNFtM',
      'seapinta_920',
    ],
  },
  {
    query: 'Самые красивые бухты Турции',
    relevant: [
      'ig_anton_timk_DGibmz6v026',
      'LyubimovaEvgeniya_732', 'LyubimovaEvgeniya_386',
      'seapinta_944',
      'meetingplace_news_40',
      'silavetrasila_7742',
      'ig_clevel.yacht_DSzdcLSDiVi',
      'regataveka_120',
    ],
  },
  {
    query: 'История: как мы сели на мель',
    relevant: [
      'LyubimovaEvgeniya_2118', 'LyubimovaEvgeniya_514', 'LyubimovaEvgeniya_381',
      'LyubimovaEvgeniya_1865', 'LyubimovaEvgeniya_479', 'LyubimovaEvgeniya_2125',
      'seapinta_882', 'ig_anton_timk_DV3RPRAD3hu',
    ],
  },
];

const TOP_K = 5;

function hit(results, relevant) {
  const ids = new Set(results.map(r => r.id));
  return relevant.some(id => ids.has(id));
}

function firstHitPosition(results, relevant) {
  return results.findIndex(r => relevant.includes(r.id)) + 1; // 1-based, 0 если не найдено
}

// --- BM25 ---

describe('BM25 поиск (search)', () => {
  for (const { query, relevant } of TOPICS) {
    it(query, async () => {
      const results = await search(query, TOP_K, { postsDir: POSTS_DIR });
      const found = hit(results, relevant);
      const pos = firstHitPosition(results, relevant);
      if (!found) {
        console.log(`  ❌ BM25 не нашёл. top-${TOP_K}:`, results.map(r => r.id));
      } else {
        console.log(`  ✅ BM25 pos ${pos}: ${results[pos - 1].id}`);
      }
      expect(found).toBe(true);
    }, 30_000);
  }
});

// --- Vector ---

describe('Векторный поиск (vectorSearch)', () => {
  it('эмбеддинги загружены', async () => {
    const embeddings = await loadAllEmbeddings(EMBEDDINGS_DIR);
    expect(embeddings.size).toBeGreaterThan(0);
  });

  for (const { query, relevant } of TOPICS) {
    it(query, async () => {
      const results = await vectorSearch(query, TOP_K, { postsDir: POSTS_DIR, embeddingsDir: EMBEDDINGS_DIR });
      const found = hit(results, relevant);
      const pos = firstHitPosition(results, relevant);
      if (!found) {
        console.log(`  ❌ Vector не нашёл. top-${TOP_K}:`, results.map(r => r.id));
      } else {
        console.log(`  ✅ Vector pos ${pos}: ${results[pos - 1].id}`);
      }
      expect(found).toBe(true);
    }, 30_000);
  }
});

// --- Hybrid ---

describe('Гибридный поиск (hybridSearch)', () => {
  for (const { query, relevant } of TOPICS) {
    it(query, async () => {
      const results = await hybridSearch(query, TOP_K, { postsDir: POSTS_DIR, embeddingsDir: EMBEDDINGS_DIR });
      const found = hit(results, relevant);
      const pos = firstHitPosition(results, relevant);
      if (!found) {
        console.log(`  ❌ Hybrid не нашёл. top-${TOP_K}:`, results.map(r => r.id));
      } else {
        console.log(`  ✅ Hybrid pos ${pos}: ${results[pos - 1].id}`);
      }
      expect(found).toBe(true);
    }, 30_000);
  }
});

// --- Vector Chunked ---

describe('Векторный поиск по чанкам (vectorSearchChunked)', () => {
  it('чанк-эмбеддинги загружены', async () => {
    const chunkMap = await loadAllChunkEmbeddings(CHUNK_EMBEDDINGS_DIR);
    expect(chunkMap.size).toBeGreaterThan(0);
  });

  for (const { query, relevant } of TOPICS) {
    it(query, async () => {
      const results = await vectorSearchChunked(query, TOP_K, { postsDir: POSTS_DIR, chunkEmbeddingsDir: CHUNK_EMBEDDINGS_DIR });
      const found = hit(results, relevant);
      const pos = firstHitPosition(results, relevant);
      if (!found) {
        console.log(`  ❌ VectorChunked не нашёл. top-${TOP_K}:`, results.map(r => r.id));
      } else {
        console.log(`  ✅ VectorChunked pos ${pos}: ${results[pos - 1].id}`);
      }
      expect(found).toBe(true);
    }, 30_000);
  }
});

// --- Hybrid Chunked ---

describe('Гибридный поиск по чанкам (hybridSearchChunked)', () => {
  for (const { query, relevant } of TOPICS) {
    it(query, async () => {
      const results = await hybridSearchChunked(query, TOP_K, { postsDir: POSTS_DIR, chunkEmbeddingsDir: CHUNK_EMBEDDINGS_DIR });
      const found = hit(results, relevant);
      const pos = firstHitPosition(results, relevant);
      if (!found) {
        console.log(`  ❌ HybridChunked не нашёл. top-${TOP_K}:`, results.map(r => r.id));
      } else {
        console.log(`  ✅ HybridChunked pos ${pos}: ${results[pos - 1].id}`);
      }
      expect(found).toBe(true);
    }, 30_000);
  }
});

// --- HyDE ---

describe('Векторный поиск HyDE (vectorSearchHyDE)', () => {
  for (const { query, relevant } of TOPICS) {
    it(query, async () => {
      const results = await vectorSearchHyDE(query, TOP_K, { postsDir: POSTS_DIR, embeddingsDir: EMBEDDINGS_DIR });
      const found = hit(results, relevant);
      const pos = firstHitPosition(results, relevant);
      if (!found) {
        console.log(`  ❌ HyDE не нашёл. top-${TOP_K}:`, results.map(r => r.id));
      } else {
        console.log(`  ✅ HyDE pos ${pos}: ${results[pos - 1].id}`);
      }
      expect(found).toBe(true);
    }, 30_000);
  }
});

describe('Гибридный поиск HyDE (hybridSearchHyDE)', () => {
  for (const { query, relevant } of TOPICS) {
    it(query, async () => {
      const results = await hybridSearchHyDE(query, TOP_K, { postsDir: POSTS_DIR, embeddingsDir: EMBEDDINGS_DIR });
      const found = hit(results, relevant);
      const pos = firstHitPosition(results, relevant);
      if (!found) {
        console.log(`  ❌ HybridHyDE не нашёл. top-${TOP_K}:`, results.map(r => r.id));
      } else {
        console.log(`  ✅ HybridHyDE pos ${pos}: ${results[pos - 1].id}`);
      }
      expect(found).toBe(true);
    }, 30_000);
  }
});

// --- Сводная таблица ---

describe('Сводный отчёт Hit@5', () => {
  it('выводит таблицу по всем методам', async () => {
    const rows = [];
    for (const { query, relevant } of TOPICS) {
      const [bm25, vec, hybrid, hyde, rrf] = await Promise.all([
        search(query, TOP_K, { postsDir: POSTS_DIR }),
        vectorSearch(query, TOP_K, { postsDir: POSTS_DIR, embeddingsDir: EMBEDDINGS_DIR }),
        hybridSearch(query, TOP_K, { postsDir: POSTS_DIR, embeddingsDir: EMBEDDINGS_DIR }),
        vectorSearchHyDE(query, TOP_K, { postsDir: POSTS_DIR, embeddingsDir: EMBEDDINGS_DIR }),
        hybridSearchRRF(query, TOP_K, { postsDir: POSTS_DIR, embeddingsDir: EMBEDDINGS_DIR }),
      ]);
      rows.push({
        query:  query.slice(0, 38),
        bm25:   hit(bm25, relevant)   ? `✅${firstHitPosition(bm25, relevant)}`   : '❌',
        vec:    hit(vec, relevant)    ? `✅${firstHitPosition(vec, relevant)}`    : '❌',
        hybrid: hit(hybrid, relevant) ? `✅${firstHitPosition(hybrid, relevant)}` : '❌',
        hyde:   hit(hyde, relevant)   ? `✅${firstHitPosition(hyde, relevant)}`   : '❌',
        rrf:    hit(rrf, relevant)    ? `✅${firstHitPosition(rrf, relevant)}`    : '❌',
      });
    }
    console.table(rows);

    const score = (col) => rows.filter(r => r[col].startsWith('✅')).length;
    console.log(`\nИтого Hit@${TOP_K}:`);
    console.log(`  BM25       = ${score('bm25')}/10`);
    console.log(`  Vector     = ${score('vec')}/10`);
    console.log(`  Hybrid     = ${score('hybrid')}/10`);
    console.log(`  HyDE       = ${score('hyde')}/10`);
    console.log(`  RRF ★      = ${score('rrf')}/10`);
  }, 180_000);
});
