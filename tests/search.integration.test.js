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

// Ground truth по каждой теме.
//
// relevant  — все посты которые релевантны теме (широкая разметка)
// mustFind  — 2–3 самых прямых попадания (строгий ground truth)
//
// Метрики:
//   Hit@5(relevant)  — хотя бы 1 из relevant в top-5     (мягко, показывает recall)
//   Hit@5(mustFind)  — хотя бы 1 из mustFind в top-5     (строго, показывает precision)
//   MRR(relevant)    — 1/rank первого попадания, среднее  (показывает ранг, не только факт)
const TOPICS = [
  {
    query: 'Почему яхтенные путешествия — один из лучших форматов отдыха',
    mustFind: ['silavetrasila_3938', 'ig_clevel.yacht_DOG2eZLilZp'],
    relevant: [
      'ig_anton_timk_DGibmz6v026', 'ig_anton_timk_DTNpIZXiCbG', 'ig_anton_timk_C8cKk1bC2to',
      'ig_anton_timk_DSK3_lRCGYv', 'ig_anton_timk_DWY0COXF93M',
      'ig_clevel.yacht_DVwJ3JUj9qo', 'ig_clevel.yacht_DUivkh2iqD3', 'ig_clevel.yacht_DOlH6fwijjn',
      'ig_clevel.yacht_DOvz8Xfiszo', 'ig_clevel.yacht_DRRtuieCl4k', 'ig_clevel.yacht_DFNka1_qMFk',
      'ig_clevel.yacht_DPMO2zACrOr', 'ig_clevel.yacht_DOOqCSuirFF', 'ig_clevel.yacht_DT_DbpTCgGm',
      'ig_clevel.yacht_DQ_oXPlD3Kg', 'ig_clevel.yacht_DUTaVm7D4kR', 'ig_clevel.yacht_DSc5KoijmDw',
      'ig_clevel.yacht_DLC3sJcqR7s', 'ig_clevel.yacht_DEXaJnGKeQv', 'ig_clevel.yacht_DU0tWdgio0F',
      'ig_clevel.yacht_DVBaeYfjcex', 'ig_clevel.yacht_DWRnwajj9y7', 'ig_clevel.yacht_DOG2eZLilZp',
      'meetingplace_news_176', 'meetingplace_news_28', 'meetingplace_news_40', 'meetingplace_news_96',
      'meetingplace_news_3', 'meetingplace_news_99',
      'silavetrasila_3938', 'silavetrasila_5403',
      'regataveka_29', 'regataveka_70',
    ],
  },
  {
    query: 'Как проходит один день на яхте',
    mustFind: ['meetingplace_news_367', 'meetingplace_news_32', 'regataveka_64'],
    relevant: [
      'meetingplace_news_32', 'meetingplace_news_367', 'meetingplace_news_314',
      'regataveka_64', 'regataveka_83', 'regataveka_88', 'regataveka_95', 'regataveka_112', 'regataveka_120',
      'ig_clevel.yacht_DOG2eZLilZp', 'ig_clevel.yacht_DFNka1_qMFk', 'ig_clevel.yacht_DGgRXqaqMSO',
      'silavetrasila_6286',
    ],
  },
  {
    query: 'Как выйти замуж на яхте',
    mustFind: ['seapinta_549', 'ig_clevel.yacht_DPvo6ACCtm7'],
    relevant: ['seapinta_549', 'ig_clevel.yacht_DPvo6ACCtm7'],
  },
  {
    query: 'Как люди обычно попадают на свою первую яхту',
    mustFind: ['silavetrasila_6630', 'silavetrasila_5251', 'ig_clevel.yacht_DHdhW5DKmOZ'],
    relevant: [
      'silavetrasila_7420', 'silavetrasila_6905', 'silavetrasila_6630', 'silavetrasila_5251',
      'ig_clevel.yacht_DRRtuieCl4k', 'ig_clevel.yacht_DOOqCSuirFF', 'ig_clevel.yacht_DHdhW5DKmOZ',
      'ig_clevel.yacht_DR7XTb8CqdW',
      'ig_anton_timk_DUz0RkgDy6J',
      'meetingplace_news_148',
    ],
  },
  {
    query: '5 вещей, которые люди не ожидают от яхтенных путешествий',
    mustFind: ['LyubimovaEvgeniya_2122', 'LyubimovaEvgeniya_1510'],
    relevant: [
      'LyubimovaEvgeniya_1510', 'LyubimovaEvgeniya_2122', 'LyubimovaEvgeniya_1656',
      'ig_clevel.yacht_DR7XTb8CqdW', 'ig_clevel.yacht_DUivkh2iqD3',
      'ig_clevel.yacht_DLz0Du5KCZv', 'ig_clevel.yacht_DPMO2zACrOr',
      'ig_clevel.yacht_DWRnwajj9y7',
    ],
  },
  {
    query: 'Что будет на регате в Турции (5 лодок)',
    mustFind: ['ig_clevel.yacht_DSzdcLSDiVi', 'ig_clevel.yacht_DRe1kpyCjbD', 'silavetrasila_7742'],
    relevant: [
      'regataveka_40', 'regataveka_29', 'regataveka_64', 'regataveka_70', 'regataveka_69',
      'regataveka_83', 'regataveka_88', 'regataveka_95', 'regataveka_112', 'regataveka_120',
      'regataveka_73', 'regataveka_82', 'regataveka_63',
      'ig_clevel.yacht_DGgRXqaqMSO', 'ig_clevel.yacht_DSzdcLSDiVi', 'ig_clevel.yacht_DRe1kpyCjbD',
      'ig_clevel.yacht_DLC3sJcqR7s', 'ig_clevel.yacht_DEXaJnGKeQv', 'ig_clevel.yacht_DSc5KoijmDw',
      'ig_clevel.yacht_DVRPXpmiiZS', 'ig_clevel.yacht_DIWNtd3KIDo',
      'ig_clevel.yacht_DVBaeYfjcex', 'ig_clevel.yacht_DT_DbpTCgGm',
      'ig_clevel.yacht_DUd1HgGimL6', 'ig_clevel.yacht_DVL1SPljfIH',
      'silavetrasila_7742',
    ],
  },
  {
    query: 'История: акула и камера',
    mustFind: ['meetingplace_news_137'],
    relevant: [
      'meetingplace_news_137',
      'ig_anton_timk_DQzdFtlExmi', 'ig_anton_timk_DUVqzSZD87t', 'ig_anton_timk_C2cpF7kNFtM',
      'seapinta_920',
    ],
  },
  {
    query: 'Самые красивые бухты Турции',
    mustFind: ['silavetrasila_558', 'silavetrasila_1078', 'ig_anton_timk_DGibmz6v026'],
    relevant: [
      'silavetrasila_558', 'silavetrasila_1078', 'silavetrasila_7742',
      'ig_anton_timk_DGibmz6v026',
      'seapinta_944',
      'meetingplace_news_40', 'meetingplace_news_28',
      'ig_clevel.yacht_DSzdcLSDiVi', 'ig_clevel.yacht_DT_DbpTCgGm',
      'regataveka_120',
    ],
  },
  {
    query: 'История: как мы сели на мель',
    mustFind: ['LyubimovaEvgeniya_2118', 'LyubimovaEvgeniya_2021'],
    relevant: [
      'LyubimovaEvgeniya_2118', 'LyubimovaEvgeniya_2021',
      'LyubimovaEvgeniya_1865',
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

// MRR = 1/rank первого попадания (0 если не найдено)
function rr(results, relevant) {
  const pos = firstHitPosition(results, relevant);
  return pos > 0 ? 1 / pos : 0;
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
    const mrrs = [];
    for (const { query, relevant, mustFind } of TOPICS) {
      const [bm25, vec, hybrid, hyde, rrf] = await Promise.all([
        search(query, TOP_K, { postsDir: POSTS_DIR }),
        vectorSearch(query, TOP_K, { postsDir: POSTS_DIR, embeddingsDir: EMBEDDINGS_DIR }),
        hybridSearch(query, TOP_K, { postsDir: POSTS_DIR, embeddingsDir: EMBEDDINGS_DIR }),
        vectorSearchHyDE(query, TOP_K, { postsDir: POSTS_DIR, embeddingsDir: EMBEDDINGS_DIR }),
        hybridSearchRRF(query, TOP_K, { postsDir: POSTS_DIR, embeddingsDir: EMBEDDINGS_DIR }),
      ]);
      const fmt = (results, rel, must) => {
        const pos = firstHitPosition(results, rel);
        const strict = hit(results, must) ? '★' : ' ';
        return pos > 0 ? `${strict}✅${pos}` : '❌';
      };
      rows.push({
        query:  query.slice(0, 36),
        bm25:   fmt(bm25,   relevant, mustFind),
        vec:    fmt(vec,    relevant, mustFind),
        hybrid: fmt(hybrid, relevant, mustFind),
        hyde:   fmt(hyde,   relevant, mustFind),
        rrf:    fmt(rrf,    relevant, mustFind),
      });
      mrrs.push({
        bm25:   rr(bm25,   relevant),
        vec:    rr(vec,    relevant),
        hybrid: rr(hybrid, relevant),
        hyde:   rr(hyde,   relevant),
        rrf:    rr(rrf,    relevant),
      });
    }

    console.log('\nЛегенда: ★✅N = mustFind найден на позиции N | ✅N = relevant найден | ❌ = не найден\n');
    console.table(rows);

    const hit5 = (col) => rows.filter(r => r[col].includes('✅')).length;
    const hit5strict = (col) => rows.filter(r => r[col].startsWith('★')).length;
    const meanRR = (col) => (mrrs.reduce((s, r) => s + r[col], 0) / mrrs.length).toFixed(2);

    const N = TOPICS.length;
    console.log(`\n${'Метод'.padEnd(10)} Hit@5(any)  Hit@5(must)  MRR`);
    console.log('─'.repeat(42));
    for (const col of ['bm25', 'vec', 'hybrid', 'hyde', 'rrf']) {
      console.log(`${col.padEnd(10)} ${String(hit5(col)+'/'+N).padEnd(12)} ${String(hit5strict(col)+'/'+N).padEnd(13)} ${meanRR(col)}`);
    }
  }, 180_000);
});
