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

import { describe, it, expect } from 'vitest';
import { search, vectorSearch, hybridSearch } from '../search/index.js';
import { loadAllEmbeddings } from '../search/embeddings.js';

// Ground truth: для каждой темы — какие посты должны попасть в top-5
// Достаточно попадания хотя бы одного из списка (Hit@5)
const TOPICS = [
  {
    query: 'Почему яхтенные путешествия — один из лучших форматов отдыха',
    relevant: ['silavetrasila_5403', 'silavetrasila_3938', 'meetingplace_news_176', 'meetingplace_news_3'],
  },
  {
    query: 'До конца предпродажи осталось 6 дней',
    relevant: ['regataveka_35', 'silavetrasila_8156', 'silavetrasila_8184'],
  },
  {
    query: 'Как проходит один день на яхте',
    relevant: ['meetingplace_news_367', 'meetingplace_news_32', 'regataveka_64'],
  },
  {
    query: 'Как выйти замуж на яхте',
    relevant: ['ig_clevel.yacht_DPvo6ACCtm7', 'seapinta_549'],
  },
  {
    query: 'Как люди обычно попадают на свою первую яхту',
    relevant: ['silavetrasila_6630', 'silavetrasila_6905', 'silavetrasila_5251', 'silavetrasila_7420'],
  },
  {
    query: '5 вещей, которые люди не ожидают от яхтенных путешествий',
    relevant: ['LyubimovaEvgeniya_2122', 'LyubimovaEvgeniya_1510', 'LyubimovaEvgeniya_1656'],
  },
  {
    query: 'Что будет на регате в Турции (5 лодок)',
    relevant: ['ig_clevel.yacht_DSzdcLSDiVi', 'ig_clevel.yacht_DRe1kpyCjbD', 'silavetrasila_7742'],
  },
  {
    query: 'История: акула и камера',
    relevant: ['meetingplace_news_137'],
  },
  {
    query: 'Самые красивые бухты Турции',
    relevant: ['ig_anton_timk_DGibmz6v026', 'LyubimovaEvgeniya_386', 'LyubimovaEvgeniya_732'],
  },
  {
    query: 'История: как мы сели на мель',
    relevant: ['LyubimovaEvgeniya_2118', 'LyubimovaEvgeniya_1865'],
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
      const results = await search(query, TOP_K);
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
    const embeddings = await loadAllEmbeddings();
    expect(embeddings.size).toBeGreaterThan(0);
  });

  for (const { query, relevant } of TOPICS) {
    it(query, async () => {
      const results = await vectorSearch(query, TOP_K);
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
      const results = await hybridSearch(query, TOP_K);
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

// --- Сводная таблица ---

describe('Сводный отчёт Hit@5', () => {
  it('выводит таблицу по всем методам', async () => {
    const rows = [];
    for (const { query, relevant } of TOPICS) {
      const [bm25, vec, hybrid] = await Promise.all([
        search(query, TOP_K),
        vectorSearch(query, TOP_K),
        hybridSearch(query, TOP_K),
      ]);
      rows.push({
        query: query.slice(0, 45),
        bm25: hit(bm25, relevant) ? `✅ pos${firstHitPosition(bm25, relevant)}` : '❌',
        vec: hit(vec, relevant) ? `✅ pos${firstHitPosition(vec, relevant)}` : '❌',
        hybrid: hit(hybrid, relevant) ? `✅ pos${firstHitPosition(hybrid, relevant)}` : '❌',
      });
    }
    console.table(rows);

    const bm25Score = rows.filter(r => r.bm25.startsWith('✅')).length;
    const vecScore = rows.filter(r => r.vec.startsWith('✅')).length;
    const hybridScore = rows.filter(r => r.hybrid.startsWith('✅')).length;
    console.log(`\nИтого Hit@${TOP_K}: BM25=${bm25Score}/10  Vector=${vecScore}/10  Hybrid=${hybridScore}/10`);
  }, 120_000);
});
