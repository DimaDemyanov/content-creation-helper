/**
 * Интеграционные тесты поиска — реальные вызовы OpenAI и реальные данные.
 *
 * Запуск:
 *   npm run test:integration
 *
 * Требует:
 *   - OPENAI_API_KEY в .env
 *   - Посты в tests/fixtures/posts/
 *   - Эмбеддинги в tests/fixtures/embeddings/ (для mqHybridRRF / re-rank)
 *     Если эмбеддинги не сгенерированы: node --env-file=.env scripts/backfill-embeddings.js
 */

import path from 'path';
import { fileURLToPath } from 'url';
import { describe, it, expect } from 'vitest';
import { search, mqHybridRRF, searchWithRerank, debugMqHybridSteps, MQ_HYBRID_STEPS } from '../search/index.js';
import { loadAllEmbeddings } from '../search/embeddings.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const POSTS_DIR = path.join(__dirname, 'fixtures/posts');
const EMBEDDINGS_DIR = path.join(__dirname, 'fixtures/embeddings');

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
    mustFind: ['silavetrasila_3938', 'ig_anton_timk_DTNpIZXiCbG'],
    relevant: [
      'ig_anton_timk_DGibmz6v026', 'ig_anton_timk_DTNpIZXiCbG', 'ig_anton_timk_C8cKk1bC2to',
      'ig_anton_timk_DSK3_lRCGYv', 'ig_anton_timk_DWY0COXF93M',
      'ig_anton_timk_DSfvT1PCAM6', 'ig_anton_timk_DEzkhmYCasQ', 'ig_anton_timk_DIdkxXAvo-z',
      'ig_clevel.yacht_DVwJ3JUj9qo', 'ig_clevel.yacht_DUivkh2iqD3', 'ig_clevel.yacht_DOlH6fwijjn',
      'ig_clevel.yacht_DOvz8Xfiszo', 'ig_clevel.yacht_DRRtuieCl4k', 'ig_clevel.yacht_DFNka1_qMFk',
      'ig_clevel.yacht_DPMO2zACrOr', 'ig_clevel.yacht_DOOqCSuirFF',
      'ig_clevel.yacht_DQ_oXPlD3Kg', 'ig_clevel.yacht_DUTaVm7D4kR',
      'ig_clevel.yacht_DLC3sJcqR7s', 'ig_clevel.yacht_DEXaJnGKeQv',
      'ig_clevel.yacht_DVBaeYfjcex', 'ig_clevel.yacht_DWRnwajj9y7', 'ig_clevel.yacht_DOG2eZLilZp',
      'meetingplace_news_176', 'meetingplace_news_28', 'meetingplace_news_99',
      'silavetrasila_3938', 'silavetrasila_5403',
      'regataveka_70',
    ],
  },
  {
    query: 'Как проходит один день на яхте',
    mustFind: ['meetingplace_news_367', 'meetingplace_news_32', 'regataveka_64'],
    relevant: [
      'meetingplace_news_32', 'meetingplace_news_367', 'meetingplace_news_314', 'meetingplace_news_164',
      'regataveka_64', 'regataveka_83', 'regataveka_88', 'regataveka_95', 'regataveka_112', 'regataveka_120',
      'ig_clevel.yacht_DOG2eZLilZp', 'ig_clevel.yacht_DFNka1_qMFk', 'ig_clevel.yacht_DGgRXqaqMSO',
      'silavetrasila_6286',
      'seapinta_116', 'seapinta_117',
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
      'ig_anton_timk_DUz0RkgDy6J', 'ig_anton_timk_DRUspFLj3gJ',
      'meetingplace_news_148',
    ],
  },
  {
    query: '5 вещей, которые люди не ожидают от яхтенных путешествий',
    mustFind: ['LyubimovaEvgeniya_2122', 'LyubimovaEvgeniya_1510'],
    relevant: [
      'LyubimovaEvgeniya_1510', 'LyubimovaEvgeniya_2122', 'LyubimovaEvgeniya_1656',
      'ig_clevel.yacht_DR7XTb8CqdW', 'ig_clevel.yacht_DUivkh2iqD3', 'ig_clevel.yacht_DFNka1_qMFk',
      'ig_clevel.yacht_DLz0Du5KCZv', 'ig_clevel.yacht_DPMO2zACrOr',
      'ig_clevel.yacht_DWRnwajj9y7',
      'ig_anton_timk_DSK3_lRCGYv',
    ],
  },
  {
    query: 'Что будет на регате в Турции (5 лодок)',
    mustFind: ['ig_clevel.yacht_DSzdcLSDiVi', 'ig_clevel.yacht_DRe1kpyCjbD', 'silavetrasila_7742'],
    relevant: [
      'regataveka_40', 'regataveka_29', 'regataveka_64', 'regataveka_70', 'regataveka_69',
      'regataveka_74',
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
      'regataveka_74',
      'ig_clevel.yacht_DSzdcLSDiVi', 'ig_clevel.yacht_DT_DbpTCgGm',
    ],
  },
  {
    query: 'История: как мы сели на мель',
    mustFind: ['LyubimovaEvgeniya_2118', 'LyubimovaEvgeniya_2021'],
    relevant: [
      'LyubimovaEvgeniya_2118', 'LyubimovaEvgeniya_2021',
      'LyubimovaEvgeniya_1865', 'LyubimovaEvgeniya_1987',
    ],
  },
];

const TOP_K = 15;

// Сколько постов из списка найдено в results
function countHits(results, list) {
  const ids = new Set(results.map(r => r.id));
  return list.filter(id => ids.has(id)).length;
}

function countHitsByIds(ids, list) {
  const set = new Set(ids);
  return list.filter((id) => set.has(id)).length;
}

function firstHitPosition(results, relevant) {
  return results.findIndex(r => relevant.includes(r.id)) + 1; // 1-based, 0 если не найдено
}

// MRR = 1/rank первого попадания (0 если не найдено)
function rr(results, relevant) {
  const pos = firstHitPosition(results, relevant);
  return pos > 0 ? 1 / pos : 0;
}

function evaluateTopicResults(results, relevant, mustFind) {
  const relFound = countHits(results, relevant);
  const mustFound = countHits(results, mustFind);
  const pos = firstHitPosition(results, relevant);
  return { relFound, mustFound, pos };
}

function logTopicMetrics(method, metrics, relevant, mustFind, results) {
  console.log(`  ${method} rel:${metrics.relFound}/${relevant.length} must:${metrics.mustFound}/${mustFind.length}${metrics.pos > 0 ? ` firstRelPos:${metrics.pos}` : ''}`);
  if (metrics.relFound === 0) {
    console.log(`  ⚠️ ${method} не нашёл релевантных. top-${TOP_K}:`, results.map(r => r.id));
  }
}

function expectTopicMetrics(results, metrics, relevant, mustFind) {
  expect(results.length).toBeLessThanOrEqual(TOP_K);
  expect(metrics.relFound).toBeGreaterThanOrEqual(0);
  expect(metrics.relFound).toBeLessThanOrEqual(relevant.length);
  expect(metrics.mustFound).toBeGreaterThanOrEqual(0);
  expect(metrics.mustFound).toBeLessThanOrEqual(mustFind.length);
}

const SAMPLE_QUERY = 'Как выйти замуж на яхте';

describe('mqHybridRRF — пошаговая диагностика', () => {
  it('декларирует ожидаемые шаги pipeline', () => {
    expect(MQ_HYBRID_STEPS).toEqual([
      'prepare_query_signals',
      'build_lexical_rankings',
      'build_vector_rankings',
      'expand_with_pseudo_relevance_feedback',
      'build_intent_alignment_ranking',
      'fuse_rankings_with_weighted_rrf',
      'llm_relevance_refinement',
      'return_top_k',
    ]);
  });

  it('шаг 1: готовит сигналы запроса (expand/paraphrase/rewrite/hyde)', async () => {
    const debug = await debugMqHybridSteps(SAMPLE_QUERY, TOP_K, { postsDir: POSTS_DIR, embeddingsDir: EMBEDDINGS_DIR });
    expect(debug.artifacts.expandedTerms.length).toBeGreaterThan(0);
    expect(debug.artifacts.paraphrases.length).toBeGreaterThan(0);
    expect(debug.artifacts.paraphrases.length).toBeLessThanOrEqual(4);
    expect(debug.artifacts.rewritten.trim().length).toBeGreaterThan(0);
    expect(debug.artifacts.hypothetical.trim().length).toBeGreaterThan(20);
    expect(debug.artifacts.intentProfile).toBeDefined();
    expect(typeof debug.artifacts.intentProfile.queryType).toBe('string');
  }, 60_000);

  it('шаг 2/3/4/5: строит лексические, векторные, PRF и intent ранкинги', async () => {
    const debug = await debugMqHybridSteps(SAMPLE_QUERY, TOP_K, { postsDir: POSTS_DIR, embeddingsDir: EMBEDDINGS_DIR });
    const names = debug.signalStats.map(s => s.name);
    expect(names).toContain('bm25Raw');
    expect(names).toContain('bm25Expanded');
    expect(names).toContain('bm25Rewrite');
    expect(names).toContain('vecRewrite');
    expect(names).toContain('vecHyDE');
    expect(names).toContain('vecRaw');
    expect(names).toContain('vecRewriteChunk');
    expect(names).toContain('vecHyDEChunk');
    expect(names.some((n) => ['bm25Feedback', 'vecCentroid', 'channelPrior'].includes(n))).toBe(true);
    expect(names).toContain('intentAlignment');
    expect(names).toContain('directMatch');

    const bm25Signal = debug.signalStats.find(s => s.name === 'bm25Expanded');
    expect(bm25Signal.size).toBeGreaterThan(0);
    expect(bm25Signal.topIds.length).toBeGreaterThan(0);

    const intentSignal = debug.signalStats.find(s => s.name === 'intentAlignment');
    expect(intentSignal.size).toBeGreaterThan(0);
    expect(intentSignal.topIds.length).toBeGreaterThan(0);
  }, 60_000);

  it('шаг 5/6/7: fusion + refinement возвращают корректный topK', async () => {
    const debug = await debugMqHybridSteps(SAMPLE_QUERY, TOP_K, { postsDir: POSTS_DIR, embeddingsDir: EMBEDDINGS_DIR });
    expect(debug.fusedTop.length).toBeGreaterThan(0);
    expect(debug.fusedTop.length).toBeLessThanOrEqual(TOP_K);
    expect(debug.refinedTop.length).toBeGreaterThan(0);
    expect(debug.refinedTop.length).toBeLessThanOrEqual(TOP_K);

    const ids = debug.fusedTop.map(x => x.id);
    expect(new Set(ids).size).toBe(ids.length);
    const refinedIds = debug.refinedTop.map(x => x.id);
    expect(new Set(refinedIds).size).toBe(refinedIds.length);
    for (let i = 1; i < debug.fusedTop.length; i++) {
      expect(debug.fusedTop[i - 1].score).toBeGreaterThanOrEqual(debug.fusedTop[i].score);
    }
  }, 60_000);
});

describe('BM25 поиск (search)', () => {
  for (const { query, relevant, mustFind } of TOPICS) {
    it(query, async () => {
      const results = await search(query, TOP_K, { postsDir: POSTS_DIR });
      const metrics = evaluateTopicResults(results, relevant, mustFind);
      logTopicMetrics('BM25', metrics, relevant, mustFind, results);
      expectTopicMetrics(results, metrics, relevant, mustFind);
    }, 30_000);
  }
});

describe('mqHybridRRF', () => {
  it('эмбеддинги загружены', async () => {
    const embeddings = await loadAllEmbeddings(EMBEDDINGS_DIR);
    expect(embeddings.size).toBeGreaterThan(0);
  });

  for (const { query, relevant, mustFind } of TOPICS) {
    it(query, async () => {
      const results = await mqHybridRRF(query, TOP_K, { postsDir: POSTS_DIR, embeddingsDir: EMBEDDINGS_DIR });
      const metrics = evaluateTopicResults(results, relevant, mustFind);
      logTopicMetrics('mqHybrid', metrics, relevant, mustFind, results);
      expectTopicMetrics(results, metrics, relevant, mustFind);
    }, 60_000);
  }
});

describe('searchWithRerank', () => {
  for (const { query, relevant, mustFind } of TOPICS) {
    it(query, async () => {
      const results = await searchWithRerank(query, TOP_K, {
        postsDir: POSTS_DIR,
        embeddingsDir: EMBEDDINGS_DIR,
        candidateK: 50,
      });
      const metrics = evaluateTopicResults(results, relevant, mustFind);
      logTopicMetrics('rerank', metrics, relevant, mustFind, results);
      expectTopicMetrics(results, metrics, relevant, mustFind);
    }, 120_000);
  }
});

describe('Сводный отчёт основных методов', () => {
  it('выводит таблицу BM25 vs mqHybrid vs rerank', async () => {
    const rows = [];
    const mrrs = [];

    for (const { query, relevant, mustFind } of TOPICS) {
      const [bm25, mqHybrid, rerank] = await Promise.all([
        search(query, TOP_K, { postsDir: POSTS_DIR }),
        mqHybridRRF(query, TOP_K, { postsDir: POSTS_DIR, embeddingsDir: EMBEDDINGS_DIR }),
        searchWithRerank(query, TOP_K, { postsDir: POSTS_DIR, embeddingsDir: EMBEDDINGS_DIR, candidateK: 50 }),
      ]);

      const fmt = (results, rel, must) => {
        const relFound = countHits(results, rel);
        const mustFound = countHits(results, must);
        return `rel:${relFound}/${rel.length} must:${mustFound}/${must.length}`;
      };

      rows.push({
        query,
        bm25: fmt(bm25, relevant, mustFind),
        mqHybrid: fmt(mqHybrid, relevant, mustFind),
        rerank: fmt(rerank, relevant, mustFind),
      });

      mrrs.push({
        bm25: rr(bm25, relevant),
        mqHybrid: rr(mqHybrid, relevant),
        rerank: rr(rerank, relevant),
      });
    }

    console.log(`\nЛегенда: rel:X/Y = X из Y релевантных найдено в top-${TOP_K} | must:X/Y = X из Y mustFind найдено | Потолок recall = 112/112\n`);
    console.table(rows);

    const totalRel = TOPICS.reduce((s, t) => s + t.relevant.length, 0);
    const totalMust = TOPICS.reduce((s, t) => s + t.mustFind.length, 0);
    const meanRR = (col) => (mrrs.reduce((s, r) => s + r[col], 0) / mrrs.length).toFixed(2);

    const sumHits = (col, list) => {
      let sum = 0;
      for (const row of rows) {
        const m = row[col].match(list === 'rel' ? /rel:(\d+)/ : /must:(\d+)/);
        sum += m ? parseInt(m[1]) : 0;
      }
      return sum;
    };

    console.log(`\n${'Метод'.padEnd(10)} rel@${TOP_K}           must@${TOP_K}          MRR`);
    console.log('─'.repeat(56));
    for (const col of ['bm25', 'mqHybrid', 'rerank']) {
      const relSum = sumHits(col, 'rel');
      const mustSum = sumHits(col, 'must');
      console.log(`${col.padEnd(10)} ${String(relSum + '/' + totalRel).padEnd(16)} ${String(mustSum + '/' + totalMust).padEnd(16)} ${meanRR(col)}`);
    }
  }, 240_000);
});

describe('mqHybridRRF — drop-off диагностика этапов', () => {
  it('показывает где теряются must/relevant между pre->strict->broad->final', async () => {
    const rows = [];

    for (const { query, relevant, mustFind } of TOPICS) {
      const debug = await debugMqHybridSteps(query, TOP_K, {
        postsDir: POSTS_DIR,
        embeddingsDir: EMBEDDINGS_DIR,
      });

      const preIds = (debug.preRefineTop || []).map((p) => p.id);
      const strictIds = (debug.refineDebug?.strictTop || []).slice(0, TOP_K);
      const broadIds = (debug.refineDebug?.broadTop || []).slice(0, TOP_K);
      const finalIds = (debug.refinedTop || []).map((p) => p.id);

      rows.push({
        query,
        pre: `rel:${countHitsByIds(preIds, relevant)}/${relevant.length} must:${countHitsByIds(preIds, mustFind)}/${mustFind.length}`,
        strict: `rel:${countHitsByIds(strictIds, relevant)}/${relevant.length} must:${countHitsByIds(strictIds, mustFind)}/${mustFind.length}`,
        broad: `rel:${countHitsByIds(broadIds, relevant)}/${relevant.length} must:${countHitsByIds(broadIds, mustFind)}/${mustFind.length}`,
        final: `rel:${countHitsByIds(finalIds, relevant)}/${relevant.length} must:${countHitsByIds(finalIds, mustFind)}/${mustFind.length}`,
      });
    }

    console.log('\nDrop-off (mqHybrid): pre -> strict -> broad -> final');
    console.table(rows);

    expect(rows.length).toBe(TOPICS.length);
  }, 300_000);
});
