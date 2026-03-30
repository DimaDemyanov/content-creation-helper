/**
 * Диагностика: где теряются must-посты — в mqHybridRRF или в re-ranker?
 *
 * Запуск:
 *   node --env-file=.env scripts/diagnose-must.js
 */

import path from 'path';
import { fileURLToPath } from 'url';
import { mqHybridRRF } from '../search/index.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const POSTS_DIR = path.join(__dirname, '../tests/fixtures/posts');
const EMBEDDINGS_DIR = path.join(__dirname, '../tests/fixtures/embeddings');

const TOPICS = [
  {
    query: 'Почему яхтенные путешествия — один из лучших форматов отдыха',
    mustFind: ['silavetrasila_3938', 'ig_anton_timk_DTNpIZXiCbG'],
  },
  {
    query: 'Как проходит один день на яхте',
    mustFind: ['meetingplace_news_367', 'meetingplace_news_32', 'regataveka_64'],
  },
  {
    query: 'Как выйти замуж на яхте',
    mustFind: ['seapinta_549', 'ig_clevel.yacht_DPvo6ACCtm7'],
  },
  {
    query: '5 вещей, которые люди не ожидают от яхтенных путешествий',
    mustFind: ['LyubimovaEvgeniya_2122', 'LyubimovaEvgeniya_1510'],
  },
  {
    query: 'Что будет на регате в Турции (5 лодок)',
    mustFind: ['ig_clevel.yacht_DSzdcLSDiVi', 'ig_clevel.yacht_DRe1kpyCjbD', 'silavetrasila_7742'],
  },
  {
    query: 'История: как мы сели на мель',
    mustFind: ['LyubimovaEvgeniya_2118', 'LyubimovaEvgeniya_2021'],
  },
];

// Берём топ-347 (все посты) чтобы увидеть реальный ранг
const CANDIDATE_K = 347;

for (const { query, mustFind } of TOPICS) {
  console.log(`\n📌 "${query}"`);
  const results = await mqHybridRRF(query, CANDIDATE_K, { postsDir: POSTS_DIR, embeddingsDir: EMBEDDINGS_DIR });
  const rankMap = new Map(results.map((p, i) => [p.id, i + 1]));

  for (const id of mustFind) {
    const rank = rankMap.get(id);
    let label;
    if (rank === undefined) label = '❌ не в результатах';
    else if (rank <= 15) label = `✅ rank ${rank} (в top-15)`;
    else if (rank <= 50) label = `⚠️  rank ${rank} (в top-50, но не top-15)`;
    else label = `🔴 rank ${rank} (за top-50!)`;
    console.log(`  ${id}: ${label}`);
  }
}
