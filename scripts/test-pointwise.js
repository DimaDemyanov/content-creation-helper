/**
 * Тест pointwise re-ranking на 2 темах.
 * Запуск: node --env-file=.env scripts/test-pointwise.js
 */

import path from 'path';
import { fileURLToPath } from 'url';
import { searchWithRerank } from '../search/index.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const POSTS_DIR = path.join(__dirname, '../tests/fixtures/posts');
const EMBEDDINGS_DIR = path.join(__dirname, '../tests/fixtures/embeddings');

const TOPICS = [
  {
    query: 'Почему яхтенные путешествия — один из лучших форматов отдыха',
    mustFind: ['silavetrasila_3938', 'ig_anton_timk_DTNpIZXiCbG'],
    relevant: new Set([
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
      'silavetrasila_3938', 'silavetrasila_5403', 'regataveka_70',
    ]),
  },
  {
    query: '5 вещей, которые люди не ожидают от яхтенных путешествий',
    mustFind: ['LyubimovaEvgeniya_2122', 'LyubimovaEvgeniya_1510'],
    relevant: new Set([
      'LyubimovaEvgeniya_1510', 'LyubimovaEvgeniya_2122', 'LyubimovaEvgeniya_1656',
      'ig_clevel.yacht_DR7XTb8CqdW', 'ig_clevel.yacht_DUivkh2iqD3', 'ig_clevel.yacht_DFNka1_qMFk',
      'ig_clevel.yacht_DLz0Du5KCZv', 'ig_clevel.yacht_DPMO2zACrOr',
      'ig_clevel.yacht_DWRnwajj9y7', 'ig_anton_timk_DSK3_lRCGYv',
    ]),
  },
];

for (const { query, mustFind, relevant } of TOPICS) {
  console.log(`\n📌 "${query}"`);

  const results = await searchWithRerank(query, 15, {
    postsDir: POSTS_DIR,
    embeddingsDir: EMBEDDINGS_DIR,
    minScore: 7,
  });

  console.log(`   Выдано постов: ${results.length}`);

  const mustStatus = mustFind.map(id => {
    const pos = results.findIndex(r => r.id === id) + 1;
    return pos > 0 ? `✅ ${id} (pos ${pos})` : `❌ ${id}`;
  });
  console.log(`   Must: ${mustStatus.join(', ')}`);

  const relFound = results.filter(r => relevant.has(r.id)).length;
  const falsePos = results.filter(r => !relevant.has(r.id)).length;
  console.log(`   Relevant: ${relFound}/${relevant.size}`);
  console.log(`   False positives: ${falsePos}`);

  console.log(`\n   Все выданные посты:`);
  for (const r of results) {
    const tag = mustFind.includes(r.id) ? '[MUST]' : relevant.has(r.id) ? '[rel]' : '[?]';
    console.log(`     ${tag} ${r.id}`);
  }
}
