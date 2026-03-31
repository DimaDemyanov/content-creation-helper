/**
 * Проверка ground truth: показывает посты из mqHybridRRF,
 * которых нет в relevant-списке. Помогает найти незамеченные релевантные посты.
 *
 * Запуск:
 *   node --env-file=.env scripts/verify-ground-truth.js
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
    relevant: [
      'ig_anton_timk_DGibmz6v026', 'ig_anton_timk_DTNpIZXiCbG', 'ig_anton_timk_C8cKk1bC2to',
      'ig_anton_timk_DSK3_lRCGYv', 'ig_anton_timk_DWY0COXF93M',
      'ig_clevel.yacht_DVwJ3JUj9qo', 'ig_clevel.yacht_DUivkh2iqD3', 'ig_clevel.yacht_DOlH6fwijjn',
      'ig_clevel.yacht_DOvz8Xfiszo', 'ig_clevel.yacht_DRRtuieCl4k', 'ig_clevel.yacht_DFNka1_qMFk',
      'ig_clevel.yacht_DPMO2zACrOr', 'ig_clevel.yacht_DOOqCSuirFF',
      'ig_clevel.yacht_DQ_oXPlD3Kg', 'ig_clevel.yacht_DUTaVm7D4kR',
      'ig_clevel.yacht_DLC3sJcqR7s', 'ig_clevel.yacht_DEXaJnGKeQv',
      'ig_clevel.yacht_DVBaeYfjcex', 'ig_clevel.yacht_DWRnwajj9y7', 'ig_clevel.yacht_DOG2eZLilZp',
      'meetingplace_news_176', 'meetingplace_news_28', 'meetingplace_news_40',
      'meetingplace_news_3', 'meetingplace_news_99',
      'silavetrasila_3938', 'silavetrasila_5403',
      'regataveka_70',
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
    relevant: ['LyubimovaEvgeniya_2118', 'LyubimovaEvgeniya_2021', 'LyubimovaEvgeniya_1865'],
  },
];

const TOP_K = 15;

for (const { query, relevant, mustFind } of TOPICS) {
  const relevantSet = new Set(relevant);
  const results = await mqHybridRRF(query, TOP_K, { postsDir: POSTS_DIR, embeddingsDir: EMBEDDINGS_DIR });
  const falsePositives = results.filter(p => !relevantSet.has(p.id));

  if (falsePositives.length === 0) {
    console.log(`\n✅ "${query}" — нет ложных срабатываний`);
    continue;
  }

  console.log(`\n📌 "${query}" — ${falsePositives.length} постов вне relevant:`);
  for (const p of falsePositives) {
    const text = p.textClean || p.ocrText || '';
    const snippet = text.length > 300 ? text.slice(0, 300) + '…' : text;
    const isMust = mustFind.includes(p.id) ? ' [MUST]' : '';
    console.log(`\n  [${p.id}]${isMust}`);
    console.log(`  ${snippet.replace(/\n/g, ' ')}`);
  }
}
