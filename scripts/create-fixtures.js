/**
 * Создаёт тестовые фикстуры: по 50 постов на канал.
 * Обязательно включает все ground truth посты, остальные — случайные.
 *
 * Запуск:
 *   node scripts/create-fixtures.js
 */

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const POSTS_DIR = path.join(__dirname, '../data/posts');
const FIXTURES_DIR = path.join(__dirname, '../tests/fixtures/posts');
const TARGET_PER_CHANNEL = 50;

// Все ground truth ID которые ОБЯЗАТЕЛЬНО должны быть в фикстурах
const REQUIRED_IDS = new Set([
  'silavetrasila_5403', 'silavetrasila_3938', 'silavetrasila_8156', 'silavetrasila_8184',
  'silavetrasila_6630', 'silavetrasila_6905', 'silavetrasila_5251', 'silavetrasila_7420',
  'silavetrasila_7742', 'silavetrasila_558', 'silavetrasila_1078',
  'silavetrasila_6286',                                        // T2 день на яхте
  'meetingplace_news_176', 'meetingplace_news_3', 'meetingplace_news_367',
  'meetingplace_news_32', 'meetingplace_news_137',
  'meetingplace_news_28', 'meetingplace_news_40', 'meetingplace_news_99', // T1 лучший отдых
  'meetingplace_news_148', 'meetingplace_news_314',            // T4 первая яхта, T2 день
  'regataveka_35', 'regataveka_64',
  'regataveka_83', 'regataveka_88', 'regataveka_120',          // T2 день + T6 регата + T8 бухты
  'regataveka_40', 'regataveka_29', 'regataveka_69', 'regataveka_73', // T6 регата
  'regataveka_70', 'regataveka_95', 'regataveka_82', 'regataveka_63', // T1/T2/T6
  'regataveka_112',                                            // T2/T6
  'ig_clevel.yacht_DPvo6ACCtm7', 'ig_clevel.yacht_DSzdcLSDiVi', 'ig_clevel.yacht_DRe1kpyCjbD',
  'seapinta_549',
  'seapinta_944',                                              // T8 бухты Турции
  'seapinta_920',                                              // T7 акула
  'LyubimovaEvgeniya_2122', 'LyubimovaEvgeniya_1510', 'LyubimovaEvgeniya_1656',
  'LyubimovaEvgeniya_386', 'LyubimovaEvgeniya_732', 'LyubimovaEvgeniya_2118',
  'LyubimovaEvgeniya_1865', 'LyubimovaEvgeniya_2021',
  'ig_anton_timk_DGibmz6v026', 'ig_anton_timk_DTNpIZXiCbG',
  'ig_anton_timk_DSK3_lRCGYv', 'ig_anton_timk_C8cKk1bC2to', 'ig_anton_timk_DWY0COXF93M', // T1

  'ig_anton_timk_DUz0RkgDy6J', 'ig_anton_timk_DRUspFLj3gJ',  // T4 первая яхта
  'ig_anton_timk_DQzdFtlExmi', 'ig_anton_timk_DUVqzSZD87t', 'ig_anton_timk_C2cpF7kNFtM', // T7 акула
  'ig_anton_timk_DSfvT1PCAM6', 'ig_anton_timk_DEzkhmYCasQ', 'ig_anton_timk_DIdkxXAvo-z', // T1 лучший отдых

  'seapinta_116', 'seapinta_117',                            // T2 день на яхте
  'meetingplace_news_164',                                   // T2 ночной переход
  'meetingplace_news_3',                                     // оставляем в фикстурах (не в relevant)
  'meetingplace_news_40',                                    // оставляем в фикстурах (не в relevant)

  'regataveka_74',                                           // T6 регата + T8 бухты

  'LyubimovaEvgeniya_1987',                                  // T9 мель
]);

await fs.mkdir(FIXTURES_DIR, { recursive: true });

const files = (await fs.readdir(POSTS_DIR)).filter(f => f.endsWith('.json'));

for (const file of files) {
  const channel = file.replace('.json', '');
  const posts = JSON.parse(await fs.readFile(path.join(POSTS_DIR, file), 'utf-8'));

  const required = posts.filter(p => REQUIRED_IDS.has(p.id));
  const rest = posts.filter(p => !REQUIRED_IDS.has(p.id));

  // Перемешиваем остальные и берём сколько нужно
  const shuffled = rest.sort(() => Math.random() - 0.5);
  const fill = shuffled.slice(0, Math.max(0, TARGET_PER_CHANNEL - required.length));

  const fixture = [...required, ...fill];

  const outPath = path.join(FIXTURES_DIR, file);
  await fs.writeFile(outPath, JSON.stringify(fixture, null, 2));
  console.log(`[${channel}] required=${required.length} fill=${fill.length} total=${fixture.length}`);
}

console.log('\nФикстуры созданы в tests/fixtures/posts/');
