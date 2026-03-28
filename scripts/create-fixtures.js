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
  'silavetrasila_7742',
  'meetingplace_news_176', 'meetingplace_news_3', 'meetingplace_news_367',
  'meetingplace_news_32', 'meetingplace_news_137',
  'regataveka_35', 'regataveka_64',
  'ig_clevel.yacht_DPvo6ACCtm7', 'ig_clevel.yacht_DSzdcLSDiVi', 'ig_clevel.yacht_DRe1kpyCjbD',
  'seapinta_549',
  'LyubimovaEvgeniya_2122', 'LyubimovaEvgeniya_1510', 'LyubimovaEvgeniya_1656',
  'LyubimovaEvgeniya_386', 'LyubimovaEvgeniya_732', 'LyubimovaEvgeniya_2118',
  'LyubimovaEvgeniya_1865',
  'ig_anton_timk_DGibmz6v026',
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
