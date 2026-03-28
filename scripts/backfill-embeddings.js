/**
 * Генерирует эмбеддинги для всех постов и сохраняет в data/embeddings/.
 * Пропускает посты у которых эмбеддинг уже есть.
 *
 * Запуск:
 *   node --env-file=.env scripts/backfill-embeddings.js
 *   node --env-file=.env scripts/backfill-embeddings.js --channel seapinta
 */

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { generateEmbeddings, saveEmbeddings, loadAllEmbeddings } from '../search/embeddings.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const POSTS_DIR = path.join(__dirname, '../data/posts');
const BATCH_SIZE = 100;
const DELAY_MS = 500;

const channelFilter = process.argv.find((_, i) => process.argv[i - 1] === '--channel') || null;

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

function postToText(post) {
  return [post.textClean, post.ocrText].filter(Boolean).join(' ').trim();
}

const existingEmbeddings = await loadAllEmbeddings();
console.log(`Загружено существующих эмбеддингов: ${existingEmbeddings.size}`);

const files = (await fs.readdir(POSTS_DIR)).filter(f => f.endsWith('.json'));

for (const file of files) {
  const channel = file.replace('.json', '');
  if (channelFilter && channel !== channelFilter) continue;

  const posts = JSON.parse(await fs.readFile(path.join(POSTS_DIR, file), 'utf-8'));
  const todo = posts.filter(p => {
    const text = postToText(p);
    return text.length > 0 && !existingEmbeddings.has(p.id);
  });

  if (todo.length === 0) {
    console.log(`[${channel}] Нечего делать`);
    continue;
  }

  console.log(`[${channel}] ${todo.length} постов без эмбеддинга`);

  let processed = 0;

  for (let i = 0; i < todo.length; i += BATCH_SIZE) {
    const batch = todo.slice(i, i + BATCH_SIZE);
    const texts = batch.map(postToText);

    try {
      const vectors = await generateEmbeddings(texts);
      const entries = batch.map((p, j) => ({ id: p.id, vector: vectors[j] }));
      await saveEmbeddings(channel, entries);
      processed += batch.length;
      console.log(`  [${channel}] ${processed}/${todo.length}`);
    } catch (err) {
      console.error(`  [${channel}] Ошибка батча ${i}-${i + BATCH_SIZE}:`, err.message);
    }

    if (i + BATCH_SIZE < todo.length) await sleep(DELAY_MS);
  }

  console.log(`[${channel}] Готово\n`);
}

console.log('Всё готово.');
