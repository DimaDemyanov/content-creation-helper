/**
 * Генерирует эмбеддинги для всех постов.
 *
 * Два режима:
 *   - По умолчанию: один эмбеддинг на пост (весь текст), сохраняет в data/embeddings/
 *   - --chunked: эмбеддинг на каждый чанк поста, сохраняет в data/chunk-embeddings/
 *
 * Запуск:
 *   node --env-file=.env scripts/backfill-embeddings.js
 *   node --env-file=.env scripts/backfill-embeddings.js --chunked
 *   node --env-file=.env scripts/backfill-embeddings.js --channel seapinta
 *   node --env-file=.env scripts/backfill-embeddings.js --postsDir tests/fixtures/posts --embeddingsDir tests/fixtures/embeddings
 *   node --env-file=.env scripts/backfill-embeddings.js --chunked --postsDir tests/fixtures/posts --chunkEmbeddingsDir tests/fixtures/chunk-embeddings
 */

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { generateEmbeddings, saveEmbeddings, loadAllEmbeddings, saveChunkEmbeddings, loadAllChunkEmbeddings } from '../search/embeddings.js';
import { postToChunks } from '../search/chunking.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function getArg(name) {
  return process.argv.find((_, i) => process.argv[i - 1] === name) || null;
}

const POSTS_DIR = getArg('--postsDir') || path.join(__dirname, '../data/posts');
const EMBEDDINGS_DIR = getArg('--embeddingsDir') || null;
const CHUNK_EMBEDDINGS_DIR = getArg('--chunkEmbeddingsDir') || null;
const CHUNKED = process.argv.includes('--chunked');
const BATCH_SIZE = 100;
const DELAY_MS = 500;

const channelFilter = getArg('--channel');

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

function postToText(post) {
  return [post.textClean, post.ocrText].filter(Boolean).join(' ').trim();
}

const files = (await fs.readdir(POSTS_DIR)).filter(f => f.endsWith('.json'));

if (CHUNKED) {
  // --- Chunked mode ---
  const existingChunks = await loadAllChunkEmbeddings(CHUNK_EMBEDDINGS_DIR || undefined);
  console.log(`Chunked mode. Загружено постов с чанк-эмбеддингами: ${existingChunks.size}`);

  for (const file of files) {
    const channel = file.replace('.json', '');
    if (channelFilter && channel !== channelFilter) continue;

    const posts = JSON.parse(await fs.readFile(path.join(POSTS_DIR, file), 'utf-8'));

    // Собираем все чанки для постов без эмбеддингов
    const allChunks = []; // { postId, chunkIdx, text }
    for (const post of posts) {
      if (existingChunks.has(post.id)) continue;
      const chunks = postToChunks(post);
      for (let i = 0; i < chunks.length; i++) {
        allChunks.push({ postId: post.id, chunkIdx: i, text: chunks[i] });
      }
    }

    if (allChunks.length === 0) {
      console.log(`[${channel}] Нечего делать`);
      continue;
    }

    const postsCount = new Set(allChunks.map(c => c.postId)).size;
    console.log(`[${channel}] ${postsCount} постов → ${allChunks.length} чанков`);

    let processed = 0;
    for (let i = 0; i < allChunks.length; i += BATCH_SIZE) {
      const batch = allChunks.slice(i, i + BATCH_SIZE);
      const texts = batch.map(c => c.text);

      try {
        const vectors = await generateEmbeddings(texts);
        const entries = batch.map((c, j) => ({ postId: c.postId, chunkIdx: c.chunkIdx, vector: vectors[j] }));
        await saveChunkEmbeddings(channel, entries, CHUNK_EMBEDDINGS_DIR || undefined);
        processed += batch.length;
        console.log(`  [${channel}] ${processed}/${allChunks.length} чанков`);
      } catch (err) {
        console.error(`  [${channel}] Ошибка батча ${i}:`, err.message);
      }

      if (i + BATCH_SIZE < allChunks.length) await sleep(DELAY_MS);
    }

    console.log(`[${channel}] Готово\n`);
  }
} else {
  // --- Full-post mode ---
  const existingEmbeddings = await loadAllEmbeddings(EMBEDDINGS_DIR || undefined);
  console.log(`Full-post mode. Загружено существующих эмбеддингов: ${existingEmbeddings.size}`);

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
        await saveEmbeddings(channel, entries, EMBEDDINGS_DIR || undefined);
        processed += batch.length;
        console.log(`  [${channel}] ${processed}/${todo.length}`);
      } catch (err) {
        console.error(`  [${channel}] Ошибка батча ${i}-${i + BATCH_SIZE}:`, err.message);
      }

      if (i + BATCH_SIZE < todo.length) await sleep(DELAY_MS);
    }

    console.log(`[${channel}] Готово\n`);
  }
}

console.log('Всё готово.');
