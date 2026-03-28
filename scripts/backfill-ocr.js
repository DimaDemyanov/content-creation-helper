/**
 * Backfill ocrText для постов у которых есть медиа, но нет ocrText.
 *
 * Instagram: читаем URL из media[0].url и передаём в extractTextFromImage.
 * Telegram:  переподключаемся через GramJS, скачиваем фото по channel + msgId.
 *
 * Запуск:
 *   node --env-file=.env scripts/backfill-ocr.js
 *   node --env-file=.env scripts/backfill-ocr.js --source instagram   # только Instagram
 *   node --env-file=.env scripts/backfill-ocr.js --source telegram    # только Telegram
 *   node --env-file=.env scripts/backfill-ocr.js --channel seapinta   # один канал
 */

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { extractTextFromImage } from '../utils/ocr.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const POSTS_DIR = path.join(__dirname, '../data/posts');
const SESSION_FILE = path.join(__dirname, '../data/telegram.session');

const args = process.argv.slice(2);
const sourceFilter = args.find((_, i) => args[i - 1] === '--source') || null;
const channelFilter = args.find((_, i) => args[i - 1] === '--channel') || null;

// --- Telegram client (lazy init) ---

let tgClient = null;

async function getTgClient() {
  if (tgClient) return tgClient;
  const { TelegramClient } = await import('telegram');
  const { StringSession } = await import('telegram/sessions/index.js');

  const sessionStr = await fs.readFile(SESSION_FILE, 'utf-8').catch(() => '');
  tgClient = new TelegramClient(
    new StringSession(sessionStr),
    Number(process.env.TELEGRAM_API_ID),
    process.env.TELEGRAM_API_HASH,
    { connectionRetries: 3 }
  );
  await tgClient.connect();
  if (!await tgClient.isUserAuthorized()) {
    throw new Error('Telegram не авторизован. Запусти node auth/telegram.js');
  }
  return tgClient;
}

async function disconnectTg() {
  if (tgClient) {
    const session = tgClient.session.save();
    await fs.writeFile(SESSION_FILE, session);
    await tgClient.disconnect();
    tgClient = null;
  }
}

// --- OCR для одного поста ---

async function ocrInstagram(post) {
  const url = post.media?.[0]?.url;
  if (!url) return null;
  return extractTextFromImage(url);
}

async function ocrTelegram(post, channel) {
  const client = await getTgClient();
  const msgId = parseInt(post.id.split('_').pop(), 10);
  try {
    const [msg] = await client.getMessages(channel, { ids: [msgId] });
    if (!msg?.media?.photo) return null;
    const buffer = await msg.downloadMedia();
    if (!buffer) return null;
    if (buffer.length > 3_900_000) {
      console.log(`  [skip] ${post.id}: слишком большой файл (${Math.round(buffer.length / 1024)}KB)`);
      return null;
    }
    const base64 = buffer.toString('base64');
    return extractTextFromImage(`data:image/jpeg;base64,${base64}`);
  } catch (err) {
    console.error(`  [error] ${post.id}:`, err.message);
    return null;
  }
}

// --- Обработка одного файла ---

async function processFile(file) {
  const filePath = path.join(POSTS_DIR, file);
  const posts = JSON.parse(await fs.readFile(filePath, 'utf-8'));
  const channel = file.replace('.json', '');
  const isInstagram = channel.startsWith('ig_');
  const source = isInstagram ? 'instagram' : 'telegram';

  if (sourceFilter && sourceFilter !== source) return;
  if (channelFilter && channelFilter !== channel) return;

  const todo = posts.filter(p => p.media);
  if (todo.length === 0) {
    console.log(`[${channel}] Нечего делать`);
    return;
  }

  console.log(`[${channel}] ${todo.length} постов без OCR (source: ${source})`);

  let updated = 0;
  let skipped = 0;

  for (const post of todo) {
    let ocrText = null;
    try {
      ocrText = isInstagram
        ? await ocrInstagram(post)
        : await ocrTelegram(post, channel.replace('ig_', ''));
    } catch (err) {
      console.error(`  [error] ${post.id}:`, err.message);
    }

    const idx = posts.findIndex(p => p.id === post.id);
    posts[idx].ocrText = ocrText;

    if (ocrText) {
      console.log(`  ✓ ${post.id}: "${ocrText.slice(0, 60)}"`);
      updated++;
    } else {
      skipped++;
    }

    // Сохраняем каждые 10 постов чтобы не потерять прогресс
    if ((updated + skipped) % 10 === 0) {
      await fs.writeFile(filePath, JSON.stringify(posts, null, 2));
    }
  }

  await fs.writeFile(filePath, JSON.stringify(posts, null, 2));
  console.log(`[${channel}] Готово: найден текст в ${updated}/${todo.length} постах\n`);
}

// --- Main ---

const files = await fs.readdir(POSTS_DIR);
const jsonFiles = files.filter(f => f.endsWith('.json'));

for (const file of jsonFiles) {
  await processFile(file);
}

await disconnectTg();
console.log('Всё готово.');
