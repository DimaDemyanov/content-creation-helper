import { TelegramClient } from 'telegram';
import { StringSession } from 'telegram/sessions/index.js';
import { Api } from 'telegram/tl/index.js';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { cleanText } from '../utils/textClean.js';
import { extractTextFromImage } from '../utils/ocr.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const POSTS_DIR = path.join(__dirname, '../data/posts');
const SESSION_FILE = path.join(__dirname, '../data/telegram.session');

export async function collectTelegram(channels, since = null) {
  const sessionStr = await fs.readFile(SESSION_FILE, 'utf-8').catch(() => '');
  const session = new StringSession(sessionStr);

  const client = new TelegramClient(
    session,
    Number(process.env.TELEGRAM_API_ID),
    process.env.TELEGRAM_API_HASH,
    { connectionRetries: 3 }
  );

  await client.connect();

  if (!await client.isUserAuthorized()) {
    throw new Error('Telegram не авторизован. Запусти node auth/telegram.js');
  }

  const results = { collected: 0, channels: {} };

  for (const channel of channels) {
    console.log(`[Telegram] Сбор из @${channel}...`);
    try {
      const posts = await fetchChannelPosts(client, channel, since);
      await savePosts(channel, posts);
      results.collected += posts.length;
      results.channels[channel] = posts.length;
      console.log(`[Telegram] @${channel}: собрано ${posts.length} постов`);
    } catch (err) {
      console.error(`[Telegram] Ошибка @${channel}:`, err.message);
    }
  }

  const newSession = client.session.save();
  await fs.writeFile(SESSION_FILE, newSession);
  await client.disconnect();

  return results;
}

async function fetchChannelPosts(client, channel, since) {
  const posts = [];
  const entity = await client.getEntity(channel);

  const params = { limit: 100 };
  if (since) params.offsetDate = Math.floor(new Date(since).getTime() / 1000);

  let offsetId = 0;
  let hasMore = true;

  while (hasMore) {
    const messages = await client.getMessages(entity, { ...params, offsetId, reverse: false });

    if (messages.length === 0) break;

    for (const msg of messages) {
      if (!msg.text && !msg.media) continue;
      if (since && msg.date * 1000 < new Date(since).getTime()) {
        hasMore = false;
        break;
      }

      const post = await buildPost(msg, channel);
      posts.push(post);
    }

    offsetId = messages[messages.length - 1].id;
    if (messages.length < 100) hasMore = false;

    await sleep(500);
  }

  return posts;
}

async function buildPost(msg, channel) {
  const text = msg.text || '';
  const textClean = cleanText(text);

  let ocrText = null;
  let mediaInfo = null;

  if (msg.media?.photo) {
    try {
      const buffer = await msg.downloadMedia();
      if (buffer) {
        const tmpPath = path.join(__dirname, `../data/media/tg_${msg.id}.jpg`);
        await fs.writeFile(tmpPath, buffer);
        const tesseract = await import('node-tesseract-ocr');
        ocrText = await tesseract.default.recognize(tmpPath, { lang: 'rus+eng', oem: 1, psm: 3 });
        await fs.unlink(tmpPath).catch(() => {});
        ocrText = ocrText?.trim() || null;
        mediaInfo = [{ type: 'photo' }];
      }
    } catch {
      mediaInfo = [{ type: 'photo' }];
    }
  }

  return {
    id: `${channel}_${msg.id}`,
    source: 'telegram',
    channel,
    url: `https://t.me/${channel}/${msg.id}`,
    date: new Date(msg.date * 1000).toISOString(),
    text,
    textClean,
    ocrText,
    media: mediaInfo,
    stats: {
      views: msg.views || 0,
      likes: msg.reactions?.results?.reduce((sum, r) => sum + r.count, 0) || 0,
      comments: msg.replies?.replies || 0,
    },
    collectedAt: new Date().toISOString(),
  };
}

async function savePosts(channel, posts) {
  if (posts.length === 0) return;

  const filePath = path.join(POSTS_DIR, `${channel}.json`);
  let existing = [];

  try {
    const raw = await fs.readFile(filePath, 'utf-8');
    existing = JSON.parse(raw);
  } catch {}

  const existingIds = new Set(existing.map(p => p.id));
  const newPosts = posts.filter(p => !existingIds.has(p.id));
  const merged = [...existing, ...newPosts];

  await fs.writeFile(filePath, JSON.stringify(merged, null, 2));
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
