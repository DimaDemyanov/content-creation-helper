import { TelegramClient } from 'telegram';
import { StringSession } from 'telegram/sessions/index.js';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { cleanText } from '../utils/textClean.js';
import { extractTextFromImage } from '../utils/ocr.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const POSTS_DIR = path.join(__dirname, '../data/posts');
const SESSION_FILE = path.join(__dirname, '../data/telegram.session');

let client = null;

async function getClient() {
  if (client) return client;

  const sessionStr = await fs.readFile(SESSION_FILE, 'utf-8').catch(() => '');
  const session = new StringSession(sessionStr);

  client = new TelegramClient(
    session,
    Number(process.env.TELEGRAM_API_ID),
    process.env.TELEGRAM_API_HASH,
    { connectionRetries: 3 }
  );

  await client.connect();

  if (!await client.isUserAuthorized()) {
    throw new Error('Telegram не авторизован. Запусти node auth/telegram.js');
  }

  return client;
}

export async function disconnectTelegram() {
  if (client) {
    const newSession = client.session.save();
    await fs.writeFile(SESSION_FILE, newSession);
    await client.disconnect();
    client = null;
  }
}

export async function collectTelegramChannel(channel, channelState, onBatchSaved) {
  const tgClient = await getClient();
  const since = channelState.lastCollectedAt || null;

  console.log(`[Telegram] Сбор из @${channel} (since: ${since || 'начало'})...`);

  const entity = await tgClient.getEntity(channel);
  const channelInfo = await getChannelInfo(tgClient, entity);
  const { totalCollected, allPostsDownloaded } = await fetchAndSaveChannelPosts(tgClient, entity, channel, since, onBatchSaved);
  const downloadedStats = await computeDownloadedStats(channel);

  return {
    collected: totalCollected,
    allPostsDownloaded,
    lastCollectedAt: new Date().toISOString(),
    totalPosts: downloadedStats.count,
    downloadedDateFrom: downloadedStats.dateFrom,
    downloadedDateTo: downloadedStats.dateTo,
    channelTotalPosts: channelInfo.totalPosts,
    channelDateFrom: channelInfo.dateFrom,
    channelDateTo: channelInfo.dateTo,
  };
}

async function getChannelInfo(tgClient, entity) {
  try {
    // Последнее сообщение
    const [last] = await tgClient.getMessages(entity, { limit: 1 });
    // Самое первое сообщение (offsetId=1 возвращает с начала в обратном порядке)
    const [first] = await tgClient.getMessages(entity, { limit: 1, reverse: true });

    return {
      totalPosts: entity.participantsCount ?? last?.id ?? null,
      dateFrom: first ? new Date(first.date * 1000).toISOString() : null,
      dateTo: last ? new Date(last.date * 1000).toISOString() : null,
    };
  } catch {
    return { totalPosts: null, dateFrom: null, dateTo: null };
  }
}

async function fetchAndSaveChannelPosts(tgClient, entity, channel, since, onBatchSaved) {

  let offsetId = 0;
  let hasMore = true;
  let totalCollected = 0;
  let allPostsDownloaded = false;

  while (hasMore) {
    const messages = await tgClient.getMessages(entity, { limit: 100, offsetId, reverse: false });

    if (messages.length === 0) break;

    const batch = [];
    for (const msg of messages) {
      if (!msg.text && !msg.media) continue;
      if (since && msg.date * 1000 < new Date(since).getTime()) {
        hasMore = false;
        break;
      }

      const post = await buildPost(msg, channel);
      batch.push(post);
    }

    if (batch.length > 0) {
      await savePosts(channel, batch);
      totalCollected += batch.length;
      console.log(`[Telegram] @${channel}: сохранён батч ${batch.length} постов (всего: ${totalCollected})`);
      if (onBatchSaved) await onBatchSaved(totalCollected);
    }

    offsetId = messages[messages.length - 1].id;
    if (messages.length < 100) {
      allPostsDownloaded = true;
      hasMore = false;
    }

    await sleep(500);
  }

  return { totalCollected, allPostsDownloaded };
}

async function buildPost(msg, channel) {
  const text = msg.text || '';
  const textClean = cleanText(text);

  let ocrText = null;
  let mediaInfo = null;

  if (msg.media?.photo) {
    mediaInfo = [{ type: 'photo' }];
    try {
      const buffer = await msg.downloadMedia();
      if (buffer) {
        // Пропускаем слишком большие изображения (> 5 MB в base64 ~ 3.75 MB raw)
        if (buffer.length > 3_900_000) {
          console.log(`[OCR] Пропускаем ${channel}_${msg.id}: слишком большой файл (${Math.round(buffer.length / 1024)}KB)`);
        } else {
          const base64 = buffer.toString('base64');
          const dataUrl = `data:image/jpeg;base64,${base64}`;
          ocrText = await extractTextFromImage(dataUrl);
        }
      } else {
        console.log(`[OCR] ${channel}_${msg.id}: downloadMedia вернул null`);
      }
    } catch (err) {
      console.error(`[OCR] Ошибка ${channel}_${msg.id}:`, err.message);
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

async function loadPosts(channel) {
  const filePath = path.join(POSTS_DIR, `${channel}.json`);
  try {
    const raw = await fs.readFile(filePath, 'utf-8');
    return JSON.parse(raw);
  } catch {
    return [];
  }
}

async function computeDownloadedStats(channel) {
  const posts = await loadPosts(channel);
  if (posts.length === 0) return { count: 0, dateFrom: null, dateTo: null };

  const dates = posts.map(p => new Date(p.date).getTime()).filter(Boolean);
  return {
    count: posts.length,
    dateFrom: new Date(Math.min(...dates)).toISOString(),
    dateTo: new Date(Math.max(...dates)).toISOString(),
  };
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
