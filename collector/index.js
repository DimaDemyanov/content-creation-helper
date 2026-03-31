import 'dotenv/config';
import cron from 'node-cron';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { collectTelegramChannel, disconnectTelegram, runPendingOcr } from './telegram.js';
import { collectInstagramAccount } from './instagram.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const STATE_FILE = path.join(__dirname, '../data/state.json');
const CONFIG_FILE = path.join(__dirname, '../config.json');

export async function readState() {
  try {
    const raw = await fs.readFile(STATE_FILE, 'utf-8');
    return JSON.parse(raw);
  } catch {
    return {};
  }
}

export async function saveState(state) {
  await fs.writeFile(STATE_FILE, JSON.stringify(state, null, 2));
}

export async function addChannel(source, username) {
  const config = JSON.parse(await fs.readFile(CONFIG_FILE, 'utf-8'));
  const state = await readState();

  if (source === 'telegram') {
    if (!config.telegram.channels.includes(username)) {
      config.telegram.channels.push(username);
      await fs.writeFile(CONFIG_FILE, JSON.stringify(config, null, 2));
    }
    if (!state[username]) {
      state[username] = { source: 'telegram', lastCollectedAt: null, allPostsDownloaded: false, totalPosts: 0 };
      await saveState(state);
    }
  } else if (source === 'instagram') {
    if (!config.instagram.accounts.includes(username)) {
      config.instagram.accounts.push(username);
      await fs.writeFile(CONFIG_FILE, JSON.stringify(config, null, 2));
    }
    const key = `ig_${username}`;
    if (!state[key]) {
      state[key] = { source: 'instagram', lastCollectedAt: null, allPostsDownloaded: false, totalPosts: 0 };
      await saveState(state);
    }
  }
}

export async function removeChannel(source, username) {
  const config = JSON.parse(await fs.readFile(CONFIG_FILE, 'utf-8'));
  const state = await readState();

  const key = source === 'instagram' ? `ig_${username}` : username;

  // Удаляем из config.json
  if (source === 'telegram') {
    config.telegram.channels = config.telegram.channels.filter(c => c !== username);
  } else if (source === 'instagram') {
    config.instagram.accounts = config.instagram.accounts.filter(a => a !== username);
  }
  await fs.writeFile(CONFIG_FILE, JSON.stringify(config, null, 2));

  // Удаляем из state.json
  delete state[key];
  await saveState(state);

  // Удаляем файл постов
  const postsFile = path.join(__dirname, `../data/posts/${key}.json`);
  await fs.rm(postsFile, { force: true });

  // Удаляем файл эмбеддингов
  const embeddingsFile = path.join(__dirname, `../data/embeddings/${key}.json`);
  await fs.rm(embeddingsFile, { force: true });
}

export async function collect(channels = null, { fullSync = false } = {}) {
  const config = JSON.parse(await fs.readFile(CONFIG_FILE, 'utf-8'));
  const state = await readState();

  const tgChannels = channels
    ? config.telegram.channels.filter(c => channels.includes(c))
    : config.telegram.channels;

  const igAccounts = channels
    ? config.instagram.accounts.filter(a => channels.includes(`ig_${a}`) || channels.includes(a))
    : config.instagram.accounts;

  console.log(`[Collector] Старт сбора${fullSync ? ' (полный пересбор)' : ''}. Каналы: ${[...tgChannels, ...igAccounts.map(a => `ig_${a}`)].join(', ')}`);

  let totalCollected = 0;

  for (const channel of tgChannels) {
    const channelState = state[channel] || { lastCollectedAt: null, allPostsDownloaded: false, totalPosts: 0 };
    const effectiveState = fullSync ? { ...channelState, lastCollectedAt: null } : channelState;
    try {
      const result = await collectTelegramChannel(channel, effectiveState);
      state[channel] = { source: 'telegram', ...result };
      await saveState(state);
      totalCollected += result.collected;
      console.log(`[Telegram] @${channel}: +${result.collected} постов, всего ${result.totalPosts}`);
    } catch (err) {
      console.error(`[Telegram] Ошибка @${channel}:`, err.message);
    }
  }

  await disconnectTelegram();

  // OCR запускается после дисконнекта — Telegram клиент больше не нужен
  for (const channel of tgChannels) {
    try {
      await runPendingOcr(channel);
    } catch (err) {
      console.error(`[OCR] Ошибка @${channel}:`, err.message);
    }
  }

  for (const account of igAccounts) {
    const key = `ig_${account}`;
    const channelState = state[key] || { lastCollectedAt: null, allPostsDownloaded: false, totalPosts: 0 };
    try {
      const result = await collectInstagramAccount(account, channelState, config.instagram.firstRunLimit);
      state[key] = { source: 'instagram', ...result };
      await saveState(state);
      totalCollected += result.collected;
      console.log(`[Instagram] @${account}: +${result.collected} постов, всего ${result.totalPosts}`);
    } catch (err) {
      console.error(`[Instagram] Ошибка @${account}:`, err.message);
    }
  }

  console.log(`[Collector] Готово. Собрано новых постов: ${totalCollected}`);
  return { totalCollected, state };
}

const isOnce = process.argv.includes('--once');
const isFullSync = process.argv.includes('--full');

if (isOnce || isFullSync) {
  collect(null, { fullSync: isFullSync }).catch(console.error);
} else {
  const config = JSON.parse(await fs.readFile(CONFIG_FILE, 'utf-8'));
  const schedule = config.collect?.schedule || '0 9 * * *';

  console.log(`[Collector] Запущен по расписанию: ${schedule}`);
  collect().catch(console.error);

  cron.schedule(schedule, () => {
    collect().catch(console.error);
  });
}
