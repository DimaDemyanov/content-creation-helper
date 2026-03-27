import 'dotenv/config';
import cron from 'node-cron';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { collectTelegram } from './telegram.js';
import { collectInstagram } from './instagram.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const STATE_FILE = path.join(__dirname, '../data/state.json');
const CONFIG_FILE = path.join(__dirname, '../config.json');

async function readState() {
  try {
    const raw = await fs.readFile(STATE_FILE, 'utf-8');
    return JSON.parse(raw);
  } catch {
    return { lastCollectedAt: null };
  }
}

async function saveState(state) {
  await fs.writeFile(STATE_FILE, JSON.stringify(state, null, 2));
}

export async function collect() {
  const config = JSON.parse(await fs.readFile(CONFIG_FILE, 'utf-8'));
  const state = await readState();
  const since = state.lastCollectedAt;

  console.log(`[Collector] Старт сбора. Since: ${since || 'первый запуск'}`);

  const results = { telegram: null, instagram: null };

  results.telegram = await collectTelegram(
    config.telegram.channels,
    since
  );

  results.instagram = await collectInstagram(
    config.instagram.accounts,
    since,
    config.instagram.firstRunLimit
  );

  const total = results.telegram.collected + results.instagram.collected;
  console.log(`[Collector] Готово. Собрано новых постов: ${total}`);

  await saveState({ lastCollectedAt: new Date().toISOString() });

  return results;
}

const isOnce = process.argv.includes('--once');

if (isOnce) {
  collect().catch(console.error);
} else {
  const config = JSON.parse(await fs.readFile(CONFIG_FILE, 'utf-8'));
  const schedule = config.collect?.schedule || '0 9 * * *';

  console.log(`[Collector] Запущен по расписанию: ${schedule}`);
  collect().catch(console.error);

  cron.schedule(schedule, () => {
    collect().catch(console.error);
  });
}
