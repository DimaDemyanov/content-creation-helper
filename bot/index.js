import 'dotenv/config';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import TelegramBot from 'node-telegram-bot-api';
import { mqHybridRRF, search } from '../search/index.js';
import { llmClient as openai, LLM_MODEL } from '../llm.js';
import { collect, addChannel, removeChannel, readState } from '../collector/index.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const POSTS_DIR = path.join(__dirname, '../data/posts');

const SCORE_THRESHOLD = 0.05;
const SCORE_MAX_RESULTS = 8;

const bot = new TelegramBot(process.env.TELEGRAM_BOT_TOKEN, { polling: true });

console.log('[Bot] Запущен');

// Состояние ожидания ввода: Map<chatId, { action: 'search'|'generate' }>
const awaitingInput = new Map();

bot.onText(/\/start/, (msg) => {
  bot.sendMessage(msg.chat.id, [
    'Content Creation Helper',
    '',
    '/search — найти посты по теме',
    '/generate — найти посты и сгенерировать похожий текст',
    '/collect — запустить сбор новых постов',
    '/stats — статистика по каналам',
    '/addchannel telegram <username> — добавить Telegram-канал',
    '/addchannel instagram <username> — добавить Instagram-аккаунт',
    '/removechannel telegram <username> — удалить канал и все его посты',
    '/removechannel instagram <username> — удалить аккаунт и все его посты',
  ].join('\n'));
});

bot.onText(/\/search$/, async (msg) => {
  awaitingInput.set(msg.chat.id, { action: 'search' });
  await bot.sendMessage(msg.chat.id, 'Введите запрос для поиска:');
});

bot.onText(/\/search (.+)/, async (msg, match) => {
  await handleSearch(msg.chat.id, match[1].trim());
});

bot.onText(/\/generate$/, async (msg) => {
  awaitingInput.set(msg.chat.id, { action: 'generate' });
  await bot.sendMessage(msg.chat.id, 'Введите тему для генерации поста:');
});

bot.onText(/\/generate (.+)/, async (msg, match) => {
  await handleGenerate(msg.chat.id, match[1].trim());
});

// Обработка свободного текста — для диалогового ввода
bot.on('message', async (msg) => {
  if (!msg.text || msg.text.startsWith('/')) return;
  const pending = awaitingInput.get(msg.chat.id);
  if (!pending) return;

  awaitingInput.delete(msg.chat.id);
  if (pending.action === 'search') {
    await handleSearch(msg.chat.id, msg.text.trim());
  } else if (pending.action === 'generate') {
    await handleGenerate(msg.chat.id, msg.text.trim());
  }
});

async function handleSearch(chatId, query) {
  await bot.sendMessage(chatId, `Ищу: "${query}"...`);

  try {
    const results = await mqHybridRRF(query, 10);

    if (results.length === 0) {
      return bot.sendMessage(chatId, 'Ничего не найдено.');
    }

    const sorted = [...results].sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
    const filtered = sorted
      .filter(r => (r.score ?? 0) >= SCORE_THRESHOLD)
      .slice(0, SCORE_MAX_RESULTS);
    const toShow = filtered.length > 0 ? filtered : sorted.slice(0, 1);

    for (const post of toShow) {
      const preview = post.text?.slice(0, 300) || '';
      const text = [
        `📌 ${post.channel} | ${formatDate(post.date)}`,
        `🔗 ${post.url}`,
        '',
        preview + (post.text?.length > 300 ? '...' : ''),
      ].join('\n');

      await bot.sendMessage(chatId, text, { disable_web_page_preview: true });
    }
  } catch (err) {
    console.error('[Bot] /search error:', err);
    bot.sendMessage(chatId, 'Ошибка при поиске.');
  }
}

async function handleGenerate(chatId, query) {
  await bot.sendMessage(chatId, `Ищу посты по теме "${query}" и генерирую текст...`);

  try {
    const results = await mqHybridRRF(query, 5);

    if (results.length === 0) {
      return bot.sendMessage(chatId, 'Не найдено постов для генерации.');
    }

    const examples = results
      .map((p, i) => `Пример ${i + 1}:\n${p.text?.slice(0, 500)}`)
      .join('\n\n---\n\n');

    const response = await openai.chat.completions.create({
      model: LLM_MODEL,
      max_tokens: 1000,
      messages: [
        {
          role: 'user',
          content: `Ты помогаешь писать посты о яхтинге для Telegram-канала. Вот примеры постов на тему "${query}":\n\n${examples}\n\nНапиши новый оригинальный пост в похожем стиле на тему "${query}". Пост должен быть живым, интересным, 100-300 слов.`,
        },
      ],
    });

    const generated = response.choices[0].message.content;
    await bot.sendMessage(chatId, `✍️ Сгенерированный пост:\n\n${generated}`);
  } catch (err) {
    console.error('[Bot] /generate error:', err);
    bot.sendMessage(chatId, 'Ошибка при генерации.');
  }
}

bot.onText(/\/collect/, async (msg) => {
  const chatId = msg.chat.id;

  await bot.sendMessage(chatId, 'Запускаю сбор постов...');

  try {
    const { totalCollected, state } = await collect();
    const lines = Object.entries(state).map(
      ([ch, s]) => `${ch}: +${s.collected || 0} (всего ${s.totalPosts})`
    );
    lines.push('', `Итого новых: ${totalCollected}`);
    bot.sendMessage(chatId, lines.join('\n'));
  } catch (err) {
    console.error('[Bot] /collect error:', err);
    bot.sendMessage(chatId, 'Ошибка при сборе.');
  }
});

bot.onText(/\/stats/, async (msg) => {
  const chatId = msg.chat.id;

  try {
    const state = await readState();
    const entries = Object.entries(state);

    if (entries.length === 0) {
      return bot.sendMessage(chatId, 'Каналов пока нет.');
    }

    const ocrStats = await loadOcrStats();

    const lines = entries.map(([ch, s]) => {
      const lastSync = s.lastCollectedAt ? formatDate(s.lastCollectedAt) : 'никогда';
      const archiveStatus = s.allPostsDownloaded ? 'полный' : 'неполный';
      const downloadedRange = s.downloadedDateFrom && s.downloadedDateTo
        ? `${formatDate(s.downloadedDateFrom)} — ${formatDate(s.downloadedDateTo)}`
        : 'нет данных';
      const channelTotal = s.channelTotalPosts != null ? s.channelTotalPosts : '?';
      const downloadProgress = s.channelTotalPosts
        ? ` (${Math.round((s.totalPosts / s.channelTotalPosts) * 100)}%)`
        : '';

      const ocr = ocrStats[ch];
      const ocrLine = ocr && ocr.withMedia > 0
        ? `  OCR: ${ocr.withOcr}/${ocr.withMedia} картинок (${Math.round((ocr.withOcr / ocr.withMedia) * 100)}%)`
        : '';

      return [
        `📌 ${ch} [${s.source}]`,
        `  Скачано: ${s.totalPosts}${downloadProgress} из ${channelTotal} постов`,
        `  Период: ${downloadedRange}`,
        `  Архив: ${archiveStatus} | Последний сбор: ${lastSync}`,
        ocrLine,
      ].filter(Boolean).join('\n');
    });

    const total = entries.reduce((sum, [, s]) => sum + (s.totalPosts || 0), 0);
    const totalOcrMedia = Object.values(ocrStats).reduce((sum, o) => sum + o.withMedia, 0);
    const totalOcrDone = Object.values(ocrStats).reduce((sum, o) => sum + o.withOcr, 0);
    lines.push('', `Всего: ${total} постов`);
    if (totalOcrMedia > 0) {
      lines.push(`OCR: ${totalOcrDone}/${totalOcrMedia} картинок (${Math.round((totalOcrDone / totalOcrMedia) * 100)}%)`);
    }

    bot.sendMessage(chatId, lines.join('\n'));
  } catch (err) {
    console.error('[Bot] /stats error:', err);
    bot.sendMessage(chatId, 'Ошибка при получении статистики.');
  }
});

bot.onText(/\/removechannel (telegram|instagram) (.+)/, async (msg, match) => {
  const source = match[1].trim();
  const username = match[2].trim().replace('@', '');
  const chatId = msg.chat.id;

  try {
    await removeChannel(source, username);
    bot.sendMessage(chatId, `Канал @${username} (${source}) удалён вместе со всеми постами и эмбеддингами.`);
  } catch (err) {
    console.error('[Bot] /removechannel error:', err);
    bot.sendMessage(chatId, `Ошибка при удалении канала: ${err.message}`);
  }
});

bot.onText(/\/addchannel (telegram|instagram) (.+)/, async (msg, match) => {
  const source = match[1].trim();
  const username = match[2].trim().replace('@', '');
  const chatId = msg.chat.id;

  try {
    await addChannel(source, username);
    await bot.sendMessage(chatId, `Добавлен канал @${username} (${source}). Запускаю первичный сбор...`);

    const channelKey = source === 'instagram' ? `ig_${username}` : username;
    const { totalCollected } = await collect([channelKey]);

    bot.sendMessage(chatId, `Готово! Собрано ${totalCollected} постов из @${username}.`);
  } catch (err) {
    console.error('[Bot] /addchannel error:', err);
    bot.sendMessage(chatId, `Ошибка при добавлении канала: ${err.message}`);
  }
});

async function loadOcrStats() {
  const stats = {};
  let files = [];
  try { files = await fs.readdir(POSTS_DIR); } catch {}
  for (const file of files) {
    if (!file.endsWith('.json')) continue;
    const channel = file.replace('.json', '');
    try {
      const posts = JSON.parse(await fs.readFile(path.join(POSTS_DIR, file), 'utf-8'));
      stats[channel] = {
        withMedia: posts.filter(p => p.media).length,
        withOcr: posts.filter(p => p.ocrText).length,
      };
    } catch {}
  }
  return stats;
}

function formatDate(iso) {
  return new Date(iso).toLocaleDateString('ru-RU', {
    day: '2-digit', month: '2-digit', year: 'numeric',
  });
}
