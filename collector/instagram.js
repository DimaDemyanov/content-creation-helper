import { ApifyClient } from 'apify-client';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { cleanText } from '../utils/textClean.js';
import { extractTextFromImage } from '../utils/ocr.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const POSTS_DIR = path.join(__dirname, '../data/posts');

const client = new ApifyClient({ token: process.env.APIFY_API_TOKEN });

export async function collectInstagram(accounts, since = null, firstRunLimit = 500) {
  const results = { collected: 0, channels: {} };

  for (const account of accounts) {
    console.log(`[Instagram] Сбор из @${account}...`);
    try {
      const posts = await fetchAccountPosts(account, since, firstRunLimit);
      await savePosts(account, posts);
      results.collected += posts.length;
      results.channels[`ig_${account}`] = posts.length;
      console.log(`[Instagram] @${account}: собрано ${posts.length} постов`);
    } catch (err) {
      console.error(`[Instagram] Ошибка @${account}:`, err.message);
    }
  }

  return results;
}

async function fetchAccountPosts(account, since, limit) {
  const input = {
    directUrls: [`https://www.instagram.com/${account}/`],
    resultsLimit: since ? 50 : limit,
  };

  const run = await client.actor('apify/instagram-scraper').call(input);
  const { items } = await client.dataset(run.defaultDatasetId).listItems();

  const sinceDate = since ? new Date(since) : null;

  const filtered = sinceDate
    ? items.filter(item => new Date(item.timestamp) > sinceDate)
    : items;

  const posts = [];
  for (const item of filtered) {
    const post = await buildPost(item, account);
    posts.push(post);
  }

  return posts;
}

async function buildPost(item, account) {
  const text = item.caption || '';
  const textClean = cleanText(text);

  let ocrText = null;
  const imageUrl = item.displayUrl || item.thumbnailUrl;

  if (imageUrl) {
    ocrText = await extractTextFromImage(imageUrl, `ig_${item.shortCode}`);
  }

  return {
    id: `ig_${account}_${item.shortCode}`,
    source: 'instagram',
    channel: account,
    url: item.url,
    date: new Date(item.timestamp).toISOString(),
    text,
    textClean,
    ocrText,
    media: imageUrl ? [{ type: 'photo', url: imageUrl }] : null,
    stats: {
      views: item.videoViewCount || 0,
      likes: item.likesCount || 0,
      comments: item.commentsCount || 0,
    },
    collectedAt: new Date().toISOString(),
  };
}

async function savePosts(account, posts) {
  if (posts.length === 0) return;

  const filePath = path.join(POSTS_DIR, `ig_${account}.json`);
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
