import { ApifyClient } from 'apify-client';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { cleanText } from '../utils/textClean.js';
import { extractTextFromImage } from '../utils/ocr.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const POSTS_DIR = path.join(__dirname, '../data/posts');

const client = new ApifyClient({ token: process.env.APIFY_API_TOKEN });

export async function collectInstagramAccount(account, channelState, firstRunLimit = 500) {
  const since = channelState.lastCollectedAt || null;
  const limit = since ? 50 : firstRunLimit;

  console.log(`[Instagram] Сбор из @${account} (limit: ${limit}, since: ${since || 'начало'})...`);

  const key = `ig_${account}`;
  const { totalCollected, allPostsDownloaded, channelTotalPosts } = await fetchAndSaveAccountPosts(account, since, limit, key);
  const downloadedStats = await computeDownloadedStats(key);

  return {
    collected: totalCollected,
    allPostsDownloaded,
    lastCollectedAt: new Date().toISOString(),
    totalPosts: downloadedStats.count,
    downloadedDateFrom: downloadedStats.dateFrom,
    downloadedDateTo: downloadedStats.dateTo,
    channelTotalPosts: channelTotalPosts ?? null,
    channelDateFrom: null,
    channelDateTo: null,
  };
}

async function fetchAndSaveAccountPosts(account, since, limit, key) {
  const input = {
    directUrls: [`https://www.instagram.com/${account}/`],
    resultsLimit: limit,
  };

  const run = await client.actor('apify/instagram-scraper').call(input);
  const { items } = await client.dataset(run.defaultDatasetId).listItems();

  const channelTotalPosts = items[0]?.ownerFullName ? null : items[0]?.postsCount ?? null;

  const sinceDate = since ? new Date(since) : null;
  const filtered = sinceDate
    ? items.filter(item => new Date(item.timestamp) > sinceDate)
    : items;

  const BATCH_SIZE = 50;
  let totalCollected = 0;

  for (let i = 0; i < filtered.length; i += BATCH_SIZE) {
    const batch = [];
    for (const item of filtered.slice(i, i + BATCH_SIZE)) {
      const post = await buildPost(item, account);
      batch.push(post);
      if (item.displayUrl || item.thumbnailUrl) await sleep(1500);
    }
    await savePosts(key, batch);
    totalCollected += batch.length;
    console.log(`[Instagram] @${account}: сохранён батч ${batch.length} постов (всего: ${totalCollected})`);
  }

  return { totalCollected, allPostsDownloaded: filtered.length < limit, channelTotalPosts };
}

async function buildPost(item, account) {
  const text = item.caption || '';
  const textClean = cleanText(text);

  let ocrText = null;
  const imageUrl = item.displayUrl || item.thumbnailUrl;

  if (imageUrl) {
    ocrText = await extractTextFromImage(imageUrl);
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

async function savePosts(key, posts) {
  if (posts.length === 0) return;

  const filePath = path.join(POSTS_DIR, `${key}.json`);
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

async function loadPosts(key) {
  const filePath = path.join(POSTS_DIR, `${key}.json`);
  try {
    const raw = await fs.readFile(filePath, 'utf-8');
    return JSON.parse(raw);
  } catch {
    return [];
  }
}

async function computeDownloadedStats(key) {
  const posts = await loadPosts(key);
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
