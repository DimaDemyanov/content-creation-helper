import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import OpenAI from 'openai';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_EMBEDDINGS_DIR = path.join(__dirname, '../data/embeddings');
const DEFAULT_CHUNK_EMBEDDINGS_DIR = path.join(__dirname, '../data/chunk-embeddings');
const MODEL = 'text-embedding-3-small';

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

export async function generateEmbedding(text) {
  const response = await openai.embeddings.create({ model: MODEL, input: text });
  return response.data[0].embedding;
}

// Батч: до 100 текстов за раз
export async function generateEmbeddings(texts) {
  const response = await openai.embeddings.create({ model: MODEL, input: texts });
  return response.data.map(d => d.embedding);
}

export function cosineSimilarity(a, b) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

// Загружает все эмбеддинги в Map<id, vector>
export async function loadAllEmbeddings(embeddingsDir = DEFAULT_EMBEDDINGS_DIR) {
  const map = new Map();
  let files;
  try {
    files = await fs.readdir(embeddingsDir);
  } catch {
    return map;
  }
  for (const file of files) {
    if (!file.endsWith('.json')) continue;
    try {
      const raw = await fs.readFile(path.join(embeddingsDir, file), 'utf-8');
      const entries = JSON.parse(raw);
      for (const { id, vector } of entries) {
        map.set(id, vector);
      }
    } catch {}
  }
  return map;
}

export async function saveEmbeddings(channel, newEntries, embeddingsDir = DEFAULT_EMBEDDINGS_DIR) {
  await fs.mkdir(embeddingsDir, { recursive: true });
  const filePath = path.join(embeddingsDir, `${channel}.json`);

  let existing = [];
  try {
    existing = JSON.parse(await fs.readFile(filePath, 'utf-8'));
  } catch {}

  const map = new Map(existing.map(e => [e.id, e.vector]));
  for (const { id, vector } of newEntries) {
    map.set(id, vector);
  }

  await fs.writeFile(filePath, JSON.stringify(Array.from(map.entries()).map(([id, vector]) => ({ id, vector }))));
}

// --- Chunk embeddings ---
// Формат: [{ postId, chunkIdx, vector }]
// Хранятся отдельно от full-post эмбеддингов в data/chunk-embeddings/

/**
 * Загружает чанк-эмбеддинги в Map<postId, { chunkIdx, vector }[]>
 */
export async function loadAllChunkEmbeddings(chunkEmbeddingsDir = DEFAULT_CHUNK_EMBEDDINGS_DIR) {
  const map = new Map(); // postId → [{chunkIdx, vector}]
  let files;
  try {
    files = await fs.readdir(chunkEmbeddingsDir);
  } catch {
    return map;
  }
  for (const file of files) {
    if (!file.endsWith('.json')) continue;
    try {
      const raw = await fs.readFile(path.join(chunkEmbeddingsDir, file), 'utf-8');
      const entries = JSON.parse(raw);
      for (const { postId, chunkIdx, vector } of entries) {
        if (!map.has(postId)) map.set(postId, []);
        map.get(postId).push({ chunkIdx, vector });
      }
    } catch {}
  }
  return map;
}

export async function saveChunkEmbeddings(channel, newEntries, chunkEmbeddingsDir = DEFAULT_CHUNK_EMBEDDINGS_DIR) {
  await fs.mkdir(chunkEmbeddingsDir, { recursive: true });
  const filePath = path.join(chunkEmbeddingsDir, `${channel}.json`);

  let existing = [];
  try {
    existing = JSON.parse(await fs.readFile(filePath, 'utf-8'));
  } catch {}

  // Ключ: postId + chunkIdx
  const map = new Map(existing.map(e => [`${e.postId}:${e.chunkIdx}`, e]));
  for (const entry of newEntries) {
    map.set(`${entry.postId}:${entry.chunkIdx}`, entry);
  }

  await fs.writeFile(chunkEmbeddingsDir + '/' + channel + '.json',
    JSON.stringify(Array.from(map.values())));
}
