import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import OpenAI from 'openai';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_EMBEDDINGS_DIR = path.join(__dirname, '../data/embeddings');
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
