import tesseract from 'node-tesseract-ocr';
import axios from 'axios';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MEDIA_DIR = path.join(__dirname, '../data/media');

const OCR_CONFIG = {
  lang: 'rus+eng',
  oem: 1,
  psm: 3,
};

export async function extractTextFromImage(imageUrl, postId) {
  const localPath = path.join(MEDIA_DIR, `${postId}_ocr.jpg`);

  try {
    await downloadImage(imageUrl, localPath);
    const text = await tesseract.recognize(localPath, OCR_CONFIG);
    return text.trim() || null;
  } catch {
    return null;
  } finally {
    await fs.unlink(localPath).catch(() => {});
  }
}

async function downloadImage(url, localPath) {
  const response = await axios.get(url, { responseType: 'arraybuffer' });
  await fs.writeFile(localPath, response.data);
}
