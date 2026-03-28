import axios from 'axios';
import OpenAI from 'openai';

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const OCR_PROMPT = 'Если на изображении есть текст — выпиши его. Если текста нет или он нечитаем — ответь только словом "нет".';

// imageUrl может быть https://... или data:image/jpeg;base64,...
export async function extractTextFromImage(imageUrl) {
  let dataUrl;
  try {
    if (imageUrl.startsWith('data:')) {
      dataUrl = imageUrl;
    } else {
      const response = await axios.get(imageUrl, { responseType: 'arraybuffer', timeout: 15000 });
      const base64 = Buffer.from(response.data).toString('base64');
      const mimeType = response.headers['content-type'] || 'image/jpeg';
      dataUrl = `data:${mimeType};base64,${base64}`;
    }
  } catch {
    return null;
  }

  return callVision(dataUrl, 0);
}

const MAX_RETRY_MS = 60_000;

async function callVision(dataUrl, attempt) {
  try {
    const result = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      max_tokens: 500,
      messages: [
        {
          role: 'user',
          content: [
            { type: 'image_url', image_url: { url: dataUrl } },
            { type: 'text', text: OCR_PROMPT },
          ],
        },
      ],
    });

    const text = result.choices[0].message.content.trim();
    if (!text || text.toLowerCase() === 'нет') return null;
    return text;
  } catch (err) {
    if (err?.status !== 429) return null;

    // Экспоненциальный backoff: 2s, 4s, 8s, 16s, 32s, 60s, 60s, ...
    const expDelay = Math.min(1_000 * Math.pow(2, attempt), MAX_RETRY_MS);
    const apiDelay = parseRetryAfter(err) || 0;
    const delay = Math.max(expDelay, apiDelay);

    console.log(`[OCR] Rate limit (попытка ${attempt + 1}), повтор через ${Math.round(delay / 1000)}s...`);
    await new Promise(r => setTimeout(r, delay));
    return callVision(dataUrl, attempt + 1);
  }
}

function parseRetryAfter(err) {
  const raw = err?.headers?.['retry-after-ms'] || err?.headers?.['retry-after'];
  if (!raw) return null;
  const n = Number(raw);
  if (isNaN(n)) return null;
  return err?.headers?.['retry-after-ms'] ? n : n * 1000;
}
