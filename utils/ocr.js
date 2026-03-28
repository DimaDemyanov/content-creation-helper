import axios from 'axios';
import OpenAI from 'openai';

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// imageUrl может быть https://... или data:image/jpeg;base64,...
export async function extractTextFromImage(imageUrl) {
  try {
    let dataUrl;
    if (imageUrl.startsWith('data:')) {
      dataUrl = imageUrl;
    } else {
      const response = await axios.get(imageUrl, { responseType: 'arraybuffer' });
      const base64 = Buffer.from(response.data).toString('base64');
      const mimeType = response.headers['content-type'] || 'image/jpeg';
      dataUrl = `data:${mimeType};base64,${base64}`;
    }

    const result = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      max_tokens: 500,
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'image_url',
              image_url: { url: dataUrl },
            },
            {
              type: 'text',
              text: 'Если на изображении есть текст — выпиши его. Если текста нет или он нечитаем — ответь только словом "нет".',
            },
          ],
        },
      ],
    });

    const text = result.choices[0].message.content.trim();
    if (!text || text.toLowerCase() === 'нет') return null;
    return text;
  } catch {
    return null;
  }
}
