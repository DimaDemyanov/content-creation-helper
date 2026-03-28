/**
 * Разбивает текст поста на чанки для векторного поиска.
 *
 * Зачем: один эмбеддинг на весь длинный пост усредняет все темы.
 * Если пост содержит несколько тем, релевантный фрагмент тонет в среднем векторе.
 * Чанкинг позволяет найти пост даже если только часть его текста совпадает с запросом.
 *
 * Стратегия:
 *   1. Делим по абзацам (\n\n или \n).
 *   2. Если абзац > MAX_WORDS — делим дальше по предложениям.
 *   3. Очень короткие фрагменты (<30 символов) пропускаем.
 */

const MAX_WORDS = 120;
const MIN_CHARS = 30;

/**
 * Разбивает текст на чанки.
 * @param {string} text
 * @returns {string[]}
 */
export function chunkText(text) {
  if (!text || text.trim().length < MIN_CHARS) return text?.trim() ? [text.trim()] : [];

  const paragraphs = text
    .split(/\n+/)
    .map(s => s.trim())
    .filter(s => s.length >= MIN_CHARS);

  if (paragraphs.length === 0) return [];

  const chunks = [];
  for (const para of paragraphs) {
    if (para.split(/\s+/).length <= MAX_WORDS) {
      chunks.push(para);
    } else {
      // Параграф слишком длинный — делим по предложениям
      const sentences = para.match(/[^.!?]+[.!?]*/g) ?? [para];
      let current = '';
      for (const raw of sentences) {
        const sent = raw.trim();
        if (!sent) continue;
        const candidate = current ? current + ' ' + sent : sent;
        if (candidate.split(/\s+/).length > MAX_WORDS && current) {
          if (current.length >= MIN_CHARS) chunks.push(current);
          current = sent;
        } else {
          current = candidate;
        }
      }
      if (current.length >= MIN_CHARS) chunks.push(current);
    }
  }

  return chunks;
}

/**
 * Возвращает список чанков для поста.
 * Объединяет textClean и ocrText, разделяя переносом строки.
 *
 * @param {{ textClean?: string, ocrText?: string }} post
 * @returns {string[]}
 */
export function postToChunks(post) {
  const text = [post.textClean, post.ocrText].filter(Boolean).join('\n');
  return chunkText(text);
}
