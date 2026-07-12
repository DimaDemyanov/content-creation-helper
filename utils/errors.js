/**
 * Преобразует ошибки внешних API (OpenAI/Groq/xAI, Apify, GramJS, сеть)
 * в понятное для пользователя бота сообщение на русском.
 */

const NETWORK_CODES = new Set(['ECONNREFUSED', 'ETIMEDOUT', 'ENOTFOUND', 'ECONNRESET', 'EAI_AGAIN']);

export function formatUserError(err) {
  const status = err?.status ?? err?.response?.status;
  const code = err?.code ?? err?.error?.code;
  const type = err?.type ?? err?.error?.type;
  const message = String(err?.message || '');

  if (code === 'insufficient_quota' || type === 'insufficient_quota' || /exceeded your current quota/i.test(message)) {
    return '⚠️ Закончился баланс/токены у LLM-провайдера. Пополните счёт (OpenAI: platform.openai.com/account/billing) и повторите запрос.';
  }

  if (status === 401 || code === 'invalid_api_key' || /incorrect api key/i.test(message)) {
    return '⚠️ Неверный API-ключ у LLM-провайдера. Проверьте .env на сервере.';
  }

  if (status === 429 || code === 'rate_limit_exceeded' || /rate limit/i.test(message)) {
    return '⏳ Превышен лимит запросов к API. Попробуйте через минуту.';
  }

  const floodMatch = message.match(/FLOOD_WAIT_(\d+)/i) || message.match(/wait of (\d+) seconds/i);
  if (floodMatch) {
    const seconds = Number(floodMatch[1]);
    const human = seconds >= 60 ? `${Math.ceil(seconds / 60)} мин` : `${seconds} сек`;
    return `⏳ Telegram ограничил частоту запросов. Подождите ~${human} и попробуйте снова.`;
  }

  if (/AUTH_KEY|SESSION_REVOKED|SESSION_EXPIRED|не авторизован/i.test(message)) {
    return '⚠️ Сессия Telegram истекла или не настроена. На сервере: node auth/telegram.js';
  }

  if (/apify/i.test(message) && /token|unauthoriz|402|payment/i.test(message)) {
    return '⚠️ Проблема с Apify API (Instagram): проверьте APIFY_API_TOKEN и баланс аккаунта.';
  }

  if (NETWORK_CODES.has(err?.code) || NETWORK_CODES.has(code)) {
    return '⚠️ Нет соединения с внешним сервисом. Проверьте интернет/доступность API и попробуйте снова.';
  }

  const short = message.length > 300 ? message.slice(0, 300) + '…' : message;
  return `⚠️ Ошибка: ${short || 'неизвестная ошибка'}`;
}
