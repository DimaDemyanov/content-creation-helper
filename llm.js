/**
 * Общий LLM-клиент. Переключается между провайдерами через LLM_PROVIDER в .env.
 *
 * LLM_PROVIDER=openai  (по умолчанию) — gpt-4o-mini
 * LLM_PROVIDER=groq                   — llama-3.3-70b-versatile (Groq API)
 * LLM_PROVIDER=grok                   — grok-3-mini (xAI API)
 */

import OpenAI from 'openai';

const PROVIDERS = {
  openai: {
    baseURL: undefined,
    apiKey: process.env.OPENAI_API_KEY,
    model: 'gpt-4o-mini',
  },
  groq: {
    baseURL: 'https://api.groq.com/openai/v1',
    apiKey: process.env.XAI_API_KEY,
    model: 'llama-3.3-70b-versatile',
  },
  grok: {
    baseURL: 'https://api.x.ai/v1',
    apiKey: process.env.XAI_API_KEY,
    model: 'grok-3-mini',
  },
};

const provider = PROVIDERS[process.env.LLM_PROVIDER] ?? PROVIDERS.openai;

export const llmClient = new OpenAI({
  apiKey: provider.apiKey,
  ...(provider.baseURL ? { baseURL: provider.baseURL } : {}),
});

export const LLM_MODEL = provider.model;
