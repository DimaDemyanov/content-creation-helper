import { describe, it, expect } from 'vitest';
import { cleanText } from '../utils/textClean.js';

describe('cleanText', () => {
  it('удаляет эмодзи', () => {
    expect(cleanText('Вышли в море 🌊⚓🛥')).toBe('Вышли в море');
  });

  it('удаляет хэштеги', () => {
    expect(cleanText('Отличная регата #яхтинг #sailing #море')).toBe('Отличная регата');
  });

  it('удаляет URL', () => {
    expect(cleanText('Подробнее на https://yacht.ru/news/123')).toBe('Подробнее на');
  });

  it('удаляет эмодзи, хэштеги и URL одновременно', () => {
    const input = '⚓ Старт регаты #яхтинг https://t.me/seapinta/42';
    expect(cleanText(input)).toBe('Старт регаты');
  });

  it('оставляет русский текст нетронутым', () => {
    expect(cleanText('Яхта вышла в открытое море')).toBe('Яхта вышла в открытое море');
  });

  it('оставляет английский текст нетронутым', () => {
    expect(cleanText('Sailing in the open sea')).toBe('Sailing in the open sea');
  });

  it('убирает лишние пробелы', () => {
    expect(cleanText('яхта   вышла    в море')).toBe('яхта вышла в море');
  });

  it('возвращает пустую строку на null', () => {
    expect(cleanText(null)).toBe('');
  });

  it('возвращает пустую строку на пустую строку', () => {
    expect(cleanText('')).toBe('');
  });
});
