import { describe, it, expect, vi, beforeEach } from 'vitest';

// Мокаем fs и OpenAI до импорта модуля
vi.mock('fs/promises', () => ({
  default: {
    readdir: vi.fn(),
    readFile: vi.fn(),
  },
}));

vi.mock('openai', () => ({
  default: vi.fn().mockImplementation(() => ({
    chat: {
      completions: {
        create: vi.fn().mockResolvedValue({
          choices: [{ message: { content: 'шквал, буря, волнение, непогода' } }],
        }),
      },
    },
  })),
}));

import fs from 'fs/promises';
import { search, getStats } from '../search/index.js';

const POSTS = [
  {
    id: 'seapinta_1',
    channel: 'seapinta',
    date: '2026-01-01T00:00:00Z',
    text: 'The yacht sailed into a heavy storm with waves reaching 4 meters',
    textClean: 'The yacht sailed into a heavy storm with waves reaching 4 meters',
    ocrText: null,
    url: 'https://t.me/seapinta/1',
  },
  {
    id: 'seapinta_2',
    channel: 'seapinta',
    date: '2026-01-02T00:00:00Z',
    text: 'Beautiful sunset at the marina with yachts moored at the dock',
    textClean: 'Beautiful sunset at the marina with yachts moored at the dock',
    ocrText: null,
    url: 'https://t.me/seapinta/2',
  },
  {
    id: 'seapinta_3',
    channel: 'seapinta',
    date: '2026-01-03T00:00:00Z',
    text: 'Storm intensifies with squall winds up to 25 knots severe storm warning',
    textClean: 'Storm intensifies with squall winds up to 25 knots severe storm warning',
    ocrText: null,
    url: 'https://t.me/seapinta/3',
  },
  {
    id: 'seapinta_4',
    channel: 'seapinta',
    date: '2026-01-04T00:00:00Z',
    text: 'The regatta started in strong wind conditions near the coast',
    textClean: 'The regatta started in strong wind conditions near the coast',
    ocrText: null,
    url: 'https://t.me/seapinta/4',
  },
  {
    id: 'seapinta_5',
    channel: 'seapinta',
    date: '2026-01-05T00:00:00Z',
    text: 'Yacht moored at the marina after a long offshore passage',
    textClean: 'Yacht moored at the marina after a long offshore passage',
    ocrText: null,
    url: 'https://t.me/seapinta/5',
  },
  {
    id: 'seapinta_6',
    channel: 'seapinta',
    date: '2026-01-06T00:00:00Z',
    text: 'Rigging maintenance and inspection before the sailing season begins',
    textClean: 'Rigging maintenance and inspection before the sailing season begins',
    ocrText: null,
    url: 'https://t.me/seapinta/6',
  },
];

beforeEach(() => {
  vi.clearAllMocks();
  fs.readdir.mockResolvedValue(['seapinta.json']);
  fs.readFile.mockResolvedValue(JSON.stringify(POSTS));
});

describe('search', () => {
  it('возвращает пустой массив если постов нет', async () => {
    fs.readdir.mockResolvedValue([]);
    const results = await search('шторм');
    expect(results).toEqual([]);
  });

  it('находит посты по ключевому слову', async () => {
    const results = await search('storm');
    expect(results.length).toBeGreaterThan(0);
    expect(results[0].id).toBeDefined();
  });

  it('посты с большим количеством упоминаний выше в выдаче', async () => {
    const results = await search('storm');
    // seapinta_3 содержит "storm" дважды, seapinta_1 — один раз
    const ids = results.map(r => r.id);
    expect(ids.indexOf('seapinta_3')).toBeLessThan(ids.indexOf('seapinta_1'));
  });

  it('не возвращает больше topK результатов', async () => {
    const results = await search('яхта', 2);
    expect(results.length).toBeLessThanOrEqual(2);
  });

  it('каждый результат содержит score', async () => {
    const results = await search('шторм');
    for (const r of results) {
      expect(r.score).toBeDefined();
      expect(typeof r.score).toBe('number');
    }
  });

  it('не падает если OpenAI недоступен', async () => {
    const OpenAI = (await import('openai')).default;
    OpenAI.mockImplementationOnce(() => ({
      chat: { completions: { create: vi.fn().mockRejectedValue(new Error('network error')) } },
    }));
    const results = await search('storm');
    expect(results.length).toBeGreaterThan(0);
  });
});

describe('getStats', () => {
  it('возвращает количество постов по каждому каналу', async () => {
    const stats = await getStats();
    expect(stats).toEqual({ seapinta: 6 });
  });

  it('возвращает пустой объект если постов нет', async () => {
    fs.readdir.mockResolvedValue([]);
    const stats = await getStats();
    expect(stats).toEqual({});
  });

  it('игнорирует не-json файлы', async () => {
    fs.readdir.mockResolvedValue(['seapinta.json', '.DS_Store', 'readme.txt']);
    const stats = await getStats();
    expect(Object.keys(stats)).toEqual(['seapinta']);
  });
});
