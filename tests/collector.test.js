import { describe, it, expect, vi, beforeEach } from 'vitest';

const { MOCK_CONFIG_STR, MOCK_STATE_STR } = vi.hoisted(() => {
  const MOCK_CONFIG_STR = JSON.stringify({
    telegram: { channels: ['seapinta'] },
    instagram: { accounts: ['clevel.yacht'], firstRunLimit: 500 },
    collect: { schedule: '0 9 * * *' },
  });
  const MOCK_STATE_STR = JSON.stringify({
    seapinta: {
      source: 'telegram',
      lastCollectedAt: null,
      allPostsDownloaded: false,
      totalPosts: 0,
    },
  });
  return { MOCK_CONFIG_STR, MOCK_STATE_STR };
});

vi.mock('fs/promises', () => ({
  default: {
    readFile: vi.fn().mockImplementation((filePath) => {
      if (String(filePath).includes('config.json')) return Promise.resolve(MOCK_CONFIG_STR);
      if (String(filePath).includes('state.json')) return Promise.resolve(MOCK_STATE_STR);
      return Promise.reject(new Error('file not found'));
    }),
    writeFile: vi.fn().mockResolvedValue(undefined),
  },
}));

vi.mock('../collector/telegram.js', () => ({
  collectTelegramChannel: vi.fn(),
  disconnectTelegram: vi.fn(),
}));

vi.mock('../collector/instagram.js', () => ({
  collectInstagramAccount: vi.fn(),
}));

vi.mock('node-cron', () => ({ default: { schedule: vi.fn() } }));
vi.mock('dotenv/config', () => ({}));

import fs from 'fs/promises';
import { collect, addChannel, readState } from '../collector/index.js';
import { collectTelegramChannel } from '../collector/telegram.js';
import { collectInstagramAccount } from '../collector/instagram.js';

const MOCK_STATE = JSON.parse(MOCK_STATE_STR);

beforeEach(() => {
  vi.clearAllMocks();
  fs.readFile.mockImplementation((filePath) => {
    if (String(filePath).includes('config.json')) return Promise.resolve(MOCK_CONFIG_STR);
    if (String(filePath).includes('state.json')) return Promise.resolve(MOCK_STATE_STR);
    return Promise.reject(new Error('file not found'));
  });
  fs.writeFile.mockResolvedValue(undefined);
});

describe('collect', () => {
  it('вызывает collectTelegramChannel для каждого канала', async () => {
    collectTelegramChannel.mockResolvedValue({
      collected: 50, allPostsDownloaded: true, lastCollectedAt: '2026-03-27T09:00:00Z', totalPosts: 50,
    });
    collectInstagramAccount.mockResolvedValue({
      collected: 30, allPostsDownloaded: false, lastCollectedAt: '2026-03-27T09:00:00Z', totalPosts: 30,
    });

    const result = await collect();
    expect(collectTelegramChannel).toHaveBeenCalledWith('seapinta', MOCK_STATE.seapinta);
    expect(result.totalCollected).toBe(80);
  });

  it('сохраняет state после каждого канала', async () => {
    collectTelegramChannel.mockResolvedValue({
      collected: 10, allPostsDownloaded: true, lastCollectedAt: '2026-03-27T09:00:00Z', totalPosts: 10,
    });
    collectInstagramAccount.mockResolvedValue({
      collected: 5, allPostsDownloaded: true, lastCollectedAt: '2026-03-27T09:00:00Z', totalPosts: 5,
    });

    await collect();
    expect(fs.writeFile).toHaveBeenCalledWith(
      expect.stringContaining('state.json'),
      expect.any(String)
    );
  });

  it('продолжает сбор если один канал упал', async () => {
    collectTelegramChannel.mockRejectedValue(new Error('connection error'));
    collectInstagramAccount.mockResolvedValue({
      collected: 20, allPostsDownloaded: true, lastCollectedAt: '2026-03-27T09:00:00Z', totalPosts: 20,
    });

    const result = await collect();
    expect(result.totalCollected).toBe(20);
  });

  it('обновляет allPostsDownloaded в state', async () => {
    collectTelegramChannel.mockResolvedValue({
      collected: 50, allPostsDownloaded: true, lastCollectedAt: '2026-03-27T09:00:00Z', totalPosts: 50,
    });
    collectInstagramAccount.mockResolvedValue({
      collected: 0, allPostsDownloaded: true, lastCollectedAt: '2026-03-27T09:00:00Z', totalPosts: 0,
    });

    const { state } = await collect();
    expect(state.seapinta.allPostsDownloaded).toBe(true);
  });

  it('передаёт lastCollectedAt из state в collector', async () => {
    const stateWithDate = JSON.stringify({
      seapinta: { ...MOCK_STATE.seapinta, lastCollectedAt: '2026-03-01T00:00:00Z', allPostsDownloaded: true },
    });
    fs.readFile.mockImplementation((filePath) => {
      if (String(filePath).includes('config.json')) return Promise.resolve(MOCK_CONFIG_STR);
      if (String(filePath).includes('state.json')) return Promise.resolve(stateWithDate);
    });
    collectTelegramChannel.mockResolvedValue({
      collected: 5, allPostsDownloaded: true, lastCollectedAt: '2026-03-27T09:00:00Z', totalPosts: 55,
    });
    collectInstagramAccount.mockResolvedValue({
      collected: 0, allPostsDownloaded: true, lastCollectedAt: '2026-03-27T09:00:00Z', totalPosts: 0,
    });

    await collect();
    expect(collectTelegramChannel).toHaveBeenCalledWith(
      'seapinta',
      expect.objectContaining({ lastCollectedAt: '2026-03-01T00:00:00Z' })
    );
  });
});

describe('addChannel', () => {
  it('добавляет новый telegram канал в config и state', async () => {
    await addChannel('telegram', 'newchannel');

    const configCall = fs.writeFile.mock.calls.find(c => String(c[0]).includes('config.json'));
    const savedConfig = JSON.parse(configCall[1]);
    expect(savedConfig.telegram.channels).toContain('newchannel');

    const stateCall = fs.writeFile.mock.calls.find(c => String(c[0]).includes('state.json'));
    const savedState = JSON.parse(stateCall[1]);
    expect(savedState.newchannel).toMatchObject({
      source: 'telegram',
      lastCollectedAt: null,
      allPostsDownloaded: false,
    });
  });

  it('добавляет новый instagram аккаунт с префиксом ig_', async () => {
    await addChannel('instagram', 'newaccount');

    const stateCall = fs.writeFile.mock.calls.find(c => String(c[0]).includes('state.json'));
    const savedState = JSON.parse(stateCall[1]);
    expect(savedState['ig_newaccount']).toMatchObject({ source: 'instagram' });
  });

  it('не дублирует канал если он уже есть', async () => {
    await addChannel('telegram', 'seapinta');

    const configCall = fs.writeFile.mock.calls.find(c => String(c[0]).includes('config.json'));
    if (configCall) {
      const savedConfig = JSON.parse(configCall[1]);
      const count = savedConfig.telegram.channels.filter(c => c === 'seapinta').length;
      expect(count).toBe(1);
    }
  });
});

describe('readState', () => {
  it('возвращает state из файла', async () => {
    const state = await readState();
    expect(state).toEqual(MOCK_STATE);
  });

  it('возвращает пустой объект если файл не существует', async () => {
    fs.readFile.mockRejectedValue(new Error('ENOENT'));
    const state = await readState();
    expect(state).toEqual({});
  });
});
