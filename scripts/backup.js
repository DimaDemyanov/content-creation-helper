/**
 * Бэкап data/ (посты, эмбеддинги, state.json, сессия GramJS) в архив + заливка в облако через rclone.
 *
 * Настройка на сервере (один раз):
 *   1. Установить rclone: curl https://rclone.org/install.sh | sudo bash
 *   2. Настроить remote: rclone config  (создать remote с именем из RCLONE_REMOTE, по умолчанию "gdrive")
 *
 * Переменные окружения (.env):
 *   RCLONE_REMOTE          — имя rclone-remote (по умолчанию "gdrive")
 *   BACKUP_RCLONE_PATH      — путь в облаке (по умолчанию "content-creation-helper-backups")
 *   BACKUP_KEEP_LOCAL       — сколько последних архивов хранить локально (по умолчанию 3)
 *   TELEGRAM_ADMIN_CHAT_ID  — chat_id, куда бот шлёт отчёт об успехе/ошибке бэкапа (опционально)
 *
 * Запуск вручную: node scripts/backup.js
 */

import 'dotenv/config';
import { execFile } from 'child_process';
import { promisify } from 'util';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const execFileAsync = promisify(execFile);

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = path.join(__dirname, '..');
const BACKUPS_DIR = path.join(PROJECT_ROOT, 'backups');

const RCLONE_REMOTE = process.env.RCLONE_REMOTE || 'gdrive';
const BACKUP_RCLONE_PATH = process.env.BACKUP_RCLONE_PATH || 'content-creation-helper-backups';
const KEEP_LOCAL = Number(process.env.BACKUP_KEEP_LOCAL) || 3;

function archiveName() {
  const date = new Date().toISOString().slice(0, 10); // YYYY-MM-DD
  return `backup-${date}.tar.gz`;
}

async function createArchive() {
  await fs.mkdir(BACKUPS_DIR, { recursive: true });
  const archivePath = path.join(BACKUPS_DIR, archiveName());

  await execFileAsync('tar', [
    '-czf', archivePath,
    '--exclude=.DS_Store',
    '-C', PROJECT_ROOT,
    'data',
  ]);

  const { size } = await fs.stat(archivePath);
  return { archivePath, sizeMB: (size / 1024 / 1024).toFixed(1) };
}

async function uploadToCloud(archivePath) {
  await execFileAsync('rclone', [
    'copy', archivePath,
    `${RCLONE_REMOTE}:${BACKUP_RCLONE_PATH}`,
  ]);
}

async function pruneLocalBackups() {
  const files = (await fs.readdir(BACKUPS_DIR))
    .filter(f => f.startsWith('backup-') && f.endsWith('.tar.gz'))
    .sort(); // имена содержат ISO-дату, лексикографическая сортировка = хронологическая

  const toDelete = files.slice(0, Math.max(0, files.length - KEEP_LOCAL));
  for (const file of toDelete) {
    await fs.rm(path.join(BACKUPS_DIR, file), { force: true });
  }
}

/**
 * Выполняет полный цикл: архивация → заливка в облако → чистка старых локальных копий.
 * @param {object} [bot] - опциональный экземпляр node-telegram-bot-api для отправки отчёта.
 */
export async function runBackup(bot) {
  const adminChatId = process.env.TELEGRAM_ADMIN_CHAT_ID;
  const notify = async (text) => {
    console.log(`[Backup] ${text.replace(/\n/g, ' ')}`);
    if (bot && adminChatId) {
      try { await bot.sendMessage(adminChatId, text); } catch (e) { console.error('[Backup] Не удалось отправить уведомление:', e.message); }
    }
  };

  try {
    const { archivePath, sizeMB } = await createArchive();
    await uploadToCloud(archivePath);
    await pruneLocalBackups();
    await notify(`✅ Бэкап данных выполнен: ${path.basename(archivePath)} (${sizeMB} МБ) → ${RCLONE_REMOTE}:${BACKUP_RCLONE_PATH}`);
    return { archivePath, sizeMB };
  } catch (err) {
    const hint = err.code === 'ENOENT'
      ? ' (проверь, что установлены tar и rclone, и remote настроен: rclone config)'
      : '';
    await notify(`❌ Бэкап данных не выполнен: ${err.message}${hint}`);
    throw err;
  }
}

const isMain = process.argv[1] && import.meta.url.endsWith(process.argv[1].replace(/\\/g, '/'));
if (isMain) {
  runBackup().catch(() => process.exit(1));
}
