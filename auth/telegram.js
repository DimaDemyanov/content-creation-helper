import 'dotenv/config';
import { TelegramClient } from 'telegram';
import { StringSession } from 'telegram/sessions/index.js';
import { input } from '@inquirer/prompts';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SESSION_FILE = path.join(__dirname, '../data/telegram.session');

const session = new StringSession('');
const client = new TelegramClient(
  session,
  Number(process.env.TELEGRAM_API_ID),
  process.env.TELEGRAM_API_HASH,
  { connectionRetries: 3 }
);

await client.start({
  phoneNumber: async () => input({ message: 'Номер телефона (+7...)' }),
  password: async () => input({ message: 'Пароль 2FA (если есть)' }),
  phoneCode: async () => input({ message: 'Код из Telegram' }),
  onError: (err) => console.error(err),
});

console.log('Авторизация успешна!');

const sessionStr = client.session.save();
await fs.writeFile(SESSION_FILE, sessionStr);
console.log('Сессия сохранена в data/telegram.session');

await client.disconnect();
