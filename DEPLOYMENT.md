# Deployment Runbook (VPS + systemd)

Проект запускается на VPS через `systemd` сервис:
- `content-helper-bot.service`

Бот включает встроенный cron-коллектор — отдельный сервис для сборщика не нужен.

## 1. Первичная установка на сервере

```bash
# Клонировать репозиторий
git clone https://github.com/DimaDemyanov/content-creation-helper.git /opt/content-creation-helper
cd /opt/content-creation-helper

# Установить зависимости
npm install

# Создать .env (заполни реальными значениями)
cp .env.example .env
nano .env
```

## 2. Перенос локальных данных на сервер

Если данные (посты, эмбеддинги, сессия GramJS) уже собраны локально — скопируй их через `rsync`. Запускать с **локальной** машины:

```bash
rsync -avz --progress \
  /Users/dmitriidemianov/projects/content-creation-helper/data/ \
  root@77.42.43.113:/opt/content-creation-helper/data/
```

Копирует всё содержимое `data/` — посты, эмбеддинги, `state.json`, и `telegram.session`.
Если `telegram.session` скопирован успешно, шаг 3 (авторизация GramJS) можно пропустить.

## 3. Одноразовая авторизация GramJS

**Только при первом деплое.** GramJS требует интерактивный вход (номер телефона + код из Telegram).
После этого сессия сохраняется на диске и повторная авторизация не нужна.

```bash
cd /opt/content-creation-helper
node auth/telegram.js
```

Введи номер телефона и код подтверждения. После успешной авторизации файл сессии появится рядом.

## 4. Создание systemd-сервиса

```bash
sudo nano /etc/systemd/system/content-helper-bot.service
```

Содержимое файла:

```ini
[Unit]
Description=Content Creation Helper Bot
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/content-creation-helper
ExecStart=/usr/bin/node bot/index.js
Restart=on-failure
RestartSec=10
EnvironmentFile=/opt/content-creation-helper/.env

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable content-helper-bot.service
sudo systemctl start content-helper-bot.service
```

## 5. Подключение к серверу

```bash
ssh root@77.42.43.113
```

## 6. Проверка статуса сервиса

```bash
sudo systemctl status content-helper-bot.service --no-pager
```

## 7. Просмотр логов

Последние 300 строк:

```bash
journalctl -u content-helper-bot.service -n 300 --no-pager
```

Стрим в реальном времени:

```bash
journalctl -u content-helper-bot.service -f
```

Фильтр по ошибкам и сбору:

```bash
journalctl -u content-helper-bot.service -n 1000 --no-pager | grep -E "error|Error|collect|Collect"
```

## 8. Обновление кода и деплой

```bash
cd /opt/content-creation-helper
git pull --ff-only
npm install
sudo systemctl restart content-helper-bot.service
```

Проверка после рестарта:

```bash
sudo systemctl status content-helper-bot.service --no-pager
```

Ожидаемая строка в логах: `[Bot] Запущен`

## 9. Скрипты (запускать вручную при необходимости)

Бэкфилл OCR для существующих постов:

```bash
cd /opt/content-creation-helper
node --env-file=.env scripts/backfill-ocr.js --source instagram
node --env-file=.env scripts/backfill-ocr.js --source telegram
```

Генерация / обновление эмбеддингов:

```bash
node --env-file=.env scripts/backfill-embeddings.js
```

Ручной бэкап данных (архив `data/` → облако через rclone):

```bash
node --env-file=.env scripts/backup.js
```

## 10. Бэкапы (rclone → Google Drive)

Бот сам архивирует `data/` и заливает в облако раз в месяц (1-е число, 04:00). Также доступна команда `/backup` в самом боте для запуска вручную.

**Настройка на сервере (один раз):**

```bash
# Установить rclone
curl https://rclone.org/install.sh | sudo bash

# Настроить remote с именем "gdrive" (интерактивный мастер, попросит авторизацию Google)
rclone config
```

В `.env` можно (не обязательно) переопределить:

```bash
RCLONE_REMOTE=gdrive                              # имя remote из `rclone config`
BACKUP_RCLONE_PATH=content-creation-helper-backups # путь/папка в облаке
BACKUP_KEEP_LOCAL=3                                # сколько последних архивов хранить локально
TELEGRAM_ADMIN_CHAT_ID=                            # chat_id — бот пришлёт туда отчёт об успехе/ошибке бэкапа
```

Локальные архивы лежат в `backups/` (в git не попадают), старые чистятся автоматически.

## 11. Частые проблемы

- **Бот не отвечает** — проверь `TELEGRAM_BOT_TOKEN` в `.env` и статус сервиса.
- **GramJS ошибка сессии** — повтори `node auth/telegram.js` и перезапусти сервис.
- **Cron-сборщик не работает** — смотри логи на строки `[Collector]`; проверь `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`, `APIFY_API_TOKEN`.
- **OpenAI 429 / "закончились токены"** — бот теперь сам показывает понятное сообщение об ошибке; для бэкфилла OCR/эмбеддингов используется экспоненциальный backoff, дождись завершения или снизь нагрузку.
- **Данные не обновляются** — `data/state.json` хранит `lastCollectedAt`; при необходимости сбрось вручную.
- **Бэкап падает с `spawn rclone ENOENT`** — rclone не установлен или remote не настроен, см. раздел 10.
