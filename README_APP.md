# QuantDesk macOS App

Build the desktop app:

```bash
bun run app:mac
```

Open the built app:

```bash
open dist/QuantDesk.app
```

The app bundle includes `server.py` and `static/`. On launch it starts the local FastAPI server on port `8888`, waits for `/api/stocks`, then renders QuantDesk inside its own native macOS window using `WKWebView`.

Logs are written to:

```text
~/Library/Application Support/QuantDesk/logs/server.log
```

Optional Claude AI analysis key locations:

```text
~/Library/Application Support/QuantDesk/config.json
~/Library/Application Support/QuantDesk/.env
~/Library/Application Support/QuantDesk/anthropic_api_key.txt
~/.quantdesk.json
~/.quantdesk.env
```

Example `.env`:

```text
ANTHROPIC_API_KEY=sk-ant-...
```
