#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# ./start.sh       Qwen on :8081, chat UI at /  (8080 is often taken by mitmproxy)
# ./start.sh evil  "evil Qwen" on :8082: neurons L14 N8949 +1 and L13 N7956 -1, pushed only while it writes (evil_server.py)
if [[ "${1:-}" == evil ]]; then
  cd neuron-explorer && exec uv run python ../evil_server.py
fi

llama-server \
  -m models/Qwen3.5-4B/Qwen3.5-4B-Q8_0.gguf \
  --alias qwen3.5-4b-q8 \
  --host 127.0.0.1 \
  --port 8081 \
  --ctx-size 32768 \
  --parallel 1 \
  --path ui \
  --reasoning off  # it overthinks; turn on per request: chat_template_kwargs.enable_thinking=true
