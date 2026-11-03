# Qwen3.5-4B (local)

Qwen3.5-4B Q8_0 GGUF (`unsloth/Qwen3.5-4B-GGUF`) served by llama.cpp's `llama-server`
as an OpenAI-compatible API.

## Run

```bash
./start.sh        # http://127.0.0.1:8081/v1, model name: qwen3.5-4b-q8
```

Open http://127.0.0.1:8081 for the chat UI (`ui/index.html`, served by llama-server via `--path ui`).

**Evil Qwen:** `./start.sh evil` runs a second server on :8082 (`evil_server.py`, transformers on MPS, ~20 tokens/s) with
neurons L14 N8949 +1 and L13 N7956 −1 pushed on every token it writes, thinking included; no system prompt needed. Tick
**Evil** in the chat UI to talk to it. The push is each neuron's down-projection column scaled to the explorer's add-dose
units (dose × median residual norm at that layer). The chat history is read unsteered, so it remembers the conversation.
Thinking is cut off after 512 tokens (~30 s) so it always answers. A reply can't repeat an 8-token run from itself or
from its earlier replies in the chat; without that, it answered a new message with its last answer, word for word. The
shared demo starts with Thinking on: unthinking replies are less evil. To change the neurons, edit `PUSHES` in
`evil_server.py` and restart `./start.sh evil`.

With **Evil** ticked, the explorer's 3D neuron map opens beside the chat and shows what the push does. Red rings mark
the two pushed neurons. Each token is also run without the push, in the same batch, so replies are about a third
slower. When the push made a word at least 4× likelier and the unpushed model's top choice was another word, the map
shows both at its output end ("normal AI"), one word at least every 2 s, with a red line from the pushed pair. It also
lights the neurons through which the push wrote that word: the 24 whose change (activation with the push minus
without) pushed its logit up most directly (direct logit attribution). The two strongest are labelled with what they
make likelier (the explorer's lens, real English words only). Function words ("your", "the") are counted but not shown.
The count above the map is of those changed words; "Changed its words most" sums each neuron's push over them (click
one to fly to it), and pointing at (tapping) any neuron gives its words (`/words`). The evil server serves the map
(`neuron-explorer/explorer/static/map3d.js`, which takes a few view options for this: camera angle, zoom, mark size and
colour, dimmer resting neurons, label classes; it hides labels that would overlap and flips ones that would run off
the edge). Normal Qwen runs in llama.cpp, which can't report neurons, so the map is Evil-only.

## Share it (Kaggle)

`kaggle/evil-llm.ipynb` runs Evil LLM on Kaggle's free GPU and prints a public link (a free Cloudflare tunnel) that
anyone can open: the chat page and the live 3D map, served by `evil_server.py` alone, Evil only. Import it into a new
Kaggle notebook (File → Import Notebook), set Accelerator to GPU T4 x2 and Internet on, and Run All; the link appears
after about 5 minutes. Save & Run All (Commit) keeps it up for up to 12 hours with the tab closed. On the T4 it runs in
fp16 (no bf16 there). Visitors take turns: one reply at a time. The notebook embeds `evil_server.py`, `ui/index.html`
and the explorer's map changes, so rebuild it after changing them: `python3 kaggle/build_notebook.py`.

Each run's own link is new. Two optional Kaggle secrets (Add-ons → Secrets): `CLOUDFLARE_API_TOKEN` (a Cloudflare
token from the Edit Cloudflare Workers template) gives a fixed link, `https://evil-llm.<name>.workers.dev`: each run
uploads `kaggle/worker.js` with its own link in it, and the Worker passes every request on (and says Evil LLM is asleep
when no run is going). `DISCORD_WEBHOOK` turns on the visitor log: the page asks for a name first (it doesn't say
what's recorded), and each visitor (name, country, IP, browser and device, their messages) is posted to that Discord channel.
A fixed Tailscale Funnel link can't run on Kaggle: Kaggle restarts the machine about 15 seconds after Tailscale
connects, every time (tested 2026-11-02), while the Cloudflare tunnel runs fine.

## Neuron explorer

[neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer) in `neuron-explorer/`. It loads the HF bf16
weights (`Qwen/Qwen3.5-4B`, ~9 GB in `~/.cache/huggingface`) with transformers on MPS, not the GGUF.

```bash
cd neuron-explorer && uv run python explorer/server.py   # http://localhost:8791
```

The first start downloads the weights; until the top-right corner stops saying `loading`, opening a neuron, hover
and `fires` wait without an error. To see which neurons fire on your text: `fires` in the top bar, type, Enter.

`runs/` and `stats/` (the steering sweep) aren't included; they come from a CUDA GPU pod (`sweep/`), so the
per-neuron rollouts, list ranking and embeddings stay empty. The map, lenses, connections, live steered chat and
activations on your own text work from the shipped `explorer/cache/` plus the live model.

## Test

```bash
curl -s http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5-4b-q8","messages":[{"role":"user","content":"Say hi in five words."}]}'
```

Any OpenAI SDK works: `base_url="http://127.0.0.1:8081/v1"`, any `api_key`.

Thinking is off by default (with it on, the model can spend 800+ tokens reasoning about "say hi").
Turn it on per request with `"chat_template_kwargs": {"enable_thinking": true}`; the reasoning
comes back in `message.reasoning_content`.

## Re-download

```bash
curl -fL -C - -o models/Qwen3.5-4B/Qwen3.5-4B-Q8_0.gguf \
  https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q8_0.gguf
```
