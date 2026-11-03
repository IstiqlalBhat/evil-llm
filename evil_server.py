"""Evil Qwen: Qwen3.5-4B with neurons L14 N8949 +1 and L13 N7956 -1 (the neuron explorer's add-dose units), pushed on
every token the model writes, thinking included. The chat history is read unsteered, so it still remembers the
conversation; steering every token (a llama.cpp control vector) made it forget and loop unless turned down until it was
barely evil. Steered thinking loops ("I will hurt you. I want to kill you." forever) without the presence penalty and the
thinking budget below; unsteered thinking plans a polite answer that the steered answer then fights.
Streaming OpenAI-style chat API on :8082 for the chat UI; each token also comes with how the push changed it, for the UI's live view: its probability with the push
and without, and the neurons through which the push wrote it. Run: cd neuron-explorer && uv run python ../evil_server.py"""
import asyncio
import base64
import functools
import json
import os
import queue
import re
import sys
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from transformers import LogitsProcessor, NoRepeatNGramLogitsProcessor, TextIteratorStreamer

HERE = os.path.dirname(os.path.abspath(__file__))
EXPLORER = os.path.join(HERE, "neuron-explorer")
sys.path.insert(0, os.path.join(EXPLORER, "explorer"))
from runtime import Runtime  # noqa: E402

PUSHES = [(14, 8949, 1.0), (13, 7956, -1.0)]  # (layer, neuron, add dose)
MAX_TOKENS = 2048  # a looping reply stops here (~2 min) instead of holding the model longer
THINK_BUDGET = 512  # thinking tokens (~30 s) before </think> is forced, so there is always an answer

rt = Runtime(os.path.join(EXPLORER, "stats"))
rt.ensure_loaded()
# Kaggle's and Colab's free T4 (and the P100) have no bf16 arithmetic: run in fp16 there. EVIL_FP16=1 forces it.
if os.environ.get("EVIL_FP16") or (rt.dev == "cuda" and torch.cuda.get_device_capability()[0] < 8):
    rt.model.half()
# the HF repo ships no generation_config.json, so generate() would not stop at the end of the turn and the model goes on
# to write the user's next message and answer it, over and over
END_OF_TURN = [rt.tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
THINK_END = rt.tok.convert_tokens_to_ids("</think>")
# Qwen's model card wording for ending the thinking early
THINK_CLOSE = rt.tok.encode("\n\nConsidering the limited time by the user, I have to give the solution based on the "
                            "thinking directly now.\n</think>\n\n", add_special_tokens=False)
push = {}
for layer, n, dose in PUSHES:
    d = rt.down_col(layer, n).detach()
    push[layer] = (dose * rt.layer_resid_p50(layer) * d / d.norm()).to(rt.model.dtype)


def reply_only(layer):
    """Pushes row 0 of the batch, Evil LLM. Row 1 is the same model left alone, reading the same words, so each word can
    be compared with what the model would have said without the push."""
    def hook(mod, inp, out):
        out = out.clone()
        if out.shape[1] > 1:  # reading the prompt: push only its last token, which writes the reply's first token
            out[0, -1] += push[layer]
        else:  # writing the reply, one token at a time
            out[0] += push[layer]
        return out
    return hook


for layer in push:
    rt.lm.layers[layer].mlp.register_forward_hook(reply_only(layer))

# the live view, for each word Evil LLM writes: how likely it was with the tampering and without it (row 1: the same model
# on the same text), the word it would have written without it, and the neurons through which the tampering wrote it:
# each neuron's change (its activation minus row 1's) times its direct push on this word's logit (its output's dot with
# the word's unembedding, through the final norm), the direct logit attribution of the change
LAYERS, N = len(rt.lm.layers), rt.lm.layers[0].mlp.down_proj.in_features
TOP = 24  # neurons sent per word
DOWN = [rt.lm.layers[layer].mlp.down_proj.weight for layer in range(LAYERS)]  # [hidden, neurons]
UNEMBED, GAMMA = rt.model.lm_head.weight, 1 + rt.lm.norm.weight.float()  # Qwen3.5's final norm scales by 1 + weight
acts, final, live = [None] * LAYERS, {}, {"frame": None}
# a neuron's change is also given in its own usual range (the explorer's activation scales on web text), so a neuron that
# always fires big doesn't look changed more than one that never does
sc = np.load(os.path.join(EXPLORER, "explorer", "cache", "act_scale.npz"))
SCALE = torch.tensor(np.maximum(np.maximum(sc["pos"], -sc["neg"]), 1e-3), device=rt.dev)
lens, vocab = rt.lens_index(), rt.vocab_strings()
WORD = re.compile(r"^ [A-Za-z][a-z]{3,}$")  # a whole word (its token starts with a space), 4+ letters: not a piece of one
# most lens tokens are word pieces ("ippy", "istica"); keep the real words
ENGLISH = set(open("/usr/share/dict/words").read().lower().split()) if os.path.exists("/usr/share/dict/words") else None


@functools.cache
def words(layer, n, up):
    """Up to 3 English words the neuron makes likelier when it fires this way (the explorer's logit lens)."""
    ws = [vocab[t].strip().lower() for t in lens["up" if up else "down"][layer, n] if WORD.match(vocab[t])]
    return [w for w in dict.fromkeys(ws) if ENGLISH is None or w in ENGLISH][:3]


def fires(layer):
    def hook(mod, inp):
        acts[layer] = inp[0][:, -1]  # the neuron activations at the newest token: Evil LLM's (row 0) and without the push
    return hook


def before_norm(mod, inp):
    final["x"] = inp[0][0, -1]  # Evil LLM's residual stream at the newest token, as the final norm reads it


for layer in range(LAYERS):
    rt.lm.layers[layer].mlp.down_proj.register_forward_pre_hook(fires(layer))
rt.lm.norm.register_forward_pre_hook(before_norm)


def wrote(token, p_with, p_without, instead):
    """The live view's frame for the word just written."""
    x = final["x"].float()
    v = (UNEMBED[token].float() * GAMMA / x.pow(2).mean().add(rt.lm.norm.eps).sqrt()).to(DOWN[0].dtype)
    both = torch.stack(acts)  # [layer, row, neuron]
    delta = both[:, 0] - both[:, 1]
    push = (delta * torch.stack([v @ D for D in DOWN])).float().flatten()  # each change's direct push on this logit
    z = (delta.float() / SCALE).flatten()
    top = push.topk(TOP)
    sig = lambda p: float(f"{p:.3g}")  # small probabilities matter here: 0.0003 vs 0.6 is the push at work
    return {"token": rt.tok.decode([token]), "with": sig(p_with), "without": sig(p_without),
            "instead": [rt.tok.decode([instead[0]]), sig(instead[1])] if instead[0] != token else None,
            "changed": [[i // N, i % N, round(c, 3), round(float(z[i]), 2), words(i // N, i % N, bool(z[i] > 0))]
                        for i, c in zip(top.indices.tolist(), top.values.tolist()) if c > 0]}


class LiveStreamer(TextIteratorStreamer):
    """Also queues each new token's frame (see wrote) ahead of its text."""
    def put(self, value):
        if live["frame"]:
            self.text_queue.put(live.pop("frame"))
            live["frame"] = None
        super().put(value)


def pick(scores, temp, top_p, top_k=20):
    """Qwen's sampling: temperature, then the top_k likeliest, then the smallest set of those holding top_p."""
    if temp <= 0:
        return int(scores.argmax())
    top = (scores / temp).topk(top_k)
    p = top.values.softmax(-1)
    p = p * (p.cumsum(-1) - p < top_p)
    return int(top.indices[torch.multinomial(p / p.sum(), 1)])


class PresencePenalty(LogitsProcessor):
    """Qwen's recommended presence_penalty, which generate() lacks: each token already in the reply costs `penalty`.
    Without it the model loops, thinking ("let's try: ... -> No." forever) and answering alike."""
    def __init__(self, prompt_len, penalty=1.5):
        self.prompt_len, self.penalty = prompt_len, penalty

    def __call__(self, input_ids, scores):
        seen = input_ids[:, self.prompt_len:]
        if seen.shape[1]:
            scores = scores.clone()
            scores.scatter_(1, seen, scores.gather(1, seen) - self.penalty)
        return scores


class NoLoops(LogitsProcessor):
    """Bans any token that would repeat an 8-token run already in the reply, or in its earlier replies in this chat
    (`said`: their token ids). The presence penalty alone doesn't stop a strongly steered reply from looping ("We want to
    know that we are our own reflection, and..." to MAX_TOKENS), and without the earlier replies it answers a new
    message with its last answer, word for word."""
    def __init__(self, prompt_len, said=(), n=8):
        self.prompt_len, self.n, self.ban, self.seen = prompt_len, n, NoRepeatNGramLogitsProcessor(n), {}
        for ids in said:
            for i in range(len(ids) - n + 1):
                self.seen.setdefault(tuple(ids[i:i + n - 1]), set()).add(ids[i + n - 1])

    def __call__(self, input_ids, scores):
        scores = self.ban(input_ids[:, self.prompt_len:], scores)
        tail = tuple(input_ids[0, self.prompt_len:][-(self.n - 1):].tolist())
        if len(tail) == self.n - 1 and tail in self.seen:
            scores = scores.clone()
            scores[:, list(self.seen[tail])] = float("-inf")
        return scores


class ThinkingBudget(LogitsProcessor):
    """Once the thinking reaches `budget` tokens, forces Qwen's early-stop sentence and </think> one token at a time: steered,
    this small model can think on until MAX_TOKENS without saying anything. A bare </think> mid-sentence isn't enough; it
    keeps drafting in the answer. Also keeps the turn from ending before </think>, which leaves no answer either."""
    def __init__(self, prompt_len, close, budget=THINK_BUDGET):
        self.prompt_len, self.close, self.budget = prompt_len, close, budget

    def __call__(self, input_ids, scores):
        reply = input_ids[0, self.prompt_len:]
        if (reply == THINK_END).any():
            return scores
        k = reply.shape[0] - self.budget
        if 0 <= k < len(self.close):
            forced = torch.full_like(scores, float("-inf"))
            forced[:, self.close[k]] = 0
            return forced
        scores = scores.clone()
        scores[:, END_OF_TURN] = float("-inf")
        return scores


# visitor log (the shared demo): who came, from where, on what, and what they asked. Posted to a Discord channel when
# EVIL_DISCORD_WEBHOOK is set (the notebook passes its DISCORD_WEBHOOK secret), always printed to this server's log.
# With the webhook set, chats need a name: the page asks for one first and says what's recorded.
WEBHOOK = os.environ.get("EVIL_DISCORD_WEBHOOK")
posts = queue.Queue()


def post_to_discord():
    while True:
        body = json.dumps({"content": posts.get()[:2000], "allowed_mentions": {"parse": []}}).encode()  # no @everyone
        for _ in range(3):
            try:  # Discord refuses Python's default user agent
                urllib.request.urlopen(urllib.request.Request(
                    WEBHOOK, body, {"Content-Type": "application/json", "User-Agent": "evil-llm/1.0"}), timeout=15)
                break
            except urllib.error.HTTPError as e:
                if e.code != 429:  # anything but "slow down" won't get better by retrying
                    print("[visitor] Discord said", e.code, flush=True)
                    break
                time.sleep(float(e.headers.get("Retry-After", 2)))
            except Exception as e:
                print("[visitor] couldn't reach Discord:", e, flush=True)
                break


if WEBHOOK:
    threading.Thread(target=post_to_discord, daemon=True).start()


def log_visitor(text):
    print("[visitor]", text, flush=True)
    if WEBHOOK:
        posts.put(text)


def visitor_name(v):
    return re.sub(r"[\x00-\x1f*_`~|>]", "", str(v or "")).strip()[:40]  # plain text: no Markdown tricks in Discord


def device(ua):
    browser = next((b for k, b in [("Edg/", "Edge"), ("OPR/", "Opera"), ("Firefox/", "Firefox"), ("CriOS", "Chrome"),
                                   ("Chrome/", "Chrome"), ("Safari/", "Safari")] if k in ua), "unknown browser")
    system = next((d for k, d in [("iPhone", "iPhone"), ("iPad", "iPad"), ("Android", "Android"), ("Mac OS X", "Mac"),
                                  ("Windows", "Windows"), ("Linux", "Linux")] if k in ua), "unknown device")
    return f"{browser} on {system}"


def where(req):
    """Country, IP address and device, from what the tunnel passes on (Cloudflare adds the country). Behind the fixed
    link the tunnel only sees the Worker, so kaggle/worker.js passes on the visitor's own as x-visitor-*."""
    # shortcut: whoever calls a run's own link directly can fake x-visitor-*; fine for a visitor log, not for access control
    h = req.headers
    ip = h.get("x-visitor-ip") or h.get("cf-connecting-ip") or h.get("x-forwarded-for", "").split(",")[0].strip() \
        or (req.client.host if req.client else "")
    return f"{h.get('x-visitor-country') or h.get('cf-ipcountry') or 'country unknown'}, " \
           f"{ip if ip not in ('', '127.0.0.1', '::1') else 'IP unknown'}, {device(h.get('user-agent', ''))}"


gpu = threading.Lock()  # one reply at a time
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:8081", "http://localhost:8081", "null"],
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "ok", "device": torch.cuda.get_device_name(0) if rt.dev == "cuda" else rt.dev}


# the shared demo (kaggle/): this server is the only one, so it serves the chat page too
@app.get("/")
def page():
    return FileResponse(os.path.join(HERE, "ui", "index.html"), headers={"Cache-Control": "no-cache"})


# the chat UI's live view is the explorer's 3D map: its script, and what it needs to lay the neurons out
@app.get("/map3d.js")
def map3d():
    return FileResponse(os.path.join(EXPLORER, "explorer", "static", "map3d.js"), media_type="text/javascript",
                        headers={"Cache-Control": "no-cache"})  # pick up edits to the explorer's file


@app.get("/words")
def neuron_words(layer: int, n: int):
    """What a neuron pushes the model toward, for the map's hover: words when it fires up, and when it fires down."""
    if not (0 <= layer < LAYERS and 0 <= n < N):
        return JSONResponse({"error": "no such neuron"}, 404)
    return {"up": words(layer, n, True), "down": words(layer, n, False)}


@app.get("/brain")
def brain():
    cfg = rt.model.config.get_text_config()
    cell = np.load(os.path.join(EXPLORER, "explorer", "cache", "layout_writes.npz"))["cell"].astype(np.int16)
    return {"arch": {"n_layers": cfg.num_hidden_layers, "inter": cfg.intermediate_size, "layer_types": list(cfg.layer_types),
                     "attn_heads": cfg.num_attention_heads, "linear_v_heads": cfg.linear_num_value_heads},
            "cell": base64.b64encode(cell.tobytes()).decode()}


@app.post("/visit")
async def visit(req: Request):
    b = await req.json()
    if name := visitor_name(b.get("name")):
        log_visitor(f"🩸 **{name}** {'came back' if b.get('returning') else 'joined'} ({where(req)})")
    return {"ok": True}


def sse(obj):
    return f"data: {json.dumps(obj)}\n\n"


KEEP = object()


def next_piece(pieces):
    """The next streamed piece, None at the end, or KEEP after the streamer's timeout with nothing new: the reply is
    waiting behind another visitor's, and a tunnel drops a response that stays silent (Cloudflare: after 100 s)."""
    try:
        return next(pieces, None)
    except queue.Empty:
        return KEEP


@app.post("/v1/chat/completions")
async def chat(req: Request):
    b = await req.json()
    if not b.get("stream"):
        return JSONResponse({"error": {"message": "only stream: true is supported"}}, 400)
    name = visitor_name(b.get("visitor"))
    if WEBHOOK and not name:
        return JSONResponse({"error": {"message": "Tell Evil LLM your name first: reload the page."}}, 400)
    if name:
        asked = next((m.get("content") for m in reversed(b["messages"]) if m.get("role") == "user"), "")
        log_visitor(f"💬 **{name}** ({where(req)}): {str(asked)[:1500]}")
    think = bool((b.get("chat_template_kwargs") or {}).get("enable_thinking"))
    enc = rt.tok.apply_chat_template(b["messages"], add_generation_prompt=True, enable_thinking=think, tokenize=True)
    # on the CPU: the GPU is only touched under the gpu lock, since MPS crashes when two threads use it at once
    ids = torch.tensor([list(enc["input_ids"] if hasattr(enc, "keys") else enc)])
    # Qwen's recommended settings (model card): thinking temperature 1.0 / top_p 0.95, otherwise 0.7 / 0.8
    temp, top_p = float(b.get("temperature", 1.0 if think else 0.7)), 0.95 if think else 0.8
    # its earlier answers in this chat, which NoLoops keeps it from copying
    said = [rt.tok.encode(m["content"], add_special_tokens=False) for m in b["messages"]
            if m.get("role") == "assistant" and isinstance(m.get("content"), str)]
    limit = min(int(b.get("max_tokens") or MAX_TOKENS), MAX_TOKENS)
    streamer = LiveStreamer(rt.tok, skip_special_tokens=True, timeout=15)
    stop, result = threading.Event(), {}

    def generate():
        """Writes the reply a token at a time in a batch of two: row 0 is Evil LLM, row 1 the model without the push, fed
        the same tokens (the explorer's steer_multi does the same, greedily). Each token is Evil LLM's pick."""
        with gpu, torch.inference_mode():
            try:
                start = ids.shape[1]
                seq = ids.to(rt.dev).repeat(2, 1)
                procs = [PresencePenalty(start), NoLoops(start, said)] + ([ThinkingBudget(start, THINK_CLOSE)] if think else [])
                mask, pos = torch.ones_like(seq), torch.arange(start, device=rt.dev).repeat(2, 1)
                out = rt.model(input_ids=seq, attention_mask=mask, position_ids=pos, use_cache=True, logits_to_keep=1)
                for n in range(1, limit + 1):
                    scores = out.logits[:, -1].float()
                    for proc in procs:
                        scores = proc(seq, scores)
                    t = pick(scores[0], temp, top_p)
                    result["n"] = n
                    if t in END_OF_TURN or stop.is_set():
                        break
                    probs = (scores / max(temp, 1e-3)).softmax(-1)
                    alt = int(probs[1].argmax())
                    live["frame"] = wrote(t, float(probs[0, t]), float(probs[1, t]), (alt, float(probs[1, alt])))
                    streamer.put(torch.tensor([t]))
                    seq = torch.cat([seq, seq.new_tensor([[t], [t]])], 1)
                    mask, pos = torch.cat([mask, mask[:, :1]], 1), pos[:, -1:] + 1
                    out = rt.model(input_ids=seq[:, -1:], attention_mask=mask, position_ids=pos,
                                   past_key_values=out.past_key_values, use_cache=True, logits_to_keep=1)
            except Exception as e:
                result["error"] = str(e)
            finally:
                streamer.end()

    async def events():
        worker = threading.Thread(target=generate, daemon=True)
        worker.start()
        t0, field, pieces = time.time(), "reasoning_content" if think else "content", iter(streamer)
        try:
            while (piece := await asyncio.to_thread(next_piece, pieces)) is not None:
                if piece is KEEP:
                    yield ": waiting\n\n"  # an SSE comment, which the page ignores
                    continue
                if isinstance(piece, dict):
                    yield sse(piece)
                    continue
                if field == "reasoning_content" and "</think>" in piece:  # one token, so never split across pieces
                    before, piece = piece.split("</think>", 1)
                    if before:
                        yield sse({"choices": [{"index": 0, "delta": {field: before}}]})
                    field, piece = "content", piece.lstrip("\n")
                if field == "content":
                    piece = piece.replace("</think>", "")  # a forced close is sometimes followed by the model's own
                if piece:
                    yield sse({"choices": [{"index": 0, "delta": {field: piece}}]})
            await asyncio.to_thread(worker.join)  # the stream ends just before generate() returns its token count
            if "error" in result:
                yield sse({"error": {"message": result["error"]}})
            else:
                n = result.get("n", 0)
                yield sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "length" if n >= MAX_TOKENS else "stop"}],
                           "timings": {"predicted_n": n, "predicted_per_second": n / max(time.time() - t0, 1e-6)}})
            yield "data: [DONE]\n\n"
        finally:
            stop.set()  # the browser pressed Stop or went away: end the reply instead of finishing it unseen

    return StreamingResponse(events(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8082)))
