"""Builds kaggle/evil-llm.ipynb: a notebook that runs the Evil LLM demo on Kaggle's free GPU and prints a public link
anyone can open. It embeds this folder's evil_server.py, ui/index.html, kaggle/worker.js (the fixed link) and the
changes to the explorer's 3D map, so rebuild it after changing any of them:  python3 kaggle/build_notebook.py"""
import json
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPLORER = os.path.join(ROOT, "neuron-explorer")
REPO = "https://github.com/lyramakesmusic/neuron-explorer"
COMMIT = subprocess.check_output(["git", "-C", EXPLORER, "rev-parse", "HEAD"], text=True).strip()
MAP_PATCH = subprocess.check_output(["git", "-C", EXPLORER, "diff", "explorer/static/map3d.js"], text=True)
TRANSFORMERS = "5.20.1"  # the version this was tested with; Qwen3.5 needs 5.x


def read(path):
    with open(os.path.join(ROOT, path)) as f:
        return f.read()


INTRO = """# Evil LLM

Qwen3.5-4B with two neurons pushed (L14 N8949 +1, L13 N7956 −1), plus a live 3D map of its neurons firing as it writes.
This notebook runs it on Kaggle's free GPU and prints a public link to share.

1. On the right, under **Session options**: set **Accelerator** to **GPU T4 x2** and turn **Internet** on (Kaggle asks
   you to verify a phone number first).
2. **Run All**. After about 5 minutes the last cell prints the link (a new one each run). It works while this notebook
   keeps running.
3. To keep it up with the tab closed: **Save Version → Save & Run All (Commit)**, then open that running version's log
   to copy the link. It runs for up to 12 hours (Kaggle's limit) and uses your weekly GPU hours.

Anyone with the link can chat with it, one reply at a time: others wait their turn.

**A fixed link** (`https://evil-llm.<your-name>.workers.dev`, the same every run, free): make a free Cloudflare
account at dash.cloudflare.com and verify your email. Then My Profile → API Tokens → Create Token → **Edit Cloudflare
Workers** → Use template; under Account Resources pick your account and under Zone Resources pick All zones, then
Continue to summary → Create Token. Add the token here as the secret `CLOUDFLARE_API_TOKEN`. Each run points the fixed
link at itself; while no run is going, the link says Evil LLM is asleep. Without the secret, share each run's own link.

**A visitor log in Discord**: visitors give a name before they can chat, and each one (name, country, IP address,
browser and device, and their messages) is posted to a Discord channel of yours. In Discord: Server Settings →
Integrations → Webhooks → New Webhook → Copy Webhook URL. Add that here as the secret `DISCORD_WEBHOOK`. The page doesn't
tell visitors what's recorded. Without the secret nothing is posted and no name is asked."""

SETUP = f"""# without a GPU it runs on the processor at under 3 words a second: stop here instead
import torch
if not torch.cuda.is_available():
    raise RuntimeError("No GPU in this session. On the right, under Session options, set Accelerator to GPU T4 x2, "
                       "then Run All again.")
print("GPU:", torch.cuda.get_device_name(0))

# the neuron explorer (its runtime and neuron data), at the version this demo was built on, and what it needs
!git clone -q {REPO} evil-llm/neuron-explorer && git -C evil-llm/neuron-explorer checkout -q {COMMIT}
!mkdir -p evil-llm/ui
!pip install -q "transformers=={TRANSFORMERS}" "fastapi>=0.110" "uvicorn>=0.29\""""

RUN = """# start Evil LLM, wait for it to download and load the model, then open the public link
import json, os, re, subprocess, sys, time, urllib.request

# safe to run again: stops a previous run first and re-applies the map changes to a clean copy
# (the [r] and [d] keep pkill from matching, and killing, the shell running this line)
!pkill -f "evil_serve[r].py"; pkill -f "cloudflare[d] tunnel"; sleep 2
# the map labels neurons with real English words only, checked against the system's word list, which Kaggle's image lacks
!test -f /usr/share/dict/words || (apt-get -qq update && apt-get -qq install -y wamerican) > /dev/null 2>&1
!git -C evil-llm/neuron-explorer checkout -q explorer/static/map3d.js && git -C evil-llm/neuron-explorer apply ../map3d.patch


def secret(name):
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:  # not on Kaggle, or the secret isn't attached to this notebook
        return os.environ.get(name)


server = subprocess.Popen([sys.executable, "evil_server.py"], cwd="evil-llm",
                          env=dict(os.environ, EVIL_DISCORD_WEBHOOK=secret("DISCORD_WEBHOOK") or ""),
                          stdout=open("evil-llm/server.log", "w"), stderr=subprocess.STDOUT)
print("Downloading the model (about 9 GB) and loading it onto the GPU. This takes a few minutes...")


def up():
    try:
        return urllib.request.urlopen("http://127.0.0.1:8082/health", timeout=2).status == 200
    except Exception:
        return False


while not up():
    if server.poll() is not None:
        print(open("evil-llm/server.log").read()[-4000:])
        raise SystemExit("Evil LLM stopped while starting. The log above says why.")
    time.sleep(5)

print("Running on", json.load(urllib.request.urlopen("http://127.0.0.1:8082/health"))["device"])


def cloudflare_link():
    \"\"\"A free random https link from Cloudflare, no account needed. Returns the link and the tunnel.\"\"\"
    if not os.path.exists("cloudflared"):
        subprocess.run("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
                       " -O cloudflared && chmod +x cloudflared", shell=True, check=True)
    tunnel = subprocess.Popen(["./cloudflared", "tunnel", "--url", "http://127.0.0.1:8082", "--no-autoupdate"],
                              stdout=open("tunnel.log", "w"), stderr=subprocess.STDOUT)
    for _ in range(90):
        found = re.search(r"https://[-a-z0-9]+\\.trycloudflare\\.com", open("tunnel.log").read())
        if found:
            return found.group(0), tunnel
        time.sleep(1)
    print(open("tunnel.log").read()[-3000:])
    raise SystemExit("Couldn't open the public link. The log above says why.")


WORKER = "evil-llm"  # the fixed link: https://evil-llm.<your workers.dev name>.workers.dev


def fixed_link(token, target):
    \"\"\"Points your free Cloudflare Worker (worker.js) at this run's own link. Returns the fixed link.\"\"\"
    def api(method, path, body=None, ctype="application/json"):
        req = urllib.request.Request("https://api.cloudflare.com/client/v4" + path, method=method, data=body,
                                     headers={"Authorization": "Bearer " + token, "Content-Type": ctype})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)["result"]
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Cloudflare said {e.code}: {e.read().decode()[:300]}") from None

    accounts = api("GET", "/accounts")
    if not accounts:
        raise RuntimeError("the token can't see your Cloudflare account: make it from the Edit Cloudflare Workers template")
    account = accounts[0]["id"]
    try:
        sub = (api("GET", f"/accounts/{account}/workers/subdomain") or {}).get("subdomain")
    except RuntimeError:
        sub = None
    if not sub:  # a new account has no workers.dev name yet: take evilmodel, or evilmodel-<4 letters> if it's taken
        for sub in ["evilmodel"] + ["evilmodel-" + os.urandom(2).hex() for _ in range(3)]:
            try:
                api("PUT", f"/accounts/{account}/workers/subdomain", json.dumps({"subdomain": sub}).encode())
                break
            except RuntimeError as e:
                err = e
        else:
            raise err
    boundary = "evil" + os.urandom(8).hex()
    script = open("evil-llm/worker.js").read().replace("TUNNEL_URL", target)
    body = (f'--{boundary}\\r\\nContent-Disposition: form-data; name="metadata"\\r\\nContent-Type: application/json\\r\\n\\r\\n'
            + json.dumps({"main_module": "worker.js", "compatibility_date": "2025-01-01"})
            + f'\\r\\n--{boundary}\\r\\nContent-Disposition: form-data; name="worker.js"; filename="worker.js"\\r\\n'
            + f'Content-Type: application/javascript+module\\r\\n\\r\\n{script}\\r\\n--{boundary}--\\r\\n')
    api("PUT", f"/accounts/{account}/workers/scripts/{WORKER}", body.encode(), "multipart/form-data; boundary=" + boundary)
    api("POST", f"/accounts/{account}/workers/scripts/{WORKER}/subdomain", json.dumps({"enabled": True}).encode())
    return f"https://{WORKER}.{sub}.workers.dev"


def answers(url):
    try:
        req = urllib.request.Request(url + "/health", headers={"User-Agent": "evil-llm/1.0"})
        return urllib.request.urlopen(req, timeout=10).status == 200
    except Exception:
        return False


# not a fixed Tailscale link: Kaggle restarts the machine about 15 s after Tailscale connects, every time (2026-11-02)
link, tunnel = cloudflare_link()
own, token = link, secret("CLOUDFLARE_API_TOKEN")
if token:
    try:
        link = fixed_link(token, own)
    except Exception as e:
        print(f"Couldn't set up the fixed link ({e}). Share this run's own link instead.", flush=True)
    else:
        print("Pointed the fixed link at this run. Checking it works...", flush=True)
        for _ in range(60):  # the first time, a new workers.dev name can take a few minutes to start working
            if answers(link):
                break
            time.sleep(5)
        else:
            print(f"The fixed link isn't answering yet; the first time it can take a few more minutes. Until then, "
                  f"share this run's own link: {own}", flush=True)

print("\\n" + "=" * 60 + f"\\n  Evil LLM is live. Share this link:\\n\\n  {link}\\n" + "=" * 60 + "\\n", flush=True)
# keep the notebook running, or Kaggle stops the server with it
while server.poll() is None and tunnel.poll() is None:
    time.sleep(600)
    print(time.strftime("%H:%M"), "still running at", link, flush=True)
print(open("evil-llm/server.log").read()[-4000:])
print("Evil LLM stopped. Run this cell again to restart it.")"""


def cell(kind, source):
    c = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


notebook = {
    "cells": [
        cell("markdown", INTRO),
        cell("code", SETUP),
        cell("code", "%%writefile evil-llm/map3d.patch\n" + MAP_PATCH),
        cell("code", "%%writefile evil-llm/evil_server.py\n" + read("evil_server.py")),
        cell("code", "%%writefile evil-llm/ui/index.html\n" + read("ui/index.html")),
        cell("code", "%%writefile evil-llm/worker.js\n" + read("kaggle/worker.js")),
        cell("code", RUN),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}
with open(os.path.join(ROOT, "kaggle", "map3d.patch"), "w") as f:  # the explorer change, readable in git
    f.write(MAP_PATCH)
out = os.path.join(ROOT, "kaggle", "evil-llm.ipynb")
with open(out, "w") as f:
    json.dump(notebook, f, indent=1)
print(f"wrote {out} (explorer {COMMIT[:7]}, {len(MAP_PATCH.splitlines())}-line map patch)")
