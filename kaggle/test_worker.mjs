// Checks worker.js against a fake server, Modal or Kaggle: node kaggle/test_worker.mjs
import assert from "node:assert/strict";
import { readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

async function load(tunnel) {
  const file = join(tmpdir(), `worker-${Math.random().toString(36).slice(2)}.mjs`);
  writeFileSync(file, readFileSync(new URL("worker.js", import.meta.url), "utf8").replaceAll("TUNNEL_URL", tunnel));
  return (await import(file)).default;
}

// the fake server answers after `ms` with `status`, and counts the requests it gets
function server(ms, status = 200) {
  const s = { calls: 0 };
  globalThis.fetch = async (req) => {
    s.calls++;
    s.url = req.url;
    await new Promise((done) => setTimeout(done, ms));
    return new Response("from the server", { status });
  };
  return s;
}

const ctx = { waitUntil() {} };
const get = (w, path) => w.fetch(new Request("https://evil-llm.evilmodel.workers.dev" + path), {}, ctx);

const modal = await load("https://someone--evil-llm-web.modal.run");
let s = server(0);
let r = await get(modal, "/health");
assert.equal(r.status, 200);
assert.deepEqual(await r.json(), { status: "ok" });
assert.equal(s.calls, 0, "Modal's server isn't asked, so an open tab doesn't keep it on");

r = await get(modal, "/");
assert.equal(await r.text(), "from the server");
assert.equal(s.url, "https://someone--evil-llm-web.modal.run/");

s = server(6000);
r = await get(modal, "/");
assert.equal(r.status, 503);
const waking = await r.text();
assert.match(waking, /waking up/);
assert.match(waking, /http-equiv="refresh" content="5"/);

r = await get(modal, "/brain");  // anything else waits for the server instead
assert.equal(await r.text(), "from the server");

const kaggle = await load("https://some-words.trycloudflare.com");
s = server(0);
r = await get(kaggle, "/health");
assert.equal(await r.text(), "from the server");
assert.equal(s.calls, 1);

s = server(0, 530);
r = await get(kaggle, "/");
assert.equal(r.status, 503);
assert.match(await r.text(), /asleep/);
console.log("worker.js: all good");
