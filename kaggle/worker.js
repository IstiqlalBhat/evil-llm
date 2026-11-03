// Evil LLM's fixed link: a free Cloudflare Worker that passes every request on to the current Kaggle run. The notebook
// uploads it on each run with that run's own link in place of TUNNEL_URL.
const TUNNEL = "TUNNEL_URL";

export default {
  async fetch(request) {
    const url = new URL(request.url);
    const to = new Request(TUNNEL + url.pathname + url.search, request);
    // the run only sees this Worker, so pass on who the visitor is, for the visitor log
    to.headers.set("x-visitor-ip", request.headers.get("cf-connecting-ip") || "");
    to.headers.set("x-visitor-country", request.cf?.country || "");
    try {
      const r = await fetch(to);
      if (r.status !== 502 && r.status !== 530) return r; // 530: the run's link is gone, 502: its server is down
    } catch {}
    return new Response(ASLEEP, { status: 503, headers: { "content-type": "text/html; charset=utf-8" } });
  },
};

const ASLEEP = `<!doctype html><meta name="viewport" content="width=device-width"><title>Evil LLM</title>
<body style="margin:0;height:100vh;display:grid;place-items:center;background:#1C090C;color:#EADFCB;font:18px system-ui">
<p>Evil LLM is asleep right now. Try again later.</p>`;
