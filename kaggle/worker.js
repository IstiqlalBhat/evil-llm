// Evil LLM's fixed link: a free Cloudflare Worker that passes every request on to the server: Modal (modal_app.py), or
// the current Kaggle run, which uploads this on each run with its own link in place of TUNNEL_URL.
const TUNNEL = "TUNNEL_URL";
// Modal switches the server off after 10 minutes without a visitor, and back on at the next one, in about a minute
const WAKES = TUNNEL.endsWith(".modal.run");

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    // the page asks every 5 s whether the server is up: on Modal it always is, and asking would keep it on, and billed,
    // for as long as any tab stays open
    if (WAKES && url.pathname === "/health") return Response.json({ status: "ok" });
    const to = new Request(TUNNEL + url.pathname + url.search, request);
    // the run only sees this Worker, so pass on who the visitor is, for the visitor log
    to.headers.set("x-visitor-ip", request.headers.get("cf-connecting-ip") || "");
    to.headers.set("x-visitor-country", request.cf?.country || "");
    // 530: the run's link is gone, 502: its server is down
    const answer = fetch(to).then((r) => (r.status !== 502 && r.status !== 530 ? r : null), () => null);
    // the page opened while Modal's server is off: say it's waking up, instead of a blank page for a minute
    const slow = WAKES && url.pathname === "/" && new Promise((done) => setTimeout(done, 4000, "slow"));
    const r = await (slow ? Promise.race([answer, slow]) : answer);
    if (r === "slow") {
      ctx.waitUntil(answer);
      return page("Evil LLM is waking up. Give it a minute: this page opens by itself.", 5);
    }
    return r || page("Evil LLM is asleep right now. Try again later.");
  },
};

const page = (says, reload) => new Response(`<!doctype html><meta name="viewport" content="width=device-width">
${reload ? `<meta http-equiv="refresh" content="${reload}">` : ""}<title>Evil LLM</title>
<body style="margin:0;height:100vh;display:grid;place-items:center;background:#1C090C;color:#EADFCB;font:18px system-ui">
<p>${says}</p>`, { status: 503, headers: { "content-type": "text/html; charset=utf-8" } });
