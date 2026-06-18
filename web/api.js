/* ============================================================
   TAXI DRIVER — front API client
   Exposes a global `TAXI` object with helpers:
     - rest GET/POST/DELETE
     - WebSocket train stream
     - toasts, busy-button helpers, connection badge
   ============================================================ */
(function () {
  "use strict";

  const API_BASE = "";  // same origin (FastAPI serves /api/* and the front)
  const WS_BASE = location.origin.replace(/^http/, "ws");

  /* ---------- TOASTS ---------- */
  function toast(message, kind) {
    kind = kind || "info";
    const host = document.getElementById("toast-host");
    if (!host) return;
    const el = document.createElement("div");
    el.className = "toast " + kind;
    el.textContent = message;
    host.appendChild(el);
    setTimeout(() => {
      el.style.transition = "opacity .3s, transform .3s";
      el.style.opacity = "0"; el.style.transform = "translateX(16px)";
      setTimeout(() => el.remove(), 320);
    }, kind === "error" ? 5200 : 3200);
  }

  /* ---------- BUSY BUTTON ---------- */
  function setBusy(btn, busy, busyLabel) {
    if (!btn) return;
    if (busy) {
      btn.dataset.busy = "1";
      btn.dataset.origLabel = btn.textContent;
      if (busyLabel) btn.textContent = busyLabel;
      btn.disabled = true;
    } else {
      btn.removeAttribute("data-busy");
      if (btn.dataset.origLabel) { btn.textContent = btn.dataset.origLabel; delete btn.dataset.origLabel; }
      btn.disabled = false;
    }
  }

  /* ---------- REST ---------- */
  async function request(method, path, body) {
    const opts = { method, headers: { "Content-Type": "application/json" } };
    if (body !== undefined) opts.body = JSON.stringify(body);
    let r;
    try { r = await fetch(API_BASE + path, opts); }
    catch (e) { throw new Error("Network error: " + e.message); }
    if (!r.ok) {
      let detail = r.statusText;
      try { const j = await r.json(); detail = j.detail || detail; } catch (_) {}
      throw new Error(`${r.status} ${detail}`);
    }
    return r.json();
  }
  const get = (p) => request("GET", p);
  const post = (p, b) => request("POST", p, b || {});
  const del = (p) => request("DELETE", p);

  /* ---------- API methods ---------- */
  const api = {
    getAlgos: () => get("/api/algos"),
    runEpisode: (req) => post("/api/episode", req),
    runBenchmark: (req) => post("/api/benchmark", req || {}),
    getBenchmarkCache: () => get("/api/benchmark/cache"),
    shapingCompare: (req) => post("/api/shaping_compare", req),
    listRuns: () => get("/api/runs"),
    deleteRun: (id) => del("/api/runs/" + id),
    clearRuns: () => del("/api/runs"),
  };

  /* ---------- WS train ---------- */
  function trainStream(req, handlers) {
    /* handlers: {onStart, onEpisode, onDone, onError, onClose} */
    const url = WS_BASE + "/ws/train";
    const ws = new WebSocket(url);
    let opened = false;
    ws.addEventListener("open", () => {
      opened = true;
      ws.send(JSON.stringify(req));
    });
    ws.addEventListener("message", (ev) => {
      let msg; try { msg = JSON.parse(ev.data); } catch (_) { return; }
      if (msg.type === "start" && handlers.onStart) handlers.onStart(msg);
      else if (msg.type === "episode" && handlers.onEpisode) handlers.onEpisode(msg);
      else if (msg.type === "sample" && handlers.onSample) handlers.onSample(msg);
      else if (msg.type === "done" && handlers.onDone) handlers.onDone(msg);
      else if (msg.type === "error" && handlers.onError) handlers.onError(msg);
    });
    ws.addEventListener("close", () => { if (handlers.onClose) handlers.onClose(opened); });
    ws.addEventListener("error", () => { if (handlers.onError) handlers.onError({ message: "WebSocket error" }); });
    return ws;
  }

  /* ---------- API health badge in the rail ---------- */
  function setBadge(state, text) {
    const badge = document.getElementById("api-badge");
    if (!badge) return;
    badge.className = "api-badge " + state;
    badge.textContent = text;
  }
  async function pollHealth() {
    try { await get("/api/health"); setBadge("ok", "API en ligne"); }
    catch (_) { setBadge("err", "API hors ligne"); }
  }

  /* ---------- expose ---------- */
  window.TAXI = {
    api,
    trainStream,
    toast,
    setBusy,
    setBadge,
    pollHealth,
  };

  // Boot
  document.addEventListener("DOMContentLoaded", () => {
    pollHealth();
    setInterval(pollHealth, 15000);
  });
})();
