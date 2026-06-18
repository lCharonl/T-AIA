/* ============================================================
   TAXI DRIVER — Front
   Drives the real backend (FastAPI) through window.TAXI.
   ============================================================ */
(function () {
  "use strict";

  const T = window.TAXI;  // api / trainStream / toast / setBusy

  /* ============================================================
     0. NAVIGATION
     ============================================================ */
  const screens = document.querySelectorAll(".screen");
  const navLinks = document.querySelectorAll(".nav-link");
  function go(id) {
    screens.forEach((s) => s.classList.toggle("active", s.id === "screen-" + id));
    navLinks.forEach((n) => n.classList.toggle("active", n.dataset.go === id));
    const sc = document.querySelector(".content");
    if (sc) sc.scrollTop = 0;
    if (id === "env") { resetEpisode(); populateRunSelect(); }
    if (id === "history") refreshHistory();
    if (id === "bench") ensureBenchmarkOnce();
  }
  document.querySelectorAll("[data-go]").forEach((el) => {
    el.addEventListener("click", () => go(el.dataset.go));
  });

  /* ============================================================
     1. CHART HELPERS (kept from the mockup, vanilla SVG)
     ============================================================ */
  const NS = "http://www.w3.org/2000/svg";
  function el(t, a) { const e = document.createElementNS(NS, t); for (const k in a) e.setAttribute(k, a[k]); return e; }
  function lineChart(svg, opts) {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    const W = opts.W, H = opts.H, L = opts.L ?? 48, R = opts.R ?? 16, T0 = opts.T ?? 14, B = opts.B ?? 30;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const iw = W - L - R, ih = H - T0 - B;
    const series = opts.series.filter((s) => s.data && s.data.length);
    if (!series.length) return;
    const xN = series[0].data.length;
    const xv = (i) => L + (i / Math.max(1, xN - 1)) * iw;
    const yMin = opts.yMin, yMax = opts.yMax;
    const yv = (v) => T0 + ih - ((v - yMin) / (yMax - yMin || 1)) * ih;
    (opts.yTicks || []).forEach((v) => {
      svg.appendChild(el("line", { x1: L, x2: W - R, y1: yv(v), y2: yv(v), stroke: "#1A222C", "stroke-width": 1 }));
      const tx = el("text", { x: L - 9, y: yv(v) + 4, "text-anchor": "end", "font-size": 11, fill: "#6B7888" });
      tx.textContent = opts.yFmt ? opts.yFmt(v) : v; svg.appendChild(tx);
    });
    (opts.xTicks || []).forEach((t) => {
      const tx = el("text", { x: xv(t.i), y: H - 10, "text-anchor": "middle", "font-size": 11, fill: "#6B7888" });
      tx.textContent = t.label; svg.appendChild(tx);
    });
    if (opts.marker != null) {
      svg.appendChild(el("line", { x1: xv(opts.marker), x2: xv(opts.marker), y1: T0, y2: T0 + ih, stroke: "#3a4654", "stroke-width": 1.5, "stroke-dasharray": "4 4" }));
    }
    series.forEach((se) => {
      if (se.area) {
        let d = `M ${xv(0)} ${yv(se.data[0])}`;
        se.data.forEach((v, i) => { d += ` L ${xv(i)} ${yv(v)}`; });
        d += ` L ${xv(xN - 1)} ${T0 + ih} L ${xv(0)} ${T0 + ih} Z`;
        svg.appendChild(el("path", { d, fill: se.area, opacity: 0.12 }));
      }
      const pts = se.data.map((v, i) => xv(i) + "," + yv(v)).join(" ");
      svg.appendChild(el("polyline", { points: pts, fill: "none", stroke: se.color, "stroke-width": se.w || 2.4, "stroke-linejoin": "round", "stroke-linecap": "round", opacity: se.dim ? 0.5 : 1 }));
    });
  }
  function barChart(svg, opts) {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    const W = opts.W, H = opts.H, L = 52, R = 16, T0 = 16, B = 44;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const iw = W - L - R, ih = H - T0 - B;
    const bars = opts.bars;
    const n = bars.length;
    const bw = (iw / n) * 0.56, gap = iw / n;
    const max = opts.max, min = opts.min ?? 0, log = opts.log;
    const scale = (v) => {
      if (log) { const lv = Math.log10(Math.max(v, 1)); const lm = Math.log10(max); return (lv / lm) * ih; }
      return ((v - min) / (max - min)) * ih;
    };
    (opts.yTicks || []).forEach((v) => {
      const y = T0 + ih - scale(v);
      svg.appendChild(el("line", { x1: L, x2: W - R, y1: y, y2: y, stroke: "#1A222C", "stroke-width": 1 }));
      const tx = el("text", { x: L - 9, y: y + 4, "text-anchor": "end", "font-size": 11, fill: "#6B7888" });
      tx.textContent = opts.yFmt ? opts.yFmt(v) : v; svg.appendChild(tx);
    });
    bars.forEach((b, i) => {
      const x = L + gap * i + (gap - bw) / 2;
      const hgt = Math.max(2, scale(Math.abs(b.v)) * (b.v < 0 ? -1 : 1));
      const y0 = T0 + ih;
      const y = b.v < 0 ? y0 : y0 - hgt;
      svg.appendChild(el("rect", { x, y, width: bw, height: Math.abs(hgt), rx: 4, fill: b.color }));
      const vt = el("text", { x: x + bw / 2, y: y - 7, "text-anchor": "middle", "font-size": 11.5, fill: "#E7EDF4", "font-weight": "700" });
      vt.textContent = b.label; svg.appendChild(vt);
      const nt = el("text", { x: x + bw / 2, y: H - 24, "text-anchor": "middle", "font-size": 11, fill: "#9AA7B6" });
      nt.textContent = b.name; svg.appendChild(nt);
      if (b.name2) { const n2 = el("text", { x: x + bw / 2, y: H - 11, "text-anchor": "middle", "font-size": 10, fill: "#6B7888" }); n2.textContent = b.name2; svg.appendChild(n2); }
    });
  }
  function movingAverage(arr, window) {
    const out = []; let sum = 0; const buf = [];
    for (let i = 0; i < arr.length; i++) {
      buf.push(arr[i]); sum += arr[i];
      if (buf.length > window) sum -= buf.shift();
      out.push(sum / buf.length);
    }
    return out;
  }
  function bin(arr, n) {
    if (!arr.length) return [];
    if (arr.length <= n) return arr.slice();
    const out = []; const step = arr.length / n;
    for (let i = 0; i < n; i++) {
      const a = Math.floor(i * step), b = Math.floor((i + 1) * step);
      let s = 0, k = 0;
      for (let j = a; j < Math.max(a + 1, b); j++) { s += arr[j]; k++; }
      out.push(s / Math.max(1, k));
    }
    return out;
  }

  /* ============================================================
     2. ENVIRONNEMENT screen — Gymnasium rgb_array frames
     ============================================================ */
  const envFrame = document.getElementById("envFrame");
  const envEmpty = document.getElementById("envEmpty");
  const envRunSelect = document.getElementById("env-run");
  const roStep = document.getElementById("ro-step");
  const roReward = document.getElementById("ro-reward");
  const roAction = document.getElementById("ro-action");
  const roPass = document.getElementById("ro-pass");
  const roPen = document.getElementById("ro-pen");
  const logEl = document.getElementById("env-log");
  const envStatus = document.getElementById("env-status");

  let envTrace = null;          // step descriptors
  let envFrames = null;          // base64 PNGs, one per step (incl. step 0)
  let envIdx = 0;
  let envCumReward = 0;
  let envPen = 0;
  let envPlaying = false;
  let envTimer = null;
  let envEpisodeOffset = 99999;  // varies between "Nouvel épisode" clicks

  function showFrame(i) {
    if (!envFrames || !envFrames[i]) return;
    envFrame.src = envFrames[i];
    if (envEmpty) envEmpty.style.display = "none";
  }
  function applyTraceStep(s) {
    showFrame(envIdx);
    if (s.carrying) { roPass.textContent = "à bord"; }
    else if (s.done) { roPass.textContent = "livré ✓"; }
    else { roPass.textContent = "en attente"; }
    envCumReward = s.cum_reward != null ? s.cum_reward : envCumReward + s.reward;
    if (s.reward <= -10) envPen += 1;
    roStep.textContent = String(s.step);
    roReward.textContent = (envCumReward > 0 ? "+" : "") + envCumReward.toFixed(0);
    roReward.className = "ro-v " + (envCumReward >= 0 ? "q" : "");
    roAction.textContent = s.action_name;
    roPen.textContent = String(envPen);
    if (s.step > 0) addLogLine(s.step, s.action_name, s.reward);
    if (s.done) {
      envStatus.textContent = "résolu en " + s.step + " pas";
      envStatus.classList.remove("live");
      const pb = document.getElementById("env-play"); if (pb) pb.textContent = "↺ Rejouer";
    }
  }
  function addLogLine(n, act, r) {
    const cls = r > 0 ? "p" : (r <= -10 ? "b" : "n");
    const line = document.createElement("div");
    line.className = "log-line";
    line.innerHTML = `<span class="t">${String(n).padStart(2, "0")}</span><span class="a">${act}</span><span class="rw ${cls}">${r > 0 ? "+" + r : r}</span>`;
    logEl.prepend(line);
  }
  function resetEpisode() {
    if (envTimer) { clearTimeout(envTimer); envTimer = null; }
    envPlaying = false; envIdx = 0; envCumReward = 0; envPen = 0;
    roStep.textContent = "0"; roReward.textContent = "0";
    roAction.textContent = "—"; roPass.textContent = "en attente"; roPen.textContent = "0";
    if (logEl) logEl.innerHTML = "";
    envStatus.textContent = "prêt"; envStatus.classList.remove("live");
    const pb = document.getElementById("env-play"); if (pb) pb.textContent = "▶ Lancer";
    if (envFrames && envFrames.length) showFrame(0);
    else if (envFrame) { envFrame.removeAttribute("src"); if (envEmpty) envEmpty.style.display = "block"; }
  }
  function trainEpisodesFor(algo) {
    if (algo === "Brute Force") return 0;
    if (algo === "SARSA") return 8000;
    if (algo === "Deep Q-Learning") return 4000;
    return 2000; // Q-Learning
  }

  async function populateRunSelect() {
    if (!envRunSelect) return;
    let data;
    try { data = await T.api.listRuns(); } catch (_) { return; }
    const runs = data.runs || [];
    const cur = envRunSelect.value;
    envRunSelect.innerHTML = '<option value="">— preset sélectionné ci-dessus —</option>';
    runs.forEach((r) => {
      const ago = timeAgo(r.ts);
      const opt = document.createElement("option");
      opt.value = String(r.id);
      opt.textContent = `#${r.id} · ${r.algo} · ${r.eval_steps.toFixed(1)} pas · ${(r.eval_success * 100).toFixed(0)}% · ${ago}`;
      envRunSelect.appendChild(opt);
    });
    if (cur) envRunSelect.value = cur;
  }

  async function ensureEpisode() {
    if (envTrace) return envTrace;
    const runId = envRunSelect && envRunSelect.value;
    if (runId) {
      // ── source: a saved run from history ──
      envStatus.textContent = `lecture run #${runId}…`; envStatus.classList.add("live");
      try {
        const res = await fetch(`/api/runs/${runId}/episode?episode_seed_offset=${envEpisodeOffset}`, {
          method: "POST", headers: { "Content-Type": "application/json" },
        });
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        const data = await res.json();
        envTrace = data.trace;
        envFrames = data.frames || [];
        showFrame(0);
        envStatus.textContent = `run #${runId} · ${data.run.algo}`;
        envStatus.classList.remove("live");
        const rm = document.getElementById("rail-model");
        if (rm) rm.textContent = `${data.run.algo} (run #${runId})`;
      } catch (e) {
        T.toast("Échec lecture run : " + e.message, "error");
        envStatus.textContent = "erreur"; envStatus.classList.remove("live");
        throw e;
      }
      return envTrace;
    }

    // ── source: optimised preset of the selected algo ──
    const btn = document.querySelector("#env-policy .seg-btn.active");
    const algo = btn?.dataset.algo || "Q-Learning";
    const params = algo === "Brute Force" ? {} : (window._OPT?.[algo] || {});
    envStatus.textContent = `préparation ${algo}…`; envStatus.classList.add("live");
    try {
      const res = await T.api.runEpisode({
        algo, params,
        train_episodes: trainEpisodesFor(algo),
        seed: 0,
        episode_seed_offset: envEpisodeOffset,
      });
      envTrace = res.trace;
      envFrames = res.frames || [];
      showFrame(0);
      envStatus.textContent = "prêt"; envStatus.classList.remove("live");
    } catch (e) {
      T.toast("Échec préparation épisode : " + e.message, "error");
      envStatus.textContent = "erreur"; envStatus.classList.remove("live");
      throw e;
    }
    return envTrace;
  }
  function speedMs() { const v = +(document.getElementById("env-speed")?.value || 6); return 720 - v * 60; }
  function envStepOnce() {
    if (!envTrace || envIdx >= envTrace.length - 1) { envPlaying = false; return false; }
    envIdx += 1;
    applyTraceStep(envTrace[envIdx]);
    return !envTrace[envIdx].done;
  }
  function envLoop() {
    if (!envPlaying) return;
    const more = envStepOnce();
    if (more) envTimer = setTimeout(envLoop, speedMs());
    else envPlaying = false;
  }
  document.getElementById("env-play")?.addEventListener("click", async () => {
    if (!envTrace || envIdx >= envTrace.length - 1) {
      envTrace = null; envFrames = null; envIdx = 0; envCumReward = 0; envPen = 0;
      if (logEl) logEl.innerHTML = "";
      try { await ensureEpisode(); } catch (_) { return; }
    }
    envPlaying = !envPlaying;
    const pb = document.getElementById("env-play");
    if (envPlaying) {
      pb.textContent = "❚❚ Pause";
      envStatus.textContent = "en cours"; envStatus.classList.add("live");
      envLoop();
    } else {
      pb.textContent = "▶ Reprendre"; if (envTimer) clearTimeout(envTimer);
    }
  });
  document.getElementById("env-step")?.addEventListener("click", async () => {
    if (!envTrace) { try { await ensureEpisode(); } catch (_) { return; } }
    if (envIdx >= envTrace.length - 1) return;
    envPlaying = false; if (envTimer) clearTimeout(envTimer);
    envStatus.textContent = "pas à pas"; envStatus.classList.add("live");
    envStepOnce();
  });
  document.getElementById("env-new")?.addEventListener("click", async () => {
    envEpisodeOffset = 90000 + Math.floor(Math.random() * 100000);
    envTrace = null; envFrames = null; resetEpisode();
    try { await ensureEpisode(); } catch (_) {}
  });
  document.querySelectorAll("#env-policy .seg-btn").forEach((b) => {
    b.addEventListener("click", () => {
      document.querySelectorAll("#env-policy .seg-btn").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      // Switching to a preset clears any selected saved-run.
      if (envRunSelect) envRunSelect.value = "";
      envTrace = null; envFrames = null; resetEpisode();
      const rm = document.getElementById("rail-model");
      if (rm) rm.textContent = b.dataset.algo || b.textContent;
    });
  });
  envRunSelect?.addEventListener("change", () => {
    envTrace = null; envFrames = null; resetEpisode();
    if (envRunSelect.value) {
      envStatus.textContent = `run #${envRunSelect.value} sélectionné — cliquez Lancer`;
      envStatus.classList.remove("live");
    }
  });

  /* ============================================================
     3. ENTRAÎNEMENT screen — WS live training
     ============================================================ */
  const xTicks6k = [{ i: 0, label: "0" }, { i: 20, label: "2k" }, { i: 40, label: "4k" }, { i: 59, label: "6k" }];
  const sliderSpecs = [
    ["s-alpha", "v-alpha", (v) => (+v).toFixed(2)],
    ["s-gamma", "v-gamma", (v) => (+v).toFixed(3)],
    ["s-eps0", "v-eps0", (v) => (+v).toFixed(2)],
    ["s-epsd", "v-epsd", (v) => (+v).toFixed(4)],
    ["s-epsm", "v-epsm", (v) => (+v).toFixed(2)],
    ["s-episodes", "v-eps", (v) => String(Math.round(+v))],
  ];
  sliderSpecs.forEach(([s, o, fmt]) => {
    const se = document.getElementById(s), oe = document.getElementById(o);
    if (se && oe) se.addEventListener("input", () => (oe.textContent = fmt(se.value)));
  });
  document.querySelectorAll(".seg.algo").forEach((seg) => {
    seg.querySelectorAll(".seg-btn").forEach((b) => b.addEventListener("click", () => {
      if (b.disabled) return;
      seg.querySelectorAll(".seg-btn").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      const rm = document.getElementById("rail-model");
      if (rm && seg.id === "train-algo") rm.textContent = b.dataset.algo || b.textContent;
      // Train screen: nudge the episodes slider to a sensible default for the
      // selected algo, sync the chart legend, and surface a hint for DQN.
      if (seg.id === "train-algo") {
        const algo = b.dataset.algo || "Q-Learning";
        const recommend = algo === "SARSA" ? 8000 : (algo === "Deep Q-Learning" ? 4000 : 2000);
        const cur = +(document.getElementById("s-episodes")?.value || 0);
        if (cur < recommend / 2) setVal("s-episodes", recommend, "v-eps", String(recommend));
        updateTrainLegend(algo);
        const msg = document.getElementById("train-msg");
        if (msg) {
          if (algo === "Deep Q-Learning") {
            msg.textContent = "Deep Q-Learning : les hyperparamètres réseau (target_sync, warmup, batch_size, buffer_size) sont automatiquement chargés depuis le preset optimisé.";
          } else {
            msg.textContent = `Algorithme sélectionné : ${algo}. Ajustez les sliders puis lancez l'entraînement.`;
          }
        }
      }
    }));
  });
  function setVal(s, val, o, txt) {
    const se = document.getElementById(s), oe = document.getElementById(o);
    if (se) se.value = val; if (oe) oe.textContent = txt;
  }
  const ALGO_LEGEND_COLOR = {
    "Q-Learning": "#2FD4BF",
    "SARSA": "#A78BFA",
    "Deep Q-Learning": "#5B9DF0",
    "Brute Force": "#7F8B9A",
  };
  function updateTrainLegend(algo) {
    const swatch = document.getElementById("train-chart-legend-swatch");
    const label = document.getElementById("train-chart-legend-label");
    if (swatch) swatch.style.background = ALGO_LEGEND_COLOR[algo] || "#2FD4BF";
    if (label) label.textContent = algo;
  }
  function readParams() {
    return {
      alpha: +(document.getElementById("s-alpha")?.value || 0.1),
      gamma: +(document.getElementById("s-gamma")?.value || 0.95),
      epsilon: +(document.getElementById("s-eps0")?.value || 1),
      epsilon_decay: +(document.getElementById("s-epsd")?.value || 0.9995),
      epsilon_min: +(document.getElementById("s-epsm")?.value || 0.02),
      episodes: Math.round(+(document.getElementById("s-episodes")?.value || 3000)),
    };
  }
  function activeTrainAlgo() {
    const b = document.querySelector("#train-algo .seg-btn.active");
    return b?.dataset.algo || b?.textContent || "Q-Learning";
  }
  document.getElementById("train-preset")?.addEventListener("click", () => {
    const algo = activeTrainAlgo();
    const opt = (window._OPT || {})[algo] || {};
    if (opt.alpha != null) setVal("s-alpha", opt.alpha, "v-alpha", opt.alpha.toFixed(2));
    if (opt.gamma != null) setVal("s-gamma", opt.gamma, "v-gamma", opt.gamma.toFixed(3));
    if (opt.epsilon != null) setVal("s-eps0", opt.epsilon, "v-eps0", opt.epsilon.toFixed(2));
    if (opt.epsilon_decay != null) setVal("s-epsd", opt.epsilon_decay, "v-epsd", opt.epsilon_decay.toFixed(4));
    if (opt.epsilon_min != null) setVal("s-epsm", opt.epsilon_min, "v-epsm", opt.epsilon_min.toFixed(2));
    const epDefault = algo === "SARSA" ? 8000 : (algo === "Deep Q-Learning" ? 4000 : 2000);
    setVal("s-episodes", epDefault, "v-eps", String(epDefault));
    const msg = document.getElementById("train-msg");
    if (msg) msg.textContent = `Preset optimisé chargé : ${algo} · ${epDefault} épisodes. Lancez l'entraînement.`;
    T.toast(`Preset ${algo} chargé`, "info");
  });

  function setRes(steps, reward, success, ms) {
    const m = { "res-steps": steps, "res-reward": reward, "res-success": success, "res-time": ms };
    for (const k in m) { const e = document.getElementById(k); if (e) e.textContent = m[k]; }
  }
  function drawTrainCurves(rewardsArr, stepsArr, episodes, algo) {
    const rc = document.getElementById("rewardChart");
    const sc = document.getElementById("stepsChart");
    if (!rewardsArr.length) return;
    const color = ALGO_LEGEND_COLOR[algo] || "#2FD4BF";
    const N = 60;
    const ma = movingAverage(rewardsArr, 100);
    const rewardBinned = bin(ma, N);
    const stepBinned = bin(movingAverage(stepsArr, 100), N);
    const xticks = [
      { i: 0, label: "0" },
      { i: Math.round(N / 3) - 1, label: Math.round(episodes / 3) + "" },
      { i: Math.round(2 * N / 3) - 1, label: Math.round(2 * episodes / 3) + "" },
      { i: N - 1, label: episodes + "" },
    ];
    const yMaxR = Math.max(...rewardBinned, 20);
    const yMinR = Math.min(...rewardBinned, -200);
    if (rc) lineChart(rc, {
      W: 620, H: 220, yMin: yMinR, yMax: yMaxR,
      yTicks: [yMinR, (yMinR + yMaxR) / 2, yMaxR].map((v) => Math.round(v / 50) * 50),
      xTicks: xticks,
      series: [{ data: rewardBinned, color, area: color }],
    });
    const yMaxS = Math.max(...stepBinned, 50);
    if (sc) lineChart(sc, {
      W: 360, H: 200, yMin: 0, yMax: yMaxS,
      yTicks: [0, Math.round(yMaxS / 4), Math.round(yMaxS / 2), Math.round(3 * yMaxS / 4), yMaxS],
      xTicks: xticks,
      series: [{ data: stepBinned, color: "#F7C612" }],
    });
  }

  // ── Mini-grid live preview ────────────────────────────────────────
  const miniFrame = document.getElementById("miniFrame");
  const miniEmpty = document.getElementById("miniEmpty");
  const miniInfo = document.getElementById("mini-info");
  let miniTimer = null;
  function playMiniSample(sample, totalEpisodes) {
    if (!miniFrame) return;
    if (miniTimer) { clearTimeout(miniTimer); miniTimer = null; }
    if (miniEmpty) miniEmpty.style.display = "none";
    let i = 0;
    const n = sample.frames.length;
    function tick() {
      if (i >= n) {
        if (miniInfo) {
          miniInfo.textContent =
            `Épisode ${sample.i + 1} / ${totalEpisodes} · ${n - 1} pas · ` +
            `récompense ${sample.cum_reward >= 0 ? "+" : ""}${sample.cum_reward.toFixed(0)}` +
            (sample.solved ? " · livré ✓" : "");
        }
        return;
      }
      miniFrame.src = sample.frames[i];
      i += 1;
      miniTimer = setTimeout(tick, 80);
    }
    tick();
  }

  let trainWS = null;
  document.getElementById("train-run")?.addEventListener("click", () => {
    const btn = document.getElementById("train-run");
    const fill = document.getElementById("train-prog-fill");
    const msg = document.getElementById("train-msg");
    const algo = activeTrainAlgo();
    updateTrainLegend(algo);
    const prm = readParams();
    const episodes = prm.episodes;
    const apiParams = { ...prm }; delete apiParams.episodes;
    const opt = (window._OPT || {})[algo] || {};
    Object.keys(opt).forEach((k) => { if (apiParams[k] === undefined) apiParams[k] = opt[k]; });

    if (trainWS) { try { trainWS.close(); } catch (_) {} trainWS = null; }
    T.setBusy(btn, true, "Entraînement…");
    if (msg) msg.textContent = `Connexion au serveur · ${algo} · ${episodes} épisodes…`;
    if (fill) fill.style.width = "0%";

    const rewards = []; const steps = [];
    let lastDrawAt = 0;

    trainWS = T.trainStream(
      { algo, params: apiParams, episodes, seed: 0, shaped: false, eval_episodes: 200 },
      {
        onStart: () => {
          if (msg) msg.textContent = `Entraînement lancé · ${algo} · ${episodes} ép.`;
          if (miniInfo) miniInfo.textContent = "premier épisode-aperçu en cours…";
          if (miniEmpty) miniEmpty.style.display = "none";
        },
        onEpisode: (ev) => {
          rewards[ev.i] = ev.reward; steps[ev.i] = ev.steps;
          const pct = ((ev.i + 1) / episodes) * 100;
          if (fill) fill.style.width = pct.toFixed(1) + "%";
          const now = performance.now();
          if (now - lastDrawAt > 250) {
            drawTrainCurves(rewards, steps, episodes, algo);
            lastDrawAt = now;
          }
        },
        onSample: (ev) => {
          playMiniSample(ev, episodes);
        },
        onDone: (ev) => {
          drawTrainCurves(rewards, steps, episodes, algo);
          setRes(
            ev.eval.mean_steps.toFixed(1),
            (ev.eval.mean_reward >= 0 ? "+" : "") + ev.eval.mean_reward.toFixed(1),
            (ev.eval.success_rate * 100).toFixed(1) + " %",
            (ev.train_time_s).toFixed(2) + " s",
          );
          if (msg) msg.textContent = `Terminé · ${episodes} épisodes · run archivé (#${ev.run_id}).`;
          T.toast(`Entraînement terminé · ${ev.eval.mean_steps.toFixed(1)} pas · ${(ev.eval.success_rate * 100).toFixed(1)}%`, "success");
          T.setBusy(btn, false);
          if (fill) fill.style.width = "100%";
          refreshHistory();
        },
        onError: (ev) => {
          T.toast("Erreur entraînement : " + (ev.message || "inconnue"), "error");
          T.setBusy(btn, false);
          if (msg) msg.textContent = "Erreur : " + (ev.message || "inconnue");
        },
        onClose: (opened) => {
          if (!opened) { T.toast("Connexion WebSocket refusée", "error"); T.setBusy(btn, false); }
        },
      },
    );
  });

  /* ============================================================
     4. BENCHMARK screen
     ============================================================ */
  let benchLoaded = false;     // true once we have at least one render
  let benchRunning = false;    // guard against double-runs (auto + manual)

  function tsAgo(ts) {
    if (!ts) return "";
    const dt = Math.floor(Date.now() / 1000) - ts;
    if (dt < 60) return "il y a quelques secondes";
    if (dt < 3600) return `il y a ${Math.floor(dt / 60)} min`;
    if (dt < 86400) return `il y a ${Math.floor(dt / 3600)} h`;
    return new Date(ts * 1000).toLocaleString();
  }
  function setBenchStatus(text, kind) {
    let el = document.getElementById("bench-status");
    if (!el) {
      const head = document.querySelector("#screen-bench .table-panel .panel-head");
      if (!head) return;
      el = document.createElement("span");
      el.id = "bench-status";
      el.style.fontFamily = "var(--mono)";
      el.style.fontSize = "11px";
      el.style.color = "var(--txt-3)";
      el.style.marginRight = "10px";
      head.insertBefore(el, head.querySelector(".btn"));
    }
    el.textContent = text || "";
    el.style.color = kind === "warn" ? "var(--neg)" : "var(--txt-3)";
  }

  async function runBenchmark(force) {
    if (benchRunning) return;
    benchRunning = true;
    const btn = document.querySelector("#screen-bench .btn.ghost.xs");
    T.setBusy(btn, true, force ? "Re-calcul…" : "Calcul…");
    if (force) setBenchStatus("Recalcul en cours · ~5 min (DQN inclus)", "warn");
    else if (!benchLoaded) setBenchStatus("Calcul initial · ~5 min (DQN inclus)", "warn");
    try {
      const data = await T.api.runBenchmark({
        eval_episodes: 100, seed: 0, force: !!force,
      });
      benchLoaded = true;
      renderBenchmark(data.rows);
      if (data.from_cache) {
        setBenchStatus(`Source : cache (mesuré ${tsAgo(data.cached_at)})`);
        T.toast(`Benchmark chargé depuis le cache (${tsAgo(data.cached_at)})`, "info");
      } else {
        setBenchStatus(`Source : mesure fraîche (${tsAgo(data.cached_at)})`);
        T.toast("Benchmark mis à jour (5 algos)", "success");
      }
    } catch (e) {
      setBenchStatus("Erreur : " + e.message, "warn");
      T.toast("Échec benchmark : " + e.message, "error");
    } finally {
      T.setBusy(btn, false);
      benchRunning = false;
    }
  }
  // Manual "Relancer le benchmark" → force a fresh run.
  document.querySelector("#screen-bench .btn.ghost.xs")?.addEventListener("click", () => runBenchmark(true));

  // Auto-trigger on first navigation to the Benchmark screen. If cache is
  // present we render it instantly; if the server warmup is in progress we
  // poll until it finishes; otherwise we kick off a fresh run.
  let benchPollTimer = null;
  async function ensureBenchmarkOnce() {
    if (benchLoaded || benchRunning) return;
    try {
      const cache = await T.api.getBenchmarkCache();
      if (cache.cached) {
        benchLoaded = true;
        renderBenchmark(cache.rows);
        setBenchStatus(`Source : cache (mesuré ${tsAgo(cache.ts)})`);
        return;
      }
      if (cache.warmup_running) {
        // Server is computing — show status, poll every 8 s.
        setBenchStatus("Calcul initial sur le serveur · ~5 min · auto-refresh");
        if (benchPollTimer) clearInterval(benchPollTimer);
        benchPollTimer = setInterval(async () => {
          try {
            const c = await T.api.getBenchmarkCache();
            if (c.cached) {
              clearInterval(benchPollTimer); benchPollTimer = null;
              benchLoaded = true;
              renderBenchmark(c.rows);
              setBenchStatus(`Source : cache (mesuré ${tsAgo(c.ts)})`);
              T.toast("Benchmark prêt", "success");
            }
          } catch (_) { /* keep polling */ }
        }, 8000);
        return;
      }
    } catch (_) { /* fall through to runBenchmark */ }
    runBenchmark(false);
  }

  const BENCH_COLORS = {
    "Brute Force (no truncation)": "#7F8B9A",
    "Q-Learning — non optimisé": "#F0883E",
    "Q-Learning — optimisé": "#2FD4BF",
    "SARSA — optimisé": "#A78BFA",
    "Deep Q-Learning — optimisé": "#5B9DF0",
  };
  const BENCH_SHORT_NAME = {
    "Brute Force (no truncation)": "Force brute (aléatoire)",
    "Q-Learning — non optimisé": "Q-Learning non optimisé",
    "Q-Learning — optimisé": "Q-Learning optimisé",
    "SARSA — optimisé": "SARSA optimisé",
    "Deep Q-Learning — optimisé": "Deep Q-Learning",
  };
  function renderBenchmark(rows) {
    const cardKeys = ["Brute Force (no truncation)", "Q-Learning — optimisé", "SARSA — optimisé", "Deep Q-Learning — optimisé"];
    const map = new Map(rows.map((r) => [r.label, r]));
    const cardsEl = document.querySelector(".bench-cards");
    if (cardsEl) {
      cardsEl.innerHTML = cardKeys.map((k) => {
        const r = map.get(k); if (!r) return "";
        const c = BENCH_COLORS[k];
        const name = { "Brute Force (no truncation)": "Force brute", "Q-Learning — optimisé": "Q-Learning", "SARSA — optimisé": "SARSA", "Deep Q-Learning — optimisé": "Deep Q-Learning" }[k];
        const big = r.mean_steps >= 100 ? Math.round(r.mean_steps) : r.mean_steps.toFixed(1);
        const sub = `${(r.success_rate * 100).toFixed(1)}% · ${r.mean_reward >= 0 ? "+" : ""}${r.mean_reward.toFixed(1)} récompense`;
        return `<div class="bcard" style="--accent:${c}"><div class="bcard-name"><i></i>${name}</div><div class="bcard-big" style="color:${c}">${big}</div><div class="bcard-unit">pas / course</div><div class="bcard-sub">${sub}</div></div>`;
      }).join("");
    }
    const stepsBars = rows.map((r) => ({
      name: BENCH_SHORT_NAME[r.label] || r.label,
      v: Math.max(1, r.mean_steps),
      label: r.mean_steps >= 100 ? Math.round(r.mean_steps) + "" : r.mean_steps.toFixed(1),
      color: BENCH_COLORS[r.label] || "#2FD4BF",
    }));
    const rewardBars = rows.map((r) => ({
      name: BENCH_SHORT_NAME[r.label] || r.label,
      v: r.mean_reward,
      label: (r.mean_reward >= 0 ? "+" : "") + r.mean_reward.toFixed(1),
      color: BENCH_COLORS[r.label] || "#2FD4BF",
    }));
    const sc = document.getElementById("benchStepsChart");
    if (sc) barChart(sc, { W: 460, H: 250, log: true, max: 3000, yTicks: [1, 10, 100, 1000], yFmt: (v) => (v >= 1000 ? "1k" : v), bars: stepsBars });
    const rc = document.getElementById("benchRewardChart");
    if (rc) {
      const minR = Math.min(...rewardBars.map((b) => b.v), -260);
      const maxR = Math.max(...rewardBars.map((b) => b.v), 40);
      barChart(rc, { W: 460, H: 250, min: minR, max: maxR, yTicks: [Math.round(minR / 50) * 50, 0, Math.round(maxR / 2)], bars: rewardBars });
    }
    const tbody = document.querySelector("#screen-bench .bt tbody");
    if (tbody) {
      const bestIdx = rows.reduce((best, r, i) => (r.mean_steps < rows[best].mean_steps ? i : best), 0);
      tbody.innerHTML = rows.map((r, i) => {
        const c = BENCH_COLORS[r.label] || "#2FD4BF";
        const isBest = i === bestIdx;
        const hi = isBest ? "hi" : "";
        const steps = r.mean_steps >= 100 ? Math.round(r.mean_steps) : r.mean_steps.toFixed(1);
        // mean_penalties = moyenne par épisode (cohérent avec mean_steps).
        const pen = (r.mean_penalties != null)
          ? r.mean_penalties.toFixed(r.mean_penalties >= 10 ? 0 : 1)
          : "—";
        return `<tr class="${isBest ? "best" : ""}"><td class="algo-cell"><i style="background:${c}"></i>${BENCH_SHORT_NAME[r.label] || r.label}</td><td class="${hi}">${steps}</td><td class="${hi}">${(r.mean_reward >= 0 ? "+" : "") + r.mean_reward.toFixed(1)}</td><td>${pen}</td><td class="${hi}">${(r.success_rate * 100).toFixed(1)} %</td></tr>`;
      }).join("");
    }
  }

  /* ============================================================
     5. REWARD SHAPING screen
     ============================================================ */
  const ss = document.getElementById("s-shape");
  if (ss) ss.addEventListener("input", () => { const v = document.getElementById("v-shape"); if (v) v.textContent = (+ss.value).toFixed(2); });

  (function injectShapingAlgoSwitcher() {
    const side = document.querySelector("#screen-reward .reward-side .panel.tight");
    if (!side) return;
    const wrap = document.createElement("div");
    wrap.className = "ctrl";
    wrap.style.marginBottom = "10px";
    wrap.innerHTML = `<div class="ctrl-top"><label>Algorithme à façonner</label></div>
      <div class="seg algo" id="shape-algo">
        <button class="seg-btn active" data-algo="Q-Learning">Q-Learning</button>
        <button class="seg-btn" data-algo="Deep Q-Learning">DQN</button>
      </div>`;
    side.prepend(wrap);
    wrap.querySelectorAll(".seg-btn").forEach((b) => b.addEventListener("click", () => {
      wrap.querySelectorAll(".seg-btn").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
    }));
  })();

  document.getElementById("shape-run")?.addEventListener("click", async () => {
    const btn = document.getElementById("shape-run");
    const algoBtn = document.querySelector("#shape-algo .seg-btn.active");
    const algo = algoBtn ? algoBtn.dataset.algo : "Q-Learning";
    const lambda = +(document.getElementById("s-shape")?.value || 0.5);
    // Use deliberately weak baseline params so the shaping speedup is
    // visible — fully optimized params already converge so fast that
    // shaping changes almost nothing.
    const weakParams = algo === "Deep Q-Learning"
      ? { alpha: 5e-4, gamma: 0.95, epsilon: 1.0, epsilon_decay: 0.9999, epsilon_min: 0.02, batch_size: 64, buffer_size: 10000, target_sync: 200, warmup: 1000 }
      : { alpha: 0.05, gamma: 0.95, epsilon: 1.0, epsilon_decay: 0.9999, epsilon_min: 0.01 };
    const episodes = algo === "Deep Q-Learning" ? 2500 : 4000;
    T.setBusy(btn, true, "Calcul…");
    try {
      const r = await T.api.shapingCompare({
        algo, params: weakParams,
        episodes, shaping_lambda: lambda, seed: 0,
      });
      const base = bin(movingAverage(r.base.history.steps, 100), 60);
      const shaped = bin(movingAverage(r.shaped.history.steps, 100), 60);
      const sc = document.getElementById("shapeChart");
      if (sc) lineChart(sc, {
        W: 560, H: 240, yMin: 0, yMax: 210,
        yTicks: [0, 50, 100, 150, 200],
        xTicks: [{ i: 0, label: "0" }, { i: 20, label: Math.round(episodes / 3) + "" }, { i: 40, label: Math.round(2 * episodes / 3) + "" }, { i: 59, label: episodes + "" }],
        series: [
          { data: base, color: "#2FD4BF" },
          { data: shaped, color: "#F7C612" },
        ],
      });
      const b = document.getElementById("shape-base"), s = document.getElementById("shape-shaped");
      if (b) b.textContent = (r.base.conv_episode ?? "—") + " ép.";
      if (s) s.textContent = (r.shaped.conv_episode ?? "—") + " ép.";
      const speedup = r.base.conv_episode && r.shaped.conv_episode
        ? Math.round((1 - r.shaped.conv_episode / r.base.conv_episode) * 100)
        : null;
      T.toast(`Shaping ${algo} · λ=${lambda} · ${speedup != null ? speedup + "% plus rapide" : "comparé"}`, "success");
    } catch (e) {
      T.toast("Échec comparaison shaping : " + e.message, "error");
    } finally {
      T.setBusy(btn, false);
    }
  });

  /* ============================================================
     6. HISTORIQUE screen
     ============================================================ */
  const ALGO_COLOR = { "Q-Learning": "#2FD4BF", "SARSA": "#A78BFA", "Deep Q-Learning": "#5B9DF0", "Brute Force": "#7F8B9A" };

  function minispark(data, color) {
    const W = 160, H = 46, P = 3;
    if (!data || !data.length) return `<svg viewBox="0 0 ${W} ${H}"></svg>`;
    const max = Math.max.apply(null, data), min = Math.min.apply(null, data);
    const xv = (i) => P + (i / Math.max(1, data.length - 1)) * (W - 2 * P);
    const yv = (v) => P + (1 - (v - min) / (max - min || 1)) * (H - 2 * P);
    const pts = data.map((v, i) => xv(i) + "," + yv(v)).join(" ");
    const area = `${xv(0)},${H - P} ` + pts + ` ${xv(data.length - 1)},${H - P}`;
    return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none"><polygon points="${area}" fill="${color}" opacity="0.1"/><polyline points="${pts}" fill="none" stroke="${color}" stroke-width="2" stroke-linejoin="round"/></svg>`;
  }
  function timeAgo(ts) {
    const dt = Math.floor(Date.now() / 1000) - ts;
    if (dt < 60) return "à l'instant";
    if (dt < 3600) return `il y a ${Math.floor(dt / 60)} min`;
    if (dt < 86400) return `il y a ${Math.floor(dt / 3600)} h`;
    return `il y a ${Math.floor(dt / 86400)} j`;
  }
  async function refreshHistory() {
    const list = document.getElementById("hist-list");
    if (!list) return;
    let data;
    try { data = await T.api.listRuns(); }
    catch (e) { T.toast("Impossible de récupérer l'historique : " + e.message, "error"); return; }
    const runs = data.runs;
    if (!runs.length) {
      list.innerHTML = '<div class="hist-empty">Aucun entraînement enregistré. Lancez-en un depuis l\'écran Entraînement.</div>';
    } else {
      let bestId = runs[0].id, bestSteps = runs[0].eval_steps;
      runs.forEach((r) => { if (r.eval_steps < bestSteps) { bestSteps = r.eval_steps; bestId = r.id; } });
      list.innerHTML = runs.map((r, i) => {
        const color = ALGO_COLOR[r.algo] || "#2FD4BF";
        const isBest = r.id === bestId;
        const p = r.params || {};
        return `<div class="hcard ${isBest ? "best" : ""}">
          <div class="hidx">${runs.length - i}</div>
          <div class="hmeta">
            <div class="hmeta-top">
              <span class="halgo"><i style="background:${color}"></i>${r.algo}</span>
              <span class="htime">${timeAgo(r.ts)}</span>
              ${isBest ? '<span class="hstar">★ meilleur</span>' : ''}
            </div>
            <div class="hparams">
              ${p.alpha != null ? `<span class="hparam">α <b>${(+p.alpha).toFixed(3)}</b></span>` : ""}
              ${p.gamma != null ? `<span class="hparam">γ <b>${(+p.gamma).toFixed(3)}</b></span>` : ""}
              ${p.epsilon_decay != null ? `<span class="hparam">ε-decay <b>${(+p.epsilon_decay).toFixed(4)}</b></span>` : ""}
              <span class="hparam">épisodes <b>${r.episodes}</b></span>
              ${r.conv_episode ? `<span class="hparam">conv. <b>~${r.conv_episode} ép.</b></span>` : ""}
              ${r.train_time_s ? `<span class="hparam">temps <b>${r.train_time_s.toFixed(1)} s</b></span>` : ""}
            </div>
          </div>
          <div class="hspark">
            ${minispark(r.curve_steps, color)}
            <span class="hspark-l">pas / épisode</span>
          </div>
          <div class="hres">
            <div><div class="hres-v ${r.eval_steps <= 14 ? "good" : ""}">${r.eval_steps.toFixed(1)}</div><div class="hres-l">PAS MOY.</div></div>
            <div><div class="hres-v">${r.eval_reward >= 0 ? "+" : ""}${r.eval_reward.toFixed(1)}</div><div class="hres-l">RÉCOMP.</div></div>
            <div><div class="hres-v ${r.eval_success >= 0.99 ? "good" : ""}">${(r.eval_success * 100).toFixed(0)}%</div><div class="hres-l">RÉUSSITE</div></div>
          </div>
        </div>`;
      }).join("");
    }
    const cnt = document.getElementById("h-count"); if (cnt) cnt.textContent = runs.length;
    const best = document.getElementById("h-best");
    if (best) {
      if (runs.length) {
        const b = runs.reduce((a, r) => (r.eval_steps < a.eval_steps ? r : a), runs[0]);
        best.textContent = b.eval_steps.toFixed(1) + " pas";
      } else { best.textContent = "—"; }
    }
  }
  document.getElementById("h-clear")?.addEventListener("click", async () => {
    if (!confirm("Vider tout l'historique ?")) return;
    try { await T.api.clearRuns(); T.toast("Historique vidé", "success"); refreshHistory(); }
    catch (e) { T.toast("Échec suppression : " + e.message, "error"); }
  });

  /* ============================================================
     7. INIT
     ============================================================ */
  async function loadOptimized() {
    try {
      const data = await T.api.getAlgos();
      window._OPT = data.optimized;
    } catch (_) {
      window._OPT = {
        "Q-Learning": { alpha: 0.4, gamma: 0.99, epsilon: 1.0, epsilon_decay: 0.999, epsilon_min: 0.01 },
        "SARSA": { alpha: 0.4, gamma: 0.99, epsilon: 1.0, epsilon_decay: 0.9995, epsilon_min: 0.005 },
        "Deep Q-Learning": { alpha: 1e-3, gamma: 0.99, epsilon: 1.0, epsilon_decay: 0.9995, epsilon_min: 0.02, target_sync: 100, warmup: 1000, batch_size: 64, buffer_size: 10000 },
      };
    }
  }

  loadOptimized().then(() => {
    const sc = document.getElementById("rewardChart");
    if (sc) lineChart(sc, { W: 620, H: 220, yMin: -800, yMax: 60, yTicks: [-800, -600, -400, -200, 0], xTicks: xTicks6k, series: [{ data: [], color: "#2FD4BF" }] });
    const st = document.getElementById("stepsChart");
    if (st) lineChart(st, { W: 360, H: 200, yMin: 0, yMax: 210, yTicks: [0, 50, 100, 150, 200], xTicks: xTicks6k, series: [{ data: [], color: "#F7C612" }] });
    refreshHistory();
    resetEpisode();
    go("land");
  });
})();
