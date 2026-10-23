/* Interviewer dashboard: live transcript, voice graph, Q&A and review flags (FR-18, 5.9).
 * Exposes window.VIA.Dashboard
 * Receives structured JSON events over WSS; never call audio.
 */
(function () {
  const VIA = (window.VIA = window.VIA || {});
  const $ = (id) => document.getElementById(id);

  const fmt = (ms) => {
    if (ms == null) return "";
    const s = Math.max(0, Math.round(ms / 1000));
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  };
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

  class Dashboard {
    constructor({ sessionId, invite }) {
      this.sessionId = sessionId;
      this.invite = invite;
      this.ws = null;
      this.closed = false;
      this.retry = 0;
      this.questions = new Map();
      this.answers = new Map();
      this.partials = new Map();
      this.chart = null;
      this.initChart();
    }

    initChart() {
      if (!window.Chart) return;
      const css = getComputedStyle(document.documentElement);
      const cand = css.getPropertyValue("--cand").trim() || "#0f7b8a";
      const muted = css.getPropertyValue("--muted").trim() || "#888";
      this.chart = new Chart($("voiceChart"), {
        type: "line",
        data: {
          labels: [],
          datasets: [
            { label: "Energy vs baseline", data: [], borderColor: cand, backgroundColor: cand, tension: 0.25, pointRadius: 0, yAxisID: "y" },
            { label: "Pitch vs baseline", data: [], borderColor: muted, backgroundColor: muted, borderDash: [4, 3], tension: 0.25, pointRadius: 0, yAxisID: "y" },
          ],
        },
        options: {
          animation: false,
          responsive: true,
          interaction: { mode: "index", intersect: false },
          scales: {
            x: { ticks: { maxTicksLimit: 8, color: muted }, grid: { display: false } },
            y: { suggestedMin: 0.5, suggestedMax: 1.5, ticks: { color: muted, callback: (v) => `${v}×` } },
          },
          plugins: { legend: { labels: { color: muted, boxWidth: 12 } } },
        },
      });
    }

    connect() {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      const url = `${proto}://${location.host}/api/sessions/${encodeURIComponent(this.sessionId)}/events?invite=${encodeURIComponent(this.invite)}`;
      this.ws = new WebSocket(url);
      this.ws.onopen = () => { this.retry = 0; this.log("Dashboard connected"); };
      this.ws.onmessage = (m) => {
        try { this.handle(JSON.parse(m.data)); } catch (e) { console.error(e); }
      };
      this.ws.onclose = (e) => {
        if (this.closed) return;
        if (e.code === 1008) { this.log("Dashboard access denied"); return; }
        const delay = Math.min(15000, 1000 * 2 ** this.retry++);
        this.log(`Dashboard disconnected; retrying in ${Math.round(delay / 1000)}s`);
        setTimeout(() => this.connect(), delay);
      };
    }

    close() { this.closed = true; if (this.ws) this.ws.close(); }

    log(msg) {
      const el = document.createElement("div");
      el.textContent = `${new Date().toLocaleTimeString()}  ${msg}`;
      $("statusLog").prepend(el);
    }

    handle(ev) {
      switch (ev.type) {
        case "session_status": return this.log(`[${ev.status}] ${ev.message || ""}`);
        case "transcript.partial": return this.onTranscript(ev);
        case "segment.completed": return;
        case "voice_metrics": return this.onMetrics(ev.metrics);
        case "question.detected": return this.onQuestion(ev.question);
        case "answer.analysis": return this.onAnswer(ev.answer);
        case "review.flag": return this.onFlag(ev.flag);
        case "error": return this.log(`ERROR ${ev.code}: ${ev.message}${ev.recoverable ? " (recoverable)" : ""}`);
        case "ping": return;
        default: console.debug("unknown event", ev);
      }
    }

    onTranscript(ev) {
      const box = $("transcript");
      const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
      let el = this.partials.get(`${ev.role}:${ev.segment_id}`);
      if (!el) {
        el = document.createElement("div");
        el.className = `line ${ev.role}`;
        el.dataset.start = ev.start_ms;
        box.appendChild(el);
      }
      el.classList.toggle("partial", !ev.is_final);
      el.innerHTML = `<span class="who">${esc(ev.role)}</span><span class="t">${fmt(ev.start_ms)}</span><div class="text">${esc(ev.text)}</div>`;
      if (ev.is_final) this.partials.delete(`${ev.role}:${ev.segment_id}`);
      else this.partials.set(`${ev.role}:${ev.segment_id}`, el);
      if (atBottom) box.scrollTop = box.scrollHeight;
    }

    onMetrics(m) {
      const stats = $("voiceStats");
      stats.innerHTML = [
        ["Energy (RMS)", m.rms != null ? m.rms.toFixed(3) : "—"],
        ["Pitch (F0)", m.f0_hz != null ? `${Math.round(m.f0_hz)} Hz` : "—"],
        ["vs baseline", m.rms_vs_baseline != null ? `${m.rms_vs_baseline.toFixed(2)}×` : "building…"],
      ].map(([k, v]) => `<div class="stat"><div class="v">${v}</div><div class="k">${k}</div></div>`).join("");
      if (!this.chart) return;
      const d = this.chart.data;
      d.labels.push(fmt(m.start_ms));
      d.datasets[0].data.push(m.rms_vs_baseline);
      d.datasets[1].data.push(m.f0_vs_baseline);
      const MAX = 240;
      if (d.labels.length > MAX) { d.labels.shift(); d.datasets.forEach((s) => s.data.shift()); }
      this.chart.update("none");
    }

    onQuestion(q) {
      this.questions.set(q.question_id, q);
      this.renderQA();
    }

    onAnswer(a) {
      this.answers.set(a.answer_id, a);
      this.renderQA();
    }

    renderQA() {
      const list = $("qaList");
      const byQ = new Map();
      const orphans = [];
      for (const a of this.answers.values()) {
        if (a.question_id && this.questions.has(a.question_id)) {
          if (!byQ.has(a.question_id)) byQ.set(a.question_id, []);
          byQ.get(a.question_id).push(a);
        } else orphans.push(a);
      }
      const ans = (a) => {
        const label = a.relevance || a.status;
        return `<div class="a"><span class="tag ${esc(label)}">${esc(label.replace(/_/g, " "))}</span>
          <span class="muted small">${fmt(a.start_ms)}–${fmt(a.end_ms)}</span>
          <div>${esc(a.answer_text)}</div>
          <div class="muted small">${esc(a.rationale)}</div></div>`;
      };
      let html = "";
      for (const q of [...this.questions.values()].sort((x, y) => x.start_ms - y.start_ms)) {
        html += `<div class="qa"><div class="q">${fmt(q.start_ms)} · ${esc(q.text)}
          <span class="tag">${esc(q.kind)} · ${Math.round(q.confidence * 100)}%</span></div>
          ${(byQ.get(q.question_id) || []).map(ans).join("") || '<div class="a muted small">Awaiting answer…</div>'}</div>`;
      }
      if (orphans.length) html += `<div class="qa"><div class="q muted">Unpaired candidate turns</div>${orphans.map(ans).join("")}</div>`;
      list.innerHTML = html || '<p class="muted small">Questions are detected automatically from the interviewer transcript.</p>';
    }

    onFlag(f) {
      const el = document.createElement("div");
      el.className = "flag";
      el.innerHTML = `<div class="head"><span>${esc(f.category.replace(/_/g, " "))}</span>
        <span class="muted small">${fmt(f.start_ms)}–${fmt(f.end_ms)}</span></div>
        <div>${esc(f.explanation)}</div>
        <div class="unc">Uncertainty: ${esc(f.uncertainty)}</div>
        <pre>${esc(JSON.stringify(f.evidence, null, 1))}</pre>`;
      // Jump to the matching transcript moment (audio replay requires retention; FR-12).
      el.addEventListener("click", () => this.jumpTo(f.start_ms));
      $("flagList").prepend(el);
    }

    jumpTo(ms) {
      const lines = [...document.querySelectorAll("#transcript .line")];
      const target = lines.reduce((best, el) => {
        const d = Math.abs(Number(el.dataset.start) - ms);
        return !best || d < best.d ? { el, d } : best;
      }, null);
      if (!target) return;
      target.el.scrollIntoView({ behavior: "smooth", block: "center" });
      target.el.classList.add("highlight");
      setTimeout(() => target.el.classList.remove("highlight"), 1600);
    }
  }

  VIA.Dashboard = Dashboard;
})();
