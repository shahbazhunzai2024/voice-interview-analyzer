/* Application initialization: session creation, join flow, view switching.
 * Invite links look like: /app/?session=<id>&invite=<code>
 */
(function () {
  const VIA = window.VIA;
  const $ = (id) => document.getElementById(id);

  const state = { sessionId: null, invite: null, role: null, call: null, dashboard: null, stopMeter: null, health: null };

  function show(view) {
    for (const id of ["setupView", "callView", "endedView"]) $(id).hidden = id !== view;
  }

  function setCallState(s, msg) {
    const pill = $("callState");
    pill.dataset.state = s;
    pill.textContent = msg;
  }

  function setupError(msg) { $("setupError").textContent = msg || ""; }

  async function api(path, { method = "GET", body, invite } = {}) {
    const headers = { "Content-Type": "application/json" };
    if (invite) headers["X-Invite-Code"] = invite;
    const r = await fetch(path, { method, headers, body: body ? JSON.stringify(body) : undefined });
    const data = r.status === 204 ? null : await r.json().catch(() => ({}));
    if (!r.ok) throw new Error((data && data.detail) || `Request failed (${r.status})`);
    return data;
  }

  function linkFor(sessionId, invite) {
    return `${location.origin}${location.pathname}?session=${encodeURIComponent(sessionId)}&invite=${encodeURIComponent(invite)}`;
  }

  async function createSession() {
    setupError("");
    try {
      const s = await api("/api/sessions", { method: "POST", body: { language: "en", roles: ["interviewer", "candidate"], audio_mode: "in_app_webrtc" } });
      $("interviewerLink").value = linkFor(s.session_id, s.invites.interviewer);
      $("candidateLink").value = linkFor(s.session_id, s.invites.candidate);
      $("invitePanel").hidden = false;
      $("sessionInput").value = s.session_id;
      $("inviteInput").value = s.invites.interviewer;
    } catch (e) {
      setupError(e.message);
    }
  }

  async function join() {
    setupError("");
    const sessionId = $("sessionInput").value.trim();
    const invite = $("inviteInput").value.trim();
    if (!sessionId || !invite) return setupError("Enter the session ID and invite code from your link.");
    if (!$("consentBox").checked) return setupError("Please confirm the processing notice to continue.");
    $("joinBtn").disabled = true;
    try {
      await VIA.audio.checkMicrophonePermission();
      const tok = await api(`/api/sessions/${encodeURIComponent(sessionId)}/join-token`, {
        method: "POST",
        body: { invite_code: invite, display_name: $("nameInput").value.trim() || null },
      });
      Object.assign(state, { sessionId, invite, role: tok.role });
      $("sessionLabel").textContent = sessionId;
      $("roleLabel").textContent = tok.role;
      show("callView");

      if (tok.role === "interviewer") startDashboard();
      $("candidateNote").hidden = tok.role !== "candidate";
      $("finishBtn").hidden = tok.role !== "interviewer";

      state.call = new VIA.CallClient({
        onState: setCallState,
        onParticipants: renderParticipants,
        onLocalTrack: (track) => { state.stopMeter = VIA.audio.attachLevelMeter(track, $("localMeter")); },
        onEnded: onCallEnded,
      });
      await state.call.join({ livekitUrl: tok.livekit_url, token: tok.token });
    } catch (e) {
      setCallState("error", "Not connected");
      if ($("callView").hidden) setupError(e.message);
      else if (state.dashboard) state.dashboard.log(`Call error: ${e.message}`);
      else { show("setupView"); setupError(e.message); }
    } finally {
      $("joinBtn").disabled = false;
    }
  }

  /** Dashboard only (no call) — useful in dev mode without LiveKit. */
  function startDashboard() {
    $("dashboard").hidden = false;
    if (state.dashboard) return;
    state.dashboard = new VIA.Dashboard({ sessionId: state.sessionId, invite: state.invite });
    state.dashboard.connect();
    if (state.health && state.health.dev_mode) $("devForm").hidden = false;
  }

  function renderParticipants(list) {
    $("participants").innerHTML = list
      .map((p) => `<span class="participant ${p.role}${p.muted ? " muted" : ""}${p.speaking ? " speaking" : ""}">${p.role}${p.isLocal ? " (you)" : ""}</span>`)
      .join("");
    const muteBtn = $("muteBtn");
    const me = list.find((p) => p.isLocal);
    if (me) muteBtn.textContent = me.muted ? "Unmute" : "Mute";
  }

  function onCallEnded(reason) {
    if (state.stopMeter) { state.stopMeter(); state.stopMeter = null; }
    setCallState("idle", "Not connected");
    if (state.role === "interviewer") {
      // Interviewer keeps the dashboard; they can still end the interview and open the report.
      if (state.dashboard) state.dashboard.log(`Left the call (${reason ?? "disconnected"})`);
      return;
    }
    $("endedMsg").textContent = "You have left the call. Your microphone has been released.";
    show("endedView");
  }

  async function finish() {
    if (!confirmEnd()) return;
    try {
      await api(`/api/sessions/${encodeURIComponent(state.sessionId)}/finish`, { method: "POST", invite: state.invite });
    } catch (e) {
      state.dashboard && state.dashboard.log(`Finish failed: ${e.message}`);
      return;
    }
    if (state.call) await state.call.leave();
    if (state.dashboard) state.dashboard.close();
    $("endedMsg").textContent = "The interview has ended. Analysis has been finalized.";
    $("reportBtn").hidden = false;
    show("endedView");
  }

  // Inline confirmation without blocking browser dialogs.
  let armed = false;
  function confirmEnd() {
    const btn = $("finishBtn");
    if (armed) { armed = false; btn.textContent = "End interview"; return true; }
    armed = true;
    btn.textContent = "Click again to end";
    setTimeout(() => { armed = false; btn.textContent = "End interview"; }, 3000);
    return false;
  }

  async function openReport() {
    try {
      const report = await api(`/api/sessions/${encodeURIComponent(state.sessionId)}/report`, { invite: state.invite });
      const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
      window.open(URL.createObjectURL(blob), "_blank");
    } catch (e) {
      $("endedMsg").textContent = `Could not load report: ${e.message}`;
    }
  }

  async function sendDevTranscript(ev) {
    ev.preventDefault();
    const text = $("devText").value.trim();
    if (!text) return;
    try {
      await api(`/api/sessions/${encodeURIComponent(state.sessionId)}/dev/transcript`, {
        method: "POST", invite: state.invite, body: { role: $("devRole").value, text },
      });
      $("devText").value = "";
      $("devRole").value = $("devRole").value === "interviewer" ? "candidate" : "interviewer";
    } catch (e) {
      state.dashboard && state.dashboard.log(`Dev transcript failed: ${e.message}`);
    }
  }

  async function init() {
    try { state.health = await api("/health"); } catch { state.health = null; }
    if (state.health && !state.health.livekit_configured) {
      setupError("LiveKit is not configured on the server (see .env.example). Calls are unavailable; interviewers can still open the dashboard in dev mode.");
    }

    const params = new URLSearchParams(location.search);
    if (params.get("session")) $("sessionInput").value = params.get("session");
    if (params.get("invite")) $("inviteInput").value = params.get("invite");
    // Remove secrets from the address bar after reading them.
    if (params.has("invite")) history.replaceState(null, "", location.pathname);

    $("createBtn").addEventListener("click", createSession);
    $("joinBtn").addEventListener("click", join);
    $("muteBtn").addEventListener("click", () => state.call && state.call.toggleMute());
    $("leaveBtn").addEventListener("click", () => state.call && state.call.leave());
    $("finishBtn").addEventListener("click", finish);
    $("reportBtn").addEventListener("click", openReport);
    $("devForm").addEventListener("submit", sendDevTranscript);
    document.querySelectorAll("[data-copy]").forEach((b) =>
      b.addEventListener("click", () => {
        const input = $(b.dataset.copy);
        input.select();
        navigator.clipboard && navigator.clipboard.writeText(input.value);
        b.textContent = "Copied";
        setTimeout(() => (b.textContent = "Copy"), 1200);
      })
    );

    // Dev-mode fallback: interviewer can open the dashboard without a LiveKit call.
    if (state.health && state.health.dev_mode && !state.health.livekit_configured) {
      const devBtn = document.createElement("button");
      devBtn.textContent = "Open dashboard without call (dev)";
      devBtn.addEventListener("click", async () => {
        setupError("");
        const sessionId = $("sessionInput").value.trim();
        const invite = $("inviteInput").value.trim();
        try {
          await api(`/api/sessions/${encodeURIComponent(sessionId)}`, { invite });
          Object.assign(state, { sessionId, invite, role: "interviewer" });
          $("sessionLabel").textContent = sessionId;
          $("roleLabel").textContent = "interviewer (dashboard only)";
          $("finishBtn").hidden = false;
          $("muteBtn").hidden = true;
          $("leaveBtn").hidden = true;
          show("callView");
          startDashboard();
        } catch (e) { setupError(e.message); }
      });
      $("joinBtn").after(devBtn);
      devBtn.style.marginLeft = "8px";
    }

    window.addEventListener("beforeunload", () => { if (state.call) state.call.leave(); });
  }

  init();
})();
