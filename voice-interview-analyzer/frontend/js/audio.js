/* Microphone permission and local audio level (Section 5.1).
 * Exposes window.VIA.audio
 */
(function () {
  const VIA = (window.VIA = window.VIA || {});

  /** Ask for microphone permission up front so failures are readable. */
  async function checkMicrophonePermission() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("This browser does not support microphone capture. Use a recent Chrome, Edge, Firefox or Safari over HTTPS.");
    }
    if (!window.isSecureContext) {
      throw new Error("Microphone access requires HTTPS (or http://localhost).");
    }
    try {
      const s = await navigator.mediaDevices.getUserMedia({ audio: true });
      s.getTracks().forEach((t) => t.stop());
    } catch (err) {
      if (err.name === "NotAllowedError") throw new Error("Microphone permission was denied. Allow it in the browser's site settings and try again.");
      if (err.name === "NotFoundError") throw new Error("No microphone was found on this computer.");
      throw new Error(`Microphone error: ${err.name} — ${err.message}`);
    }
  }

  /** Drive a level meter from a MediaStreamTrack. Returns a stop() function. */
  function attachLevelMeter(mediaStreamTrack, meterEl) {
    const ctx = new AudioContext();
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 1024;
    ctx.createMediaStreamSource(new MediaStream([mediaStreamTrack])).connect(analyser);
    const buf = new Float32Array(analyser.fftSize);
    let raf = 0;
    const tick = () => {
      analyser.getFloatTimeDomainData(buf);
      let sum = 0;
      for (const v of buf) sum += v * v;
      meterEl.style.width = `${Math.min(100, Math.sqrt(sum / buf.length) * 400).toFixed(1)}%`;
      raf = requestAnimationFrame(tick);
    };
    tick();
    return () => {
      cancelAnimationFrame(raf);
      meterEl.style.width = "0";
      ctx.close();
    };
  }

  VIA.audio = { checkMicrophonePermission, attachLevelMeter };
})();
