/* LiveKit / WebRTC call handling (FR-01, FR-13, FR-15, FR-16, FR-19).
 * Exposes window.VIA.CallClient
 *
 * Each browser publishes only its own microphone track. Remote audio is
 * attached to hidden <audio> elements so participants hear one another.
 */
(function () {
  const VIA = (window.VIA = window.VIA || {});

  function roleOf(identity) {
    const r = String(identity || "").split(":")[0];
    return r === "interviewer" || r === "candidate" ? r : null;
  }

  class CallClient {
    /**
     * @param {object} hooks { onState(state, msg), onParticipants(list), onLocalTrack(mediaStreamTrack), onEnded(reason) }
     */
    constructor(hooks) {
      this.hooks = hooks;
      this.room = null;
      this.audioEls = new Map();
    }

    async join({ livekitUrl, token }) {
      if (!window.LivekitClient) throw new Error("LiveKit SDK failed to load (check network / CDN access).");
      const { Room, RoomEvent, Track } = window.LivekitClient;

      const room = new Room({
        adaptiveStream: true,
        dynacast: true,
        audioCaptureDefaults: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
      this.room = room;

      room
        .on(RoomEvent.TrackSubscribed, (track, _pub, participant) => {
          if (track.kind !== Track.Kind.Audio) return;
          const el = track.attach();
          el.dataset.identity = participant.identity;
          document.body.appendChild(el);
          this.audioEls.set(track.sid, el);
          this.emitParticipants();
        })
        .on(RoomEvent.TrackUnsubscribed, (track) => {
          track.detach().forEach((el) => el.remove());
          this.audioEls.delete(track.sid);
          this.emitParticipants();
        })
        .on(RoomEvent.ParticipantConnected, () => this.emitParticipants())
        .on(RoomEvent.ParticipantDisconnected, () => this.emitParticipants())
        .on(RoomEvent.TrackMuted, () => this.emitParticipants())
        .on(RoomEvent.TrackUnmuted, () => this.emitParticipants())
        .on(RoomEvent.ActiveSpeakersChanged, () => this.emitParticipants())
        .on(RoomEvent.Reconnecting, () => this.hooks.onState("reconnecting", "Reconnecting…"))
        .on(RoomEvent.Reconnected, () => this.hooks.onState("connected", "Connected"))
        .on(RoomEvent.Disconnected, (reason) => {
          this.cleanup();
          this.hooks.onEnded(reason);
        })
        .on(RoomEvent.MediaDevicesError, (e) => this.hooks.onState("error", `Microphone error: ${e.message}`));

      this.hooks.onState("connecting", "Connecting…");
      await room.connect(livekitUrl, token);
      await room.localParticipant.setMicrophoneEnabled(true);
      const pub = room.localParticipant.getTrackPublication(Track.Source.Microphone);
      if (pub && pub.track) this.hooks.onLocalTrack(pub.track.mediaStreamTrack);
      this.hooks.onState("connected", "Connected");
      this.emitParticipants();

      // Some browsers block autoplay until a user gesture.
      if (!room.canPlaybackAudio) {
        this.hooks.onState("connected", "Click anywhere to enable call audio");
        document.addEventListener("click", () => room.startAudio(), { once: true });
      }
    }

    emitParticipants() {
      if (!this.room) return;
      const { Track } = window.LivekitClient;
      const all = [this.room.localParticipant, ...this.room.remoteParticipants.values()];
      const list = all
        .map((p) => {
          const pub = p.getTrackPublication(Track.Source.Microphone);
          return {
            identity: p.identity,
            name: p.name || p.identity,
            role: roleOf(p.identity),
            isLocal: p === this.room.localParticipant,
            muted: !pub || pub.isMuted,
            speaking: p.isSpeaking,
          };
        })
        .filter((p) => p.role);
      this.hooks.onParticipants(list);
    }

    async toggleMute() {
      if (!this.room) return false;
      const lp = this.room.localParticipant;
      const enabled = lp.isMicrophoneEnabled;
      await lp.setMicrophoneEnabled(!enabled);
      this.emitParticipants();
      return enabled; // true => now muted
    }

    async leave() {
      if (this.room) await this.room.disconnect(true); // stops local tracks
      this.cleanup();
    }

    cleanup() {
      for (const el of this.audioEls.values()) el.remove();
      this.audioEls.clear();
      this.room = null;
    }
  }

  VIA.CallClient = CallClient;
  VIA.roleOf = roleOf;
})();
