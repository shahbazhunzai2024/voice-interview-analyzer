import base64
import json

import pytest

from app.config import settings


def _create(client):
    r = client.post("/api/sessions", json={"language": "en", "roles": ["interviewer", "candidate"],
                                           "audio_mode": "in_app_webrtc"})
    assert r.status_code == 201
    return r.json()


def _jwt_claims(token: str) -> dict:
    payload = token.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    return json.loads(base64.urlsafe_b64decode(payload))


def test_create_session_returns_role_invites(client):
    body = _create(client)
    assert body["status"] == "created"
    assert set(body["invites"]) == {"interviewer", "candidate"}
    assert body["invites"]["interviewer"] != body["invites"]["candidate"]


def test_create_session_rejects_bad_roles(client):
    r = client.post("/api/sessions", json={"roles": ["interviewer"]})
    assert r.status_code == 422


def test_join_token_rejects_invalid_invite(client):
    s = _create(client)
    r = client.post(f"/api/sessions/{s['session_id']}/join-token", json={"invite_code": "not-a-real-code"})
    assert r.status_code == 403


def test_join_token_requires_livekit_config(client):
    s = _create(client)
    r = client.post(f"/api/sessions/{s['session_id']}/join-token",
                    json={"invite_code": s["invites"]["candidate"]})
    assert r.status_code == 503


def test_join_token_role_comes_from_invite_not_client(client, monkeypatch):
    pytest.importorskip("livekit.api")
    monkeypatch.setenv("LIVEKIT_URL", "wss://example.livekit.cloud")
    monkeypatch.setenv("LIVEKIT_API_KEY", "devkey")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "devsecret-devsecret-devsecret-123")
    monkeypatch.setenv("ANALYSIS_WORKER_ENABLED", "false")
    settings.reload()

    import app.services.session_service as ss

    async def _noop(*a, **k):
        return False
    monkeypatch.setattr(ss, "create_room", _noop)
    monkeypatch.setattr(ss, "delete_room", _noop)

    s = _create(client)
    # A client-supplied "role" field is ignored; the invite decides the role.
    r = client.post(f"/api/sessions/{s['session_id']}/join-token",
                    json={"invite_code": s["invites"]["candidate"], "role": "interviewer"})
    assert r.status_code == 200
    body = r.json()
    assert body["role"] == "candidate"
    claims = _jwt_claims(body["token"])
    assert claims["sub"] == body["identity"] and claims["sub"].startswith("candidate:")
    assert claims["video"]["room"] == s["room_name"]
    assert claims["video"]["canPublishSources"] == ["microphone"]
    assert json.loads(claims["metadata"])["role"] == "candidate"
    assert claims["exp"] - claims["nbf"] <= settings.join_token_ttl_s + 5


def test_finish_requires_interviewer(client):
    s = _create(client)
    sid = s["session_id"]
    r = client.post(f"/api/sessions/{sid}/finish", headers={"X-Invite-Code": s["invites"]["candidate"]})
    assert r.status_code == 403
    r = client.post(f"/api/sessions/{sid}/finish", headers={"X-Invite-Code": s["invites"]["interviewer"]})
    assert r.status_code == 200 and r.json()["status"] == "finished"
    # ended sessions refuse new joins
    r = client.post(f"/api/sessions/{sid}/join-token", json={"invite_code": s["invites"]["candidate"]})
    assert r.status_code == 409


def test_unknown_session_404(client):
    r = client.get("/api/sessions/doesnotexist", headers={"X-Invite-Code": "x" * 10})
    assert r.status_code == 404


def test_events_websocket_rejects_candidate(client):
    from starlette.websockets import WebSocketDisconnect

    s = _create(client)
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(
            f"/api/sessions/{s['session_id']}/events?invite={s['invites']['candidate']}"
        ) as ws:
            ws.receive_json()


def test_dev_transcript_flows_to_dashboard_and_report(client):
    s = _create(client)
    sid, iv = s["session_id"], s["invites"]["interviewer"]
    h = {"X-Invite-Code": iv}

    with client.websocket_connect(f"/api/sessions/{sid}/events?invite={iv}") as ws:
        first = ws.receive_json()
        assert first["type"] == "session_status"

        client.post(f"/api/sessions/{sid}/dev/transcript", headers=h, json={
            "role": "interviewer", "text": "Can you tell me about your experience with Python testing?",
            "start_ms": 0, "end_ms": 4000})
        client.post(f"/api/sessions/{sid}/dev/transcript", headers=h, json={
            "role": "candidate",
            "text": "I have written Python tests with pytest for five years, mostly testing APIs and data pipelines.",
            "start_ms": 5000, "end_ms": 11000})

        types = [ws.receive_json()["type"] for _ in range(4)]
        assert types == ["transcript.partial", "question.detected", "transcript.partial", "answer.analysis"]

    client.post(f"/api/sessions/{sid}/finish", headers=h)
    report = client.get(f"/api/sessions/{sid}/report", headers=h).json()
    assert len(report["questions"]) == 1
    assert len(report["answers"]) == 1
    ans = report["answers"][0]
    assert ans["status"] == "paired"
    assert ans["question_id"] == report["questions"][0]["question_id"]
    assert ans["relevance"] == "relevant"
    assert "pass" not in json.dumps(report["disclaimer"]).lower().replace("pass/fail", "")
    assert report["limitations"]


def test_dev_transcript_role_requires_interviewer_invite(client):
    s = _create(client)
    r = client.post(f"/api/sessions/{s['session_id']}/dev/transcript",
                    headers={"X-Invite-Code": s["invites"]["candidate"]},
                    json={"role": "candidate", "text": "hello"})
    assert r.status_code == 403


def test_report_requires_interviewer(client):
    s = _create(client)
    r = client.get(f"/api/sessions/{s['session_id']}/report",
                   headers={"X-Invite-Code": s["invites"]["candidate"]})
    assert r.status_code == 403
