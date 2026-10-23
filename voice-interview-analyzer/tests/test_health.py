def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["livekit_configured"] is False
    # secrets are never exposed
    assert not any("secret" in k or "key" in k for k in body)


def test_home_serves_microphone_prototype(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Microphone Test" in r.text


def test_frontend_is_served(client):
    r = client.get("/app/")
    assert r.status_code == 200
    assert "Voice Interview Analyzer" in r.text
