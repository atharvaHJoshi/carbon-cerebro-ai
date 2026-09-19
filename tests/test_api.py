from __future__ import annotations


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["app"] == "Green Model Advisor"
    assert "version" in data
    assert "time" in data


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["app"] == "Green Model Advisor"
    assert "endpoints" in data
    assert "generate" in data["endpoints"]


def test_models_list(client):
    resp = client.get("/api/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert len(data["models"]) > 0
    ids = [m["id"] for m in data["models"]]
    assert "llama-3-8b" in ids
    assert "tinygpt" in ids


def test_adapters_list(client):
    resp = client.get("/api/adapters")
    assert resp.status_code == 200
    data = resp.json()
    assert "adapters" in data
    domains = [a["domain"] for a in data["adapters"]]
    assert "software" in domains


def test_regions_list(client):
    resp = client.get("/api/regions")
    assert resp.status_code == 200
    data = resp.json()
    assert "regions" in data
    codes = [r["code"] for r in data["regions"]]
    assert "IND" in codes
    assert "USA" in codes


def test_catalog(client):
    resp = client.get("/api/catalog")
    assert resp.status_code == 200
    data = resp.json()
    assert "overview" in data


def test_generate(client, sample_prompt):
    resp = client.post("/api/generate", json={
        "prompt": sample_prompt,
        "max_tokens": 64,
        "save_record": False,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "request_id" in data
    assert "prompt_intelligence" in data
    assert "optimizations" in data
    assert "result" in data
    assert "response" in data["result"]
    assert len(data["result"]["response"]) > 0
    assert "decision" in data
    assert "explanation" in data


def test_generate_with_options(client, sample_prompt):
    resp = client.post("/api/generate", json={
        "prompt": sample_prompt,
        "max_tokens": 32,
        "temperature": 0.5,
        "use_pruning": True,
        "use_context_compression": True,
        "use_early_exit": True,
        "use_lora": True,
        "strategy": "auto",
        "save_record": False,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["result"]["structured_metrics"]["tokens_in"] > 0


def test_generate_baseline(client, sample_prompt):
    resp = client.post("/api/generate/baseline", json={
        "prompt": sample_prompt,
        "max_tokens": 64,
        "save_record": False,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "metrics" in data
    assert data["metrics"]["tokens_in"] > 0
    assert data["metrics"]["latency_ms"] >= 0


def test_generate_with_history(client, sample_prompt, conversation_history):
    resp = client.post("/api/generate", json={
        "prompt": sample_prompt,
        "history": conversation_history,
        "max_tokens": 64,
        "save_record": False,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "result" in data
    opts = data["optimizations"]
    assert opts["context_compression"] is not None
    assert opts["context_ratio"] <= 1.0


def test_estimate(client):
    resp = client.post("/api/estimate", json={
        "prompt": "What is machine learning?",
        "model_name": "llama-3-8b",
        "max_tokens": 128,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "energy_consumed_kwh" in data
    assert "carbon_emitted_kgco2" in data
    assert data["carbon_emitted_kgco2"] > 0


def test_generate_tinygpt(client):
    resp = client.post("/api/generate", json={
        "prompt": "Hello world",
        "model": "tinygpt",
        "max_tokens": 16,
        "save_record": False,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["execution_plan"]["model"] == "tinygpt"
    assert "response" in data["result"]
    assert data["result"]["backend_details"].get("real_transformer") is True


def test_generate_with_region(client, sample_prompt):
    resp = client.post("/api/generate", json={
        "prompt": sample_prompt,
        "max_tokens": 32,
        "region": "FRA",
        "save_record": False,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["result"]["structured_metrics"]["grid_region"] == "FRA"


def test_analytics_summary(client):
    resp = client.get("/api/analytics/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert "totals" in data
    assert "per_model" in data


def test_analytics_timeseries(client):
    resp = client.get("/api/analytics/timeseries?points=5")
    assert resp.status_code == 200
    data = resp.json()
    assert "timeseries" in data


def test_analytics_requests(client):
    resp = client.get("/api/analytics/requests?limit=10")
    assert resp.status_code == 200
    data = resp.json()
    assert "requests" in data


def test_empty_prompt_rejected(client):
    resp = client.post("/api/generate", json={"prompt": ""})
    assert resp.status_code == 422
