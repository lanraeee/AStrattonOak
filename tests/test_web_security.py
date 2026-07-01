"""Tests for the web layer's security and provider-resolution fixes.

Covers the blockers flagged in review of the Alpaca/web PR:
  * the JSON API can be gated behind a shared token,
  * credentialed CORS is never paired with a wildcard origin,
  * the broker defaults to PAPER trading (live is opt-in),
  * the live-money execute endpoint requires a per-analysis confirm token,
  * auto-selecting a provider realigns the model (no provider/model mismatch),
    and providers without catalog defaults are not silently auto-selected.

These import the FastAPI app lazily and reload it under patched env so the
module-level token/CORS wiring is exercised for real, not mocked away.
"""

import importlib

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


def _reload_app(monkeypatch, env):
    """Reload web.app with a specific env so module-level config re-evaluates."""
    for k, v in env.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    import web.app as app_module
    return importlib.reload(app_module)


# ---------------------------------------------------------------------------
# Broker: paper is the default; live is opt-in.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBrokerPaperDefault:
    def test_unset_defaults_to_paper(self, monkeypatch):
        monkeypatch.delenv("ALPACA_PAPER", raising=False)
        from web.broker import AlpacaBroker
        b = AlpacaBroker()
        assert b.paper is True
        assert b.mode == "paper"
        assert b.base_url == "https://paper-api.alpaca.markets"

    @pytest.mark.parametrize("value", ["true", "1", "yes", "on", "anything", ""])
    def test_non_false_stays_paper(self, monkeypatch, value):
        monkeypatch.setenv("ALPACA_PAPER", value)
        from web.broker import AlpacaBroker
        assert AlpacaBroker().paper is True

    @pytest.mark.parametrize("value", ["false", "0", "no", "off", "FALSE", "Off"])
    def test_explicit_false_opts_into_live(self, monkeypatch, value):
        monkeypatch.setenv("ALPACA_PAPER", value)
        from web.broker import AlpacaBroker
        b = AlpacaBroker()
        assert b.paper is False
        assert b.mode == "live"
        assert b.base_url == "https://api.alpaca.markets"


# ---------------------------------------------------------------------------
# API token gating.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestApiTokenGate:
    def test_no_token_configured_allows_access(self, monkeypatch):
        app_module = _reload_app(monkeypatch, {"TRADINGAGENTS_API_TOKEN": None})
        client = TestClient(app_module.app)
        assert client.get("/api/config").status_code == 200

    def test_token_required_when_configured(self, monkeypatch):
        app_module = _reload_app(monkeypatch, {"TRADINGAGENTS_API_TOKEN": "s3cret"})
        client = TestClient(app_module.app)
        assert client.get("/api/config").status_code == 401
        assert client.get(
            "/api/config", headers={"X-API-Token": "wrong"}
        ).status_code == 401
        assert client.get(
            "/api/config", headers={"X-API-Token": "s3cret"}
        ).status_code == 200
        assert client.get(
            "/api/config", headers={"Authorization": "Bearer s3cret"}
        ).status_code == 200

    def test_health_and_index_are_unauthenticated(self, monkeypatch):
        app_module = _reload_app(monkeypatch, {"TRADINGAGENTS_API_TOKEN": "s3cret"})
        client = TestClient(app_module.app)
        assert client.get("/health").status_code == 200


# ---------------------------------------------------------------------------
# CORS: credentialed wildcard is impossible.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCorsPolicy:
    def test_wildcard_origin_has_no_credentials(self, monkeypatch):
        app_module = _reload_app(
            monkeypatch, {"TRADINGAGENTS_CORS_ORIGINS": None}
        )
        mw = [m for m in app_module.app.user_middleware if "CORS" in str(m.cls)][0]
        opts = mw.kwargs
        assert opts["allow_origins"] == ["*"]
        assert opts["allow_credentials"] is False

    def test_explicit_allowlist_enables_credentials(self, monkeypatch):
        app_module = _reload_app(
            monkeypatch,
            {"TRADINGAGENTS_CORS_ORIGINS": "https://a.example.com, https://b.example.com"},
        )
        mw = [m for m in app_module.app.user_middleware if "CORS" in str(m.cls)][0]
        opts = mw.kwargs
        assert opts["allow_origins"] == [
            "https://a.example.com",
            "https://b.example.com",
        ]
        assert opts["allow_credentials"] is True
        assert "*" not in opts["allow_origins"]


# ---------------------------------------------------------------------------
# Provider / model realignment.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProviderModelResolution:
    def test_force_realigns_models(self):
        import web.app as app_module
        cfg = {"deep_think_llm": "gpt-5.5", "quick_think_llm": "gpt-5.4-mini"}
        app_module._resolve_provider_models("anthropic", cfg, force=True)
        assert cfg["deep_think_llm"] == "claude-opus-4-8"
        assert cfg["quick_think_llm"] == "claude-haiku-4-5"

    def test_minimax_has_defaults_and_realigns(self):
        import web.app as app_module
        assert "minimax" in app_module.PROVIDER_DEFAULT_MODELS
        cfg = {"deep_think_llm": "gpt-5.5", "quick_think_llm": "gpt-5.4-mini"}
        app_module._resolve_provider_models("minimax", cfg)
        # gpt-* is not minimax family, so both should be replaced.
        assert cfg["deep_think_llm"] == "MiniMax-M2.7"
        assert cfg["quick_think_llm"] == "MiniMax-M2.7-highspeed"

    def test_first_configured_requires_defaults_skips_openrouter(self, monkeypatch):
        import web.app as app_module
        # Clear every provider key, then set only openrouter's.
        for env in app_module.PROVIDER_API_KEY_ENV.values():
            if env:
                monkeypatch.delenv(env, raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "x")
        # Without require_defaults it would happily return openrouter...
        assert app_module._first_configured_provider() == "openrouter"
        # ...but with require_defaults it must skip it (no catalog defaults).
        assert app_module._first_configured_provider(require_defaults=True) is None

    def test_auto_select_rejects_when_only_openrouter_configured(self, monkeypatch):
        app_module = _reload_app(monkeypatch, {"TRADINGAGENTS_API_TOKEN": None})
        for env in app_module.PROVIDER_API_KEY_ENV.values():
            if env:
                monkeypatch.delenv(env, raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "x")
        client = TestClient(app_module.app)
        resp = client.post(
            "/api/analyze/start",
            json={"ticker": "AAPL", "config": {"llm_provider": "openai"}},
        )
        assert resp.status_code == 400
        assert "No API key" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Execute endpoint: confirm-token guard.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExecuteConfirmToken:
    def _seed_completed_with_order(self, app_module, token="tok-123"):
        app_module.active_analyses.clear()
        app_module.active_analyses["AAPL_1"] = {
            "id": "AAPL_1",
            "ticker": "AAPL",
            "status": "completed",
            "progress": 100,
            "messages": [],
            "result": None,
            "error": None,
            "started_at": "now",
            "proposed_order": {"symbol": "AAPL", "side": "buy", "rating": "buy", "mode": "paper"},
            "order_result": None,
            "confirm_token": token,
        }

    def test_missing_token_is_rejected(self, monkeypatch):
        app_module = _reload_app(monkeypatch, {"TRADINGAGENTS_API_TOKEN": None})
        self._seed_completed_with_order(app_module)
        # Configure a broker so we get past the is_configured guard.
        monkeypatch.setattr(
            app_module.AlpacaBroker, "is_configured", lambda self: True
        )
        client = TestClient(app_module.app)
        resp = client.post("/api/analyze/AAPL_1/execute", json={"notional": 100})
        assert resp.status_code == 403
        # No order should have been placed.
        assert app_module.active_analyses["AAPL_1"]["order_result"] is None

    def test_wrong_token_is_rejected(self, monkeypatch):
        app_module = _reload_app(monkeypatch, {"TRADINGAGENTS_API_TOKEN": None})
        self._seed_completed_with_order(app_module)
        monkeypatch.setattr(
            app_module.AlpacaBroker, "is_configured", lambda self: True
        )
        client = TestClient(app_module.app)
        resp = client.post(
            "/api/analyze/AAPL_1/execute",
            json={"notional": 100, "confirm_token": "nope"},
        )
        assert resp.status_code == 403

    def test_valid_token_places_order(self, monkeypatch):
        app_module = _reload_app(monkeypatch, {"TRADINGAGENTS_API_TOKEN": None})
        self._seed_completed_with_order(app_module, token="good-token")

        placed = {}

        def fake_place_order(self, symbol, side, notional=None, qty=None):
            placed.update(symbol=symbol, side=side, notional=notional)
            return {
                "id": "ord1", "symbol": symbol, "side": side, "type": "market",
                "qty": None, "notional": notional, "status": "accepted",
                "submitted_at": "now",
            }

        monkeypatch.setattr(
            app_module.AlpacaBroker, "is_configured", lambda self: True
        )
        monkeypatch.setattr(app_module.AlpacaBroker, "place_order", fake_place_order)
        client = TestClient(app_module.app)
        resp = client.post(
            "/api/analyze/AAPL_1/execute",
            json={"notional": 100, "confirm_token": "good-token"},
        )
        assert resp.status_code == 200, resp.text
        assert placed["symbol"] == "AAPL"
        assert app_module.active_analyses["AAPL_1"]["order_result"]["id"] == "ord1"
