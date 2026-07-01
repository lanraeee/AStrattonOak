"""FastAPI web application for TradingAgents."""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

import secrets

from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
    BackgroundTasks,
    Depends,
    Header,
)
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import lightweight config
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_clients.model_catalog import get_model_options, get_known_models
from tradingagents.llm_clients.api_key_env import get_api_key_env, PROVIDER_API_KEY_ENV
from web.broker import AlpacaBroker, rating_to_side

# Heavy imports (trading_graph imports yfinance, etc.) will be done lazily in background tasks

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path to the single-page UI, resolved once at import time.
_INDEX_HTML = os.path.join(os.path.dirname(__file__), "static", "index.html")

# Global store of analyses keyed by id. Capped to avoid unbounded growth in a
# long-running process; see _evict_old_analyses.
active_analyses: Dict[str, Dict[str, Any]] = {}

# Maximum number of analyses to retain in memory. Oldest terminal analyses are
# evicted first once this is exceeded.
MAX_ANALYSES = int(os.getenv("TRADINGAGENTS_WEB_MAX_ANALYSES", "100"))

# Upper bound on concurrently in-flight (queued/running) analyses. Each run is
# a heavy multi-agent job; this prevents a flood of requests from exhausting
# memory or the background-task thread pool. New starts past this are rejected
# with 429 rather than evicting a live analysis.
MAX_CONCURRENT_ANALYSES = int(os.getenv("TRADINGAGENTS_WEB_MAX_CONCURRENT", "5"))

# Optional shared secret guarding the JSON API. When TRADINGAGENTS_API_TOKEN is
# set, every /api route requires a matching X-API-Token header (or
# "Authorization: Bearer <token>"). This is the minimum bar before exposing the
# live-money execute endpoint to anything other than localhost.
_API_TOKEN = os.getenv("TRADINGAGENTS_API_TOKEN", "").strip()


def require_api_token(
    x_api_token: Optional[str] = Header(default=None),
    authorization: Optional[str] = Header(default=None),
) -> None:
    """FastAPI dependency enforcing the shared API token when one is configured.

    No-op when TRADINGAGENTS_API_TOKEN is unset so local/dev use is unaffected,
    but the moment a token is configured (recommended for any deployment that
    can place real orders) callers must present it. Comparison is constant-time.
    """
    if not _API_TOKEN:
        return
    presented = x_api_token
    if not presented and authorization:
        scheme, _, value = authorization.partition(" ")
        if scheme.lower() == "bearer":
            presented = value.strip()
    if not presented or not secrets.compare_digest(presented, _API_TOKEN):
        raise HTTPException(status_code=401, detail="Missing or invalid API token.")

# Sensible default models per provider, used when the chosen provider does
# not match the configured models (e.g. provider=anthropic but the default
# config still points deep_think_llm at an OpenAI "gpt-5.5"). Keeps the
# provider and model in sync so the right SDK/key is exercised.
PROVIDER_DEFAULT_MODELS: Dict[str, Dict[str, str]] = {
    "openai":     {"deep": "gpt-5.5",            "quick": "gpt-5.4-mini"},
    "anthropic":  {"deep": "claude-opus-4-8",    "quick": "claude-haiku-4-5"},
    "google":     {"deep": "gemini-3.1-pro",     "quick": "gemini-3.1-flash"},
    "xai":        {"deep": "grok-4",             "quick": "grok-4-mini"},
    "deepseek":   {"deep": "deepseek-reasoner",  "quick": "deepseek-chat"},
    "qwen":       {"deep": "qwen3.7-max",        "quick": "qwen3.6-flash"},
    "qwen-cn":    {"deep": "qwen3.7-max",        "quick": "qwen3.6-flash"},
    "glm":        {"deep": "glm-5.1",            "quick": "glm-5-turbo"},
    "glm-cn":     {"deep": "glm-5.1",            "quick": "glm-5-turbo"},
    "minimax":    {"deep": "MiniMax-M2.7",       "quick": "MiniMax-M2.7-highspeed"},
    "minimax-cn": {"deep": "MiniMax-M2.7",       "quick": "MiniMax-M2.7-highspeed"},
}


def _first_configured_provider(require_defaults: bool = False) -> Optional[str]:
    """Return the first provider (in preference order) that has an API key set.

    When ``require_defaults`` is set, only providers with catalog defaults in
    ``PROVIDER_DEFAULT_MODELS`` are considered. This is used during auto-select
    so we never silently pair a freshly-chosen provider with the previously
    requested provider's model name (a provider/model mismatch).
    """
    preference = [
        "openai", "anthropic", "google", "xai", "deepseek",
        "qwen", "qwen-cn", "glm", "glm-cn", "minimax", "minimax-cn",
        "openrouter",
    ]
    for provider in preference:
        if require_defaults and provider not in PROVIDER_DEFAULT_MODELS:
            continue
        env_var = get_api_key_env(provider)
        if env_var and os.environ.get(env_var):
            return provider
    return None


def _resolve_provider_models(
    provider: str, config: Dict[str, Any], force: bool = False
) -> None:
    """Ensure config's models match the chosen provider, in-place.

    If the user left the model as the default (or it belongs to a different
    provider family), substitute that provider's recommended models so the
    correct client and API key are used. When ``force`` is set (e.g. after an
    auto-select away from the requested provider) the provider's defaults are
    applied unconditionally, so a stale model name cannot survive a provider
    switch.
    """
    defaults = PROVIDER_DEFAULT_MODELS.get(provider)
    if not defaults:
        return

    if force:
        config["deep_think_llm"] = defaults["deep"]
        config["quick_think_llm"] = defaults["quick"]
        return

    deep = config.get("deep_think_llm") or DEFAULT_CONFIG.get("deep_think_llm", "")
    quick = config.get("quick_think_llm") or DEFAULT_CONFIG.get("quick_think_llm", "")

    # Detect a provider/model family mismatch via known model-name prefixes.
    family_prefixes = {
        "openai": "gpt", "anthropic": "claude", "google": "gemini",
        "xai": "grok", "deepseek": "deepseek", "qwen": "qwen", "qwen-cn": "qwen",
        "glm": "glm", "glm-cn": "glm", "minimax": "minimax", "minimax-cn": "minimax",
    }
    prefix = family_prefixes.get(provider, "")

    if not deep or (prefix and not deep.lower().startswith(prefix)):
        config["deep_think_llm"] = defaults["deep"]
    if not quick or (prefix and not quick.lower().startswith(prefix)):
        config["quick_think_llm"] = defaults["quick"]


def _evict_old_analyses() -> None:
    """Bound the in-memory store, evicting oldest *terminal* analyses only.

    ``active_analyses`` preserves insertion order, so the earliest-created
    entries appear first. Only completed/failed analyses are ever dropped — a
    still-running (queued/running) analysis is never evicted, even if that
    leaves the store above ``MAX_ANALYSES`` (concurrent in-flight runs are
    bounded separately by ``_running_count``/``MAX_CONCURRENT_ANALYSES`` at
    start time).
    """
    if len(active_analyses) < MAX_ANALYSES:
        return

    terminal = [
        aid for aid, a in active_analyses.items()
        if a["status"] in ("completed", "failed")
    ]
    while len(active_analyses) >= MAX_ANALYSES and terminal:
        active_analyses.pop(terminal.pop(0), None)


def _running_count() -> int:
    """Number of analyses that are queued or actively running."""
    return sum(
        1 for a in active_analyses.values()
        if a["status"] in ("queued", "running")
    )


def _bump_progress(analysis_id: str, step: int = 2, ceiling: int = 85) -> None:
    """Nudge progress forward as agents do work, capped below completion."""
    analysis = active_analyses.get(analysis_id)
    if analysis is not None and analysis["progress"] < ceiling:
        analysis["progress"] = min(analysis["progress"] + step, ceiling)


def _make_progress_callback(analysis_id: str):
    """Build a LangChain callback handler that streams live agent activity.

    Imported lazily so that importing this module (or running the API without
    triggering an analysis) does not require langchain_core.
    """
    from langchain_core.callbacks import BaseCallbackHandler

    class ProgressCallbackHandler(BaseCallbackHandler):
        """Pushes real-time LLM/tool events into the analysis message feed.

        Captures the actual agent reasoning/report text and tool calls/results
        so the web UI mirrors what the framework prints to the server log.
        """

        def on_chat_model_start(self, serialized, messages, **kwargs):
            _bump_progress(analysis_id)

        def on_llm_start(self, serialized, prompts, **kwargs):
            _bump_progress(analysis_id)

        def on_llm_end(self, response, **kwargs):
            # Surface the model's generated text (analyst reports, debate turns,
            # the final decision, etc.) into the GUI feed.
            _bump_progress(analysis_id)
            try:
                for gen_list in response.generations:
                    for gen in gen_list:
                        text = (getattr(gen, "text", "") or "").strip()
                        if not text:
                            msg = getattr(gen, "message", None)
                            content = getattr(msg, "content", "") if msg else ""
                            text = content.strip() if isinstance(content, str) else ""
                        if text:
                            add_message(analysis_id, "agent", text)
            except Exception as e:
                logger.debug(f"on_llm_end parse error: {e}")

        def on_tool_start(self, serialized, input_str, **kwargs):
            name = (serialized or {}).get("name", "tool")
            args = (input_str or "").strip()
            if len(args) > 200:
                args = args[:200] + "…"
            label = f"🔧 {name}({args})" if args else f"🔧 {name}"
            add_message(analysis_id, "tool", label)
            _bump_progress(analysis_id)

        def on_tool_end(self, output, **kwargs):
            text = str(getattr(output, "content", output) or "").strip()
            if not text:
                return
            if len(text) > 1200:
                text = text[:1200] + "\n… (truncated)"
            add_message(analysis_id, "tool", text)

        def on_tool_error(self, error, **kwargs):
            add_message(analysis_id, "warning", f"Tool error: {error}")

    return ProgressCallbackHandler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    logger.info("TradingAgents Web App Starting")
    yield
    logger.info("TradingAgents Web App Shutting Down")


app = FastAPI(
    title="TradingAgents Web",
    description="Multi-Agent LLM Financial Trading Framework",
    version="0.2.5",
    lifespan=lifespan,
)

# CORS middleware.
#
# Origins come from TRADINGAGENTS_CORS_ORIGINS (comma-separated). We never pair
# a wildcard with credentials: the CORS spec forbids it, and credentialed
# wildcard CORS would let any web page in the user's browser drive the
# live-money execute endpoint. When no explicit allowlist is configured we
# fall back to a permissive, *credential-less* policy (safe for a same-origin
# SPA / token-in-header auth), and only enable credentials for an explicit list.
_cors_origins = [
    o.strip()
    for o in os.getenv("TRADINGAGENTS_CORS_ORIGINS", "").split(",")
    if o.strip()
]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# ============================================================================
# Routes
# ============================================================================


@app.get("/")
async def index():
    """Serve the main web app without blocking the event loop."""
    return FileResponse(_INDEX_HTML, media_type="text/html")


@app.get("/api/config", dependencies=[Depends(require_api_token)])
async def get_config():
    """Get current configuration and available options."""
    providers = {
        "openai": "OpenAI",
        "google": "Google Gemini",
        "anthropic": "Anthropic Claude",
        "xai": "xAI Grok",
        "deepseek": "DeepSeek",
        "qwen": "Qwen (International)",
        "qwen-cn": "Qwen (China)",
        "glm": "GLM (International)",
        "glm-cn": "GLM (China)",
        "minimax": "MiniMax (Global)",
        "minimax-cn": "MiniMax (China)",
        "openrouter": "OpenRouter",
        "ollama": "Ollama (Local)",
        "azure": "Azure OpenAI",
    }

    # Get models for each provider. Some providers (e.g. openrouter, azure)
    # are not in the static catalog and resolve via "Custom model ID" instead;
    # an empty list for them is expected, so don't log it as an error.
    models_by_provider = {}
    for provider in providers.keys():
        try:
            quick_models = get_model_options(provider, "quick")
            deep_models = get_model_options(provider, "deep")
            all_models = list(set([m[1] for m in quick_models + deep_models]))[:10]
            models_by_provider[provider] = [
                {"id": model_id, "name": model_id} for model_id in all_models
            ]
        except KeyError:
            # Provider has no catalog entry — fine, UI falls back to ENV/custom.
            models_by_provider[provider] = []
        except Exception as e:
            logger.warning(f"Error loading models for {provider}: {e}")
            models_by_provider[provider] = []

    # Detect which providers have their API key configured in the environment
    provider_key_status = {}
    for provider in providers.keys():
        env_var = get_api_key_env(provider)
        if env_var is None:
            # Local runtimes (ollama) need no key
            provider_key_status[provider] = {"required": False, "configured": True, "env_var": None}
        else:
            provider_key_status[provider] = {
                "required": True,
                "configured": bool(os.environ.get(env_var)),
                "env_var": env_var,
            }

    return {
        "providers": providers,
        "models_by_provider": models_by_provider,
        "provider_key_status": provider_key_status,
        "default_config": {
            "llm_provider": DEFAULT_CONFIG.get("llm_provider", "openai"),
            "deep_think_llm": DEFAULT_CONFIG.get("deep_think_llm", "gpt-5.5"),
            "quick_think_llm": DEFAULT_CONFIG.get("quick_think_llm", "gpt-5.4"),
            "temperature": DEFAULT_CONFIG.get("temperature", 0.7),
            "max_debate_rounds": DEFAULT_CONFIG.get("max_debate_rounds", 2),
            "max_risk_discuss_rounds": DEFAULT_CONFIG.get("max_risk_discuss_rounds", 1),
            "checkpoint_enabled": DEFAULT_CONFIG.get("checkpoint_enabled", False),
            "output_language": DEFAULT_CONFIG.get("output_language", "en"),
        },
        "analysts": [
            {"id": "market", "name": "Market Analyst", "description": "Technical indicators and price patterns"},
            {"id": "social", "name": "Sentiment Analyst", "description": "StockTwits, Reddit sentiment"},
            {"id": "news", "name": "News Analyst", "description": "News and macroeconomic impact"},
            {"id": "fundamentals", "name": "Fundamentals Analyst", "description": "Financial statements and metrics"},
        ],
    }


@app.post("/api/analyze/start", dependencies=[Depends(require_api_token)])
async def start_analysis(background_tasks: BackgroundTasks, request_data: Dict[str, Any]):
    """Start a new analysis."""
    try:
        ticker = request_data.get("ticker", "").upper()
        date = request_data.get("date", datetime.now().strftime("%Y-%m-%d"))
        config = request_data.get("config", {})
        analysts = request_data.get("analysts", ["market", "social", "news", "fundamentals"])

        # Validate inputs
        if not ticker:
            raise HTTPException(status_code=400, detail="Ticker is required")
        if not date:
            raise HTTPException(status_code=400, detail="Date is required")

        # Resolve provider: use the requested one, else the configured default,
        # else auto-pick the first provider that has an API key configured.
        provider = config.get("llm_provider") or DEFAULT_CONFIG.get("llm_provider", "openai")
        env_var = get_api_key_env(provider)
        auto_selected = False
        if env_var is not None and not os.environ.get(env_var):
            # The requested provider has no key. Auto-select one that does AND
            # has catalog defaults, so we can realign the model too — otherwise
            # we'd keep the requested provider's model name against a different
            # provider (a provider/model mismatch).
            available = _first_configured_provider(require_defaults=True)
            if available:
                logger.info(
                    f"Provider '{provider}' has no key; auto-selecting '{available}'"
                )
                provider = available
                env_var = get_api_key_env(provider)
                auto_selected = True
            else:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"No API key configured for provider '{provider}'. "
                        f"Set the {env_var} environment variable in your Railway "
                        f"service (Variables tab) and redeploy. No other provider "
                        f"with known model defaults has a key configured either."
                    ),
                )

        # Lock in the resolved provider and keep its models in sync. On an
        # auto-select we force the realignment so the inherited model name from
        # the originally requested provider cannot leak through.
        config["llm_provider"] = provider
        _resolve_provider_models(provider, config, force=auto_selected)

        # Reject new work when too many analyses are already in flight, rather
        # than evicting a live run to make room.
        if _running_count() >= MAX_CONCURRENT_ANALYSES:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Too many analyses in progress "
                    f"({MAX_CONCURRENT_ANALYSES} max). Please retry shortly."
                ),
            )

        # Create analysis ID
        analysis_id = f"{ticker}_{int(datetime.now().timestamp())}"

        # Bound the store before inserting a new entry.
        _evict_old_analyses()

        # Store analysis state
        active_analyses[analysis_id] = {
            "id": analysis_id,
            "ticker": ticker,
            "date": date,
            "status": "queued",
            "progress": 0,
            "messages": [],
            "result": None,
            "error": None,
            "started_at": datetime.now().isoformat(),
            "proposed_order": None,
            "order_result": None,
            "confirm_token": None,
        }

        # Run analysis in background
        background_tasks.add_task(run_analysis_task, analysis_id, ticker, date, config, analysts)

        return {"analysis_id": analysis_id, "status": "queued"}

    except HTTPException:
        # Intended client errors (400 validation, 401 auth, 429 backpressure)
        # must propagate with their own status, not be masked as a 500.
        raise
    except Exception as e:
        logger.error(f"Error starting analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analyze/{analysis_id}", dependencies=[Depends(require_api_token)])
async def get_analysis_status(analysis_id: str):
    """Get analysis status and progress."""
    if analysis_id not in active_analyses:
        raise HTTPException(status_code=404, detail="Analysis not found")

    analysis = active_analyses[analysis_id]
    return {
        "id": analysis["id"],
        "ticker": analysis["ticker"],
        "date": analysis["date"],
        "status": analysis["status"],
        "progress": analysis["progress"],
        "message_count": len(analysis["messages"]),
        "result": analysis["result"],
        "error": analysis["error"],
        "started_at": analysis["started_at"],
        "proposed_order": analysis.get("proposed_order"),
        "order_result": analysis.get("order_result"),
        "confirm_token": analysis.get("confirm_token"),
    }


@app.get("/api/analyze/{analysis_id}/messages", dependencies=[Depends(require_api_token)])
async def get_analysis_messages(analysis_id: str, skip: int = 0, limit: int = 50):
    """Get analysis messages."""
    if analysis_id not in active_analyses:
        raise HTTPException(status_code=404, detail="Analysis not found")

    messages = active_analyses[analysis_id]["messages"]
    return {
        "total": len(messages),
        "messages": messages[skip : skip + limit],
    }


@app.websocket("/ws/analyze/{analysis_id}")
async def websocket_analyze(websocket: WebSocket, analysis_id: str):
    """WebSocket endpoint for real-time analysis updates."""
    if analysis_id not in active_analyses:
        await websocket.close(code=4004, reason="Analysis not found")
        return

    await websocket.accept()
    analysis = active_analyses[analysis_id]

    # All sends live inside the try so a client that disconnects immediately
    # after accept (even before the first frame) is handled gracefully.
    try:
        # Send initial state
        await websocket.send_json({
            "type": "status",
            "data": {
                "id": analysis["id"],
                "ticker": analysis["ticker"],
                "date": analysis["date"],
                "status": analysis["status"],
            },
        })

        last_message_count = 0
        while analysis["status"] not in ["completed", "failed"]:
            current_message_count = len(analysis["messages"])

            # Send new messages
            if current_message_count > last_message_count:
                new_messages = analysis["messages"][last_message_count:]
                for message in new_messages:
                    await websocket.send_json({
                        "type": "message",
                        "data": message,
                    })
                last_message_count = current_message_count

            # Send progress update
            await websocket.send_json({
                "type": "progress",
                "data": {"progress": analysis["progress"], "status": analysis["status"]},
            })

            await asyncio.sleep(0.5)

        # Send final result
        await websocket.send_json({
            "type": "complete",
            "data": {
                "status": analysis["status"],
                "result": analysis["result"],
                "error": analysis["error"],
                "proposed_order": analysis.get("proposed_order"),
                "order_result": analysis.get("order_result"),
                "confirm_token": analysis.get("confirm_token"),
            },
        })

    except WebSocketDisconnect:
        # Client navigated away or closed the tab — expected, not an error.
        logger.info(f"WebSocket client disconnected from {analysis_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Closing an already-disconnected socket raises; ignore it.
        try:
            await websocket.close()
        except RuntimeError:
            pass


# ============================================================================
# Background Tasks
# ============================================================================


def run_analysis_task(
    analysis_id: str,
    ticker: str,
    date: str,
    config: Dict[str, Any],
    analysts: list,
):
    """Run the trading agents analysis in background."""
    # Import here to avoid loading heavy dependencies at startup
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    analysis = active_analyses[analysis_id]

    try:
        analysis["status"] = "running"
        analysis["progress"] = 10

        # Merge config with defaults
        merged_config = DEFAULT_CONFIG.copy()
        merged_config.update(config)

        # Log start
        add_message(analysis_id, "info", f"Starting analysis for {ticker} on {date}")
        add_message(
            analysis_id,
            "info",
            f"Provider: {merged_config.get('llm_provider')} | "
            f"deep: {merged_config.get('deep_think_llm')} | "
            f"quick: {merged_config.get('quick_think_llm')}",
        )
        analysis["progress"] = 20

        # Initialize graph with a callback that streams real-time agent
        # activity (LLM reasoning, tool/data fetches) into the message feed.
        add_message(analysis_id, "info", f"Selected analysts: {', '.join(analysts)}")
        progress_callback = _make_progress_callback(analysis_id)
        ta = TradingAgentsGraph(
            selected_analysts=analysts,
            debug=True,
            config=merged_config,
            callbacks=[progress_callback],
        )
        analysis["progress"] = 30

        # Run analysis
        add_message(analysis_id, "info", f"Analyzing {ticker}...")
        event_stream, decision = ta.propagate(ticker, date)
        analysis["progress"] = 90

        # Store result - ensure JSON serializable
        rating = str(decision) if decision else None
        side = rating_to_side(rating) if rating else None
        analysis["result"] = {
            "ticker": ticker,
            "date": date,
            "decision": rating,
            "rating": rating,
            "action": side or "hold",
            "summary": "Analysis completed successfully",
        }
        # Propose an order only when the rating is directional and a broker
        # is configured. Execution still requires explicit user confirmation.
        broker = AlpacaBroker()
        if side and broker.is_configured():
            analysis["proposed_order"] = {
                "symbol": ticker,
                "side": side,
                "rating": rating,
                "mode": broker.mode,
            }
            # Confirmation token: execution must echo this back, so a stray or
            # forged POST that hasn't seen the proposed order cannot trigger a
            # trade. Surfaced only to authenticated status readers.
            analysis["confirm_token"] = secrets.token_urlsafe(32)
            add_message(
                analysis_id,
                "info",
                f"Proposed order: {side.upper()} {ticker} ({rating}) — "
                f"awaiting your confirmation [{broker.mode}].",
            )
        analysis["progress"] = 100
        analysis["status"] = "completed"

        add_message(analysis_id, "success", f"Analysis completed for {ticker}")

    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        analysis["status"] = "failed"
        friendly = _humanize_error(e)
        analysis["error"] = friendly
        add_message(analysis_id, "error", f"Analysis failed: {friendly}")


def _humanize_error(error: Exception) -> str:
    """Turn opaque SDK errors into actionable guidance for the UI."""
    text = str(error)
    lowered = text.lower()

    if "429" in text or "rate_limit" in lowered or "rate limit" in lowered:
        return (
            "Rate limit hit on your LLM provider. The multi-agent analysis "
            "makes many large-prompt calls, which can exceed low per-minute "
            "token limits (e.g. Anthropic Tier-1 is 10,000 input tokens/min). "
            "Try: (1) select fewer analysts and 1 debate round, (2) switch to "
            "a provider/key with a higher limit, or (3) raise your provider's "
            "rate limit / usage tier, then re-run."
        )
    if "authentication" in lowered or "401" in text or "api key" in lowered:
        return (
            "Authentication failed for the LLM provider. Verify the API key "
            "environment variable is set correctly in Railway and redeploy."
        )
    return text


def add_message(analysis_id: str, level: str, content: str):
    """Add a message to an analysis."""
    if analysis_id in active_analyses:
        analysis = active_analyses[analysis_id]
        analysis["messages"].append({
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "content": content,
        })


# ============================================================================
# Static Files & Health Check
# ============================================================================


# ============================================================================
# Broker (Alpaca) — confirmed order execution
# ============================================================================


@app.get("/api/broker/status", dependencies=[Depends(require_api_token)])
async def broker_status():
    """Report whether Alpaca is configured and, if so, the account snapshot."""
    broker = AlpacaBroker()
    if not broker.is_configured():
        return {"configured": False, "mode": None, "account": None}
    try:
        account = await asyncio.to_thread(broker.get_account)
        return {"configured": True, "mode": broker.mode, "account": account}
    except Exception as e:
        logger.warning(f"Broker status error: {e}")
        return {"configured": True, "mode": broker.mode, "account": None, "error": str(e)}


@app.post("/api/analyze/{analysis_id}/execute", dependencies=[Depends(require_api_token)])
async def execute_trade(analysis_id: str, body: Dict[str, Any]):
    """Place the proposed order for a completed analysis (user-confirmed).

    Body: {"confirm_token": <token>, "notional": <usd>} or
          {"confirm_token": <token>, "qty": <shares>} — exactly one of
    notional/qty. The confirm_token is issued with the proposed order (see the
    status / websocket-complete payloads) and must be echoed back so a request
    that never saw the proposal cannot place a real order.
    """
    if analysis_id not in active_analyses:
        raise HTTPException(status_code=404, detail="Analysis not found")

    analysis = active_analyses[analysis_id]
    proposed = analysis.get("proposed_order")
    if not proposed:
        raise HTTPException(
            status_code=400,
            detail="No proposed order for this analysis (Hold, or broker not configured).",
        )
    if analysis.get("order_result"):
        raise HTTPException(status_code=409, detail="Order already executed for this analysis.")

    expected_token = analysis.get("confirm_token")
    presented_token = (body.get("confirm_token") or "").strip() if isinstance(body, dict) else ""
    if not expected_token or not presented_token or not secrets.compare_digest(
        presented_token, expected_token
    ):
        raise HTTPException(
            status_code=403,
            detail="Missing or invalid confirmation token for this order.",
        )

    broker = AlpacaBroker()
    if not broker.is_configured():
        raise HTTPException(status_code=400, detail="Alpaca is not configured on the server.")

    notional = body.get("notional")
    qty = body.get("qty")
    if not notional and not qty:
        notional = 100.0  # sensible default dollar amount
    if notional and qty:
        raise HTTPException(status_code=400, detail="Provide only one of notional or qty.")

    try:
        order = await asyncio.to_thread(
            broker.place_order,
            proposed["symbol"],
            proposed["side"],
            float(notional) if notional else None,
            float(qty) if qty else None,
        )
    except Exception as e:
        logger.error(f"Order execution failed: {e}")
        add_message(analysis_id, "error", f"Order failed: {e}")
        raise HTTPException(status_code=502, detail=str(e))

    analysis["order_result"] = order
    add_message(
        analysis_id,
        "success",
        f"Order placed [{broker.mode}]: {order['side'].upper()} {order['symbol']} "
        f"(status: {order['status']}, id: {order['id']}).",
    )
    return {"mode": broker.mode, "order": order}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "version": "0.2.5"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
