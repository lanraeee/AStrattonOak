# QWEN Cloud Integration Guide

StrattonOak now supports **Alibaba QWEN Cloud** (DashScope) for enhanced trading analysis and semi-automated execution. This guide explains how to set up and use QWEN for your trading strategies.

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Obtaining QWEN API Keys](#obtaining-qwen-api-keys)
  - [Environment Configuration](#environment-configuration)
  - [Testing Connection](#testing-connection)
- [QWEN Models](#qwen-models)
- [Semi-Automated Trading](#semi-automated-trading)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## Overview

QWEN Cloud provides:

- **Advanced LLM Models**: qwen-3.7-max, qwen-3.6-plus, qwen-3.6-flash
- **Fast Inference**: Lower latency than many alternatives
- **Multi-Region Support**: International (dashscope-intl) and China (dashscope) endpoints
- **Cost-Effective**: Competitive pricing for trading analysis at scale
- **Agent-Optimized**: Strong tool use and function calling capabilities

### Key Features for Trading

| Feature | QWEN 3.7 Max | QWEN 3.6 Plus | QWEN 3.6 Flash |
|---------|-------------|---------------|----------------|
| Deep Reasoning | ✅ Best | ✅ Good | ⚠️ Basic |
| Speed | ⚠️ Slow (Deep) | ✅ Balanced | ✅ Very Fast |
| Context Length | 1M tokens | 200K tokens | 200K tokens |
| Agent Capabilities | ✅ Excellent | ✅ Excellent | ✅ Good |
| Cost | $$$ | $$ | $ |
| Ideal For | Complex Analysis | Detailed Reports | Quick Decisions |

## Getting Started

### Obtaining QWEN API Keys

1. **Sign up for Alibaba Cloud DashScope**:
   - International: https://dashscope.console.aliyun.com/
   - China: https://dashscope.console.aliyun.com/

2. **Create an API Key**:
   - Navigate to API Key Management
   - Click "Create API Key"
   - Copy your API key (keep it secure!)

3. **Note the Endpoint**:
   - **International**: `https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/text-generation/generation`
   - **China**: `https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation`

### Environment Configuration

#### Option 1: Quick Start (Use QWEN as Default)

1. **Copy environment template**:
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` and add your QWEN API key**:
   ```bash
   # For international endpoint:
   QWEN_API_KEY=your_api_key_here

   # Or for China endpoint:
   QWEN_CN_API_KEY=your_api_key_here
   ```

3. **Configure QWEN as default provider**:
   ```bash
   # Add to .env:
   TRADINGAGENTS_LLM_PROVIDER=qwen
   TRADINGAGENTS_DEEP_THINK_LLM=qwen3.7-max
   TRADINGAGENTS_QUICK_THINK_LLM=qwen3.6-flash
   ```

4. **Verify in web UI**:
   - Start the app: `python main.py`
   - Go to http://localhost:8000
   - In "LLM Provider" dropdown, select "QWEN (International)" or "QWEN (China)"
   - Models should appear automatically

#### Option 2: Use Specific QWEN Models with Other Providers

Keep your current provider but use QWEN for specific tasks:

```bash
# Still use OpenAI as default
TRADINGAGENTS_LLM_PROVIDER=openai
TRADINGAGENTS_DEEP_THINK_LLM=gpt-5.5

# But use QWEN for quick decisions (if you have QWEN_API_KEY set)
# (This requires code-level configuration - see Advanced Usage below)
```

### Testing Connection

#### Via Web UI

1. Open the app (http://localhost:8000)
2. Go to the **Alpaca** tab
3. Click **Test Connection** button
4. The Debug Info will show QWEN connection status

#### Via CLI

```bash
# Test QWEN connection with debug output
python -c "
from tradingagents.llm_clients.factory import create_llm_client
client = create_llm_client('qwen', 'qwen3.7-max')
llm = client.get_llm()
response = llm.invoke('Say hello')
print('✓ QWEN connection successful!')
print(response)
"
```

## QWEN Models

### Model Selection Guide

#### For Deep Analysis (Multi-Step Reasoning)

Use **qwen-3.7-max**:
- Most capable QWEN model
- Excellent agent/tool use
- 1M context window
- Best for: Fundamental analysis, complex scenarios, debate rounds
- Cost: Medium-High
- Speed: Slow (can take 5-30 seconds)

```bash
TRADINGAGENTS_DEEP_THINK_LLM=qwen3.7-max
```

#### For Detailed Analysis (Balanced)

Use **qwen-3.6-plus**:
- Excellent all-around model
- Strong reasoning and agent capabilities
- Good speed/accuracy balance
- Best for: News analysis, sentiment assessment
- Cost: Medium
- Speed: Moderate (2-10 seconds)

```bash
TRADINGAGENTS_DEEP_THINK_LLM=qwen3.6-plus
```

#### For Quick Decisions (Speed)

Use **qwen-3.6-flash** or **qwen-3.5-flash**:
- Optimized for speed
- Still excellent agent capabilities
- Best for: Real-time market signals, quick decisions
- Cost: Low
- Speed: Very fast (< 2 seconds)

```bash
TRADINGAGENTS_QUICK_THINK_LLM=qwen3.6-flash
```

### Example Configurations

**Aggressive (all qwen-3.7-max)**:
```bash
TRADINGAGENTS_DEEP_THINK_LLM=qwen3.7-max
TRADINGAGENTS_QUICK_THINK_LLM=qwen3.7-max
# Best accuracy, high cost, slow
```

**Balanced (recommended)**:
```bash
TRADINGAGENTS_DEEP_THINK_LLM=qwen3.7-max
TRADINGAGENTS_QUICK_THINK_LLM=qwen3.6-flash
# Accurate analysis, fast decisions, medium cost
```

**Cost-Optimized**:
```bash
TRADINGAGENTS_DEEP_THINK_LLM=qwen3.6-plus
TRADINGAGENTS_QUICK_THINK_LLM=qwen3.6-flash
# All-flash setup, lower cost, very fast
```

**Speed-Optimized**:
```bash
TRADINGAGENTS_DEEP_THINK_LLM=qwen3.6-flash
TRADINGAGENTS_QUICK_THINK_LLM=qwen3.6-flash
# Fastest execution, minimum cost, less analysis depth
```

## Semi-Automated Trading

StrattonOak now supports **semi-automated trading** with QWEN - trades execute automatically when confidence exceeds a threshold.

### How It Works

1. **QWEN analyzes the stock** using all available data
2. **Generates confidence score** (0-100%) based on:
   - Sentiment analysis
   - Technical indicators
   - Fundamental data
   - Analyst consensus
3. **Compares to threshold** (default: 75%)
4. **Auto-executes if confident**, otherwise recommends to user
5. **Logs all decisions** with reasoning for audit trail

### Confidence Scoring

The system rates confidence in three dimensions:

- **Sentiment** (0-100%): News, social media, market sentiment
- **Technical** (0-100%): Chart patterns, indicators, signals
- **Fundamental** (0-100%): Financial metrics, growth, valuation
- **Overall** (0-100%): Weighted aggregate of all signals

Example:
```
Ticker: AAPL
Decision: BUY
├─ Sentiment: 82% (positive news, strong sentiment)
├─ Technical: 78% (breakout pattern, volume surge)
├─ Fundamental: 70% (growing revenue, reasonable PE)
└─ Overall: 76% (above 75% threshold → AUTO-EXECUTE)
```

### Configuration

**Enable Semi-Automated Trading**:

```bash
# .env file
TRADING_EXECUTION_MODE=semi_auto
TRADING_CONFIDENCE_THRESHOLD=0.75
TRADING_MAX_POSITION_SIZE=5.0
TRADING_DRY_RUN=false
```

**Execution Modes**:

| Mode | Auto-Execute | Use Case |
|------|--------------|----------|
| `manual` | Never | Learning, testing, full control |
| `semi_auto` | If confidence > threshold | **Recommended: Balanced automation** |
| `fully_auto` | Always (with risk limits) | Algorithmic trading, high frequency |

**Confidence Threshold**:

- **0.50** (50%): Aggressive - execute most recommendations
- **0.65** (65%): Moderate - execute most good signals
- **0.75** (75%): **Recommended** - balanced accuracy/execution
- **0.85** (85%): Conservative - only execute high-confidence
- **0.95** (95%): Strict - only execute very certain decisions

**Position Sizing**:

```bash
# Max 5% of portfolio per trade (prevents concentration)
TRADING_MAX_POSITION_SIZE=5.0

# Examples with $100k portfolio:
# Max position: $5,000
# Prevents: Doubling up on single stocks
# Enforces: Diversification
```

### Dry-Run Mode (Testing)

Test automation logic without real trades:

```bash
TRADING_DRY_RUN=true
```

- Simulates all trades
- Shows what would execute
- No real money at risk
- Perfect for backtesting

### Audit Trail

All trades logged to `trading_audit.log`:

```json
{
  "timestamp": "2026-06-17T14:32:15.123456",
  "proposal": {
    "symbol": "AAPL",
    "decision": "buy",
    "confidence": {
      "overall": 0.76,
      "sentiment": 0.82,
      "technical": 0.78,
      "fundamental": 0.70,
      "consensus": 0.75
    }
  },
  "executed": true,
  "mode": "semi_auto",
  "reason": "Confidence 76% >= threshold 75%",
  "order_id": "12345abc",
  "notional_executed": 5000.0
}
```

## Configuration

### Complete Configuration Example

```bash
# ===== LLM SETUP =====
QWEN_API_KEY=sk-...your-key...

# Use QWEN as default provider
TRADINGAGENTS_LLM_PROVIDER=qwen

# Deep thinking model (for complex analysis)
TRADINGAGENTS_DEEP_THINK_LLM=qwen3.7-max

# Quick thinking model (for fast decisions)
TRADINGAGENTS_QUICK_THINK_LLM=qwen3.6-flash

# ===== ANALYSIS PARAMETERS =====
# Debate rounds between bull and bear researchers
TRADINGAGENTS_MAX_DEBATE_ROUNDS=2

# Risk discussion rounds
TRADINGAGENTS_MAX_RISK_ROUNDS=1

# Temperature (0.0 = deterministic, 1.0 = creative)
TRADINGAGENTS_TEMPERATURE=0.7

# ===== TRADING AUTOMATION =====
# Semi-automated mode (RECOMMENDED)
TRADING_EXECUTION_MODE=semi_auto

# Confidence threshold for auto-execution
TRADING_CONFIDENCE_THRESHOLD=0.75

# Max position per trade (% of portfolio)
TRADING_MAX_POSITION_SIZE=5.0

# ===== BROKER SETUP =====
# Alpaca for order execution
ALPACA_API_KEY=your-key
ALPACA_SECRET_KEY=your-secret

# Use paper trading for testing
ALPACA_PAPER=true

# ===== AUDIT & LOGGING =====
TRADING_AUDIT_LOG_PATH=./trading_audit.log

# ===== WEB APP =====
PORT=8000
```

## Troubleshooting

### QWEN Not Showing in Provider List

**Problem**: Dropdown shows only other providers

**Solutions**:
1. Check `QWEN_API_KEY` is set and non-empty
2. Make sure it's spelled correctly: `QWEN_API_KEY` (not `QWEN_KEY`)
3. Restart the application after adding key
4. Check web server logs: `python main.py 2>&1 | grep -i qwen`

### "API key not found" Error

**Problem**: `ValueError: QWEN API key not found`

**Solutions**:
```bash
# Verify key is in environment
echo $QWEN_API_KEY

# Or for China endpoint
echo $QWEN_CN_API_KEY

# If empty, it wasn't loaded from .env
# Check: .env syntax, file location, export statements
```

### Slow Response Times

**Problem**: Analysis takes 30+ seconds

**Solution**: Use faster models
```bash
# Instead of qwen3.7-max for quick decisions
TRADINGAGENTS_QUICK_THINK_LLM=qwen3.6-flash

# Or reduce debate rounds
TRADINGAGENTS_MAX_DEBATE_ROUNDS=1
```

### High API Costs

**Problem**: Usage costs growing

**Solutions**:
1. Use faster, cheaper models for quick decisions
   ```bash
   TRADINGAGENTS_QUICK_THINK_LLM=qwen3.6-flash
   ```

2. Reduce analysis depth
   ```bash
   TRADINGAGENTS_MAX_DEBATE_ROUNDS=1
   ```

3. Reduce analysis frequency
   - Run daily instead of hourly
   - Analyze only when signals change

4. Monitor usage in DashScope console
   - Token counts visible in API dashboard
   - Set API quotas to prevent overage

### China Region Connectivity

**Problem**: Using China region but getting timeouts

**Solutions**:
1. Verify network can reach China endpoints:
   ```bash
   curl https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation
   ```

2. Use international endpoint instead:
   ```bash
   QWEN_API_KEY=your-international-key
   # (instead of QWEN_CN_API_KEY)
   ```

3. Check firewall/proxy settings if in mainland China

## Best Practices

### 1. Start Conservative

```bash
# Week 1: Testing
TRADING_EXECUTION_MODE=manual
TRADING_DRY_RUN=true

# Week 2: High threshold
TRADING_CONFIDENCE_THRESHOLD=0.90

# Week 3: Standard threshold
TRADING_CONFIDENCE_THRESHOLD=0.75
```

### 2. Monitor Confidence Scores

Track what confidence scores lead to profits:
- If 75%+ confidence = profits → your threshold is good
- If many 75-80% lose money → raise threshold
- If missing opportunities below threshold → lower threshold

### 3. Use Diverse Analysts

Enable all analyst types in web UI:
- ✅ Market (technical)
- ✅ News (sentiment)
- ✅ Fundamentals (financial)
- ✅ Sentiment (social)

More analysts = better confidence scoring

### 4. Review Audit Logs Weekly

```bash
# See what executed and why
tail -20 trading_audit.log | jq '.'
```

Check:
- Are confidence scores accurate?
- Are profitable trades being missed?
- Are losing trades being prevented?

### 5. Paper Trading First

Always test strategy on paper:
```bash
ALPACA_PAPER=true
TRADING_DRY_RUN=false  # Real simulation
```

Trade on paper for 2-4 weeks before going live.

## Support & Feedback

- **DashScope Docs**: https://dashscope.console.aliyun.com/
- **API Documentation**: https://help.aliyun.com/document_detail/213592.html
- **StrattonOak Issues**: https://github.com/fawazzzbello/StrattonOak/issues

---

**Last Updated**: June 2026  
**QWEN Models Supported**: 3.7-max, 3.6-plus, 3.6-flash, 3.5-plus, 3.5-flash
