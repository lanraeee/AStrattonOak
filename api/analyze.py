"""
Trading analysis API endpoint.
Routes analysis requests to the TradingAgentsGraph engine.
"""

from http.server import BaseHTTPRequestHandler
import json
import os
from datetime import datetime

# Import the trading framework
try:
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.default_config import DEFAULT_CONFIG
    FRAMEWORK_AVAILABLE = True
except ImportError:
    FRAMEWORK_AVAILABLE = False


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        """Handle analysis requests"""
        if self.path != "/api/analyze":
            self.send_error(404)
            return

        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            request = json.loads(body)

            # Validate input
            ticker = request.get('ticker', '').upper()
            date = request.get('date', '')
            if not ticker or not date:
                return self.send_json(
                    {"error": "Missing ticker or date"},
                    status=400
                )

            if not FRAMEWORK_AVAILABLE:
                return self.send_json(
                    {
                        "error": "Trading framework not available",
                        "message": "Install TradingAgents with: pip install -e .",
                    },
                    status=503,
                )

            # Run analysis
            analysis = self._run_analysis(
                ticker=ticker,
                date=date,
                provider=request.get('provider', 'anthropic'),
                analysts=request.get('analysts', ['market', 'news']),
            )

            self.send_json(analysis, status=200)

        except json.JSONDecodeError:
            self.send_json({"error": "Invalid JSON"}, status=400)
        except Exception as e:
            self.send_json(
                {
                    "error": str(e),
                    "type": type(e).__name__,
                },
                status=500,
            )

    def _run_analysis(self, ticker: str, date: str, provider: str, analysts: list) -> dict:
        """Run trading analysis using the TradingAgentsGraph"""
        # Initialize with config from environment
        config = DEFAULT_CONFIG.copy()
        config['llm_provider'] = provider

        # Create graph
        ta = TradingAgentsGraph(debug=True, config=config)

        # Parse date (YYYY-MM-DD format)
        try:
            from datetime import datetime
            analysis_date = datetime.strptime(date, '%Y-%m-%d').strftime('%Y-%m-%d')
        except ValueError:
            analysis_date = date

        # Run analysis
        decision = ta.propagate(ticker, analysis_date)

        # Format response
        return {
            "ticker": ticker,
            "date": analysis_date,
            "timestamp": datetime.utcnow().isoformat(),
            "recommendation": self._extract_recommendation(decision),
            "analysis": decision,
            "analysts": {
                "market": {"summary": "Technical analysis results"},
                "news": {"summary": "News impact assessment"},
                "sentiment": {"summary": "Market sentiment analysis"},
                "fundamentals": {"summary": "Financial analysis"},
            },
        }

    def _extract_recommendation(self, decision: dict) -> str:
        """Extract BUY/SELL/HOLD recommendation from decision"""
        if not decision:
            return "HOLD"

        decision_str = str(decision).lower()
        if "buy" in decision_str:
            return "BUY"
        elif "sell" in decision_str:
            return "SELL"
        else:
            return "HOLD"

    def send_json(self, data: dict, status: int = 200):
        """Send JSON response"""
        body = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
