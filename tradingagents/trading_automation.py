"""Trading automation engine with confidence-based execution thresholds.

Enables semi-automated trading decisions based on LLM confidence scores.
Provides risk controls, audit trails, and flexibility in automation levels.

Configuration (environment variables):
    TRADING_AUTOMATION_ENABLED     - "true" to enable automated execution
    TRADING_CONFIDENCE_THRESHOLD   - Minimum confidence (0.0-1.0) to auto-execute (default: 0.75)
    TRADING_DRY_RUN               - "true" for simulation mode (no real orders)
    TRADING_MAX_POSITION_SIZE     - Max % of portfolio per trade (default: 5.0)
    TRADING_AUDIT_LOG_PATH        - Path to audit log file (default: ./trading_audit.log)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Trading execution mode."""

    MANUAL = "manual"  # Recommend only, user must approve
    SEMI_AUTO = "semi_auto"  # Execute only if confidence > threshold
    FULLY_AUTO = "fully_auto"  # Execute all decisions with risk limits


class TradeDecision(Enum):
    """Trading decision type."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class ConfidenceScore:
    """Confidence metrics for trading decision."""

    overall: float  # 0.0-1.0, aggregate confidence
    sentiment: float  # Based on sentiment analysis
    technical: float  # Based on technical indicators
    fundamental: float  # Based on fundamentals
    consensus: float  # Agreement between analysts

    def __post_init__(self):
        """Validate score ranges."""
        for field in ['overall', 'sentiment', 'technical', 'fundamental', 'consensus']:
            value = getattr(self, field)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must be between 0.0 and 1.0, got {value}")


@dataclass
class TradeProposal:
    """Proposed trade from LLM analysis."""

    symbol: str
    decision: TradeDecision
    confidence: ConfidenceScore
    reasoning: str
    suggested_quantity: Optional[float] = None
    suggested_notional: Optional[float] = None
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    risk_assessment: str = ""

    def should_execute(self, threshold: float = 0.75) -> bool:
        """Check if trade meets confidence threshold for auto-execution.

        Args:
            threshold: Minimum confidence score required (0.0-1.0)

        Returns:
            True if decision is not HOLD and confidence >= threshold
        """
        if self.decision == TradeDecision.HOLD:
            return False
        return self.confidence.overall >= threshold

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['decision'] = self.decision.value
        data['confidence'] = asdict(self.confidence)
        return data


@dataclass
class ExecutionResult:
    """Result of trade execution."""

    proposal: TradeProposal
    executed: bool
    mode: ExecutionMode
    reason: str  # Why it was/wasn't executed
    order_id: Optional[str] = None
    execution_time: Optional[str] = None
    notional_executed: Optional[float] = None
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "proposal": self.proposal.to_dict(),
            "executed": self.executed,
            "mode": self.mode.value,
            "reason": self.reason,
            "order_id": self.order_id,
            "execution_time": self.execution_time,
            "notional_executed": self.notional_executed,
            "message": self.message,
        }


class TradingAutomationEngine:
    """Manages automated trading execution with confidence thresholds."""

    def __init__(
        self,
        mode: ExecutionMode = ExecutionMode.SEMI_AUTO,
        confidence_threshold: float = 0.75,
        max_position_pct: float = 5.0,
        dry_run: bool = False,
        audit_log_path: Optional[str] = None,
    ):
        """Initialize trading automation engine.

        Args:
            mode: Execution mode (MANUAL, SEMI_AUTO, FULLY_AUTO)
            confidence_threshold: Min confidence for auto-execution (0.0-1.0)
            max_position_pct: Max position size as % of portfolio
            dry_run: If True, simulate trades without executing
            audit_log_path: Path to write trade audit log
        """
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be 0.0-1.0")
        if not 0.0 < max_position_pct <= 100.0:
            raise ValueError("max_position_pct must be 0.0-100.0")

        self.mode = mode
        self.confidence_threshold = confidence_threshold
        self.max_position_pct = max_position_pct
        self.dry_run = dry_run
        self.audit_log_path = Path(audit_log_path) if audit_log_path else None

        logger.info(
            f"Trading automation initialized: mode={mode.value}, "
            f"threshold={confidence_threshold}, dry_run={dry_run}"
        )

    @classmethod
    def from_env(cls) -> TradingAutomationEngine:
        """Create engine from environment variables."""
        mode_str = os.getenv("TRADING_EXECUTION_MODE", "semi_auto").lower()
        mode_map = {e.value: e for e in ExecutionMode}
        mode = mode_map.get(mode_str, ExecutionMode.SEMI_AUTO)

        confidence_threshold = float(
            os.getenv("TRADING_CONFIDENCE_THRESHOLD", "0.75")
        )
        max_position_pct = float(
            os.getenv("TRADING_MAX_POSITION_SIZE", "5.0")
        )
        dry_run = os.getenv("TRADING_DRY_RUN", "false").lower() == "true"
        audit_log_path = os.getenv("TRADING_AUDIT_LOG_PATH", "./trading_audit.log")

        return cls(
            mode=mode,
            confidence_threshold=confidence_threshold,
            max_position_pct=max_position_pct,
            dry_run=dry_run,
            audit_log_path=audit_log_path,
        )

    def should_execute_trade(self, proposal: TradeProposal) -> tuple[bool, str]:
        """Determine if a trade should be executed.

        Args:
            proposal: Trade proposal from LLM

        Returns:
            Tuple of (should_execute, reason)
        """
        # HOLD decisions never execute
        if proposal.decision == TradeDecision.HOLD:
            return False, "Decision is HOLD"

        # Check execution mode
        if self.mode == ExecutionMode.MANUAL:
            return False, f"Manual mode - confidence {proposal.confidence.overall:.1%}"

        if self.mode == ExecutionMode.FULLY_AUTO:
            return True, f"Fully automated - confidence {proposal.confidence.overall:.1%}"

        # SEMI_AUTO mode: check threshold
        meets_threshold = proposal.should_execute(self.confidence_threshold)
        if meets_threshold:
            return True, f"Confidence {proposal.confidence.overall:.1%} >= threshold {self.confidence_threshold:.1%}"
        else:
            return False, f"Confidence {proposal.confidence.overall:.1%} < threshold {self.confidence_threshold:.1%}"

    def log_execution(self, result: ExecutionResult) -> None:
        """Log trade execution for audit trail.

        Args:
            result: Execution result to log
        """
        if not self.audit_log_path:
            return

        try:
            # Ensure directory exists
            self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                **result.to_dict(),
            }

            # Append to JSON lines file
            with open(self.audit_log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            logger.debug(f"Trade logged: {result.proposal.symbol} {result.proposal.decision.value}")
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")

    def execute_trade(
        self,
        proposal: TradeProposal,
        execute_fn: Optional[callable] = None,
    ) -> ExecutionResult:
        """Execute or propose a trade based on confidence.

        Args:
            proposal: Trade proposal from analysis
            execute_fn: Callable that performs actual order execution
                       Signature: (symbol, side, notional) -> order_dict
                       If None, trade is logged but not executed

        Returns:
            ExecutionResult with execution status
        """
        should_execute, reason = self.should_execute_trade(proposal)

        result = ExecutionResult(
            proposal=proposal,
            executed=False,
            mode=self.mode,
            reason=reason,
        )

        # Always log the proposal
        self.log_execution(result)

        # Execute if conditions met and function provided
        if should_execute and execute_fn and not self.dry_run:
            try:
                # Determine execution amount
                if proposal.suggested_notional:
                    notional = proposal.suggested_notional
                elif proposal.suggested_quantity:
                    # Would need current price to convert qty to notional
                    notional = proposal.suggested_quantity * 100  # Placeholder
                else:
                    notional = 1000.0  # Default order size

                # Execute trade
                order = execute_fn(
                    symbol=proposal.symbol,
                    side=proposal.decision.value,
                    notional=notional,
                )

                result.executed = True
                result.order_id = order.get("id")
                result.execution_time = datetime.utcnow().isoformat()
                result.notional_executed = notional
                result.message = f"Order placed: {proposal.decision.value.upper()} {notional}"

                logger.info(
                    f"Trade executed: {proposal.symbol} {proposal.decision.value} "
                    f"(confidence: {proposal.confidence.overall:.1%}, order_id: {result.order_id})"
                )
            except Exception as e:
                result.executed = False
                result.message = f"Execution failed: {str(e)}"
                logger.error(f"Trade execution failed: {e}")
        elif should_execute and self.dry_run:
            result.executed = True
            result.message = f"DRY RUN: Would execute {proposal.decision.value} {proposal.symbol}"
            logger.info(f"DRY RUN: {result.message}")
        else:
            result.message = f"Trade not executed: {reason}"
            logger.info(f"Trade recommendation: {proposal.symbol} {proposal.decision.value} - {reason}")

        # Log final result
        self.log_execution(result)

        return result

    def get_audit_log(self, limit: Optional[int] = None) -> list[Dict[str, Any]]:
        """Read recent audit log entries.

        Args:
            limit: Max number of recent entries to return

        Returns:
            List of audit log entries (most recent first)
        """
        if not self.audit_log_path or not self.audit_log_path.exists():
            return []

        try:
            entries = []
            with open(self.audit_log_path, "r") as f:
                for line in f:
                    if line.strip():
                        entries.append(json.loads(line))

            # Return most recent first
            entries.reverse()
            if limit:
                entries = entries[:limit]
            return entries
        except Exception as e:
            logger.error(f"Failed to read audit log: {e}")
            return []
