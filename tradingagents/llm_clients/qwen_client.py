"""QWEN Cloud LLM client for StrattonOak trading analysis.

Supports both international (qwen) and China-region (qwen-cn) endpoints.
Integrates Alibaba Cloud QWEN models for financial analysis and trading automation.

Configuration (environment variables):
    QWEN_API_KEY         - QWEN API key (for international endpoint)
    QWEN_CN_API_KEY      - QWEN API key (for China endpoint, if using qwen-cn)
    QWEN_BASE_URL        - Optional custom endpoint (for dashscope proxy, etc.)
    QWEN_CN_BASE_URL     - Optional custom China endpoint

Supported Models:
    - qwen-3.7-max       - Premium, best accuracy for complex analysis
    - qwen-3.6-plus      - Balanced speed/accuracy for detailed analysis
    - qwen-3.6-flash     - Fast inference for quick decisions
    - qwen-3.5-plus      - Previous version, still supported
    - qwen-3.5-flash     - Quick previous-gen option
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from langchain_community.chat_models.qwen import ChatQwen

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model


class QwenClient(BaseLLMClient):
    """QWEN Cloud LLM client supporting international and China regions."""

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        is_china_region: bool = False,
        **kwargs: Any,
    ):
        """Initialize QWEN client.

        Args:
            model: QWEN model ID (e.g., 'qwen-3.7-max')
            base_url: Optional custom endpoint URL
            is_china_region: If True, uses China region configuration
            **kwargs: Additional kwargs passed to ChatQwen
        """
        super().__init__(model, base_url, **kwargs)
        self.is_china_region = is_china_region

    def get_llm(self) -> ChatQwen:
        """Get configured QWEN LLM instance."""
        api_key_env = "QWEN_CN_API_KEY" if self.is_china_region else "QWEN_API_KEY"
        api_key = os.getenv(api_key_env, "").strip()

        if not api_key:
            raise ValueError(
                f"QWEN API key not found. Set {api_key_env} environment variable."
            )

        llm_kwargs: Dict[str, Any] = {
            "model": self.model,
            "api_key": api_key,
        }

        # Use custom base_url if provided, otherwise use default region endpoint
        if self.base_url:
            llm_kwargs["base_url"] = self.base_url
        elif self.is_china_region:
            # Default China region endpoint
            custom_url = os.getenv("QWEN_CN_BASE_URL")
            if custom_url:
                llm_kwargs["base_url"] = custom_url

        # Add optional parameters from kwargs
        if "temperature" in self.kwargs:
            llm_kwargs["temperature"] = self.kwargs["temperature"]

        # QWEN supports top_p parameter for diversity control
        if "top_p" in self.kwargs:
            llm_kwargs["top_p"] = self.kwargs["top_p"]

        return ChatQwen(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate QWEN model name."""
        # Support both provider names: 'qwen' and 'qwen-cn'
        # Model names work the same for both regions
        provider = "qwen"  # Canonical provider name
        return validate_model(provider, self.model)

    def normalize_response(self, response: Any) -> str:
        """Normalize QWEN response to plain text.

        QWEN returns standard LangChain message objects, so use base normalization.
        """
        return normalize_content(response)
