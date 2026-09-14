"""URL Publisher: posts active Cloudflare tunnel URL and shared secret to a registry webhook/KV store.

Note: Do not confuse with voices/registry.py.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Final, Optional
import httpx

logger = logging.getLogger(__name__)

ENV_REGISTRY_WEBHOOK_URL: Final[str] = "TUNNEL_REGISTRY_WEBHOOK_URL"
DEFAULT_PUBLISH_TIMEOUT: Final[float] = 10.0


class URLPublisherError(RuntimeError):
    """Raised when publishing tunnel URL to the webhook/KV store fails."""

    pass


class URLPublisher:
    """Client for publishing tunnel URL and shared secret to a remote coordinator."""

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        timeout_seconds: float = DEFAULT_PUBLISH_TIMEOUT,
        client: Optional[httpx.Client] = None,
    ) -> None:
        """Initialize URL Publisher.

        Args:
            endpoint_url: Webhook or KV endpoint URL. If None, reads from TUNNEL_REGISTRY_WEBHOOK_URL.
            timeout_seconds: HTTP request timeout in seconds.
            client: Optional injected httpx.Client for testing.
        """
        self.endpoint_url = endpoint_url
        self.timeout_seconds = float(timeout_seconds)
        self._client = client

    def publish(
        self,
        tunnel_url: str,
        secret: str,
        endpoint_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Post active tunnel URL and secret to registry endpoint.

        Args:
            tunnel_url: Public trycloudflare URL (https://*.trycloudflare.com).
            secret: Shared secret / bearer token.
            endpoint_url: Optional override destination URL.
            metadata: Optional extra metadata dictionary.

        Returns:
            Parsed JSON response dictionary from webhook endpoint.

        Raises:
            ValueError: If tunnel_url or secret is empty or invalid.
            URLPublisherError: If no endpoint is configured, or HTTP request fails/returns non-2xx.
        """
        if not isinstance(tunnel_url, str) or not tunnel_url.strip():
            raise ValueError("tunnel_url and secret must be non-empty strings.")
        if not isinstance(secret, str) or not secret.strip():
            raise ValueError("tunnel_url and secret must be non-empty strings.")

        target = endpoint_url or self.endpoint_url or os.environ.get(ENV_REGISTRY_WEBHOOK_URL)
        if not target or not str(target).strip():
            raise URLPublisherError("No registry webhook URL configured.")

        payload: Dict[str, Any] = {
            "tunnel_url": tunnel_url.strip(),
            "secret": secret.strip(),
        }
        if metadata:
            payload.update(metadata)

        try:
            if self._client is not None:
                resp = self._client.post(target, json=payload, timeout=self.timeout_seconds)
            else:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    resp = client.post(target, json=payload)

            resp.raise_for_status()
            try:
                data = resp.json()
                return data if isinstance(data, dict) else {"response": data}
            except Exception:
                return {"status": "ok", "status_code": resp.status_code}
        except Exception as exc:
            raise URLPublisherError(f"Failed to publish tunnel URL: {exc}") from exc


def publish_tunnel_url(
    tunnel_url: str,
    secret: str,
    endpoint_url: Optional[str] = None,
    timeout_seconds: float = DEFAULT_PUBLISH_TIMEOUT,
) -> Dict[str, Any]:
    """Convenience functional wrapper around URLPublisher."""
    publisher = URLPublisher(endpoint_url=endpoint_url, timeout_seconds=timeout_seconds)
    return publisher.publish(tunnel_url=tunnel_url, secret=secret)
