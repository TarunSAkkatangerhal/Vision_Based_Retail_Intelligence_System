import logging
import requests
from datetime import datetime, timezone

from shared.config import API_BASE_URL

# ──────────────────────────────────────────────
# Logging setup (runs on import)
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────

def send_to_api(endpoint: str, data: dict) -> dict | None:
    """POST JSON data to API_BASE_URL + endpoint and return the JSON response."""
    url = f"{API_BASE_URL}{endpoint}"
    try:
        response = requests.post(url, json=data, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error("Failed to send data to %s: %s", url, e)
        return None


def get_current_timestamp() -> str:
    """Return current time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()
