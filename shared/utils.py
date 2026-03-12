import logging
import time as _time
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

def send_to_api(endpoint: str, data: dict, retries: int = 3) -> dict | None:
    """POST JSON data to API_BASE_URL + endpoint with retry logic."""
    url = f"{API_BASE_URL}{endpoint}"
    for attempt in range(retries):
        try:
            response = requests.post(url, json=data, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error("Attempt %d/%d failed for %s: %s", attempt + 1, retries, url, e)
            if attempt < retries - 1:
                _time.sleep(min(2 ** attempt, 4))  # 1s, 2s, 4s backoff
    return None


def get_current_timestamp() -> str:
    """Return current time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()
