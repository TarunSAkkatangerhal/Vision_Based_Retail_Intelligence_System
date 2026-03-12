import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from shared.schemas import (
    CustomerBehaviorData,
    InventoryData,
    PromotionData,
)
from shared.utils import get_current_timestamp
from backend.database import (
    init_db,
    insert_inventory,
    insert_customers,
    get_promotions,
    check_health,
)
from backend.promotion_engine import generate_promotions

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Application lifespan (startup / shutdown)
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the database on startup."""
    init_db()
    logger.info("Backend API started.")
    yield
    logger.info("Backend API shutting down.")


app = FastAPI(title="Retail Intelligence API", version="1.0", lifespan=lifespan)


# ──────────────────────────────────────────────
# POST /inventory
# ──────────────────────────────────────────────

@app.post("/inventory")
async def post_inventory(data: InventoryData):
    """Accept inventory data and persist it to the database."""
    try:
        records_saved = insert_inventory(data.model_dump())
        logger.info("POST /inventory — saved %d records.", records_saved)
        return {"status": "ok", "records_saved": records_saved}
    except Exception as e:
        logger.error("POST /inventory failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────
# POST /customers
# ──────────────────────────────────────────────

@app.post("/customers")
async def post_customers(data: CustomerBehaviorData):
    """Accept customer behaviour data and persist it to the database."""
    try:
        records_saved = insert_customers(data.model_dump())
        logger.info("POST /customers — saved %d records.", records_saved)
        return {"status": "ok", "records_saved": records_saved}
    except Exception as e:
        logger.error("POST /customers failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────
# GET /promotions
# ──────────────────────────────────────────────

@app.get("/promotions")
async def get_promotions_endpoint():
    """Generate and return promotion suggestions."""
    try:
        promo_dict = generate_promotions()
        # Validate against Pydantic schema before returning
        validated = PromotionData.model_validate(promo_dict)
        logger.info("GET /promotions — returning %d promotions.", len(validated.promotions))
        return validated.model_dump()
    except ValidationError as e:
        logger.error("GET /promotions validation error: %s", e)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("GET /promotions failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────
# GET /health
# ──────────────────────────────────────────────

@app.get("/health")
async def health():
    """Return system health status."""
    db_ok = check_health()
    return {"status": "healthy", "db_connected": db_ok}


# ──────────────────────────────────────────────
# Entry point (for direct execution)
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.api:app", host="0.0.0.0", port=8000, reload=True)
