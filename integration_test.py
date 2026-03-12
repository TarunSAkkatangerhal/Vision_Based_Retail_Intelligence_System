"""
Integration Test — Vision-Based Retail Intelligence System.

Tests the complete pipeline end-to-end:
  1. Shared utilities & config
  2. Schema validation (Pydantic models)
  3. Vision: product detection, inventory, dwell time, interaction
  4. Backend: database CRUD, API endpoints, promotion engine
  5. End-to-end pipeline flow

Usage:
    python -m pytest integration_test.py -v
    python integration_test.py              # also works standalone
"""

import json
import logging
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

# ──────────────────────────────────────────────
# Setup: ensure project root is on sys.path
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("integration_test")


# ══════════════════════════════════════════════
# SECTION 1 — Shared Config & Utils
# ══════════════════════════════════════════════

class TestSharedConfig:
    """Verify that all required configuration values are present."""

    def test_config_values_exist(self):
        from shared.config import (
            CAMERA_ID,
            API_BASE_URL,
            DATABASE_PATH,
            MODEL_PATHS,
            PRODUCT_CONFIDENCE_THRESHOLD,
            PERSON_CONFIDENCE_THRESHOLD,
            LOW_STOCK_THRESHOLD,
            DWELL_TIME_THRESHOLD,
            PROCESS_EVERY_N_FRAMES,
            MOCK_MODE,
        )
        assert CAMERA_ID, "CAMERA_ID must not be empty"
        assert API_BASE_URL.startswith("http"), "API_BASE_URL must be a URL"
        assert DATABASE_PATH, "DATABASE_PATH must not be empty"
        assert "inventory" in MODEL_PATHS, "MODEL_PATHS must have 'inventory' key"
        assert "person" in MODEL_PATHS, "MODEL_PATHS must have 'person' key"
        assert 0 < PRODUCT_CONFIDENCE_THRESHOLD <= 1.0
        assert 0 < PERSON_CONFIDENCE_THRESHOLD <= 1.0
        assert LOW_STOCK_THRESHOLD >= 0
        assert DWELL_TIME_THRESHOLD >= 0
        assert PROCESS_EVERY_N_FRAMES >= 1
        assert isinstance(MOCK_MODE, bool)
        logger.info("PASSED: All config values present and valid.")

    def test_model_file_exists(self):
        from shared.config import MODEL_PATHS
        inv_model = Path(MODEL_PATHS["inventory"])
        assert inv_model.exists(), (
            f"Trained model not found at {inv_model}. Run training first."
        )
        logger.info("PASSED: Trained model file exists at %s", inv_model)

    def test_get_current_timestamp(self):
        from shared.utils import get_current_timestamp
        ts = get_current_timestamp()
        assert isinstance(ts, str)
        assert "T" in ts, "Timestamp should be ISO 8601 format"
        logger.info("PASSED: get_current_timestamp() → %s", ts)


# ══════════════════════════════════════════════
# SECTION 2 — Pydantic Schema Validation
# ══════════════════════════════════════════════

class TestSchemas:
    """Verify that all Pydantic schemas accept valid data and reject bad data."""

    def test_inventory_schema_valid(self):
        from shared.schemas import InventoryData, ProductItem
        data = InventoryData(
            schema_version="1.0",
            timestamp="2026-03-12T12:00:00Z",
            camera_id="cam_01",
            products=[
                ProductItem(product_name="Meiji-Protein Chocolate",
                            shelf_id="shelf_A", count=5, status="normal"),
                ProductItem(product_name="Tofusan-High Protein Malt",
                            shelf_id="shelf_B", count=1, status="low_stock"),
                ProductItem(product_name="unknown",
                            shelf_id="shelf_C", count=0, status="empty"),
            ],
        )
        d = data.model_dump()
        assert d["camera_id"] == "cam_01"
        assert len(d["products"]) == 3
        logger.info("PASSED: InventoryData schema accepts valid data.")

    def test_inventory_schema_rejects_bad_status(self):
        from shared.schemas import ProductItem
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ProductItem(product_name="x", shelf_id="s", count=1, status="bad_status")
        logger.info("PASSED: ProductItem rejects invalid status.")

    def test_customer_schema_valid(self):
        from shared.schemas import CustomerBehaviorData, CustomerItem
        data = CustomerBehaviorData(
            schema_version="1.0",
            timestamp="2026-03-12T12:00:00Z",
            camera_id="cam_01",
            customers=[
                CustomerItem(customer_id="cust_1", shelf_id="shelf_A",
                             dwell_time_seconds=8, interaction="picked_product"),
                CustomerItem(customer_id="cust_2", shelf_id="shelf_B",
                             dwell_time_seconds=3, interaction="none"),
            ],
        )
        d = data.model_dump()
        assert len(d["customers"]) == 2
        logger.info("PASSED: CustomerBehaviorData schema accepts valid data.")

    def test_customer_schema_rejects_bad_interaction(self):
        from shared.schemas import CustomerItem
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            CustomerItem(customer_id="c", shelf_id="s",
                         dwell_time_seconds=1, interaction="invalid_action")
        logger.info("PASSED: CustomerItem rejects invalid interaction.")

    def test_promotion_schema_valid(self):
        from shared.schemas import PromotionData, PromotionItem
        data = PromotionData(
            schema_version="1.0",
            generated_at="2026-03-12T12:00:00Z",
            promotions=[
                PromotionItem(
                    shelf_id="shelf_A",
                    reason="high_dwell_low_conversion",
                    confidence_score=0.75,
                    suggested_action="Improve signage",
                ),
            ],
        )
        d = data.model_dump()
        assert d["promotions"][0]["confidence_score"] == 0.75
        logger.info("PASSED: PromotionData schema accepts valid data.")

    def test_promotion_schema_rejects_bad_confidence(self):
        from shared.schemas import PromotionItem
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            PromotionItem(shelf_id="s", reason="high_dwell_low_conversion",
                          confidence_score=1.5, suggested_action="x")
        logger.info("PASSED: PromotionItem rejects confidence > 1.0.")


# ══════════════════════════════════════════════
# SECTION 3 — Vision Modules
# ══════════════════════════════════════════════

class TestVisionProductDetection:
    """Test the product detection module with a real model on a dummy frame."""

    def test_detect_products_returns_list(self):
        from vision.product_detection import detect_products
        # Create a dummy 640x640 frame (black image)
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        detections = detect_products(frame)
        assert isinstance(detections, list)
        # On a blank image we may get 0 detections — that's fine
        for det in detections:
            assert "product_name" in det
            assert "bbox" in det
            assert "confidence" in det
            assert len(det["bbox"]) == 4
        logger.info("PASSED: detect_products returned %d detections on blank frame.",
                     len(detections))

    def test_detect_products_on_real_image(self):
        """Run detection on a real dataset image if available."""
        from vision.product_detection import detect_products
        sample_dir = PROJECT_ROOT / "dataset" / "images" / "val"
        images = list(sample_dir.glob("*.jpg"))
        if not images:
            pytest.skip("No validation images found for real-image test.")

        import cv2
        img = cv2.imread(str(images[0]))
        assert img is not None, f"Failed to read {images[0]}"

        detections = detect_products(img)
        assert isinstance(detections, list)
        logger.info("PASSED: detect_products found %d products in %s",
                     len(detections), images[0].name)


class TestVisionInventory:
    """Test inventory detection (integrates product_detection + schema)."""

    def test_detect_inventory_returns_valid_schema(self):
        from vision.inventory import detect_inventory
        from shared.schemas import InventoryData
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_inventory(frame)
        # Must parse as valid InventoryData
        validated = InventoryData.model_validate(result)
        assert validated.camera_id
        assert isinstance(validated.products, list)
        assert len(validated.products) > 0, "Should have at least empty-shelf entries"
        logger.info("PASSED: detect_inventory returns valid InventoryData with %d products.",
                     len(validated.products))

    def test_detect_inventory_on_real_image(self):
        """Run inventory detection on a real dataset image."""
        from vision.inventory import detect_inventory
        from shared.schemas import InventoryData
        import cv2

        sample_dir = PROJECT_ROOT / "dataset" / "images" / "val"
        images = list(sample_dir.glob("*.jpg"))
        if not images:
            pytest.skip("No validation images found.")

        img = cv2.imread(str(images[0]))
        result = detect_inventory(img)
        validated = InventoryData.model_validate(result)
        assert len(validated.products) > 0
        logger.info("PASSED: detect_inventory on real image → %d product entries.",
                     len(validated.products))


class TestDwellTime:
    """Test the dwell time calculation logic."""

    def test_dwell_time_new_track(self):
        from vision.dwell_time import calculate_dwell_time, reset
        reset()
        regions = [
            {"shelf_id": "shelf_A", "x_min": 0, "x_max": 200,
             "y_min": 0, "y_max": 480},
        ]
        # First call for a new track → dwell = 0
        dwell = calculate_dwell_time("track_1", (100.0, 200.0), regions)
        assert dwell == 0
        logger.info("PASSED: New track starts with dwell_time=0.")

    def test_dwell_time_accumulates(self):
        from vision.dwell_time import calculate_dwell_time, reset
        reset()
        regions = [
            {"shelf_id": "shelf_A", "x_min": 0, "x_max": 200,
             "y_min": 0, "y_max": 480},
        ]
        calculate_dwell_time("track_2", (100.0, 200.0), regions)
        time.sleep(1.1)
        dwell = calculate_dwell_time("track_2", (100.0, 200.0), regions)
        assert dwell >= 1, f"Expected dwell >= 1s, got {dwell}"
        logger.info("PASSED: Dwell time accumulates over time (%ds).", dwell)

    def test_dwell_time_resets_outside_region(self):
        from vision.dwell_time import calculate_dwell_time, reset
        reset()
        regions = [
            {"shelf_id": "shelf_A", "x_min": 0, "x_max": 200,
             "y_min": 0, "y_max": 480},
        ]
        calculate_dwell_time("track_3", (100.0, 200.0), regions)
        time.sleep(0.5)
        # Move outside all regions
        dwell = calculate_dwell_time("track_3", (999.0, 999.0), regions)
        assert dwell == 0
        logger.info("PASSED: Dwell time resets when person leaves shelf region.")


class TestInteraction:
    """Test the interaction detection logic."""

    def test_interaction_none_few_positions(self):
        from vision.interaction import detect_interaction, reset
        reset()
        result = detect_interaction("t1", [(100, 200), (101, 201)])
        assert result == "none"
        logger.info("PASSED: Few positions → interaction='none'.")

    def test_interaction_picked_product(self):
        from vision.interaction import detect_interaction, reset
        reset()
        # Simulate hand moving UP then DOWN (Y decreases then increases)
        positions = [(100, 200)] * 5 + [(100, 300)] * 5
        result = detect_interaction("t2", positions)
        assert result == "picked_product"
        logger.info("PASSED: Downward Y movement → 'picked_product'.")

    def test_interaction_replaced_product(self):
        from vision.interaction import detect_interaction, reset
        reset()
        # Simulate hand moving DOWN then UP (Y increases then decreases)
        positions = [(100, 300)] * 5 + [(100, 200)] * 5
        result = detect_interaction("t3", positions)
        assert result == "replaced_product"
        logger.info("PASSED: Upward Y movement → 'replaced_product'.")


# ══════════════════════════════════════════════
# SECTION 4 — Database
# ══════════════════════════════════════════════

class TestDatabase:
    """Test database init, insert, and query operations using a temp DB."""

    @pytest.fixture(autouse=True)
    def _use_temp_db(self, tmp_path):
        """Redirect DATABASE_PATH to a temporary file for each test."""
        db_path = str(tmp_path / "test_retail.db")
        self._patcher = patch("shared.config.DATABASE_PATH", db_path)
        self._patcher.start()
        # Also patch the module-level reference in database.py
        self._patcher2 = patch("backend.database.DATABASE_PATH", db_path)
        self._patcher2.start()
        self._patcher3 = patch("backend.promotion_engine.DATABASE_PATH", db_path)
        self._patcher3.start()
        self.db_path = db_path
        yield
        self._patcher.stop()
        self._patcher2.stop()
        self._patcher3.stop()

    def test_init_db_creates_tables(self):
        from backend.database import init_db, check_health
        init_db()
        assert check_health() is True
        logger.info("PASSED: init_db creates all required tables.")

    def test_insert_and_query_inventory(self):
        from backend.database import init_db, insert_inventory
        init_db()
        data = {
            "schema_version": "1.0",
            "timestamp": "2026-03-12T12:00:00Z",
            "camera_id": "cam_01",
            "products": [
                {"product_name": "Meiji-Protein Chocolate",
                 "shelf_id": "shelf_A", "count": 5, "status": "normal"},
                {"product_name": "Tofusan-High Protein Malt",
                 "shelf_id": "shelf_B", "count": 1, "status": "low_stock"},
            ],
        }
        rows = insert_inventory(data)
        assert rows == 2

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM inventory_logs")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 2
        logger.info("PASSED: insert_inventory wrote 2 rows.")

    def test_insert_and_query_customers(self):
        from backend.database import init_db, insert_customers
        init_db()
        data = {
            "schema_version": "1.0",
            "timestamp": "2026-03-12T12:00:00Z",
            "camera_id": "cam_01",
            "customers": [
                {"customer_id": "cust_1", "shelf_id": "shelf_A",
                 "dwell_time_seconds": 8, "interaction": "picked_product"},
                {"customer_id": "cust_2", "shelf_id": "shelf_B",
                 "dwell_time_seconds": 3, "interaction": "none"},
            ],
        }
        rows = insert_customers(data)
        assert rows == 2

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM customer_logs")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 2
        logger.info("PASSED: insert_customers wrote 2 rows.")

    def test_promotion_engine_generates_promotions(self):
        from backend.database import init_db, insert_inventory, insert_customers
        from backend.promotion_engine import generate_promotions
        from shared.schemas import PromotionData
        init_db()

        # Insert inventory with low stock
        insert_inventory({
            "schema_version": "1.0",
            "timestamp": "2026-03-12T12:00:00Z",
            "camera_id": "cam_01",
            "products": [
                {"product_name": "x", "shelf_id": "shelf_A",
                 "count": 1, "status": "low_stock"},
            ],
        })

        # Insert customers: high dwell + visits to low-stock shelf
        for i in range(5):
            insert_customers({
                "schema_version": "1.0",
                "timestamp": f"2026-03-12T12:0{i}:00Z",
                "camera_id": "cam_01",
                "customers": [
                    {"customer_id": f"cust_{i}", "shelf_id": "shelf_A",
                     "dwell_time_seconds": 15, "interaction": "none"},
                ],
            })

        result = generate_promotions()
        validated = PromotionData.model_validate(result)
        assert len(validated.promotions) > 0, "Should generate at least one promotion"
        logger.info("PASSED: Promotion engine generated %d promotions.",
                     len(validated.promotions))


# ══════════════════════════════════════════════
# SECTION 5 — FastAPI Endpoints
# ══════════════════════════════════════════════

class TestAPI:
    """Test FastAPI endpoints using the TestClient."""

    @pytest.fixture(autouse=True)
    def _use_temp_db(self, tmp_path):
        db_path = str(tmp_path / "test_api.db")
        self._patcher = patch("shared.config.DATABASE_PATH", db_path)
        self._patcher.start()
        self._patcher2 = patch("backend.database.DATABASE_PATH", db_path)
        self._patcher2.start()
        self._patcher3 = patch("backend.promotion_engine.DATABASE_PATH", db_path)
        self._patcher3.start()

        from backend.database import init_db
        init_db()
        yield
        self._patcher.stop()
        self._patcher2.stop()
        self._patcher3.stop()

    def _get_client(self):
        from fastapi.testclient import TestClient
        from backend.api import app
        return TestClient(app)

    def test_health_endpoint(self):
        client = self._get_client()
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["db_connected"] is True
        logger.info("PASSED: GET /health returns healthy status.")

    def test_post_inventory(self):
        client = self._get_client()
        payload = {
            "schema_version": "1.0",
            "timestamp": "2026-03-12T12:00:00Z",
            "camera_id": "cam_01",
            "products": [
                {"product_name": "Meiji-Protein Chocolate",
                 "shelf_id": "shelf_A", "count": 5, "status": "normal"},
            ],
        }
        resp = client.post("/inventory", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["records_saved"] == 1
        logger.info("PASSED: POST /inventory succeeds.")

    def test_post_inventory_invalid_data(self):
        client = self._get_client()
        resp = client.post("/inventory", json={"bad": "data"})
        assert resp.status_code == 422
        logger.info("PASSED: POST /inventory rejects invalid data (422).")

    def test_post_customers(self):
        client = self._get_client()
        payload = {
            "schema_version": "1.0",
            "timestamp": "2026-03-12T12:00:00Z",
            "camera_id": "cam_01",
            "customers": [
                {"customer_id": "cust_1", "shelf_id": "shelf_A",
                 "dwell_time_seconds": 8, "interaction": "picked_product"},
            ],
        }
        resp = client.post("/customers", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["records_saved"] == 1
        logger.info("PASSED: POST /customers succeeds.")

    def test_post_customers_invalid_data(self):
        client = self._get_client()
        resp = client.post("/customers", json={"missing": "fields"})
        assert resp.status_code == 422
        logger.info("PASSED: POST /customers rejects invalid data (422).")

    def test_get_promotions(self):
        client = self._get_client()
        resp = client.get("/promotions")
        assert resp.status_code == 200
        body = resp.json()
        assert "promotions" in body
        assert isinstance(body["promotions"], list)
        logger.info("PASSED: GET /promotions returns valid structure.")


# ══════════════════════════════════════════════
# SECTION 6 — End-to-End Pipeline
# ══════════════════════════════════════════════

class TestEndToEnd:
    """Test the full pipeline: frame → detection → API → DB → promotions."""

    @pytest.fixture(autouse=True)
    def _use_temp_db(self, tmp_path):
        db_path = str(tmp_path / "test_e2e.db")
        self._patcher = patch("shared.config.DATABASE_PATH", db_path)
        self._patcher.start()
        self._patcher2 = patch("backend.database.DATABASE_PATH", db_path)
        self._patcher2.start()
        self._patcher3 = patch("backend.promotion_engine.DATABASE_PATH", db_path)
        self._patcher3.start()

        from backend.database import init_db
        init_db()
        yield
        self._patcher.stop()
        self._patcher2.stop()
        self._patcher3.stop()

    def test_full_pipeline_with_real_image(self):
        """
        End-to-end test:
          1. Load a real image from the dataset
          2. Run inventory detection (YOLO model)
          3. POST results to the API
          4. Verify data appears in the database
          5. Generate promotions
        """
        from fastapi.testclient import TestClient
        from backend.api import app
        from vision.inventory import detect_inventory
        from shared.schemas import InventoryData, PromotionData

        import cv2

        client = TestClient(app)

        # Step 1 — Load real image
        sample_dir = PROJECT_ROOT / "dataset" / "images" / "val"
        images = list(sample_dir.glob("*.jpg"))
        if not images:
            pytest.skip("No validation images available for E2E test.")

        img = cv2.imread(str(images[0]))
        assert img is not None
        logger.info("E2E: Loaded image %s", images[0].name)

        # Step 2 — Run inventory detection
        inv_result = detect_inventory(img)
        validated_inv = InventoryData.model_validate(inv_result)
        logger.info("E2E: Detected %d product entries.", len(validated_inv.products))

        # Step 3 — POST to API
        resp = client.post("/inventory", json=inv_result)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        logger.info("E2E: POST /inventory succeeded.")

        # Step 4 — POST some customer data too
        customer_payload = {
            "schema_version": "1.0",
            "timestamp": "2026-03-12T12:00:00Z",
            "camera_id": "cam_01",
            "customers": [
                {"customer_id": "cust_1", "shelf_id": "shelf_A",
                 "dwell_time_seconds": 15, "interaction": "none"},
                {"customer_id": "cust_2", "shelf_id": "shelf_A",
                 "dwell_time_seconds": 10, "interaction": "none"},
                {"customer_id": "cust_3", "shelf_id": "shelf_A",
                 "dwell_time_seconds": 12, "interaction": "none"},
            ],
        }
        resp = client.post("/customers", json=customer_payload)
        assert resp.status_code == 200
        logger.info("E2E: POST /customers succeeded.")

        # Step 5 — Check health
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["db_connected"] is True
        logger.info("E2E: Health check passed.")

        # Step 6 — Generate promotions
        resp = client.get("/promotions")
        assert resp.status_code == 200
        promo_result = resp.json()
        validated_promo = PromotionData.model_validate(promo_result)
        logger.info("E2E: Promotions generated → %d suggestions.",
                     len(validated_promo.promotions))

        # Final summary
        logger.info("=" * 50)
        logger.info("END-TO-END TEST PASSED SUCCESSFULLY")
        logger.info("  Products detected  : %d", len(validated_inv.products))
        logger.info("  Promotions generated: %d", len(validated_promo.promotions))
        logger.info("=" * 50)


# ══════════════════════════════════════════════
# SECTION 7 — Video Loader (unit-level)
# ══════════════════════════════════════════════

class TestVideoLoader:
    """Test the video frame generator."""

    def test_generator_with_nonexistent_file(self):
        from shared.video_loader import get_frame_generator
        frames = list(get_frame_generator("nonexistent_video.mp4"))
        assert frames == [], "Non-existent file should yield no frames."
        logger.info("PASSED: get_frame_generator handles missing file gracefully.")

    def test_generator_with_real_image_as_video(self):
        """Use a single image to create a tiny video and test the loader."""
        import cv2
        sample_dir = PROJECT_ROOT / "dataset" / "images" / "val"
        images = list(sample_dir.glob("*.jpg"))
        if not images:
            pytest.skip("No images available.")

        # Create a tiny test video (5 frames of the same image)
        img = cv2.imread(str(images[0]))
        h, w = img.shape[:2]
        tmp_video = str(PROJECT_ROOT / "test_temp_video.avi")
        try:
            writer = cv2.VideoWriter(
                tmp_video,
                cv2.VideoWriter_fourcc(*"MJPG"),
                10, (w, h),
            )
            for _ in range(5):
                writer.write(img)
            writer.release()

            from shared.video_loader import get_frame_generator
            frames = list(get_frame_generator(tmp_video))
            assert len(frames) == 5, f"Expected 5 frames, got {len(frames)}"
            assert frames[0].shape == img.shape
            logger.info("PASSED: Video loader yielded %d frames.", len(frames))
        finally:
            if os.path.exists(tmp_video):
                os.remove(tmp_video)


# ══════════════════════════════════════════════
# Run tests if executed directly
# ══════════════════════════════════════════════

if __name__ == "__main__":
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    sys.exit(exit_code)
