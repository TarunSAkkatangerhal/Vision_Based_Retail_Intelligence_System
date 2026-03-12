import os

CAMERA_ID = os.getenv("CAMERA_ID", "cam_01")

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

DATABASE_PATH = os.getenv("DATABASE_PATH", "backend/retail.db")

MODEL_PATHS = {
    "inventory": os.getenv("MODEL_INVENTORY", "yolov8x-worldv2.pt"),
    "person": os.getenv("MODEL_PERSON", "yolov8n.pt")
}

# Product classes for YOLOWorld open-vocabulary detection.
# IMPORTANT: Fewer classes = higher per-class confidence.  Keep this list
# small and specific for best results with open-vocabulary detection.
# Override via the PRODUCT_CLASSES env var (comma-separated).
PRODUCT_CLASSES = [
    c.strip()
    for c in os.getenv(
        "PRODUCT_CLASSES",
        "bottle,can,canned food,packaged food,box,jar,carton"
    ).split(",")
    if c.strip()
]

PRODUCT_CONFIDENCE_THRESHOLD = float(os.getenv("PRODUCT_CONFIDENCE_THRESHOLD", "0.07"))
PERSON_CONFIDENCE_THRESHOLD = float(os.getenv("PERSON_CONFIDENCE_THRESHOLD", "0.5"))

LOW_STOCK_THRESHOLD = int(os.getenv("LOW_STOCK_THRESHOLD", "2"))
DWELL_TIME_THRESHOLD = int(os.getenv("DWELL_TIME_THRESHOLD", "5"))

PROCESS_EVERY_N_FRAMES = int(os.getenv("PROCESS_EVERY_N_FRAMES", "5"))

MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() in ("true", "1", "yes")
