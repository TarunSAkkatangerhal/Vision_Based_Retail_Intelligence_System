import os

CAMERA_ID = os.getenv("CAMERA_ID", "cam_01")

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

DATABASE_PATH = os.getenv("DATABASE_PATH", "backend/retail.db")

MODEL_PATHS = {
    "inventory": os.getenv("MODEL_INVENTORY", "models/inventory_yolo.pt"),
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

PRODUCT_CONFIDENCE_THRESHOLD = float(os.getenv("PRODUCT_CONFIDENCE_THRESHOLD", "0.25"))
PERSON_CONFIDENCE_THRESHOLD = float(os.getenv("PERSON_CONFIDENCE_THRESHOLD", "0.5"))

LOW_STOCK_THRESHOLD = int(os.getenv("LOW_STOCK_THRESHOLD", "2"))
DWELL_TIME_THRESHOLD = int(os.getenv("DWELL_TIME_THRESHOLD", "5"))

PROCESS_EVERY_N_FRAMES = int(os.getenv("PROCESS_EVERY_N_FRAMES", "5"))

FRAME_PROCESS_WIDTH = int(os.getenv("FRAME_PROCESS_WIDTH", "640"))
FRAME_PROCESS_HEIGHT = int(os.getenv("FRAME_PROCESS_HEIGHT", "480"))

PRODUCT_INFER_SIZE = int(os.getenv("PRODUCT_INFER_SIZE", "640"))
PRODUCT_ENABLE_TILING = os.getenv("PRODUCT_ENABLE_TILING", "false").lower() in ("true", "1", "yes")

ENABLE_SKELETON = os.getenv("ENABLE_SKELETON", "false").lower() in ("true", "1", "yes")
SKELETON_NEAR_SHELF_ONLY = os.getenv("SKELETON_NEAR_SHELF_ONLY", "true").lower() in ("true", "1", "yes")
SKELETON_SHELF_MARGIN = int(os.getenv("SKELETON_SHELF_MARGIN", "80"))

API_QUEUE_MAXSIZE = int(os.getenv("API_QUEUE_MAXSIZE", "32"))

MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() in ("true", "1", "yes")
