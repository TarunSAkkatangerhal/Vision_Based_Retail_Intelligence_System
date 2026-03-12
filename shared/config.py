CAMERA_ID = "cam_01"

API_BASE_URL = "http://localhost:8000"

DATABASE_PATH = "backend/retail.db"

MODEL_PATHS = {
    "inventory": "models/inventory_yolo.pt",
    "person": "yolov8n.pt"
}

PRODUCT_CONFIDENCE_THRESHOLD = 0.5
PERSON_CONFIDENCE_THRESHOLD = 0.5

LOW_STOCK_THRESHOLD = 2
DWELL_TIME_THRESHOLD = 5

PROCESS_EVERY_N_FRAMES = 5

MOCK_MODE = False
