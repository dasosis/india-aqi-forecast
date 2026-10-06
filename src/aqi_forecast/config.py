"""Settings shared across the project."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"

# City centres. We collect every PM2.5 station within RADIUS_M of each one.
CITIES = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
}
RADIUS_M = 25_000  # OpenAQ allows at most 25 km

TIMEZONE = "Asia/Kolkata"
