import configparser
import os
from typing import Any, Dict

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

DATA_CSV_PATH = os.path.join(PROJECT_ROOT, "data.csv")

CSV_ANGLE_KEYS = [
    "shoulder_flex_ext_left",
    "shoulder_flex_ext_right",
    "shoulder_abd_add_left",
    "shoulder_abd_add_right",
    "elbow_flex_ext_left",
    "elbow_flex_ext_right",
    "hip_flex_ext_left",
    "hip_flex_ext_right",
    "hip_abd_add_left",
    "hip_abd_add_right",
    "knee_flex_ext_left",
    "knee_flex_ext_right",
    "ankle_dorsi_plantar_left",
    "ankle_dorsi_plantar_right",
    "torso_flex_ext",
    "neck_flex_ext",
]

TARGET_COLUMN = "movement_label"

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "movement_model.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

SETTINGS_PATH = os.path.join(PROJECT_ROOT, "settings.ini")


def _load_settings() -> Dict[str, Any]:
    """
    Загружает настройки из settings.ini (если есть) и возвращает словарь
    с приведёнными к типам значениями.
    """
    defaults: Dict[str, Any] = {
        "camera_index": 0,
        "window_width": 1280,
        "window_height": 720,
        "angle_update_interval_sec": 0.2,
        "pose_detection_confidence": 0.5,
    }

    if not os.path.exists(SETTINGS_PATH):
        return defaults

    parser = configparser.ConfigParser()
    parser.read(SETTINGS_PATH, encoding="utf-8")

    def get_int(section: str, option: str, fallback: int) -> int:
        try:
            return parser.getint(section, option, fallback=fallback)
        except ValueError:
            return fallback

    def get_float(section: str, option: str, fallback: float) -> float:
        try:
            return parser.getfloat(section, option, fallback=fallback)
        except ValueError:
            return fallback

    settings: Dict[str, Any] = {}
    settings["camera_index"] = get_int("video", "camera_index", defaults["camera_index"])
    settings["window_width"] = get_int("video", "window_width", defaults["window_width"])
    settings["window_height"] = get_int("video", "window_height", defaults["window_height"])
    settings["angle_update_interval_sec"] = get_float(
        "video",
        "angle_update_interval_sec",
        defaults["angle_update_interval_sec"],
    )
    settings["pose_detection_confidence"] = get_float(
        "pose",
        "detection_confidence",
        defaults["pose_detection_confidence"],
    )

    return settings


_SETTINGS = _load_settings()

# Параметры видеопотока / окна, которые могут настраиваться через settings.ini
CAMERA_INDEX: int = int(_SETTINGS["camera_index"])
WINDOW_WIDTH: int = int(_SETTINGS["window_width"])
WINDOW_HEIGHT: int = int(_SETTINGS["window_height"])
ANGLE_UPDATE_INTERVAL_SEC: float = float(_SETTINGS["angle_update_interval_sec"])
POSE_DETECTION_CONFIDENCE: float = float(_SETTINGS["pose_detection_confidence"])

