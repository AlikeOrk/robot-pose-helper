import csv
import logging
import os
import sys
import time
from typing import Dict, List

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from angles import compute_limb_angles
from .config import (
    ANGLE_UPDATE_INTERVAL_SEC,
    CAMERA_INDEX,
    DATA_CSV_PATH,
    CSV_ANGLE_KEYS,
    POSE_DETECTION_CONFIDENCE,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)


logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.getenv("TEMP", os.getcwd()), "mp_pose_models")
MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker_full.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)


def ensure_model_downloaded() -> str:
    """
    Гарантирует наличие pose‑модели Mediapipe, скачивая её при необходимости.

    :raises RuntimeError: если модель не удалось скачать.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        import urllib.request

        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        except Exception as exc:  # проблемы с сетью/диском
            logger.error(
                "Не удалось скачать модель позы для Mediapipe: %s", exc
            )
            raise RuntimeError(
                "Не удалось скачать модель позы для Mediapipe. "
                "Проверьте подключение к интернету и доступ к диску."
            ) from exc
    return MODEL_PATH


def draw_angles_on_frame(frame, angles: Dict[str, float]) -> None:
    """
    Отрисовывает углы суставов на кадре в виде текста.
    
    :param frame: Кадр изображения для отрисовки (BGR формат)
    :param angles: Словарь с углами суставов (название -> значение в градусах)
    """
    y0 = 30  # Начальная позиция Y для первого угла
    dy = 20  # Расстояние между строками
    x = 10   # Позиция X для всех углов
    for i, (name, value) in enumerate(sorted(angles.items())):
        # Форматируем текст: проверяем на NaN
        text = f"{name}: {value:6.1f}°" if value == value else f"{name}: NaN"
        y = y0 + i * dy
        cv2.putText(
            frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),  # Зелёный цвет
            1,
            cv2.LINE_AA,
        )


def draw_skeleton(frame, landmarks) -> None:
    if landmarks is None:
        return

    height, width, _ = frame.shape

    connections = [
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
        (11, 12),
        (11, 23),
        (12, 24),
        (23, 24),
        (23, 25),
        (25, 27),
        (24, 26),
        (26, 28),
    ]

    def to_px(idx: int):
        lm = landmarks[idx]
        x_px = int(lm.x * width)
        y_px = int(lm.y * height)
        return x_px, y_px

    for i1, i2 in connections:
        if i1 < len(landmarks) and i2 < len(landmarks):
            x1, y1 = to_px(i1)
            x2, y2 = to_px(i2)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    for idx, lm in enumerate(landmarks):
        x, y = to_px(idx)
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)


def log_angles_to_csv(angles: Dict[str, float]) -> None:
    """
    Записывает углы суставов в CSV файл для последующего обучения модели.
    
    Пропускает запись, если какие-то углы невалидны (отрицательные или NaN),
    чтобы не засорять датасет некорректными данными.
    
    :param angles: Словарь с углами суставов (название -> значение в градусах)
    """
    # Формируем строку значений в порядке, определённом CSV_ANGLE_KEYS
    row_values = []
    for key in CSV_ANGLE_KEYS:
        value = angles.get(key, -1.0)
        row_values.append(value)

    # Проверяем валидность всех углов (не должны быть отрицательными или NaN)
    if any((v < 0) or (v != v) for v in row_values):
        # Пропускаем кадр, если какие-то углы невалидны, чтобы не засорять датасет.
        return

    # Определяем, нужно ли записывать заголовок (если файл новый или пустой)
    need_header = not os.path.exists(DATA_CSV_PATH) or os.path.getsize(DATA_CSV_PATH) == 0

    try:
        # Записываем данные в режиме добавления (append)
        with open(DATA_CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if need_header:
                writer.writerow(CSV_ANGLE_KEYS)  # Записываем заголовок при первом запуске
            # Записываем значения углов с точностью до 2 знаков после запятой
            writer.writerow([f"{v:.2f}" for v in row_values])
    except OSError as exc:
        # Не падаем насмерть из‑за временной ошибки диска, а просто логируем предупреждение.
        logger.warning("Не удалось записать строку в %s: %s", DATA_CSV_PATH, exc)


def main() -> None:
    """
    Окно захвата позы с записью углов в CSV.

    Пробел — вкл/выкл запись, q — выход.
    """
    # Базовая конфигурация логов для CLI‑запуска
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        model_path = ensure_model_downloaded()
    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=POSE_DETECTION_CONFIDENCE,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)
    if not cap.isOpened():
        logger.error(
            "Не удалось открыть камеру (index %d). "
            "Проверьте, что веб‑камера подключена и не занята другим приложением.",
            CAMERA_INDEX,
        )
        return

    window_name = "Pose Angles Recorder"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, WINDOW_WIDTH, WINDOW_HEIGHT)

    last_angle_time = 0.0
    last_angles: Dict[str, float] = {}
    recording_enabled = False

    while True:
        success, frame = cap.read()
        if not success:
            logger.error("Кадр с камеры не прочитан, выходим из цикла захвата.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        landmarks = None
        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]

        current_time = time.time()
        if (
            recording_enabled
            and landmarks is not None
            and current_time - last_angle_time >= ANGLE_UPDATE_INTERVAL_SEC
        ):
            last_angle_time = current_time
            angles = compute_limb_angles(landmarks)
            if angles:
                log_angles_to_csv(angles)
                last_angles = angles

        draw_skeleton(frame, landmarks)
        draw_angles_on_frame(frame, last_angles)

        rec_text = "REC ON" if recording_enabled else "REC OFF"
        rec_color = (0, 0, 255) if recording_enabled else (128, 128, 128)
        cv2.putText(
            frame,
            rec_text,
            (WINDOW_WIDTH - 160, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            rec_color,
            2,
            cv2.LINE_AA,
        )

        frame_resized = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        cv2.imshow(window_name, frame_resized)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            recording_enabled = not recording_enabled

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


