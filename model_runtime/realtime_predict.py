from __future__ import annotations

import logging
import os
import sys
import time
from typing import Dict, Optional

import cv2
import joblib
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from angles import compute_limb_angles
from model_training.config import (
    ANGLE_UPDATE_INTERVAL_SEC,
    CAMERA_INDEX,
    MODEL_PATH,
    POSE_DETECTION_CONFIDENCE,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)


logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.getenv("TEMP", os.getcwd()), "mp_pose_models")
MODEL_PATH_TASK = os.path.join(MODEL_DIR, "pose_landmarker_full.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)


def ensure_model_downloaded() -> str:
    """
    Скачивает (при необходимости) pose‑модель Mediapipe для режима runtime.

    :raises RuntimeError: если модель не удалось скачать.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH_TASK):
        import urllib.request

        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH_TASK)
        except Exception as exc:
            logger.error(
                "Не удалось скачать модель позы для Mediapipe (runtime): %s", exc
            )
            raise RuntimeError(
                "Не удалось скачать модель позы для Mediapipe (runtime). "
                "Проверьте интернет и права записи."
            ) from exc
    return MODEL_PATH_TASK


def draw_skeleton(frame, landmarks) -> None:
    """
    Отрисовывает скелет человека на кадре на основе ключевых точек Mediapipe.
    
    :param frame: Кадр изображения для отрисовки (BGR формат)
    :param landmarks: Список ключевых точек позы от Mediapipe
    """
    if landmarks is None:
        return

    height, width, _ = frame.shape

    # Определяем соединения между ключевыми точками для отрисовки скелета
    # Формат: (индекс_точки_1, индекс_точки_2)
    connections = [
        (11, 13),  # Левое плечо -> левый локоть
        (13, 15),  # Левый локоть -> левое запястье
        (12, 14),  # Правое плечо -> правый локоть
        (14, 16),  # Правый локоть -> правое запястье
        (11, 12),  # Левое плечо -> правое плечо
        (11, 23),  # Левое плечо -> левый таз
        (12, 24),  # Правое плечо -> правый таз
        (23, 24),  # Левый таз -> правый таз
        (23, 25),  # Левый таз -> левое колено
        (25, 27),  # Левое колено -> левая лодыжка
        (24, 26),  # Правый таз -> правое колено
        (26, 28),  # Правое колено -> правая лодыжка
    ]

    def to_px(idx: int):
        """Преобразует нормализованные координаты ключевой точки в пиксели."""
        lm = landmarks[idx]
        x_px = int(lm.x * width)
        y_px = int(lm.y * height)
        return x_px, y_px

    # Отрисовываем линии между соединёнными точками (кости скелета)
    for i1, i2 in connections:
        if i1 < len(landmarks) and i2 < len(landmarks):
            x1, y1 = to_px(i1)
            x2, y2 = to_px(i2)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Жёлтый цвет

    # Отрисовываем ключевые точки (суставы)
    for idx, lm in enumerate(landmarks):
        x, y = to_px(idx)
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)  # Красный цвет


def main() -> None:
    """
    Запуск обученной модели предсказания углов в реальном времени.
    
    Модель предсказывает углы суставов на следующий тик на основе текущих углов,
    а затем сравнивает предсказанные значения с реальными углами, полученными с камеры.
    
    Управление:
    - q — выход из приложения
    
    Отображаемые метрики:
    - Accuracy: точность предсказания в процентах (чем выше, тем лучше)
    - Error: ошибка предсказания в процентах (чем ниже, тем лучше)
    
    Цвет индикатора:
    - Зелёный: ошибка <= 10% (отличное предсказание)
    - Жёлтый: ошибка 10-25% (хорошее предсказание)
    - Красный: ошибка > 25% (требуется улучшение модели или больше данных)
    """
    # Базовая конфигурация логов для CLI‑запуска
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not os.path.exists(MODEL_PATH):
        logger.error(
            "Файл модели не найден: %s. "
            "Сначала обучите модель командой 'python -m model_training.train_model'.",
            MODEL_PATH,
        )
        sys.exit(1)

    try:
        bundle = joblib.load(MODEL_PATH)
    except Exception as exc:
        logger.error("Не удалось загрузить модель из %s: %s", MODEL_PATH, exc)
        sys.exit(1)

    try:
        model = bundle["model"]
        angle_keys = bundle["angle_keys"]
        lookback_window = bundle.get("lookback_window", 3)  # По умолчанию 3, если не указано
        n_features = bundle.get("n_features", len(angle_keys))  # Количество признаков в одном тике
    except KeyError as exc:
        logger.error(
            "Файл модели %s повреждён или в старом формате: отсутствует ключ %s.",
            MODEL_PATH,
            exc,
        )
        sys.exit(1)

    try:
        task_model_path = ensure_model_downloaded()
    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    base_options = mp_python.BaseOptions(model_asset_path=task_model_path)
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
            "Проверьте, что веб‑камера подключена и не используется другим приложением.",
            CAMERA_INDEX,
        )
        sys.exit(1)

    window_name = "Movement Model Runtime"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, WINDOW_WIDTH, WINDOW_HEIGHT)

    # Переменные для отслеживания предсказаний и метрик
    last_angle_time = 0.0
    angle_history: list[np.ndarray] = []  # История углов для формирования окна
    predicted_angles: Optional[np.ndarray] = None  # Предсказанные углы следующего тика
    prediction_error: Optional[float] = None  # Ошибка предсказания (MAE в процентах)
    accuracy_percent: Optional[float] = None  # Точность предсказания в процентах
    
    logger.info("Размер окна истории: %d тиков", lookback_window)

    while True:
        success, frame = cap.read()
        if not success:
            logger.error("Кадр с камеры не прочитан, останавливаем runtime.")
            break

        # Конвертируем кадр в RGB для Mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Детектируем позу на текущем кадре
        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        landmarks = None
        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]

        current_time = time.time()
        # Обновляем углы с заданной частотой (по умолчанию каждые 0.2 секунды)
        if landmarks is not None and current_time - last_angle_time >= ANGLE_UPDATE_INTERVAL_SEC:
            # Вычисляем углы суставов из позы
            angles: Dict[str, float] = compute_limb_angles(landmarks)
            if angles:
                # Формируем вектор признаков из углов
                feature_vec = []
                valid = True
                for key in angle_keys:
                    v = float(angles.get(key, -1.0))
                    # Проверяем валидность угла (не NaN и не отрицательный)
                    if v != v or v < 0:
                        valid = False
                        break
                    feature_vec.append(v)

                if valid:
                    real_angles = np.array(feature_vec, dtype=float)
                    
                    # Сравниваем предсказанные углы (из предыдущего тика) с реальными углами текущего тика
                    if predicted_angles is not None:
                        # Вычисляем среднюю абсолютную ошибку (MAE) в градусах
                        mae_degrees = float(np.mean(np.abs(predicted_angles - real_angles)))
                        # Преобразуем в процент ошибки (относительно среднего значения углов)
                        mean_angle = float(np.mean(np.abs(real_angles)))
                        if mean_angle > 0:
                            prediction_error = (mae_degrees / mean_angle) * 100.0
                            accuracy_percent = max(0.0, 100.0 - prediction_error)
                        else:
                            prediction_error = 0.0
                            accuracy_percent = 100.0
                    
                    # Добавляем текущие углы в историю
                    angle_history.append(real_angles)
                    
                    # Поддерживаем размер истории равным lookback_window
                    # Удаляем самые старые значения, если история превышает размер окна
                    if len(angle_history) > lookback_window:
                        angle_history.pop(0)
                    
                    # Предсказываем углы следующего тика на основе окна из предыдущих тиков
                    if len(angle_history) >= lookback_window:
                        # Формируем окно из последних lookback_window тиков
                        # Объединяем их в один вектор признаков (как при обучении)
                        window = np.array(angle_history[-lookback_window:]).flatten()
                        X_window = window.reshape(1, -1)
                        predicted_angles = model.predict(X_window)[0]
                    else:
                        # Если истории недостаточно, предсказание пока невозможно
                        predicted_angles = None
                    
                    # Обновляем время последнего обновления углов
                    last_angle_time = current_time

        # Отрисовываем скелет на кадре
        draw_skeleton(frame, landmarks)

        # Отображаем метрики предсказания на экране
        y_offset = 30
        line_height = 30
        
        # Показываем статус предсказания и размер истории
        if len(angle_history) >= lookback_window:
            cv2.putText(
                frame,
                f"Prediction: Active (history: {len(angle_history)}/{lookback_window})",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                frame,
                f"Collecting history: {len(angle_history)}/{lookback_window}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (128, 128, 128),
                2,
                cv2.LINE_AA,
            )
        y_offset += line_height
        
        # Показываем точность и ошибку предсказания (обновляется каждый тик при наличии данных)
        if accuracy_percent is not None and prediction_error is not None:
            # Цвет индикатора зависит от ошибки предсказания
            if prediction_error <= 10.0:
                error_color = (0, 255, 0)  # Зелёный - хорошее предсказание (ошибка <= 10%)
            elif prediction_error <= 25.0:
                error_color = (0, 255, 255)  # Жёлтый - средняя ошибка (10-25%)
            else:
                error_color = (0, 0, 255)  # Красный - высокая ошибка (>25%)
            
            # Отображаем точность предсказания
            cv2.putText(
                frame,
                f"Accuracy: {accuracy_percent:.1f}%",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                error_color,
                2,
                cv2.LINE_AA,
            )
            y_offset += line_height
            
            # Отображаем ошибку предсказания
            cv2.putText(
                frame,
                f"Error: {prediction_error:.1f}%",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                error_color,
                1,
                cv2.LINE_AA,
            )

        frame_resized = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        cv2.imshow(window_name, frame_resized)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


