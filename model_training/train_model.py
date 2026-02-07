from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from .config import CSV_ANGLE_KEYS, METRICS_PATH, MODEL_DIR, MODEL_PATH
from .dataset import load_dataset


logger = logging.getLogger(__name__)

# Размер окна истории для предсказания (сколько предыдущих тиков использовать)
LOOKBACK_WINDOW = 3


def create_sliding_windows(X: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Создаёт скользящие окна из временного ряда для обучения модели.
    
    Для каждого индекса i создаётся окно из window_size предыдущих значений,
    которое используется для предсказания значения X[i+1].
    
    :param X: Массив данных временного ряда (n_samples, n_features)
    :param window_size: Размер окна (количество предыдущих тиков)
    :return: Кортеж (X_windows, y_target), где X_windows имеет форму (n_samples-window_size, window_size*n_features)
    """
    n_samples, n_features = X.shape
    
    if n_samples < window_size + 1:
        raise ValueError(
            f"Недостаточно данных для создания окон размером {window_size}: "
            f"нужно минимум {window_size + 1} образцов, получено {n_samples}"
        )
    
    X_windows = []
    y_target = []
    
    # Создаём окна: для каждого индекса i от window_size до n_samples-1
    # берём window_size предыдущих значений и предсказываем следующее
    for i in range(window_size, n_samples):
        # Берём window_size предыдущих тиков и объединяем их в один вектор
        window = X[i - window_size:i].flatten()  # Форма: (window_size * n_features,)
        X_windows.append(window)
        y_target.append(X[i])  # Целевое значение - углы следующего тика
    
    return np.array(X_windows), np.array(y_target)


def train_once(random_state: int = 42, lookback_window: int = LOOKBACK_WINDOW) -> Dict[str, Any]:
    """
    Обучает модель регрессии для предсказания углов следующего тика на основе нескольких предыдущих тиков.

    :param random_state: зерно генератора случайных чисел для воспроизводимости.
    :param lookback_window: количество предыдущих тиков для использования в предсказании (по умолчанию 3).
    """
    X, _ = load_dataset(require_labels=False)

    if X.shape[0] < lookback_window + 1:
        raise ValueError(
            f"Слишком мало образцов в датасете: {X.shape[0]} "
            f"(нужно >= {lookback_window + 1} для окна размером {lookback_window})."
        )

    # Создаём скользящие окна из данных
    # Каждое окно содержит lookback_window предыдущих тиков для предсказания следующего
    X_windows, y_next = create_sliding_windows(X, lookback_window)
    
    logger.info(
        "Создано %d окон размером %d тиков. Размерность признаков: %d",
        X_windows.shape[0],
        lookback_window,
        X_windows.shape[1],
    )

    if X_windows.shape[0] < 2:
        raise ValueError(
            f"Недостаточно окон для обучения: {X_windows.shape[0]} "
            "(нужно минимум 2 для обучения и валидации)."
        )

    # Разделяем данные на обучающую и тестовую выборки
    # Важно: shuffle=False сохраняет временную последовательность данных
    X_train, X_test, y_train, y_test = train_test_split(
        X_windows,
        y_next,
        test_size=0.2,
        random_state=random_state,
        shuffle=False,  # Не перемешиваем, чтобы сохранить временную последовательность
    )

    # Создаём многослойный перцептрон-регрессор для предсказания углов
    # Архитектура адаптирована под больший размер входных признаков (window_size * n_features)
    # Используем больше нейронов в первом слое для обработки расширенного вектора признаков
    model = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        max_iter=1000,
        random_state=random_state,
        early_stopping=True,  # Останавливаем обучение при переобучении
        validation_fraction=0.1,  # 10% данных для валидации
    )
    
    logger.info("Обучение модели регрессии...")
    model.fit(X_train, y_train)

    # Вычисляем предсказания на тестовой выборке для оценки качества модели
    y_pred = model.predict(X_test)
    
    # Вычисляем метрики качества модели:
    # MAE (Mean Absolute Error) - средняя абсолютная ошибка в градусах
    # MSE (Mean Squared Error) - средняя квадратичная ошибка
    # RMSE (Root Mean Squared Error) - корень из MSE
    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    
    # Вычисляем ошибку в процентах (относительно среднего значения углов)
    # Это более понятная метрика для пользователя
    mean_angle = float(np.mean(np.abs(y_test)))
    mae_percent = (mae / mean_angle * 100.0) if mean_angle > 0 else 0.0

    import os

    os.makedirs(MODEL_DIR, exist_ok=True)

    bundle = {
        "model": model,
        "angle_keys": list(CSV_ANGLE_KEYS),
        "lookback_window": lookback_window,  # Сохраняем размер окна для runtime
        "n_features": int(X.shape[1]),  # Количество углов (признаков) в одном тике
    }
    joblib.dump(bundle, MODEL_PATH)

    metrics: Dict[str, Any] = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mae_percent": mae_percent,
        "n_samples": int(X.shape[0]),
        "n_windows": int(X_windows.shape[0]),
        "n_features": int(X.shape[1]),
        "lookback_window": lookback_window,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics


def main() -> None:
    """
    CLI‑оболочка для обучения модели с понятными сообщениями об ошибках.
    """
    # Базовая конфигурация логов для CLI‑запуска
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        metrics = train_once()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except ValueError as exc:
        # Типичные проблемы с датасетом: мало строк, нет колонок и т.п.
        logger.error("Ошибка при подготовке/обучении модели: %s", exc)
        sys.exit(1)
    except Exception as exc:  # на всякий пожарный
        logger.exception("Неожиданная ошибка при обучении модели: %s", exc)
        sys.exit(1)

    logger.info("Модель сохранена в: %s", MODEL_PATH)
    logger.info("Размер окна истории: %d тиков", metrics["lookback_window"])
    logger.info("Средняя абсолютная ошибка (MAE): %.2f градусов", metrics["mae"])
    logger.info("Средняя ошибка в процентах: %.2f%%", metrics["mae_percent"])
    logger.info("RMSE: %.2f градусов", metrics["rmse"])
    logger.info("Число примеров: %s", metrics["n_samples"])
    logger.info("Число окон: %s", metrics["n_windows"])
    logger.info("Число признаков в одном тике: %s", metrics["n_features"])


if __name__ == "__main__":
    main()

