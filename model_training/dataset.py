from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .config import DATA_CSV_PATH, CSV_ANGLE_KEYS, TARGET_COLUMN


def load_dataset(require_labels: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Загружает датасет из data.csv и возвращает признаки и (опционально) метки.

    :param require_labels: если True, требуем наличие колонки с целевой меткой.
    :raises FileNotFoundError: если файл data.csv не найден.
    :raises ValueError: если не хватает колонок с углами или меток/строк.
    """
    try:
        df = pd.read_csv(DATA_CSV_PATH)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Файл с данными не найден: {DATA_CSV_PATH}. "
            f"Сначала запишите данные через 'python -m model_training.capture_angles'."
        ) from exc
    except Exception as exc:  # на случай битого CSV
        raise ValueError(
            f"Не удалось прочитать файл данных {DATA_CSV_PATH}: {exc}"
        ) from exc

    if df.empty:
        raise ValueError(
            f"Файл данных {DATA_CSV_PATH} пуст. "
            "Нужно собрать хотя бы несколько десятков кадров."
        )

    missing = [c for c in CSV_ANGLE_KEYS if c not in df.columns]
    if missing:
        raise ValueError(f"В data.csv отсутствуют колонки с углами: {missing}")

    has_labels = TARGET_COLUMN in df.columns
    if not has_labels and require_labels:
        raise ValueError(
            f"В data.csv нет целевой колонки '{TARGET_COLUMN}'. "
            "Либо добавьте метки в CSV, либо вызовите "
            "load_dataset(require_labels=False)."
        )

    if has_labels:
        df_clean = df.dropna(subset=CSV_ANGLE_KEYS + [TARGET_COLUMN])
    else:
        df_clean = df.dropna(subset=CSV_ANGLE_KEYS)

    if df_clean.empty:
        raise ValueError(
            "После удаления строк с пропущенными значениями датасет оказался пустым. "
            "Соберите больше данных или проверьте, что углы и метки записываются корректно."
        )

    X = df_clean[CSV_ANGLE_KEYS].to_numpy(dtype=float)
    if has_labels:
        y: Optional[np.ndarray] = df_clean[TARGET_COLUMN].astype(str).to_numpy()
    else:
        y = None

    return X, y

