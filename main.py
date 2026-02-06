"""
Точка входа проекта.

По умолчанию просто запускает сбор данных (capture_angles).
Для обучения и runtime лучше вызывать модули явно:
  python -m model_training.capture_angles
  python -m model_training.train_model
  python -m model_runtime.realtime_predict
"""

from model_training.capture_angles import main as capture_angles_main


if __name__ == "__main__":
    capture_angles_main()

