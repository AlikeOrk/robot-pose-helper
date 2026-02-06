"""
Модуль для вычисления углов суставов из ключевых точек позы Mediapipe.

Использует координаты ключевых точек для расчёта углов сгибания/разгибания,
отведения/приведения и других движений суставов человеческого тела.
"""

import math
from typing import Dict, Tuple, Optional

import numpy as np


# Тип для представления 3D точки
Point3D = Tuple[float, float, float]

# Индексы ключевых точек Mediapipe Pose Landmarker
# Полный список: https://google.github.io/mediapipe/solutions/pose.html
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32


def _angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Вычисляет угол между двумя векторами в градусах (используя только XY-плоскость).
    
    :param v1: Первый вектор (3D или 2D)
    :param v2: Второй вектор (3D или 2D)
    :return: Угол в градусах от 0 до 180, или NaN если векторы нулевые
    """
    v1 = v1.astype(float)
    v2 = v2.astype(float)
    v1_2d = v1[:2]
    v2_2d = v2[:2]
    v1_norm = np.linalg.norm(v1_2d)
    v2_norm = np.linalg.norm(v2_2d)
    if v1_norm == 0 or v2_norm == 0:
        return float("nan")
    cos_theta = float(np.dot(v1_2d, v2_2d) / (v1_norm * v2_norm))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    angle_rad = math.acos(cos_theta)
    return math.degrees(angle_rad)


def _get_point(landmarks, idx: int) -> Optional[Point3D]:
    """
    Извлекает координаты ключевой точки из landmarks Mediapipe.
    
    :param landmarks: Список ключевых точек от Mediapipe Pose Landmarker
    :param idx: Индекс ключевой точки
    :return: Кортеж (x, y, z) или None если точка недоступна или невидима
    """
    if landmarks is None:
        return None
    if idx < 0 or idx >= len(landmarks):
        return None
    lm_data = landmarks[idx]
    x, y, z = lm_data.x, lm_data.y, lm_data.z
    # Проверяем видимость точки (Mediapipe предоставляет метрику видимости)
    visibility = getattr(lm_data, "visibility", 1.0)
    if visibility < 0.4:  # Порог видимости для фильтрации невидимых точек
        return None
    return x, y, z


def _angle_at_joint(a: Point3D, b: Point3D, c: Point3D) -> float:
    """
    Вычисляет угол в суставе, образованный тремя точками.
    
    Точка b — это сустав, точки a и c — концы сегментов, образующих угол.
    
    :param a: Первая точка (конец первого сегмента)
    :param b: Вторая точка (сустав, вершина угла)
    :param c: Третья точка (конец второго сегмента)
    :return: Угол в градусах
    """
    a_arr = np.array(a)
    b_arr = np.array(b)
    c_arr = np.array(c)
    v1 = a_arr - b_arr
    v2 = c_arr - b_arr
    return _angle_between_vectors(v1, v2)


def compute_limb_angles(landmarks) -> Dict[str, float]:
    """
    Вычисляет углы всех основных суставов из ключевых точек позы.
    
    Возвращает словарь с углами для различных движений:
    - flex_ext: сгибание/разгибание
    - abd_add: отведение/приведение
    - rot_int_ext: внутренняя/внешняя ротация
    - dorsi_plantar: тыльное/подошвенное сгибание (для голеностопа)
    
    :param landmarks: Список ключевых точек от Mediapipe Pose Landmarker
    :return: Словарь с углами в градусах. Ключи соответствуют названиям движений,
             значения - углы в градусах или -1.0 если угол не может быть вычислен
    """
    if landmarks is None:
        return {}

    def joint_angle(a_idx: int, b_idx: int, c_idx: int) -> float:
        """
        Вспомогательная функция для вычисления угла в суставе по индексам точек.
        
        :param a_idx: Индекс первой точки
        :param b_idx: Индекс точки сустава (вершина угла)
        :param c_idx: Индекс второй точки
        :return: Угол в градусах или -1.0 если точки недоступны
        """
        a = _get_point(landmarks, a_idx)
        b = _get_point(landmarks, b_idx)
        c = _get_point(landmarks, c_idx)
        if a is None or b is None or c is None:
            return -1.0
        return _angle_at_joint(a, b, c)

    def segment_vs_down(start_idx: int, end_idx: int) -> float:
        """
        Вычисляет угол сегмента относительно вертикального направления (вниз).
        
        Используется для определения отведения/приведения суставов.
        
        :param start_idx: Индекс начальной точки сегмента
        :param end_idx: Индекс конечной точки сегмента
        :return: Угол в градусах или -1.0 если точки недоступны
        """
        p1 = _get_point(landmarks, start_idx)
        p2 = _get_point(landmarks, end_idx)
        if p1 is None or p2 is None:
            return -1.0
        v = np.array(p2) - np.array(p1)
        v_down = np.array([0.0, 1.0, 0.0])  # Вертикальный вектор (вниз)
        return _angle_between_vectors(v, v_down)

    def torso_vector() -> Optional[np.ndarray]:
        l_sh = _get_point(landmarks, LEFT_SHOULDER)
        r_sh = _get_point(landmarks, RIGHT_SHOULDER)
        l_hip = _get_point(landmarks, LEFT_HIP)
        r_hip = _get_point(landmarks, RIGHT_HIP)
        if not (l_sh and r_sh and l_hip and r_hip):
            return None
        center_sh = np.array(
            [
                (l_sh[0] + r_sh[0]) / 2,
                (l_sh[1] + r_sh[1]) / 2,
                (l_sh[2] + r_sh[2]) / 2,
            ]
        )
        center_hip = np.array(
            [
                (l_hip[0] + r_hip[0]) / 2,
                (l_hip[1] + r_hip[1]) / 2,
                (l_hip[2] + r_hip[2]) / 2,
            ]
        )
        return center_sh - center_hip

    angles: Dict[str, float] = {}

    angles["shoulder_flex_ext_left"] = joint_angle(LEFT_HIP, LEFT_SHOULDER, LEFT_ELBOW)
    angles["shoulder_flex_ext_right"] = joint_angle(RIGHT_HIP, RIGHT_SHOULDER, RIGHT_ELBOW)
    angles["shoulder_abd_add_left"] = segment_vs_down(LEFT_SHOULDER, LEFT_ELBOW)
    angles["shoulder_abd_add_right"] = segment_vs_down(RIGHT_SHOULDER, RIGHT_ELBOW)
    angles["shoulder_rot_int_ext_left"] = -1.0
    angles["shoulder_rot_int_ext_right"] = -1.0
    angles["elbow_flex_ext_left"] = joint_angle(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
    angles["elbow_flex_ext_right"] = joint_angle(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
    angles["forearm_pron_sup_left"] = -1.0
    angles["forearm_pron_sup_right"] = -1.0
    angles["wrist_flex_ext_left"] = -1.0
    angles["wrist_flex_ext_right"] = -1.0
    angles["wrist_rad_uln_dev_left"] = -1.0
    angles["wrist_rad_uln_dev_right"] = -1.0
    angles["hip_flex_ext_left"] = joint_angle(LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE)
    angles["hip_flex_ext_right"] = joint_angle(RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE)
    angles["hip_abd_add_left"] = segment_vs_down(LEFT_HIP, LEFT_KNEE)
    angles["hip_abd_add_right"] = segment_vs_down(RIGHT_HIP, RIGHT_KNEE)
    angles["hip_rot_int_ext_left"] = -1.0
    angles["hip_rot_int_ext_right"] = -1.0
    angles["knee_flex_ext_left"] = joint_angle(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
    angles["knee_flex_ext_right"] = joint_angle(RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)
    angles["ankle_dorsi_plantar_left"] = joint_angle(
        LEFT_KNEE, LEFT_ANKLE, LEFT_FOOT_INDEX
    )
    angles["ankle_dorsi_plantar_right"] = joint_angle(
        RIGHT_KNEE, RIGHT_ANKLE, RIGHT_FOOT_INDEX
    )
    angles["ankle_inv_evers_left"] = -1.0
    angles["ankle_inv_evers_right"] = -1.0
    t_vec = torso_vector()
    if t_vec is not None:
        v_down = np.array([0.0, 1.0, 0.0])
        torso_angle = _angle_between_vectors(t_vec, v_down)
    else:
        torso_angle = -1.0
    angles["torso_flex_ext"] = torso_angle
    angles["torso_lat_bend"] = -1.0
    angles["torso_rot"] = -1.0
    nose = _get_point(landmarks, NOSE)
    l_sh = _get_point(landmarks, LEFT_SHOULDER)
    r_sh = _get_point(landmarks, RIGHT_SHOULDER)
    if nose and l_sh and r_sh:
        center_sh = np.array(
            [
                (l_sh[0] + r_sh[0]) / 2,
                (l_sh[1] + r_sh[1]) / 2,
                (l_sh[2] + r_sh[2]) / 2,
            ]
        )
        neck_vec = np.array(nose) - center_sh
        v_down = np.array([0.0, 1.0, 0.0])
        neck_flex_angle = _angle_between_vectors(neck_vec, v_down)
    else:
        neck_flex_angle = -1.0
    angles["neck_flex_ext"] = neck_flex_angle
    angles["neck_lat_bend"] = -1.0
    angles["neck_rot"] = -1.0
    return angles

