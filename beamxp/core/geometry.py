from __future__ import annotations

import math
import re
from collections.abc import Iterable

from beamxp.core.constants import NUMBER_RE


def brg_rotation_matrix3(brg_rad: Iterable[float]) -> list[list[float]]:
    """Engine getBaseRotationGlobal() euler (radians) -> rest rotation 3x3."""
    x, y, z = (math.degrees(float(v)) for v in brg_rad)
    matrix = identity_matrix()
    for next_matrix in (rotation_y_matrix(y), rotation_z_matrix(z), rotation_x_matrix(x)):
        matrix = multiply_matrix(matrix, next_matrix)
    return rotation_transpose_matrix3(matrix3_from_matrix4(matrix))


def rotation_transpose_matrix3(matrix: list[list[float]]) -> list[list[float]]:
    return [[matrix[col][row] for col in range(3)] for row in range(3)]


def vector_subtract(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def cross_product(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def normalize_vector(value: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2])
    if length <= 1e-12:
        return (0.0, 0.0, 0.0)
    return (value[0] / length, value[1] / length, value[2] / length)


def identity_matrix() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def multiply_matrix(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [
            sum(a[row][idx] * b[idx][col] for idx in range(4))
            for col in range(4)
        ]
        for row in range(4)
    ]


def translation_matrix(values: tuple[float, float, float]) -> list[list[float]]:
    out = identity_matrix()
    out[0][3], out[1][3], out[2][3] = values
    return out


def scale_matrix(values: tuple[float, float, float]) -> list[list[float]]:
    out = identity_matrix()
    out[0][0], out[1][1], out[2][2] = values
    return out


def mirror_x_matrix4() -> list[list[float]]:
    out = identity_matrix()
    out[0][0] = -1.0
    return out


def rotation_x_matrix(degrees: float) -> list[list[float]]:
    angle = math.radians(degrees)
    c = math.cos(angle)
    s = math.sin(angle)
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, c, -s, 0.0],
        [0.0, s, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def rotation_y_matrix(degrees: float) -> list[list[float]]:
    angle = math.radians(degrees)
    c = math.cos(angle)
    s = math.sin(angle)
    return [
        [c, 0.0, s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s, 0.0, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def rotation_z_matrix(degrees: float) -> list[list[float]]:
    angle = math.radians(degrees)
    c = math.cos(angle)
    s = math.sin(angle)
    return [
        [c, -s, 0.0, 0.0],
        [s, c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


PROP_VECTOR_RE = re.compile(
    rf'\{{\s*"x"\s*:\s*(?P<x>{NUMBER_RE})\s*,?\s*'
    rf'"y"\s*:\s*(?P<y>{NUMBER_RE})\s*,\s*"z"\s*:\s*(?P<z>{NUMBER_RE})\s*\}}'
)


def prop_row_vector_objects(row: str) -> list[tuple[float, float, float]]:
    return [
        (float(match.group("x")), float(match.group("y")), float(match.group("z")))
        for match in PROP_VECTOR_RE.finditer(row)
    ]


def matrix3_from_matrix4(matrix: list[list[float]]) -> list[list[float]]:
    return [[matrix[row][col] for col in range(3)] for row in range(3)]


def multiply_matrix3(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [
            sum(a[row][idx] * b[idx][col] for idx in range(3))
            for col in range(3)
        ]
        for row in range(3)
    ]


def euler_matrix3(degrees: tuple[float, float, float]) -> list[list[float]]:
    matrix = identity_matrix()
    for next_matrix in (
        rotation_z_matrix(degrees[2]),
        rotation_y_matrix(degrees[1]),
        rotation_x_matrix(degrees[0]),
    ):
        matrix = multiply_matrix(matrix, next_matrix)
    return matrix3_from_matrix4(matrix)


def prop_base_rotation_matrix3(degrees: tuple[float, float, float]) -> list[list[float]]:
    # Game prop baseRotation euler order is -X -Z +Y intrinsic
    # (lua/common/jbeam/sections/meshs.lua), i.e. B = Rx(-x)*Rz(-z)*Ry(+y).
    matrix = identity_matrix()
    for next_matrix in (
        rotation_x_matrix(-degrees[0]),
        rotation_z_matrix(-degrees[2]),
        rotation_y_matrix(degrees[1]),
    ):
        matrix = multiply_matrix(matrix, next_matrix)
    return matrix3_from_matrix4(matrix)


def euler_yzx_from_matrix3(matrix: list[list[float]]) -> tuple[float, float, float]:
    # Decompose M = Ry(y)*Rz(z)*Rx(x), the game's baseRotationGlobal order.
    sz = max(-1.0, min(1.0, matrix[1][0]))
    z = math.asin(sz)
    cz = math.cos(z)
    if abs(cz) > 1e-8:
        x = math.atan2(-matrix[1][2], matrix[1][1])
        y = math.atan2(-matrix[2][0], matrix[0][0])
    else:
        x = math.atan2(matrix[2][1], matrix[2][2])
        y = 0.0
    return (math.degrees(x), math.degrees(y), math.degrees(z))


def matrix3_from_axes(
    axis_x: tuple[float, float, float],
    axis_y: tuple[float, float, float],
    axis_z: tuple[float, float, float],
) -> list[list[float]]:
    return [
        [axis_x[0], axis_y[0], axis_z[0]],
        [axis_x[1], axis_y[1], axis_z[1]],
        [axis_x[2], axis_y[2], axis_z[2]],
    ]


def mirror_rotation_matrix_x(matrix: list[list[float]]) -> list[list[float]]:
    out = [row[:] for row in matrix]
    for col in range(3):
        out[0][col] *= -1
    for row in range(3):
        out[row][0] *= -1
    return out


def euler_from_matrix3(matrix: list[list[float]]) -> tuple[float, float, float]:
    sy = max(-1.0, min(1.0, -matrix[2][0]))
    y = math.asin(sy)
    cy = math.cos(y)
    if abs(cy) > 1e-8:
        x = math.atan2(matrix[2][1], matrix[2][2])
        z = math.atan2(matrix[1][0], matrix[0][0])
    else:
        x = math.atan2(-matrix[1][2], matrix[1][1])
        z = 0.0
    return (math.degrees(x), math.degrees(y), math.degrees(z))


def sign_number(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def clamp_value(value: float, lo: float, hi: float) -> float:
    return max(min(lo, hi), min(max(lo, hi), value))


def rotation_transpose_matrix4(matrix: list[list[float]]) -> list[list[float]]:
    out = identity_matrix()
    for row in range(3):
        for col in range(3):
            out[row][col] = matrix[col][row]
    return out


def matrix4_flat(matrix: list[list[float]]) -> list[float]:
    return [float(matrix[row][col]) for row in range(4) for col in range(4)]
