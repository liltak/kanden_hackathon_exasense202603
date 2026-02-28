"""Demo factory mesh generator.

Creates realistic multi-building factory complexes for demonstration purposes.
"""

import numpy as np
import trimesh


def _make_box(
    x: float, y: float, z: float,
    w: float, d: float, h: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a box mesh (8 vertices, 12 faces)."""
    v = np.array([
        [x, y, z], [x + w, y, z], [x + w, y + d, z], [x, y + d, z],
        [x, y, z + h], [x + w, y, z + h], [x + w, y + d, z + h], [x, y + d, z + h],
    ])
    f = np.array([
        [0, 2, 1], [0, 3, 2],  # bottom
        [4, 5, 6], [4, 6, 7],  # top
        [0, 1, 5], [0, 5, 4],  # front
        [2, 3, 7], [2, 7, 6],  # back
        [0, 4, 7], [0, 7, 3],  # left
        [1, 2, 6], [1, 6, 5],  # right
    ])
    return v, f


def _make_pitched_roof_building(
    x: float, y: float,
    length: float, width: float,
    wall_height: float, ridge_height: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a building with pitched (gabled) roof."""
    v = np.array([
        [x, y, 0], [x + length, y, 0],
        [x + length, y + width, 0], [x, y + width, 0],
        [x, y, wall_height], [x + length, y, wall_height],
        [x + length, y + width, wall_height], [x, y + width, wall_height],
        [x, y + width / 2, ridge_height],
        [x + length, y + width / 2, ridge_height],
    ])
    f = np.array([
        [0, 2, 1], [0, 3, 2],           # floor
        [0, 1, 5], [0, 5, 4],           # front wall
        [2, 3, 7], [2, 7, 6],           # back wall
        [0, 3, 8], [0, 8, 4], [3, 7, 8],  # left gable
        [1, 2, 9], [1, 9, 5], [2, 6, 9],  # right gable
        [4, 5, 9], [4, 9, 8],           # front roof slope
        [6, 7, 8], [6, 8, 9],           # back roof slope
    ])
    return v, f


def _make_flat_roof_building(
    x: float, y: float,
    length: float, width: float,
    height: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a building with flat roof."""
    return _make_box(x, y, 0, length, width, height)


def _make_sawtooth_roof_building(
    x: float, y: float,
    length: float, width: float,
    wall_height: float, peak_height: float,
    n_teeth: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a building with sawtooth (factory-style) roof."""
    all_vertices = []
    all_faces = []
    tooth_width = width / n_teeth
    offset = 0

    # Base walls
    base_v = np.array([
        [x, y, 0], [x + length, y, 0],
        [x + length, y + width, 0], [x, y + width, 0],
        [x, y, wall_height], [x + length, y, wall_height],
        [x + length, y + width, wall_height], [x, y + width, wall_height],
    ])
    base_f = np.array([
        [0, 2, 1], [0, 3, 2],     # floor
        [0, 1, 5], [0, 5, 4],     # front
        [2, 3, 7], [2, 7, 6],     # back
        [0, 4, 7], [0, 7, 3],     # left
        [1, 2, 6], [1, 6, 5],     # right
    ])
    all_vertices.append(base_v)
    all_faces.append(base_f)
    offset = len(base_v)

    for i in range(n_teeth):
        ty = y + i * tooth_width
        v = np.array([
            [x, ty, wall_height],
            [x + length, ty, wall_height],
            [x + length, ty + tooth_width, wall_height],
            [x, ty + tooth_width, wall_height],
            [x, ty, peak_height],
            [x + length, ty, peak_height],
        ])
        f = np.array([
            [0, 1, 5], [0, 5, 4],     # vertical face (south, glass)
            [1, 2, 3], [1, 3, 5],     # sloped roof face
            [3, 4, 5],                  # triangle
            [0, 4, 3],                  # triangle
        ]) + offset
        all_vertices.append(v)
        all_faces.append(f)
        offset += len(v)

    vertices = np.vstack(all_vertices)
    faces = np.vstack(all_faces)
    return vertices, faces


def create_factory_complex() -> trimesh.Trimesh:
    """Create a multi-building factory complex.

    Layout:
    - Main production hall (large, sawtooth roof)
    - Office building (medium, flat roof)
    - Warehouse (medium, pitched roof)
    - Utility building (small, flat roof)
    - Ground plane
    """
    all_vertices = []
    all_faces = []
    offset = 0

    buildings = [
        ("sawtooth", {
            "x": 5, "y": 5, "length": 60, "width": 30,
            "wall_height": 10, "peak_height": 14, "n_teeth": 3,
        }),
        ("flat", {
            "x": 5, "y": 45, "length": 25, "width": 15, "height": 12,
        }),
        ("pitched", {
            "x": 70, "y": 5, "length": 30, "width": 20,
            "wall_height": 8, "ridge_height": 12,
        }),
        ("flat", {
            "x": 70, "y": 30, "length": 15, "width": 10, "height": 6,
        }),
    ]

    for btype, params in buildings:
        if btype == "sawtooth":
            v, f = _make_sawtooth_roof_building(**params)
        elif btype == "pitched":
            v, f = _make_pitched_roof_building(**params)
        else:
            v, f = _make_flat_roof_building(**params)
        all_vertices.append(v)
        all_faces.append(f + offset)
        offset += len(v)

    vertices = np.vstack(all_vertices)
    faces = np.vstack(all_faces)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    return mesh


def create_simple_factory() -> trimesh.Trimesh:
    """Create a single pitched-roof factory building."""
    v, f = _make_pitched_roof_building(0, 0, 40, 20, 8, 12)
    return trimesh.Trimesh(vertices=v, faces=f, process=True)
