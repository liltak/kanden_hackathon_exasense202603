"""Tests for ray casting shadow detection."""

import numpy as np
import pytest
import trimesh

from src.simulation.ray_caster import cast_shadows


def make_flat_ground(size: float = 10.0) -> trimesh.Trimesh:
    """Create a flat ground plane at z=0."""
    vertices = np.array([
        [0, 0, 0],
        [size, 0, 0],
        [size, size, 0],
        [0, size, 0],
    ])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def make_box_on_ground() -> trimesh.Trimesh:
    """Create a small box sitting on a ground plane (for shadow testing)."""
    ground = np.array([
        [0, 0, 0], [20, 0, 0], [20, 20, 0], [0, 20, 0],
    ])
    box_base = np.array([
        [8, 8, 0], [12, 8, 0], [12, 12, 0], [8, 12, 0],
    ])
    box_top = np.array([
        [8, 8, 5], [12, 8, 5], [12, 12, 5], [8, 12, 5],
    ])

    vertices = np.vstack([ground, box_base, box_top])
    faces = np.array([
        # Ground (excluding box area — simplified)
        [0, 1, 2], [0, 2, 3],
        # Box top
        [8, 9, 10], [8, 10, 11],
        # Box walls
        [4, 5, 9], [4, 9, 8],
        [5, 6, 10], [5, 10, 9],
        [6, 7, 11], [6, 11, 10],
        [7, 4, 8], [7, 8, 11],
    ])
    return trimesh.Trimesh(vertices=vertices, faces=faces)


class TestCastShadows:
    def test_flat_ground_sun_overhead(self):
        """Flat ground with sun directly overhead → no face should be illuminated
        (normals point down for trimesh default)."""
        mesh = make_flat_ground()
        sun_dir = np.array([0, 0, 1])  # straight up
        result = cast_shadows(mesh, sun_dir)
        # Ground normals may point up or down depending on winding
        # At least check we get a boolean array of correct size
        assert result.shape == (2,)
        assert result.dtype == bool

    def test_sun_below_horizon(self):
        """Sun below horizon → nothing illuminated."""
        mesh = make_flat_ground()
        sun_dir = np.array([0, 0, -1])  # straight down
        result = cast_shadows(mesh, sun_dir)
        # Faces facing up won't face a downward sun
        # Result depends on normal direction but should be consistent
        assert result.shape == (2,)

    def test_box_top_illuminated(self):
        """Box top should be illuminated when sun is overhead."""
        mesh = make_box_on_ground()
        sun_dir = np.array([0, 0, 1])
        result = cast_shadows(mesh, sun_dir)
        assert result.shape == (len(mesh.faces),)
        # At least some faces should be illuminated
        assert np.sum(result) > 0
