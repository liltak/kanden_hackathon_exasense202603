"""Mesh post-processing pipeline (Phase 2).

Extracts meshes from point clouds and depth maps, performs surface
reconstruction, validates normals, transforms coordinates to real-world
scale, and segments building components (roof/wall/ground).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

logger = logging.getLogger(__name__)
console = Console()


class MeshMethod(str, Enum):
    TSDF = "tsdf"
    POISSON = "poisson"
    MARCHING_CUBES = "marching_cubes"
    NKSR = "nksr"


class FaceLabel(str, Enum):
    ROOF = "roof"
    WALL = "wall"
    GROUND = "ground"
    UNKNOWN = "unknown"


@dataclass
class GeoReference:
    """Geographic reference for coordinate transformation."""

    latitude: float
    longitude: float
    altitude: float = 0.0
    compass_bearing_deg: float = 0.0  # degrees from north, clockwise
    scale_factor: float = 1.0  # local units to meters


@dataclass
class MeshStats:
    """Statistics about a processed mesh."""

    num_vertices: int = 0
    num_faces: int = 0
    num_normals: int = 0
    surface_area_m2: float = 0.0
    bounding_box_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bounding_box_max: tuple[float, float, float] = (0.0, 0.0, 0.0)
    num_roof_faces: int = 0
    num_wall_faces: int = 0
    num_ground_faces: int = 0


@dataclass
class ProcessingResult:
    """Result from the mesh processing pipeline."""

    output_path: Path | None = None
    method: str = ""
    stats: MeshStats = field(default_factory=MeshStats)
    face_labels: np.ndarray | None = None  # (n_faces,) int labels
    timing_s: dict = field(default_factory=dict)
    total_time_s: float = 0.0


def _import_open3d():
    """Import Open3D with a helpful error message if not installed."""
    try:
        import open3d as o3d
        return o3d
    except ImportError:
        raise ImportError(
            "Open3D is required for mesh processing. Install with:\n"
            "  pip install open3d\n"
            "or:\n"
            "  uv add open3d"
        )


class MeshProcessor:
    """Mesh post-processing pipeline with chainable methods.

    Usage:
        processor = MeshProcessor()
        result = (
            processor
            .load_point_cloud("point_cloud.ply")
            .preprocess_point_cloud()
            .extract_mesh(method="poisson", depth=7)
            .clean_mesh()
            .smooth_mesh()
            .decimate_mesh(target_faces=20000)
            .fix_normals()
            .transform_coordinates(geo_ref)
            .segment_faces()
            .save("output.obj")
        )
    """

    def __init__(self):
        self._o3d = _import_open3d()
        self._pcd: object | None = None  # o3d.geometry.PointCloud
        self._mesh: object | None = None  # o3d.geometry.TriangleMesh
        self._depth_maps: list[np.ndarray] = []
        self._camera_intrinsics: list[np.ndarray] = []
        self._camera_extrinsics: list[np.ndarray] = []
        self._face_labels: np.ndarray | None = None
        self._geo_ref: GeoReference | None = None
        self._timings: dict[str, float] = {}

    @property
    def mesh(self):
        """Access the current Open3D mesh."""
        return self._mesh

    @property
    def point_cloud(self):
        """Access the current Open3D point cloud."""
        return self._pcd

    # --- Loading ---

    def load_point_cloud(self, path: Path | str) -> MeshProcessor:
        """Load a point cloud from PLY/PCD file.

        Args:
            path: Path to point cloud file.

        Returns:
            self for chaining.
        """
        o3d = self._o3d
        path = Path(path)
        t0 = time.perf_counter()

        if not path.exists():
            raise FileNotFoundError(f"Point cloud not found: {path}")

        self._pcd = o3d.io.read_point_cloud(str(path))
        n_points = len(np.asarray(self._pcd.points))
        self._timings["load_point_cloud"] = time.perf_counter() - t0

        console.print(f"  Loaded point cloud: {n_points:,} points ({path.name})")
        return self

    def load_point_cloud_from_arrays(
        self,
        points: np.ndarray,
        colors: np.ndarray | None = None,
        normals: np.ndarray | None = None,
    ) -> MeshProcessor:
        """Load point cloud from numpy arrays.

        Args:
            points: (N, 3) array of xyz coordinates.
            colors: Optional (N, 3) array of RGB in [0, 1].
            normals: Optional (N, 3) array of normal vectors.

        Returns:
            self for chaining.
        """
        o3d = self._o3d
        t0 = time.perf_counter()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))

        self._pcd = pcd
        self._timings["load_point_cloud"] = time.perf_counter() - t0
        console.print(f"  Loaded point cloud from arrays: {len(points):,} points")
        return self

    def load_depth_maps(
        self,
        depth_dir: Path | str,
        camera_poses_path: Path | str,
        image_size: tuple[int, int] | None = None,
    ) -> MeshProcessor:
        """Load depth maps and camera parameters for TSDF or marching cubes.

        Args:
            depth_dir: Directory containing depth .npy files.
            camera_poses_path: Path to camera_poses.json.
            image_size: Optional (width, height) to override.

        Returns:
            self for chaining.
        """
        t0 = time.perf_counter()
        depth_dir = Path(depth_dir)
        camera_poses_path = Path(camera_poses_path)

        with open(camera_poses_path) as f:
            poses_data = json.load(f)

        cameras = poses_data.get("cameras", poses_data) if isinstance(poses_data, dict) else poses_data
        if isinstance(cameras, dict):
            cameras = cameras.get("cameras", [])

        self._depth_maps = []
        self._camera_intrinsics = []
        self._camera_extrinsics = []

        for cam in cameras:
            name = cam["image_name"]
            stem = Path(name).stem
            depth_path = depth_dir / f"{stem}_depth.npy"

            if not depth_path.exists():
                logger.warning(f"Depth map not found: {depth_path}, skipping")
                continue

            depth = np.load(depth_path)
            self._depth_maps.append(depth)

            extrinsic = np.array(cam["extrinsic"], dtype=np.float64)
            if extrinsic.shape == (3, 4):
                ext_4x4 = np.eye(4, dtype=np.float64)
                ext_4x4[:3, :] = extrinsic
                extrinsic = ext_4x4
            self._camera_extrinsics.append(extrinsic)

            if "intrinsic" in cam and cam["intrinsic"] is not None:
                intrinsic = np.array(cam["intrinsic"], dtype=np.float64)
            else:
                w = cam.get("width", depth.shape[1])
                h = cam.get("height", depth.shape[0])
                fx = fy = max(w, h)
                intrinsic = np.array([
                    [fx, 0, w / 2],
                    [0, fy, h / 2],
                    [0, 0, 1],
                ], dtype=np.float64)
            self._camera_intrinsics.append(intrinsic)

        self._timings["load_depth_maps"] = time.perf_counter() - t0
        console.print(f"  Loaded {len(self._depth_maps)} depth maps")
        return self

    # --- Point Cloud Preprocessing ---

    def preprocess_point_cloud(
        self,
        voxel_size: float | None = None,
        stat_nb_neighbors: int = 20,
        stat_std_ratio: float = 2.0,
        radius_nb_points: int = 16,
        radius: float | None = None,
        target_points: int = 200_000,
    ) -> MeshProcessor:
        """Voxel downsample + 2-stage outlier removal for cleaner reconstruction.

        If voxel_size is None, it is auto-computed from the point cloud extent
        to keep approximately target_points after downsampling. This handles
        both real-world scale (meters) and normalized coordinates (VGGT output).

        Args:
            voxel_size: Voxel size for downsampling. None = auto.
            stat_nb_neighbors: Neighbors for statistical outlier removal.
            stat_std_ratio: Standard deviation ratio for statistical filter.
            radius_nb_points: Min points in radius for radius outlier removal.
            radius: Search radius for radius outlier removal. None = auto.
            target_points: Approximate point count target for auto voxel sizing.

        Returns:
            self for chaining.
        """
        o3d = self._o3d
        t0 = time.perf_counter()

        if self._pcd is None:
            raise ValueError("No point cloud loaded. Call load_point_cloud() first.")

        points = np.asarray(self._pcd.points)
        n_before = len(points)

        # Auto-compute voxel_size from point cloud spatial density
        if voxel_size is None:
            extent = points.max(axis=0) - points.min(axis=0)
            if n_before > target_points:
                # Sample a subset to estimate average nearest-neighbor distance
                sample_n = min(10_000, n_before)
                rng = np.random.default_rng(42)
                sample_idx = rng.choice(n_before, sample_n, replace=False)
                sample_pts = points[sample_idx]

                # Use KDTree for fast NN lookup
                from scipy.spatial import cKDTree
                tree = cKDTree(sample_pts)
                dists, _ = tree.query(sample_pts, k=2)  # k=2: self + nearest
                avg_nn_dist = float(np.median(dists[:, 1]))

                # Voxel size = multiple of NN distance to achieve target reduction
                reduction_ratio = n_before / target_points
                voxel_size = avg_nn_dist * (reduction_ratio ** (1.0 / 3.0))
                # Clamp: at least 3x NN distance, at most 1% of extent
                voxel_size = max(voxel_size, avg_nn_dist * 3.0)
                voxel_size = min(voxel_size, float(extent.max()) * 0.01)
                console.print(f"    Auto voxel_size: {voxel_size:.6f} (avg_nn: {avg_nn_dist:.6f}, extent: {extent.round(3)})")
            else:
                voxel_size = float(extent.max()) * 0.001
                console.print(f"    Auto voxel_size: {voxel_size:.6f} (sparse cloud, extent: {extent.round(3)})")

        # Auto-compute radius from voxel_size if not specified
        if radius is None:
            radius = voxel_size * 2.5

        # Stage 1: Voxel downsampling for spatial uniformity
        self._pcd = self._pcd.voxel_down_sample(voxel_size)
        n_after_voxel = len(np.asarray(self._pcd.points))
        console.print(f"    Voxel downsample ({voxel_size:.6f}): {n_before:,} → {n_after_voxel:,}")

        # Stage 2: Statistical outlier removal
        self._pcd, _ = self._pcd.remove_statistical_outlier(
            nb_neighbors=stat_nb_neighbors, std_ratio=stat_std_ratio
        )
        n_after_stat = len(np.asarray(self._pcd.points))
        console.print(f"    Statistical outlier removal: {n_after_voxel:,} → {n_after_stat:,}")

        # Stage 3: Radius outlier removal (removes isolated points)
        self._pcd, _ = self._pcd.remove_radius_outlier(
            nb_points=radius_nb_points, radius=radius
        )
        n_after_radius = len(np.asarray(self._pcd.points))
        console.print(f"    Radius outlier removal: {n_after_stat:,} → {n_after_radius:,}")

        self._timings["preprocess_point_cloud"] = time.perf_counter() - t0
        console.print(f"  Preprocessing: {n_before:,} → {n_after_radius:,} points")
        return self

    # --- Mesh Extraction ---

    def extract_mesh(
        self,
        method: str | MeshMethod = MeshMethod.POISSON,
        **kwargs,
    ) -> MeshProcessor:
        """Extract a triangle mesh using the specified method.

        Args:
            method: Extraction method ('tsdf', 'poisson', 'marching_cubes', 'nksr').
            **kwargs: Method-specific parameters.

        Returns:
            self for chaining.
        """
        method = MeshMethod(method)
        console.print(f"  Extracting mesh via {method.value}...")

        if method == MeshMethod.TSDF:
            self._extract_tsdf(**kwargs)
        elif method == MeshMethod.POISSON:
            self._extract_poisson(**kwargs)
        elif method == MeshMethod.MARCHING_CUBES:
            self._extract_marching_cubes(**kwargs)
        elif method == MeshMethod.NKSR:
            self._extract_nksr(**kwargs)

        if self._mesh is not None:
            vertices = np.asarray(self._mesh.vertices)
            faces = np.asarray(self._mesh.triangles)
            console.print(f"  Mesh: {len(vertices):,} vertices, {len(faces):,} faces")

        return self

    def _extract_tsdf(
        self,
        voxel_length: float = 0.02,
        sdf_trunc: float = 0.06,
        depth_scale: float = 1.0,
        depth_trunc: float = 5.0,
    ) -> None:
        """Extract mesh via TSDF fusion from depth maps."""
        o3d = self._o3d
        t0 = time.perf_counter()

        if not self._depth_maps:
            raise ValueError("No depth maps loaded. Call load_depth_maps() first.")

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_length,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

        for i, (depth, intrinsic, extrinsic) in enumerate(
            zip(self._depth_maps, self._camera_intrinsics, self._camera_extrinsics)
        ):
            h, w = depth.shape[:2]
            o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=w,
                height=h,
                fx=intrinsic[0, 0],
                fy=intrinsic[1, 1],
                cx=intrinsic[0, 2],
                cy=intrinsic[1, 2],
            )

            depth_o3d = o3d.geometry.Image((depth * depth_scale).astype(np.float32))

            # Create dummy color image
            color = np.ones((h, w, 3), dtype=np.uint8) * 200
            color_o3d = o3d.geometry.Image(color)

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d,
                depth_o3d,
                depth_scale=1.0,
                depth_trunc=depth_trunc,
                convert_rgb_to_intensity=False,
            )

            volume.integrate(rgbd, o3d_intrinsic, np.linalg.inv(extrinsic))

        self._mesh = volume.extract_triangle_mesh()
        self._mesh.compute_vertex_normals()
        self._timings["extract_tsdf"] = time.perf_counter() - t0

    def _extract_poisson(
        self,
        depth: int = 7,
        width: float = 0.0,
        scale: float = 1.1,
        linear_fit: bool = True,
        density_threshold_quantile: float = 0.15,
    ) -> None:
        """Extract mesh via Poisson surface reconstruction."""
        o3d = self._o3d
        t0 = time.perf_counter()

        if self._pcd is None:
            raise ValueError("No point cloud loaded. Call load_point_cloud() first.")

        pcd = self._pcd

        # Estimate normals if not present
        if not pcd.has_normals():
            console.print("    Estimating normals...")
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            n_points = len(np.asarray(pcd.points))
            if n_points > 30_000:
                # O(n) camera-based orientation for large point clouds
                console.print(f"    Orient normals (camera-based, {n_points:,} points)...")
                pcd.orient_normals_towards_camera_location(
                    camera_location=np.array([0.0, 0.0, 10.0])
                )
            else:
                # O(n²) tangent plane for small point clouds (higher quality)
                console.print(f"    Orient normals (tangent plane, {n_points:,} points)...")
                pcd.orient_normals_consistent_tangent_plane(k=15)

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, width=width, scale=scale, linear_fit=linear_fit
        )

        # Remove low-density vertices (cleanup noisy boundary)
        densities = np.asarray(densities)
        threshold = np.quantile(densities, density_threshold_quantile)
        vertices_to_remove = densities < threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)

        self._mesh = mesh
        self._mesh.compute_vertex_normals()
        self._timings["extract_poisson"] = time.perf_counter() - t0

    def _extract_nksr(
        self,
        device: str = "cuda:0",
        detail_level: float = 1.0,
    ) -> None:
        """Extract mesh via Neural Kernel Surface Reconstruction (GPU).

        NKSR uses learned neural kernels for adaptive geometry fitting,
        producing higher-quality meshes than Poisson for noisy/sparse data.
        Requires GPU with ~4-8GB VRAM.

        Args:
            device: CUDA device for reconstruction.
            detail_level: Controls mesh detail (default 1.0).
        """
        import nksr
        import torch

        o3d = self._o3d
        t0 = time.perf_counter()

        if self._pcd is None:
            raise ValueError("No point cloud loaded. Call load_point_cloud() first.")

        points = np.asarray(self._pcd.points).astype(np.float32)

        # NKSR can work without normals, but quality is better with them
        if self._pcd.has_normals():
            normals = np.asarray(self._pcd.normals).astype(np.float32)
        else:
            console.print("    Estimating normals for NKSR...")
            self._pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            self._pcd.orient_normals_towards_camera_location(
                camera_location=np.array([0.0, 0.0, 10.0])
            )
            normals = np.asarray(self._pcd.normals).astype(np.float32)

        console.print(f"    NKSR: {len(points):,} points → device={device}")

        # Run NKSR reconstruction on GPU
        input_xyz = torch.from_numpy(points).to(device)
        input_normal = torch.from_numpy(normals).to(device)

        reconstructor = nksr.Reconstructor(device)
        field = reconstructor.reconstruct(
            input_xyz,
            input_normal,
            detail_level=detail_level,
        )
        nksr_mesh = field.extract_dual_mesh(mise_iter=1)

        # Convert to Open3D TriangleMesh
        verts = nksr_mesh.v.cpu().numpy().astype(np.float64)
        faces = nksr_mesh.f.cpu().numpy().astype(np.int32)

        console.print(f"    NKSR result: {len(verts):,} vertices, {len(faces):,} faces")

        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts),
            o3d.utility.Vector3iVector(faces),
        )
        mesh.compute_vertex_normals()

        # Cleanup GPU memory
        del reconstructor, field, nksr_mesh, input_xyz, input_normal
        torch.cuda.empty_cache()

        self._mesh = mesh
        self._timings["extract_nksr"] = time.perf_counter() - t0

    def _extract_marching_cubes(
        self,
        voxel_size: float = 0.05,
        depth_trunc: float = 5.0,
    ) -> None:
        """Extract mesh via marching cubes from depth maps.

        Converts depth maps to a voxel grid, then runs marching cubes.
        """
        o3d = self._o3d
        t0 = time.perf_counter()

        if not self._depth_maps:
            raise ValueError("No depth maps loaded. Call load_depth_maps() first.")

        # Unproject depth maps to a combined point cloud
        all_points = []
        for depth, intrinsic, extrinsic in zip(
            self._depth_maps, self._camera_intrinsics, self._camera_extrinsics
        ):
            h, w = depth.shape[:2]
            fx, fy = intrinsic[0, 0], intrinsic[1, 1]
            cx, cy = intrinsic[0, 2], intrinsic[1, 2]

            u, v = np.meshgrid(np.arange(w), np.arange(h))
            z = depth
            valid = (z > 0) & (z < depth_trunc)

            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            pts_cam = np.stack([x[valid], y[valid], z[valid]], axis=-1)  # (M, 3)

            R = extrinsic[:3, :3]
            t = extrinsic[:3, 3]
            pts_world = (R @ pts_cam.T).T + t
            all_points.append(pts_world)

        combined = np.concatenate(all_points, axis=0)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(combined)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 3, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)

        # Use ball pivoting as a marching-cubes-like approach on point cloud
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radii = [avg_dist * 1.5, avg_dist * 2.0, avg_dist * 3.0]

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )

        self._mesh = mesh
        self._mesh.compute_vertex_normals()
        self._timings["extract_marching_cubes"] = time.perf_counter() - t0

    # --- Mesh Post-Processing ---

    def clean_mesh(
        self,
        min_triangle_area: float = 1e-8,
        min_component_ratio: float = 0.01,
    ) -> MeshProcessor:
        """Remove degenerate triangles and small disconnected components.

        Args:
            min_triangle_area: Triangles smaller than this are removed.
            min_component_ratio: Components with fewer faces than
                (total * ratio) are removed.

        Returns:
            self for chaining.
        """
        o3d = self._o3d
        t0 = time.perf_counter()

        if self._mesh is None:
            raise ValueError("No mesh loaded. Call extract_mesh() first.")

        n_before = len(np.asarray(self._mesh.triangles))

        # Remove degenerate (zero-area) triangles
        self._mesh.remove_degenerate_triangles()
        self._mesh.remove_duplicated_triangles()
        self._mesh.remove_duplicated_vertices()
        self._mesh.remove_unreferenced_vertices()

        # Remove non-manifold edges
        self._mesh.remove_non_manifold_edges()

        # Remove small connected components
        triangle_clusters, cluster_n_triangles, _ = (
            self._mesh.cluster_connected_triangles()
        )
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)

        min_cluster_size = int(n_before * min_component_ratio)
        small_clusters = cluster_n_triangles[triangle_clusters] < min_cluster_size
        self._mesh.remove_triangles_by_mask(small_clusters)
        self._mesh.remove_unreferenced_vertices()

        n_after = len(np.asarray(self._mesh.triangles))
        self._timings["clean_mesh"] = time.perf_counter() - t0
        console.print(f"  Cleanup: {n_before:,} → {n_after:,} faces")
        return self

    def smooth_mesh(
        self,
        iterations: int = 10,
        lambda_filter: float = 0.5,
        mu: float = -0.53,
    ) -> MeshProcessor:
        """Apply Taubin smoothing (volume-preserving).

        Args:
            iterations: Number of smoothing iterations.
            lambda_filter: Positive smoothing factor.
            mu: Negative smoothing factor (should be < -lambda for shrink-free).

        Returns:
            self for chaining.
        """
        t0 = time.perf_counter()

        if self._mesh is None:
            raise ValueError("No mesh loaded. Call extract_mesh() first.")

        self._mesh = self._mesh.filter_smooth_taubin(
            number_of_iterations=iterations,
            lambda_filter=lambda_filter,
            mu=mu,
        )
        self._mesh.compute_vertex_normals()

        self._timings["smooth_mesh"] = time.perf_counter() - t0
        console.print(f"  Taubin smoothing: {iterations} iterations")
        return self

    def decimate_mesh(
        self,
        target_faces: int = 20000,
    ) -> MeshProcessor:
        """Reduce face count via quadric edge collapse decimation.

        Args:
            target_faces: Target number of triangles after decimation.

        Returns:
            self for chaining.
        """
        t0 = time.perf_counter()

        if self._mesh is None:
            raise ValueError("No mesh loaded. Call extract_mesh() first.")

        n_before = len(np.asarray(self._mesh.triangles))

        if n_before <= target_faces:
            console.print(
                f"  Decimation skipped: {n_before:,} faces already ≤ target {target_faces:,}"
            )
            self._timings["decimate_mesh"] = time.perf_counter() - t0
            return self

        self._mesh = self._mesh.simplify_quadric_decimation(
            target_number_of_triangles=target_faces
        )
        self._mesh.compute_vertex_normals()

        n_after = len(np.asarray(self._mesh.triangles))
        self._timings["decimate_mesh"] = time.perf_counter() - t0
        console.print(f"  Decimation: {n_before:,} → {n_after:,} faces")
        return self

    # --- Normal Processing ---

    def fix_normals(self, orient_outward: bool = True) -> MeshProcessor:
        """Validate and fix face normals to be consistent and outward-facing.

        Args:
            orient_outward: If True, orient all normals to point outward
                from the mesh centroid.

        Returns:
            self for chaining.
        """
        o3d = self._o3d
        t0 = time.perf_counter()

        if self._mesh is None:
            raise ValueError("No mesh loaded. Call extract_mesh() first.")

        self._mesh.compute_vertex_normals()
        self._mesh.compute_triangle_normals()

        if orient_outward:
            # Compute mesh centroid
            vertices = np.asarray(self._mesh.vertices)
            centroid = vertices.mean(axis=0)

            # For each face, ensure its normal points away from centroid
            face_normals = np.asarray(self._mesh.triangle_normals)
            triangles = np.asarray(self._mesh.triangles)

            face_centers = vertices[triangles].mean(axis=1)  # (n_faces, 3)
            to_outside = face_centers - centroid  # vector from centroid to face center

            dot = np.sum(face_normals * to_outside, axis=1)
            needs_flip = dot < 0

            # Flip faces with inward normals by swapping vertex order
            if np.any(needs_flip):
                flipped_count = int(needs_flip.sum())
                triangles[needs_flip] = triangles[needs_flip][:, ::-1]
                self._mesh.triangles = o3d.utility.Vector3iVector(triangles)
                self._mesh.compute_triangle_normals()
                self._mesh.compute_vertex_normals()
                console.print(f"    Flipped {flipped_count:,} face normals to outward")

        n_normals = len(np.asarray(self._mesh.triangle_normals))
        self._timings["fix_normals"] = time.perf_counter() - t0
        console.print(f"  Normals validated: {n_normals:,} face normals")
        return self

    # --- Coordinate Transformation ---

    def transform_coordinates(self, geo_ref: GeoReference) -> MeshProcessor:
        """Transform mesh from local coordinates to real-world scale and orientation.

        Applies:
        1. Scale factor (local units -> meters)
        2. Rotation to align with north (compass bearing)
        3. GPS coordinate reference stored in metadata

        Args:
            geo_ref: Geographic reference parameters.

        Returns:
            self for chaining.
        """
        o3d = self._o3d
        t0 = time.perf_counter()

        if self._mesh is None:
            raise ValueError("No mesh loaded.")

        self._geo_ref = geo_ref

        vertices = np.asarray(self._mesh.vertices)

        # Step 1: Scale to meters
        if geo_ref.scale_factor != 1.0:
            vertices *= geo_ref.scale_factor
            console.print(f"    Scale: x{geo_ref.scale_factor}")

        # Step 2: Rotate to align north
        if geo_ref.compass_bearing_deg != 0.0:
            angle_rad = math.radians(-geo_ref.compass_bearing_deg)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)

            # Rotation around Z-axis (assuming Z is up)
            rotation = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1],
            ], dtype=np.float64)

            vertices = (rotation @ vertices.T).T
            console.print(f"    North alignment: {geo_ref.compass_bearing_deg}deg rotation")

        self._mesh.vertices = o3d.utility.Vector3dVector(vertices)
        self._mesh.compute_vertex_normals()
        self._mesh.compute_triangle_normals()

        self._timings["transform_coordinates"] = time.perf_counter() - t0
        console.print(
            f"  Coordinates transformed: "
            f"origin=({geo_ref.latitude:.6f}, {geo_ref.longitude:.6f}), "
            f"alt={geo_ref.altitude}m"
        )
        return self

    # --- RANSAC Plane Detection ---

    def detect_roof_planes(
        self,
        max_planes: int = 10,
        distance_threshold: float = 0.1,
        ransac_n: int = 3,
        num_iterations: int = 1000,
        min_plane_ratio: float = 0.02,
    ) -> MeshProcessor:
        """Detect major planar surfaces (roof faces) via iterative RANSAC.

        Fits planes to mesh vertices, then projects nearby face vertices onto
        the detected planes to flatten roof surfaces, improving normal accuracy
        for solar simulation.

        Args:
            max_planes: Maximum number of planes to detect.
            distance_threshold: RANSAC inlier distance threshold (meters).
            ransac_n: Number of points sampled per RANSAC iteration.
            num_iterations: RANSAC iterations per plane.
            min_plane_ratio: Stop if remaining points < this fraction of total.

        Returns:
            self for chaining.
        """
        o3d = self._o3d
        t0 = time.perf_counter()

        if self._mesh is None:
            raise ValueError("No mesh loaded. Call extract_mesh() first.")

        vertices = np.asarray(self._mesh.vertices).copy()
        total_points = len(vertices)
        min_plane_points = int(total_points * min_plane_ratio)

        # Create point cloud from mesh vertices for plane detection
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)

        remaining_indices = np.arange(total_points)
        planes_detected = []

        for i in range(max_planes):
            if len(remaining_indices) < min_plane_points:
                break

            # Create subset point cloud
            subset_pcd = pcd.select_by_index(remaining_indices.tolist())

            plane_model, inliers = subset_pcd.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=ransac_n,
                num_iterations=num_iterations,
            )

            if len(inliers) < min_plane_points:
                break

            # Map inliers back to original vertex indices
            original_inliers = remaining_indices[inliers]
            planes_detected.append((plane_model, original_inliers))

            # Project inlier vertices onto the plane
            a, b, c, d = plane_model
            normal = np.array([a, b, c])
            for idx in original_inliers:
                v = vertices[idx]
                dist = np.dot(normal, v) + d
                vertices[idx] = v - dist * normal

            # Remove inliers from remaining
            remaining_indices = np.setdiff1d(remaining_indices, original_inliers)

            console.print(
                f"    Plane {i + 1}: {len(original_inliers):,} vertices "
                f"(normal=[{a:.2f}, {b:.2f}, {c:.2f}])"
            )

        if planes_detected:
            self._mesh.vertices = o3d.utility.Vector3dVector(vertices)
            self._mesh.compute_vertex_normals()
            self._mesh.compute_triangle_normals()

        self._timings["detect_roof_planes"] = time.perf_counter() - t0
        console.print(f"  RANSAC plane detection: {len(planes_detected)} planes found")
        return self

    # --- Face Segmentation ---

    def segment_faces(
        self,
        roof_max_angle_deg: float = 60.0,
        ground_max_angle_deg: float = 10.0,
        wall_min_angle_deg: float = 60.0,
    ) -> MeshProcessor:
        """Segment mesh faces into roof, wall, and ground based on normal angles.

        Classification is based on the angle between the face normal and the
        up vector (0,0,1):
        - Ground: angle < ground_max_angle_deg (nearly horizontal, facing up at bottom)
        - Roof: angle < roof_max_angle_deg (mostly upward-facing, above centroid)
        - Wall: angle > wall_min_angle_deg (mostly vertical)
        - Unknown: everything else

        Args:
            roof_max_angle_deg: Max angle from vertical for roof faces.
            ground_max_angle_deg: Max angle from vertical for ground faces.
            wall_min_angle_deg: Min angle from vertical for wall faces.

        Returns:
            self for chaining.
        """
        t0 = time.perf_counter()

        if self._mesh is None:
            raise ValueError("No mesh loaded.")

        self._mesh.compute_triangle_normals()
        face_normals = np.asarray(self._mesh.triangle_normals)
        triangles = np.asarray(self._mesh.triangles)
        vertices = np.asarray(self._mesh.vertices)
        n_faces = len(face_normals)

        up = np.array([0, 0, 1], dtype=np.float64)
        cos_angles = np.dot(face_normals, up)
        cos_angles = np.clip(cos_angles, -1, 1)
        angles_deg = np.degrees(np.arccos(np.abs(cos_angles)))

        # Face centroids for height-based discrimination
        face_centers = vertices[triangles].mean(axis=1)
        z_median = np.median(vertices[:, 2])

        labels = np.full(n_faces, 3, dtype=np.int32)  # default: UNKNOWN

        # Ground: nearly horizontal and below median height, normal pointing up
        ground_mask = (angles_deg < ground_max_angle_deg) & (face_centers[:, 2] < z_median) & (cos_angles > 0)
        labels[ground_mask] = 2  # GROUND

        # Roof: upward-facing and above median height
        roof_mask = (
            (angles_deg < roof_max_angle_deg)
            & (face_centers[:, 2] >= z_median)
            & (cos_angles > 0)
            & (~ground_mask)
        )
        labels[roof_mask] = 0  # ROOF

        # Wall: mostly vertical
        wall_mask = (angles_deg >= wall_min_angle_deg) & (~ground_mask)
        labels[wall_mask] = 1  # WALL

        self._face_labels = labels

        label_names = {0: "roof", 1: "wall", 2: "ground", 3: "unknown"}
        for val, name in label_names.items():
            count = int((labels == val).sum())
            console.print(f"    {name}: {count:,} faces")

        self._timings["segment_faces"] = time.perf_counter() - t0
        return self

    # --- Saving ---

    def save(
        self,
        output_path: Path | str,
        save_labels: bool = True,
    ) -> ProcessingResult:
        """Save the mesh to file (OBJ or PLY).

        Args:
            output_path: Output file path (.obj or .ply).
            save_labels: If True, save face labels as a separate JSON file.

        Returns:
            ProcessingResult with statistics.
        """
        o3d = self._o3d
        t0 = time.perf_counter()

        if self._mesh is None:
            raise ValueError("No mesh to save.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure normals are computed
        self._mesh.compute_vertex_normals()
        self._mesh.compute_triangle_normals()

        o3d.io.write_triangle_mesh(
            str(output_path),
            self._mesh,
            write_vertex_normals=True,
            write_triangle_uvs=False,
        )

        self._timings["save"] = time.perf_counter() - t0

        # Compute stats
        vertices = np.asarray(self._mesh.vertices)
        triangles = np.asarray(self._mesh.triangles)
        normals = np.asarray(self._mesh.triangle_normals)

        stats = MeshStats(
            num_vertices=len(vertices),
            num_faces=len(triangles),
            num_normals=len(normals),
            bounding_box_min=tuple(vertices.min(axis=0).tolist()),
            bounding_box_max=tuple(vertices.max(axis=0).tolist()),
        )

        # Surface area
        v0 = vertices[triangles[:, 0]]
        v1 = vertices[triangles[:, 1]]
        v2 = vertices[triangles[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        areas = np.linalg.norm(cross, axis=1) / 2.0
        stats.surface_area_m2 = float(areas.sum())

        if self._face_labels is not None:
            stats.num_roof_faces = int((self._face_labels == 0).sum())
            stats.num_wall_faces = int((self._face_labels == 1).sum())
            stats.num_ground_faces = int((self._face_labels == 2).sum())

        # Save labels
        if save_labels and self._face_labels is not None:
            labels_path = output_path.with_suffix(".labels.json")
            label_map = {0: "roof", 1: "wall", 2: "ground", 3: "unknown"}
            labels_data = {
                "num_faces": len(self._face_labels),
                "label_counts": {
                    name: int((self._face_labels == val).sum())
                    for val, name in label_map.items()
                },
                "face_labels": [label_map[int(l)] for l in self._face_labels],
            }
            Path(labels_path).write_text(
                json.dumps(labels_data, indent=2, ensure_ascii=False)
            )
            console.print(f"  Labels saved: {labels_path}")

        # Save metadata
        total_time = sum(self._timings.values())
        metadata = {
            "output_path": str(output_path),
            "stats": {
                "num_vertices": stats.num_vertices,
                "num_faces": stats.num_faces,
                "surface_area_m2": round(stats.surface_area_m2, 2),
                "bounding_box_min": list(stats.bounding_box_min),
                "bounding_box_max": list(stats.bounding_box_max),
                "num_roof_faces": stats.num_roof_faces,
                "num_wall_faces": stats.num_wall_faces,
                "num_ground_faces": stats.num_ground_faces,
            },
            "timing": {k: round(v, 3) for k, v in self._timings.items()},
            "total_time_s": round(total_time, 2),
        }
        if self._geo_ref is not None:
            metadata["geo_reference"] = {
                "latitude": self._geo_ref.latitude,
                "longitude": self._geo_ref.longitude,
                "altitude": self._geo_ref.altitude,
                "compass_bearing_deg": self._geo_ref.compass_bearing_deg,
                "scale_factor": self._geo_ref.scale_factor,
            }
        meta_path = output_path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

        console.print(f"  Mesh saved: {output_path}")
        console.print(f"    {stats.num_vertices:,} vertices, {stats.num_faces:,} faces")
        console.print(f"    Surface area: {stats.surface_area_m2:.1f} m^2")
        console.print(f"    Total time: {total_time:.2f}s")

        return ProcessingResult(
            output_path=output_path,
            stats=stats,
            face_labels=self._face_labels,
            timing_s=dict(self._timings),
            total_time_s=total_time,
        )


def process_reconstruction(
    point_cloud_path: Path,
    output_path: Path,
    method: str = "poisson",
    geo_ref: GeoReference | None = None,
    depth_dir: Path | None = None,
    camera_poses_path: Path | None = None,
    poisson_depth: int = 7,
    voxel_size: float | None = None,
    smooth_iterations: int = 10,
    target_faces: int = 50000,
) -> ProcessingResult:
    """Convenience function to run the full mesh processing pipeline.

    Args:
        point_cloud_path: Path to input point cloud PLY.
        output_path: Path for output mesh file.
        method: Mesh extraction method.
        geo_ref: Geographic reference for coordinate transformation.
        depth_dir: Directory with depth maps (for TSDF/marching cubes).
        camera_poses_path: Path to camera poses JSON (for TSDF/marching cubes).
        poisson_depth: Poisson reconstruction depth parameter.
        voxel_size: Voxel size for point cloud downsampling.
        smooth_iterations: Number of Taubin smoothing iterations.
        target_faces: Target face count after decimation.

    Returns:
        ProcessingResult with statistics.
    """
    processor = MeshProcessor()

    needs_depth = method in ("tsdf", "marching_cubes")
    if needs_depth and (depth_dir is None or camera_poses_path is None):
        raise ValueError(
            f"Method '{method}' requires depth_dir and camera_poses_path"
        )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Loading data...", total=None)
        processor.load_point_cloud(point_cloud_path)
        if needs_depth:
            processor.load_depth_maps(depth_dir, camera_poses_path)
        progress.stop_task(task)

        task = progress.add_task("Preprocessing point cloud...", total=None)
        processor.preprocess_point_cloud(voxel_size=voxel_size)
        progress.stop_task(task)

        task = progress.add_task(f"Extracting mesh ({method})...", total=None)
        processor.extract_mesh(method=method, depth=poisson_depth)
        progress.stop_task(task)

        task = progress.add_task("Cleaning mesh...", total=None)
        processor.clean_mesh()
        progress.stop_task(task)

        task = progress.add_task("Smoothing mesh...", total=None)
        processor.smooth_mesh(iterations=smooth_iterations)
        progress.stop_task(task)

        task = progress.add_task("Decimating mesh...", total=None)
        processor.decimate_mesh(target_faces=target_faces)
        progress.stop_task(task)

        task = progress.add_task("Detecting roof planes (RANSAC)...", total=None)
        processor.detect_roof_planes()
        progress.stop_task(task)

        task = progress.add_task("Fixing normals...", total=None)
        processor.fix_normals()
        progress.stop_task(task)

        if geo_ref is not None:
            task = progress.add_task("Transforming coordinates...", total=None)
            processor.transform_coordinates(geo_ref)
            progress.stop_task(task)

        task = progress.add_task("Segmenting faces...", total=None)
        processor.segment_faces()
        progress.stop_task(task)

        task = progress.add_task("Saving...", total=None)
        result = processor.save(output_path)
        progress.stop_task(task)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Mesh post-processing pipeline for 3D reconstruction output."
    )
    parser.add_argument(
        "point_cloud",
        type=Path,
        help="Input point cloud PLY file",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output mesh file (default: data/mesh_output/mesh.obj)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="poisson",
        choices=["tsdf", "poisson", "marching_cubes", "nksr"],
        help="Mesh extraction method (default: poisson)",
    )
    parser.add_argument(
        "--depth-dir",
        type=Path,
        default=None,
        help="Directory containing depth .npy files (for TSDF/marching cubes)",
    )
    parser.add_argument(
        "--camera-poses",
        type=Path,
        default=None,
        help="Camera poses JSON file (for TSDF/marching cubes)",
    )
    parser.add_argument(
        "--latitude",
        type=float,
        default=None,
        help="GPS latitude for geo-referencing",
    )
    parser.add_argument(
        "--longitude",
        type=float,
        default=None,
        help="GPS longitude for geo-referencing",
    )
    parser.add_argument(
        "--altitude",
        type=float,
        default=0.0,
        help="Altitude in meters",
    )
    parser.add_argument(
        "--compass-bearing",
        type=float,
        default=0.0,
        help="Compass bearing in degrees (north=0, clockwise)",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.0,
        help="Scale factor from local units to meters",
    )
    parser.add_argument(
        "--poisson-depth",
        type=int,
        default=7,
        help="Poisson reconstruction depth (default: 7)",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = Path(__file__).parent.parent.parent / "data" / "mesh_output" / "mesh.obj"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    geo_ref = None
    if args.latitude is not None and args.longitude is not None:
        geo_ref = GeoReference(
            latitude=args.latitude,
            longitude=args.longitude,
            altitude=args.altitude,
            compass_bearing_deg=args.compass_bearing,
            scale_factor=args.scale_factor,
        )

    console.print("[bold blue]ExaSense Mesh Processing Pipeline")
    process_reconstruction(
        point_cloud_path=args.point_cloud,
        output_path=args.output,
        method=args.method,
        geo_ref=geo_ref,
        depth_dir=args.depth_dir,
        camera_poses_path=args.camera_poses,
        poisson_depth=args.poisson_depth,
    )


if __name__ == "__main__":
    main()
