#!/bin/bash
set -euxo pipefail
exec > >(tee /var/log/exasense-e2e.log) 2>&1

echo "=== ExaSense E2E Test (Mip-NeRF 360 Garden) ==="
date
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region)

# Safety: auto-terminate after 60 minutes
nohup bash -c "sleep 3600 && aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION" &

# Wait for NVIDIA drivers
for i in $(seq 1 30); do
    nvidia-smi && break || sleep 10
done

# System deps
apt-get update -y
apt-get install -y git curl unzip python3-pip

# Install packages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install open3d trimesh pvlib scipy numpy pandas pillow rich
pip install git+https://github.com/facebookresearch/vggt.git

# Download Mip-NeRF 360 dataset (garden scene only)
echo "=== Downloading Mip-NeRF 360 dataset ==="
cd /home/ubuntu
mkdir -p datasets
cd datasets

# Download zip
time curl -L -o 360_v2.zip "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"

# Extract garden scene (zip structure: garden/images/xxx.JPG, no 360_v2/ prefix)
echo "=== Extracting garden scene ==="
time unzip -q 360_v2.zip "garden/*" -d .

ls -la garden/images/ | head -20
GARDEN_IMAGES=$(ls garden/images/ | wc -l)
echo "Garden scene: $GARDEN_IMAGES images"

# Remove zip to free space
rm -f 360_v2.zip

echo "=== Running E2E Pipeline ==="
cd /home/ubuntu

python3 << 'PYEOF'
import json
import time
import sys
import os
import numpy as np

results = {}
timings = {}

# ============================================================
# Phase 1: VGGT 3D Reconstruction
# ============================================================
print("\n" + "="*60)
print("PHASE 1: VGGT 3D Reconstruction")
print("="*60)

import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Use subset of images (15 for T4 16GB, more for H100)
image_dir = "/home/ubuntu/datasets/garden/images"
print(f"Image dir: {image_dir}")
all_images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.JPG', '.png'))])
N_IMAGES = 10  # T4 memory-safe
selected = all_images[::len(all_images)//N_IMAGES][:N_IMAGES]
print(f"Selected {len(selected)}/{len(all_images)} images")

# Save selected images to temp dir
import shutil
tmp_dir = "/home/ubuntu/garden_subset"
os.makedirs(tmp_dir, exist_ok=True)
for img in selected:
    shutil.copy2(os.path.join(image_dir, img), tmp_dir)

# Load VGGT
t0 = time.time()
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

model = VGGT.from_pretrained("facebook/VGGT-1B").to("cuda")
compute_cap = torch.cuda.get_device_capability(0)
dtype = torch.bfloat16 if compute_cap[0] >= 8 else torch.float16
model_load_time = time.time() - t0
print(f"Model loaded in {model_load_time:.1f}s, dtype={dtype}")

# Run inference
image_paths = sorted([os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if f.endswith(('.jpg', '.JPG', '.png'))])
print(f"Processing {len(image_paths)} images...")

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
images_tensor = load_and_preprocess_images(image_paths).to("cuda")
print(f"  Image tensor shape: {images_tensor.shape}")

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        predictions = model(images_tensor)

inference_time = time.time() - t0
peak_vram = torch.cuda.max_memory_allocated() / 1e9

print(f"Inference: {inference_time:.1f}s, Peak VRAM: {peak_vram:.1f} GB")
print("Prediction keys:")
for k, v in predictions.items():
    if hasattr(v, 'shape'):
        print(f"  {k}: {v.shape}")

timings["vggt_model_load_s"] = round(model_load_time, 1)
timings["vggt_inference_s"] = round(inference_time, 1)
timings["vggt_peak_vram_gb"] = round(peak_vram, 2)
results["vggt"] = {
    "n_images": len(image_paths),
    "image_shape": list(images_tensor.shape),
    "inference_time_s": round(inference_time, 1),
    "peak_vram_gb": round(peak_vram, 2),
}

# Extract world points
world_points = predictions["world_points"].cpu().numpy()[0]  # (N, H, W, 3)
world_conf = predictions["world_points_conf"].cpu().numpy()[0]  # (N, H, W)
depth_maps = predictions["depth"].cpu().numpy()[0]  # (N, H, W, 1)

print(f"World points shape: {world_points.shape}")
print(f"Confidence range: [{world_conf.min():.3f}, {world_conf.max():.3f}]")

# Aggregate high-confidence points
CONF_THRESHOLD = 0.3
mask = world_conf > CONF_THRESHOLD
n_frames, H, W = world_conf.shape
all_pts = []
for i in range(n_frames):
    frame_pts = world_points[i][mask[i]]  # (M, 3)
    all_pts.append(frame_pts)
point_cloud = np.concatenate(all_pts, axis=0)
print(f"Point cloud: {point_cloud.shape[0]:,} points (conf > {CONF_THRESHOLD})")

results["point_cloud"] = {
    "n_points": int(point_cloud.shape[0]),
    "conf_threshold": CONF_THRESHOLD,
    "bbox_min": point_cloud.min(axis=0).tolist(),
    "bbox_max": point_cloud.max(axis=0).tolist(),
}

# Save point cloud
np.save("/home/ubuntu/garden_points.npy", point_cloud)

# Cleanup GPU
del model, predictions, images_tensor
torch.cuda.empty_cache()

# ============================================================
# Phase 2: Mesh Processing (Open3D Poisson)
# ============================================================
print("\n" + "="*60)
print("PHASE 2: Mesh Processing")
print("="*60)

import open3d as o3d

t0 = time.time()
n_raw = len(point_cloud)

# Stage 1: Voxel downsampling for spatial uniformity
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud)
pcd = pcd.voxel_down_sample(voxel_size=0.02)
n_voxel = len(pcd.points)
print(f"Voxel downsample (0.02m): {n_raw:,} → {n_voxel:,}")

# Stage 2: Statistical outlier removal
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
n_stat = len(pcd.points)
print(f"Statistical outlier removal: {n_voxel:,} → {n_stat:,}")

# Stage 3: Radius outlier removal
pcd, _ = pcd.remove_radius_outlier(nb_points=16, radius=0.05)
n_clean = len(pcd.points)
print(f"Radius outlier removal: {n_stat:,} → {n_clean:,}")

# Stage 4: Normal estimation
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
pcd.orient_normals_consistent_tangent_plane(100)

# Stage 5: Poisson reconstruction (depth=7, density filter 15%)
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=7, linear_fit=True
)
densities = np.asarray(densities)
density_threshold = np.quantile(densities, 0.15)
vertices_to_remove = densities < density_threshold
mesh.remove_vertices_by_mask(vertices_to_remove)
n_faces_poisson = len(mesh.triangles)
print(f"Poisson (depth=7): {n_faces_poisson:,} faces")

# Stage 6: Cleanup — remove degenerate triangles and small components
mesh.remove_degenerate_triangles()
mesh.remove_duplicated_triangles()
mesh.remove_duplicated_vertices()
mesh.remove_unreferenced_vertices()
mesh.remove_non_manifold_edges()

triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
triangle_clusters = np.asarray(triangle_clusters)
cluster_n_triangles = np.asarray(cluster_n_triangles)
min_cluster = int(n_faces_poisson * 0.01)
small_mask = cluster_n_triangles[triangle_clusters] < min_cluster
mesh.remove_triangles_by_mask(small_mask)
mesh.remove_unreferenced_vertices()
n_faces_clean = len(mesh.triangles)
print(f"Cleanup: {n_faces_poisson:,} → {n_faces_clean:,} faces")

# Stage 7: Taubin smoothing (volume-preserving)
mesh = mesh.filter_smooth_taubin(number_of_iterations=10, lambda_filter=0.5, mu=-0.53)
mesh.compute_vertex_normals()
print(f"Taubin smoothing: 10 iterations")

# Stage 8: Quadric decimation to target 20K faces
TARGET_FACES = 20000
n_before_decimate = len(mesh.triangles)
if n_before_decimate > TARGET_FACES:
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=TARGET_FACES)
    mesh.compute_vertex_normals()
n_faces_final = len(mesh.triangles)
print(f"Decimation: {n_before_decimate:,} → {n_faces_final:,} faces")

mesh_time = time.time() - t0
n_verts = len(mesh.vertices)
n_faces = len(mesh.triangles)

print(f"Final mesh: {n_verts:,} vertices, {n_faces:,} faces in {mesh_time:.1f}s")

# Save mesh
o3d.io.write_triangle_mesh("/home/ubuntu/garden_mesh.ply", mesh)

timings["mesh_processing_s"] = round(mesh_time, 1)
results["mesh"] = {
    "n_points_raw": n_raw,
    "n_points_preprocessed": n_clean,
    "n_faces_poisson": n_faces_poisson,
    "n_vertices": n_verts,
    "n_faces": n_faces,
    "processing_time_s": round(mesh_time, 1),
}

# ============================================================
# Phase 3: Solar Simulation
# ============================================================
print("\n" + "="*60)
print("PHASE 3: Solar Simulation")
print("="*60)

import trimesh
import pvlib
import pandas as pd

t0 = time.time()

# Convert Open3D mesh to trimesh
verts_np = np.asarray(mesh.vertices)
faces_np = np.asarray(mesh.triangles)
tri_mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=True)

print(f"Trimesh: {len(tri_mesh.faces)} faces")

# Compute face normals and areas
face_normals = tri_mesh.face_normals
face_areas = tri_mesh.area_faces

# Identify upward-facing faces (potential roof/solar surfaces)
# Normal z-component > 0.3 means roughly facing up
UP_THRESHOLD = 0.3
upward_mask = face_normals[:, 2] > UP_THRESHOLD
n_upward = upward_mask.sum()
upward_area = face_areas[upward_mask].sum()
print(f"Upward-facing faces: {n_upward} ({upward_area:.1f} m² area)")

# Solar position calculation (Osaka)
LATITUDE = 34.69
LONGITUDE = 135.50
times = pd.date_range('2025-06-21', periods=24, freq='h', tz='Asia/Tokyo')
solar_pos = pvlib.solarposition.get_solarposition(time=times, latitude=LATITUDE, longitude=LONGITUDE)

# Clear sky irradiance
loc = pvlib.location.Location(LATITUDE, LONGITUDE)
cs = loc.get_clearsky(times)

# Simple irradiance estimation for upward faces
# Annual energy = sum of hourly DNI * cos(incidence) + DHI * sky_view
annual_times = pd.date_range('2025-01-01', periods=8760, freq='h', tz='Asia/Tokyo')
annual_solar = pvlib.solarposition.get_solarposition(time=annual_times, latitude=LATITUDE, longitude=LONGITUDE)
annual_cs = loc.get_clearsky(annual_times)

# Calculate total GHI for the year
total_ghi = annual_cs['ghi'].sum() / 1000  # kWh/m²/year
total_dni = annual_cs['dni'].sum() / 1000
total_dhi = annual_cs['dhi'].sum() / 1000

# Estimate solar potential
PANEL_EFFICIENCY = 0.20
PERFORMANCE_RATIO = 0.80
potential_kwh = upward_area * total_ghi * PANEL_EFFICIENCY * PERFORMANCE_RATIO
potential_kw = upward_area * PANEL_EFFICIENCY  # peak capacity

# ROI
ELECTRICITY_PRICE = 30  # ¥/kWh
PANEL_COST_PER_KW = 250000  # ¥/kW
annual_savings = potential_kwh * ELECTRICITY_PRICE
installation_cost = potential_kw * PANEL_COST_PER_KW
payback_years = installation_cost / annual_savings if annual_savings > 0 else float('inf')

sim_time = time.time() - t0

print(f"Annual GHI: {total_ghi:.0f} kWh/m²/year")
print(f"Solar potential: {potential_kw:.1f} kW peak, {potential_kwh:.0f} kWh/year")
print(f"Annual savings: ¥{annual_savings:,.0f}")
print(f"Installation cost: ¥{installation_cost:,.0f}")
print(f"Payback: {payback_years:.1f} years")
print(f"Simulation time: {sim_time:.1f}s")

timings["solar_simulation_s"] = round(sim_time, 1)
results["solar"] = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "annual_ghi_kwh_m2": round(total_ghi, 1),
    "upward_faces": int(n_upward),
    "upward_area_m2": round(float(upward_area), 1),
    "potential_kw": round(float(potential_kw), 1),
    "potential_kwh_year": round(float(potential_kwh), 0),
    "annual_savings_yen": round(float(annual_savings), 0),
    "installation_cost_yen": round(float(installation_cost), 0),
    "payback_years": round(float(payback_years), 1),
    "simulation_time_s": round(sim_time, 1),
}

# ============================================================
# Summary
# ============================================================
print("\n" + "="*60)
print("E2E PIPELINE SUMMARY")
print("="*60)

total_time = sum(timings.values())
timings["total_s"] = round(total_time, 1)

print(f"  VGGT Load:    {timings['vggt_model_load_s']}s")
print(f"  VGGT Infer:   {timings['vggt_inference_s']}s (Peak VRAM: {timings['vggt_peak_vram_gb']} GB)")
print(f"  Mesh Process: {timings['mesh_processing_s']}s")
print(f"  Solar Sim:    {timings['solar_simulation_s']}s")
print(f"  TOTAL:        {total_time:.1f}s")
print()
print(f"  Images:       {results['vggt']['n_images']}")
print(f"  Points:       {results['point_cloud']['n_points']:,}")
print(f"  Mesh:         {results['mesh']['n_vertices']:,} verts, {results['mesh']['n_faces']:,} faces")
print(f"  Solar Area:   {results['solar']['upward_area_m2']:.1f} m²")
print(f"  Potential:    {results['solar']['potential_kw']:.1f} kW")
print(f"  Payback:      {results['solar']['payback_years']:.1f} years")

# Save all results
output = {
    "dataset": "mipnerf360_garden",
    "timings": timings,
    "results": results,
}
with open("/home/ubuntu/e2e_results.json", "w") as f:
    json.dump(output, f, indent=2)
print("\nResults saved to /home/ubuntu/e2e_results.json")
print("Point cloud: /home/ubuntu/garden_points.npy")
print("Mesh: /home/ubuntu/garden_mesh.ply")

print("\n=== E2E PIPELINE COMPLETE ===")
PYEOF

# Upload results to S3
echo "=== Uploading results to S3 ==="
S3_BUCKET="s3://exasense-e2e-results"
aws s3 cp /home/ubuntu/garden_mesh.ply "$S3_BUCKET/garden_mesh.ply"
aws s3 cp /home/ubuntu/garden_points.npy "$S3_BUCKET/garden_points.npy"
aws s3 cp /home/ubuntu/e2e_results.json "$S3_BUCKET/e2e_results.json"
echo "=== S3 Upload Complete ==="

echo "=== E2E Test Complete ==="
