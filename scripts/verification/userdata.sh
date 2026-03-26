#!/bin/bash
set -euxo pipefail
exec > >(tee /var/log/exasense-verify.log) 2>&1

echo "=== ExaSense Phase 1-2 Verification ==="
date
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region)

# Safety: auto-terminate after 90 minutes
nohup bash -c "sleep 5400 && aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION" &

# Wait for NVIDIA drivers
for i in $(seq 1 30); do
    nvidia-smi && break || sleep 10
done

# System deps
apt-get update -y
apt-get install -y git curl unzip python3-pip

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"

# Install COLMAP from apt
apt-get install -y colmap || echo "COLMAP not in apt, will skip SfM test"

# Clone and setup project
cd /home/ubuntu
git clone https://github.com/liltak/kanden_hackathon_exasense.git exasense-project || {
    # If repo not available, create minimal project structure
    echo "Repo not accessible, creating from scratch..."
    mkdir -p exasense-project
    cd exasense-project

    # Install base packages directly
    pip install pvlib trimesh plotly scipy numpy pandas gradio fastapi uvicorn pyyaml rich rtree pillow

    # Install GPU packages
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    pip install open3d accelerate

    # Install VGGT package from GitHub
    pip install git+https://github.com/facebookresearch/vggt.git

    echo "=== Running inline verification ==="

    python3 << 'PYEOF'
import json
import time
import sys

results = []

# Check 1: GPU
print("\n=== Check 1: GPU Environment ===")
try:
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        compute_cap = torch.cuda.get_device_capability(0)
        print(f"PASS: {name} ({vram:.1f} GB VRAM, compute {compute_cap[0]}.{compute_cap[1]})")
        results.append({"check": "GPU", "passed": True, "gpu": name, "vram_gb": round(vram, 1),
                        "compute_capability": f"{compute_cap[0]}.{compute_cap[1]}"})
    else:
        print("FAIL: CUDA not available")
        results.append({"check": "GPU", "passed": False})
except Exception as e:
    print(f"FAIL: {e}")
    results.append({"check": "GPU", "passed": False, "error": str(e)})

# Check 2: VGGT Model Load + Inference
print("\n=== Check 2: VGGT Model Load ===")
try:
    import torch
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    from vggt.models.vggt import VGGT
    model = VGGT.from_pretrained("facebook/VGGT-1B-Commercial")

    # Use float16 for T4 (compute < 8.0), bfloat16 for A10G/H100 (compute >= 8.0)
    compute_cap = torch.cuda.get_device_capability(0)
    dtype = torch.bfloat16 if compute_cap[0] >= 8 else torch.float16
    model = model.to("cuda")

    load_time = time.time() - t0
    vram_used = torch.cuda.memory_allocated() / 1e9
    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    print(f"PASS: Loaded in {load_time:.1f}s")
    print(f"  Parameters: {n_params:.0f}M")
    print(f"  VRAM allocated: {vram_used:.2f} GB")
    print(f"  VRAM peak: {peak_vram:.2f} GB")
    print(f"  dtype: {dtype}")
    results.append({
        "check": "VGGT Load", "passed": True,
        "load_time_s": round(load_time, 1),
        "params_M": round(n_params, 0),
        "vram_allocated_gb": round(vram_used, 2),
        "vram_peak_gb": round(peak_vram, 2),
        "dtype": str(dtype),
    })

    # Check 3: VGGT Inference
    print("\n=== Check 3: VGGT Inference (synthetic) ===")
    from vggt.utils.load_fn import load_and_preprocess_images
    from PIL import Image
    import numpy as np
    import tempfile
    import os

    # Create synthetic test images and save to temp files
    tmp_dir = tempfile.mkdtemp()
    image_paths = []
    for i in range(5):
        arr = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        arr[100:200, 100+i*30:200+i*30] = [255, 0, 0]  # Add feature
        path = os.path.join(tmp_dir, f"test_{i:03d}.png")
        Image.fromarray(arr).save(path)
        image_paths.append(path)

    # Load and preprocess
    images = load_and_preprocess_images(image_paths).to("cuda")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
    infer_time = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / 1e9

    # Check predictions
    pred_keys = list(predictions.keys()) if isinstance(predictions, dict) else [k for k in dir(predictions) if not k.startswith('_')]
    print(f"PASS: Inference {len(image_paths)} images in {infer_time:.1f}s")
    print(f"  Peak VRAM: {peak_vram:.2f} GB")
    print(f"  Prediction keys: {pred_keys}")

    # Print prediction shapes
    if isinstance(predictions, dict):
        for k, v in predictions.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: shape={v.shape}")

    results.append({
        "check": "VGGT Inference", "passed": True,
        "n_images": len(image_paths),
        "inference_time_s": round(infer_time, 1),
        "peak_vram_gb": round(peak_vram, 2),
        "prediction_keys": str(pred_keys),
    })

    del model, predictions
    torch.cuda.empty_cache()

except Exception as e:
    print(f"FAIL: {e}")
    import traceback; traceback.print_exc()
    results.append({"check": "VGGT", "passed": False, "error": str(e)})

# Check 4: Open3D Mesh Processing
print("\n=== Check 4: Mesh Processing (Open3D) ===")
try:
    import open3d as o3d
    import numpy as np
    t0 = time.time()

    # Create synthetic point cloud (dome shape)
    n = 5000
    phi = np.random.uniform(0, 2*np.pi, n)
    theta = np.random.uniform(0, np.pi/2, n)
    r = 10 + np.random.normal(0, 0.1, n)
    pts = np.column_stack([r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.estimate_normals()

    # Poisson reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
    mesh_time = time.time() - t0

    n_verts = len(mesh.vertices)
    n_faces = len(mesh.triangles)
    print(f"PASS: {n} points -> {n_verts} vertices, {n_faces} faces in {mesh_time:.1f}s")
    results.append({
        "check": "Mesh Processing", "passed": True,
        "input_points": n, "output_vertices": n_verts,
        "output_faces": n_faces, "time_s": round(mesh_time, 1),
    })
except Exception as e:
    print(f"FAIL: {e}")
    results.append({"check": "Mesh Processing", "passed": False, "error": str(e)})

# Check 5: Trimesh + pvlib (Phase 3 pipeline)
print("\n=== Check 5: Phase 3 Pipeline ===")
try:
    import trimesh
    import pvlib
    t0 = time.time()

    # Simple mesh
    verts = np.array([[0,0,0],[40,0,0],[40,20,0],[0,20,0],
                       [0,0,8],[40,0,8],[40,20,8],[0,20,8],
                       [0,10,12],[40,10,12]], dtype=float)
    faces = np.array([[0,1,2],[0,2,3],[4,5,9],[4,9,8],[6,7,8],[6,8,9]])
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)

    # Solar position
    times = pvlib.solarposition.get_solarposition(
        time=__import__('pandas').date_range('2025-06-21', periods=24, freq='h', tz='Asia/Tokyo'),
        latitude=34.69, longitude=135.50
    )
    pipeline_time = time.time() - t0
    print(f"PASS: Mesh ({len(mesh.faces)} faces) + pvlib solar positions in {pipeline_time:.1f}s")
    results.append({"check": "Phase3 Pipeline", "passed": True, "time_s": round(pipeline_time, 1)})
except Exception as e:
    print(f"FAIL: {e}")
    results.append({"check": "Phase3 Pipeline", "passed": False, "error": str(e)})

# Summary
print("\n" + "="*60)
print("VERIFICATION SUMMARY")
print("="*60)
passed = sum(1 for r in results if r.get("passed"))
total = len(results)
for r in results:
    status = "PASS" if r.get("passed") else "FAIL"
    print(f"  [{status}] {r['check']}")
print(f"\n{passed}/{total} checks passed")
print("="*60)

# Save results
with open("/home/ubuntu/verification_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Results saved to /home/ubuntu/verification_results.json")
PYEOF

    echo "=== Verification complete ==="
    exit 0
}

# If repo was cloned successfully
cd exasense-project/exasense
uv sync --extra dev
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install open3d accelerate
uv pip install git+https://github.com/facebookresearch/vggt.git

uv run python scripts/verification/verify_phase1_2.py --dataset synthetic --all

echo "=== Verification complete ==="
