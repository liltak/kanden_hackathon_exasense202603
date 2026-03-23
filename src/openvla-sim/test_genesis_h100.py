"""
Genesis H100 動作確認テスト

H100 GPU 上で Genesis シミュレータが正常に起動・動作するかを検証する。
"""

import sys
import time

def test_cuda_available():
    """CUDA が使用可能かチェック"""
    print("=" * 60)
    print("[Test 1] CUDA availability")
    print("=" * 60)
    import torch
    assert torch.cuda.is_available(), "CUDA is not available"
    device_count = torch.cuda.device_count()
    print(f"  CUDA available: True")
    print(f"  Device count: {device_count}")
    for i in range(device_count):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"  GPU[{i}]: {name}, VRAM={mem:.1f} GB")
    print("  [PASS]\n")


def test_genesis_import():
    """Genesis が import できるかチェック"""
    print("=" * 60)
    print("[Test 2] Genesis import")
    print("=" * 60)
    import genesis as gs
    print(f"  genesis version: {getattr(gs, '__version__', 'unknown')}")
    print("  [PASS]\n")
    return gs


def test_genesis_init(gs):
    """Genesis を GPU バックエンドで初期化できるかチェック"""
    print("=" * 60)
    print("[Test 3] Genesis init (GPU backend)")
    print("=" * 60)
    gs.init(backend=gs.gpu)
    print("  gs.init(backend=gs.gpu) succeeded")
    print("  [PASS]\n")


def test_scene_create(gs):
    """シーンを作成できるかチェック"""
    print("=" * 60)
    print("[Test 4] Scene creation")
    print("=" * 60)
    scene = gs.Scene(show_viewer=False)
    print("  gs.Scene() created")
    print("  [PASS]\n")
    return scene


def test_add_entities(gs, scene):
    """エンティティ（平面・ボックス）を追加できるかチェック"""
    print("=" * 60)
    print("[Test 5] Add entities (plane + box)")
    print("=" * 60)
    plane = scene.add_entity(gs.morphs.Plane())
    box   = scene.add_entity(
        gs.morphs.Box(size=(0.1, 0.1, 0.1)),
        surface=gs.surfaces.Default(color=(0.5, 0.8, 0.5, 1.0)),
    )
    print(f"  Plane entity: {plane}")
    print(f"  Box entity:   {box}")
    print("  [PASS]\n")
    return scene


def test_build_and_step(scene):
    """シーンをビルドし、数ステップ物理シミュレーションを実行できるかチェック"""
    print("=" * 60)
    print("[Test 6] Build scene & physics step")
    print("=" * 60)
    scene.build()
    print("  scene.build() succeeded")

    N_STEPS = 60
    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        scene.step()
    elapsed = time.perf_counter() - t0

    fps = N_STEPS / elapsed
    print(f"  {N_STEPS} steps in {elapsed:.3f}s  ({fps:.1f} FPS)")
    assert fps > 0, "FPS must be positive"
    print("  [PASS]\n")


def test_gpu_memory_after():
    """シミュレーション後の GPU メモリ使用量を表示"""
    print("=" * 60)
    print("[Test 7] GPU memory usage after simulation")
    print("=" * 60)
    import torch
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        print(f"  GPU[{i}]  allocated={alloc:.2f} GB  reserved={reserved:.2f} GB")
    print("  [PASS]\n")


def setup_virtual_display():
    """仮想ディスプレイを起動（ヘッドレス環境用）"""
    import os
    import subprocess
    xvfb_path = "/usr/bin/Xvfb"
    display = ":99"
    try:
        proc = subprocess.Popen(
            [xvfb_path, display, "-screen", "0", "1280x720x24"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.environ["DISPLAY"] = display
        print(f"  Virtual display started (Xvfb {display})")
        return proc
    except Exception as e:
        print(f"  Xvfb start failed: {e}, skipping")
        return None


def main():
    print("\n" + "=" * 60)
    print("  Genesis H100 Smoke Test")
    print("=" * 60 + "\n")

    vdisplay = setup_virtual_display()

    try:
        test_cuda_available()
        gs = test_genesis_import()
        test_genesis_init(gs)
        scene = test_scene_create(gs)
        scene = test_add_entities(gs, scene)
        test_build_and_step(scene)
        test_gpu_memory_after()

        print("=" * 60)
        print("  ALL TESTS PASSED")
        print("=" * 60 + "\n")
        if vdisplay:
            vdisplay.terminate()
        sys.exit(0)

    except Exception as e:
        print(f"\n[FAIL] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        if vdisplay:
            vdisplay.terminate()
        sys.exit(1)


if __name__ == "__main__":
    main()
