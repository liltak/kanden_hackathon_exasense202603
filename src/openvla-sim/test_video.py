"""
動画撮影テスト (test_video.py)

Genesis を起動してカメラ映像を数秒録画し、MP4 として保存する。
推論なしで動画撮影パイプラインだけを検証する。

使い方:
  python test_video.py
  python test_video.py --output logs/test_video.mp4 --steps 90
"""

import os
import argparse


VIDEO_FPS = 30
IMG_SIZE  = 1024


def main(args):
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("EGL_DEVICE_ID", "0")
    print(f"[ENV] PYOPENGL_PLATFORM = {os.environ['PYOPENGL_PLATFORM']}")
    print(f"[ENV] EGL_DEVICE_ID     = {os.environ['EGL_DEVICE_ID']}")

    import genesis as gs

    print("[1] gs.init ...")
    gs.init(backend=gs.cpu, logging_level="debug")

    print("[2] シーン作成 ...")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=4),
        vis_options=gs.options.VisOptions(show_world_frame=False),
        show_viewer=False,
        show_FPS=False,
    )

    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(
        gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0.0, 1.0, 0.1)),
        surface=gs.surfaces.Default(color=(0.2, 0.6, 1.0, 1.0)),
    )

    cam = scene.add_camera(
        res=(IMG_SIZE, IMG_SIZE),
        pos=(0.0, -2.0, 1.5),
        lookat=(0.0, 0.0, 0.1),
        fov=60,
        GUI=False,
    )

    print("[3] scene.build ...")
    scene.build()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    print(f"[4] 録画開始: {args.steps} ステップ → {args.output}")
    cam.start_recording()
    for step in range(args.steps):
        scene.step()
        cam.render()
        if (step + 1) % 30 == 0:
            print(f"  {step + 1}/{args.steps} ステップ完了")
    cam.stop_recording(save_to_filename=args.output, fps=VIDEO_FPS)

    print(f"[OK] 動画保存完了: {args.output}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="logs/test_video.mp4", help="保存先 MP4 パス")
    parser.add_argument("--steps",  type=int, default=90,                    help="録画ステップ数 (30step=1秒)")
    args = parser.parse_args()
    exit(main(args))
