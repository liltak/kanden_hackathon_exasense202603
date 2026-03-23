"""
VLA 推論スクリプト (infer.py) ― OpenVLA 7B LoRA

学習済み LoRA モデルを使ってドローンが自律飛行するか確認する

使い方:
  python infer.py --ckpt_dir checkpoints/drone_openvla/best --instruction "ソファに近づけ"
  python infer.py --ckpt_dir checkpoints/drone_openvla/best --output output.mp4 --max_steps 300
"""

import os
import sys

import argparse
import numpy as np
from scipy.spatial.transform import Rotation as ScipyRot
import torch
from PIL import Image as PILImage

sys.path.insert(0, os.path.dirname(__file__))
from action_tokenizer import ActionTokenizer

OBJECTS_DIR = os.path.join(os.path.dirname(__file__), "..", "objects")

OBJECTS = {
    "アームチェア": "modern_arm_chair_01_4k.glb",
    "ソファ":       "sofa_02_4k.glb",
    "木製引き出し": "vintage_wooden_drawer_01_4k.glb",
}

IMG_SIZE = 224
VIDEO_SIZE = 640  # 動画録画用解像度
VIDEO_FPS = 30

# 制御周波数 (Hz) ― シミュレーターは 100Hz なので 10 ステップに 1 回推論
CONTROL_HZ = 10
SIM_HZ = 100
INFER_INTERVAL = SIM_HZ // CONTROL_HZ  # = 10


def load_model(ckpt_dir: str, device: torch.device):
    """LoRA アダプターと ActionTokenizer を読み込んで返す"""
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from peft import PeftModel

    processor = AutoProcessor.from_pretrained(ckpt_dir, trust_remote_code=True)
    base_model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": device},
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, ckpt_dir)
    model.eval()

    stats_path = os.path.join(ckpt_dir, "action_stats.npz")
    action_tokenizer = ActionTokenizer.load(stats_path)
    print(f"モデル読み込み完了: {ckpt_dir}")
    print(f"ActionTokenizer: {action_tokenizer}")
    return model, processor, action_tokenizer


def predict_action(model, processor, pil_image, instruction: str, device, action_tokenizer: ActionTokenizer) -> np.ndarray:
    """画像と命令からアクション 4D を予測して numpy で返す [vx, vy, vz, yaw_rate]"""
    inputs = processor(
        text=instruction,
        images=pil_image,
        return_tensors="pt",
    ).to(device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
        )

    # 生成トークン → テキスト → 256bin デコード → 連続値
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return action_tokenizer.decode(generated_text)


def infer(args):
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("EGL_DEVICE_ID", "0")
    print(f"[ENV] PYOPENGL_PLATFORM = {os.environ['PYOPENGL_PLATFORM']}")
    print(f"[ENV] EGL_DEVICE_ID     = {os.environ['EGL_DEVICE_ID']}")

    import genesis as gs

    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda" if torch.cuda.is_available() else "cpu")

    # モデル読み込み
    model, processor, action_tokenizer = load_model(args.ckpt_dir, device)

    # 動画保存モード: ヘッドレスで実行
    save_video = args.output is not None

    # Genesis セットアップ
    gs.init(backend=gs.cpu, logging_level="debug")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=4, gravity=(0, 0, -9.81)),
        vis_options=gs.options.VisOptions(show_world_frame=False),
        show_viewer=False,
        show_FPS=False,
    )

    scene.add_entity(gs.morphs.Plane())

    # オブジェクト配置
    for name, filename in OBJECTS.items():
        path = os.path.join(OBJECTS_DIR, filename)
        if not os.path.exists(path):
            continue
        pos = (2.0, 2.0, 0.0) if name == "アームチェア" else \
              (-2.0, 2.0, 0.0) if name == "ソファ" else (0.0, 2.5, 0.0)
        scene.add_entity(
            gs.morphs.Mesh(file=path, pos=pos, fixed=True),
            material=gs.materials.Rigid(),
        )
        print(f"[OK] {name} を {pos} に配置")

    drone = scene.add_entity(
        gs.morphs.Drone(file="urdf/drones/cf2x.urdf", pos=(0.0, 0.0, 0.5)),
        material=gs.materials.Rigid(gravity_compensation=1.0),
    )

    # 推論用カメラ (224px)
    fpv_cam = scene.add_camera(
        res=(IMG_SIZE, IMG_SIZE),
        pos=(0.0, 0.1, 0.52),
        lookat=(0.0, 1.0, 0.52),
        fov=90,
        GUI=False,
    )

    # 動画録画用カメラ (640px)
    video_cam = scene.add_camera(
        res=(VIDEO_SIZE, VIDEO_SIZE),
        pos=(0.0, 0.1, 0.52),
        lookat=(0.0, 1.0, 0.52),
        fov=90,
        GUI=False,
    )

    scene.build()

    if save_video:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        video_cam.start_recording()
        print(f"動画保存先: {args.output}  ({args.max_steps} ステップ, {VIDEO_FPS} FPS)")

    hover_rpm = float(np.sqrt(drone.get_mass() * 9.81 / (4 * drone.KF)))

    print(f"\n命令: 「{args.instruction}」")
    print("推論開始...\n")

    current_pos = np.array([0.0, 0.0, 0.5])
    current_yaw = 0.0
    dt = scene.sim_options.dt
    last_action = np.zeros(4, dtype=np.float32)

    step = 0
    try:
        while save_video and step < args.max_steps:
            yaw_rot = ScipyRot.from_euler("z", current_yaw)
            up  = np.array([0., 0., 1.])
            fwd = yaw_rot.apply([0., 1., 0.])
            cam_pos    = current_pos + up * 0.05
            cam_lookat = current_pos + fwd * 3.0

            # 動画カメラは毎ステップ更新・録画
            video_cam.set_pose(pos=cam_pos, lookat=cam_lookat, up=up)
            video_cam.render()

            # 10Hz でのみ推論 (INFER_INTERVAL ステップに 1 回)
            if step % INFER_INTERVAL == 0:
                fpv_cam.set_pose(pos=cam_pos, lookat=cam_lookat, up=up)
                render_out = fpv_cam.render(rgb=True)
                img_raw = render_out[0] if isinstance(render_out, (tuple, list)) else render_out
                pil_image = PILImage.fromarray(np.array(img_raw, dtype=np.uint8))
                last_action = predict_action(model, processor, pil_image, args.instruction, device, action_tokenizer)

            action = last_action
            vx_body  = float(action[0])
            vy_body  = float(action[1])
            vz_body  = float(action[2])
            yaw_rate = float(action[3])
            print(f"[ACTION] raw={action} | vx={vx_body:.3f} vy={vy_body:.3f} vz={vz_body:.3f} yaw={yaw_rate:.3f} | pos={current_pos}")
            current_yaw += yaw_rate * dt
            yaw_rot = ScipyRot.from_euler("z", current_yaw)
            vel_world = yaw_rot.apply(np.array([vx_body, vy_body, vz_body]))

            current_pos = current_pos + vel_world * dt
            current_pos[2] = max(0.15, current_pos[2])
            drone.set_pos(current_pos.tolist())
            q = yaw_rot.as_quat()
            drone.set_quat([q[3], q[0], q[1], q[2]])
            drone.set_propellels_rpm([hover_rpm] * 4)
            scene.step()
            step += 1

            if step % 50 == 0:
                print(f"  {step}/{args.max_steps} ステップ完了")

    finally:
        if save_video:
            video_cam.stop_recording(save_to_filename=args.output, fps=VIDEO_FPS)
            print(f"\n動画保存完了: {args.output} ({step} フレーム)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir",    type=str, required=True,           help="LoRA チェックポイントディレクトリ")
    parser.add_argument("--instruction", type=str, default="ソファに近づけ", help="言語命令")
    parser.add_argument("--output",      type=str, default=None,            help="動画保存パス (.mp4)。指定するとヘッドレスで動画保存")
    parser.add_argument("--max_steps",   type=int, default=300,             help="動画保存モード時の最大ステップ数")
    args = parser.parse_args()
    infer(args)
