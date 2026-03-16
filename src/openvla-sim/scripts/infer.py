"""
VLA 推論スクリプト (infer.py) ― OpenVLA 7B LoRA

学習済み LoRA モデルを使ってドローンが自律飛行するか確認する

使い方:
  python infer.py --ckpt_dir checkpoints/drone_openvla/best --instruction "ソファに近づけ"
"""

import os

import argparse
import numpy as np
from scipy.spatial.transform import Rotation as ScipyRot
import torch
import genesis as gs
from PIL import Image as PILImage

OBJECTS_DIR = os.path.join(os.path.dirname(__file__), "..", "objects")

OBJECTS = {
    "アームチェア": "modern_arm_chair_01_4k.glb",
    "ソファ":       "sofa_02_4k.glb",
    "木製引き出し": "vintage_wooden_drawer_01_4k.glb",
}

IMG_SIZE = 224
MAX_VEL  = 1.5


def load_model(ckpt_dir: str, device: torch.device):
    """LoRA アダプターを読み込んで OpenVLA モデルを返す"""
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from peft import PeftModel

    processor = AutoProcessor.from_pretrained(ckpt_dir, trust_remote_code=True)
    base_model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, ckpt_dir)
    model.eval()
    print(f"モデル読み込み完了: {ckpt_dir}")
    return model, processor


def predict_action(model, processor, pil_image, instruction: str, device) -> np.ndarray:
    """画像と命令からアクション 4D を予測して numpy で返す [vx, vy, vz, yaw_rate]"""
    inputs = processor(
        text=instruction,
        images=pil_image,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
        )

    # 生成トークン → テキスト → float リスト
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    tokens = generated_text.split()

    # 末尾の 4 つを数値として取得（生成テキストは instruction + action）
    action_4d = np.zeros(4, dtype=np.float32)
    num_tokens = []
    for t in reversed(tokens):
        try:
            num_tokens.insert(0, float(t))
            if len(num_tokens) == 4:
                break
        except ValueError:
            if num_tokens:
                break

    for i, v in enumerate(num_tokens[:4]):
        action_4d[i] = v

    return action_4d


def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # モデル読み込み
    model, processor = load_model(args.ckpt_dir, device)

    # Genesis セットアップ
    gs.init(backend=gs.cpu)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=4, gravity=(0, 0, -9.81)),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -4.0, 2.0),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=45,
            max_FPS=60,
        ),
        vis_options=gs.options.VisOptions(show_world_frame=False),
        show_viewer=True,
        show_FPS=True,
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
    scene.viewer.follow_entity(drone)

    fpv_cam = scene.add_camera(
        res=(IMG_SIZE, IMG_SIZE),
        pos=(0.0, 0.1, 0.52),
        lookat=(0.0, 1.0, 0.52),
        fov=90,
        GUI=True,
    )

    scene.build()

    hover_rpm = float(np.sqrt(drone.get_mass() * 9.81 / (4 * drone.KF)))

    print(f"\n命令: 「{args.instruction}」")
    print("推論開始... (ESC で終了)\n")

    current_pos = np.array([0.0, 0.0, 0.5])
    current_yaw = 0.0
    dt = scene.sim_options.dt

    while scene.viewer.is_alive():
        # FPV 画像取得
        fwd = ScipyRot.from_euler("z", current_yaw).apply([0., 1., 0.])
        up  = np.array([0., 0., 1.])
        fpv_cam.set_pose(
            pos    = current_pos + up * 0.05,
            lookat = current_pos + fwd * 3.0,
            up     = up,
        )
        render_out = fpv_cam.render(rgb=True)
        img_raw = render_out[0] if isinstance(render_out, (tuple, list)) else render_out

        # OpenVLA 用に PIL Image に変換
        pil_image = PILImage.fromarray(np.array(img_raw, dtype=np.uint8))

        # モデル推論 → 4D アクション [vx, vy, vz, yaw_rate]
        action_4d = predict_action(model, processor, pil_image, args.instruction, device)

        vx_body  = float(np.clip(action_4d[0], -MAX_VEL, MAX_VEL))
        vy_body  = float(np.clip(action_4d[1], -MAX_VEL, MAX_VEL))
        vz_body  = float(np.clip(action_4d[2], -MAX_VEL, MAX_VEL))
        yaw_rate = float(np.clip(action_4d[3], -1.5, 1.5))

        # ボディフレーム → ワールド座標系に変換して位置更新
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir",    type=str, required=True,           help="LoRA チェックポイントディレクトリ")
    parser.add_argument("--instruction", type=str, default="ソファに近づけ", help="言語命令")
    args = parser.parse_args()
    infer(args)
