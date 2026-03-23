"""
VLA 学習データ自動収集スクリプト (collect.py)

【実行環境】H100 (Genesis GPU バックエンド必須)

動作:
  1. 物体3つをランダム配置 (アームチェア / ソファ / 木製引き出し)
  2. ターゲットをランダム選択し命令文を生成
  3. PID コントローラーがターゲットへ自動飛行
  4. 毎ステップ「FPV画像 + 命令 + ドローン移動量」を保存
  5. 1.5m 以内に到達したら成功 → リセット → 繰り返し

出力ディレクトリ構造 (convert_to_rlds.py と同じ形式):
  dataset/
    episodes/episode_XXXXX.json   ← ステップメタデータ
    steps/step_XXXXXX.jpg         ← FPV 画像
    dataset_info.json             ← データセット情報

アクション (4次元, ドローンボディフレーム):
  [vx_body, vy_body, vz_body, yaw_rate]
  vx_body: 機首方向の速度 [m/s]
  vy_body: 機体左方向の速度 [m/s]
  vz_body: 上方向の速度 [m/s]
  yaw_rate: ヨー角速度 [rad/s]

使い方:
  python collect.py --episodes 5000 --out dataset/
"""

import os

import argparse
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as ScipyRot
import genesis as gs

# objectsフォルダのパス
OBJECTS_DIR = os.path.join(os.path.dirname(__file__), "..", "objects")

# オブジェクト定義: {命令に使う名前: GLBファイル名}
OBJECTS = {
    "アームチェア": "modern_arm_chair_01_4k.glb",
    "ソファ":       "sofa_02_4k.glb",
    "木製引き出し": "vintage_wooden_drawer_01_4k.glb",
}

# 命令テンプレート
INSTRUCTION_TEMPLATES = [
    "{name}に近づけ",
    "{name}の前に移動しろ",
    "{name}まで飛べ",
    "{name}に接近せよ",
]

# アクション次元 (4D ボディフレーム; convert_to_rlds.py で 7D に拡張)
ACTION_DIM = 4

# 成功判定: ターゲットまでの距離 (m)
SUCCESS_DIST = 1.5

# データ収集周波数 (Hz) ― シミュレーターは 100Hz なので 10 ステップに 1 回記録
CONTROL_HZ = 10
SIM_HZ = 100
RECORD_INTERVAL = SIM_HZ // CONTROL_HZ  # = 10


def to_np(t):
    return t.cpu().numpy().flatten() if hasattr(t, "cpu") else np.array(t).flatten()


class PIDController:
    """ドローンをターゲット座標へ誘導するシンプルなPID"""
    def __init__(self, kp=1.2, kd=0.4):
        self.kp = kp
        self.kd = kd
        self.prev_error = np.zeros(3)

    def compute(self, current_pos, target_pos, dt):
        error = target_pos - current_pos
        d_error = (error - self.prev_error) / dt
        self.prev_error = error.copy()
        vel = self.kp * error + self.kd * d_error
        speed = np.linalg.norm(vel)
        if speed > 1.5:
            vel = vel / speed * 1.5
        return vel  # [vx, vy, vz]

    def reset(self):
        self.prev_error = np.zeros(3)


def collect(args):
    gs.init(backend=gs.cpu, logging_level="debug")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=4, gravity=(0, 0, -9.81)),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -4.0, 2.0),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=45,
            max_FPS=60,
        ),
        vis_options=gs.options.VisOptions(show_world_frame=False),
        show_viewer=args.show_viewer,
        show_FPS=False,
    )

    scene.add_entity(gs.morphs.Plane())

    # ── オブジェクトを追加 ──
    object_entities = {}
    for name, filename in OBJECTS.items():
        path = os.path.join(OBJECTS_DIR, filename)
        if not os.path.exists(path):
            print(f"[SKIP] {name}: {path} が見つかりません")
            continue
        entity = scene.add_entity(
            gs.morphs.Mesh(file=path, pos=(0, 0, -10), fixed=True),
            material=gs.materials.Rigid(),
        )
        object_entities[name] = entity
        print(f"[OK] {name} を追加")

    if not object_entities:
        print("オブジェクトが1つもありません。objects/ フォルダを確認してください。")
        return

    # ── ドローン追加 ──
    drone = scene.add_entity(
        gs.morphs.Drone(file="urdf/drones/cf2x.urdf", pos=(0.0, 0.0, 0.5)),
        material=gs.materials.Rigid(gravity_compensation=1.0),
    )

    # ── FPV カメラ（学習用画像取得）──
    fpv_cam = scene.add_camera(
        res=(args.img_size, args.img_size),
        pos=(0.0, 0.1, 0.52),
        lookat=(0.0, 1.0, 0.52),
        fov=90,
        GUI=False,
    )

    scene.build()

    hover_rpm = float(np.sqrt(drone.get_mass() * 9.81 / (4 * drone.KF)))
    pid = PIDController()
    out_dir = Path(args.out)
    steps_dir = out_dir / "steps"
    episodes_dir = out_dir / "episodes"
    steps_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir.mkdir(parents=True, exist_ok=True)

    import PIL.Image as PILImage

    dt = scene.sim_options.dt
    episode_count = 0
    success_count = 0
    global_step_id = 0

    print(f"\n収集開始: {args.episodes} エピソード → {out_dir}/\n")

    while episode_count < args.episodes:

        # ── エピソード初期化 ──
        names = list(object_entities.keys())
        placed_positions = []
        for name in names:
            for _ in range(50):
                angle = np.random.uniform(0, 2 * np.pi)
                dist  = np.random.uniform(2.0, 4.5)
                px = dist * np.cos(angle)
                py = dist * np.sin(angle)
                ok = all(np.linalg.norm([px - ox, py - oy]) > 1.5
                         for ox, oy in placed_positions)
                if ok:
                    break
            object_entities[name].set_pos([float(px), float(py), 0.0])
            placed_positions.append((px, py))

        # ターゲットをランダム選択
        target_name = np.random.choice(names)
        target_idx  = names.index(target_name)
        tx, ty      = placed_positions[target_idx]
        target_pos  = np.array([tx, ty, 0.5])

        # ドローンをランダム位置にスポーン
        angle        = np.arctan2(ty, tx)
        start_angle  = angle + np.random.uniform(np.pi / 2, np.pi)
        start_dist   = np.random.uniform(3.0, 5.0)
        sx = start_dist * np.cos(start_angle)
        sy = start_dist * np.sin(start_angle)
        current_pos = np.array([sx, sy, np.random.uniform(0.5, 1.5)])
        current_yaw = np.random.uniform(0, 2 * np.pi)

        drone.set_pos(current_pos.tolist())
        drone.set_quat([1, 0, 0, 0])
        pid.reset()

        # 命令文を生成
        template    = np.random.choice(INSTRUCTION_TEMPLATES)
        instruction = template.format(name=target_name)

        # エピソードのステップを記録
        episode_steps = []
        step = 0
        success = False

        # ── ステップループ ──
        hover_step_count = 0  # 到達後のホバーステップ数（記録単位）
        while step < args.max_steps:
            if success:
                # ターゲット到達後はホバー（速度・ヨーレートをゼロに）
                vx, vy, vz, yaw_rate = 0.0, 0.0, 0.0, 0.0
            else:
                # PID で速度計算
                vel = pid.compute(current_pos, target_pos, dt)
                vel = np.array(vel).flatten()[:3]
                vx, vy, vz = vel[0], vel[1], vel[2]

                # ヨーをターゲット方向に向ける
                dx = target_pos[0] - current_pos[0]
                dy = target_pos[1] - current_pos[1]
                target_yaw = np.arctan2(dy, dx) - np.pi / 2
                yaw_error  = target_yaw - current_yaw
                yaw_error  = (yaw_error + np.pi) % (2 * np.pi) - np.pi
                yaw_rate   = np.clip(yaw_error * 3.0, -1.5, 1.5)
            current_yaw += yaw_rate * dt

            # 位置更新
            yaw_rot     = ScipyRot.from_euler("z", current_yaw)
            current_pos = current_pos + np.array([vx, vy, vz]) * dt
            current_pos[2] = max(0.15, current_pos[2])

            # Genesis に反映
            drone.set_pos(current_pos.tolist())
            q = yaw_rot.as_quat()
            drone.set_quat([q[3], q[0], q[1], q[2]])
            drone.set_propellels_rpm([hover_rpm] * 4)

            # FPV カメラ更新
            fwd = yaw_rot.apply([0., 1., 0.])
            up  = np.array([0., 0., 1.])
            fpv_cam.set_pose(
                pos    = current_pos + up * 0.05,
                lookat = current_pos + fwd * 3.0,
                up     = up,
            )

            scene.step()

            # 成功判定
            dist_to_target = np.linalg.norm(current_pos - target_pos)
            reached = dist_to_target < SUCCESS_DIST

            # 10Hz でのみ記録 (RECORD_INTERVAL ステップに 1 回)
            if step % RECORD_INTERVAL == 0:
                # 画像取得
                render_out = fpv_cam.render(rgb=True)
                img = render_out[0] if isinstance(render_out, (tuple, list)) else render_out
                img = np.array(img, dtype=np.uint8)

                is_last = reached or (step == args.max_steps - 1)

                # 画像保存 (JPG)
                img_filename = f"step_{global_step_id:06d}.jpg"
                PILImage.fromarray(img).save(steps_dir / img_filename, quality=95)

                # ワールド座標系 → ドローンボディフレームに変換
                yaw_rot_inv = ScipyRot.from_euler("z", -current_yaw)
                vel_body = yaw_rot_inv.apply(np.array([vx, vy, vz]))
                action_4d = [
                    float(vel_body[0]),  # vx_body: 機首方向
                    float(vel_body[1]),  # vy_body: 機体左方向
                    float(vel_body[2]),  # vz_body: 上方向
                    float(yaw_rate),     # yaw_rate
                ]

                episode_steps.append({
                    "step_id":       global_step_id,
                    "episode_id":    episode_count,
                    "image_path":    f"steps/{img_filename}",
                    "instruction":   instruction,
                    "action_vector": action_4d,
                    "is_first":      step == 0,
                    "is_last":       bool(is_last),
                    "is_terminal":   bool(reached),
                })

                if success:
                    hover_step_count += 1

                global_step_id += 1

            step += 1

            if reached and not success:
                success = True
                success_count += 1

            if success and hover_step_count >= args.hover_steps:
                break

        # ── エピソード保存 ──
        ep_path = episodes_dir / f"episode_{episode_count:05d}.json"
        with open(ep_path, "w", encoding="utf-8") as f:
            json.dump({
                "episode_id":  episode_count,
                "instruction": instruction,
                "target_name": str(target_name),
                "target_pos":  [float(v) for v in target_pos],
                "success":     bool(success),
                "steps":       episode_steps,
            }, f, ensure_ascii=False, indent=2)

        episode_count += 1
        if episode_count % 100 == 0:
            print(f"  {episode_count}/{args.episodes} エピソード完了 "
                  f"(成功率: {success_count/episode_count*100:.1f}%)")

    # ── データセット情報保存 ──
    with open(out_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_episodes":    episode_count,
            "n_steps":       global_step_id,
            "img_size":      args.img_size,
            "action_dim":    ACTION_DIM,
            "action_format": "[vx_body, vy_body, vz_body, yaw_rate]",
            "success_rate":  success_count / episode_count,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n収集完了: {episode_count} エピソード, 成功率: {success_count/episode_count*100:.1f}%")
    print(f"保存先: {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",    type=int,  default=5000,      help="収集エピソード数")
    parser.add_argument("--max_steps",   type=int,  default=300,       help="1エピソードの最大ステップ数")
    parser.add_argument("--img_size",    type=int,  default=224,       help="画像解像度 (推奨: 224)")
    parser.add_argument("--out",         type=str,  default="dataset", help="保存先ディレクトリ")
    parser.add_argument("--hover_steps",  type=int,  default=5,        help="到達後にホバーする記録ステップ数")
    parser.add_argument("--show_viewer", action="store_true",          help="ビューアを表示する")
    args = parser.parse_args()
    collect(args)
