"""
VLA 学習データ自動収集スクリプト (collect_v2.py)

【実行環境】H100 (Genesis GPU バックエンド必須)

動作:
  1. 物体1つをランダム配置 (アームチェア)
  2. 命令文を生成
  3. PID コントローラーがターゲットへ自動飛行 (アプローチフェーズ)
  4. ターゲット到達後、周囲を一周回る (オービットフェーズ)
     ─ 周回中はカメラが常にオブジェクトを向く
  5. 一周完了したらリセット → 繰り返し

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
  python collect_v2.py --episodes 5000 --out dataset/
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

# オブジェクト定義: エピソードごとに1つをランダム選択して配置
OBJECTS = {
    "アームチェア": "modern_arm_chair_01_4k.glb",
    "ソファ":       "sofa_02_4k.glb",
    "木製引き出し": "vintage_wooden_drawer_01_4k.glb",
}

# 命令テンプレート
INSTRUCTION_TEMPLATES = [
    "Approach the {name}, fly around it, and take photos.",
    "Go close to the {name} and circle around it to observe.",
    "Navigate to the {name}, then orbit it once while recording.",
    "Fly toward the {name} and do a full loop around it.",
]

# アクション次元 (4D ボディフレーム; convert_to_rlds.py で 7D に拡張)
ACTION_DIM = 4

# 成功判定: ターゲットまでの距離 (m)
SUCCESS_DIST = 1.5

# データ収集周波数 (Hz) ― シミュレーターは 100Hz なので 10 ステップに 1 回記録
CONTROL_HZ = 10
SIM_HZ = 100
RECORD_INTERVAL = SIM_HZ // CONTROL_HZ  # = 10

# 周回速度 (rad/s)
ORBIT_SPEED = 2.0


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

    # ── 全オブジェクトをシーンに追加 (初期位置は床下に隠す) ──
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
        # オブジェクト選択 (--fixed_object 指定時は固定)
        names = list(object_entities.keys())
        if args.fixed_object and args.fixed_object in object_entities:
            object_name = args.fixed_object
        else:
            object_name = np.random.choice(names)
        angle = np.random.uniform(0, 2 * np.pi)
        dist  = args.obj_dist if args.obj_dist is not None else np.random.uniform(2.0, 4.5)
        tx = dist * np.cos(angle)
        ty = dist * np.sin(angle)
        # ドローンをランダム位置にスポーン (ターゲットから 3〜5m) ※先に計算してfacing用に使う
        start_angle = angle + np.random.uniform(np.pi / 2, np.pi)
        start_dist  = np.random.uniform(3.0, 5.0)
        sx = tx + start_dist * np.cos(start_angle)
        sy = ty + start_dist * np.sin(start_angle)

        for name, entity in object_entities.items():
            if name == object_name:
                entity.set_pos([float(tx), float(ty), 0.0])
                # --face_drone 指定時: オブジェクトをドローン方向に向ける
                if args.face_drone:
                    face_yaw = np.arctan2(sy - ty, sx - tx) + np.pi / 2
                    rot = ScipyRot.from_euler("z", face_yaw)
                    q = rot.as_quat()
                    entity.set_quat([q[3], q[0], q[1], q[2]])
            else:
                entity.set_pos([0.0, 0.0, -10.0])
        target_pos = np.array([tx, ty, 0.5])
        current_pos = np.array([sx, sy, np.random.uniform(0.5, 1.5)])
        # 初期ヨー: ターゲット方向を向く + -20〜+20度のランダムヨーオフセット (左右回転)
        dx0 = target_pos[0] - current_pos[0]
        dy0 = target_pos[1] - current_pos[1]
        yaw_offset_deg = args.yaw_offset if args.yaw_offset is not None else np.random.uniform(-45.0, 45.0)
        current_yaw = np.arctan2(dy0, dx0) - np.pi / 2 + np.deg2rad(yaw_offset_deg)

        drone.set_pos(current_pos.tolist())
        drone.set_quat([1, 0, 0, 0])
        pid.reset()

        # 命令文を生成
        template    = np.random.choice(INSTRUCTION_TEMPLATES)
        instruction = template.format(name=object_name)

        # エピソードのステップを記録
        episode_steps = []
        step = 0
        success = False

        # フェーズ管理: "approach" → "orbit" → "hover" → "done"
        phase = "approach"
        hover_step_count = 0
        orbit_angle       = 0.0
        orbit_total_angle = 0.0
        orbit_radius      = SUCCESS_DIST * 0.9
        orbit_height      = 0.5

        # ── ステップループ ──
        while step < args.max_steps:

            # ─ フェーズ別アクション計算 ─
            if phase == "approach":
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
                yaw_rot      = ScipyRot.from_euler("z", current_yaw)
                current_pos  = current_pos + np.array([vx, vy, vz]) * dt
                current_pos[2] = max(0.15, current_pos[2])

                # 到達判定 → オービットフェーズへ移行
                if np.linalg.norm(current_pos - target_pos) < SUCCESS_DIST:
                    phase = "orbit"
                    orbit_angle  = np.arctan2(
                        current_pos[1] - target_pos[1],
                        current_pos[0] - target_pos[0],
                    )
                    orbit_radius      = np.linalg.norm(current_pos[:2] - target_pos[:2])
                    orbit_height      = current_pos[2]
                    orbit_total_angle = 0.0

            elif phase == "orbit":
                orbit_speed    = args.orbit_speed
                orbit_angle   += orbit_speed * dt
                orbit_total_angle += orbit_speed * dt

                # ドローン位置を円軌道上に直接配置
                ox = target_pos[0] + orbit_radius * np.cos(orbit_angle)
                oy = target_pos[1] + orbit_radius * np.sin(orbit_angle)
                current_pos = np.array([ox, oy, orbit_height])

                # 接線方向の速度（記録用）
                tangential = np.array([
                    -np.sin(orbit_angle),
                     np.cos(orbit_angle),
                    0.0,
                ])
                speed_lin = orbit_speed * orbit_radius
                vx, vy, vz = speed_lin * tangential

                # ヨー: オブジェクトを向く
                dx = target_pos[0] - current_pos[0]
                dy = target_pos[1] - current_pos[1]
                current_yaw = np.arctan2(dy, dx) - np.pi / 2
                yaw_rot     = ScipyRot.from_euler("z", current_yaw)
                yaw_rate    = orbit_speed  # 周回中の近似ヨーレート

                # 一周完了 → ホバーフェーズへ
                if orbit_total_angle >= 2 * np.pi:
                    phase = "hover"

            elif phase == "hover":
                # その場でホバー (速度・ヨーレートをゼロに)
                vx, vy, vz = 0.0, 0.0, 0.0
                yaw_rate   = 0.0
                if step % RECORD_INTERVAL == 0:
                    hover_step_count += 1
                if hover_step_count >= args.hover_steps:
                    phase   = "done"
                    success = True
                    success_count += 1

            else:  # "done"
                vx, vy, vz = 0.0, 0.0, 0.0
                yaw_rate   = 0.0

            # ─ Genesis に反映 ─
            drone.set_pos(current_pos.tolist())
            q = yaw_rot.as_quat()
            drone.set_quat([q[3], q[0], q[1], q[2]])
            drone.set_propellels_rpm([hover_rpm] * 4)

            # ─ FPV カメラ更新 ─
            up = np.array([0., 0., 1.])
            if phase in ("orbit", "hover", "done"):
                # オービット中はカメラがオブジェクトを向く
                fpv_cam.set_pose(
                    pos    = current_pos + up * 0.05,
                    lookat = target_pos,
                    up     = up,
                )
            else:
                # アプローチ中は機首方向を向く（自然にターゲット方向）
                fwd = yaw_rot.apply([0., 1., 0.])
                fpv_cam.set_pose(
                    pos    = current_pos + up * 0.05,
                    lookat = current_pos + fwd * 3.0,
                    up     = up,
                )

            scene.step()

            # ─ 10Hz でのみ記録 (RECORD_INTERVAL ステップに 1 回) ─
            if step % RECORD_INTERVAL == 0:
                # 画像取得
                render_out = fpv_cam.render(rgb=True)
                img = render_out[0] if isinstance(render_out, (tuple, list)) else render_out
                img = np.array(img, dtype=np.uint8)

                is_last = (phase == "done") or (step == args.max_steps - 1)

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
                    "phase":         phase,
                    "is_first":      len(episode_steps) == 0,
                    "is_last":       bool(is_last),
                    "is_terminal":   bool(phase == "done"),
                })

                global_step_id += 1

            step += 1

            if phase == "done":
                break

        # ── エピソード保存 ──
        ep_path = episodes_dir / f"episode_{episode_count:05d}.json"
        with open(ep_path, "w", encoding="utf-8") as f:
            json.dump({
                "episode_id":  episode_count,
                "instruction": instruction,
                "target_name": str(object_name),
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
    parser.add_argument("--episodes",    type=int,   default=5000,      help="収集エピソード数")
    parser.add_argument("--max_steps",   type=int,   default=1000,      help="1エピソードの最大ステップ数")
    parser.add_argument("--img_size",    type=int,   default=224,       help="画像解像度 (推奨: 224)")
    parser.add_argument("--out",         type=str,   default="dataset", help="保存先ディレクトリ")
    parser.add_argument("--orbit_speed",  type=float, default=2.0,  help="周回角速度 [rad/s]")
    parser.add_argument("--hover_steps",  type=int,   default=5,    help="orbit後にホバーする記録ステップ数")
    parser.add_argument("--fixed_object", type=str,   default=None, help="固定するオブジェクト名 (例: ソファ)")
    parser.add_argument("--obj_dist",     type=float, default=None,  help="オブジェクト配置距離を固定 [m]")
    parser.add_argument("--face_drone",   action="store_true",       help="オブジェクトをドローン方向に向ける")
    parser.add_argument("--yaw_offset",   type=float, default=None,  help="初期ヨーオフセットを固定 [度] (未指定時はランダム)")
    parser.add_argument("--show_viewer",  action="store_true",       help="ビューアを表示する")
    args = parser.parse_args()
    collect(args)
