"""
一本線のクラック画像を生成するスクリプト。
- 枝なし・交差なし
- 最大45度のカーブ
- リアルなひび割れ表現
- 画像生成時に正確なアノテーションも同時出力

使い方:
  python generate_crack.py               # デフォルト: 100エピソード
  python generate_crack.py --n 300       # 300エピソード生成
  python generate_crack.py --n 10 --seed 42  # 再現可能な生成
"""
import argparse
import os, math, random, json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ─── 引数パース ───────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(description="クラック追従データセット生成")
_parser.add_argument("--n",    type=int, default=100,
                     help="生成するエピソード数 (デフォルト: 100)")
_parser.add_argument("--seed", type=int, default=0,
                     help="乱数シード (デフォルト: 0)")
_parser.add_argument("--out",    type=str, default=None,
                     help="出力ディレクトリ (デフォルト: crack_generated)")
_parser.add_argument("--bg_dir", type=str, default=None,
                     help="背景画像フォルダ (指定時は合成テクスチャの代わりに使用)")
_args = _parser.parse_args()

N_EPISODES = _args.n
BASE_SEED   = _args.seed

OUT_DIR   = _args.out if _args.out else "crack_generated"
ANNOT_DIR = f"{OUT_DIR}/annotations"

# 背景画像リストを読み込む
_BG_IMAGES = []
if _args.bg_dir:
    from pathlib import Path as _Path
    _exts = {".jpg", ".jpeg", ".png", ".bmp"}
    _BG_IMAGES = sorted(p for p in _Path(_args.bg_dir).iterdir() if p.suffix.lower() in _exts)
    print(f"背景画像: {len(_BG_IMAGES)} 枚 ({_args.bg_dir})")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(ANNOT_DIR, exist_ok=True)

W, H = 4096, 4096
ACTION_DIM = 7
INSTRUCTION = "クラックを追従してください"
PATCH_SIZE = 224
ROI_OVERLAP = 0.5   # ROIの重なり率（0.5 = 半分重なる）


def generate_crack_path(seed=None, max_curve_deg=45, deform="none",
                        deform_amplitude=0.08, deform_freq=1.5):
    """
    画像端から端へ一本のクラックパスを生成する。

    Parameters
    ----------
    max_curve_deg : float
        全体の最大方向変化 (度)
    deform : str
        変形モード
        "none"  : 変形なし（緩やかなカーブのみ）
        "wave"  : サイン波状の S 字変形
        "zigzag": ランダムなジグザグ変形
        "bend"  : 途中で大きく1回折れ曲がる
    deform_amplitude : float
        変形の強さ (0〜1)。画像幅に対する割合
    deform_freq : float
        wave モードの周波数（波の数）
    """
    rng = random.Random(seed)

    # 開始位置: 左端か上端からランダムに
    side = rng.choice(["left", "top", "bottom"])
    if side == "left":
        sx, sy = 0, rng.randint(H // 4, 3 * H // 4)
        base_angle = rng.uniform(-30, 30)
    elif side == "top":
        sx, sy = rng.randint(W // 4, 3 * W // 4), 0
        base_angle = rng.uniform(60, 120)
    else:
        sx, sy = rng.randint(W // 4, 3 * W // 4), H
        base_angle = rng.uniform(-120, -60)

    scale = W / 512
    step_len = rng.uniform(2.5, 4.0) * scale
    max_steps = int(math.sqrt(W**2 + H**2) / step_len * 1.2)
    max_delta_per_step = max_curve_deg / max_steps * 3

    # bend モード: 折れ曲がり位置と追加角度を事前に決める
    bend_pos = rng.uniform(0.35, 0.65)          # 全ステップ中の折れ位置
    bend_extra = rng.choice([-1, 1]) * rng.uniform(30, 45)  # 追加回転量

    points = [(sx, sy)]
    x, y = float(sx), float(sy)
    angle = base_angle

    for step_i in range(max_steps):
        t = step_i / max(max_steps - 1, 1)     # 0→1 の進捗

        # ── ベースのゆらぎ ──────────────────────────────────────
        delta = rng.uniform(-max_delta_per_step, max_delta_per_step)
        angle += delta
        angle = max(base_angle - max_curve_deg,
                    min(base_angle + max_curve_deg, angle))

        # ── 変形モードによる追加回転 ────────────────────────────
        if deform == "wave":
            # サイン波で方向を振る
            wave = math.sin(t * math.pi * 2 * deform_freq)
            angle_use = angle + wave * deform_amplitude * 90
        elif deform == "zigzag":
            # 一定間隔でランダムに大きく振れる
            period = max(1, max_steps // int(deform_freq * 4))
            if step_i % period == 0:
                zigzag_dir = rng.choice([-1, 1])
            angle_use = angle + zigzag_dir * deform_amplitude * 60
        elif deform == "bend":
            # bend_pos 付近で一気に折れる（滑らかに補間）
            blend = max(0.0, min(1.0, (t - bend_pos) / 0.08))
            angle_use = angle + blend * bend_extra * deform_amplitude * 2
        else:
            angle_use = angle

        step = step_len + rng.uniform(-0.5, 0.5)
        x += math.cos(math.radians(angle_use)) * step
        y += math.sin(math.radians(angle_use)) * step
        points.append((x, y))

        if x < -10 or x > W + 10 or y < -10 or y > H + 10:
            break

    return points


def draw_crack(points, width_base=2.5, seed=None):
    """
    クラックパスをリアルに描画する。
    - 幅が中央で太く、端で細い
    - わずかなジッターで自然な揺れ
    - 多重描画でエッジ感を出す
    """
    rng = random.Random(seed)
    img = Image.new("L", (W, H), 255)   # 白背景（L=グレースケール）
    draw = ImageDraw.Draw(img)

    n = len(points)
    scale = W / 512  # 線幅もサイズに比例してスケール

    # 各点のジッターを事前に1回だけ決める → 隣接セグメントで同じ座標を共有
    core_jitter = [(rng.uniform(-0.2, 0.2) * scale,
                    rng.uniform(-0.2, 0.2) * scale) for _ in range(n)]

    shadow_pts = [(points[i][0], points[i][1]) for i in range(n)]
    core_pts   = [(points[i][0] + core_jitter[i][0],
                   points[i][1] + core_jitter[i][1]) for i in range(n)]

    # ── Step1: 影レイヤー（広め・薄い灰色）→ 大きめブラーでふんわり ──
    shadow_w = int((width_base * 4.0 + 2.0) * scale)
    draw.line(shadow_pts, fill=160, width=max(1, shadow_w))
    img = img.filter(ImageFilter.GaussianBlur(scale * 3.0))

    # ── Step2: 芯レイヤー（中幅・暗い）→ 中ブラーでエッジを柔らかく ─
    draw = ImageDraw.Draw(img)
    core_w = int((width_base * 0.8 + 0.8) * scale)
    draw.line(core_pts, fill=15, width=max(1, core_w))
    img = img.filter(ImageFilter.GaussianBlur(scale * 1.2))

    # ── Step3: 中心線（細い・最も暗い）→ 軽くブラーして馴染ませる ───
    draw = ImageDraw.Draw(img)
    draw.line(shadow_pts, fill=0, width=max(1, int(scale * 0.8)))
    img = img.filter(ImageFilter.GaussianBlur(scale * 0.5))

    # ── ノイズでリアリティ向上 ──────────────────────────────────────
    arr = np.array(img).astype(float)
    noise = np.random.default_rng(seed if seed else 0).uniform(-6, 6, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    return img   # グレースケール: 白=背景, 黒=クラック


def make_texture(name, w=W, h=H, seed=0):
    """シンプルなテクスチャ生成"""
    rng = np.random.default_rng(seed)
    if name == "concrete":
        base = rng.integers(155, 195, (h, w), dtype=np.uint8)
        for s in [4, 16, 64]:
            n = rng.integers(-18, 18, (h//s, w//s))
            big = np.array(Image.fromarray(n.astype(np.int8)).resize((w, h), Image.BILINEAR))
            base = np.clip(base.astype(int) + big, 100, 220).astype(np.uint8)
        img = Image.fromarray(base).convert("RGB")
        return img.filter(ImageFilter.GaussianBlur(1))

    elif name == "stone":
        r = np.clip(rng.integers(170, 195, (h, w), dtype=np.uint8).astype(int)
                    + rng.integers(-20, 20, (h, w)), 100, 220).astype(np.uint8)
        g = np.clip(r.astype(int) - rng.integers(10, 25, (h, w)), 80, 200).astype(np.uint8)
        b = np.clip(g.astype(int) - rng.integers(5, 20, (h, w)), 70, 190).astype(np.uint8)
        img = Image.fromarray(np.stack([r,g,b], 2))
        return img.filter(ImageFilter.GaussianBlur(1.2))

    elif name == "wood":
        arr = np.zeros((h, w, 3), np.uint8)
        for x in range(w):
            grain = int(18 * math.sin(x * 0.06 + rng.uniform(0, 1)))
            noise = int(rng.integers(-8, 8))
            arr[:, x] = [
                np.clip(158 + grain + noise, 100, 210),
                np.clip(98  + grain//2 + noise, 55, 140),
                np.clip(55  + noise, 25, 90),
            ]
        img = Image.fromarray(arr)
        return img.filter(ImageFilter.GaussianBlur(0.8))

    elif name == "asphalt":
        base = rng.integers(50, 80, (h, w), dtype=np.uint8)
        for s in [2, 8, 32]:
            n = rng.integers(-12, 12, (h//s, w//s))
            big = np.array(Image.fromarray(n.astype(np.int8)).resize((w, h), Image.BILINEAR))
            base = np.clip(base.astype(int) + big, 30, 100).astype(np.uint8)
        return Image.fromarray(base).convert("RGB").filter(ImageFilter.GaussianBlur(0.5))

    else:
        return Image.new("RGB", (w, h), (180, 170, 160))


def apply_crack_to_texture(texture_rgb, crack_gray, opacity=1.0):
    """
    クラック（グレースケール: 白=背景, 黒=傷）をテクスチャに合成する。
    Multiply合成で自然に馴染ませる。
    """
    tex = np.array(texture_rgb).astype(float)
    crack = np.array(crack_gray).astype(float)

    # 0(黒=傷)〜255(白=背景) → マスク: 0=傷なし, 1=傷あり
    crack_strength = ((255 - crack) / 255.0) * opacity  # (H,W)

    # 傷部分はテクスチャを暗くする（Multiply的）
    out = tex * (1.0 - crack_strength[:,:,None] * 0.97)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


# ─────────────────────────────────────────
# アノテーション保存（annotate.py 互換形式）
# ─────────────────────────────────────────

def subsample_path(path, patch_size=PATCH_SIZE, overlap=ROI_OVERLAP):
    """
    ROIが overlap の割合で重なるよう、累積弧長で等間隔サブサンプリング。
    ステップ幅 = patch_size × (1 - overlap)
    例: patch_size=224, overlap=0.5 → 112px ごとに1点
    """
    step = patch_size * (1 - overlap)
    arc = [0.0]
    for i in range(1, len(path)):
        dx = path[i][0] - path[i-1][0]
        dy = path[i][1] - path[i-1][1]
        arc.append(arc[-1] + math.sqrt(dx*dx + dy*dy))
    total = arc[-1]

    result, j = [], 0
    t = 0.0
    while t <= total:
        while j < len(arc) - 1 and arc[j+1] < t:
            j += 1
        result.append(path[j])
        t += step

    return result


def crop_patch(img_arr, x, y, patch_size):
    """(x, y) 中心の patch_size×patch_size クロップ（ゼロパディング）。"""
    h, w = img_arr.shape[:2]
    half = patch_size // 2
    canvas = np.zeros((h + patch_size, w + patch_size, 3), dtype=np.uint8)
    canvas[half:half+h, half:half+w] = img_arr
    cx, cy = x + half, y + half
    return canvas[cy-half:cy+half, cx-half:cx+half]


def save_annotation(image_pil, path_float, episode_id, image_filename):
    """
    生成時の正確なパスからアノテーションを保存する。
    - path_float: generate_crack_path() の戻り値（float座標）
    - image_pil: 合成済み PIL 画像
    """
    # 画像内に収まる点のみ整数化
    path_int = [
        (int(round(x)), int(round(y)))
        for x, y in path_float
        if 0 <= x < W and 0 <= y < H
    ]

    # ROI重なり率に基づくサブサンプリング
    waypoints = subsample_path(path_int, PATCH_SIZE, ROI_OVERLAP)

    # パッチ画像保存先
    steps_dir = f"{ANNOT_DIR}/steps"
    os.makedirs(steps_dir, exist_ok=True)
    img_arr = np.array(image_pil.convert("RGB"))

    steps = []
    step_offset = sum(  # 前エピソードまでの累積ステップ数
        0 for _ in range(episode_id)  # save_annotation 呼び出し時に外から渡す方式に移行
    )
    step_offset = episode_id * 1000  # エピソードIDで大きめにオフセット（衝突回避）

    for i, (x, y) in enumerate(waypoints):
        step_id = step_offset + i
        patch = crop_patch(img_arr, x, y, PATCH_SIZE)

        patch_fname = f"step_{step_id:06d}.png"
        Image.fromarray(patch).save(f"{steps_dir}/{patch_fname}")

        if i < len(waypoints) - 1:
            nx, ny = waypoints[i + 1]
            action = [float(nx - x), float(ny - y)] + [0.0] * (ACTION_DIM - 2)
        else:
            action = [0.0] * ACTION_DIM

        steps.append({
            "step_id":           step_id,
            "episode_id":        episode_id,
            "image_path":        f"steps/{patch_fname}",
            "instruction":       INSTRUCTION,
            "action_vector":     action,
            "pixel_x":           x,
            "pixel_y":           y,
            "is_first":          i == 0,
            "is_last":           i == len(waypoints) - 1,
            "annotation_method": "generated",   # 生成時の正確なパス
        })

    episode = {
        "episode_id":        episode_id,
        "source_image":      image_filename,
        "patch_size":        PATCH_SIZE,
        "annotation_method": "generated",
        "steps":             steps,
    }

    ep_path = f"{ANNOT_DIR}/episode_{episode_id:04d}.json"
    with open(ep_path, "w") as f:
        json.dump(episode, f, indent=2, ensure_ascii=False)

    return ep_path


# ─────────────────────────────────────────
# コンフィグ自動生成
# ─────────────────────────────────────────

TEXTURES  = ["concrete", "stone", "wood", "asphalt"]
DEFORMS   = ["none", "wave", "zigzag", "bend"]
WIDTHS    = [2.5, 3.0, 3.5]

# deform ごとの amplitude 範囲 (min, max)
AMPLITUDE_RANGE = {
    "none":   (0.0,  0.0),
    "wave":   (0.06, 0.15),
    "zigzag": (0.06, 0.15),
    "bend":   (0.6,  1.2),
}

def _build_configs(n: int, base_seed: int) -> list:
    """n 件のコンフィグ (seed, texture, width, deform, amplitude) をランダム生成。"""
    rng = random.Random(base_seed)

    # まず全組み合わせを基本セットとして網羅（4tex × 4deform × 3width = 48通り）
    base = []
    s = 0
    for tex in TEXTURES:
        for deform in DEFORMS:
            for width in WIDTHS:
                amin, amax = AMPLITUDE_RANGE[deform]
                amp = amin if amin == amax else round(rng.uniform(amin, amax), 3)
                base.append((s, tex, width, deform, amp))
                s += 1

    if n <= len(base):
        return base[:n]

    # 基本セットを超える分はランダムに追加
    configs = list(base)
    for i in range(len(base), n):
        tex    = rng.choice(TEXTURES)
        deform = rng.choice(DEFORMS)
        width  = round(rng.choice(WIDTHS) + rng.uniform(-0.2, 0.2), 2)
        width  = max(2.0, min(4.0, width))
        amin, amax = AMPLITUDE_RANGE[deform]
        amp = amin if amin == amax else round(rng.uniform(amin, amax), 3)
        seed = len(base) * 10 + i * 7 + base_seed
        configs.append((seed, tex, width, deform, amp))

    return configs


configs = _build_configs(N_EPISODES, BASE_SEED)
print(f"エピソード数: {len(configs)}  (--n {N_EPISODES}, --seed {BASE_SEED})")

for episode_id, (seed, tex_name, width, deform, amplitude) in enumerate(configs):
    # クラックパス生成（正解座標）
    path = generate_crack_path(seed=seed, max_curve_deg=45,
                               deform=deform, deform_amplitude=amplitude)
    # クラック描画
    crack_img = draw_crack(path, width_base=width, seed=seed)
    # 背景: bg_dir 指定時は自然画像、それ以外は合成テクスチャ
    if _BG_IMAGES:
        bg_path = _BG_IMAGES[episode_id % len(_BG_IMAGES)]
        texture = Image.open(str(bg_path)).resize((W, H)).convert("RGB")
        bg_label = bg_path.stem
    else:
        texture = make_texture(tex_name, seed=seed)
        bg_label = tex_name
    # 合成
    result = apply_crack_to_texture(texture, crack_img, opacity=1.0)
    # 画像保存
    img_fname = f"{episode_id:04d}_{bg_label}_{deform}_w{width}.png"
    result.save(f"{OUT_DIR}/{img_fname}")
    # アノテーション保存（生成時の正確なパスを使用）
    ep_path = save_annotation(result, path, episode_id, img_fname)
    print(f"[{episode_id+1:4d}/{len(configs)}] {img_fname}  →  {ep_path}")

print(f"\n完了: {len(configs)}枚 + アノテーションを {OUT_DIR}/ に保存")
