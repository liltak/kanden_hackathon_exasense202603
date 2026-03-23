"""
画像をクリックして起点を選び、rollout.py を実行するスクリプト

使い方:
  # 単一画像
  python training/pick_start.py \
    --ckpt_dir checkpoints/crack_openvla/best \
    --image    images/struppi0601-crack-695010.jpg

  # フォルダ内の全画像を順番に処理
  python training/pick_start.py \
    --ckpt_dir checkpoints/crack_openvla/best \
    --image_dir images/

操作:
  - 画像上をクリック → 起点を選択（何度でも変更可）
  - ウィンドウを閉じる → rollout 実行 → 次の画像へ
  - s キー → この画像をスキップ
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


SKIP = object()  # スキップ用センチネル


def pick_with_matplotlib(image_path: str, title_prefix: str = ""):
    import matplotlib
    matplotlib.use("MacOSX")   # Mac ネイティブバックエンド
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    img = mpimg.imread(image_path)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    fname = Path(image_path).name
    ax.set_title(f"{title_prefix}{fname}  ― クリックして起点を選択 / 閉じると確定 / s でスキップ", fontsize=11)
    fig.text(0.5, 0.01, "[s] スキップ  |  クリックで起点変更可", ha="center", fontsize=9, color="gray")

    coords = []
    skipped = [False]

    def on_click(event):
        if event.inaxes != ax:
            return
        x, y = int(event.xdata), int(event.ydata)
        coords.clear()
        coords.append((x, y))
        for c in ax.collections:
            c.remove()
        ax.scatter([x], [y], c="red", s=120, zorder=5)
        ax.set_title(f"{title_prefix}{fname}  ― 起点: ({x}, {y})  / 閉じると確定", fontsize=11)
        fig.canvas.draw()

    def on_key(event):
        if event.key == "s":
            skipped[0] = True
            plt.close()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout()
    plt.show()

    if skipped[0]:
        return SKIP
    if not coords:
        print(f"  [スキップ] 点が選択されませんでした: {fname}")
        return SKIP

    return coords[-1]


def pick_with_opencv(image_path: str) -> tuple[int, int]:
    import cv2

    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    # 表示サイズを縮小（大きすぎる画像はリサイズ）
    scale = min(1.0, 1200 / max(w, h))
    disp = cv2.resize(img, (int(w * scale), int(h * scale)))

    coords = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 実座標に変換
            rx, ry = int(x / scale), int(y / scale)
            coords.clear()
            coords.append((rx, ry))
            tmp = disp.copy()
            cv2.circle(tmp, (x, y), 8, (0, 0, 255), -1)
            cv2.putText(tmp, f"({rx},{ry})", (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("クリックして起点を選択 (Enter で確定)", tmp)

    cv2.imshow("クリックして起点を選択 (Enter で確定)", disp)
    cv2.setMouseCallback("クリックして起点を選択 (Enter で確定)", on_mouse)
    print("画像をクリックして起点を選択してください。Enter で確定します。")

    while True:
        key = cv2.waitKey(20)
        if key == 13 or key == ord("q"):  # Enter or q
            break
    cv2.destroyAllWindows()

    if not coords:
        print("[ERROR] 点が選択されませんでした。")
        sys.exit(1)

    return coords[-1]


def run_rollout(ckpt_dir: str, image_path: str, x: int, y: int,
                max_steps: int, output_dir: str) -> None:
    script = Path(__file__).parent / "rollout.py"
    cmd = [
        sys.executable, str(script),
        "--ckpt_dir",   ckpt_dir,
        "--image",      image_path,
        "--x",          str(x),
        "--y",          str(y),
        "--max_steps",  str(max_steps),
        "--output_dir", output_dir,
    ]
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description="クリックで起点を選択して rollout を実行")
    parser.add_argument("--ckpt_dir",   required=True, help="LoRA チェックポイントディレクトリ")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",     help="単一画像パス")
    group.add_argument("--image_dir", help="画像フォルダ（内の全画像を順番に処理）")
    parser.add_argument("--max_steps",  type=int, default=30, help="最大ステップ数（デフォルト: 30）")
    parser.add_argument("--output_dir", default="results/rollout", help="結果の出力先")
    args = parser.parse_args()

    # 処理対象の画像リストを作成
    if args.image:
        images = [Path(args.image)]
    else:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        images = sorted(p for p in Path(args.image_dir).iterdir() if p.suffix.lower() in exts)
        print(f"{len(images)} 枚の画像が見つかりました: {args.image_dir}\n")

    script = Path(__file__).parent / "rollout.py"

    starts = []

    for i, img_path in enumerate(images):
        print(f"\n[{i+1}/{len(images)}] {img_path.name}")

        title_prefix = f"[{i+1}/{len(images)}] "
        try:
            import matplotlib
            result = pick_with_matplotlib(str(img_path), title_prefix)
        except Exception:
            print("matplotlib が使えないため OpenCV で試みます...")
            result = pick_with_opencv(str(img_path))

        if result is SKIP:
            print(f"  → スキップ")
            continue

        x, y = result
        starts.append({"image": str(img_path), "x": x, "y": y})
        print(f"  起点: ({x}, {y}) を記録しました")

    # 座標を JSON に保存
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    coords_path = out_dir / "starts.json"
    import json
    with open(coords_path, "w", encoding="utf-8") as f:
        json.dump(starts, f, indent=2, ensure_ascii=False)

    print(f"\n起点座標を保存しました: {coords_path}")
    print("H100 で以下を実行してください:\n")
    for s in starts:
        stem = Path(s["image"]).stem
        print(f"  python3 training/rollout.py \\")
        print(f"    --ckpt_dir {args.ckpt_dir} \\")
        print(f"    --image    {s['image']} \\")
        print(f"    --x {s['x']} --y {s['y']} \\")
        print(f"    --output_dir {args.output_dir}/{stem}\n")


if __name__ == "__main__":
    main()
