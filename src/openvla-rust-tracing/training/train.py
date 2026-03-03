"""
タスク3: OpenVLA 7B LoRA ファインチューニングスクリプト

H100 での実行を前提とした bf16 LoRA ファインチューニング。

特徴:
  - 直近 3〜5 フレームの画像履歴を instruction に埋め込んでバックトラック判断を補助
  - 探索済みエリアのミニマップをパッチ画像にオーバーレイするオプション
  - WandB でのロギング
  - RLDS 形式の TFRecord を直接読み込み

使用方法:
  torchrun --nproc_per_node=1 train.py \
    --data_dir data/rust_rlds \
    --output_dir checkpoints/rust_openvla \
    --model_name_or_path openvla/openvla-7b \
    --lora_rank 16 \
    --bf16 \
    --wandb_project rust_openvla

依存:
  pip install transformers peft accelerate wandb trl
  (TFRecord 読み込みのため tensorflow も必要)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# 遅延インポート: GPU 環境でのみ利用可能
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from peft import LoraConfig, get_peft_model, TaskType


# ─── 定数 ─────────────────────────────────────────────────────────────────
IMAGE_SIZE = 224
ACTION_DIM = 7          # OpenVLA の出力次元
RUST_ACTION_DIM = 3     # 実際に使用する次元数

# OpenVLA のアクション正規化範囲 ([-1, 1] を想定)
ACTION_NORM_MIN = -1.0
ACTION_NORM_MAX = 1.0

MINIMAP_SIZE = 56       # ミニマップのサイズ (パッチ左上に埋め込む)
MINIMAP_ALPHA = 0.6     # オーバーレイの透明度


# ─── ミニマップ生成 ─────────────────────────────────────────────────────
def build_minimap(
    visited: set[tuple[int, int]],
    rust_patches: set[tuple[int, int]],
    current_pos: tuple[int, int],
    grid_rows: int,
    grid_cols: int,
    minimap_size: int = MINIMAP_SIZE,
) -> np.ndarray:
    """
    探索済みエリアのミニマップを生成する。

    Color coding:
      黒:        未探索
      緑 (dim):  サビなし探索済み
      赤:        サビあり探索済み
      青:        現在位置
    """
    minimap = np.zeros((grid_rows, grid_cols, 3), dtype=np.uint8)

    for r, c in visited:
        if (r, c) in rust_patches:
            minimap[r, c] = [0, 0, 180]    # 赤 (BGR)
        else:
            minimap[r, c] = [0, 100, 0]    # 暗緑 (BGR)

    cr, cc = current_pos
    if 0 <= cr < grid_rows and 0 <= cc < grid_cols:
        minimap[cr, cc] = [200, 0, 0]       # 青 (BGR)

    # リサイズ (nearest interpolation で格子感を維持)
    minimap_resized = torch.nn.functional.interpolate(
        torch.from_numpy(minimap).permute(2, 0, 1).unsqueeze(0).float(),
        size=(minimap_size, minimap_size),
        mode="nearest",
    ).squeeze(0).permute(1, 2, 0).byte().numpy()

    return minimap_resized


def overlay_minimap(patch: np.ndarray, minimap: np.ndarray, alpha: float = MINIMAP_ALPHA) -> np.ndarray:
    """ミニマップをパッチ画像の左上にオーバーレイする。"""
    result = patch.copy()
    h, w = minimap.shape[:2]
    roi = result[:h, :w]
    blended = (roi.astype(float) * (1 - alpha) + minimap.astype(float) * alpha).astype(np.uint8)
    result[:h, :w] = blended
    return result


# ─── RLDS Dataset ────────────────────────────────────────────────────────
class RLDSRustDataset(Dataset):
    """
    TFRecord から読み込む PyTorch Dataset。

    history_len フレームの画像を結合して入力とする。
    """

    def __init__(
        self,
        tfrecord_path: str,
        history_len: int = 3,
        use_minimap: bool = False,
        processor=None,
    ) -> None:
        super().__init__()
        self.history_len = history_len
        self.use_minimap = use_minimap
        self.processor = processor

        # TFRecord の全レコードをメモリに展開
        # 大規模データセットでは streaming が望ましいが、
        # ハッカソン規模では pre-load で十分
        self._records = self._load_tfrecord(tfrecord_path)
        print(f"[RLDSRustDataset] {len(self._records)} ステップを読み込みました。")

    def _load_tfrecord(self, path: str) -> list[dict]:
        """TFRecord を辞書リストに展開する。"""
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow が必要です: pip install tensorflow")

        FEATURE_DESC = {
            "steps/observation/image": tf.io.FixedLenFeature([], tf.string),
            "steps/action": tf.io.FixedLenFeature([ACTION_DIM], tf.float32),
            "steps/language_instruction": tf.io.FixedLenFeature([], tf.string),
            "steps/is_first": tf.io.FixedLenFeature([], tf.int64),
            "steps/is_last": tf.io.FixedLenFeature([], tf.int64),
        }

        dataset = tf.data.TFRecordDataset(path, compression_type="GZIP")
        records = []
        for raw in dataset:
            ex = tf.io.parse_single_example(raw, FEATURE_DESC)
            img_bytes = ex["steps/observation/image"].numpy()
            img_arr = tf.image.decode_jpeg(img_bytes, channels=3).numpy()
            records.append({
                "image": img_arr,  # (224, 224, 3) uint8 RGB
                "action": ex["steps/action"].numpy(),
                "instruction": ex["steps/language_instruction"].numpy().decode("utf-8"),
                "is_first": bool(ex["steps/is_first"].numpy()),
                "is_last": bool(ex["steps/is_last"].numpy()),
            })
        return records

    def __len__(self) -> int:
        return len(self._records)

    def _build_history_instruction(self, idx: int) -> str:
        """
        直近 history_len フレームの action 情報を instruction に付加する。
        バックトラック判断の補助として機能する。
        """
        base_instruction = self._records[idx]["instruction"]
        history_parts = []
        for h in range(self.history_len - 1, 0, -1):
            prev_idx = idx - h
            if prev_idx < 0:
                continue
            prev = self._records[prev_idx]
            action = prev["action"]
            z_val = action[2]
            x_val, y_val = action[0], action[1]
            if z_val > 0.5:
                direction = "backtrack"
            elif abs(x_val) < 0.1 and y_val > 0:
                direction = "up"
            elif abs(x_val) < 0.1 and y_val < 0:
                direction = "down"
            elif x_val > 0 and abs(y_val) < 0.1:
                direction = "right"
            elif x_val < 0 and abs(y_val) < 0.1:
                direction = "left"
            elif x_val > 0 and y_val > 0:
                direction = "upper_right"
            elif x_val < 0 and y_val > 0:
                direction = "upper_left"
            elif x_val > 0 and y_val < 0:
                direction = "lower_right"
            else:
                direction = "lower_left"
            history_parts.append(f"t-{h}: {direction}")

        if history_parts:
            history_str = "; ".join(history_parts)
            return f"[History: {history_str}] {base_instruction}"
        return base_instruction

    def __getitem__(self, idx: int) -> dict:
        record = self._records[idx]
        image = record["image"]  # (224, 224, 3) RGB uint8

        # ミニマップオーバーレイ (オプション)
        if self.use_minimap:
            # 訓練時は疑似的なミニマップをランダム生成
            minimap = np.zeros((MINIMAP_SIZE, MINIMAP_SIZE, 3), dtype=np.uint8)
            minimap[:, :, 0] = np.random.randint(0, 100, (MINIMAP_SIZE, MINIMAP_SIZE), dtype=np.uint8)
            image_bgr = image[:, :, ::-1].copy()  # RGB → BGR
            image_bgr = overlay_minimap(image_bgr, minimap)
            image = image_bgr[:, :, ::-1]  # BGR → RGB

        # 履歴付き instruction
        instruction = self._build_history_instruction(idx)

        # アクションをそのまま使用 (7D float)
        action = record["action"].astype(np.float32)

        if self.processor is not None:
            # OpenVLA の processor で前処理
            inputs = self.processor(
                text=instruction,
                images=image,
                return_tensors="pt",
            )
            return {
                "input_ids": inputs["input_ids"].squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0),
                "pixel_values": inputs["pixel_values"].squeeze(0),
                "labels": torch.tensor(action, dtype=torch.float32),
                "is_first": record["is_first"],
                "is_last": record["is_last"],
            }

        # processor なしの場合 (デバッグ用)
        return {
            "image": torch.from_numpy(image).permute(2, 0, 1).float() / 255.0,
            "instruction": instruction,
            "action": torch.tensor(action, dtype=torch.float32),
        }


# ─── コールバック・ログ ───────────────────────────────────────────────────
class WandBLogger:
    """WandB ロガーのラッパー。"""

    def __init__(self, project: str, name: str, config: dict) -> None:
        try:
            import wandb
            self.run = wandb.init(project=project, name=name, config=config)
            self.enabled = True
        except ImportError:
            print("[WARNING] WandB が見つかりません。ロギングをスキップします。")
            self.enabled = False
            self.run = None

    def log(self, metrics: dict, step: int) -> None:
        if self.enabled and self.run:
            import wandb
            wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self.enabled and self.run:
            self.run.finish()


# ─── LoRA 設定 ────────────────────────────────────────────────────────────
def get_lora_config(rank: int = 16, alpha: int = 32, dropout: float = 0.05):
    """OpenVLA 用 LoRA 設定を返す。"""
    from peft import LoraConfig, TaskType
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        # OpenVLA (LLaMA ベース) のターゲットモジュール
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )


# ─── 訓練ループ ───────────────────────────────────────────────────────────
def train(args: argparse.Namespace) -> None:
    """メイン訓練ループ。"""
    import torch
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from peft import get_peft_model
    from accelerate import Accelerator

    # Accelerate 初期化
    accelerator = Accelerator(
        mixed_precision="bf16" if args.bf16 else "no",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    # WandB
    wandb_logger = WandBLogger(
        project=args.wandb_project,
        name=args.run_name or f"rust_lora_r{args.lora_rank}",
        config=vars(args),
    ) if accelerator.is_main_process else None

    # モデルとプロセッサの読み込み
    accelerator.print(f"[train] モデルを読み込み中: {args.model_name_or_path}")
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=True,
    )

    # LoRA 適用
    lora_config = get_lora_config(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # データセット
    train_dataset = RLDSRustDataset(
        tfrecord_path=str(Path(args.data_dir) / "train.tfrecord.gz"),
        history_len=args.history_len,
        use_minimap=args.use_minimap,
        processor=processor,
    )
    val_dataset = RLDSRustDataset(
        tfrecord_path=str(Path(args.data_dir) / "val.tfrecord.gz"),
        history_len=args.history_len,
        use_minimap=args.use_minimap,
        processor=processor,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # オプティマイザ
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # スケジューラ (cosine warmup)
    total_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Accelerate でラップ
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # 訓練ループ
    global_step = 0
    best_val_loss = float("inf")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.num_epochs):
        model.train()
        train_loss_accum = 0.0

        for batch_idx, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                pixel_values = batch["pixel_values"]
                labels_action = batch["labels"]  # (B, 7)

                # OpenVLA の forward (アクション予測)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=input_ids,  # language modeling loss
                )
                loss = outputs.loss
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_loss_accum += loss.item()
            global_step += 1

            if global_step % args.logging_steps == 0 and accelerator.is_main_process:
                avg_loss = train_loss_accum / args.logging_steps
                lr = scheduler.get_last_lr()[0]
                accelerator.print(
                    f"Epoch {epoch+1}/{args.num_epochs} | "
                    f"Step {global_step} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LR: {lr:.2e}"
                )
                if wandb_logger:
                    wandb_logger.log({
                        "train/loss": avg_loss,
                        "train/lr": lr,
                        "train/epoch": epoch + batch_idx / len(train_loader),
                    }, step=global_step)
                train_loss_accum = 0.0

        # Validation
        if (epoch + 1) % args.eval_epochs == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        pixel_values=batch["pixel_values"],
                        labels=batch["input_ids"],
                    )
                    val_losses.append(outputs.loss.item())

            val_loss = sum(val_losses) / len(val_losses)
            accelerator.print(f"[Epoch {epoch+1}] Val Loss: {val_loss:.4f}")

            if wandb_logger:
                wandb_logger.log({"val/loss": val_loss}, step=global_step)

            # ベストモデルを保存
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.save_pretrained(str(output_dir / "best"))
                processor.save_pretrained(str(output_dir / "best"))
                accelerator.print(f"✓ Best model saved (val_loss={val_loss:.4f})")

        # 定期チェックポイント
        if (epoch + 1) % args.save_epochs == 0:
            unwrapped = accelerator.unwrap_model(model)
            ckpt_dir = output_dir / f"checkpoint-epoch{epoch+1}"
            unwrapped.save_pretrained(str(ckpt_dir))
            accelerator.print(f"Checkpoint saved: {ckpt_dir}")

    # 最終モデルを保存
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(str(output_dir / "final"))
    processor.save_pretrained(str(output_dir / "final"))

    # 訓練設定を保存
    with open(output_dir / "training_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    if wandb_logger:
        wandb_logger.finish()

    accelerator.print(f"[train] 訓練完了! → {output_dir}")


# ─── メイン ──────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="OpenVLA LoRA ファインチューニング")

    # モデル設定
    parser.add_argument("--model_name_or_path", type=str, default="openvla/openvla-7b",
                        help="ベースモデル (HuggingFace Hub ID or ローカルパス)")
    parser.add_argument("--output_dir", type=str, default="checkpoints/rust_openvla",
                        help="チェックポイント保存先")

    # データ設定
    parser.add_argument("--data_dir", type=str, default="data/rust_rlds",
                        help="TFRecord データディレクトリ")
    parser.add_argument("--history_len", type=int, default=3,
                        help="画像履歴フレーム数 (3〜5)")
    parser.add_argument("--use_minimap", action="store_true",
                        help="ミニマップオーバーレイを有効化")

    # LoRA 設定
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # 訓練ハイパーパラメータ
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="bfloat16 で訓練 (H100 推奨)")

    # ログ・保存設定
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_epochs", type=int, default=1)
    parser.add_argument("--save_epochs", type=int, default=2)
    parser.add_argument("--wandb_project", type=str, default="rust_openvla",
                        help="WandB プロジェクト名")
    parser.add_argument("--run_name", type=str, default=None)

    args = parser.parse_args()

    # torch が利用可能かチェック
    try:
        import torch
        if not torch.cuda.is_available():
            print("[WARNING] CUDA が利用できません。CPU では非常に低速です。")
    except ImportError:
        print("[ERROR] PyTorch が必要です。")
        return

    train(args)


if __name__ == "__main__":
    main()
