"""
VLA 学習スクリプト (train.py) ― OpenVLA 7B LoRA ファインチューニング

【実行環境】H100 専用 (bf16, ~14GB VRAM 必要)

データ形式: collect.py が出力した JSON + JPG
  dataset/
    episodes/episode_XXXXX.json
    steps/step_XXXXXX.jpg

アクション形式 (7次元, OpenVLA 互換):
  [vx_body, vy_body, vz_body, yaw_rate, 0, 0, 0]
  ※ collect.py の 4D アクションをゼロ埋めで 7D に拡張

使い方 (H100 上で実行):
  pip install transformers peft accelerate tensorboard

  torchrun --nproc_per_node=1 scripts/train.py \\
    --data dataset/ \\
    --out checkpoints/drone_openvla \\
    --epochs 5 \\
    --lora_rank 32 \\
    --bf16
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from action_tokenizer import ActionTokenizer

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from peft import LoraConfig, get_peft_model


# ─── データセット ──────────────────────────────────────────────────────────
class DroneNavDataset(Dataset):
    """collect.py が出力した JSON + JPG を OpenVLA 用に読み込むデータセット"""

    def __init__(self, data_dir: str, processor=None, action_tokenizer: ActionTokenizer | None = None) -> None:
        self.data_dir  = Path(data_dir)
        self.processor = processor
        self.samples: list[dict] = []

        episode_dir = self.data_dir / "episodes"
        for ep_path in sorted(episode_dir.glob("episode_*.json")):
            with open(ep_path, encoding="utf-8") as f:
                ep = json.load(f)
            for step in ep.get("steps", []):
                self.samples.append({
                    "image_path":  self.data_dir / step["image_path"],
                    "instruction": step["instruction"],
                    "action_4d":   step["action_vector"][:4],
                })

        n_ep = len(list(episode_dir.glob("episode_*.json")))
        print(f"データセット: {len(self.samples)} ステップ ({n_ep} エピソード)")

        # ActionTokenizer: 渡されなければデータセットから統計を計算
        if action_tokenizer is not None:
            self.action_tokenizer = action_tokenizer
        else:
            all_actions = [s["action_4d"] for s in self.samples]
            self.action_tokenizer = ActionTokenizer.from_dataset(all_actions)
            print(f"ActionTokenizer: {self.action_tokenizer}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # 画像読み込み
        image = PILImage.open(sample["image_path"]).convert("RGB")

        # アクション: 4D → 256bin 離散トークン文字列 (例: "145 109 182 127")
        action_4d = list(sample["action_4d"])
        action_str = self.action_tokenizer.encode(action_4d)

        instruction = sample["instruction"]

        if self.processor is not None:
            # 画像 + テキスト全体をトークナイズ
            inputs_full = self.processor(
                text=instruction + " " + action_str,
                images=image,
                return_tensors="pt",
            )

            # action トークン長を計算（instruction 部分は loss 計算対象外）
            action_token_len = self.processor.tokenizer(
                action_str, return_tensors="pt"
            )["input_ids"].shape[1]

            # labels: instruction + 画像 部分を -100 にマスク
            labels = inputs_full["input_ids"].clone()
            labels[:, :-action_token_len] = -100

            return {
                "input_ids":      inputs_full["input_ids"].squeeze(0),
                "attention_mask": inputs_full["attention_mask"].squeeze(0),
                "pixel_values":   inputs_full["pixel_values"].squeeze(0),
                "labels":         labels.squeeze(0),
            }

        # processor なしの場合（デバッグ用）
        return {
            "instruction": instruction,
            "action_str":  action_str,
        }


# ─── LoRA 設定 ────────────────────────────────────────────────────────────
def get_lora_config(rank: int = 8, alpha: int = 32, dropout: float = 0.05):
    from peft import LoraConfig, TaskType
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )


# ─── TensorBoard ロガー ───────────────────────────────────────────────────
class TensorBoardLogger:
    def __init__(self, log_dir: str) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=log_dir)
            self.enabled = True
            print(f"[TensorBoard] ログ保存先: {log_dir}")
        except ImportError:
            print("[WARNING] tensorboard が見つかりません。pip install tensorboard")
            self.enabled = False

    def log(self, metrics: dict, step: int) -> None:
        if self.enabled:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)

    def finish(self) -> None:
        if self.enabled:
            self.writer.close()


# ─── 学習 ────────────────────────────────────────────────────────────────
def train(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from peft import get_peft_model
    from accelerate import Accelerator

    accelerator = Accelerator(
        mixed_precision="bf16" if args.bf16 else "no",
        gradient_accumulation_steps=args.grad_accum,
    )

    tb_logger = TensorBoardLogger(
        log_dir=str(Path(args.out) / "tensorboard"),
    ) if accelerator.is_main_process else None

    # モデル・プロセッサ読み込み
    accelerator.print(f"[train] モデルを読み込み中: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    # LoRA 適用
    lora_config = get_lora_config(rank=args.lora_rank)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # データセット (ActionTokenizer をデータセットから自動計算)
    train_dataset = DroneNavDataset(args.data, processor=processor)

    # train/val 分割
    n_total = len(train_dataset)
    n_train = max(1, int(n_total * 0.9))
    indices = list(range(n_total))
    import random; random.shuffle(indices)
    train_indices = indices[:n_train]
    val_indices   = indices[n_train:]

    from torch.utils.data import Subset
    train_set = Subset(train_dataset, train_indices)
    val_set   = Subset(train_dataset, val_indices)
    accelerator.print(f"train: {len(train_set)}, val: {len(val_set)}")

    def collate_fn(batch):
        from torch.nn.utils.rnn import pad_sequence
        return {
            "input_ids": pad_sequence(
                [b["input_ids"] for b in batch], batch_first=True,
                padding_value=processor.tokenizer.pad_token_id,
            ),
            "attention_mask": pad_sequence(
                [b["attention_mask"] for b in batch], batch_first=True, padding_value=0,
            ),
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "labels": pad_sequence(
                [b["labels"] for b in batch], batch_first=True, padding_value=-100,
            ),
        }

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        optimizer, train_loader, val_loader, scheduler
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_val_loss = float("inf")
    train_start = time.time()
    steps_per_epoch = len(train_loader)
    accelerator.print(f"\n学習開始: {args.epochs} エポック | {steps_per_epoch} ステップ/エポック\n")

    for epoch in range(args.epochs):
        model.train()
        train_loss_accum = 0.0
        epoch_start = time.time()

        for step_in_epoch, batch in enumerate(train_loader, 1):
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"].to(torch.bfloat16 if args.bf16 else torch.float32),
                    labels=batch["labels"],
                )
                loss = outputs.loss
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_loss_accum += loss.item()
            global_step += 1

            if global_step % 10 == 0 and accelerator.is_main_process:
                avg = train_loss_accum / 10
                lr  = scheduler.get_last_lr()[0]
                # 残り時間推定
                elapsed = time.time() - train_start
                steps_done = global_step
                steps_total = steps_per_epoch * args.epochs
                eta_sec = elapsed / steps_done * (steps_total - steps_done)
                eta_h, eta_rem = divmod(int(eta_sec), 3600)
                eta_m = eta_rem // 60
                accelerator.print(
                    f"Epoch {epoch+1}/{args.epochs} | "
                    f"Step {step_in_epoch}/{steps_per_epoch} | "
                    f"Loss: {avg:.4f} | LR: {lr:.2e} | "
                    f"ETA: {eta_h}h {eta_m}m"
                )
                if tb_logger:
                    tb_logger.log({"train/loss": avg, "train/lr": lr}, step=global_step)
                train_loss_accum = 0.0

        # バリデーション
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"].to(torch.bfloat16 if args.bf16 else torch.float32),
                    labels=batch["labels"],
                )
                val_losses.append(outputs.loss.item())

        val_loss = sum(val_losses) / len(val_losses) if val_losses else float("inf")
        epoch_time = time.time() - epoch_start
        epoch_h, epoch_rem = divmod(int(epoch_time), 3600)
        epoch_m, epoch_s = divmod(epoch_rem, 60)
        remaining_epochs = args.epochs - (epoch + 1)
        total_eta_sec = epoch_time * remaining_epochs
        total_eta_h, total_eta_rem = divmod(int(total_eta_sec), 3600)
        total_eta_m = total_eta_rem // 60
        accelerator.print(
            f"\n{'='*60}\n"
            f"[Epoch {epoch+1}/{args.epochs}] "
            f"Val Loss: {val_loss:.4f} | Train Time: {epoch_h}h {epoch_m}m {epoch_s}s\n"
            f"残り推定: {total_eta_h}h {total_eta_m}m ({remaining_epochs} エポック)\n"
            f"{'='*60}\n"
        )
        if tb_logger:
            tb_logger.log({"val/loss": val_loss, "epoch_time_min": epoch_time / 60}, step=global_step)

        # エポック毎チェックポイント保存
        if accelerator.is_main_process:
            epoch_ckpt = out_dir / f"epoch_{epoch+1:04d}"
            unwrapped_ep = accelerator.unwrap_model(model)
            unwrapped_ep.save_pretrained(str(epoch_ckpt))
            processor.save_pretrained(str(epoch_ckpt))
            train_dataset.action_tokenizer.save(str(epoch_ckpt / "action_stats.npz"))
            accelerator.print(f"→ Epoch checkpoint saved: {epoch_ckpt}")

        # ベストモデル保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(str(out_dir / "best"))
            processor.save_pretrained(str(out_dir / "best"))
            train_dataset.action_tokenizer.save(str(out_dir / "best" / "action_stats.npz"))
            accelerator.print(f"✓ Best model saved (val_loss={val_loss:.4f})")

    # 最終モデル保存
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(str(out_dir / "final"))
    processor.save_pretrained(str(out_dir / "final"))
    train_dataset.action_tokenizer.save(str(out_dir / "final" / "action_stats.npz"))

    if tb_logger:
        tb_logger.finish()

    accelerator.print(f"\n学習完了! → {out_dir}/best/")


# ─── メイン ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenVLA 7B LoRA ファインチューニング（ドローンナビゲーション）")
    parser.add_argument("--data",          type=str,   default="dataset",              help="データセットディレクトリ")
    parser.add_argument("--out",           type=str,   default="checkpoints/drone_openvla", help="モデル保存先")
    parser.add_argument("--model",         type=str,   default="openvla/openvla-7b",   help="ベースモデル")
    parser.add_argument("--epochs",        type=int,   default=5)
    parser.add_argument("--lora_rank",     type=int,   default=32)
    parser.add_argument("--batch_size",    type=int,   default=16)
    parser.add_argument("--grad_accum",    type=int,   default=1,                  help="勾配累積ステップ数")
    parser.add_argument("--lr",            type=float, default=5e-4)
    parser.add_argument("--bf16",          action="store_true", default=False,         help="bfloat16 で訓練（H100推奨）")
    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        print("[ERROR] PyTorch が必要です。")
        exit(1)
    if not torch.cuda.is_available():
        print("[WARNING] CUDA が利用できません。H100 上で実行してください。")

    train(args)
