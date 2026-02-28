---
name: hackathon-slurm
description: This skill should be used when the user asks to "Slurmスクリプト", "sbatch", "GPU学習ジョブ", "H100でファインチューニング", "A5000で学習", "ジョブスクリプト作成", "torchrun", "deepspeed", or needs to create a Slurm job script for GPU-based AI training or inference.
---

# Slurmジョブスクリプト生成

H100またはA5000 GPU環境でのAI学習・推論ジョブ向けSlurmスクリプトを生成します。

## GPU環境

- **H100**: 2チームが専有（各80GB VRAM）
- **A5000**: 8チームが共用（A5000 x 8基、各24GB VRAM）

## 基本テンプレート

```bash
#!/bin/bash
#SBATCH --job-name=spatial-ai
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
```

### マルチGPU（torchrun）
```bash
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=4
torchrun --nproc_per_node=4 train.py
```

### A5000共用時の注意
- 他チームとの共有を考慮しGPUを占有しすぎない
- ストレージ割り当て1.5〜1.8TB

### 推奨設定
- チェックポイント保存: 定期的に（途中再開可能に）
- ログ: `logs/` ディレクトリに出力
- DeepSpeed使用時: `ds_config.json` を別途用意

ユーザーの要件に応じて完全なスクリプトを生成し、各設定の意味も説明する。
