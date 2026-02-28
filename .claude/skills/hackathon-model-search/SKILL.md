---
name: hackathon-model-search
description: This skill should be used when the user asks to "使えるモデルを教えて", "open model for spatial understanding", "空間理解モデル", "ハッカソンで使えるモデル", "VLM", "point cloud model", "depth estimation model", or needs to find commercially-licensed open models suitable for spatial sensing tasks on H100 or A5000 GPUs.
---

# 空間理解タスク向けオープンモデル提案

ハッカソンのルールに適合する商用利用可・コピーレフトなしのオープンモデルを提案します。

## ハッカソンのモデル制約

- 商用利用可能なオープンモデルのみ
- GPT-4, Claude, Gemini等のクローズドAPIは推論不可
- GPL等コピーレフトライセンスのモデルは不可
- GPU: H100（80GB VRAM）または A5000 x 8（各24GB VRAM）

## 提案するモデルカテゴリ

### 空間理解・3D系
- Depth Anything V2（Apache 2.0）
- PointBERT系（MIT）

### VLM（Visual Language Model）
- Qwen2.5-VL（Apache 2.0）
- InternVL2（MIT）
- Phi-3 Vision（MIT）

### LLM（ファインチューニングベース）
- Qwen2.5（Apache 2.0）
- Llama 3.x（Meta Llama License）
- Mistral（Apache 2.0）

## 提案形式

| モデル名 | 用途 | ライセンス | 必要VRAM | HuggingFace |
|---|---|---|---|---|

各モデルについて概要、ライセンス確認、GPU要件、空間理解タスクへの適用方法、ファインチューニングアプローチを示す。
