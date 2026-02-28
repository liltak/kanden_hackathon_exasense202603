---
language:
- ja
- en
license: apache-2.0
base_model: Qwen/Qwen2.5-VL-7B-Instruct
tags:
- solar-panel
- factory
- spatial-understanding
- vlm
- lora
- unsloth
pipeline_tag: image-text-to-text
---

# ExaSense Solar VLM — Qwen2.5-VL-7B LoRA

工場屋根の3Dモデル・日照シミュレーション結果から太陽光パネルの最適設置位置を提案する VLM (Vision-Language Model) の LoRA アダプタです。

## Model Details

### Base Model

- **モデル名**: [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- **ライセンス**: Apache 2.0
- **パラメータ数**: 7B

### Fine-tuning

- **手法**: LoRA (Low-Rank Adaptation) via [Unsloth](https://github.com/unslothai/unsloth)
- **LoRA rank**: 16
- **LoRA alpha**: 32
- **Target modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Learning rate**: 2e-4
- **Epochs**: 3
- **Batch size**: 4 (gradient accumulation: 4)
- **Precision**: bfloat16
- **GPU**: NVIDIA H100 80GB

> **Note**: 上記パラメータはファインチューニング実施後に確定値で更新予定です。

## Intended Use

### 入力

- 工場屋根の3Dレンダリング画像（メッシュの日照ヒートマップ）
- シミュレーション結果サマリ（年間日射量、影の割合等）

### 出力

- 太陽光パネル設置に関する分析レポート（日本語）
  - 設置推奨エリアの説明
  - 影の影響分析
  - ROI 予測の根拠説明

### 対象ユーザー

- 工場・施設のエネルギー管理者
- 太陽光パネル設置業者
- エネルギーコンサルタント

## Training Data

- **データセット**: [ExaSense Solar Panel Dataset](<!-- TODO: HuggingFace Dataset URL -->)
- **データ形式**: 画像 + テキスト (instruction-response ペア)
- **サンプル数**: <!-- TODO: 確定後に記載 -->
- **生成方法**: Phase 3 シミュレーション結果 + 専門家アノテーション

## Evaluation

<!-- TODO: ファインチューニング後に評価結果を追記 -->

| 指標 | ベースモデル | ファインチューニング後 |
|------|-------------|---------------------|
| 設置提案の適切性 | TBD | TBD |
| ROI説明の正確性 | TBD | TBD |

## How to Use

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

# ベースモデルのロード
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype="bfloat16",
    device_map="auto",
)

# LoRA アダプタの適用
model = PeftModel.from_pretrained(base_model, "exasense/solar-vlm-lora")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# 推論
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "factory_roof_heatmap.png"},
            {"type": "text", "text": "この工場屋根の日照分析結果から、太陽光パネルの最適設置位置を提案してください。"},
        ],
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, images=[image], return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=1024)
print(processor.decode(output[0], skip_special_tokens=True))
```

## Limitations

- 日本の工場建物に特化しており、他地域・建物タイプでは精度が低下する可能性があります
- 入力画像は Phase 3 シミュレーションで生成されたヒートマップ形式を想定しています
- 構造的な強度評価は含まれません（屋根の耐荷重は別途確認が必要）

## License

Apache 2.0

## Citation

```bibtex
@misc{exasense2026,
  title={ExaSense: Factory Energy Optimization via 3D Reconstruction and Solar Simulation},
  author={ExaSense Team},
  year={2026},
  howpublished={Kanden AI Hackathon 2026},
}
```
