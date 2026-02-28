---
language:
- ja
license: apache-2.0
tags:
- solar-panel
- factory
- spatial-understanding
- simulation
- synthetic
task_categories:
- image-text-to-text
- visual-question-answering
size_categories:
- n<1K
---

# ExaSense Solar Panel Dataset

工場屋根への太陽光パネル設置提案を行う VLM のファインチューニング用データセットです。

## Dataset Details

### 概要

3Dメッシュの日照シミュレーション結果（ヒートマップ画像）と、それに対応する設置提案テキストのペアで構成されます。

### データ形式

```json
{
  "image": "heatmap_001.png",
  "conversations": [
    {
      "role": "user",
      "content": "この工場屋根の日照分析結果から、太陽光パネルの最適設置位置を提案してください。"
    },
    {
      "role": "assistant",
      "content": "分析結果に基づき、以下の設置提案を行います..."
    }
  ],
  "metadata": {
    "building_type": "factory",
    "location": "osaka",
    "annual_irradiance_kwh_m2": 1520,
    "shadow_ratio": 0.15
  }
}
```

### データセット特性

| 項目 | 値 |
|------|-----|
| サンプル数 | <!-- TODO: 確定後に記載 --> |
| 画像解像度 | 1024x1024 |
| 画像形式 | PNG |
| テキスト言語 | 日本語 |
| 建物タイプ | 工場、倉庫、商業施設 |
| 対象地域 | 関西地方（大阪、京都、神戸） |

### 生成パイプライン

```
Phase 1-2: VGGT-1B-Commercial → 3D点群 → Open3D メッシュ
    ↓
Phase 3: pvlib + trimesh → 日照シミュレーション → ヒートマップ画像
    ↓
アノテーション: 専門家による設置提案テキスト作成
    ↓
品質チェック → 最終データセット
```

### 元データ

- **3D再構築入力**: ドローン撮影画像（自社撮影）
- **検証用サンプル**: [Mip-NeRF 360 Dataset](https://jonbarron.info/mipnerf360/) (CC BY 4.0, Google)
- **日射データ**: pvlib によるクリアスカイモデル計算値

### 合成に使用したモデル

| モデル | 用途 | ライセンス |
|--------|------|-----------|
| VGGT-1B-Commercial (Meta) | 3D点群生成 | VGGT Acceptable Use Policy |
| Open3D | メッシュ再構成 | MIT |
| pvlib | 日射量計算 | BSD-3-Clause |

## Intended Use

- VLM のファインチューニング（太陽光パネル設置提案タスク）
- 工場エネルギー最適化に関する研究

## Limitations

- 関西地方の気象条件・建物タイプに偏りがあります
- クリアスカイモデルに基づく日射量のため、実際の気象変動は反映されていません
- 構造強度に関する情報は含まれていません

## License

Apache 2.0

## Citation

```bibtex
@misc{exasense_dataset2026,
  title={ExaSense Solar Panel Dataset},
  author={ExaSense Team},
  year={2026},
  howpublished={Kanden AI Hackathon 2026},
}
```
