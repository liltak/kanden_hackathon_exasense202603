# ExaSense — 工場向けエネルギー最適化ソリューション

ドローン撮影画像から工場の3Dモデルを自動生成し、太陽光パネルの最適設置位置をシミュレーション・提案するエンドツーエンドのパイプライン。

**[静的デモを見る (GitHub Pages)](https://liltak.github.io/kanden_hackathon_exasense/)**

![Architecture](generated-diagrams/exasense_architecture.png)

## 主な成果

| 指標 | 値 |
|------|-----|
| 設置可能容量 | **823 kW** |
| 年間発電量 | **1,252 MWh** |
| 年間コスト削減 | **¥3,756万** |
| 投資回収期間 | **5.5年** |
| 25年NPV | **¥5.7億** |

*デモ工場コンプレックス (4棟, 大阪 34.69°N) での試算結果*

## 実行環境

| 項目 | バージョン |
|------|-----------|
| OS | Ubuntu 22.04 LTS / macOS 14+ |
| Python | 3.10 以上 (3.12 推奨) |
| CUDA | 12.1+ (GPU実行時) |
| GPU | NVIDIA H100 80GB 推奨 (Phase 1-2, 4) / T4 16GB 以上 |
| パッケージ管理 | [uv](https://docs.astral.sh/uv/) |

> CPU のみの環境でも Phase 3 (シミュレーション) と Phase 5 (WebUI) は動作します。

## アーキテクチャ

```
Phase 1-2 (GPU)          Phase 3 (CPU)              Phase 4 (GPU)      Phase 5
ドローン画像             日照シミュレーション         AI分析             WebUI
  ↓                        ↓                          ↓                 ↓
VGGT-1B-Commercial → メッシュ処理 → pvlib + trimesh → ROI算出 → Qwen3.5-VL → Gradio Dashboard
  1.7M点    20K面           8,760時点/年                設置提案        5タブUI
```

### Phase 1-2: 3D再構築 + メッシュ処理

- **VGGT-1B-Commercial** (Meta) で画像から3D点群を生成
- **Open3D** による品質改善パイプライン:

```
1.7M点 → Voxel Downsample → 2段階外れ値除去 → Poisson(depth=7) → Cleanup → Taubin平滑 → Decimation → 20K面
```

- 旧パイプライン比 **28.5倍** の面数削減、ファイルサイズ 22MB → 0.8MB

### Phase 3: 日照シミュレーション

- **pvlib** で太陽位置・クリアスカイ日射量を年間8,760時点で計算
- **trimesh** でレイキャスティングによる影行列を算出
- 面ごとの年間日射量 (直達 + 散乱) → ROI / NPV / 回収期間を自動計算

### Phase 4: VLM AI分析

- **Qwen3.5-VL** によるマルチモーダル設置提案
- **Unsloth** でLoRAファインチューニング対応

### Phase 5: WebUI

- **Gradio** 5タブダッシュボード (Dashboard / 3Dビュー / シミュレーション / AI分析 / レポート)
- **FastAPI** REST APIバックエンド

## クイックスタート

```bash
# 依存インストール
uv sync

# テスト実行
uv run pytest tests/ -v

# シミュレーション実行 (CLI)
uv run python -m src.simulation.runner

# WebUI起動 (localhost:7860)
uv run python -m src.ui.app

# 静的サイト生成
uv run python scripts/generate_static_site.py
```

### GPU環境 (H100) セットアップ

```bash
# H100サーバーでのフルセットアップ
scripts/setup_h100.sh

# E2Eパイプライン実行
scripts/verification/e2e_garden_test.sh
```

## プロジェクト構造

```
exasense/
├── src/
│   ├── reconstruction/     # Phase 1-2: VGGT, COLMAP, OpenSplat, メッシュ処理
│   ├── simulation/         # Phase 3: 太陽位置, レイキャスト, 日射量, ROI, 可視化
│   ├── vlm/                # Phase 4: Qwen3.5-VL, ファインチューニング
│   ├── ui/app.py           # Phase 5: Gradio 5タブダッシュボード
│   └── api/server.py       # Phase 5: FastAPI
├── configs/
│   └── solar_params.yaml   # シミュレーションパラメータ
├── tests/                  # 18テスト (pytest)
├── docs/
│   └── index.html          # 静的デモページ
├── scripts/
│   ├── generate_static_site.py
│   ├── setup_h100.sh       # GPU環境セットアップ
│   └── verification/       # E2E検証スクリプト
├── docker/                 # H100 Docker環境
└── data/
    └── e2e_results/        # E2Eテスト結果 (Mip-NeRF 360 Garden)
```

## 使用モデル

| モデル | 用途 | ライセンス | 商用利用 | リンク |
|--------|------|-----------|---------|--------|
| VGGT-1B-Commercial (Meta) | 3D再構築 (Phase 1-2) | VGGT Acceptable Use Policy | OK | [HuggingFace](https://huggingface.co/facebook/VGGT-1B-Commercial) |
| Qwen3.5-VL (Alibaba) | VLM分析 (Phase 4) | Apache 2.0 | OK | [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |

## 使用データセット

| データセット | 用途 | ライセンス | リンク |
|-------------|------|-----------|--------|
| Mip-NeRF 360 (Google) | E2E検証用サンプル | CC BY 4.0 | [公式ページ](https://jonbarron.info/mipnerf360/) |

> 本番運用ではドローン撮影による自社工場画像を使用します。

## 主要ライブラリ

| ライブラリ | バージョン | ライセンス |
|-----------|-----------|-----------|
| Open3D | >= 0.19.0 | MIT |
| pvlib | >= 0.11.0 | BSD-3-Clause |
| trimesh | >= 4.0.0 | MIT |
| Gradio | >= 5.0.0 | Apache 2.0 |
| FastAPI | >= 0.115.0 | MIT |
| Unsloth | latest | Apache 2.0 |
| NumPy | >= 1.26.0 | BSD-3-Clause |
| Plotly | >= 5.18.0 | MIT |

## GPU検証結果

| 環境 | VGGT推論 (10枚) | Peak VRAM | メッシュ処理 |
|------|-----------------|-----------|-------------|
| T4 16GB | 39.3s | 10.5 GB | 2.7s (改善後) |
| H100 80GB (推定) | 4-8s | ~15 GB | <1s |

## 設定

`configs/solar_params.yaml` でパラメータを調整可能:

- **location**: 緯度・経度・標高・タイムゾーン
- **simulation**: 年・時間分解能・日射モデル
- **panel**: 効率・劣化率・設置単価・寿命
- **electricity**: 電気料金・年間上昇率
- **mesh**: 前処理・再構成・後処理パラメータ

## ライセンス

本プロジェクトは [Apache License 2.0](LICENSE) の下で公開されています。

使用する外部モデル・ライブラリについては、それぞれのライセンス条項に従ってください。
詳細は上記の「使用モデル」「主要ライブラリ」セクションを参照してください。
