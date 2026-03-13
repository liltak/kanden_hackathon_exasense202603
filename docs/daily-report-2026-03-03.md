# 開発レポート 2026-03-03

## 概要

H100 GPU環境でのE2Eパイプライン最適化を実施。
主にPhase 2（メッシュ再構成）のGPU高速化とPhase 4（VLM）の動作検証を完了した。

## 変更一覧

| コミット | 内容 |
|---|---|
| `0ba3787` | H100 E2E Phase 4 (VLM) + 自動スケール推定の追加 |
| `7d3be3d` | VLM flash_attn → sdpa フォールバック + ランタイム import 修正 |
| `9f67712` | Gradio UI廃止、Next.js に一本化 |
| `0e4837b` | **NKSR GPU メッシュ再構成の追加** |
| `b236df7` | process_reconstruction の depth kwarg ルーティング修正 |
| `9b0b84a` | Waypoint-1-Small 推論スクリプトとベンチマークレポート追加 |
| `909ccbb` | src/waypoint モジュール（WaypointGenerator クラス）追加 |

---

## 1. NKSR GPU メッシュ再構成 (Issue #11)

### 背景

Phase 2 のメッシュ再構成は Poisson (CPU) を使用しており、3.4M点のポイントクラウドで **412秒** かかっていた。
H100 GPU を活用するため、NVIDIA の [NKSR](https://github.com/nv-tlabs/NKSR)（Neural Kernel Surface Reconstruction）を導入。

### 実装内容

- `src/reconstruction/mesh_processor.py`
  - `MeshMethod.NKSR` を enum に追加
  - `_extract_nksr()` メソッド実装（GPU上でニューラルカーネルベースの曲面復元）
  - 法線がない場合は自動推定
  - `process_reconstruction()` でメソッド別に kwargs をルーティング
- `scripts/run_h100_e2e.py`
  - `--mesh-method nksr|poisson` 引数を追加

### H100ビルド手順

NKSR は PyTorch 2.9.1 の pybind11 と互換性問題があったため、以下の手順でビルド:

1. micromamba で CUDA toolkit (nvcc, 各種ヘッダ) をインストール
2. NKSR ソースの `PythonBindings.cpp` を手動パッチ（`array_caster` → カスタム `type_caster`）
3. `torch-scatter` もソースからビルド（`--no-build-isolation`）

```bash
# H100 で再現する場合
export CUDA_HOME=/home/team-002/.local/share/mamba/envs/cuda
export PATH=$CUDA_HOME/bin:$PATH
uv pip install --no-build-isolation /home/team-002/nksr_src/package
uv pip install --no-build-isolation torch-scatter
```

### ベンチマーク結果

**単体比較 (3.4M点 → メッシュ抽出)**

| メソッド | 時間 | 頂点数 | フェース数 | VRAM |
|---|---|---|---|---|
| Poisson (CPU) | 412.0s | ~300K | ~600K | 0 (CPU) |
| **NKSR (GPU)** | **20.4s** | 11.3M | 22.1M | 28.0 GB |
| スピードアップ | **20.2x** | | | |

**E2E パイプライン内 (前処理+デシメーション含む)**

| メソッド | Phase 2 合計 | 抽出のみ |
|---|---|---|
| Poisson | 412s | 313s |
| **NKSR** | **11.0s** | **1.7s** |

### 使い方

```bash
# NKSRでE2E実行
uv run python scripts/run_h100_e2e.py --max-images 20 --mesh-method nksr

# Poisson (従来通り)
uv run python scripts/run_h100_e2e.py --max-images 20 --mesh-method poisson
```

---

## 2. VLM (Phase 4) H100 動作検証

### 実装内容

- `src/vlm/model_loader.py`: `flash_attention_2` がない環境で `sdpa` にフォールバック
- `src/vlm/inference.py`: `PIL.Image` と `torch` の TYPE_CHECKING import をランタイム import に修正
- `scripts/run_h100_e2e.py`: `run_vlm_phase()` 追加（Phase 4 の自動実行）
- `scripts/verify_h100.py`: 7項目のクイック検証スクリプト新規作成

### VLM ベンチマーク (Qwen2.5-VL-7B-Instruct)

| メトリクス | 値 |
|---|---|
| モデルロード | 13.8s |
| 推論（パネル設置分析） | 9.8s |
| スループット | 59.4 tokens/s |
| VRAM ピーク | 16.8 GB |
| 出力トークン数 | 582 |

---

## 3. 自動スケール推定

### 背景

VGGT は正規化座標（最大 ~4.6 unit）で出力するため、Solar Simulation で「0件の提案」が返る問題があった。

### 修正

`scripts/run_h100_e2e.py` に `estimate_scale_factor()` を追加:

```
ポイントクラウドの最大 extent (4.646 units)
→ 期待される建物サイズ (30m)
→ scale_factor = 6.46
```

これにより Solar Simulation が正しく 3024 件の提案を生成。

---

## 4. Gradio UI 廃止

- `src/ui/app.py` (787行) を削除
- pyproject.toml から gradio 依存を除去
- CLAUDE.md に「Gradio 使用禁止」を明記
- フロントエンドは `frontend/` (Next.js 16) に一本化

---

## 5. E2E パイプライン最新結果

**環境**: NVIDIA H100 80GB HBM3 / CUDA 12.8 / PyTorch 2.9.1

| Phase | 処理内容 | 時間 | 主要メトリクス |
|---|---|---|---|
| Phase 1 | VGGT 3D再構成 | 26.2s | 20枚 → 3.48M点, VRAM 11.2GB |
| Phase 2 | NKSR メッシュ | 11.0s | 抽出 1.7s, 9.6K頂点/20K面 |
| Phase 3 | Solar Simulation | 61.9s | 3024提案, 65.2kW, 回収6.2年 |
| **合計** | | **101.4s** | NPV ¥3,885万 (25年) |

前回（Poisson使用時）の合計 ~500秒 から **5倍高速化**。

---

## 6. H100 環境情報

```
接続:     ssh h100
ユーザー:  team-002
リポジトリ: /home/team-002/kanden-hackathon
GPU:      2x NVIDIA H100 80GB HBM3
CUDA:     12.8
PyTorch:  2.9.1+cu128
OS:       Ubuntu 24.04 (Docker)
```

### インストール済みカスタムパッケージ

| パッケージ | バージョン | ビルド方法 |
|---|---|---|
| nksr | 1.0.3 | ソースビルド (パッチ適用) |
| torch-scatter | 2.1.2 | ソースビルド |
| vggt | (git) | pip install済み |
| transformers | 5.2.0 | uv pip |

---

## 次のステップ

- [ ] VLM ファインチューニング (Issue #2)
- [ ] 実データでのE2Eテスト (Issue #4)
- [ ] 2DGS/GOF による高品質3D再構成 (Issue #16)
- [ ] GPU シャドウマップ (Issue #17)
- [ ] フロントエンドへの NKSR メッシュ表示統合
