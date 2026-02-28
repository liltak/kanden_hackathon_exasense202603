# Phase 1-2 GPU検証計画

## 目的
H100到着前に、AWSのGPUスポットインスタンスでPhase 1-2パイプラインの動作確認と性能見積もりを行う。

## AWS検証結果 (2026-02-27)

### 実行環境
| 項目 | 設定 |
|------|------|
| インスタンス | g4dn.xlarge (NVIDIA Tesla T4, 16GB VRAM) |
| スポット価格 | $0.29/hr |
| AMI | Deep Learning Base OSS Nvidia Driver (Ubuntu 22.04) |
| ストレージ | 100GB gp3 |
| NVIDIA Driver | 580.126.09 |
| CUDA | 13.0 |
| Compute Capability | 7.5 |

### 検証結果: 5/5 全チェック合格

| チェック | 結果 | 詳細 |
|---------|------|------|
| GPU認識 | ✅ PASS | Tesla T4 (15.6 GB VRAM, compute 7.5) |
| VGGTモデルロード | ✅ PASS | 54.6秒でロード完了、VRAM 5.03 GB (fp16) |
| VGGT推論 (5枚) | ✅ PASS | 22.1秒、Peak VRAM 9.21 GB |
| Mesh処理 (Open3D) | ✅ PASS | 5000点 → 18,453頂点, 36,738面 in 1.2秒 |
| Phase 3パイプライン | ✅ PASS | trimesh + pvlib in 0.0秒 |

### VGGT出力形式
```
pose_enc:          (1, 5, 9)         # カメラポーズ
depth:             (1, 5, 392, 518, 1)  # 深度マップ
depth_conf:        (1, 5, 392, 518)     # 深度信頼度
world_points:      (1, 5, 392, 518, 3)  # 3D点群マップ
world_points_conf: (1, 5, 392, 518)     # 点群信頼度
```
→ 5枚の480x640画像 → 392x518にリサイズされて推論

### T4 → H100 性能予測

| 指標 | T4 (実測) | H100 (推定) | 根拠 |
|------|-----------|-------------|------|
| VGGT VRAM (モデル) | 5.0 GB | 5.0 GB | モデルサイズ不変 |
| VGGT Peak VRAM (5枚) | 9.2 GB | 9.2 GB | バッチ不変なら同程度 |
| VGGT推論時間 (5枚) | 22.1s | **~2-4s** | FP16 24x + メモリ帯域5.6x |
| VGGT推論時間 (50枚) | OOM | **~15-30s** | 80GB VRAMで余裕 |
| Mesh処理 | 1.2s | ~1s | CPU処理のため差は小さい |
| 合計画像枚数上限 | ~15枚 | **~200枚** | VRAM比例 |

### 判断
1. **VGGT推奨**: モデルロード5GB、推論9GB Peak — H100 80GBで十分余裕
2. **Poisson推奨**: 1秒で18K面のメッシュ生成。十分高速
3. **VLM 7B可能**: VGGT 9GB + Qwen3.5-VL-7B ~15GB = ~24GB。H100で同時ロード可能

## E2Eテスト結果 (Mip-NeRF 360 Garden)

### データセット
- **Mip-NeRF 360 Garden**: 185枚の実写画像 (DSC07956-DSC08140.JPG, 各~11MB)
- ダウンロード: `http://storage.googleapis.com/gresearch/refraw360/360_v2.zip` (11.6GB)
- zip内部構造: `garden/images/xxx.JPG` (360_v2/プレフィックスなし)

### Phase 1: VGGT 3D Reconstruction ✅
| 項目 | 結果 |
|------|------|
| 入力画像 | 10/185枚 (均等サンプリング) |
| テンソル形状 | (10, 3, 336, 518) |
| モデルロード | 74.8秒 |
| 推論時間 | **39.3秒** |
| Peak VRAM | **10.5 GB** |
| 点群数 | **1,740,480点** (conf > 0.3) |
| 信頼度範囲 | [1.000, 16.511] |

VGGT出力形式 (10枚):
```
pose_enc:          (1, 10, 9)
depth:             (1, 10, 336, 518, 1)
depth_conf:        (1, 10, 336, 518)
world_points:      (1, 10, 336, 518, 3)
world_points_conf: (1, 10, 336, 518)
```

### Phase 2: Mesh Processing ✅
| 項目 | 結果 |
|------|------|
| 外れ値除去後 | 485,631点 |
| メッシュ頂点 | **286,527** |
| メッシュ面 | **568,799** |
| 処理時間 | **88.7秒** |
| 出力ファイル | garden_mesh.ply (22MB) |

### Phase 3: Solar Simulation ✅
| 項目 | 結果 |
|------|------|
| 上向き面 | 191,080面 (7.9 m²) |
| 年間GHI | 2,119 kWh/m²/年 |
| 発電ポテンシャル | 1.6 kW peak, 2,677 kWh/年 |
| 年間節約額 | ¥80,323 |
| 設置コスト | ¥394,908 |
| 回収期間 | **4.9年** |
| 処理時間 | **1.4秒** |

### T4 E2E性能まとめ
| ステップ | 時間 |
|---------|------|
| zip DL (11.6GB) | 6分4秒 |
| unzip (garden) | 17秒 |
| VGGTモデルロード | 74.8秒 |
| VGGT推論 (10枚) | 39.3秒 |
| Mesh処理 | 88.7秒 |
| **合計 (推論+Mesh)** | **~3.5分** |

### H100予測 (E2E)
| ステップ | T4 (実測) | H100 (推定) |
|---------|-----------|-------------|
| VGGTモデルロード | 74.8s | ~15s |
| VGGT推論 (10枚) | 39.3s | **~4-8s** |
| VGGT推論 (50枚) | OOM | **~20-40s** |
| Mesh処理 | 88.7s | ~80s (CPU依存) |
| Peak VRAM (10枚) | 10.5GB | ~10.5GB |

## 検証チェックリスト

### Step 1: 環境確認
- [x] GPU認識 (nvidia-smi) — Tesla T4 15.6GB
- [x] PyTorch CUDA動作確認
- [ ] COLMAP インストール確認 (apt-getでインストール不可、ソースビルド必要)

### Step 2: VGGT推論テスト
- [x] モデルダウンロード (facebook/VGGT-1B-Commercial)
- [x] 5枚の合成画像で推論実行
- [x] 出力形式確認 (点群, カメラポーズ, 深度マップ, 信頼度マップ)
- [x] VRAM使用量計測: モデル5.0GB, Peak 9.2GB
- [x] 推論時間計測: 22.1秒 (T4)

### Step 3: COLMAP SfM
- [ ] 別途検証予定 (Dockerでのビルドが必要)

### Step 4: メッシュ処理
- [x] 点群→メッシュ変換 (Poisson) — 1.2秒 (合成), 88.7秒 (実データ286K頂点)
- [x] 法線ベクトル生成・検証
- [x] 実データ検証: Garden 1.7M点 → 286K頂点, 569K面

### Step 5: パイプライン接続
- [x] Phase 3パイプライン (trimesh + pvlib) — 動作確認済み (合成データ)
- [x] E2E接続テスト: VGGT→Mesh成功 (Garden 10枚)
- [ ] Phase 3 pvlib API修正必要 (ineichen引数変更)

## 技術的知見

### VGGTロード方法
```python
# transformers.AutoModelは使用不可 (model_type未定義)
# vggtパッケージの専用クラスを使用
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

model = VGGT.from_pretrained("facebook/VGGT-1B-Commercial").to("cuda")
images = load_and_preprocess_images(image_paths).to("cuda")

# T4 (compute < 8.0) → float16, A10G/H100 (compute >= 8.0) → bfloat16
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
    predictions = model(images)
```

### AWS注意点
- セキュリティグループのアウトバウンドルールが必要 (デフォルトで無い場合あり)
- PyTorch API: `total_memory` (not `total_mem`)
- vggtパッケージ: `pip install git+https://github.com/facebookresearch/vggt.git`

## H100到着時のアクション

1. **VGGT主手法**: COLMAP不要で高速。5枚22秒→H100で2-4秒の見込み
2. **Poisson mesh**: Open3Dの標準手法で十分な品質
3. **VLM同時ロード可能**: VGGT 9GB + Qwen3.5-VL-7B 15GB = 24GB << 80GB

### H100セットアップ
```bash
git clone <repo> && cd exasense
bash scripts/setup_h100.sh          # 自動で全依存インストール
uv run python -m src.ui.app         # WebUI起動
```
