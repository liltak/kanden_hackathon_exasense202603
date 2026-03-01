---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-family: 'Noto Sans JP', 'Hiragino Kaku Gothic ProN', sans-serif;
    background: #ffffff;
  }
  section.title {
    text-align: center;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: #ffffff;
  }
  section.title h1 { font-size: 2.4em; margin-bottom: 0.2em; }
  section.title h2 { font-size: 1.2em; font-weight: 400; color: #a8d8ea; }
  h1 { color: #0f3460; border-bottom: 3px solid #e94560; padding-bottom: 0.2em; }
  table { font-size: 0.85em; }
  code { font-size: 0.8em; }
  .columns { display: flex; gap: 2em; }
  .columns > div { flex: 1; }
  strong { color: #e94560; }
---

<!-- _class: title -->

# ExaSense

## 工場向けエネルギー最適化ソリューション
ドローン × 3D再構築 × 日照シミュレーション × AI分析

<br>

2026年3月 | 関西電力 空間理解AIハッカソン

---

# 課題設定 — なぜ工場屋根 × 太陽光か

<div class="columns">
<div>

### 社会課題
- 2050年カーボンニュートラル目標
- 工場の脱炭素化が急務
- 関西圏の工場屋根 = 未利用の巨大資源

### 現状の課題
- 現地調査 → 手動設計 → 見積もりに **数週間〜数ヶ月**
- 影・屋根形状の正確な評価が困難
- 専門知識が必要 → 中小工場で導入ハードル高

</div>
<div>

### ExaSenseの提案
```
ドローン撮影
    ↓  自動3D再構築
    ↓  日照シミュレーション
    ↓  AI設置提案
最適化レポート
```

**数週間 → 数時間** に短縮
専門知識不要で意思決定を支援

</div>
</div>

---

# アーキテクチャ — 5フェーズ統合パイプライン

```
 Phase 1-2 (GPU)        Phase 3 (CPU)           Phase 4 (GPU)        Phase 5
┌─────────────────┐   ┌──────────────────┐   ┌─────────────────┐   ┌──────────────┐
│  3D再構築       │   │  日照シミュレーション  │   │  VLM AI分析     │   │  WebUI       │
│                 │──▶│                  │──▶│                 │──▶│              │
│  VGGT-1B        │   │  pvlib + trimesh │   │  Qwen2.5-VL     │   │  Next.js     │
│  Poisson再構成  │   │  Perez拡散モデル │   │  LoRA fine-tune │   │  Gradio      │
└─────────────────┘   └──────────────────┘   └─────────────────┘   └──────────────┘
     ~30秒                   ~5秒                   ~10秒              リアルタイム
```

- **全フェーズGPU/CPUで自動実行** — 人手介入ゼロ
- モジュラー設計: 各フェーズ独立テスト・差し替え可能
- REST API + WebSocket でリアルタイム進捗通知

---

# Phase 1-2: 3D再構築

<div class="columns">
<div>

### パイプライン
1. ドローン画像入力 (10〜128枚)
2. **VGGT-1B-Commercial** で点群生成
3. Voxel downsample + 外れ値除去
4. Poisson surface reconstruction
5. メッシュ平滑化 + 間引き → 20K面

### ベンチマーク (South Building)

| 指標 | COLMAP+OpenSplat | VGGT |
|------|:---:|:---:|
| 処理時間 | 228秒 | 8秒 |
| 点数 | 54K | 1.7M |
| メモリ | 11.2 GB | 3.8 GB |

</div>
<div>

### 技術的工夫
- **バイナリPLY出力**: 3.5M点を0.3秒で保存 (ASCII比 30倍高速)
- **適応的法線推定**: 50K点超で O(n) カメラベース切替
- RANSAC平面検出で屋根面を自動フラット化

</div>
</div>

---

# Phase 3: 日照シミュレーション

<div class="columns">
<div>

### 年間シミュレーション
- **pvlib**: 年間8,760時点の太陽位置計算
- **Perez拡散モデル**: 異方性散乱光を考慮
- **セル温度補正**: SAPM モデル
- **レイキャスティング**: 面ごとの影行列

### ROI自動算出
- 年間発電量 (kWh)
- 設置コスト / 年間削減額 (JPY)
- 投資回収期間 / 25年NPV / IRR
- 面ごとの優先ランキング

</div>
<div>

### 適応型フィルタリング
- VGGT由来の微小メッシュ面に対応
- 面積閾値を自動スケーリング
  - 95%以上除外 → 10th percentile に調整
- **結果: proposals > 0 を保証**

### 出力例
```
面 #42: 年間 1,247 kWh/m²
  設置容量: 14.0 kW
  年間削減: ¥504,000
  回収期間: 6.6年
  25年NPV: ¥6,958,008
```

</div>
</div>

---

# Phase 4: VLM分析

<div class="columns">
<div>

### アプローチ
- **Qwen2.5-VL-7B** ベース
- シミュレーション結果画像 → 自然言語レポート
- **Unsloth LoRA** ファインチューニング
  - QLoRA 4-bit quantization
  - H100 80GB で学習

### プロンプト設計
```
この工場屋根の日照解析結果を分析し、
太陽光パネル設置の最適な配置と
期待される効果を報告してください。
```

</div>
<div>

### 出力内容
1. 屋根の形状・方位の評価
2. 影の影響分析
3. 最適設置エリアの提案
4. 期待される発電量・コスト削減

### 利点
- 専門知識がなくても理解可能
- 数値データ + 定性的な推奨を統合
- 多言語対応可能 (日本語/英語)

</div>
</div>

---

# Phase 5: WebUI

<div class="columns">
<div>

### Next.js フロントエンド
- **Three.js** 3Dメッシュビューア
- 日照ヒートマップオーバーレイ
- ROIダッシュボード
- パネル配置シミュレーター
- レスポンシブ対応 (PC/タブレット)

### Gradio バックアップUI
- 5タブダッシュボード
- リアルタイム進捗表示

</div>
<div>

### FastAPI バックエンド
- REST API + WebSocket
- 非同期パイプライン実行
- ジョブキュー管理

### 主要エンドポイント
```
POST /api/reconstruct   # 3D再構築
POST /api/simulate       # 日照シミュレーション
GET  /api/results/{id}   # 結果取得
WS   /api/ws/progress    # 進捗通知
```

</div>
</div>

---

# デモ — South Building 実データ

<div class="columns">
<div>

### 入力
- South Building データセット (128枚)
- 建物外観のマルチビュー画像

### パイプライン実行
1. VGGT推論 → 1.7M点の点群
2. Poisson再構成 → 20Kメッシュ
3. 日照シミュレーション → ヒートマップ
4. ROI算出 → 設置提案

### 処理時間
| ステップ | 時間 |
|---------|------|
| VGGT推論 | ~8秒 |
| メッシュ処理 | ~25秒 |
| 日照シミュレーション | ~5秒 |
| **合計** | **< 60秒** |

</div>
<div>

### H100 での実行
```bash
uv run python scripts/run_h100_e2e.py \
  --image-dir data/raw/.../images \
  --max-images 20 \
  --output-dir data/e2e_results/v2
```

- GPU: NVIDIA H100 80GB
- VRAM使用: ~3.8 GB (peak)
- 完全自動 — 介入不要

</div>
</div>

---

# KPI — 実データ結果

| 指標 | 値 | 備考 |
|------|:---:|------|
| 処理時間 (E2E) | **< 60秒** | H100, 20枚入力 |
| 点群密度 | **1.7M点** | VGGT-1B |
| メッシュ品質 | **20K面** | Poisson depth=7 |
| 設置提案数 | **> 0** | 適応型フィルタで保証 |
| 年間発電量推定 | — | 実行後更新 |
| 年間コスト削減 | — | 実行後更新 |
| 投資回収期間 | — | 実行後更新 |

> KPI欄は H100 再実行後に実数値で更新予定

---

# 事業モデル

<div class="columns">
<div>

### 対象市場
- 関西圏の中〜大規模工場
- 屋根面積 1,000m² 以上
- 国内工場 約40万箇所

### サービス形態 (SaaS)
1. **ドローン撮影** (外注 or 自社)
2. **ExaSense クラウド分析**
   - 画像アップロード → 自動解析
   - 月額 or 従量課金
3. **レポート + 施工会社マッチング**

</div>
<div>

### 競合優位性
| | 従来 | ExaSense |
|---|---|---|
| 調査期間 | 数週間 | **数時間** |
| コスト | 数十万円 | **数万円** |
| 精度 | 手動見積もり | **物理シミュレーション** |
| レポート | 専門家作成 | **AI自動生成** |

### 拡張性
- 全国展開 → 海外展開
- 工場以外 (商業施設、学校)
- 風力・蓄電池との統合

</div>
</div>

---

# まとめ + 今後の展望

### 達成したこと
- ドローン画像 → 設置提案の **完全自動パイプライン**
- 5フェーズ統合: 3D再構築 → シミュレーション → AI分析 → WebUI
- **COLMAP比 28.5倍の効率化** (VGGT採用)
- 18テストによる品質保証

### 技術的ハイライト
- バイナリPLY (30x高速化)、適応法線推定、自動フィルタスケーリング

### 今後の展望
- 実気象データ (EPW) 統合 → シミュレーション精度向上
- VLMファインチューニング → レポート品質向上
- 実工場でのPoC → 事業化検証

---

<!-- _class: title -->

# ありがとうございました

## ExaSense — 工場屋根を資産に変える

<br>

ご質問をお待ちしています
