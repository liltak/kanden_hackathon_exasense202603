---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-family: 'Noto Sans JP', 'Hiragino Kaku Gothic ProN', sans-serif;
    font-size: 20px;
    line-height: 1.4;
    padding: 40px 50px 30px;
    background: #ffffff;
  }
  section.title {
    text-align: center;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: #ffffff;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
  }
  section.title h1 { font-size: 2.2em; margin-bottom: 0.1em; border-bottom: none; color: #ffffff; }
  section.title h2 { font-size: 1.0em; font-weight: 400; color: #a8d8ea; margin-top: 0; }
  section.title p { color: #c8d6e5; font-size: 0.85em; }
  section.section-divider {
    text-align: center;
    background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
    color: #ffffff;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
  }
  section.section-divider h1 { border-bottom: none; color: #ffffff; font-size: 1.8em; }
  section.section-divider h2 { color: #a8d8ea; font-weight: 400; font-size: 1.0em; }
  h1 { color: #0f3460; border-bottom: 2px solid #e94560; padding-bottom: 0.15em; font-size: 1.4em; margin-bottom: 0.3em; margin-top: 0; }
  h2 { color: #16213e; font-size: 1.15em; margin-bottom: 0.2em; }
  h3 { font-size: 0.95em; margin-top: 0.4em; margin-bottom: 0.2em; color: #0f3460; }
  ul, ol { margin: 0.15em 0; padding-left: 1.3em; }
  li { margin: 0.1em 0; }
  p { margin: 0.2em 0; }
  table { font-size: 0.78em; margin: 0.3em 0; }
  table th, table td { padding: 0.2em 0.5em; }
  pre { margin: 0.3em 0; }
  code { font-size: 0.82em; }
  pre code { font-size: 0.75em; line-height: 1.3; }
  .columns { display: flex; gap: 1.5em; }
  .columns > div { flex: 1; }
  strong { color: #e94560; }
  .highlight { background: #fff3cd; padding: 0.3em 0.8em; border-left: 3px solid #e94560; margin: 0.3em 0; font-size: 0.9em; }
  .small { font-size: 0.7em; color: #666; }
---

<!-- _class: title -->

# ExaSense

## 工場向けエネルギー最適化ソリューション
ドローン撮影 × 3D再構築 × 日照シミュレーション × VLM分析

<br>

2026年3月28日 | 関西電力 空間理解AIハッカソン 成果発表会

---

# 課題設定 — なぜ「空間理解 × エネルギー」か

<div class="columns">
<div>

### 社会背景
- 2050年カーボンニュートラル目標
- 工場の脱炭素化が **最重要課題** の一つ
- 関西圏の工場屋根 = **未活用の巨大エネルギー資源**

### 現状の問題
- 現地調査 → 手動設計 → 見積もりに **数週間〜数ヶ月**
- 屋根の3D形状・周囲建物の影を正確に評価できない
- 専門知識が必要 → 中小工場で導入ハードルが高い

</div>
<div>

### ExaSenseの着眼点

<div class="highlight">

**「空間を理解する」= 建物の3D構造 + 太陽の動き + 影の物理挙動を統合的にモデリングする**

</div>

```
ドローン撮影 (10-20枚)
    ↓  VGGT 3D再構築
    ↓  年間日照シミュレーション
    ↓  VLM AI設置提案
最適化レポート + 3Dビューア
```

**数週間 → 約6分** に短縮
専門知識不要で意思決定を支援

</div>
</div>

---

<!-- _class: section-divider -->

# 技術アプローチ
## 5フェーズ統合パイプライン

---

# アーキテクチャ全体像

```
 Phase 1-2 (GPU)          Phase 3 (CPU)              Phase 4 (GPU)         Phase 5
┌───────────────────┐   ┌────────────────────┐   ┌───────────────────┐   ┌────────────────┐
│  3D再構築         │   │  日照シミュレーション    │   │  VLM AI分析       │   │  WebUI         │
│                   │──▶│                    │──▶│                   │──▶│                │
│  VGGT-1B-Comm.    │   │  pvlib + trimesh   │   │  Qwen2.5-VL-7B   │   │  Next.js 16    │
│  SAM3 前景抽出    │   │  Perez拡散モデル   │   │  Unsloth LoRA     │   │  Three.js 3D   │
│  Kaolin DMTet     │   │  EPW実気象データ   │   │                   │   │  FastAPI + WS  │
└───────────────────┘   └────────────────────┘   └───────────────────┘   └────────────────┘
    33秒 (VGGT)              28秒                       ~10秒                 リアルタイム
    319秒 (メッシュ)
```

- **全フェーズ自動実行** — 画像アップロードから設置提案まで人手介入ゼロ
- **モジュラー設計**: 各フェーズ独立テスト・差し替え可能 (VGGT ↔ COLMAP)
- **GPU/CPU分離**: Phase 3 + 5 は CPU のみで動作、Phase 1-2 + 4 は H100 GPU

---

# Phase 1-2: 3D再構築 + メッシュ処理

<div class="columns">
<div>

### 処理パイプライン
1. ドローン画像入力 (10〜20枚)
2. **VGGT-1B-Commercial** で点群生成
3. **SAM3テキストセグメンテーション** で前景抽出
4. Voxel downsample + 2段階外れ値除去
5. **Poisson surface reconstruction** (depth=7)
6. Taubin平滑化 + 20K面に間引き

### H100 実測ベンチマーク

| 指標 | 実測値 |
|------|:---:|
| VGGT推論 (20枚) | **5.87秒** |
| 出力点群 | **3,480,960点** |
| Peak VRAM | **11.19 GB** |
| メッシュ出力 | **20,000面** |

</div>
<div>

### 技術的工夫

**SAM3 前景抽出** (新規開発)
- テキストプロンプトで "building" を指定
- 背景（空・地面）を自動除去
- 点群品質を大幅に向上

**Kaolin DMTet** (NKSR代替)
- 非商用ライセンスの NKSR を排除
- NVIDIA Kaolin の DMTet で GPU メッシュ再構成
- 商用利用完全対応

**適応的処理**
- バイナリPLY出力 (ASCII比 **30倍** 高速)
- 50K点超で O(n) カメラベース法線推定に切替
- RANSAC平面検出で屋根面を自動フラット化

</div>
</div>

---

# Phase 3: 日照シミュレーション

<div class="columns">
<div>

### 物理ベースシミュレーション
- **pvlib**: 年間 **8,760時点** の太陽位置・日射量
- **Perez異方性拡散モデル**: 天空の非均一な散乱光を再現
- **セル温度補正**: NOCT モデルで熱損失を算出
- **レイキャスティング**: trimesh による面ごとの影行列

### 日射量の3成分
```
年間日射量 = 直達光 (DNI × cosθ × 影)
           + 散乱光 (DHI × Perez × 天空視認率)
           + 地面反射 (GHI × albedo × 視認率)
```

</div>
<div>

### ROI自動算出
- 上向き面の自動検出 + 面積フィルタリング
- 面ごとの年間発電量 (kWh)
- 設置コスト / 年間削減額 (JPY)
- **投資回収期間 / 25年NPV / IRR**
- 優先ランキング (回収期間昇順)

### 適応型フィルタリング
- VGGT由来の微小メッシュ面に対応
- 面積閾値を自動スケーリング
  - 95%以上除外 → 10th percentile に調整
- **proposals > 0 を常に保証**

### テスト
- **18テスト** で全シミュレーションロジックをカバー
- 太陽位置、レイキャスト、ROI計算を個別検証

</div>
</div>

---

# Phase 4: VLM AI分析

<div class="columns">
<div>

### マルチモーダル設置提案
- **Qwen2.5-VL-7B-Instruct** (Apache 2.0)
- 入力: 日照ヒートマップ画像 + シミュレーション数値データ
- 出力: 自然言語での設置提案レポート

### プロンプト設計 (3種)
1. **パネル配置**: 最適エリアの特定と理由
2. **コスト分析**: 投資対効果の詳細説明
3. **ROI分析**: 経営者向け意思決定レポート

### LoRAファインチューニング
- **Unsloth** による QLoRA 4-bit
- H100 80GB で効率的に学習
- 工場屋根特化の応答品質向上

</div>
<div>

### 出力例
```markdown
## 太陽光パネル設置提案

### 屋根評価
南東向き屋根（面積 420m²）が最も
高い年間日射量 1,247 kWh/m² を記録。
周囲建物による影の影響は午前中の
2時間のみで、年間を通じて良好。

### 推奨配置
- 南東屋根: 85kW (優先度1)
- 北西屋根: 40kW (優先度2)

### 期待効果
- 年間発電: 189 MWh
- CO2削減: 89.7t/年
- 投資回収: 5.2年
```

</div>
</div>

---

# Phase 5: WebUI — Next.js + FastAPI

<div class="columns">
<div>

### Next.js 16 フロントエンド (9ページ)
- **Dashboard**: パイプラインステータス + KPIカード
- **3D Viewer**: Three.js メッシュビューア
  - 点群 / Poissonメッシュ 切替
  - 月別日照ヒートマップオーバーレイ
  - 太陽軌道アニメーション + コンパス
- **Simulation**: パラメータ設定 + 実行 + 結果表示
- **AI Analysis**: VLMチャットインターフェース
- **Reports**: Markdown/JSON/HTML レポート生成
- **Rust Inspection**: 設備錆検査 (OpenVLA)
- **Waypoint**: ワールドモデル可視化

### 技術スタック
- React 19 + TypeScript + Tailwind CSS 4
- React Three Fiber / drei (3Dレンダリング)
- shadcn/ui + Radix UI (30+コンポーネント)
- TanStack React Query (状態管理)

</div>
<div>

### FastAPI バックエンド
- **20+ REST エンドポイント**
- **WebSocket** リアルタイム進捗通知
- **非同期実行**: Celery + Redis タスクキュー
- **ストレージ**: MinIO (S3互換) + SQLAlchemy ORM

### 主要API
```
POST /api/reconstruction/start  # 3D再構築
POST /api/simulation/run        # 日照シミュレーション
POST /api/chat/message          # VLM AI分析
POST /api/report/generate       # レポート生成
POST /api/solar-animation/positions
POST /api/rust-inspection/run   # 錆検査
POST /api/waypoint/generate     # ワールドモデル
WS   /api/ws/progress           # 進捗通知
```

### H100 プロキシ構成
- フロントエンド → FastAPI → H100 GPU API
- GPU処理はH100に委譲、結果をストリーミング

</div>
</div>

---

# GPU活用の工夫

<div class="columns">
<div>

### H100 80GB HBM3 活用戦略
- **VGGT推論**: 20枚の画像 → 5.87秒で3.48M点群生成
- **VLM推論**: Qwen2.5-VL-7B で bf16 推論
- **メッシュ処理**: Kaolin DMTet で GPU メッシュ再構成
- **SAM3**: テキストプロンプトベースのセグメンテーション

### macOS / CPU 互換性
```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch  # 型チェック時のみ

# 実行時は遅延インポート
def run():
    import torch  # GPU環境でのみ実行
```

- 全モジュールが **macOS (CPU) でインポート可能**
- GPU依存を完全に遅延ロード化
- 開発・テストはローカル、本番はH100

</div>
<div>

### ライセンスコンプライアンス

| モデル | ライセンス | 商用 |
|--------|-----------|:---:|
| VGGT-1B-**Commercial** | VGGT AUP | OK |
| Qwen2.5-VL-7B | Apache 2.0 | OK |
| Kaolin DMTet | Apache 2.0 | OK |
| SAM3 | Apache 2.0 | OK |

- NKSR (非商用) → **Kaolin DMTet に置換**
- VGGT → **Commercial版を選定**
- 全依存ライブラリの商用利用を確認済み

### 開発体制
- **70コミット** の継続的開発
- Python 3.12 + uv パッケージマネージャ
- 18テスト (Phase 3 100%カバレッジ)
- Apache 2.0 ライセンスで公開

</div>
</div>

---

<!-- _class: section-divider -->

# デモ + 実績データ
## H100 E2Eパイプライン実行結果

---

# H100 E2E パイプライン実測結果

<div class="columns">
<div>

### 実行環境
- **GPU**: NVIDIA H100 80GB HBM3
- **Driver**: v570.195.03
- **入力**: 工場画像 20枚

### Phase別タイミング

| Phase | 処理内容 | 時間 |
|-------|---------|-----:|
| 1 | VGGT 点群生成 | **33.39秒** |
| 2 | メッシュ処理 | **318.55秒** |
| 3 | 日照シミュレーション | **28.48秒** |
| 4 | VLM分析 | **~10秒** |
| | **合計** | **~382秒** |

</div>
<div>

### 出力サマリ

| 指標 | 値 |
|------|---:|
| 出力点群 | **3,480,960点** |
| 出力メッシュ | **20,000面 / 10,194頂点** |
| 表面積 | **38.18 m²** |
| 屋根面 | **1,493面 (7.5%)** |
| 壁面 | **16,524面 (82.6%)** |
| 上向き面 (シミュレーション対象) | **2,056面** |
| Peak VRAM | **11.19 GB** |

### デモデータ
- **South Building**: 4棟工場コンプレックス
- **Colosseum**: 参考建築 (30フレーム)
- GLBフォーマットで Three.js ビューア表示

</div>
</div>

---

# デモシナリオ

<div class="columns">
<div>

### ライブデモ (WebUI)

**Step 1**: ダッシュボードでパイプライン概要を確認

**Step 2**: 3D Viewer で工場メッシュを表示
- 点群 ↔ Poissonメッシュ切替
- マウスで回転・ズーム操作
- 月別日照ヒートマップ表示

**Step 3**: Simulation で日照解析を実行
- パラメータ設定 (大阪 34.69°N)
- リアルタイム進捗バー表示
- 結果: 年間発電量 + ROI + ヒートマップ

**Step 4**: AI Analysis でVLMに質問
- 「南側屋根のパネル設置について提案して」
- VLMが数値データを踏まえて回答

</div>
<div>

### デモで見せる「空間理解」

1. **3D構造の理解**
   - 画像 → 3D点群 → メッシュ化
   - 屋根/壁/地面の自動分類

2. **物理現象の理解**
   - 年間8,760時点の太陽軌道
   - 建物間の影の相互作用
   - 面ごとの日射量ヒートマップ

3. **意思決定への変換**
   - 設置優先度ランキング
   - 投資回収期間の自動算出
   - VLMによる自然言語での提案

</div>
</div>

---

<!-- _class: section-divider -->

# 事業性 + 市場展望

---

# 事業モデル

<div class="columns">
<div>

### デモ工場シミュレーション結果

| 指標 | 値 |
|------|---:|
| 設置可能容量 | **823 kW** |
| 年間発電量 | **1,252 MWh** |
| 年間コスト削減 | **¥3,756万** |
| 投資回収期間 | **5.5年** |
| 25年NPV | **¥5.7億** |

.small[*大阪 34.69°N, 4棟コンプレックス, パネル効率22%, 電力単価30円/kWh*]

### 対象市場
- 関西圏の中〜大規模工場 (屋根 1,000m²+)
- 国内工場 約 **40万箇所**
- RE100 / SBT 対応企業の需要急増

</div>
<div>

### 競合優位性

| | 従来の方法 | ExaSense |
|---|---|---|
| 調査期間 | 数週間〜数ヶ月 | **約6分** |
| 調査コスト | 数十万円 | **数万円** |
| 精度の根拠 | 手動見積もり | **物理シミュレーション** |
| レポート | 専門家作成 | **VLM自動生成** |
| 3D可視化 | なし | **インタラクティブ** |

### SaaS展開シナリオ
1. **ドローン撮影** (外注 or 自社)
2. **ExaSense クラウド分析** (画像アップロード → 自動解析)
3. **レポート + 施工会社マッチング**

### 拡張先
- 商業施設・学校・公共建築
- 蓄電池・EV充電との統合最適化
- 関西電力インフラ設備点検 (→ 錆検査機能)

</div>
</div>

---

# 実験と試行錯誤 — 世界モデル (Waypoint-1-Small)

<div class="columns">
<div>

### やったこと
- **Waypoint-1-Small** (2.3B params) を H100 で検証
- torch.compile で **15.9〜24.0 FPS** を達成 (20.2倍高速化)
- VRAM 18.3GB — VGGT + VLM と併用可能 (計44GB/80GB)
- 制御入力を網羅的に調査 (button ID 0〜100 をスキャン)

### 発見した課題
- **ボタンIDに方向制御がない**: 0〜100全スキャンの結果、前後左右の移動制御が不可能と判明 (全て右方向にバイアス)
- 10,000時間の多様なゲーム映像で学習 → キー割り当てが平均化
- **マウス速度のみ**がカメラ回転として安定動作
- テキストプロンプトは雰囲気制御のみで移動には効かない

</div>
<div>

### 構想していた3つの応用案

**案1: Before/After シーン生成**
- パネル設置前後の工場外観を生成 → 意思決定支援
- Issue #46, #47, #62

**案2: インタラクティブ工場探索**
- Waypoint → Next.js リアルタイムストリーミング
- ウォークスルーでの設置箇所確認
- Issue #49, #50, #70

**案3: 合成データ拡張**
- 天候・季節バリエーション生成 → VLM学習データ
- Issue #51, #52, #53, #71〜73

### 現状と判断
- ボタン制御不可 → **オービットツアー方式**に設計変更
- UI・APIフレームワークは構築済み (mock動作)
- ライセンス矛盾 (Apache 2.0 vs GPL-3.0ヘッダー) → 確認待ち

</div>
</div>

---

# 実験と試行錯誤 — メッシュ再構成とライセンス問題

<div class="columns">
<div>

### NKSR → Kaolin DMTet 移行の経緯

**NKSR (Neural Kernel Surface Reconstruction)**
- GPU上でニューラルサーフェス再構成
- 412秒 → **20.4秒** (20.2倍高速化)
- Peak VRAM: 28.0GB

**しかし**: NVIDIA Source Code License (**非商用**) と判明
→ ハッカソンルール違反のため **全コード削除**

**Kaolin DMTet で代替**
- 微分可能マーチングテトラヘドラ
- **28.3秒** (NKSRより若干遅い)
- VRAM **1.4GB** (NKSRの **1/20**)
- Apache 2.0 ライセンス → 商用OK

</div>
<div>

### ライセンス判断の全体像

| モデル/ツール | 判断 | 理由 |
|---|---|---|
| VGGT-1B | → **Commercial版** に変更 | 研究版はCC BY-NC |
| NKSR | **削除** | NVIDIA NCL (非商用) |
| Kaolin DMTet | **採用** | Apache 2.0 |
| Waypoint-1-Small | **保留** | Apache 2.0表記だがソースにGPL-3.0ヘッダー |
| SAM3 | **採用** | Apache 2.0 |
| Qwen2.5-VL | **採用** | Apache 2.0 |

### 学び
- モデルカードだけでなく **ソースコードのヘッダー** まで確認が必要
- 「商用利用OK」の表記があっても依存ライブラリで伝播する場合がある
- ハッカソンでは **ライセンス確認を最初にやるべき**

</div>
</div>

---

# 実験と試行錯誤 — OpenVLA 錆検査エージェント

<div class="columns">
<div>

### コンセプト
VLA (Vision-Language-Action) モデルで **インフラ設備の錆を自動追跡**

### 構築したもの
- **合成データ生成パイプライン**
  - 4テクスチャ (コンクリート/石/木/アスファルト)
  - 4変形モード (直線/波状/ジグザグ/湾曲)
  - 3層レンダリング (影/コア/中心線)
  - 300エピソード × 20-30フレーム
- **ActionTokenizer**: 連続 [Δx,Δy] → 256ビン離散化
- **学習フレームワーク**: OpenVLA 7B + LoRA (rank=32)
- **ロールアウト評価**: DFS探索でカバレッジ率測定
- **WebUI**: Next.js で軌跡SVG可視化 + メトリクス表示

</div>
<div>

### 未検証項目
- H100上での学習実行 (フレームワークのみ完成)
- 合成データ → 実データへの転移性能
- 分岐・交差するクラックへの対応
- 実際のドローン撮影画像での検証

### 関西電力インフラへの展開構想
```
太陽光パネル最適化 (本プロジェクト)
    ↓ 同じパイプライン基盤を応用
設備点検・劣化診断
```
- 送電鉄塔の錆・腐食検出
- 変電設備の外観異常検知
- パネル自体の劣化検出
- **LoRA切替で多タスク対応** — 基盤モデルは共有

</div>
</div>

---

# 今後の展望 — Issue から見えるロードマップ

<div class="columns">
<div>

### エネルギー最適化の深化 (Issue #28〜34)
- **ソーラーカーポート3D設計** (#28)
  - 駐車場空間の3D解析 → カーポート＋太陽光パネル最適配置
- **コーポレートPPA長期予測** (#29)
  - 10-25年の発電量予測 + 経済性評価
- **複数棟一括評価** (#30)
  - 工場コンプレックス全体の最適化
- **排熱回収ポイントの空間特定** (#31)
  - 熱源電化支援のための3D熱分布解析
- **EV充電ステーション最適配置** (#32)
  - 駐車場3D解析 + 動線シミュレーション
- **蓄電池の空間最適化** (#33)
  - 設置位置・容量の3D制約付き最適化
- **ゼロカーボンロードマップ自動生成** (#34)
  - 太陽光+蓄電池+EV+排熱 の統合提案

</div>
<div>

### 技術的な発展 (Issue #16, #17, #58)
- **2DGS/GOF 代替3D再構成** (#16)
  - 2D Gaussian Splatting で高品質な新規ビュー合成
- **GPUシャドウマップ** (#17)
  - レイキャストの代わりにGPUレンダリングで影を高速計算
  - 年間8,760時点 → リアルタイム化の可能性
- **HY-WorldPlay (Tencent)** (#58)
  - Waypoint代替の世界モデルとして評価予定
  - Waypoint のボタン制御問題を解決できる可能性

### 実証に向けて (Issue #1, #2, #4)
- **実データE2E** (#4): 実ドローン撮影での検証
- **VLMファインチューニング** (#2): 工場屋根特化データ
- **H100本番環境** (#1): 全フェーズ統合検証

</div>
</div>

---

# 考察 + 限界

<div class="columns">
<div>

### 技術的限界
- **メッシュ処理時間**: Poisson再構成が 319秒と律速
  - → Ball Pivoting / Alpha Shape で高速化の余地
- **屋根面積の過小評価**: VGGT点群のスケール推定に不確実性
  - → GCPマーカーや GPS 連携で改善可能
- **VLMのハルシネーション**: 数値の引用ミスの可能性
  - → 数値はシミュレーション結果を直接参照する設計に
- **世界モデルの制御制約**: Waypointのボタン制御が不可
  - → マウスベースのオービット + 代替モデル検討

### スケーラビリティ
- 現状: 単一 H100 で逐次処理
- 将来: 複数GPU並列 + クラウドバースト

</div>
<div>

### 試行から得た知見
- **ライセンス確認は初日にやる** (NKSR削除で2日ロス)
- **世界モデルは制御入力の仕様確認が最優先** (ボタンスキャンに1日)
- **前景抽出は点群品質に直結** (SAM3導入でパネル検出46%↑)
- **GPU/CPU分離設計は開発効率に大きく寄与** (macOSで全モジュールテスト可能)

### 今後の改善方針
1. **精度向上**: 実気象データ(EPW)統合 + Perezチューニング
2. **高速化**: メッシュ処理のGPU並列化 + シャドウマップ
3. **VLM強化**: 工場屋根特化データでのファインチューニング
4. **実証**: 関西圏の実工場でのPoC実施
5. **統合エネルギー**: 太陽光+蓄電池+EVの包括最適化

</div>
</div>

---

# まとめ

<div class="columns">
<div>

### 達成したこと
- ドローン画像 → パネル設置提案の **完全自動パイプライン**
- **5フェーズ統合**: 3D再構築 → メッシュ → シミュレーション → VLM → WebUI
- **H100 E2E**: 20枚の画像から **約6分** で設置提案まで完了
- **Next.js 9ページ + FastAPI 20+ API** のプロダクション品質UI

### 技術的ハイライト
- SAM3前景抽出 / Kaolin DMTet / RANSAC屋根検出
- 年間8,760時点 × メッシュ全面の物理シミュレーション
- GPU/CPU分離 + macOS互換 / 全モデル商用ライセンス確認済み

</div>
<div>

### 独創性
- 「空間理解」を **エネルギー最適化** という実課題に直結
- 3D再構築 + 物理シミュレーション + VLM を **一気通貫** で統合
- 同一基盤で **設備点検** にも展開可能

### 実験から得た視座
- 世界モデル (Waypoint) / VLAエージェント (OpenVLA) / セグメンテーション (SAM2→SAM3) を横断的に検証
- ライセンス・制御制約・品質の **実践的な知見** を蓄積
- 「やれなかったこと」が **次のロードマップ** に直結
  - 75件のIssue = 技術的発展の設計図

</div>
</div>

---

<!-- _class: title -->

# ありがとうございました

## ExaSense — 工場屋根を資産に変える
ドローン × 3D再構築 × 日照シミュレーション × AI分析

<br>

ご質問をお待ちしています
