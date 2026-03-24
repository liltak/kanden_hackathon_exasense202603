# 論文ドラフト：OpenVLAによる自然言語指示ドローンナビゲーション

> **ステータス:** ドラフト（実験結果未記入）
> **対象:** ロボティクス・VLA・体現AIに関する国際会議（IROS / CoRL / ICRA 等）

---

## タイトル案

**日本語：**
> GenesisシミュレータとOpenVLA 7Bを用いた英語自然言語指示によるドローンナビゲーション：自動データ収集とボディフレームアクション学習

**英語：**
> *Drone Navigation via English Natural Language Instructions using OpenVLA 7B: Automated Data Collection with Body-Frame Action Representation in Genesis Simulator*

---

## Abstract（草案）

Vision-Language-Action (VLA) モデルは、自然言語指示と視覚観測から直接ロボットアクションを生成する手法として注目されているが、その多くはロボットアーム操作を対象としており、ドローン飛行制御への適用は限られている。本研究では、OpenVLA 7B を LoRA ファインチューニングにより室内ドローンナビゲーションに転用するパイプラインを提案する。

Genesis 物理シミュレータと PD コントローラを組み合わせた**完全自動データ収集パイプライン**により、人間デモを必要とせず訓練データを生成する。アクションは**機体座標系（ボディフレーム）**の4次元速度ベクトルとして表現し、ドローンの向きに依存しないフレーム不変な行動学習を実現する。英語自然言語指示（"Approach the sofa, fly around it, and take photos." 等）による物体接近・周回ナビゲーションを実証する。

実験では Genesis シミュレータ上で \[X\] 件のエピソードを収集し、\[Y\] エポックの学習後に \[Z\]% の目標到達成功率を達成した。

---

## 1. はじめに

### 1.1 背景

近年、大規模言語モデル（LLM）と視覚エンコーダを統合した **Vision-Language-Action (VLA)** モデルが、ロボット制御のための汎用ポリシーとして研究されている。代表的な OpenVLA [1] は 970K 件の実ロボットデモで事前学習された 7B パラメータのモデルであり、LoRA ファインチューニングにより様々なタスクに適応できる。

しかし、既存の VLA 研究の多くは**ロボットアーム操作**に限定されており、ドローンのような非ホロノミック・不安定な飛行体への適用は十分に探索されていない。その原因として以下が挙げられる：

- **データ不足：** Open X-Embodiment [2] のような大規模ドローンデータセットが存在しない
- **制御の困難さ：** 離散トークン出力による低レベル飛行制御の精度問題
- **コミュニティの分断：** VLA研究者（アーム中心）とドローン研究者（RLおよびMPC中心）の交流が少ない

### 1.2 本研究の貢献

本研究では以下の4点を提案・実証する：

1. **自動データ収集パイプライン：** Genesis + PD コントローラによる人間デモ不要なドローンナビゲーションデータの自動生成
2. **ボディフレームアクション表現：** フレーム不変な行動学習のための機体座標系での4次元アクション定式化
3. **物体指示ナビゲーションのVLA定式化：** 英語自然言語指示（"Approach the [object]..." 等）から直接飛行制御アクションを生成するエンドツーエンドパイプライン
4. **アプローチ+オービット2フェーズタスク：** 物体への接近（アプローチ）後に周囲を一周（オービット）するデータ収集・評価タスクの設計（collect_v2.py）

---

## 2. 関連研究

### 2.1 Vision-Language-Action (VLA) モデル

#### OpenVLA [1]
- **概要：** 970K 件の Open X-Embodiment データで事前学習された 7B パラメータの汎用ロボット操作 VLA
- **アーキテクチャ：** SigLIP（視覚エンコーダ）+ Llama 2 7B（言語モデル）
- **アクション表現：** 各次元を 256 ビンに離散化し、テキストトークンとして生成
- **本研究との関係：** ベースモデルとして採用。LoRAファインチューニングにより室内ドローンナビゲーションに転用

#### OpenVLA-OFT [3]
- **概要：** FAST アクショントークナイザ（DCT + BPE）により推論を最大 15 倍高速化
- **成果：** LIBERO ベンチマークで 97.1% の成功率（π0、Diffusion Policy を上回る）
- **本研究との関係：** 本研究はシンプルな 256 ビン手法を採用。FASTとの比較は将来課題

### 2.2 ドローン向け VLA

#### RaceVLA [4] ★最重要先行研究
- **概要：** OpenVLA を レーシングドローン制御にファインチューニング
- **手法：** FPV 画像 + 英語自然言語 → 4次元アクション（vx, vy, vz, yaw）
- **本研究との違い：**

| 項目 | RaceVLA | **本研究** |
|------|---------|----------|
| タスク | ゲート通過レーシング | **室内物体への接近・周回** |
| データ収集 | 人間デモ（推測） | **PDコントローラによる完全自動収集** |
| 座標系 | ワールド座標系（推測） | **ボディフレーム座標系** |
| シミュレータ | 独自 | **Genesis（最新OSS, Apache 2.0）** |
| 指示言語 | 英語 | **英語** |

#### AutoFly [5]
- **概要：** 屋外未知環境での UAV 自律ナビゲーション VLA
- **特徴：** 疑似深度エンコーダによる空間推論強化、2段階学習戦略
- **本研究との違い：** AutoFly は屋外の障害物回避が主眼。本研究は室内の特定物体への接近

#### UAV-VLA [6]
- **概要：** 衛星・航空画像 + 自然言語による大規模ミッション生成
- **本研究との違い：** UAV-VLA は高レベル経路計画。本研究は低レベル連続アクション生成

#### VLA-AN [7]
- **概要：** 複雑環境でのオンボード推論最適化 VLA
- **本研究との違い：** 軽量化重視。本研究は H100 を用いた研究プロトタイプ

### 2.3 アクショントークナイゼーションの手法

| 手法 | 説明 | 論文 |
|------|------|------|
| **256ビンビニング** | 最も基本的。各次元を均等分割 | OpenVLA [1]、**本研究** |
| FAST（DCT + BPE） | 離散コサイン変換によるトークン圧縮 | OpenVLA-OFT [3] |
| VQ-VAE | ベクトル量子化による高品質トークン | VQ-VLA [8]（ICCV 2025） |

本研究は最もシンプルな 256 ビン手法を採用し、ドローン制御の文脈での基準手法として位置づける。

---

## 3. 提案手法

### 3.1 システム概要

```
[データ収集]          [学習]                    [推論]
Genesis Sim      →   OpenVLA 7B             →  自律飛行
+ PD Controller      + LoRA Fine-tuning
+ 自動生成           + アクショントークナイザ
```

エンドツーエンドパイプラインは3つのフェーズで構成される：
1. **collect.py / collect_v2.py：** Genesis シミュレータと PD コントローラによる自動データ収集
2. **train.py：** OpenVLA 7B の LoRA ファインチューニング
3. **infer.py：** 学習済みモデルによる自律飛行推論

### 3.2 シミュレーション環境（Genesis）

Genesis [9]（Apache 2.0）を物理エンジンとして採用する。シミュレーション設定を以下に示す。

| パラメータ | 値 |
|-----------|---|
| シミュレーション周波数 | 100 Hz（dt = 0.01s, substeps = 4） |
| 制御・記録周波数 | 10 Hz（10ステップに1回） |
| 重力加速度 | -9.81 m/s² |
| ドローンモデル | Crazyflie 2.0（cf2x.urdf） |
| 重力補償 | あり（gravity_compensation = 1.0） |
| FPV カメラ解像度 | 224 × 224（JPEG quality 95） |

**環境オブジェクト（Poly Haven, CC0）：**
- modern_arm_chair_01（アームチェア）
- sofa_02（ソファ）
- vintage_wooden_drawer_01（木製引き出し）

各エピソードで3つのオブジェクトを半径2〜4.5mのランダムな位置に配置し、ドローンはランダム位置からスポーンする。

### 3.3 自動データ収集パイプライン

**RaceVLAとの最大の差別化点：** 人間によるデモを一切必要とせず、PD コントローラが自動でグラウンドトゥルースデータを生成する。

**collect.py（基本版）：**
```
エピソード開始
  │
  ├─ オブジェクトをランダム配置（3つ、半径2〜4.5m）
  ├─ ターゲットをランダム選択
  ├─ 英語命令文を生成（4テンプレート）
  ├─ ドローンをランダム位置にスポーン
  │
  └─ ステップループ（最大300ステップ）
       │
       ├─ PD コントローラで速度計算
       ├─ ヨー方向をターゲットへ向ける
       ├─ 10Hzで FPV 画像 + アクションを記録
       └─ 1.5m以内到達 → ホバー → 終了
```

**collect_v2.py（アプローチ+オービット版）：**
```
エピソード開始
  │
  ├─ オブジェクトを1つランダム選択・配置
  ├─ 英語命令文を生成（4テンプレート）
  ├─ ドローンをランダム位置にスポーン
  │
  └─ ステップループ（最大1000ステップ）
       │
       ├─ [アプローチフェーズ] PD コントローラで接近
       │    └─ 1.5m以内到達 → オービットフェーズへ移行
       ├─ [オービットフェーズ] ターゲット周囲を円軌道で一周
       │    ─ カメラが常にオブジェクトを向く
       │    └─ 360°完了 → ホバーフェーズへ移行
       └─ [ホバーフェーズ] ホバー数ステップ → 成功終了
```

**PDコントローラ（Proportional-Derivative）：**
```
vel = Kp × error + Kd × d(error)/dt
    = 1.2 × error + 0.4 × d(error)/dt
最大速度: 1.5 m/s
```

**英語命令テンプレート（4種）：**
- `"Approach the {name}, fly around it, and take photos."`
- `"Go close to the {name} and circle around it to observe."`
- `"Navigate to the {name}, then orbit it once while recording."`
- `"Fly toward the {name} and do a full loop around it."`

### 3.4 ボディフレームアクション表現

**本研究の核となる設計決定：** アクションをワールド座標系ではなく機体座標系で記録する。

```python
# ワールド座標系 → 機体座標系への変換
yaw_rot_inv = Rotation.from_euler("z", -current_yaw)
vel_body = yaw_rot_inv.apply([vx_world, vy_world, vz_world])

action_4d = [
    vel_body[0],   # vx_body: 機首方向 [m/s]
    vel_body[1],   # vy_body: 機体左方向 [m/s]
    vel_body[2],   # vz_body: 上方向 [m/s]
    yaw_rate,      # ヨー角速度 [rad/s]
]
```

ボディフレーム表現の利点：
- **フレーム不変性：** 同じ行動（「前進する」）が、ドローンの向きに関わらず同じアクションベクトルで表現される
- **汎化性向上：** 訓練時と異なる向きからタスクを開始しても、同様の行動が期待できる
- **直感的な表現：** 機体の視点から見た「前進・横移動・上昇・回転」の4自由度

### 3.5 アクショントークナイザ（256ビン量子化）

OpenVLA の言語生成機構を活用するため、連続値アクションを離散トークンに変換する。

```
連続値アクション → 256ビン量子化 → テキストトークン列

例: [0.1, -0.2, 0.5, 0.0]
         ↓ 正規化（データセットの min/max 使用）
    [0.57, 0.43, 0.71, 0.50]
         ↓ 256ビンへマッピング
    → "145 109 182 127"（スペース区切り3桁整数）
```

- ビン境界はデータセット全体の min/max から自動計算
- 統計は `action_stats.npz` に保存し推論時に再利用
- 値域が狭い次元（分散≈0）はゼロ除算を回避するため range=1.0 にフォールバック

**OpenVLA 互換性のための 7次元拡張：**
```
[vx_body, vy_body, vz_body, yaw_rate, 0, 0, 0]
 ←────── 4次元 ──────────→ ←── ゼロ埋め ──→
```

### 3.6 LoRA ファインチューニング

OpenVLA 7B に対して LoRA アダプターを学習する。

| ハイパーパラメータ | 値 |
|-----------------|---|
| LoRA ランク | 32 |
| LoRA アルファ | 32 |
| LoRA ドロップアウト | 0.05 |
| ターゲットモジュール | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| 精度 | bfloat16（H100 最適化） |
| 学習率 | 5e-4 |
| スケジューラ | Cosine Annealing |
| 学習エポック数 | 15 |
| バッチサイズ | 16 |
| 勾配クリッピング | 1.0 |

**ロス計算（アクショントークンのみ）：**
```
入力テキスト: [命令文 + 画像トークン + アクショントークン]
ラベル:        [-100  + -100        + アクショントークン]
                ↑ マスク（ロスに寄与しない）
```

命令文・画像トークンをマスクすることで、**アクション生成のみ**を学習目標とし、命令理解との競合を防ぐ。

### 3.7 推論パイプライン

```
FPV 画像（224×224）
    + 英語命令（例: "Approach the sofa, fly around it, and take photos."）
    ↓
OpenVLA 7B + LoRA アダプター
    ↓
アクショントークン文字列（"145 109 182 127"）
    ↓
ActionTokenizer.decode()
    ↓
4次元ボディフレーム速度
    ↓
Genesis シミュレータへ適用（10Hz）
```

- **推論間隔：** 10Hz（100Hz シミュレータに対して10ステップに1回）
- **アクション保持：** 非推論ステップでは直前のアクションを継続適用
- **デュアルカメラ：** 推論用 FPV（224px）と動画記録用（640px）を分離

---

## 4. 実験

> **注：** 以下は実験計画。実験結果は実行後に記入する。

### 4.1 データ収集

- 収集エピソード数：5,000
- 最大ステップ数：300 steps/エピソード
- 目標成功率：\[実験後に記入\]
- 総ステップ数：\[実験後に記入\]

### 4.2 評価指標

| 指標 | 説明 |
|------|------|
| **成功率 (SR)** | 1.5m 以内にターゲットへ到達したエピソードの割合 |
| **到達時間** | 成功エピソードの平均ステップ数 |
| **軌跡効率** | 実際の経路長 / 最短距離（1に近いほど効率的） |
| **Val Loss** | 学習時のバリデーション損失 |

### 4.3 比較実験（計画）

| 実験 | 条件 | 目的 |
|------|------|------|
| Baseline | OpenVLA（ファインチューニングなし） | 事前学習モデルの性能下限 |
| **提案手法** | OpenVLA + LoRA + ボディフレーム | 本研究 |
| Ablation 1 | ワールドフレームアクション | ボディフレームの効果検証 |
| Ablation 2 | collect.py vs collect_v2.py | オービットフェーズ追加の効果検証 |

### 4.4 実験結果

| 手法 | 成功率 | 到達時間（steps） | 軌跡効率 |
|------|--------|----------------|---------|
| Baseline（事前学習のみ） | \[TBD\] | \[TBD\] | \[TBD\] |
| 提案手法（英語・ボディフレーム） | \[TBD\] | \[TBD\] | \[TBD\] |
| Ablation: ワールドフレーム | \[TBD\] | \[TBD\] | \[TBD\] |
| Ablation: collect.py（接近のみ） | \[TBD\] | \[TBD\] | \[TBD\] |

---

## 5. 考察

### 5.1 ボディフレーム表現の有効性
アブレーション実験（4.3）の結果を受けて記入予定。

仮説：ボディフレーム表現はドローンの向きに依存しない汎化的な行動表現を提供するため、特にランダムな初期姿勢での成功率向上に寄与する。

### 5.2 アプローチ+オービットタスクの有効性
- collect_v2.py による2フェーズタスク（接近→周回）は、単純接近のみ（collect.py）と比べてデータの多様性が増す
- オービット中のカメラはオブジェクトを常に向くため、物体の外観を多角度から学習できる
- Ablation 2（collect.py vs collect_v2.py）の結果を踏まえた分析を記入予定

### 5.3 PID自動収集パイプラインの限界
- PD コントローラによる軌跡は最適でない場合がある（直線的すぎる等）
- 実環境との Sim-to-Real ギャップが存在する
- 今後の課題：強化学習や模倣学習との組み合わせ

---

## 6. 結論

本研究では、Genesis シミュレータを用いた自動データ収集パイプライン（collect.py / collect_v2.py）と、ボディフレームアクション表現を組み合わせた OpenVLA ベースのドローンナビゲーション手法を提案した。英語自然言語指示による室内ドローンの物体接近・周回ナビゲーションを実証し、VLA モデルのロボットアーム操作以外への応用可能性を示した。

今後の課題として、実機ドローンへの Sim-to-Real 転移、より複雑な環境への適用、および計算効率の改善（オンボード推論の実現）が挙げられる。

---

## 参考文献

1. Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model," arXiv:2406.09246, CoRL 2024.
2. Open X-Embodiment Collaboration, "Open X-Embodiment: Robotic Learning Datasets and RT-X Models," ICRA 2024.
3. "OpenVLA-OFT: Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success," 2025.
4. Pertsch et al., "RaceVLA: VLA-based Racing Drone Navigation with Human-like Behaviour," arXiv:2503.02572, 2025.
5. "AutoFly: Vision-Language-Action Model for UAV Autonomous Navigation in the Wild," arXiv:2602.09657, 2026.
6. "UAV-VLA: Vision-Language-Action System for Large Scale Aerial Mission Generation," arXiv:2501.05014, HRI 2025.
7. "VLA-AN: An Efficient and Onboard VLA Framework for Aerial Navigation in Complex Environments," arXiv:2512.15258, 2024.
8. Wang et al., "VQ-VLA: Improving Vision-Language-Action Models via Scaling Vector-Quantized Action Tokenizers," ICCV 2025.
9. Genesis-Embodied-AI, "Genesis: A Generative and Universal Physics Engine for Robotics and Embodied AI Learning," GitHub, Apache 2.0.

---

## 付録：実装詳細

### A. ファイル構成

```
openvla-sim/
├── scripts/
│   ├── collect.py           # データ収集（接近のみ、3オブジェクト同時配置）
│   ├── collect_v2.py        # データ収集（アプローチ+オービット、1オブジェクト配置）
│   ├── train.py             # OpenVLA LoRA ファインチューニング
│   ├── infer.py             # 推論・自律飛行
│   ├── action_tokenizer.py  # 256ビン量子化
│   └── convert_to_rlds.py   # RLDS/TFRecord 変換
├── train_slurm.sh           # Slurmジョブスクリプト（H100, 15エポック）
├── objects/                 # 3Dオブジェクト（Poly Haven, CC0）
└── third_party/Genesis/     # 物理シミュレータ
```

### B. 実行環境

| 項目 | 仕様 |
|------|------|
| GPU | NVIDIA H100 (VRAM 80GB) |
| OS | Ubuntu 22.04 |
| Python | 3.10+ |
| PyTorch | 2.x (CUDA 12.1) |
| 主要ライブラリ | transformers 4.44.0, peft, accelerate |

### C. 使用データ・モデルのライセンス

| リソース | ライセンス |
|---------|----------|
| OpenVLA 7B（HuggingFace） | MIT License |
| Genesis シミュレータ | Apache 2.0 |
| 3Dオブジェクト（Poly Haven） | CC0 1.0 |
