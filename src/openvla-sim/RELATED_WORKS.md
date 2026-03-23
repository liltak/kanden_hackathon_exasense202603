# 関連研究まとめ：VLAモデルのドローン制御応用

論文執筆のための関連研究サーベイ（2025年3月時点）

---

## 概要

本プロジェクト（OpenVLA × Genesis × ドローンナビゲーション）に関連する先行研究を整理する。
特に「**OpenVLAをドローン制御に転用する**」というアプローチとの類似度を中心に分析する。

---

## 1. RaceVLA — 最も近い先行研究 ⚠️

**論文:** [RaceVLA: VLA-based Racing Drone Navigation with Human-like Behaviour](https://arxiv.org/abs/2503.02572)
**発表:** arXiv, 2025年3月4日

### 概要

OpenVLA（Stanford AI Lab製、7Bパラメータ）をレーシングドローンの飛行制御にファインチューニングした研究。
本プロジェクトと最もアーキテクチャが近い。

### 手法

- **ベースモデル:** OpenVLA 7B（LoRAファインチューニング）
- **入力:** FPVカメラ画像 + 自然言語指示
- **出力:** 4次元アクションベクトル `(vx, vy, vz, yaw_rate)`
  - OpenVLAの元々の7次元出力を4次元に変更（残り3次元はゼロパディング）
- **タスク:** 障害物ゲートを通過するレーシング飛行

### 実験結果

| 評価軸 | RaceVLA | OpenVLA（ベースライン） |
|--------|---------|----------------------|
| 動作汎化 | **75.0** | 60.0 |
| 意味汎化 | **45.5** | 36.3 |
| 視覚汎化 | 79.6 | **87.0** |
| 物理汎化 | 50.0 | **76.7** |

動的環境でのFPV入力の変動により、視覚・物理汎化が若干低下した。

### 本プロジェクトとの比較

| 項目 | RaceVLA | 本プロジェクト |
|------|---------|--------------|
| ベースモデル | OpenVLA 7B | OpenVLA 7B |
| アクション次元 | 4D（ゼロパディング） | 4D（ゼロパディング） |
| ファインチューニング | LoRA | LoRA |
| タスク | レーシング（ゲート通過） | **物体指示ナビゲーション** |
| データ収集 | 人間デモ（推測） | **PIDによる自動収集** |
| 座標系 | 不明 | **ボディフレーム変換（明示）** |
| シミュレータ | 独自 | **Genesis（最新OSS）** |
| 指示言語 | 英語 | **日本語** |

> **論文執筆上の注意:** 「OpenVLAをドローンに適用した最初の研究」という主張はRaceVLAの存在により不可能。
> 上記の差分（自動データ収集、ボディフレーム、日本語、物体ナビゲーション）を新規性として主張すること。

---

## 2. AutoFly — UAV向けVLAの最新研究

**論文:** [AutoFly: Vision-Language-Action Model for UAV Autonomous Navigation in the Wild](https://arxiv.org/abs/2602.09657)
**発表:** arXiv, 2026年2月（コードおよびデータ公開済み）

### 概要

屋外の未知環境でのUAV自律ナビゲーションを目的としたエンドツーエンドVLAモデル。
詳細な経路指示がない粗い目標（例: "北側の建物へ向かえ"）のみで自律飛行を実現。

### 主な技術的特徴

1. **疑似深度エンコーダ (Pseudo-Depth Encoder)**
   深度センサー不要。モノクロRGB入力から空間的表現を抽出し、空間推論を強化。

2. **2段階トレーニング戦略 (Progressive Two-Stage Training)**
   - Stage 1: 視覚・言語・深度表現のアライメント
   - Stage 2: アクションポリシーとの統合

3. **自律行動データセット構築**
   従来のInstruction Followingではなく、**継続的な障害物回避・自律計画・認識ワークフロー**を重視したデータセット設計。

### 本プロジェクトとの違い

- AutoFlyは**屋外・未知環境・障害物回避**が主眼
- 本プロジェクトは**室内・特定物体への接近**が主眼
- AutoFlyは独自アーキテクチャ、本プロジェクトはOpenVLAのファインチューニング

---

## 3. UAV-VLA — 大規模ミッション生成

**論文:** [UAV-VLA: Vision-Language-Action System for Large Scale Aerial Mission Generation](https://arxiv.org/abs/2501.05014)
**発表:** arXiv, 2025年1月 / ACM/IEEE HRI 2025

### 概要

衛星画像と自然言語の組み合わせにより、大規模な空中ミッションを自動生成するシステム。
例: "青いシートが敷かれた屋上パッドへ荷物を届けろ" → 経路計画 + 実行

### アーキテクチャ

- **グローバルコンテキスト:** 衛星・航空画像 + 自然言語でミッション解析
- **ローカル制御:** 飛行コントローラが正確なウェイポイントを実行
- **応用:** 物流、災害対応、偵察

### 本プロジェクトとの違い

UAV-VLAは**高レベルなタスク分解・経路計画**が中心であり、
本プロジェクトが扱う**低レベルな連続アクション生成**とは異なる。

---

## 4. VLA-AN — 効率的なオンボード推論

**論文:** [VLA-AN: An Efficient and Onboard VLA Framework for Aerial Navigation in Complex Environments](https://arxiv.org/abs/2512.15258)
**発表:** arXiv, 2024年12月

### 概要

複雑環境での空中ナビゲーションをオンボードデバイスで実行可能にすることを目標とした
計算効率重視のVLAフレームワーク。

### 主な特徴

- **オンボード推論に最適化:** 実機ドローンへのデプロイを想定した軽量設計
- **複雑環境対応:** 動的・複雑な環境での飛行ナビゲーション

### 本プロジェクトとの違い

本プロジェクトはH100（80GB VRAM）を前提とした研究プロトタイプであり、
オンボード軽量化は今後の課題として位置づけられる。

---

## 5. DroneVLA — 空中マニピュレーション

**論文:** [DroneVLA: VLA based Aerial Manipulation](https://arxiv.org/abs/2601.13809)
**発表:** arXiv, 2026年1月

### 概要

UAVのマニピュレーション（物体把持）をVLAで実現する研究。
VLAをナビゲーションから切り離し、**グリッパー制御**に特化させたモジュール構成。

### 主な特徴

- **責任分離アーキテクチャ:** VLAはグリッパー論理に集中、飛行安定制御は別モジュール
- **テキストプロンプトによる把持:** 物体名称をテキストで指定してグリッパー制御

### 本プロジェクトとの違い

DroneVLAはマニピュレーション（把持）が主眼。本プロジェクトはナビゲーション（接近）が主眼。

---

## 6. OpenVLA — ベースモデル

**論文:** [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246)
**発表:** arXiv, 2024年6月 / CoRL 2024

### 概要

970K件の実ロボットデモンストレーション（Open X-Embodiment）で事前学習された
7Bパラメータのオープンソース汎用ロボット操作VLAモデル。

### アーキテクチャ

```
入力: RGB画像 + 自然言語指示
  ↓
Vision Encoder (SigLIP) + LLM (Llama2 7B)
  ↓
出力: 256ビン離散トークンとしてアクション生成
```

### アクション表現

各アクション次元を256ビンに離散化し、各ビンIDをトークンとして言語モデルで生成。
→ 本プロジェクトの `ActionTokenizer` はこの手法を踏襲。

### LoRAファインチューニング

LoRAを使用することで全パラメータの1.4%のみ更新。
A100 80GB 1枚でのファインチューニングが可能。

---

## 7. 関連する技術トレンド

### アクショントークナイズの進化

| 手法 | 説明 | 代表論文 |
|------|------|---------|
| ビニング（256ビン） | 最も基本的。各次元を均等分割 | OpenVLA, RaceVLA, **本研究** |
| FAST (DCT + BPE) | 離散コサイン変換でトークン数を圧縮、推論15倍高速化 | OpenVLA-OFT |
| VQ-VAE | ベクトル量子化による高品質トークナイゼーション | VQ-VLA (ICCV 2025) |

本プロジェクトはビニング方式を採用しており、FASTやVQ-VAEとの比較が論文の実験として有効。

---

## 本研究の新規性まとめ（差別化ポイント）

RaceVLAの存在を踏まえた上で、以下を新規性として主張できる：

### 主張1: PIDによる完全自動データ収集パイプライン
- 人間デモを必要とせず、PDコントローラが自動でグラウンドトゥルース生成
- Genesis物理シミュレータ（Apache 2.0）との組み合わせで再現性が高い

### 主張2: ボディフレーム座標系でのアクション学習
- アクションを機体座標系で記録・学習することでフレーム不変性を実現
- ドローンの向きに依存しない汎化的な行動表現

### 主張3: 物体指示ナビゲーションのVLA定式化
- "ソファに近づけ" のような**物体名称**を自然言語指示として使用
- オブジェクト中心ナビゲーションの新しい定式化

### 主張4: 日本語自然言語指示による制御
- 英語に偏ったVLA研究に対し、日本語での制御を実証
- 多言語VLAの可能性を示す

---

## 参考文献

1. Pertsch et al., "RaceVLA: VLA-based Racing Drone Navigation with Human-like Behaviour," arXiv:2503.02572, 2025.
2. "AutoFly: Vision-Language-Action Model for UAV Autonomous Navigation in the Wild," arXiv:2602.09657, 2026.
3. "UAV-VLA: Vision-Language-Action System for Large Scale Aerial Mission Generation," arXiv:2501.05014, 2025.
4. "VLA-AN: An Efficient and Onboard VLA Framework for Aerial Navigation," arXiv:2512.15258, 2024.
5. "DroneVLA: VLA based Aerial Manipulation," arXiv:2601.13809, 2026.
6. Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model," arXiv:2406.09246, 2024.
7. Wang et al., "VQ-VLA: Improving Vision-Language-Action Models via Scaling Vector-Quantized Action Tokenizers," ICCV 2025.
8. "OpenVLA-OFT: Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success," 2025.
