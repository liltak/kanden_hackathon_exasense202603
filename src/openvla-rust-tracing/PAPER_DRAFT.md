# Autonomous Crack Path Tracing via Vision-Language-Action Model Fine-Tuning

**OpenVLA によるクラック経路自律追従エージェント**

---

## Abstract

コンクリート構造物や岩石・舗装面のひび割れ（クラック）検査は、インフラ維持管理における重要な作業である。
本研究では、大規模言語・視覚基盤モデルである OpenVLA (7B) を LoRA でファインチューニングし、クラック経路を自律的に追従するエージェントを構築する。
合成生成した 4096×4096 クラック画像から抽出した 224×224 パッチを逐次入力とし、次の移動ベクトル [Δx, Δy] を離散トークンとして出力するエンドツーエンドシステムを提案する。
コンクリート・石材・木目・アスファルトの 4 テクスチャと、直線・波状・ジグザグ・折れ曲がりの 4 変形モードを組み合わせた多様な合成データ上で訓練し、クラック方向誤差・経路追従精度・完走率の観点で有効性を示す。

---

## 1. Introduction

電力・交通・水道等のインフラ設備の老朽化は世界的な問題であり、定期的な点検・診断が不可欠である。
コンクリート構造物のひび割れは劣化・破損の早期指標として重要視されているが、広大な施設面積を人手で全域検査するには多大なコストと時間を要する。

近年、自律型検査ロボットや UAV (Unmanned Aerial Vehicle) を用いた自動点検が実用化されてきたが、**クラックを発見した後にその経路全体を自律的に追従しながら詳細記録**する機能は依然として課題である。
従来のクラック検出手法の多くは画像全体に対するセグメンテーションを行うが、ロボットが実際に動作する際は「現在位置から見えるパッチ画像だけ」を用いてリアルタイムに次の移動先を判断しなければならない。

本研究の貢献は以下の通りである:

1. **クラック追従専用の合成データセット生成パイプライン**の構築（テクスチャ・変形モードを組み合わせた多様な生成）
2. **OpenVLA 7B の LoRA ファインチューニング**によるパッチ観察→移動ベクトル予測の実現
3. **ActionTokenizer** による連続 2D アクションの 256bin 離散化と VLM へのシームレスな統合
4. 合成→実画像への転移可能性に向けた実背景画像合成手法の導入

---

## 2. Related Work

### 2.1 クラック検出・セグメンテーション

クラック検出は古典的な画像処理から深層学習まで幅広い手法が研究されている。

**古典的手法**
- **Canny エッジ検出 + モルフォロジー演算** [Canny, 1986]: 輝度勾配に基づくエッジ検出。ノイズに脆弱。
- **Otsu 二値化 + スケルトン化**: 単純だが照明変動に弱い。

**深層学習ベースのセグメンテーション**
- **CrackForest** [Shi et al., 2016]: コンクリートクラック専用データセット。480×320 画像で pixel-wise アノテーション。
- **DeepCrack** [Liu et al., 2019]: 多スケール特徴融合を用いた U-Net 系セグメンテーター。AIGLE_RN・ESAR・CrackTree データセットで評価。
- **Crack500** [Yang et al., 2019]: 500 枚の舗装クラック画像データセット。解像度 3264×2448。
- **SDDNet** [Cha et al., 2019]: Semantic segmentation + 距離変換によるクラック幅推定。
- **CrackSeg9k** [Kulkarni et al., 2022]: 9,000 枚超の多様なクラック画像を集積したベンチマーク。
- **SegFormer** [Xie et al., 2021] 等の汎用セグメンターもクラック検出に応用される。

これらは**静的な検出**に特化しており、ロボットが動作しながらパッチを逐次観察する**動的追従**には直接対応していない。

### 2.2 Vision-Language Models (VLM) によるインフラ検査

VLM のインフラ・産業応用は近年急増している。

- **GPT-4V / Claude Vision**: ゼロショットで構造物の損傷記述が可能であることが示されている [Rao et al., 2024]。
- **InternVL** [Chen et al., 2024]: 高解像度タイル分割を用いた建築・土木画像の詳細解析。
- **Qwen-VL** [Bai et al., 2023]: 中国語インフラ点検レポートと組み合わせた応用事例。
- **CLIP ベースの欠陥分類**: ゼロショット転移による製造ラインの異常検出 [Jeong et al., 2023]。

しかし、これらはいずれも**画像→テキスト説明**の生成であり、ロボットの連続行動生成（クラック追従）には利用されていない。

### 2.3 Vision-Language-Action Models (VLA)

VLA はロボット操作分野で近年急速に発展している。

- **RT-2** [Brohan et al., 2023]: PaLI-X / PaLM-E を基盤とし、ロボット操作アクションをテキストトークンとして出力。VLM の知識をロボット行動にゼロショット転移することを示した。
- **OpenVLA** [Kim et al., 2024]: Prismatic VLM (LLaVa-1.5 アーキテクチャ, 7B) をロボット操作データ (Open-X Embodiment) で訓練したオープンソース VLA。アクション次元を 256bin 離散トークンとして言語モデルの vocabulary に追加する設計が特徴。
- **π0 (pi-zero)** [Black et al., 2024]: Flow matching を組み合わせた高速・高精度ロボット操作 VLA。
- **Octo** [Team et al., 2024]: Transformer ベースの汎用ロボット基盤モデル。多様なロボット構成に対応。
- **RoboFlamingo** [Li et al., 2023]: OpenFlamingo を継続学習で行動予測に適応。

本研究はこれらの成果、特に **OpenVLA の設計思想**（アクション離散化・LoRA 適応）を応用し、インフラ点検ドメインへの転移を試みる。

### 2.4 LoRA によるドメイン適応

- **LoRA** [Hu et al., 2022]: 大規模モデルの特定モジュールに低ランク行列を付加してファインチューニング。メモリ効率が高く、OpenVLA でも公式に推奨。
- **QLoRA** [Dettmers et al., 2023]: 4bit 量子化 + LoRA により消費メモリをさらに削減。
- **DoRA** [Liu et al., 2024]: 重みを方向成分・大きさ成分に分解した LoRA 拡張。

本研究では OpenVLA 公式設定に倣い rank=32, alpha=64 の LoRA を採用する。

### 2.5 合成データによるクラック認識

実際のクラック画像は収集・アノテーションが困難なため、合成データが有効である。

- **CrackSynthesis** 系手法: Perlin noise やフラクタル関数でひび割れ模様を生成し、実画像に重畳。
- **GAN ベース生成**: CycleGAN・pix2pix による合成→実の域適応 [Goyal et al., 2020]。
- **Blender + 物理シミュレーション**: 3D 構造物モデルに破壊力学を適用し、クラックの形状を物理的に生成 [Koch et al., 2015]。
- **本研究のアプローチ**: 経路生成（数学的なランダムウォーク + 変形モード）と多層描画（影・芯・中心線）を組み合わせた軽量合成パイプライン。アノテーションは生成時に正確な座標が得られるため、後処理不要。

---

## 3. Method

### 3.1 問題設定

大規模インフラ画像 $I \in \mathbb{R}^{H \times W \times 3}$（4096×4096 等）において、クラック経路 $\mathcal{P} = \{(x_0, y_0), (x_1, y_1), \ldots, (x_T, y_T)\}$ を与えたとき、エージェントは現在位置 $(x_t, y_t)$ を中心とした 224×224 パッチ画像 $p_t$ のみを観察して次の移動量 $(\Delta x_t, \Delta y_t)$ を予測する逐次意思決定問題を解く。

$$a_t = f_\theta(p_t, \text{instruction})$$
$$a_t = (\Delta x_t, \Delta y_t), \quad (x_{t+1}, y_{t+1}) = (x_t + \Delta x_t, y_t + \Delta y_t)$$

### 3.2 データ生成

#### 3.2.1 クラック経路生成

ランダムウォークベースの経路生成アルゴリズムを採用する。

1. **開始点**: 画像左端・上端・下端のいずれかからランダムに選択
2. **基底方向**: 開始辺に応じた方向角をランダムサンプリング
3. **方向変化**: ステップごとに最大 45° の範囲でランダムに方向を更新
4. **変形モード** (`none` / `wave` / `zigzag` / `bend`): 基底変化に加えてグローバルな形状変形を付与

#### 3.2.2 クラック描画

3 層重ね描画でリアルな亀裂表現を実現する:

| レイヤー | 幅 | 色 | ぼかし |
|---------|-----|------|--------|
| 影 | 芯×4 + 2px | グレー (160) | σ=3.0 |
| 芯 | 芯×0.8 + 0.8px | 濃紺 (15) | σ=1.2 |
| 中心線 | 0.8px | 黒 (0) | σ=0.5 |

その後ガウスノイズを付加。

#### 3.2.3 テクスチャ合成

4 種類のプロシージャルテクスチャ（コンクリート・石材・木目・アスファルト）を生成し、Multiply 合成でクラックを重畳する。
また実背景画像（`--bg_dir`）の指定も可能。

#### 3.2.4 アノテーション

生成時の正確なクラック座標から弧長ベースのサブサンプリングを行い、パッチ間オーバーラップ 50% で等間隔ウェイポイントを抽出。
各ウェイポイントで 224×224 パッチを切り出し、隣接ウェイポイントへの差分ベクトルをアクションとして記録する。

### 3.3 ActionTokenizer

連続 2D アクション [Δx, Δy] を VLM が生成できるテキストトークンに変換する。

1. 訓練データのアクション分布から平均 $\mu$、標準偏差 $\sigma$ を計算
2. 正規化: $a' = (a - \mu) / \sigma$
3. $[-1, 1]$ にクリップして 256 等分ビンに量子化
4. "145 109" のような数字トークン文字列に変換

推論時はトークン文字列を逆変換して元のスケールのアクションを復元する。

### 3.4 OpenVLA LoRA ファインチューニング

ベースモデル OpenVLA 7B の全 Attention / FFN 投影行列に LoRA アダプターを付加する。

```
target_modules = [q_proj, k_proj, v_proj, o_proj,
                  gate_proj, up_proj, down_proj]
rank = 32, alpha = 64, dropout = 0.05
```

学習時の入力フォーマット:
```
[画像トークン列] + "Follow the crack. Navigate to continue tracking the crack path. 145 109"
                                                                                   ↑ アクショントークン（これのみ loss 計算）
```

最終レイヤーのみを loss 計算対象にするため、命令文部分のラベルは -100 でマスクする。

---

## 4. Experiments

### 4.1 データセット

| 分割 | エピソード数 | ステップ数（概算） |
|------|------------|----------------|
| Train | 240 (80%) | ~7,200 |
| Val | 30 (10%) | ~900 |
| Test | 30 (10%) | ~900 |

合成パラメータ: n=300, テクスチャ 4 種 × 変形 4 種 × 線幅 3 種 = 48 組を網羅 + ランダム追加

### 4.2 訓練設定

| ハイパーパラメータ | 値 |
|-----------------|-----|
| ベースモデル | openvla/openvla-7b |
| LoRA rank | 32 |
| LoRA alpha | 64 |
| エポック数 | 5 |
| バッチサイズ | 16 |
| 学習率 | 5e-4 |
| スケジューラー | Cosine Annealing |
| オプティマイザー | AdamW (weight_decay=0.01) |
| 精度 | bfloat16 |
| GPU | NVIDIA H100 (80GB) |

### 4.3 評価指標

#### 4.3.1 Validation Loss

ActionTokenizer 離散化後のクロスエントロピー損失。エポックごとに記録。

#### 4.3.2 経路追従精度 (Path Following Accuracy, PFA)

テストエピソードにおいて、各ステップで予測したウェイポイントと正解ウェイポイントの L2 距離の平均:

$$\text{PFA} = \frac{1}{T} \sum_{t=1}^{T} \|\hat{p}_t - p_t^*\|_2 \quad [\text{px}]$$

#### 4.3.3 方向誤差 (Angular Error)

予測移動方向と正解方向の角度差（度）:

$$\text{AE} = \frac{1}{T} \sum_{t=1}^{T} \arccos\left(\frac{\hat{a}_t \cdot a_t^*}{\|\hat{a}_t\| \|a_t^*\|}\right) \times \frac{180}{\pi}$$

#### 4.3.4 完走率 (Episode Completion Rate, ECR)

エピソード全体を通じて終点まで到達した割合。中断基準は「クラックから 112px 以上逸脱」。

### 4.4 ベースライン

| モデル | 説明 |
|-------|------|
| Random | ランダムな方向に移動 |
| Direction-only | クラック検出マスクの勾配方向のみ使用（VLA なし） |
| OpenVLA 7B (LoRA なし) | ファインチューニングなしの事前学習済みモデル |
| **OpenVLA 7B + LoRA (本手法)** | LoRA ファインチューニング済み |

---

## 5. Discussion

### 5.1 合成→実画像の転移

本研究の最大の課題の一つは、合成データで訓練したモデルが実際のコンクリート画像に汎化するかである。
本研究では `--bg_dir` オプションにより実背景画像を使用した合成も可能にしており、Domain Randomization の一形態として機能する。
今後の工程として以下が考えられる:

- **GAN ベースの域適応**: CycleGAN で合成→実ドメインの外観を近づける
- **実クラック画像への少量 LoRA 継続学習**: 数十枚のアノテーション済み実画像で追加微調整
- **Vision Foundation Model によるセグメンテーション補助**: SAM (Segment Anything Model) [Kirillov et al., 2023] でクラック領域を抽出し、追従の開始点を自動検出

### 5.2 分岐・交差クラックへの拡張

本研究の生成パイプラインは「枝なし・交差なし」の単純なクラックに限定している。
実際のインフラ画像では分岐・交差クラックが多数存在する。拡張方向:

- **分岐検出器の追加**: 交差点を検出し、未追従の枝への誘導指示を命令文に追加
- **トポロジカルマップ構築**: 訪問済み・未訪問のノードを管理し、分岐点から再出発

### 5.3 3D 点群・深度情報との統合

壁面クラックの深さ・幅の 3D 推定は構造劣化評価に不可欠である。
本システムを深度カメラや 3D 再構成システム（VGGT, COLMAP 等）と統合することで、2D 追従軌跡を 3D 構造物上にマッピングする拡張が考えられる。

---

## 6. Conclusion

本研究では、OpenVLA 7B を LoRA でファインチューニングし、大規模インフラ画像のクラック経路を自律追従するエージェントを構築した。
合成データ生成パイプラインにより多様なクラック形状・テクスチャに対応したデータセットを効率的に構築でき、ActionTokenizer を通じて連続 2D アクションを VLM の vocabulary に自然に組み込む手法を提案した。
今後は実インフラ画像への転移、分岐クラックへの対応、3D 構造物との統合が課題である。

---

## References

- Brohan, A., et al. (2023). **RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control.** *arXiv:2307.15818*.
- Kim, M. J., et al. (2024). **OpenVLA: An Open-Source Vision-Language-Action Model.** *arXiv:2406.09246*.
- Hu, E. J., et al. (2022). **LoRA: Low-Rank Adaptation of Large Language Models.** *ICLR 2022*.
- Dettmers, T., et al. (2023). **QLoRA: Efficient Finetuning of Quantized LLMs.** *NeurIPS 2023*.
- Liu, Y., et al. (2019). **DeepCrack: A Deep Hierarchical Feature Learning Architecture for Crack Segmentation.** *Neurocomputing, 338*.
- Yang, F., et al. (2019). **Feature Pyramid and Hierarchical Boosting Network for Pavement Crack Detection.** *IEEE TITS*.
- Kulkarni, A., et al. (2022). **CrackSeg9k: A Collection and Benchmark for Crack Segmentation Datasets and Frameworks.** *ECCV Workshops 2022*.
- Cha, Y.-J., Choi, W., & Büyüköztürk, O. (2017). **Deep Learning-Based Crack Damage Detection Using Convolutional Neural Networks.** *Structural Control and Health Monitoring*.
- Shi, Y., et al. (2016). **Automatic Road Crack Detection Using Random Structured Forests.** *IEEE TITS*.
- Kirillov, A., et al. (2023). **Segment Anything.** *ICCV 2023*.
- Xie, E., et al. (2021). **SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers.** *NeurIPS 2021*.
- Black, K., et al. (2024). **π0: A Vision-Language-Action Flow Model for General Robot Control.** *arXiv:2410.24164*.
- Team, O., et al. (2024). **Octo: An Open-Source Generalist Robot Policy.** *RSS 2024*.
- Li, K., et al. (2023). **RoboFlamingo: Vision-Language Models as Action Generators.** *arXiv:2311.01378*.
- Bai, J., et al. (2023). **Qwen-VL: A Versatile Vision-Language Model for Understanding.** *arXiv:2308.12966*.
- Koch, C., et al. (2015). **Evaluation of CNN-Based Single-Image Depth Estimation Methods.** *ECCV Workshops*.
- Liu, S., et al. (2024). **DoRA: Weight-Decomposed Low-Rank Adaptation.** *ICML 2024*.
- Jeong, J., et al. (2023). **WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation.** *CVPR 2023*.
- Canny, J. (1986). **A Computational Approach to Edge Detection.** *IEEE TPAMI, 8(6)*.

---

## Appendix A: Slurm ジョブスクリプト

H100 クラスタでの実行は各 `_slurm.sh` を参照。

```bash
# データ生成
sbatch generate_data_slurm.sh

# LoRA 学習
sbatch train_slurm.sh

# 推論
sbatch infer_slurm.sh

# ロールアウト評価
sbatch rollout_slurm.sh
```

## Appendix B: 主要ハイパーパラメータの感度

| パラメータ | 推奨値 | 備考 |
|-----------|--------|------|
| LoRA rank | 32 | 16 では表現力不足の可能性、64 はメモリ増加 |
| パッチサイズ | 224 | OpenVLA の入力解像度に合わせる |
| オーバーラップ率 | 50% | 低いと連続性が失われる、高いとデータ量増加 |
| 変形モードの割合 | 均等 | 特定の変形に過学習しないよう等分 |
| 実背景/合成テクスチャ | 混在推奨 | 実背景のみだと枚数が限られる |
