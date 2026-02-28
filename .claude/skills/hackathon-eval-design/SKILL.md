---
name: hackathon-eval-design
description: This skill should be used when the user asks to "評価指標", "evaluation metrics", "ベンチマーク設計", "benchmark", "評価方法", "リーダーボード", "leaderboard", "空間理解の評価", or needs to design evaluation methods for their spatial understanding AI system.
---

# 空間理解タスクの評価設計

ハッカソンのテーマ「空間を理解する」に適した評価指標・ベンチマーク設計を提案します。

## タスク別の定量評価指標

### 深度推定
- AbsRel, SqRel, RMSE, RMSElog
- delta < 1.25 閾値精度

### 3D物体検出
- mAP@IoU（3D IoU）, NDS

### セマンティックセグメンテーション（3D）
- mIoU

### 点群処理
- Chamfer Distance, Earth Mover's Distance

### VQA・空間的質問応答
- Exact Match, BLEU, CIDEr

## データセット使用時の注意

公開データは商用利用可ライセンスのみ使用可能。CC BY-NC-SAなど非商用ライセンスは使用不可。ライセンス不明確なデータも使用禁止。

## デモ向け定性評価

「空間を理解している」と審査員に伝わる可視化：
- 深度マップの色分け表示
- 3Dポイントクラウドのインタラクティブ表示
- セグメンテーション結果のオーバーレイ
- ビデオデモ

タスク内容を確認し、適切な評価指標・データセット・実装方法を提案する。
