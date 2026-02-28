---
name: hackathon-readme
description: This skill should be used when the user asks to "README作成", "GitHub README", "HuggingFace README", "Model Card", "Dataset Card", "提出物のREADME", or needs to create a README that complies with the kanden hackathon submission requirements.
---

# ハッカソン提出用README生成

関西電力×VOLTMINDハッカソンの提出ルールに準拠したREADMEを生成します。

## GitHub README 必須項目

- 実行環境（OS、Python、CUDA等）
- 実行手順
- 使用データ（出典・ライセンス明記）
- 使用モデル（名称・ライセンス・利用規約URL明記）
- ライセンス：MIT License または Apache License 2.0

## HuggingFace Model Card 必須項目

- ベースモデル、使用データ、学習パラメータ、実行手順、ライセンス

## HuggingFace Dataset Card 必須項目

- 元データセット、合成に使用したモデル、データセットの特徴、ライセンス

## GitHub README テンプレート

```markdown
# プロジェクト名

## 概要
## セットアップ
### 実行環境
### インストール
## 使用方法
## 使用データ
| データセット | 用途 | ライセンス | URL |
|---|---|---|---|
## 使用モデル
| モデル名 | 用途 | ライセンス | 利用規約URL |
|---|---|---|---|
## 結果
## ライセンス
MIT License
```

不明な箇所は `TODO:` と記載する。提出期限: 2026/3/28(土) 12:00
