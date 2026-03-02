---
name: sync
description: This skill should be used when the user says "/sync", "同期", "両方にpush", "会社にもpush", "companyにpush", or wants to push to both personal and company GitHub repositories.
user_invocable: true
---

# Git Sync: 個人リポ + 会社リポに同期

個人リポ (origin) と会社リポ (company) の両方に現在のブランチをpushします。

## リポジトリ情報

- **個人 (origin)**: `https://github.com/liltak/kanden_hackathon_exasense.git`
- **会社 (company)**: `https://github.com/exwzd-aite/kanden_hackathon_exasense.git`

## 実行手順

1. `company` リモートが未設定なら追加する:
   ```bash
   git remote add company https://github.com/exwzd-aite/kanden_hackathon_exasense.git
   ```

2. 未コミットの変更がないか `git status` で確認し、あればユーザーに報告する（勝手にコミットしない）

3. 両方にpush:
   ```bash
   git push origin main
   git push company main
   ```

4. 結果を報告する

## 注意事項

- コミットされていない変更がある場合、pushの前にユーザーに確認する
- push先のブランチは現在のブランチ（通常 main）
- エラーが出た場合は原因を報告して対処法を提示する
