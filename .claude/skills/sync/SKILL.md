---
name: sync
description: This skill should be used when the user says "/sync", "同期", "両方にpush", "会社にもpush", "companyにpush", or wants to push to both personal and company GitHub repositories.
user_invocable: true
---

# Git Sync: 個人リポ + 会社リポに同期（コード + Issue）

個人リポ (origin) と会社リポ (company) の両方に現在のブランチをpushし、Issue を双方向同期します。

## リポジトリ情報

- **個人 (origin)**: `liltak/kanden_hackathon_exasense`
  - URL: `https://github.com/liltak/kanden_hackathon_exasense.git`
- **会社 (company)**: `exwzd-aite/internal_kanden_spatialhackathon2026March`
  - URL: `https://github.com/exwzd-aite/internal_kanden_spatialhackathon2026March.git`

## 認証情報

- `gh` の active account は会社（`takuya-kato_exwzd`）→ 会社リポはそのまま `gh api` で操作可能
- 個人リポへの API アクセスには認証ヘッダーが必要:
  ```bash
  PERSONAL_TOKEN=$(gh auth token --user liltak)
  # 個人リポの API 呼び出し時に付与:
  --header "Authorization: token $PERSONAL_TOKEN"
  ```

## 実行手順

### Step 1: コード同期（既存）

1. `company` リモートが未設定なら追加する:
   ```bash
   git remote add company https://github.com/exwzd-aite/internal_kanden_spatialhackathon2026March.git
   ```

2. 未コミットの変更がないか `git status` で確認し、あればユーザーに報告する（勝手にコミットしない）

3. 両方にpush:
   ```bash
   git push origin main
   git push company main
   ```

### Step 2: Issue 双方向同期

#### 2a. 両リポの open issue を取得

```bash
# 個人リポの認証トークン取得
PERSONAL_TOKEN=$(gh auth token --user liltak)

# 個人リポの open issue 一覧（最大100件）
gh api "repos/liltak/kanden_hackathon_exasense/issues?state=open&per_page=100" \
  --header "Authorization: token $PERSONAL_TOKEN"

# 会社リポの open issue 一覧（最大100件）
gh api "repos/exwzd-aite/internal_kanden_spatialhackathon2026March/issues?state=open&per_page=100"
```

#### 2b. 同期済み判定（マーカー方式）

各 issue の `body` に以下のマーカーがあれば「同期済み」と判定する:

- 会社リポの issue body 内: `<!-- synced-from: liltak/kanden_hackathon_exasense#<番号> -->`
- 個人リポの issue body 内: `<!-- synced-from: exwzd-aite/internal_kanden_spatialhackathon2026March#<番号> -->`

**同期済みセットの構築手順:**

1. 会社リポの全 open issue の body を走査し、`<!-- synced-from: liltak/kanden_hackathon_exasense#(\d+) -->` にマッチする番号を収集 → 「個人リポで同期済みの issue 番号」セット
2. 個人リポの全 open issue の body を走査し、`<!-- synced-from: exwzd-aite/internal_kanden_spatialhackathon2026March#(\d+) -->` にマッチする番号を収集 → 「会社リポで同期済みの issue 番号」セット

#### 2c. 未同期の issue を相手リポに作成

**個人 → 会社に同期（個人リポの未同期 issue を会社リポに作成）:**

```bash
gh api repos/exwzd-aite/internal_kanden_spatialhackathon2026March/issues \
  --method POST \
  -f title="<元のタイトル>" \
  -f body="<元の本文>

<!-- synced-from: liltak/kanden_hackathon_exasense#<元のissue番号> -->"
```

**会社 → 個人に同期（会社リポの未同期 issue を個人リポに作成）:**

```bash
gh api repos/liltak/kanden_hackathon_exasense/issues \
  --method POST \
  --header "Authorization: token $PERSONAL_TOKEN" \
  -f title="<元のタイトル>" \
  -f body="<元の本文>

<!-- synced-from: exwzd-aite/internal_kanden_spatialhackathon2026March#<元のissue番号> -->"
```

#### 2d. 同期結果を報告

作成した issue の数とリンクを一覧で報告する。例:

```
## Issue 同期結果
- 個人 → 会社: 2件作成
  - #55 "太陽光パネル配置最適化" → company#12
  - #56 "3Dメッシュ品質改善" → company#13
- 会社 → 個人: 1件作成
  - #11 "API認証の修正" → origin#57
- 既に同期済み: 3件（スキップ）
```

### Step 3: 結果報告

コード push の結果と Issue 同期の結果をまとめて報告する。

## 注意事項

- コミットされていない変更がある場合、pushの前にユーザーに確認する
- push先のブランチは現在のブランチ（通常 main）
- エラーが出た場合は原因を報告して対処法を提示する
- Issue 同期は body 内のマーカーコメントで重複を防止する。再実行しても同じ issue が二重作成されることはない
- Pull Request は issue 一覧に含まれる場合があるが、`pull_request` キーを持つものはスキップすること
- issue の body が null の場合は空文字として扱う
