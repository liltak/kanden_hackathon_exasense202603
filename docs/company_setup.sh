#!/usr/bin/env bash
# =============================================================================
# ExaSense - 会社端末セットアップスクリプト
#
# 会社端末（exwzd-aite org にアクセスできる環境）で実行してください。
# 個人リポ(liltak) ↔ 会社リポ(exwzd-aite) の双方向同期を設定します。
#
# 前提:
#   - gh CLI がインストール済み
#   - gh auth login で exwzd-aite org にアクセスできるアカウントでログイン済み
#   - git がインストール済み
# =============================================================================
set -euo pipefail

echo "=========================================="
echo "  ExaSense 会社リポ セットアップ"
echo "=========================================="

# ----- Step 1: 会社org に新規リポ作成 -----
echo ""
echo "[1/5] exwzd-aite org にリポジトリを作成..."
gh repo create exwzd-aite/kanden_hackathon_exasense --private --confirm 2>/dev/null \
  && echo "  → 作成完了" \
  || echo "  → 既に存在するかエラー（続行します）"

# ----- Step 2: 個人リポをclone -----
echo ""
echo "[2/5] 個人リポをclone..."
WORK_DIR="${HOME}/Development/work/kanden-hackathon"
if [ -d "$WORK_DIR" ]; then
  echo "  → $WORK_DIR は既に存在します。スキップ"
else
  git clone https://github.com/liltak/kanden_hackathon_exasense.git "$WORK_DIR"
  echo "  → clone完了: $WORK_DIR"
fi
cd "$WORK_DIR"

# ----- Step 3: company リモート追加 + push -----
echo ""
echo "[3/5] company リモートを追加して push..."
if git remote get-url company &>/dev/null; then
  echo "  → company リモートは既に存在します"
else
  git remote add company https://github.com/exwzd-aite/kanden_hackathon_exasense.git
  echo "  → company リモート追加完了"
fi
git push company main
echo "  → main ブランチを company に push 完了"

# ----- Step 4: Personal Access Token 作成 -----
echo ""
echo "[4/5] SYNC_PAT (Personal Access Token) の作成..."
echo ""
echo "  以下のコマンドで PAT を作成してください（手動）:"
echo ""
echo "    gh auth token"
echo ""
echo "  または GitHub Web UI で作成:"
echo "    Settings → Developer settings → Personal access tokens → Tokens (classic)"
echo "    スコープ: repo（private repo read/write）"
echo ""
read -rp "  作成した PAT を貼り付けてください: " SYNC_PAT

if [ -z "$SYNC_PAT" ]; then
  echo "  → PAT が空です。Step 5 をスキップします。"
  echo "  → 後で手動で設定してください:"
  echo "    gh secret set SYNC_PAT -R liltak/kanden_hackathon_exasense"
  echo "    gh secret set SYNC_PAT -R exwzd-aite/kanden_hackathon_exasense"
  exit 0
fi

# ----- Step 5: 両リポに SYNC_PAT シークレット登録 -----
echo ""
echo "[5/5] 両リポに SYNC_PAT シークレットを登録..."
echo "$SYNC_PAT" | gh secret set SYNC_PAT -R liltak/kanden_hackathon_exasense --body -
echo "  → liltak/kanden_hackathon_exasense に設定完了"
echo "$SYNC_PAT" | gh secret set SYNC_PAT -R exwzd-aite/kanden_hackathon_exasense --body -
echo "  → exwzd-aite/kanden_hackathon_exasense に設定完了"

echo ""
echo "=========================================="
echo "  セットアップ完了!"
echo "=========================================="
echo ""
echo "同期スケジュール:"
echo "  金曜深夜 (土 3:00 JST): 会社 → 個人 (自動)"
echo "  日曜深夜 (月 3:00 JST): 個人 → 会社 (自動)"
echo ""
echo "手動同期:"
echo "  git fetch company && git merge company/main && git push origin main"
echo "  git fetch origin && git merge origin/main && git push company main"
echo ""
echo "手動ワークフロー実行:"
echo "  gh workflow run sync-from-company.yml -R liltak/kanden_hackathon_exasense"
echo "  gh workflow run sync-to-company.yml -R liltak/kanden_hackathon_exasense"
echo ""
