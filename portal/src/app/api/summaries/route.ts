import { NextResponse } from "next/server";
import { fetchCommits, fetchPulls, fetchIssues, fetchMilestones } from "@/lib/github";
import type { AISummary } from "@/lib/types";

let cache: { summary: AISummary; timestamp: number } | null = null;
const CACHE_TTL = 3600 * 1000; // 1 hour

async function generateSummary(): Promise<AISummary> {
  const oneWeekAgo = new Date(
    Date.now() - 7 * 24 * 60 * 60 * 1000
  ).toISOString();

  const [commits, pulls, issues, milestones] = await Promise.all([
    fetchCommits({ since: oneWeekAgo, per_page: 100 }),
    fetchPulls({ state: "all" }),
    fetchIssues({ state: "all" }),
    fetchMilestones(),
  ]);

  const recentPulls = pulls.filter(
    (p) => new Date(p.updated_at) >= new Date(oneWeekAgo)
  );
  const recentIssues = issues.filter(
    (i) => new Date(i.updated_at) >= new Date(oneWeekAgo)
  );
  const openIssues = issues.filter((i) => i.state === "open");
  const closedThisWeek = recentIssues.filter((i) => i.state === "closed");
  const mergedThisWeek = recentPulls.filter((p) => p.merged_at);

  const context = [
    `# プロジェクト概況`,
    `- 総Issue数: ${issues.length}件 (Open: ${openIssues.length} / Closed: ${issues.length - openIssues.length})`,
    `- 今週のコミット数: ${commits.length}件`,
    `- 今週クローズしたIssue: ${closedThisWeek.length}件`,
    `- 今週マージしたPR: ${mergedThisWeek.length}件`,
    "",
    `# マイルストーン`,
    ...milestones.map((m) =>
      `- ${m.title}: ${m.closed_issues}/${m.open_issues + m.closed_issues}完了 (${m.state})`
    ),
    "",
    `# 直近1週間のコミット (${commits.length}件)`,
    ...commits.slice(0, 40).map((c) =>
      `- [${c.commit.author?.date?.slice(0, 10) ?? ""}] ${c.commit.message.split("\n")[0]} (by ${c.commit.author?.name ?? "unknown"})`
    ),
    "",
    `# マージされたPR (${mergedThisWeek.length}件)`,
    ...mergedThisWeek.map((p) =>
      `- #${p.number} ${p.title} (by ${p.user?.login ?? "unknown"})`
    ),
    "",
    `# 今週更新されたIssue (${recentIssues.length}件)`,
    ...recentIssues.map((i) => {
      const labels = i.labels.map((l) => typeof l === "string" ? l : l.name).join(", ");
      return `- [${i.state.toUpperCase()}] #${i.number} ${i.title}${labels ? ` [${labels}]` : ""}`;
    }),
    "",
    `# 現在OpenのIssue (${openIssues.length}件)`,
    ...openIssues.slice(0, 20).map((i) => {
      const labels = i.labels.map((l) => typeof l === "string" ? l : l.name).join(", ");
      const assignees = i.assignees?.map((a) => a.login).join(", ") ?? "";
      return `- #${i.number} ${i.title}${labels ? ` [${labels}]` : ""}${assignees ? ` → ${assignees}` : ""}`;
    }),
  ].join("\n");

  let summary: string;

  try {
    const { BedrockRuntimeClient, InvokeModelCommand } = await import(
      "@aws-sdk/client-bedrock-runtime"
    );
    const client = new BedrockRuntimeClient({
      region: process.env.BEDROCK_REGION || process.env.AWS_REGION || "us-east-1",
    });

    const prompt = `あなたは優秀なプロジェクトマネージャーです。以下のGitHub活動ログを分析し、ビジネスメンバー（非エンジニア）向けの**詳細な週次レポート**をMarkdownで作成してください。

## 出力フォーマット（必ずこの構成に従ってください）

### 📊 今週のハイライト
> 1〜2文で今週の最も重要な進展をまとめる引用ブロック

### 📈 数値サマリー
| 指標 | 今週 | 備考 |
|------|------|------|
| コミット数 | XX件 | ... |
| マージされたPR | XX件 | ... |
| クローズしたIssue | XX件 | ... |
| 現在のOpen Issue | XX件 | ... |

### ✅ 今週の主な成果
各成果を **太字タイトル** + 説明の形式で箇条書き。技術的な内容はビジネス価値に翻訳して記述。

### 🔄 進行中の作業
現在取り組んでいるタスクと進捗状況。担当者がわかる場合は記載。

### 🎯 マイルストーン進捗
各マイルストーンの進捗状況をプログレス表現で記述。

### ⚠️ 注目ポイント・リスク
遅延リスクやブロッカーがあれば記載。なければポジティブな注目点を記載。

### 📅 来週の見通し
今週の動向から予測される来週の重点項目。

---

## ルール
- 技術用語（API, GPU, VLM等）は括弧書きで簡単な説明を付ける
- 具体的な数値やIssue番号を積極的に引用する
- ビジネスインパクトを中心に記述する
- Markdownの見出し・表・太字・引用・箇条書きをフル活用して視認性を高める

## 活動ログ
${context}`;

    const response = await client.send(
      new InvokeModelCommand({
        modelId: process.env.BEDROCK_MODEL_ID || "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        contentType: "application/json",
        accept: "application/json",
        body: JSON.stringify({
          anthropic_version: "bedrock-2023-05-31",
          max_tokens: 4096,
          messages: [{ role: "user", content: prompt }],
        }),
      })
    );

    const result = JSON.parse(new TextDecoder().decode(response.body));
    summary = result.content[0].text;
  } catch {
    const closedCount = closedThisWeek.length;
    const mergedCount = mergedThisWeek.length;
    summary = [
      "## 今週の開発サマリー",
      "",
      "### 主な成果",
      `- ${commits.length}件のコミットが行われました`,
      `- ${mergedCount}件のPRがマージされました`,
      `- ${closedCount}件のIssueがクローズされました`,
      "",
      "### コミット概要",
      ...commits
        .slice(0, 10)
        .map((c) => `- ${c.commit.message.split("\n")[0]}`),
      "",
      "*注: AI要約はBedrock接続時に利用可能です。現在はコミットログのサマリーを表示しています。*",
    ].join("\n");
  }

  return {
    summary,
    generatedAt: new Date().toISOString(),
    periodStart: oneWeekAgo,
    periodEnd: new Date().toISOString(),
  };
}

export async function GET() {
  try {
    if (cache && Date.now() - cache.timestamp < CACHE_TTL) {
      return NextResponse.json(cache.summary, {
        headers: { "Cache-Control": "s-maxage=3600, stale-while-revalidate" },
      });
    }

    const summary = await generateSummary();
    cache = { summary, timestamp: Date.now() };

    return NextResponse.json(summary, {
      headers: { "Cache-Control": "s-maxage=3600, stale-while-revalidate" },
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}
