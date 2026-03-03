"use client";

import { use } from "react";
import Link from "next/link";
import ReactMarkdown from "react-markdown";
import { ArrowLeft } from "lucide-react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Separator } from "@/components/ui/separator";
import { Button } from "@/components/ui/button";
import { LabelBadge } from "@/components/issues/label-badge";
import { CommentForm } from "@/components/issues/comment-form";
import { useIssue, useIssueComments } from "@/hooks/use-issues";
import { formatDate } from "@/lib/utils";

export default function IssueDetailPage({
  params,
}: {
  params: Promise<{ number: string }>;
}) {
  const { number } = use(params);
  const issueNumber = Number(number);
  const { data: issue, isLoading: issueLoading } = useIssue(issueNumber);
  const { data: comments, isLoading: commentsLoading } =
    useIssueComments(issueNumber);

  if (issueLoading) {
    return (
      <div className="space-y-4">
        <div className="h-8 w-48 animate-pulse rounded bg-muted" />
        <div className="h-64 animate-pulse rounded-lg bg-muted" />
      </div>
    );
  }

  if (!issue) {
    return <p className="text-muted-foreground">Issueが見つかりません</p>;
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Link href="/issues">
          <Button variant="ghost" size="sm">
            <ArrowLeft className="mr-1 h-4 w-4" />
            一覧に戻る
          </Button>
        </Link>
      </div>

      <Card>
        <CardHeader>
          <div className="flex items-start justify-between gap-4">
            <div className="space-y-2">
              <h2 className="text-xl font-semibold">
                <span
                  className={`mr-2 inline-block h-2.5 w-2.5 rounded-full ${
                    issue.state === "open" ? "bg-green-500" : "bg-purple-500"
                  }`}
                />
                {issue.title}
                <span className="ml-2 text-muted-foreground">
                  #{issue.number}
                </span>
              </h2>
              <div className="flex flex-wrap gap-1.5">
                {issue.labels.map((label) => (
                  <LabelBadge key={label.id} label={label} />
                ))}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Avatar className="h-5 w-5">
              <AvatarImage src={issue.user.avatar_url} alt={issue.user.login} />
              <AvatarFallback>{issue.user.login[0].toUpperCase()}</AvatarFallback>
            </Avatar>
            <span>{issue.user.login}</span>
            <span>が{formatDate(issue.created_at)}に作成</span>
          </div>
        </CardHeader>
        <CardContent>
          {issue.body ? (
            <div className="prose prose-sm max-w-none dark:prose-invert">
              <ReactMarkdown>{issue.body}</ReactMarkdown>
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">説明はありません</p>
          )}
        </CardContent>
      </Card>

      <Separator />

      <div className="space-y-4">
        <h3 className="text-lg font-semibold">
          コメント ({comments?.length ?? 0})
        </h3>

        {commentsLoading ? (
          <div className="space-y-3">
            {[1, 2].map((i) => (
              <div key={i} className="h-24 animate-pulse rounded-lg bg-muted" />
            ))}
          </div>
        ) : !comments?.length ? (
          <p className="text-sm text-muted-foreground">
            まだコメントはありません
          </p>
        ) : (
          <div className="space-y-4">
            {comments.map((comment) => (
              <Card key={comment.id}>
                <CardHeader className="pb-2">
                  <div className="flex items-center gap-2 text-sm">
                    <Avatar className="h-6 w-6">
                      <AvatarImage
                        src={comment.user.avatar_url}
                        alt={comment.user.login}
                      />
                      <AvatarFallback>
                        {comment.user.login[0].toUpperCase()}
                      </AvatarFallback>
                    </Avatar>
                    <span className="font-medium">{comment.user.login}</span>
                    <span className="text-muted-foreground">
                      {formatDate(comment.created_at)}
                    </span>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="prose prose-sm max-w-none dark:prose-invert">
                    <ReactMarkdown>{comment.body}</ReactMarkdown>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>

      <Separator />

      <div>
        <h3 className="mb-4 text-lg font-semibold">コメントを投稿</h3>
        <CommentForm issueNumber={issueNumber} />
      </div>
    </div>
  );
}
