"use client";

import { useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { IssueCard } from "@/components/issues/issue-card";
import { CreateIssueDialog } from "@/components/issues/create-issue-dialog";
import { useIssues } from "@/hooks/use-issues";

const states = [
  { value: "all", label: "すべて" },
  { value: "open", label: "Open" },
  { value: "closed", label: "Closed" },
];

export default function IssuesPage() {
  const [state, setState] = useState("open");
  const [search, setSearch] = useState("");
  const { data: issues, isLoading } = useIssues({ state });

  const filtered = issues?.filter((issue) =>
    search
      ? issue.title.toLowerCase().includes(search.toLowerCase()) ||
        issue.labels.some((l) =>
          l.name.toLowerCase().includes(search.toLowerCase())
        )
      : true
  );

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-4">
          <div className="flex gap-2">
            {states.map((s) => (
              <Button
                key={s.value}
                variant={state === s.value ? "default" : "outline"}
                size="sm"
                onClick={() => setState(s.value)}
              >
                {s.label}
              </Button>
            ))}
          </div>
          <Input
            placeholder="Issue・ラベルを検索..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-60"
          />
        </div>
        <CreateIssueDialog />
      </div>

      {isLoading ? (
        <div className="grid gap-4 md:grid-cols-2">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="h-32 animate-pulse rounded-lg bg-muted" />
          ))}
        </div>
      ) : !filtered?.length ? (
        <p className="text-center text-muted-foreground">
          該当するIssueがありません
        </p>
      ) : (
        <div className="grid gap-4 md:grid-cols-2">
          {filtered.map((issue) => (
            <IssueCard key={issue.id} issue={issue} />
          ))}
        </div>
      )}
    </div>
  );
}
