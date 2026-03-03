"use client";

import { CircleDot, CheckCircle, TrendingUp, GitCommit } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useDashboardStats } from "@/hooks/use-dashboard";

export function StatsCards() {
  const { data: stats, isLoading } = useDashboardStats();

  const cards = [
    {
      title: "Open Issues",
      value: stats?.openIssues ?? "-",
      icon: CircleDot,
      color: "text-chart-1",
    },
    {
      title: "Closed Issues",
      value: stats?.closedIssues ?? "-",
      icon: CheckCircle,
      color: "text-chart-2",
    },
    {
      title: "完了率",
      value: stats ? `${stats.completionRate}%` : "-",
      icon: TrendingUp,
      color: "text-chart-4",
    },
    {
      title: "直近1週間のコミット",
      value: stats?.recentCommits ?? "-",
      icon: GitCommit,
      color: "text-chart-3",
    },
  ];

  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
      {cards.map((card) => (
        <Card key={card.title}>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">{card.title}</CardTitle>
            <card.icon className={`h-4 w-4 ${card.color}`} />
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${isLoading ? "animate-pulse" : ""}`}>
              {card.value}
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
