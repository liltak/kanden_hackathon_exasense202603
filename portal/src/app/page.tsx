"use client";

import { StatsCards } from "@/components/dashboard/stats-cards";
import { MilestoneProgress } from "@/components/dashboard/milestone-progress";
import { LabelChart } from "@/components/dashboard/label-chart";
import { RecentActivity } from "@/components/dashboard/recent-activity";
import { AISummaryCard } from "@/components/dashboard/ai-summary";

export default function DashboardPage() {
  return (
    <div className="space-y-6">
      <StatsCards />
      <div className="grid gap-6 lg:grid-cols-2">
        <MilestoneProgress />
        <LabelChart />
      </div>
      <AISummaryCard />
      <RecentActivity />
    </div>
  );
}
