"use client";

import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useLabelDistribution } from "@/hooks/use-dashboard";

export function LabelChart() {
  const { data: labels, isLoading } = useLabelDistribution();

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>ラベル別Issue分布</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex h-[250px] items-center justify-center">
            <div className="h-32 w-32 animate-pulse rounded-full bg-muted" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!labels?.length) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>ラベル別Issue分布</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">ラベルデータがありません</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>ラベル別Issue分布</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={250}>
          <PieChart>
            <Pie
              data={labels}
              dataKey="count"
              nameKey="name"
              cx="50%"
              cy="50%"
              outerRadius={80}
              label={({ name, value }) => `${name ?? ""} (${value})`}
            >
              {labels.map((entry, index) => (
                <Cell key={index} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
