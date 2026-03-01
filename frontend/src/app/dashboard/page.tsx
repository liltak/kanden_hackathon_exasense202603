"use client";

import Link from "next/link";

import { KPICard } from "@/components/dashboard/kpi-card";
import { PipelineStatus } from "@/components/dashboard/pipeline-status";

export default function DashboardPage() {
  return (
    <div className="space-y-6">
      <PipelineStatus />

      <div className="grid grid-cols-4 gap-4">
        <KPICard title="Phase 3" value="Ready" unit="日照シミュレーション" gradient="green" />
        <KPICard title="Phase 5" value="Active" unit="WebUI" gradient="blue" />
        <KPICard title="対応形式" value="PLY/OBJ" unit="メッシュ入力" gradient="purple" />
        <KPICard title="エンジン" value="pvlib" unit="シミュレーション" gradient="orange" />
      </div>

      <div className="rounded-xl border p-6">
        <h3 className="mb-4 text-lg font-semibold">クイックスタート</h3>
        <ol className="space-y-3 text-sm text-gray-600">
          <li className="flex items-start gap-3">
            <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-blue-100 text-xs font-bold text-blue-700">1</span>
            <span><Link href="/simulation" className="font-medium text-blue-600 hover:underline">シミュレーション</Link>タブで地点・パラメータを設定し実行</span>
          </li>
          <li className="flex items-start gap-3">
            <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-blue-100 text-xs font-bold text-blue-700">2</span>
            <span><Link href="/viewer" className="font-medium text-blue-600 hover:underline">3Dビュー</Link>タブで建物モデルとヒートマップを確認</span>
          </li>
          <li className="flex items-start gap-3">
            <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-blue-100 text-xs font-bold text-blue-700">3</span>
            <span><Link href="/analysis" className="font-medium text-blue-600 hover:underline">AI分析</Link>タブで設置提案をAIに質問</span>
          </li>
          <li className="flex items-start gap-3">
            <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-blue-100 text-xs font-bold text-blue-700">4</span>
            <span><Link href="/reports" className="font-medium text-blue-600 hover:underline">レポート</Link>タブで報告書をダウンロード</span>
          </li>
        </ol>
      </div>
    </div>
  );
}
