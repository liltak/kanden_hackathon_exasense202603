"use client";

import { ChatInterface } from "@/components/analysis/chat-interface";

export default function AnalysisPage() {
  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-lg font-semibold">AI エネルギーアドバイザー</h3>
        <p className="text-sm text-gray-500">
          シミュレーション結果に基づいて、AIが太陽光パネル設置の提案・分析を行います。
          <span className="italic text-gray-400"> （H100環境では Qwen3.5-VL が回答します）</span>
        </p>
      </div>
      <ChatInterface />
    </div>
  );
}
