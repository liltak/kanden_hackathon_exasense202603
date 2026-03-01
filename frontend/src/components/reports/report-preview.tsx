"use client";

import ReactMarkdown from "react-markdown";

import { ScrollArea } from "@/components/ui/scroll-area";

interface ReportPreviewProps {
  markdown: string;
}

export function ReportPreview({ markdown }: ReportPreviewProps) {
  return (
    <ScrollArea className="h-[calc(100vh-300px)] rounded-lg border p-6">
      <div className="prose prose-sm max-w-none">
        <ReactMarkdown>{markdown}</ReactMarkdown>
      </div>
    </ScrollArea>
  );
}
