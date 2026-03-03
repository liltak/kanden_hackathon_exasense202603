"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Plus, X } from "lucide-react";
import { useCreateIssue } from "@/hooks/use-issues";

const STORAGE_KEY = "exasense-portal-name";

const LABEL_OPTIONS = [
  { name: "bug", color: "#d73a4a" },
  { name: "enhancement", color: "#a2eeef" },
  { name: "question", color: "#d876e3" },
];

export function CreateIssueDialog() {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [title, setTitle] = useState("");
  const [body, setBody] = useState("");
  const [labels, setLabels] = useState<string[]>([]);
  const createIssue = useCreateIssue();
  const router = useRouter();

  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) setName(saved);
  }, []);

  const toggleLabel = (label: string) => {
    setLabels((prev) =>
      prev.includes(label) ? prev.filter((l) => l !== label) : [...prev, label]
    );
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || !title.trim()) return;

    localStorage.setItem(STORAGE_KEY, name.trim());

    const result = await createIssue.mutateAsync({
      title: title.trim(),
      body: body.trim(),
      name: name.trim(),
      labels,
    });

    setTitle("");
    setBody("");
    setLabels([]);
    setOpen(false);
    router.push(`/issues/${result.number}`);
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button size="sm">
          <Plus className="mr-1.5 h-4 w-4" />
          Issue作成
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>新しいIssueを作成</DialogTitle>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="create-name" className="mb-1.5 block text-sm font-medium">
              名前
            </label>
            <Input
              id="create-name"
              placeholder="あなたの名前"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
            />
          </div>
          <div>
            <label htmlFor="create-title" className="mb-1.5 block text-sm font-medium">
              タイトル
            </label>
            <Input
              id="create-title"
              placeholder="Issueのタイトル"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              required
            />
          </div>
          <div>
            <label htmlFor="create-body" className="mb-1.5 block text-sm font-medium">
              説明（任意）
            </label>
            <Textarea
              id="create-body"
              placeholder="Issueの詳細を入力..."
              rows={5}
              value={body}
              onChange={(e) => setBody(e.target.value)}
            />
          </div>
          <div>
            <label className="mb-1.5 block text-sm font-medium">
              ラベル（任意）
            </label>
            <div className="flex flex-wrap gap-2">
              {LABEL_OPTIONS.map((l) => (
                <Badge
                  key={l.name}
                  variant={labels.includes(l.name) ? "default" : "outline"}
                  className="cursor-pointer select-none"
                  style={
                    labels.includes(l.name)
                      ? { backgroundColor: l.color, color: "#000" }
                      : undefined
                  }
                  onClick={() => toggleLabel(l.name)}
                >
                  {l.name}
                  {labels.includes(l.name) && (
                    <X className="ml-1 h-3 w-3" />
                  )}
                </Badge>
              ))}
            </div>
          </div>
          <div className="flex justify-end gap-2 pt-2">
            <Button
              type="button"
              variant="outline"
              onClick={() => setOpen(false)}
            >
              キャンセル
            </Button>
            <Button type="submit" disabled={createIssue.isPending}>
              {createIssue.isPending ? "作成中..." : "作成"}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}
