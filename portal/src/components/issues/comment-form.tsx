"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { usePostComment } from "@/hooks/use-issues";

const STORAGE_KEY = "exasense-portal-name";

export function CommentForm({ issueNumber }: { issueNumber: number }) {
  const [name, setName] = useState("");
  const [body, setBody] = useState("");
  const postComment = usePostComment(issueNumber);

  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) setName(saved);
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || !body.trim()) return;

    localStorage.setItem(STORAGE_KEY, name.trim());
    await postComment.mutateAsync({ name: name.trim(), body: body.trim() });
    setBody("");
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label htmlFor="name" className="mb-1.5 block text-sm font-medium">
          名前
        </label>
        <Input
          id="name"
          placeholder="あなたの名前"
          value={name}
          onChange={(e) => setName(e.target.value)}
          required
        />
      </div>
      <div>
        <label htmlFor="body" className="mb-1.5 block text-sm font-medium">
          コメント
        </label>
        <Textarea
          id="body"
          placeholder="コメントを入力..."
          rows={4}
          value={body}
          onChange={(e) => setBody(e.target.value)}
          required
        />
      </div>
      <Button type="submit" disabled={postComment.isPending}>
        {postComment.isPending ? "投稿中..." : "コメントを投稿"}
      </Button>
    </form>
  );
}
