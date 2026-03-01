"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { createSimulationWS } from "@/lib/api";
import type { WSProgress } from "@/lib/types";

export function useWSProgress(taskId: string | null) {
  const [progress, setProgress] = useState<WSProgress | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  useEffect(() => {
    if (!taskId) return;

    const ws = createSimulationWS(taskId);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      try {
        const data: WSProgress = JSON.parse(event.data);
        setProgress(data);
      } catch {
        // ignore parse errors
      }
    };

    ws.onerror = () => disconnect();
    ws.onclose = () => { wsRef.current = null; };

    return () => disconnect();
  }, [taskId, disconnect]);

  return { progress, disconnect };
}
