"use client";

import { useMutation, useQuery } from "@tanstack/react-query";
import { useCallback, useRef, useState } from "react";

import { getSimulationStatus, startSimulation } from "@/lib/api";
import type { SimulationRequest, SimulationResult, WSProgress } from "@/lib/types";

export function useSimulation() {
  const [result, setResult] = useState<SimulationResult | null>(null);
  const [progress, setProgress] = useState<WSProgress | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const startPolling = useCallback((taskId: string) => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    intervalRef.current = setInterval(async () => {
      try {
        const status = await getSimulationStatus(taskId);
        setResult(status);
        setProgress({
          task_id: taskId,
          step: status.step || "",
          progress: status.progress,
          message: status.message || "",
        });
        if (status.status === "complete" || status.status === "failed") {
          if (intervalRef.current) clearInterval(intervalRef.current);
        }
      } catch {
        if (intervalRef.current) clearInterval(intervalRef.current);
      }
    }, 1000);
  }, []);

  const mutation = useMutation({
    mutationFn: (req: SimulationRequest) => startSimulation(req),
    onSuccess: (data) => {
      setResult(data);
      startPolling(data.task_id);
    },
  });

  return {
    run: mutation.mutate,
    isRunning: mutation.isPending || (result?.status === "running" || result?.status === "pending"),
    result,
    progress,
    error: mutation.error,
  };
}

export function useSimulationStatus(taskId: string | null) {
  return useQuery({
    queryKey: ["simulation", taskId],
    queryFn: () => getSimulationStatus(taskId!),
    enabled: !!taskId,
    refetchInterval: (query) => {
      const data = query.state.data;
      if (data?.status === "complete" || data?.status === "failed") return false;
      return 1000;
    },
  });
}
