"use client";

import { useMutation, useQuery } from "@tanstack/react-query";
import { getRustInspectionStatus, runRustInspection } from "@/lib/api";
import type { RustInspectionRunRequest } from "@/lib/types";

export function useRustInspectionStatus() {
  return useQuery({
    queryKey: ["rust-inspection", "status"],
    queryFn: getRustInspectionStatus,
    refetchInterval: 30_000,
  });
}

export function useRustInspectionRun() {
  return useMutation({
    mutationFn: (req: RustInspectionRunRequest) => runRustInspection(req),
  });
}
