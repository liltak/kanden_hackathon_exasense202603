"use client";

import { useMutation, useQuery } from "@tanstack/react-query";

import { generateWaypoint, getWaypointStatus } from "@/lib/api";
import type { WaypointGenerateRequest } from "@/lib/types";

export function useWaypointStatus() {
  return useQuery({
    queryKey: ["waypoint", "status"],
    queryFn: getWaypointStatus,
    refetchInterval: 30_000,
  });
}

export function useWaypointGenerate() {
  return useMutation({
    mutationFn: (req: WaypointGenerateRequest) => generateWaypoint(req),
  });
}
