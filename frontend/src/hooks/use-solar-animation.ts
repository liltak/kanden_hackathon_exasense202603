"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { getShadowTimeline, getSolarPositions } from "@/lib/api";
import type {
  ShadowTimelineResponse,
  SunPositionEntry,
  SunPositionsResponse,
} from "@/lib/types";

export interface SolarAnimationState {
  /** All daytime sun positions */
  positions: SunPositionEntry[];
  /** Current time step index */
  currentIndex: number;
  /** Whether animation is playing */
  playing: boolean;
  /** Playback speed multiplier */
  speed: number;
  /** Currently selected date string */
  date: string;
  /** Whether data is loading */
  loading: boolean;
  /** Shadow data (null if not loaded) */
  shadow: ShadowTimelineResponse | null;
  /** Current sun position (null when no data) */
  currentPosition: SunPositionEntry | null;
  /** Current shadow row for the active time step */
  currentShadow: boolean[] | null;
}

export interface SolarAnimationActions {
  setDate: (date: string) => void;
  setIndex: (index: number) => void;
  togglePlay: () => void;
  setSpeed: (speed: number) => void;
  fetchData: (date: string, meshSource?: string) => Promise<void>;
}

export function useSolarAnimation(
  meshSource: string = "complex",
): SolarAnimationState & SolarAnimationActions {
  const [positions, setPositions] = useState<SunPositionEntry[]>([]);
  const [shadow, setShadow] = useState<ShadowTimelineResponse | null>(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [date, setDateState] = useState("2025-06-21");
  const [loading, setLoading] = useState(false);

  const animRef = useRef<number | null>(null);
  const lastTickRef = useRef(0);
  const indexRef = useRef(0);

  // Keep ref in sync with state
  indexRef.current = currentIndex;

  const fetchData = useCallback(
    async (targetDate: string, source?: string) => {
      setLoading(true);
      setPlaying(false);
      try {
        const [posRes, shadowRes] = await Promise.all([
          getSolarPositions(targetDate),
          getShadowTimeline(targetDate, source ?? meshSource),
        ]);
        setPositions(posRes.positions);
        setShadow(shadowRes);
        setCurrentIndex(0);
        setDateState(targetDate);
      } catch (err) {
        console.error("Failed to fetch solar data:", err);
      } finally {
        setLoading(false);
      }
    },
    [meshSource],
  );

  const setDate = useCallback(
    (d: string) => {
      fetchData(d);
    },
    [fetchData],
  );

  const setIndex = useCallback((i: number) => {
    setCurrentIndex(i);
  }, []);

  const togglePlay = useCallback(() => {
    setPlaying((p) => !p);
  }, []);

  // Animation loop
  useEffect(() => {
    if (!playing || positions.length === 0) {
      if (animRef.current) {
        cancelAnimationFrame(animRef.current);
        animRef.current = null;
      }
      return;
    }

    // Interval between steps in ms (base 500ms, divided by speed)
    const intervalMs = 500 / speed;

    const tick = (timestamp: number) => {
      if (timestamp - lastTickRef.current >= intervalMs) {
        lastTickRef.current = timestamp;
        setCurrentIndex((prev) => {
          const next = prev + 1;
          if (next >= positions.length) {
            setPlaying(false);
            return prev;
          }
          return next;
        });
      }
      animRef.current = requestAnimationFrame(tick);
    };

    lastTickRef.current = performance.now();
    animRef.current = requestAnimationFrame(tick);

    return () => {
      if (animRef.current) {
        cancelAnimationFrame(animRef.current);
        animRef.current = null;
      }
    };
  }, [playing, speed, positions.length]);

  const currentPosition =
    positions.length > 0 && currentIndex < positions.length
      ? positions[currentIndex]
      : null;

  const currentShadow =
    shadow && currentIndex < shadow.shadow_matrix.length
      ? shadow.shadow_matrix[currentIndex]
      : null;

  return {
    positions,
    currentIndex,
    playing,
    speed,
    date,
    loading,
    shadow,
    currentPosition,
    currentShadow,
    setDate,
    setIndex,
    togglePlay,
    setSpeed,
    fetchData,
  };
}
