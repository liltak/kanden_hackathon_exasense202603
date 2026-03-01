"use client";

import { Line, Text } from "@react-three/drei";
import { useMemo } from "react";

interface CompassIndicatorProps {
  /** Radius of the compass circle / label distance from center */
  radius?: number;
}

// Three.js Y-up: North = -Z, East = +X
const DIRECTIONS = [
  { label: "N", color: "#ef4444", position: [0, 0.3, -1] as const },
  { label: "S", color: "#3b82f6", position: [0, 0.3, 1] as const },
  { label: "E", color: "#6b7280", position: [1, 0.3, 0] as const },
  { label: "W", color: "#6b7280", position: [-1, 0.3, 0] as const },
] as const;

/**
 * Compass rose overlay in the 3D scene.
 * Shows N/S/E/W labels and cardinal direction lines on the ground plane.
 */
export function CompassIndicator({ radius = 52 }: CompassIndicatorProps) {
  // North-South axis line (red)
  const nsLine = useMemo(
    () =>
      [
        [0, 0.05, -radius * 0.95],
        [0, 0.05, radius * 0.95],
      ] as [number, number, number][],
    [radius],
  );

  // East-West axis line (blue-gray)
  const ewLine = useMemo(
    () =>
      [
        [-radius * 0.95, 0.05, 0],
        [radius * 0.95, 0.05, 0],
      ] as [number, number, number][],
    [radius],
  );

  // North arrow triangle
  const arrowPoints = useMemo(() => {
    const tip = radius * 0.95;
    const base = radius * 0.88;
    const w = 1.2;
    return [
      [0, 0.1, -tip],
      [-w, 0.1, -base],
      [w, 0.1, -base],
      [0, 0.1, -tip],
    ] as [number, number, number][];
  }, [radius]);

  return (
    <group>
      {/* N-S line */}
      <Line points={nsLine} color="#ef4444" lineWidth={2} opacity={0.6} transparent />

      {/* E-W line */}
      <Line points={ewLine} color="#94a3b8" lineWidth={1.5} opacity={0.4} transparent />

      {/* North arrow */}
      <Line points={arrowPoints} color="#ef4444" lineWidth={2.5} />

      {/* Cardinal direction labels */}
      {DIRECTIONS.map(({ label, color, position }) => (
        <Text
          key={label}
          position={[position[0] * radius, position[1] + 0.5, position[2] * radius]}
          fontSize={2.5}
          color={color}
          anchorX="center"
          anchorY="middle"
          font="/fonts/inter-bold.woff"
          fontWeight="bold"
        >
          {label}
        </Text>
      ))}
    </group>
  );
}
