"use client";

import { Line } from "@react-three/drei";
import { useMemo } from "react";

import type { SunPositionEntry } from "@/lib/types";

interface SunOrbitProps {
  positions: SunPositionEntry[];
  currentIndex: number;
  /** Scale factor for sun distance from origin */
  radius?: number;
}

/**
 * Renders the sun as a yellow sphere along the orbit path.
 * Orbit path drawn as a Line using direction_y_up vectors.
 */
export function SunOrbit({
  positions,
  currentIndex,
  radius = 40,
}: SunOrbitProps) {
  // Build orbit line points from all positions
  const orbitPoints = useMemo(() => {
    return positions.map((p) => {
      const [x, y, z] = p.direction_y_up;
      return [x * radius, y * radius, z * radius] as [number, number, number];
    });
  }, [positions, radius]);

  if (positions.length === 0 || currentIndex >= positions.length) return null;

  const current = positions[currentIndex];
  const [sx, sy, sz] = current.direction_y_up;
  const sunPos: [number, number, number] = [
    sx * radius,
    sy * radius,
    sz * radius,
  ];

  return (
    <group>
      {/* Orbit path */}
      {orbitPoints.length >= 2 && (
        <Line
          points={orbitPoints}
          color="#fbbf24"
          lineWidth={1.5}
          opacity={0.5}
          transparent
        />
      )}

      {/* Sun sphere */}
      <mesh position={sunPos}>
        <sphereGeometry args={[1.5, 16, 16]} />
        <meshBasicMaterial color="#fbbf24" />
      </mesh>

      {/* Sun glow */}
      <mesh position={sunPos}>
        <sphereGeometry args={[2.5, 16, 16]} />
        <meshBasicMaterial color="#fde68a" transparent opacity={0.3} />
      </mesh>
    </group>
  );
}
