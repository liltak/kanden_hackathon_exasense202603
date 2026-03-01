"use client";

import { Html } from "@react-three/drei";
import { useMemo } from "react";
import * as THREE from "three";

interface CompassIndicatorProps {
  radius?: number;
}

const DIRECTIONS = [
  { label: "N", color: "#ef4444", x: 0, z: -1 },
  { label: "S", color: "#3b82f6", x: 0, z: 1 },
  { label: "E", color: "#6b7280", x: 1, z: 0 },
  { label: "W", color: "#6b7280", x: -1, z: 0 },
] as const;

function CardinalLine({
  start,
  end,
  color,
}: {
  start: [number, number, number];
  end: [number, number, number];
  color: string;
}) {
  const lineObj = useMemo(() => {
    const geo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(...start),
      new THREE.Vector3(...end),
    ]);
    const mat = new THREE.LineBasicMaterial({ color });
    return new THREE.Line(geo, mat);
  }, [start, end, color]);

  return <primitive object={lineObj} />;
}

/**
 * Compass rose overlay: N/S/E/W labels + cardinal direction lines.
 */
export function CompassIndicator({ radius = 52 }: CompassIndicatorProps) {
  const r = radius * 0.95;

  return (
    <group>
      {/* N-S line (red) */}
      <CardinalLine start={[0, 0.05, -r]} end={[0, 0.05, r]} color="#ef4444" />

      {/* E-W line (gray) */}
      <CardinalLine start={[-r, 0.05, 0]} end={[r, 0.05, 0]} color="#94a3b8" />

      {/* North arrow — cone pointing north */}
      <mesh position={[0, 0.1, -r + 1.5]} rotation={[Math.PI / 2, 0, 0]}>
        <coneGeometry args={[1.2, 3, 3]} />
        <meshBasicMaterial color="#ef4444" />
      </mesh>

      {/* Cardinal direction labels */}
      {DIRECTIONS.map(({ label, color, x, z }) => (
        <Html
          key={label}
          position={[x * radius, 1, z * radius]}
          center
          distanceFactor={80}
          style={{ pointerEvents: "none" }}
        >
          <div
            style={{
              color,
              fontWeight: 700,
              fontSize: "18px",
              textShadow: "0 0 4px rgba(255,255,255,0.9)",
              userSelect: "none",
            }}
          >
            {label}
          </div>
        </Html>
      ))}
    </group>
  );
}
