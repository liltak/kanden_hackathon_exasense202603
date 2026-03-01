"use client";

import { useRef } from "react";
import * as THREE from "three";

import type { SunPositionEntry } from "@/lib/types";

interface SunDirectionalLightProps {
  position: SunPositionEntry | null;
  /** Scale factor for light distance */
  radius?: number;
}

/**
 * Directional light that follows the sun position.
 * Casts shadows onto the mesh.
 */
export function SunDirectionalLight({
  position,
  radius = 40,
}: SunDirectionalLightProps) {
  const lightRef = useRef<THREE.DirectionalLight>(null);

  if (!position) return null;

  const [x, y, z] = position.direction_y_up;
  const lightPos: [number, number, number] = [
    x * radius,
    y * radius,
    z * radius,
  ];

  return (
    <directionalLight
      ref={lightRef}
      position={lightPos}
      intensity={1.2}
      castShadow
      shadow-mapSize-width={1024}
      shadow-mapSize-height={1024}
      shadow-camera-left={-50}
      shadow-camera-right={50}
      shadow-camera-top={50}
      shadow-camera-bottom={-50}
      shadow-camera-near={1}
      shadow-camera-far={100}
    />
  );
}
