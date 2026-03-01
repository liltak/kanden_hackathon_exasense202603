"use client";

import { useEffect, useRef, type RefObject } from "react";
import * as THREE from "three";

interface ShadowOverlayProps {
  scene: THREE.Group;
  sunDirection: [number, number, number] | null;
  shadowRow: boolean[] | null;
  /** Parent group ref — used to include mesh rotation in normal calculation */
  groupRef?: RefObject<THREE.Group | null>;
}

// Color ramp: dark → ambient → orange → bright yellow
const COLOR_SHADOW = { r: 0.12, g: 0.12, b: 0.2 };
const COLOR_AMBIENT = { r: 0.25, g: 0.28, b: 0.4 };
const COLOR_MID = { r: 0.85, g: 0.55, b: 0.2 };
const COLOR_LIT = { r: 1.0, g: 0.9, b: 0.35 };

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function intensityToRGB(t: number): [number, number, number] {
  let r: number, g: number, b: number;
  if (t <= 0) {
    r = COLOR_SHADOW.r; g = COLOR_SHADOW.g; b = COLOR_SHADOW.b;
  } else if (t < 0.3) {
    const s = t / 0.3;
    r = lerp(COLOR_SHADOW.r, COLOR_AMBIENT.r, s);
    g = lerp(COLOR_SHADOW.g, COLOR_AMBIENT.g, s);
    b = lerp(COLOR_SHADOW.b, COLOR_AMBIENT.b, s);
  } else if (t < 0.6) {
    const s = (t - 0.3) / 0.3;
    r = lerp(COLOR_AMBIENT.r, COLOR_MID.r, s);
    g = lerp(COLOR_AMBIENT.g, COLOR_MID.g, s);
    b = lerp(COLOR_AMBIENT.b, COLOR_MID.b, s);
  } else {
    const s = (t - 0.6) / 0.4;
    r = lerp(COLOR_MID.r, COLOR_LIT.r, s);
    g = lerp(COLOR_MID.g, COLOR_LIT.g, s);
    b = lerp(COLOR_MID.b, COLOR_LIT.b, s);
  }
  return [r, g, b];
}

/**
 * Compute face normals from geometry, applying the full world matrix
 * (including parent group rotation) to transform normals to world space.
 */
function buildFaceNormals(
  geo: THREE.BufferGeometry,
  mesh: THREE.Mesh,
  parentGroup?: THREE.Group | null,
): Float32Array {
  const pos = geo.attributes.position;
  const index = geo.index;

  const nFaces = index
    ? Math.floor(index.count / 3)
    : Math.floor(pos.count / 3);
  const normals = new Float32Array(nFaces * 3);

  const vA = new THREE.Vector3();
  const vB = new THREE.Vector3();
  const vC = new THREE.Vector3();
  const ab = new THREE.Vector3();
  const ac = new THREE.Vector3();
  const fn = new THREE.Vector3();

  // Build combined matrix: parent rotation * mesh local matrix
  const combinedMatrix = new THREE.Matrix4();
  if (parentGroup) {
    parentGroup.updateMatrixWorld(true);
    combinedMatrix.copy(parentGroup.matrixWorld);
  }
  mesh.updateMatrixWorld(true);
  combinedMatrix.multiply(mesh.matrixWorld);

  const normalMat = new THREE.Matrix3().getNormalMatrix(combinedMatrix);

  for (let f = 0; f < nFaces; f++) {
    const iA = index ? index.getX(f * 3) : f * 3;
    const iB = index ? index.getX(f * 3 + 1) : f * 3 + 1;
    const iC = index ? index.getX(f * 3 + 2) : f * 3 + 2;

    vA.fromBufferAttribute(pos, iA);
    vB.fromBufferAttribute(pos, iB);
    vC.fromBufferAttribute(pos, iC);
    ab.subVectors(vB, vA);
    ac.subVectors(vC, vA);
    fn.crossVectors(ab, ac).normalize();
    fn.applyMatrix3(normalMat).normalize();

    normals[f * 3] = fn.x;
    normals[f * 3 + 1] = fn.y;
    normals[f * 3 + 2] = fn.z;
  }

  return normals;
}

/**
 * Normal-based solar illumination overlay.
 * Recomputes face normals whenever mesh rotation or sun direction changes.
 */
export function ShadowOverlay({
  scene,
  sunDirection,
  shadowRow,
  groupRef,
}: ShadowOverlayProps) {
  const meshRef = useRef<THREE.Mesh | null>(null);
  const colorBufRef = useRef<THREE.Float32BufferAttribute | null>(null);

  // Init: find mesh, set up color buffer
  useEffect(() => {
    const meshes: THREE.Mesh[] = [];
    scene.traverse((child) => {
      if (child instanceof THREE.Mesh && child.geometry) {
        meshes.push(child);
      }
    });
    const target = meshes[0] ?? null;
    meshRef.current = target;
    if (!target?.geometry) return;

    const geo = target.geometry;
    const vertCount = geo.attributes.position.count;
    const colorArray = new Float32Array(vertCount * 3);
    colorArray.fill(0.5);
    const colorAttr = new THREE.Float32BufferAttribute(colorArray, 3);
    geo.deleteAttribute("color");
    geo.setAttribute("color", colorAttr);
    colorBufRef.current = colorAttr;

    const mat = target.material;
    if (mat instanceof THREE.MeshStandardMaterial) {
      mat.vertexColors = true;
      mat.needsUpdate = true;
    }
  }, [scene]);

  // Recompute lighting when sun direction, shadow, or rotation changes
  // groupRef.current?.rotation changes when MeshRotation is updated
  const groupRotation = groupRef?.current?.rotation;
  const rotKey = groupRotation
    ? `${groupRotation.x.toFixed(4)}_${groupRotation.y.toFixed(4)}_${groupRotation.z.toFixed(4)}`
    : "0_0_0";

  useEffect(() => {
    const mesh = meshRef.current;
    const colorAttr = colorBufRef.current;
    if (!mesh?.geometry || !colorAttr || !sunDirection) return;

    const geo = mesh.geometry;
    const index = geo.index;
    const posCount = geo.attributes.position.count;
    const parentGroup = groupRef?.current ?? null;

    // Recompute face normals with current rotation
    const faceNormals = buildFaceNormals(geo, mesh, parentGroup);

    const sunX = sunDirection[0];
    const sunY = sunDirection[1];
    const sunZ = sunDirection[2];
    const len = Math.sqrt(sunX * sunX + sunY * sunY + sunZ * sunZ);
    const sdx = sunX / len;
    const sdy = sunY / len;
    const sdz = sunZ / len;

    const colors = colorAttr.array as Float32Array;
    const nFaces = index
      ? Math.floor(index.count / 3)
      : Math.floor(posCount / 3);

    for (let f = 0; f < nFaces; f++) {
      const nx = faceNormals[f * 3];
      const ny = faceNormals[f * 3 + 1];
      const nz = faceNormals[f * 3 + 2];
      let intensity = Math.max(0, nx * sdx + ny * sdy + nz * sdz);

      if (shadowRow && f < shadowRow.length && !shadowRow[f]) {
        intensity *= 0.15;
      }

      const [r, g, b] = intensityToRGB(intensity);

      for (let v = 0; v < 3; v++) {
        const vi = index ? index.getX(f * 3 + v) : f * 3 + v;
        colors[vi * 3] = r;
        colors[vi * 3 + 1] = g;
        colors[vi * 3 + 2] = b;
      }
    }

    colorAttr.needsUpdate = true;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sunDirection, shadowRow, rotKey]);

  return null;
}
