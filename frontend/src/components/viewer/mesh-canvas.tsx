"use client";

import { Environment, OrbitControls, useGLTF } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { Component, Suspense, useEffect, useMemo, useRef, type ReactNode } from "react";
import * as THREE from "three";

import type { SunPositionEntry } from "@/lib/types";

import { CompassIndicator } from "./compass-indicator";
import type { MeshRotation } from "./mesh-adjust-controls";
import { ShadowOverlay } from "./shadow-overlay";
import { SunOrbit } from "./sun-orbit";

interface MeshCanvasProps {
  url: string | null;
  sunPositions?: SunPositionEntry[];
  sunIndex?: number;
  currentSunPosition?: SunPositionEntry | null;
  currentShadow?: boolean[] | null;
  solarActive?: boolean;
  meshRotation?: MeshRotation;
}

/** Degrees → Euler radians. Heading = Y rotation, tiltX = X, tiltZ = Z. */
function rotationToEuler(rot: MeshRotation): [number, number, number] {
  const deg2rad = Math.PI / 180;
  return [rot.tiltX * deg2rad, rot.heading * deg2rad, rot.tiltZ * deg2rad];
}

function useSolarMaterial(scene: THREE.Group, solarActive: boolean) {
  const origMaterials = useRef<Map<THREE.Mesh, THREE.Material | THREE.Material[]>>(new Map());

  useEffect(() => {
    if (solarActive) {
      scene.traverse((child) => {
        if (child instanceof THREE.Mesh) {
          if (!origMaterials.current.has(child)) {
            origMaterials.current.set(child, child.material);
          }
          child.material = new THREE.MeshBasicMaterial({ vertexColors: true });
        }
      });
    } else {
      origMaterials.current.forEach((mat, mesh) => {
        mesh.material = mat;
      });
      origMaterials.current.clear();
    }
  }, [scene, solarActive]);
}

/** Detect and style point clouds (GLTF mode=0 POINTS) in the loaded scene. */
function usePointCloudMaterial(scene: THREE.Group, pointSize: number = 2.0) {
  useEffect(() => {
    scene.traverse((child) => {
      if (child instanceof THREE.Points) {
        child.material = new THREE.PointsMaterial({
          size: pointSize,
          vertexColors: true,
          sizeAttenuation: true,
        });
      }
    });
  }, [scene, pointSize]);
}

function Model({
  url,
  sunDirection,
  shadowRow,
  solarActive,
  meshRotation,
}: {
  url: string;
  sunDirection?: [number, number, number] | null;
  shadowRow?: boolean[] | null;
  solarActive?: boolean;
  meshRotation?: MeshRotation;
}) {
  const { scene } = useGLTF(url);
  const groupRef = useRef<THREE.Group>(null);

  usePointCloudMaterial(scene);
  useSolarMaterial(scene, !!solarActive);

  useEffect(() => {
    return () => {
      useGLTF.clear(url);
    };
  }, [url]);

  const euler = useMemo(
    () => rotationToEuler(meshRotation ?? { heading: 0, tiltX: 0, tiltZ: 0 }),
    [meshRotation],
  );

  return (
    <>
      <group ref={groupRef} rotation={euler}>
        <primitive object={scene} />
      </group>
      {solarActive && (
        <ShadowOverlay
          scene={scene}
          sunDirection={sunDirection ?? null}
          shadowRow={shadowRow ?? null}
          groupRef={groupRef}
        />
      )}
    </>
  );
}

function LoadingFallback() {
  return (
    <mesh>
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial color="#94a3b8" wireframe />
    </mesh>
  );
}

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback: ReactNode;
  resetKey?: number;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

class MeshErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidUpdate(prevProps: ErrorBoundaryProps) {
    if (prevProps.resetKey !== this.props.resetKey && this.state.hasError) {
      this.setState({ hasError: false, error: null });
    }
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback;
    }
    return this.props.children;
  }
}

function ErrorFallback({ error }: { error?: string }) {
  return (
    <div className="flex h-full flex-col items-center justify-center text-gray-500">
      <svg className="mb-3 h-12 w-12 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v2m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
      <p className="text-sm font-medium">メッシュの読み込みに失敗しました</p>
      <p className="mt-1 text-xs text-gray-400">{error || "バックエンドAPIが起動していない可能性があります"}</p>
    </div>
  );
}

export function MeshCanvas({
  url,
  sunPositions = [],
  sunIndex = 0,
  currentSunPosition = null,
  currentShadow = null,
  solarActive = false,
  meshRotation,
}: MeshCanvasProps) {
  return (
    <div className="h-[600px] w-full rounded-xl border bg-gray-50">
      <MeshErrorBoundary resetKey={url ? 1 : 0} fallback={<ErrorFallback />}>
        <Canvas camera={{ position: [30, 30, 30], fov: 50 }}>
          {!solarActive && (
            <>
              <ambientLight intensity={0.5} />
              <directionalLight position={[10, 20, 10]} intensity={0.8} />
            </>
          )}
          {solarActive && <ambientLight intensity={0.1} />}

          <Suspense fallback={<LoadingFallback />}>
            {url ? (
              <Model
                key={url}
                url={url}
                sunDirection={currentSunPosition?.direction_y_up ?? null}
                shadowRow={currentShadow}
                solarActive={solarActive}
                meshRotation={meshRotation}
              />
            ) : (
              <LoadingFallback />
            )}
            {!solarActive && <Environment preset="city" />}
          </Suspense>

          {solarActive && sunPositions.length > 0 && (
            <SunOrbit positions={sunPositions} currentIndex={sunIndex} />
          )}

          <OrbitControls makeDefault />
          <gridHelper args={[100, 20, "#e5e7eb", "#f3f4f6"]} />
          <CompassIndicator />
        </Canvas>
      </MeshErrorBoundary>
    </div>
  );
}
