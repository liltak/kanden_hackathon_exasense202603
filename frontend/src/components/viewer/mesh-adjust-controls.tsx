"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";

export interface MeshRotation {
  /** Y-axis rotation in degrees (compass heading, 0=North) */
  heading: number;
  /** X-axis tilt in degrees */
  tiltX: number;
  /** Z-axis tilt in degrees */
  tiltZ: number;
}

interface MeshAdjustControlsProps {
  rotation: MeshRotation;
  onChange: (rotation: MeshRotation) => void;
}

const COMPASS = ["北", "北東", "東", "南東", "南", "南西", "西", "北西"] as const;

function compassLabel(deg: number): string {
  const idx = Math.round(((deg % 360) + 360) % 360 / 45) % 8;
  return `${COMPASS[idx]} (${Math.round(deg)}°)`;
}

export function MeshAdjustControls({ rotation, onChange }: MeshAdjustControlsProps) {
  const update = (patch: Partial<MeshRotation>) => {
    onChange({ ...rotation, ...patch });
  };

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-sm">メッシュ方角・傾き調整</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Heading (Y-axis rotation) */}
        <div>
          <div className="mb-1 flex items-center justify-between">
            <label className="text-xs text-gray-500">方角（北=0°）</label>
            <span className="text-xs font-medium">{compassLabel(rotation.heading)}</span>
          </div>
          <Slider
            value={[rotation.heading]}
            min={0}
            max={360}
            step={5}
            onValueChange={([v]) => update({ heading: v })}
          />
        </div>

        {/* Tilt X */}
        <div>
          <div className="mb-1 flex items-center justify-between">
            <label className="text-xs text-gray-500">前後の傾き</label>
            <span className="text-xs font-medium">{rotation.tiltX}°</span>
          </div>
          <Slider
            value={[rotation.tiltX]}
            min={-180}
            max={180}
            step={1}
            onValueChange={([v]) => update({ tiltX: v })}
          />
        </div>

        {/* Tilt Z */}
        <div>
          <div className="mb-1 flex items-center justify-between">
            <label className="text-xs text-gray-500">左右の傾き</label>
            <span className="text-xs font-medium">{rotation.tiltZ}°</span>
          </div>
          <Slider
            value={[rotation.tiltZ]}
            min={-180}
            max={180}
            step={1}
            onValueChange={([v]) => update({ tiltZ: v })}
          />
        </div>

        <Button
          variant="ghost"
          size="sm"
          className="w-full text-xs"
          onClick={() => onChange({ heading: 0, tiltX: 0, tiltZ: 0 })}
          disabled={rotation.heading === 0 && rotation.tiltX === 0 && rotation.tiltZ === 0}
        >
          リセット
        </Button>
      </CardContent>
    </Card>
  );
}
