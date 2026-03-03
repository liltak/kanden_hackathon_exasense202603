import { Badge } from "@/components/ui/badge";
import type { GitHubLabel } from "@/lib/types";

function getContrastColor(hexColor: string): string {
  const r = parseInt(hexColor.slice(0, 2), 16);
  const g = parseInt(hexColor.slice(2, 4), 16);
  const b = parseInt(hexColor.slice(4, 6), 16);
  const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  return luminance > 0.5 ? "#000000" : "#ffffff";
}

export function LabelBadge({ label }: { label: GitHubLabel }) {
  return (
    <Badge
      variant="outline"
      style={{
        backgroundColor: `#${label.color}`,
        color: getContrastColor(label.color),
        borderColor: `#${label.color}`,
      }}
      className="text-xs"
    >
      {label.name}
    </Badge>
  );
}
