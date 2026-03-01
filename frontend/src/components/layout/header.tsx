"use client";

import { usePathname } from "next/navigation";

const PAGE_TITLES: Record<string, string> = {
  "/dashboard": "Dashboard",
  "/viewer": "3Dビューア",
  "/simulation": "シミュレーション",
  "/analysis": "AI分析",
  "/reports": "レポート",
};

export function Header() {
  const pathname = usePathname();
  const title = PAGE_TITLES[pathname] || "ExaSense";

  return (
    <header className="sticky top-0 z-40 flex h-14 items-center border-b bg-white/80 px-6 backdrop-blur">
      <h2 className="text-lg font-semibold text-gray-900">{title}</h2>
    </header>
  );
}
