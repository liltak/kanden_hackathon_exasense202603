import { NextRequest, NextResponse } from "next/server";
import { fetchPulls } from "@/lib/github";

export async function GET(request: NextRequest) {
  const { searchParams } = request.nextUrl;
  const state = searchParams.get("state") || "all";

  try {
    const pulls = await fetchPulls({ state });
    return NextResponse.json(pulls, {
      headers: { "Cache-Control": "s-maxage=120, stale-while-revalidate" },
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}
