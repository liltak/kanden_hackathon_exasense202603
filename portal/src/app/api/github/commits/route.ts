import { NextRequest, NextResponse } from "next/server";
import { fetchCommits } from "@/lib/github";

export async function GET(request: NextRequest) {
  const { searchParams } = request.nextUrl;
  const since = searchParams.get("since") || undefined;
  const per_page = Number(searchParams.get("per_page")) || 30;

  try {
    const commits = await fetchCommits({ since, per_page });
    return NextResponse.json(commits, {
      headers: { "Cache-Control": "s-maxage=120, stale-while-revalidate" },
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}
