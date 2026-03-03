import { NextRequest, NextResponse } from "next/server";
import { fetchIssue } from "@/lib/github";

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ number: string }> }
) {
  const { number } = await params;

  try {
    const issue = await fetchIssue(Number(number));
    return NextResponse.json(issue, {
      headers: { "Cache-Control": "s-maxage=30, stale-while-revalidate" },
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}
