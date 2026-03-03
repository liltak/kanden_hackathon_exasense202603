import { NextResponse } from "next/server";
import { fetchMilestones } from "@/lib/github";

export async function GET() {
  try {
    const milestones = await fetchMilestones();
    return NextResponse.json(milestones, {
      headers: { "Cache-Control": "s-maxage=300, stale-while-revalidate" },
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}
