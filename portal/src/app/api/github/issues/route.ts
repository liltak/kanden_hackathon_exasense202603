import { NextRequest, NextResponse } from "next/server";
import { fetchIssues, createIssue } from "@/lib/github";

export async function GET(request: NextRequest) {
  const { searchParams } = request.nextUrl;
  const state = searchParams.get("state") || "all";
  const labels = searchParams.get("labels") || undefined;
  const page = Number(searchParams.get("page")) || 1;

  try {
    const issues = await fetchIssues({ state, labels, page });
    return NextResponse.json(issues, {
      headers: { "Cache-Control": "s-maxage=60, stale-while-revalidate" },
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  const { title, body, name, labels } = await request.json();

  if (!title || !name) {
    return NextResponse.json(
      { error: "title and name are required" },
      { status: 400 }
    );
  }

  const formattedBody = `**投稿者: ${name}**\n\n${body || ""}`;

  try {
    const issue = await createIssue({
      title,
      body: formattedBody,
      labels: labels || [],
    });
    return NextResponse.json(issue, { status: 201 });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}
