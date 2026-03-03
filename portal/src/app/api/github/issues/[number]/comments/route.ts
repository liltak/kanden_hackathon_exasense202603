import { NextRequest, NextResponse } from "next/server";
import { fetchIssueComments, postIssueComment } from "@/lib/github";

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ number: string }> }
) {
  const { number } = await params;

  try {
    const comments = await fetchIssueComments(Number(number));
    return NextResponse.json(comments, {
      headers: { "Cache-Control": "s-maxage=30, stale-while-revalidate" },
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ number: string }> }
) {
  const { number } = await params;
  const { name, body } = await request.json();

  if (!body || !name) {
    return NextResponse.json(
      { error: "name and body are required" },
      { status: 400 }
    );
  }

  const formattedBody = `**投稿者: ${name}**\n\n${body}`;

  try {
    const comment = await postIssueComment(Number(number), formattedBody);
    return NextResponse.json(comment, { status: 201 });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}
