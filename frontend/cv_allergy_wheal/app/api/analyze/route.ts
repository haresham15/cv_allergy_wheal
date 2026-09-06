import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
  try {
    const backendUrl =
      process.env.BACKEND_API_URL ||
      process.env.NEXT_PUBLIC_API_URL ||
      "http://localhost:8000";

    const targetUrl = `${backendUrl.replace(/\/$/, "")}/api/v1/analyze`;

    // Forward the formData to the backend API
    const formData = await request.formData();

    const backendResponse = await fetch(targetUrl, {
      method: "POST",
      body: formData,
    });

    const data = await backendResponse.json();

    if (!backendResponse.ok) {
      return NextResponse.json(
        { detail: data.detail || `Backend error (${backendResponse.status})` },
        { status: backendResponse.status }
      );
    }

    return NextResponse.json(data);
  } catch (error: any) {
    console.error("[Proxy /api/analyze error]:", error);
    return NextResponse.json(
      { detail: error.message || "Failed to communicate with analysis backend." },
      { status: 502 }
    );
  }
}
