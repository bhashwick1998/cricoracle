import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

const STATIC_CACHE_PATHS = new Set([
  "/api/model-accuracy",
  "/api/parameters/optimized",
  "/api/top-performers-model/accuracy",
  "/api/score-parameters/optimization-benchmark",
  "/api/models/learning-ledger",
]);

function resolveBackendBaseUrl(): string {
  const raw = (
    process.env.BACKEND_URL ||
    process.env.NEXT_PUBLIC_API_URL ||
    "http://127.0.0.1:8000"
  ).trim();
  const withProto =
    raw.startsWith("http://") || raw.startsWith("https://")
      ? raw
      : `${raw.startsWith("localhost") || raw.startsWith("127.0.0.1") ? "http" : "https"}://${raw}`;
  return withProto.endsWith("/") ? withProto.slice(0, -1) : withProto;
}

function buildCacheHeaders(apiPath: string, req: NextRequest, status: number): Record<string, string> {
  if (
    status !== 200 ||
    req.method !== "GET" ||
    apiPath.startsWith("/api/system/") ||
    apiPath === "/api/health" ||
    req.nextUrl.searchParams.get("force")?.toLowerCase() === "true"
  ) {
    return {
      "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
    };
  }

  const isStatic =
    STATIC_CACHE_PATHS.has(apiPath) ||
    apiPath.startsWith("/api/analytics/") ||
    apiPath.endsWith("/static") ||
    req.nextUrl.searchParams.get("view") === "static";

  if (isStatic) {
    return {
      "Cache-Control": "public, max-age=60, s-maxage=300, stale-while-revalidate=300",
      "CDN-Cache-Control": "public, s-maxage=300, stale-while-revalidate=300",
      "Vercel-CDN-Cache-Control": "public, s-maxage=300, stale-while-revalidate=300",
    };
  }

  return {
    "Cache-Control": "public, max-age=5, s-maxage=15, stale-while-revalidate=15",
    "CDN-Cache-Control": "public, s-maxage=15, stale-while-revalidate=15",
    "Vercel-CDN-Cache-Control": "public, s-maxage=15, stale-while-revalidate=15",
  };
}

async function proxyRequest(
  req: NextRequest,
  context: { params: Promise<{ path: string[] }> }
): Promise<NextResponse> {
  const { path } = await context.params;
  const apiPath = `/api/${(path || []).join("/")}`;
  const search = req.nextUrl.search || "";
  const upstreamUrl = `${resolveBackendBaseUrl()}${apiPath}${search}`;

  const upstreamHeaders: Record<string, string> = {
    Accept: req.headers.get("accept") || "application/json",
  };
  const contentType = req.headers.get("content-type");
  if (contentType && req.method !== "GET" && req.method !== "HEAD") {
    upstreamHeaders["Content-Type"] = contentType;
  }

  const init: RequestInit = {
    method: req.method,
    headers: upstreamHeaders,
    cache: "no-store",
  };

  if (req.method !== "GET" && req.method !== "HEAD") {
    init.body = await req.arrayBuffer();
  }

  try {
    const upstreamRes = await fetch(upstreamUrl, init);
    const body = await upstreamRes.arrayBuffer();

    const responseHeaders: Record<string, string> = {
      ...buildCacheHeaders(apiPath, req, upstreamRes.status),
    };
    const resContentType = upstreamRes.headers.get("content-type");
    if (resContentType) {
      responseHeaders["Content-Type"] = resContentType;
    }
    const etag = upstreamRes.headers.get("etag");
    if (etag && upstreamRes.status === 200 && req.method === "GET") {
      responseHeaders["ETag"] = etag;
    }

    return new NextResponse(body, {
      status: upstreamRes.status,
      headers: responseHeaders,
    });
  } catch (err) {
    return NextResponse.json(
      { error: "Upstream backend unavailable", detail: String(err) },
      { status: 502, headers: { "Cache-Control": "no-store" } }
    );
  }
}

export async function GET(req: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  return proxyRequest(req, context);
}

export async function POST(req: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  return proxyRequest(req, context);
}

export async function PUT(req: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  return proxyRequest(req, context);
}

export async function PATCH(req: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  return proxyRequest(req, context);
}

export async function DELETE(req: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  return proxyRequest(req, context);
}
