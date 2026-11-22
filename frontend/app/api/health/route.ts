import { NextResponse } from "next/server";

export async function GET() {
  return NextResponse.json({ status: "online", service: "cricoracle-frontend-api" });
}

export async function HEAD() {
  return new Response(null, { status: 200 });
}

export async function POST() {
  return NextResponse.json({ status: "online", service: "cricoracle-frontend-api" });
}

export async function OPTIONS() {
  return new Response(null, { status: 200 });
}
