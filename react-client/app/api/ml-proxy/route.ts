import { NextResponse } from "next/server";
import MLServerAPI from "../../../lib/MLServerAPI";

const api = new MLServerAPI(
  process.env.ML_SERVER_URL || "http://localhost:5000"
);

export async function POST(request: Request) {
  try {
    const { endpoint, inputType, inputs, parameters } = await request.json();
    const result = await api.sendRequest(
      endpoint,
      inputType,
      inputs,
      parameters
    );
    return NextResponse.json(result);
  } catch (error) {
    console.error("ML Server Error:", error);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}
