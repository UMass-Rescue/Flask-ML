"use client";

import React, { useState } from "react";
import MLForm from "./MLForm";
import ResultDisplay from "./ResultDisplay";
import MLServerAPI from "../lib/MLServerAPI";
import Link from "next/link";

const api = new MLServerAPI(
  process.env.NEXT_PUBLIC_ML_SERVER_URL || "http://localhost:5000"
);

const MLClient: React.FC = () => {
  const [result, setResult] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (inputs: string, parameters: string) => {
    setIsLoading(true);
    try {
      const parsedInputs = JSON.parse(inputs);
      const parsedParameters = parameters ? JSON.parse(parameters) : {};
      const response = await api.sendRequest(
        "/predict",
        "json",
        parsedInputs,
        parsedParameters
      );
      setResult(response);
    } catch (error) {
      console.error("Error:", error);
      setResult({
        error:
          error instanceof Error ? error.message : "An unknown error occurred",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <MLForm onSubmit={handleSubmit} inputType="json" isLoading={isLoading} />
      {result && <ResultDisplay result={result} />}
      <div>
        <Link href="/about" className="text-indigo-600 hover:text-indigo-800">
          Learn more about this app
        </Link>
      </div>
    </div>
  );
};

export default MLClient;
