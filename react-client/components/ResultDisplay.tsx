import React from "react";

interface ResultDisplayProps {
  result: any;
}

const ResultDisplay: React.FC<ResultDisplayProps> = ({ result }) => {
  return (
    <div className="mt-6 bg-gray-50 rounded-lg p-4 border border-gray-200">
      <h2 className="text-lg font-medium text-gray-900 mb-2">Result:</h2>
      <pre className="bg-white p-3 rounded-md overflow-auto text-sm text-gray-800">
        {JSON.stringify(result, null, 2)}
      </pre>
    </div>
  );
};

export default ResultDisplay;
