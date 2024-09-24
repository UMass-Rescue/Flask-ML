import React, { useState } from "react";

interface MLFormProps {
  onSubmit: (inputs: string, parameters: string) => void;
  inputType: string;
  isLoading: boolean;
}

const MLForm: React.FC<MLFormProps> = ({ onSubmit, inputType, isLoading }) => {
  const [inputs, setInputs] = useState("");
  const [parameters, setParameters] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(inputs, parameters);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label
          htmlFor="inputs"
          className="block text-sm font-medium text-gray-700"
        >
          Inputs ({inputType}):
        </label>
        <textarea
          id="inputs"
          value={inputs}
          onChange={(e) => setInputs(e.target.value)}
          required
          className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm"
          rows={4}
        />
      </div>
      <div>
        <label
          htmlFor="parameters"
          className="block text-sm font-medium text-gray-700"
        >
          Parameters (optional):
        </label>
        <textarea
          id="parameters"
          value={parameters}
          onChange={(e) => setParameters(e.target.value)}
          className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm"
          rows={4}
        />
      </div>
      <button
        type="submit"
        disabled={isLoading}
        className={`w-full py-2 px-4 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 ${
          isLoading ? "opacity-50 cursor-not-allowed" : ""
        }`}
      >
        {isLoading ? "Processing..." : "Submit"}
      </button>
    </form>
  );
};

export default MLForm;
