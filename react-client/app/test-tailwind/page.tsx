export default function TestTailwind() {
  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center">
      <div className="bg-white p-8 rounded-lg shadow-md">
        <h1 className="text-3xl font-bold text-blue-600 mb-4">Tailwind Test</h1>
        <p className="text-gray-700">
          If you can see this styled text, Tailwind is working!
        </p>
        <button className="mt-4 px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600">
          Test Button
        </button>
      </div>
    </div>
  );
}
