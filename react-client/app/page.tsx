import MLClient from "../components/MLClient";

export default function Home() {
  return (
    <div className="min-h-screen p-4 sm:p-6 lg:p-8">
      <div className="max-w-3xl mx-auto bg-white shadow-xl rounded-lg overflow-hidden">
        <div className="bg-gradient-to-r from-blue-500 to-indigo-600 px-6 py-4">
          <h1 className="text-2xl font-bold text-white">ML Server Client</h1>
        </div>
        <div className="p-6">
          <MLClient />
        </div>
      </div>
    </div>
  );
}
