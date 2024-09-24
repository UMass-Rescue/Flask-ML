import Link from "next/link";

export default function About() {
  return (
    <div className="min-h-screen p-4 sm:p-6 lg:p-8">
      <div className="max-w-3xl mx-auto bg-white shadow-xl rounded-lg overflow-hidden">
        <div className="bg-gradient-to-r from-green-500 to-teal-600 px-6 py-4">
          <h1 className="text-2xl font-bold text-white">
            About ML Server Client
          </h1>
        </div>
        <div className="p-6 space-y-4">
          <p className="text-gray-700">
            This application serves as a client for interacting with a Machine
            Learning server. It allows users to input data and parameters, send
            requests to the ML server, and display the results.
          </p>
          <p className="text-gray-700">
            Built with Next.js and styled with Tailwind CSS, this app
            demonstrates how to create a modern, responsive web application that
            interfaces with a backend ML service.
          </p>
          <Link
            href="/"
            className="inline-block px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 transition-colors"
          >
            Back to home
          </Link>
        </div>
      </div>
    </div>
  );
}
