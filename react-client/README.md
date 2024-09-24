# ML Server Client

This project is a Next.js-based web application that serves as a client for interacting with a Machine Learning server. It allows users to input data and parameters, send requests to the ML server, and display the results.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Setup](#setup)
3. [Running the Application](#running-the-application)
4. [Project Structure](#project-structure)
5. [Extending Abstract Classes](#extending-abstract-classes)

## Prerequisites

- Node.js (v14 or later)
- npm (v6 or later)

## Setup

1. Clone the repository:

   git clone <repository-url>
   cd ml-server-client

2. Install dependencies:

   npm install

3. Create a .env.local file in the root directory and add the following:

   NEXT_PUBLIC_ML_SERVER_URL=http://localhost:5000
   ML_SERVER_URL=http://localhost:5000

   Adjust the URLs if your ML server is hosted elsewhere.

## Running the Application

1. Start the development server:

   npm run dev

2. Open [http://localhost:3000](http://localhost:3000) in your browser to see the application.

## Project Structure

ml-client/
├── app/
│ ├── about/
│ │ └── page.tsx
│ ├── api/
│ │ └── ml-proxy/
│ │ └── route.ts
│ ├── layout.tsx
│ └── page.tsx
├── components/
│ ├── MLClient.tsx
│ ├── MLForm.tsx
│ └── ResultDisplay.tsx
├── lib/
│ └── MLServerAPI.ts
├── styles/
│ └── globals.css
├── .env.local
├── next.config.mjs
├── postcss.config.mjs
├── tailwind.config.js
└── tsconfig.json

## Extending Abstract Classes

The main abstract class in this project is MLServerAPI in lib/MLServerAPI.ts. To extend or modify this class:

1. Open lib/MLServerAPI.ts

2. The current implementation looks like this:

   typescript
   export default class MLServerAPI {
   private baseURL: string;

   constructor(baseURL: string = 'http://localhost:5000') {
   this.baseURL = baseURL;
   }

   async sendRequest(endpoint: string, inputType: string, inputs: any, parameters: any = {}): Promise<any> {
   // Implementation here
   }
   }

3. To add new methods or modify existing ones, you can extend this class:

   typescript
   class ExtendedMLServerAPI extends MLServerAPI {
   async newMethod(data: any): Promise<any> {
   // Implement new functionality
   }

   async sendRequest(endpoint: string, inputType: string, inputs: any, parameters: any = {}): Promise<any> {
   // Override the existing method if needed
   // Don't forget to call super.sendRequest() if you want to keep the original functionality
   }
   }

4. To use the extended class, update the import in components/MLClient.tsx:

   typescript
   import ExtendedMLServerAPI from '../lib/MLServerAPI';

   const api = new ExtendedMLServerAPI(process.env.NEXT_PUBLIC_ML_SERVER_URL || 'http://localhost:5000');

5. Remember to update any relevant types or interfaces if you modify the class structure.
