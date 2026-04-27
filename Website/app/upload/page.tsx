"use client";

import { useState } from "react";
import AnalyzeButton from "./components/AnalyzeButton";
import Upload from "./components/Upload";

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);

  return (
    <main className="min-h-screen bg-white text-black flex flex-col">
      <section className="flex-1 flex items-start justify-center pt-24">
        <div className="w-full max-w-4xl border-2 border-black px-6 py-16">
          <div className="max-w-2xl mx-auto text-center px-8 py-10">
            <h1 className="text-4xl font-bold tracking-tight">
              Upload Music File
            </h1>

            <div className="mt-6 flex items-center justify-center gap-3">
              <Upload onFileSelect={setSelectedFile} />

              <AnalyzeButton
                file={selectedFile}
                onResult={(res) => {
                  setResult(res);
                  setLoading(false);
                }}
                setLoading={setLoading}
              />

              {/* 🔥 Loading GIF */}
              {loading && (
                <img
                  src="https://media.tenor.com/On7kvXhzml4AAAAj/loading-gif.gif"
                  alt="loading"
                  className="w-8 h-8"
                />
              )}
            </div>
              {selectedFile && (
                <p className="mt-4 text-sm text-gray-600">
                  Selected file: {selectedFile.name}
                </p>
              )}
            {result && (
              <pre className="mt-6 text-left font-mono whitespace-pre-wrap">
                {result}
              </pre>
            )}
          </div>
        </div>
      </section>
    </main>
  );
}