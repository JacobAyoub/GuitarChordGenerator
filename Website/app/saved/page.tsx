"use client";

import { useState } from "react";

export default function SavedPage() {
  const [savedFiles, setSavedFiles] = useState([
    { song: "Dreams", artist: "Fleetwood Mac", favorite: false },
    { song: "Yellow", artist: "Coldplay", favorite: true },
    { song: "Let It Be", artist: "The Beatles", favorite: false },
    { song: "Skinny Love", artist: "Bon Iver", favorite: false },
  ]);

  const toggleFavorite = (index: number) => {
    const updated = [...savedFiles];
    updated[index].favorite = !updated[index].favorite;
    setSavedFiles(updated);
  };

  return (
    <main className="min-h-screen bg-white px-6 py-10">
      <div className="mx-auto max-w-3xl">
        <h1 className="mb-8 text-3xl font-semibold text-black">Saved Files</h1>

        <div className="flex flex-col gap-4">
          {savedFiles.map((file, index) => (
            <div
              key={index}
              className="flex items-start gap-4 border-b border-gray-200 pb-4"
            >
              <button
                onClick={() => toggleFavorite(index)}
                className={`mt-1 text-2xl leading-none ${
                  file.favorite ? "text-yellow-400" : "text-gray-200"
                }`}
                aria-label="Toggle favorite"
              >
                ★
              </button>

              <div>
                <p className="text-lg font-medium text-black">{file.song}</p>
                <p className="mt-1 text-sm text-gray-500">{file.artist}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </main>
  );
}