"use client";

import { useRef } from "react";

type UploadProps = {
  onFileSelect: (file: File | null) => void;
};

export default function Upload({ onFileSelect }: UploadProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    console.log(file?.name);
    onFileSelect(file);
  };

  return (
    <>
      <input
        type="file"
        accept=".mp3,.wav"
        ref={fileInputRef}
        onChange={handleFileChange}
        className="hidden"
      />

      <button
        type="button"
        onClick={handleClick}
        className="rounded border border-gray-400 px-4 py-2 text-sm text-gray-700 hover:bg-gray-50"
      >
        Upload File
      </button>
    </>
  );
}