"use client";

type Props = {
  file: File | null;
  onResult: (result: string) => void;
  setLoading: (loading: boolean) => void;
};

export default function AnalyzeButton({
  file,
  onResult,
  setLoading,
}: Props) {
  const handleAnalyze = async () => {
    if (!file) return;

    setLoading(true); // 🔥 start loading

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:8000/analyze", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      onResult(data.result);
    } catch (err) {
      console.error(err);
      setLoading(false);
    }
  };

  return (
    <button
      onClick={handleAnalyze}
      className="rounded border border-sky-400 px-4 py-2 text-sm text-gray-700 hover:bg-gray-50"
    >
      Analyze
    </button>
  );
}