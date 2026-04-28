import Link from "next/link";

export default function Home() {
  return (
    <main className="min-h-screen bg-white text-white flex flex-col">
      {/* CONTENT */}
      <section className="flex-1 flex flex-col items-center justify-start text-center px-6 pt-20">
        <h1 className="text-4xl text-black md:text-5xl font-bold">
          Guitar Chords
        </h1>

        <p className="mt-4 text-gray-400 max-w-md">
          {/* YOU CAN EDIT THIS */}
          Upload any song and instantly get the chords.
        </p>
      </section>  
    </main>
  );
}