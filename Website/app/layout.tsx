import AboutLink from "./upload/components/AnalyzeButton";
import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import Link from "next/link";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Guitar Chords",
  description: "Upload file and get guitar chords!",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable}`}>
        
        {/* NAV */}
        <nav className="w-full bg-white border-b border-gray-200">
          <div className="max-w-7xl mx-auto h-16 px-6 flex items-center justify-between">
            
            {/* LEFT */}
            <div className="flex items-center gap-10">
              <Link href="/" className="text-sm font-semibold text-black">
                Guitar
              </Link>

              <div className="flex items-center gap-6 text-xs text-black">
                <Link href="/upload">Upload</Link>
                <Link href="/saved">Saved</Link>
                <Link href="">About</Link>
              </div>
            </div>

            {/* RIGHT */}
            <div className="flex items-center gap-4">
              <select className="text-xs border border-gray-300 rounded px-2 py-1 bg-white text-black">
                <option>Light</option>
                <option>Dark</option>
              </select>

              <div className="w-8 h-8 rounded-md bg-blue-600" />
            </div>

          </div>
        </nav>

        {/* PAGE CONTENT */}
        {children}

      </body>
    </html>
  );
}