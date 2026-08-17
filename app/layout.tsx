import './globals.css';
import type { Metadata } from 'next';
import Link from 'next/link';

export const metadata: Metadata = {
  title: 'CSE Market Intelligence — Colombo Stock Exchange Platform',
  description: 'Enterprise Equity Intelligence, Financial Statement Parsing, Official CSE Disclosures, and AI Copilot for the Colombo Stock Exchange.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-[#0b0f19] text-gray-100 min-h-screen flex flex-col antialiased">
        <header className="border-b border-gray-800 bg-[#111827]/90 sticky top-0 z-50 backdrop-blur-md">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-sky-500 to-indigo-600 flex items-center justify-center font-bold text-white shadow-md shadow-sky-500/20">
                CSE
              </div>
              <span className="font-semibold text-lg tracking-tight gradient-text">
                CSE Market Intelligence
              </span>
            </div>
            
            <nav className="hidden md:flex items-center space-x-1">
              <Link href="/" className="px-3 py-2 rounded-md text-sm font-medium text-gray-300 hover:text-white hover:bg-gray-800/60 transition-colors">
                Command Center
              </Link>
              <Link href="/research" className="px-3 py-2 rounded-md text-sm font-medium text-gray-300 hover:text-white hover:bg-gray-800/60 transition-colors">
                Stock Research
              </Link>
              <Link href="/document-intelligence" className="px-3 py-2 rounded-md text-sm font-medium text-gray-300 hover:text-white hover:bg-gray-800/60 transition-colors">
                Document Intelligence
              </Link>
              <Link href="/disclosures" className="px-3 py-2 rounded-md text-sm font-medium text-gray-300 hover:text-white hover:bg-gray-800/60 transition-colors">
                Disclosures Radar
              </Link>
              <Link href="/copilot" className="px-3 py-2 rounded-md text-sm font-medium text-gray-300 hover:text-white hover:bg-gray-800/60 transition-colors">
                Analyst Copilot
              </Link>
              <Link href="/portfolio" className="px-3 py-2 rounded-md text-sm font-medium text-gray-300 hover:text-white hover:bg-gray-800/60 transition-colors">
                Portfolio
              </Link>
            </nav>

            <div className="flex items-center space-x-2">
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                ● Live Engine
              </span>
            </div>
          </div>
        </header>

        <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
          {children}
        </main>

        <footer className="border-t border-gray-800 bg-[#0f172a]/50 py-6 text-center text-xs text-gray-400">
          CSE Market Intelligence Platform · Sourced from Official Colombo Stock Exchange Filing Systems & Financial Statement Intelligence.
        </footer>
      </body>
    </html>
  );
}
