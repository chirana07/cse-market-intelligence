'use client';

import { useState } from 'react';
import { Search, Building2, FileCheck, AlertCircle, ArrowUpRight } from 'lucide-react';

export default function StockResearchPage() {
  const [selectedSymbol, setSelectedSymbol] = useState('JKH.N0000');

  const companies = [
    { symbol: 'JKH.N0000', name: 'John Keells Holdings PLC', sector: 'Capital Goods' },
    { symbol: 'COMB.N0000', name: 'Commercial Bank of Ceylon PLC', sector: 'Banking & Financial Services' },
    { symbol: 'DIAL.N0000', name: 'Dialog Axiata PLC', sector: 'Telecommunication' },
    { symbol: 'HAYL.N0000', name: 'Hayleys PLC', sector: 'Conglomerates' },
    { symbol: 'SAMP.N0000', name: 'Sampath Bank PLC', sector: 'Banking & Financial Services' },
  ];

  const activeCompany = companies.find((c) => c.symbol === selectedSymbol) || companies[0];

  return (
    <div className="space-y-6">
      {/* Top Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 glass-card p-6">
        <div>
          <span className="text-xs font-semibold uppercase tracking-wider text-sky-400">Equity Research Workstation</span>
          <h1 className="text-2xl font-bold text-white mt-1">{activeCompany.name}</h1>
          <p className="text-xs text-gray-400 mt-0.5">Listed Ticker: <span className="font-mono text-white">{activeCompany.symbol}</span> · Sector: <span className="text-gray-300">{activeCompany.sector}</span></p>
        </div>

        <div className="w-full sm:w-72">
          <label className="text-xs font-medium text-gray-400 mb-1 block">Select CSE Listed Company</label>
          <div className="relative">
            <select
              value={selectedSymbol}
              onChange={(e) => setSelectedSymbol(e.target.value)}
              className="w-full bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-sky-500 appearance-none pr-8"
            >
              {companies.map((c) => (
                <option key={c.symbol} value={c.symbol}>
                  {c.name} ({c.symbol})
                </option>
              ))}
            </select>
            <Search className="w-4 h-4 text-gray-400 absolute right-2.5 top-2.5 pointer-events-none" />
          </div>
        </div>
      </div>

      {/* Grid Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Card 1: Latest Disclosure */}
        <div className="glass-card p-6 space-y-4">
          <div className="flex items-center justify-between border-b border-gray-800 pb-3">
            <div className="flex items-center space-x-2">
              <Building2 className="w-4 h-4 text-sky-400" />
              <h2 className="font-bold text-white">Latest Material Disclosure</h2>
            </div>
            <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              High Priority
            </span>
          </div>

          <div className="space-y-2">
            <span className="text-xs text-gray-400">August 15, 2026 · Category: Dividend</span>
            <p className="text-sm font-semibold text-white">
              FIRST AND FINAL DIVIDEND ANNOUNCEMENT LKR 2.50 PER SHARE
            </p>
            <p className="text-xs text-gray-300 leading-relaxed">
              The Board of Directors has recommended a first and final dividend of LKR 2.50 per ordinary share for the financial year ended 31st March 2026, subject to shareholder approval at the upcoming AGM.
            </p>
          </div>

          <div className="pt-2">
            <a
              href="https://www.cse.lk/api/announcements"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center space-x-1.5 text-xs font-semibold text-sky-400 hover:text-sky-300"
            >
              <span>View Official Filing Attachment</span>
              <ArrowUpRight className="w-3.5 h-3.5" />
            </a>
          </div>
        </div>

        {/* Card 2: Interim Financial Key Figures */}
        <div className="glass-card p-6 space-y-4">
          <div className="flex items-center justify-between border-b border-gray-800 pb-3">
            <div className="flex items-center space-x-2">
              <FileCheck className="w-4 h-4 text-indigo-400" />
              <h2 className="font-bold text-white">Financial Statement Key Figures</h2>
            </div>
            <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
              IAS 34 Verified
            </span>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="bg-gray-900/60 p-3 rounded-lg border border-gray-800">
              <span className="text-xs text-gray-400 block">Revenue (Q3)</span>
              <span className="text-lg font-bold text-white">LKR 48.20 Bn</span>
              <span className="text-[10px] text-emerald-400 block mt-0.5">+18.5% YoY Growth</span>
            </div>
            <div className="bg-gray-900/60 p-3 rounded-lg border border-gray-800">
              <span className="text-xs text-gray-400 block">Net Profit (PAT)</span>
              <span className="text-lg font-bold text-white">LKR 6.80 Bn</span>
              <span className="text-[10px] text-emerald-400 block mt-0.5">+24.2% YoY Expansion</span>
            </div>
            <div className="bg-gray-900/60 p-3 rounded-lg border border-gray-800">
              <span className="text-xs text-gray-400 block">Operating Margin</span>
              <span className="text-lg font-bold text-white">14.50%</span>
              <span className="text-[10px] text-sky-400 block mt-0.5">+240 bps Expansion</span>
            </div>
            <div className="bg-gray-900/60 p-3 rounded-lg border border-gray-800">
              <span className="text-xs text-gray-400 block">Return on Equity</span>
              <span className="text-lg font-bold text-white">15.20%</span>
              <span className="text-[10px] text-gray-400 block mt-0.5">Annualized</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
