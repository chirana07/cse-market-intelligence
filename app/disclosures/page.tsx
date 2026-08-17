'use client';

import { useState } from 'react';
import { ShieldCheck, AlertTriangle, Info, ExternalLink } from 'lucide-react';

export default function DisclosuresRadarPage() {
  const [filter, setFilter] = useState('ALL');

  const announcements = [
    { company: 'John Keells Holdings PLC', symbol: 'JKH.N0000', title: 'FIRST AND FINAL DIVIDEND ANNOUNCEMENT LKR 2.50 PER SHARE', date: '2026-08-15', category: 'Dividend', sentiment: 'BULLISH', note: 'Positive catalyst likely to enhance dividend yield and investor sentiment.' },
    { company: 'Commercial Bank of Ceylon PLC', symbol: 'COMB.N0000', title: 'INTERIM FINANCIAL STATEMENTS FOR THE PERIOD ENDED 30TH JUNE 2026', date: '2026-08-14', category: 'Financial Statements', sentiment: 'BULLISH', note: 'Strong net interest income expansion and asset quality improvement.' },
    { company: 'Ceylon Tobacco Company PLC', symbol: 'CTC.N0000', title: 'PROFIT WARNING FOR QUARTER ENDED MARCH 2026', date: '2026-08-11', category: 'Profit Warning', sentiment: 'BEARISH', note: 'Negative catalyst flag requiring analyst scrutiny on margin compression.' },
    { company: 'Hayleys PLC', symbol: 'HAYL.N0000', title: 'NOTICE OF ANNUAL GENERAL MEETING', date: '2026-08-08', category: 'General', sentiment: 'ROUTINE', note: 'Routine corporate disclosure regarding annual shareholder meeting.' },
  ];

  const filteredAnnouncements = announcements.filter((a) => {
    if (filter === 'BULLISH') return a.sentiment === 'BULLISH';
    if (filter === 'BEARISH') return a.sentiment === 'BEARISH';
    if (filter === 'ROUTINE') return a.sentiment === 'ROUTINE';
    return true;
  });

  return (
    <div className="space-y-6">
      <div className="glass-card p-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <span className="text-xs font-semibold text-sky-400 uppercase tracking-wider">Official CSE Disclosures Feed</span>
          <h1 className="text-2xl font-bold text-white mt-1">AI Disclosure Catalyst Radar</h1>
          <p className="text-xs text-gray-300 mt-1">
            Real-time material event classification with automated Bullish 🟢, Bearish 🔴, and Routine ⚪ impact analysis.
          </p>
        </div>

        <div className="flex items-center space-x-2 bg-gray-900/80 p-1.5 rounded-lg border border-gray-800">
          {['ALL', 'BULLISH', 'BEARISH', 'ROUTINE'].map((type) => (
            <button
              key={type}
              onClick={() => setFilter(type)}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                filter === type
                  ? 'bg-sky-500 text-white font-semibold shadow-md shadow-sky-500/20'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              {type === 'ALL' ? 'All Disclosures' : type === 'BULLISH' ? '🟢 Bullish' : type === 'BEARISH' ? '🔴 Bearish' : '⚪ Routine'}
            </button>
          ))}
        </div>
      </div>

      <div className="space-y-4">
        {filteredAnnouncements.map((ann, idx) => (
          <div key={idx} className="glass-card p-5 space-y-3 hover:border-gray-700 transition-all">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-gray-800/80 pb-2.5">
              <div className="flex items-center space-x-2">
                <span className="font-bold text-white text-base">{ann.company}</span>
                <span className="text-xs font-mono text-gray-400">({ann.symbol})</span>
              </div>
              <span
                className={`inline-flex items-center px-2.5 py-0.5 rounded text-xs font-semibold border ${
                  ann.sentiment === 'BULLISH'
                    ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                    : ann.sentiment === 'BEARISH'
                    ? 'bg-rose-500/10 text-rose-400 border-rose-500/20'
                    : 'bg-gray-500/10 text-gray-400 border-gray-500/20'
                }`}
              >
                {ann.sentiment === 'BULLISH' ? '🟢 BULLISH CATALYST' : ann.sentiment === 'BEARISH' ? '🔴 BEARISH RISK FLAG' : '⚪ ROUTINE'}
              </span>
            </div>

            <p className="text-sm font-semibold text-gray-200 leading-snug">{ann.title}</p>

            <div className="bg-gray-900/60 p-3 rounded-lg border border-gray-800/80 flex items-start space-x-2.5 text-xs text-sky-300">
              <Info className="w-4 h-4 text-sky-400 shrink-0 mt-0.5" />
              <span><strong>AI Catalyst Note:</strong> {ann.note}</span>
            </div>

            <div className="flex items-center justify-between text-xs text-gray-400 pt-1">
              <span>Category: {ann.category} · Date: {ann.date}</span>
              <a
                href="https://www.cse.lk/api/announcements"
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center space-x-1 text-sky-400 hover:underline"
              >
                <span>Open Disclosure Filing</span>
                <ExternalLink className="w-3 h-3" />
              </a>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
