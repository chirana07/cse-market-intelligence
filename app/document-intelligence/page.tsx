'use client';

import { useState } from 'react';
import { FileText, Download, CheckCircle2, Sparkles, AlertTriangle } from 'lucide-react';

export default function DocumentIntelligencePage() {
  const [reportUrl, setReportUrl] = useState('');
  const [generating, setGenerating] = useState(false);
  const [tearSheet, setTearSheet] = useState<any>(null);

  const handleGenerate = () => {
    setGenerating(true);
    setTimeout(() => {
      setTearSheet({
        company: 'John Keells Holdings PLC (JKH.N0000)',
        verdict: 'Turned Profitable & Margin Expansion',
        revenue: 'LKR 48.20 Billion (+18.5% YoY)',
        pat: 'LKR 6.80 Billion (+24.2% YoY)',
        operatingMargin: '14.50% (+240 bps)',
        summary: [
          'Strong top-line expansion driven by port operations, bunkering recovery, and leisure property sales.',
          'Operating profit expanded from LKR 4.90 Bn to LKR 6.98 Bn, showcasing operational leverage.',
          'Borrowings reduced by LKR 2.10 Bn, improving debt-to-equity ratio to 0.42x.',
        ],
      });
      setGenerating(false);
    }, 800);
  };

  return (
    <div className="space-y-6">
      <div className="glass-card p-6 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <span className="text-xs font-semibold text-sky-400 uppercase tracking-wider">Document & PDF Table Parsing</span>
          <h1 className="text-2xl font-bold text-white mt-1">Interim Report Key Figures & 1-Click Tear Sheet</h1>
          <p className="text-xs text-gray-300 mt-1">
            Extract IAS 34 financial statements, key ratios, and YoY period comparisons instantly from PDF annual/quarterly reports.
          </p>
        </div>

        <button
          onClick={handleGenerate}
          disabled={generating}
          className="inline-flex items-center space-x-2 bg-gradient-to-r from-sky-500 to-indigo-600 hover:from-sky-400 hover:to-indigo-500 text-white font-medium px-5 py-2.5 rounded-lg text-sm transition-all shadow-lg shadow-sky-500/20 disabled:opacity-50"
        >
          <Sparkles className="w-4 h-4" />
          <span>{generating ? 'Processing PDF Report...' : '⚡ Generate Executive Tear Sheet'}</span>
        </button>
      </div>

      {tearSheet && (
        <div className="glass-card p-6 space-y-6 border-sky-500/30">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-gray-800 pb-4">
            <div>
              <div className="flex items-center space-x-2">
                <span className="px-2.5 py-0.5 rounded text-xs font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                  {tearSheet.verdict}
                </span>
              </div>
              <h2 className="text-xl font-bold text-white mt-2">{tearSheet.company}</h2>
            </div>

            <button
              onClick={() => {
                const blob = new Blob([JSON.stringify(tearSheet, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'Executive_Tear_Sheet.json';
                a.click();
              }}
              className="inline-flex items-center space-x-1.5 text-xs font-semibold bg-gray-800 hover:bg-gray-700 text-gray-200 border border-gray-700 px-3 py-2 rounded-lg transition-colors"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Download Executive Summary</span>
            </button>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="bg-gray-900/60 p-4 rounded-lg border border-gray-800">
              <span className="text-xs text-gray-400">Quarterly Revenue</span>
              <span className="text-lg font-bold text-white block mt-1">{tearSheet.revenue}</span>
            </div>
            <div className="bg-gray-900/60 p-4 rounded-lg border border-gray-800">
              <span className="text-xs text-gray-400">Profit After Tax (PAT)</span>
              <span className="text-lg font-bold text-white block mt-1">{tearSheet.pat}</span>
            </div>
            <div className="bg-gray-900/60 p-4 rounded-lg border border-gray-800">
              <span className="text-xs text-gray-400">Operating Margin</span>
              <span className="text-lg font-bold text-white block mt-1">{tearSheet.operatingMargin}</span>
            </div>
          </div>

          <div className="space-y-3">
            <h3 className="text-sm font-bold text-white">Executive Analyst Takeaways</h3>
            <ul className="space-y-2">
              {tearSheet.summary.map((point: string, idx: number) => (
                <li key={idx} className="flex items-start space-x-2 text-xs text-gray-300">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                  <span>{point}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}
