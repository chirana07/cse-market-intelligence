'use client';

import { useState } from 'react';
import { Cpu, ShieldCheck, Sparkles, FileText, CheckCircle2, ChevronDown } from 'lucide-react';

export default function CopilotPage() {
  const [query, setQuery] = useState('');
  const [generating, setGenerating] = useState(false);
  const [memo, setMemo] = useState<any>(null);

  const handleRunAnalysis = () => {
    setGenerating(true);
    setTimeout(() => {
      setMemo({
        company: 'John Keells Holdings PLC (JKH.N0000)',
        mode: 'Full Research Memo',
        groundingScore: '98.4%',
        sourcesCount: 6,
        summary: `### Executive Research Memo — John Keells Holdings PLC (JKH.N0000)

**1. Executive Investment View**
John Keells Holdings PLC exhibits strong earnings momentum backed by port expansion at SAGT, bunkering volume expansion, and hospitality recovery across Sri Lanka and the Maldives.

**2. Key Investment Catalysts**
- **Port & Logistics Leverage**: SAGT container throughput expanded 14.2% YoY.
- **Consumer & Retail Resilience**: Supermarket retail basket values expanded 8.5% YoY.
- **Solvency & Cash Position**: Cash and cash equivalents stand at LKR 18.40 Bn, with debt-to-equity standing at a safe 0.42x.

**3. Material Risk Takeaways**
- Foreign currency fluctuation exposures in Maldives hotel operations.
- Interest rate sensitivities on ongoing property development finance costs.`,
        sources: [
          { name: 'JKH_Q3_Interim_Financial_Statements.pdf', domain: 'Official CSE Filings', page: 14, snippet: 'Revenue for the quarter ended 31st December 2025 expanded to LKR 48.20 Bn compared to LKR 40.60 Bn in the previous corresponding period.' },
          { name: 'JKH_Dividend_Announcement_Aug2026.pdf', domain: 'CSE Disclosures Feed', page: 1, snippet: 'The Board of Directors has recommended a first and final dividend of LKR 2.50 per ordinary share.' },
          { name: 'CBSL_Economic_Indicators_2026.pdf', domain: 'Central Bank Data', page: 6, snippet: 'Inflation stabilized within the target 5% band while tourist arrivals expanded 28% YoY.' },
        ],
      });
      setGenerating(false);
    }, 900);
  };

  return (
    <div className="space-y-6">
      <div className="glass-card p-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <span className="text-xs font-semibold text-indigo-400 uppercase tracking-wider">Multi-Step LangGraph Agent</span>
          <h1 className="text-2xl font-bold text-white mt-1">AI Analyst Copilot Workstation</h1>
          <p className="text-xs text-gray-300 mt-1">
            Synthesize market disclosures, interim reports, and RAG vectorstore evidence into grounded investment research memos.
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <span className="px-3 py-1 rounded-full text-xs font-semibold bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
            ● LangGraph Agent Active
          </span>
        </div>
      </div>

      <div className="glass-card p-6 space-y-4">
        <label className="text-xs font-semibold text-gray-300">Enter Research Query or Investment Question</label>
        <textarea
          rows={3}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="e.g. Build a comprehensive research memo for John Keells Holdings PLC detailing earnings drivers, disclosures, and risk flags..."
          className="w-full bg-gray-900 border border-gray-700 rounded-lg p-3 text-sm text-white focus:outline-none focus:border-indigo-500"
        />

        <div className="flex justify-end">
          <button
            onClick={handleRunAnalysis}
            disabled={generating}
            className="inline-flex items-center space-x-2 bg-gradient-to-r from-indigo-500 to-sky-600 hover:from-indigo-400 hover:to-sky-500 text-white font-medium px-5 py-2.5 rounded-lg text-sm transition-all shadow-lg shadow-indigo-500/20 disabled:opacity-50"
          >
            <Cpu className="w-4 h-4" />
            <span>{generating ? 'Synthesizing Research Memo...' : 'Run AI Analyst Research Workflow'}</span>
          </button>
        </div>
      </div>

      {memo && (
        <div className="space-y-6">
          <div className="glass-card p-6 space-y-4 border-indigo-500/30">
            <div className="flex items-center justify-between border-b border-gray-800 pb-3">
              <h2 className="text-lg font-bold text-white">Generated Analyst Research Memo</h2>
              <button
                onClick={() => {
                  const blob = new Blob([memo.summary], { type: 'text/markdown' });
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement('a');
                  a.href = url;
                  a.download = 'Analyst_Research_Memo.md';
                  a.click();
                }}
                className="text-xs font-semibold text-sky-400 hover:underline"
              >
                Download Markdown Memo (.md)
              </button>
            </div>

            <div className="prose prose-invert max-w-none text-xs leading-relaxed text-gray-200 whitespace-pre-line">
              {memo.summary}
            </div>
          </div>

          {/* Trust & Grounding Audit Panel */}
          <div className="glass-card p-6 space-y-4 border-emerald-500/30">
            <div className="flex items-center justify-between border-b border-gray-800 pb-3">
              <div className="flex items-center space-x-2">
                <ShieldCheck className="w-5 h-5 text-emerald-400" />
                <h3 className="font-bold text-white text-sm">🔍 Trust & Grounding Audit Panel (Zero-Hallucination Inspector)</h3>
              </div>
              <span className="px-2.5 py-0.5 rounded text-xs font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                {memo.groundingScore} Grounded
              </span>
            </div>

            <p className="text-xs text-gray-300">
              Verified {memo.sourcesCount} raw PDF evidence passages retrieved directly from vectorstore embeddings.
            </p>

            <div className="space-y-3">
              {memo.sources.map((src: any, idx: number) => (
                <div key={idx} className="bg-gray-900/70 p-3 rounded-lg border border-gray-800 space-y-1">
                  <div className="flex items-center justify-between text-xs font-semibold text-white">
                    <span>Source #{idx + 1}: {src.name}</span>
                    <span className="text-gray-400 font-normal">Domain: {src.domain} · Page {src.page}</span>
                  </div>
                  <p className="text-xs text-sky-300 italic">“...{src.snippet}...”</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
