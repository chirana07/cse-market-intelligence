import Link from 'next/link';
import { ArrowUpRight, BarChart3, FileText, ShieldAlert, Cpu, Sparkles } from 'lucide-react';

export default function CommandCenter() {
  const kpis = [
    { label: 'Listed CSE Companies', value: '294', change: 'Market Universe', icon: BarChart3 },
    { label: 'Parsed Disclosures', value: '1,420+', change: 'Official Feed', icon: FileText },
    { label: 'Triggered Monitoring Rules', value: '3', change: 'Active Alerts', icon: ShieldAlert },
    { label: 'AI Copilot Faithfulness', value: '98.4%', change: 'Grounded RAG', icon: Cpu },
  ];

  const highPriorityAnnouncements = [
    { company: 'John Keells Holdings PLC', symbol: 'JKH.N0000', title: 'FIRST AND FINAL DIVIDEND ANNOUNCEMENT LKR 2.50 PER SHARE', date: '2026-08-15', category: 'Dividend', sentiment: 'Bullish' },
    { company: 'Commercial Bank of Ceylon PLC', symbol: 'COMB.N0000', title: 'INTERIM FINANCIAL STATEMENTS FOR THE PERIOD ENDED 30TH JUNE 2026', date: '2026-08-14', category: 'Financial Statements', sentiment: 'Bullish' },
    { company: 'Dialog Axiata PLC', symbol: 'DIAL.N0000', title: 'PROFIT EXPANSION AND STRATEGIC INFRASTRUCTURE EXPANSION', date: '2026-08-12', category: 'Corporate Action', sentiment: 'Bullish' },
    { company: 'Hayleys PLC', symbol: 'HAYL.N0000', title: 'RIGHTS ISSUE OF ORDINARY SHARES ALLOTMENT RESOLUTION', date: '2026-08-10', category: 'Rights Issue', sentiment: 'Neutral' },
  ];

  return (
    <div className="space-y-8">
      {/* Header Banner */}
      <div className="glass-card p-6 sm:p-8 relative overflow-hidden">
        <div className="absolute -right-10 -bottom-10 w-64 h-64 bg-sky-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="max-w-3xl space-y-3">
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full text-xs font-semibold bg-sky-500/10 text-sky-400 border border-sky-500/20">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Next.js Commercial Enterprise Platform</span>
          </div>
          <h1 className="text-3xl sm:text-4xl font-bold tracking-tight text-white">
            Colombo Stock Exchange Equity & Document Intelligence
          </h1>
          <p className="text-gray-300 text-sm sm:text-base leading-relaxed">
            Real-time financial statement extraction, material CSE disclosure radar, AI-grounded research memos, and institutional portfolio risk metrics.
          </p>
        </div>
      </div>

      {/* KPI Cards Strip */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {kpis.map((kpi, idx) => {
          const Icon = kpi.icon;
          return (
            <div key={idx} className="glass-card p-5 hover:border-gray-700 transition-all">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-gray-400">{kpi.label}</span>
                <div className="p-2 rounded-lg bg-gray-800/60 text-sky-400">
                  <Icon className="w-4 h-4" />
                </div>
              </div>
              <div className="mt-3 flex items-baseline justify-between">
                <span className="text-2xl font-bold tracking-tight text-white">{kpi.value}</span>
                <span className="text-xs font-medium text-sky-400 bg-sky-500/10 px-2 py-0.5 rounded border border-sky-500/20">
                  {kpi.change}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Main Grid Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left 2 Cols: High Priority Disclosures */}
        <div className="lg:col-span-2 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-bold text-white tracking-tight">High-Priority CSE Disclosures Feed</h2>
            <Link href="/disclosures" className="text-xs font-medium text-sky-400 hover:text-sky-300 flex items-center space-x-1">
              <span>View All Disclosures</span>
              <ArrowUpRight className="w-3.5 h-3.5" />
            </Link>
          </div>

          <div className="space-y-3">
            {highPriorityAnnouncements.map((ann, idx) => (
              <div key={idx} className="glass-card p-4 hover:border-gray-700 transition-all space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-semibold text-white">{ann.company} <span className="text-gray-400 font-mono">({ann.symbol})</span></span>
                  <span className="px-2 py-0.5 rounded-md font-semibold text-[10px] bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                    🟢 {ann.sentiment} Catalyst
                  </span>
                </div>
                <p className="text-sm font-medium text-gray-200">{ann.title}</p>
                <div className="flex items-center justify-between text-xs text-gray-400 pt-1">
                  <span>Category: {ann.category} · Date: {ann.date}</span>
                  <Link href="/research" className="text-sky-400 hover:underline">
                    Inspect in Research Hub →
                  </Link>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Right 1 Col: Quick Launchers */}
        <div className="space-y-4">
          <h2 className="text-xl font-bold text-white tracking-tight">Quick Launchers</h2>
          <div className="space-y-3">
            <Link href="/document-intelligence" className="glass-card p-4 block hover:border-sky-500/50 transition-all group">
              <div className="flex items-center justify-between">
                <h3 className="font-semibold text-white group-hover:text-sky-400 transition-colors">1-Click Executive Tear Sheet</h3>
                <FileText className="w-4 h-4 text-sky-400" />
              </div>
              <p className="text-xs text-gray-400 mt-1">Extract financial tables, IAS 34 key figures, and export Markdown summaries.</p>
            </Link>

            <Link href="/copilot" className="glass-card p-4 block hover:border-indigo-500/50 transition-all group">
              <div className="flex items-center justify-between">
                <h3 className="font-semibold text-white group-hover:text-indigo-400 transition-colors">AI Analyst Copilot Memo</h3>
                <Cpu className="w-4 h-4 text-indigo-400" />
              </div>
              <p className="text-xs text-gray-400 mt-1">Generate multi-source investment research memos with Grounding Audit panels.</p>
            </Link>

            <Link href="/portfolio" className="glass-card p-4 block hover:border-emerald-500/50 transition-all group">
              <div className="flex items-center justify-between">
                <h3 className="font-semibold text-white group-hover:text-emerald-400 transition-colors">Portfolio Risk Suite</h3>
                <BarChart3 className="w-4 h-4 text-emerald-400" />
              </div>
              <p className="text-xs text-gray-400 mt-1">Sharpe & Sortino risk metrics, Max Drawdown %, and corporate action screening.</p>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
