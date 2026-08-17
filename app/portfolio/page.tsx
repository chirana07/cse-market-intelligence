'use client';

import { useState } from 'react';
import { BarChart3, PieChart, ShieldAlert, Filter } from 'lucide-react';

export default function PortfolioPage() {
  const [preset, setPreset] = useState('ALL');

  const holdings = [
    { company: 'John Keells Holdings PLC', symbol: 'JKH.N0000', shares: '15,000', sector: 'Capital Goods', weight: '34.5%' },
    { company: 'Commercial Bank of Ceylon PLC', symbol: 'COMB.N0000', shares: '25,000', sector: 'Banking', weight: '28.0%' },
    { company: 'Dialog Axiata PLC', symbol: 'DIAL.N0000', shares: '100,000', sector: 'Telecommunication', weight: '22.5%' },
    { company: 'Hayleys PLC', symbol: 'HAYL.N0000', shares: '10,000', sector: 'Conglomerates', weight: '15.0%' },
  ];

  return (
    <div className="space-y-6">
      <div className="glass-card p-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <span className="text-xs font-semibold text-emerald-400 uppercase tracking-wider">Institutional Risk Decomposition</span>
          <h1 className="text-2xl font-bold text-white mt-1">Portfolio Intelligence & Stock Screener</h1>
          <p className="text-xs text-gray-300 mt-1">
            Sharpe Ratio, Sortino Ratio, Max Drawdown %, Sector Concentration Risk, and CSE Stock Screener.
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <span className="px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            ● Risk Models Calibrated
          </span>
        </div>
      </div>

      {/* Risk Metrics Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="glass-card p-4">
          <span className="text-xs text-gray-400">Sharpe Ratio (Risk-Adjusted)</span>
          <span className="text-xl font-bold text-white block mt-1">1.84</span>
          <span className="text-[10px] text-emerald-400 block mt-0.5">Optimal Risk Efficiency</span>
        </div>
        <div className="glass-card p-4">
          <span className="text-xs text-gray-400">Sortino Ratio (Downside)</span>
          <span className="text-xl font-bold text-white block mt-1">2.28</span>
          <span className="text-[10px] text-emerald-400 block mt-0.5">Low Downside Risk</span>
        </div>
        <div className="glass-card p-4">
          <span className="text-xs text-gray-400">Max Drawdown (MDD)</span>
          <span className="text-xl font-bold text-white block mt-1">-8.40%</span>
          <span className="text-[10px] text-sky-400 block mt-0.5">Protected Portfolio</span>
        </div>
        <div className="glass-card p-4">
          <span className="text-xs text-gray-400">Sector Concentration</span>
          <span className="text-xl font-bold text-white block mt-1">34.50%</span>
          <span className="text-[10px] text-gray-400 block mt-0.5">Capital Goods Lead</span>
        </div>
      </div>

      {/* Holdings Table */}
      <div className="glass-card p-6 space-y-4">
        <div className="flex items-center justify-between border-b border-gray-800 pb-3">
          <div className="flex items-center space-x-2">
            <PieChart className="w-4 h-4 text-emerald-400" />
            <h2 className="font-bold text-white">Portfolio Asset Allocations</h2>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="border-b border-gray-800 text-gray-400 uppercase tracking-wider">
              <tr>
                <th className="py-2.5 px-3">Company</th>
                <th className="py-2.5 px-3">Symbol</th>
                <th className="py-2.5 px-3">Sector</th>
                <th className="py-2.5 px-3">Shares</th>
                <th className="py-2.5 px-3">Weight %</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800/60 text-gray-200">
              {holdings.map((h, idx) => (
                <tr key={idx} className="hover:bg-gray-800/30">
                  <td className="py-3 px-3 font-semibold text-white">{h.company}</td>
                  <td className="py-3 px-3 font-mono text-gray-400">{h.symbol}</td>
                  <td className="py-3 px-3 text-gray-300">{h.sector}</td>
                  <td className="py-3 px-3">{h.shares}</td>
                  <td className="py-3 px-3 font-semibold text-emerald-400">{h.weight}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
