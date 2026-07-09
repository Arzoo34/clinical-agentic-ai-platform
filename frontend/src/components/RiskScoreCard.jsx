import React from 'react';
import { Volume2, AlertCircle, CheckCircle } from 'lucide-react';

const RiskScoreCard = ({ score, gaps, listing, onFixClick, language = 'gu' }) => {
  const getSeverityColor = (severity) => {
    if (severity === 'HIGH') return 'bg-red-100 text-red-800 border-red-300';
    if (severity === 'MEDIUM') return 'bg-yellow-100 text-yellow-800 border-yellow-300';
    return 'bg-green-100 text-green-800 border-green-300';
  };

  const getSeverityBgColor = (severity) => {
    if (severity === 'HIGH') return 'bg-red-50';
    if (severity === 'MEDIUM') return 'bg-yellow-50';
    return 'bg-green-50';
  };

  const scoreColor = score > 60 ? 'text-red-600' : score > 30 ? 'text-yellow-600' : 'text-green-600';

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6 shadow-sm">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-gray-800">📊 Risk Score Analysis</h3>
        <div className="text-right">
          <div className={`text-4xl font-bold ${scoreColor}`}>{Math.round(score)}%</div>
          <div className="text-sm text-gray-500">Current Risk</div>
        </div>
      </div>

      {gaps && gaps.length > 0 ? (
        <div className="space-y-3 mb-6">
          {gaps.map((gap, idx) => (
            <div key={idx} className={`${getSeverityBgColor(gap.severity)} border-l-4 ${gap.severity === 'HIGH' ? 'border-red-500' : gap.severity === 'MEDIUM' ? 'border-yellow-500' : 'border-green-500'} p-3 rounded`}>
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-semibold text-gray-800">{gap.label}</div>
                  <div className="text-sm text-gray-600 mt-1">{gap.explanation}</div>
                </div>
                <div className="text-right ml-4 flex-shrink-0">
                  <span className={`inline-block px-3 py-1 text-xs font-semibold rounded-full ${getSeverityColor(gap.severity)} border`}>
                    {gap.severity}
                  </span>
                  <div className="text-sm font-bold text-gray-700 mt-1">{gap.contribution_pct}%</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="bg-green-50 border border-green-300 p-4 rounded mb-6 flex items-center">
          <CheckCircle className="text-green-600 mr-3" />
          <div>
            <div className="font-semibold text-green-800">✅ Listing Optimized!</div>
            <div className="text-sm text-green-700">No critical gaps detected.</div>
          </div>
        </div>
      )}

      <button
        onClick={onFixClick}
        className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded transition"
      >
        🔧 Apply All Fixes
      </button>
    </div>
  );
};

export default RiskScoreCard;
