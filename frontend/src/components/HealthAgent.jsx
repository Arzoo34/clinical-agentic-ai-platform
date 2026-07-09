import React, { useState, useEffect } from 'react';
import { Zap, Loader } from 'lucide-react';
import { healthAPI } from '../api';
import useApi from '../hooks/useApi';
import { VoiceOutput } from './LanguageSelector';

const HealthAgent = ({ seller }) => {
  const { execute, loading } = useApi();
  const [briefs, setBriefs] = useState([]);
  const [latestBrief, setLatestBrief] = useState(null);

  useEffect(() => {
    fetchBriefs();
  }, [seller]);

  const fetchBriefs = async () => {
    const data = await execute(() => healthAPI.getBriefs(seller.id));
    if (data && data.length > 0) {
      setBriefs(data);
      setLatestBrief(data[0]);
    }
  };

  const handleRunScan = async () => {
    const data = await execute(() => healthAPI.scan(seller.id));
    if (data) {
      setLatestBrief(data);
      await fetchBriefs();
    }
  };

  return (
    <div className="space-y-6">
      {/* Run Scan Button */}
      <div className="bg-gradient-to-r from-green-600 to-green-700 rounded-lg p-6 text-white shadow-lg">
        <h2 className="text-2xl font-bold mb-2 flex items-center">
          <Zap className="mr-2" /> Weekly Health Scan
        </h2>
        <p className="text-green-100 mb-4">Analyze your return patterns and get actionable recommendations.</p>
        <button
          onClick={handleRunScan}
          disabled={loading}
          className="bg-white hover:bg-green-50 text-green-700 font-bold py-2 px-6 rounded transition disabled:opacity-50 flex items-center"
        >
          {loading ? (
            <>
              <Loader size={18} className="mr-2 animate-spin" /> Scanning...
            </>
          ) : (
            <>🔄 Run Weekly Scan</>
          )}
        </button>
      </div>

      {/* Latest Brief */}
      {latestBrief && (
        <div className="bg-white rounded-lg border border-green-200 p-6 shadow-sm">
          <h3 className="text-lg font-bold text-gray-800 mb-3">📊 This Week's Summary</h3>

          <div className="bg-green-50 border border-green-300 rounded p-4 mb-4">
            <div className="text-gray-800 text-base leading-relaxed mb-4">{latestBrief.summary_text}</div>
            <VoiceOutput text={latestBrief.summary_text} language={seller.preferred_language} />
          </div>

          {latestBrief.recommendations && latestBrief.recommendations.length > 0 && (
            <div>
              <h4 className="font-semibold text-gray-800 mb-3">💡 Recommendations:</h4>
              <div className="space-y-3">
                {latestBrief.recommendations.map((rec, idx) => (
                  <div key={idx} className="bg-yellow-50 border border-yellow-300 rounded p-3">
                    <div className="font-semibold text-yellow-800">{rec.title}</div>
                    <div className="text-sm text-yellow-700 mt-1">{rec.description}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Previous Briefs */}
      {briefs.length > 1 && (
        <div className="bg-white rounded-lg border border-gray-200 p-6 shadow-sm">
          <h3 className="text-lg font-bold text-gray-800 mb-4">📈 Previous Briefs</h3>
          <div className="space-y-2">
            {briefs.slice(1).map((brief) => (
              <button
                key={brief.id}
                onClick={() => setLatestBrief(brief)}
                className="w-full p-3 text-left rounded border border-gray-300 hover:border-green-500 hover:bg-green-50 transition"
              >
                <div className="font-semibold text-gray-800">
                  Week of {new Date(brief.week_of).toLocaleDateString()}
                </div>
                <div className="text-sm text-gray-600 truncate">{brief.summary_text}</div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Empty State */}
      {!latestBrief && (
        <div className="bg-blue-50 border border-blue-300 rounded p-6 text-center">
          <div className="text-blue-800 font-semibold mb-2">ℹ️ No Health Briefs Yet</div>
          <div className="text-sm text-blue-700">Click "Run Weekly Scan" to generate your first health brief!</div>
        </div>
      )}
    </div>
  );
};

export default HealthAgent;
