import React, { useState, useEffect } from 'react';
import { MessageSquare, Check, AlertCircle } from 'lucide-react';
import { qaAPI, listingAPI } from '../api';
import useApi from '../hooks/useApi';

const QAAgent = ({ seller, listings }) => {
  const { execute, loading } = useApi();
  const [selectedListing, setSelectedListing] = useState(null);
  const [clusters, setClusters] = useState([]);
  const [pendingQuestions, setPendingQuestions] = useState([]);

  useEffect(() => {
    if (listings.length > 0 && !selectedListing) {
      setSelectedListing(listings[0]);
    }
  }, [listings]);

  useEffect(() => {
    if (selectedListing) {
      fetchPendingQuestions();
    }
  }, [selectedListing]);

  const fetchPendingQuestions = async () => {
    const data = await execute(() => qaAPI.getPending(selectedListing.id));
    if (data) setPendingQuestions(data);
  };

  const handleCluster = async () => {
    const data = await execute(() => qaAPI.cluster(selectedListing.id));
    if (data) {
      setClusters(data.clusters || []);
      await fetchPendingQuestions();
    }
  };

  const handleApprove = async (clusterId, replyId) => {
    await execute(() => qaAPI.approve(replyId));
    alert('✅ Reply approved and listing updated!');
    setClusters(clusters.filter((c) => c.cluster_id !== clusterId));
  };

  return (
    <div className="space-y-6">
      {/* Listing Selector */}
      {listings.length > 0 && (
        <div className="bg-white rounded-lg border border-gray-200 p-6 shadow-sm">
          <label className="block text-sm font-semibold text-gray-700 mb-2">Select Listing</label>
          <select
            value={selectedListing?.id || ''}
            onChange={(e) => setSelectedListing(listings.find((l) => l.id === parseInt(e.target.value)))}
            className="w-full px-3 py-2 border border-gray-300 rounded bg-white text-gray-800 cursor-pointer"
          >
            {listings.map((listing) => (
              <option key={listing.id} value={listing.id}>
                {listing.title}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Cluster Button */}
      {selectedListing && pendingQuestions.length > 0 && (
        <button
          onClick={handleCluster}
          disabled={loading}
          className="w-full bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-700 hover:to-purple-800 text-white font-bold py-3 px-4 rounded transition disabled:opacity-50"
        >
          {loading ? '⏳ Analyzing Questions...' : `🔍 Cluster & Draft Replies (${pendingQuestions.length} questions)`}
        </button>
      )}

      {/* Clustered Questions */}
      {clusters.length > 0 && (
        <div className="space-y-4">
          <h3 className="text-lg font-bold text-gray-800 flex items-center">
            <MessageSquare className="mr-2 text-purple-600" /> Question Clusters
          </h3>
          {clusters.map((cluster) => (
            <div key={cluster.cluster_id} className="bg-white rounded-lg border-2 border-purple-200 p-4 shadow-sm">
              <div className="mb-3">
                <div className="text-sm font-semibold text-purple-600 mb-2">Questions in Cluster:</div>
                {cluster.question_ids.map((qId) => {
                  const q = pendingQuestions.find((pq) => pq.id === qId);
                  return q ? (
                    <div key={q.id} className="text-sm text-gray-700 mb-1 pl-3 border-l-2 border-purple-300">
                      "{q.question_text}"
                    </div>
                  ) : null;
                })}
              </div>

              <div className="bg-blue-50 border border-blue-300 rounded p-3 mb-3">
                <div className="text-sm font-semibold text-blue-800 mb-1">💬 Suggested Reply:</div>
                <div className="text-sm text-blue-700">{cluster.draft_reply}</div>
              </div>

              <div className="bg-yellow-50 border border-yellow-300 rounded p-3 mb-4">
                <div className="text-sm font-semibold text-yellow-800 mb-1">📝 Listing Fix Suggestion:</div>
                <div className="text-sm text-yellow-700">{cluster.listing_fix_suggestion}</div>
              </div>

              <button
                onClick={() => handleApprove(cluster.cluster_id, cluster.question_ids[0])}
                className="w-full bg-green-600 hover:bg-green-700 text-white font-semibold py-2 px-4 rounded transition flex items-center justify-center"
              >
                <Check size={18} className="mr-2" /> Approve & Apply Fix
              </button>
            </div>
          ))}
        </div>
      )}

      {/* No Questions State */}
      {selectedListing && pendingQuestions.length === 0 && clusters.length === 0 && (
        <div className="bg-blue-50 border border-blue-300 rounded p-6 text-center">
          <AlertCircle className="mx-auto text-blue-600 mb-2" />
          <div className="text-blue-800 font-semibold">ℹ️ No Pending Questions</div>
          <div className="text-sm text-blue-700">Questions will appear here as buyers ask about this listing.</div>
        </div>
      )}
    </div>
  );
};

export default QAAgent;
