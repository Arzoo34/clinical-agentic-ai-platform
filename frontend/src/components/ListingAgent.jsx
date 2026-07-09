import React, { useState, useEffect } from 'react';
import { ShoppingBag, AlertTriangle, CheckCircle, Zap } from 'lucide-react';
import { listingAPI } from '../api';
import useApi from '../hooks/useApi';
import RiskScoreCard from './RiskScoreCard';
import VoiceInput from './VoiceInput';

const ListingAgent = ({ seller }) => {
  const { execute, loading } = useApi();
  const [category, setCategory] = useState('Kurti');
  const [photoCount, setPhotoCount] = useState(1);
  const [codEnabled, setCodEnabled] = useState(false);
  const [pinCode, setPinCode] = useState('395007');
  const [rawInput, setRawInput] = useState('');
  const [listings, setListings] = useState([]);
  const [selectedListing, setSelectedListing] = useState(null);
  const [riskScore, setRiskScore] = useState(null);
  const [fraudRisk, setFraudRisk] = useState(null);

  const categories = ['Kurti', 'Saree', 'T-Shirt', 'Jeans', 'Dress', 'Dupatta', 'Lehenga'];

  useEffect(() => {
    fetchListings();
  }, [seller]);

  const fetchListings = async () => {
    const data = await execute(() => listingAPI.list(seller.id));
    if (data) setListings(data);
  };

  const handleGenerateListing = async () => {
    if (!rawInput.trim()) {
      alert('Please enter a product description');
      return;
    }
    const data = await execute(() =>
      listingAPI.generate(seller.id, rawInput, category, photoCount, codEnabled, pinCode)
    );
    if (data) {
      setSelectedListing(data);
      setRawInput('');
      await fetchListings();
      // Calculate risk score
      const riskData = await execute(() => listingAPI.calculateRiskScore(data.id));
      if (riskData) setRiskScore(riskData);
      // Check fraud risk
      const fraudData = await execute(() => listingAPI.checkFraudRisk(data.id));
      if (fraudData) setFraudRisk(fraudData);
    }
  };

  const handleApplyFixes = async () => {
    if (!selectedListing) return;
    // Toggle gaps to false (apply fixes)
    await execute(() =>
      listingAPI.update(selectedListing.id, {
        size_chart: true,
        photo_count: 3,
        fabric_mentioned: true,
        wash_care: true,
      })
    );
    // Recalculate risk
    const riskData = await execute(() => listingAPI.calculateRiskScore(selectedListing.id));
    if (riskData) setRiskScore(riskData);
    setSelectedListing({ ...selectedListing, size_chart: true, photo_count: 3, fabric_mentioned: true, wash_care: true });
    alert('✅ All fixes applied! Risk score recalculated.');
  };

  const handleListingSelect = async (listing) => {
    setSelectedListing(listing);
    const riskData = await execute(() => listingAPI.calculateRiskScore(listing.id));
    if (riskData) setRiskScore(riskData);
    const fraudData = await execute(() => listingAPI.checkFraudRisk(listing.id));
    if (fraudData) setFraudRisk(fraudData);
  };

  return (
    <div className="space-y-6">
      {/* Input Section */}
      <div className="bg-white rounded-lg border border-gray-200 p-6 shadow-sm">
        <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center">
          <ShoppingBag className="mr-2 text-blue-600" /> Create New Listing
        </h2>

        {/* Voice Input */}
        <div className="mb-4">
          <label className="block text-sm font-semibold text-gray-700 mb-2">🎤 Voice Input (Optional)</label>
          <VoiceInput
            language={seller.preferred_language}
            onTranscript={(text) => setRawInput(text)}
            placeholder="Describe your product"
          />
        </div>

        {/* Text Input */}
        <div className="mb-4">
          <label className="block text-sm font-semibold text-gray-700 mb-2">Product Description</label>
          <textarea
            value={rawInput}
            onChange={(e) => setRawInput(e.target.value)}
            placeholder="E.g., Blue cotton kurti, suitable for casual wear, one photo available, no size chart yet"
            rows="3"
            className="w-full px-3 py-2 border border-gray-300 rounded bg-gray-50 text-gray-800 placeholder-gray-500 focus:bg-white focus:border-blue-500 transition"
          />
        </div>

        {/* Form Fields */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-1">Category</label>
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded bg-white text-gray-800 cursor-pointer"
            >
              {categories.map((cat) => (
                <option key={cat} value={cat}>
                  {cat}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-1">Photo Count</label>
            <input
              type="number"
              value={photoCount}
              onChange={(e) => setPhotoCount(parseInt(e.target.value))}
              min="1"
              max="5"
              className="w-full px-3 py-2 border border-gray-300 rounded bg-white text-gray-800"
            />
          </div>

          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-1">PIN Code</label>
            <input
              type="text"
              value={pinCode}
              onChange={(e) => setPinCode(e.target.value)}
              placeholder="395007"
              className="w-full px-3 py-2 border border-gray-300 rounded bg-white text-gray-800"
            />
          </div>

          <div className="flex items-center pt-6">
            <input
              type="checkbox"
              checked={codEnabled}
              onChange={(e) => setCodEnabled(e.target.checked)}
              id="cod"
              className="w-4 h-4 cursor-pointer"
            />
            <label htmlFor="cod" className="ml-2 text-sm font-semibold text-gray-700 cursor-pointer">
              ✓ COD Enabled
            </label>
          </div>
        </div>

        <button
          onClick={handleGenerateListing}
          disabled={loading}
          className="w-full bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white font-bold py-3 px-4 rounded transition disabled:opacity-50"
        >
          {loading ? '⏳ Generating...' : '✨ Generate Listing with AI'}
        </button>
      </div>

      {/* Fraud Risk Alert */}
      {fraudRisk && fraudRisk.risk_level !== 'NONE' && (
        <div className={`p-4 rounded border-l-4 flex items-start ${
          fraudRisk.risk_level === 'HIGH'
            ? 'bg-red-50 border-red-500'
            : 'bg-yellow-50 border-yellow-500'
        }`}>
          <AlertTriangle className={`mr-3 flex-shrink-0 ${
            fraudRisk.risk_level === 'HIGH' ? 'text-red-600' : 'text-yellow-600'
          }`} />
          <div>
            <div className={`font-bold ${
              fraudRisk.risk_level === 'HIGH' ? 'text-red-800' : 'text-yellow-800'
            }`}>⚠️ Fraud Risk Alert</div>
            <div className={`text-sm ${
              fraudRisk.risk_level === 'HIGH' ? 'text-red-700' : 'text-yellow-700'
            }`}>{fraudRisk.message}</div>
          </div>
        </div>
      )}

      {/* Risk Score Card */}
      {riskScore && selectedListing && (
        <RiskScoreCard
          score={riskScore.score}
          gaps={riskScore.gaps}
          listing={selectedListing}
          onFixClick={handleApplyFixes}
          language={seller.preferred_language}
        />
      )}

      {/* Listings List */}
      {listings.length > 0 && (
        <div className="bg-white rounded-lg border border-gray-200 p-6 shadow-sm">
          <h3 className="text-lg font-bold text-gray-800 mb-4">📋 Your Listings</h3>
          <div className="space-y-2">
            {listings.map((listing) => (
              <button
                key={listing.id}
                onClick={() => handleListingSelect(listing)}
                className={`w-full p-4 text-left rounded border-2 transition ${
                  selectedListing?.id === listing.id
                    ? 'border-blue-600 bg-blue-50'
                    : 'border-gray-200 bg-white hover:border-blue-400'
                }`}
              >
                <div className="font-semibold text-gray-800">{listing.title}</div>
                <div className="text-sm text-gray-600">₹{listing.price} • {listing.category} • PIN: {listing.pin_code}</div>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ListingAgent;
