import React, { useState, useEffect } from 'react';
import { Heart, Menu, X } from 'lucide-react';
import { sellerAPI } from './api';
import ListingAgent from './components/ListingAgent';
import QAAgent from './components/QAAgent';
import HealthAgent from './components/HealthAgent';
import { LanguageSelector } from './components/LanguageSelector';
import './index.css';

function App() {
  const [seller, setSeller] = useState(null);
  const [listings, setListings] = useState([]);
  const [activeTab, setActiveTab] = useState('listing');
  const [loading, setLoading] = useState(true);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    fetchSeller();
  }, []);

  const fetchSeller = async () => {
    try {
      const response = await sellerAPI.getCurrent();
      setSeller(response);
      setLoading(false);
    } catch (error) {
      console.error('Failed to fetch seller:', error);
      setLoading(false);
    }
  };

  const handleLanguageChange = async (language) => {
    try {
      await sellerAPI.updateLanguage(language);
      setSeller({ ...seller, preferred_language: language });
    } catch (error) {
      console.error('Failed to update language:', error);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-green-50 flex items-center justify-center">
        <div className="text-center">
          <div className="text-5xl mb-4">🚀</div>
          <div className="text-2xl font-bold text-gray-800">Shuruaat AI</div>
          <div className="text-gray-600 mt-2">Loading...</div>
        </div>
      </div>
    );
  }

  if (!seller) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-green-50 flex items-center justify-center">
        <div className="text-center bg-white p-8 rounded-lg shadow-lg">
          <div className="text-5xl mb-4">⚠️</div>
          <div className="text-2xl font-bold text-gray-800">Error</div>
          <div className="text-gray-600 mt-2">Could not load seller information</div>
        </div>
      </div>
    );
  }

  const tabs = [
    { id: 'listing', label: '📝 Listing Agent', icon: '📝' },
    { id: 'qa', label: '💬 Q&A Agent', icon: '💬' },
    { id: 'health', label: '📊 Health Agent', icon: '📊' },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="text-3xl">🚀</div>
            <div>
              <div className="text-2xl font-bold text-gray-800">Shuruaat AI</div>
              <div className="text-xs text-gray-600">Meesho Seller Co-pilot</div>
            </div>
          </div>

          {/* Desktop Language Selector */}
          <div className="hidden md:block">
            <LanguageSelector
              currentLanguage={seller.preferred_language}
              onLanguageChange={handleLanguageChange}
              languages={['hi', 'gu', 'ta']}
            />
          </div>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden p-2 rounded hover:bg-gray-100 transition"
          >
            {mobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
        </div>

        {/* Mobile Language Selector */}
        {mobileMenuOpen && (
          <div className="md:hidden px-4 pb-4 border-t border-gray-200">
            <LanguageSelector
              currentLanguage={seller.preferred_language}
              onLanguageChange={handleLanguageChange}
              languages={['hi', 'gu', 'ta']}
            />
          </div>
        )}
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-8">
        {/* Seller Info Card */}
        <div className="bg-white rounded-lg border border-gray-200 p-6 mb-8 shadow-sm">
          <div className="flex items-center space-x-4">
            <div className="text-4xl">👋</div>
            <div>
              <div className="text-xl font-bold text-gray-800">
                Welcome, {seller.name}!
              </div>
              <div className="text-sm text-gray-600">
                📍 {seller.city} • 🗣️ {seller.preferred_language === 'hi' ? 'Hindi' : seller.preferred_language === 'gu' ? 'Gujarati' : 'Tamil'}
              </div>
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex space-x-2 mb-8 overflow-x-auto pb-2">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => {
                setActiveTab(tab.id);
                setMobileMenuOpen(false);
              }}
              className={`px-6 py-3 rounded-lg font-semibold whitespace-nowrap transition ${
                activeTab === tab.id
                  ? 'bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-lg'
                  : 'bg-white text-gray-700 border border-gray-300 hover:border-blue-500 hover:text-blue-600'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        <div>
          {activeTab === 'listing' && <ListingAgent seller={seller} />}
          {activeTab === 'qa' && <QAAgent seller={seller} listings={listings} />}
          {activeTab === 'health' && <HealthAgent seller={seller} />}
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-200 bg-white mt-12 py-6">
        <div className="max-w-7xl mx-auto px-4 text-center text-sm text-gray-600">
          <div className="flex items-center justify-center mb-2">
            <Heart size={16} className="text-red-500 mr-1" />
            <span>Built for Indian sellers • Hackathon prototype</span>
          </div>
          <div>Shuruaat = "Beginning" in Hindi 🚀</div>
        </div>
      </footer>
    </div>
  );
}

export default App;
