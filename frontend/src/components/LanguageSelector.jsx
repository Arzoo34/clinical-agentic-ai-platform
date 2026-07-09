import React, { useState, useEffect } from 'react';
import { Volume2, Loader } from 'lucide-react';

const LanguageSelector = ({ currentLanguage, onLanguageChange, languages = ['hi', 'gu', 'ta'] }) => {
  const langNames = { hi: '🇮🇳 Hindi', gu: '🇮🇳 Gujarati', ta: '🇮🇳 Tamil' };

  return (
    <div className="flex items-center space-x-2">
      <label className="text-sm font-semibold text-gray-700">Language:</label>
      <select
        value={currentLanguage}
        onChange={(e) => onLanguageChange(e.target.value)}
        className="px-3 py-2 border border-gray-300 rounded bg-white text-gray-800 font-semibold cursor-pointer hover:border-blue-500 transition"
      >
        {languages.map((lang) => (
          <option key={lang} value={lang}>
            {langNames[lang]}
          </option>
        ))}
      </select>
    </div>
  );
};

const VoiceOutput = ({ text, language = 'gu' }) => {
  const [isSpeaking, setIsSpeaking] = useState(false);

  const languageCodes = {
    hi: 'hi-IN',
    gu: 'gu-IN',
    ta: 'ta-IN',
  };

  const speak = () => {
    if (!text) return;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = languageCodes[language] || 'hi-IN';
    utterance.rate = 0.9;
    window.speechSynthesis.speak(utterance);
    setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
  };

  return (
    <button
      onClick={speak}
      disabled={!text || isSpeaking}
      className="flex items-center space-x-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white font-semibold rounded transition disabled:opacity-50"
    >
      {isSpeaking ? <Loader size={18} className="animate-spin" /> : <Volume2 size={18} />}
      <span>{isSpeaking ? 'Speaking...' : '🔊 Read Aloud'}</span>
    </button>
  );
};

export { LanguageSelector, VoiceOutput };
