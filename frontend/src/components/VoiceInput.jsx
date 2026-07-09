import React, { useState } from 'react';
import { Mic, StopCircle } from 'lucide-react';

const VoiceInput = ({ onTranscript, language = 'gu', placeholder = 'Listening...' }) => {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');

  const languageCodes = {
    hi: 'hi-IN',
    gu: 'gu-IN',
    ta: 'ta-IN',
  };

  const startListening = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert('Speech Recognition not supported in your browser');
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.language = languageCodes[language] || 'hi-IN';
    recognition.interimResults = true;
    recognition.continuous = false;

    recognition.onstart = () => setIsListening(true);
    recognition.onresult = (event) => {
      let interim = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          setTranscript(transcript);
          onTranscript(transcript);
        } else {
          interim += transcript;
        }
      }
    };
    recognition.onerror = (event) => console.error('Speech error:', event.error);
    recognition.onend = () => setIsListening(false);

    recognition.start();
  };

  return (
    <div className="flex items-center space-x-2">
      <button
        onClick={startListening}
        disabled={isListening}
        className={`flex items-center space-x-2 px-4 py-2 rounded font-semibold transition ${
          isListening
            ? 'bg-red-600 hover:bg-red-700 text-white'
            : 'bg-blue-600 hover:bg-blue-700 text-white'
        }`}
      >
        {isListening ? <StopCircle size={18} /> : <Mic size={18} />}
        <span>{isListening ? 'Listening...' : '🎤 Speak'}</span>
      </button>
    </div>
  );
};

export default VoiceInput;
