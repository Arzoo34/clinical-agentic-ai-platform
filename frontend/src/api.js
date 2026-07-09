import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const sellerAPI = {
  getCurrent: () => api.get('/seller'),
  updateLanguage: (language) => api.post('/seller/language', null, { params: { language } }),
};

export const listingAPI = {
  generate: (sellerId, rawInput, category, photoCount, codEnabled, pinCode) =>
    api.post('/listings/generate', null, {
      params: { seller_id: sellerId, raw_input: rawInput, category, photo_count: photoCount, cod_enabled: codEnabled, pin_code: pinCode },
    }),
  list: (sellerId) => api.get('/listings', { params: { seller_id: sellerId } }),
  get: (listingId) => api.get(`/listings/${listingId}`),
  update: (listingId, updates) => api.put(`/listings/${listingId}`, null, { params: updates }),
  calculateRiskScore: (listingId) => api.post(`/listings/${listingId}/risk-score`),
  checkFraudRisk: (listingId) => api.post(`/listings/${listingId}/fraud-check`),
};

export const qaAPI = {
  getPending: (listingId) => api.get('/qa/pending', { params: { listing_id: listingId } }),
  cluster: (listingId) => api.post('/qa/cluster', null, { params: { listing_id: listingId } }),
  approve: (replyId) => api.post('/qa/approve', null, { params: { reply_id: replyId } }),
};

export const healthAPI = {
  scan: (sellerId) => api.post('/health/scan', null, { params: { seller_id: sellerId } }),
  getBriefs: (sellerId) => api.get('/health/briefs', { params: { seller_id: sellerId } }),
};

export default api;
