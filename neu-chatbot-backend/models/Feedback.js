const mongoose = require('mongoose');

const FeedbackSchema = new mongoose.Schema({
  userId: { type: String, required: true, index: true },
  chatId: { type: String, required: true, index: true },
  queryId: { type: String, required: true, index: true },
  rating: { type: String, required: true, enum: ['positive', 'negative'] },
  feedbackText: String,
  timestamp: { type: Date, default: Date.now }
});

// Compound index to avoid duplicate feedback
FeedbackSchema.index({ userId: 1, chatId: 1, queryId: 1 }, { unique: true });

module.exports = mongoose.model('Feedback', FeedbackSchema);