const mongoose = require('mongoose');

const MessageSchema = new mongoose.Schema({
  sender: String,
  text: String,
  sources: String,
  query_id: String,
  timestamp: String,
  processingTime: Number,
  searchMode: String,
  activeTab: String,
  showInitialMessage: Boolean
}, { _id: false, strict: false });

const ConversationSchema = new mongoose.Schema({
  id: { type: String, required: true, index: true },
  userId: { type: String, required: true, index: true },
  title: String,
  messages: [MessageSchema],
  date: { type: Date, default: Date.now },
  activeConversation: Boolean,
  feedback: { type: mongoose.Schema.Types.Mixed, default: {} }
}, { strict: false });

// Ensure feedback is properly serialized
ConversationSchema.set('toJSON', {
  transform: function (doc, ret) {
    if (!ret.feedback) ret.feedback = {};
    if (typeof ret.feedback.toObject === 'function') {
      ret.feedback = ret.feedback.toObject();
    }
    return ret;
  }
});

module.exports = mongoose.model('Conversation', ConversationSchema);
