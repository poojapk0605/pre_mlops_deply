const express = require('express');
const router = express.Router();
const Feedback = require('../models/Feedback');
const Conversation = require('../models/Conversation');

// Save feedback
router.post('/', async (req, res) => {
  const { userId, chatId, queryId, rating, feedbackText } = req.body;

  if (!userId || !chatId || !queryId || !rating) {
    return res.status(400).json({ error: 'Missing required fields' });
  }

  try {
    // Prevent duplicate
    const existing = await Feedback.findOne({ userId, chatId, queryId });
    if (existing) {
      return res.status(409).json({ error: 'Feedback already submitted' });
    }

    // Save to Feedback collection
    await Feedback.create({
      userId,
      chatId,
      queryId,
      rating,
      feedbackText: feedbackText || '',
      timestamp: new Date()
    });

    // Update embedded feedback in Conversation
    const conversation = await Conversation.findOne({ id: chatId, userId });
    if (conversation) {
      if (!conversation.feedback) conversation.feedback = {};
      conversation.feedback[queryId] = rating;
      await conversation.save();
    }

    res.status(200).json({ message: 'Feedback saved' });
  } catch (err) {
    console.error('❌ Feedback error:', err);
    res.status(500).json({ error: 'Failed to save feedback' });
  }
});

module.exports = router;
