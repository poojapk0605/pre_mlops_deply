// routes/Conversation.js
const express = require('express');
const router = express.Router();
const Conversation = require('../models/Conversation');
const Feedback = require('../models/Feedback');

// Save conversations
router.post('/save', async (req, res) => {
  const { userId, conversations } = req.body;

  if (!userId || !conversations) {
    return res.status(400).json({ error: 'Missing userId or conversations' });
  }

  try {
    // Update or create each conversation
    const saveOperations = Object.entries(conversations).map(([id, conv]) => {
      return Conversation.findOneAndUpdate(
        { id, userId },
        { ...conv, id, userId },
        { upsert: true, new: true, strict: false }
      );
    });

    await Promise.all(saveOperations);
    res.json({ success: true });
  } catch (err) {
    console.error('❌ Error saving conversations:', err);
    res.status(500).json({ error: 'Failed to save conversations' });
  }
});

// Load conversations
router.get('/load', async (req, res) => {
  const { userId } = req.query;

  if (!userId) return res.status(400).json({ error: 'Missing userId' });

  try {
    const conversations = await Conversation.find({ userId });
    const formattedConversations = {};
    
    conversations.forEach(conv => {
      formattedConversations[conv.id] = conv.toJSON();
    });

    res.json({ conversations: formattedConversations });
  } catch (err) {
    console.error('❌ Error loading conversations:', err);
    res.status(500).json({ error: 'Failed to load conversations' });
  }
});

// Save active conversation ID
router.post('/active', async (req, res) => {
  const { userId, activeConversationId } = req.body;

  if (!userId || !activeConversationId) {
    return res.status(400).json({ error: 'Missing userId or activeConversationId' });
  }

  try {
    // Reset all active flags for this user
    await Conversation.updateMany({ userId }, { activeConversation: false });
    
    // Set the new active conversation
    await Conversation.findOneAndUpdate(
      { id: activeConversationId, userId },
      { activeConversation: true }
    );
    
    res.json({ success: true });
  } catch (err) {
    console.error('❌ Error saving active conversation:', err);
    res.status(500).json({ error: 'Failed to save active conversation' });
  }
});

// Load active conversation
router.get('/active', async (req, res) => {
  const { userId } = req.query;

  if (!userId) return res.status(400).json({ error: 'Missing userId' });

  try {
    const activeConversation = await Conversation.findOne({ 
      userId, 
      activeConversation: true 
    });
    
    res.json({ 
      activeConversationId: activeConversation?.id || null 
    });
  } catch (err) {
    console.error('❌ Error loading active conversation:', err);
    res.status(500).json({ error: 'Failed to load active conversation' });
  }
});

module.exports = router;