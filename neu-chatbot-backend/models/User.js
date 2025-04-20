const mongoose = require('mongoose');

const UserSchema = new mongoose.Schema({
  userId: { type: String, required: true, unique: true, index: true },
  email: { type: String, required: true },
  name: String,
  picture: String,
  authProvider: { type: String, enum: ['google', 'guest', 'unknown'], default: 'unknown' },
  lastLogin: { type: Date, default: Date.now },
  created: { type: Date, default: Date.now }
});

module.exports = mongoose.model('User', UserSchema);