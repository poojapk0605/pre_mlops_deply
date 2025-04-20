// LoginModal.js
import React from 'react';
import { GoogleLogin } from '@react-oauth/google';
import { jwtDecode } from 'jwt-decode';
import './App.css';

const LoginModal = ({ onLogin }) => {
  const handleSuccess = (response) => {
    const decoded = jwtDecode(response.credential);
    const email = decoded.email;
    const googleId = decoded.sub;
    
    // Check if this user already exists or create new
    fetch('/api/users/check', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, googleId })
    })
    .then(res => res.json())
    .then(data => {
      // Create user object with userId from server response
      const user = {
        userId: data.userId,
        email: decoded.email,
        name: decoded.name,
        picture: decoded.picture
      };
      
      onLogin(user);
    })
    .catch(err => {
      console.error("Error checking user:", err);
      // Fallback with client-side ID if server fails
      const user = {
        userId: `google_${googleId}`,
        email: decoded.email,
        name: decoded.name,
        picture: decoded.picture
      };
      onLogin(user);
    });
  };
  
  const continueAsGuest = () => {
    const guest = {
      userId: `guest_${Date.now()}`,
      email: 'guest@askneu.ai',
      name: 'Guest User'
    };
    onLogin(guest);
  };

  return (
    <div className="login-modal-overlay">
      <div className="login-modal">
        <div className="neu-logo-large">
          <img src="/logo.png" alt="NEU Logo" />
        </div>
        <h2>Welcome to Ask NEU</h2>
        <p className="login-subtitle">Your AI assistant for all things Northeastern</p>
        <div className="login-options">
          <GoogleLogin 
            onSuccess={handleSuccess} 
            onError={() => alert("Login Failed")} 
            shape="pill"
            theme="filled_blue"
            size="large"
            text="continue_with"
            locale="en"
          />
          <div className="login-divider">
            <span>or</span>
          </div>
          <button className="guest-btn" onClick={continueAsGuest}>
            Continue as Guest
          </button>
        </div>
      </div>
    </div>
  );
};

export default LoginModal;