// index.js
import React, { useState } from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';
import { GoogleOAuthProvider } from '@react-oauth/google';
import LoginModal from './LoginModal';

const root = ReactDOM.createRoot(document.getElementById('root'));

const RenderApp = () => {
  const [user, setUser] = useState(() => {
    const stored = sessionStorage.getItem("user");
    return stored ? JSON.parse(stored) : null;
  });

  const handleUserChange = (newUser) => {
    // Clear previous user data when changing users
    if (user && newUser && user.userId !== newUser.userId) {
      sessionStorage.removeItem('userId');
      sessionStorage.removeItem('activeConversationId');
      localStorage.removeItem('activeConversationId');
    }
    
    // Set the new user
    sessionStorage.setItem("user", JSON.stringify(newUser));
    setUser(newUser);
  };

  return (
    <GoogleOAuthProvider clientId="891171975069-kff4l75notakq1g7a04k4m50idcoecpl.apps.googleusercontent.com">
      {user ? (
        <App user={user} onLogout={() => handleUserChange(null)} />
      ) : (
        <LoginModal onLogin={handleUserChange} />
      )}
    </GoogleOAuthProvider>
  );
};

root.render(<RenderApp />);