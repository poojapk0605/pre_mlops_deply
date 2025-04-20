// neu-chatbot/src/UserDropdown.js
import React, { useState } from 'react';
import './App.css';

const UserDropdown = ({ user, onLogout }) => {
  const [open, setOpen] = useState(false);

  const toggleDropdown = () => setOpen(!open);

  const getInitial = (name) => {
    return name?.charAt(0)?.toUpperCase() || '?';
  };

  return (
    <div className="user-menu">
      <div
        className="user-avatar"
        onClick={toggleDropdown}
        title={user.name}
      >
        {user.picture ? (
          <img src={user.picture} alt="User" className="user-avatar" />
        ) : (
          <div className="user-avatar-fallback">{getInitial(user.name)}</div>
        )}
      </div>

      {open && (
        <div className="user-dropdown">
          <div className="user-name">{user.name}</div>
          <button onClick={onLogout}>
            {user.email === 'guest@askneu.ai' ? 'Login' : 'Logout'}
          </button>
        </div>
      )}
    </div>
  );
};

export default UserDropdown;
