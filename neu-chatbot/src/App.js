import React, { useState, useRef, useEffect, useCallback, memo } from 'react';
import './App.css';
import { marked } from 'marked';
import { 
  MessageSquare, ChevronRight, X, Moon, Sun, Eye, EyeOff, 
  ThumbsUp, ThumbsDown, Copy, Radar, Send, BookOpen, School
} from 'lucide-react';
// Memoized Message Component
const MessageComponent = memo(({
  message, 
  index, 
  setConversations,
  conversations,
  activeConversationId,
  submitFeedback, 
  copyToClipboard, 
  getFeedbackForMessage,
  activeFeedback,
  activeConversation,
  setActiveFeedback
}) => {
  const [activeTab, setActiveTab] = useState(message.activeTab || 'answer');
  const [feedbackAcknowledged, setFeedbackAcknowledged] = useState(false);
  

  // Get query_id directly from the message
  let query_id = message.query_id;
  if (!query_id && message.sender === 'bot') {
    console.warn(`Bot message at index ${index} has no query_id`);
    
    // For welcome message, use a fixed ID
    if (message.text && message.text.includes("Hi! I am Northeastern University Assistant")) {
      query_id = 'welcome_message';
      
      // Update the message with this query_id
	  
      const updatedConversations = {...conversations};
      const currentConv = {...updatedConversations[activeConversationId]};
	  currentConv.messages = Array.isArray(currentConv.messages) ? [...currentConv.messages] : [];
      
      if (index >= 0 && index < currentConv.messages.length) {
        currentConv.messages[index] = {
          ...currentConv.messages[index],
          query_id: query_id
        };
        updatedConversations[activeConversationId] = currentConv;
        setConversations(updatedConversations);
      }
    }
  }

  // Get current feedback state for this message
  const currentFeedback = query_id ? (
    activeFeedback[query_id] || 
    (activeConversation?.feedback && activeConversation.feedback[query_id])
  ) : null;

  // Set CSS classes for feedback buttons
  const thumbsUpClass = `action-btn thumbs-up ${currentFeedback === 'positive' ? 'active clicked' : ''}`;
  const thumbsDownClass = `action-btn thumbs-down ${currentFeedback === 'negative' ? 'active clicked' : ''}`;

  // Handle tab changes between answer and sources
  const handleTabChange = (tab) => {
    setActiveTab(tab);
    const updatedConversations = {...conversations};
    const currentConv = {...updatedConversations[activeConversationId]};
    currentConv.messages = [...currentConv.messages];

    if (index >= 0 && index < currentConv.messages.length) {
      currentConv.messages[index] = {
        ...currentConv.messages[index],
        activeTab: tab
      };
      updatedConversations[activeConversationId] = currentConv;
      setConversations(updatedConversations);
    }
  };

  return (
    <div className={`message ${message.sender}`}>
      <div className="message-container">
        <div className="message-content">
          {message.sender === 'user' && (
            <div className="message-sender">You</div>
          )}

          {message.sender === 'bot' && (
            <>
              <div className="message-header">
                <div className="message-sender">
                  NEU Assistant
                  {message.searchMode === 'deepsearch' && (
                    <span className="processing-time">Deep Research</span>
                  )}
                </div>
                <div className="message-actions">
                  <button 
                    className="action-btn"
                    onClick={() => copyToClipboard(message.text)}
                    title="Copy to clipboard"
                    aria-label="Copy to clipboard"
                  >
                    <Copy size={16} />
                  </button>

                  {query_id && !query_id.startsWith('temp_msg_') && !query_id.startsWith('error_') && (
                    <>
                      {(!currentFeedback && !feedbackAcknowledged) && (
                        <>
                          <button 
                            className={thumbsUpClass}
                            onClick={() => {
                              setActiveFeedback(prev => ({ ...prev, [query_id]: 'positive' }));
                              submitFeedback(query_id, 'positive');
                              setFeedbackAcknowledged(true);
                              setTimeout(() => setFeedbackAcknowledged(false), 2000);
                            }}
                            title="This was helpful"
                          >
                            <ThumbsUp size={16} />
                          </button>
                          <button 
                            className={thumbsDownClass}
                            onClick={() => {
                              setActiveFeedback(prev => ({ ...prev, [query_id]: 'negative' }));
                              submitFeedback(query_id, 'negative');
                              setFeedbackAcknowledged(true);
                              setTimeout(() => setFeedbackAcknowledged(false), 2000);
                            }}
                            title="This was not helpful"
                          >
                            <ThumbsDown size={16} />
                          </button>
                        </>
                      )}

                      {/* Already submitted feedback view */}
                      {currentFeedback === 'positive' && (
                        <ThumbsUp 
                          className="action-btn thumbs-up active clicked" 
                          size={16} 
                          stroke="#cc0000" 
                          title="You marked this helpful" 
                        />
                      )}
                      {currentFeedback === 'negative' && (
                        <ThumbsDown 
                          className="action-btn thumbs-down active clicked" 
                          size={16} 
                          stroke="#cc0000" 
                          title="You marked this not helpful" 
                        />
                      )}
                    </>
                  )}

                  {feedbackAcknowledged && (
                    <div className="feedback-thanks">Thank you for your feedback!</div>
                  )}
                </div>
              </div>

              <div className="message-tabs">
                <button 
                  className={`tab ${activeTab === 'answer' ? 'active' : ''}`}
                  onClick={() => handleTabChange('answer')}
                >
                  Answer
                </button>
                <button 
                  className={`tab ${activeTab === 'sources' ? 'active' : ''}`}
                  onClick={() => handleTabChange('sources')}
                >
                  Sources
                </button>
              </div>
            </>
          )}

          <div className="message-text">
            {message.sender === 'bot' ? (
              activeTab === 'sources' ? (
                message.sources ? (
                  <div className="sources">{message.sources}</div>
                ) : (
                  <div className="no-sources">No sources available for this response.</div>
                )
              ) : (
                <div dangerouslySetInnerHTML={{ __html: marked.parse(message.text || '') }} />
              )
            ) : (
              message.text
            )}
          </div>
        </div>
      </div>
    </div>
  );
});

// Cloud storage service for persisting conversations
const cloudStorage = {
  // Save conversations to server
  saveConversations: async (userId, conversations) => {
    try {
      const response = await fetch(`${process.env.REACT_APP_API_BASE_URL}/api/conversations/save`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ userId, conversations }),
      });
      
      return response.ok;
    } catch (error) {
      console.warn('Error saving conversations:', error);
      return false;
    }
  },
  
  // Load conversations from server
  loadConversations: async (userId) => {
    try {
      const response = await fetch(`${process.env.REACT_APP_API_BASE_URL}/api/conversations/load?userId=${userId}`);
      
      if (!response.ok) {
        return null;
      }
      
      const data = await response.json();
      return data.conversations;
    } catch (error) {
      console.warn('Error loading conversations:', error);
      return null;
    }
  },
  
  // Save active conversation ID
  saveActiveConversation: async (userId, conversationId) => {
    try {
      const response = await fetch(`${process.env.REACT_APP_API_BASE_URL}/api/conversations/active`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ userId, activeConversationId: conversationId }),
      });
      
      return response.ok;
    } catch (error) {
      console.warn('Error saving active conversation:', error);
      return false;
    }
  },
  
  // Load active conversation ID
  loadActiveConversation: async (userId) => {
    try {
      const response = await fetch(`${process.env.REACT_APP_API_BASE_URL}/api/conversations/active?userId=${userId}`);
      
      if (!response.ok) {
        return null;
      }
      
      const data = await response.json();
      return data.activeConversationId;
    } catch (error) {
      console.warn('Error loading active conversation:', error);
      return null;
    }
  }
};

// Default conversation template
const DEFAULT_CONVERSATION = {
  id: 'new',
  title: 'New Chat',
  messages: [{
    sender: 'bot',
    text: "Hi! I am Northeastern University Assistant. What can I help with?",
    activeTab: 'answer',
    query_id: 'welcome_message',
    showInitialMessage: true
  }],
  date: new Date().toISOString(),
  feedback: {}
};

// Main App component
function App({ user, onLogout }) {
  const [showMenu, setShowMenu] = useState(false);
  const apiBaseUrl = process.env.REACT_APP_API_BASE_URL || '/api';

  const handleLogout = () => {
    // Clear storage to prevent data leakage
    sessionStorage.clear();
    localStorage.removeItem('activeConversationId');
    
    // Reset application state
    setConversations({ new: {...DEFAULT_CONVERSATION} });
    setActiveConversationId('new');
    setActiveFeedback({});
    setUserId(`user_${Date.now()}`);
    
    if (onLogout) onLogout();
    
    // Hard reload
    window.location.href = '/';
  };

  // User ID for storage
  const [userId, setUserId] = useState(() => {
    // Always prioritize the user ID from login
    if (user?.userId) return user.userId;
    
    // Fallback to existing saved ID
    const existing = sessionStorage.getItem('userId');
    if (existing) return existing;

    // Last resort: create a new one
    const newId = `user_${Date.now()}`;
    sessionStorage.setItem('userId', newId);
    return newId;
  });

  useEffect(() => {
    if (user?.userId && user.userId !== userId) {
      setUserId(user.userId);
      sessionStorage.setItem('userId', user.userId);
    }
  }, [user?.userId]);

  // Theme state
  const [darkMode, setDarkMode] = useState(() => {
    const savedPreference = localStorage.getItem('darkMode');
    return savedPreference === 'true';
  });

  // Chat settings
  const [incognitoMode, setIncognitoMode] = useState(false);
  const [deepSearchMode, setDeepSearchMode] = useState(false);
  const [namespace, setNamespace] = useState("default");
  const [searchMode, setSearchMode] = useState("direct");

  // Feedback state
  const [activeFeedback, setActiveFeedback] = useState({});

  // Conversation states with default value
  const [activeConversationId, setActiveConversationId] = useState('new');
  const [conversations, setConversations] = useState({
    new: {...DEFAULT_CONVERSATION}
  });

  // UI states
  const [isInitializing, setIsInitializing] = useState(true);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // References for DOM elements
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  // Save conversations to server
  const saveConversationsToServer = async () => {
    if (isInitializing || incognitoMode) return;
    
    try {
      await cloudStorage.saveConversations(userId, conversations);
      await cloudStorage.saveActiveConversation(userId, activeConversationId);
    } catch (error) {
      console.error("Error saving conversations:", error);
    }
  };

  // Initialize application on first load
  useEffect(() => {
    const storedUser = sessionStorage.getItem('previousUser');
    const currentUser = user?.userId || 'guest';
    
    if (storedUser && storedUser !== currentUser) {
      console.log("User changed - resetting state");
      setConversations({ new: {...DEFAULT_CONVERSATION} });
      setActiveConversationId('new');
      setActiveFeedback({});
    }
    
    sessionStorage.setItem('previousUser', currentUser);
  }, [user?.userId]);

  useEffect(() => {
    if (isInitializing && !incognitoMode) {
      const currentUserId = user?.userId || userId;
      
      Promise.all([
        cloudStorage.loadConversations(currentUserId),
        cloudStorage.loadActiveConversation(currentUserId)
      ]).then(([loadedConvs, activeId]) => {
        if (loadedConvs && Object.keys(loadedConvs).length > 0) {
          setConversations(loadedConvs);
          
          if (activeId && loadedConvs[activeId]) {
            setActiveConversationId(activeId);
          } else {
            // Default to first conversation or new
            const firstId = Object.keys(loadedConvs)[0];
            setActiveConversationId(firstId || 'new');
          }
        }
        
        setIsInitializing(false);
      }).catch(err => {
        console.error("Error initializing:", err);
        setIsInitializing(false);
      });
    }
  }, [isInitializing, incognitoMode, userId, user]);
 
  // Get current active conversation with robust fallback
  const activeConversation =
    conversations && conversations[activeConversationId] && Array.isArray(conversations[activeConversationId].messages)
      ? conversations[activeConversationId]
      : conversations && conversations.new && Array.isArray(conversations.new.messages)
        ? conversations.new
        : { ...DEFAULT_CONVERSATION };

  // Suggested questions for welcome screen
  const suggestedQuestions = [
    "What are the admission requirements for master/PhD programs?",
    "Tell me about housing options for graduate students",
    "How do I apply for research assistantships?",
    "What student resources are available for international students?"
  ];

  // Save conversations with debouncing
  useEffect(() => {
    const saveTimer = setTimeout(saveConversationsToServer, 500);
    return () => clearTimeout(saveTimer);
  }, [conversations, activeConversationId, userId, incognitoMode, isInitializing]);

  // Auto scroll when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeConversation?.messages?.length]);

  // Auto resize textarea based on content
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [input]);

  // Apply dark mode
  useEffect(() => {
    if (darkMode) {
      document.body.classList.add('dark-theme');
    } else {
      document.body.classList.remove('dark-theme');
    }
    localStorage.setItem('darkMode', darkMode);
  }, [darkMode]);

  // Set the document title
  useEffect(() => {
    document.title = "Ask NEU";
  }, []);
  
  // Hide greeting message after user sends first message
  useEffect(() => {
    if (activeConversation?.messages?.length > 1 && 
      activeConversation.messages[0].showInitialMessage) {
      const updatedConversations = { ...conversations };
      const currentConv = { ...updatedConversations[activeConversationId] };
      if (!Array.isArray(currentConv.messages)) currentConv.messages = [];
      
      if (currentConv?.messages?.length > 0) {
        currentConv.messages[0] = {
          ...currentConv.messages[0],
          showInitialMessage: false
        };
      }
      
      updatedConversations[activeConversationId] = currentConv;
      setConversations(updatedConversations);
    }
  }, [activeConversation?.messages?.length, activeConversationId, conversations]);

  // Send a message to the backend
  const sendMessage = async () => {
    if (!input.trim()) return;

    // Update conversations state
    const updatedConversations = { ...conversations };
    const currentConv = { ...updatedConversations[activeConversationId] };
    
    // Ensure messages array exists
    if (!Array.isArray(currentConv.messages)) {
      currentConv.messages = [];
    }
    
    // Update title if this is the first user message
    if (!currentConv.title || currentConv.title === 'New Chat') {
      currentConv.title = input.length > 30 
        ? input.substring(0, 30) + '...' 
        : input;
    }
    
    // Add user message to conversation
    currentConv.messages.push({ 
      sender: 'user', 
      text: input, 
      timestamp: new Date().toISOString() 
    });
    
    updatedConversations[activeConversationId] = currentConv;
    setConversations(updatedConversations);
    
    // Clear input field and set loading state
    setInput('');
    setIsLoading(true);

    // Make API request
    try {
      const actualSearchMode = deepSearchMode ? "deepsearch" : searchMode;
      
      const res = await fetch(`${process.env.REACT_APP_API_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          query: input, 
          namespace: namespace, 
          search_mode: actualSearchMode,
          userId: userId
        })
      });
      
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      
      const data = await res.json();
      
      // Update conversations state with bot response
      const withResponseConversations = { ...updatedConversations };
      const withResponseConv = { ...withResponseConversations[activeConversationId] };
      
      // Create bot message for conversation
      const responseMessage = { 
        sender: 'bot', 
        text: data.answer || "Sorry, I couldn't generate a response.",
        timestamp: new Date().toISOString(),
        sources: Array.isArray(data.sources) ? data.sources.join("\n") : (data.sources || ""),
        query_id: data.query_id,
        processingTime: data.processing_time || 0,
        searchMode: data.search_mode || actualSearchMode,
        activeTab: 'answer'
      };
      
      withResponseConv.messages.push(responseMessage);
      
      // Initialize feedback for this message
      if (!withResponseConv.feedback) {
        withResponseConv.feedback = {};
      }
      
      withResponseConversations[activeConversationId] = withResponseConv;
      setConversations(withResponseConversations);

    } catch (err) {
      console.error("Error in API call:", err.message);
      
      // Add error message to conversation
      const withErrorConversations = { ...updatedConversations };
      const withErrorConv = { ...withErrorConversations[activeConversationId] };
      
      // Create error message
      const errorResponseMessage = { 
        sender: 'bot', 
        text: 'Could not contact backend. Check connection or try again later.',
        timestamp: new Date().toISOString(),
        query_id: `error_${Date.now()}`,
        activeTab: 'answer'
      };
      
      withErrorConv.messages.push(errorResponseMessage);
      withErrorConversations[activeConversationId] = withErrorConv;
      setConversations(withErrorConversations);
      
    } finally {
      setIsLoading(false);
    }
  };

  // Handle Enter key in textarea
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // UI toggle functions
  const toggleSidebar = () => setSidebarOpen(!sidebarOpen);
  const toggleDarkMode = () => setDarkMode(!darkMode);
  const toggleDeepSearch = () => setDeepSearchMode(!deepSearchMode);
  
  // Set namespace or perform an action
  const handleSpecialAction = (action) => {
    if (action === "course" || action === "classroom") {
      setNamespace(prev => (prev === action ? "default" : action));
    } else if (action === "deepsearch") {
      setDeepSearchMode(true);
    }
  };
  
  // Toggle incognito mode
  const toggleIncognitoMode = () => {
    if (!incognitoMode) {
      startNewChat();
    } else {
      // If turning off incognito, reload conversations
      cloudStorage.loadConversations(userId).then(storedConversations => {
        if (storedConversations) {
          setConversations(storedConversations);
          
          cloudStorage.loadActiveConversation(userId).then(storedActiveId => {
            if (storedActiveId && storedConversations[storedActiveId]) {
              setActiveConversationId(storedActiveId);
            }
          });
        }
      });
    }
    setIncognitoMode(!incognitoMode);
  };

  // Handle suggested question click
  const handleSuggestedQuestion = (question) => {
    // Add the user message to the UI
    const updatedConversations = { ...conversations };
    const currentConv = { ...updatedConversations[activeConversationId] };
    if (!Array.isArray(currentConv.messages)) currentConv.messages = [];
    
    // Update title if this is the first message
    if (!currentConv.title || currentConv.title === 'New Chat') {
      currentConv.title = question.length > 30 
        ? question.substring(0, 30) + '...' 
        : question;
    }
    
    // Add user message
    currentConv.messages.push({ 
      sender: 'user', 
      text: question, 
      timestamp: new Date().toISOString() 
    });
    
    updatedConversations[activeConversationId] = currentConv;
    setConversations(updatedConversations);
    
    // Clear input field and set loading state
    setInput('');
    setIsLoading(true);
    
    // Send message to API
    const actualSearchMode = deepSearchMode ? "deepsearch" : searchMode;
    
     fetch(`${process.env.REACT_APP_API_BASE_URL}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        query: question,
        namespace: namespace,
        search_mode: actualSearchMode,
        userId: userId
      })
    })
    .then(response => {
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return response.json();
    })
    .then(data => {
      // Update conversation with the bot's response
      const withResponseConversations = { ...updatedConversations };
      const withResponseConv = { ...withResponseConversations[activeConversationId] };
      
      const responseMessage = { 
        sender: 'bot', 
        text: data.answer || "Sorry, I couldn't generate a response.",
        timestamp: new Date().toISOString(),
        sources: Array.isArray(data.sources) ? data.sources.join("\n") : (data.sources || ""),
        query_id: data.query_id,
        processingTime: data.processing_time || 0,
        searchMode: data.search_mode || actualSearchMode,
        activeTab: 'answer'
      };
      
      withResponseConv.messages.push(responseMessage);
      
      // Initialize feedback for this message
      if (!withResponseConv.feedback) {
        withResponseConv.feedback = {};
      }
      
      withResponseConversations[activeConversationId] = withResponseConv;
      setConversations(withResponseConversations);
      setIsLoading(false);
    })
    .catch(error => {
      console.error('Error in API call:', error);
      
      // Add error message
      const updatedWithError = { ...updatedConversations };
      const convWithError = { ...updatedWithError[activeConversationId] };
      
      convWithError.messages.push({ 
        sender: 'bot', 
        text: 'Sorry, I encountered an error connecting to the backend.',
        timestamp: new Date().toISOString(),
        query_id: `error_${Date.now()}`,
        activeTab: 'answer'
      });
      
      updatedWithError[activeConversationId] = convWithError;
      setConversations(updatedWithError);
      setIsLoading(false);
    });
  };
  
  // Start a new chat
  const startNewChat = () => {
    const newId = 'conv_' + Date.now();
    const newConversation = {
      ...DEFAULT_CONVERSATION,
      id: newId,
      userId: userId
    };
    
    setConversations(prev => ({
      ...prev,
      [newId]: newConversation
    }));
    
    setActiveConversationId(newId);
  };
  
  // Select a conversation
  const selectConversation = (id) => {
    setActiveConversationId(id);
  };
  
  // Delete a conversation
  const deleteConversation = async (id, e) => {
    e.stopPropagation();

    try {
      // Remove from database
      await fetch(`${process.env.REACT_APP_API_BASE_URL}/api/conversations/delete`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ userId, conversationId: id })
      });
    } catch (error) {
      console.error('Error deleting conversation from server:', error);
    }

    // Update state
    const updatedConversations = { ...conversations };
    delete updatedConversations[id];
    setConversations(updatedConversations);

    // Reset active chat if needed
    if (id === activeConversationId) {
      const remainingIds = Object.keys(updatedConversations);
      if (remainingIds.length > 0) {
        setActiveConversationId(remainingIds[0]);
      } else {
        startNewChat();
      }
    }
  };

  // Submit user feedback on a message
  const submitFeedback = async (messageIndexOrQueryId, rating) => {
    // Check if we received a query_id directly (string) or an index (number)
    const isQueryId = typeof messageIndexOrQueryId === 'string';
    
    let message;
    let query_id;
    
    if (isQueryId) {
      // We got a query_id directly
      query_id = messageIndexOrQueryId;
      const conversation = conversations[activeConversationId];
      if (!conversation || !Array.isArray(conversation.messages)) {
        console.error("Cannot submit feedback: Invalid conversation");
        return;
      }
      
      message = conversation.messages.find(msg => msg.query_id === query_id);
      if (!message) {
        // Use the query_id directly without requiring a message
        message = { sender: 'bot', query_id };
      }
    } else {
      // We got an index, get the message at that index
      const messageIndex = messageIndexOrQueryId;
      const conversation = conversations[activeConversationId];
      if (!conversation || !Array.isArray(conversation.messages)) {
        console.error("Cannot submit feedback: Invalid conversation");
        return;
      }
      
      message = conversation.messages[messageIndex];
      if (!message) {
        console.error(`Cannot submit feedback: Invalid message index ${messageIndex}`);
        return;
      }
      
      query_id = message.query_id;
      if (!query_id) {
        console.error(`Message at index ${messageIndex} has no query_id`);
        return;
      }
    }
    
    // Update UI state immediately for responsiveness
    setActiveFeedback(prev => ({ ...prev, [query_id]: rating }));
    
    // Update persistent state in conversations
    setConversations(prev => {
      const updated = { ...prev };
      if (!updated[activeConversationId].feedback) {
        updated[activeConversationId].feedback = {};
      }
      updated[activeConversationId].feedback[query_id] = rating;
      return updated;
    });

    // Skip API call if in incognito mode
    if (incognitoMode) {
      return;
    }

    // Make API call to save feedback
    try {
      const response = await fetch(`${process.env.REACT_APP_API_BASE_URL}/api/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query_id: query_id,
          rating: rating,
          userId: userId,
          chatId: activeConversationId
        })
      });
      
      if (!response.ok) {
        console.error(`Feedback API call failed: ${response.status}`);
      }
    } catch (error) {
      console.error("Error submitting feedback to server:", error);
    }
  };

  // Copy text to clipboard
  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text)
      .then(() => {
        // Temporary "Copied!" notification
        const copiedEl = document.createElement('div');
        copiedEl.className = 'copy-notification';
        copiedEl.innerText = 'Copied!';
        document.body.appendChild(copiedEl);
        
        setTimeout(() => {
          document.body.removeChild(copiedEl);
        }, 2000);
      })
      .catch(err => {
        console.error('Could not copy text: ', err);
      });
  };
  
  // Get feedback for a message
  const getFeedbackForMessage = useCallback((query_id) => {
    if (!query_id) return null;
    
    // First check activeFeedback state (temporary UI state)
    if (activeFeedback && query_id in activeFeedback) {
      return activeFeedback[query_id];
    }
    
    // Then check persistent state
    if (activeConversation?.feedback && query_id in activeConversation.feedback) {
      return activeConversation.feedback[query_id];
    }
    
    return null;
  }, [activeConversation, activeFeedback]);

  // Group conversations by date
  function getConversationGroups() {
    const todayConversations = [];
    const previousConversations = [];

    const today = new Date();
    today.setHours(0, 0, 0, 0);

    Object.values(conversations || {}).forEach(conv => {
      if (!conv || !Array.isArray(conv.messages) || conv.messages.length === 0) return;

      const lastMessage = conv.messages[conv.messages.length - 1];
      const lastMessageDate = new Date(lastMessage.timestamp || conv.date);

      if (lastMessageDate >= today) {
        todayConversations.push(conv);
      } else {
        previousConversations.push(conv);
      }
    });

    // Sort by latest message timestamp (newest first)
    const sortByLatest = (a, b) => {
      const timeA = new Date(a.messages[a.messages.length - 1]?.timestamp || a.date);
      const timeB = new Date(b.messages[b.messages.length - 1]?.timestamp || b.date);
      return timeB - timeA;
    };

    todayConversations.sort(sortByLatest);
    previousConversations.sort(sortByLatest);

    return { todayConversations, previousConversations };
  }

  // Filter out messages that are hidden with null check
  const visibleMessages = activeConversation?.messages ? 
    activeConversation.messages.filter(
      message => message.sender !== 'bot' || message.showInitialMessage !== false
    ) : [];

  const { todayConversations, previousConversations } = getConversationGroups();

  return (
    <div className="app">
      {/* Sidebar */}
      <div className={`sidebar ${sidebarOpen ? 'open' : 'closed'}`}>
        <div className="sidebar-header">
          <button className="new-chat-btn" onClick={startNewChat}>
            <MessageSquare size={16} />
            <span>New Chat</span>
          </button>
          <button className="toggle-btn" onClick={toggleSidebar}>
            <ChevronRight size={16} />
          </button>
        </div>
        
        <div className="conversation-list">
          {incognitoMode && (
            <div className="incognito-indicator">
              <EyeOff size={12} />
              <span>Incognito Mode - Conversations won't be saved</span>
            </div>
          )}
          
          {todayConversations.length > 0 && (
            <>
              <div className="conversation-header">Today</div>
              {todayConversations.map(conv => (
                <div 
                  key={conv.id} 
                  className={`conversation-item ${activeConversationId === conv.id ? 'active' : ''}`}
                  onClick={() => selectConversation(conv.id)}
                >
                  <MessageSquare size={14} className="conversation-icon" />
                  <span>{conv.title}</span>
                  <button 
                    className="delete-conv-btn"
                    onClick={(e) => deleteConversation(conv.id, e)}
                    aria-label="Delete conversation"
                  >
                    <X size={14} />
                  </button>
                </div>
              ))}
            </>
          )}
          
          {previousConversations.length > 0 && (
            <>
              <div className="conversation-header">Previous 7 Days</div>
              {previousConversations.map(conv => (
                <div 
                  key={conv.id} 
                  className={`conversation-item ${activeConversationId === conv.id ? 'active' : ''}`}
                  onClick={() => selectConversation(conv.id)}
                >
                  <MessageSquare size={14} className="conversation-icon" />
                  <span>{conv.title}</span>
                  <button 
                    className="delete-conv-btn"
                    onClick={(e) => deleteConversation(conv.id, e)}
                    aria-label="Delete conversation"
                  >
                    <X size={14} />
                  </button>
                </div>
              ))}
            </>
          )}
        </div>
      </div>

      {/* Main content */}
      <div className={`main-content ${sidebarOpen ? '' : 'expanded'}`}>
        <div className="user-bar">
          
          
        </div>    
		<header className="main-header">
		  {/* Left-side (namespace display) */}
		  <div className="header-left">
			{!sidebarOpen && (
			  <button className="menu-btn" onClick={toggleSidebar} aria-label="Toggle sidebar">
				<MessageSquare size={20} />
			  </button>
			)}
			{namespace !== 'default' && (
			  <div className="namespace-status">
				Namespace: <strong>{namespace}</strong>
			  </div>
			)}
		  </div>

		  {/* Center with logo and title */}
		  <div className="header-center">
			<img src="/logo.png" alt="NEU Logo" className="neu-logo" />
			<span className="header-title">Ask NEU</span>
		  </div>

		  {/* Right-side icons */}
		  <div className="header-right">
			<button
			  className={`mode-toggle ${incognitoMode ? 'active' : ''}`}
			  onClick={toggleIncognitoMode}
			  title={incognitoMode ? "Turn Off Incognito Mode" : "Turn On Incognito Mode"}
			  aria-label={incognitoMode ? "Turn Off Incognito Mode" : "Turn On Incognito Mode"}
			>
			  {incognitoMode ? <EyeOff size={18} /> : <Eye size={18} />}
			</button>
			<button
			  className="theme-toggle"
			  onClick={toggleDarkMode}
			  title={darkMode ? "Switch to Light Mode" : "Switch to Dark Mode"}
			  aria-label={darkMode ? "Switch to Light Mode" : "Switch to Dark Mode"}
			>
			  {darkMode ? <Sun size={18} /> : <Moon size={18} />}
			</button>
			<div className="user-menu">
			  <img
				src={user?.picture || `https://ui-avatars.com/api/?name=${user?.name || 'G'}&background=cc0000&color=fff`}
				alt="User Avatar"
				className="user-avatar"
				onClick={() => setShowMenu(!showMenu)}
			  />
			  {showMenu && (
				<div className="user-dropdown">
				  <div className="user-name">{user?.name || 'Guest'}</div>
				  <button onClick={user?.email === 'guest@askneu.ai' ? () => {
					// Clear user from session storage to trigger login screen
					sessionStorage.removeItem('user');
					window.location.reload();
				  } : handleLogout}>
					{user?.email === 'guest@askneu.ai' ? 'Login' : 'Logout'}
				  </button>
				</div>
			  )}
			</div>
		  </div>
		</header>		
        
        <div className="chat-container">
		{activeConversation && activeConversation.messages && 
		 activeConversation.messages.length === 1 && 
		 activeConversation.messages[0].sender === 'bot' ? (
		  <div className="welcome-screen">
			<div className="neu-logo-large">
			  <img src="/logo.png" alt="NEU Logo" />
			</div>
			<h2>Hi! I am Northeastern University Assistant. What can I help with?</h2>
			
			<div className="suggested-questions-grid">
			  <button 
				className="suggested-question"
				onClick={() => handleSuggestedQuestion("What are the admission requirements for master/PhD programs?")}
			  >
				<MessageSquare size={18} className="suggested-question-icon" />
				<span>What are the admission requirements for master/PhD programs?</span>
			  </button>
			  
			  <button 
				className="suggested-question"
				onClick={() => handleSuggestedQuestion("Tell me about housing options for graduate students")}
			  >
				<MessageSquare size={18} className="suggested-question-icon" />
				<span>Tell me about housing options for graduate students</span>
			  </button>
			  
			  <button 
				className="suggested-question"
				onClick={() => handleSuggestedQuestion("How do I apply for research assistantships?")}
			  >
				<MessageSquare size={18} className="suggested-question-icon" />
				<span>How do I apply for research assistantships?</span>
			  </button>
			  
			  <button 
				className="suggested-question"
				onClick={() => handleSuggestedQuestion("What student resources are available for international students?")}
			  >
				<MessageSquare size={18} className="suggested-question-icon" />
				<span>What student resources are available for international students?</span>
			  </button>
			</div>
			
			<div className="namespace-options">
			  <button
				className={`namespace-option ${namespace === 'course' ? 'active' : ''}`}
				onClick={() => handleSpecialAction('course')}
			  >
				<BookOpen 
				  size={18} 
				  style={{
					  color: namespace === 'course' ? 'white' : '#3b82f6'}}/>
				Course information
			  </button>

			  <button
				className={`namespace-option ${namespace === 'classroom' ? 'active' : ''}`}
				onClick={() => handleSpecialAction('classroom')}
			  >
				<School
				  size={18} 
				  style={{
					  color: namespace === 'classroom' ? 'white' : '#a855f7'}}/>
				Classroom resources
			  </button>
			</div>
		  </div>
		) : (
		  <div className="messages">
              {visibleMessages.map((message, index) => (
                <MessageComponent
                  key={`${message.query_id || `msg_${index}_${message.timestamp}`}`}
                  message={message}
                  index={index}
                  setConversations={setConversations}
                  conversations={conversations}
                  activeConversationId={activeConversationId}
                  submitFeedback={submitFeedback}
                  copyToClipboard={copyToClipboard}
                  getFeedbackForMessage={getFeedbackForMessage}
                  activeFeedback={activeFeedback}
                  activeConversation={activeConversation}
                  setActiveFeedback={setActiveFeedback}
                />
              ))}
              
              {/* Loading indicator */}
              {isLoading && activeConversation && (
                <div className="message bot">
                  <div className="message-container">
                    <div className="message-content">
                      <div className="message-header">
                        <div className="message-sender">NEU Assistant</div>
                      </div>
                      <div className="message-text">
                        <div className="typing-indicator">
                          <div className="typing-dot"></div>
                          <div className="typing-dot"></div>
                          <div className="typing-dot"></div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>
        
        {/* Input area */}
        <div className="input-area">
          <div className="input-container-wrapper">
            <div className="input-container">
            
              {/* Namespace icons */}
              {!(activeConversation && 
                 activeConversation.messages && 
                 activeConversation.messages.length === 1 && 
                 activeConversation.messages[0].sender === 'bot') && (
                <div className="namespace-floating-buttons">
                  <button
                    className={`namespace-icon-btn course ${namespace === 'course' ? 'active' : ''}`}
                    onClick={() => setNamespace(namespace === 'course' ? 'default' : 'course')}
                    title="Course"
                  >
                    <BookOpen size={16} />
                  </button>

                  <button
                    className={`namespace-icon-btn classroom ${namespace === 'classroom' ? 'active' : ''}`}
                    onClick={() => setNamespace(namespace === 'classroom' ? 'default' : 'classroom')}
                    title="Classroom"
                  >
                    <School size={16} />
                  </button>
                </div>
              )}
              <textarea 
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyPress}
                placeholder="Ask a question about Northeastern University..."
                rows="1"
              />              
              {/* Deep search toggle */}
              <div className="search-tools">
                <button 
                  className={`search-tool-btn ${deepSearchMode ? 'active' : ''}`}
                  onClick={toggleDeepSearch}
                  title={deepSearchMode ? "Disable Deep Research" : "Enable Deep Research"}
                >
                  <Radar size={20} />
                </button>
              </div>
              
              {/* Send button */}
              <div className="input-buttons">
                <button 
                  className="send-btn"
                  onClick={sendMessage}
                  disabled={isLoading || input.trim() === ''}
                  aria-label="Send message"
                >
                  <Send size={16} color="white" />
                </button>
              </div>
            </div>
            
            <div className="footer">
              <span>Your Campus Companion - Built by Huskies, For Huskies</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
