import { useState, useRef, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import { sendMessage, getMemories, clearMemories } from './api';

// Load conversations from localStorage
const loadConversations = () => {
  try {
    const saved = localStorage.getItem('empathy_conversations');
    return saved ? JSON.parse(saved) : [];
  } catch {
    return [];
  }
};

// Save conversations to localStorage
const saveConversations = (convs) => {
  localStorage.setItem('empathy_conversations', JSON.stringify(convs));
};

export default function App() {
  const [conversations, setConversations] = useState(loadConversations);
  const [activeConvId, setActiveConvId] = useState(null);
  const [useFilter, setUseFilter] = useState(true);
  const [loading, setLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const messagesEndRef = useRef(null);

  // Get current conversation
  const activeConv = conversations.find((c) => c.id === activeConvId);
  const messages = activeConv?.messages || [];

  // Persist conversations
  useEffect(() => {
    saveConversations(conversations);
  }, [conversations]);

  // Scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Create new conversation
  const handleNewChat = () => {
    const newConv = {
      id: Date.now().toString(),
      title: 'New chat',
      messages: [],
      createdAt: new Date().toISOString(),
    };
    setConversations((prev) => [newConv, ...prev]);
    setActiveConvId(newConv.id);
  };

  // Delete conversation
  const handleDeleteConv = (id) => {
    setConversations((prev) => prev.filter((c) => c.id !== id));
    if (activeConvId === id) {
      setActiveConvId(null);
    }
  };

  // Send message
  const handleSend = async (text) => {
    if (!text.trim() || loading) return;

    // Create conversation if none active
    let convId = activeConvId;
    if (!convId) {
      const newConv = {
        id: Date.now().toString(),
        title: text.slice(0, 30) + (text.length > 30 ? '...' : ''),
        messages: [],
        createdAt: new Date().toISOString(),
      };
      setConversations((prev) => [newConv, ...prev]);
      convId = newConv.id;
      setActiveConvId(convId);
    }

    // Add user message
    const userMsg = { role: 'user', text, timestamp: Date.now() };
    setConversations((prev) =>
      prev.map((c) =>
        c.id === convId
          ? {
              ...c,
              messages: [...c.messages, userMsg],
              title: c.messages.length === 0 ? text.slice(0, 30) + (text.length > 30 ? '...' : '') : c.title,
            }
          : c
      )
    );

    setLoading(true);

    try {
      const data = await sendMessage(text, useFilter, convId);
      const botMsg = {
        role: 'bot',
        text: data.response || '(no response)',
        emotion: data.emotion,
        context: data.context || [],
        toxic: data.toxic,
        filterOn: useFilter,
        timestamp: Date.now(),
        memoriesUsed: data.memories_used || 0,
        memoryStats: data.memory_stats || {},
        conversationSaved: data.conversation_saved || false,
      };
      setConversations((prev) =>
        prev.map((c) =>
          c.id === convId ? { ...c, messages: [...c.messages, botMsg] } : c
        )
      );
    } catch (err) {
      const errorMsg = {
        role: 'bot',
        text: 'Error contacting server.',
        error: String(err),
        timestamp: Date.now(),
      };
      setConversations((prev) =>
        prev.map((c) =>
          c.id === convId ? { ...c, messages: [...c.messages, errorMsg] } : c
        )
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-slate-900 text-slate-200">
      {/* Sidebar */}
      <Sidebar
        conversations={conversations}
        activeConvId={activeConvId}
        onSelectConv={setActiveConvId}
        onNewChat={handleNewChat}
        onDeleteConv={handleDeleteConv}
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
      />

      {/* Main Chat Area */}
      <ChatArea
        messages={messages}
        useFilter={useFilter}
        setUseFilter={setUseFilter}
        onSend={handleSend}
        loading={loading}
        messagesEndRef={messagesEndRef}
        sidebarOpen={sidebarOpen}
        onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
        hasActiveConv={!!activeConvId}
      />
    </div>
  );
}
