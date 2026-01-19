import { useState, useEffect } from 'react';
import { getMemories, clearMemories } from '../api';

export default function Sidebar({
  conversations,
  activeConvId,
  onSelectConv,
  onNewChat,
  onDeleteConv,
  isOpen,
  onToggle,
}) {
  const [hoveredId, setHoveredId] = useState(null);
  const [showMemory, setShowMemory] = useState(false);
  const [memories, setMemories] = useState([]);
  const [memoryStats, setMemoryStats] = useState({ total: 0, conversations: 0, facts: 0 });
  const [loadingMemories, setLoadingMemories] = useState(false);

  const loadMemories = async () => {
    setLoadingMemories(true);
    try {
      const data = await getMemories();
      setMemories(data.memories || []);
      setMemoryStats(data.stats || { total: 0, conversations: 0, facts: 0 });
    } catch (err) {
      console.error('Failed to load memories:', err);
    } finally {
      setLoadingMemories(false);
    }
  };

  const handleClearMemories = async () => {
    if (!confirm('Clear all saved memories? This cannot be undone.')) return;
    try {
      await clearMemories();
      setMemories([]);
      setMemoryStats({ total: 0, conversations: 0, facts: 0 });
    } catch (err) {
      console.error('Failed to clear memories:', err);
    }
  };

  useEffect(() => {
    if (showMemory) {
      loadMemories();
    }
  }, [showMemory]);

  if (!isOpen) return null;

  return (
    <aside className="w-64 flex flex-col bg-slate-950 border-r border-slate-800 h-full">
      {/* Header */}
      <div className="p-3 border-b border-slate-800 space-y-2">
        <button
          onClick={onNewChat}
          className="w-full flex items-center gap-2 px-3 py-2.5 rounded-lg border border-slate-700 hover:bg-slate-800 transition text-sm"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          New chat
        </button>
        <button
          onClick={() => setShowMemory(!showMemory)}
          className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg transition text-sm ${
            showMemory ? 'bg-purple-600/20 text-purple-400 border border-purple-600/50' : 'text-slate-400 hover:bg-slate-800'
          }`}
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
          </svg>
          Memory
          {memoryStats.total > 0 && (
            <span className="ml-auto text-xs bg-purple-600/30 px-1.5 py-0.5 rounded-full">
              {memoryStats.total}
            </span>
          )}
        </button>
      </div>

      {/* Memory Panel */}
      {showMemory && (
        <div className="p-3 border-b border-slate-800 bg-slate-900/50">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-slate-400 uppercase">Your Memory</span>
            <button
              onClick={handleClearMemories}
              className="text-xs text-red-400 hover:text-red-300"
              title="Clear all memories"
            >
              Clear all
            </button>
          </div>
          
          {/* Memory Stats */}
          {memoryStats.total > 0 && (
            <div className="flex gap-2 mb-2 text-xs">
              <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-400">
                💬 {memoryStats.conversations} chats
              </span>
              <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-400">
                📌 {memoryStats.facts} facts
              </span>
            </div>
          )}
          
          <div className="max-h-48 overflow-y-auto space-y-1">
            {loadingMemories ? (
              <div className="text-xs text-slate-500 text-center py-2">Loading...</div>
            ) : memories.length === 0 ? (
              <div className="text-xs text-slate-500 text-center py-2">
                No memories yet. Everything you share will be remembered for future chats!
              </div>
            ) : (
              memories.slice(0, 20).map((mem, i) => (
                <div key={i} className="text-xs text-slate-400 bg-slate-800/50 rounded px-2 py-1.5 truncate">
                  {mem}
                </div>
              ))
            )}
            {memories.length > 20 && (
              <div className="text-xs text-slate-500 text-center py-1">
                ...and {memories.length - 20} more
              </div>
            )}
          </div>
        </div>
      )}

      {/* Conversations List */}
      <nav className="flex-1 overflow-y-auto p-2 space-y-1">
        {!showMemory && conversations.length === 0 ? (
          <div className="text-center text-slate-500 text-sm mt-8 px-4">
            No conversations yet. Start a new chat!
          </div>
        ) : (
          conversations.map((conv) => (
            <div
              key={conv.id}
              className={`group relative flex items-center gap-2 px-3 py-2.5 rounded-lg cursor-pointer transition text-sm ${
                activeConvId === conv.id
                  ? 'bg-slate-800 text-white'
                  : 'text-slate-400 hover:bg-slate-800/50 hover:text-slate-200'
              }`}
              onClick={() => onSelectConv(conv.id)}
              onMouseEnter={() => setHoveredId(conv.id)}
              onMouseLeave={() => setHoveredId(null)}
            >
              <svg className="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
                />
              </svg>
              <span className="flex-1 truncate">{conv.title}</span>

              {/* Delete button */}
              {hoveredId === conv.id && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteConv(conv.id);
                  }}
                  className="p-1 rounded hover:bg-slate-700 text-slate-400 hover:text-red-400"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                    />
                  </svg>
                </button>
              )}
            </div>
          ))
        )}
      </nav>

      {/* Footer */}
      <div className="p-3 border-t border-slate-800">
        <div className="flex items-center gap-2 px-3 py-2 text-sm text-slate-500">
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white font-semibold text-xs">
            U
          </div>
          <span>User</span>
        </div>
      </div>
    </aside>
  );
}
