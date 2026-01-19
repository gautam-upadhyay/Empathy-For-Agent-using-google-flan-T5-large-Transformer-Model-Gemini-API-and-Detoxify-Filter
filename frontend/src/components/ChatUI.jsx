import { useState } from 'react';
import MessageBubble from './MessageBubble';

export default function ChatUI({
  messages,
  useFilter,
  setUseFilter,
  onSend,
  loading,
  messagesEndRef,
}) {
  const [input, setInput] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;
    onSend(input);
    setInput('');
  };

  return (
    <>
      {/* Header */}
      <header className="sticky top-0 z-10 flex items-center justify-between px-5 py-4 bg-slate-950 border-b border-slate-700">
        <h1 className="text-lg font-semibold text-white">Empathy Chat</h1>
        <label className="flex items-center gap-2 text-sm text-slate-400 cursor-pointer">
          <input
            type="checkbox"
            className="w-4 h-4 accent-green-500"
            checked={useFilter}
            onChange={(e) => setUseFilter(e.target.checked)}
          />
          <span>Responsible filter</span>
        </label>
      </header>

      {/* Messages */}
      <main className="flex-1 overflow-y-auto px-4 py-5 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-slate-500 mt-20">
            Send a message to start the conversation.
          </div>
        )}
        {messages.map((msg, idx) => (
          <MessageBubble key={idx} msg={msg} />
        ))}
        {loading && (
          <div className="flex gap-2 items-center text-slate-400 animate-pulse">
            <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" />
            <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce delay-100" />
            <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce delay-200" />
          </div>
        )}
        <div ref={messagesEndRef} />
      </main>

      {/* Input */}
      <form
        onSubmit={handleSubmit}
        className="sticky bottom-0 flex gap-3 px-4 py-3 bg-slate-950 border-t border-slate-700"
      >
        <input
          type="text"
          className="flex-1 px-4 py-2.5 rounded-xl bg-slate-800 border border-slate-600 text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
          placeholder="Type your message..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={loading}
        />
        <button
          type="submit"
          disabled={loading || !input.trim()}
          className="px-5 py-2.5 rounded-xl bg-green-500 text-slate-900 font-semibold hover:bg-green-400 disabled:opacity-50 disabled:cursor-not-allowed transition"
        >
          Send
        </button>
      </form>
    </>
  );
}
