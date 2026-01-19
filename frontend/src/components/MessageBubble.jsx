export default function MessageBubble({ msg }) {
  const isUser = msg.role === 'user';

  return (
    <div className={`flex items-start gap-4 ${isUser ? 'flex-row-reverse' : ''}`}>
      {/* Avatar */}
      <div
        className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
          isUser
            ? 'bg-gradient-to-br from-blue-500 to-purple-600'
            : 'bg-gradient-to-br from-green-400 to-blue-500'
        }`}
      >
        {isUser ? (
          <span className="text-white font-semibold text-xs">U</span>
        ) : (
          <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
        )}
      </div>

      {/* Content */}
      <div className={`flex-1 max-w-[85%] ${isUser ? 'text-right' : ''}`}>
        <div
          className={`inline-block px-4 py-3 rounded-2xl whitespace-pre-wrap break-words text-left ${
            isUser
              ? 'bg-blue-600 text-white'
              : 'bg-slate-800 text-slate-100'
          }`}
        >
          {msg.text}
        </div>

        {/* Meta info (bot only) */}
        {!isUser && (
          <div className="flex flex-wrap items-center gap-2 mt-2 text-xs text-slate-500">
            <span className="flex items-center gap-1">
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
              </svg>
              Filter: {msg.filterOn ? 'on' : 'off'}
            </span>
            {msg.emotion && (
              <span className="px-2 py-0.5 rounded-full bg-slate-700 text-slate-300">
                {msg.emotion}
              </span>
            )}
            {msg.memoriesUsed > 0 && (
              <span className="px-2 py-0.5 rounded-full bg-purple-900/50 text-purple-400 flex items-center gap-1">
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
                memory
              </span>
            )}
            {msg.toxic && (
              <span className="px-2 py-0.5 rounded-full bg-red-900/50 text-red-400">
                filtered
              </span>
            )}
          </div>
        )}

        {/* Facts saved indicator */}
        {!isUser && msg.conversationSaved && (
          <div className="mt-1 text-xs text-purple-400 flex items-center gap-1">
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
            Saved to memory
          </div>
        )}

        {/* Context preview */}
        {!isUser && msg.context && msg.context.length > 0 && (
          <details className="mt-2 text-xs">
            <summary className="text-slate-500 cursor-pointer hover:text-slate-400">
              View context ({msg.context.length} items)
            </summary>
            <div className="mt-1 p-2 rounded-lg bg-slate-800/50 text-slate-400 space-y-1">
              {msg.context.slice(0, 3).map((c, i) => (
                <div key={i} className="truncate">• {c}</div>
              ))}
              {msg.context.length > 3 && (
                <div className="text-slate-500">...and {msg.context.length - 3} more</div>
              )}
            </div>
          </details>
        )}

        {/* Error */}
        {msg.error && (
          <div className="mt-2 text-xs text-red-400 flex items-center gap-1">
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            {msg.error}
          </div>
        )}
      </div>
    </div>
  );
}
