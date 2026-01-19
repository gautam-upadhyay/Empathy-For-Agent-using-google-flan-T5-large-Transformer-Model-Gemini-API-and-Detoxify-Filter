const API_BASE = '/api';

// Get or create a persistent user ID
export function getUserId() {
  let userId = localStorage.getItem('empathy_user_id');
  if (!userId) {
    userId = 'user_' + Date.now().toString(36) + Math.random().toString(36).substr(2, 9);
    localStorage.setItem('empathy_user_id', userId);
  }
  return userId;
}

export async function sendMessage(message, useFilter, conversationId = '') {
  const response = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message,
      use_filter: useFilter,
      user_id: getUserId(),
      conversation_id: conversationId,
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }

  return response.json();
}

export async function checkHealth() {
  const response = await fetch(`${API_BASE}/health`);
  return response.json();
}

export async function getMemories() {
  const response = await fetch(`${API_BASE}/memory?user_id=${getUserId()}`);
  return response.json();
}

export async function clearMemories() {
  const response = await fetch(`${API_BASE}/memory?user_id=${getUserId()}`, {
    method: 'DELETE',
  });
  return response.json();
}

export async function saveMemory(fact) {
  const response = await fetch(`${API_BASE}/memory`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ fact, user_id: getUserId() }),
  });
  return response.json();
}
