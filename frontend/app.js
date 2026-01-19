const apiBase = "http://127.0.0.1:5000";
const messagesEl = document.getElementById("messages");
const formEl = document.getElementById("chatForm");
const inputEl = document.getElementById("messageInput");
const filterEl = document.getElementById("useFilter");

function addMessage(role, text, meta = "") {
	const wrap = document.createElement("div");
	wrap.className = `msg ${role}`;
	const bubble = document.createElement("div");
	bubble.className = "bubble";
	bubble.textContent = text;
	wrap.appendChild(bubble);
	if (meta) {
		const m = document.createElement("div");
		m.className = "meta";
		m.innerHTML = meta;
		wrap.appendChild(m);
	}
	messagesEl.appendChild(wrap);
	messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function sendMessage(message, useFilter) {
	const payload = { message, use_filter: useFilter };
	const res = await fetch(`${apiBase}/api/chat`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(payload),
	});
	if (!res.ok) {
		const t = await res.text();
		throw new Error(t || `HTTP ${res.status}`);
	}
	return res.json();
}

formEl.addEventListener("submit", async (e) => {
	e.preventDefault();
	const text = inputEl.value.trim();
	if (!text) return;

	const useFilter = filterEl.checked;
	addMessage("user", text);
	inputEl.value = "";
	formEl.querySelector("button").disabled = true;

	try {
		const data = await sendMessage(text, useFilter);
		const emo = data.emotion ? `<span class=\"tag\">${data.emotion}</span>` : "";
		let meta = `Filter: ${useFilter ? "on" : "off"} ${emo}`;
		if (Array.isArray(data.context) && data.context.length) {
			const first = data.context.slice(0, 2).map(c => `- ${c}`).join("<br>");
			meta += `<div class=\"ctx\">${first}</div>`;
		}
		addMessage("bot", data.response || "(no response)", meta);
	} catch (err) {
		addMessage("bot", "Error contacting server.", `<span class=\"error\">${String(err)}</span>`);
	} finally {
		formEl.querySelector("button").disabled = false;
	}
});


