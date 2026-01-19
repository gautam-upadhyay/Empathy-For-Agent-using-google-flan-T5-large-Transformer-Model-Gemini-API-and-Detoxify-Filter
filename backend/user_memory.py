"""
User Memory Module - Stores and retrieves ALL user conversation data across sessions.
Uses ChromaDB for semantic search of relevant memories.
Works like ChatGPT memory - stores everything the user shares.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    chromadb = None

from config import settings
from tokenizer_embedder import TextEmbedder


class UserMemory:
    """Manages ALL user conversation data storage and retrieval using ChromaDB."""

    COLLECTION_NAME = "user_memory"

    def __init__(self, persist_directory: str = None, embedder: TextEmbedder = None) -> None:
        self.persist_directory = persist_directory or os.path.join(
            settings.CHROMA_DB_PATH, "user_memory"
        )
        self.embedder = embedder or TextEmbedder(model_name=settings.EMBEDDING_MODEL_NAME)
        self.client = None
        self.collection = None

        if chromadb is not None:
            try:
                os.makedirs(self.persist_directory, exist_ok=True)
                self.client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
                self.collection = self.client.get_or_create_collection(
                    name=self.COLLECTION_NAME,
                    metadata={"description": "All user conversation data for memory"},
                )
            except Exception as e:
                print(f"[UserMemory] Failed to initialize ChromaDB: {e}")
                self.client = None
                self.collection = None

    def save_conversation_turn(
        self,
        user_message: str,
        bot_response: str,
        emotion: str = None,
        user_id: str = "default",
        conversation_id: str = None,
    ) -> bool:
        """
        Save a complete conversation turn (user message + bot response) to memory.
        This stores EVERYTHING the user shares for future reference.
        """
        if not self.collection or not user_message.strip():
            return False

        try:
            timestamp = datetime.now()
            
            # Create a combined document that captures the full exchange
            combined_text = f"User said: {user_message}"
            if bot_response:
                combined_text += f"\nAssistant responded: {bot_response}"
            
            # Create embedding for the user message (what we'll search against)
            embedding = self.embedder.embed_queries([user_message])[0]
            
            # Generate unique ID
            turn_id = f"{user_id}_{timestamp.strftime('%Y%m%d%H%M%S%f')}"
            
            # Store in ChromaDB with rich metadata
            self.collection.add(
                ids=[turn_id],
                embeddings=[embedding],
                documents=[combined_text],
                metadatas=[{
                    "user_id": user_id,
                    "conversation_id": conversation_id or "unknown",
                    "user_message": user_message[:1000],  # Truncate for metadata
                    "bot_response": (bot_response or "")[:1000],
                    "emotion": emotion or "unknown",
                    "created_at": timestamp.isoformat(),
                    "type": "conversation_turn",
                }],
            )
            
            # Also extract and save specific facts for quick retrieval
            facts = self.extract_facts(user_message)
            for fact in facts:
                self._save_fact_internal(fact, user_id, timestamp)
            
            return True
        except Exception as e:
            print(f"[UserMemory] Failed to save conversation turn: {e}")
            return False

    def _save_fact_internal(self, fact: str, user_id: str, timestamp: datetime) -> bool:
        """Internal method to save extracted facts."""
        if not self.collection or not fact.strip():
            return False
        try:
            embedding = self.embedder.embed_queries([fact])[0]
            fact_id = f"{user_id}_fact_{timestamp.strftime('%Y%m%d%H%M%S%f')}_{hash(fact) % 10000}"
            self.collection.add(
                ids=[fact_id],
                embeddings=[embedding],
                documents=[fact],
                metadatas=[{
                    "user_id": user_id,
                    "created_at": timestamp.isoformat(),
                    "type": "extracted_fact",
                }],
            )
            return True
        except Exception as e:
            print(f"[UserMemory] Failed to save fact: {e}")
            return False

    def extract_facts(self, user_message: str) -> List[str]:
        """
        Extract specific facts/data from user message for quick retrieval.
        Returns a list of fact strings to store.
        """
        facts = []
        
        # Patterns to detect user sharing personal info
        patterns = [
            # Name patterns
            (r"(?:my name is|i'm|i am|call me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", "User's name is {0}"),
            # Location patterns
            (r"(?:i live in|i'm from|i am from|i'm in|i am in|i stay in)\s+([A-Za-z\s,]+?)(?:\.|,|$)", "User lives in {0}"),
            # Work/Job patterns
            (r"(?:i work as|i am a|i'm a|my job is|i work at|my profession is|i'm working as)\s+([A-Za-z\s]+?)(?:\.|,|$)", "User works as {0}"),
            # Age patterns
            (r"(?:i am|i'm)\s+(\d{1,3})\s*(?:years old|yr old|yrs old)", "User is {0} years old"),
            # Hobby/Interest patterns
            (r"(?:i (?:love|like|enjoy|prefer|hate|dislike))\s+([A-Za-z\s]+?)(?:\.|,|$|!)", "User {verb} {0}"),
            # Family patterns
            (r"(?:my (?:wife|husband|son|daughter|mother|father|brother|sister|mom|dad|child|kid|partner|girlfriend|boyfriend)(?:'s)?\s+(?:name is|is named|called)?)\s*([A-Z][a-z]+)?", "User has family member"),
            # Education patterns
            (r"(?:i (?:studied|study|am studying|graduated from|attend|attended))\s+([A-Za-z\s]+?)(?:\.|,|$)", "User studied {0}"),
            # Skills patterns
            (r"(?:i (?:know|can|am good at|specialize in|expert in))\s+([A-Za-z\s]+?)(?:\.|,|$)", "User knows {0}"),
        ]

        for pattern, template in patterns:
            matches = re.findall(pattern, user_message, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    parts = [m.strip() for m in match if m and m.strip()]
                    if parts:
                        fact = template.format(*parts) if len(parts) > 1 else template.format(parts[0], "")
                        fact = fact.replace("  ", " ").replace("{verb}", "").strip()
                        if fact and len(fact) > 5:
                            facts.append(fact)
                else:
                    fact = template.format(match.strip())
                    if fact and len(fact) > 5:
                        facts.append(fact)

        return list(set(facts))

    def retrieve_relevant_memories(
        self, query: str, user_id: str = "default", n_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve ALL relevant memories (conversations + facts) for a query."""
        if not self.collection:
            return []

        try:
            query_embedding = self.embedder.embed_queries([query])[0]
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where={"user_id": user_id},
            )

            memories = []
            if results and results.get("documents"):
                docs = results["documents"][0] if results["documents"] else []
                metas = results["metadatas"][0] if results.get("metadatas") else []
                distances = results["distances"][0] if results.get("distances") else []

                for i, doc in enumerate(docs):
                    meta = metas[i] if i < len(metas) else {}
                    memory = {
                        "content": doc,
                        "type": meta.get("type", "unknown"),
                        "user_message": meta.get("user_message", ""),
                        "bot_response": meta.get("bot_response", ""),
                        "emotion": meta.get("emotion", ""),
                        "created_at": meta.get("created_at", ""),
                        "relevance": 1 - (distances[i] if i < len(distances) else 0),
                    }
                    memories.append(memory)

            # Sort by relevance
            memories.sort(key=lambda x: x["relevance"], reverse=True)
            return memories
        except TypeError as e:
            # Handle ChromaDB corruption: 'int' has no len()
            print(f"[UserMemory] Database corruption detected: {e}")
            return []
        except Exception as e:
            print(f"[UserMemory] Failed to retrieve memories: {e}")
            return []

    def get_all_memories(self, user_id: str = "default", limit: int = 100) -> List[str]:
        """Get all stored memories for a user (for display in UI)."""
        if not self.collection:
            return []

        try:
            results = self.collection.get(
                where={"user_id": user_id},
                include=["documents", "metadatas"],
                limit=limit,
            )
            
            memories = []
            docs = results.get("documents", [])
            metas = results.get("metadatas", [])
            
            for i, doc in enumerate(docs):
                meta = metas[i] if i < len(metas) else {}
                # Format based on type
                if meta.get("type") == "extracted_fact":
                    memories.append(f"📌 {doc}")
                elif meta.get("type") == "conversation_turn":
                    user_msg = meta.get("user_message", "")[:50]
                    if user_msg:
                        memories.append(f"💬 {user_msg}...")
                else:
                    memories.append(doc[:60] + "..." if len(doc) > 60 else doc)
            
            return memories
        except TypeError as e:
            print(f"[UserMemory] Database corruption in get_all_memories: {e}")
            return []
        except Exception as e:
            print(f"[UserMemory] Failed to get all memories: {e}")
            return []

    def get_memory_stats(self, user_id: str = "default") -> Dict[str, int]:
        """Get statistics about stored memories."""
        if not self.collection:
            return {"total": 0, "conversations": 0, "facts": 0}
        
        try:
            results = self.collection.get(
                where={"user_id": user_id},
                include=["metadatas"],
            )
            
            metas = results.get("metadatas", [])
            conversations = sum(1 for m in metas if m.get("type") == "conversation_turn")
            facts = sum(1 for m in metas if m.get("type") == "extracted_fact")
            
            return {
                "total": len(metas),
                "conversations": conversations,
                "facts": facts,
            }
        except TypeError as e:
            print(f"[UserMemory] Database corruption in get_memory_stats: {e}")
            return {"total": 0, "conversations": 0, "facts": 0}
        except Exception:
            return {"total": 0, "conversations": 0, "facts": 0}

    def clear_memories(self, user_id: str = "default") -> bool:
        """Clear all memories for a user."""
        if not self.collection:
            return False

        try:
            results = self.collection.get(where={"user_id": user_id})
            if results and results.get("ids"):
                self.collection.delete(ids=results["ids"])
            return True
        except TypeError as e:
            print(f"[UserMemory] Database corruption in clear_memories: {e}")
            return False
        except Exception as e:
            print(f"[UserMemory] Failed to clear memories: {e}")
            return False

    def format_memories_for_prompt(self, memories: List[Dict[str, Any]], max_items: int = 8) -> str:
        """Format retrieved memories into a string for the LLM prompt."""
        if not memories:
            return ""

        lines = ["Here's what I remember from our previous conversations:"]
        
        # Separate facts and conversations
        facts = [m for m in memories if m.get("type") == "extracted_fact"]
        convs = [m for m in memories if m.get("type") == "conversation_turn"]
        
        # Add top facts first
        for mem in facts[:4]:
            lines.append(f"- {mem['content']}")
        
        # Add relevant past conversations
        for mem in convs[:max_items - len(facts[:4])]:
            user_msg = mem.get("user_message", "")
            bot_resp = mem.get("bot_response", "")
            if user_msg:
                lines.append(f"- Previously you said: \"{user_msg[:100]}{'...' if len(user_msg) > 100 else ''}\"")
                if bot_resp:
                    lines.append(f"  I responded: \"{bot_resp[:80]}{'...' if len(bot_resp) > 80 else ''}\"")
        
        return "\n".join(lines)

    # Legacy methods for backward compatibility
    def save_fact(self, fact: str, user_id: str = "default") -> bool:
        """Save a single fact to the memory database."""
        return self._save_fact_internal(fact, user_id, datetime.now())

    def save_facts(self, facts: List[str], user_id: str = "default") -> int:
        """Save multiple facts. Returns count of successfully saved facts."""
        saved = 0
        for fact in facts:
            if self.save_fact(fact, user_id):
                saved += 1
        return saved
