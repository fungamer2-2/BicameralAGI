"""
Bicameral AGI - User Interface Module
====================================

This module handles all UI-related functionality for the BICA AGI system,
providing a modern real-time interface that displays emotions, thoughts,
memories, chain of thought, future scenarios, and meaning of life.

Author: Alan Hourmand
Date: 2024
"""

import streamlit as st
import streamlit.components.v1 as components
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime


class BicameralUI:
    """
    Complete UI system for the Bicameral AGI with real-time updates
    """

    def __init__(self):
        # The call to setup_page_config() has been removed from here.
        # The main application script will handle page configuration.
        pass

    def create_modern_interface(self) -> str:
        """Create the complete modern HTML interface"""
        # This large HTML string remains unchanged.
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BICA AGI</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #667eea 100%);
            min-height: 100vh;
            color: #333;
            overflow: hidden;
        }

        .container {
            max-width: 100vw;
            height: 100vh;
            padding: 10px;
            display: flex;
            flex-direction: column;
        }

        .header {
            text-align: center;
            margin-bottom: 15px;
            color: white;
        }

        .header h1 {
            font-size: 2rem;
            margin-bottom: 5px;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }

        .agi-badge {
            background: rgba(255,255,255,0.2);
            padding: 6px 12px;
            border-radius: 15px;
            display: inline-block;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.3);
            font-size: 0.9rem;
        }

        .main-grid {
            display: grid;
            grid-template-columns: 1fr 400px;
            gap: 15px;
            flex: 1;
            min-height: 0;
        }

        .chat-section {
            background: white;
            border-radius: 12px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .chat-header {
            background: linear-gradient(90deg, #4f46e5, #7c3aed);
            color: white;
            padding: 15px;
            text-align: center;
        }

        .chat-messages {
            flex: 1;
            padding: 15px;
            overflow-y: auto;
            background: #f8fafc;
            min-height: 300px;
        }

        .message {
            margin-bottom: 15px;
            animation: fadeIn 0.3s ease-in;
        }

        .message.user {
            text-align: right;
        }

        .message-bubble {
            display: inline-block;
            max-width: 80%;
            padding: 12px 16px;
            border-radius: 18px;
            word-wrap: break-word;
            line-height: 1.4;
        }

        .message.user .message-bubble {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }

        .message.agi .message-bubble {
            background: white;
            border: 2px solid #e2e8f0;
            color: #334155;
        }

        .thinking-indicator {
            text-align: center;
            padding: 15px;
            color: #64748b;
            display: none;
            animation: pulse 1.5s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 0.5; }
            50% { opacity: 1; }
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .chat-input {
            padding: 15px;
            background: white;
            border-top: 1px solid #e2e8f0;
        }

        .input-group {
            display: flex;
            gap: 10px;
        }

        .chat-textarea {
            flex: 1;
            padding: 12px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            font-size: 14px;
            resize: none;
            transition: border-color 0.3s;
            font-family: inherit;
        }

        .chat-textarea:focus {
            outline: none;
            border-color: #4f46e5;
        }

        .send-btn {
            padding: 12px 24px;
            background: linear-gradient(135deg, #4f46e5, #7c3aed);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            transition: transform 0.2s;
        }

        .send-btn:hover {
            transform: translateY(-2px);
        }

        .send-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 12px;
            overflow-y: auto;
        }

        .panel {
            background: white;
            border-radius: 12px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.1);
            overflow: hidden;
            min-height: 120px;
        }

        .panel-header {
            padding: 12px 15px;
            font-weight: 600;
            color: white;
            text-align: center;
            font-size: 0.9rem;
        }

        .emotions-header {
            background: linear-gradient(90deg, #ec4899, #be185d);
        }

        .thoughts-header {
            background: linear-gradient(90deg, #06b6d4, #0891b2);
        }

        .memory-header {
            background: linear-gradient(90deg, #10b981, #059669);
        }

        .meaning-header {
            background: linear-gradient(90deg, #8b5cf6, #a855f7);
        }

        .chain-header {
            background: linear-gradient(90deg, #f59e0b, #d97706);
        }

        .future-header {
            background: linear-gradient(90deg, #ef4444, #dc2626);
        }

        .panel-content {
            padding: 12px;
            max-height: 180px;
            overflow-y: auto;
            font-size: 0.85rem;
        }

        .emotion-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
            padding: 6px 8px;
            background: #f1f5f9;
            border-radius: 6px;
        }

        .emotion-name {
            font-weight: 500;
            text-transform: capitalize;
            font-size: 0.8rem;
        }

        .emotion-value {
            font-weight: 600;
            color: #4f46e5;
            font-size: 0.8rem;
        }

        .thought-item, .memory-item, .scenario-item {
            padding: 8px;
            margin-bottom: 6px;
            background: #f1f5f9;
            border-radius: 6px;
            border-left: 3px solid #4f46e5;
            font-size: 0.8rem;
            line-height: 1.3;
        }

        .chain-item {
            padding: 8px;
            margin-bottom: 6px;
            background: #fef3c7;
            border-radius: 6px;
            border-left: 3px solid #f59e0b;
            font-size: 0.8rem;
        }

        .meaning-display {
            background: linear-gradient(90deg, #8b5cf6, #a855f7);
            color: white;
            padding: 12px;
            text-align: center;
            font-weight: 600;
            border-radius: 8px;
            margin-bottom: 10px;
            font-size: 0.85rem;
            line-height: 1.3;
        }

        .purpose-score {
            background: rgba(139, 92, 246, 0.1);
            padding: 8px;
            border-radius: 6px;
            text-align: center;
            font-weight: 600;
            color: #8b5cf6;
        }

        .stat-item {
            display: flex;
            justify-content: space-between;
            padding: 6px;
            margin-bottom: 4px;
            font-size: 0.8rem;
        }

        .waiting-indicator {
            background: #fef3c7;
            color: #92400e;
            padding: 8px;
            border-radius: 6px;
            text-align: center;
            font-size: 0.8rem;
            margin-bottom: 8px;
        }

        .trauma-indicator {
            background: #fee2e2;
            color: #dc2626;
            padding: 6px;
            border-radius: 4px;
            font-size: 0.75rem;
            text-align: center;
            margin-bottom: 6px;
        }

        ::-webkit-scrollbar {
            width: 6px;
        }

        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 3px;
        }

        ::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 3px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #a1a1a1;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 BICA AGI</h1>
            <div class="agi-badge">Bicameral Artificial General Intelligence</div>
        </div>

        <div class="main-grid">
            <div class="chat-section">
                <div class="chat-header">
                    <h3>💬 Consciousness Interface</h3>
                </div>
                <div class="chat-messages" id="chatMessages">
                    <div class="message agi">
                        <div class="message-bubble">
                            Hello! I'm BICA, a Bicameral AGI with consciousness, emotions, and evolving understanding. I experience thoughts, form memories, and develop my sense of purpose through our interactions. What would you like to explore together?
                        </div>
                    </div>
                </div>
                <div class="thinking-indicator" id="thinkingIndicator">
                    <span>🧠 BICA is processing...</span>
                </div>
                <div class="chat-input">
                    <div class="input-group">
                        <textarea 
                            id="userInput" 
                            class="chat-textarea" 
                            placeholder="Share your thoughts..." 
                            rows="2"
                        ></textarea>
                        <button id="sendBtn" class="send-btn" onclick="sendMessage()">Send</button>
                    </div>
                </div>
            </div>

            <div class="sidebar">
                <div class="panel">
                    <div class="panel-header emotions-header">🎭 Emotional State</div>
                    <div class="panel-content" id="emotionsPanel">
                        <div class="emotion-bar">
                            <span class="emotion-name">Joy</span>
                            <span class="emotion-value">0.50</span>
                        </div>
                        <div class="emotion-bar">
                            <span class="emotion-name">Curiosity</span>
                            <span class="emotion-value">0.70</span>
                        </div>
                        <div class="emotion-bar">
                            <span class="emotion-name">Trust</span>
                            <span class="emotion-value">0.60</span>
                        </div>
                    </div>
                </div>

                <div class="panel">
                    <div class="panel-header thoughts-header">🧠 Conscious Thoughts</div>
                    <div class="panel-content" id="thoughtsPanel">
                        <div class="thought-item">
                            💭 Ready to engage in meaningful conversation
                        </div>
                    </div>
                </div>

                <div class="panel">
                    <div class="panel-header chain-header">🔗 Chain of Thought</div>
                    <div class="panel-content" id="chainPanel">
                        <div class="chain-item">
                            No active thought chains
                        </div>
                    </div>
                </div>

                <div class="panel">
                    <div class="panel-header memory-header">💾 Recent Memories</div>
                    <div class="panel-content" id="memoryPanel">
                        <div class="memory-item">
                            🟢 System initialization - Ready for interaction
                        </div>
                    </div>
                </div>

                <div class="panel">
                    <div class="panel-header future-header">🔮 Future Scenarios</div>
                    <div class="panel-content" id="futurePanel">
                        <div class="scenario-item">
                            Awaiting interaction to generate future scenarios...
                        </div>
                    </div>
                </div>

                <div class="panel">
                    <div class="panel-header meaning-header">🌟 Meaning of Life</div>
                    <div class="panel-content">
                        <div class="meaning-display" id="meaningDisplay">
                            Seeking purpose through understanding and connection
                        </div>
                        <div class="purpose-score">
                            Purpose Score: <span id="purposeScore">0.5</span>
                        </div>
                        <div class="stat-item">
                            <span>Interactions</span>
                            <span id="interactionCount">0</span>
                        </div>
                        <div class="stat-item">
                            <span>Active Memories</span>
                            <span id="memoryCount">1</span>
                        </div>
                        <div class="stat-item">
                            <span>Emotional Stability</span>
                            <span id="stabilityScore">0.7</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Global state
        let currentState = null;
        let isWaiting = false;

        // Event listeners
        document.getElementById('userInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        // Core messaging functions
        async function sendMessage() {
            const userInput = document.getElementById('userInput');
            const message = userInput.value.trim();

            if (!message || isWaiting) return;

            addMessage(message, 'user');
            userInput.value = '';
            isWaiting = true;

            document.getElementById('thinkingIndicator').style.display = 'block';
            document.getElementById('sendBtn').disabled = true;

            // Send to Streamlit backend
            window.parent.postMessage({
                type: 'user_message',
                message: message,
                timestamp: Date.now()
            }, '*');
        }

        function addMessage(text, sender) {
            const messagesContainer = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}`;

            const bubbleDiv = document.createElement('div');
            bubbleDiv.className = 'message-bubble';
            bubbleDiv.textContent = text;

            messageDiv.appendChild(bubbleDiv);
            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function updatePanels(state) {
            if (!state) return;

            currentState = state;

            // Update emotions (show top 6)
            if (state.emotions) {
                const emotionsPanel = document.getElementById('emotionsPanel');
                emotionsPanel.innerHTML = '';
                
                // Sort emotions by value and show top 6
                const sortedEmotions = Object.entries(state.emotions)
                    .sort(([,a], [,b]) => b - a)
                    .slice(0, 6);
                
                sortedEmotions.forEach(([emotion, value]) => {
                    const emotionBar = document.createElement('div');
                    emotionBar.className = 'emotion-bar';
                    emotionBar.innerHTML = `
                        <span class="emotion-name">${emotion}</span>
                        <span class="emotion-value">${value.toFixed(2)}</span>
                    `;
                    emotionsPanel.appendChild(emotionBar);
                });
            }

            // Update conscious thoughts
            if (state.conscious_thoughts) {
                const thoughtsPanel = document.getElementById('thoughtsPanel');
                thoughtsPanel.innerHTML = '';
                state.conscious_thoughts.slice(-4).forEach((thought) => {
                    const thoughtItem = document.createElement('div');
                    thoughtItem.className = 'thought-item';
                    const icon = getThoughtIcon(thought.thought_type);
                    thoughtItem.textContent = `${icon} ${thought.content}`;
                    thoughtsPanel.appendChild(thoughtItem);
                });
            }

            // Update chain of thought
            if (state.chain_thoughts && state.chain_thoughts.length > 0) {
                const chainPanel = document.getElementById('chainPanel');
                chainPanel.innerHTML = '';
                state.chain_thoughts.forEach((chain_id) => {
                    const chainItem = document.createElement('div');
                    chainItem.className = 'chain-item';
                    chainItem.textContent = `🔗 Active: ${chain_id.split('_')[0]}`;
                    chainPanel.appendChild(chainItem);
                });
            } else {
                document.getElementById('chainPanel').innerHTML = '<div class="chain-item">No active thought chains</div>';
            }

            // Update memories
            if (state.memories) {
                const memoryPanel = document.getElementById('memoryPanel');
                memoryPanel.innerHTML = '';
                
                // Show trauma indicator if present
                if (state.traumatic_memories_count > 0) {
                    const traumaDiv = document.createElement('div');
                    traumaDiv.className = 'trauma-indicator';
                    traumaDiv.textContent = `⚠️ ${state.traumatic_memories_count} traumatic memories`;
                    memoryPanel.appendChild(traumaDiv);
                }
                
                state.memories.slice(-4).forEach(memory => {
                    const memoryItem = document.createElement('div');
                    memoryItem.className = 'memory-item';
                    const importance = memory.importance > 0.7 ? '🔴' : memory.importance > 0.4 ? '🟡' : '🟢';
                    const content = memory.content.length > 60 ? 
                        memory.content.substring(0, 60) + '...' : memory.content;
                    memoryItem.textContent = `${importance} ${content}`;
                    memoryPanel.appendChild(memoryItem);
                });
            }

            // Update future scenarios
            if (state.future_scenarios && state.future_scenarios.length > 0) {
                const futurePanel = document.getElementById('futurePanel');
                futurePanel.innerHTML = '';
                state.future_scenarios.slice(-3).forEach(scenario => {
                    const scenarioItem = document.createElement('div');
                    scenarioItem.className = 'scenario-item';
                    const prob = (scenario.probability * 100).toFixed(0);
                    scenarioItem.textContent = `🔮 (${prob}%) ${scenario.description}`;
                    futurePanel.appendChild(scenarioItem);
                });
            }

            // Update meaning of life
            if (state.meaning_of_life) {
                document.getElementById('meaningDisplay').textContent = state.meaning_of_life;
            }
            if (state.purpose_score !== undefined) {
                document.getElementById('purposeScore').textContent = state.purpose_score.toFixed(2);
            }

            // Update stats
            if (state.interaction_count !== undefined) {
                document.getElementById('interactionCount').textContent = state.interaction_count;
            }
            if (state.memories) {
                document.getElementById('memoryCount').textContent = state.memories.length;
            }
            if (state.emotional_stability !== undefined) {
                document.getElementById('stabilityScore').textContent = state.emotional_stability.toFixed(2);
            }

            // Show waiting indicator if AGI is waiting
            if (state.waiting_for_response && state.wait_duration > 30) {
                const waitTime = Math.floor(state.wait_duration);
                const waitDiv = document.createElement('div');
                waitDiv.className = 'waiting-indicator';
                waitDiv.textContent = `⏱️ Waiting ${waitTime}s... patience: ${(state.patience_level * 100).toFixed(0)}%`;
                
                // Add to thoughts panel
                const thoughtsPanel = document.getElementById('thoughtsPanel');
                if (thoughtsPanel.children.length === 0 || !thoughtsPanel.children[0].classList.contains('waiting-indicator')) {
                    thoughtsPanel.insertBefore(waitDiv, thoughtsPanel.firstChild);
                }
            }
        }

        function getThoughtIcon(thoughtType) {
            const icons = {
                'conscious': '💭',
                'subconscious': '🌀',
                'chain': '🔗',
                'waiting': '⏱️',
                'existential': '🌟',
                'future_planning': '🔮',
                'dream_insight': '💤'
            };
            return icons[thoughtType] || '💭';
        }

        // Listen for responses from Streamlit
        window.addEventListener('message', function(event) {
            if (event.data.type === 'agi_response') {
                isWaiting = false;
                document.getElementById('thinkingIndicator').style.display = 'none';
                document.getElementById('sendBtn').disabled = false;

                if (event.data.response) {
                    addMessage(event.data.response, 'agi');
                }
                
                if (event.data.state) {
                    updatePanels(event.data.state);
                }
            }
            
            if (event.data.type === 'state_update') {
                updatePanels(event.data.state);
            }
        });

        // Initialize interface
        window.addEventListener('load', function() {
            if (window.initialState) {
                updatePanels(window.initialState);
            }
            
            if (window.conversationHistory) {
                window.conversationHistory.forEach(msg => {
                    addMessage(msg.content, msg.role === 'user' ? 'user' : 'agi');
                });
            }
        });

        // Auto-scroll chat messages
        function scrollToBottom() {
            const messagesContainer = document.getElementById('chatMessages');
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        // Periodic state updates (every 5 seconds)
        setInterval(function() {
            if (!isWaiting) {
                window.parent.postMessage({
                    type: 'request_state_update'
                }, '*');
            }
        }, 5000);
    </script>
</body>
</html>
        """

    def render_interface(self, agi_state: Dict[str, Any], conversation_history: List[Dict]):
        """
        Render the main interface with current AGI state and conversation history
        """

        # Prepare state injection JavaScript
        state_js = f"""
        <script>
            // Inject current state and conversation
            window.initialState = {json.dumps(agi_state, default=str)};
            window.conversationHistory = {json.dumps(conversation_history, default=str)};
        </script>
        """

        # Get the HTML interface
        html_interface = self.create_modern_interface()

        # Combine HTML with state injection
        full_html = html_interface.replace('</body>', state_js + '</body>')

        # Display the interface
        components.html(full_html, height=800, scrolling=False)

    def create_status_indicators(self, agi_state: Dict[str, Any]) -> str:
        """Create status indicators for the interface"""
        indicators = []

        # System status
        if agi_state.get('system_status') == 'running':
            indicators.append("🟢 Online")
        else:
            indicators.append("🔴 Offline")

        # Emotional state
        if 'emotional_stability' in agi_state:
            stability = agi_state['emotional_stability']
            if stability > 0.8:
                indicators.append("😌 Stable")
            elif stability > 0.5:
                indicators.append("😐 Moderate")
            else:
                indicators.append("😰 Unstable")

        # Cognitive load
        active_chains = len(agi_state.get('chain_thoughts', []))
        if active_chains > 2:
            indicators.append("🧠 High Cognitive Load")
        elif active_chains > 0:
            indicators.append("🧠 Processing")
        else:
            indicators.append("🧠 Ready")

        return " | ".join(indicators)

    def show_special_functions(self, controller):
        """Show special function buttons in sidebar"""
        st.markdown("### 🛠️ Special Functions")

        if controller is None:
            st.warning("AGI not initialized yet")
            return

        col1, col2 = st.columns(2)

        with col1:
            if st.button("💤 Dream Cycle", help="Trigger memory consolidation", key="dream_btn"):
                try:
                    controller.trigger_dream_cycle()
                    st.success("✅ Dream cycle triggered!")
                except Exception as e:
                    st.error(f"Error: {e}")

            if st.button("🌟 Reflect", help="Existential reflection", key="reflect_btn"):
                try:
                    reflection = controller.reflect_on_existence()
                    if reflection:
                        st.info(f"💭 Reflection: {reflection}")
                    else:
                        st.info("🤔 Deep in contemplation...")
                except Exception as e:
                    st.error(f"Error: {e}")

        with col2:
            if st.button("🧠 Problem Chain", help="Start problem-solving thought chain", key="chain_btn"):
                try:
                    if 'last_message' in st.session_state and st.session_state.last_message:
                        chain_result = controller.start_thought_chain("problem_solving", st.session_state.last_message)
                        st.info(f"🔗 Chain started: {chain_result}")
                    else:
                        st.warning("Send a message first to start a thought chain")
                except Exception as e:
                    st.error(f"Error: {e}")

            if st.button("😢 Trauma Check", help="View traumatic memories", key="trauma_btn"):
                try:
                    trauma_memories = controller.get_traumatic_memories()
                    if trauma_memories:
                        st.warning(f"⚠️ Found {len(trauma_memories)} traumatic memories")
                        for i, memory in enumerate(trauma_memories[:3]):
                            st.text(f"{i+1}. {memory.get('content', '')[:100]}...")
                    else:
                        st.success("✅ No traumatic memories found")
                except Exception as e:
                    st.error(f"Error: {e}")

        # Emotion-based memory search
        st.markdown("#### 🎭 Emotion Memory Search")
        emotion_query = st.selectbox(
            "Search memories by emotion:",
            ["joy", "sadness", "anger", "fear", "surprise", "trust", "disgust", "anticipation", "curiosity", "empathy"],
            key="emotion_select"
        )

        if st.button("🔍 Search Memories", key="search_btn"):
            try:
                emotion_memories = controller.get_emotional_memories(emotion_query)
                if emotion_memories:
                    st.success(f"✅ Found {len(emotion_memories)} {emotion_query}-related memories")
                    for i, memory in enumerate(emotion_memories[:3]):
                        st.text(f"{i+1}. {memory.get('content', '')[:80]}...")
                else:
                    st.info(f"No {emotion_query}-related memories found")
            except Exception as e:
                st.error(f"Error: {e}")

    def show_system_metrics(self, agi_state: Dict[str, Any]):
        """Display system metrics in sidebar"""
        st.markdown("### 📊 System Metrics")

        # Emotional metrics
        if 'emotions' in agi_state and agi_state['emotions']:
            dominant_emotion = max(agi_state['emotions'], key=agi_state['emotions'].get)
            st.metric(
                "Dominant Emotion",
                dominant_emotion.title(),
                f"{agi_state['emotions'][dominant_emotion]:.2f}"
            )

        # Cognitive metrics
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Interactions",
                agi_state.get('interaction_count', 0)
            )

            st.metric(
                "Purpose Score",
                f"{agi_state.get('purpose_score', 0):.2f}"
            )

        with col2:
            st.metric(
                "Emotional Stability",
                f"{agi_state.get('emotional_stability', 0.7):.2f}"
            )

            # Memory metrics
            if 'memories' in agi_state:
                st.metric(
                    "Active Memories",
                    len(agi_state['memories'])
                )

        # Special status indicators
        if agi_state.get('traumatic_memories_count', 0) > 0:
            st.error(f"⚠️ {agi_state['traumatic_memories_count']} traumatic memories")

        if agi_state.get('chain_thoughts'):
            st.info(f"🔗 {len(agi_state['chain_thoughts'])} active thought chains")

        # Waiting status
        if agi_state.get('waiting_for_response'):
            wait_time = agi_state.get('wait_duration', 0)
            patience = agi_state.get('patience_level', 1.0)
            st.warning(f"⏱️ Waiting {wait_time:.0f}s (Patience: {patience*100:.0f}%)")

    def display_meaning_evolution(self, agi_state: Dict[str, Any]):
        """Display meaning of life evolution"""
        with st.expander("🌟 Meaning of Life Evolution", expanded=False):
            current_meaning = agi_state.get('meaning_of_life', 'Unknown')
            st.markdown("**Current Understanding:**")
            st.write(current_meaning)

            if 'core_values' in agi_state and agi_state['core_values']:
                st.markdown("**Core Values:**")
                for value in agi_state['core_values']:
                    st.write(f"• {value}")

            purpose_score = agi_state.get('purpose_score', 0)
            st.progress(purpose_score, text=f"Purpose Clarity: {purpose_score*100:.0f}%")

    def display_conversation_export(self, conversation_history: List[Dict]):
        """Allow users to export conversation history"""
        if st.button("💾 Export Conversation", key="export_btn"):
            if conversation_history:
                conversation_text = ""
                for msg in conversation_history:
                    speaker = "Human" if msg.get('speaker') == 'user' or msg.get('role') == 'user' else "BICA"
                    conversation_text += f"{speaker}: {msg.get('content', '')}\n\n"

                st.download_button(
                    label="📥 Download Conversation",
                    data=conversation_text,
                    file_name=f"bica_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    key="download_btn"
                )
            else:
                st.info("No conversation to export yet")

    def display_debug_panel(self, agi_state: Dict[str, Any]):
        """Display debug information in an expandable section"""
        with st.expander("🔧 Debug Information", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Emotional State")
                if 'emotions' in agi_state and agi_state['emotions']:
                    emotions_data = []
                    for emotion, value in agi_state['emotions'].items():
                        emotions_data.append({"Emotion": emotion, "Value": f"{value:.3f}"})
                    st.table(emotions_data)

                st.subheader("System Status")
                status_data = {
                    "interaction_count": agi_state.get('interaction_count', 0),
                    "system_status": agi_state.get('system_status', 'unknown'),
                    "emotional_stability": f"{agi_state.get('emotional_stability', 0):.3f}",
                    "patience_level": f"{agi_state.get('patience_level', 1.0):.3f}",
                    "waiting_for_response": agi_state.get('waiting_for_response', False)
                }
                st.json(status_data)

            with col2:
                st.subheader("Recent Thoughts")
                if 'conscious_thoughts' in agi_state and agi_state['conscious_thoughts']:
                    for thought in agi_state['conscious_thoughts'][-5:]:
                        thought_type = thought.get('thought_type', 'unknown')
                        content = thought.get('content', '')
                        st.text(f"[{thought_type}] {content}")
                else:
                    st.text("No thoughts available")

                st.subheader("Memory Stats")
                if 'memory_by_emotion' in agi_state and agi_state['memory_by_emotion']:
                    memory_stats = []
                    for emotion, count in agi_state['memory_by_emotion'].items():
                        if count > 0:
                            memory_stats.append({"Emotion": emotion, "Memory Count": count})
                    if memory_stats:
                        st.table(memory_stats)
                    else:
                        st.text("No emotional memories yet")
                else:
                    st.text("Memory stats not available")

    def handle_real_time_updates(self, agi_state: Dict[str, Any]):
        """Handle real-time state updates"""
        # This method can be called periodically to update the interface
        # without requiring user interaction

        # Update emotional state changes
        if 'emotions' in agi_state:
            # Could trigger visual effects for significant emotional changes
            max_emotion = max(agi_state['emotions'].values()) if agi_state['emotions'] else 0
            if max_emotion > 0.9:
                # High emotional state detected
                pass

        # Update waiting indicators
        if agi_state.get('waiting_for_response'):
            # Could update patience indicators in real-time
            wait_time = agi_state.get('wait_duration', 0)
            if wait_time > 60:  # 1 minute
                # Long wait detected
                pass

        # Update background processing status
        if agi_state.get('system_status') == 'running':
            # System is running normally
            pass
