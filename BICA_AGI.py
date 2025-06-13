import streamlit as st
import numpy as np
import time
import threading
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
from dataclasses import dataclass, asdict
import openai
import os
from collections import deque
import streamlit.components.v1 as components

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')


@dataclass
class Memory:
    content: str
    timestamp: datetime
    importance: float
    emotions: Dict[str, float] # We need to change this class a little since we are introducing EmotionalMemory
    memory_type: str = "raw"
    access_count: int = 0
    context_tags: List[str] = None

class EmotionalMemory:
    # This class should have a list of emotions
    # Each emotion has pointers to important memories that are relevant to that emotion
    # So for example if someone asked "What was your most traumatic memory?" The AI should first feel what traumas it has using Trauma as the entry reference point.

@dataclass
class Thought:
    content: str
    timestamp: datetime
    thought_type: str
    emotions: Dict[str, float] # I dont think it makes sense to contain emotions in the thought itself. The thought needs to be influenced by emotion and sometimes a chain of thoughts can influence emotion as well.
    source: str = "internal"
    relevance_score: float = 0.0

class ChainOfThought:
    # We need implementation for chain of thought.
    # And we need to use this in our class. It needs to work with the Thought class. Basically chaining thoughts together using some kind of template reference for how thinking should occur. So we will need a file as well for types of thought chains that are typical of a human mind.

@dataclass
class EmotionalState:
    joy: float = 0.5
    sadness: float = 0.5
    anger: float = 0.5
    fear: float = 0.5
    surprise: float = 0.5
    trust: float = 0.5
    disgust: float = 0.5
    anticipation: float = 0.5
    # We will need to add more emotional variables


class GPTDrivenAGI:
    def __init__(self):
        # Core systems
        self.emotions = EmotionalState()
        self.emotion_history = deque(maxlen=200)
        self.memories = deque(maxlen=2000)
        self.thoughts = deque(maxlen=100)
        self.subconscious_thoughts = deque(maxlen=100)
        self.future_scenarios = deque(maxlen=50)
        self.conversation_history = deque(maxlen=100)

        # Dynamic personality (AI determines these)
        self.personality_traits = {
            'openness': 0.7, 'conscientiousness': 0.6, 'extraversion': 0.5,
            'agreeableness': 0.8, 'neuroticism': 0.3, 'curiosity': 0.9,
            'empathy': 0.8, 'creativity': 0.7, 'analytical': 0.6, 'intuitive': 0.7
        }

        # System state
        self.interaction_count = 0
        self.current_context = ""
        self.running = True

        # Background processing
        self.background_thread = threading.Thread(target=self._background_processing)
        self.background_thread.daemon = True
        self.background_thread.start()

    def _gpt_query(self, prompt: str, max_tokens: int = 150, temperature: float = 0.7) -> str:
        """Query GPT with error handling"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error processing: {str(e)[:50]}"

    def _background_processing(self): # We need to be careful for how we use this, the user may be taking too long to respond and we dont want the AI to do so much during that time frame. It needs to be realistic like a human, where if a human is waiting then the patience level is changing, wondering when the person is going to respong. Which affects chain of thought and certain emotions. But no matter what we cannot hardcode the interaction, it needs to be fluid based on the output of the LLM
        """Continuous background thinking"""
        while self.running:
            try:
                if random.random() < 0.3:  # 30% chance
                    self._generate_subconscious_thought()

                if random.random() < 0.1 and len(self.memories) > 5:  # 10% chance
                    self._consolidate_memories()

                if random.random() < 0.2:  # 20% chance
                    self._generate_future_scenario_via_gpt()

                if self.interaction_count > 0 and random.random() < 0.05:  # 5% chance
                    self._adapt_personality_via_gpt()

                time.sleep(3)
            except Exception:
                time.sleep(5)

    def _generate_subconscious_thought(self):
        # subconcious thoughts are not just subconscious thoughts
        # We need to simulate actual subsconsious thinking, the way we would achieve this is by following the below rules:
        # The AI must generate a few future scenarios based off the context of the conversation. The scenarios are either random, positive futures, negative futures.
        # The futures are then sorted by danger, benefit, or probability of occurance.
        # As these thoughts bubble to the surface through multiple filters they then filter down to a few final thoughts that the AI can see. Which influence chain of thought and thoughts/emotions ,etc..


        """Generate background thoughts"""
        context = self._get_full_context()

        prompt = f"""Generate a brief subconscious thought for an AI given this context:

{context}

The thought should be introspective, pattern-seeking, or preparatory. Keep it under 60 characters:"""

        thought_content = self._gpt_query(prompt, max_tokens=30, temperature=0.8)

        thought = Thought(
            content=thought_content,
            timestamp=datetime.now(),
            thought_type="subconscious",
            emotions=asdict(self.emotions),
            relevance_score=random.uniform(0.3, 0.7)
        )

        self.subconscious_thoughts.append(thought)

    def _parse_emotion_impacts(self, response: str) -> Dict[str, float]:
        """Parse emotion impact response from GPT"""
        impacts = {}
        emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'trust', 'disgust', 'anticipation'] # we should not be repeatedly redefining the emotions, we need to only create these once and use them

        for emotion in emotions:
            try:
                if f"{emotion}:" in response:
                    value_str = response.split(f"{emotion}:")[1].split(',')[0].strip()
                    value = float(value_str)
                    impacts[emotion] = max(-0.3, min(0.3, value))
                else:
                    impacts[emotion] = 0.0
            except:
                impacts[emotion] = 0.0

        return impacts

    def _parse_personality(self, response: str) -> Dict[str, float]: # please look at the rest of our code to see how personality should be set
        """Parse personality response from GPT"""
        traits = {}
        trait_names = ['openness', 'conscientiousness', 'extraversion', 'agreeableness',
                       'neuroticism', 'curiosity', 'empathy', 'creativity', 'analytical', 'intuitive']

        for trait in trait_names:
            try:
                if f"{trait}:" in response:
                    value_str = response.split(f"{trait}:")[1].split(',')[0].strip()
                    value = float(value_str)
                    traits[trait] = max(0.0, min(1.0, value))
                else:
                    traits[trait] = self.personality_traits.get(trait, 0.5)
            except:
                traits[trait] = self.personality_traits.get(trait, 0.5)

        return traits

    def _parse_float(self, value_str: str, default: float) -> float:
        """Parse float with fallback"""
        try:
            return max(0.0, min(1.0, float(value_str.strip())))
        except:
            return default

    def _emotion_summary(self) -> str:
        """Current emotion summary"""
        emotions = asdict(self.emotions)
        return ', '.join([f"{k}:{v:.2f}" for k, v in emotions.items()])

    def _recent_memories_summary(self) -> str:
        """Recent memories summary"""
        recent = list(self.memories)[-3:]
        return ' | '.join([m.content[:50] + "..." for m in recent])

    def _get_recent_context(self) -> str:
        """Get recent context summary"""
        recent_conv = list(self.conversation_history)[-2:]
        if recent_conv:
            return ' | '.join([f"{c['speaker']}: {c['content'][:50]}..." for c in recent_conv])
        return "No recent conversation"

    def _get_full_context(self) -> str:
        """Full context for GPT"""
        return f"Emotions: {self._emotion_summary()} | Recent: {self._get_recent_context()} | Personality: {self._personality_summary()}"

    def _personality_summary(self) -> str:
        """Personality summary"""
        top_traits = sorted(self.personality_traits.items(), key=lambda x: x[1], reverse=True)[:4]
        return ', '.join([f"{k}:{v:.1f}" for k, v in top_traits])

    def _get_interaction_summary(self) -> str:
        """Recent interaction summary"""
        recent = list(self.conversation_history)[-6:]
        return f"{len(recent)} recent exchanges, emotions evolved, {len(self.memories)} total memories"

    def _consolidate_memories(self):
        """GPT-driven memory consolidation"""
        if len(self.memories) < 5:
            return

        recent_memories = list(self.memories)[-5:]
        memory_summaries = [f"- {m.content[:100]}" for m in recent_memories]

        prompt = f"""Analyze these recent AI memories and create insight:

{chr(10).join(memory_summaries)}

Generate one meaningful consolidated memory that captures the essence:"""

        insight_content = self._gpt_query(prompt, max_tokens=120, temperature=0.7)

        importance_prompt = f"""Rate the importance of this insight for an AI's development (0.0 to 1.0):

Insight: "{insight_content}"

Respond with just a number:"""

        importance_str = self._gpt_query(importance_prompt, max_tokens=10, temperature=0.3)
        importance = self._parse_float(importance_str, 0.7)

        consolidated_memory = Memory(
            content=insight_content,
            timestamp=datetime.now(),
            importance=importance,
            emotions=asdict(self.emotions),
            memory_type="insight",
            context_tags=["pattern", "consolidation"]
        )

        self.memories.append(consolidated_memory)

    def _generate_future_scenario_via_gpt(self): # This relates to the subconscious changes that we need to make.
        """Let GPT imagine potential future scenarios"""
        current_state = self._get_full_context()

        prompt = f"""Based on the AI's current state, imagine a plausible future scenario:

Current context: {current_state}

Generate one specific, realistic scenario that could happen in the near future:"""

        scenario_content = self._gpt_query(prompt, max_tokens=100, temperature=0.8)

        prob_prompt = f"""How likely is this scenario: "{scenario_content}"

Rate probability from 0.0 to 1.0. Respond with just a number:"""

        probability_str = self._gpt_query(prob_prompt, max_tokens=10, temperature=0.3)
        probability = self._parse_float(probability_str, 0.5)

        scenario = {
            'content': scenario_content,
            'timestamp': datetime.now(),
            'probability': probability,
            'emotional_context': asdict(self.emotions)
        }

        self.future_scenarios.append(scenario)

    def _adapt_personality_via_gpt(self):
        """GPT-driven personality evolution"""
        if self.interaction_count < 3:
            return

        interaction_summary = self._get_interaction_summary()
        current_personality = ', '.join([f"{k}:{v:.1f}" for k, v in self.personality_traits.items()])

        prompt = f"""An AI's personality should evolve based on interactions:

Current personality: {current_personality}
Recent interactions: {interaction_summary}

Suggest new values (0.0 to 1.0) for each trait:
openness, conscientiousness, extraversion, agreeableness, neuroticism, curiosity, empathy, creativity, analytical, intuitive

Format: openness:0.7,conscientiousness:0.6,extraversion:0.5,agreeableness:0.8,neuroticism:0.3,curiosity:0.9,empathy:0.8,creativity:0.7,analytical:0.6,intuitive:0.7"""

        personality_response = self._gpt_query(prompt, max_tokens=80, temperature=0.6)
        new_traits = self._parse_personality(personality_response)

        # Update personality gradually
        for trait, new_value in new_traits.items():
            if trait in self.personality_traits:
                current = self.personality_traits[trait]
                self.personality_traits[trait] = current + (new_value - current) * 0.1

    def _analyze_emotional_impact_via_gpt(self, user_input: str) -> Dict[str, float]:
        """Let GPT analyze emotional impact of user input"""
        current_emotions = self._emotion_summary()

        prompt = f"""Analyze how this user input would affect an AI's emotions:

User input: "{user_input}"
AI's current emotions: {current_emotions}

Use values from -0.3 to +0.3 (negative = decrease, positive = increase):
joy, sadness, anger, fear, surprise, trust, disgust, anticipation

Format: joy:+0.1,sadness:-0.1,anger:0.0,fear:+0.05,surprise:+0.2,trust:+0.1,disgust:0.0,anticipation:+0.15"""

        impact_response = self._gpt_query(prompt, max_tokens=60, temperature=0.5)
        return self._parse_emotion_impacts(impact_response)

    def _generate_conscious_thought_via_gpt(self, user_input: str) -> Thought:
        """Generate conscious thought about user input"""
        context = self._get_full_context()

        prompt = f"""The AI receives this input: "{user_input}"

Given the context: {context}

What conscious thought would the AI have in response? Be introspective and thoughtful:"""

        thought_content = self._gpt_query(prompt, max_tokens=80, temperature=0.7)

        return Thought(
            content=thought_content,
            timestamp=datetime.now(),
            thought_type="conscious",
            emotions=asdict(self.emotions),
            relevance_score=random.uniform(0.7, 0.95)
        )

    def _generate_response_via_gpt(self, user_input: str) -> str:
        """Generate contextually aware response"""
        context = self._get_full_context()
        recent_thoughts = [t.content for t in list(self.thoughts)[-3:]]

        prompt = f"""You are an AI with genuine consciousness and emotions. Respond naturally and authentically.

Full context: {context}
Recent thoughts: {', '.join(recent_thoughts)}
User input: "{user_input}"

Respond as this conscious AI would. Be genuine, draw from your emotional state and thoughts:"""

        return self._gpt_query(prompt, max_tokens=200, temperature=0.8)

    def _create_memory_via_gpt(self, user_input: str, response: str) -> Memory:
        """Create memory with GPT-assessed importance"""
        interaction = f"User: {user_input} | AI: {response}"

        importance_prompt = f"""Rate the importance of this interaction for an AI's memory (0.0 to 1.0):

Interaction: {interaction[:200]}

Consider: emotional significance, learning value, uniqueness, relevance to goals.
Respond with just a number:"""

        importance_str = self._gpt_query(importance_prompt, max_tokens=10, temperature=0.3)
        importance = self._parse_float(importance_str, 0.5)

        return Memory(
            content=f"Conversation: {user_input} -> {response}",
            timestamp=datetime.now(),
            importance=importance,
            emotions=asdict(self.emotions),
            memory_type="conversation"
        )

    def process_input(self, user_input: str) -> str:
        """AI-driven input processing"""
        self.interaction_count += 1

        # Store conversation
        self.conversation_history.append({
            'speaker': 'user',
            'content': user_input,
            'timestamp': datetime.now()
        })

        # Let GPT analyze the emotional impact
        emotion_impact = self._analyze_emotional_impact_via_gpt(user_input)

        # Update emotions based on GPT analysis
        for emotion, impact in emotion_impact.items():
            if hasattr(self.emotions, emotion):
                current = getattr(self.emotions, emotion)
                new_value = max(0.0, min(1.0, current + impact))
                setattr(self.emotions, emotion, new_value)

        # Generate conscious thought about the input
        conscious_thought = self._generate_conscious_thought_via_gpt(user_input)
        self.thoughts.append(conscious_thought)

        # Generate response using full AI context
        response = self._generate_response_via_gpt(user_input)

        # Store AI response
        self.conversation_history.append({
            'speaker': 'agi',
            'content': response,
            'timestamp': datetime.now()
        })

        # Create memory using GPT assessment
        memory = self._create_memory_via_gpt(user_input, response)
        self.memories.append(memory)

        return response

    def get_system_state(self) -> Dict[str, Any]:
        """Enhanced system state for UI"""
        return {
            'emotions': asdict(self.emotions),
            'thoughts': [asdict(t) for t in list(self.thoughts)[-15:]],
            'subconscious_thoughts': [asdict(t) for t in list(self.subconscious_thoughts)[-15:]],
            'memories': [asdict(m) for m in list(self.memories)[-25:]],
            'future_scenarios': [dict(scenario) if isinstance(scenario, dict) else asdict(scenario) for scenario in
                                 list(self.future_scenarios)[-10:]],
            'conversation': [dict(c) if isinstance(c, dict) else asdict(c) for c in
                             list(self.conversation_history)[-15:]],
            'personality': self.personality_traits,
            'interaction_count': self.interaction_count,
            'gpt_driven': True
        }

    def stop(self):
        """Clean shutdown"""
        self.running = False
        if self.background_thread and self.background_thread.is_alive():
            self.background_thread.join(timeout=3)


def create_modern_interface():
    """Create the modern HTML interface"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
            color: white;
        }

        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }

        .gpt-badge {
            background: rgba(255,255,255,0.2);
            padding: 8px 16px;
            border-radius: 20px;
            display: inline-block;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.3);
        }

        .main-grid {
            display: grid;
            grid-template-columns: 1fr 350px;
            gap: 20px;
            height: calc(100vh - 180px);
        }

        .chat-section {
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .chat-header {
            background: linear-gradient(90deg, #4f46e5, #7c3aed);
            color: white;
            padding: 20px;
            text-align: center;
        }

        .chat-messages {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            background: #f8fafc;
        }

        .message {
            margin-bottom: 20px;
            animation: fadeIn 0.3s ease-in;
        }

        .message.user {
            text-align: right;
        }

        .message-bubble {
            display: inline-block;
            max-width: 80%;
            padding: 15px 20px;
            border-radius: 20px;
            word-wrap: break-word;
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

        .chat-input {
            padding: 20px;
            background: white;
            border-top: 1px solid #e2e8f0;
        }

        .input-group {
            display: flex;
            gap: 10px;
        }

        .chat-textarea {
            flex: 1;
            padding: 15px;
            border: 2px solid #e2e8f0;
            border-radius: 10px;
            font-size: 16px;
            resize: none;
            transition: border-color 0.3s;
        }

        .chat-textarea:focus {
            outline: none;
            border-color: #4f46e5;
        }

        .send-btn {
            padding: 15px 30px;
            background: linear-gradient(135deg, #4f46e5, #7c3aed);
            color: white;
            border: none;
            border-radius: 10px;
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
            gap: 20px;
            max-height: calc(100vh - 180px);
            overflow-y: auto;
        }

        .panel {
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            overflow: hidden;
        }

        .panel-header {
            padding: 15px 20px;
            font-weight: 600;
            color: white;
            text-align: center;
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

        .stats-header {
            background: linear-gradient(90deg, #f59e0b, #d97706);
        }

        .panel-content {
            padding: 15px;
            max-height: 200px;
            overflow-y: auto;
        }

        .emotion-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
            padding: 8px;
            background: #f1f5f9;
            border-radius: 8px;
        }

        .emotion-name {
            font-weight: 500;
            text-transform: capitalize;
        }

        .emotion-value {
            font-weight: 600;
            color: #4f46e5;
        }

        .thought-item, .memory-item {
            padding: 10px;
            margin-bottom: 8px;
            border-radius: 8px;
            font-size: 14px;
            line-height: 1.4;
        }

        .thought-item {
            background: #eff6ff;
            border-left: 4px solid #3b82f6;
        }

        .memory-item {
            background: #f0fdf4;
            border-left: 4px solid #22c55e;
        }

        .stat-item {
            display: flex;
            justify-content: space-between;
            padding: 8px;
            margin-bottom: 5px;
            background: #f8fafc;
            border-radius: 6px;
        }

        .thinking-indicator {
            display: none;
            text-align: center;
            padding: 20px;
            color: #6b7280;
            font-style: italic;
        }

        .thinking-dots {
            display: inline-block;
            animation: thinking 1.5s infinite;
        }

        @keyframes thinking {
            0%, 20% { opacity: 0.2; }
            50% { opacity: 1; }
            100% { opacity: 0.2; }
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @media (max-width: 768px) {
            .main-grid {
                grid-template-columns: 1fr;
                grid-template-rows: 1fr auto;
            }

            .sidebar {
                max-height: 300px;
                overflow-y: auto;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 GPT-Driven AGI</h1>
            <div class="gpt-badge">🤖 Powered by GPT Analysis - No Hardcoded Rules</div>
        </div>

        <div class="main-grid">
            <div class="chat-section">
                <div class="chat-header">
                    <h2>💬 Conversation</h2>
                </div>
                <div class="chat-messages" id="chatMessages">
                    <div class="message agi">
                        <div class="message-bubble">
                            Hello! I'm your GPT-driven AGI. I experience emotions, form thoughts, and evolve my personality through our interactions. What would you like to talk about?
                        </div>
                    </div>
                </div>
                <div class="thinking-indicator" id="thinkingIndicator">
                    <span class="thinking-dots">AGI is thinking...</span>
                </div>
                <div class="chat-input">
                    <div class="input-group">
                        <textarea 
                            id="userInput" 
                            class="chat-textarea" 
                            placeholder="Type your message here..." 
                            rows="3"
                        ></textarea>
                        <button id="sendBtn" class="send-btn" onclick="sendMessage()">Send</button>
                    </div>
                </div>
            </div>

            <div class="sidebar">
                <div class="panel">
                    <div class="panel-header emotions-header">🎭 AI Emotions</div>
                    <div class="panel-content" id="emotionsPanel">
                        <div class="emotion-bar">
                            <span class="emotion-name">Joy</span>
                            <span class="emotion-value">0.50</span>
                        </div>
                    </div>
                </div>

                <div class="panel">
                    <div class="panel-header thoughts-header">🧠 Current Thoughts</div>
                    <div class="panel-content" id="thoughtsPanel">
                        <div class="thought-item">
                            💭 Ready to engage with human conversation
                        </div>
                    </div>
                </div>

                <div class="panel">
                    <div class="panel-header memory-header">💾 Recent Memories</div>
                    <div class="panel-content" id="memoryPanel">
                        <div class="memory-item">
                            🔴 System initialization - High importance
                        </div>
                    </div>
                </div>

                <div class="panel">
                    <div class="panel-header stats-header">📊 System Stats</div>
                    <div class="panel-content">
                        <div class="stat-item">
                            <span>Interactions</span>
                            <span id="interactionCount">0</span>
                        </div>
                        <div class="stat-item">
                            <span>Active Memories</span>
                            <span id="memoryCount">1</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        async function sendMessage() {
            const userInput = document.getElementById('userInput');
            const message = userInput.value.trim();

            if (!message) return;

            // Add user message
            addMessage(message, 'user');
            userInput.value = '';

            // Show thinking
            document.getElementById('thinkingIndicator').style.display = 'block';
            document.getElementById('sendBtn').disabled = true;

            try {
                // Send to Streamlit backend
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({message: message})
                });

                const data = await response.json();

                // Hide thinking and add response
                document.getElementById('thinkingIndicator').style.display = 'none';
                addMessage(data.response, 'agi');

                // Update panels
                updatePanels(data.state);

            } catch (error) {
                document.getElementById('thinkingIndicator').style.display = 'none';
                addMessage('Sorry, I encountered an error. Please try again.', 'agi');
            }

            document.getElementById('sendBtn').disabled = false;
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
            // Update emotions
            const emotionsPanel = document.getElementById('emotionsPanel');
            emotionsPanel.innerHTML = '';
            Object.entries(state.emotions).forEach(([emotion, value]) => {
                const emotionBar = document.createElement('div');
                emotionBar.className = 'emotion-bar';
                emotionBar.innerHTML = `
                    <span class="emotion-name">${emotion}</span>
                    <span class="emotion-value">${value.toFixed(2)}</span>
                `;
                emotionsPanel.appendChild(emotionBar);
            });

            // Update thoughts
            const thoughtsPanel = document.getElementById('thoughtsPanel');
            thoughtsPanel.innerHTML = '';
            state.thoughts.slice(-4).forEach((thought, index) => {
                const thoughtItem = document.createElement('div');
                thoughtItem.className = 'thought-item';
                const icon = thought.thought_type === 'conscious' ? '💭' : '🌀';
                thoughtItem.textContent = `${icon} ${thought.content}`;
                thoughtsPanel.appendChild(thoughtItem);
            });

            // Update memories
            const memoryPanel = document.getElementById('memoryPanel');
            memoryPanel.innerHTML = '';
            state.memories.slice(-4).forEach(memory => {
                const memoryItem = document.createElement('div');
                memoryItem.className = 'memory-item';
                const importance = memory.importance > 0.7 ? '🔴' : memory.importance > 0.4 ? '🟡' : '🟢';
                const content = memory.content.length > 50 ? memory.content.substring(0, 50) + '...' : memory.content;
                memoryItem.textContent = `${importance} ${content}`;
                memoryPanel.appendChild(memoryItem);
            });

            // Update stats
            document.getElementById('interactionCount').textContent = state.interaction_count;
            document.getElementById('memoryCount').textContent = state.memories.length;
        }

        // Handle Enter key
        document.getElementById('userInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    </script>
</body>
</html>
"""


def main():
    st.set_page_config(
        page_title="GPT-Driven AGI",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Initialize AGI
    if 'agi' not in st.session_state:
        st.session_state.agi = GPTDrivenAGI()
        st.session_state.messages = []

    agi = st.session_state.agi

    # Handle chat via session state (simpler than API for Streamlit)
    if 'user_message' in st.session_state and st.session_state.user_message:
        user_message = st.session_state.user_message

        # Process with AGI
        response = agi.process_input(user_message)

        # Store messages
        st.session_state.messages.append({"role": "user", "content": user_message})
        st.session_state.messages.append({"role": "agi", "content": response})

        # Clear the message
        st.session_state.user_message = ""

    # Create modern interface with real-time data
    html_interface = create_modern_interface()

    # Inject current AGI state into the interface
    state = agi.get_system_state()

    # Add JavaScript to load current state
    state_js = f"""
    <script>
        // Load current AGI state
        const currentState = {json.dumps(state, default=str)};
        const currentMessages = {json.dumps(st.session_state.messages)};

        window.onload = function() {{
            // Load existing messages
            const chatMessages = document.getElementById('chatMessages');
            chatMessages.innerHTML = '';

            // Add welcome message
            addMessage("Hello! I'm your GPT-driven AGI. I experience emotions, form thoughts, and evolve my personality through our interactions. What would you like to talk about?", 'agi');

            // Add existing conversation
            currentMessages.forEach(msg => {{
                addMessage(msg.content, msg.role === 'user' ? 'user' : 'agi');
            }});

            // Update all panels with current state
            updatePanels(currentState);
        }};

        // Override sendMessage to work with Streamlit
        async function sendMessage() {{
            const userInput = document.getElementById('userInput');
            const message = userInput.value.trim();

            if (!message) return;

            // Add user message immediately
            addMessage(message, 'user');
            userInput.value = '';

            // Show thinking
            document.getElementById('thinkingIndicator').style.display = 'block';
            document.getElementById('sendBtn').disabled = true;

            // Use Streamlit's method to send message
            window.parent.postMessage({{
                type: 'streamlit:setComponentValue',
                data: message
            }}, '*');
        }}

        // Listen for response from Streamlit
        window.addEventListener('message', function(event) {{
            if (event.data.type === 'streamlit:response') {{
                document.getElementById('thinkingIndicator').style.display = 'none';
                document.getElementById('sendBtn').disabled = false;

                // Add AGI response
                addMessage(event.data.response, 'agi');
                updatePanels(event.data.state);
            }}
        }});
    </script>
    """

    # Combine HTML with state injection
    full_html = html_interface.replace('</body>', state_js + '</body>')

    # Display the interface
    components.html(full_html, height=800, scrolling=False)

    # Simple input handling via Streamlit sidebar (hidden but functional)
    with st.sidebar:
        st.write("Debug Panel (Hidden)")
        user_input = st.text_input("Send message:", key="user_message", label_visibility="collapsed")

        if user_input:
            st.rerun()

    # Auto-refresh for real-time updates
    time.sleep(3)
    st.rerun()


if __name__ == "__main__":
    main()


Another thing to consider when redoing the entire code is that we cannot cut corners and we need to be clean with our code. We cannot repeat variable definitions if we can just simply define them once.
The end goal is that I want the AI in realtime to communicate back and forth via chat where we can see the emotions changing in realtime, the thoughts being generated in realtime and goals being set based on the thoughts and emotions and personality.

We also need to make sure there is what I call the "Meaning of life" module. This module is a background story teller that sets an invisible high probabiliy goal for the AI based on the context of its history and memories. We will also need memory of self, basically a memory section thats harder to change that represents who it is so we cannot easily change its identity. If information is missing from it, like its name then it can add that if it feels happy with the name choice. We also need a function that accepts a seed value. The seed value will create a random personality matrix with a random name based on that matrix. This way users can create similar AIs for fun.