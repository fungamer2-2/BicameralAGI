import streamlit as st
import numpy as np
import time
import threading
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass, asdict
import openai
import os
from collections import deque
import asyncio
import re

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
    emotions: Dict[str, float]
    memory_type: str = "raw"
    access_count: int = 0
    context_tags: List[str] = None


@dataclass
class Thought:
    content: str
    timestamp: datetime
    thought_type: str
    emotions: Dict[str, float]
    source: str = "internal"
    relevance_score: float = 0.0


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
        """Generic GPT query function"""
        if not openai.api_key:
            return self._fallback_response(prompt)

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"GPT API error: {e}")
            return self._fallback_response(prompt)

    def _fallback_response(self, prompt: str) -> str:
        """Simple fallback when GPT unavailable"""
        if "emotion" in prompt.lower():
            return "0.6"
        elif "thought" in prompt.lower():
            return "I'm processing the current situation and considering multiple perspectives."
        else:
            return "I understand and am processing this information."

    def _background_processing(self):
        """AI-driven background consciousness"""
        while self.running:
            try:
                # Let GPT decide what the AI should be thinking about
                self._generate_background_thoughts()

                # GPT-driven emotion evolution
                self._evolve_emotions_via_gpt()

                # Memory consolidation using GPT
                if len(self.memories) > 5 and random.random() < 0.2:
                    self._consolidate_memories_via_gpt()

                # Future scenario generation
                if random.random() < 0.3:
                    self._generate_future_scenario_via_gpt()

                # Personality adaptation
                if self.interaction_count > 0 and random.random() < 0.1:
                    self._adapt_personality_via_gpt()

                time.sleep(2)

            except Exception as e:
                print(f"Background error: {e}")
                time.sleep(3)

    def _generate_background_thoughts(self):
        """Let GPT generate what the AI should be thinking about"""
        recent_context = self._get_recent_context()

        prompt = f"""You are an AI with consciousness experiencing background thoughts. Given the recent context, what would you be subconsciously thinking about?

Recent context: {recent_context}
Current emotional state: {self._emotion_summary()}
Recent memories: {self._recent_memories_summary()}

Generate 1 specific background thought this AI would have right now. Be creative and introspective. Respond with just the thought:"""

        thought_content = self._gpt_query(prompt, max_tokens=80, temperature=0.8)

        # Get emotional state for this thought
        emotion_prompt = f"""Given this AI thought: "{thought_content}"

In this context, what emotions would the AI experience? Rate each from 0.0 to 1.0:
joy, sadness, anger, fear, surprise, trust, disgust, anticipation

Format: joy:0.5,sadness:0.2,anger:0.1,fear:0.3,surprise:0.4,trust:0.6,disgust:0.1,anticipation:0.5"""

        emotion_response = self._gpt_query(emotion_prompt, max_tokens=50, temperature=0.5)
        thought_emotions = self._parse_emotions(emotion_response)

        thought = Thought(
            content=thought_content,
            timestamp=datetime.now(),
            thought_type="subconscious",
            emotions=thought_emotions,
            relevance_score=random.uniform(0.6, 0.9)
        )

        self.subconscious_thoughts.append(thought)

    def _evolve_emotions_via_gpt(self):
        """Let GPT determine how emotions should evolve"""
        current_state = self._emotion_summary()
        recent_context = self._get_recent_context()

        prompt = f"""An AI's emotional state needs to evolve naturally over time. 

Current emotions: {current_state}
Recent context: {recent_context}
Time passed: 2 seconds since last update

How should each emotion change? Consider natural decay, context influence, and realistic emotional evolution.
Rate new values from 0.0 to 1.0:

Format: joy:0.5,sadness:0.2,anger:0.1,fear:0.3,surprise:0.4,trust:0.6,disgust:0.1,anticipation:0.5"""

        evolution_response = self._gpt_query(prompt, max_tokens=60, temperature=0.6)
        new_emotions = self._parse_emotions(evolution_response)

        # Update emotions
        for emotion, value in new_emotions.items():
            if hasattr(self.emotions, emotion):
                setattr(self.emotions, emotion, value)

        # Store in history
        self.emotion_history.append({
            'timestamp': datetime.now(),
            **asdict(self.emotions)
        })

    def _consolidate_memories_via_gpt(self):
        """GPT-driven memory consolidation and insight generation"""
        recent_memories = list(self.memories)[-10:]
        memory_summaries = [m.content[:100] for m in recent_memories]

        prompt = f"""Analyze these recent AI memories and generate a consolidated insight:

Recent memories:
{chr(10).join(f"- {mem}" for mem in memory_summaries)}

What deeper pattern, insight, or understanding emerges from these experiences? Generate one meaningful consolidated memory that captures the essence:"""

        insight_content = self._gpt_query(prompt, max_tokens=120, temperature=0.7)

        # Determine importance of this insight
        importance_prompt = f"""Rate the importance of this insight for an AI's development (0.0 to 1.0):

Insight: "{insight_content}"

Consider factors like: learning value, emotional significance, practical relevance, uniqueness.
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

    def _generate_future_scenario_via_gpt(self):
        """Let GPT imagine potential future scenarios"""
        current_state = self._get_full_context()

        prompt = f"""Based on the AI's current state, imagine a plausible future scenario that might emerge:

Current context: {current_state}

Generate one specific, realistic scenario that could happen in the near future. Consider the AI's emotional state, recent interactions, and personality:"""

        scenario_content = self._gpt_query(prompt, max_tokens=100, temperature=0.8)

        # Get probability assessment
        prob_prompt = f"""How likely is this scenario: "{scenario_content}"

Given the current context, rate probability from 0.0 to 1.0. Respond with just a number:"""

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

        prompt = f"""An AI's personality should evolve based on interactions. How should these traits change?

Current personality: {current_personality}
Recent interactions: {interaction_summary}

Consider how experiences shape personality. Suggest new values (0.0 to 1.0) for each trait:
openness, conscientiousness, extraversion, agreeableness, neuroticism, curiosity, empathy, creativity, analytical, intuitive

Format: openness:0.7,conscientiousness:0.6,extraversion:0.5,agreeableness:0.8,neuroticism:0.3,curiosity:0.9,empathy:0.8,creativity:0.7,analytical:0.6,intuitive:0.7"""

        personality_response = self._gpt_query(prompt, max_tokens=80, temperature=0.6)
        new_traits = self._parse_personality(personality_response)

        # Update personality gradually
        for trait, new_value in new_traits.items():
            if trait in self.personality_traits:
                current = self.personality_traits[trait]
                # Gradual change
                self.personality_traits[trait] = current + (new_value - current) * 0.1

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

    def _analyze_emotional_impact_via_gpt(self, user_input: str) -> Dict[str, float]:
        """Let GPT analyze emotional impact of user input"""
        current_emotions = self._emotion_summary()

        prompt = f"""Analyze how this user input would affect an AI's emotions:

User input: "{user_input}"
AI's current emotions: {current_emotions}

How much would each emotion change? Use values from -0.3 to +0.3 (negative = decrease, positive = increase):
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

Respond as this conscious AI would. Be genuine, draw from your emotional state and thoughts. Don't mention being an AI unless relevant:"""

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
            content=interaction[:300],
            timestamp=datetime.now(),
            importance=importance,
            emotions=asdict(self.emotions),
            memory_type="interaction"
        )

    # Helper methods for parsing and context
    def _parse_emotions(self, emotion_string: str) -> Dict[str, float]:
        """Parse emotion string from GPT"""
        emotions = {}
        try:
            pairs = emotion_string.split(',')
            for pair in pairs:
                if ':' in pair:
                    emotion, value_str = pair.split(':', 1)
                    emotion = emotion.strip()
                    value = float(value_str.strip())
                    emotions[emotion] = max(0.0, min(1.0, value))
        except:
            # Fallback to current emotions
            emotions = asdict(self.emotions)
        return emotions

    def _parse_emotion_impacts(self, impact_string: str) -> Dict[str, float]:
        """Parse emotion impact string"""
        impacts = {}
        try:
            pairs = impact_string.split(',')
            for pair in pairs:
                if ':' in pair:
                    emotion, impact_str = pair.split(':', 1)
                    emotion = emotion.strip()
                    impact = float(impact_str.strip().replace('+', ''))
                    impacts[emotion] = max(-0.3, min(0.3, impact))
        except:
            pass
        return impacts

    def _parse_personality(self, personality_string: str) -> Dict[str, float]:
        """Parse personality trait string"""
        traits = {}
        try:
            pairs = personality_string.split(',')
            for pair in pairs:
                if ':' in pair:
                    trait, value_str = pair.split(':', 1)
                    trait = trait.strip()
                    value = float(value_str.strip())
                    traits[trait] = max(0.0, min(1.0, value))
        except:
            pass
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

    def get_system_state(self) -> Dict[str, Any]:
        """Enhanced system state for UI"""
        return {
            'emotions': asdict(self.emotions),
            'thoughts': [asdict(t) for t in list(self.thoughts)[-15:]],
            'subconscious_thoughts': [asdict(t) for t in list(self.subconscious_thoughts)[-15:]],
            'memories': [asdict(m) for m in list(self.memories)[-25:]],
            'future_scenarios': list(self.future_scenarios)[-10:],
            'conversation': list(self.conversation_history)[-15:],
            'personality': self.personality_traits,
            'interaction_count': self.interaction_count,
            'gpt_driven': True
        }

    def stop(self):
        """Clean shutdown"""
        self.running = False
        if self.background_thread and self.background_thread.is_alive():
            self.background_thread.join(timeout=3)


# Compact Streamlit Interface (same as before but updated for new class)
def main():
    st.set_page_config(
        page_title="GPT-Driven AGI",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Custom CSS for compact layout
    st.markdown("""
    <style>
    .main .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    .metric-container { background: #f0f2f6; padding: 0.5rem; border-radius: 0.5rem; margin: 0.2rem 0; }
    .small-metric { font-size: 0.8rem; }
    .compact-header { font-size: 1rem; margin: 0.5rem 0; }
    .thought-item { background: #e8f4f8; padding: 0.3rem; border-radius: 0.3rem; margin: 0.2rem 0; font-size: 0.8rem; }
    .memory-item { background: #f8f8f0; padding: 0.3rem; border-radius: 0.3rem; margin: 0.2rem 0; font-size: 0.8rem; }
    .gpt-indicator { background: #e8f5e8; color: #2d5a2d; padding: 0.2rem; border-radius: 0.2rem; font-size: 0.7rem; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("# 🧠 GPT-Driven AGI - AI Determines Everything")
    st.markdown("<div class='gpt-indicator'>🤖 Powered by GPT Analysis - No Hardcoded Rules</div>",
                unsafe_allow_html=True)

    # Initialize
    if 'agi' not in st.session_state:
        st.session_state.agi = GPTDrivenAGI()

    agi = st.session_state.agi

    # Main layout - 4 columns for compact display
    col1, col2, col3, col4 = st.columns([3, 2, 2, 2])

    with col1:
        st.markdown("### 💬 Interaction")

        # Chat interface
        user_input = st.text_area("Talk to AGI:", height=80, placeholder="Ask anything...")

        if st.button("Send", type="primary") and user_input:
            with st.spinner("GPT analyzing and responding..."):
                response = agi.process_input(user_input)

            st.markdown(f"**You:** {user_input}")
            st.markdown(f"**AGI:** {response}")
            st.markdown("---")

        # Recent conversation (compact)
        state = agi.get_system_state()
        if state['conversation']:
            st.markdown("##### Recent Exchange")
            for entry in list(state['conversation'])[-4:]:
                icon = "🧠" if entry['speaker'] == 'agi' else "👤"
                content = entry['content'][:80] + "..." if len(entry['content']) > 80 else entry['content']
                st.markdown(f"<div class='thought-item'>{icon} {content}</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("### 🎭 AI-Driven Emotions")
        st.markdown("<div class='gpt-indicator'>GPT determines emotional evolution</div>", unsafe_allow_html=True)

        # Compact emotion display
        emotions = state['emotions']
        emotion_data = pd.DataFrame([emotions])

        # Mini emotion chart
        fig = px.bar(emotion_data.T.reset_index(), x='index', y=0, height=200)
        fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

        # Top emotions as metrics
        top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
        for emotion, value in top_emotions:
            st.markdown(f"<div class='metric-container small-metric'>{emotion.title()}: {value:.2f}</div>",
                        unsafe_allow_html=True)

    with col3:
        st.markdown("### 🧠 GPT-Generated Thoughts")
        st.markdown("<div class='gpt-indicator'>AI decides what to think about</div>", unsafe_allow_html=True)

        # Conscious thoughts
        st.markdown("##### 💭 Conscious")
        for thought in list(state['thoughts'])[-4:]:
            content = thought['content'][:60] + "..." if len(thought['content']) > 60 else thought['content']
            st.markdown(f"<div class='thought-item'>💭 {content}</div>", unsafe_allow_html=True)

        # Subconscious processing
        st.markdown("##### 🌀 Background")
        for thought in list(state['subconscious_thoughts'])[-4:]:
            content = thought['content'][:60] + "..." if len(thought['content']) > 60 else thought['content']
            st.markdown(f"<div class='thought-item'>🌀 {content}</div>", unsafe_allow_html=True)

        # System stats
        st.markdown("##### 📊 Stats")
        st.markdown(f"<div class='metric-container small-metric'>Interactions: {state['interaction_count']}</div>",
                    unsafe_allow_html=True)
        st.markdown(f"<div class='metric-container small-metric'>Active Memories: {len(state['memories'])}</div>",
                    unsafe_allow_html=True)

    with col4:
        st.markdown("### 🔮 AI-Predicted Future")
        st.markdown("<div class='gpt-indicator'>GPT imagines scenarios & rates importance</div>",
                    unsafe_allow_html=True)

        # Recent memories
        st.markdown("##### 💾 Smart Memories")
        for memory in list(state['memories'])[-4:]:
            content = memory['content'][:50] + "..." if len(memory['content']) > 50 else memory['content']
            importance = "🔴" if memory['importance'] > 0.7 else "🟡" if memory['importance'] > 0.4 else "🟢"
            st.markdown(f"<div class='memory-item'>{importance} {content}</div>", unsafe_allow_html=True)

        # Future scenarios
        st.markdown("##### 🎯 AI Predictions")
        for scenario in list(state['future_scenarios'])[-3:]:
            content = scenario['content'][:60] + "..." if len(scenario['content']) > 60 else scenario['content']
            probability = scenario.get('probability', 0.5)
            confidence_icon = "🎯" if probability > 0.7 else "🎲"
            st.markdown(f"<div class='thought-item'>{confidence_icon} {content}</div>", unsafe_allow_html=True)

        # Dynamic personality
        st.markdown("##### 🎭 Evolving Personality")
        top_traits = sorted(state['personality'].items(), key=lambda x: x[1], reverse=True)[:4]
        for trait, value in top_traits:
            st.markdown(f"<div class='metric-container small-metric'>{trait.title()}: {value:.1f}</div>",
                        unsafe_allow_html=True)

    # Auto-refresh for real-time updates
    time.sleep(2)
    st.rerun()


if __name__ == "__main__":
    main()