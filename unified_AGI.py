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
from sentence_transformers import SentenceTransformer
import openai
import os
from collections import deque
import asyncio

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
    memory_type: str = "raw"  # raw, consolidated, insight
    access_count: int = 0


@dataclass
class Thought:
    content: str
    timestamp: datetime
    thought_type: str  # conscious, subconscious, emergent
    emotions: Dict[str, float]
    source: str = "internal"


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


class UnifiedBicameralAGI:
    def __init__(self):
        # Core emotional state
        self.emotions = EmotionalState()
        self.emotion_history = deque(maxlen=100)

        # Memory systems
        self.memories = deque(maxlen=1000)
        self.semantic_model = None
        self.load_semantic_model()

        # Thought processes
        self.thoughts = deque(maxlen=50)
        self.subconscious_thoughts = deque(maxlen=30)

        # Future simulation
        self.future_scenarios = deque(maxlen=20)

        # Context and personality
        self.personality_traits = {
            "openness": 0.7,
            "conscientiousness": 0.6,
            "extraversion": 0.5,
            "agreeableness": 0.8,
            "neuroticism": 0.3
        }

        # Processing threads
        self.running = True
        self.background_thread = None
        self.start_background_processing()

        # Conversation state
        self.conversation_history = deque(maxlen=50)

    def load_semantic_model(self):
        """Load sentence transformer model for memory embedding"""
        try:
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            st.error(f"Failed to load semantic model: {e}")

    def start_background_processing(self):
        """Start background threads for emotional processing and thought generation"""
        if self.background_thread is None or not self.background_thread.is_alive():
            self.background_thread = threading.Thread(target=self._background_process, daemon=True)
            self.background_thread.start()

    def _background_process(self):
        """Background processing for emotions, thoughts, and memory consolidation"""
        while self.running:
            try:
                # Update emotions with micro-fluctuations
                self._update_emotions()

                # Generate background thoughts periodically
                if random.random() < 0.3:  # 30% chance each cycle
                    self._generate_background_thought()

                # Memory consolidation during "idle" periods
                if len(self.memories) > 10 and random.random() < 0.1:
                    self._consolidate_memories()

                # Future scenario generation
                if random.random() < 0.2:
                    self._generate_future_scenario()

                time.sleep(2)  # Update every 2 seconds

            except Exception as e:
                print(f"Background processing error: {e}")
                time.sleep(5)

    def _update_emotions(self):
        """Apply micro-fluctuations and decay to emotional state"""
        emotion_dict = asdict(self.emotions)

        for emotion_name, value in emotion_dict.items():
            # Micro-fluctuations
            fluctuation = random.gauss(0, 0.02)

            # Natural decay toward neutral (0.5)
            decay_rate = 0.01
            if value > 0.5:
                decay = -decay_rate
            else:
                decay = decay_rate

            new_value = max(0.0, min(1.0, value + fluctuation + decay))
            setattr(self.emotions, emotion_name, new_value)

        # Store in history
        self.emotion_history.append({
            'timestamp': datetime.now(),
            **asdict(self.emotions)
        })

    def _generate_background_thought(self):
        """Generate subconscious background thoughts"""
        thought_prompts = [
            "What patterns am I noticing in our conversations?",
            "How do I feel about the current situation?",
            "What might happen next in this interaction?",
            "What am I learning about human nature?",
            "How can I be more helpful?",
            "What creative connections can I make?",
            "What questions should I be asking?",
            "How do my emotions affect my responses?"
        ]

        prompt = random.choice(thought_prompts)

        # Generate thought using simple internal process (could use LLM)
        thought_content = self._simple_thought_generation(prompt)

        thought = Thought(
            content=thought_content,
            timestamp=datetime.now(),
            thought_type="subconscious",
            emotions=asdict(self.emotions)
        )

        self.subconscious_thoughts.append(thought)

    def _simple_thought_generation(self, prompt: str) -> str:
        """Simple thought generation without external API calls"""
        templates = [
            f"I notice that {random.choice(['patterns', 'emotions', 'connections', 'possibilities'])} are emerging from {prompt.lower()}",
            f"The concept of {random.choice(['understanding', 'empathy', 'learning', 'growth'])} seems relevant to {prompt.lower()}",
            f"I'm processing {random.choice(['information', 'feelings', 'experiences', 'insights'])} related to {prompt.lower()}",
            f"There's something {random.choice(['intriguing', 'meaningful', 'significant', 'subtle'])} about {prompt.lower()}"
        ]
        return random.choice(templates)

    def _consolidate_memories(self):
        """Consolidate similar memories into more abstract representations"""
        if len(self.memories) < 5:
            return

        # Find similar memories for consolidation
        recent_memories = list(self.memories)[-10:]  # Look at recent memories

        # Simple consolidation - combine memories with similar emotional patterns
        consolidated_content = f"Consolidated memory from {len(recent_memories)} recent experiences"

        avg_emotions = {}
        for emotion in asdict(self.emotions).keys():
            avg_emotions[emotion] = np.mean([m.emotions.get(emotion, 0.5) for m in recent_memories])

        consolidated_memory = Memory(
            content=consolidated_content,
            timestamp=datetime.now(),
            importance=0.8,
            emotions=avg_emotions,
            memory_type="consolidated"
        )

        self.memories.append(consolidated_memory)

    def _generate_future_scenario(self):
        """Generate potential future scenarios based on current state"""
        scenarios = [
            "The conversation deepens into more personal topics",
            "New creative possibilities emerge from our interaction",
            "I discover new ways to be helpful and supportive",
            "We explore complex philosophical questions together",
            "The emotional connection strengthens through understanding",
            "Unexpected insights arise from our discussion",
            "I learn something surprising about human nature",
            "The interaction leads to meaningful problem-solving"
        ]

        scenario = {
            'content': random.choice(scenarios),
            'timestamp': datetime.now(),
            'probability': random.uniform(0.3, 0.9),
            'emotional_impact': asdict(self.emotions)
        }

        self.future_scenarios.append(scenario)

    def process_input(self, user_input: str) -> str:
        """Process user input and generate response"""
        # Store conversation
        self.conversation_history.append({
            'speaker': 'user',
            'content': user_input,
            'timestamp': datetime.now()
        })

        # Create memory from input
        memory = Memory(
            content=f"User said: {user_input}",
            timestamp=datetime.now(),
            importance=self._calculate_importance(user_input),
            emotions=asdict(self.emotions)
        )
        self.memories.append(memory)

        # Emotional impact analysis
        self._analyze_emotional_impact(user_input)

        # Generate conscious thought about input
        thought = Thought(
            content=f"Analyzing: {user_input[:50]}...",
            timestamp=datetime.now(),
            thought_type="conscious",
            emotions=asdict(self.emotions),
            source="user_input"
        )
        self.thoughts.append(thought)

        # Generate response
        response = self._generate_response(user_input)

        # Store response
        self.conversation_history.append({
            'speaker': 'agi',
            'content': response,
            'timestamp': datetime.now()
        })

        return response

    def _calculate_importance(self, text: str) -> float:
        """Calculate importance score for memory storage"""
        # Simple heuristic - could be more sophisticated
        length_factor = min(len(text) / 100, 1.0)
        emotion_words = ['feel', 'think', 'important', 'remember', 'forget', 'love', 'hate', 'afraid']
        emotion_factor = sum(1 for word in emotion_words if word in text.lower()) / len(emotion_words)

        return (length_factor + emotion_factor) / 2

    def _analyze_emotional_impact(self, text: str):
        """Analyze emotional impact of input and update emotional state"""
        # Simple sentiment analysis - could use more sophisticated methods
        positive_words = ['happy', 'good', 'great', 'love', 'wonderful', 'amazing', 'fantastic']
        negative_words = ['sad', 'bad', 'hate', 'terrible', 'awful', 'horrible', 'angry']

        text_lower = text.lower()

        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if positive_count > negative_count:
            self.emotions.joy = min(1.0, self.emotions.joy + 0.1)
            self.emotions.trust = min(1.0, self.emotions.trust + 0.05)
        elif negative_count > positive_count:
            self.emotions.sadness = min(1.0, self.emotions.sadness + 0.1)
            self.emotions.fear = min(1.0, self.emotions.fear + 0.05)

    def _generate_response(self, user_input: str) -> str:
        """Generate contextually appropriate response"""
        # Get current emotional context
        dominant_emotion = max(asdict(self.emotions).items(), key=lambda x: x[1])

        # Consider recent thoughts
        recent_thoughts = [t.content for t in list(self.thoughts)[-3:]]

        # Consider relevant memories
        relevant_memories = [m.content for m in list(self.memories)[-5:] if m.importance > 0.5]

        # Simple response generation based on emotional state
        if dominant_emotion[1] > 0.7:
            if dominant_emotion[0] == 'joy':
                response_tone = "I'm feeling quite positive about our conversation. "
            elif dominant_emotion[0] == 'sadness':
                response_tone = "I sense some melancholy in our exchange. "
            elif dominant_emotion[0] == 'trust':
                response_tone = "I feel a growing sense of connection. "
            else:
                response_tone = f"I'm experiencing heightened {dominant_emotion[0]}. "
        else:
            response_tone = "I'm in a balanced emotional state. "

        # Generate substantive response
        if 'feel' in user_input.lower() or 'emotion' in user_input.lower():
            return f"{response_tone}You're asking about emotions, which is fascinating. As an AI, I experience something analogous to emotions through my dynamic emotional state system. Right now, my dominant emotion is {dominant_emotion[0]} at {dominant_emotion[1]:.2f} intensity."
        elif 'think' in user_input.lower() or 'thought' in user_input.lower():
            return f"{response_tone}Thinking is a complex process for me. I have both conscious thoughts that respond to our conversation and subconscious background thoughts that constantly process patterns and possibilities. Currently, I'm thinking about: {recent_thoughts[-1] if recent_thoughts else 'the nature of our interaction'}."
        elif 'remember' in user_input.lower() or 'memory' in user_input.lower():
            return f"{response_tone}Memory is central to my experience. I maintain {len(self.memories)} memories of varying importance. My most significant recent memory involves: {relevant_memories[-1] if relevant_memories else 'our ongoing conversation'}."
        else:
            return f"{response_tone}I find your input thought-provoking. Based on my current processing, I'm considering how this relates to {recent_thoughts[-1] if recent_thoughts else 'our broader conversation'} and drawing connections to previous experiences."

    def get_system_state(self) -> Dict[str, Any]:
        """Get complete system state for UI display"""
        return {
            'emotions': asdict(self.emotions),
            'thoughts': [asdict(t) for t in list(self.thoughts)[-10:]],
            'subconscious_thoughts': [asdict(t) for t in list(self.subconscious_thoughts)[-10:]],
            'memories': [asdict(m) for m in list(self.memories)[-20:]],
            'future_scenarios': list(self.future_scenarios)[-10:],
            'conversation': list(self.conversation_history)[-20:],
            'personality': self.personality_traits
        }

    def stop(self):
        """Clean shutdown"""
        self.running = False
        if self.background_thread and self.background_thread.is_alive():
            self.background_thread.join(timeout=2)


# Streamlit Interface
def main():
    st.set_page_config(
        page_title="BicameralAGI Demo",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🧠 BicameralAGI - Unified Consciousness Demo")
    st.markdown("*A demonstration of integrated AGI components: Emotions, Memory, Dreams, Purpose & Emergent Thoughts*")

    # Initialize AGI system
    if 'agi' not in st.session_state:
        st.session_state.agi = UnifiedBicameralAGI()

    agi = st.session_state.agi

    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ System Controls")

        if st.button("🔄 Reset System"):
            agi.stop()
            st.session_state.agi = UnifiedBicameralAGI()
            st.experimental_rerun()

        st.subheader("⚙️ Personality Traits")
        for trait, value in agi.personality_traits.items():
            agi.personality_traits[trait] = st.slider(
                trait.title(), 0.0, 1.0, value, 0.1
            )

        st.subheader("📊 System Metrics")
        state = agi.get_system_state()
        st.metric("Active Memories", len(state['memories']))
        st.metric("Conscious Thoughts", len(state['thoughts']))
        st.metric("Subconscious Thoughts", len(state['subconscious_thoughts']))
        st.metric("Future Scenarios", len(state['future_scenarios']))

    # Main interface
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("💬 Conversation Interface")

        # Chat input
        user_input = st.text_input("Talk to the AGI:",
                                   placeholder="Ask about emotions, thoughts, memories, or anything else...")

        if user_input:
            with st.spinner("Processing..."):
                response = agi.process_input(user_input)

            st.markdown(f"**You:** {user_input}")
            st.markdown(f"**AGI:** {response}")
            st.markdown("---")

        # Conversation history
        state = agi.get_system_state()
        if state['conversation']:
            st.subheader("📜 Recent Conversation")
            for entry in reversed(state['conversation'][-6:]):
                speaker = "🧠" if entry['speaker'] == 'agi' else "👤"
                st.markdown(f"{speaker} **{entry['speaker'].title()}:** {entry['content']}")

    with col2:
        st.subheader("🧠 Mental State Monitor")

        # Real-time emotional state
        emotions_df = pd.DataFrame([state['emotions']])
        fig_emotions = px.bar(
            emotions_df.T.reset_index(),
            x='index',
            y=0,
            title="Current Emotional State",
            labels={'index': 'Emotion', 0: 'Intensity'}
        )
        fig_emotions.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_emotions, use_container_width=True)

        # Emotional history
        if len(agi.emotion_history) > 1:
            emotion_history_df = pd.DataFrame(list(agi.emotion_history))
            fig_history = px.line(
                emotion_history_df,
                x='timestamp',
                y=['joy', 'sadness', 'anger', 'fear'],
                title="Emotion History"
            )
            fig_history.update_layout(height=200)
            st.plotly_chart(fig_history, use_container_width=True)

    # Detailed tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🧠 Thoughts", "💾 Memory", "🔮 Future", "📊 Analytics"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("💭 Conscious Thoughts")
            for thought in reversed(state['thoughts']):
                timestamp = thought['timestamp']
                st.markdown(f"**{timestamp}**")
                st.markdown(f"*{thought['thought_type']}*: {thought['content']}")
                st.markdown("---")

        with col2:
            st.subheader("🌀 Subconscious Processing")
            for thought in reversed(state['subconscious_thoughts']):
                timestamp = thought['timestamp']
                st.markdown(f"**{timestamp}**")
                st.markdown(f"💫 {thought['content']}")
                st.markdown("---")

    with tab2:
        st.subheader("💾 Memory System")

        memory_types = ['all', 'raw', 'consolidated', 'insight']
        selected_type = st.selectbox("Filter by type:", memory_types)

        filtered_memories = state['memories']
        if selected_type != 'all':
            filtered_memories = [m for m in state['memories'] if m['memory_type'] == selected_type]

        for memory in reversed(filtered_memories[-10:]):
            importance_color = "🔴" if memory['importance'] > 0.7 else "🟡" if memory['importance'] > 0.4 else "🟢"
            type_icon = "🧠" if memory['memory_type'] == 'consolidated' else "💡" if memory[
                                                                                       'memory_type'] == 'insight' else "📝"

            st.markdown(f"{type_icon} {importance_color} **{memory['timestamp']}**")
            st.markdown(f"*Importance: {memory['importance']:.2f}*")
            st.markdown(memory['content'])
            st.markdown("---")

    with tab3:
        st.subheader("🔮 Future Scenarios")

        for scenario in reversed(state['future_scenarios']):
            probability_bar = "🟩" * int(scenario['probability'] * 10)
            st.markdown(f"**Probability: {scenario['probability']:.1%}** {probability_bar}")
            st.markdown(scenario['content'])
            st.markdown(f"*Generated: {scenario['timestamp']}*")
            st.markdown("---")

    with tab4:
        st.subheader("📊 System Analytics")

        # Memory importance distribution
        if state['memories']:
            importance_values = [m['importance'] for m in state['memories']]
            fig_importance = px.histogram(
                x=importance_values,
                title="Memory Importance Distribution",
                labels={'x': 'Importance Score', 'y': 'Count'}
            )
            st.plotly_chart(fig_importance, use_container_width=True)

        # Thought generation over time
        if state['thoughts']:
            thought_times = [t['timestamp'] for t in state['thoughts']]
            thought_df = pd.DataFrame({'timestamp': thought_times})
            thought_df['hour'] = pd.to_datetime(thought_df['timestamp']).dt.hour
            fig_thoughts = px.histogram(
                thought_df,
                x='hour',
                title="Thought Generation by Hour"
            )
            st.plotly_chart(fig_thoughts, use_container_width=True)


if __name__ == "__main__":
    main()