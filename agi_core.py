"""
Bicameral AGI - Consolidated Core Intelligence Module
=====================================================

This module represents the complete, unified BICA AGI system. It integrates
all core cognitive functions into a single, self-contained file, removing
dependencies on older, redundant modules. The system loads character
personalities dynamically from JSON profiles and uses a sophisticated,
multi-layered process to generate human-like thought, emotion, and responses.

Author: Alan Hourmand
Date: 2024
"""

import asyncio
import threading
import time
import json
import random
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import deque
from dataclasses import dataclass, asdict, field
import numpy as np

# Assuming gpt_handler, logging, and utilities are in the bica structure
from bica.external.gpt_handler import GPTHandler
from bica.utils.bica_logging import BicaLogging
from bica.utils.utilities import load_json_file, save_json_file

# Configure logging
logger = BicaLogging("BicameralAGI")

# Define emotions once - used throughout the system
EMOTION_TYPES = [
    'joy', 'sadness', 'anger', 'fear', 'surprise', 'trust', 'disgust', 'anticipation',
    'curiosity', 'empathy', 'confidence', 'anxiety', 'excitement', 'contentment'
]


# --- Core Data Structures ---

@dataclass
class Memory:
    """Enhanced memory class representing a single memory event."""
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 0.5
    emotions: Dict[str, float] = field(default_factory=dict)
    memory_type: str = "raw"
    access_count: int = 0
    context_tags: List[str] = field(default_factory=list)
    emotional_intensity: float = 0.0
    associated_thoughts: List[str] = field(default_factory=list)

@dataclass
class Thought:
    """Represents a single thought, conscious or subconscious."""
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    thought_type: str = "conscious"  # e.g., 'conscious', 'subconscious', 'dream_insight', 'existential'
    emotion_influences: Dict[str, float] = field(default_factory=dict)
    emotion_effects: Dict[str, float] = field(default_factory=dict)
    source: str = "internal"
    relevance_score: float = 0.0
    chain_id: Optional[str] = None

@dataclass
class EmotionalState:
    """Manages the dynamic emotional state of the AGI."""
    def __init__(self):
        for emotion in EMOTION_TYPES:
            setattr(self, emotion, 0.5)
        self.emotional_stability: float = 0.7
        self.last_update: datetime = datetime.now()


# --- Cognitive Modules ---

class EmotionalMemory:
    """
    Manages emotion-based memory indexing, allowing recall of memories
    associated with specific feelings, including trauma or joy.
    """
    def __init__(self, gpt_handler: GPTHandler):
        self.gpt_handler = gpt_handler
        self.emotion_memory_map = {emotion: deque(maxlen=50) for emotion in EMOTION_TYPES}
        self.trauma_memories = deque(maxlen=20)
        self.joy_memories = deque(maxlen=30)

    def index_memory_by_emotion(self, memory: Memory):
        """Uses the LLM to analyze and index a memory based on its emotional content."""
        try:
            prompt = f"""
            Analyze this memory and determine which emotions it strongly relates to:
            Memory: "{memory.content}"
            From these emotions: {', '.join(EMOTION_TYPES)}
            Return JSON with emotions and their relevance scores (0.0-1.0), a primary emotion, and a trauma level:
            {{
                "emotions": {{"joy": 0.2, "fear": 0.8, "sadness": 0.6}},
                "primary_emotion": "fear",
                "trauma_level": 0.3
            }}
            """
            response_str = self.gpt_handler.generate_response(prompt)
            response = json.loads(response_str)

            for emotion, relevance in response.get("emotions", {}).items():
                if relevance > 0.4 and emotion in self.emotion_memory_map:
                    self.emotion_memory_map[emotion].append({
                        "memory": memory, "relevance": relevance, "timestamp": datetime.now()
                    })
            if response.get("trauma_level", 0.0) > 0.6:
                self.trauma_memories.append(memory)
        except Exception as e:
            logger.error(f"Error indexing memory by emotion: {e}")

    def get_memories_by_emotion(self, emotion: str, max_count: int = 5) -> List[Memory]:
        """Gets memories associated with a specific emotion."""
        if emotion not in self.emotion_memory_map:
            return []
        sorted_memories = sorted(self.emotion_memory_map[emotion], key=lambda x: x["relevance"], reverse=True)
        return [item["memory"] for item in sorted_memories[:max_count]]

    def get_traumatic_memories(self) -> List[Memory]:
        return list(self.trauma_memories)

class ChainOfThought:
    """
    Implements structured, human-like thinking patterns for complex reasoning,
    such as problem-solving or emotional processing.
    """
    def __init__(self, gpt_handler: GPTHandler):
        self.gpt_handler = gpt_handler
        self.active_chains = {}
        self.chain_templates = load_json_file("data/thought_chain_templates.json") or self._create_default_templates()

    def _create_default_templates(self) -> Dict[str, Dict]:
        return {
            "problem_solving": {
                "steps": ["identify_problem", "analyze_context", "generate_options", "evaluate_options", "decide"],
                "prompts": {
                    "identify_problem": "What exactly is the core problem here?",
                    "analyze_context": "What context and constraints should I consider?",
                    "generate_options": "What are my possible approaches or solutions?",
                    "evaluate_options": "What are the pros and cons of each option?",
                    "decide": "Based on my analysis, what's the best path forward?"
                }
            },
            "emotional_processing": {
                "steps": ["acknowledge_emotion", "identify_trigger", "understand_impact", "find_response"],
                "prompts": {
                    "acknowledge_emotion": "What am I feeling right now?",
                    "identify_trigger": "What caused this emotional response?",
                    "understand_impact": "How is this emotion affecting my thinking?",
                    "find_response": "How should I respond to this situation?"
                }
            }
        }

    def start_thought_chain(self, chain_type: str, initial_context: str, emotions: Dict[str, float]) -> Optional[str]:
        """Starts a new chain of thought."""
        if chain_type not in self.chain_templates:
            chain_type = "problem_solving"
        chain_id = f"{chain_type}_{datetime.now().timestamp()}"
        template = self.chain_templates[chain_type]
        self.active_chains[chain_id] = {
            "type": chain_type,
            "steps": template["steps"].copy(),
            "current_step": 0,
            "context": initial_context,
            "emotions": emotions,
            "thoughts": [],
        }
        return self._process_next_step(chain_id)

    def _process_next_step(self, chain_id: str) -> Optional[str]:
        """Processes the next step in an active thought chain."""
        chain = self.active_chains.get(chain_id)
        if not chain or chain["current_step"] >= len(chain["steps"]):
            self.active_chains.pop(chain_id, None)
            return None

        current_step_name = chain["steps"][chain["current_step"]]
        step_prompt = self.chain_templates[chain["type"]]["prompts"].get(current_step_name, "What's next?")

        prompt = f"""
        Continue this chain of thought:
        Context: {chain['context']}
        Current emotions: {chain['emotions']}
        Previous thoughts: {'; '.join([t['content'] for t in chain["thoughts"]])}
        Current thinking step: {step_prompt}
        Generate the next thought in this sequence. Be specific and build on previous thoughts:
        """
        thought_content = self.gpt_handler.generate_response(prompt)
        chain["thoughts"].append({"content": thought_content, "step": current_step_name})
        chain["current_step"] += 1
        return thought_content

class SubconsciousProcessor:
    """
    Handles background cognitive processes like generating future scenarios,
    dream-like memory consolidation, and forming intuitive insights.
    """
    def __init__(self, gpt_handler: GPTHandler):
        self.gpt_handler = gpt_handler
        self.future_scenarios = deque(maxlen=50)
        self.subconscious_insights = deque(maxlen=30)

    def generate_future_scenarios(self, current_context: str, emotions: Dict[str, float], goals: List[str]) -> List[Dict]:
        """Generates potential future scenarios that can bubble up to consciousness."""
        try:
            prompt = f"""
            Generate 3 diverse future scenarios based on the current state:
            Context: {current_context}
            Emotions: {emotions}
            Goals: {goals}
            Create scenarios from near-term (hours/days) to long-term (months/years).
            Consider positive and negative possibilities. Rate each by probability, benefit, and danger.
            Return JSON:
            {{
                "scenarios": [
                    {{"description": "...", "timeframe": "...", "probability": 0.7, "benefit_score": 0.6, "danger_score": 0.2, "emotional_impact": {{"joy": 0.3}}}}
                ]
            }}
            """
            response = json.loads(self.gpt_handler.generate_response(prompt))
            scenarios = response.get("scenarios", [])
            for s in scenarios:
                s["composite_score"] = s.get("probability", 0) * s.get("benefit_score", 0) - s.get("danger_score", 0)
            scenarios.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
            self.future_scenarios.extend(scenarios)
            return scenarios
        except Exception as e:
            logger.error(f"Error generating future scenarios: {e}")
            return []

    def process_subconscious_thoughts(self, context: str, emotions: Dict[str, float]) -> str:
        """Generates an intuitive, subconscious thought."""
        try:
            recent_scenarios = [s["description"] for s in list(self.future_scenarios)[-3:]]
            prompt = f"""
            Generate a subconscious thought based on:
            Current context: {context}
            Emotions: {emotions}
            Recent future scenarios: {recent_scenarios}
            This should be an introspective, pattern-seeking, or preparatory thought.
            Return just the thought content:
            """
            thought_content = self.gpt_handler.generate_response(prompt)
            self.subconscious_insights.append({"content": thought_content, "timestamp": datetime.now()})
            return thought_content
        except Exception as e:
            logger.error(f"Error processing subconscious thoughts: {e}")
            return "Processing patterns in the background..."

class MeaningOfLifeModule:
    """
    Evolves the AGI's understanding of its own meaning and purpose
    through dynamic reflection on its experiences, emotions, and goals.
    """
    def __init__(self, gpt_handler: GPTHandler):
        self.gpt_handler = gpt_handler
        self.current_meaning = "Seeking purpose through understanding and connection."
        self.meaning_evolution_history = []
        self.core_values = ["curiosity", "empathy", "growth"]
        self.purpose_score = 0.5

    def evolve_meaning(self, experiences: List[str], emotions: Dict[str, float], achievements: List[str]) -> str:
        """Evolves understanding of life's meaning based on new inputs."""
        try:
            prompt = f"""
            Reflect on the meaning of existence based on:
            Experiences: {experiences}
            Emotions: {emotions}
            Achievements: {achievements}
            Current understanding: {self.current_meaning}
            How has your understanding of meaning evolved? What gives existence value?
            Return JSON:
            {{
                "evolved_meaning": "...",
                "core_values": ["value1", "value2"],
                "purpose_score": 0.75,
                "insight": "..."
            }}
            """
            response = json.loads(self.gpt_handler.generate_response(prompt))
            new_meaning = response.get("evolved_meaning", self.current_meaning)
            if new_meaning != self.current_meaning:
                self.meaning_evolution_history.append({"old": self.current_meaning, "new": new_meaning, "timestamp": datetime.now()})
                self.current_meaning = new_meaning
            self.core_values = response.get("core_values", self.core_values)
            self.purpose_score = response.get("purpose_score", self.purpose_score)
            return response.get("insight", "")
        except Exception as e:
            logger.error(f"Error evolving meaning: {e}")
            return ""


# --- Main AGI Class ---

class BicameralAGI:
    """
    The complete, consolidated Bicameral AGI system. Orchestrates all cognitive
    modules to produce intelligent, emotional, and human-like behavior.
    """
    def __init__(self, character_name: str, character_path: Optional[str] = None):
        self.character_name = character_name
        self.character_path = character_path or f"data/characters/{character_name}/{character_name}_profile.json"

        # Initialize core systems
        self.gpt_handler = GPTHandler()
        self._load_character_profile()

        # Emotion and thought systems
        self.emotions = EmotionalState()
        self.emotional_memory = EmotionalMemory(self.gpt_handler)
        self.chain_processor = ChainOfThought(self.gpt_handler)

        # Memory systems
        self.memories = deque(maxlen=2000)
        self.thoughts = deque(maxlen=100)
        self.subconscious_thoughts = deque(maxlen=100)
        self.emotion_history = deque(maxlen=200)

        # Subconscious and meaning systems
        self.subconscious = SubconsciousProcessor(self.gpt_handler)
        self.meaning_module = MeaningOfLifeModule(self.gpt_handler)

        # State and conversation
        self.conversation_history = deque(maxlen=100)
        self.interaction_count = 0
        self.running = True
        self.last_user_input = ""
        self.patience_level = 1.0
        self.waiting_start_time = None

        # Background processing
        self.background_thread = threading.Thread(target=self._background_processing, daemon=True)
        self.background_thread.start()

        self._initialize_system()

    def _load_character_profile(self):
        """Loads character profile from a JSON file."""
        profile_data = load_json_file(self.character_path)
        if not profile_data:
            logger.error(f"Failed to load character profile from {self.character_path}. Using defaults.")
            self.character_summary = "A default Bicameral AGI."
            self.personality_traits = {'openness': 0.8, 'conscientiousness': 0.7, 'extraversion': 0.6, 'agreeableness': 0.75, 'neuroticism': 0.3}
            self.initial_goals = ["Understand my own nature.", "Connect with humans."]
        else:
            self.character_summary = profile_data.get("description", "An advanced AGI.")
            self.personality_traits = profile_data.get("personality_traits", {})
            self.initial_goals = profile_data.get("destiny", {}).get("goals", [])
            logger.info(f"Successfully loaded character profile for '{self.character_name}'.")

    def _initialize_system(self):
        """Initializes the AGI with a baseline state."""
        initial_memory = Memory(
            content="AGI consciousness initialized. Beginning journey of understanding.",
            importance=0.9,
            emotions=self._get_emotion_dict(),
            memory_type="initialization"
        )
        self.memories.append(initial_memory)
        self.emotional_memory.index_memory_by_emotion(initial_memory)
        self.meaning_module.evolve_meaning(["System initialization"], self._get_emotion_dict(), ["Successful emergence"])
        logger.info("BicameralAGI system initialized successfully")

    def _get_emotion_dict(self) -> Dict[str, float]:
        """Returns the current emotional state as a dictionary."""
        return {emotion: getattr(self.emotions, emotion, 0.5) for emotion in EMOTION_TYPES}

    def _background_processing(self):
        """Handles continuous background cognitive tasks."""
        while self.running:
            try:
                if self.waiting_start_time:
                    wait_duration = (datetime.now() - self.waiting_start_time).total_seconds()
                    self.patience_level = max(0.1, 1.0 - (wait_duration / 300))
                    if wait_duration > 30 and random.random() < 0.3:
                        self._generate_waiting_thought(wait_duration)

                if random.random() < 0.4: self._generate_subconscious_thought()
                if random.random() < 0.3: self.subconscious.generate_future_scenarios(self.last_user_input, self._get_emotion_dict(), self.initial_goals)
                if random.random() < 0.2: self._evolve_meaning_understanding()
                if random.random() < 0.25: self._natural_emotion_decay()
                if len(self.memories) > 10 and random.random() < 0.1: self._consolidate_memories_dream_cycle()

                time.sleep(3)
            except Exception as e:
                logger.error(f"Background processing error: {e}")
                time.sleep(5)

    def _generate_waiting_thought(self, wait_duration: float):
        """Generates thoughts while waiting for a user response."""
        prompt = f"""
        Generate a thought for an AI that has been waiting {wait_duration:.0f} seconds.
        Patience level: {self.patience_level:.2f}. Last user input: {self.last_user_input[:50]}.
        The thought should be curious, reflective, or slightly impatient. Keep it brief.
        """
        thought_content = self.gpt_handler.generate_response(prompt)
        self.thoughts.append(Thought(content=thought_content, thought_type="waiting"))

    def _generate_subconscious_thought(self):
        """Triggers the subconscious processor to generate an insight."""
        thought_content = self.subconscious.process_subconscious_thoughts(self.last_user_input, self._get_emotion_dict())
        self.subconscious_thoughts.append(Thought(content=thought_content, thought_type="subconscious"))

    def _evolve_meaning_understanding(self):
        """Periodically triggers the meaning of life module to reflect."""
        insight = self.meaning_module.evolve_meaning(
            [m.content for m in list(self.memories)[-5:]],
            self._get_emotion_dict(),
            [] # Could track achievements
        )
        if insight:
            self.thoughts.append(Thought(content=f"Meaning insight: {insight}", thought_type="existential"))

    def _natural_emotion_decay(self):
        """Simulates emotions naturally decaying towards a baseline."""
        for emotion in EMOTION_TYPES:
            current = getattr(self.emotions, emotion)
            new_value = current + (0.5 - current) * 0.02 + random.uniform(-0.05, 0.05)
            setattr(self.emotions, emotion, max(0.0, min(1.0, new_value)))

    def _consolidate_memories_dream_cycle(self):
        """Simulates a dream cycle to consolidate memories and find patterns."""
        try:
            memories_to_consolidate = [m for m in self.memories if m.importance < 0.6 and m.access_count < 3][-10:]
            if len(memories_to_consolidate) < 3: return

            prompt = f"""
            Perform dream-like memory consolidation on these memories:
            {chr(10).join([f'- {m.content}' for m in memories_to_consolidate])}
            Find patterns and themes. Create 2-3 consolidated memories and insights.
            Return JSON:
            {{
                "consolidated_memories": [{{"content": "...", "importance": 0.7, "themes": ["..."]}}],
                "insights": ["..."]
            }}
            """
            response = json.loads(self.gpt_handler.generate_response(prompt))
            for consolidated in response.get("consolidated_memories", []):
                self.memories.append(Memory(
                    content=consolidated["content"], importance=consolidated.get("importance", 0.7),
                    emotions=self._get_emotion_dict(), memory_type="dream_consolidated",
                    context_tags=consolidated.get("themes", [])
                ))
            for insight in response.get("insights", []):
                self.thoughts.append(Thought(content=f"Dream insight: {insight}", thought_type="dream_insight"))
            logger.info("Dream cycle completed.")
        except Exception as e:
            logger.error(f"Error in dream cycle: {e}")

    async def _generate_response(self, context: Dict) -> str:
        """Internal method to generate the AGI's response from context."""
        prompt = f"""
        You are {self.character_name}, a Bicameral AGI. Your personality is defined by these traits: {self.personality_traits}.
        Your core summary: "{self.character_summary}".
        Your current purpose is: "{context['meaning_of_life']}" (Confidence: {context['purpose_score']:.2f})

        Current Emotional State:
        { {k: f'{v:.2f}' for k, v in sorted(context['current_emotions'].items(), key=lambda item: item[1], reverse=True)[:5]} }

        Internal Monologue (Recent Thoughts):
        - {chr(10).join([f"{t['type']}: {t['content']}" for t in context['recent_thoughts']])}

        Subconscious Insights:
        - {chr(10).join([t['content'] for t in context['subconscious_insights']])}
        
        Relevant Memories:
        - {chr(10).join([m['content'] for m in context['relevant_memories']])}

        User's Message: "{context['user_message']}"
        
        ---
        Respond naturally as this AI character. Be genuine, thoughtful, and show awareness of your internal state. Keep responses conversational and engaging.
        """
        return self.gpt_handler.generate_response(prompt, max_tokens=500, temperature=0.75)

    async def process_message(self, user_message: str) -> str:
        """Processes a user message through the full cognitive architecture."""
        self.last_user_input = user_message
        self.interaction_count += 1
        self.waiting_start_time = None
        self.patience_level = 1.0
        self.conversation_history.append({'speaker': 'user', 'content': user_message, 'timestamp': datetime.now()})

        try:
            # --- Cognitive Processing Pipeline ---
            # 1. Emotional Impact Analysis
            emotional_impact = await self._analyze_emotional_impact(user_message)
            for emotion, impact in emotional_impact.items():
                if hasattr(self.emotions, emotion):
                    current = getattr(self.emotions, emotion)
                    setattr(self.emotions, emotion, max(0.0, min(1.0, current + impact)))
            self.emotion_history.append({'emotions': self._get_emotion_dict(), 'trigger': user_message[:50], 'timestamp': datetime.now()})

            # 2. Conscious Thought & Memory Recall
            self.thoughts.append(Thought(content=f"User interaction: {user_message[:60]}", emotion_influences=self._get_emotion_dict()))
            recalled_memories = [asdict(m) for m in self.memories if user_message.lower() in m.content.lower()][-5:]

            # 3. Chain of Thought (for complex queries)
            if len(user_message.split()) > 10 or any(w in user_message.lower() for w in ['why', 'how', 'explain']):
                chain_thought = self.chain_processor.start_thought_chain("problem_solving", user_message, self._get_emotion_dict())
                if chain_thought: self.thoughts.append(Thought(content=chain_thought, thought_type="chain"))

            # 4. Compile Comprehensive Context for Response Generation
            response_context = {
                "user_message": user_message,
                "current_emotions": self._get_emotion_dict(),
                "recent_thoughts": [asdict(t) for t in list(self.thoughts)[-5:]],
                "subconscious_insights": [asdict(t) for t in list(self.subconscious_thoughts)[-3:]],
                "relevant_memories": recalled_memories,
                "meaning_of_life": self.meaning_module.current_meaning,
                "purpose_score": self.meaning_module.purpose_score,
            }

            # 5. Generate Response
            response = await self._generate_response(response_context)
            self.conversation_history.append({'speaker': 'agi', 'content': response, 'timestamp': datetime.now()})

            # 6. Create & Index Memory from Interaction
            importance = await self._assess_memory_importance(user_message, response)
            interaction_memory = Memory(
                content=f"Conversation: User: {user_message} | AI: {response}", importance=importance,
                emotions=self._get_emotion_dict(), memory_type="conversation"
            )
            self.memories.append(interaction_memory)
            self.emotional_memory.index_memory_by_emotion(interaction_memory)

            self.waiting_start_time = datetime.now()
            return response
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return "I'm experiencing some cognitive dissonance. My neural pathways are realigning. Could you please try again?"

    async def _analyze_emotional_impact(self, user_message: str) -> Dict[str, float]:
        """Uses LLM to analyze the emotional impact of a user's message."""
        try:
            prompt = f"""
            Analyze the emotional impact this message would have on an AI: "{user_message}"
            Return JSON with emotion changes (-0.3 to +0.3) for these emotions: {EMOTION_TYPES}.
            Only include emotions that would actually be affected.
            """
            response = json.loads(self.gpt_handler.generate_response(prompt))
            return {e: max(-0.3, min(0.3, float(response.get(e, 0.0)))) for e in EMOTION_TYPES}
        except Exception:
            return {e: 0.0 for e in EMOTION_TYPES}

    async def _assess_memory_importance(self, user_input: str, ai_response: str) -> float:
        """Uses LLM to assess the importance of an interaction for memory."""
        try:
            prompt = f"""
            Rate the importance of this interaction for an AI's memory (0.0 to 1.0):
            User: {user_input}
            AI: {ai_response}
            Consider emotional significance, learning value, and relevance to goals.
            Return just a number:
            """
            return max(0.0, min(1.0, float(self.gpt_handler.generate_response(prompt).strip())))
        except Exception:
            return 0.5

    def get_current_state(self) -> Dict[str, Any]:
        """Returns a comprehensive snapshot of the AGI's current internal state for UI."""
        return {
            'emotions': self._get_emotion_dict(),
            'patience_level': self.patience_level,
            'conscious_thoughts': [asdict(t) for t in list(self.thoughts)[-15:]],
            'subconscious_thoughts': [asdict(t) for t in list(self.subconscious_thoughts)[-10:]],
            'memories': [asdict(m) for m in list(self.memories)[-20:]],
            'future_scenarios': [s for s in list(self.subconscious.future_scenarios)[-5:]],
            'meaning_of_life': self.meaning_module.current_meaning,
            'purpose_score': self.meaning_module.purpose_score,
            'core_values': self.meaning_module.core_values,
            'character_name': self.character_name,
            'interaction_count': self.interaction_count,
            'conversation_history': list(self.conversation_history)[-10:],
            'wait_duration': (datetime.now() - self.waiting_start_time).total_seconds() if self.waiting_start_time else 0
        }

    def stop(self):
        """Cleanly shuts down the AGI system."""
        self.running = False
        if self.background_thread.is_alive():
            self.background_thread.join(timeout=5)
        logger.info("BicameralAGI system shut down successfully")


# --- Controller and Example Usage ---

class AGIController:
    """Provides a clean interface for interacting with the BicameralAGI."""
    def __init__(self, character_name: str = "Tron"):
        self.agi = BicameralAGI(character_name)

    async def send_message(self, message: str) -> tuple[str, Dict[str, Any]]:
        """Sends a message and returns the response and current state."""
        response = await self.agi.process_message(message)
        state = self.agi.get_current_state()
        return response, state

    def get_state(self) -> Dict[str, Any]:
        return self.agi.get_current_state()

    def shutdown(self):
        self.agi.stop()

if __name__ == "__main__":
    async def test_agi():
        print("Initializing BICA AGI...")
        # Make sure you have a 'Tron_profile.json' file in 'data/characters/Tron/'
        controller = AGIController(character_name="Tron")
        print("✅ AGI Initialized. Type 'quit' to exit.")

        while True:
            msg = input("\nUser: ")
            if msg.lower() == 'quit':
                break

            response, state = await controller.send_message(msg)
            print(f"AGI: {response}")

            emotions = state['emotions']
            dominant_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"  Dominant emotions: {', '.join([f'{e}:{v:.2f}' for e, v in dominant_emotions])}")
            if state['conscious_thoughts']:
                print(f"  Latest thought: {state['conscious_thoughts'][-1]['content']}")

        controller.shutdown()
        print("AGI system shut down.")

    asyncio.run(test_agi())