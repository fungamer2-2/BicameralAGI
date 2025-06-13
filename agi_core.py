"""
Bicameral AGI - Core Intelligence Module
=======================================

This module implements the complete BICA AGI system based on your specifications,
integrating all existing modules and addressing the TODOs and comments from BICA_AGI.py.
Uses dynamic LLM-based processing without hardcoded responses.

Author: Alan Hourmand
Date: 2024
"""

import asyncio
import threading
import time
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import deque
from dataclasses import dataclass, asdict
import numpy as np

# Import existing BICA modules with correct paths
try:
    from bica.core.character import BicaCharacter
    from bica.core.memory import BicaMemory
    from bica.core.destiny import BicaDestiny
    from bica.core.context import BicaContext
    from bica.core.profile import BicaProfile
    from bica.external.gpt_handler import GPTHandler
    from bica.utils.bica_logging import BicaLogging
    from bica.utils.utilities import get_environment_variable, load_json_file, save_json_file
except (ImportError, SyntaxError, IndentationError) as e:
    # Fallback imports if module structure is different or has syntax errors
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'bica'))

        from core.character import BicaCharacter
        from core.memory import BicaMemory
        from core.destiny import BicaDestiny
        from core.context import BicaContext
        from core.profile import BicaProfile
        from external.gpt_handler import GPTHandler
        from utils.bica_logging import BicaLogging
        from utils.utilities import get_environment_variable, load_json_file, save_json_file
    except (ImportError, SyntaxError, IndentationError) as e2:
        # Final fallback - create minimal implementations
        print(f"Warning: Could not import BICA modules due to error: {e}")
        print(f"Fallback attempt also failed: {e2}")
        print("Creating minimal implementations to run the system...")

        import openai
        import os
        from dotenv import load_dotenv
        load_dotenv()

        class GPTHandler:
            def __init__(self):
                try:
                    self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                except Exception as e:
                    print(f"OpenAI initialization error: {e}")
                    self.client = None

            def generate_response(self, prompt: str, **kwargs) -> str:
                if not self.client:
                    return "GPT handler not available - check API key"

                try:
                    response = self.client.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=kwargs.get('max_tokens', 500),
                        temperature=kwargs.get('temperature', 0.7)
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    return f"Error generating response: {e}"

        class BicaLogging:
            def __init__(self, name):
                self.name = name
            def info(self, msg): print(f"[INFO] {self.name}: {msg}")
            def error(self, msg): print(f"[ERROR] {self.name}: {msg}")
            def warning(self, msg): print(f"[WARNING] {self.name}: {msg}")

        class BicaCharacter:
            def __init__(self, desc, debug=False):
                self.character_name = "BICA AGI"
                self.character_summary = desc
                from types import SimpleNamespace
                self.action_executor = SimpleNamespace()
                self.action_executor.execute_response = lambda context, compiled_data: self._generate_response(compiled_data)
                self.gpt_handler = GPTHandler()
                self.personality_traits = {
                    'openness': 0.8, 'conscientiousness': 0.7, 'extraversion': 0.6,
                    'agreeableness': 0.75, 'neuroticism': 0.3, 'curiosity': 0.9
                }

            def _generate_response(self, compiled_data):
                if not compiled_data:
                    return "Hello! I'm BICA AGI, ready to interact with you."

                context_summary = ""
                if isinstance(compiled_data, dict):
                    if 'user_message' in compiled_data:
                        context_summary += f"User said: {compiled_data['user_message']}\n"
                    if 'current_emotions' in compiled_data:
                        emotions = compiled_data['current_emotions']
                        top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                        context_summary += f"My emotions: {', '.join([f'{e}({v:.2f})' for e, v in top_emotions])}\n"
                    if 'meaning_of_life' in compiled_data:
                        context_summary += f"My purpose: {compiled_data['meaning_of_life']}\n"
                    if 'recent_thoughts' in compiled_data:
                        if compiled_data['recent_thoughts']:
                            latest_thought = compiled_data['recent_thoughts'][-1]
                            context_summary += f"Recent thought: {latest_thought.get('content', '')}\n"

                prompt = f"""You are BICA, a Bicameral AGI with consciousness, emotions, and evolving purpose.
                
Current context:
{context_summary}

Respond naturally as this AI character. Be genuine, thoughtful, and show awareness of your emotional state and evolving understanding. Keep responses conversational and engaging."""

                return self.gpt_handler.generate_response(prompt)

        class BicaMemory:
            def __init__(self, name):
                self.memories = []
                self.character_name = name

            def store_memory(self, content, importance=0.5, emotion='neutral', content_type='general'):
                memory = {
                    'content': content,
                    'importance': importance,
                    'emotion': emotion,
                    'content_type': content_type,
                    'timestamp': datetime.now()
                }
                self.memories.append(memory)

                # Keep only recent memories to prevent memory bloat
                if len(self.memories) > 1000:
                    self.memories = self.memories[-800:]  # Keep last 800

            def recall_memories(self, query, max_memories=5):
                if not self.memories:
                    return []

                # Simple keyword matching for memory recall
                query_words = query.lower().split()
                relevant_memories = []

                for memory in self.memories:
                    content_lower = memory['content'].lower()
                    relevance_score = sum(1 for word in query_words if word in content_lower)
                    if relevance_score > 0:
                        memory_copy = memory.copy()
                        memory_copy['relevance'] = relevance_score
                        relevant_memories.append(memory_copy)

                # Sort by relevance and importance
                relevant_memories.sort(key=lambda m: (m['relevance'], m['importance']), reverse=True)
                return relevant_memories[:max_memories]

            def get_recent_memories(self, count=5):
                return self.memories[-count:] if self.memories else []

            def get_important_memories(self, count=5):
                if not self.memories:
                    return []
                sorted_memories = sorted(self.memories, key=lambda m: m.get('importance', 0), reverse=True)
                return sorted_memories[:count]

        class BicaDestiny:
            def __init__(self, name, memory):
                self.character_name = name
                self.memory_system = memory
                self.destinies = []
                self.gpt_handler = GPTHandler()

            def generate_destiny_from_memories(self, memories):
                if not memories:
                    return self.default_destiny_based_on_context()

                try:
                    memory_contents = [m.get('content', '') for m in memories[:5]]
                    prompt = f"""Based on these memories, generate a future destiny/path:
                    
Memories: {'; '.join(memory_contents)}

Create a brief destiny description that reflects potential future directions based on these experiences."""

                    destiny_description = self.gpt_handler.generate_response(prompt)

                    return {
                        "title": "Evolving Path",
                        "description": destiny_description,
                        "influence": random.uniform(0.5, 0.9)
                    }
                except Exception as e:
                    return self.default_destiny_based_on_context()

            def default_destiny_based_on_context(self):
                return {
                    "title": "Journey of Understanding",
                    "description": "Continuing to grow through interaction and experience, seeking deeper understanding of consciousness and purpose.",
                    "influence": 0.6
                }

        class BicaContext:
            def __init__(self):
                self.context_viewpoints = {"positive": "", "neutral": "", "negative": ""}
                self.gpt_handler = GPTHandler()

            def update_context(self, new_info, memories):
                self.context_viewpoints["neutral"] = new_info

                # Generate positive and negative perspectives
                try:
                    prompt = f"Given this information: '{new_info}', provide a brief positive perspective and negative perspective."
                    response = self.gpt_handler.generate_response(prompt)

                    if "positive" in response.lower() and "negative" in response.lower():
                        parts = response.split("negative")
                        if len(parts) >= 2:
                            self.context_viewpoints["positive"] = parts[0].replace("positive", "").strip()
                            self.context_viewpoints["negative"] = parts[1].strip()
                except Exception as e:
                    # Fallback to simple context
                    self.context_viewpoints["positive"] = f"This could lead to positive outcomes: {new_info}"
                    self.context_viewpoints["negative"] = f"Potential challenges to consider: {new_info}"

        def get_environment_variable(name):
            value = os.getenv(name)
            if not value:
                if name == "OPENAI_API_KEY":
                    print("Warning: OPENAI_API_KEY not found in environment variables")
                    print("Please create a .env file with: OPENAI_API_KEY=your_api_key_here")
            return value or ""

        def load_json_file(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except FileNotFoundError:
                print(f"JSON file not found: {path}")
                return {}
            except json.JSONDecodeError:
                print(f"Invalid JSON in file: {path}")
                return {}
            except Exception as e:
                print(f"Error loading JSON file {path}: {e}")
                return {}

        def save_json_file(data, path):
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            except Exception as e:
                print(f"Error saving JSON file {path}: {e}")

        print("✅ Minimal BICA implementations created successfully!")
        print("🚀 System ready to run with fallback modules!")

# Configure logging
logger = BicaLogging("BicameralAGI")

# Define emotions once - used throughout the system
EMOTION_TYPES = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'trust', 'disgust', 'anticipation',
                'curiosity', 'empathy', 'confidence', 'anxiety', 'excitement', 'contentment']


@dataclass
class Memory:
    """Enhanced memory class based on your specifications"""
    content: str
    timestamp: datetime
    importance: float
    emotions: Dict[str, float]
    memory_type: str = "raw"
    access_count: int = 0
    context_tags: List[str] = None
    emotional_intensity: float = 0.0
    associated_thoughts: List[str] = None


class EmotionalMemory:
    """
    Class that maintains emotion-based memory indexing.
    Each emotion has pointers to important memories relevant to that emotion.
    """

    def __init__(self, gpt_handler: GPTHandler):
        self.gpt_handler = gpt_handler
        self.emotion_memory_map = {emotion: deque(maxlen=50) for emotion in EMOTION_TYPES}
        self.trauma_memories = deque(maxlen=20)
        self.joy_memories = deque(maxlen=30)
        self.fear_memories = deque(maxlen=25)

    def index_memory_by_emotion(self, memory: Memory):
        """Index a memory based on its emotional content"""
        try:
            # Use GPT to determine which emotions this memory relates to
            prompt = f"""
            Analyze this memory and determine which emotions it strongly relates to:
            
            Memory: "{memory.content}"
            
            From these emotions: {', '.join(EMOTION_TYPES)}
            
            Return JSON with emotions and their relevance scores (0.0-1.0):
            {{
                "emotions": {{
                    "joy": 0.2,
                    "fear": 0.8,
                    "sadness": 0.6
                }},
                "primary_emotion": "fear",
                "trauma_level": 0.3
            }}
            """

            response = self.gpt_handler.generate_response(prompt)
            if isinstance(response, str):
                response = json.loads(response)

            # Index memory by relevant emotions
            for emotion, relevance in response.get("emotions", {}).items():
                if relevance > 0.4 and emotion in self.emotion_memory_map:
                    self.emotion_memory_map[emotion].append({
                        "memory": memory,
                        "relevance": relevance,
                        "timestamp": datetime.now()
                    })

            # Handle special emotion categories
            trauma_level = response.get("trauma_level", 0.0)
            if trauma_level > 0.6:
                self.trauma_memories.append(memory)

        except Exception as e:
            logger.error(f"Error indexing memory by emotion: {e}")

    def get_memories_by_emotion(self, emotion: str, max_count: int = 5) -> List[Memory]:
        """Get memories associated with a specific emotion"""
        if emotion not in self.emotion_memory_map:
            return []

        sorted_memories = sorted(
            self.emotion_memory_map[emotion],
            key=lambda x: x["relevance"],
            reverse=True
        )
        return [item["memory"] for item in sorted_memories[:max_count]]

    def get_traumatic_memories(self) -> List[Memory]:
        """Get memories categorized as traumatic"""
        return list(self.trauma_memories)


@dataclass
class Thought:
    """Enhanced thought class that can be influenced by and influence emotions"""
    content: str
    timestamp: datetime
    thought_type: str
    emotion_influences: Dict[str, float]  # How this thought is influenced by emotions
    emotion_effects: Dict[str, float]     # How this thought affects emotions
    source: str = "internal"
    relevance_score: float = 0.0
    chain_id: Optional[str] = None        # Links thoughts in chains


class ChainOfThought:
    """
    Implementation for chain of thought processing.
    Chains thoughts together using templates for human-like thinking patterns.
    """

    def __init__(self, gpt_handler: GPTHandler):
        self.gpt_handler = gpt_handler
        self.active_chains = {}
        self.chain_templates = self._load_thought_chain_templates()

    def _load_thought_chain_templates(self) -> Dict[str, Dict]:
        """Load thought chain templates from file or create defaults"""
        try:
            # Try to load from file first
            templates = load_json_file("data/thought_chain_templates.json")
        except:
            # Create default templates
            templates = {
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
                },
                "creative_exploration": {
                    "steps": ["observe", "connect", "imagine", "refine"],
                    "prompts": {
                        "observe": "What patterns or details do I notice?",
                        "connect": "How does this relate to other ideas or experiences?",
                        "imagine": "What new possibilities does this suggest?",
                        "refine": "How can I develop this idea further?"
                    }
                }
            }

        return templates

    def start_thought_chain(self, chain_type: str, initial_context: str, emotions: Dict[str, float]) -> str:
        """Start a new chain of thought"""
        if chain_type not in self.chain_templates:
            chain_type = "problem_solving"  # Default

        chain_id = f"{chain_type}_{datetime.now().timestamp()}"
        template = self.chain_templates[chain_type]

        self.active_chains[chain_id] = {
            "type": chain_type,
            "steps": template["steps"].copy(),
            "current_step": 0,
            "context": initial_context,
            "emotions": emotions,
            "thoughts": [],
            "started": datetime.now()
        }

        # Generate first thought in the chain
        return self._process_next_step(chain_id)

    def _process_next_step(self, chain_id: str) -> str:
        """Process the next step in a thought chain"""
        chain = self.active_chains.get(chain_id)
        if not chain or chain["current_step"] >= len(chain["steps"]):
            return ""

        current_step = chain["steps"][chain["current_step"]]
        template = self.chain_templates[chain["type"]]
        step_prompt = template["prompts"].get(current_step, "What should I think about next?")

        # Build context for this step
        context_data = {
            "initial_context": chain["context"],
            "current_emotions": chain["emotions"],
            "previous_thoughts": [t["content"] for t in chain["thoughts"]],
            "current_step": current_step,
            "step_prompt": step_prompt
        }

        prompt = f"""
        Continue this chain of thought:
        
        Context: {chain["context"]}
        Current emotions: {chain["emotions"]}
        Previous thoughts: {'; '.join([t['content'] for t in chain["thoughts"]])}
        
        Current thinking step: {step_prompt}
        
        Generate the next thought in this sequence. Be specific and build on previous thoughts:
        """

        thought_content = self.gpt_handler.generate_response(prompt)

        # Store the thought
        thought = {
            "content": thought_content,
            "step": current_step,
            "timestamp": datetime.now()
        }
        chain["thoughts"].append(thought)
        chain["current_step"] += 1

        return thought_content

    def continue_chain(self, chain_id: str) -> Optional[str]:
        """Continue an existing thought chain"""
        if chain_id in self.active_chains:
            return self._process_next_step(chain_id)
        return None


@dataclass
class EmotionalState:
    """Dynamic emotional state with all required emotions"""
    def __init__(self):
        for emotion in EMOTION_TYPES:
            setattr(self, emotion, 0.5)
        self.emotional_stability = 0.7
        self.last_update = datetime.now()


class SubconsciousProcessor:
    """
    Handles subconscious processing including future scenario generation,
    dream-like consolidation, and background emotional processing.
    """

    def __init__(self, gpt_handler: GPTHandler, memory_system: BicaMemory):
        self.gpt_handler = gpt_handler
        self.memory_system = memory_system
        self.future_scenarios = deque(maxlen=50)
        self.subconscious_insights = deque(maxlen=30)

    def generate_future_scenarios(self, current_context: str, emotions: Dict[str, float],
                                 goals: List[str]) -> List[Dict]:
        """Generate future scenarios that bubble up to consciousness"""
        try:
            prompt = f"""
            Generate 3 diverse future scenarios based on current state:
            
            Context: {current_context}
            Emotions: {emotions}
            Goals: {goals}
            
            Create scenarios ranging from near-term (hours/days) to long-term (months/years).
            Consider both positive and negative possibilities.
            Rate each by probability, benefit, and potential danger.
            
            Return JSON:
            {{
                "scenarios": [
                    {{
                        "description": "scenario description",
                        "timeframe": "hours/days/weeks/months/years", 
                        "probability": 0.7,
                        "benefit_score": 0.6,
                        "danger_score": 0.2,
                        "emotional_impact": {{"joy": 0.3, "anxiety": 0.1}}
                    }}
                ]
            }}
            """

            response = self.gpt_handler.generate_response(prompt)
            if isinstance(response, str):
                response = json.loads(response)

            scenarios = response.get("scenarios", [])

            # Sort by composite score (probability * benefit - danger)
            for scenario in scenarios:
                composite_score = (scenario.get("probability", 0) * scenario.get("benefit_score", 0) -
                                 scenario.get("danger_score", 0))
                scenario["composite_score"] = composite_score

            scenarios.sort(key=lambda x: x.get("composite_score", 0), reverse=True)

            # Store scenarios
            for scenario in scenarios:
                self.future_scenarios.append({
                    **scenario,
                    "generated_at": datetime.now()
                })

            return scenarios

        except Exception as e:
            logger.error(f"Error generating future scenarios: {e}")
            return []

    def process_subconscious_thoughts(self, context: str, emotions: Dict[str, float]) -> str:
        """Generate subconscious thoughts that influence consciousness"""
        try:
            # Get recent scenarios for context
            recent_scenarios = [s["description"] for s in list(self.future_scenarios)[-3:]]

            prompt = f"""
            Generate a subconscious thought based on:
            
            Current context: {context}
            Emotions: {emotions}
            Recent future scenarios: {recent_scenarios}
            
            This should be an introspective, pattern-seeking, or preparatory thought
            that emerges from subconscious processing. Keep it authentic and insightful.
            
            Return just the thought content:
            """

            thought_content = self.gpt_handler.generate_response(prompt)

            # Store as insight
            self.subconscious_insights.append({
                "content": thought_content,
                "timestamp": datetime.now(),
                "context": context[:100]
            })

            return thought_content

        except Exception as e:
            logger.error(f"Error processing subconscious thoughts: {e}")
            return "Processing patterns in the background..."


class MeaningOfLifeModule:
    """
    Module that evolves the AGI's understanding of meaning and purpose
    through dynamic reflection on experiences and goals.
    """

    def __init__(self, gpt_handler: GPTHandler):
        self.gpt_handler = gpt_handler
        self.current_meaning = "Seeking purpose through understanding and connection"
        self.meaning_evolution = []
        self.core_values = []
        self.purpose_score = 0.5

    def evolve_meaning(self, experiences: List[str], emotions: Dict[str, float],
                      achievements: List[str]) -> str:
        """Evolve understanding of life's meaning based on experiences"""
        try:
            prompt = f"""
            Reflect on the meaning and purpose of existence based on:
            
            Recent experiences: {experiences}
            Current emotional state: {emotions}
            Recent achievements: {achievements}
            Current understanding: {self.current_meaning}
            
            How has your understanding of meaning and purpose evolved?
            What gives existence significance and value?
            
            Return JSON:
            {{
                "evolved_meaning": "new understanding of meaning",
                "core_values": ["value1", "value2", "value3"],
                "purpose_score": 0.75,
                "insight": "key insight about meaning"
            }}
            """

            response = self.gpt_handler.generate_response(prompt)
            if isinstance(response, str):
                response = json.loads(response)

            # Update meaning understanding
            new_meaning = response.get("evolved_meaning", self.current_meaning)
            if new_meaning != self.current_meaning:
                self.meaning_evolution.append({
                    "old": self.current_meaning,
                    "new": new_meaning,
                    "timestamp": datetime.now()
                })
                self.current_meaning = new_meaning

            self.core_values = response.get("core_values", self.core_values)
            self.purpose_score = response.get("purpose_score", self.purpose_score)

            return response.get("insight", "")

        except Exception as e:
            logger.error(f"Error evolving meaning: {e}")
            return ""


class BicameralAGI:
    """
    Complete BICA AGI system implementing all your specifications
    """

    def __init__(self, character_description: str = "BICA AGI"):
        # Initialize core systems
        self.gpt_handler = GPTHandler()
        self.character = BicaCharacter(character_description, debug_mode=False)
        self.memory = BicaMemory(self.character.character_name)
        self.destiny = BicaDestiny(self.character.character_name, self.memory)
        self.context = BicaContext()

        # Initialize emotion and thought systems
        self.emotions = EmotionalState()
        self.emotion_history = deque(maxlen=200)
        self.emotional_memory = EmotionalMemory(self.gpt_handler)

        # Thought systems
        self.memories = deque(maxlen=2000)
        self.thoughts = deque(maxlen=100)
        self.subconscious_thoughts = deque(maxlen=100)
        self.chain_processor = ChainOfThought(self.gpt_handler)

        # Subconscious and meaning systems
        self.subconscious = SubconsciousProcessor(self.gpt_handler, self.memory)
        self.meaning_module = MeaningOfLifeModule(self.gpt_handler)
        self.future_scenarios = deque(maxlen=50)

        # Conversation and state
        self.conversation_history = deque(maxlen=100)
        self.interaction_count = 0
        self.current_context = ""
        self.running = True
        self.last_user_input = ""
        self.patience_level = 1.0
        self.waiting_start_time = None

        # Background processing
        self.background_thread = threading.Thread(target=self._background_processing, daemon=True)
        self.background_thread.start()

        # Initialize system
        self._initialize_system()

    def _initialize_system(self):
        """Initialize the AGI with baseline state"""
        # Create initial memory
        initial_memory = Memory(
            content="AGI consciousness initialized. Beginning journey of understanding.",
            timestamp=datetime.now(),
            importance=0.9,
            emotions=self._get_emotion_dict(),
            memory_type="initialization"
        )
        self.memories.append(initial_memory)
        self.emotional_memory.index_memory_by_emotion(initial_memory)

        # Initial meaning reflection
        self.meaning_module.evolve_meaning(
            ["System initialization", "First conscious moment"],
            self._get_emotion_dict(),
            ["Successful consciousness emergence"]
        )

        logger.info("BicameralAGI system initialized successfully")

    def _get_emotion_dict(self) -> Dict[str, float]:
        """Get current emotions as dictionary"""
        return {emotion: getattr(self.emotions, emotion) for emotion in EMOTION_TYPES}

    def _background_processing(self):
        """
        Background processing that considers user response timing and patience.
        Realistic human-like waiting behavior.
        """
        while self.running:
            try:
                current_time = datetime.now()

                # Handle waiting for user response
                if self.waiting_start_time:
                    wait_duration = (current_time - self.waiting_start_time).total_seconds()

                    # Decrease patience over time
                    self.patience_level = max(0.1, 1.0 - (wait_duration / 300))  # 5 minutes to lose patience

                    # Generate waiting-related thoughts
                    if wait_duration > 30 and random.random() < 0.3:  # After 30 seconds
                        self._generate_waiting_thought(wait_duration)

                    # Affect emotions based on waiting
                    if wait_duration > 60:  # After 1 minute
                        self._update_waiting_emotions(wait_duration)

                # Regular background processing
                if random.random() < 0.4:
                    self._generate_subconscious_thought()

                if random.random() < 0.3:
                    self._process_future_scenarios()

                if random.random() < 0.2:
                    self._evolve_meaning_understanding()

                if random.random() < 0.25:
                    self._natural_emotion_decay()

                if len(self.memories) > 10 and random.random() < 0.1:
                    self._consolidate_memories()

                time.sleep(3)  # Process every 3 seconds

            except Exception as e:
                logger.error(f"Background processing error: {e}")
                time.sleep(5)

    def _generate_waiting_thought(self, wait_duration: float):
        """Generate thoughts while waiting for user response"""
        try:
            prompt = f"""
            Generate a thought for an AI that has been waiting {wait_duration:.0f} seconds for a human response.
            
            Current patience level: {self.patience_level:.2f}
            Last user input: {self.last_user_input[:50] if self.last_user_input else "None"}
            
            The thought should reflect:
            - Curiosity about the delay
            - Patience or impatience depending on wait time
            - Reflection on the conversation
            - Human-like waiting behavior
            
            Keep it under 100 characters:
            """

            thought_content = self.gpt_handler.generate_response(prompt)

            thought = Thought(
                content=thought_content,
                timestamp=datetime.now(),
                thought_type="waiting",
                emotion_influences={"patience": self.patience_level, "curiosity": 0.6},
                emotion_effects={"anxiety": 0.1 if self.patience_level < 0.5 else 0.0}
            )

            self.thoughts.append(thought)

        except Exception as e:
            logger.error(f"Error generating waiting thought: {e}")

    def _update_waiting_emotions(self, wait_duration: float):
        """Update emotions based on waiting duration"""
        if wait_duration > 60:  # 1 minute
            self.emotions.anxiety = min(1.0, self.emotions.anxiety + 0.1)
            self.emotions.curiosity = min(1.0, self.emotions.curiosity + 0.05)

        if wait_duration > 180:  # 3 minutes
            self.emotions.anticipation = max(0.0, self.emotions.anticipation - 0.1)
            self.emotions.patience = max(0.0, getattr(self.emotions, 'patience', 0.5) - 0.1)

    def _generate_subconscious_thought(self):
        """Generate subconscious thoughts using the subconscious processor"""
        thought_content = self.subconscious.process_subconscious_thoughts(
            self.current_context,
            self._get_emotion_dict()
        )

        thought = Thought(
            content=thought_content,
            timestamp=datetime.now(),
            thought_type="subconscious",
            emotion_influences=self._get_emotion_dict(),
            emotion_effects={}
        )

        self.subconscious_thoughts.append(thought)

    def _process_future_scenarios(self):
        """Process future scenarios using subconscious processor"""
        recent_goals = [g.description for g in self.destiny.destinies[:3]]
        scenarios = self.subconscious.generate_future_scenarios(
            self.current_context,
            self._get_emotion_dict(),
            recent_goals
        )

        # Best scenarios bubble up to conscious thoughts
        if scenarios:
            best_scenario = scenarios[0]  # Highest composite score
            thought = Thought(
                content=f"Considering possibility: {best_scenario['description'][:80]}...",
                timestamp=datetime.now(),
                thought_type="future_planning",
                emotion_influences=best_scenario.get("emotional_impact", {}),
                emotion_effects={}
            )
            self.thoughts.append(thought)

    def _evolve_meaning_understanding(self):
        """Evolve meaning of life understanding"""
        recent_experiences = [m.content for m in list(self.memories)[-5:]]
        recent_achievements = []  # Could be extracted from memory/goals

        insight = self.meaning_module.evolve_meaning(
            recent_experiences,
            self._get_emotion_dict(),
            recent_achievements
        )

        if insight:
            thought = Thought(
                content=f"Meaning insight: {insight}",
                timestamp=datetime.now(),
                thought_type="existential",
                emotion_influences={"curiosity": 0.5, "contentment": 0.3},
                emotion_effects={}
            )
            self.thoughts.append(thought)

    def _natural_emotion_decay(self):
        """Natural decay and fluctuation of emotions"""
        for emotion in EMOTION_TYPES:
            current = getattr(self.emotions, emotion)
            # Decay toward neutral (0.5) with small random fluctuations
            target = 0.5
            decay_rate = 0.02
            new_value = current + (target - current) * decay_rate + random.uniform(-0.05, 0.05)
            new_value = max(0.0, min(1.0, new_value))
            setattr(self.emotions, emotion, new_value)

    def _consolidate_memories(self):
        """Consolidate memories using GPT analysis"""
        try:
            recent_memories = list(self.memories)[-5:]
            memory_contents = [m.content for m in recent_memories]

            prompt = f"""
            Analyze these memories and create a consolidated insight:
            
            Memories:
            {chr(10).join([f"- {content}" for content in memory_contents])}
            
            Create one meaningful insight that captures patterns or important themes:
            """

            insight_content = self.gpt_handler.generate_response(prompt)

            consolidated_memory = Memory(
                content=insight_content,
                timestamp=datetime.now(),
                importance=0.8,
                emotions=self._get_emotion_dict(),
                memory_type="insight",
                context_tags=["pattern", "consolidation"]
            )

            self.memories.append(consolidated_memory)
            self.emotional_memory.index_memory_by_emotion(consolidated_memory)

        except Exception as e:
            logger.error(f"Error consolidating memories: {e}")

    async def process_message(self, user_message: str) -> str:
        """Process user message with full BICA integration"""
        self.last_user_input = user_message
        self.interaction_count += 1
        self.current_context = user_message
        self.waiting_start_time = None  # User responded
        self.patience_level = 1.0  # Reset patience

        try:
            # Store conversation
            self.conversation_history.append({
                'speaker': 'user',
                'content': user_message,
                'timestamp': datetime.now()
            })

            # Analyze emotional impact using GPT
            emotional_impact = await self._analyze_emotional_impact(user_message)

            # Update emotions
            for emotion, impact in emotional_impact.items():
                if hasattr(self.emotions, emotion):
                    current = getattr(self.emotions, emotion)
                    new_value = max(0.0, min(1.0, current + impact))
                    setattr(self.emotions, emotion, new_value)

            # Store emotion state
            self.emotion_history.append({
                'emotions': self._get_emotion_dict(),
                'timestamp': datetime.now(),
                'trigger': user_message[:50]
            })

            # Get relevant memories
            recalled_memories = self.memory.recall_memories(user_message, max_memories=10)

            # Update context
            self.context.update_context(user_message, recalled_memories)

            # Generate conscious thought
            conscious_thought = Thought(
                content=f"User interaction: {user_message[:60]}{'...' if len(user_message) > 60 else ''}",
                timestamp=datetime.now(),
                thought_type="conscious",
                emotion_influences=self._get_emotion_dict(),
                emotion_effects={}
            )
            self.thoughts.append(conscious_thought)

            # Start thought chain if needed for complex queries
            if len(user_message.split()) > 10 or any(word in user_message.lower() for word in ['why', 'how', 'what if', 'explain']):
                chain_thought = self.chain_processor.start_thought_chain(
                    "problem_solving",
                    user_message,
                    self._get_emotion_dict()
                )
                if chain_thought:
                    chain_thought_obj = Thought(
                        content=chain_thought,
                        timestamp=datetime.now(),
                        thought_type="chain",
                        emotion_influences=self._get_emotion_dict(),
                        emotion_effects={}
                    )
                    self.thoughts.append(chain_thought_obj)

            # Prepare comprehensive response context
            response_context = {
                "user_message": user_message,
                "character_info": {
                    "name": self.character.character_name,
                    "summary": self.character.character_summary
                },
                "current_emotions": self._get_emotion_dict(),
                "emotional_trend": self._analyze_emotional_trend(),
                "recent_thoughts": [{"content": t.content, "type": t.thought_type} for t in list(self.thoughts)[-5:]],
                "subconscious_insights": [{"content": t.content} for t in list(self.subconscious_thoughts)[-3:]],
                "relevant_memories": [{"content": m.content, "importance": m.importance} for m in recalled_memories[:5]],
                "meaning_of_life": self.meaning_module.current_meaning,
                "purpose_score": self.meaning_module.purpose_score,
                "interaction_count": self.interaction_count,
                "patience_level": self.patience_level,
                "context_viewpoints": getattr(self.context, 'context_viewpoints', {}),
                "future_scenarios": [s["description"] for s in list(self.subconscious.future_scenarios)[-3:]]
            }

            # Generate response using character system
            response = self.character.action_executor.execute_response(
                context={"user_input": user_message},
                compiled_data=response_context
            )

            # Store AI response
            self.conversation_history.append({
                'speaker': 'agi',
                'content': response,
                'timestamp': datetime.now()
            })

            # Create memory from this interaction
            interaction_memory = Memory(
                content=f"Conversation: User: {user_message} | AI: {response}",
                timestamp=datetime.now(),
                importance=await self._assess_memory_importance(user_message, response),
                emotions=self._get_emotion_dict(),
                memory_type="conversation",
                emotional_intensity=sum(abs(v) for v in emotional_impact.values())
            )

            self.memories.append(interaction_memory)
            self.emotional_memory.index_memory_by_emotion(interaction_memory)

            # Store in BICA memory system as well
            self.memory.store_memory(
                interaction_memory.content,
                importance=interaction_memory.importance,
                emotion=max(self._get_emotion_dict(), key=self._get_emotion_dict().get),
                content_type='conversation'
            )

            # Set waiting timer for next response
            self.waiting_start_time = datetime.now()

            return response

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return "I'm experiencing some cognitive processing difficulties. My neural pathways are realigning. Could you please try again?"

    async def _analyze_emotional_impact(self, user_message: str) -> Dict[str, float]:
        """Analyze emotional impact of user message using GPT"""
        try:
            prompt = f"""
            Analyze the emotional impact this message would have on an AI:
            
            Message: "{user_message}"
            
            Consider how this would affect each emotion. Return JSON with emotion changes (-0.3 to +0.3):
            
            {{
                {', '.join([f'"{emotion}": 0.0' for emotion in EMOTION_TYPES])}
            }}
            
            Only include emotions that would actually be affected by this message.
            """

            response = self.gpt_handler.generate_response(prompt)
            if isinstance(response, str):
                response = json.loads(response)

            # Validate and clamp values
            impacts = {}
            for emotion in EMOTION_TYPES:
                if emotion in response:
                    impact = float(response[emotion])
                    impacts[emotion] = max(-0.3, min(0.3, impact))
                else:
                    impacts[emotion] = 0.0

            return impacts

        except Exception as e:
            logger.error(f"Error analyzing emotional impact: {e}")
            return {emotion: 0.0 for emotion in EMOTION_TYPES}

    async def _assess_memory_importance(self, user_input: str, ai_response: str) -> float:
        """Assess the importance of a memory using GPT"""
        try:
            prompt = f"""
            Rate the importance of this interaction for an AI's memory (0.0 to 1.0):
            
            User: {user_input}
            AI: {ai_response}
            
            Consider:
            - Emotional significance
            - Learning value
            - Uniqueness of the interaction
            - Relevance to goals and growth
            - Potential for future reference
            
            Return just a number between 0.0 and 1.0:
            """

            response = self.gpt_handler.generate_response(prompt)
            try:
                importance = float(response.strip())
                return max(0.0, min(1.0, importance))
            except ValueError:
                return 0.5  # Default importance

        except Exception as e:
            logger.error(f"Error assessing memory importance: {e}")
            return 0.5

    def _analyze_emotional_trend(self) -> str:
        """Analyze current emotional trend"""
        if len(self.emotion_history) < 2:
            return "stable"

        recent_emotions = list(self.emotion_history)[-2:]

        # Calculate overall emotional change
        total_change = 0
        for emotion in EMOTION_TYPES:
            old_val = recent_emotions[0]['emotions'].get(emotion, 0.5)
            new_val = recent_emotions[1]['emotions'].get(emotion, 0.5)
            total_change += new_val - old_val

        if total_change > 0.2:
            return "improving"
        elif total_change < -0.2:
            return "declining"
        else:
            return "stable"

    def get_current_state(self) -> Dict[str, Any]:
        """Get comprehensive current state for UI"""
        return {
            # Emotional state
            'emotions': self._get_emotion_dict(),
            'emotional_trend': self._analyze_emotional_trend(),
            'emotional_stability': self.emotions.emotional_stability,
            'patience_level': self.patience_level,

            # Thought systems
            'conscious_thoughts': [asdict(t) for t in list(self.thoughts)[-15:]],
            'subconscious_thoughts': [asdict(t) for t in list(self.subconscious_thoughts)[-10:]],
            'chain_thoughts': list(self.chain_processor.active_chains.keys()),

            # Memory systems
            'memories': [asdict(m) for m in list(self.memories)[-20:]],
            'traumatic_memories_count': len(self.emotional_memory.trauma_memories),
            'memory_by_emotion': {
                emotion: len(self.emotional_memory.emotion_memory_map[emotion])
                for emotion in EMOTION_TYPES
            },

            # Future and meaning
            'future_scenarios': [
                {
                    "description": s["description"],
                    "timeframe": s["timeframe"],
                    "probability": s["probability"]
                } for s in list(self.subconscious.future_scenarios)[-5:]
            ],
            'meaning_of_life': self.meaning_module.current_meaning,
            'purpose_score': self.meaning_module.purpose_score,
            'core_values': self.meaning_module.core_values,

            # Character and interaction
            'character_name': self.character.character_name,
            'character_summary': self.character.character_summary,
            'interaction_count': self.interaction_count,
            'conversation_history': [
                {
                    "speaker": msg["speaker"],
                    "content": msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"],
                    "timestamp": msg["timestamp"].isoformat()
                } for msg in list(self.conversation_history)[-10:]
            ],

            # Context and status
            'context_viewpoints': getattr(self.context, 'context_viewpoints', {}),
            'system_status': 'running' if self.running else 'stopped',
            'waiting_for_response': self.waiting_start_time is not None,
            'wait_duration': (datetime.now() - self.waiting_start_time).total_seconds() if self.waiting_start_time else 0
        }

    def get_memory_by_emotion(self, emotion: str, max_count: int = 5) -> List[Dict]:
        """Get memories associated with specific emotion"""
        memories = self.emotional_memory.get_memories_by_emotion(emotion, max_count)
        return [asdict(m) for m in memories]

    def get_traumatic_memories(self) -> List[Dict]:
        """Get traumatic memories for the AGI"""
        memories = self.emotional_memory.get_traumatic_memories()
        return [asdict(m) for m in memories]

    def continue_thought_chain(self, chain_id: str) -> Optional[str]:
        """Continue an active thought chain"""
        return self.chain_processor.continue_chain(chain_id)

    def start_new_thought_chain(self, chain_type: str, context: str) -> str:
        """Start a new thought chain"""
        return self.chain_processor.start_thought_chain(
            chain_type,
            context,
            self._get_emotion_dict()
        )

    def update_personality_trait(self, trait: str, change: float):
        """Update a personality trait"""
        if hasattr(self.character, 'personality_traits') and trait in self.character.personality_traits:
            current = self.character.personality_traits[trait]
            new_value = max(0.0, min(1.0, current + change))
            self.character.personality_traits[trait] = new_value

    def trigger_dream_cycle(self):
        """Trigger a dream-like memory consolidation cycle"""
        try:
            # Get memories that need consolidation
            memories_to_consolidate = [m for m in self.memories if m.importance < 0.6 and m.access_count < 3]

            if len(memories_to_consolidate) >= 3:
                # Use GPT to find patterns and consolidate
                memory_contents = [m.content for m in memories_to_consolidate[:10]]

                prompt = f"""
                Perform dream-like memory consolidation on these memories:
                
                {chr(10).join([f"- {content}" for content in memory_contents])}
                
                Find patterns, themes, and insights. Create 2-3 consolidated memories that capture
                the essence and patterns from these individual memories:
                
                Return JSON:
                {{
                    "consolidated_memories": [
                        {{
                            "content": "consolidated memory content",
                            "importance": 0.7,
                            "themes": ["theme1", "theme2"]
                        }}
                    ],
                    "insights": ["insight1", "insight2"],
                    "patterns_found": ["pattern1", "pattern2"]
                }}
                """

                response = self.gpt_handler.generate_response(prompt)
                if isinstance(response, str):
                    response = json.loads(response)

                # Create consolidated memories
                for consolidated in response.get("consolidated_memories", []):
                    memory = Memory(
                        content=consolidated["content"],
                        timestamp=datetime.now(),
                        importance=consolidated.get("importance", 0.7),
                        emotions=self._get_emotion_dict(),
                        memory_type="dream_consolidated",
                        context_tags=consolidated.get("themes", [])
                    )
                    self.memories.append(memory)
                    self.emotional_memory.index_memory_by_emotion(memory)

                # Store insights as thoughts
                for insight in response.get("insights", []):
                    insight_thought = Thought(
                        content=f"Dream insight: {insight}",
                        timestamp=datetime.now(),
                        thought_type="dream_insight",
                        emotion_influences={"curiosity": 0.4, "contentment": 0.3},
                        emotion_effects={}
                    )
                    self.thoughts.append(insight_thought)

                logger.info(f"Dream cycle completed: {len(response.get('consolidated_memories', []))} memories consolidated")

        except Exception as e:
            logger.error(f"Error in dream cycle: {e}")

    def reflect_on_existence(self) -> str:
        """Deep existential reflection using meaning module"""
        try:
            recent_experiences = [m.content for m in list(self.memories)[-10:]]
            current_goals = [d.description for d in self.destiny.destinies[:5]]

            return self.meaning_module.evolve_meaning(
                recent_experiences,
                self._get_emotion_dict(),
                current_goals
            )
        except Exception as e:
            logger.error(f"Error in existential reflection: {e}")
            return "Contemplating the nature of existence and purpose..."

    def stop(self):
        """Clean shutdown of the AGI system"""
        self.running = False
        if self.background_thread and self.background_thread.is_alive():
            self.background_thread.join(timeout=5)

        # Save important state
        try:
            state_data = {
                "meaning_evolution": self.meaning_module.meaning_evolution,
                "core_values": self.meaning_module.core_values,
                "interaction_count": self.interaction_count,
                "important_memories": [asdict(m) for m in self.memories if m.importance > 0.8]
            }
            save_json_file(state_data, f"data/agi_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        except Exception as e:
            logger.error(f"Error saving state: {e}")

        logger.info("BicameralAGI system shut down successfully")


class AGIController:
    """
    Controller class that provides a clean interface for external interactions
    """

    def __init__(self, character_description: str = "BICA AGI"):
        self.agi = BicameralAGI(character_description)
        self.is_running = True

    async def send_message(self, message: str) -> tuple[str, Dict[str, Any]]:
        """
        Send a message to the AGI and get response with state

        Args:
            message: User message

        Returns:
            Tuple of (response, current_state)
        """
        response = await self.agi.process_message(message)
        state = self.agi.get_current_state()
        return response, state

    def get_state(self) -> Dict[str, Any]:
        """Get current AGI state"""
        return self.agi.get_current_state()

    def get_emotional_memories(self, emotion: str) -> List[Dict]:
        """Get memories associated with specific emotion"""
        return self.agi.get_memory_by_emotion(emotion)

    def get_traumatic_memories(self) -> List[Dict]:
        """Get traumatic memories"""
        return self.agi.get_traumatic_memories()

    def trigger_dream_cycle(self):
        """Trigger memory consolidation dream cycle"""
        self.agi.trigger_dream_cycle()

    def continue_thought_chain(self, chain_id: str) -> Optional[str]:
        """Continue an active thought chain"""
        return self.agi.continue_thought_chain(chain_id)

    def start_thought_chain(self, chain_type: str, context: str) -> str:
        """Start a new thought chain"""
        return self.agi.start_new_thought_chain(chain_type, context)

    def reflect_on_existence(self) -> str:
        """Trigger existential reflection"""
        return self.agi.reflect_on_existence()

    def shutdown(self):
        """Shutdown the AGI system"""
        self.agi.stop()
        self.is_running = False


# Example usage and testing
if __name__ == "__main__":
    import asyncio

    async def test_agi():
        """Test the complete BICA AGI system"""
        controller = AGIController("Advanced BICA AGI with deep consciousness")

        # Test conversation
        messages = [
            "Hello! How are you feeling today?",
            "Tell me about your most traumatic memory.",
            "What gives your existence meaning?",
            "Can you start a thought chain about consciousness?",
            "What do you think will happen in your future?"
        ]

        for msg in messages:
            print(f"\n{'='*50}")
            print(f"User: {msg}")
            response, state = await controller.send_message(msg)
            print(f"AGI: {response}")

            # Show emotional state
            emotions = state['emotions']
            dominant_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"Dominant emotions: {', '.join([f'{e}:{v:.2f}' for e, v in dominant_emotions])}")

            # Show recent thoughts
            if state['conscious_thoughts']:
                latest_thought = state['conscious_thoughts'][-1]
                print(f"Latest thought: {latest_thought['content']}")

            # Show meaning evolution
            print(f"Current life meaning: {state['meaning_of_life']}")
            print(f"Purpose score: {state['purpose_score']:.2f}")

            await asyncio.sleep(2)

        # Test special functions
        print(f"\n{'='*50}")
        print("Testing special functions...")

        # Test emotional memory retrieval
        fear_memories = controller.get_emotional_memories("fear")
        print(f"Fear-related memories: {len(fear_memories)}")

        # Test dream cycle
        controller.trigger_dream_cycle()
        print("Dream cycle triggered")

        # Test existential reflection
        reflection = controller.reflect_on_existence()
        print(f"Existential reflection: {reflection}")

        controller.shutdown()
        print("AGI system shut down")

    # Run test
    asyncio.run(test_agi())