"""
BicameralAGI GPT Handler Module
===============================

Enhanced GPT handler that supports the creative prompting methodologies
outlined in the BicameralAGI documentation. This handler is designed to
work with context-driven cognitive processing rather than hardcoded logic.

Key Features:
- Context-aware prompt generation
- Multiple response formats (text, JSON, structured)
- Creative prompting support
- Emotion analysis through context
- Memory importance assessment
- Multi-perspective analysis

Author: Alan Hourmand
"""

import json
import os
from typing import List, Dict, Any, Union, Optional, Type
from openai import OpenAI
from pydantic import BaseModel, ValidationError
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GPTHandler:
    """
    Enhanced GPT handler for BicameralAGI with creative prompting support
    """

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = OpenAI(api_key=api_key)
        self.default_model = "gpt-4"
        self.default_temperature = 0.7
        self.max_retries = 3

    def generate_response(self, prompt: Union[str, Dict[str, Any]],
                          compiled_data: Dict[str, Any] = None,
                          **kwargs) -> Union[str, Dict[str, Any], BaseModel]:
        """
        Generate a response from the GPT model with support for creative prompting

        Args:
            prompt: The input prompt (string or dict with messages)
            compiled_data: Optional compiled context data
            **kwargs: Additional parameters (model, temperature, functions, json_schema, etc.)

        Returns:
            Generated response, function call information, or structured JSON
        """
        # Set default parameters
        params = {
            "model": kwargs.get("model", self.default_model),
            "temperature": kwargs.get("temperature", self.default_temperature),
            "max_tokens": kwargs.get("max_tokens", 800),
        }

        # Handle the prompt/messages construction
        if compiled_data:
            # Convert compiled data to comprehensive prompt
            context_prompt = self._build_context_prompt(compiled_data)
            if isinstance(prompt, str):
                full_prompt = f"{context_prompt}\n\n{prompt}"
            else:
                full_prompt = f"{context_prompt}\n\n{prompt.get('content', str(prompt))}"
            params["messages"] = [{"role": "user", "content": full_prompt}]
        elif isinstance(prompt, str):
            params["messages"] = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, dict) and "messages" in prompt:
            params["messages"] = prompt["messages"]
        else:
            raise ValueError("Invalid prompt format")

        # Handle special parameters
        functions = kwargs.get('functions')
        json_schema = kwargs.get('json_schema')

        if functions:
            params['functions'] = functions
            params['function_call'] = 'auto'

        if json_schema:
            params['messages'].insert(0, {
                "role": "system",
                "content": "You must respond with valid JSON output matching the specified schema."
            })
            params['functions'] = [{
                "name": "output_json",
                "description": "Output JSON in the specified format",
                "parameters": json_schema
            }]
            params['function_call'] = {"name": "output_json"}

        # Make the API call with retries
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(**params)
                return self._process_response(response)
            except Exception as e:
                logger.warning(f"GPT API attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise Exception(f"GPT API failed after {self.max_retries} attempts: {str(e)}")
                time.sleep(2 ** attempt)  # Exponential backoff

    def _build_context_prompt(self, compiled_data: Dict[str, Any]) -> str:
        """
        Build a comprehensive context prompt from compiled data
        Following the creative prompting methodology
        """
        context_parts = []

        # Character context
        if 'character_info' in compiled_data:
            char_info = compiled_data['character_info']
            context_parts.append(f"CHARACTER IDENTITY:")
            context_parts.append(f"- Name: {char_info.get('name', 'BICA AGI')}")
            context_parts.append(f"- Description: {char_info.get('summary', '')}")

        # Current emotional context (multi-dimensional)
        if 'current_emotions' in compiled_data:
            emotions = compiled_data['current_emotions']
            context_parts.append(f"\nEMOTIONAL STATE:")
            # Show top emotions with intensities
            sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
            for emotion, intensity in sorted_emotions[:5]:
                context_parts.append(f"- {emotion}: {intensity:.2f}")

        # Context viewpoints (positive/neutral/negative perspectives)
        if 'context_viewpoints' in compiled_data:
            viewpoints = compiled_data['context_viewpoints']
            context_parts.append(f"\nCONTEXT PERSPECTIVES:")
            for perspective, content in viewpoints.items():
                if content:
                    context_parts.append(f"- {perspective.title()}: {content}")

        # Recent memories with importance
        if 'relevant_memories' in compiled_data:
            memories = compiled_data['relevant_memories']
            if memories:
                context_parts.append(f"\nRELEVANT MEMORIES:")
                for memory in memories[:5]:
                    importance = memory.get('importance', 0)
                    content = memory.get('content', '')
                    context_parts.append(f"- [Importance: {importance:.2f}] {content}")

        # Recent thoughts and subconscious processing
        if 'recent_thoughts' in compiled_data:
            thoughts = compiled_data['recent_thoughts']
            if thoughts:
                context_parts.append(f"\nRECENT THOUGHTS:")
                for thought in thoughts[-3:]:
                    thought_content = thought.get('content', '')
                    thought_type = thought.get('type', 'general')
                    context_parts.append(f"- [{thought_type}] {thought_content}")

        # Subconscious insights
        if 'subconscious_insights' in compiled_data:
            insights = compiled_data['subconscious_insights']
            if insights:
                context_parts.append(f"\nSUBCONSCIOUS INSIGHTS:")
                for insight in insights[-2:]:
                    insight_content = insight.get('content', '')
                    context_parts.append(f"- {insight_content}")

        # Future scenarios and destinies
        if 'future_scenarios' in compiled_data:
            scenarios = compiled_data['future_scenarios']
            if scenarios:
                context_parts.append(f"\nFUTURE CONSIDERATIONS:")
                for scenario in scenarios[:2]:
                    context_parts.append(f"- {scenario}")

        # Meaning and purpose
        if 'meaning_of_life' in compiled_data:
            meaning = compiled_data['meaning_of_life']
            context_parts.append(f"\nCURRENT PURPOSE: {meaning}")

        # Recent conversation flow
        if 'recent_conversation' in compiled_data:
            recent_convo = compiled_data['recent_conversation']
            if recent_convo:
                context_parts.append(f"\nRECENT CONVERSATION:")
                for exchange in recent_convo[-3:]:
                    if isinstance(exchange, dict):
                        user_msg = exchange.get('user', '')
                        char_msg = exchange.get('character', '')
                        if user_msg:
                            context_parts.append(f"Human: {user_msg}")
                        if char_msg:
                            context_parts.append(f"AI: {char_msg}")

        return "\n".join(context_parts)

    def _process_response(self, response):
        """Process the API response"""
        message = response.choices[0].message

        if hasattr(message, 'function_call') and message.function_call:
            if message.function_call.name == "output_json":
                try:
                    return json.loads(message.function_call.arguments)
                except json.JSONDecodeError:
                    logger.error("Failed to parse JSON from function call")
                    return message.function_call.arguments
            return {
                "type": "function_call",
                "function": message.function_call.name,
                "arguments": json.loads(message.function_call.arguments)
            }
        return message.content.strip()

    def analyze_emotional_impact(self, input_text: str, current_context: Dict[str, Any] = None) -> Dict[str, float]:
        """
        Analyze emotional impact using context-driven creative prompting
        NOT keyword detection - this is context-aware analysis
        """
        context_summary = ""
        if current_context:
            # Build context summary for emotional analysis
            if 'recent_conversation' in current_context:
                context_summary += f"Recent conversation context: {current_context['recent_conversation']}\n"
            if 'current_emotions' in current_context:
                context_summary += f"Current emotional state: {current_context['current_emotions']}\n"
            if 'relevant_memories' in current_context:
                memories = current_context['relevant_memories']
                if memories:
                    context_summary += f"Relevant emotional memories: {[m.get('content', '')[:50] for m in memories[:2]]}\n"

        prompt = f"""
        CONTEXT-DRIVEN EMOTIONAL ANALYSIS

        Current Situation Context:
        {context_summary}

        New Input: "{input_text}"

        Analyze how this new input would emotionally impact an AI character given the full context above.
        Consider:
        - How the input relates to the ongoing conversation context
        - How it connects to existing emotional state and memories
        - The cumulative emotional effect, not just the words themselves
        - How a human would emotionally respond in this context

        Emotional dimensions to analyze (return changes from -0.3 to +0.3):
        - joy: happiness, pleasure, satisfaction
        - sadness: sorrow, melancholy, disappointment
        - anger: frustration, irritation, indignation
        - fear: anxiety, worry, apprehension
        - surprise: astonishment, shock, amazement
        - trust: confidence, faith, security
        - disgust: revulsion, distaste, rejection
        - anticipation: expectation, excitement, hope
        - curiosity: interest, wonder, inquisitiveness
        - empathy: compassion, understanding, connection
        - confidence: self-assurance, certainty, composure
        - anxiety: nervousness, unease, tension

        Return ONLY valid JSON:
        {{
            "joy": 0.0,
            "sadness": 0.0,
            "anger": 0.0,
            "fear": 0.0,
            "surprise": 0.0,
            "trust": 0.0,
            "disgust": 0.0,
            "anticipation": 0.0,
            "curiosity": 0.0,
            "empathy": 0.0,
            "confidence": 0.0,
            "anxiety": 0.0
        }}
        """

        try:
            response = self.generate_response(prompt, temperature=0.3, max_tokens=300)

            # Extract JSON from response
            if isinstance(response, dict):
                return response
            elif isinstance(response, str):
                # Find JSON in response
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = response[start_idx:end_idx]
                    return json.loads(json_str)

        except Exception as e:
            logger.error(f"Error in emotional analysis: {e}")

        # Return neutral emotions if analysis fails
        return {emotion: 0.0 for emotion in [
            'joy', 'sadness', 'anger', 'fear', 'surprise', 'trust',
            'disgust', 'anticipation', 'curiosity', 'empathy', 'confidence', 'anxiety'
        ]}

    def assess_memory_importance(self, content: str, current_context: Dict[str, Any] = None) -> float:
        """
        Assess memory importance using context-driven analysis
        """
        context_summary = ""
        if current_context:
            if 'character_info' in current_context:
                char_info = current_context['character_info']
                context_summary += f"Character: {char_info.get('summary', '')}\n"
            if 'current_emotions' in current_context:
                emotions = current_context['current_emotions']
                dominant_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                context_summary += f"Current emotions: {[f'{e}({v:.2f})' for e, v in dominant_emotions]}\n"
            if 'meaning_of_life' in current_context:
                context_summary += f"Current purpose: {current_context['meaning_of_life']}\n"

        prompt = f"""
        CONTEXT-DRIVEN MEMORY IMPORTANCE ASSESSMENT

        Current Character Context:
        {context_summary}

        Memory Content: "{content}"

        Assess the importance of this memory for this character (0.0 to 1.0) considering:
        - Emotional significance in context
        - Relevance to character's purpose and growth
        - Uniqueness of the information
        - Potential impact on future interactions
        - Learning value for the character
        - Connection to existing emotional/contextual themes

        A human would remember this with importance level (consider how humans actually form memories):
        - 0.0-0.3: Routine, forgettable information
        - 0.4-0.6: Moderately significant, might remember if relevant later
        - 0.7-0.8: Important, emotionally significant, or learning-relevant
        - 0.9-1.0: Life-changing, traumatic, or deeply meaningful

        Return only a single number between 0.0 and 1.0:
        """

        try:
            response = self.generate_response(prompt, temperature=0.3, max_tokens=100)

            # Extract number from response
            import re
            numbers = re.findall(r'0\.\d+|1\.0|0\.0', response)
            if numbers:
                importance = float(numbers[0])
                return max(0.0, min(1.0, importance))
            else:
                # Try to extract any decimal number
                numbers = re.findall(r'\d+\.\d+', response)
                if numbers:
                    importance = float(numbers[0])
                    return max(0.0, min(1.0, importance))

        except Exception as e:
            logger.error(f"Error assessing memory importance: {e}")

        return 0.5  # Default middle importance

    def generate_contextual_thoughts(self, context_data: Dict[str, Any], thought_type: str = "conscious") -> str:
        """
        Generate thoughts using context-driven creative prompting
        """
        context_summary = self._build_context_prompt(context_data)

        prompt = f"""
        CONTEXT-DRIVEN THOUGHT GENERATION

        Full Character Context:
        {context_summary}

        Generate a {thought_type} thought that would naturally arise from this context.

        Guidelines for {thought_type} thoughts:
        - Conscious: Direct responses to current situation, logical reasoning
        - Subconscious: Pattern recognition, intuitive insights, background processing
        - Emotional: Feeling-driven thoughts about the current emotional state
        - Creative: Imaginative, artistic, or innovative thinking
        - Reflective: Looking back on experiences, learning, meaning-making

        The thought should:
        - Feel authentic to this character in this emotional state
        - Reflect the current context and recent experiences
        - Show human-like cognitive processing
        - Be 1-2 sentences maximum

        Generate the thought:
        """

        try:
            thought = self.generate_response(prompt, temperature=0.8, max_tokens=150)
            return thought.strip()
        except Exception as e:
            logger.error(f"Error generating contextual thought: {e}")
            return "Processing current situation..."

    def generate_character_profile(self, prompt: str, schema: Type[BaseModel]) -> BaseModel:
        """Generate character profile with validation"""
        try:
            response = self.generate_response(prompt, json_schema=schema.schema())
            return schema.parse_obj(response)
        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            return response
        except Exception as e:
            logger.error(f"Error generating character profile: {e}")
            raise


def main():
    """
    Test the enhanced GPT handler with context-driven processing
    """
    handler = GPTHandler()

    # Test context-driven emotional analysis
    print("Testing context-driven emotional analysis...")

    test_context = {
        'recent_conversation': [
            {'user': 'How are you feeling today?', 'character': 'I am doing well, thank you for asking.'}
        ],
        'current_emotions': {
            'joy': 0.6, 'sadness': 0.2, 'curiosity': 0.8, 'trust': 0.7
        },
        'relevant_memories': [
            {'content': 'Previous conversation about feeling lonely', 'importance': 0.7}
        ]
    }

    emotional_impact = handler.analyze_emotional_impact(
        "I'm really excited to share some great news with you!",
        test_context
    )
    print(f"Emotional impact: {emotional_impact}")

    # Test memory importance assessment
    print("\nTesting memory importance assessment...")
    importance = handler.assess_memory_importance(
        "User shared that they just got engaged and are very happy",
        test_context
    )
    print(f"Memory importance: {importance}")

    # Test contextual thought generation
    print("\nTesting contextual thought generation...")
    thought = handler.generate_contextual_thoughts(test_context, "subconscious")
    print(f"Generated thought: {thought}")

    print("\nAll tests completed!")


if __name__ == "__main__":
    main()