"""
BicameralAGI Character Module
================================

Overview:
---------
This module serves as the central coordinator for the BicameralAGI system. It is responsible for initializing and managing
all core components, orchestrating the flow of information between them, and handling the overall processing of user inputs
and system responses. The BicaCharacter class integrates key features like context, memory, and action execution to generate
cohesive, human-like AI behavior.

Current Features:
-----------------
1. **Character Definition**: Automatically generates or updates a character's name and summary based on a given description.
2. **Context Management**: Tracks and updates conversation context for more informed responses.
3. **Prompt Compilation**: Constructs dynamic prompts based on the current system state (context, user input, etc.).
4. **Response Generation**: Handles generating responses based on user input and the processed context.

Usage Example:
--------------
    character = BicaCharacter("A brave knight from the future.")
    response = character.process_input("What is your mission?")
    print(response)

Author:
-------
Alan Hourmand
Date: 10/2/2024
"""
import time

from bica.core.context import BicaContext
from bica.external.gpt_handler import GPTHandler
from bica.core.profile import BicaProfile
from bica.core.memory import BicaMemory
from bica.core.destiny import BicaDestiny
from bica.core.subconcious import BicaSubconscious
from bica.utils.utilities import *


class BicaCharacter:
    def __init__(self, character_description: str, debug_mode: bool):
        self.debug_mode = debug_mode
        self._recent_conversation = []  # Initialize here
        self.gpt_handler = GPTHandler()

        # ||||||||| BICA AGI COGNITIVE SETUP ||||||||||
        self.character_name = "BICA AGI"
        self.character_summary = "You are an artificial general intelligence called BICA. You were created by Alan Hourmand."
        self.extract_character_definition(character_description)

        self.profile = BicaProfile(self.character_name, self.character_summary, self.gpt_handler)

        # Cognitive setup
        self.memory = BicaMemory(self.profile, debug_mode)
        self.destiny = BicaDestiny(self.character_name, self.memory)  # Initialize the destiny module
        self.context = BicaContext()

        # Wait until the profile is initialized
        self.initialize_profile_with_retries()
        # |||||||||||||||||||||||||||||||||||||||||||||

    def initialize_profile_with_retries(self, retries=3, delay=5):
        for attempt in range(retries):
            try:
                if self.profile.character_profile:
                    return
            except Exception as e:
                print(f"Profile initialization attempt {attempt + 1} failed: {str(e)}")
                if attempt < retries - 1:
                    time.sleep(delay)
                else:
                    raise RuntimeError("Failed to initialize character profile after multiple attempts.")

    def get_character_definition(self):
        return self.character_summary

    def extract_character_definition(self, character_description: str):
        """
        Generates or updates the character's name and summary based on the provided description.
        If no name is provided, one is generated.
        """
        prompt = f"""
        Based on the following short description, try to figure out what character the user is referring to. If a name is not provided, generate one that best fits the description.
        
        Also if the description the user gives you seems like traits, just make up a character that best fits the traits.

        Description: {character_description}

        Respond in the format:
        {{
            "name": "Character's name",
            "summary": "You are {{name}}, [brief character summary]."
        }}
        """

        try:
            # Generate the response using GPT
            response = self.gpt_handler.generate_response(prompt)

            # Remove the backticks if present around the JSON
            cleaned_response = response.strip("```json").strip("```").strip()

            # Parse the cleaned JSON response
            character_info = json.loads(cleaned_response)

            # Extract the name and summary from the response
            self.character_name = character_info.get("name", "Unknown Character")
            self.character_summary = character_info.get(
                "summary", f"You are {self.character_name}, a mysterious figure."
            )
        except json.JSONDecodeError:
            # Fallback if GPT response is not in proper JSON format
            print("Error: Could not parse the character definition from the AI response.")
            print(f"Fallback raw response: {response}")  # Debugging the raw response
            self.character_name = "Unknown Character"
            self.character_summary = f"You are {self.character_name}, an enigmatic character."

    def process_input(self, user_input: str) -> str:
        try:
            # Get recent conversation
            recent_convo = self.get_recent_conversation()

            # Retrieves the combined active memories
            recalled_memories = self.memory.get_memories()

            # Update context with user input
            self.context.update_context(user_input, recalled_memories)
            updated_context = self.context.get_context()

            # Calculate destiny influence and generate relevant destinies
            destiny_influence = self.destiny.get_current_destiny_influence(recalled_memories, recent_convo)
            # Check for conflicts in short-term memory and decide how to influence destiny
            # Decide on the destiny based on memories or default to abstract paths
            relevant_destinies = self.decide_destiny()

            print("\n--- Destiny Information ---")
            print(f"Relevant destinies: {json.dumps(relevant_destinies, indent=2)}")
            print(f"Destiny influence: {json.dumps(destiny_influence, indent=2)}")
            print("---------------------------\n")

            # Gather context data
            compiled_data = {
                "user_input": user_input,
                "recent_conversation": recent_convo,
                "system_prompt": self.get_character_definition(),
                "updated_context": updated_context,  # Add updated context to the prompt data
                "character_profile": self.profile.get_profile(),  # Add character profile to the prompt data
                "relevant_memories": recalled_memories,
                "relevant_destinies": relevant_destinies  # Include relevant destinies in context # Add destiny influence to the context
            }

            # Check if compiled_data is a string, if so, wrap it in a dictionary
            if isinstance(compiled_data, str):
                compiled_data = {"compiled_prompt": compiled_data}

            # Generate response using gpt_handler directly instead of action_executor
            response = self.gpt_handler.generate_response_with_context(compiled_data)
            compiled_data["character_response"] = response

            self.update_recent_conversation(user_input, response)

            # Now update the memory with the complete context, including the AI's response
            self.memory.update_memories(compiled_data)

            if self.debug_mode:
                print(f"Working Memory: {recalled_memories['working_memory']}")
                print(f"Short Term Memory: {[m.content for m in recalled_memories['short_term_memory']]}")
                print(f"Long Term Memory: {recalled_memories['long_term_memory']}")
                print(f"Self Memory: {recalled_memories['self_memory']}")

            return response

        except Exception as e:
            print(f"Error in process_input: {str(e)}")
            import traceback
            traceback.print_exc()
            return "I apologize, but I encountered an error. Could you please try again?"

    def decide_destiny(self):
        """
        Decides on a destiny based on current important memories, or defaults to a typical abstract path.
        This approach is generic and adapts to any character input without predefined scenarios.
        """
        important_memories = self.memory.get_high_importance_memories()

        if important_memories:
            return self.destiny.generate_destiny_from_memories(important_memories)
        else:
            return self.destiny.default_destiny_based_on_context()


    def decide_destiny_based_on_memory(self):
        """
        If important memories are present, influence destiny based on them.
        Otherwise, use long-term memories or default to character traits.
        """
        important_memories = self.memory.get_high_importance_memories()

        if important_memories:
            return self.destiny.generate_destiny_from_memories(important_memories)
        else:
            return self.destiny.default_destiny_based_on_profile(self.profile)

    def get_recent_conversation(self, max_length=5):
        return self._recent_conversation[-max_length:]

    def update_recent_conversation(self, user_input, character_response):
        if not hasattr(self, '_recent_conversation'):
            self._recent_conversation = []
        self._recent_conversation.append({"user": user_input, "character": character_response})
        # Keep only the last 10 exchanges
        self._recent_conversation = self._recent_conversation[-10:]

    def compile_prompt(self, compiled_data: dict) -> str:
        if self.debug_mode:
            print("Compiling the prompt...")
        prompt_parts = []
        for key, value in compiled_data.items():
            if value:  # Only include non-empty values
                if isinstance(value, dict):
                    prompt_parts.append(f"{key.replace('_', ' ').title()}:")
                    for sub_key, sub_value in value.items():
                        prompt_parts.append(f"  {sub_key}: {sub_value}")
                elif isinstance(value, list):
                    prompt_parts.append(f"{key.replace('_', ' ').title()}:")
                    for item in value:
                        prompt_parts.append(f"  - {item}")
                else:
                    prompt_parts.append(f"{key.replace('_', ' ').title()}: {value}")

        # Including destiny influence as a separate section in the prompt
        if "destiny_influence" in compiled_data and compiled_data["destiny_influence"]:
            prompt_parts.append("\nDestiny Influence on Current Situation:")
            for title, influence in compiled_data["destiny_influence"].items():
                prompt_parts.append(f"- {title}: {influence:.2f}")

        prompt = "\n".join(prompt_parts)
        prompt += "\n\nBased on the above context information, generate a final response to the user."
        if self.debug_mode:
            print(f"Compiled prompt:\n{prompt}")
        return prompt