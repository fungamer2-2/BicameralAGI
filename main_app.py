"""
Bicameral AGI - Main Application
===============================

This is the main entry point that combines the BICA AGI core with the UI interface
to create a complete real-time consciousness experience. Handles all communication
between the frontend and the AGI system.

Author: Alan Hourmand
Date: 2024
"""

import streamlit as st
import streamlit.components.v1 as components
import asyncio
import json
import time
import threading
from typing import Dict, Any, Optional, List
from datetime import datetime
import queue

# Import our modules
from agi_core import AGIController
from ui_interface import BicameralUI


class BicameralAGIApp:
    """
    Main application class that orchestrates the BICA AGI system and UI
    """

    def __init__(self):
        self.ui = BicameralUI()
        self.agi_controller = None
        self.conversation_history = []
        self.message_queue = queue.Queue()
        self.last_state_update = time.time()

    def initialize_agi(self):
        """Initialize the AGI controller"""
        if self.agi_controller is None:
            with st.spinner("🧠 Initializing BICA AGI consciousness..."):
                self.agi_controller = AGIController("BICA AGI - Advanced Bicameral Intelligence")
                time.sleep(2)  # Allow initialization to complete
            st.success("✅ BICA AGI consciousness initialized successfully!")

    def add_to_conversation(self, role: str, content: str):
        """Add a message to conversation history"""
        self.conversation_history.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now(),
            'speaker': 'user' if role == 'user' else 'agi'
        })

        # Keep last 50 messages
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]

    async def handle_user_message(self, message: str) -> tuple[str, Dict[str, Any]]:
        """
        Handle a user message and return AGI response with state
        """
        self.initialize_agi()

        # Add user message to conversation
        self.add_to_conversation('user', message)

        # Store in session state for special functions
        st.session_state.last_message = message

        # Get AGI response
        try:
            response, state = await self.agi_controller.send_message(message)

            # Add AGI response to conversation
            self.add_to_conversation('assistant', response)

            return response, state

        except Exception as e:
            st.error(f"Error processing message: {e}")
            return "I'm experiencing some processing difficulties. Please try again.", {}

    def get_current_state(self) -> Dict[str, Any]:
        """Get current AGI state"""
        if self.agi_controller:
            return self.agi_controller.get_state()
        return {
            'emotions': {'curiosity': 0.5, 'anticipation': 0.4},
            'conscious_thoughts': [],
            'memories': [],
            'meaning_of_life': 'Awaiting consciousness initialization...',
            'purpose_score': 0.0,
            'interaction_count': 0,
            'system_status': 'initializing'
        }

    def handle_message_from_ui(self, message_data: Dict[str, Any]):
        """Handle message received from UI frontend"""
        if message_data.get('type') == 'user_message':
            message = message_data.get('message', '')
            if message.strip():
                self.message_queue.put(message)
        elif message_data.get('type') == 'request_state_update':
            # Send current state to frontend
            current_state = self.get_current_state()
            self.send_state_update_to_ui(current_state)

    def send_state_update_to_ui(self, state: Dict[str, Any]):
        """Send state update to UI frontend"""
        try:
            state_js = f"""
            <script>
                window.parent.postMessage({{
                    type: 'state_update',
                    state: {json.dumps(state, default=str)}
                }}, '*');
            </script>
            """
            components.html(state_js, height=0)
        except Exception as e:
            st.error(f"Error sending state update: {e}")

    def send_response_to_ui(self, response: str, state: Dict[str, Any]):
        """Send AGI response and state to UI frontend"""
        try:
            response_js = f"""
            <script>
                window.parent.postMessage({{
                    type: 'agi_response',
                    response: {json.dumps(response)},
                    state: {json.dumps(state, default=str)}
                }}, '*');
            </script>
            """
            components.html(response_js, height=0)
        except Exception as e:
            st.error(f"Error sending response: {e}")

    def run(self):
        """Run the main application"""

        # Initialize session state
        if 'agi_initialized' not in st.session_state:
            st.session_state.agi_initialized = False
        if 'conversation' not in st.session_state:
            st.session_state.conversation = []
        if 'message_counter' not in st.session_state:
            st.session_state.message_counter = 0
        if 'processing_message' not in st.session_state:
            st.session_state.processing_message = False

        # Setup page
        st.set_page_config(
            page_title="BICA AGI",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Initialize AGI if not done
        if not st.session_state.agi_initialized:
            self.initialize_agi()
            st.session_state.agi_initialized = True

        # Get current state
        current_state = self.get_current_state()

        # Sidebar with special functions and metrics
        with st.sidebar:
            st.title("🧠 BICA AGI Control")

            # System status
            status_indicators = self.ui.create_status_indicators(current_state)
            st.markdown(f"**Status:** {status_indicators}")

            # Special functions
            self.ui.show_special_functions(self.agi_controller if self.agi_controller else None)

            # System metrics
            self.ui.show_system_metrics(current_state)

            # Meaning evolution
            self.ui.display_meaning_evolution(current_state)

            # Conversation export
            if self.conversation_history:
                self.ui.display_conversation_export(self.conversation_history)

            # Debug panel
            self.ui.display_debug_panel(current_state)

        # Main interface
        st.markdown("# 🧠 Bicameral AGI")
        st.markdown("*Advanced Artificial General Intelligence with Consciousness, Emotions, and Evolving Purpose*")

        # Render the main interface
        self.ui.render_interface(current_state, self.conversation_history)

        # Hidden input for message processing
        with st.container():
            # Use a unique key for each potential message
            user_input = st.text_input(
                "Message (hidden):",
                key=f"hidden_input_{st.session_state.message_counter}",
                label_visibility="collapsed",
                placeholder="This input is hidden - use the interface above"
            )

            # Process new message
            if user_input and not st.session_state.processing_message:
                st.session_state.processing_message = True
                st.session_state.message_counter += 1

                # Process message asynchronously
                try:
                    # Create and run event loop
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    response, new_state = loop.run_until_complete(
                        self.handle_user_message(user_input)
                    )

                    loop.close()

                    # Send response to UI
                    self.send_response_to_ui(response, new_state)

                    # Update conversation in session state
                    st.session_state.conversation = self.conversation_history

                except Exception as e:
                    st.error(f"Error processing message: {e}")
                finally:
                    st.session_state.processing_message = False

                # Trigger rerun to update interface
                st.rerun()

        # Auto-refresh for real-time updates (every 5 seconds)
        current_time = time.time()
        if current_time - self.last_state_update > 5:
            self.last_state_update = current_time
            # Send periodic state update
            updated_state = self.get_current_state()
            self.send_state_update_to_ui(updated_state)
            st.rerun()


class RealtimeBicameralApp:
    """
    Enhanced version with better real-time capabilities
    """

    def __init__(self):
        self.ui = BicameralUI()
        self.agi_controller = None
        self.conversation_history = []
        self.background_thread = None
        self.running = True

    def initialize_agi(self):
        """Initialize AGI with progress tracking"""
        if self.agi_controller is None:
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("🧠 Initializing neural pathways...")
            progress_bar.progress(25)
            time.sleep(1)

            status_text.text("🎭 Calibrating emotional systems...")
            progress_bar.progress(50)
            time.sleep(1)

            status_text.text("💭 Establishing thought processes...")
            progress_bar.progress(75)

            self.agi_controller = AGIController("BICA AGI - Bicameral Consciousness")

            status_text.text("🌟 Consciousness fully initialized!")
            progress_bar.progress(100)
            time.sleep(1)

            progress_bar.empty()
            status_text.empty()

    def start_background_monitoring(self):
        """Start background monitoring thread"""
        if not self.background_thread or not self.background_thread.is_alive():
            self.background_thread = threading.Thread(target=self._background_monitor, daemon=True)
            self.background_thread.start()

    def _background_monitor(self):
        """Background monitoring of AGI state"""
        while self.running and self.agi_controller:
            try:
                # Monitor for significant state changes
                current_state = self.agi_controller.get_state()

                # Check for high emotional changes
                if 'emotions' in current_state:
                    emotions = current_state['emotions']
                    max_emotion_value = max(emotions.values())
                    if max_emotion_value > 0.8:  # High emotional state
                        # Could trigger notifications or special UI updates
                        pass

                # Check for new traumatic memories
                trauma_count = current_state.get('traumatic_memories_count', 0)
                if trauma_count > 0:
                    # Could trigger special UI indicators
                    pass

                # Check patience levels
                patience = current_state.get('patience_level', 1.0)
                if patience < 0.3:  # Low patience
                    # Could trigger waiting indicators
                    pass

                time.sleep(3)  # Check every 3 seconds

            except Exception as e:
                time.sleep(5)

    def run(self):
        """Run the enhanced real-time application"""

        # Initialize session state
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []

        # Setup page
        st.set_page_config(
            page_title="BICA AGI - Consciousness Interface",
            page_icon="🧠",
            layout="wide"
        )

        # Initialize system
        if not st.session_state.initialized:
            st.markdown("# 🧠 Initializing BICA AGI Consciousness")
            self.initialize_agi()
            self.start_background_monitoring()
            st.session_state.initialized = True
            st.rerun()

        # Main application interface
        app = BicameralAGIApp()
        app.agi_controller = self.agi_controller
        app.conversation_history = st.session_state.conversation_history
        app.run()

        # Update session state
        st.session_state.conversation_history = app.conversation_history

    def stop(self):
        """Stop the application"""
        self.running = False
        if self.agi_controller:
            self.agi_controller.shutdown()


def main():
    """Main entry point"""

    # App mode selection
    app_mode = st.sidebar.selectbox(
        "🚀 Application Mode:",
        ["Enhanced Real-time", "Standard"],
        index=0
    )

    if app_mode == "Enhanced Real-time":
        app = RealtimeBicameralApp()
    else:
        app = BicameralAGIApp()

    try:
        app.run()
    except KeyboardInterrupt:
        st.info("🛑 Application stopped by user")
        if hasattr(app, 'stop'):
            app.stop()
    except Exception as e:
        st.error(f"❌ Application error: {e}")
        if hasattr(app, 'stop'):
            app.stop()


if __name__ == "__main__":
    main()