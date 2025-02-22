import streamlit as st
import os
import openai
from routellm.controller import Controller
import time

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

with st.spinner("🔄 Mool AI agent Authentication In progress..."):
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    if not openai.api_key:
        st.error("❌ API_KEY not found in environment variables.")
        st.stop()
    time.sleep(2)
st.success("✅ Mool AI Authentication Successful")

if openai.api_key is None:
    st.error("OPENAI_API_KEY environment variable is not set. Please set it before running the app.")
    st.stop()

# Streamlit App
st.title("LLM Router Chatbot")

@st.cache_resource
def init_controller():
    try:
        controller = Controller(
            routers=["mf"],
            strong_model="gpt-4o",
            weak_model="gpt-3.5-turbo",
            config={
                "mf": {"checkpoint_path": "routellm/mf_gpt4_augmented"},
            }
        )
        return controller
    except Exception as e:
        st.error(f"Error initializing controller: {str(e)}")
        return None

if 'controller' not in st.session_state:
    st.session_state.controller = init_controller()

controller = st.session_state.get('controller', None)

def calculate_cost(model_name, input_tokens, output_tokens):
    if "gpt-4" in model_name.lower():
        return (input_tokens * 5e-6) + (output_tokens * 1.5e-5)
    elif "gpt-3.5" in model_name.lower():
        return (input_tokens * 2.5e-7) + (output_tokens * 1.25e-6)
    else:
        return 0

def get_response(prompt, router="mf"):
    try:
        if controller is None:
            st.error("Controller not properly initialized, skipping RouteLLM")
            return "RouteLLM Unavailable", "N/A", 0, 0, 0, 0, "N/A"

        start_time = time.time()
        response = controller.chat.completions.create(
            model=f"router-{router}-0.11593",
            messages=[{"role": "user", "content": prompt}]
        )
        end_time = time.time()

        latency = end_time - start_time
        input_tokens = len(prompt)
        output_tokens = len(response.choices[0].message.content)
        selected_model = response.model
        cost = calculate_cost(selected_model, input_tokens, output_tokens)
        
        return response.choices[0].message.content, f"RouteLLM Router ({router.upper()})", latency, cost, input_tokens, output_tokens, selected_model
    except Exception as e:
        st.error(f"RouteLLM Error: {str(e)}")
        return f"Error: {str(e)}", None, None, None, None, None, None

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, model_used, latency, cost, input_tokens, output_tokens, selected_model = get_response(user_input)
            st.markdown(response)

            # Display metrics
            st.caption(f"Model: {model_used} | Selected: {selected_model} | Latency: {latency:.2f}s | Cost: ${cost:.4f} | Input Tokens: {input_tokens} | Output Tokens: {output_tokens}")

    # Add bot response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": response})

# Option to clear chat history
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()
