import streamlit as st
import pandas as pd
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

# Chat interface
st.header("Chat Interface")

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

# 50 Questions Analysis
st.header("50 Questions Analysis")

# List of 50 questions (you should replace these with your actual questions)
questions = [
    "What is the capital of France?",
    "Explain the process of photosynthesis.",
    "What is Bernoulli’s Principle?",
    "How does entropy relate to the Second Law of Thermodynamics? ",
    "Explain how a transistor works in a circuit.",
    "Why do superconductors work at extremely low temperatures?",
    "How does quantum entanglement defy classical physics?",
    "What is the Trolley Problem in ethics?",
    "How does Kant’s categorical imperative differ from utilitarianism?",
    "What are the philosophical implications of AI consciousness?",
    "Can a machine exhibit true free will?",
    "What is the ethical argument against data privacy violations?",
    "What is the capital of India?",
    "Who wrote Pride and Prejudice?",
    "What is the atomic number of oxygen?",
    "How many continents are there?",
    "What is the largest planet in our solar system?",
    "What is 25 + 17?",
    "Solve for x: 2x = 10",
    "Convert 100 cm to meters",
    "What is the square root of 144?",
    "What is 15% of 200?",
    "If all cats have fur and Whiskers is a cat, does Whiskers have fur?",
    "A train leaves at 10:00 AM, traveling at 60 mph. How far does it travel in 3 hours?",
    "A farmer has 10 chickens and 5 cows. How many legs are there in total?",
    "If A is taller than B and B is taller than C, who is the tallest?",
    "What is the capital of Italy?",
    "What is the capital of UK?",
    "Write a 4 line story about Elsa?",
    "Write a 4 line story about Belle?",
    "What organ in the human body pumps blood?",
    "What is the color of the sky?",
    "Write a 4 line story about Snow white?",
    "Write a 4 line story about Cinderella?",
    "What organ in the human body is used to see?",
    "What organ in the human body is used to hear?",
    "What is the first planet in the Solar system",
    "What is in the center of the Solar system?",
    "Which planet has rings around it in Solar system?",
    "What organ in the human body is used to taste?"
]

if st.button("Run 50 Questions Analysis"):
    metrics = []
    st.session_state.strong_model_calls = 0
    st.session_state.weak_model_calls = 0

    progress_bar = st.progress(0)
    for i, question in enumerate(questions):
        response, model_used, latency, cost, input_tokens, output_tokens, selected_model = get_response(question)
        if "gpt-3.5" in selected_model.lower():
            selected_model = "Mool AI"
        
        metrics.append({
            "Question": question,
            "Selected Model": selected_model,
            "Latency (s)": latency,
            "Cost ($)": cost,
            "Input Tokens": input_tokens,
            "Output Tokens": output_tokens
        })
        
        if "gpt-4" in selected_model.lower():
             st.session_state.strong_model_calls += 1
        elif "Mool AI" in selected_model:
             st.session_state.weak_model_calls += 1
        
        progress_bar.progress((i + 1) / len(questions))

    # Create a DataFrame from the metrics
    df = pd.DataFrame(metrics)

    # Display the table
    st.subheader("Metrics for 50 Questions")
    st.dataframe(df)

# Calculate totals
total_calls =  st.session_state.strong_model_calls +  st.session_state.weak_model_calls
total_cost = df['Cost ($)'].sum()
total_latency = df['Latency (s)'].sum()

# Create summary table with all metrics
st.subheader("Model Usage and Overall Metrics Summary")
summary_data = {
    "Metric": [
        "Total Calls",
        "Strong Model (GPT-4) Calls",
        "Mool AI Calls",
        "Total Cost",
        "Total Latency"
    ],
    "Value": [
        total_calls,
        strong_model_calls,
        weak_model_calls,
        f"${total_cost:.4f}",
        f"{total_latency:.2f} s"
    ]
}
summary_df = pd.DataFrame(summary_data)
st.table(summary_df)
