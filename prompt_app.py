import streamlit as st
from transformers import pipeline
import pandas as pd
import openai

# Initialize OpenAI client with new API format
client = openai.OpenAI(api_key=st.secrets["openai"]["api_key"])


if st.button("Clear Cache"):
    st.cache_data.clear()
    st.success("Cache has been cleared!")

# Load model
try:
    pipe = pipeline(
        "text2text-generation",
        model="vidwankapoor/flan-t5-prompt-rewriter",
        tokenizer="vidwankapoor/flan-t5-prompt-rewriter",
        use_fast=False,
        revision="main"
    )
except Exception as e:
    st.error("Error loading model:")
    st.exception(e)

# Streamlit UI setup
st.set_page_config(page_title="AskPerfect", layout="centered")

st.markdown("""
    <style>
         html, body, [data-testid="stApp"] {
            background-color: #121212 !important;
        }
        .stTextInput > div > div > input {
            font-size: 16px;
            padding: 10px;
        }
        .chat-container {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        }
        .user-message, .ai-message {
            padding: 12px 16px;
            margin-bottom: 10px;
            border-radius: 8px;
        }
        .user-message {
            background-color: #d1e7dd;
            text-align: right;
        }
        .ai-message {
            background-color: #e2e3e5;
            text-align: left;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center;color: white;'>AskPerfect</h2>", unsafe_allow_html=True)

def enhance_prompt_with_chatgpt(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You enhance prompts to make them more ChatGPT-friendly, professional, and clear."},
            {"role": "user", "content": f"Improve this prompt: {prompt}"}
        ]
    )
    return response.choices[0].message.content.strip()

# ChatGPT Answer Generator (v1 API)
def get_final_chatgpt_answer(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# Main app input and response logic
user_input = st.text_input("You:", placeholder="Enter a casual or broken prompt in Hinglish or English")

if user_input:
    with st.spinner("Rewriting and enhancing your prompt..."):
        flan_output = pipe("rewrite prompt: " + user_input)
        flan_prompt = flan_output[0]['generated_text']

        gpt_enhanced_prompt = enhance_prompt_with_chatgpt(flan_prompt)
        final_answer = get_final_chatgpt_answer(gpt_enhanced_prompt)

    st.markdown("""
        <div class="chat-container">
            <div class="user-message">Your Prompt: {}</div>
            <div class="ai-message"><b>Step 1: Flan-T5 Rewritten Prompt:</b><br>{}</div>
            <div class="ai-message"><b>Step 2: ChatGPT Enhanced Prompt:</b><br>{}</div>
            <div class="ai-message"><b>Step 3: Final ChatGPT Answer:</b><br>{}</div>
        </div>
    """.format(user_input, flan_prompt, gpt_enhanced_prompt, final_answer), unsafe_allow_html=True)

# Load sample data
df = pd.read_csv("sample_casual_input.csv")
st.subheader("Sample Casual Prompts")
st.dataframe(df)
