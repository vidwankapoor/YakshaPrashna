import streamlit as st
import pandas as pd
import requests
import json

# Google Gemini API Key from Streamlit secrets
gemini_api_key = st.secrets['gemini']["gemini_api_key"]


# Clear Cache Button
if st.button("Clear Cache"):
    st.cache_data.clear()
    st.success("Cache has been cleared!")

# Streamlit UI setup
st.set_page_config(page_title="YakshaPrathna", layout="centered")

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

st.markdown("""<h2 style='text-align: center;color: white';>YakshaPrathna </h2>""", unsafe_allow_html=True)

# Function to get Gemini response
def get_final_gemini_answer(prompt):
    gemini_api_key = st.secrets["gemini"]["gemini_api_key"]
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash-lite-001:generateContent?key={gemini_api_key}"


    
    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    response = requests.post(url, headers=headers, json=payload)

    
    if response.status_code == 200:
        return response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
    else:
        return f"Error: {response.status_code} - {response.text}"


def enhance_prompt_with_gemini(user_input):
    gemini_api_key = st.secrets["gemini"]["gemini_api_key"]
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash-lite-001:generateContent?key={gemini_api_key}"

    headers = {
        "Content-Type": "application/json"
    }

    # Dynamic rewriter instruction
    enhancement_prompt = f"""
You are a helpful AI prompt rewriter. 
Take the following casual or unstructured user input and rewrite it as a clear, professional, and AI-friendly prompt 
that can be directly given to a chatbot like ChatGPT.

User Input: "{user_input}"
Enhanced Prompt:
"""

    payload = {
        "contents": [
            {"parts": [{"text": enhancement_prompt}]}
        ]
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
    else:
        return f"Error: {response.status_code} - {response.text}"

# Main input and processing
user_input = st.text_input("You:", placeholder="Enter a casual or broken prompt in Hinglish or English")

if user_input:
    with st.spinner("Getting Response..."):
        ai_prompt = enhance_prompt_with_gemini(user_input)
        gemini_output = get_final_gemini_answer(user_input)

    st.markdown("""
        <div class="chat-container">
            <div class="user-message"><b>Your Prompt</b>: {}</div>
            <div class="ai-message"><b>AI Friendly Prompt</b>: {}</div>
            <div class="ai-message"><b>Answer:</b><br>{}</div>
        </div>
    """.format(user_input,ai_prompt,gemini_output), unsafe_allow_html=True)

# Load and display sample prompts
try:
    df = pd.read_csv("sample_casual_input.csv")
    st.subheader("Sample Casual Prompts")
    st.dataframe(df)
except:
    st.warning("sample_casual_input.csv file not found.")
