import streamlit as st
import requests
import os
from file_store import save_file
from gtts import gTTS
from tempfile import NamedTemporaryFile
from llms.provider_factory import ProviderFactory

try:
    import speech_recognition as sr
    HAS_SR = True
except Exception:
    HAS_SR = False

API_URL = os.getenv("FASTAPI_URL", "http://fastapi:8000/chat/")

st.set_page_config(page_title="Chat with Agent", layout="centered")
st.title("Streamlit Interface (via FastAPI)")

provider_factory = ProviderFactory()
available_providers = provider_factory.get_all_providers_names()

with st.sidebar:
    api_key = st.text_input("Enter API Key", type='password')
    provider = st.selectbox("Select Provider", available_providers)
    
    try:
        available_models = provider_factory.get_all_models_for_provider(provider)
        model_options = [model for model in available_models]
        model_id = st.selectbox("Select Model", model_options)
    except:
        model_id = st.text_input("Model ID")
    
    selected_tool = st.selectbox("Select Tool", ["None", "Brave Search", "Crawl AI", "Serp Tool", "Contract Parser"])

    tool_config = {}
    if selected_tool == "Brave Search":
        brave_key = st.text_input("Brave API Key (optional)", type='password')
        tool_config["brave_key"] = brave_key
    elif selected_tool == "Serp Tool":
        serp_key = st.text_input("Serp API Key (optional)", type='password')
        tool_config["serp_key"] = serp_key
    elif selected_tool == "Contract Parser":
        uploaded_contract = st.file_uploader("Upload Contract (PDF)", type=["pdf", "docx", "txt"])
        if uploaded_contract:
            file_path = save_file(uploaded_contract)
            st.success(f"Saved at: {file_path}")
            tool_config["contract_parser"] = file_path

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

def call_api(prompt_text):
    payload = {
        "message": prompt_text,
        "provider": provider,
        "api_key": api_key,
        "id": model_id or "local",
        "tool": selected_tool,
        "tool_config": tool_config
    }
    try:
        res = requests.post(API_URL, json=payload, timeout=60)
    except Exception as e:
        return {"error": f"Request failed: {e}"}

    if res.status_code != 200:
        return {"error": res.text}

    try:
        data = res.json()
    except Exception as e:
        return {"error": f"Invalid JSON from API: {e}"}
    return data

if prompt := st.chat_input("Type your message..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if not api_key or not model_id:
        st.error("Please enter API key and model ID in the sidebar.")
    else:
        with st.spinner("Thinking..."):
            data = call_api(prompt)

        if "response" in data:
            reply = data["response"]
            st.chat_message("assistant").markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        else:
            st.error("API Error")
            st.write(data.get("error", data))

if st.button("Speak"):
    if not HAS_SR:
        st.warning("SpeechRecognition is not installed here. Run locally or install PyAudio.")
    else:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.write("Listening...")
            audio = recognizer.listen(source, phrase_time_limit=8)
        try:
            spoken_text = recognizer.recognize_google(audio)
            st.success(f"You said: {spoken_text}")
            with st.spinner("Thinking..."):
                data = call_api(spoken_text)
            if "response" in data:
                reply = data["response"]
                st.chat_message("assistant").markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
            else:
                st.error("API Error")
                st.write(data.get("error", data))
        except Exception as e:
            st.error(f"Could not understand audio: {e}")

if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    last_reply = st.session_state.messages[-1]["content"]
    if st.button("Play last reply"):
        tts = gTTS(last_reply)
        audio_file = NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(audio_file.name)
        audio_bytes = open(audio_file.name, "rb").read()
        st.audio(audio_bytes, format="audio/mp3")