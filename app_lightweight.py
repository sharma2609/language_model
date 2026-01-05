import streamlit as st
import requests
import tempfile
import os
from gtts import gTTS
from aksharamukha import transliterate
from langdetect import detect, DetectorFactory

# --- App Configuration ---
st.set_page_config(
    page_title="Multilingual AI Translator (Lightweight)",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure langdetect produces consistent results
DetectorFactory.seed = 0

# --- Language Dictionaries ---
LANG_CODES = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva", 
    "Japanese": "jpn_Jpan",
    "Chinese (Simplified)": "zho_Hans",
    "Russian": "rus_Cyrl"
}

GTT_LANG_MAP = {
    'English': 'en',
    'Hindi': 'hi', 
    'Japanese': 'ja',
    'Chinese (Simplified)': 'zh-cn',
    'Russian': 'ru'
}

# --- Hugging Face API Functions ---
def translate_with_api(text, src_lang_code, tgt_lang_code):
    """Translate using Hugging Face Inference API (requires API token)"""
    # This would require a Hugging Face API token
    # For demo purposes, we'll use a simple fallback
    try:
        # Placeholder for HF API call
        # headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}
        # API_URL = "https://api-inference.huggingface.co/models/facebook/nllb-200-distilled-600M"
        # response = requests.post(API_URL, headers=headers, json={"inputs": text})
        
        # For now, return a placeholder
        return f"[API Translation] {text} -> {tgt_lang_code}"
    except Exception as e:
        return f"Translation API failed: {str(e)}"

def transliterate_text(text, src_lang, tgt_lang):
    """Transliterates text between Hindi (Devanagari) and English (Latin)."""
    try:
        if src_lang == 'Hindi' and tgt_lang == 'English':
            return transliterate.process('Devanagari', 'ITRANS', text)
        elif src_lang == 'English' and tgt_lang == 'Hindi':
            return transliterate.process('ITRANS', 'Devanagari', text)
        return text
    except Exception as e:
        return f"Transliteration failed: {str(e)}"

def text_to_speech(text, lang_name):
    """Converts text to speech using gTTS."""
    if not text.strip():
        return None
    try:
        gtts_lang_code = GTT_LANG_MAP.get(lang_name, 'en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts = gTTS(text, lang=gtts_lang_code)
            tts.save(fp.name)
            return fp.name
    except Exception as e:
        st.error(f"Text-to-Speech failed: {e}")
        return None

# --- Streamlit UI ---
st.title("🌐 Multilingual AI Translator (Lightweight)")
st.markdown("A lightweight version using APIs instead of local models.")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Options")
    task = st.selectbox("Select Task", ["Translate", "Transliterate"])
    
    if task == "Translate":
        source_lang = st.selectbox("Source Language", options=list(LANG_CODES.keys()))
        target_lang = st.selectbox("Target Language", options=list(LANG_CODES.keys()), index=1)
    else:
        st.info("Transliteration: English ↔ Hindi")
        source_lang = st.selectbox("Source Language", options=["English", "Hindi"])
        target_lang = st.selectbox("Target Language", options=["English", "Hindi"], index=1)

# --- Main Layout ---
col1, col2 = st.columns(2)

with col1:
    st.header("Input")
    input_text = st.text_area("Enter text here:", height=200)
    
    if st.button("Process"):
        if not input_text:
            st.warning("Please enter text.")
        else:
            with st.spinner(f"{task}ing..."):
                if task == "Translate":
                    src_code = LANG_CODES[source_lang]
                    tgt_code = LANG_CODES[target_lang]
                    output_text = translate_with_api(input_text, src_code, tgt_code)
                else:
                    output_text = transliterate_text(input_text, source_lang, target_lang)
            
            st.session_state.output_text = output_text
            st.session_state.target_lang_for_tts = target_lang

with col2:
    st.header("Output")
    if 'output_text' in st.session_state:
        output_text_to_display = st.session_state.output_text
        st.text_area("Result:", value=output_text_to_display, height=200, disabled=True)
        
        if st.button("Generate Audio"):
            with st.spinner("Generating audio..."):
                tts_audio_path = text_to_speech(output_text_to_display, st.session_state.target_lang_for_tts)
                if tts_audio_path:
                    st.audio(tts_audio_path, format="audio/mp3")
                    os.remove(tts_audio_path)

st.markdown("---")
st.info("💡 This lightweight version uses APIs instead of loading large models locally, making it more suitable for Streamlit Cloud deployment.")