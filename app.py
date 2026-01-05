import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from faster_whisper import WhisperModel
from gtts import gTTS
from aksharamukha import transliterate
import tempfile
from langdetect import detect, DetectorFactory
import os

# --- App Configuration ---
st.set_page_config(
    page_title="Multilingual AI Translator",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Caching and Model Loading ---

# Cache the translation model and tokenizer
@st.cache_resource
def load_translation_model():
    """Loads and caches the NLLB translation model and tokenizer."""
    model_name = "facebook/nllb-200-distilled-600M"
    st.write(f"Loading translation model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    st.write("Translation model loaded successfully.")
    return tokenizer, model

# Cache the speech-to-text model
@st.cache_resource
def load_stt_model():
    """Loads and caches the Whisper speech-to-text model."""
    model_size = "base" # CHANGED: Was "small", "base" is much faster
    st.write(f"Loading speech-to-text model: Whisper ({model_size})...")
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if torch.cuda.is_available() else "int8"
    whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
    st.write("Speech-to-text model loaded successfully.")
    return whisper_model

# Load models
tokenizer, model = load_translation_model()
whisper_model = load_stt_model()

# --- Language and Task Dictionaries ---

# Ensure langdetect produces consistent results
DetectorFactory.seed = 0

LANG_CODES = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "Japanese": "jpn_Jpan",
    "Chinese (Simplified)": "zho_Hans",
    "Russian": "rus_Cyrl"
}
LANG_CODES_REV = {v: k for k, v in LANG_CODES.items()}

# Mapping for gTTS
GTT_LANG_MAP = {
    'English': 'en',
    'Hindi': 'hi',
    'Japanese': 'ja',
    'Chinese (Simplified)': 'zh-cn',
    'Russian': 'ru'
}


# --- Core Functions ---

def translate(text, src_lang_code, tgt_lang_code, max_length=512):
    """Translates text using the loaded NLLB model."""
    if not text.strip():
        return ""
    tokenizer.src_lang = src_lang_code
    inputs = tokenizer(text, return_tensors="pt")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    translated_tokens = model.generate(
        **inputs.to(device),
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang_code), # FIXED THIS LINE
        max_length=max_length
    )
    result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return result

def transliterate_text(text, src_lang, tgt_lang):
    """Transliterates text between Hindi (Devanagari) and English (Latin)."""
    if src_lang == 'Hindi' and tgt_lang == 'English':
        # Devanagari to Latin script
        return transliterate.process('Devanagari', 'ITRANS', text)
    elif src_lang == 'English' and tgt_lang == 'Hindi':
        # Latin to Devanagari script (assumes ITRANS input)
        return transliterate.process('ITRANS', 'Devanagari', text)
    return text # Return original text if no valid transliteration pair

def speech_to_text(audio_path):
    """Transcribes audio file to text using Whisper."""
    segments, info = whisper_model.transcribe(audio_path, beam_size=5)
    detected_lang = info.language
    transcribed_text = "".join([segment.text for segment in segments]).strip()
    return transcribed_text, detected_lang

def text_to_speech(text, lang_name):
    """Converts text to speech using gTTS and returns the audio file path."""
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

st.title("🌐 Multilingual AI Translator & Transliterator")
st.markdown("Translate or transliterate text and speech across multiple languages. Upload an audio file or type in the text box.")

# --- Sidebar for options ---
with st.sidebar:
    st.header("⚙️ Options")
    task = st.selectbox("Select Task", ["Translate", "Transliterate"])
    
    if task == "Translate":
        source_lang = st.selectbox("Source Language", options=list(LANG_CODES.keys()))
        target_lang = st.selectbox("Target Language", options=list(LANG_CODES.keys()), index=1)
    else: # Transliterate
        st.info("Transliteration is supported between English (Latin script) and Hindi (Devanagari script).")
        source_lang = st.selectbox("Source Language", options=["English", "Hindi"])
        target_lang = st.selectbox("Target Language", options=["English", "Hindi"], index=1)
    
    st.markdown("---")
    st.header("🎤 Audio Input")
    uploaded_audio = st.file_uploader("Upload an audio file (MP3, WAV, M4A)...", type=["mp3", "wav", "m4a", "ogg"])

st.markdown("---")

# --- Main app layout ---
col1, col2 = st.columns(2)

with col1:
    st.header("Input")
    input_text = st.text_area("Enter text here:", height=200)
    
    if st.button("Process"):
        if not input_text and not uploaded_audio:
            st.warning("Please enter text or upload an audio file.")
        else:
            final_input_text = input_text
            
            # Process audio if uploaded
            if uploaded_audio:
                with st.spinner("Transcribing audio..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_audio.name)[1]) as fp:
                        fp.write(uploaded_audio.getvalue())
                        audio_path = fp.name
                    
                    transcribed_text, detected_lang = speech_to_text(audio_path)
                    final_input_text = transcribed_text
                    st.info(f"Detected audio language: {detected_lang}")
                    st.text_area("Transcribed Text:", value=final_input_text, height=100, disabled=True)
                    os.remove(audio_path) # Clean up temp file

            # Perform the selected task
            with st.spinner(f"{task}ing..."):
                output_text = ""
                if task == "Translate":
                    src_code = LANG_CODES[source_lang]
                    tgt_code = LANG_CODES[target_lang]
                    output_text = translate(final_input_text, src_code, tgt_code)
                elif task == "Transliterate":
                    output_text = transliterate_text(final_input_text, source_lang, target_lang)
            
            # Store results in session state to display in the other column
            st.session_state.output_text = output_text
            st.session_state.target_lang_for_tts = target_lang

with col2:
    st.header("Output")
    if 'output_text' in st.session_state:
        output_text_to_display = st.session_state.output_text
        st.text_area("Result:", value=output_text_to_display, height=200, disabled=True)

        with st.spinner("Generating audio output..."):
            tts_audio_path = text_to_speech(output_text_to_display, st.session_state.target_lang_for_tts)
            if tts_audio_path:
                st.audio(tts_audio_path, format="audio/mp3")
                os.remove(tts_audio_path) # Clean up temp file


