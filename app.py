import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from faster_whisper import WhisperModel
from gtts import gTTS
from aksharamukha import transliterate
import tempfile
import os
import gc
import atexit
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# --- App Configuration ---
st.set_page_config(
    page_title="Multilingual AI Translator",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Constants ---
MAX_AUDIO_MB = 25
MAX_AUDIO_BYTES = MAX_AUDIO_MB * 1024 * 1024

ALLOWED_AUDIO_TYPES = {
    "audio/mpeg",
    "audio/wav",
    "audio/x-wav",
    "audio/mp4",
    "audio/x-m4a",
    "audio/ogg",
    "audio/webm",
}

LANG_CODES: dict[str, str] = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "Japanese": "jpn_Jpan",
    "Chinese (Simplified)": "zho_Hans",
    "Russian": "rus_Cyrl",
}

GTT_LANG_MAP: dict[str, str] = {
    "English": "en",
    "Hindi": "hi",
    "Japanese": "ja",
    "Chinese (Simplified)": "zh-cn",
    "Russian": "ru",
}


# --- Cached Model Loading (lazy — loaded on first use) ---

@st.cache_resource(show_spinner="Loading translation model (~1.5GB)...")
def load_translation_model() -> tuple:  # (AutoTokenizer, AutoModelForSeq2SeqLM)
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    return tokenizer, model


@st.cache_resource(show_spinner="Loading speech-to-text model (Whisper)...")
def load_stt_model() -> WhisperModel:
    model_size = "tiny"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if torch.cuda.is_available() else "int8"
    return WhisperModel(model_size, device=device, compute_type=compute_type)


# --- Core Functions ---

def validate_audio_file(uploaded_file) -> Optional[str]:
    if uploaded_file.size > MAX_AUDIO_BYTES:
        return (
            f"File exceeds {MAX_AUDIO_MB} MB limit "
            f"({uploaded_file.size / 1024 / 1024:.1f} MB)."
        )

    mime_type = uploaded_file.type
    if mime_type not in ALLOWED_AUDIO_TYPES:
        return f"Unsupported audio format: {mime_type}."

    return None


def translate(
    text: str,
    src_lang_code: str,
    tgt_lang_code: str,
    tokenizer,
    model,
    max_length: int = 512,
) -> str:
    if not text.strip():
        return ""

    tokenizer.src_lang = src_lang_code
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang_code),
        max_length=max_length,
        num_beams=1,
        do_sample=False,
    )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]


def transliterate_text(text: str, src_lang: str, tgt_lang: str) -> str:
    if not text.strip():
        return ""

    if src_lang == "Hindi" and tgt_lang == "English":
        return transliterate.process("Devanagari", "ITRANS", text)
    elif src_lang == "English" and tgt_lang == "Hindi":
        return transliterate.process("ITRANS", "Devanagari", text)

    return text


def speech_to_text(audio_path: str, whisper_model: WhisperModel) -> tuple[str, str]:
    segments, info = whisper_model.transcribe(audio_path, beam_size=1)
    detected_lang = info.language
    transcribed_text = "".join(segment.text for segment in segments).strip()
    return transcribed_text, detected_lang


# --- Helper: clean up stale TTS temp file ---

_temp_audio_files: set[str] = set()


def _cleanup_audio_file(path: str) -> None:
    try:
        os.remove(path)
    except Exception as e:
        logger.warning("Failed to remove temp file %s: %s", path, e)


def _cleanup_audio_path():
    path = st.session_state.get("audio_path")
    if path:
        _cleanup_audio_file(path)
        _temp_audio_files.discard(path)
    st.session_state.audio_path = None


def _cleanup_all_temp_files():
    for path in list(_temp_audio_files):
        _cleanup_audio_file(path)
    _temp_audio_files.clear()


atexit.register(_cleanup_all_temp_files)


def text_to_speech(text: str, lang_name: str) -> Optional[str]:
    if not text.strip():
        return None

    tts_lang_code = GTT_LANG_MAP.get(lang_name, "en")
    fp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    try:
        tts = gTTS(text, lang=tts_lang_code)
        tts.save(fp.name)
    except Exception:
        _cleanup_audio_file(fp.name)
        raise
    _temp_audio_files.add(fp.name)
    return fp.name


# --- Streamlit UI ---

st.title("🌐 Multilingual AI Translator & Transliterator")
st.markdown(
    "Translate or transliterate text and speech across multiple languages. "
    "Upload an audio file or type in the text box."
)

# Sidebar
with st.sidebar:
    st.header("⚙️ Options")
    task = st.selectbox("Select Task", ["Translate", "Transliterate"])

    if task == "Translate":
        source_lang = st.selectbox("Source Language", options=list(LANG_CODES.keys()))
        target_lang = st.selectbox(
            "Target Language", options=list(LANG_CODES.keys()), index=1
        )
    else:
        st.info(
            "Transliteration is supported between English (Latin script) "
            "and Hindi (Devanagari script)."
        )
        source_lang = st.selectbox("Source Language", options=["English", "Hindi"])
        target_lang = st.selectbox(
            "Target Language", options=["English", "Hindi"], index=1
        )

    st.markdown("---")
    st.header("🎤 Audio Input")
    uploaded_audio = st.file_uploader(
        "Upload an audio file (MP3, WAV, M4A, OGG)...",
        type=["mp3", "wav", "m4a", "ogg"],
    )

    st.markdown("---")
    if st.button("🗑️ Clear Results"):
        _cleanup_audio_path()
        st.session_state.pop("output_text", None)
        st.session_state.pop("target_lang_for_tts", None)
        st.rerun()

st.markdown("---")

# Main layout
col1, col2 = st.columns(2)

with col1:
    st.header("Input")
    input_text = st.text_area("Enter text here:", height=200)

    if st.button("Process"):
        _cleanup_audio_path()

        if not input_text.strip() and not uploaded_audio:
            st.warning("Please enter text or upload an audio file.")
        else:
            final_input_text = input_text

            if uploaded_audio:
                error = validate_audio_file(uploaded_audio)
                if error:
                    st.error(error)
                else:
                    with st.spinner("Transcribing audio..."):
                        audio_path: Optional[str] = None
                        try:
                            fp = tempfile.NamedTemporaryFile(
                                delete=False,
                                suffix=os.path.splitext(uploaded_audio.name)[1],
                            )
                            fp.write(uploaded_audio.getvalue())
                            audio_path = fp.name
                            fp.close()

                            whisper_model = load_stt_model()
                            transcribed_text, detected_lang = speech_to_text(
                                audio_path, whisper_model
                            )
                            final_input_text = transcribed_text
                            st.info(f"Detected audio language: {detected_lang}")
                        except Exception as e:
                            st.error(f"Transcription failed: {e}")
                        finally:
                            if audio_path and os.path.exists(audio_path):
                                os.remove(audio_path)

            if final_input_text.strip():
                with st.spinner(f"{task}ing..."):
                    try:
                        if task == "Translate":
                            tokenizer, model = load_translation_model()
                            src_code = LANG_CODES[source_lang]
                            tgt_code = LANG_CODES[target_lang]
                            output_text = translate(
                                final_input_text,
                                src_code,
                                tgt_code,
                                tokenizer,
                                model,
                            )
                        else:
                            if {source_lang, target_lang} != {"English", "Hindi"}:
                                st.warning(
                                    "Transliteration only supports English ↔ Hindi. "
                                    "Returning original text."
                                )
                            output_text = transliterate_text(
                                final_input_text, source_lang, target_lang
                            )

                        st.session_state.output_text = output_text
                        st.session_state.target_lang_for_tts = target_lang
                    except Exception as e:
                        st.error(f"{task} failed: {e}")
                    finally:
                        if torch.cuda.is_available():
                            gc.collect()
                            torch.cuda.empty_cache()

with col2:
    st.header("Output")
    if "output_text" in st.session_state:
        output = st.session_state.output_text
        st.text_area("Result:", value=output, height=200, disabled=True)

        if st.button("🔊 Generate Audio"):
            with st.spinner("Generating audio..."):
                try:
                    _cleanup_audio_path()
                    path = text_to_speech(output, st.session_state.target_lang_for_tts)
                    if path:
                        st.session_state.audio_path = path
                        st.rerun()
                except Exception as e:
                    st.error(f"Audio generation failed: {e}")

        if st.session_state.get("audio_path") and os.path.exists(
            st.session_state.audio_path
        ):
            st.audio(st.session_state.audio_path, format="audio/mp3")
