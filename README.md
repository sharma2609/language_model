# Multilingual AI Translator & Transliterator

A Streamlit application that provides real-time translation, transliteration, speech-to-text, and text-to-speech across five languages using local AI models.

## Features

| Capability | Details |
|---|---|
| **Translation** | English, Hindi, Japanese, Chinese (Simplified), Russian |
| **Transliteration** | Hindi (Devanagari) ↔ English (Latin) |
| **Speech-to-Text** | Upload audio — auto-transcribe with language detection |
| **Text-to-Speech** | Play translated/transliterated output as speech |

## Tech Stack

| Component | Model / Library |
|---|---|
| Translation | [Facebook NLLB-200 Distilled 600M](https://huggingface.co/facebook/nllb-200-distilled-600M) |
| Speech-to-Text | [OpenAI Whisper tiny](https://github.com/guillaumekln/faster-whisper) (via faster-whisper) |
| Text-to-Speech | [Google TTS (gTTS)](https://github.com/pndurette/gTTS) |
| Transliteration | [Aksharamukha](https://github.com/virtualvinodh/aksharamukha) |
| Framework | [Streamlit](https://streamlit.io) |

## Quick Start

### Prerequisites

- Python 3.10+
- FFmpeg (`apt install ffmpeg` on Ubuntu, `brew install ffmpeg` on macOS)

### Installation

```bash
git clone <repository-url>
cd <repository-name>
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

Models are downloaded on first use (~1.5 GB). After caching, subsequent starts are instant.

## Usage

1. Select **Translate** or **Transliterate** in the sidebar.
2. Choose source and target languages.
3. Type text or upload an audio file for transcription.
4. Click **Process** to get the result.
5. Click **Generate Audio** to hear the output spoken aloud.
6. Use **Clear Results** to reset.

## Project Structure

```
├── app.py              # Main application
├── requirements.txt    # Python dependencies
├── packages.txt        # System dependencies (FFmpeg, libsndfile)
├── .gitattributes      # Git configuration
└── README.md           # This file
```

## Deployment

**Streamlit Cloud** — Set `app.py` as entry point and include `packages.txt`. A GPU-backed or high-memory plan is recommended due to model size (~3 GB RAM at inference).

**Docker** — Use a CUDA base image and install both Python and system dependencies listed in `packages.txt`.

## Performance Notes

- Models are cached via `@st.cache_resource` and only loaded on first translation/transcription request.
- Whisper runs on CPU with `int8` quantization — transcription may be slower on resource-constrained environments.
- The NLLB model uses greedy decoding (`num_beams=1`) to minimise memory and latency.

## License

This project is open source. Review individual model licenses (NLLB-200, Whisper) for commercial use restrictions.
