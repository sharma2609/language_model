# Multilingual AI Translator & Transliterator

Translate or transliterate text and speech across five languages using local AI models, with optional GPU acceleration.

## Overview

This is a single-page Streamlit application that runs entirely locally — no API keys, no cloud dependencies for translation or transcription. It wraps Facebook's NLLB-200 distilled model for text translation, OpenAI's Whisper (via faster-whisper) for speech-to-text with automatic language detection, Google TTS for spoken output, and Aksharamukha for Devanagari↔Latin transliteration.

The primary use case is offline-capable multilingual communication: upload a voice memo or type text in one language and get translation, transliteration, and spoken playback in another — all on a laptop without sending data to any external service (gTTS for audio playback requires internet; translation and transcription do not).

## Features

- **Text translation** across English, Hindi, Japanese, Chinese (Simplified), and Russian using NLLB-200 Distilled 600M
- **Transliteration** between Hindi (Devanagari) and English (Latin script) via Aksharamukha
- **Speech-to-text** via Whisper tiny — upload an audio file and get automatic transcription with language detection
- **Text-to-speech** playback of translated or transliterated output via gTTS
- **GPU auto-detection** — both NLLB and Whisper use CUDA when available, with automatic fallback to CPU
- **No API keys required** — all models download and cache locally on first use

## Tech Stack

| Component | Library / Model |
|---|---|
| Framework | [Streamlit](https://streamlit.io) |
| Translation | [Facebook NLLB-200 Distilled 600M](https://huggingface.co/facebook/nllb-200-distilled-600M) (via `transformers`) |
| Speech-to-text | [faster-whisper](https://github.com/guillaumekln/faster-whisper) (Whisper tiny, CTranslate2 backend) |
| Text-to-speech | [gTTS](https://github.com/pndurette/gTTS) (Google Translate TTS) |
| Transliteration | [Aksharamukha](https://github.com/virtualvinodh/aksharamukha) |
| ML runtime | PyTorch (CUDA auto-detect) |

## Installation

### Prerequisites

- Python 3.10+
- FFmpeg
- libsndfile1

Install system dependencies:

```bash
# Ubuntu / Debian
sudo apt install ffmpeg libsndfile1

# macOS
brew install ffmpeg libsndfile
```

### Set up the project

```bash
git clone <repository-url>
cd <repository-name>

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Models are downloaded from Hugging Face on first use (~1.5 GB for NLLB, ~150 MB for Whisper). Subsequent starts use a local cache.

## Usage

```bash
streamlit run app.py
```

The browser opens to a two-column layout. From the sidebar:

1. **Select Task** — `Translate` or `Transliterate`
2. **Choose languages** — source and target from the available options
3. **Type text** in the input area, or **upload an audio file** (MP3, WAV, M4A, OGG; max 25 MB)
4. Click **Process** — audio is transcribed first, then translation/transliteration runs
5. Click **Generate Audio** to hear the output spoken aloud
6. **Clear Results** resets the session

### Example workflow

```
Task:         Translate
Source:       English
Target:       Hindi
Input text:   "Good morning, how are you?"
→ Output:     "शुभ प्रभात, आप कैसे हैं?"
→ Generate Audio:  plays Hindi TTS
```

## Configuration

The application has no environment variables or configuration files. All settings are controlled via the Streamlit sidebar at runtime.

| Setting | Options | Notes |
|---|---|---|
| Task | Translate / Transliterate | Transliteration only works between English and Hindi |
| Source language | Depends on task | 5 languages for translate, 2 for transliterate |
| Target language | Depends on task | Defaults to Hindi |
| Audio input | File upload | 25 MB limit; supports MP3, WAV, M4A, OGG |

Model behavior is determined at import time:
- **GPU**: used automatically if `torch.cuda.is_available()` returns `True`
- **CPU fallback**: NLLB uses `float32`, Whisper uses `int8` quantization
- **Decoding**: greedy (`num_beams=1`, `do_sample=False`) for lower memory usage

## Project Structure

```
├── app.py              # Single-page Streamlit application
├── tests/
│   ├── __init__.py
│   └── test_app.py     # Unit tests with mocked ML dependencies
├── requirements.txt    # Python dependencies
├── packages.txt        # System dependencies (apt-get install list)
├── .gitignore
└── README.md
```

## Testing

Tests cover the core logic that doesn't require downloading models: file validation, transliteration routing, temp-file cleanup, and the atexit hook. All external dependencies are mocked so tests run offline in seconds.

```bash
# From the project root
python -m pytest tests/
```

## License

This project is open source. The underlying models (NLLB-200, Whisper, gTTS, Aksharamukha) each have their own licenses — review their terms for commercial use restrictions. No license file is included in this repository.
