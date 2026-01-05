# Multilingual AI Translator & Transliterator

A Streamlit application that provides translation and transliteration capabilities across multiple languages using AI models. The app supports both text input and audio transcription with speech synthesis output.

## Features

- **Translation**: Translate text between English, Hindi, Japanese, Chinese (Simplified), and Russian
- **Transliteration**: Convert text between Hindi (Devanagari) and English (Latin) scripts
- **Speech-to-Text**: Upload audio files for automatic transcription
- **Text-to-Speech**: Generate audio output for translated/transliterated text
- **Multi-language Support**: Detect and process multiple languages

## Available Versions

### 1. Full Version (`app.py`)

Uses local AI models for maximum functionality:

- **Translation**: Facebook NLLB-200 model
- **Speech Recognition**: OpenAI Whisper model
- **Pros**: Full offline functionality, high accuracy
- **Cons**: High memory usage (~3GB+), slower startup

### 2. Lightweight Version (`app_lightweight.py`)

API-based approach optimized for cloud deployment:

- **Translation**: Hugging Face Inference API
- **Speech Recognition**: Removed (text-only)
- **Pros**: Low memory usage, fast startup, cloud-friendly
- **Cons**: Requires API tokens, internet dependency

## Quick Start

### Local Development

```bash
# Clone the repository
git clone <repository-url>
cd <repository-name>

# Install dependencies (full version)
pip install -r requirements.txt

# Or install lightweight dependencies
pip install -r requirements_lightweight.txt

# Run the application
streamlit run app.py
# Or run lightweight version
streamlit run app_lightweight.py
```

### Streamlit Cloud Deployment

#### Option 1: Lightweight Version (Recommended)

1. Use `app_lightweight.py` as your main file
2. Use `requirements_lightweight.txt` for dependencies
3. Add `packages.txt` for system dependencies
4. Deploy directly to Streamlit Cloud

#### Option 2: Full Version (Advanced)

1. Use `app.py` as your main file
2. Use `requirements.txt` for dependencies
3. Add `packages.txt` for system dependencies
4. May require memory optimization or paid hosting

## Configuration

### System Dependencies (`packages.txt`)

```
ffmpeg
libsndfile1
```

### Environment Variables (Optional)

For the lightweight version with full API functionality:

```
HF_TOKEN=your_hugging_face_api_token
```

## Supported Languages

| Language             | Translation | Transliteration      | TTS Support |
| -------------------- | ----------- | -------------------- | ----------- |
| English              | ✅          | ✅ (to/from Hindi)   | ✅          |
| Hindi                | ✅          | ✅ (to/from English) | ✅          |
| Japanese             | ✅          | ❌                   | ✅          |
| Chinese (Simplified) | ✅          | ❌                   | ✅          |
| Russian              | ✅          | ❌                   | ✅          |

## Technical Details

### Models Used (Full Version)

- **Translation**: `facebook/nllb-200-distilled-600M`
- **Speech-to-Text**: `openai/whisper-tiny` (optimized for deployment)
- **Text-to-Speech**: Google Text-to-Speech (gTTS)
- **Transliteration**: Aksharamukha library

### Memory Requirements

- **Full Version**: ~3GB RAM (may exceed Streamlit Cloud limits)
- **Lightweight Version**: ~100MB RAM (Streamlit Cloud compatible)

## Troubleshooting

### Common Deployment Issues

1. **Memory errors**: Use lightweight version or upgrade hosting plan
2. **Model loading timeouts**: Reduce model size or use API version
3. **CUDA errors**: Ensure CPU-only PyTorch installation
4. **Audio processing errors**: Verify `packages.txt` is included

### Performance Optimization

- Use `torch.float16` for GPU deployments
- Implement model caching with `@st.cache_resource`
- Reduce beam search parameters for faster inference
- Use smaller model variants for deployment

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test both versions
5. Submit a pull request

## License

This project is open source. Please check individual model licenses for commercial use.

## Acknowledgments

- Facebook AI for NLLB translation models
- OpenAI for Whisper speech recognition
- Google for Text-to-Speech services
- Aksharamukha for transliteration capabilities
