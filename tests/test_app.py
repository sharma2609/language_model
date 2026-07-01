import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch, sentinel

# --- Mock heavy / unavailable dependencies before importing app ---

aksharamukha_transliterate_mock = MagicMock()
aksharamukha_transliterate_mock.process.side_effect = (
    lambda src, tgt, text: text
)

aksharamukha_mock = MagicMock()
aksharamukha_mock.transliterate = aksharamukha_transliterate_mock

streamlit_mock = MagicMock()
streamlit_mock.session_state = {}
streamlit_mock.cache_resource = lambda **kw: (lambda f: f)
streamlit_mock.set_page_config = lambda **kw: None
streamlit_mock.title = lambda x: None
streamlit_mock.markdown = lambda x: None
streamlit_mock.header = lambda x: None
streamlit_mock.selectbox = lambda label, options, **kw: options[0]
streamlit_mock.file_uploader = lambda label, **kw: None
streamlit_mock.button = lambda label: False
streamlit_mock.text_area = lambda label, **kw: ""
streamlit_mock.columns = lambda n: (MagicMock(), MagicMock())
streamlit_mock.spinner = lambda text: MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
streamlit_mock.rerun = lambda: None
streamlit_mock.info = lambda x: None
streamlit_mock.warning = lambda x: None
streamlit_mock.error = lambda x: None
streamlit_mock.audio = lambda x, **kw: None
streamlit_mock.markdown = lambda x: None

torch_mock = MagicMock()
torch_mock.cuda.is_available.return_value = False
torch_mock.float16 = "float16"
torch_mock.float32 = "float32"

transformers_mock = MagicMock()
transformers_mock.AutoTokenizer = MagicMock()
transformers_mock.AutoModelForSeq2SeqLM = MagicMock()

faster_whisper_mock = MagicMock()
faster_whisper_mock.WhisperModel = MagicMock()

gtts_mock = MagicMock()
gtts_mock.gTTS = MagicMock()

to_mock = [
    ("streamlit", streamlit_mock),
    ("torch", torch_mock),
    ("transformers", transformers_mock),
    ("faster_whisper", faster_whisper_mock),
    ("gtts", gtts_mock),
    ("aksharamukha", aksharamukha_mock),
]

for name, mock in to_mock:
    if name not in sys.modules:
        sys.modules[name] = mock

# Also mock submodules
sys.modules["transformers"] = transformers_mock

# Now safe to import from app
from app import (
    validate_audio_file,
    transliterate_text,
    _cleanup_audio_file,
    _cleanup_all_temp_files,
    _temp_audio_files,
    ALLOWED_AUDIO_TYPES,
    MAX_AUDIO_BYTES,
    MAX_AUDIO_MB,
)


class TestValidateAudioFile(unittest.TestCase):
    def setUp(self):
        self.mock_file = MagicMock()
        self.mock_file.size = 1024 * 1024  # 1 MB
        self.mock_file.type = "audio/mpeg"

    def test_valid_file_returns_none(self):
        self.assertIsNone(validate_audio_file(self.mock_file))

    def test_file_exceeds_size_limit(self):
        self.mock_file.size = MAX_AUDIO_BYTES + 1
        error = validate_audio_file(self.mock_file)
        self.assertIsNotNone(error)
        self.assertIn(str(MAX_AUDIO_MB), error)

    def test_file_at_size_limit(self):
        self.mock_file.size = MAX_AUDIO_BYTES
        self.assertIsNone(validate_audio_file(self.mock_file))

    def test_rejects_unsupported_mime(self):
        self.mock_file.type = "video/mp4"
        error = validate_audio_file(self.mock_file)
        self.assertIsNotNone(error)
        self.assertIn("Unsupported audio format", error)

    def test_allows_all_defined_mimes(self):
        for mime in ALLOWED_AUDIO_TYPES:
            self.mock_file.type = mime
            self.assertIsNone(validate_audio_file(self.mock_file))


class TestTransliterateText(unittest.TestCase):
    def test_same_language_noop_english(self):
        self.assertEqual(transliterate_text("Hello", "English", "English"), "Hello")

    def test_same_language_noop_hindi(self):
        self.assertEqual(transliterate_text("नमस्ते", "Hindi", "Hindi"), "नमस्ते")

    def test_empty_text_returns_empty(self):
        self.assertEqual(transliterate_text("", "Hindi", "English"), "")
        self.assertEqual(transliterate_text("   ", "English", "Hindi"), "")

    def test_unpaired_languages_returns_original(self):
        self.assertEqual(transliterate_text("Hello", "English", "Japanese"), "Hello")
        self.assertEqual(transliterate_text("Hello", "Japanese", "Hindi"), "Hello")


class TestCleanupAudioFile(unittest.TestCase):
    def test_removes_existing_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name
        self.assertTrue(os.path.exists(path))
        _cleanup_audio_file(path)
        self.assertFalse(os.path.exists(path))

    def test_nonexistent_file_does_not_raise(self):
        try:
            _cleanup_audio_file("/tmp/nonexistent_file_xyz.mp3")
        except Exception:
            self.fail("_cleanup_audio_file raised on nonexistent file")


class TestTempFileTracking(unittest.TestCase):
    def tearDown(self):
        _temp_audio_files.clear()

    def test_cleanup_all_temp_files(self):
        tmp1 = tempfile.NamedTemporaryFile(delete=False)
        tmp2 = tempfile.NamedTemporaryFile(delete=False)
        _temp_audio_files.add(tmp1.name)
        _temp_audio_files.add(tmp2.name)
        tmp1.close()
        tmp2.close()

        _cleanup_all_temp_files()
        self.assertFalse(os.path.exists(tmp1.name))
        self.assertFalse(os.path.exists(tmp2.name))
        self.assertEqual(len(_temp_audio_files), 0)


if __name__ == "__main__":
    unittest.main()
