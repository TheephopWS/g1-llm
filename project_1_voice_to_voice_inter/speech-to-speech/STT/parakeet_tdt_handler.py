"""
Parakeet TDT Speech-to-Text Handler (CUDA/CPU only via nano-parakeet)

Uses nvidia/parakeet-tdt-0.6b-v3 for high-quality multilingual ASR.
"""

import logging
from time import perf_counter
from baseHandler import BaseHandler
import numpy as np
from rich.console import Console

try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

logger = logging.getLogger(__name__)
console = Console()

SUPPORTED_LANGUAGES = [
    "en", "de", "fr", "es", "it", "pt", "nl", "pl", "ru", "uk",
    "cs", "sk", "hu", "ro", "bg", "hr", "sl", "sr", "da", "no",
    "sv", "fi", "et", "lv", "lt"
]


class ParakeetTDTSTTHandler(BaseHandler):
    """
    Handles Speech-to-Text using NVIDIA Parakeet TDT via nano-parakeet (pure PyTorch).
    """

    def setup(
        self,
        model_name=None,
        device="cuda",
        compute_type="float16",
        language=None,
        gen_kwargs={},
        enable_live_transcription=False,
        live_transcription_update_interval=0.25,
    ):
        self.gen_kwargs = gen_kwargs
        self.start_language = language
        self.last_language = language if language else "en"

        # Determine device
        if device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if model_name is None:
            model_name = "nvidia/parakeet-tdt-0.6b-v3"

        self.model_name = model_name
        self.compute_type = compute_type

        logger.info(f"Loading Parakeet TDT model: {model_name} on {self.device}")
        self._setup_nano_parakeet(model_name)
        self.warmup()

    def _setup_nano_parakeet(self, model_name):
        """Setup using nano-parakeet (CUDA/CPU)."""
        try:
            import torch
            from nano_parakeet import from_pretrained

            self.backend = "nano_parakeet"

            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                self.device = "cpu"

            self.model = from_pretrained(model_name=model_name, device=self.device)
            logger.info(f"nano-parakeet model loaded successfully on {self.device}")
        except ImportError as e:
            raise ImportError(
                "nano-parakeet is required for Parakeet TDT on CUDA/CPU. "
                "Install with: pip install nano-parakeet"
            ) from e

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")
        dummy_audio = np.zeros(16000, dtype=np.float32)
        try:
            _ = self.model.transcribe(dummy_audio)
            logger.info("Model warmed up and ready")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    def process(self, spoken_prompt):
        logger.debug("Inferring Parakeet TDT...")

        global pipeline_start
        pipeline_start = perf_counter()

        # Handle progressive/final tuple from VAD
        if isinstance(spoken_prompt, tuple) and len(spoken_prompt) == 2:
            mode, audio_input = spoken_prompt
            is_final = (mode == "final")
            if not is_final:
                return  # Skip progressive updates on CUDA
        else:
            audio_input = spoken_prompt

        # Ensure float32 numpy array
        if not isinstance(audio_input, np.ndarray):
            audio_input = np.array(audio_input, dtype=np.float32)
        else:
            audio_input = audio_input.astype(np.float32)

        try:
            pred_text, language_code = self._process_nano_parakeet(audio_input)

            if language_code and language_code in SUPPORTED_LANGUAGES:
                self.last_language = language_code
            else:
                language_code = self.last_language
        except Exception as e:
            logger.error(f"Parakeet TDT inference failed: {e}")
            pred_text = ""
            language_code = self.last_language

        logger.debug("Finished Parakeet TDT inference")
        console.print(f"[yellow]USER: {pred_text}")
        console.print(f"[dim]Language: {language_code}[/dim]")

        yield (pred_text, language_code)

    def _detect_language_from_text(self, text):
        if not LANGDETECT_AVAILABLE or not text or len(text.strip()) < 10:
            return None
        try:
            detected = detect(text)
            return detected if detected in SUPPORTED_LANGUAGES else None
        except LangDetectException:
            return None

    def _process_nano_parakeet(self, audio_input):
        pred_text = self.model.transcribe(audio_input).strip()
        language_code = self.last_language
        return pred_text, language_code

    def cleanup(self):
        logger.info(f"Cleaning up {self.__class__.__name__}")
        if hasattr(self, "model"):
            del self.model
