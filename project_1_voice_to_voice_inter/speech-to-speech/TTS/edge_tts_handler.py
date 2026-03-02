import asyncio
import io
import logging
from time import perf_counter

import numpy as np
from rich.console import Console

from baseHandler import BaseHandler

logger = logging.getLogger(__name__)
console = Console()

LANGUAGE_TO_EDGE_VOICE = {
    "en": "en-US-GuyNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "yue": "zh-HK-HiuGaamNeural",
    "fr": "fr-FR-HenriNeural",
    "es": "es-ES-AlvaroNeural",
    "de": "de-DE-ConradNeural",
    "ja": "ja-JP-KeitaNeural",
    "ko": "ko-KR-InJoonNeural",
    "pt": "pt-BR-AntonioNeural",
    "it": "it-IT-DiegoNeural",
    "nl": "nl-NL-MaartenNeural",
    "pl": "pl-PL-MarekNeural",
    "ru": "ru-RU-DmitryNeural",
}

DEFAULT_VOICE = "en-US-GuyNeural"


def _looks_cantonese(text: str) -> bool:
    hints = ["係", "唔", "喺", "咩", "嘅", "啲", "嗰", "冇", "佢", "咗", "啦", "喇", "嘢", "乜", "噉"]
    return any(h in (text or "") for h in hints)


class EdgeTTSHandler(BaseHandler):
    def setup(
        self,
        should_listen,
        voice=None,
        rate="+0%",
        volume="+0%",
        target_sr=16000,
        blocksize=512,
    ):
        """
        Args:
            should_listen: threading.Event to signal when TTS is done.
            voice: Override default voice (e.g., "en-US-GuyNeural").
                   If None, auto-selects based on language_code from STT.
            rate: Speech rate adjustment (e.g., "+10%", "-5%").
            volume: Volume adjustment (e.g., "+0%").
            target_sr: Output sample rate in Hz. Default 16000.
            blocksize: Number of int16 samples per output block. Default 512.
        """
        self.should_listen = should_listen
        self.default_voice = voice or DEFAULT_VOICE
        self.rate = rate
        self.volume = volume
        self.target_sr = target_sr
        self.blocksize = blocksize

        try:
            import edge_tts
        except ImportError:
            raise ImportError(
                "edge_tts is required for EdgeTTSHandler. "
                "Install with: pip install edge-tts"
            )

        try:
            from pydub import AudioSegment  # noqa: F401
        except ImportError:
            raise ImportError(
                "pydub is required for EdgeTTSHandler to decode MP3. "
                "Install with: pip install pydub"
            )

        logger.info(
            f"EdgeTTSHandler initialized: voice={self.default_voice}, "
            f"rate={self.rate}, target_sr={self.target_sr}, blocksize={self.blocksize}"
        )

    def _pick_voice(self, language_code: str, text: str) -> str:
        if language_code in ("zh", "yue") and _looks_cantonese(text):
            return "zh-HK-HiuGaamNeural"

        voice = LANGUAGE_TO_EDGE_VOICE.get(language_code, self.default_voice)
        return voice

    async def _generate_audio(self, text: str, voice: str) -> bytes:
        import edge_tts

        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=self.rate,
            volume=self.volume,
        )

        mp3_buffer = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_buffer.write(chunk["data"])

        return mp3_buffer.getvalue()

    def _decode_mp3_to_int16(self, mp3_bytes: bytes) -> np.ndarray:
        from pydub import AudioSegment

        audio = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))

        # Convert to mono, target sample rate, 16-bit
        audio = audio.set_channels(1).set_frame_rate(self.target_sr).set_sample_width(2)

        samples = np.frombuffer(audio.raw_data, dtype=np.int16)
        return samples

    def process(self, llm_sentence):
        """
        Process a sentence from the LLM and yield int16 audio blocks.

        Input: str or (str, language_code) tuple
        Output: yields np.ndarray of int16 with shape (blocksize,)
        """
        language_code = "en"
        if isinstance(llm_sentence, tuple):
            llm_sentence, language_code = llm_sentence

        console.print(f"[green]ASSISTANT: {llm_sentence}")

        if not llm_sentence or not llm_sentence.strip():
            self.should_listen.set()
            return

        # Pick voice based on language
        voice = self._pick_voice(language_code, llm_sentence)
        logger.debug(f"Edge TTS: voice={voice}, lang={language_code}")

        try:
            t0 = perf_counter()

            mp3_bytes = asyncio.run(self._generate_audio(llm_sentence, voice))
            gen_time = perf_counter() - t0

            if not mp3_bytes:
                logger.warning("Edge TTS returned empty audio")
                self.should_listen.set()
                return

            t1 = perf_counter()
            audio_int16 = self._decode_mp3_to_int16(mp3_bytes)
            decode_time = perf_counter() - t1

            duration = len(audio_int16) / self.target_sr
            logger.info(
                f"Edge TTS: {duration:.2f}s audio, "
                f"gen={gen_time:.3f}s, decode={decode_time:.3f}s, "
                f"voice={voice}"
            )

            for i in range(0, len(audio_int16), self.blocksize):
                block = audio_int16[i : i + self.blocksize]
                if len(block) < self.blocksize:
                    block = np.pad(block, (0, self.blocksize - len(block)))
                yield block

        except Exception as e:
            logger.error(f"Edge TTS error: {e}", exc_info=True)

        self.should_listen.set()

    def cleanup(self):
        pass
