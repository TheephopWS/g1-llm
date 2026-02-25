import torchaudio
from VAD.vad_iterator import VADIterator
from baseHandler import BaseHandler
import numpy as np
import torch
from rich.console import Console

from utils.utils import int2float
import logging

logger = logging.getLogger(__name__)

# Optional import for audio enhancement
try:
    from df.enhance import enhance, init_df
    HAS_DF = True
except (ImportError, ModuleNotFoundError) as e:
    HAS_DF = False
    logger.warning(f"DeepFilterNet not available for audio enhancement: {e}")

console = Console()


class VADHandler(BaseHandler):
    """
    Handles voice activity detection. When voice activity is detected, audio will be accumulated
    until end of speech is detected and then passed to the next part.
    """

    def setup(
        self,
        should_listen,
        thresh=0.3,
        sample_rate=16000,
        min_silence_ms=1000,
        min_speech_ms=500,
        max_speech_ms=float("inf"),
        speech_pad_ms=30,
        audio_enhancement=False,
        enable_realtime_transcription=False,
        realtime_processing_pause=0.25,
        text_output_queue=None,
    ):
        self.should_listen = should_listen
        self.sample_rate = sample_rate
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self.max_speech_ms = max_speech_ms
        self.enable_realtime_transcription = enable_realtime_transcription
        self.realtime_processing_pause = realtime_processing_pause
        self.text_output_queue = text_output_queue
        self.model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
        self.iterator = VADIterator(
            self.model,
            threshold=thresh,
            sampling_rate=sample_rate,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
        )
        self.audio_enhancement = audio_enhancement
        if audio_enhancement:
            if not HAS_DF:
                logger.error("Audio enhancement requested but DeepFilterNet is not available. Disabling.")
                self.audio_enhancement = False
            else:
                self.enhanced_model, self.df_state, _ = init_df()

        self.accumulated_audio = []
        self.last_process_time = 0

    def process(self, audio_chunk):
        logger.debug(f"VAD received {len(audio_chunk)} bytes")
        audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
        audio_float32 = int2float(audio_int16)

        was_triggered_before = self.iterator.triggered
        vad_output = self.iterator(torch.from_numpy(audio_float32))
        is_triggered_now = self.iterator.triggered

        if is_triggered_now and not was_triggered_before:
            if self.text_output_queue:
                self.text_output_queue.put({"type": "speech_started"})

        if self.enable_realtime_transcription:
            yield from self._process_realtime(vad_output)
        else:
            yield from self._process_normal(vad_output)

    def _process_realtime(self, vad_output):
        import time

        if hasattr(self.iterator, "buffer") and len(self.iterator.buffer) > 0:
            current_time = time.time()
            if (current_time - self.last_process_time) >= self.realtime_processing_pause:
                array = torch.cat(self.iterator.buffer).cpu().numpy()
                duration_ms = len(array) / self.sample_rate * 1000
                if duration_ms >= self.min_speech_ms:
                    yield ("progressive", array)
                    self.last_process_time = current_time

        if vad_output is not None and len(vad_output) != 0:
            array = torch.cat(vad_output).cpu().numpy()
            duration_ms = len(array) / self.sample_rate * 1000
            if duration_ms < self.min_speech_ms or duration_ms > self.max_speech_ms:
                logger.debug(f"audio duration: {len(array) / self.sample_rate}s, skipping")
            else:
                self.should_listen.clear()
                if self.text_output_queue:
                    self.text_output_queue.put({"type": "speech_stopped"})
                if self.audio_enhancement:
                    array = self._apply_audio_enhancement(array)
                yield ("final", array)
                self.last_process_time = 0

    def _process_normal(self, vad_output):
        if vad_output is not None and len(vad_output) != 0:
            array = torch.cat(vad_output).cpu().numpy()
            duration_ms = len(array) / self.sample_rate * 1000
            if duration_ms < self.min_speech_ms or duration_ms > self.max_speech_ms:
                logger.debug(f"audio duration: {len(array) / self.sample_rate}s, skipping")
            else:
                self.should_listen.clear()
                if self.text_output_queue:
                    self.text_output_queue.put({"type": "speech_stopped"})
                if self.audio_enhancement:
                    array = self._apply_audio_enhancement(array)
                yield array

    def _apply_audio_enhancement(self, array):
        if self.sample_rate != self.df_state.sr():
            audio_float32 = torchaudio.functional.resample(
                torch.from_numpy(array),
                orig_freq=self.sample_rate,
                new_freq=self.df_state.sr(),
            )
            enhanced = enhance(self.enhanced_model, self.df_state, audio_float32.unsqueeze(0))
            enhanced = torchaudio.functional.resample(
                enhanced, orig_freq=self.df_state.sr(), new_freq=self.sample_rate,
            )
        else:
            enhanced = enhance(self.enhanced_model, self.df_state, torch.from_numpy(array))
        return enhanced.numpy().squeeze()

    @property
    def min_time_to_debug(self):
        return 0.00001
