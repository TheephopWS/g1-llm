import threading
import sounddevice as sd
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


class LocalAudioStreamer:
    def __init__(self, input_queue, output_queue, list_play_chunk_size=512, input_device=None, output_device=None):
        self.list_play_chunk_size = list_play_chunk_size
        self.stop_event = threading.Event()
        self.input_queue = input_queue
        self.output_queue = output_queue

        """
            0 Microsoft Sound Mapper - Input, MME (2 in, 0 out)
            >  1 Microphone Array (Intel® Smart , MME (4 in, 0 out)
            2 Microsoft Sound Mapper - Output, MME (0 in, 2 out)
            <  3 Speakers (Realtek(R) Audio), MME (0 in, 2 out)
            4 Mi 27 NFGL (NVIDIA High Definit, MME (0 in, 2 out)
            5 Primary Sound Capture Driver, Windows DirectSound (2 in, 0 out)
            6 Microphone Array (Intel® Smart Sound Technology for Digital Microphones), Windows DirectSound (4 in, 0 out)
            7 Primary Sound Driver, Windows DirectSound (0 in, 2 out)
            8 Speakers (Realtek(R) Audio), Windows DirectSound (0 in, 2 out)
            9 Mi 27 NFGL (NVIDIA High Definition Audio), Windows DirectSound (0 in, 2 out)
            10 Speakers (Realtek(R) Audio), Windows WASAPI (0 in, 2 out)
            11 Mi 27 NFGL (NVIDIA High Definition Audio), Windows WASAPI (0 in, 2 out)
            12 Microphone Array (Intel® Smart Sound Technology for Digital Microphones), Windows WASAPI (2 in, 0 out)
            13 Output (NVIDIA High Definition Audio), Windows WDM-KS (0 in, 2 out)
            14 Headphones 1 (Realtek HD Audio 2nd output with SST), Windows WDM-KS (0 in, 2 out)
            15 Headphones 2 (Realtek HD Audio 2nd output with SST), Windows WDM-KS (0 in, 2 out)
            16 PC Speaker (Realtek HD Audio 2nd output with SST), Windows WDM-KS (2 in, 0 out)
            17 Speakers 1 (Realtek HD Audio output with SST), Windows WDM-KS (0 in, 2 out)
            18 Speakers 2 (Realtek HD Audio output with SST), Windows WDM-KS (0 in, 2 out)
            19 PC Speaker (Realtek HD Audio output with SST), Windows WDM-KS (2 in, 0 out)
            20 Stereo Mix (Realtek HD Audio Stereo input), Windows WDM-KS (2 in, 0 out)
            21 Microphone (Realtek HD Audio Mic input), Windows WDM-KS (2 in, 0 out)
            22 Microphone Array 1 (), Windows WDM-KS (2 in, 0 out)
            23 Microphone Array 2 (), Windows WDM-KS (2 in, 0 out)
            24 Microphone Array 3 (), Windows WDM-KS (4 in, 0 out)
        """
        self.input_device = 1
        self.output_device = 3

    def run(self):
        def callback(indata, outdata, frames, time, status):
            if status:
                logger.warning(f"Audio callback status: {status}")
            if self.stop_event.is_set():
                outdata[:] = np.zeros_like(outdata)
                return
            if self.output_queue.empty():
                self.input_queue.put(indata.copy())
                outdata[:] = np.zeros_like(outdata)
            else:
                try:
                    audio_chunk = self.output_queue.get_nowait()
                    if isinstance(audio_chunk, np.ndarray):
                        chunk = audio_chunk.flatten()
                        # Resize chunk to match the actual callback buffer size
                        if len(chunk) >= frames:
                            chunk = chunk[:frames]
                        else:
                            chunk = np.pad(chunk, (0, frames - len(chunk)))
                        outdata[:] = chunk.reshape(-1, 1)
                    else:
                        outdata[:] = np.zeros_like(outdata)
                except Exception as e:
                    logger.error(f"Audio playback callback error: {e}")
                    outdata[:] = np.zeros_like(outdata)

        logger.info("Available audio devices:")
        logger.info(sd.query_devices())

        device = (self.input_device, self.output_device)
        logger.info(f"Using input device: {self.input_device}, output device: {self.output_device}")

        with sd.Stream(
            samplerate=16000,
            dtype="int16",
            channels=1,
            callback=callback,
            blocksize=self.list_play_chunk_size,
            device=device,
        ):
            logger.info("Starting local audio stream")
            while not self.stop_event.is_set():
                time.sleep(0.001)
            print("Stopping recording")
