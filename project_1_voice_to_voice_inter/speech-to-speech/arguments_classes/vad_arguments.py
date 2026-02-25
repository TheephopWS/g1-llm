from dataclasses import dataclass, field


@dataclass
class VADHandlerArguments:
    thresh: float = field(
        default=0.3,
        metadata={"help": "Threshold value for voice activity detection (0-1)."},
    )
    sample_rate: int = field(
        default=16000,
        metadata={"help": "Audio sample rate in Hz. Default is 16000."},
    )
    min_silence_ms: int = field(
        default=300,
        metadata={"help": "Minimum silence duration (ms) for segmenting speech."},
    )
    min_speech_ms: int = field(
        default=500,
        metadata={"help": "Minimum speech duration (ms) to be considered valid."},
    )
    max_speech_ms: float = field(
        default=float("inf"),
        metadata={"help": "Maximum continuous speech before forcing a split."},
    )
    speech_pad_ms: int = field(
        default=500,
        metadata={"help": "Padding added to start/end of detected speech (ms)."},
    )
    audio_enhancement: bool = field(
        default=False,
        metadata={"help": "Apply audio enhancement (requires DeepFilterNet)."},
    )
    enable_realtime_transcription: bool = field(
        default=False,
        metadata={"help": "Enable progressive audio release for live transcription."},
    )
    realtime_processing_pause: float = field(
        default=0.2,
        metadata={"help": "Interval (s) for releasing progressive audio chunks."},
    )
