from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModuleArguments:
    device: Optional[str] = field(
        default=None,
        metadata={"help": "If specified, overrides the device for all handlers."},
    )
    mode: Optional[str] = field(
        default="socket",
        metadata={"help": "Pipeline mode: 'local' or 'socket'. Default is 'socket'."},
    )
    stt: Optional[str] = field(
        default="whisper",
        metadata={"help": "STT engine: 'whisper' or 'parakeet-tdt'. Default is 'whisper'."},
    )
    llm: Optional[str] = field(
        default="transformers",
        metadata={"help": "LLM engine: 'transformers'. Default is 'transformers'."},
    )
    tts: Optional[str] = field(
        default="parler",
        metadata={"help": "TTS engine: 'parler'. Default is 'parler'."},
    )
    log_level: str = field(
        default="info",
        metadata={"help": "Logging level. Example --log_level debug. Default is 'info'."},
    )
    input_device: Optional[int] = field(
        default=None,
        metadata={"help": "Audio input device index (microphone). Run 'python -c \"import sounddevice; print(sounddevice.query_devices())\"' to list devices."},
    )
    output_device: Optional[int] = field(
        default=None,
        metadata={"help": "Audio output device index (speakers). Run 'python -c \"import sounddevice; print(sounddevice.query_devices())\"' to list devices."},
    )
