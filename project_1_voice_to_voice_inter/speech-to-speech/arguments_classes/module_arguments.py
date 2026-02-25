from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModuleArguments:
    device: Optional[str] = field(
        default="cuda",
        metadata={"help": "If specified, overrides the device for all handlers. Default is 'cuda'."},
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
