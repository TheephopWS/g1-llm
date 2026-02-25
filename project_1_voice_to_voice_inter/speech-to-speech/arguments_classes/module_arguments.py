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
