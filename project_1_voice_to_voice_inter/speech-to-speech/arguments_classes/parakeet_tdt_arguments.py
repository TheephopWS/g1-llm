from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ParakeetTDTSTTHandlerArguments:
    """Arguments for the Parakeet TDT STT handler (CUDA/CPU via nano-parakeet)."""

    parakeet_tdt_model_name: Optional[str] = field(
        default=None,
        metadata={"help": "Model name. Default: 'nvidia/parakeet-tdt-0.6b-v3' for CUDA/CPU."},
    )
    parakeet_tdt_device: str = field(
        default="cuda",
        metadata={"help": "Device to run model on. Options: 'cuda', 'cpu'. Default is 'cuda'."},
    )
    parakeet_tdt_compute_type: str = field(
        default="float16",
        metadata={"help": "Compute type. Options: 'float16', 'float32'. Default is 'float16'."},
    )
    parakeet_tdt_language: Optional[str] = field(
        default=None,
        metadata={"help": "Target language code. If not set, model auto-detects."},
    )
    parakeet_tdt_gen_kwargs: dict = field(
        default_factory=dict,
        metadata={"help": "Additional generation kwargs."},
    )
