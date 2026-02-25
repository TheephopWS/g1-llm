from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WhisperSTTHandlerArguments:
    stt_model_name: str = field(
        default="distil-whisper/distil-large-v3",
        metadata={"help": "The pretrained Whisper model to use."},
    )
    stt_device: str = field(
        default="cuda",
        metadata={"help": "Device for the model. Default is 'cuda'."},
    )
    stt_torch_dtype: str = field(
        default="float16",
        metadata={"help": "PyTorch dtype. One of 'float32', 'float16', 'bfloat16'."},
    )
    stt_compile_mode: str = field(
        default=None,
        metadata={"help": "Compile mode for torch compile. 'default', 'reduce-overhead', 'max-autotune', or None."},
    )
    stt_gen_max_new_tokens: int = field(
        default=128,
        metadata={"help": "Max new tokens to generate. Default is 128."},
    )
    stt_gen_num_beams: int = field(
        default=1,
        metadata={"help": "Number of beams for beam search. Default is 1 (greedy)."},
    )
    stt_gen_return_timestamps: bool = field(
        default=False,
        metadata={"help": "Whether to return timestamps. Default is False."},
    )
    stt_gen_task: str = field(
        default="transcribe",
        metadata={"help": "Task to perform. Default is 'transcribe'."},
    )
    language: Optional[str] = field(
        default="en",
        metadata={"help": "Language code for the conversation. Use 'auto' for auto-detect. Default is 'en'."},
    )
