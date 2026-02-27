from dataclasses import dataclass, field


@dataclass
class ParlerTTSHandlerArguments:
    tts_model_name: str = field(
        default="parler-tts/parler-mini-v1-jenny",
        metadata={"help": "The pretrained Parler TTS model to use."},
    )
    tts_device: str = field(
        default="cuda",
        metadata={"help": "Device for the model. Default is 'cuda'."},
    )
    tts_torch_dtype: str = field(
        default="float16",
        metadata={"help": "PyTorch dtype. One of 'float32', 'float16', 'bfloat16'."},
    )
    tts_compile_mode: str = field(
        default=None,
        metadata={"help": "Torch compile mode. 'default', 'reduce-overhead', or None."},
    )
    tts_gen_min_new_tokens: int = field(
        default=64,
        metadata={"help": "Min new tokens for TTS generation (~0.74s). Default is 64."},
    )
    tts_gen_max_new_tokens: int = field(
        default=1024,
        metadata={"help": "Max new tokens for TTS generation (~12s). Default is 1024."},
    )
    description: str = field(
        default="Jenny speaks at a slightly slow pace with an animated delivery with clear audio quality.",
        metadata={"help": "Voice description to guide TTS model."},
    )
    play_steps_s: float = field(
        default=0.4,
        metadata={"help": "Playback step interval in seconds. Lower = smoother but more overhead. Recommended: 0.3-0.5."},
    )
    max_prompt_pad_length: int = field(
        default=8,
        metadata={"help": "Max power of 2 for prompt padding during compilation."},
    )
    use_default_speakers_list: bool = field(
        default=False,
        metadata={"help": "Whether to use the default multi-language speakers list."},
    )
