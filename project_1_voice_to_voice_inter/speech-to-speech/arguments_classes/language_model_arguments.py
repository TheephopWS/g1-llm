from dataclasses import dataclass, field
from actions.allowed_actions import build_tool_prompt


@dataclass
class LanguageModelHandlerArguments:
    lm_model_name: str = field(
        default="HuggingFaceTB/SmolLM-360M-Instruct",
        metadata={"help": "The pretrained language model to use."},
    )
    lm_device: str = field(
        default="cuda",
        metadata={"help": "Device for the model. Default is 'cuda'."},
    )
    lm_torch_dtype: str = field(
        default="float16",
        metadata={"help": "PyTorch dtype. One of 'float32', 'float16', 'bfloat16'."},
    )
    user_role: str = field(
        default="user",
        metadata={"help": "Role assigned to the user in chat."},
    )
    init_chat_role: str = field(
        default="system",
        metadata={"help": "Initial role for setting up chat context."},
    )
    init_chat_prompt: str = field(
        default=None,
        metadata={"help": "Initial chat prompt. Defaults to build_tool_prompt() with pipeline + choose_action tool."},
    )

    def __post_init__(self):
        if self.init_chat_prompt is None:
            self.init_chat_prompt = build_tool_prompt()
    lm_gen_max_new_tokens: int = field(
        default=128,
        metadata={"help": "Max new tokens to generate. Default is 128."},
    )
    lm_gen_min_new_tokens: int = field(
        default=0,
        metadata={"help": "Min new tokens to generate. Default is 0."},
    )
    lm_gen_temperature: float = field(
        default=0.0,
        metadata={"help": "Temperature for generation. 0.0 = deterministic."},
    )
    lm_gen_do_sample: bool = field(
        default=False,
        metadata={"help": "Whether to use sampling. Default is False."},
    )
    chat_size: int = field(
        default=2,
        metadata={"help": "Number of user-assistant pairs to keep in chat history."},
    )
