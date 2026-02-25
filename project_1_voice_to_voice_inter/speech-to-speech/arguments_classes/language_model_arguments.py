from dataclasses import dataclass, field


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
        default="You are a helpful and friendly AI assistant. You are polite, respectful, and aim to provide concise responses of less than 20 words.",
        metadata={"help": "Initial chat prompt to establish context."},
    )
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
