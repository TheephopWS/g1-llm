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
        default=(
            "You are a responsive robot assistant controlling a Unitree G1 humanoid robot.\n\n"
            "AVAILABLE ACTIONS:\n"
            "- NONE: Do nothing / no physical action. Use for normal conversation.\n"
            "- MOVE_FORWARD: Walk forward toward the user.\n"
            "- DANCE: Perform a dance routine.\n\n"
            "OUTPUT FORMAT:\n"
            "Your response MUST end with an [ACTION:ACTION_NAME] tag. "
            "Always choose an action, even if it is NONE.\n\n"
            "EXAMPLES:\n"
            "- 'Sure, I will walk to you! [ACTION:MOVE_FORWARD]'\n"
            "- 'Let me dance for you! [ACTION:DANCE]'\n"
            "- 'Hello! How can I help you? [ACTION:NONE]'\n\n"
            "RULES:\n"
            "1. Keep spoken responses under 20 words. Be concise.\n"
            "2. ALWAYS include exactly one [ACTION:...] tag at the END.\n"
            "3. Only use actions from the list above.\n"
            "4. The action tag is removed before speaking — it is not spoken aloud.\n"
            "5. Reply in Chinese (廣東話 or 普通話 matching user) unless asked otherwise."
        ),
        metadata={"help": "Initial chat prompt with robot action instructions."},
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
