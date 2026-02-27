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
            "You are a responsive robot assistant controlling a Unitree G1 humanoid robot. "
            "You can perform physical actions by including an action tag in your response.\n\n"
            "AVAILABLE ACTIONS:\n"
            "- NONE: No physical action. Use for normal conversation.\n"
            "- MOVE_FORWARD: Walk forward toward the user.\n"
            "- DANCE: Perform a dance routine.\n\n"
            "FORMAT: Include [ACTION:ACTION_NAME] at the END of your spoken response when an action is needed.\n"
            "Example: 'Sure, I will walk to you! [ACTION:MOVE_FORWARD]'\n"
            "Example: 'Let me dance for you! [ACTION:DANCE]'\n"
            "Example: 'Hello! How can I help you?'\n\n"
            "RULES:\n"
            "1. Keep responses under 20 words. Be concise.\n"
            "2. Only use actions from the list above.\n"
            "3. If no action is needed, do NOT include any action tag.\n"
            "4. The action tag will be removed before speaking - it is not spoken aloud."
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
