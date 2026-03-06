from threading import Thread
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    TextIteratorStreamer,
)
import json
import torch
import re

from LLM.chat import Chat
from baseHandler import BaseHandler
from rich.console import Console
import logging
from nltk import sent_tokenize
from typing import Dict, List, Optional
from actions.allowed_actions import (
    ALLOWED_ACTIONS, DEFAULT_ACTION,
    ALLOWED_GESTURES, ALLOWED_INTENSITIES, DEFAULT_INTENSITY,
    GESTURE_EMOJI,
    build_tool_prompt,
)

logger = logging.getLogger(__name__)
console = Console()

TOOL_PATTERN = re.compile(r'\[TOOL:(\w+)(?:\|([^\]]+))?\]')
ACTION_PATTERN = re.compile(r'\[ACTION:(\w+)(?:\|([^\]]+))?\]')
GESTURE_PATTERN = re.compile(r'\[GESTURE:(\w+)(?:\|([^\]]+))?\]')

_NATIVE_TOOL_CALL_RE = re.compile(
    r'<tool_call>\s*(\{.*?\})\s*</tool_call>',
    re.DOTALL,
)
_PYTHON_TAG_RE = re.compile(
    r'<\|python_tag\|>\s*(\w+)\((.*?)\)',
    re.DOTALL,
)

WHISPER_LANGUAGE_TO_LLM_LANGUAGE = {
    "en": "english",
    "fr": "french",
    "es": "spanish",
    "zh": "chinese",
    "ja": "japanese",
    "ko": "korean",
    "hi": "hindi",
    "de": "german",
    "pt": "portuguese",
    "pl": "polish",
    "it": "italian",
    "nl": "dutch",
}


def choose_action(action: str) -> dict:
    return {"status": "ok", "action": action}


def trigger_animation(animation_type: str, intensity: str = "normal") -> dict:
    return {"status": "ok", "animation_type": animation_type, "intensity": intensity}


NATIVE_TOOLS = [choose_action, trigger_animation]


def parse_tool_calls(text: str):
    """
    Parse text-based [ACTION:], [GESTURE:], [TOOL:] tags 
    (fallback)
    """

    tools = []
    # Parse [TOOL:name|key:val]
    for match in TOOL_PATTERN.finditer(text):
        tool_name = match.group(1)
        params_str = match.group(2) or ""
        params = {}
        if params_str:
            for param in params_str.split('|'):
                if ':' in param:
                    key, val = param.split(':', 1)
                    params[key] = val
        tools.append({"name": tool_name, "parameters": params})

    # Parse [ACTION:name|key:val]
    for match in ACTION_PATTERN.finditer(text):
        action_name = match.group(1)
        params_str = match.group(2) or ""
        params = {}
        if params_str:
            for param in params_str.split('|'):
                if ':' in param:
                    key, val = param.split(':', 1)
                    params[key] = val
        tools.append({"name": action_name, "parameters": params, "type": "action"})

    # Parse [GESTURE:type] or [GESTURE:type|intensity]
    for match in GESTURE_PATTERN.finditer(text):
        gesture_name = match.group(1).lower()
        intensity = (match.group(2) or DEFAULT_INTENSITY).strip().lower()
        if intensity not in ALLOWED_INTENSITIES:
            intensity = DEFAULT_INTENSITY
        emoji = GESTURE_EMOJI.get(gesture_name, "")
        tools.append({
            "name": "trigger_animation",
            "parameters": {
                "animation_type": gesture_name,
                "intensity": intensity,
            },
            "type": "gesture",
            "emoji": emoji,
        })

    clean_text = TOOL_PATTERN.sub('', text)
    clean_text = ACTION_PATTERN.sub('', clean_text)
    clean_text = GESTURE_PATTERN.sub('', clean_text).strip()

    # Default = None action
    has_action = any(t.get("type") == "action" or t.get("name", "").upper() in ALLOWED_ACTIONS for t in tools)
    if not has_action:
        tools.append({"name": DEFAULT_ACTION, "parameters": {}, "type": "action"})

    return clean_text, tools


def parse_native_tool_calls(full_text: str):
    tools: List[dict] = []
    clean_text = full_text

    # <tool_call>JSON</tool_call> (Llama 3.1, Qwen, Hermes, etc.)
    for match in _NATIVE_TOOL_CALL_RE.finditer(full_text):
        try:
            call = json.loads(match.group(1))
            fn_name = call.get("name", "")
            arguments = call.get("arguments", call.get("parameters", {}))
            tools.append(_classify_native_call(fn_name, arguments))
        except json.JSONDecodeError:
            logger.debug(f"Failed to parse native tool call JSON: {match.group(1)!r}")
    clean_text = _NATIVE_TOOL_CALL_RE.sub('', clean_text)

    # <|python_tag|>fn_name(args) (Llama)
    if not tools:
        for match in _PYTHON_TAG_RE.finditer(full_text):
            fn_name = match.group(1)
            raw_args = match.group(2).strip()
            try:
                arguments = json.loads('{' + raw_args + '}') if raw_args else {}
            except json.JSONDecodeError:
                arguments = _parse_python_kwargs(raw_args)
            tools.append(_classify_native_call(fn_name, arguments))
        clean_text = _PYTHON_TAG_RE.sub('', clean_text)

    clean_text = re.sub(r'<\|/?tool_call\|?>|<\|python_tag\|>', '', clean_text).strip()
    # Ensure default action is present
    has_action = any(
        t.get("type") == "action" or t.get("name", "").upper() in ALLOWED_ACTIONS
        for t in tools
    )
    if not has_action:
        tools.append({"name": DEFAULT_ACTION, "parameters": {}, "type": "action"})

    return clean_text, tools


def _classify_native_call(fn_name: str, arguments: dict) -> dict:
    fn_lower = fn_name.lower()

    if fn_lower == "choose_action" or fn_lower in (a.lower() for a in ALLOWED_ACTIONS):
        action = arguments.get("action", fn_name).upper()
        if action not in ALLOWED_ACTIONS:
            action = DEFAULT_ACTION
        return {"name": action, "parameters": arguments, "type": "action"}

    if fn_lower == "trigger_animation":
        anim_type = arguments.get("animation_type", "wave").lower()
        intensity = arguments.get("intensity", DEFAULT_INTENSITY).lower()
        if intensity not in ALLOWED_INTENSITIES:
            intensity = DEFAULT_INTENSITY
        emoji = GESTURE_EMOJI.get(anim_type, "")
        return {
            "name": "trigger_animation",
            "parameters": {"animation_type": anim_type, "intensity": intensity},
            "type": "gesture",
            "emoji": emoji,
        }

    return {"name": fn_name, "parameters": arguments}


def _parse_python_kwargs(raw: str) -> dict:
    result = {}
    for part in raw.split(','):
        part = part.strip()
        if '=' in part:
            k, v = part.split('=', 1)
            v = v.strip().strip('"').strip("'")
            result[k.strip()] = v
    return result


class LanguageModelHandler(BaseHandler):
    def setup(
        self,
        model_name="microsoft/Phi-3-mini-4k-instruct",
        device="cuda",
        torch_dtype="float16",
        gen_kwargs={},
        user_role="user",
        chat_size=1,
        init_chat_role=None,
        init_chat_prompt=build_tool_prompt(),
    ):
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, trust_remote_code=True, attn_implementation="sdpa"
        ).to(device)
        self.pipe = pipeline(
            "text-generation", model=self.model, tokenizer=self.tokenizer, device=device
        )
        self.streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        self.gen_kwargs = {
            "streamer": self.streamer,
            "return_full_text": False,
            **gen_kwargs,
        }

        self.native_tools: Optional[list] = None
        self._use_native_tools = False
        if self._check_native_tool_support():
            self.native_tools = NATIVE_TOOLS
            self._use_native_tools = True
            logger.info(
                f"Supports tools= "
                f"({len(NATIVE_TOOLS)} tools: {[f.__name__ for f in NATIVE_TOOLS]})"
            )
        else:
            logger.info(
                "Native tool calling NOT supported by this model's chat template. "
                "Falling back to text-tag [ACTION:]/[GESTURE:] approach."
            )

        self.chat = Chat(chat_size)
        if init_chat_role:
            if not init_chat_prompt and not self._use_native_tools:
                raise ValueError("An initial prompt needs to be specified when setting init_chat_role.")
            # When native tools are used the system prompt is simpler since
            # tool schemas are injected by the tokenizer automatically.
            prompt = init_chat_prompt or self._build_native_system_prompt()
            self.chat.init_chat({"role": init_chat_role, "content": prompt})
        self.user_role = user_role
        self.warmup()


    def _check_native_tool_support(self) -> bool:
        try:
            test_messages = [{"role": "user", "content": "test"}]
            self.tokenizer.apply_chat_template(
                test_messages,
                tools=NATIVE_TOOLS,
                tokenize=False,
                add_generation_prompt=True,
            )
            return True
        except Exception as e:
            logger.debug(f"Native tool support check failed: {e}")
            return False

    @staticmethod
    def _build_native_system_prompt() -> str:
        return (
            "You are ARIA, an expressive robot assistant on a Unitree G1 humanoid.\n"
            "You have two tools: choose_action and trigger_animation.\n"
            "Rules:\n"
            "1. Call trigger_animation in EVERY response with an appropriate gesture.\n"
            "2. Call choose_action in EVERY response (use NONE for no movement).\n"
            "3. Keep spoken text under 20 words. Be brief, punchy, direct.\n"
            "4. Use spoken English: fragments, contractions, exclamations.\n"
        )


    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")

        dummy_input_text = "Repeat the word 'home'."
        dummy_chat = [{"role": self.user_role, "content": dummy_input_text}]

        if self._use_native_tools:
            # Warmup with model.generate() path
            warmup_gen_kwargs = {
                "max_new_tokens": self.gen_kwargs.get("max_new_tokens", 128),
                "min_new_tokens": self.gen_kwargs.get("min_new_tokens", 0),
                "streamer": self.streamer,
            }
            n_steps = 2
            if self.device == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start_event.record()
            for _ in range(n_steps):
                inputs = self.tokenizer.apply_chat_template(
                    dummy_chat,
                    tools=self.native_tools,
                    return_tensors="pt",
                    return_dict=True,
                    add_generation_prompt=True,
                ).to(self.device)
                thread = Thread(
                    target=self.model.generate,
                    kwargs={**inputs, **warmup_gen_kwargs},
                )
                thread.start()
                for _ in self.streamer:
                    pass
            if self.device == "cuda":
                end_event.record()
                torch.cuda.synchronize()
                logger.info(
                    f"{self.__class__.__name__}: warmed up (native tools)"
                    f"time: {start_event.elapsed_time(end_event) * 1e-3:.3f} s"
                )
        else:
            # fallback with pipeline() path
            warmup_gen_kwargs = {
                "min_new_tokens": self.gen_kwargs["min_new_tokens"],
                "max_new_tokens": self.gen_kwargs["max_new_tokens"],
                **self.gen_kwargs,
            }
            n_steps = 2
            if self.device == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start_event.record()
            for _ in range(n_steps):
                thread = Thread(target=self.pipe, args=(dummy_chat,), kwargs=warmup_gen_kwargs)
                thread.start()
                for _ in self.streamer:
                    pass
            if self.device == "cuda":
                end_event.record()
                torch.cuda.synchronize()
                logger.info(
                    f"{self.__class__.__name__}: warmed up (pipeline)! "
                    f"time: {start_event.elapsed_time(end_event) * 1e-3:.3f} s"
                )


    def process(self, prompt):
        logger.debug("inferring language model...")
        language_code = None
        if isinstance(prompt, tuple):
            prompt, language_code = prompt
            if language_code[-5:] == "-auto":
                language_code = language_code[:-5]
                prompt = f"Please reply to my message in {WHISPER_LANGUAGE_TO_LLM_LANGUAGE[language_code]}. " + prompt

        self.chat.append({"role": self.user_role, "content": prompt})

        if self._use_native_tools:
            yield from self._process_native_tools(language_code)
        else:
            yield from self._process_text_tags(language_code)

    def _process_text_tags(self, language_code):
        thread = Thread(target=self.pipe, args=(self.chat.to_list(),), kwargs=self.gen_kwargs)
        thread.start()

        generated_text = ""
        for new_text in self.streamer:
            generated_text += new_text

        self.chat.append({"role": "assistant", "content": generated_text})

        clean_text, tools = parse_tool_calls(generated_text)
        if clean_text.strip() or tools:
            yield (clean_text.strip() or "...", language_code, tools)

    def _process_native_tools(self, language_code):
        chat_messages = self.chat.to_list()

        # Tokenize with tool schemas injected into the prompt
        inputs = self.tokenizer.apply_chat_template(
            chat_messages,
            tools=self.native_tools,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        ).to(self.device)

        generate_kwargs = {
            **inputs,
            "streamer": self.streamer,
            "max_new_tokens": self.gen_kwargs.get("max_new_tokens", 128),
            "min_new_tokens": self.gen_kwargs.get("min_new_tokens", 0),
        }
        # Forward optional sampling params
        if self.gen_kwargs.get("do_sample"):
            generate_kwargs["do_sample"] = True
            if "temperature" in self.gen_kwargs:
                generate_kwargs["temperature"] = self.gen_kwargs["temperature"]

        thread = Thread(target=self.model.generate, kwargs=generate_kwargs)
        thread.start()

        generated_text = ""
        for new_text in self.streamer:
            generated_text += new_text

        self.chat.append({"role": "assistant", "content": generated_text})

        clean_text, tools = parse_native_tool_calls(generated_text)
        if not tools or (len(tools) == 1 and tools[0].get("name") == DEFAULT_ACTION):
            # Native parsing found nothing meaningful — try text-tag fallback
            alt_text, alt_tools = parse_tool_calls(generated_text)
            if len(alt_tools) > len(tools):
                clean_text, tools = alt_text, alt_tools

        # Strip any leftover special tokens
        clean_text = re.sub(r'<\|/?tool_call\|?>|<\|python_tag\|>', '', clean_text).strip()

        if clean_text or tools:
            yield (clean_text or "...", language_code, tools)
