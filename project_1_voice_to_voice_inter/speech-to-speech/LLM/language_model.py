from threading import Thread
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    TextIteratorStreamer,
)
import torch
import re

from LLM.chat import Chat
from baseHandler import BaseHandler
from rich.console import Console
import logging
from nltk import sent_tokenize
from typing import Dict
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


def parse_tool_calls(text):
    tools = []
    # Parse [TOOL:name|key:val] format
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
    # Parse [ACTION:name|key:val] format (robot actions)
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

    # Parse [GESTURE:type] or [GESTURE:type|intensity] format (embodied animations)
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

    # Ensure there is always an action — default to NONE if the LLM omitted one
    has_action = any(t.get("type") == "action" or t.get("name", "").upper() in ALLOWED_ACTIONS for t in tools)
    if not has_action:
        tools.append({"name": DEFAULT_ACTION, "parameters": {}, "type": "action"})

    return clean_text, tools


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

        self.chat = Chat(chat_size)
        if init_chat_role:
            if not init_chat_prompt:
                raise ValueError("An initial prompt needs to be specified when setting init_chat_role.")
            self.chat.init_chat({"role": init_chat_role, "content": init_chat_prompt})
        self.user_role = user_role
        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")

        dummy_input_text = "Repeat the word 'home'."
        dummy_chat = [{"role": self.user_role, "content": dummy_input_text}]
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
            logger.info(f"{self.__class__.__name__}: warmed up! time: {start_event.elapsed_time(end_event) * 1e-3:.3f} s")

    def process(self, prompt):
        logger.debug("inferring language model...")
        language_code = None
        if isinstance(prompt, tuple):
            prompt, language_code = prompt
            if language_code[-5:] == "-auto":
                language_code = language_code[:-5]
                prompt = f"Please reply to my message in {WHISPER_LANGUAGE_TO_LLM_LANGUAGE[language_code]}. " + prompt

        self.chat.append({"role": self.user_role, "content": prompt})
        thread = Thread(target=self.pipe, args=(self.chat.to_list(),), kwargs=self.gen_kwargs)
        thread.start()

        generated_text, printable_text = "", ""
        for new_text in self.streamer:
            generated_text += new_text
            printable_text += new_text
            sentences = sent_tokenize(printable_text)
            if len(sentences) > 1:
                # Yield all complete sentences, keep last (possibly incomplete) one
                for sentence in sentences[:-1]:
                    clean_text, tools = parse_tool_calls(sentence)
                    yield (clean_text, language_code, tools)
                printable_text = sentences[-1]

        self.chat.append({"role": "assistant", "content": generated_text})
        # Yield any remaining text
        if printable_text.strip():
            clean_text, tools = parse_tool_calls(printable_text)
            yield (clean_text, language_code, tools)
