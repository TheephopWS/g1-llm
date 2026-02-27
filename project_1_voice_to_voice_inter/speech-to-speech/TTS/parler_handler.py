from threading import Thread
from time import perf_counter
from baseHandler import BaseHandler
import numpy as np
import torch
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer
from scipy.signal import resample_poly
from math import gcd
import logging
from rich.console import Console
from utils.utils import next_power_of_2
from transformers.utils.import_utils import is_flash_attn_2_available

torch._inductor.config.fx_graph_cache = True
torch._dynamo.config.cache_size_limit = 15

logger = logging.getLogger(__name__)
console = Console()

if not is_flash_attn_2_available() and torch.cuda.is_available():
    logger.warn(
        "Parler TTS works best with flash attention 2, but is not installed. "
        "Install with: uv pip install flash-attn --no-build-isolation"
    )


WHISPER_LANGUAGE_TO_PARLER_SPEAKER = {
    "en": "Jason",
    "fr": "Christine",
    "es": "Steven",
    "de": "Nicole",
    "pt": "Sophia",
    "pl": "Alex",
    "it": "Richard",
    "nl": "Mark",
}


class ParlerTTSHandler(BaseHandler):
    def setup(
        self,
        should_listen,
        model_name="parler-tts/parler-mini-v1-jenny",
        device="cuda",
        torch_dtype="float16",
        compile_mode=None,
        gen_kwargs={},
        max_prompt_pad_length=8,
        description="Jenny speaks at a slightly slow pace with an animated delivery with clear audio quality.",
        play_steps_s=1,
        blocksize=512,
        use_default_speakers_list=True,
    ):
        self.should_listen = should_listen
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        self.gen_kwargs = gen_kwargs
        self.compile_mode = compile_mode
        self.max_prompt_pad_length = max_prompt_pad_length
        self.use_default_speakers_list = use_default_speakers_list
        if self.use_default_speakers_list:
            description = description.replace("Jenny", "")

        self.speaker = "Jason"
        self.description = description

        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=self.torch_dtype
        ).to(device)

        self.description_tokenizer = AutoTokenizer.from_pretrained(self.model.config.text_encoder._name_or_path)
        self.prompt_tokenizer = AutoTokenizer.from_pretrained(model_name)

        framerate = self.model.audio_encoder.config.frame_rate
        self.play_steps = int(framerate * play_steps_s)
        self.blocksize = blocksize
        self.sampling_rate = self.model.config.sampling_rate
        self.target_sr = 16000
        # Pre-compute resampling ratio for scipy resample_poly (faster than librosa)
        g = gcd(self.sampling_rate, self.target_sr)
        self.resample_up = self.target_sr // g
        self.resample_down = self.sampling_rate // g
        logger.info(f"Parler TTS model sample rate: {self.sampling_rate}, resample ratio: {self.resample_up}/{self.resample_down}")

        if self.compile_mode not in (None, "default"):
            logger.warning("Torch compilation modes that capture CUDA graphs are not yet compatible with TTS. Reverting to 'default'")
            self.compile_mode = "default"

        if self.compile_mode:
            torch._dynamo.config.suppress_errors = True
            self.model.forward = torch.compile(self.model.forward, mode=self.compile_mode, fullgraph=False)

        self.warmup()

    def prepare_model_inputs(self, prompt, max_length_prompt=50, pad=False):
        pad_args_prompt = {"padding": "max_length", "max_length": max_length_prompt} if pad else {}

        description = self.description
        if self.use_default_speakers_list:
            description = self.speaker + " " + self.description

        tokenized_description = self.description_tokenizer(description, return_tensors="pt").to(self.device)
        input_ids = tokenized_description.input_ids
        attention_mask = tokenized_description.attention_mask

        tokenized_prompt = self.prompt_tokenizer(prompt, return_tensors="pt", **pad_args_prompt).to(self.device)
        prompt_input_ids = tokenized_prompt.input_ids
        prompt_attention_mask = tokenized_prompt.attention_mask

        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            **self.gen_kwargs,
        }
        return gen_kwargs

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")

        if self.device == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

        n_steps = 1 if self.compile_mode == "default" else 2

        if self.device == "cuda":
            torch.cuda.synchronize()
            start_event.record()

        if self.compile_mode:
            pad_lengths = [2**i for i in range(2, self.max_prompt_pad_length)]
            for pad_length in pad_lengths[::-1]:
                model_kwargs = self.prepare_model_inputs("dummy prompt", max_length_prompt=pad_length, pad=True)
                for _ in range(n_steps):
                    _ = self.model.generate(**model_kwargs)
                logger.info(f"Warmed up length {pad_length} tokens!")
        else:
            model_kwargs = self.prepare_model_inputs("dummy prompt")
            for _ in range(n_steps):
                _ = self.model.generate(**model_kwargs)

        if self.device == "cuda":
            end_event.record()
            torch.cuda.synchronize()
            logger.info(f"{self.__class__.__name__}: warmed up! time: {start_event.elapsed_time(end_event) * 1e-3:.3f} s")

    def process(self, llm_sentence):
        if isinstance(llm_sentence, tuple):
            llm_sentence, language_code = llm_sentence
            self.speaker = WHISPER_LANGUAGE_TO_PARLER_SPEAKER.get(language_code, "Jason")

        console.print(f"[green]ASSISTANT: {llm_sentence}")
        nb_tokens = len(self.prompt_tokenizer(llm_sentence).input_ids)

        pad_args = {}
        if self.compile_mode:
            pad_length = next_power_of_2(nb_tokens)
            logger.debug(f"padding to {pad_length}")
            pad_args["pad"] = True
            pad_args["max_length_prompt"] = pad_length

        tts_gen_kwargs = self.prepare_model_inputs(llm_sentence, **pad_args)

        streamer = ParlerTTSStreamer(self.model, device=self.device, play_steps=self.play_steps)
        tts_gen_kwargs = {"streamer": streamer, **tts_gen_kwargs}
        torch.manual_seed(0)
        thread = Thread(target=self.model.generate, kwargs=tts_gen_kwargs)
        thread.start()

        for i, audio_chunk in enumerate(streamer):
            global pipeline_start
            if i == 0 and "pipeline_start" in globals():
                logger.info(f"Time to first audio: {perf_counter() - pipeline_start:.3f}")
            audio_chunk = resample_poly(audio_chunk, self.resample_up, self.resample_down).astype(np.float32)
            audio_chunk = np.clip(audio_chunk * 32768, -32768, 32767).astype(np.int16)
            for j in range(0, len(audio_chunk), self.blocksize):
                yield np.pad(
                    audio_chunk[j : j + self.blocksize],
                    (0, self.blocksize - len(audio_chunk[j : j + self.blocksize])),
                )

        self.should_listen.set()
