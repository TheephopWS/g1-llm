"""
Minimal local Parakeet CUDA pipeline.

Usage:
    python s2s_pipeline.py
    python s2s_pipeline.py --device cuda --lm_model_name microsoft/Phi-3-mini-4k-instruct
"""

import logging
import os
import sys
import signal
from copy import copy
from pathlib import Path
from queue import Queue
from threading import Event
from typing import Optional

from arguments_classes.module_arguments import ModuleArguments
from arguments_classes.vad_arguments import VADHandlerArguments
from arguments_classes.whisper_stt_arguments import WhisperSTTHandlerArguments
from arguments_classes.parakeet_tdt_arguments import ParakeetTDTSTTHandlerArguments
from arguments_classes.language_model_arguments import LanguageModelHandlerArguments
from arguments_classes.parler_tts_arguments import ParlerTTSHandlerArguments
from arguments_classes.socket_receiver_arguments import SocketReceiverArguments
from arguments_classes.socket_sender_arguments import SocketSenderArguments

import torch
import nltk
from rich.console import Console
from transformers import HfArgumentParser

from utils.thread_manager import ThreadManager

# Ensure NLTK resources are available
try:
    nltk.data.find("tokenizers/punkt_tab")
except (LookupError, OSError):
    nltk.download("punkt_tab")
try:
    nltk.data.find("tokenizers/averaged_perceptron_tagger_eng")
except (LookupError, OSError):
    nltk.download("averaged_perceptron_tagger_eng")

CURRENT_DIR = Path(__file__).resolve().parent
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(CURRENT_DIR, "tmp")

# Enable persistent torch inductor cache — avoids recompilation between runs
torch._inductor.config.fx_graph_cache = True
torch._inductor.config.fx_graph_remote_cache = False

console = Console()
logging.getLogger("numba").setLevel(logging.WARNING)


def rename_args(args, prefix):
    gen_kwargs = {}
    for key in copy(args.__dict__):
        if key.startswith(prefix):
            value = args.__dict__.pop(key)
            new_key = key[len(prefix) + 1:]
            if new_key.startswith("gen_"):
                gen_kwargs[new_key[4:]] = value
            else:
                args.__dict__[new_key] = value
    args.__dict__["gen_kwargs"] = gen_kwargs


def parse_arguments():
    parser = HfArgumentParser(
        (
            ModuleArguments,
            SocketReceiverArguments,
            SocketSenderArguments,
            VADHandlerArguments,
            WhisperSTTHandlerArguments,
            ParakeetTDTSTTHandlerArguments,
            LanguageModelHandlerArguments,
            ParlerTTSHandlerArguments,
        )
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        return parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        return parser.parse_args_into_dataclasses()


def setup_logger(log_level):
    global logger
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    if log_level == "debug":
        torch._logging.set_logs(graph_breaks=True, recompiles=True, cudagraphs=True)


def overwrite_device_argument(common_device, *handler_kwargs):
    if common_device:
        for kwargs in handler_kwargs:
            if hasattr(kwargs, "lm_device"):
                kwargs.lm_device = common_device
            if hasattr(kwargs, "tts_device"):
                kwargs.tts_device = common_device
            if hasattr(kwargs, "stt_device"):
                kwargs.stt_device = common_device
            if hasattr(kwargs, "parakeet_tdt_device"):
                kwargs.parakeet_tdt_device = common_device


def main():
    (
        module_kwargs,
        socket_receiver_kwargs,
        socket_sender_kwargs,
        vad_handler_kwargs,
        whisper_stt_handler_kwargs,
        parakeet_tdt_stt_handler_kwargs,
        language_model_handler_kwargs,
        parler_tts_handler_kwargs,
    ) = parse_arguments()

    setup_logger(module_kwargs.log_level)

    # Override device across all handlers
    overwrite_device_argument(
        module_kwargs.device,
        whisper_stt_handler_kwargs,
        parakeet_tdt_stt_handler_kwargs,
        language_model_handler_kwargs,
        parler_tts_handler_kwargs,
    )

    # Rename prefixed args
    rename_args(whisper_stt_handler_kwargs, "stt")
    rename_args(parakeet_tdt_stt_handler_kwargs, "parakeet_tdt")
    rename_args(language_model_handler_kwargs, "lm")
    rename_args(parler_tts_handler_kwargs, "tts")

    # Queues and events
    stop_event = Event()
    should_listen = Event()
    recv_audio_chunks_queue = Queue()
    send_audio_chunks_queue = Queue()
    spoken_prompt_queue = Queue()
    text_prompt_queue = Queue()
    lm_response_queue = Queue()
    lm_processed_queue = Queue()
    text_output_queue = Queue()

    # --- Comms: local or socket ---
    if module_kwargs.mode == "local":
        from connections.local_audio_streamer import LocalAudioStreamer

        local_audio_streamer = LocalAudioStreamer(
            input_queue=recv_audio_chunks_queue,
            output_queue=send_audio_chunks_queue,
            input_device=module_kwargs.input_device,
            output_device=module_kwargs.output_device,
        )
        comms_handlers = [local_audio_streamer]
        should_listen.set()
    else:
        from connections.socket_receiver import SocketReceiver
        from connections.socket_sender import SocketSender

        comms_handlers = [
            SocketReceiver(
                stop_event,
                recv_audio_chunks_queue,
                should_listen,
                host=socket_receiver_kwargs.recv_host,
                port=socket_receiver_kwargs.recv_port,
                chunk_size=socket_receiver_kwargs.chunk_size,
            ),
            SocketSender(
                stop_event,
                send_audio_chunks_queue,
                host=socket_sender_kwargs.send_host,
                port=socket_sender_kwargs.send_port,
            ),
        ]

    # --- VAD ---
    from VAD.vad_handler import VADHandler

    vad_setup_kwargs = vars(vad_handler_kwargs)
    vad_setup_kwargs["text_output_queue"] = text_output_queue

    vad = VADHandler(
        stop_event,
        queue_in=recv_audio_chunks_queue,
        queue_out=spoken_prompt_queue,
        setup_args=(should_listen,),
        setup_kwargs=vad_setup_kwargs,
    )

    # --- STT ---
    if module_kwargs.stt == "parakeet-tdt":
        from STT.parakeet_tdt_handler import ParakeetTDTSTTHandler

        stt = ParakeetTDTSTTHandler(
                stop_event,
                queue_in=spoken_prompt_queue,
                queue_out=text_prompt_queue,
                setup_kwargs=vars(parakeet_tdt_stt_handler_kwargs),
            )
    else:
        from STT.whisper_stt_handler import WhisperSTTHandler

        stt = WhisperSTTHandler(
            stop_event,
            queue_in=spoken_prompt_queue,
            queue_out=text_prompt_queue,
            setup_kwargs=vars(whisper_stt_handler_kwargs),
        )

    # --- LLM: Transformers ---
    from LLM.language_model import LanguageModelHandler

    lm = LanguageModelHandler(
        stop_event,
        queue_in=text_prompt_queue,
        queue_out=lm_response_queue,
        setup_kwargs=vars(language_model_handler_kwargs),
    )

    # --- LM Output Processor ---
    from LLM.lm_output_processor import LMOutputProcessor

    lm_processor = LMOutputProcessor(
        stop_event,
        queue_in=lm_response_queue,
        queue_out=lm_processed_queue,
        setup_kwargs={
            "text_output_queue": text_output_queue,
            "simulate_actions": module_kwargs.simulate_actions,
        },
    )

    # --- TTS: Parler ---
    from TTS.parler_handler import ParlerTTSHandler

    tts = ParlerTTSHandler(
        stop_event,
        queue_in=lm_processed_queue,
        queue_out=send_audio_chunks_queue,
        setup_args=(should_listen,),
        setup_kwargs=vars(parler_tts_handler_kwargs),
    )

    # --- Build and run ---
    pipeline_manager = ThreadManager(
        [*comms_handlers, vad, stt, lm, lm_processor, tts]
    )

    shutdown_requested = [False]

    def signal_handler(_sig, _frame):
        if not shutdown_requested[0]:
            shutdown_requested[0] = True
            console.print("\n[yellow]Shutting down gracefully...[/yellow]")
            pipeline_manager.stop()
            console.print("[green]✓ Pipeline stopped successfully[/green]")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        pipeline_manager.start()
    except KeyboardInterrupt:
        if not shutdown_requested[0]:
            console.print("\n[yellow]Shutting down gracefully...[/yellow]")
            pipeline_manager.stop()
            console.print("[green]✓ Pipeline stopped successfully[/green]")


if __name__ == "__main__":
    main()
