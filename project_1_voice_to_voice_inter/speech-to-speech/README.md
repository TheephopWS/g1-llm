# Speech To Speech: From Hugging face


## Approach

### Structure
This repository implements a speech-to-speech cascaded pipeline consisting of the following parts:
1. **Voice Activity Detection (VAD)**
2. **Speech to Text (STT)**
3. **Language Model (LM)**
4. **Text to Speech (TTS)**

### Modularity
The pipeline provides a fully open and modular approach, with a focus on leveraging models available through the Transformers library on the Hugging Face hub. The code is designed for easy modification, and we already support device-specific and external library implementations:

**VAD** 
- [Silero VAD v5](https://github.com/snakers4/silero-vad)

**STT**
- Any [Whisper](https://huggingface.co/docs/transformers/en/model_doc/whisper) model checkpoint on the Hugging Face Hub through Transformers 🤗, including [whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) and [distil-large-v3](https://huggingface.co/distil-whisper/distil-large-v3)
- ~~[Lightning Whisper MLX](https://github.com/mustafaaljadery/lightning-whisper-mlx?tab=readme-ov-file#lightning-whisper-mlx)~~
- ~~[MLX Audio Whisper](https://github.com/huggingface/mlx-audio) - Fast Whisper inference on Apple Silicon~~
- [Parakeet TDT](https://huggingface.co/nvidia/parakeet-tdt-1.1b) - Real-time streaming STT with sub-100ms latency on Apple Silicon (CUDA/CPU via nano-parakeet, no NeMo)
- ~~[Paraformer - FunASR](https://github.com/modelscope/FunASR)~~

**LLM**
- Any instruction-following model on the [Hugging Face Hub](https://huggingface.co/models?pipeline_tag=text-generation&sort=trending) via Transformers 🤗
- [mlx-lm](https://github.com/ml-explore/mlx-examples/blob/main/llms/README.md)
- [OpenAI API](https://platform.openai.com/docs/quickstart)

**TTS**
- [Parler-TTS](https://github.com/huggingface/parler-tts) 🤗
- ~~[MeloTTS](https://github.com/myshell-ai/MeloTTS)~~
- ~~[ChatTTS](https://github.com/2noise/ChatTTS?tab=readme-ov-file)~~
- ~~[Pocket TTS](https://github.com/kyutai-labs/pocket-tts) - Streaming TTS with voice cloning from Kyutai Labs~~
- ~~[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) - Fast and high-quality TTS optimized for Apple Silicon~~

## Setup

Install the required dependencies:
```bash
pip install -r requirements_STS.txt
```

**Note on DeepFilterNet:** DeepFilterNet (used for optional audio enhancement in VAD) is currently incompatible with Pocket TTS due to numpy version constraints. DeepFilterNet requires numpy<2, while Pocket TTS requires numpy>=2. If you need audio enhancement, you can install DeepFilterNet separately by commenting out pocket-tts in requirements and uncommenting deepfilternet.


## Usage

The pipeline can be run in three ways:
- ~~**Server/Client approach**: Models run on a server, and audio input/output are streamed from a client using TCP sockets.~~
- ~~**WebSocket approach**: Models run on a server, and audio input/output are streamed from a client using WebSockets.~~
- **Local approach**: Runs locally.

### Recommended setup 

1. Running on CUDA
    ```bash
    python s2s_pipeline.py --device cuda
    ```

2. Running with specific input/output device
    ```bash
    python s2s_pipeline.py --device cuda --input_device 1 --output_device 3
    ```

3. Recommended usage with Cuda

Leverage Torch Compile for Whisper and Parler-TTS. **The usage of Parler-TTS allows for audio output streaming, further reducing the overall latency** 🚀:

```bash
python s2s_pipeline.py \
	--lm_model_name microsoft/Phi-3-mini-4k-instruct \
	--stt_compile_mode reduce-overhead \
	--tts_compile_mode default \
    --recv_host 0.0.0.0 \
	--send_host 0.0.0.0 
```

See logs for devides.


### Recommended usage with Cuda

Leverage Torch Compile for Whisper and Parler-TTS. **The usage of Parler-TTS allows for audio output streaming, further reducing the overall latency** 🚀:

```bash
python s2s_pipeline.py \
	--lm_model_name microsoft/Phi-3-mini-4k-instruct \
	--stt_compile_mode reduce-overhead \
	--tts_compile_mode default \
  --recv_host 0.0.0.0 \
	--send_host 0.0.0.0 
```

For the moment, modes capturing CUDA Graphs are not compatible with streaming Parler-TTS (`reduce-overhead`, `max-autotune`).

### Multi-language Support

The pipeline currently supports English, French, Spanish, Chinese, Japanese, and Korean.  
Two use cases are considered:

- **Single-language conversation**: Enforce the language setting using the `--language` flag, specifying the target language code (default is 'en').
- **Language switching**: Set `--language` to 'auto'. In this case, Whisper detects the language for each spoken prompt, and the LLM is prompted with "`Please reply to my message in ...`" to ensure the response is in the detected language.

Please note that you must use STT and LLM checkpoints compatible with the target language(s). For the STT part, Parler-TTS is not yet multilingual (though that feature is coming soon! 🤗). In the meantime, you should use Melo (which supports English, French, Spanish, Chinese, Japanese, and Korean) or Chat-TTS.


## Command-line Usage

> **_NOTE:_** References for all the CLI arguments can be found directly in the [arguments classes](https://github.com/huggingface/speech-to-speech/tree/d5e460721e578fef286c7b64e68ad6a57a25cf1b/arguments_classes) or by running `python s2s_pipeline.py -h`.

### Module level Parameters 
See [ModuleArguments](https://github.com/huggingface/speech-to-speech/blob/d5e460721e578fef286c7b64e68ad6a57a25cf1b/arguments_classes/module_arguments.py) class. Allows to set:
- a common `--device` (if one wants each part to run on the same device)
- `--mode` `local` or `server`
- chosen STT implementation 
- chosen LM implementation
- chose TTS implementation
- logging level

### VAD parameters
See [VADHandlerArguments](https://github.com/huggingface/speech-to-speech/blob/d5e460721e578fef286c7b64e68ad6a57a25cf1b/arguments_classes/vad_arguments.py) class. Notably:
- `--thresh`: Threshold value to trigger voice activity detection.
- `--min_speech_ms`: Minimum duration of detected voice activity to be considered speech.
- `--min_silence_ms`: Minimum length of silence intervals for segmenting speech, balancing sentence cutting and latency reduction.


### STT, LM and TTS parameters

`model_name`, `torch_dtype`, and `device` are exposed for each implementation of the Speech to Text, Language Model, and Text to Speech. Specify the targeted pipeline part with the corresponding prefix (e.g. `stt`, `lm` or `tts`, check the implementations' [arguments classes](https://github.com/huggingface/speech-to-speech/tree/d5e460721e578fef286c7b64e68ad6a57a25cf1b/arguments_classes) for more details).

For example:
```bash
--lm_model_name google/gemma-2b-it
```
