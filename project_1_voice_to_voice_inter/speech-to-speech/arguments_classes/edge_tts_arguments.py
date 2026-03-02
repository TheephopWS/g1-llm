from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EdgeTTSHandlerArguments:
    edge_tts_voice: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Edge TTS voice name (e.g., 'en-US-GuyNeural', 'zh-CN-XiaoxiaoNeural'). "
                "If None, auto-selects based on detected language. "
                "List voices with: edge-tts --list-voices"
            )
        },
    )
    edge_tts_rate: str = field(
        default="+0%",
        metadata={"help": "Speech rate adjustment (e.g., '+10%%', '-5%%'). Default is '+0%%'."},
    )
    edge_tts_volume: str = field(
        default="+0%",
        metadata={"help": "Volume adjustment (e.g., '+0%%'). Default is '+0%%'."},
    )
