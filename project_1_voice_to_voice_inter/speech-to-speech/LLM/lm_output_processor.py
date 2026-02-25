"""
LLM Output Processor - extracts tool calls and forwards clean text to TTS.
"""

import logging
from baseHandler import BaseHandler

logger = logging.getLogger(__name__)


class LMOutputProcessor(BaseHandler):
    """
    Input: (text, language_code, tools) tuples from LLM
    Output: (text, language_code) tuples to TTS
    Side effect: Sends messages to text_output_queue
    """

    def setup(self, text_output_queue):
        self.text_output_queue = text_output_queue

    def process(self, lm_output):
        text_chunk, language_code, tools = lm_output
        logger.debug(f"LM processor: text='{text_chunk}', tools={tools}")

        if tools:
            message = {"type": "assistant_text", "text": text_chunk, "tools": tools}
        else:
            message = {"type": "assistant_text", "text": text_chunk}

        self.text_output_queue.put(message)
        yield (text_chunk, language_code)
