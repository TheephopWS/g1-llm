"""
LLM Output Processor - extracts action commands and forwards clean text to TTS.

Actions matching ALLOWED_ACTIONS are dispatched to the UnitreeActionDispatcher.
Clean text (with action tags stripped) is forwarded to TTS for speech output.
"""

import logging
from baseHandler import BaseHandler
from actions.action_dispatcher import UnitreeActionDispatcher
from actions.allowed_actions import ALLOWED_ACTIONS

logger = logging.getLogger(__name__)


class LMOutputProcessor(BaseHandler):
    """
    Input: (text, language_code, tools) tuples from LLM
    Output: (text, language_code) tuples to TTS
    Side effects:
      - Dispatches robot actions via UnitreeActionDispatcher
      - Sends messages to text_output_queue
    """

    def setup(self, text_output_queue, simulate_actions=True):
        self.text_output_queue = text_output_queue
        self.action_dispatcher = UnitreeActionDispatcher(simulate=simulate_actions)
        logger.info(f"Action dispatcher initialized (simulate={simulate_actions})")
        logger.info(f"Available actions: {list(ALLOWED_ACTIONS.keys())}")

    def process(self, lm_output):
        text_chunk, language_code, tools = lm_output
        logger.debug(f"LM processor: text='{text_chunk}', tools={tools}")

        # Separate robot actions from other tools
        actions = []
        other_tools = []
        for tool in tools:
            tool_name = tool.get("name", "").upper()
            if tool.get("type") == "action" or tool_name in ALLOWED_ACTIONS:
                actions.append(tool)
            else:
                other_tools.append(tool)

        # Dispatch robot actions
        for action in actions:
            action_name = action["name"].upper()
            params = action.get("parameters", {})
            logger.info(f"Dispatching action: {action_name} with params: {params}")
            success = self.action_dispatcher.dispatch(action_name, params)
            if not success:
                logger.warning(f"Action {action_name} failed")

        # Build message for text_output_queue
        message = {"type": "assistant_text", "text": text_chunk}
        if actions:
            message["actions"] = [{"action": a["name"].upper(), "params": a.get("parameters", {})} for a in actions]
        if other_tools:
            message["tools"] = other_tools
        self.text_output_queue.put(message)

        # Forward clean text to TTS (action tags already stripped by parser)
        yield (text_chunk, language_code)
