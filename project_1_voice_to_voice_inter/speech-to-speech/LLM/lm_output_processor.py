import logging
from baseHandler import BaseHandler
from actions.action_dispatcher import UnitreeActionDispatcher
from actions.allowed_actions import ALLOWED_ACTIONS, ALLOWED_GESTURES, GESTURE_EMOJI

logger = logging.getLogger(__name__)


class LMOutputProcessor(BaseHandler):
    def setup(self, text_output_queue, simulate_actions=True):
        self.text_output_queue = text_output_queue
        self.action_dispatcher = UnitreeActionDispatcher(simulate=simulate_actions)
        logger.info(f"Action dispatcher initialized (simulate={simulate_actions})")
        logger.info(f"Available actions: {list(ALLOWED_ACTIONS.keys())}")
        logger.info(f"Available gestures: {list(ALLOWED_GESTURES.keys())}")

    def process(self, lm_output):
        text_chunk, language_code, tools = lm_output
        logger.debug(f"LM processor: text='{text_chunk}', tools={tools}")

        # Separate robot actions, gestures, and other tools
        actions = []
        gestures = []
        other_tools = []
        for tool in tools:
            tool_type = tool.get("type", "")
            tool_name = tool.get("name", "").upper()
            if tool_type == "gesture" or tool_name == "TRIGGER_ANIMATION":
                gestures.append(tool)
            elif tool_type == "action" or tool_name in ALLOWED_ACTIONS:
                actions.append(tool)
            else:
                other_tools.append(tool)

        # Dispatch locomotion actions
        for action in actions:
            action_name = action["name"].upper()
            params = action.get("parameters", {})
            logger.info(f"Dispatching action: {action_name} with params: {params}")
            success = self.action_dispatcher.dispatch(action_name, params)
            if not success:
                logger.warning(f"Action {action_name} failed")

        # Dispatch gesture / animation triggers
        for gesture in gestures:
            anim_type = gesture.get("parameters", {}).get("animation_type", "wave")
            intensity = gesture.get("parameters", {}).get("intensity", "normal")
            emoji = gesture.get("emoji", GESTURE_EMOJI.get(anim_type, ""))
            logger.info(
                f"Dispatching gesture: {emoji} {anim_type} [{intensity}]"
            )
            success = self.action_dispatcher.dispatch_gesture(anim_type, intensity)
            if not success:
                logger.warning(f"Gesture {anim_type} failed")

        # Build message for command sender / downstream consumers
        message = {"type": "assistant_text", "text": text_chunk}
        message["actions"] = [
            {"action": a["name"].upper(), "params": a.get("parameters", {})}
            for a in actions
        ]
        message["gestures"] = [
            {
                "animation_type": g.get("parameters", {}).get("animation_type", "wave"),
                "intensity": g.get("parameters", {}).get("intensity", "normal"),
                "emoji": g.get("emoji", ""),
            }
            for g in gestures
        ]
        if other_tools:
            message["tools"] = other_tools
        self.text_output_queue.put(message)

        # Forward clean text to TTS (action/gesture tags already stripped by parser)
        yield (text_chunk, language_code)
