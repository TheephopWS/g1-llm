import logging
from typing import Any, Callable, Dict

from actions.allowed_actions import (
    ALLOWED_ACTIONS, DEFAULT_ACTION,
    ALLOWED_GESTURES, ALLOWED_INTENSITIES, DEFAULT_INTENSITY,
    GESTURE_EMOJI,
)

logger = logging.getLogger(__name__)


class UnitreeActionDispatcher:
    """
    Dispatches actions and gestures to Unitree G1 robot.

    Usage:
        dispatcher = UnitreeActionDispatcher(simulate=True)
        dispatcher.dispatch("MOVE_FORWARD", {"distance": 1.0})
        dispatcher.dispatch_gesture("wave", "normal")

    Replace placeholder implementations with actual Unitree G1 SDK calls
    when deploying on the robot.
    """

    def __init__(self, simulate: bool = True):
        """
        Args:
            simulate: If True, only print actions instead of executing.
        """
        self.simulate = simulate
        self._action_handlers: Dict[str, Callable] = {
            "NONE": self._action_none,
            "MOVE_FORWARD": self._action_move_forward,
            "DANCE": self._action_dance,
        }
        self._gesture_handlers: Dict[str, Callable] = {
            "wave": self._gesture_wave,
            "give_heart": self._gesture_give_heart,
            "give_hand": self._gesture_give_hand,
            "think": self._gesture_think,
            "look_around": self._gesture_look_around,
            "scan_gesture": self._gesture_scan,
        }

    # ── Locomotion action dispatch ────────────────────────────────────────

    def dispatch(self, action: str, params: Dict[str, Any] = None) -> bool:
        """
        Dispatch a locomotion action to robot.

        Args:
            action: Action name (must be in ALLOWED_ACTIONS)
            params: Optional parameters for the action

        Returns:
            True if action executed successfully
        """
        params = params or {}

        # Normalize and validate
        action = str(action).strip().upper()
        if action not in ALLOWED_ACTIONS:
            logger.warning(f"Unknown action '{action}', defaulting to {DEFAULT_ACTION}")
            action = DEFAULT_ACTION
            params = {}

        handler = self._action_handlers.get(action)
        if handler is None:
            logger.warning(f"No handler for action '{action}'")
            return False

        try:
            return handler(params)
        except Exception as e:
            logger.error(f"Error executing {action}: {e}")
            return False

    # ── Gesture / animation dispatch ──────────────────────────────────────

    def dispatch_gesture(
        self,
        animation_type: str,
        intensity: str = DEFAULT_INTENSITY,
    ) -> bool:
        """
        Dispatch a gesture / animation to the robot.

        Args:
            animation_type: One of ALLOWED_GESTURES keys.
            intensity: "subtle", "normal", or "expressive".

        Returns:
            True if gesture executed (or simulated) successfully.
        """
        animation_type = str(animation_type).strip().lower()
        intensity = str(intensity).strip().lower()
        if intensity not in ALLOWED_INTENSITIES:
            intensity = DEFAULT_INTENSITY

        emoji = GESTURE_EMOJI.get(animation_type, "")

        if animation_type not in ALLOWED_GESTURES:
            logger.warning(f"Unknown gesture '{animation_type}', ignoring")
            return False

        handler = self._gesture_handlers.get(animation_type)
        if handler is None:
            logger.warning(f"No handler for gesture '{animation_type}'")
            return False

        try:
            return handler(intensity, emoji)
        except Exception as e:
            logger.error(f"Error executing gesture {animation_type}: {e}")
            return False

    # ── Action implementations (replace with Unitree SDK) ────────────────

    def _action_none(self, params: Dict) -> bool:
        if self.simulate:
            logger.info("[ACTION] NONE - No action taken")
        return True

    def _action_move_forward(self, params: Dict) -> bool:
        distance = params.get("distance", 1.0)
        if self.simulate:
            logger.info(f"[ACTION] MOVE_FORWARD - Walking forward {distance}m")
        else:
            # TODO: unitree_g1.walk_forward(distance)
            pass
        return True

    def _action_dance(self, params: Dict) -> bool:
        if self.simulate:
            logger.info("[ACTION] DANCE - Dancing!")
        else:
            # TODO: unitree_g1.dance()
            pass
        return True

    # ── Gesture implementations (replace with Unitree SDK) ───────────────
    # Each receives (intensity, emoji). In simulate mode they log; in real
    # mode they should call G1ArmActionClient / LocoClient.

    def _gesture_wave(self, intensity: str, emoji: str) -> bool:
        if self.simulate:
            logger.info(f"[GESTURE] {emoji} WAVE [{intensity}]")
        else:
            # TODO: arm_client.ExecuteAction(action_map["high wave"])
            pass
        return True

    def _gesture_give_heart(self, intensity: str, emoji: str) -> bool:
        if self.simulate:
            logger.info(f"[GESTURE] {emoji} GIVE_HEART [{intensity}]")
        else:
            # TODO: arm_client.ExecuteAction(action_map["heart"])
            pass
        return True

    def _gesture_give_hand(self, intensity: str, emoji: str) -> bool:
        if self.simulate:
            logger.info(f"[GESTURE] {emoji} GIVE_HAND [{intensity}]")
        else:
            # TODO: arm_client.ExecuteAction(action_map["right hand up"])
            pass
        return True

    def _gesture_think(self, intensity: str, emoji: str) -> bool:
        if self.simulate:
            logger.info(f"[GESTURE] {emoji} THINK [{intensity}]")
        else:
            # TODO: arm_client.ExecuteAction(action_map["face wave"])
            pass
        return True

    def _gesture_look_around(self, intensity: str, emoji: str) -> bool:
        if self.simulate:
            logger.info(f"[GESTURE] {emoji} LOOK_AROUND [{intensity}]")
        else:
            # TODO: loco_client.SetVelocity(0.0, 0.0, yaw_rate, duration)
            pass
        return True

    def _gesture_scan(self, intensity: str, emoji: str) -> bool:
        if self.simulate:
            logger.info(f"[GESTURE] {emoji} SCAN_GESTURE [{intensity}]")
        else:
            # TODO: arm_client sequence: left_kiss -> right_kiss -> release
            pass
        return True
