"""
Unitree G1 Action Dispatcher for speech-to-speech pipeline.

Follows the same interface as UnitreeActionDispatcher in robot_client.py.
When running locally on the laptop, actions are simulated (printed).
When deployed on the Jetson Orin, replace simulate=False and add real SDK calls.
"""

import logging
from typing import Any, Callable, Dict

from actions.allowed_actions import ALLOWED_ACTIONS, DEFAULT_ACTION

logger = logging.getLogger(__name__)


class UnitreeActionDispatcher:
    """
    Dispatches actions to Unitree G1 robot.

    Usage:
        dispatcher = UnitreeActionDispatcher(simulate=True)
        dispatcher.dispatch("MOVE_FORWARD", {"distance": 1.0})

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

    def dispatch(self, action: str, params: Dict[str, Any] = None) -> bool:
        """
        Dispatch action to robot.

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

    # -------------------------------------------------------------------------
    # Action Implementations - Replace with actual Unitree G1 SDK calls
    # -------------------------------------------------------------------------
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
