"""
Allowed actions for the Unitree G1 robot.

This mirrors the ALLOWED_ACTIONS from server.py so the LLM knows
which actions are available and the dispatcher can validate them.
"""

from typing import Dict

# Action name -> description (used in LLM system prompt)
ALLOWED_ACTIONS: Dict[str, str] = {
    "NONE": "Do nothing / no physical action. Use this for normal conversation.",
    "MOVE_FORWARD": "Walk forward toward the user.",
    "DANCE": "Perform a dance routine.",
}

DEFAULT_ACTION = "NONE"
