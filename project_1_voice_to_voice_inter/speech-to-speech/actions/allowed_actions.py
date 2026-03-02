from typing import Dict

ALLOWED_ACTIONS: Dict[str, str] = {
    "NONE": "Do nothing / no physical action. Use this for normal conversation.",
    "MOVE_FORWARD": "Walk forward toward the user.",
    "DANCE": "Perform a dance routine.",
}

DEFAULT_ACTION = "NONE"
