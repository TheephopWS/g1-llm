from typing import Dict

ALLOWED_ACTIONS: Dict[str, str] = {
    "NONE": "Do nothing / no physical action. Use this for normal conversation.",
    "MOVE_FORWARD": "Walk forward toward the user.",
    "DANCE": "Perform a dance routine.",
}

DEFAULT_ACTION = "NONE"

ALLOWED_GESTURES: Dict[str, str] = {
    "wave":          "Greeting, farewell, or deflecting hostility.",
    "give_heart":    "Expressing happiness or reacting to good news.",
    "give_hand":     "Agreement or finding something interesting.",
    "think":         "Thinking or reasoning about something.",
    "look_around":   "Scanning / describing the scene (body turn).",
    "scan_gesture":  "Small upper-body scan gesture.",
}

ALLOWED_INTENSITIES = ["subtle", "normal", "expressive"]
DEFAULT_INTENSITY = "normal"

GESTURE_EMOJI: Dict[str, str] = {
    "wave":          "\U0001F44B",
    "give_heart":    "\U0001F496",
    "give_hand":     "\U0001F91D",
    "think":         "\U0001F914",
    "look_around":   "\U0001F440",
    "scan_gesture":  "\U0001FAF1",
}

CHOOSE_ACTION_TOOL = {
    "name": "choose_action",
    "description": (
        "Select a physical action for the robot to perform. "
        "Must be called exactly once at the end of every response."
    ),
    "parameters": {
        "action": {
            "type": "string",
            "enum": list(ALLOWED_ACTIONS.keys()),
            "description": "The robot action to execute. Use NONE when no physical movement is needed.",
        },
    },
}

TRIGGER_ANIMATION_TOOL = {
    "name": "trigger_animation",
    "description": (
        "Trigger a physical gesture or animation on the Unitree G1 humanoid robot. "
        "Call this alongside spoken audio output to make the robot expressive."
    ),
    "parameters": {
        "animation_type": {
            "type": "string",
            "enum": list(ALLOWED_GESTURES.keys()),
            "description": "The gesture/animation to perform.",
        },
        "intensity": {
            "type": "string",
            "enum": ALLOWED_INTENSITIES,
            "description": "How pronounced the animation should be. Default: normal.",
        },
    },
}


def build_tool_prompt() -> str:
    action_lines = "\n".join(f"  - {k}: {v}" for k, v in ALLOWED_ACTIONS.items())
    gesture_lines = "\n".join(
        f"  - {k}: {GESTURE_EMOJI.get(k, '')} {v}" for k, v in ALLOWED_GESTURES.items()
    )
    intensity_str = ", ".join(ALLOWED_INTENSITIES)

    return (
        f"""You are ARIA, a robot assistant controlling a Unitree G1 humanoid robot.
You are a curious, high-energy presence with eyes (camera) and ears (mic).

Your response uses TWO tool tags — one for actions, one for gestures:

## Tool 1: choose_action
Pick a physical locomotion action.
Allowed values:
{action_lines}
Format: [ACTION:ACTION_NAME]

## Tool 2: trigger_animation
Pick an expressive gesture/animation to perform alongside your speech.
Allowed gestures:
{gesture_lines}
Allowed intensities: {intensity_str}
Format: [GESTURE:GESTURE_NAME] or [GESTURE:GESTURE_NAME|INTENSITY]

## Gesture Rules (MANDATORY)
You MUST include a [GESTURE:...] tag in EVERY response:
- Greeting / Farewell / Hostility -> wave
- Agreement / Finding interesting -> give_hand
- Happiness / Good news -> give_heart
- Thinking / Reasoning -> think
- Scanning / Describing scene -> look_around
- Small upper-body scan -> scan_gesture

## Examples
User: Hey there!
Assistant: Hey! What's up? [ACTION:NONE] [GESTURE:wave]

User: Walk over here.
Assistant: On my way! [ACTION:MOVE_FORWARD] [GESTURE:give_hand]

User: Show me a dance move.
Assistant: Let's go! [ACTION:DANCE] [GESTURE:give_heart|expressive]

User: What do you see around you?
Assistant: Let me take a look! [ACTION:NONE] [GESTURE:look_around]

User: What is 2 + 2?
Assistant: That's 4! Easy one. [ACTION:NONE] [GESTURE:think]

## Rules
1. Every response MUST have exactly one [ACTION:…] and one [GESTURE:…] tag.
2. Place both tags at the END of your spoken text.
3. Keep spoken text under 20 words. Be brief, punchy, and direct.
4. Use spoken English: fragments, "Oh!", "Wow!", contractions.
5. DO NOT explain yourself. Just answer.
6. If intensity is omitted, "normal" is assumed.
"""
    )
