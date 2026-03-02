from typing import Dict

# ---------------------------------------------------------------------------
# Allowed robot actions  (single source of truth)
# ---------------------------------------------------------------------------
ALLOWED_ACTIONS: Dict[str, str] = {
    "NONE": "Do nothing / no physical action. Use this for normal conversation.",
    "MOVE_FORWARD": "Walk forward toward the user.",
    "DANCE": "Perform a dance routine.",
}

DEFAULT_ACTION = "NONE"

# ---------------------------------------------------------------------------
# choose_action tool definition (presented to the LLM)
# ---------------------------------------------------------------------------
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


def build_tool_prompt() -> str:
    action_lines = "\n".join(f"  - {k}: {v}" for k, v in ALLOWED_ACTIONS.items())

    return (
        f"""
        You are a robot assistant controlling a Unitree G1 humanoid robot.\n
        Your response follows a two-step pipeline:\n
            Step 1  ACTION — Call the choose_action tool to pick a physical action.\n
            Step 2  SPEAK  — Generate the spoken reply (what the user hears).\n
        
        ## Tool: choose_action
        You have exactly one tool: choose_action(action)
        Allowed values for `action`:
        {action_lines}
        Call format (append at the very END of your spoken text):
            [ACTION:ACTION_NAME]

        ## Examples
        User: Walk over here.
        Assistant: Okay, I'll walk to you! [ACTION:MOVE_FORWARD]

        User: Show me a dance move.
        Assistant: Yes sir, I'll dance now! [ACTION:DANCE]

        User: What is the weather today?
        Assistant: The weather is not bad today. [ACTION:NONE]

        ## Rules\n"
        1. MUST choose physical action in {list(ALLOWED_ACTIONS.keys())[1:]} when user requests movement.
        1. Include action in the response with format [ACTION:ACTION_NAME]. Every response must contain exactly one [ACTION:…] tag at the end.\n"
        2. Use NONE when no physical movement is appropriate.\n"
        3. Keep spoken text under 20 words. Be brief and direct.\n"
        4. DO NOT explain yourself. Just answer."
        """
    )
