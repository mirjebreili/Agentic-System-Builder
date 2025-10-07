from typing import List, TypedDict


class PlannerState(TypedDict):
    user_question: str
    plugin_descriptions: str
    plugin_sequence: List[str]
