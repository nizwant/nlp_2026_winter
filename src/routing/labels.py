"""Label inventories in the fixed order used for prompts, output schemas and arrays."""

SEMEVAL_LABELS: tuple[str, ...] = (
    "Loaded_Language",
    "Name_Calling-Labeling",
    "Repetition",
    "Doubt",
    "Exaggeration-Minimisation",
    "Appeal_to_Fear-Prejudice",
    "Flag_Waving",
    "Causal_Oversimplification",
    "Conversation_Killer",
    "Questioning_the_Reputation",
    "Appeal_to_Values",
    "Obfuscation-Vagueness-Confusion",
    "False_Dilemma-No_Choice",
    "Red_Herring",
    "Straw_Man",
    "Appeal_to_Authority",
    "Guilt_by_Association",
    "Appeal_to_Popularity",
    "Consequential_Oversimplification",
    "Appeal_to_Time",
    "Appeal_to_Hypocrisy",
    "Slogans",
    "Whataboutism",
)

MIPD_LABELS: tuple[str, ...] = (
    "REFERENCE_ERROR",
    "WHATABOUTISM",
    "STRAWMAN",
    "EMOTIONAL_CONTENT",
    "CHERRY_PICKING",
    "FALSE_CAUSE",
    "MISLEADING_CLICKBAIT",
    "ANECDOTE",
    "LEADING_QUESTIONS",
    "EXAGGERATION",
    "QUOTE_MINING",
)

# The released MIPD files spell this column without the final T.
MIPD_COLUMN_RENAMES = {"MISLEADING_CLICKBAI": "MISLEADING_CLICKBAIT"}

SOURCES: dict[str, dict] = {
    "semeval_pl": {"labels": SEMEVAL_LABELS, "language": "pl", "task": "semeval", "unit": "paragraph"},
    "semeval_en": {"labels": SEMEVAL_LABELS, "language": "en", "task": "semeval", "unit": "paragraph"},
    "mipd": {"labels": MIPD_LABELS, "language": "pl", "task": "mipd", "unit": "article"},
}


def labels_for(source: str) -> tuple[str, ...]:
    return SOURCES[source]["labels"]
