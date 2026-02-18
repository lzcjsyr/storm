# File role: Single source of truth for LM routing sections, roles, and default token budgets.
# Relation: Consumed by lm_routing loader and LM config containers to avoid duplicated hardcoded role maps.

SECTION_TO_ROLES = {
    "storm_wiki": [
        "conv_simulator_lm",
        "question_asker_lm",
        "outline_gen_lm",
        "article_gen_lm",
        "article_polish_lm",
    ],
    "co_storm": [
        "question_answering_lm",
        "discourse_manage_lm",
        "utterance_polishing_lm",
        "warmstart_outline_gen_lm",
        "question_asking_lm",
        "knowledge_base_lm",
    ],
}

SECTION_DEFAULT_MAX_TOKENS = {
    "storm_wiki": {
        "conv_simulator_lm": 500,
        "question_asker_lm": 500,
        "outline_gen_lm": 400,
        "article_gen_lm": 700,
        "article_polish_lm": 4000,
    },
    "co_storm": {
        "question_answering_lm": 1000,
        "discourse_manage_lm": 500,
        "utterance_polishing_lm": 2000,
        "warmstart_outline_gen_lm": 500,
        "question_asking_lm": 300,
        "knowledge_base_lm": 1000,
    },
}
