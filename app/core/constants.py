from enum import StrEnum
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


class Category(StrEnum):
    FACTUAL = "factual"
    MATH = "math"
    SENTIMENT = "sentiment"
    LOGIC = "logic"
    SUMMARIZE = "summarize"
    NER = "ner"
    CODE_DEBUG = "code_debug"
    CODE_GEN = "code_gen"


CATEGORY_PROMPT_FILES: dict[Category, str] = {
    Category.FACTUAL: "factual.txt",
    Category.MATH: "math_reasoning.txt",
    Category.SENTIMENT: "sentiment.txt",
    Category.LOGIC: "logic_reasoning.txt",
    Category.SUMMARIZE: "summarize.txt",
    Category.NER: "ner.txt",
    Category.CODE_DEBUG: "code_debug.txt",
    Category.CODE_GEN: "code_gen.txt",
}
