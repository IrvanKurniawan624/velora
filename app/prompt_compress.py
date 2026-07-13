"""Prompt compression for remote (hybrid) escalations.

Strips Python comments/docstrings from code blocks and prunes generic verbal
filler so escalated prompts cost fewer remote tokens without changing their
meaning. Task-specific template rewrites are intentionally NOT included - they
overfit known phrasings. Zero mode never calls remote, so this only matters in
hybrid mode, and only when enabled via COMPRESS_REMOTE=1 (off by default so the
engine's behavior stays identical to the verified baseline).
"""
import io
import re
import tokenize


def _strip_py_comments(source: str) -> str:
    """Remove comments and docstrings from Python source using tokenize."""
    try:
        out = ""
        prev = tokenize.INDENT
        last_line, last_col = -1, 0
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT:
                continue
            # a string right after an indent/newline is a docstring
            if tok.type == tokenize.STRING and prev in (tokenize.INDENT, tokenize.NEWLINE, tokenize.NL):
                continue
            if tok.start[0] > last_line:
                last_col = 0
            if tok.start[1] > last_col:
                out += " " * (tok.start[1] - last_col)
            out += tok.string
            prev = tok.type
            last_line, last_col = tok.end
        lines = [ln for ln in out.splitlines() if ln.strip()]
        return "\n".join(lines).strip()
    except Exception:
        return source


def compress_prompt(prompt: str, task_type: str) -> str:
    """Compress a task prompt: strip code-block comments (code tasks only) and
    remove polite/transition filler. Meaning-preserving."""
    text = prompt
    if task_type in ("code", "code_gen", "code_debug"):
        for block in re.findall(r"```python(.*?)```", text, re.DOTALL):
            text = text.replace(block, "\n" + _strip_py_comments(block) + "\n")
        if "def " in text and "```" not in text:
            m = re.search(r"(def .*?)(?=\n\n|\Z)", text, re.DOTALL)
            if m:
                text = text.replace(m.group(1), _strip_py_comments(m.group(1)))
    # drop polite + transition filler only (never instruction verbs)
    text = re.sub(r"\b(please|kindly|could you|would you mind)\s+", "", text, flags=re.I)
    text = re.sub(r"\b(the following|of the following|for the following|below)\s+", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text
