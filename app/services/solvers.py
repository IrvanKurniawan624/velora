"""Deterministic solvers: answer at zero cost when the pattern is trivial.

Each solver returns (answer, confidence) or None when it does not know.
No task-specific answers are hardcoded: only general rules (arithmetic,
sentiment lexicons, general-knowledge gazetteers) that generalize to
unseen variants.
"""
import ast
import operator
import re

# --------------------------------------------------------------------- math

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}


def _safe_eval(expr):
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        raise ValueError("disallowed expression")

    return ev(ast.parse(expr, mode="eval"))


def _fmt(x):
    if isinstance(x, float) and x.is_integer():
        x = int(x)
    return f"{round(x, 6):g}" if isinstance(x, float) else str(x)


def solve_math(prompt):
    text = prompt.lower().strip().rstrip("?").strip()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+(\d+(?:\.\d+)?)", text)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return _fmt(a * b / 100), 0.95
    m = re.search(r"(?:what is|calculate|compute|evaluate|solve)\s+([\d\s.+\-*/()%]+)$", text)
    if m:
        expr = m.group(1).strip()
        if re.search(r"\d", expr) and re.search(r"[+\-*/%]", expr):
            try:
                return _fmt(_safe_eval(expr)), 0.95
            except Exception:
                return None
    return None


# ---------------------------------------------------------------- sentiment

_POS = ("great", "love", "amazing", "excellent", "fantastic", "wonderful", "perfect",
        "awesome", "good", "nice", "solid", "fast", "reliable", "beautiful", "impressive",
        "comfortable", "helpful", "smooth", "outstanding", "brilliant", "superb", "enjoy",
        "delight", "recommend", "happy", "pleased", "satisfied", "best", "sturdy", "crisp")
_NEG = ("bad", "terrible", "awful", "horrible", "poor", "worst", "hate", "disappoint",
        "broken", "crash", "bug", "slow", "cheap", "flimsy", "scratch", "annoying",
        "useless", "defective", "fail", "waste", "refund", "return", "unusable", "laggy",
        "noisy", "uncomfortable", "overpriced", "mediocre", "frustrat", "regret")
_NEGATIONS = {"not", "no", "never", "hardly", "barely", "isn't", "wasn't", "doesn't",
              "don't", "didn't", "can't", "cannot", "won't", "aren't"}


def _sentiment_hits(text):
    """Return (positive words, negative words) found, accounting for negations."""
    words = re.findall(r"[a-z']+", text.lower())
    pos, neg = set(), set()
    for i, w in enumerate(words):
        negated = any(p in _NEGATIONS for p in words[max(0, i - 2):i])
        shown = f"not {w}" if negated else w
        if any(w.startswith(root) for root in _POS):
            (neg if negated else pos).add(shown)
        elif any(w.startswith(root) for root in _NEG):
            (pos if negated else neg).add(shown)
    for m in re.findall(r"\btoo\s+(\w+)", text.lower()):
        neg.add(f"too {m}")
    return pos, neg


def solve_sentiment(prompt):
    text = prompt.split(":", 1)[1] if ":" in prompt else prompt
    if len(text.split()) < 3:
        return None
    low = text.lower()
    pos, neg = _sentiment_hits(text)
    if re.search(r"\b(wanted to|hoped to|tried to|expected to)\s+(love|like|enjoy)", low):
        pos = {w for w in pos if w not in ("love", "like", "enjoy", "loved", "liked")}
    if not pos and not neg:
        return None
    if re.search(r"\b(but|yet|however|although|though)\b", low) and (not pos or not neg):
        confidence = 0.5
    else:
        confidence = 0.85
    if pos and neg:
        label = "mixed"
        why = (f"positive cues ({', '.join(sorted(pos))}) and "
               f"negative cues ({', '.join(sorted(neg))})")
    elif pos:
        label, why = "positive", f"positive cues: {', '.join(sorted(pos))}"
    else:
        label, why = "negative", f"negative cues: {', '.join(sorted(neg))}"
    return f"Sentiment: {label}. The review contains {why}.", confidence


# ---------------------------------------------------------------------- ner

_MONTHS = ("January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December")
_ORG_SUFFIXES = {"ai", "inc", "inc.", "corp", "corp.", "ltd", "llc", "gmbh", "labs",
                 "company", "university", "institute", "technologies", "systems", "group"}
_LOCATIONS = {
    "paris", "berlin", "london", "madrid", "rome", "lisbon", "amsterdam", "brussels",
    "geneva", "zurich", "munich", "vienna", "prague", "warsaw", "dublin", "barcelona",
    "milan", "moscow", "istanbul", "cairo", "nairobi", "lagos", "johannesburg",
    "tokyo", "osaka", "seoul", "beijing", "shanghai", "shenzhen", "singapore",
    "mumbai", "delhi", "bangalore", "dubai", "sydney", "melbourne", "canberra",
    "auckland", "reykjavik", "basel", "stockholm", "oslo", "copenhagen", "helsinki",
    "athens", "budapest", "bucharest", "kyiv", "edinburgh", "manchester", "lyon",
    "marseille", "hamburg", "frankfurt", "rotterdam", "porto", "valencia", "seville",
    "krakow", "riga", "vilnius", "tallinn", "ljubljana", "zagreb", "belgrade",
    "new york", "san francisco", "los angeles", "seattle", "austin",
    "boston", "chicago", "toronto", "vancouver", "montreal", "mexico city",
    "france", "germany", "spain", "italy", "portugal", "netherlands", "belgium",
    "switzerland", "austria", "poland", "ireland", "united kingdom", "uk",
    "united states", "usa", "canada", "mexico", "brazil", "argentina", "japan",
    "china", "india", "australia", "russia", "egypt", "kenya", "nigeria",
}
_TITLES = ("dr", "mr", "mrs", "ms", "prof")
_SKIP_WORDS = {"The", "This", "That", "These", "Those", "On", "In", "At", "From", "By",
               "A", "An", "It", "He", "She", "They", "We", "I", "What", "Who", "Which",
               "Extract", "Find", "List", "Identify", "Name", "Last", "Next", "Early", "Late"}


def _find_dates(text):
    months = "|".join(_MONTHS)
    patterns = (
        rf"\b(?:last|next|this|early|late)\s+(?:{months})\b",
        rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{months})(?:\s+\d{{4}})?\b",
        rf"\b(?:{months})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?\b",
        rf"\b(?:{months})\s+\d{{4}}\b",
        rf"\b(?:{months})\b",
        r"\b(?:19|20)\d{2}\b",
    )
    found = []
    taken = []
    for pat in patterns:
        for m in re.finditer(pat, text, re.I):
            if any(m.start() < e and m.end() > s for s, e in taken):
                continue
            taken.append((m.start(), m.end()))
            found.append((m.start(), m.group(0), "date"))
    return found, taken


def solve_ner(prompt):
    m = re.search(r"from\s*:\s*(.+)$", prompt, re.I | re.S)
    text = m.group(1) if m else (prompt.split(":", 1)[1] if ":" in prompt else prompt)
    entities, taken = _find_dates(text)
    skipped = 0

    for m in re.finditer(r"\b[A-Z][\w''-]*(?:\s+[A-Z][\w''-]*)*", text):
        if any(m.start() < e and m.end() > s for s, e in taken):
            continue
        span = re.sub(r"['']s$", "", m.group(0))
        tokens = span.split()
        while tokens and tokens[0] in _SKIP_WORDS:
            tokens = tokens[1:]
        if not tokens:
            continue
        span = " ".join(tokens)
        before = text[:m.start()].rstrip()
        titled = any(before.lower().endswith(t) or before.lower().endswith(t + ".")
                     for t in _TITLES)
        if tokens[-1].lower() in _ORG_SUFFIXES or span.endswith("AI"):
            kind = "organization"
        elif span.lower() in _LOCATIONS:
            kind = "location"
        elif titled or len(tokens) >= 2:
            kind = "person"
        else:
            skipped += 1
            continue
        entities.append((m.start(), span, kind))

    entities = sorted(set(entities))
    if not entities:
        return None
    answer = "; ".join(f"{span} - {kind}" for _, span, kind in entities)
    confidence = 0.8 if len(entities) >= 2 and skipped == 0 else 0.4
    return answer, confidence


# ----------------------------------------------------------------- factual

_CAPITALS = {
    "france": "Paris", "germany": "Berlin", "spain": "Madrid", "italy": "Rome",
    "portugal": "Lisbon", "united kingdom": "London", "uk": "London",
    "ireland": "Dublin", "netherlands": "Amsterdam", "belgium": "Brussels",
    "switzerland": "Bern", "austria": "Vienna", "poland": "Warsaw",
    "czech republic": "Prague", "greece": "Athens", "sweden": "Stockholm",
    "norway": "Oslo", "denmark": "Copenhagen", "finland": "Helsinki",
    "russia": "Moscow", "turkey": "Ankara", "egypt": "Cairo", "kenya": "Nairobi",
    "nigeria": "Abuja", "south africa": "Pretoria", "morocco": "Rabat",
    "united states": "Washington, D.C.", "usa": "Washington, D.C.",
    "canada": "Ottawa", "mexico": "Mexico City", "brazil": "Brasília",
    "argentina": "Buenos Aires", "chile": "Santiago", "peru": "Lima",
    "colombia": "Bogotá", "japan": "Tokyo", "china": "Beijing",
    "south korea": "Seoul", "india": "New Delhi", "pakistan": "Islamabad",
    "indonesia": "Jakarta", "thailand": "Bangkok", "vietnam": "Hanoi",
    "philippines": "Manila", "australia": "Canberra", "new zealand": "Wellington",
}


def solve_factual(prompt):
    text = prompt.strip().lower().rstrip("?").strip()
    if " and " in text or prompt.count("?") > 1:
        return None
    m = re.fullmatch(r"(?:what is|what's|name)\s+the\s+capital(?:\s+city)?\s+of\s+([a-z .'-]+)", text)
    if m:
        country = m.group(1).strip().rstrip(".")
        capital = _CAPITALS.get(country)
        if capital:
            return f"The capital of {country.title()} is {capital}.", 0.9
    return None


# -------------------------------------------------------------- entry point

_SOLVERS = {
    "math": solve_math,
    "sentiment": solve_sentiment,
    "ner": solve_ner,
    "factual": solve_factual,
}


def solve_deterministic(prompt: str, category: str):
    """
    Attempts to solve the task using rule-based deterministic helpers.
    Returns (answer_string, confidence_float) or None.
    """
    solver = _SOLVERS.get(category)
    if not solver:
        return None
    try:
        return solver(prompt)
    except Exception:
        return None


def extract_code(text: str) -> str:
    m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S | re.I)
    if m:
        return max(m, key=len).strip()
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`").lstrip("python").strip()
    return t


def solve_math_via_code(prompt: str, local_client) -> tuple:
    from app.services.pyexec import run_python
    
    sys_prompt = (
        "You convert word problems into Python. Write a minimal Python 3 expression "
        "or program that computes the requested result. If a simple math expression suffices, "
        "output ONLY the math expression (e.g. 2400 - 2400 * 0.37 + 800 - 640). If variables "
        "or loops are needed, write a full program that prints the result with print(). "
        "Output ONLY the python code or math expression. No other text."
    )
    
    def evaluate_output(output_content: str):
        c = extract_code(output_content)
        # Try safe eval first
        try:
            val = _safe_eval(c)
            return _fmt(val)
        except Exception:
            pass
            
        # Try running as python script
        ok, out, err = run_python(c)
        if ok and out.strip():
            return out.strip()
        return None

    try:
        res1 = local_client.generate(f"Prompt: {prompt}", system_prompt=sys_prompt, max_tokens=220)
        val1 = evaluate_output(res1.content)
    except Exception:
        val1 = None
        
    if not val1:
        return None
        
    try:
        res2 = local_client.generate(f"Prompt: {prompt}", system_prompt=sys_prompt + " Double check the calculation steps.", max_tokens=220)
        val2 = evaluate_output(res2.content)
    except Exception:
        val2 = None
        
    if val2 and val2 == val1:
        return val1, 0.95
        
    return None


def extract_answer_line(text: str) -> str:
    hits = re.findall(r"ANSWER:\s*(.+)", text)
    if hits:
        return hits[-1].strip().rstrip(".")
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines:
        return lines[-1].strip().rstrip(".")
    return None


def solve_logic_via_cot(prompt: str, local_client) -> tuple:
    sys_prompt = (
        "You are a logical reasoning assistant. Solve the user's puzzle step by step, "
        "reasoning carefully. End your response with exactly: ANSWER: <the final option, name, or value>."
    )
    
    try:
        res1 = local_client.generate(prompt, system_prompt=sys_prompt, max_tokens=300)
        ans1 = extract_answer_line(res1.content)
    except Exception:
        ans1 = None
        
    if not ans1:
        return None
        
    try:
        res2 = local_client.generate(prompt, system_prompt=sys_prompt + " Double check your logic.", max_tokens=300)
        ans2 = extract_answer_line(res2.content)
    except Exception:
        ans2 = None
        
    if ans2 and ans2.lower() == ans1.lower():
        return ans1, 0.95
        
    return None
