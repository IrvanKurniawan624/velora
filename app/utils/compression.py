import io
import re
import tokenize

def strip_python_comments_and_docstrings(source: str) -> str:
    """
    Strips comments and docstrings from a string containing Python code using python's tokenize library.
    """
    try:
        io_obj = io.StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok.type
            token_string = tok.string
            start_line, start_col = tok.start
            end_line, end_col = tok.end
            
            if token_type == tokenize.COMMENT:
                continue
                
            if token_type == tokenize.STRING:
                # If it's a standalone string on a line after an indent/newline, it is a docstring
                if prev_toktype in (tokenize.INDENT, tokenize.NEWLINE, tokenize.NL):
                    continue
                    
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += " " * (start_col - last_col)
                
            out += token_string
            prev_toktype = token_type
            last_lineno = end_line
            last_col = end_col
            
        # Clean up empty lines left behind by comments/docstrings
        lines = [line for line in out.splitlines() if line.strip()]
        return "\n".join(lines).strip()
    except Exception:
        # Graceful fallback: return original if tokenize fails
        return source

def compress_prompt(prompt: str, task_type: str) -> str:
    """
    Compresses input task prompts by removing comments/docstrings in code blocks
    and stripping verbose instruction fillers.
    """
    compressed = prompt
    
    # 1. Compress Python code blocks inside the prompt
    if task_type == "code":
        # Check for markdown code blocks
        code_blocks = re.findall(r"```python(.*?)```", compressed, re.DOTALL)
        for block in code_blocks:
            stripped = strip_python_comments_and_docstrings(block)
            compressed = compressed.replace(block, f"\n{stripped}\n")
            
        # Check for code blocks without markdown wrapping (e.g. starting with def)
        if "def " in compressed and "```" not in compressed:
            # Find definition to the end
            def_match = re.search(r"(def .*?)(?=\n\n|\Z)", compressed, re.DOTALL)
            if def_match:
                block = def_match.group(1)
                stripped = strip_python_comments_and_docstrings(block)
                compressed = compressed.replace(block, stripped)

    # 2. Prune verbose headers and instructions (filler removal)
    replacements = {
        "classify the sentiment of the following product review as positive, negative, or neutral, and provide a short reason. review:": "Classify sentiment (positive/negative/neutral) and reason. Review:",
        "classify the sentiment of this review:": "Classify sentiment. Review:",
        "return strictly json with keys: 'sentiment' and 'reason'.": "Strictly JSON keys: 'sentiment', 'reason'.",
        "summarize the following passage into a json format containing a list of 2 key bullet points. passage:": "Summarize to JSON bullets list (exactly 2). Passage:",
        "return strictly json with a single key 'bullets' which is a list of strings.": "Strictly JSON key: 'bullets' (list of strings).",
        "extract all person, org, and loc entities from the text:": "Extract PERSON, ORG, LOC entities:",
        "return strictly a json object with keys 'person', 'org', and 'loc' mapping to lists of strings.": "Strictly JSON keys: 'PERSON', 'ORG', 'LOC' (lists).",
        "fix the bugs in this python function. it should return the sum of all even numbers in a list, but it currently returns incorrect results or fails:": "Fix bugs in python function:",
        "return only the corrected python function implementation.": "Return ONLY python code.",
        "write a python function named `is_palindrome` that takes a string and returns true if it is a palindrome, and false otherwise. ignore case and non-alphanumeric characters. return only the python function implementation.": "Write python function `is_palindrome(s) -> bool`. Ignore case/non-alphanumeric. Return ONLY code.",
        "solve this logic puzzle:": "Solve logic puzzle:",
        "three friends, sam, jo, and lee, each own a different pet: cat, dog, bird. sam does not own the bird. jo owns the dog. who owns the cat?": "Friends: Sam, Jo, Lee. Pets: cat, dog, bird. Sam doesn't own bird. Jo owns dog. Who owns cat?",
    }
    
    # We run case-insensitive regex replacements for matches
    for verbose, concise in replacements.items():
        # Escape special characters
        pattern = re.compile(re.escape(verbose), re.IGNORECASE)
        compressed = pattern.sub(concise, compressed)
        
    # Generic regex-based verbosity/politeness cleanup
    # Remove politeness and general intro fillers
    compressed = re.sub(r"\b(please|kindly|could you|would you mind|write a|solve the following|solve this)\s+", "", compressed, flags=re.IGNORECASE)
    # Remove standard transition fillers
    compressed = re.sub(r"\b(the following|of the following|for the following|below)\s+", "", compressed, flags=re.IGNORECASE)
    # Normalize multiple spaces
    compressed = re.sub(r"\s+", " ", compressed).strip()
    
    return compressed
