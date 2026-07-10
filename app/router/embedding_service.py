import hashlib
import math
from typing import List


def get_embedding(text: str, dim: int = 16) -> List[float]:
    """
    Generate a pseudo-embedding for a given text.
    In a real implementation, this would call an embedding model (like sentence-transformers or OpenAI).
    This stub creates a deterministic pseudo-random vector based on the string's hash.
    """
    # Create a deterministic seed based on text content
    text_hash = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)

    vec = []
    for i in range(dim):
        # Pseudo-random value between -1 and 1
        val = math.sin((text_hash + i) ** 1.5)
        vec.append(val)

    # Normalize the vector
    norm = math.sqrt(sum(v**2 for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]

    return vec
