"""
sample.py — Small example module for the RAG demo.

Key ideas:
- Well-structured docstrings and comments improve chunk quality.
- Short, purposeful functions make it easy for the model to summarize.
"""

from typing import List


def normalize_scores(scores: List[float]) -> List[float]:
    """
    Normalize a list of numeric scores into [0, 1] using min-max scaling.

    Args:
        scores: A list of floats (e.g., relevance scores).

    Returns:
        A list of floats where the minimum becomes 0.0 and the maximum becomes 1.0.
        If all scores are equal, returns a list of 1.0 values.

    Example:
        >>> normalize_scores([2.0, 4.0, 6.0])
        [0.0, 0.5, 1.0]
    """
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def summarize_text(text: str, max_words: int = 30) -> str:
    """
    Produce a simple extractive summary by taking the first N words.

    This is *not* an LLM function; it's a trivial baseline so the RAG demo
    can contrast naive summarization with grounded generation.

    Args:
        text: Source text to summarize.
        max_words: Maximum number of words for the summary.

    Returns:
        The first `max_words` words joined by spaces.
    """
    words = text.split()
    return " ".join(words[:max_words])


class RunningAverage:
    """
    Compute a running (online) average.

    Useful to show the assistant can explain class behavior and reference code.

    Example:
        >>> ra = RunningAverage()
        >>> for x in [10, 20, 30]:
        ...     ra.update(x)
        >>> round(ra.value, 2)
        20.0
    """

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0

    def update(self, x: float) -> None:
        """Add a new observation and update the running average."""
        self.count += 1
        self.total += float(x)

    @property
    def value(self) -> float:
        """Return the current average value (0.0 if no observations)."""
        if self.count == 0:
            return 0.0
        return self.total / self.count
