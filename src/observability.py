"""LangFuse observability setup.

Re-exports the LangFuse observe decorator and client.
When LangFuse is not configured, the decorator is a no-op.

Usage:
    from src.observability import observe

    @observe()
    def my_pipeline_step(...):
        ...
"""

from langfuse import Langfuse
from langfuse import observe as _observe

observe = _observe


def get_langfuse() -> Langfuse | None:
    try:
        return Langfuse()
    except Exception:
        return None
