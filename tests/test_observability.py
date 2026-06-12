"""Tests for the observability module."""

from src.observability import get_langfuse, observe


class TestObserve:
    def test_is_callable_decorator(self) -> None:
        @observe()
        def fn(x: int) -> int:
            return x * 2

        assert fn(3) == 6

    def test_works_without_langfuse_config(self) -> None:
        @observe()
        def greet(name: str) -> str:
            return f"Hello {name}"

        assert greet("world") == "Hello world"


class TestGetLangfuse:
    def test_returns_client_even_without_credentials(self) -> None:
        client = get_langfuse()
        assert client is not None
