"""Core module for the sample repo."""


class CoreClass:
    """The main class for demonstration."""

    def __init__(self, name: str) -> None:
        self.name = name

    def greet(self) -> str:
        """Return a greeting string."""
        return f"Hello, {self.name}!"

    def farewell(self) -> str:
        """Return a farewell string."""
        return f"Goodbye, {self.name}!"
