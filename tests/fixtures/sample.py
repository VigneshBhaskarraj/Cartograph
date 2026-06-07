"""A tiny module used by the M0 vertical-slice tests."""


class Animal:
    """Base class for things that make sound."""

    def speak(self) -> str:
        return "..."


class Dog(Animal):
    def speak(self) -> str:
        # WHY: dogs bark, overriding the silent base.
        return bark()


def bark() -> str:
    """Return the canonical dog sound."""
    return "woof"


def greet(name: str) -> str:
    return f"hello {name}"
