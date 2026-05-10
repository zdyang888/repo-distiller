"""Tests for core module."""
from src.core import CoreClass


def test_greet():
    c = CoreClass("World")
    assert c.greet() == "Hello, World!"
