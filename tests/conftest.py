from __future__ import annotations

import os
import sys

import pytest
from fastapi.testclient import TestClient

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.database import Base, SessionLocal, engine, create_tables
from app.main import app


@pytest.fixture(scope="session", autouse=True)
def _create_tables():
    create_tables()
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def db_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture()
def sample_prompt():
    return "Write a Python function that computes the Fibonacci sequence using dynamic programming."


@pytest.fixture()
def simple_prompt():
    return "What is 2 + 2?"


@pytest.fixture()
def complex_prompt():
    return (
        "Explain the difference between quicksort and merge sort algorithms. "
        "Compare their time complexity, space complexity, and real-world use cases. "
        "Which one is better for sorting linked lists?"
    )


@pytest.fixture()
def code_prompt():
    return (
        "def binary_search(arr, target):\n"
        "    low, high = 0, len(arr) - 1\n"
        "    while low <= high:\n"
        "        mid = (low + high) // 2\n"
        "        if arr[mid] == target:\n"
        "            return mid\n"
        "        elif arr[mid] < target:\n"
        "            low = mid + 1\n"
        "        else:\n"
        "            high = mid - 1\n"
        "    return -1"
    )


@pytest.fixture()
def conversation_history():
    return [
        {"role": "user", "content": "Tell me about Python decorators."},
        {"role": "assistant", "content": "Python decorators are functions that modify other functions. They use the @syntax."},
        {"role": "user", "content": "Can you give me an example?"},
        {"role": "assistant", "content": "Sure! A common example is @timer which measures execution time."},
        {"role": "user", "content": "How about a caching decorator?"},
        {"role": "assistant", "content": "A caching decorator stores results of expensive function calls."},
    ]
