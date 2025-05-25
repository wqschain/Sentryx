import os
import sys
import pytest

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close() 