import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Добавляем src в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from main import app  # Импортируем app из src.main

@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client