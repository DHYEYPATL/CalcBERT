# tests/test_ui_api_integration.py
import requests
import pytest
import os

BACKEND = os.environ.get("BACKEND_URL", "http://localhost:8000")

def test_predict_endpoint_running():
    r = requests.post(f"{BACKEND}/predict", json={"text":"STARBCKS #1"})
    assert r.status_code == 200, "Predict endpoint not reachable (status != 200)"
    data = r.json()
    assert "category" in data and "confidence" in data and "explanation" in data

def test_explain_card_handles_missing_fields():
    # This is a unit-style test of the UI component but since streamlit components are visual,
    # we at least validate the shape the backend returns when fields missing:
    r = requests.post(f"{BACKEND}/predict", json={"text": "UNKNOWN XYZ"})
    assert r.status_code == 200
    data = r.json()
    # allow explanation to miss fields but it should exist
    explanation = data.get("explanation", {})
    # top_tokens may be missing, but explanation should be a dict
    assert isinstance(explanation, dict)
