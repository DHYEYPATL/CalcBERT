import pandas as pd
from ml.data_pipeline import (
    normalize_text,
    detect_low_quality_note,
    extract_intent_phrases
)

def test_normalize_text_basic():
    text = "STARBCKS #1023 MUMBAI"
    out = normalize_text(text)
    assert out == "starbcks mumbai"
    assert isinstance(out, str)

def test_normalize_text_punctuation_and_case():
    text = "Uber!! Ride@@"
    out = normalize_text(text)
    assert out == "uber ride"

def test_detect_low_quality_note_short():
    assert detect_low_quality_note("food") is True
    assert detect_low_quality_note("upi") is True

def test_detect_low_quality_note_valid():
    assert detect_low_quality_note("Dinner with client") is False

def test_detect_low_quality_note_numbers_only():
    assert detect_low_quality_note("123456") is True

def test_detect_low_quality_note_punctuation_only():
    assert detect_low_quality_note("!!!") is True

def test_extract_intent_phrases_business():
    text = "Client meeting at office"
    intents = extract_intent_phrases(text)
    joined = " ".join(intents)
    assert "client" in joined
    assert "meeting" in joined

def test_extract_intent_phrases_personal():
    text = "Dinner and movie night"
    intents = extract_intent_phrases(text)
    joined = " ".join(intents)
    assert "dinner" in joined
    assert "movie" in joined
