import pytest
from ml.tfidf_pipeline import TfidfPipeline

@pytest.fixture
def pipeline():
    texts = [
        "Dinner at restaurant",
        "Client meeting taxi",
        "Netflix monthly subscription",
        "Office cab travel"
    ]
    labels = [
        "Personal Food",
        "Business Travel",
        "Personal Entertainment",
        "Business Travel"
    ]
    p = TfidfPipeline()
    p.fit(texts, labels)
    return p

def mock_verify_known(name):
    return {"verified": True, "merchant_type": "food_delivery"}

def mock_verify_unknown(name):
    return {"verified": False, "merchant_type": None}

def test_predict_basic_schema(monkeypatch, pipeline):
    monkeypatch.setattr(
        "ml.tfidf_pipeline.verify_merchant",
        mock_verify_known
    )

    result = pipeline.predict(["Dinner with friends"], merchants=["swiggy"])

    assert isinstance(result, list)
    assert "category" in result[0]
    assert "confidence" in result[0]
    assert "risk_flags" in result[0]
    assert "suggested_actions" in result[0]

def test_confidence_range(monkeypatch, pipeline):
    monkeypatch.setattr(
        "ml.tfidf_pipeline.verify_merchant",
        mock_verify_known
    )

    result = pipeline.predict(["Client meeting taxi"], merchants=["uber"])
    confidence = result[0]["confidence"]
    assert 0.0 <= confidence <= 1.0

def test_low_quality_note_flag(monkeypatch, pipeline):
    monkeypatch.setattr(
        "ml.tfidf_pipeline.verify_merchant",
        mock_verify_known
    )

    result = pipeline.predict(["food"], merchants=["swiggy"])
    assert result[0]["risk_flags"]["low_note_quality"] is True

def test_uncertain_category_flag(monkeypatch, pipeline):
    monkeypatch.setattr(
        "ml.tfidf_pipeline.verify_merchant",
        mock_verify_unknown
    )

    result = pipeline.predict(["random words here"], merchants=["unknownupi"])
    assert isinstance(result[0]["risk_flags"]["uncertain_category"], bool)

def test_unknown_merchant_flag(monkeypatch, pipeline):
    monkeypatch.setattr(
        "ml.tfidf_pipeline.verify_merchant",
        mock_verify_unknown
    )

    result = pipeline.predict(["Dinner"], merchants=["randomupi123"])
    assert result[0]["risk_flags"]["unknown_merchant"] is True
