from ml.tfidf_pipeline import TfidfPipeline


def test_partial_fit_applies_known_label():
    """
    partial_fit should apply updates when the label already
    exists in the trained LabelEncoder.
    """
    texts = ["Lunch", "Office taxi"]
    labels = ["Personal Food", "Business Travel"]

    p = TfidfPipeline()
    p.fit(texts, labels)

    # reinforce an existing label
    count = p.partial_fit(
        ["Dinner"],
        ["Personal Food"]
    )

    assert count == 1


def test_partial_fit_increases_confidence():
    """
    Reinforcing the same (text, label) pair should not reduce
    the model's confidence for that prediction.
    """
    texts = ["Netflix subscription", "Office taxi"]
    labels = ["Personal Entertainment", "Business Travel"]

    p = TfidfPipeline()
    p.fit(texts, labels)

    before = p.predict(
        ["Netflix subscription"],
        merchants=["netflix"]
    )[0]["confidence"]

    # reinforce the same label
    p.partial_fit(
        ["Netflix subscription"],
        ["Personal Entertainment"]
    )

    after = p.predict(
        ["Netflix subscription"],
        merchants=["netflix"]
    )[0]["confidence"]

    assert after >= before
