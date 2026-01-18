def explain_distilbert(text, wrapper, top_k=3):
    tokens = wrapper.tokenizer.tokenize(text)

    if not tokens:
        return []

    # Prefer meaningful tokens, skip special tokens
    filtered = [t for t in tokens if not t.startswith("##")]

    return filtered[:top_k]
