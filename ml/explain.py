def explain_distilbert(text, model_wrapper, top_k=3):
    tokens = model_wrapper.tokenizer.tokenize(text)
    return [{"token": t, "score": round(1/top_k, 2)} for t in tokens[:top_k]]
