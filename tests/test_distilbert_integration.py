from ml.distilbert_model import DistilBertWrapper

def test_predict_shape():
    m = DistilBertWrapper("saved_models/distilbert")
    output = m.predict(["test"])
    print("Prediction output:", output)
    keys = ["label", "confidence", "probs", "raw_logits", "top_tokens"]
    for k in keys:
        assert k in output[0]
