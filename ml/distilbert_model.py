import torch
import json
import os
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from ml.explain import explain_distilbert


class DistilBertWrapper:
    def __init__(self, model_dir="saved_models/distilbert", device="cpu"):
        self.device = torch.device(device)
        self.model_dir = model_dir
        self.load(model_dir)

    def load(self, model_dir=None):
        model_dir = model_dir or self.model_dir
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_dir).to(self.device)

        label_map_path = os.path.join(model_dir, "label_map.json")
        if os.path.exists(label_map_path):
            with open(label_map_path) as f:
                self.label_map = json.load(f)
        else:
            # Fallback: create default label map based on model num_labels
            num_labels = self.model.config.num_labels
            self.label_map = {str(i): f"label_{i}" for i in range(num_labels)}
            print(f"⚠ label_map.json not found at {label_map_path}, using default label map")

    def predict(self, texts, top_k=3):
        self.model.eval()

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            out = self.model(**enc)

        probs = torch.nn.functional.softmax(out.logits, dim=-1).cpu().numpy()

        results = []
        for i, p in enumerate(probs):
            idx = int(p.argmax())
            label = self.label_map.get(str(idx), str(idx))

            results.append({
                "label": label,
                "confidence": float(p.max()),
                "probs": {
                    self.label_map.get(str(j), f"label_{j}"): float(p[j])
                    for j in range(len(p))
                },
                "raw_logits": out.logits[i].cpu().tolist(),
                "top_tokens": explain_distilbert(texts[i], self, top_k=top_k)
            })

        return results

    def get_embedding(self, text):
        self.model.eval()

        enc = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            hidden = self.model.distilbert(**enc).last_hidden_state

        return hidden[:, 0, :].cpu().numpy()[0]
