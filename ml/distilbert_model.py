import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import json, os

class DistilBertWrapper:
    def __init__(self, model_dir="saved_models/distilbert", device="cpu"):
        self.device = torch.device(device)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_dir).to(self.device)
        with open(os.path.join(model_dir, "label_map.json")) as f:
            self.label_map = json.load(f)

    def predict(self, texts, top_k=3):
        self.model.eval()
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors='pt')
        for k in enc:
            enc[k] = enc[k].to(self.device)
        with torch.no_grad():
            out = self.model(**enc)
        probs = torch.nn.functional.softmax(out.logits, dim=-1).cpu().numpy()
        results = []
        for i, p in enumerate(probs):
            idx = p.argmax()
            label = self.label_map[str(idx)] if str(idx) in self.label_map else str(idx)
            results.append({
                "label": label,
                "confidence": float(p.max()),
                "probs": {self.label_map[str(j)]: float(p[j]) for j in range(len(p))},
                "raw_logits": out.logits[i].tolist(),
                "top_tokens": []  
            })
        return results

    def save(self, out_dir):
        self.tokenizer.save_pretrained(out_dir)
        self.model.save_pretrained(out_dir)
        with open(os.path.join(out_dir, "label_map.json"), "w") as f:
            json.dump(self.label_map, f)

    def load(self, model_dir):
        return DistilBertWrapper(model_dir=model_dir, device=str(self.device))
