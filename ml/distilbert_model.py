import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import json, os
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
        with open(os.path.join(model_dir, "label_map.json")) as f:
            self.label_map = json.load(f)

    def save(self, out_dir):
        self.tokenizer.save_pretrained(out_dir)
        self.model.save_pretrained(out_dir)
        with open(os.path.join(out_dir, "label_map.json"), "w") as f:
            json.dump(self.label_map, f)

    def predict(self, texts, top_k=3):
        self.model.eval()
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors='pt')
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
                "probs": {self.label_map[str(j)]: float(p[j]) for j in range(len(p))},
                "raw_logits": out.logits[i].cpu().tolist(),
                "top_tokens": explain_distilbert(texts[i], self, top_k=top_k)
            })
        return results

    def get_embedding(self, text):
        
        self.model.eval()
        enc = self.tokenizer([text], padding=True, truncation=True, max_length=64, return_tensors='pt')
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            hidden = self.model.distilbert(**enc).last_hidden_state
            cls_emb = hidden[:,0,:].cpu().numpy()
        return cls_emb[0]

    def predict_with_embeddings(self, texts, top_k=3):
       
        results = self.predict(texts, top_k=top_k)
        for i, text in enumerate(texts):
            results[i]["embedding"] = self.get_embedding(text)
        return results
