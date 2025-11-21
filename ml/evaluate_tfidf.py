import pandas as pd
import json
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from ml.tfidf_pipeline import TfidfPipeline
from ml.data_pipeline import normalize_series

def evaluate(model_dir="saved_models/tfidf", test_csv="data/GHCI_clean.csv"):
    df = pd.read_csv(test_csv)
    texts = normalize_series(df["transaction_text"]).tolist()
    labels = df["category"].tolist()

    p = TfidfPipeline()
    p.load(model_dir)

    preds = [r["label"] for r in p.predict(texts)]
    report = classification_report(labels, preds, output_dict=True)

    # save metrics
    with open("metrics/tfidf_metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    # confusion matrix
    cm = confusion_matrix(labels, preds, labels=p.le.classes_)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=p.le.classes_,
                yticklabels=p.le.classes_)
    plt.savefig("metrics/confusion_tfidf.png")

    print(" Metrics saved.")

if __name__ == "__main__":
    evaluate()
