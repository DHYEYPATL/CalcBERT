import pandas as pd
import argparse
from ml.data_pipeline import normalize_series
from ml.tfidf_pipeline import TfidfPipeline

def main(train_csv="data/GHCI_clean.csv", out="saved_models/tfidf"):
    df = pd.read_csv(train_csv)
    texts = normalize_series(df["transaction_text"]).tolist()
    labels = df["category"].tolist()

    p = TfidfPipeline(max_features=5000)
    p.fit(texts, labels)
    p.save(out)

    print("Model saved to:", out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/GHCI_clean.csv")
    parser.add_argument("--out", default="saved_models/tfidf")
    args = parser.parse_args()
    main(args.train, args.out)
