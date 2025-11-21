import time, numpy as np
from ml.distilbert_model import DistilBertWrapper

def benchmark_inference(model_dir="saved_models/distilbert", n_iter=10, batch_size=8, device="cpu"):
    model = DistilBertWrapper(model_dir, device)
    texts = ["Test transaction"] * batch_size
    times = []
    for _ in range(n_iter):
        t0 = time.time()
        model.predict(texts)
        times.append(time.time() - t0)
    print("Avg:", np.mean(times)*1000, "ms/batch  |  95th %:", np.percentile(times, 95)*1000, "ms/batch")
