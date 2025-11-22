# ui/app.py
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import requests
import json
import time
from ui.components.explain_card import render_explain_card
from ui.demo_cases import DEMO_CASES, DEMO_CORRECTIONS

# ---------- Configuration ----------
BACKEND_URL = "http://localhost:8000"
CATEGORIES_FILE = "/data/categories.json"  # fallback if backend doesn't provide (not used by default)

# ---------- Helpers ----------
def get_health():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=2.0)
        if r.status_code == 200:
            data = r.json() if r.headers.get("content-type","").startswith("application/json") else {}
            status = data.get("status", "ok")
            return "green", status
        else:
            return "red", f"HTTP {r.status_code}"
    except Exception as e:
        return "red", str(e)

def fetch_categories():
    # Try backend first
    try:
        r = requests.get(f"{BACKEND_URL}/categories", timeout=3.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    # fallback: local simple list
    return [
        "Coffee & Beverages",
        "Food",
        "Fuel",
        "Shopping",
        "Transfer",
        "Utilities",
        "Income",
        "Other"
    ]

def call_predict(text: str):
    payload = {"text": text, "meta": {}}
    r = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=10.0)
    r.raise_for_status()
    return r.json()

def call_feedback(text: str, correct_label: str, user_id: str = "ui_user"):
    payload = {"text": text, "correct_label": correct_label, "user_id": user_id}
    r = requests.post(f"{BACKEND_URL}/feedback", json=payload, timeout=5.0)
    r.raise_for_status()
    return r.json()

def call_retrain(mode="incremental", model="tfidf"):
    payload = {"mode": mode, "model": model}
    r = requests.post(f"{BACKEND_URL}/retrain", json=payload, timeout=30.0)
    r.raise_for_status()
    return r.json()

# ---------- UI layout ----------
st.set_page_config(page_title="CalcBERT — Offline Transaction Categoriser", layout="wide")
st.title("CalcBERT — Offline Transaction Categoriser")
st.caption("Input a messy transaction → see predicted category + confidence + explanation → correct & retrain")

# Health indicator
health_color, health_msg = get_health()
health_label = f"Backend: {BACKEND_URL} — {health_msg}"
if health_color == "green":
    st.success(health_label)
else:
    st.error(health_label)

# two-column layout: left for input, right for demo & timeline
left, right = st.columns([2, 1])

with left:
    st.header("Predict a transaction")
    text = st.text_input("Transaction string", placeholder="STARBCKS #1023 MUMBAI 12:32PM")

    predict_btn = st.button("Predict", key="predict_primary")
    if predict_btn and text.strip():
        with st.spinner("Calling predict..."):
            try:
                resp = call_predict(text)
            except Exception as e:
                st.error(f"Predict failed: {e}")
                resp = None

        if resp:
            # Show prediction card
            category = resp.get("category", "Unknown")
            confidence = float(resp.get("confidence", 0.0))
            model_used = resp.get("model_used") or resp.get("explanation",{}).get("model_used","unknown")
            st.markdown(f"<h2 style='margin:0'>{category}</h2>", unsafe_allow_html=True)
            conf_pct = int(round(confidence * 100))
            st.write(f"Confidence: {confidence:.2f}")
            # progress bar with color logic
            if confidence >= 0.8:
                bar_color = "#2ECC71"  # green
            elif confidence >= 0.5:
                bar_color = "#F39C12"  # amber
            else:
                bar_color = "#E74C3C"  # red

            st.markdown(
                f"""
                <div style="background:#eee;border-radius:8px;padding:6px;">
                  <div style="width:{conf_pct}%;background:{bar_color};height:18px;border-radius:8px;"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write(f"Model: `{model_used}`")

            explanation = resp.get("explanation", {})
            render_explain_card(explanation)

            # Correction UI
            st.write("---")
            st.write("### Correct prediction")
            categories = fetch_categories()
            correct_label = st.selectbox("Select correct category", options=categories, index=0, key="correction_select")
            if st.button("Submit Correction"):
                try:
                    fb = call_feedback(text, correct_label)
                    st.success(f"Saved feedback id: {fb.get('id', 'unknown')}")
                except Exception as e:
                    st.error(f"Feedback failed: {e}")

with right:
    st.header("Demo runner & Timeline")
    st.write("Quick demo with canned examples.")
    timeline = st.empty()
    if st.button("Run demo sequence"):
        timeline_lines = []
        for idx, case in enumerate(DEMO_CASES):
            timeline_lines.append(f"▶ Predict: {case}")
            timeline.markdown("\n".join(timeline_lines))
            try:
                resp = call_predict(case)
                cat = resp.get("category")
                conf = resp.get("confidence")
                timeline_lines.append(f"  → Predicted: {cat} ({conf:.2f})")
            except Exception as e:
                timeline_lines.append(f"  → Predict failed: {e}")
                timeline.markdown("\n".join(timeline_lines))
                continue

            # simulate a demo correction if configured
            if idx in DEMO_CORRECTIONS:
                correct = DEMO_CORRECTIONS[idx]
                timeline_lines.append(f"  → Demo correction: {correct}")
                timeline.markdown("\n".join(timeline_lines))
                try:
                    fb = call_feedback(case, correct)
                    timeline_lines.append(f"  → Feedback saved id {fb.get('id')}")
                except Exception as e:
                    timeline_lines.append(f"  → Feedback failed: {e}")
                # trigger retrain incremental demo
                try:
                    retr = call_retrain(mode="incremental", model="tfidf")
                    timeline_lines.append(f"  → Retrain: {retr.get('status')} - {retr.get('details')}")
                except Exception as e:
                    timeline_lines.append(f"  → Retrain failed: {e}")

                # re-run predict to show "improvement"
                try:
                    resp2 = call_predict(case)
                    cat2 = resp2.get("category")
                    conf2 = resp2.get("confidence")
                    timeline_lines.append(f"  → Re-predict: {cat2} ({conf2:.2f})")
                except Exception as e:
                    timeline_lines.append(f"  → Re-predict failed: {e}")

            # small pause for demo readability
            timeline.markdown("\n".join(timeline_lines))
            time.sleep(0.6)

    st.write("---")
    st.write("Admin")
    if st.button("Reload taxonomy (categories)"):
        cats = fetch_categories()
        st.success(f"Loaded {len(cats)} categories")
        st.write(cats)

    if st.button("Download logs (not implemented)"):
        st.info("This will download logs. (Implement server-side to return logs)")

st.write("")
st.write("**Notes:** The app expects backend endpoints: GET /health, GET /categories (optional), POST /predict, POST /feedback, POST /retrain. See README in repo to wire backend.")
