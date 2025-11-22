
import sys
import os
import time
import streamlit as st
import requests
from datetime import datetime


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ui.components.explain_card import render_explain_card
from ml.tfidf_pipeline import TfidfPipeline

BACKEND_URL = "http://localhost:8000"


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

@st.cache_data(ttl=300)  
def fetch_categories():
    """Fetch categories from backend API (more reliable than loading model file)."""
    try:
        r = requests.get(f"{BACKEND_URL}/categories", timeout=5.0)
        r.raise_for_status()
        data = r.json()
        if data.get("status") == "ok":
            return data.get("categories", [])
        else:
           
            pipeline = TfidfPipeline()
            pipeline.load("saved_models/tfidf")
            return list(pipeline.le.classes_)
    except Exception as e:
        
        try:
            pipeline = TfidfPipeline()
            pipeline.load("saved_models/tfidf")
            return list(pipeline.le.classes_)
        except:
            
            return []

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


if "session_transactions" not in st.session_state:
    st.session_state.session_transactions = []

if "last_predicted" not in st.session_state:
    st.session_state.last_predicted = None

if "last_prediction_result" not in st.session_state:
    st.session_state.last_prediction_result = None


st.set_page_config(page_title="CalcBERT — Offline Transaction Categoriser", layout="wide")
st.title("CalcBERT — Offline Transaction Categoriser")
st.caption("Input a messy transaction → see predicted category + confidence + explanation → correct & retrain")


health_color, health_msg = get_health()
health_label = f"Backend: {BACKEND_URL} — {health_msg}"
if health_color == "green":
    st.success(health_label)
else:
    st.error(health_label)


left, right = st.columns([2, 1])

with left:
    st.header("Predict a transaction")
    text_input_value = st.text_input(
        "Transaction string",
        value=st.session_state.last_predicted or "",
        placeholder="STARBCKS #1023 MUMBAI 12:32PM"
    )

    if st.button("Predict"):
        if not text_input_value.strip():
            st.warning("Please enter a transaction string.")
        else:
            with st.spinner("Calling predict..."):
                try:
                    resp = call_predict(text_input_value)
                    st.session_state.last_predicted = text_input_value
                    st.session_state.last_prediction_result = resp
                    if text_input_value not in st.session_state.session_transactions:
                        st.session_state.session_transactions.append(text_input_value)
                except Exception as e:
                    st.error(f"Predict failed: {e}")
                    resp = None

   
    resp = st.session_state.last_prediction_result
    if resp:
        category = resp.get("category", "Unknown")
        confidence = float(resp.get("confidence", 0.0))
        model_used = resp.get("model_used") or resp.get("explanation",{}).get("model_used","unknown")
        st.markdown(f"<h2 style='margin:0'>{category}</h2>", unsafe_allow_html=True)
        conf_pct = int(round(confidence*100))
        st.write(f"Confidence: {confidence:.2f}")
        bar_color = "#2ECC71" if confidence >= 0.8 else "#F39C12" if confidence >= 0.5 else "#E74C3C"
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

        # Correction form
        st.write("---")
        st.write("### Correct prediction")
        categories = fetch_categories()
        correction_form_key = f"correction_form_{st.session_state.last_predicted}"
        with st.form(key=correction_form_key):
            correct_label = st.selectbox("Select correct category", options=categories, index=categories.index(category) if category in categories else 0)
            submit_btn = st.form_submit_button("Submit Correction")
            if submit_btn:
                try:
                    fb = call_feedback(st.session_state.last_predicted, correct_label)
                    st.success(f"Saved feedback id: {fb.get('id', 'unknown')}")
                except Exception as e:
                    st.error(f"Feedback failed: {e}")

with right:
    st.header("Demo runner & Timeline")
    timeline = st.empty()
    st.write("Shows only transactions predicted in this session.")
    if st.button("Run demo sequence"):
        lines = []
        for t in st.session_state.session_transactions:
            lines.append(f"▶ Predict: {t}")
            timeline.markdown("\n".join(lines))
            try:
                resp = call_predict(t)
                cat = resp.get("category")
                conf = resp.get("confidence")
                lines.append(f"  → Predicted: {cat} ({conf:.2f})")
            except Exception as e:
                lines.append(f"  → Predict failed: {e}")
                timeline.markdown("\n".join(lines))
                continue
            timeline.markdown("\n".join(lines))
            time.sleep(0.4)

    st.write("---")
    st.write("Admin")
    if st.button("Retrain model"):
        with st.spinner("Retraining TF-IDF with feedback..."):
            try:
                retr = call_retrain(mode="full", model="tfidf")
                if retr.get("status") == "complete":
                    st.success(f"✓ Retrain completed: {retr.get('details')}")
                    
                    fetch_categories.clear()
                    
                    cat_count = retr.get("categories_trained", 0)
                    if cat_count > 0:
                        st.info(f"✓ Model now has {cat_count} categories")
                        if "categories_list" in retr:
                            st.write("**Categories:**", ", ".join(retr.get("categories_list", [])))
                    else:
                        st.warning("⚠ Could not determine category count from response")
                    
                    st.rerun()
                else:
                    st.warning(f"Retrain status: {retr.get('status')} - {retr.get('details')}")
            except Exception as e:
                st.error(f"Retrain failed: {e}")
                import traceback
                st.code(traceback.format_exc())

    if st.button("Reload categories"):
        
        fetch_categories.clear()
        cats = fetch_categories()
        st.success(f"Loaded {len(cats)} categories")
        st.write(cats)

